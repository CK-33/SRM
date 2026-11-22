"""E09.2: proporção de AWGN com orçamento e instantes fixos."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from G5_E09_AWGN import DOSES, SPLITS, collect, csv_rows
from G5_E08_Hibrido import fit_head, metrics, standardize

FRACTIONS = (0, .125, .25, .375, .50)


def selection(origins, labels):
    """Ordena subconjuntos aninhados, estratificados por origem, classe e dose."""
    rng = np.random.default_rng(501)
    groups = []
    for origin in SPLITS["treino"]:
        odd = np.flatnonzero(origins == origin)[1::2]
        for dose in range(3):
            eligible = odd[dose::3]
            for label in (0,1):
                group = eligible[labels[eligible] == label]
                groups.append(rng.permutation(group))
    return groups


def evaluate(head, x, y, origins):
    with torch.no_grad():
        logits = head(x).flatten()
    return dict(agregado=metrics(logits,y),
                por_origem={o:metrics(logits[origins==o],y[origins==o]) for o in SPLITS["validacao"]})


def plot(results, path):
    image = Image.new("RGB",(1240,750),"white")
    d = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",19)
    title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",30)
    ink = "#182c40"
    d.text((40,25),"G5/E09.2 | Proporção de janelas AWGN no treino",font=title,fill=ink)
    d.text((40,72),"As mesmas 2.305 janelas por braço · validação em R03 e R09",font=font,fill=ink)
    colors = {"limpo":"#176ab0","50":"#437c53","40":"#875ab5","30":"#ae541c"}
    for row, metric in enumerate(("fn","fp")):
        for col, arm in enumerate(("cnn","hibrido")):
            left, top = 105+605*col, 118+278*row
            bottom = top+185
            d.text((left,top),f"{arm.upper()} · {'Omissões' if metric=='fn' else 'Falsos positivos'}",
                   font=font,fill=ink)
            ymax = max(1,max(r["modelos"][arm][c]["agregado"][metric]
                             for r in results for c in colors))
            xs = [left+108*i for i in range(len(results))]
            for i in range(5):
                yy = bottom-146*i/4
                d.line((left,yy,xs[-1],yy),fill="#dbe3ec")
                if ymax >= 4 or i in (0,4):
                    d.text((left-12,yy),f"{ymax*i/4:.0f}",anchor="rm",font=font,fill=ink)
            for xx, r in zip(xs,results):
                d.text((xx,bottom+12),f'{100*r["fracao_real"]:.1f}%',anchor="mt",font=font,fill=ink)
            for case,color in colors.items():
                points = [(xx,bottom-146*r["modelos"][arm][case]["agregado"][metric]/ymax)
                          for xx,r in zip(xs,results)]
                d.line(points,fill=color,width=3)
                for xx,yy in points:
                    d.ellipse((xx-4,yy-4,xx+4,yy+4),fill=color)
    for i,(case,color) in enumerate(colors.items()):
        d.text((120+250*i,695),"Limpo" if case=="limpo" else f"{case} dB",font=font,fill=color)
    image.save(path)


def run(gate, run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate/"modelos/pesos_pre_treinados"))
    weights = ResNet18_Weights.IMAGENET1K_V1
    encoder = resnet18(weights=weights).eval().requires_grad_(False)
    encoder.fc = nn.Identity()
    root = gate.parent/"Data and Scripts/Gate4_Exp03_Run01"
    rows = csv_rows(gate/"dados/manifestos/G5_E03_Manifesto_Canonicos.csv")
    allowed = {o for ids in SPLITS.values() for o in ids}
    assert len(rows)==3120 and {r["origem"] for r in rows}==allowed
    provenance = tuple({r["id"]:r for r in csv_rows(root/f"G4_E03_{name}.csv")
                        if r["id"].split("_")[0] in allowed} for name in ("Manifesto","Parametros"))
    base_s,base_c,y,origins = collect(root,rows,"treino","limpo",encoder,provenance)
    mix_s,mix_c,ym,om = collect(root,rows,"treino","misto",encoder,provenance)
    assert torch.equal(y,ym) and np.array_equal(origins,om)
    coverage = selection(origins,y.numpy())
    val = {c:collect(root,rows,"validacao",c,encoder,provenance) for c in ("limpo",*DOSES)}
    assert (len(y),int(y.sum()))==(2305,1741)
    assert all((len(a[2]),int(a[2].sum()))==(576,435) for a in val.values())
    reference_id = "G5_E09_1_20261122_000704_647"
    ref = json.loads((gate/"resultados/execucoes"/reference_id/"metricas.json").read_text(encoding="utf-8"))
    dest = gate/"resultados/execucoes"/run_id
    dest.mkdir()
    results = []
    report = dict(referencia=reference_id,seed_subconjuntos=501,treino_janelas=2305,
                  pesos=weights.name,torch=torch.__version__,resultados=results)
    for fraction in FRACTIONS:
        indices = np.concatenate([g[:round(len(g)*fraction/.50)] for g in coverage])
        assert len(indices)==len(set(indices))
        mask = torch.zeros(len(y),dtype=torch.bool)
        mask[indices] = True
        s = torch.where(mask[:,None],mix_s,base_s)
        c = torch.where(mask[:,None],mix_c,base_c)
        s,_,sm,ss = standardize(s,s)
        c,_,cm,cs = standardize(c,c)
        entry = dict(fracao_alvo=fraction,fracao_real=len(indices)/len(y),
                     janelas_awgn=len(indices),modelos={})
        for arm, train in (("cnn",c),("hibrido",torch.cat((c,s),1))):
            head = fit_head(train,y)
            out = {}
            for case,(sv,cv,target,origin) in val.items():
                sv,cv = (sv-sm)/ss,(cv-cm)/cs
                out[case] = evaluate(head,cv if arm=="cnn" else torch.cat((cv,sv),1),target,origin)
                if fraction in (0,.50):
                    prior = ref["resultados"]["limpo" if fraction==0 else "misto"][arm][case]["agregado"]
                    current = out[case]["agregado"]
                    assert all(current[k]==prior[k] for k in ("fp","fn","tp","tn")), (fraction,arm,case)
            entry["modelos"][arm] = out
            checkpoint = gate/"modelos/checkpoints"/f"{run_id}_{round(fraction*1000):03d}_{arm}.pt"
            with checkpoint.open("xb") as file:
                torch.save(dict(head=head.state_dict(),stats_media=sm,stats_desvio=ss,
                                cnn_media=cm,cnn_desvio=cs,pesos=weights.name,
                                fracao_real=entry["fracao_real"]),file)
            print(fraction,arm,{case:(r["agregado"]["fp"],r["agregado"]["fn"]) for case,r in out.items()},flush=True)
        results.append(entry)
        (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    figures = gate/"resultados/figuras"/run_id
    figures.mkdir()
    plot(results,figures/"G5_E09_Proporcao_AWGN.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
