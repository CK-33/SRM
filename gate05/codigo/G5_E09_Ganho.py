"""E09.3: ablação pareada de AWGN versus ganho independente de sensor."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from PIL import Image, ImageDraw, ImageFont

from G5_E09_AWGN import SPLITS, collect, csv_rows, features, record
from G5_E09_Proporcao_AWGN import evaluate, selection
from G5_E08_Hibrido import fit_head, standardize

GAINS = {"1": ("B07","B08"), "5": ("B09","B10"), "10": ("B11","B12")}


def gain_features(root, rows, split, encoder, provenance, chosen=None, level=None):
    inputs, labels, origins = [], [], []
    meta, params = provenance
    for origin in SPLITS[split]:
        pure = sorted((r for r in rows if r["origem"]==origin and r["par_puro"]=="1"),
                      key=lambda r:int(r["fim"]))
        event = int(pure[0]["evento"])
        signal = record(root,origin,split,"O00",event,provenance)
        for i,row in enumerate(pure):
            start,end = int(row["inicio_654"]),int(row["fim"])
            assert row["split"]==split and int(row["evento"])==event and end-start+1==654
            x = signal[start:end+1]
            if level is not None or (chosen is not None and chosen[len(labels)]):
                dose = level or ("1","5","10")[(i//2)%3]
                assert level is not None or i%2==1
                code = GAINS[dose][1 if level is not None else 0]
                name = f"{origin}_{code}"
                info, p = meta[name], params[name]
                assert info["grupo"]==split and int(info["registro_origem"])==int(origin[1:])
                assert info["metodo"]=="ganho_independente" and int(info["fs_Hz"])==25000
                assert int(p["ganho_modo"])==2 and np.isclose(float(p["ganho_limite"]),int(dose)/100)
                gain = np.array([float(p[f"ganho_{phase}"]) for phase in "ABC"],dtype=np.float32)
                assert np.isfinite(gain).all() and np.all(np.abs(gain-1)<=int(dose)/100+1e-6)
                x = x*gain
            inputs.append(x.T)
            labels.append(int(row["rotulo_654"]=="pos"))
            origins.append(origin)
    stats, cnn = features(torch.from_numpy(np.stack(inputs)),encoder)
    assert torch.isfinite(stats).all() and torch.isfinite(cnn).all()
    return stats,cnn,torch.tensor(labels,dtype=torch.float32),np.array(origins)


def plot(results,path):
    img = Image.new("RGB",(1250,650),"white")
    d = ImageDraw.Draw(img)
    body = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",22)
    title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",32)
    ink = "#182c40"
    d.text((42,28),"G5/E09.3 | AWGN versus ganho independente no treino",font=title,fill=ink)
    d.text((42,79),"Validação R03 e R09 · cada célula: omissões / falsos positivos",font=body,fill=ink)
    cases = ("limpo","G1","G5","G10","AWGN30")
    labels = ("Limpo","Ganho 1%","Ganho 5%","Ganho 10%","AWGN 30 dB")
    for j,label in enumerate(labels):
        d.text((440+163*j,135),label,font=body,fill=ink,anchor="mm")
    for i,(mode,arm) in enumerate((m,a) for m in results for a in ("cnn","hibrido")):
        y=169+68*i
        d.text((42,y+23),f"{mode.capitalize()} · {arm}",font=body,fill=ink,anchor="lm")
        for j,case in enumerate(cases):
            v=results[mode][arm][case]["agregado"]
            x=440+163*j
            color="#d6e8f2" if v["fn"]<=5 and v["fp"]==0 else "#f3e2d5"
            d.rounded_rectangle((x-73,y,x+73,y+48),radius=7,fill=color)
            d.text((x,y+24),f'{v["fn"]} / {v["fp"]}',font=body,fill=ink,anchor="mm")
    d.text((42,610),"Limites nominais de ganho por fase. Treinos mistos usam 288 substituições.",font=body,fill=ink)
    img.save(path)


def run(gate,run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate/"modelos/pesos_pre_treinados"))
    weights=ResNet18_Weights.IMAGENET1K_V1
    encoder=resnet18(weights=weights).eval().requires_grad_(False)
    encoder.fc=nn.Identity()
    root=gate.parent/"Data and Scripts/Gate4_Exp03_Run01"
    rows=csv_rows(gate/"dados/manifestos/G5_E03_Manifesto_Canonicos.csv")
    allowed={o for ids in SPLITS.values() for o in ids}
    assert len(rows)==3120 and {r["origem"] for r in rows}==allowed
    provenance=tuple({r["id"]:r for r in csv_rows(root/f"G4_E03_{name}.csv")
                      if r["id"].split("_")[0] in allowed} for name in ("Manifesto","Parametros"))
    s,c,y,origins=collect(root,rows,"treino","limpo",encoder,provenance)
    sa,ca,ya,oa=collect(root,rows,"treino","misto",encoder,provenance)
    assert torch.equal(y,ya) and np.array_equal(origins,oa)
    indices=np.concatenate([g[:round(len(g)*.25)] for g in selection(origins,y.numpy())])
    assert len(indices)==len(set(indices))==288
    mask=torch.zeros(len(y),dtype=torch.bool)
    mask[indices]=True
    sg,cg,yg,og=gain_features(root,rows,"treino",encoder,provenance,chosen=mask.numpy())
    assert torch.equal(y,yg) and np.array_equal(origins,og)
    train={"canonico":(s,c),"awgn":(torch.where(mask[:,None],sa,s),torch.where(mask[:,None],ca,c)),
           "ganho":(sg,cg)}
    val={"limpo":collect(root,rows,"validacao","limpo",encoder,provenance),
         "AWGN30":collect(root,rows,"validacao","30",encoder,provenance)}
    val.update({f"G{dose}":gain_features(root,rows,"validacao",encoder,provenance,level=dose)
                for dose in GAINS})
    assert (len(y),int(y.sum()))==(2305,1741)
    for _,_,target,origin in val.values():
        assert (len(target),int(target.sum()))==(576,435)
        assert torch.equal(target,val["limpo"][2]) and np.array_equal(origin,val["limpo"][3])
    dest=gate/"resultados/execucoes"/run_id
    dest.mkdir()
    results={}
    report=dict(resultados=results,amostras=654,fs_Hz=25000,seed=501,janelas=2305,
                substituicoes=288,ganhos=GAINS,pesos=weights.name,torch=torch.__version__)
    for mode,(stats,cnn) in train.items():
        results[mode]={}
        stats,_,sm,ss=standardize(stats,stats)
        cnn,_,cm,cs=standardize(cnn,cnn)
        for arm in ("cnn","hibrido"):
            head=fit_head(cnn if arm=="cnn" else torch.cat((cnn,stats),1),y)
            scores={}
            for case,(sv,cv,target,origin) in val.items():
                cv=(cv-cm)/cs
                x=cv if arm=="cnn" else torch.cat((cv,(sv-sm)/ss),1)
                scores[case]=evaluate(head,x,target,origin)
            results[mode][arm]=scores
            with (gate/"modelos/checkpoints"/f"{run_id}_{mode}_{arm}.pt").open("xb") as file:
                torch.save(dict(head=head.state_dict(),stats_media=sm,stats_desvio=ss,
                                cnn_media=cm,cnn_desvio=cs,pesos=weights.name),file)
            print(mode,arm,{k:(v["agregado"]["fp"],v["agregado"]["fn"]) for k,v in scores.items()},flush=True)
            (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    figures=gate/"resultados/figuras"/run_id
    figures.mkdir()
    plot(results,figures/"G5_E09_Ganho_vs_AWGN.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
