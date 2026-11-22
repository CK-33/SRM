"""E10.2: AWGN a 200 kHz após retenção e projeção por média de oito."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import ResNet18_Weights, resnet18

from G5_E09_AWGN import csv_rows, features, record
from G5_E09_Proporcao_AWGN import evaluate

BASE = "G5_E09_2_20261122_001522_555"
DOSES = (None, 50, 40, 30, 25, 20)
SEEDS = (801, 802, 803)
ORIGINS = ("R03", "R09")


def load_validation(gate):
    root = gate.parent / "Data and Scripts/Gate4_Exp03_Run01"
    index = {r["id"]:r for r in csv_rows(root/"G4_E03_Manifesto.csv")
             if r["id"] in {"R03_O00","R09_O00"}}
    rows = csv_rows(gate/"dados/manifestos/G5_E03_Manifesto_Canonicos.csv")
    blocks, labels, origins = [], [], []
    for origin in ORIGINS:
        manifest = sorted((r for r in rows if r["origem"]==origin),key=lambda r:int(r["fim"]))
        assert len(manifest)==312 and all(r["split"]=="validacao" for r in manifest)
        pure = [r for r in manifest if r["par_puro"]=="1"]
        marker = int(manifest[0]["evento"])
        x = record(root,origin,"validacao","O00",marker,(index,{}))
        starts = np.array([int(r["inicio_654"]) for r in pure])
        ends = np.array([int(r["fim"]) for r in pure])
        assert all(int(r["evento"])==marker for r in pure)
        np.testing.assert_array_equal(ends-starts+1,654)
        assert np.all((ends<marker)|(ends-980>=marker))
        labels.extend((ends>=marker).astype(int).tolist())
        origins.extend([origin]*len(pure))
        blocks.append((x,marker,starts))
    assert len(labels)==576 and sum(labels)==435
    return blocks,torch.tensor(labels,dtype=torch.float32),np.array(origins)


def recortes(blocks, dose, seed):
    chunks, actual = [], []
    for i,(x,marker,starts) in enumerate(blocks):
        if dose is not None:
            rng = np.random.default_rng(np.random.SeedSequence([seed,i]))
            rms = np.sqrt(np.mean(x[:marker].astype(np.float64)**2,axis=0))
            assert np.all(rms>0)
            # Cada valor de 25 kHz fica retido por oito ticks de 200 kHz.
            # Média de oito ticks e redução por oito: o sinal canônico é preservado.
            noise = rng.standard_normal((len(x),8,3))*rms*10**(-dose/20)
            projected = noise.mean(axis=1)
            high = np.sqrt(np.mean(noise[:marker]**2,axis=(0,1)))
            low = np.sqrt(np.mean(projected[:marker]**2,axis=0))
            actual.append(dict(origem=ORIGINS[i], antes_200kHz_ABC_dB=(20*np.log10(rms/high)).tolist(),
                               depois_25kHz_ABC_dB=(20*np.log10(rms/low)).tolist()))
            x = (x.astype(np.float64)+projected).astype(np.float32)
        chunks.extend(x[a:a+654].T for a in starts)
    return torch.from_numpy(np.stack(chunks)),actual


def chart(rows, path):
    image = Image.new("RGB",(1190,710),"white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",19)
    title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",30)
    ink = "#172638"
    draw.text((40,25),"G5/E10.2 | AWGN pós-retenção a 200 kHz",font=title,fill=ink)
    draw.text((40,75),"R03 e R09 · marcadores: médias · hastes: mínimo e máximo em 3 realizações",font=font,fill=ink)
    colors = {"cnn":"#176ab0","hibrido":"#ae541c"}
    for i,(metric,label) in enumerate((("fn","Omissões, de 435"),("fp","Falsos positivos, de 141"))):
        left,top = 98+580*i,150
        bottom=528
        draw.text((left,top-25),label,font=font,fill=ink)
        ymax=max(1,max(s["agregado"][metric] for row in rows for s in row["modelos"].values()))
        xs=[left+85*j for j in range(len(DOSES))]
        def yy(v):
            return bottom-330*v/ymax
        for step in range(5):
            v=ymax*step/4
            draw.line((left,yy(v),xs[-1],yy(v)),fill="#e1e8ee")
            if ymax>=4 or step in (0,4):
                draw.text((left-12,yy(v)),f"{v:.0f}",anchor="rm",font=font,fill=ink)
        for j,dose in enumerate(DOSES):
            draw.text((xs[j],550),"Limpo" if dose is None else f"{dose} dB",anchor="mt",font=font,fill=ink)
        for arm,color in colors.items():
            points=[]
            for xx,dose in zip(xs,DOSES):
                values=[r["modelos"][arm]["agregado"][metric] for r in rows if r["snr_alvo_dB"]==dose]
                points.append((xx,yy(sum(values)/len(values))))
                draw.line((xx,yy(min(values)),xx,yy(max(values))),fill=color,width=3)
            draw.line(points,fill=color,width=3)
            for xx,y in points:
                draw.ellipse((xx-4,y-4,xx+4,y+4),fill=color)
    draw.text((90,620),"CNN",font=font,fill=colors["cnn"])
    draw.text((255,620),"Híbrido",font=font,fill=colors["hibrido"])
    draw.text((90,656),"Ambos treinados com 288 substituições AWGN. SNR do eixo medida nominalmente antes da conversão.",
              font=font,fill=ink)
    image.save(path)


def run(gate,run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate/"modelos/pesos_pre_treinados"))
    weights=ResNet18_Weights.IMAGENET1K_V1
    encoder=resnet18(weights=weights).eval().requires_grad_(False)
    encoder.fc=nn.Identity()
    configs={}
    for arm,width in (("cnn",512),("hibrido",520)):
        ck=gate/"modelos/checkpoints"/f"{BASE}_125_{arm}.pt"
        p=torch.load(ck,map_location="cpu",weights_only=True)
        assert p["pesos"]==weights.name and round(p["fracao_real"]*2305)==288
        head=nn.Linear(width,1).eval()
        head.load_state_dict(p["head"])
        configs[arm]=(head,p)
    blocks,target,origin=load_validation(gate)
    dest=gate/"resultados/execucoes"/run_id
    dest.mkdir()
    rows=[]
    report=dict(referencia=BASE,substituicoes_awgn_treino=288,fs_Hz=25000, fs_ruido_Hz=200000, operador="media de 8 ticks",
                origem_snr_realizada=list(ORIGINS),sementes=list(SEEDS),torch=torch.__version__,resultados=rows)
    for dose in DOSES:
        for seed in ((None,) if dose is None else SEEDS):
            x,actual=recortes(blocks,dose,seed)
            s,c=features(x,encoder)
            scores={}
            for arm,(head,p) in configs.items():
                z=(c-p["cnn_media"])/p["cnn_desvio"]
                if arm=="hibrido":
                    z=torch.cat((z,(s-p["stats_media"])/p["stats_desvio"]),dim=1)
                scores[arm]=evaluate(head,z,target,origin)
                result=scores[arm]["agregado"]
                if dose is None:
                    assert (result["fp"],result["fn"])==(0,3 if arm=="cnn" else 2), arm
                print(dose,seed,arm,result,flush=True)
            rows.append(dict(snr_alvo_dB=dose,seed=seed,snr_realizada_ABC_dB=actual,modelos=scores))
            (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    folder=gate/"resultados/figuras"/run_id
    folder.mkdir()
    chart(rows,folder/"G5_E10_AWGN_Pos_Retencao.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
