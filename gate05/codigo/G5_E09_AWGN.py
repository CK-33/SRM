"""Ablação com orçamento fixo de janelas e AWGN existente a 25 kHz."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from PIL import Image, ImageDraw, ImageFont

from G5_E04_Transferencia import SPLITS, metrics
from G5_E08_Hibrido import fit_head, standardize, statistics

DOSES = {"50": ("A01","A02"), "40": ("A03","A04"), "30": ("A05","A06")}


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def record(root, origin, split, variant, event, provenance, dose=None):
    folder = "Originais_Canonicos" if variant=="O00" else "Familia_A_AWGN"
    name = f"{origin}_{variant}"
    meta, params = provenance
    info = meta[name]
    assert info["grupo"] == split and int(info["registro_origem"]) == int(origin[1:])
    assert int(info["fs_Hz"]) == 25000 and int(info["n_amostras"]) == 13750
    assert info["arquivo"].replace("\\","/") == f"{folder}/{name}.mat"
    if dose is not None:
        assert float(params[name]["snr_dB"]) == float(dose)
        assert int(params[name]["ruido_modo"]) == 1
    path = root / folder / f"{name}.mat"
    r = loadmat(path, variable_names=["registro"], simplify_cells=True)["registro"]
    assert r["id"] == name and r["Fs_Hz"] == 25000
    assert int(np.ceil((r["t_evento_s"]-r["t0_s"])*25000-1e-8)) == event
    x = np.asarray(r["I_A"],dtype=np.float32)
    assert x.shape == (13750,3) and np.isfinite(x).all()
    return x


def features(x, encoder):
    cnn = []
    with torch.no_grad():
        for batch in x.split(16):
            frames = batch[..., 39:].unfold(-1,164,41)
            assert frames.shape[-2] == 12
            window = torch.hann_window(164,periodic=False)
            spec = torch.fft.rfft(frames*window,n=1024).abs()/window.sum()
            spec[...,1:-1] *= 2
            spec = ((20*spec.transpose(-1,-2).clamp_min(1e-3).log10()+60)/100).clamp(0,1)
            spec = F.interpolate(spec,(128,64),mode="bilinear",align_corners=False,antialias=True)
            mean = spec.new_tensor([.485,.456,.406])[None,:,None,None]
            std = spec.new_tensor([.229,.224,.225])[None,:,None,None]
            cnn.append(encoder((spec-mean)/std))
    return statistics(x),torch.cat(cnn)


def collect(root, rows, split, condition, encoder, provenance):
    inputs, labels, origins = [], [], []
    for origin in SPLITS[split]:
        m = sorted((r for r in rows if r["origem"]==origin),key=lambda r:int(r["fim"]))
        assert len(m)==312 and all(r["split"]==split for r in m)
        np.testing.assert_array_equal([int(r["fim"]) for r in m],np.arange(980,13750,41))
        event = int(m[0]["evento"])
        assert all(int(r["evento"])==event for r in m)
        pure = [r for r in m if r["par_puro"]=="1"]
        signals = {}
        for i,r in enumerate(pure):
            end = int(r["fim"])
            start = int(r["inicio_654"])
            assert start==end-653 and r["rotulo_654"]==r["rotulo_981"]
            assert end<event or end-980>=event
            assert r["rotulo_654"]==("pre" if end<event else "pos")
            dose = None
            if condition=="misto" and i%2:
                dose = ("50","40","30")[(i//2)%3]
                variant = DOSES[dose][0]
            elif condition in DOSES:
                dose = condition
                variant = DOSES[dose][1]
            else:
                variant = "O00"
            if variant not in signals:
                signals[variant] = record(root,origin,split,variant,event,provenance,dose)
            inputs.append(signals[variant][start:end+1].T)
            labels.append(int(r["rotulo_654"]=="pos"))
            origins.append(origin)
        print(split,condition,origin,len(pure),flush=True)
    s,c = features(torch.from_numpy(np.stack(inputs)),encoder)
    assert torch.isfinite(s).all() and torch.isfinite(c).all()
    return s,c,torch.tensor(labels,dtype=torch.float32),np.array(origins)


def plot(result, path):
    image = Image.new("RGB",(1160,660),"white")
    d = ImageDraw.Draw(image)
    f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",20)
    title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",30)
    ink = "#172638"
    d.text((40,28),"G5/E09.1 | AWGN a 25 kHz com orçamento fixo",font=title,fill=ink)
    d.text((40,76),"Validação R03 e R09 · taxas em porcentagem",font=f,fill=ink)
    colors = {"limpo":"#176ab0","misto":"#ae541c"}
    conditions = ("limpo","50","40","30")
    for row,metric in enumerate(("recall","fpr")):
        for col,arm in enumerate(("estatisticas","cnn","hibrido")):
            x0,y0 = 60+380*col, 125+236*row
            d.text((x0,y0),f"{arm} · {metric.upper()}",font=f,fill=ink)
            bottom = y0+158
            for rate in (0,50,100):
                yy=bottom-rate
                d.line((x0,yy,x0+285,yy),fill="#dbe3ec")
                d.text((x0+290,yy-10),str(rate),font=f,fill=ink)
            for mode,color in colors.items():
                values = [result[mode][arm][c]["agregado"][metric]*100 for c in conditions]
                points = [(x0+18+83*i,bottom-v) for i,v in enumerate(values)]
                d.line(points,fill=color,width=3)
                for x,y in points:
                    d.ellipse((x-4,y-4,x+4,y+4),fill=color)
            for i,c in enumerate(("Limpo","50 dB","40 dB","30 dB")):
                d.text((x0+18+83*i,bottom+14),c,anchor="mt",font=f,fill=ink)
    d.text((110,610),"Treino canônico",font=f,fill=colors["limpo"])
    d.text((390,610),"Treino misto com AWGN",font=f,fill=colors["misto"])
    image.save(path)


def run(gate, run_id):
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
    val={case:collect(root,rows,"validacao",case,encoder,provenance) for case in ("limpo",*DOSES)}
    train={case:collect(root,rows,"treino",case,encoder,provenance) for case in ("limpo","misto")}
    assert torch.equal(train["limpo"][2],train["misto"][2])
    assert all(torch.equal(val["limpo"][2],block[2]) for block in val.values())
    y=train["limpo"][2]
    t=val["limpo"][2]
    assert (len(y),int(y.sum()),len(t),int(t.sum()))==(2305,1741,576,435)
    dest=gate/"resultados/execucoes"/run_id
    dest.mkdir()
    results={}
    report=dict(seed=501,pesos=weights.name,doses=DOSES,amostras=654,fs_Hz=25000,
                torch=torch.__version__,epocas=20,treino_janelas_por_braco=2305,resultados=results)
    for mode,(s,c,y,_) in train.items():
        results[mode]={}
        s,_,sm,ss=standardize(s,s)
        c,_,cm,cs=standardize(c,c)
        for arm,x in (("estatisticas",s),("cnn",c),("hibrido",torch.cat((c,s),1))):
            head=fit_head(x,y)
            results[mode][arm]={}
            for case,(sv,cv,target,origins) in val.items():
                sv,cv=(sv-sm)/ss,(cv-cm)/cs
                inputs={"estatisticas":sv,"cnn":cv,"hibrido":torch.cat((cv,sv),1)}[arm]
                with torch.no_grad():
                    logits=head(inputs).flatten()
                results[mode][arm][case]=dict(agregado=metrics(logits,target),
                    por_origem={o:metrics(logits[origins==o],target[origins==o]) for o in SPLITS["validacao"]})
            with (gate/"modelos/checkpoints"/f"{run_id}_{mode}_{arm}.pt").open("xb") as file:
                torch.save(dict(head=head.state_dict(),stats_media=sm,stats_desvio=ss,
                                cnn_media=cm,cnn_desvio=cs,pesos=weights.name),file)
            (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    figures=gate/"resultados/figuras"/run_id
    figures.mkdir()
    plot(results,figures/"G5_E09_AWGN_Recall_FPR.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
