"""E10.5: treino AWGN e robustez da alternativa AC sem fase B."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from G5_E06_Fases import fit
from G5_E09_AWGN import SPLITS, collect, csv_rows, features
from G5_E09_Proporcao_AWGN import evaluate, selection
from G5_E10_AWGN_Inedito import load_validation, recortes
from G5_E10_Perda_B import AC_ID, load_head

BASE = "G5_E10_4_20261122_021947_000"
CASES = ((None,None),(30,801),(30,802),(30,803))


def plot(rows,path):
    image=Image.new("RGB",(1150,495),"white")
    d=ImageDraw.Draw(image)
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",23)
    b=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",30)
    ink="#172638"
    d.text((42,26),"G5/E10.5 | Alternativa AC com ruído no treino",font=b,fill=ink)
    d.text((42,78),"Omissões / falsos positivos · R03 e R09 · fase B indisponível",font=f,fill=ink)
    for j,label in enumerate(("Limpo","30 dB · 801","30 dB · 802","30 dB · 803")):
        d.text((480+170*j,150),label,anchor="mm",font=f,fill=ink)
    for i,(arm,label) in enumerate((("canonico","AC canônica"),("awgn","AC treinada com AWGN"))):
        y=194+92*i
        d.text((42,y+29),label,anchor="lm",font=f,fill=ink)
        for j,row in enumerate(rows):
            m=row["modelos"][arm]["agregado"]
            x=480+170*j
            d.rounded_rectangle((x-74,y,x+74,y+57),radius=8,fill="#e6edf4" if i==0 else "#d8ece7")
            d.text((x,y+28),f'{m["fn"]} / {m["fp"]}',anchor="mm",font=b,fill=ink)
    d.text((42,417),"435 janelas pós e 141 pré por realização. Limiar fixo em 0,5.",font=f,fill=ink)
    image.save(path)


def run(gate,run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate/"modelos/pesos_pre_treinados"))
    weights=ResNet18_Weights.IMAGENET1K_V1
    encoder=resnet18(weights=weights).eval().requires_grad_(False)
    encoder.fc=nn.Identity()

    def masked(x):
        return encoder(x*x.new_tensor([1,0,1])[None,:,None,None])

    root=gate.parent/"Data and Scripts/Gate4_Exp03_Run01"
    rows=csv_rows(gate/"dados/manifestos/G5_E03_Manifesto_Canonicos.csv")
    allowed={o for ids in SPLITS.values() for o in ids}
    assert len(rows)==3120 and {r["origem"] for r in rows}==allowed
    provenance=tuple({r["id"]:r for r in csv_rows(root/f"G4_E03_{name}.csv")
                      if r["id"].split("_")[0] in allowed} for name in ("Manifesto","Parametros"))
    _,clean,y,origins=collect(root,rows,"treino","limpo",masked,provenance)
    _,noisy,yn,on=collect(root,rows,"treino","misto",masked,provenance)
    assert (len(y),int(y.sum()))==(2305,1741) and torch.equal(y,yn) and np.array_equal(origins,on)
    indices=np.concatenate([g[:round(len(g)*.25)] for g in selection(origins,y.numpy())])
    assert len(indices)==len(set(indices))==288
    mixed=clean.clone()
    mixed[indices]=noisy[indices]
    canonical,state=load_head(gate/"modelos/checkpoints"/AC_ID,512,weights)
    assert tuple(state["mascara"])==(1,0,1) and state["configuracao"]=="x164"
    awgn=fit(mixed,y)
    blocks,target,origin=load_validation(gate)
    reference=json.loads((gate/"resultados/execucoes"/BASE/"metricas.json").read_text(encoding="utf-8"))["resultados"]
    dest=gate/"resultados/execucoes"/run_id
    dest.mkdir()
    result=[]
    report=dict(referencia=BASE,ac_referencia=AC_ID,pesos=weights.name,treino=2305,
                substituicoes=288,amostras=654,fs_Hz=25000,epocas=20,lote=64,seed=501,
                mascara=[1,0,1],torch=torch.__version__,resultados=result)
    for dose,seed in CASES:
        x,actual=recortes(blocks,dose,seed)
        x[:,1]=0
        _,emb=features(x,masked)
        scores={key:evaluate(head,emb,target,origin)
                for key,head in (("canonico",canonical),("awgn",awgn))}
        prior=next(r for r in reference if r["snr_alvo_dB"]==dose and r["seed"]==seed)
        m=scores["canonico"]["agregado"]
        exp=prior["modelos"]["AC_especializado"]["agregado"]
        assert all(m[key]==exp[key] for key in ("tp","tn","fp","fn"))
        result.append(dict(snr_alvo_dB=dose,seed=seed,snr_realizada_ABC_dB=actual,modelos=scores))
        print(dose,seed,{key:(value["agregado"]["fn"],value["agregado"]["fp"])
                         for key,value in scores.items()},flush=True)
        (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    with (gate/"modelos/checkpoints"/f"{run_id}_AC_awgn.pt").open("xb") as file:
        torch.save(dict(head=awgn.state_dict(),pesos=weights.name,canais="AC",mascara=(1,0,1),
                        configuracao="x164",substituicoes=288),file)
    folder=gate/"resultados/figuras"/run_id
    folder.mkdir()
    plot(result,folder/"G5_E10_AC_AWGN.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
