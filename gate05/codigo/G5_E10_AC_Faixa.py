"""E10.6: faixa de SNR da alternativa AC em validação exploratória."""
import json
import sys
from pathlib import Path
from statistics import mean

import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import ResNet18_Weights, resnet18

from G5_E09_AWGN import features
from G5_E09_Proporcao_AWGN import evaluate
from G5_E10_AWGN_Inedito import DOSES, SEEDS, load_validation, recortes
from G5_E10_Perda_B import AC_ID, load_head

BASE = "G5_E10_5_20261122_022443_829"


def chart(summary, path):
    img = Image.new("RGB", (1240, 680), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    b = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 31)
    ink = "#172638"
    colors = {"canonico":"#176ab0","awgn":"#ae541c"}
    d.text((40,26),"G5/E10.6 | Faixa de SNR da alternativa AC",font=b,fill=ink)
    d.text((40,76),"R03 e R09 · pontos: médias · hastes: mínimo e máximo observados",font=f,fill=ink)
    for panel,(key,label) in enumerate((("recall","Recall (%)"),("fpr","Taxa de falsos positivos (%)"))):
        left, bottom = 100+590*panel, 510
        xs = [left+85*i for i in range(len(summary))]
        d.text((left,140),label,font=f,fill=ink)
        for rate in range(0,101,25):
            yy=bottom-3*rate
            d.line((left,yy,xs[-1],yy),fill="#dfe7ed")
            d.text((left-15,yy),str(rate),anchor="rm",font=f,fill=ink)
        for arm,color in colors.items():
            points=[]
            for x,row in zip(xs,summary):
                v=row["modelos"][arm][key]
                points.append((x,bottom-300*v["media"]))
                d.line((x,bottom-300*v["max"],x,bottom-300*v["min"]),fill=color,width=4)
            d.line(points,fill=color,width=3)
            for x,y in points:
                d.ellipse((x-4,y-4,x+4,y+4),fill=color)
        for x,row in zip(xs,summary):
            dose=row["snr_alvo_dB"]
            d.text((x,529),"Limpo" if dose is None else f"{dose} dB",anchor="mt",font=f,fill=ink)
    d.text((100,597),"AC canônica",font=f,fill=colors["canonico"])
    d.text((340,597),"AC treinada com AWGN",font=f,fill=colors["awgn"])
    d.text((100,636),"Três realizações por nível. O marcador de condição limpa representa uma avaliação.",font=f,fill=ink)
    img.save(path)


def run(gate, run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate/"modelos/pesos_pre_treinados"))
    weights=ResNet18_Weights.IMAGENET1K_V1
    model=resnet18(weights=weights).eval().requires_grad_(False)
    model.fc=nn.Identity()
    def masked(x):
        return model(x*x.new_tensor([1,0,1])[None,:,None,None])
    folder=gate/"modelos/checkpoints"
    heads={}
    for arm,name in (("canonico",AC_ID),("awgn",f"{BASE}_AC_awgn.pt")):
        head,state=load_head(folder/name,512,weights)
        assert tuple(state["mascara"])==(1,0,1) and state["configuracao"]=="x164"
        if arm=="awgn":
            assert state["substituicoes"]==288
        heads[arm]=head
    ref=json.loads((gate/"resultados/execucoes"/BASE/"metricas.json").read_text(encoding="utf-8"))["resultados"]
    blocks,target,origin=load_validation(gate)
    dest=gate/"resultados/execucoes"/run_id
    dest.mkdir()
    rows,summary=[],[]
    report=dict(referencia=BASE,fs_Hz=25000,sementes=SEEDS,limiar=.5,
                torch=torch.__version__,resultados=rows,resumo=summary)
    for dose in DOSES:
        current=[]
        for seed in ((None,) if dose is None else SEEDS):
            x,actual=recortes(blocks,dose,seed)
            x[:,1]=0
            _,emb=features(x,masked)
            scores={arm:evaluate(head,emb,target,origin) for arm,head in heads.items()}
            if dose in (None,30):
                prior=next(r for r in ref if r["snr_alvo_dB"]==dose and r["seed"]==seed)
                for arm,values in scores.items():
                    assert all(values["agregado"][k]==prior["modelos"][arm]["agregado"][k]
                               for k in ("fp","fn","tp","tn"))
            row=dict(snr_alvo_dB=dose,seed=seed,snr_realizada_ABC_dB=actual,modelos=scores)
            rows.append(row)
            current.append(row)
            print(dose,seed,{k:(v["agregado"]["fn"],v["agregado"]["fp"]) for k,v in scores.items()},flush=True)
        stats={}
        for arm in heads:
            stats[arm]={}
            for metric in ("recall","fpr","fn","fp","logloss"):
                vals=[r["modelos"][arm]["agregado"][metric] for r in current]
                stats[arm][metric]=dict(media=mean(vals),min=min(vals),max=max(vals))
        summary.append(dict(snr_alvo_dB=dose,modelos=stats))
        (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    fig=gate/"resultados/figuras"/run_id
    fig.mkdir()
    chart(summary,fig/"G5_E10_AC_Faixa.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
