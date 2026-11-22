"""E10.7: robustez pareada em 25 kHz, com referências clássicas ajustadas no treino."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, sosfilt
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from G5_E04_Transferencia import metrics
from G5_E07_CNN1D import load_segments
from G5_E08_Hibrido import fit_head, standardize, statistics
from G5_E09_AWGN import csv_rows, features
from G5_E09_Proporcao_AWGN import evaluate
from G5_E10_AWGN_Inedito import BASE, load_validation
from G5_E10_Perda_B import load_head

CASES = {
    "limpo":"Limpo",
    "lacunas":"Lacunas · 1,28 ms a cada 10,24 ms",
    "preso":"B presa no último valor",
    "saturacao":"B limitada ao P95 pré do treino",
    "passa_baixa":"B filtrada · passa-baixa de 1 kHz",
    "ganho":"Ganhos ABC: +15%, −15%, +10%",
    "atraso":"B atrasada em 1,60 ms",
}


def perturb(x, case, limit, origin):
    out = x.copy()
    if case == "lacunas":
        # Todos os canais perdem amostras juntos. Imputação por último valor observado.
        idx = np.arange(len(x))
        missing = ((idx + 17*origin) % 256) >= 224
        available = np.maximum.accumulate(np.where(~missing,idx,0))
        out = x[available]
    elif case == "preso":
        out[1:,1] = x[0,1]
    elif case == "saturacao":
        out[:,1] = np.clip(x[:,1],-limit,limit)
    elif case == "passa_baixa":
        sos = butter(4,1000,fs=25000,output="sos")
        out[:,1] = sosfilt(sos,x[:,1])
    elif case == "ganho":
        out *= np.array([1.15,.85,1.10],dtype=np.float32)
    elif case == "atraso":
        out[:,1] = x[np.maximum(np.arange(len(x))-40,0),1]
    assert np.isfinite(out).all()
    return out


def figure(results,path):
    img=Image.new("RGB",(1420,760),"white")
    d=ImageDraw.Draw(img)
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",22)
    title=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",33)
    ink="#172638"
    d.text((45,25),"G5/E10.7 | Bateria final de robustez",font=title,fill=ink)
    d.text((45,80),"Omissões / falsos positivos · R03 e R09 · 435 pós e 141 pré",font=font,fill=ink)
    columns=("rms","estatisticas","cnn","hibrido")
    for i,label in enumerate(("Limiar RMS B","Logística estatística","CNN ABC","Híbrido ABC")):
        d.text((545+228*i,152),label,font=font,fill=ink,anchor="mm")
    for i,(case,label) in enumerate(CASES.items()):
        y=187+68*i
        d.text((45,y+23),label,font=font,fill=ink,anchor="lm")
        for j,key in enumerate(columns):
            row=results[case][key]["agregado"]
            x=545+228*j
            total=row["fp"]+row["fn"]
            fill="#e0edf5" if total<20 else "#f3e4d8"
            d.rounded_rectangle((x-102,y,x+102,y+46),radius=8,fill=fill)
            d.text((x,y+23),f'{row["fn"]} / {row["fp"]}',font=font,fill=ink,anchor="mm")
    d.text((45,697),"Decisão forçada em todos os casos. Operação com falha exige verificação da qualidade do sensor.",
           font=font,fill=ink)
    img.save(path)


def run(gate,run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate/"modelos/pesos_pre_treinados"))
    weights=ResNet18_Weights.IMAGENET1K_V1
    encoder=resnet18(weights=weights).eval().requires_grad_(False)
    encoder.fc=nn.Identity()
    checkpoints=gate/"modelos/checkpoints"
    configs={arm:load_head(checkpoints/f"{BASE}_125_{arm}.pt",width,weights)
             for arm,width in (("cnn",512),("hibrido",520))}
    assert all(round(p["fracao_real"]*2305)==288 for _,p in configs.values())
    rows=csv_rows(gate/"dados/manifestos/G5_E03_Manifesto_Canonicos.csv")
    root=gate.parent/"Data and Scripts/Gate4_Exp03_Run01/Originais_Canonicos"
    train,y,_=load_segments(root,rows,"treino",654)
    assert (len(y),int(y.sum()))==(2305,1741)
    s=statistics(train)
    _,_,sm,ss=standardize(s,s)
    classical=fit_head((s-sm)/ss,y)
    threshold=float(torch.quantile(s[y==0,1],.99))
    limit=float(torch.quantile(train[y==0,1].abs().flatten(),.95))
    assert threshold>0 and limit>0
    blocks,target,origins=load_validation(gate)
    dest=gate/"resultados/execucoes"/run_id
    dest.mkdir()
    results={}
    report=dict(fs_Hz=25000,janelas_treino_classicos=2305,janelas_validacao=576,
                checkpoints_CNN_hibrido=BASE,torch=torch.__version__,limiar_sigmoide=.5,
                limiar_rms_B_A=threshold,saturacao_limite_B_A=limit,casos=CASES,resultados=results)
    weights_path=checkpoints/f"{run_id}_estatisticas.pt"
    with weights_path.open("xb") as file:
        torch.save(dict(head=classical.state_dict(),media=sm,desvio=ss,limiar_rms_B=threshold),file)
    for case in CASES:
        windows=[]
        for i,(x,_,starts) in enumerate(blocks):
            z=perturb(x,case,limit,i)
            windows.extend(z[a:a+654].T for a in starts)
        v,emb=features(torch.from_numpy(np.stack(windows)),encoder)
        rms=v[:,1]
        def rate(mask):
            actual=target[mask]==1
            pred=rms[mask]>threshold
            tp,tn,fp,fn=map(int,((pred&actual).sum(),(~pred&~actual).sum(),
                                (pred&~actual).sum(),(~pred&actual).sum()))
            return dict(tp=tp,tn=tn,fp=fp,fn=fn,recall=tp/(tp+fn),fpr=fp/(fp+tn),
                        f1=2*tp/(2*tp+fp+fn),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2)
        entry={"rms":dict(agregado=rate(slice(None)),por_origem={o:rate(origins==o) for o in sorted(set(origins))}),
               "estatisticas":evaluate(classical,(v-sm)/ss,target,origins)}
        for arm,(head,p) in configs.items():
            x=(emb-p["cnn_media"])/p["cnn_desvio"]
            if arm=="hibrido":
                x=torch.cat((x,(v-p["stats_media"])/p["stats_desvio"]),1)
            entry[arm]=evaluate(head,x,target,origins)
        if case=="limpo":
            for arm,fn in (("cnn",3),("hibrido",2)):
                assert (entry[arm]["agregado"]["fn"],entry[arm]["agregado"]["fp"])==(fn,0), arm
        results[case]=entry
        print(case,{k:(v["agregado"]["fn"],v["agregado"]["fp"]) for k,v in entry.items()},flush=True)
        (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    folder=gate/"resultados/figuras"/run_id
    folder.mkdir()
    figure(results,folder/"G5_E10_Bateria_Final.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
