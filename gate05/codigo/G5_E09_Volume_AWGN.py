"""E09.7: adição de variantes AWGN versus repetição de janelas canônicas."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from PIL import Image, ImageDraw, ImageFont

from G5_E09_AWGN import DOSES, SPLITS, collect, csv_rows
from G5_E09_Proporcao_AWGN import evaluate, selection
from G5_E08_Hibrido import fit_head, standardize


def plot(results,path):
    image=Image.new("RGB",(1220,690),"white")
    d=ImageDraw.Draw(image)
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",21)
    b=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",30)
    ink="#172638"
    d.text((40,26),"G5/E09.7 | Adicionar AWGN ou repetir janelas limpas",font=b,fill=ink)
    d.text((40,75),"Híbrido · omissões / falsos positivos · R03 e R09",font=f,fill=ink)
    cases=("limpo","50","40","30")
    for j,label in enumerate(("Limpo","50 dB","40 dB","30 dB")):
        d.text((487+192*j,135),label,anchor="mm",font=f,fill=ink)
    for i,entry in enumerate(results):
        y=163+64*i
        name={"base":"Canônico", "repeticao":"Repetição limpa", "awgn":"AWGN adicional"}[entry["modo"]]
        d.text((40,y+21),f'{name} · {entry["treino_total"]} janelas',font=f,fill=ink,anchor="lm")
        for j,case in enumerate(cases):
            s=entry["metricas"][case]["agregado"]
            x=487+192*j
            fill="#e2eef6" if entry["modo"]=="awgn" else "#edf0f3"
            d.rounded_rectangle((x-82,y,x+82,y+42),radius=7,fill=fill)
            d.text((x,y+21),f'{s["fn"]} / {s["fp"]}',font=f,fill=ink,anchor="mm")
    d.text((40,630),"Cada par usa os mesmos índices adicionais e o mesmo orçamento de atualizações.",font=f,fill=ink)
    image.save(path)


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
    assert (len(y),int(y.sum()))==(2305,1741) and torch.equal(y,ya) and np.array_equal(origins,oa)
    val={case:collect(root,rows,"validacao",case,encoder,provenance) for case in ("limpo",*DOSES)}
    assert all((len(v[2]),int(v[2].sum()))==(576,435) for v in val.values())
    assert all(torch.equal(v[2],val["limpo"][2]) and np.array_equal(v[3],val["limpo"][3]) for v in val.values())
    groups=selection(origins,y.numpy())
    dest=gate/"resultados/execucoes"/run_id
    dest.mkdir()
    results=[]
    report=dict(resultados=results,modelo="hibrido",pesos=weights.name,seed=501,epocas=20,
                lote=64,treino_base=2305,fs_Hz=25000,amostras=654,torch=torch.__version__)
    for fraction,count in ((0,0),(.25,288),(.5,576),(1,1152)):
        idx=np.concatenate([g[:round(len(g)*fraction)] for g in groups])
        assert len(idx)==len(set(idx))==count
        for mode in (("base",) if count==0 else ("repeticao","awgn")):
            source_s,source_c=(sa,ca) if mode=="awgn" else (s,c)
            stats=torch.cat((s,source_s[idx]),dim=0)
            cnn=torch.cat((c,source_c[idx]),dim=0)
            labels=torch.cat((y,y[idx]),dim=0)
            stats,_,sm,ss=standardize(stats,stats)
            cnn,_,cm,cs=standardize(cnn,cnn)
            head=fit_head(torch.cat((cnn,stats),dim=1),labels)
            scores={case:evaluate(head,torch.cat(((cv-cm)/cs,(sv-sm)/ss),dim=1),target,origin)
                    for case,(sv,cv,target,origin) in val.items()}
            results.append(dict(modo=mode,treino_total=len(labels),adicionais=count,
                                atualizacoes=20*((len(labels)+63)//64),metricas=scores))
            with (gate/"modelos/checkpoints"/f"{run_id}_{mode}_{count}.pt").open("xb") as f:
                torch.save(dict(head=head.state_dict(),stats_media=sm,stats_desvio=ss,
                                cnn_media=cm,cnn_desvio=cs,pesos=weights.name),f)
            print(mode,count,{case:(v["agregado"]["fp"],v["agregado"]["fn"]) for case,v in scores.items()},flush=True)
            (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    figures=gate/"resultados/figuras"/run_id
    figures.mkdir()
    plot(results,figures/"G5_E09_Volume_AWGN.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
