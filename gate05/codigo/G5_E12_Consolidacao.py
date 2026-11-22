"""E12: replay causal em CPU e comparação operacional dos candidatos fixados."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from G5_E07_CNN1D import model_1d
from G5_E09_AWGN import csv_rows, features, record
from G5_E10_AWGN_Inedito import BASE, ORIGINS, SEEDS
from G5_E10_Perda_B import load_head

CLASSICAL = "G5_E10_7_20261122_024732_045_estatisticas.pt"
TIME_MODEL = "G5_E07_3_20261121_235336_565_654.pt"
FS = 25000
METHODS = ("rms", "estatisticas", "cnn", "hibrido", "cnn1d")


def replay_metrics(scores, ends, marker, persistence, threshold, strict=False):
    scores = np.asarray(scores)
    above = scores > threshold if strict else scores >= threshold
    streak = 0
    active = np.zeros(len(above), dtype=bool)
    for i, decision in enumerate(above):
        streak = streak + 1 if decision else 0
        active[i] = streak >= persistence
    pre = ends < marker
    post = ~pre
    pure = ends - 653 >= marker
    episodes = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    after = episodes[ends[episodes] >= marker]
    # Um episódio iniciado antes do marcador continua sendo falso alarme.
    delay = float(1000*(ends[after[0]]-marker)/FS) if len(after) else None
    return dict(pre=int(pre.sum()), pos_puro=int(pure.sum()), cruzado=int((post&~pure).sum()),
                positivos_pre=int((above&pre).sum()), omissoes_pos_puro=int((~above&pure).sum()),
                episodios_falsos=int(np.count_nonzero(ends[episodes] < marker)),
                primeiro_novo_alarme_ms=delay,
                ativo_ao_cruzar=bool(active[np.flatnonzero(post)[0]]),
                inicio_episodios_amostra=ends[episodes].tolist())


def plot(traces, folder):
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",20)
    b=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",28)
    ink="#172638"
    colors={"rms":"#666666","estatisticas":"#8452a0","cnn":"#176ab0",
            "hibrido":"#ae541c","cnn1d":"#397450"}
    for origin in ORIGINS:
        image=Image.new("RGB",(1250,650),"white")
        d=ImageDraw.Draw(image)
        d.text((40,25),f"E12 | Replay causal · {origin} · condição limpa",font=b,fill=ink)
        d.text((40,70),"Escores dos quatro modelos aprendidos, com limiar fixo de 0,5",font=f,fill=ink)
        r=next(row for row in traces if row["origem"]==origin and row["seed"] is None)
        t=1000*(np.asarray(r["fim"])-r["evento_indice"])/FS
        keep=(t>=-60)&(t<=120)
        xx=lambda v: 95+(v+60)*1085/180
        yy=lambda v: 520-360*v
        for level in (0,.25,.5,.75,1):
            d.line((95,yy(level),1180,yy(level)),fill="#9caebf" if level==.5 else "#dfe8ee")
            d.text((48,yy(level)-11),str(level),font=f,fill=ink)
        d.line((xx(0),160,xx(0),520),fill=ink,width=2)
        for k in ("estatisticas","cnn","hibrido","cnn1d"):
            z=np.array(r["escores"][k])[keep]
            p=1/(1+np.exp(-np.clip(z,-80,80)))
            d.line([(xx(a),yy(v)) for a,v in zip(t[keep],p)],fill=colors[k],width=3)
        for tick in (-60,-30,0,30,60,90,120):
            d.text((xx(tick),533),str(tick),anchor="mt",font=f,fill=ink)
        d.text((460,565),"Tempo relativo ao marcador (ms)",font=f,fill=ink)
        for i,k in enumerate(("estatisticas","cnn","hibrido","cnn1d")):
            d.text((105+245*i,608),k,font=f,fill=colors[k])
        image.save(folder/f"G5_E12_Replay_{origin}.png")


def plot_summary(traces, costs, folder):
    image=Image.new("RGB",(1160,620),"white")
    d=ImageDraw.Draw(image)
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",21)
    b=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",30)
    ink="#172638"
    d.text((40,24),"E12 | Custo e alarmes por método",font=b,fill=ink)
    d.text((40,73),"Persistência de três decisões · R03 e R09",font=f,fill=ink)
    headers=("Método","CPU p50 / p95 (ms)","Alarmes falsos","Atraso no sinal¹ (ms)","Sem novo alarme")
    xs=(42,245,480,670,973)
    for x,label in zip(xs,headers):
        d.text((x,145),label,font=f,fill=ink)
    for i,k in enumerate(METHODS):
        yy=190+64*i
        d.rectangle((30,yy-5,1130,yy+44),fill="#eef3f8" if i%2==0 else "#e1ebf3")
        m=[r["decisoes"][k]["3"] for r in traces]
        delays=[r["primeiro_novo_alarme_ms"] for r in m if r["primeiro_novo_alarme_ms"] is not None]
        values=(k,f'{costs[k]["p50_ms"]:.2f} / {costs[k]["p95_ms"]:.2f}',
                str(sum(r["episodios_falsos"] for r in m)),
                f'{min(delays):.2f} a {max(delays):.2f}' if delays else "--",str(len(m)-len(delays)))
        for x,value in zip(xs,values):
            d.text((x,yy+7),value,font=f,fill=ink)
    d.text((40,535),"¹ Relógio do sinal, do marcador ao início do primeiro episódio pós-evento.",font=f,fill=ink)
    d.text((40,570),"Limpo e três realizações a 30 dB por origem. CPU com lote unitário.",font=f,fill=ink)
    image.save(folder/"G5_E12_Custo_Alarmes.png")


@torch.inference_mode()
def run(gate, run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate/"modelos/pesos_pre_treinados"))
    w=ResNet18_Weights.IMAGENET1K_V1
    encoder=resnet18(weights=w).eval().requires_grad_(False)
    encoder.fc=nn.Identity()
    cp=gate/"modelos/checkpoints"
    heads={k:load_head(cp/f"{BASE}_125_{k}.pt",n,w) for k,n in (("cnn",512),("hibrido",520))}
    assert all(round(p["fracao_real"]*2305)==288 for _,p in heads.values())
    p=torch.load(cp/CLASSICAL,map_location="cpu",weights_only=True)
    stat_head=nn.Linear(8,1).eval()
    stat_head.load_state_dict(p["head"])
    q=torch.load(cp/TIME_MODEL,map_location="cpu",weights_only=True)
    assert q["amostras"]==654
    time_model=model_1d().eval()
    time_model.load_state_dict(q["modelo"])
    from G5_E08_Hibrido import statistics

    def score(k,x):
        if k=="rms":
            return x[:,1].square().mean(1).sqrt()
        if k=="estatisticas":
            return stat_head((statistics(x)-p["media"])/p["desvio"]).flatten()
        if k=="cnn1d":
            return time_model((x-q["media"])/q["desvio"]).flatten()
        s,e=features(x,encoder)
        return spectral(k,s,e)

    def spectral(k,s,e):
        head,st=heads[k]
        z=(e-st["cnn_media"])/st["cnn_desvio"]
        if k=="hibrido":
            z=torch.cat((z,(s-st["stats_media"])/st["stats_desvio"]),1)
        return head(z).flatten()

    root=gate.parent/"Data and Scripts/Gate4_Exp03_Run01"
    index={r["id"]:r for r in csv_rows(root/"G4_E03_Manifesto.csv")
           if r["id"] in {"R03_O00","R09_O00"}}
    rows=csv_rows(gate/"dados/manifestos/G5_E03_Manifesto_Canonicos.csv")
    traces=[]
    sample=None
    for i,origin in enumerate(ORIGINS):
        own=sorted((r for r in rows if r["origem"]==origin),key=lambda r:int(r["fim"]))
        assert len(own)==312 and all(r["split"]=="validacao" for r in own)
        ends=np.array([int(r["fim"]) for r in own])
        np.testing.assert_array_equal(ends,np.arange(980,13750,41))
        marker=int(own[0]["evento"])
        assert all(int(r["evento"])==marker and int(r["inicio_654"])==int(r["fim"])-653 for r in own)
        canonical=record(root,origin,"validacao","O00",marker,(index,{}))
        for seed in (None,*SEEDS):
            x=canonical
            if seed is not None:
                rng=np.random.default_rng(np.random.SeedSequence([seed,i]))
                rms=np.sqrt(np.mean(x[:marker].astype(np.float64)**2,axis=0))
                noise=(rng.standard_normal(x.shape)*rms*10**(-30/20)).astype(np.float32)
                x=x+noise
            windows=torch.from_numpy(np.stack([x[e-653:e+1].T for e in ends]))
            assert windows.shape==(312,3,654)
            if sample is None:
                sample=windows[len(windows)//2:len(windows)//2+1].contiguous()
            s,e=features(windows,encoder)
            scores={k:(spectral(k,s,e) if k in heads else score(k,windows)).numpy() for k in METHODS}
            assert all(np.isfinite(v).all() and len(v)==312 for v in scores.values())
            decisions={k:{str(m):replay_metrics(v,ends,marker,m,
                       p["limiar_rms_B"] if k=="rms" else 0., strict=k=="rms")
                       for m in (1,3)} for k,v in scores.items()}
            traces.append(dict(origem=origin,seed=seed,evento_indice=marker,fim=ends.tolist(),
                               escores={k:v.tolist() for k,v in scores.items()},decisoes=decisions))
            print(origin,seed,{k:decisions[k]["3"]["primeiro_novo_alarme_ms"] for k in METHODS},flush=True)
    costs={}
    for k in METHODS:
        for _ in range(5):
            score(k,sample)
        timings=[]
        for _ in range(40):
            start=time.perf_counter_ns()
            score(k,sample)
            timings.append((time.perf_counter_ns()-start)/1e6)
        costs[k]=dict(p50_ms=float(np.median(timings)),p95_ms=float(np.quantile(timings,.95)),
                      excedeu_salto=sum(x>1.64 for x in timings),repeticoes=len(timings))
    dest=gate/"resultados/execucoes"/run_id
    folder=gate/"resultados/figuras"/run_id
    dest.mkdir()
    folder.mkdir()
    (dest/"metricas.json").write_text(json.dumps(dict(fs_Hz=FS,salto=41,tamanho=654,
        limiar_rms_B_A=p["limiar_rms_B"],referencias=dict(cnn_hibrido=BASE,estatisticas=CLASSICAL,cnn1d=TIME_MODEL),
        custo_CPU=costs,sequencias=traces,torch=torch.__version__,threads=4),indent=2,allow_nan=False),encoding="utf-8")
    plot(traces,folder)
    plot_summary(traces,costs,folder)
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
