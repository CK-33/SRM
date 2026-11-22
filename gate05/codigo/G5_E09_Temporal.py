"""E09.5: ablação de deformação temporal a 25 kHz no suporte comum."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from PIL import Image, ImageDraw, ImageFont

from G5_E09_AWGN import SPLITS, csv_rows, features, record
from G5_E09_Proporcao_AWGN import evaluate
from G5_E08_Hibrido import fit_head, standardize
from G5_E09_Auditar_Temporal import audit

TRAIN_CODES = ("C01","C02","C03","C04","C05","C06","C07","C08")
VAL_CASES = ("limpo","C07","C08")


def temporal(root, origin, code, meta, entry):
    name = f"{origin}_{code}"
    info = meta[name]
    assert info["arquivo"].replace("\\","/")==f"Familia_C_Temporal/{name}.mat"
    r = loadmat(root/"Familia_C_Temporal"/f"{name}.mat",variable_names=["registro"],
                simplify_cells=True)["registro"]
    assert r["id"]==name and r["Fs_Hz"]==25000 and entry["split"]==info["grupo"]
    for field,column in (("t0_s","t_inicio_s"),("tf_s","t_fim_s"),("t_evento_s","t_evento_s")):
        assert abs(r[field]-float(info[column]))<1e-9, (name,field)
    x = np.asarray(r["I_A"],dtype=np.float32)
    assert x.shape==(int(info["n_amostras"]),3) and np.isfinite(x).all()
    return x


def residual(clean, warped, offset, marker, alpha):
    """Compara no relógio de saída, com e sem compensação; exclui bordas e evento."""
    grid = offset+np.arange(len(warped))
    query = marker+alpha*(grid-marker)
    margin = 160
    keep = ((grid>=offset+margin) & (grid<offset+len(warped)-margin)
            & (query>=margin) & (query<len(clean)-margin) & (abs(grid-marker)>=margin))
    assert keep.any()
    q = query[keep]
    transformed = np.stack([np.interp(q,np.arange(len(clean)),clean[:,k]) for k in range(3)],axis=1)
    def nrmse(ref):
        return (100*np.sqrt(np.mean((warped[keep]-ref)**2,axis=0)) /
                np.sqrt(np.mean(ref**2,axis=0)).clip(1e-12)).tolist()
    return dict(amostras=int(keep.sum()), excluidas=int((~keep).sum()),
                nrmse_sem_compensacao=nrmse(clean[grid[keep]]),
                nrmse_compensado=nrmse(transformed))


def prepare(root, clocks, split, provenance, audited, encoder, reports):
    cases = ("canonico","temporal") if split=="treino" else VAL_CASES
    segments = {case:[] for case in cases}
    labels, origins = [], []
    for origin in SPLITS[split]:
        own = sorted((r for r in clocks if r["origem"]==origin),key=lambda r:int(r["fim"]))
        assert len(own)==312 and all(r["split"]==split for r in own)
        event = int(own[0]["evento"])
        entries = {code:audited[f"{origin}_{code}"] for code in TRAIN_CODES}
        start = max(r["recorte_inicio"] for r in entries.values())
        end = 13749-max(r["recorte_fim"] for r in entries.values())
        pure = [r for r in own if r["par_puro"]=="1" and
                int(r["inicio_654"])>=start and int(r["fim"])<=end]
        clean = record(root,origin,split,"O00",event,provenance)
        codes = TRAIN_CODES if split=="treino" else VAL_CASES[1:]
        warped = {code:temporal(root,origin,code,provenance[0],entries[code]) for code in codes}
        base = provenance[0][f"{origin}_O00"]
        marker = (float(base["t_evento_s"])-float(base["t_inicio_s"]))*25000
        for code,x in warped.items():
            entry=entries[code]
            reports[entry["id"]]=residual(clean,x,entry["recorte_inicio"],marker,entry["alpha"])
        chosen={}
        if split=="treino":
            # 9 pré e 27 pós por origem, distribuídos pelo trecho disponível.
            for label,count in (("pre",9),("pos",27)):
                eligible=[i for i,r in enumerate(pure) if r["rotulo_654"]==label]
                assert len(eligible)>=count
                chosen.update({eligible[int((j+.5)*len(eligible)/count)]:TRAIN_CODES[(j+len(origins))%8]
                               for j in range(count)})
            assert len(chosen)==36
        for i,r in enumerate(pure):
            a,b=int(r["inicio_654"]),int(r["fim"])+1
            assert b-a==654 and r["rotulo_654"]==r["rotulo_981"] in ("pre","pos")
            assert r["rotulo_654"]==("pre" if b-1<event else "pos") and (b-1<event or b-981>=event)
            for case in cases:
                code = chosen.get(i) if case=="temporal" else case if case in warped else None
                if code:
                    offset=entries[code]["recorte_inicio"]
                    x=warped[code][a-offset:b-offset]
                else:
                    x=clean[a:b]
                assert x.shape==(654,3)
                segments[case].append(x.T)
            labels.append(int(r["rotulo_654"]=="pos"))
            origins.append(origin)
        print(split,origin,len(pure),"janelas puras no suporte comum",flush=True)
    y,groups=torch.tensor(labels,dtype=torch.float32),np.array(origins)
    out={}
    for case,items in segments.items():
        s,c=features(torch.from_numpy(np.stack(items)),encoder)
        assert torch.isfinite(s).all() and torch.isfinite(c).all()
        out[case]=(s,c)
    return out,y,groups


def plot(report, path):
    img=Image.new("RGB",(1060,590),"white")
    d=ImageDraw.Draw(img)
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",21)
    b=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",29)
    ink="#172638"
    d.text((40,25),"G5/E09.5 | Ablação da deformação temporal",font=b,fill=ink)
    d.text((40,75),"Validação R03 e R09 · omissões / falsos positivos no suporte comum",font=f,fill=ink)
    for j,label in enumerate(("Limpo","C07 · α=0,98","C08 · α=1,02")):
        d.text((440+205*j,150),label,anchor="mm",font=f,fill=ink)
    for i,(mode,arm) in enumerate((m,a) for m in report["resultados"] for a in ("cnn","hibrido")):
        y=195+73*i
        d.text((40,y+23),f"{mode.capitalize()} · {arm}",anchor="lm",font=f,fill=ink)
        for j,case in enumerate(VAL_CASES):
            score=report["resultados"][mode][arm][case]["agregado"]
            x=440+205*j
            d.rounded_rectangle((x-83,y,x+83,y+48),radius=7,fill="#e5eff7" if mode=="canonico" else "#f3e8dd")
            d.text((x,y+24),f'{score["fn"]} / {score["fp"]}',anchor="mm",font=b,fill=ink)
    d.text((40,531),f'{report["janelas_treino"]} janelas por treino · 288 substituições na condição temporal',
           font=f,fill=ink)
    img.save(path)


def run(gate,run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate/"modelos/pesos_pre_treinados"))
    weights=ResNet18_Weights.IMAGENET1K_V1
    encoder=resnet18(weights=weights).eval().requires_grad_(False)
    encoder.fc=nn.Identity()
    root=gate.parent/"Data and Scripts/Gate4_Exp03_Run01"
    clocks=csv_rows(gate/"dados/manifestos/G5_E03_Manifesto_Canonicos.csv")
    allowed={o for ids in SPLITS.values() for o in ids}
    assert len(clocks)==3120 and {r["origem"] for r in clocks}==allowed
    provenance=tuple({r["id"]:r for r in csv_rows(root/f"G4_E03_{name}.csv")
                      if r["id"].split("_")[0] in allowed} for name in ("Manifesto","Parametros"))
    audited={r["id"]:r for r in audit(gate)}
    assert len(audited)==80
    residuals={}
    train,y,_=prepare(root,clocks,"treino",provenance,audited,encoder,residuals)
    val,target,origins=prepare(root,clocks,"validacao",provenance,audited,encoder,residuals)
    dest=gate/"resultados/execucoes"/run_id
    dest.mkdir()
    report=dict(janelas_treino=len(y),janelas_validacao=len(target),substituicoes=288,
                treinos_pos=int(y.sum()),validacao_pos=int(target.sum()),seed=501,
                fs_Hz=25000,amostras=654,pesos=weights.name,torch=torch.__version__,
                margem_residuo_amostras=160,residuos=residuals,resultados={})
    for mode,(s,c) in train.items():
        report["resultados"][mode]={}
        s,_,sm,ss=standardize(s,s)
        c,_,cm,cs=standardize(c,c)
        for arm in ("cnn","hibrido"):
            head=fit_head(c if arm=="cnn" else torch.cat((c,s),1),y)
            scores={}
            for case,(sv,cv) in val.items():
                cv=(cv-cm)/cs
                x=cv if arm=="cnn" else torch.cat((cv,(sv-sm)/ss),1)
                scores[case]=evaluate(head,x,target,origins)
            report["resultados"][mode][arm]=scores
            with (gate/"modelos/checkpoints"/f"{run_id}_{mode}_{arm}.pt").open("xb") as file:
                torch.save(dict(head=head.state_dict(),stats_media=sm,stats_desvio=ss,
                                cnn_media=cm,cnn_desvio=cs,pesos=weights.name),file)
            print(mode,arm,{k:(v["agregado"]["fp"],v["agregado"]["fn"]) for k,v in scores.items()},flush=True)
            (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    folder=gate/"resultados/figuras"/run_id
    folder.mkdir()
    plot(report,folder/"G5_E09_Abla_Temporal.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
