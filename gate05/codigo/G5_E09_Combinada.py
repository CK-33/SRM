"""E09.6: modulação e ganho+modulação+AWGN, com orçamento fixo."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import ResNet18_Weights, resnet18

from G5_E09_AWGN import SPLITS, collect, csv_rows, features, record
from G5_E09_Proporcao_AWGN import evaluate, selection
from G5_E08_Hibrido import fit_head, standardize

CODES = {"modulacao": ("D05","D06","D07","D08"),
         "combinada": ("D13","D14","D15","D16")}


def collect_d(root, rows, split, family, encoder, provenance, chosen=None, fixed=None):
    segments, labels, origins = [], [], []
    meta, params = provenance
    for origin in SPLITS[split]:
        own = sorted((r for r in rows if r["origem"]==origin),key=lambda r:int(r["fim"]))
        assert len(own)==312 and all(r["split"]==split for r in own)
        event = int(own[0]["evento"])
        canonical = record(root,origin,split,"O00",event,provenance) if fixed is None else None
        arrays = {}
        for i,row in enumerate(r for r in own if r["par_puro"]=="1"):
            a,b = int(row["inicio_654"]),int(row["fim"])+1
            assert b-a==654 and int(row["evento"])==event and (b-1<event or b-981>=event)
            assert row["rotulo_654"]==row["rotulo_981"]==("pre" if b-1<event else "pos")
            code = fixed
            if chosen is not None and chosen[len(labels)]:
                code = CODES[family][(i//2)%2*2]
            if code and code not in arrays:
                name = f"{origin}_{code}"
                info,p = meta[name],params[name]
                relative = f"Familia_D_Combinada/{name}.mat"
                assert info["arquivo"].replace("\\","/")==relative
                assert info["grupo"]==split and int(info["registro_origem"])==int(origin[1:])
                assert int(info["fs_Hz"])==25000 and int(info["n_amostras"])==13750
                assert info["t_inicio_s"]==meta[f"{origin}_O00"]["t_inicio_s"]
                assert info["t_evento_s"]==meta[f"{origin}_O00"]["t_evento_s"]
                if family=="modulacao":
                    assert info["metodo"]=="modulacao_suave" and int(p["ruido_modo"])==0
                else:
                    assert info["metodo"]=="ganho_modulacao_AWGN" and int(p["ruido_modo"])==1
                    assert np.isfinite(float(p["snr_dB"]))
                r=loadmat(root/relative,variable_names=["registro"],simplify_cells=True)["registro"]
                assert r["id"]==name and r["Fs_Hz"]==25000
                assert abs(r["t0_s"]-float(info["t_inicio_s"]))<1e-9
                assert int(np.ceil((r["t_evento_s"]-r["t0_s"])*25000-1e-8))==event
                arrays[code]=np.asarray(r["I_A"],dtype=np.float32)
                assert arrays[code].shape==(13750,3) and np.isfinite(arrays[code]).all()
            x = arrays[code] if code else canonical
            segments.append(x[a:b].T)
            labels.append(int(row["rotulo_654"]=="pos"))
            origins.append(origin)
        print(split,fixed or family,origin,flush=True)
    s,c=features(torch.from_numpy(np.stack(segments)),encoder)
    assert torch.isfinite(s).all() and torch.isfinite(c).all()
    return s,c,torch.tensor(labels,dtype=torch.float32),np.array(origins)


def plot(results, path):
    image=Image.new("RGB",(1360,770),"white")
    d=ImageDraw.Draw(image)
    normal=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",21)
    title=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",30)
    ink="#172638"
    d.text((40,27),"G5/E09.6 | Família D versus AWGN",font=title,fill=ink)
    d.text((40,77),"Omissões / falsos positivos · R03 e R09 · 576 janelas por condição",font=normal,fill=ink)
    labels=("Limpo","AWGN 30 dB","Mod. D06","Mod. D08","Comb. D14","Comb. D16")
    cases=("limpo","AWGN30","D06","D08","D14","D16")
    for j,label in enumerate(labels):
        d.text((352+168*j,139),label,anchor="mm",font=normal,fill=ink)
    for i,(mode,arm) in enumerate((m,a) for m in results for a in ("cnn","hibrido")):
        y=170+65*i
        d.text((40,y+22),f"{mode.capitalize()} · {arm}",anchor="lm",font=normal,fill=ink)
        for j,case in enumerate(cases):
            v=results[mode][arm][case]["agregado"]
            x=352+168*j
            d.rounded_rectangle((x-72,y,x+72,y+46),radius=7,
                                fill="#e1edf5" if i%2==0 else "#f0e8df")
            d.text((x,y+23),f'{v["fn"]} / {v["fp"]}',anchor="mm",font=normal,fill=ink)
    d.text((40,717),"Treinos perturbados: as mesmas 288 substituições em 2.305 janelas.",font=normal,fill=ink)
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
    assert torch.equal(y,ya) and np.array_equal(origins,oa) and (len(y),int(y.sum()))==(2305,1741)
    idx=np.concatenate([g[:round(len(g)*.25)] for g in selection(origins,y.numpy())])
    assert len(idx)==len(set(idx))==288
    mask=torch.zeros(len(y),dtype=torch.bool)
    mask[idx]=True
    train={"canonico":(s,c),"awgn":(torch.where(mask[:,None],sa,s),torch.where(mask[:,None],ca,c))}
    val={"limpo":collect(root,rows,"validacao","limpo",encoder,provenance),
         "AWGN30":collect(root,rows,"validacao","30",encoder,provenance)}
    for family,codes in CODES.items():
        ds,dc,dy,do=collect_d(root,rows,"treino",family,encoder,provenance,chosen=mask.numpy())
        assert torch.equal(y,dy) and np.array_equal(origins,do)
        train[family]=(ds,dc)
        for code in codes[1::2]:
            val[code]=collect_d(root,rows,"validacao",family,encoder,provenance,fixed=code)
    for _,_,target,origin in val.values():
        assert (len(target),int(target.sum()))==(576,435)
        assert torch.equal(target,val["limpo"][2]) and np.array_equal(origin,val["limpo"][3])
    results={}
    doses={o:{code:float(provenance[1][f"{o}_{code}"]["snr_dB"]) for code in ("D14","D16")}
           for o in SPLITS["validacao"]}
    report=dict(janelas_treino=2305,substituicoes=288,amostras=654,fs_Hz=25000,
                pesos=weights.name,seed=501,torch=torch.__version__,variantes=CODES,
                snr_dB_nominal_combinada_validacao=doses,resultados=results)
    dest=gate/"resultados/execucoes"/run_id
    dest.mkdir()
    for mode,(s,c) in train.items():
        results[mode]={}
        s,_,sm,ss=standardize(s,s)
        c,_,cm,cs=standardize(c,c)
        for arm in ("cnn","hibrido"):
            head=fit_head(c if arm=="cnn" else torch.cat((c,s),1),y)
            scores={}
            for case,(sv,cv,target,origin) in val.items():
                cv=(cv-cm)/cs
                x=cv if arm=="cnn" else torch.cat((cv,(sv-sm)/ss),1)
                scores[case]=evaluate(head,x,target,origin)
            results[mode][arm]=scores
            with (gate/"modelos/checkpoints"/f"{run_id}_{mode}_{arm}.pt").open("xb") as f:
                torch.save(dict(head=head.state_dict(),stats_media=sm,stats_desvio=ss,
                                cnn_media=cm,cnn_desvio=cs,pesos=weights.name),f)
            print(mode,arm,{k:(v["agregado"]["fp"],v["agregado"]["fn"]) for k,v in scores.items()},flush=True)
            (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    figures=gate/"resultados/figuras"/run_id
    figures.mkdir()
    plot(results,figures/"G5_E09_Combinada_vs_AWGN.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
