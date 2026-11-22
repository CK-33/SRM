"""E11 integrado: interpreta a CNN e o híbrido fixados no E09."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from G5_E09_AWGN import csv_rows, features
from G5_E09_Proporcao_AWGN import evaluate
from G5_E10_AWGN_Inedito import BASE, load_validation, recortes
from G5_E10_Perda_B import load_head


def preprocess(batch):
    frames=batch[...,39:].unfold(-1,164,41)
    win=torch.hann_window(164,periodic=False)
    spec=torch.fft.rfft(frames*win,n=1024).abs()/win.sum()
    spec[...,1:-1] *= 2
    spec=((20*spec.transpose(-1,-2).clamp_min(1e-3).log10()+60)/100).clamp(0,1)
    spec=F.interpolate(spec,(128,64),mode="bilinear",align_corners=False,antialias=True)
    return (spec-spec.new_tensor([.485,.456,.406])[None,:,None,None])/spec.new_tensor([.229,.224,.225])[None,:,None,None]


def maps(encoder, score, z):
    acts=[]
    hook=encoder.layer4.register_forward_hook(lambda _m,_i,o: acts.append(o))
    try:
        x=z.detach().requires_grad_(True)
        logit=score(encoder(x))
        activation=acts[0]
        gx,ga=torch.autograd.grad(logit.sum(),(x,activation))
        cam=(ga.mean((2,3),keepdim=True)*activation).sum(1,keepdim=True).relu()
        cam=F.interpolate(cam,(128,64),mode="bilinear",align_corners=False)
        return gx.abs().detach(),cam.detach()
    finally:
        hook.remove()


def scale(z):
    return (255*z/(z.max().clamp_min(1e-12))).clamp(0,255).to(torch.uint8).numpy()


def map_image(z):
    gray=scale(z).squeeze()
    return Image.fromarray(np.flipud(gray).copy()).resize((135,235)).convert("RGB")


def draw_maps(items,folder):
    width=1250
    img=Image.new("RGB",(width,100+280*len(items)),"white")
    d=ImageDraw.Draw(img)
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",19)
    b=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",29)
    ink="#172638"
    d.text((32,22),"E11 | Grad-CAM e saliência do logit pós-evento",font=b,fill=ink)
    for j,label in enumerate(("Entrada A","Entrada B","Entrada C","Saliência A","Saliência B","Saliência C","Grad-CAM")):
        d.text((39+173*j,68),label,font=f,fill=ink)
    for i,(label,x,g,cam) in enumerate(items):
        y=98+280*i
        d.text((32,y),label,font=f,fill=ink)
        for j,tile in enumerate([*x,*g,cam]):
            img.paste(map_image(tile),(34+173*j,y+29))
    img.save(folder/"G5_E11_Mapas.png")


def draw_pca(points, labels, origins, ratios, folder):
    image=Image.new("RGB",(920,630),"white")
    d=ImageDraw.Draw(image)
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",20)
    b=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",29)
    ink="#172638"
    d.text((42,24),"E11 | PCA dos embeddings da ResNet-18",font=b,fill=ink)
    d.text((42,69),f"R03 e R09 · PCA ajustada no treino · PC1/PC2: {100*ratios[0]:.1f}% / {100*ratios[1]:.1f}%",font=f,fill=ink)
    lo,hi=points.min(0),points.max(0)
    xy=(points-lo)/np.maximum(hi-lo,1e-12)
    d.rectangle((95,125,845,510),outline="#8496a6")
    for (x,y),label,origin in zip(xy,labels,origins):
        xx,yy=104+730*x,502-366*y
        color="#176ab0" if label==0 else "#ae541c"
        if origin=="R03":
            d.ellipse((xx-3,yy-3,xx+3,yy+3),fill=color)
        else:
            d.rectangle((xx-3,yy-3,xx+3,yy+3),outline=color,width=1)
    d.text((390,520),"PC1",font=f,fill=ink)
    d.text((45,300),"PC2",font=f,fill=ink)
    d.text((95,561),"Azul: pré   ·   Laranja: pós   ·   Círculo: R03   ·   Quadrado: R09",font=f,fill=ink)
    image.save(folder/"G5_E11_PCA.png")


def draw_occlusion(results,folder):
    image=Image.new("RGB",(1170,650),"white")
    d=ImageDraw.Draw(image)
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",19)
    b=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",30)
    ink="#172638"
    d.text((35,25),"E11 | Oclusão da entrada do extrator",font=b,fill=ink)
    d.text((35,73),"Contagens de omissões / falsos positivos em 576 janelas puras",font=f,fill=ink)
    for i,(name,row) in enumerate(results.items()):
        y=137+61*i
        d.text((40,y),name,font=f,fill=ink)
        for j,key in enumerate(("cnn","hibrido")):
            m=row[key]["agregado"]
            x=425+363*j
            d.rounded_rectangle((x,y-7,x+320,y+31),radius=6,fill="#e4edf5" if j==0 else "#f3e8de")
            d.text((x+13,y),f'{key.upper()}  {m["fn"]} / {m["fp"]}',font=f,fill=ink)
    d.text((35,575),"No híbrido, a oclusão altera apenas a entrada da CNN. Estatísticas ficam fixas.",font=f,fill=ink)
    image.save(folder/"G5_E11_Oclusao.png")


def run(gate, run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate/"modelos/pesos_pre_treinados"))
    weights=ResNet18_Weights.IMAGENET1K_V1
    encoder=resnet18(weights=weights).eval().requires_grad_(False)
    encoder.fc=nn.Identity()
    cp=gate/"modelos/checkpoints"
    heads={k:load_head(cp/f"{BASE}_125_{k}.pt",w,weights)
           for k,w in (("cnn",512),("hibrido",520))}
    assert all(round(s["fracao_real"]*2305)==288 for _,s in heads.values())
    blocks,target,origins=load_validation(gate)
    raw,_=recortes(blocks,None,None)
    z=torch.cat([preprocess(x) for x in raw.split(16)])
    # Mesmo operador das avaliações E09/E10. Verificação numérica antes de interpretar.
    s,reference=features(raw,encoder)
    with torch.no_grad():
        emb=torch.cat([encoder(x) for x in z.split(16)])
    torch.testing.assert_close(emb,reference,atol=1e-6,rtol=1e-5)

    def score(key,e,stats):
        head,p=heads[key]
        x=(e-p["cnn_media"])/p["cnn_desvio"]
        if key=="hibrido":
            x=torch.cat((x,(stats-p["stats_media"])/p["stats_desvio"]),1)
        return head(x).flatten()

    interventions={
        "Sem oclusão":None, "A mascarada":(0,slice(None),slice(None)),
        "B mascarada":(1,slice(None),slice(None)), "C mascarada":(2,slice(None),slice(None)),
        "Frequências abaixo de ~2 kHz":(slice(None),slice(0,20),slice(None)),
        "Último quarto temporal":(slice(None),slice(None),slice(48,None)),
        "Primeiro quarto temporal":(slice(None),slice(None),slice(0,16)),
    }
    results={}
    with torch.no_grad():
        for label,idx in interventions.items():
            modified=z.clone() if idx is not None else z
            if idx is not None:
                modified[(slice(None),*idx)]=0
            e=torch.cat([encoder(x) for x in modified.split(16)])
            results[label]={}
            for key,(head,p) in heads.items():
                zz=(e-p["cnn_media"])/p["cnn_desvio"]
                if key=="hibrido":
                    zz=torch.cat((zz,(s-p["stats_media"])/p["stats_desvio"]),1)
                results[label][key]=evaluate(head,zz,target,origins)
    assert results["Sem oclusão"]["cnn"]["agregado"]["fn"]==3
    assert results["Sem oclusão"]["hibrido"]["agregado"]["fn"]==2
    items=[]
    examples=[]
    for origin in ("R03","R09"):
        for label in (0,1):
            idx=np.flatnonzero((origins==origin)&(target.numpy()==label))
            j=int(idx[len(idx)//2])
            for key in ("cnn","hibrido"):
                gradient,cam=maps(encoder,lambda e:score(key,e,s[j:j+1]),z[j:j+1])
                title=f"{origin} · {'pré' if label==0 else 'pós'} · {key}"
                p=z[j:j+1]
                den=p.new_tensor([.229,.224,.225])[None,:,None,None]
                mu=p.new_tensor([.485,.456,.406])[None,:,None,None]
                items.append((title,(p*den+mu)[0].clamp(0,1),gradient[0],cam[0]))
                examples.append(dict(origem=origin,rotulo=label,indice_puro=j,modelo=key,
                                     cam_max=float(cam.max()),
                                     saliencia_media_ABC=gradient.mean((0,2,3)).tolist()))

    # PCA usa apenas embeddings canônicos de treino. A validação é somente projetada.
    from G5_E04_Transferencia import extract
    rows=csv_rows(gate/"dados/manifestos/G5_E03_Manifesto_Canonicos.csv")
    train,_,_=extract(gate,rows,"treino","x164",encoder)
    assert len(train)==2305
    center=train.mean(0)
    _,values,basis=np.linalg.svd((train-center).numpy(),full_matrices=False)
    points=(emb-center).numpy()@basis[:2].T
    ratios=(values[:2]**2/np.sum(values**2)).tolist()
    # Controle negativo: relógio sozinho como atalho, só neste diagnóstico.
    clock_train=[r for r in rows if r["split"]=="treino" and r["par_puro"]=="1"]
    clock_test=sorted((r for r in rows if r["split"]=="validacao" and r["par_puro"]=="1"),
                      key=lambda r:(r["origem"],int(r["fim"])))
    np.testing.assert_array_equal([r["origem"] for r in clock_test],origins)
    np.testing.assert_array_equal([int(r["rotulo_654"]=="pos") for r in clock_test],target.numpy())
    latest_pre=max(float(r["decisao_s"]) for r in clock_train if r["rotulo_654"]=="pre")
    earliest_pos=min(float(r["decisao_s"]) for r in clock_train if r["rotulo_654"]=="pos")
    cutoff=(latest_pre+earliest_pos)/2
    clock_pred=np.array([float(r["decisao_s"])>=cutoff for r in clock_test])
    clock_pos=target.numpy()==1
    shortcut=dict(corte_ajustado_treino_s=cutoff, fp=int(np.sum(clock_pred&~clock_pos)),
                  fn=int(np.sum(~clock_pred&clock_pos)), entrada_modelo=False)
    # Oclusões de bandas e tempo são intervenções artificiais, não falhas de sensor.
    dest=gate/"resultados/execucoes"/run_id
    folder=gate/"resultados/figuras"/run_id
    dest.mkdir()
    folder.mkdir()
    report=dict(referencia=BASE,torch=torch.__version__,fs_Hz=25000,amostras=654,
                pca_variancia=ratios,exemplos=examples,oclusoes=results,controle_relogio=shortcut,
                escala_mapas="Cada painel é escalado pelo próprio máximo. No híbrido as estatísticas ficam fixas.")
    (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    draw_maps(items,folder)
    draw_pca(points,target.numpy(),origins,ratios,folder)
    draw_occlusion(results,folder)
    for case,row in results.items():
        print(case,{k:(v["agregado"]["fn"],v["agregado"]["fp"]) for k,v in row.items()},flush=True)
    print("controle_relogio",shortcut,flush=True)
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
