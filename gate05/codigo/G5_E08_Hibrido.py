"""E08.1: comparação pareada entre estatísticas, ResNet-18 e híbrido."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from G5_E04_Transferencia import SPLITS, extract, metrics
from G5_E07_CNN1D import load_segments

NAMES = ["RMS_A", "RMS_B", "RMS_C", "crista_A", "crista_B", "crista_C",
         "RMS_trifasico", "dispersao_RMS_pct"]


def statistics(x):
    rms = x.square().mean(dim=2).sqrt()
    crest = x.abs().amax(dim=2) / rms.clamp_min(1e-8)
    total = rms.square().mean(dim=1, keepdim=True).sqrt()
    dispersion = 100*rms.std(dim=1, keepdim=True, correction=1) / rms.mean(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.cat((rms, crest, total, dispersion), dim=1)


def standardize(train, val):
    mean = train.mean(dim=0)
    std = train.std(dim=0, correction=0).clamp_min(1e-6)
    return (train-mean)/std, (val-mean)/std, mean, std


def fit_head(x,y):
    # Mesmo otimizador e número de atualizações do E06, com dimensão variável.
    torch.manual_seed(501)
    model = nn.Linear(x.shape[1],1)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(564/1741))
    for _ in range(20):
        for idx in torch.randperm(len(y)).split(64):
            opt.zero_grad()
            value = loss(model(x[idx]).flatten(),y[idx])
            assert torch.isfinite(value)
            value.backward()
            opt.step()
    return model.eval()


def plot(results, path):
    image = Image.new("RGB",(1150,645),"white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",20)
    title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",30)
    ink = "#172638"
    draw.text((40,25),"G5/E08.1 | Estatísticas, CNN e modelo híbrido",font=title,fill=ink)
    draw.text((40,76),"Validação canônica · dois ciclos · R03 e R09",font=font,fill=ink)
    aliases = {"estatisticas":"Estatísticas", "cnn":"CNN", "hibrido":"CNN + estatísticas"}
    colors = {"estatisticas":"#397450", "cnn":"#216ba4", "hibrido":"#ae541c"}
    largest = max(m["logloss"] for r in results.values()
                  for m in [r["agregado"],*r["por_origem"].values()]) or 1
    for i, scope in enumerate(("agregado","R03","R09")):
        top = 130 + 150*i
        draw.text((40,top),scope.capitalize(),font=title,fill=ink)
        for j,(key,r) in enumerate(results.items()):
            m = r["agregado"] if scope=="agregado" else r["por_origem"][scope]
            y = top+39+31*j
            draw.text((40,y),aliases[key],font=font,fill=ink)
            width = 480*m["logloss"]/largest
            draw.rectangle((247,y,247+width,y+20),fill=colors[key])
            draw.text((740,y),f"Logloss {m['logloss']:.5f}   |   FP {m['fp']}   FN {m['fn']}",font=font,fill=ink)
    draw.text((40,598),"As barras mostram logloss em escala comum. Menor é melhor.",font=font,fill=ink)
    image.save(path)


def run(gate, run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate/"modelos/pesos_pre_treinados"))
    weights = ResNet18_Weights.IMAGENET1K_V1
    encoder = resnet18(weights=weights).eval().requires_grad_(False)
    encoder.fc = nn.Identity()
    with (gate/"dados/manifestos/G5_E03_Manifesto_Canonicos.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows)==3120 and {r["origem"] for r in rows}=={o for ids in SPLITS.values() for o in ids}
    root = gate.parent/"Data and Scripts/Gate4_Exp03_Run01/Originais_Canonicos"
    blocks = {}
    for split in SPLITS:
        x, y, origins = load_segments(root,rows,split,654)
        features, target, spectral_origins = extract(gate,rows,split,"x164",encoder)
        assert torch.equal(y,target)
        np.testing.assert_array_equal(origins,spectral_origins)
        stats = statistics(x)
        assert torch.isfinite(stats).all() and torch.isfinite(features).all()
        blocks[split] = (stats,features,y,origins)
    s,c,y,_ = blocks["treino"]
    sv,cv,target,origins = blocks["validacao"]
    assert (len(y),int(y.sum()),len(target),int(target.sum()))==(2305,1741,576,435)
    s,sv,sm,ss = standardize(s,sv)
    c,cv,cm,cs = standardize(c,cv)
    # Cada assinatura entra com o mesmo peso por componente padronizado.
    variants = {"estatisticas":(s,sv), "cnn":(c,cv),
                "hibrido":(torch.cat((c,s),1),torch.cat((cv,sv),1))}
    dest = gate/"resultados/execucoes"/run_id
    dest.mkdir()
    report = dict(pesos=weights.name,torch=torch.__version__,seed=501,epocas=20,lote=64,
                  lr=.001,weight_decay=.01,peso_positivo=564/1741,resolucao=[128,64],
                  amostras=654,estatisticas=NAMES,resultados={})
    for key,(train,val) in variants.items():
        head = fit_head(train,y)
        with torch.no_grad():
            logits = head(val).flatten()
        report["resultados"][key] = dict(entradas=train.shape[1],agregado=metrics(logits,target),
                por_origem={o:metrics(logits[origins==o],target[origins==o]) for o in SPLITS["validacao"]})
        with (gate/"modelos/checkpoints"/f"{run_id}_{key}.pt").open("xb") as f:
            torch.save(dict(head=head.state_dict(),variante=key,stats_media=sm,stats_desvio=ss,
                            cnn_media=cm,cnn_desvio=cs,pesos=weights.name),f)
        print(key, report["resultados"][key]["agregado"], flush=True)
    (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    figures = gate/"resultados/figuras"/run_id
    figures.mkdir()
    plot(report["resultados"],figures/"G5_E08_Comparacao.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]),sys.argv[2])
