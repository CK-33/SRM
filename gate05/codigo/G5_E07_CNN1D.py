"""E07.3: CNN 1D sobre correntes canônicas ABC, execução local."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn

from G5_E04_Transferencia import SPLITS
from G5_E07_Treino_Do_Zero import EPOCHS, SEED, validate


def plot(report, path):
    from PIL import Image, ImageDraw, ImageFont
    from math import log10, floor, ceil

    image = Image.new("RGB", (1000, 570), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 30)
    ink = "#172638"
    draw.text((40,24), "G5/E07.3 | CNN 1D sobre correntes ABC", font=title, fill=ink)
    draw.text((40,76), "Logloss de validação por época · escala logarítmica · R03 e R09", font=font, fill=ink)
    curves = [[row["agregado"]["logloss"] for row in r["epocas"]] for r in report.values()]
    refs = [r["referencia_e04"]["logloss"] for r in report.values()]
    values = [log10(max(v,1e-12)) for curve in curves for v in curve] + [log10(max(v,1e-12)) for v in refs]
    lo = floor(min(values))
    hi = max(lo+1,ceil(max(values)))
    def y(log_value):
        return 440 - 300*(log_value-lo)/(hi-lo)
    for tick in range(lo,hi+1):
        draw.line((120,y(tick),930,y(tick)),fill="#dbe3ec")
        draw.text((40,y(tick)-10),f"1e{tick}",font=font,fill=ink)
    xs = [120+810*i/(EPOCHS-1) for i in range(EPOCHS)]
    for i,x in enumerate(xs,1):
        draw.text((x,453),str(i),anchor="mt",font=font,fill=ink)
    for i,(curve,ref,color) in enumerate(zip(curves,refs,("#176ab0","#ae541c"))):
        points = [(x,y(log10(max(v,1e-12)))) for x,v in zip(xs,curve)]
        draw.line(points,fill=color,width=4)
        for x,yy in points:
            draw.ellipse((x-4,yy-4,x+4,yy+4),fill=color)
        yy = y(log10(max(ref,1e-12)))
        for x in range(120,930,20):
            draw.line((x,yy,x+10,yy),fill=color,width=2)
        draw.text((120+350*i,493),f"{i+2} ciclos",font=font,fill=color)
    draw.text((120,530),"Tracejado: referência ResNet-18 + STFT do E07.2, época 5",font=font,fill=ink)
    image.save(path)


def load_segments(root, rows, split, length):
    segments, labels, origins = [], [], []
    for origin in SPLITS[split]:
        path = root / f"{origin}_O00.mat"
        r = loadmat(path, variable_names=["registro"], simplify_cells=True)["registro"]
        assert r["id"] == f"{origin}_O00" and r["Fs_Hz"] == 25000
        raw = np.asarray(r["I_A"], dtype=np.float32)
        assert raw.shape == (13750,3) and np.isfinite(raw).all()
        event = int(np.ceil((r["t_evento_s"]-r["t0_s"])*25000-1e-8))
        manifest = sorted((row for row in rows if row["origem"] == origin), key=lambda row: int(row["fim"]))
        assert len(manifest) == 312 and all(row["split"] == split for row in manifest)
        np.testing.assert_array_equal([int(row["fim"]) for row in manifest], np.arange(980,13750,41))
        for row in manifest:
            last = int(row["fim"])
            starts = {n: last-n+1 for n in (654,981)}
            expected = {n: "pre" if last < event else "pos" if starts[n] >= event else "cruzado" for n in starts}
            pure = expected[654] == expected[981] and expected[654] != "cruzado"
            assert int(row["evento"]) == event and int(row["par_puro"]) == int(pure)
            assert all(int(row[f"inicio_{n}"]) == starts[n] and row[f"rotulo_{n}"] == expected[n] for n in starts)
            if pure:
                segments.append(raw[starts[length]:last+1].T)
                labels.append(int(expected[length] == "pos"))
                origins.append(origin)
    return torch.from_numpy(np.stack(segments)), torch.tensor(labels,dtype=torch.float32), np.array(origins)


def model_1d():
    return nn.Sequential(
        nn.Conv1d(3,32,15,stride=2,padding=7,bias=False), nn.BatchNorm1d(32), nn.ReLU(),
        nn.Conv1d(32,64,9,stride=2,padding=4,bias=False), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Conv1d(64,128,7,stride=2,padding=3,bias=False), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128,1),
    )


def run(gate, run_id, reference_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = gate.parent / "Data and Scripts/Gate4_Exp03_Run01/Originais_Canonicos"
    with (gate/"dados/manifestos/G5_E03_Manifesto_Canonicos.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert len(rows)==3120 and {row["origem"] for row in rows} == {o for ids in SPLITS.values() for o in ids}
    ref = json.loads((gate/"resultados/execucoes"/reference_id/"metricas.json").read_text(encoding="utf-8"))
    dest = gate/"resultados/execucoes"/run_id
    dest.mkdir()
    report = dict(referencia=reference_id, fs_Hz=25000, seed=SEED, epocas=EPOCHS, lote=16,
                  lr=.0003, cpu_threads=4, torch=torch.__version__, resultados={})
    for key,length in (("x164",654),("x327",981)):
        x,y,_ = load_segments(root,rows,"treino",length)
        v,t,origins = load_segments(root,rows,"validacao",length)
        assert (len(y),int(y.sum()),len(t),int(t.sum())) == (2305,1741,576,435)
        # Um centro e uma escala por fase, fixos após ajuste apenas no treino.
        mean = x.mean(dim=(0,2),keepdim=True)
        std = x.std(dim=(0,2),keepdim=True,correction=0).clamp_min(1e-6)
        x, v = (x-mean)/std, (v-mean)/std
        torch.manual_seed(SEED)
        model = model_1d()
        opt = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(564/1741))
        history = []
        report["resultados"][key] = dict(amostras=length, media_ABC=mean.flatten().tolist(),
            desvio_ABC=std.flatten().tolist(), parametros=sum(p.numel() for p in model.parameters()),
            referencia_e04=ref["resultados"][key]["epocas"][-1]["agregado"], epocas=history)
        for epoch in range(1,EPOCHS+1):
            model.train()
            total = 0.
            for idx in torch.randperm(len(y)).split(16):
                opt.zero_grad()
                loss = criterion(model(x[idx]).flatten(),y[idx])
                assert torch.isfinite(loss)
                loss.backward()
                opt.step()
                total += float(loss.detach())*len(idx)
            scores = validate(model,v,t,origins)
            history.append(dict(epoca=epoch,perda_treino=total/len(y),**scores))
            (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
            print(length,epoch,scores["agregado"],flush=True)
        with (gate/"modelos/checkpoints"/f"{run_id}_{length}.pt").open("xb") as file:
            torch.save(dict(modelo=model.state_dict(),media=mean,desvio=std,amostras=length),file)
    figure = gate/"resultados/figuras"/run_id
    figure.mkdir()
    plot(report["resultados"],figure/"G5_E07_3_Logloss.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]),sys.argv[2],sys.argv[3])
