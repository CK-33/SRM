"""E07.2: piloto de treino integral da ResNet-18 em CPU."""
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import __version__ as vision_version
from torchvision.models import resnet18
from PIL import Image, ImageDraw, ImageFont

from G5_E04_Transferencia import SPLITS, metrics
from G5_E05_Ajuste_Seletivo import load_inputs

EPOCHS = 5
SEED = 501


def validate(model, x, y, origins):
    model.eval()
    with torch.no_grad():
        scores = torch.cat([model(batch).flatten() for batch in x.split(16)])
    return {"agregado": metrics(scores, y),
            "por_origem": {o: metrics(scores[origins == o], y[origins == o])
                           for o in SPLITS["validacao"]}}


def plot(report, path):
    image = Image.new("RGB", (1100, 640), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 28)
    ink = "#172638"
    draw.text((40, 26), "G5/E07.2 | ResNet-18 treinada do zero", fill=ink, font=title)
    draw.text((40, 72), "Logloss de validação por época · escala logarítmica · R03 e R09", fill=ink, font=font)
    from math import floor, ceil, log10
    all_values = [row["agregado"]["logloss"] for config in report.values() for row in config["epocas"]]
    all_values += [config["referencia_e04"]["logloss"] for config in report.values()]
    low = floor(log10(max(min(all_values), 1e-12)))
    high = max(low+1, ceil(log10(max(max(all_values), 1e-12))))
    left, right, top, bottom = 130, 1000, 140, 465
    def y(value):
        return bottom-(bottom-top)*(log10(max(value, 1e-12))-low)/(high-low)
    for exponent in range(low, high+1):
        yy = y(10**exponent)
        draw.line((left, yy, right, yy), fill="#dbe3ec")
        draw.text((45, yy-12), f"1e{exponent}", fill=ink, font=font)
    xs = [left+(right-left)*i/(EPOCHS-1) for i in range(EPOCHS)]
    for i, xx in enumerate(xs, 1):
        draw.text((xx, 481), str(i), anchor="mt", fill=ink, font=font)
    draw.text((565, 516), "Época", anchor="mt", fill=ink, font=font)
    for i, (key, color, label) in enumerate(zip(("x164","x327"), ("#176ab0","#ae541c"),
                                              ("2 ciclos / 164","3 ciclos / 327"))):
        config = report[key]
        points = [(xx, y(row["agregado"]["logloss"])) for xx, row in zip(xs,config["epocas"])]
        draw.line(points, fill=color, width=4)
        for xx, yy in points:
            draw.ellipse((xx-4, yy-4, xx+4, yy+4), fill=color)
        reference = y(config["referencia_e04"]["logloss"])
        for xx in range(left,right,22):
            draw.line((xx,reference,min(xx+12,right),reference), fill=color, width=2)
        draw.text((125+455*i, 558), label, fill=color, font=font)
    draw.text((125, 595), "Linhas tracejadas: referências ImageNet congeladas do E04", fill=ink, font=font)
    image.save(path)


def run(gate, run_id, baseline):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with (gate / "dados/manifestos/G5_E03_Manifesto_Canonicos.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3120
    assert {r["origem"] for r in rows} == {o for split in SPLITS.values() for o in split}
    reference = json.loads((gate / "resultados/execucoes" / baseline / "metricas.json").read_text(encoding="utf-8"))
    dest = gate / "resultados/execucoes" / run_id
    dest.mkdir()
    report = dict(baseline=baseline, pesos_iniciais=None, seed=SEED, epocas=EPOCHS, lote=16,
                  lr=.0003, weight_decay=.01, torch=torch.__version__, torchvision=vision_version,
                  cpu_threads=4, resolucao=[128,64], peso_positivo=564/1741,resultados={})
    for key in ("x164","x327"):
        train, y, _ = load_inputs(gate, rows, "treino", key)
        val, target, origins = load_inputs(gate, rows, "validacao", key)
        assert (len(y),int(y.sum()),len(target),int(target.sum())) == (2305,1741,576,435)
        torch.manual_seed(SEED)
        model = resnet18(weights=None)
        model.fc = nn.Linear(512,1)
        opt = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(564/1741))
        history = []
        report["resultados"][key] = dict(referencia_e04=reference["resultados"][key]["validacao"],epocas=history)
        for epoch in range(1,EPOCHS+1):
            model.train()
            total = 0.
            for idx in torch.randperm(len(y)).split(16):
                opt.zero_grad()
                loss = loss_fn(model(train[idx]).flatten(),y[idx])
                assert torch.isfinite(loss)
                loss.backward()
                opt.step()
                total += float(loss.detach())*len(idx)
            score = validate(model,val,target,origins)
            history.append(dict(epoca=epoch,perda_treino=total/len(y),**score))
            (dest/"metricas.json").write_text(json.dumps(report, indent=2, allow_nan=False),encoding="utf-8")
            print(key,epoch,score["agregado"],flush=True)
        path = gate / "modelos/checkpoints" / f"{run_id}_{key}.pt"
        with path.open("xb") as file:
            torch.save(dict(modelo=model.state_dict(),configuracao=key,epoca=EPOCHS,seed=SEED),file)
    figures = gate / "resultados/figuras" / run_id
    figures.mkdir()
    plot(report["resultados"],figures/"G5_E07_2_Logloss.png")
    print(f"OK: {dest / 'metricas.json'}",flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]),sys.argv[2],sys.argv[3])
