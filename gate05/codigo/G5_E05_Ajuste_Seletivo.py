"""Ajusta layer4 e a cabeça do E04, mantendo BatchNorm em avaliação."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from G5_E04_Transferencia import SPLITS, metrics


def load_inputs(gate, rows, split, key):
    inputs, labels, origins = [], [], []
    for origin in SPLITS[split]:
        group = sorted((r for r in rows if r["origem"] == origin), key=lambda r: int(r["fim"]))
        assert len(group) == 312 and all(r["split"] == split for r in group)
        path = gate / "dados/derivados/25kHz" / split / f"{origin}_STFT_v1.mat"
        raw = loadmat(path, variable_names=[key, "stop"])
        np.testing.assert_array_equal(raw["stop"].ravel(), [int(r["fim"]) for r in group])
        indices = [i for i, r in enumerate(group) if r["par_puro"] == "1"]
        assert all(group[i]["rotulo_654"] == group[i]["rotulo_981"] in ("pre", "pos") for i in indices)
        labels.extend(int(group[i]["rotulo_654"] == "pos") for i in indices)
        origins.extend([origin] * len(indices))
        for start in range(0, len(indices), 16):
            x = torch.from_numpy(np.ascontiguousarray(raw[key][:,:,:,indices[start:start+16]].transpose(3,0,2,1)))
            assert torch.isfinite(x).all() and (x >= 0).all()
            x = ((20*x.clamp_min(1e-3).log10()+60)/100).clamp(0,1)
            x = F.interpolate(x, (128,64), mode="bilinear", align_corners=False, antialias=True)
            inputs.append((x-x.new_tensor([.485,.456,.406])[None,:,None,None]) /
                          x.new_tensor([.229,.224,.225])[None,:,None,None])
    return torch.cat(inputs), torch.tensor(labels,dtype=torch.float32), np.array(origins)


def run(gate, baseline, run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate / "modelos/pesos_pre_treinados"))
    weights = ResNet18_Weights.IMAGENET1K_V1
    with (gate / "dados/manifestos/G5_E03_Manifesto_Canonicos.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3120
    assert {r["origem"] for r in rows} == {o for group in SPLITS.values() for o in group}
    dest = gate / "resultados/execucoes" / run_id
    dest.mkdir()
    report = dict(baseline=baseline, pesos=weights.name, seed=501, epocas=3,
                  lote=16, lr_layer4=1e-5, lr_cabeca=1e-4, resolucao=[128,64], resultados={})
    original = json.loads((gate / "resultados/execucoes" / baseline / "metricas.json").read_text())
    for key in ("x164","x327"):
        torch.manual_seed(501)
        model = resnet18(weights=weights)
        model.fc = nn.Linear(512,1)
        checkpoint = torch.load(gate / "modelos/checkpoints" / f"{baseline}_{key}.pt",
                                map_location="cpu", weights_only=True)
        assert checkpoint["pesos"] == weights.name and checkpoint["configuracao"] == key
        model.fc.load_state_dict(checkpoint["head"])
        model.requires_grad_(False).eval()
        model.layer4.requires_grad_(True)
        model.fc.requires_grad_(True)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(False)
        train, y, _ = load_inputs(gate, rows, "treino", key)
        val, target, origins = load_inputs(gate, rows, "validacao", key)
        assert (len(y),int(y.sum()),len(target),int(target.sum())) == (2305,1741,576,435)
        opt = torch.optim.AdamW([
            dict(params=[p for p in model.layer4.parameters() if p.requires_grad],lr=1e-5),
            dict(params=model.fc.parameters(),lr=1e-4),
        ],weight_decay=.01)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(564/1741))

        @torch.no_grad()
        def validate():
            logits = torch.cat([model(x).flatten() for x in val.split(16)])
            return dict(agregado=metrics(logits,target),
                        por_origem={o:metrics(logits[origins==o],target[origins==o])
                                    for o in SPLITS["validacao"]})

        before = validate()
        previous = original["resultados"][key]["validacao"]
        assert all(before["agregado"][m] == previous[m] for m in ("tp","tn","fp","fn"))
        assert np.isclose(before["agregado"]["logloss"], previous["logloss"], rtol=1e-4)
        history = []
        for epoch in range(1,4):
            running = 0.
            for indices in torch.randperm(len(y)).split(16):
                opt.zero_grad()
                loss = loss_fn(model(train[indices]).flatten(),y[indices])
                loss.backward()
                opt.step()
                running += float(loss.detach())*len(indices)
            scores = validate()
            history.append(dict(epoca=epoch, perda_treino=running/len(y), **scores))
            print(key, epoch, scores["agregado"], flush=True)
        path = gate / "modelos/checkpoints" / f"{run_id}_{key}.pt"
        with path.open("xb") as file:
            torch.save(dict(layer4=model.layer4.state_dict(),head=model.fc.state_dict(),
                            pesos=weights.name,configuracao=key),file)
        report["resultados"][key] = dict(antes=before,epocas=history)
        (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    plot(gate, run_id, report)
    print(f"OK: {dest / 'metricas.json'}", flush=True)


def plot(gate, run_id, report):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1040, 610), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 31)
    draw.text((45, 30), "G5/E05 | Logloss de validação por época", fill="#162938", font=title)
    draw.text((45, 81), "Época 0 = checkpoint G5/E04 · R03 e R09", fill="#162938", font=font)
    series = []
    for key, row in report["resultados"].items():
        scores = [row["antes"]] + row["epocas"]
        series.append([s["agregado"]["logloss"] for s in scores])
    ymax = max(max(values) for values in series) * 1.1 or 1
    left, right, top, bottom = 145, 955, 148, 465
    for i in range(5):
        yy = bottom - (bottom-top)*i/4
        draw.line((left, yy, right, yy), fill="#dbe3ec")
        draw.text((45, yy-12), f"{ymax*i/4:.5f}", fill="#162938", font=font)
    xs = [left + (right-left)*i/3 for i in range(4)]
    for i, xx in enumerate(xs):
        draw.text((xx, 485), str(i), anchor="mt", fill="#162938", font=font)
    for i, (values, color, label) in enumerate(zip(series, ("#176ab0","#ae541c"),
                                                   ("2 ciclos / 164","3 ciclos / 327"))):
        points = [(xx, bottom - (bottom-top)*v/ymax) for xx, v in zip(xs, values)]
        draw.line(points, fill=color, width=4)
        for xx, yy in points:
            draw.ellipse((xx-5,yy-5,xx+5,yy+5), fill=color)
        draw.text((200+360*i, 545), label, fill=color, font=font)
    folder = gate / "resultados/figuras" / run_id
    folder.mkdir()
    image.save(folder / "G5_E05_Logloss_Validacao.png")


if __name__ == "__main__":
    run(Path(sys.argv[1]),sys.argv[2],sys.argv[3])
