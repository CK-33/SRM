"""G5/E04: ResNet-18 ImageNet congelada, cabeça binária em CPU."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import __version__ as torchvision_version
from torchvision.models import ResNet18_Weights, resnet18

from G5_E04_Verificar_Lote import SPLITS


def extract(gate, rows, split, key, model):
    chunks, labels, origins = [], [], []
    for origin in SPLITS[split]:
        manifest = sorted((r for r in rows if r["origem"] == origin), key=lambda r: int(r["fim"]))
        assert len(manifest) == 312 and all(r["split"] == split for r in manifest)
        path = gate / "dados/derivados/25kHz" / split / f"{origin}_STFT_v1.mat"
        raw = loadmat(path, variable_names=[key, "stop"])
        np.testing.assert_array_equal(raw["stop"].ravel(), [int(r["fim"]) for r in manifest])
        selected = [i for i, row in enumerate(manifest) if row["par_puro"] == "1"]
        assert all(manifest[i]["rotulo_654"] == manifest[i]["rotulo_981"] in ("pre", "pos")
                   for i in selected)
        labels.extend(int(manifest[i]["rotulo_654"] == "pos") for i in selected)
        origins.extend([origin] * len(selected))
        for start in range(0, len(selected), 16):
            indices = selected[start:start+16]
            x = torch.from_numpy(np.ascontiguousarray(raw[key][:, :, :, indices].transpose(3, 0, 2, 1)))
            assert torch.isfinite(x).all() and (x >= 0).all()
            # Escala física fixa, comum a todas as fases e partições: -60 a +40 dB re 1 A.
            x = ((20 * x.clamp_min(1e-3).log10() + 60) / 100).clamp(0, 1)
            x = F.interpolate(x, size=(128, 64), mode="bilinear", align_corners=False, antialias=True)
            mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
            std = x.new_tensor([.229, .224, .225])[None, :, None, None]
            with torch.no_grad():
                chunks.append(model((x - mean) / std))
        print(f"{key} {split} {origin}: {len(selected)} janelas", flush=True)
    return torch.cat(chunks), torch.tensor(labels, dtype=torch.float32), np.array(origins)


def metrics(logits, labels):
    pred = logits >= 0
    pos = labels == 1
    tp, tn, fp, fn = map(int, ((pred & pos).sum(), (~pred & ~pos).sum(),
                               (pred & ~pos).sum(), (~pred & pos).sum()))
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, recall=tp/(tp+fn), fpr=fp/(fp+tn),
                f1=2*tp/(2*tp+fp+fn), balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,
                logloss=float(F.binary_cross_entropy_with_logits(logits, labels)))


def run(gate, run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate / "modelos/pesos_pre_treinados"))
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.requires_grad_(False).eval()
    with (gate / "dados/manifestos/G5_E03_Manifesto_Canonicos.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    allowed = {origin for origins in SPLITS.values() for origin in origins}
    assert len(rows) == 3120 and {r["origem"] for r in rows} == allowed
    report = {"peso": weights.name, "url": weights.url, "torch": torch.__version__,
              "torchvision": torchvision_version, "resolucao": [128, 64],
              "cpu_threads": 4, "seed": 501, "epocas": 20, "lr": .001,
              "peso_positivo": 564/1741, "resultados": {}}
    run_dir = gate / "resultados/execucoes" / run_id
    run_dir.mkdir(exist_ok=False)
    for key in ("x164", "x327"):
        train, y, _ = extract(gate, rows, "treino", key, model)
        val, target, origins = extract(gate, rows, "validacao", key, model)
        assert (len(y), int(y.sum()), len(target), int(target.sum())) == (2305, 1741, 576, 435)
        torch.manual_seed(501)
        head = nn.Linear(512, 1)
        opt = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(564/1741))
        losses = []
        for _ in range(20):
            total = 0
            for batch in torch.randperm(len(y)).split(64):
                opt.zero_grad()
                loss = loss_fn(head(train[batch]).flatten(), y[batch])
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(batch)
            losses.append(total/len(y))
        with torch.no_grad():
            logits = head(val).flatten()
            report["resultados"][key] = {
                "validacao": metrics(logits, target),
                "por_origem": {o: metrics(logits[origins == o], target[origins == o])
                               for o in SPLITS["validacao"]},
                "perda_treino": losses,
            }
        checkpoint = gate / "modelos/checkpoints" / f"{run_id}_{key}.pt"
        with checkpoint.open("xb") as file:
            torch.save({"head": head.state_dict(), "pesos": weights.name, "configuracao": key}, file)
        (run_dir / "metricas.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", type=Path)
    parser.add_argument("run_id")
    args = parser.parse_args()
    run(args.gate, args.run_id)
