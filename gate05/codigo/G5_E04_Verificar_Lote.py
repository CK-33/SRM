"""Verifica os tensores locais do E03 e a entrada da ResNet-18."""
import argparse
import csv
import json
from pathlib import Path

from scipy.io import loadmat
import numpy as np
import torch
from torchvision.models import resnet18

SPLITS = {
    "treino": ("R01", "R02", "R04", "R05", "R07", "R08", "R10", "R11"),
    "validacao": ("R03", "R09"),
}


def verify(gate):
    data = gate / "dados"
    with (data / "manifestos/G5_E03_Manifesto_Canonicos.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    expected = {origin: split for split, origins in SPLITS.items() for origin in origins}
    assert len(rows) == 3120 and all(expected.get(r["origem"]) == r["split"] for r in rows)
    torch.set_num_threads(4)
    model = resnet18(weights=None).eval()
    result = {}
    for split, origins in SPLITS.items():
        counts = {"pre": 0, "pos": 0, "cruzado": 0}
        batches = {}
        for origin in origins:
            manifest = sorted((r for r in rows if r["origem"] == origin), key=lambda r: int(r["fim"]))
            path = data / "derivados/25kHz" / split / f"{origin}_STFT_v1.mat"
            mat = loadmat(path, variable_names=["stop", "x164", "x327"])
            ends = np.array([int(r["fim"]) for r in manifest])
            np.testing.assert_array_equal(ends, np.arange(980, 13750, 41))
            np.testing.assert_array_equal(ends, mat["stop"].ravel())
            pure = []
            for i, row in enumerate(manifest):
                end, event = int(row["fim"]), int(row["evento"])
                labels = []
                for length in (654, 981):
                    start = end - length + 1
                    assert start == int(row[f"inicio_{length}"])
                    label = "pre" if end < event else "pos" if start >= event else "cruzado"
                    assert row[f"rotulo_{length}"] == label
                    labels.append(label)
                paired = labels[0] == labels[1] and labels[0] != "cruzado"
                assert paired == bool(int(row["par_puro"]))
                counts[labels[0] if paired else "cruzado"] += 1
                if paired:
                    pure.append((i, int(labels[0] == "pos")))
            for key, frames in (("x164", 12), ("x327", 16)):
                x = mat[key]
                assert x.shape == (3, frames, 513, 312) and x.dtype == np.float32
                assert np.isfinite(x).all() and (x >= 0).all()
                if key not in batches:
                    selected = [next(i for i, y in pure if y == label) for label in (0, 1)]
                    batch = torch.from_numpy(np.ascontiguousarray(x[:, :, :, selected].transpose(3, 0, 2, 1)))
                    with torch.inference_mode():
                        output = model(batch)
                    assert output.shape == (2, 1000) and torch.isfinite(output).all()
                    batches[key] = {"entrada": list(batch.shape), "rotulos": [0, 1], "saida": list(output.shape)}
        result[split] = {"contagens": counts, "lotes": batches}
    assert result["treino"]["contagens"] == {"pre": 564, "pos": 1741, "cruzado": 191}
    assert result["validacao"]["contagens"] == {"pre": 141, "pos": 435, "cruzado": 48}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", type=Path)
    args = parser.parse_args()
    result = verify(args.gate)
    report = json.dumps(result, indent=2)
    (args.gate / "resultados/logs/G5_E04_Verificacao_Lote.json").write_text(report + "\n", encoding="utf-8")
    print(report)
