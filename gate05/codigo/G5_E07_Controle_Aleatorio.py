"""E07.1: ResNet-18 congelada, pesos aleatórios versus ImageNet."""
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision.models import resnet18

from G5_E04_Transferencia import SPLITS, extract, metrics
from G5_E06_Fases import fit


def plot(report, path):
    im = Image.new("RGB", (1100, 580), "white")
    draw = ImageDraw.Draw(im)
    title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 31)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 21)
    ink = "#172638"
    draw.text((40, 26), "G5/E07.1 | Efeito dos pesos do extrator", fill=ink, font=title)
    draw.text((40, 74), "Logloss de validação · menor é melhor · R03 e R09", fill=ink, font=font)
    largest = max(row["agregado"]["logloss"] for configs in report.values() for row in configs.values())
    for i, (key, config) in enumerate(report.items()):
        top = 135 + 174*i
        draw.text((40, top), "2 ciclos / 164" if key=="x164" else "3 ciclos / 327", fill=ink, font=title)
        for j, (label, color) in enumerate((("ImageNet", "#176ab0"), ("Aleatório", "#ae541c"))):
            value = config[label]["agregado"]["logloss"]
            y = top+55+43*j
            draw.text((40,y), label, fill=ink, font=font)
            width = 650*value/largest
            draw.rectangle((180,y,180+width,y+25), fill=color)
            draw.text((194+width,y), f"{value:.6f}", fill=ink, font=font)
    draw.text((40, 525), "Cabeça treinada sob o mesmo protocolo · extrator congelado nas duas condições",
              fill=ink, font=font)
    im.save(path)


def run(gate, run_id, baseline):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with (gate / "dados/manifestos/G5_E03_Manifesto_Canonicos.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3120
    assert {r["origem"] for r in rows} == {o for origins in SPLITS.values() for o in origins}
    reference = json.loads((gate / "resultados/execucoes" / baseline / "metricas.json").read_text(encoding="utf-8"))
    results = {}
    report = dict(baseline=baseline, arquitetura="resnet18", pesos_aleatorio=None,
                  sementes_extrator=[501,502,503], semente_cabeca=501,
                  epocas_cabeca=20, resultados=results)
    dest = gate / "resultados/execucoes" / run_id
    dest.mkdir()
    for key in ("x164","x327"):
        original = reference["resultados"][key]
        results[key] = {"ImageNet": dict(agregado=original["validacao"], por_origem=original["por_origem"])}
        repeats = []
        for seed in report["sementes_extrator"]:
            torch.manual_seed(seed)
            model = resnet18(weights=None).eval().requires_grad_(False)
            model.fc = nn.Identity()
            train,y,_ = extract(gate,rows,"treino",key,model)
            val,target,origins = extract(gate,rows,"validacao",key,model)
            assert (len(y),int(y.sum()),len(target),int(target.sum())) == (2305,1741,576,435)
            head = fit(train,y)
            with torch.no_grad():
                logits = head(val).flatten()
            scores = metrics(logits,target)
            repeats.append(dict(seed=seed,agregado=scores,
                                por_origem={o:metrics(logits[origins==o],target[origins==o])
                                            for o in SPLITS["validacao"]}))
            path = gate / "modelos/checkpoints" / f"{run_id}_{key}_seed{seed}.pt"
            with path.open("xb") as file:
                torch.save(dict(extrator=model.state_dict(),head=head.state_dict(),
                                seed=seed,configuracao=key),file)
            print(key,seed,scores,flush=True)
        avg = {name:sum(r["agregado"][name] for r in repeats)/len(repeats)
               for name in ("logloss","f1","fpr","recall","balanced_accuracy")}
        results[key]["Aleatório"] = dict(agregado=avg, por_semente=repeats)
        (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False), encoding="utf-8")
    folder = gate / "resultados/figuras" / run_id
    folder.mkdir()
    plot(results,folder/"G5_E07_Controle_Aleatorio.png")
    print(f"OK: {dest / 'metricas.json'}",flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]),sys.argv[2],sys.argv[3])
