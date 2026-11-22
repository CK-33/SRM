"""G5/E06: ablação de fases com extrator ImageNet congelado."""
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from G5_E04_Transferencia import SPLITS, extract, metrics

MASKS = {"A": (1,0,0), "B": (0,1,0), "C": (0,0,1), "AC": (1,0,1), "ABC": (1,1,1)}


def fit(x, y):
    torch.manual_seed(501)
    head = nn.Linear(512,1)
    optimizer = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(564/1741))
    for _ in range(20):
        for idx in torch.randperm(len(y)).split(64):
            optimizer.zero_grad()
            loss(head(x[idx]).flatten(),y[idx]).backward()
            optimizer.step()
    return head.eval()


def plot(report, path):
    image = Image.new("RGB",(1200,830),"white")
    draw = ImageDraw.Draw(image)
    regular = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",19)
    bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",28)
    ink = "#182c40"
    draw.text((40,25),"G5/E06 | Efeito da disponibilidade das fases",font=bold,fill=ink)
    draw.text((40,68),"Validação canônica · F1, recall e taxa de falsos positivos (FPR)",font=regular,fill=ink)
    colors = ("#216ba4","#437c53","#ad5226")
    for row_id,(key, name) in enumerate((("x164","2 ciclos / 164"),("x327","3 ciclos / 327"))):
        top = 112 + 350*row_id
        draw.text((40,top),name,font=bold,fill=ink)
        for col,scope in enumerate(("agregado","R03","R09")):
            x0 = 45+390*col
            draw.text((x0,top+42),scope,font=regular,fill=ink)
            y0 = top+238
            draw.line((x0,y0,x0+320,y0),fill=ink,width=2)
            for rate in (0.5,1):
                yy = y0-155*rate
                draw.line((x0,yy,x0+320,yy),fill="#e0e9ee")
                draw.text((x0+321,yy-10),str(rate),font=regular,fill=ink)
            for i,phase in enumerate(MASKS):
                entry = report["resultados"][key][phase]
                scores = entry["agregado"] if scope=="agregado" else entry["por_origem"][scope]
                bx = x0+10+62*i
                for j,(metric,color) in enumerate(zip(("f1","recall","fpr"),colors)):
                    value = scores[metric]
                    if value:
                        draw.rectangle((bx+14*j,y0-155*value,bx+14*j+10,y0-1),fill=color)
                draw.text((bx+17,y0+8),phase,anchor="mt",font=regular,fill=ink)
    for j,(label,color) in enumerate(zip(("F1","Recall","FPR"),colors)):
        draw.rectangle((55+145*j,785,74+145*j,804),fill=color)
        draw.text((82+145*j,783),label,font=regular,fill=ink)
    image.save(path)


def run(gate, run_id, baseline):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate / "modelos/pesos_pre_treinados"))
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval().requires_grad_(False)
    model.fc = nn.Identity()
    with (gate / "dados/manifestos/G5_E03_Manifesto_Canonicos.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert len(rows)==3120 and {r["origem"] for r in rows} == {o for ids in SPLITS.values() for o in ids}
    reference = json.loads((gate / "resultados/execucoes" / baseline / "metricas.json").read_text(encoding="utf-8"))
    execution = gate / "resultados/execucoes" / run_id
    execution.mkdir()
    report = dict(baseline=baseline,pesos=weights.name,torch=torch.__version__,seed=501,
                  epocas=20,lote=64,lr=.001,resolucao=[128,64],
                  ordem_canais=["A","B","C"],mascaras=MASKS,resultados={})
    for key in ("x164","x327"):
        report["resultados"][key] = {}
        for phase in ("ABC","A","B","C","AC"):
            def masked(x):
                mask = x.new_tensor(MASKS[phase])[None,:,None,None]
                return model(x * mask)
            train, y, _ = extract(gate,rows,"treino",key,masked)
            val, target, origins = extract(gate,rows,"validacao",key,masked)
            assert (len(y),int(y.sum()),len(target),int(target.sum()))==(2305,1741,576,435)
            head = fit(train,y)
            with torch.no_grad():
                logits = head(val).flatten()
            scores = metrics(logits,target)
            if phase=="ABC":
                original = reference["resultados"][key]["validacao"]
                assert all(scores[k]==original[k] for k in ("tp","tn","fp","fn"))
                assert abs(scores["logloss"]-original["logloss"]) <= 1e-6
            report["resultados"][key][phase] = dict(
                agregado=scores,por_origem={o:metrics(logits[origins==o],target[origins==o])
                                            for o in SPLITS["validacao"]})
            path = gate / "modelos/checkpoints" / f"{run_id}_{key}_{phase}.pt"
            with path.open("xb") as file:
                torch.save(dict(head=head.state_dict(),pesos=weights.name,configuracao=key,
                                canais=phase,mascara=MASKS[phase]),file)
            (execution/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
            print(key,phase,scores,flush=True)
    folder = gate / "resultados/figuras" / run_id
    folder.mkdir()
    plot(report,folder/"G5_E06_Metricas_por_Fase.png")
    print(f"OK: {execution / 'metricas.json'}",flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]),sys.argv[2],sys.argv[3])
