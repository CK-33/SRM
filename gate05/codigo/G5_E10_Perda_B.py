"""E10.4: perda completa de B e avaliação do modelo especializado AC."""
import json
import sys
from pathlib import Path

import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import ResNet18_Weights, resnet18

from G5_E09_AWGN import features
from G5_E09_Proporcao_AWGN import evaluate
from G5_E10_AWGN_Inedito import BASE, load_validation, recortes

AC_ID = "G5_E06_20261121_232055_824_x164_AC.pt"
CASES = ("cnn_ABC_integro","hibrido_ABC_integro","cnn_ABC_sem_B","hibrido_ABC_sem_B","AC_especializado")


def load_head(path,width,weights):
    state=torch.load(path,map_location="cpu",weights_only=True)
    assert state["pesos"]==weights.name
    head=nn.Linear(width,1).eval()
    head.load_state_dict(state["head"])
    return head,state


def plot(rows,path):
    img=Image.new("RGB",(1240,640),"white")
    d=ImageDraw.Draw(img)
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",22)
    title=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",31)
    ink="#172638"
    d.text((42,26),"G5/E10.4 | Perda da fase B",font=title,fill=ink)
    d.text((42,76),"Omissões / falsos positivos · R03 e R09",font=font,fill=ink)
    for j,label in enumerate(("Limpo","30 dB · 801","30 dB · 802","30 dB · 803")):
        d.text((565+174*j,143),label,anchor="mm",font=font,fill=ink)
    names=("CNN ABC com B","Híbrido ABC com B","CNN ABC forçada sem B",
           "Híbrido ABC forçado sem B","CNN AC especializada")
    for i,(case,name) in enumerate(zip(CASES,names)):
        y=177+77*i
        d.text((42,y+21),name,anchor="lm",font=font,fill=ink)
        for j,row in enumerate(rows):
            x=565+174*j
            v=row["modelos"][case]["agregado"]
            d.rounded_rectangle((x-76,y,x+76,y+46),radius=8,
                                fill="#ddeef0" if i==4 else "#eef2f7")
            d.text((x,y+23),f'{v["fn"]} / {v["fp"]}',anchor="mm",font=font,fill=ink)
    d.text((42,583),"ABC sem B é uma simulação de falha. AC é o caminho alternativo avaliado.",
           font=font,fill=ink)
    img.save(path)


def run(gate,run_id):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(gate/"modelos/pesos_pre_treinados"))
    weights=ResNet18_Weights.IMAGENET1K_V1
    encoder=resnet18(weights=weights).eval().requires_grad_(False)
    encoder.fc=nn.Identity()

    def masked(x):
        return encoder(x*x.new_tensor([1,0,1])[None,:,None,None])

    folder=gate/"modelos/checkpoints"
    cnn,c=load_head(folder/f"{BASE}_125_cnn.pt",512,weights)
    hybrid,h=load_head(folder/f"{BASE}_125_hibrido.pt",520,weights)
    ac,a=load_head(folder/AC_ID,512,weights)
    assert tuple(a["mascara"])==(1,0,1) and a["configuracao"]=="x164"
    assert all(round(state["fracao_real"]*2305)==288 for state in (c,h))
    blocks,target,origin=load_validation(gate)
    dest=gate/"resultados/execucoes"/run_id
    dest.mkdir()
    rows=[]
    source="G5_E06_20261121_232055_824"
    ref=json.loads((gate/"resultados/execucoes"/source/"metricas.json").read_text(encoding="utf-8"))
    report=dict(referencia_ABC=BASE,referencia_AC=source,fs_Hz=25000,amostras=654,
                torch=torch.__version__,resultados=rows)
    for dose,seed in ((None,None),(30,801),(30,802),(30,803)):
        x,actual=recortes(blocks,dose,seed)
        stats,abc=features(x,encoder)
        # O canal indisponível não entra no extrator mascarado, nem nas estatísticas usadas.
        absent=x.clone()
        absent[:,1]=0
        stats_ac,embedding_ac=features(absent,masked)
        zstats=(stats-h["stats_media"])/h["stats_desvio"]
        zstats_ac=(stats_ac-h["stats_media"])/h["stats_desvio"]
        # Imputação pela média de treino nas entradas que requerem B.
        zstats_ac[:,[1,4,6,7]]=0
        score_inputs={
            "cnn_ABC_integro":(cnn,(abc-c["cnn_media"])/c["cnn_desvio"]),
            "hibrido_ABC_integro":(hybrid,torch.cat(((abc-h["cnn_media"])/h["cnn_desvio"],zstats),1)),
            "cnn_ABC_sem_B":(cnn,(embedding_ac-c["cnn_media"])/c["cnn_desvio"]),
            "hibrido_ABC_sem_B":(hybrid,torch.cat(((embedding_ac-h["cnn_media"])/h["cnn_desvio"],zstats_ac),1)),
            "AC_especializado":(ac,embedding_ac),
        }
        results={key:evaluate(head,z,target,origin) for key,(head,z) in score_inputs.items()}
        if dose is None:
            expected=ref["resultados"]["x164"]["AC"]["agregado"]
            assert all(results["AC_especializado"]["agregado"][k]==expected[k] for k in ("tp","tn","fp","fn"))
            assert results["cnn_ABC_integro"]["agregado"]["fn"]==3
            assert results["hibrido_ABC_integro"]["agregado"]["fn"]==2
            assert all(results[m]["agregado"]["fp"]==0 for m in CASES[:2])
        rows.append(dict(snr_alvo_dB=dose,seed=seed,snr_realizada_ABC_dB=actual,modelos=results))
        print(dose,seed,{k:(v["agregado"]["fn"],v["agregado"]["fp"]) for k,v in results.items()},flush=True)
        (dest/"metricas.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    folder=gate/"resultados/figuras"/run_id
    folder.mkdir()
    plot(rows,folder/"G5_E10_Perda_B.png")
    print(f"OK: {dest/'metricas.json'}",flush=True)


if __name__=="__main__":
    run(Path(sys.argv[1]),sys.argv[2])
