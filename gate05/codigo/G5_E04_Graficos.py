import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument("metricas", type=Path)
parser.add_argument("destino", type=Path)
args = parser.parse_args()
data = json.loads(args.metricas.read_text(encoding="utf-8-sig"))["resultados"]
args.destino.mkdir(parents=True, exist_ok=True)

def font(size, bold=False):
    return ImageFont.truetype("C:/Windows/Fonts/" + ("arialbd.ttf" if bold else "arial.ttf"), size)

image = Image.new("RGB", (1200, 780), "white")
d = ImageDraw.Draw(image)
ink = "#172638"
d.text((55, 35), "G5/E04 | ResNet-18 com extrator congelado", font=font(34, True), fill=ink)
d.text((55, 88), "Perda ponderada no treino por época · escala logarítmica", font=font(23), fill=ink)
left, right, top, bottom = 130, 1130, 155, 445
lo, hi = math.log10(.002), math.log10(.25)
def y(value):
    return bottom - (math.log10(value)-lo)/(hi-lo)*(bottom-top)
def x(epoch):
    return left + (epoch-1)/19*(right-left)
for v in (.002,.005,.01,.02,.05,.1,.2):
    yy = y(v)
    d.line((left,yy,right,yy), fill="#dbe3ec")
    d.text((55,yy-10), str(v).replace(".",","), font=font(17), fill=ink)
for n in (1,5,10,15,20):
    d.text((x(n),bottom+18), str(n), anchor="mt", font=font(18), fill=ink)
d.line((left,top,left,bottom,right,bottom), fill=ink,width=2)
colors = ("#176ab0","#ae541c")
for i,(key,label,color) in enumerate(zip(("x164","x327"),("2 ciclos / 164","3 ciclos / 327"),colors)):
    points = [(x(j),y(v)) for j,v in enumerate(data[key]["perda_treino"],1)]
    d.line(points,fill=color,width=4)
    for px,py in points:
        d.ellipse((px-4,py-4,px+4,py+4),fill=color)
    d.text((180+340*i,500), label, font=font(21,True), fill=color)
d.text((55,565),"Validação · ambas as configurações",font=font(26,True),fill=ink)
d.text((55,613),"141 pré e 435 pós classificados corretamente. Zero falsos alarmes ou omissões.",font=font(21),fill=ink)
losses = [data[k]["validacao"]["logloss"] for k in ("x164","x327")]
d.text((55,654),f"Logloss: {losses[0]:.6f} (2 ciclos)  |  {losses[1]:.6f} (3 ciclos)".replace(".",","),font=font(21),fill=ink)
d.text((55,710),"576 janelas puras de R03 e R09. Janelas da mesma origem são correlacionadas.",font=font(18),fill=ink)
image.save(args.destino / "G5_E04_Perda_e_Validacao.png")

image = Image.new("RGB",(1200,620),"white")
d = ImageDraw.Draw(image)
d.text((55,36),"G5/E04 | Matrizes de confusão na validação",font=font(34,True),fill=ink)
d.text((55,88),"Linhas = rótulo real · colunas = decisão da CNN",font=font(21),fill=ink)
for i,(key,label) in enumerate(zip(("x164","x327"),("2 ciclos / 164","3 ciclos / 327"))):
    m = data[key]["validacao"]
    sx = 175 + 550*i
    d.text((sx+160,150),label,anchor="mt",font=font(25,True),fill=ink)
    for j,category in enumerate(("Pré","Pós")):
        d.text((sx+80+160*j,212),category,anchor="mm",font=font(20),fill=ink)
        d.text((sx-20,284+105*j),category,anchor="rm",font=font(20),fill=ink)
    for j,value in enumerate((m["tn"],m["fp"],m["fn"],m["tp"])):
        xx, yy = sx+160*(j%2),232+105*(j//2)
        d.rectangle((xx,yy,xx+154,yy+99),fill="#175d94" if value else "#edf2f7")
        d.text((xx+77,yy+50),str(value),anchor="mm",font=font(37,True),fill="white" if value else ink)
d.text((55,500),"F1 = 1,00 em ambas · 576 janelas puras · R03 e R09",font=font(24,True),fill=ink)
d.text((55,550),"A validação deste ensaio cobre dois registros canônicos de simulação.",font=font(19),fill=ink)
image.save(args.destino / "G5_E04_Matrizes_Confusao.png")
