import csv
from PIL import Image, ImageDraw, ImageFont


def export_csv(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,"w",encoding="utf-8-sig",newline="") as file:
        writer=csv.DictWriter(file,keys)
        writer.writeheader()
        writer.writerows(rows)


def export_pdf(path, lines, figures):
    from textwrap import wrap
    normal=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",23)
    title=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",40)
    paragraphs=[line for row in lines for line in (wrap(str(row),80) or [""])]
    pages=[]
    for start in range(0,len(paragraphs),42):
        image=Image.new("RGB",(1240,1754),"white")
        draw=ImageDraw.Draw(image)
        draw.text((70,70),"SRM Lab · Relatório de análise",font=title,fill="#12273a")
        for i,line in enumerate(paragraphs[start:start+42]):
            draw.text((70,160+i*34),line,font=normal,fill="#12273a")
        pages.append(image)
    for name,figure in figures:
        image=Image.new("RGB",(1240,1754),"white")
        ImageDraw.Draw(image).text((70,70),name,font=title,fill="#12273a")
        copy=figure.copy().convert("RGB")
        copy.thumbnail((1100,1500),Image.Resampling.LANCZOS)
        image.paste(copy,(70,155))
        pages.append(image)
    pages[0].save(path,save_all=True,append_images=pages[1:],resolution=150)
