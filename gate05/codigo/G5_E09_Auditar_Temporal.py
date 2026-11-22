"""Auditoria documental da família C, usando somente os três manifestos CSV."""
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IDS = ("R01","R02","R04","R05","R07","R08","R10","R11","R03","R09")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def grid_index(delta):
    index = round(delta*25000)
    assert abs(index-delta*25000)<1e-6
    return index


def audit(gate):
    root = gate.parent/"Data and Scripts/Gate4_Exp03_Run01"
    manifest = {r["id"]:r for r in read(root/"G4_E03_Manifesto.csv")
                if r["id"].split("_")[0] in IDS}
    params = {r["id"]:r for r in read(root/"G4_E03_Parametros.csv")
              if r["id"].split("_")[0] in IDS}
    clocks = read(gate/"dados/manifestos/G5_E03_Manifesto_Canonicos.csv")
    assert len(clocks)==3120 and {r["origem"] for r in clocks}==set(IDS)
    rows = []
    for i,origin in enumerate(IDS):
        split = "treino" if i<8 else "validacao"
        base = manifest[f"{origin}_O00"]
        t0 = float(base["t_inicio_s"])
        te = float(base["t_evento_s"])
        n = int(base["n_amostras"])
        assert base["grupo"]==split and n==13750 and int(base["fs_Hz"])==25000
        assert grid_index(float(base["t_fim_s"])-t0)==n-1
        own = [r for r in clocks if r["origem"]==origin]
        pure = [r for r in own if r["par_puro"]=="1"]
        assert len(own)==312 and all(r["split"]==split for r in own)
        for code in range(1,9):
            name=f"{origin}_C{code:02d}"
            info=manifest[name]
            alpha=float(params[name]["alpha"])
            assert info["grupo"]==split and int(info["registro_origem"])==int(origin[1:])
            assert int(info["fs_Hz"])==25000 and int(info["familia"])==3
            assert info["metodo"]==("dilatacao_temporal" if alpha<1 else "compressao_temporal")
            assert abs(float(info["t_evento_s"])-te)<1e-10 and alpha>0
            add_start=grid_index(float(info["t_inicio_s"])-t0)
            stop=grid_index(float(info["t_fim_s"])-t0)
            assert 0<=add_start<=stop<n and stop-add_start+1==int(info["n_amostras"])
            assert t0+add_start/25000<=te<=t0+stop/25000
            # Relógio canônico em tempo de saída; não realinhar por índice local.
            counts={}
            for length in (654,981):
                kept=[r for r in pure if int(r[f"inicio_{length}"])>=add_start and int(r["fim"])<=stop]
                dropped=[r for r in pure if r not in kept]
                counts[str(length)]=dict(total=len(pure),validas=len(kept),
                    excluidas_pre=sum(r[f"rotulo_{length}"]=="pre" for r in dropped),
                    excluidas_pos=sum(r[f"rotulo_{length}"]=="pos" for r in dropped))
            displacement=1000*abs(alpha-1)*max(abs(t0+add_start/25000-te),abs(t0+stop/25000-te))
            rows.append(dict(id=name,origem=origin,split=split,alpha=alpha,recorte_inicio=add_start,
                             recorte_fim=n-1-stop,deslocamento_max_teorico_ms=displacement,contagens=counts))
    return rows


def plot(rows,path):
    image=Image.new("RGB",(1110,585),"white")
    d=ImageDraw.Draw(image)
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",22)
    b=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",30)
    ink="#172638"
    d.text((42,28),"G5/E09.4 | Suporte da deformação temporal",font=b,fill=ink)
    d.text((42,76),"Máximos por variante entre as dez origens · relógio canônico fixo",font=f,fill=ink)
    for j,label in enumerate(("Variante","α","Corte início","Corte fim","Perda 2 ciclos","Perda 3 ciclos")):
        d.text((42+175*j,143),label,font=f,fill=ink)
    for i in range(1,9):
        part=[r for r in rows if r["id"].endswith(f"C{i:02d}")]
        y=184+43*(i-1)
        d.rectangle((30,y-6,1070,y+30),fill="#eef3f8" if i%2 else "#e0ebf4")
        alpha=f'{part[0]["alpha"]:.4f}'
        loss=[max(r["contagens"][n]["total"]-r["contagens"][n]["validas"] for r in part)
              for n in ("654","981")]
        vals=(f"C{i:02d}",alpha,max(r["recorte_inicio"] for r in part),
              max(r["recorte_fim"] for r in part),*loss)
        for j,value in enumerate(vals):
            d.text((42+175*j,y),str(value),font=f,fill=ink)
    d.text((42,545),"Cortes em amostras. Perdas em janelas puras por origem.",font=f,fill=ink)
    image.save(path)


if __name__=="__main__":
    gate, dest = map(Path,sys.argv[1:3])
    rows=audit(gate)
    assert len(rows)==80
    (dest/"G5_E09_Auditoria_Temporal.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    plot(rows,dest/"G5_E09_Suporte_Temporal.png")
    print("OK: 80 variantes auditadas nos manifestos.")
    for n in ("654","981"):
        print(n, "maior perda",max(r["contagens"][n]["total"]-r["contagens"][n]["validas"] for r in rows))
