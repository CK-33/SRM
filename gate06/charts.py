import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

COLORS=("#2168c9","#bb5421","#087a66")
BG="#ffffff"


def render(t, x, title, xlabel, ylabel, width=1050, height=410, names=("A","B","C")):
    t=np.asarray(t)
    x=np.asarray(x).reshape(len(t),-1)
    im=Image.new("RGB",(width,height),BG)
    d=ImageDraw.Draw(im)
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",14)
    bold=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",18)
    fg="#233b56"
    d.text((18,13),title,font=bold,fill=fg)
    l,r,u,b=85,width-27,67,height-57
    dt=max(float(t[-1]-t[0]),1e-12)
    finite=x[np.isfinite(x)]
    low,high=(float(finite.min()),float(finite.max())) if finite.size else (0,1)
    span=max(high-low,1e-12)
    for i in range(6):
        px=l+(r-l)*i/5
        py=b-(b-u)*i/5
        d.line((px,u,px,b),fill="#e6edf5")
        d.line((l,py,r,py),fill="#e6edf5")
        d.text((px,b+7),f"{t[0]+dt*i/5:.4g}",font=f,fill=fg,anchor="mt")
        d.text((l-9,py),f"{low+span*i/5:.4g}",font=f,fill=fg,anchor="rm")
    count=max(1,len(t)//2200)
    for j,line in enumerate(x.T):
        color=COLORS[j%3]
        points=[]
        for a,v in zip(t[::count],line[::count]):
            if np.isfinite(v):
                points.append((l+(a-t[0])/dt*(r-l),b-(v-low)/span*(b-u)))
            else:
                if len(points)>1: d.line(points,fill=color,width=2)
                points=[]
        if len(points)>1: d.line(points,fill=color,width=2)
        if j<len(names):
            d.text((l+155*j,42),names[j],font=f,fill=color)
    d.text((width//2,height-24),xlabel,font=f,fill=fg,anchor="mm")
    d.text((18,43),ylabel,font=f,fill=fg)
    return im


def heatmap(z,t,freq):
    data=z[::-1]
    lo,hi=np.percentile(data,[5,99])
    q=np.clip((data-lo)/max(hi-lo,1e-8),0,1)
    rgb=np.uint8(np.stack((255*q**1.6,225*q,70+170*q**.6),-1))
    im=Image.new("RGB",(1050,410),BG)
    im.paste(Image.fromarray(rgb).resize((912,290)),(85,60))
    d=ImageDraw.Draw(im)
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",16)
    fg="#233b56"
    d.text((18,16),"Espectrograma B · 20 log10 da magnitude STFT",font=f,fill=fg)
    for y,v in ((60,freq[-1]),(337,0)):
        d.text((10,y),f"{v:.0f} Hz",font=f,fill=fg)
    d.text((85,366),f"{t[0]:.4f} s",font=f,fill=fg)
    d.text((884,366),f"{t[-1]:.4f} s",font=f,fill=fg)
    d.text((350,366),f"Escala visual: {lo:.1f} a {hi:.1f} dB",font=f,fill=fg)
    return im


class Chart(tk.Canvas):
    def __init__(self,parent,zoom):
        super().__init__(parent,height=420,bg=BG,highlightthickness=0)
        self.zoom=zoom
        self.bind("<MouseWheel>",lambda e:self.zoom(.8 if e.delta>0 else 1.25))
        self.bind("<ButtonPress-1>",lambda e:setattr(self,"start",e.x))
        self.bind("<ButtonRelease-1>",self.pan)
        self.bind("<Configure>",self.refresh)
        self.original=None
        self.placeholder="Nenhuma série para exibir"

    def empty(self,message="Selecione um arquivo ou abra a demonstração"):
        self.placeholder=message
        self.original=None
        self.refresh()

    def pan(self,event):
        delta=event.x-getattr(self,"start",event.x)
        if abs(delta)>8:
            self.zoom(1,-delta/max(self.winfo_width(),1))

    def show(self,im):
        self.original=im
        self.refresh()

    def refresh(self,_=None):
        if self.original is None:
            self.delete("all")
            x=max(0,self.winfo_width()//2)
            y=max(0,self.winfo_height()//2)
            self.create_text(x,y-32,text="∿",font=("Segoe UI Semibold",34),fill="#1763c5")
            self.create_text(x,y+13,text=self.placeholder,font=("Segoe UI Semibold",13),fill="#526b85")
            return
        im=self.original.copy()
        im.thumbnail((max(1,self.winfo_width()),max(1,self.winfo_height())))
        self.image=ImageTk.PhotoImage(im)
        self.delete("all")
        self.create_image((self.winfo_width()-im.width)//2,
                          (self.winfo_height()-im.height)//2,image=self.image,anchor="nw")
