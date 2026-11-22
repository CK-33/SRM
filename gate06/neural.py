from pathlib import Path
import numpy as np

BASE="G5_E09_2_20261122_001522_555"
TIME="G5_E07_3_20261121_235336_565_654.pt"
MODES=("CNN 1D","ResNet-18 STFT","Híbrido ABC")


def score_batch(x, model, mode, ck, torch):
    from torch.nn import functional as F
    if mode=="CNN 1D":
        return model((x-ck["media"])/ck["desvio"]).flatten()
    frames=x[...,39:].unfold(-1,164,41)
    win=torch.hann_window(164,periodic=False)
    a=torch.fft.rfft(frames*win,n=1024).abs()/win.sum()
    a[...,1:-1]*=2
    a=((20*a.transpose(-1,-2).clamp_min(1e-3).log10()+60)/100).clamp(0,1)
    a=F.interpolate(a,(128,64),mode="bilinear",align_corners=False,antialias=True)
    mu=a.new_tensor([.485,.456,.406])[None,:,None,None]
    sigma=a.new_tensor([.229,.224,.225])[None,:,None,None]
    emb,head=model
    v=(emb((a-mu)/sigma)-ck["cnn_media"])/ck["cnn_desvio"]
    if mode=="Híbrido ABC":
        rms=x.square().mean(2).sqrt()
        crest=x.abs().amax(2)/rms.clamp_min(1e-8)
        total=rms.square().mean(1,keepdim=True).sqrt()
        d=100*rms.std(1,keepdim=True,correction=1)/rms.mean(1,keepdim=True).clamp_min(1e-8)
        s=torch.cat((rms,crest,total,d),1)
        v=torch.cat((v,(s-ck["stats_media"])/ck["stats_desvio"]),1)
    return head(v).flatten()


class Neural:
    def __init__(self, folder, mode):
        import torch
        from torch import nn
        root=Path(folder)
        self.name=mode
        self.torch=torch
        torch.set_num_threads(4)
        if mode=="CNN 1D":
            f=root/"checkpoints"/TIME
            ck=torch.load(f,map_location="cpu",weights_only=True)
            if ck["amostras"]!=654:
                raise ValueError("Checkpoint 1D incompatível.")
            net=nn.Sequential(
                nn.Conv1d(3,32,15,stride=2,padding=7,bias=False),nn.BatchNorm1d(32),nn.ReLU(),
                nn.Conv1d(32,64,9,stride=2,padding=4,bias=False),nn.BatchNorm1d(64),nn.ReLU(),
                nn.Conv1d(64,128,7,stride=2,padding=3,bias=False),nn.BatchNorm1d(128),nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(128,1))
            net.load_state_dict(ck["modelo"])
            self.model=net.eval()
        else:
            from torchvision.models import resnet18
            arm="hibrido" if mode=="Híbrido ABC" else "cnn"
            f=root/"checkpoints"/f"{BASE}_125_{arm}.pt"
            ck=torch.load(f,map_location="cpu",weights_only=True)
            if ck["pesos"]!="IMAGENET1K_V1" or round(ck["fracao_real"]*2305)!=288:
                raise ValueError("Checkpoint espectral incompatível.")
            backbone=root/"pesos_pre_treinados"/"checkpoints"/"resnet18-f37072fd.pth"
            net=resnet18(weights=None)
            net.load_state_dict(torch.load(backbone,map_location="cpu",weights_only=True))
            net.fc=nn.Identity()
            head=nn.Linear(520 if arm=="hibrido" else 512,1)
            head.load_state_dict(ck["head"])
            self.model=(net.eval(),head.eval())
        self.checkpoint=ck
        self.version=f.name

    def infer(self, x, fs, jitter, units):
        from signals import quality
        if x.shape!=(654,3) or abs(fs-25000)>.1 or jitter>.001 or units!="A":
            raise ValueError("IA exige 654 amostras ABC em A e grade uniforme de 25 kHz.")
        if any(a!="Sem alerta" for a in quality(x)):
            raise ValueError("Indeterminado: qualidade insuficiente para IA.")
        t=self.torch
        with t.inference_mode():
            value=score_batch(t.from_numpy(np.ascontiguousarray(x.T,dtype=np.float32))[None],
                              self.model,self.name,self.checkpoint,t).sigmoid().item()
        if not np.isfinite(value):
            raise ValueError("Saída neural inválida.")
        return value
