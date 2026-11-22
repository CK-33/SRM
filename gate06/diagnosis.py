import numpy as np
from signals import quality,stats


class Monitor:
    def __init__(self, fs):
        self.fs=fs
        self.size=max(16,round(fs*.02616))
        self.step=max(1,round(fs*.00164))
        self.next=self.size
        self.baseline=[]
        self.center=None
        self.spread=None
        self.cusum=self.ewma=0.
        self.streak=0
        self.active=False
        self.history=[]
        self.events=[]
        self.current="Dados insuficientes"
        self.q=["Aguardando"]*3

    def process(self, record, released):
        for end in range(self.next,released+1,self.step):
            x=record.current[end-self.size:end]
            q=quality(x)
            self.q=q
            z=None
            r=None
            before=self.active
            if record.jitter>.05 or record.units!="A" or any(a!="Sem alerta" for a in q):
                self.current="Sensor ou aquisição inválida"
                self.cusum=self.ewma=0.
                self.streak=0
                self.active=False
            else:
                r=float(np.sqrt(np.mean(x[:,1]**2)))
                if self.center is None:
                    self.baseline.append(r)
                    self.current="Dados insuficientes · construindo referência inicial"
                    if end>=round(self.fs*.125) and len(self.baseline)>=8:
                        self.center=float(np.median(self.baseline))
                        self.spread=max(1.4826*float(np.median(abs(np.asarray(self.baseline)-self.center))),
                                        .02*self.center,.02)
                else:
                    z=(r-self.center)/self.spread
                    self.cusum=max(0.,self.cusum+abs(z)-.5)
                    self.ewma=.2*z+.8*self.ewma
                    self.streak=self.streak+1 if abs(z)>3 and (self.cusum>5 or abs(self.ewma)>1) else 0
                    self.active=self.streak>=3
                    self.current="Alteração detectada" if self.active else "Sem anomalia detectada nas condições avaliadas"
            timestamp=float(record.time[end-1])
            if self.active and not before:
                self.events.append((timestamp,"Alteração RMS B · CUSUM/EWMA",z))
            self.history.append(dict(tempo_s=timestamp,rms_B_A=r,z=z,cusum=self.cusum,
                                     ewma=self.ewma,estado=self.current))
            self.next=end+self.step
