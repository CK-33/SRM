import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import queue
from pathlib import Path
import time
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from signals import advanced, catalog, checked, demo, prepare_srm, quality, spectrogram, spectrum, stats
from charts import Chart, heatmap, render
from diagnosis import Monitor
from neural import MODES, Neural
from reports import export_csv, export_pdf

AREAS=("Visão geral","Análise de sinais","Diagnóstico","Análise avançada","Relatórios")
SUBTITLES=("Acompanhe a aquisição e os indicadores da sessão.",
           "Explore as correntes por fase, intervalo e frequência.",
           "Inspecione a qualidade dos sensores, os eventos e os escores.",
           "Compare periodicidade, defasagens e evolução temporal.",
           "Exporte a sessão, as métricas e os gráficos analisados.")
MODEL_ROOT=Path(__file__).resolve().parent/"modelos"


class App:
    def __init__(self, root, model_root=MODEL_ROOT):
        self.root=root
        self.model_root=Path(model_root)
        self.record=None
        self.representations={}
        self.monitor=None
        self.figures={}
        self.net=None
        self.net_value=None
        self.net_clock=None
        self.release=0
        self.playing=False
        self.anchor=0.
        self.replay=False
        self.generation=0
        self.view_generation=0
        self.monitor_busy=False
        self.chart_busy=False
        self.worker=ThreadPoolExecutor(max_workers=2)
        self.events=queue.Queue()
        root.title("SRM Lab | Gate 6")
        root.geometry("1260x830")
        root.minsize(1050,720)
        root.configure(bg="#eef3f9")
        style=ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame",background="#ffffff")
        style.configure("TLabel",background="#ffffff",foreground="#162943",font=("Segoe UI",10))
        style.configure("Work.TFrame",background="#eef3f9")
        style.configure("TButton",background="#eaf1f9",foreground="#18334f",padding=(10,7),
                        font=("Segoe UI Semibold",10),borderwidth=0,relief="flat")
        style.map("TButton",background=[("active","#dcecff")])
        style.configure("Primary.TButton",background="#1763c5",foreground="white",padding=(12,8))
        style.map("Primary.TButton",background=[("active","#0d4da6")])
        style.configure("Title.TLabel",background="#eef3f9",font=("Segoe UI Semibold",26))
        style.configure("Sub.TLabel",background="#eef3f9",foreground="#536982")
        style.configure("CardLabel.TLabel",foreground="#566e88",font=("Segoe UI Semibold",9))
        style.configure("CardValue.TLabel",foreground="#173e72",font=("Segoe UI Semibold",18))
        style.configure("Accent.Horizontal.TProgressbar",background="#3683ea",troughcolor="#e3edf9",
                        bordercolor="#e3edf9",lightcolor="#3683ea",darkcolor="#3683ea",thickness=4)
        style.configure("TCombobox",padding=4,fieldbackground="white")
        side=tk.Frame(root,bg="#101f35",width=220)
        side.pack(side="left",fill="y")
        side.pack_propagate(False)
        logo=tk.Frame(side,bg="#101f35")
        logo.pack(fill="x",padx=19,pady=(24,22))
        tk.Label(logo,text="∿",bg="#2969d2",fg="white",font=("Segoe UI Semibold",21),width=2).pack(side="left")
        tk.Label(logo,text="SRM Lab",bg="#101f35",fg="white",font=("Segoe UI Semibold",16)).pack(side="left",padx=6)
        tk.Label(side,text="ÁREAS DE TRABALHO",bg="#101f35",fg="#91a7c3",
                 font=("Segoe UI Semibold",9)).pack(anchor="w",padx=23,pady=(5,8))
        self.nav=[]
        for i,name in enumerate(AREAS):
            icon=("⌂","∿","◉","≋","▤")[i]
            nav=tk.Button(side,text=f"{icon}   {name}",command=lambda j=i:self.select(j),anchor="w",
                          bg="#101f35",activebackground="#294362",fg="#cfdef0",activeforeground="white",
                          bd=0,highlightthickness=0,font=("Segoe UI Semibold",10),padx=10,pady=11,cursor="hand2")
            nav.pack(fill="x",padx=12,pady=2)
            self.nav.append(nav)
        tk.Frame(side,bg="#2a3c54",height=1).pack(fill="x",padx=22,pady=(24,15))
        ttk.Button(side,text="+  Abrir arquivo",command=self.open,style="Primary.TButton").pack(fill="x",padx=17)
        ttk.Button(side,text="Abrir demonstração",command=lambda:self.load(demo())).pack(fill="x",padx=15,pady=7)
        tk.Label(side,text="GATE 6 · MODO LOCAL",bg="#101f35",fg="#91a7c3",
                 font=("Segoe UI Semibold",9)).pack(side="bottom",anchor="w",padx=20,pady=18)
        self.body=ttk.Frame(root,padding=(20,14,20,8),style="Work.TFrame")
        self.body.pack(side="left",fill="both",expand=True)
        self.title=ttk.Label(self.body,style="Title.TLabel")
        self.title.pack(anchor="w")
        self.subtitle=tk.StringVar()
        ttk.Label(self.body,textvariable=self.subtitle,style="Sub.TLabel").pack(anchor="w",pady=(0,11))
        session=ttk.Frame(self.body,padding=(10,8))
        session.pack(fill="x",pady=(0,7))
        ttk.Label(session,text="Máquina").pack(side="left")
        self.machine=tk.StringVar(value="SRM")
        ttk.Entry(session,textvariable=self.machine,width=11).pack(side="left",padx=(5,10))
        for label,func in (("Iniciar / retomar",self.play),("Pausar",self.pause),("Parar",self.stop),
                           ("Reiniciar",self.restart),("Arquivo inteiro",self.full)):
            ttk.Button(session,text=label,command=func,
                       style="Primary.TButton" if func==self.play else "TButton").pack(side="left",padx=2)
        self.speed=tk.StringVar(value="1×")
        speed=ttk.Combobox(session,textvariable=self.speed,values=("0.25×","0.5×","1×","2×","5×"),
                           state="readonly",width=6)
        speed.pack(side="left",padx=3)
        speed.bind("<<ComboboxSelected>>",lambda e:self.reanchor())
        self.status=tk.StringVar(value="Nenhum arquivo carregado.")
        filebar=ttk.Frame(self.body,style="Work.TFrame")
        filebar.pack(fill="x")
        self.grade=tk.StringVar(value="Original")
        self.grade_box=ttk.Combobox(filebar,textvariable=self.grade,state="disabled",width=22)
        self.grade_box.pack(side="right",padx=(6,0))
        self.grade_box.bind("<<ComboboxSelected>>",lambda _:self.switch_grade())
        ttk.Label(filebar,textvariable=self.status,style="Sub.TLabel",wraplength=590).pack(side="left",anchor="w")
        self.progress=ttk.Progressbar(self.body,style="Accent.Horizontal.TProgressbar",maximum=100)
        self.progress.pack(fill="x",pady=(5,9))
        self.cards=ttk.Frame(self.body,style="Work.TFrame")
        self.cards.pack(fill="x",pady=(0,8))
        self.card_values=[]
        for i,label in enumerate(("RMS A","RMS B","RMS C","DISPERSÃO RMS","EPISÓDIOS")):
            self.cards.columnconfigure(i,weight=1,uniform="metric")
            tile=ttk.Frame(self.cards,padding=(12,7))
            tile.grid(row=0,column=i,sticky="ew",padx=(0,7 if i<4 else 0))
            ttk.Label(tile,text=label,style="CardLabel.TLabel").pack(anchor="w")
            value=tk.StringVar(value="—")
            ttk.Label(tile,textvariable=value,style="CardValue.TLabel").pack(anchor="w")
            self.card_values.append(value)
        self.actions=ttk.Frame(self.body,padding=(8,6))
        self.actions.pack(fill="x",pady=(0,7))
        self.info=tk.StringVar()
        self.detail=ttk.Label(self.body,textvariable=self.info,wraplength=880,justify="left",padding=(10,5))
        self.detail.pack(fill="x",pady=(0,7))
        self.chart=Chart(self.body,self.zoom)
        self.chart.pack(fill="both",expand=True)
        self.footer=tk.StringVar(value="Ferramenta experimental · resultados por corrente e proxy temporal")
        ttk.Label(self.body,textvariable=self.footer,style="Sub.TLabel",wraplength=790).pack(
            before=self.chart,side="bottom",anchor="w",pady=(6,0))
        self.plot_type="Correntes"
        self.interval=None
        self.mode=tk.StringVar(value="Sem IA")
        self.tick_ms=45
        self.select(0)
        root.protocol("WM_DELETE_WINDOW",self.close)
        root.after(self.tick_ms,self.tick)

    def close(self):
        self.playing=False
        self.worker.shutdown(wait=False,cancel_futures=True)
        self.root.destroy()

    def task(self, function, done, failed=None):
        future=self.worker.submit(function)
        future.add_done_callback(lambda value:self.events.put((value,done,failed)))

    def open(self):
        path=filedialog.askopenfilename(filetypes=[("Dados","*.mat *.csv *.h5 *.hdf5 *.parquet"),("Todos","*.*")])
        if path:
            self.status.set("Lendo o arquivo local…")
            self.task(lambda:catalog(path),lambda data:self.map_dialog(path,data))

    def map_dialog(self,path,data):
        values, defaults=data
        if not values:
            raise ValueError("Não encontrei colunas ou vetores numéricos.")
        dialog=tk.Toplevel(self.root)
        dialog.title("Associar variáveis aos canais")
        frame=ttk.Frame(dialog,padding=17)
        frame.pack(fill="both",expand=True)
        bindings=[]
        for i,label in enumerate(("Tempo (s)","Corrente A","Corrente B","Corrente C")):
            ttk.Label(frame,text=label).grid(row=i,column=0,sticky="w")
            initial=defaults[i] if defaults[i] in values else ""
            box=ttk.Combobox(frame,values=list(values),state="readonly",width=55)
            box.set(initial)
            box.grid(row=i,column=1,padx=6,pady=5)
            bindings.append(box)
        units=ttk.Combobox(frame,values=("A","Sem unidade confirmada"),state="readonly")
        units.set("A")
        units.grid(row=4,column=1,sticky="w")
        ttk.Label(frame,text="Unidades").grid(row=4,column=0,sticky="w")
        prepare=tk.BooleanVar(value=tuple(defaults)==("results.time",
                            *(f"results.signals[3].values[{i}]" for i in (1,2,3))))
        ttk.Checkbutton(frame,text="Preparar grade SRM de 200 kHz e verificar retenção para IA",
                        variable=prepare).grid(row=5,column=0,columnspan=2,sticky="w",pady=4)
        def commit():
            keys=[v.get() for v in bindings]
            if any(k not in values for k in keys) or len(set(keys))!=4:
                messagebox.showerror("Associação","Selecione quatro variáveis distintas.",parent=dialog)
                return
            try:
                signal=checked(values[keys[0]],np.column_stack([values[k] for k in keys[1:]]),
                               Path(path).name,units.get())
            except ValueError as error:
                messagebox.showerror("Dados inválidos",str(error),parent=dialog)
                return
            dialog.destroy()
            signal.mapping=tuple(keys)
            self.load(signal,prepare.get())
        ttk.Button(frame,text="Aplicar associação",command=commit).grid(row=6,column=1,sticky="e",pady=9)

    def load(self,record,prepare=False):
        self.representations={"Original":record}
        selected="Original"
        warning=None
        if prepare:
            try:
                regular,canonical=prepare_srm(record)
                self.representations["Regular · 200 kHz"]=regular
                selected="Regular · 200 kHz"
                if canonical is not None:
                    selected="Canônica · 25 kHz"
                    self.representations[selected]=canonical
                else:
                    warning="Grade regular disponível. Retenção exata por 8 não confirmada. IA a 25 kHz indisponível."
            except ValueError as error:
                warning=str(error)
        self.grade_box.configure(values=list(self.representations),
                                 state="readonly" if len(self.representations)>1 else "disabled")
        self.grade.set(selected)
        self.switch_grade()
        if warning:
            self.footer.set(warning)
            self.info.set(self.info.get()+"\nPreparação: "+warning)

    def switch_grade(self):
        record=self.representations.get(self.grade.get())
        if record is None:
            return
        self.generation+=1
        self.record=record
        self.figures={}
        self.net_value=None
        self.net_clock=None
        self.net_wall=0.
        self.replay=self.playing=False
        self.release=len(record.time)
        self.interval=None
        self.monitor=None
        self.monitor_busy=False
        self.footer.set(record.preparation)
        self.render()

    def select(self,index):
        self.area=index
        self.title.configure(text=AREAS[index])
        self.subtitle.set(SUBTITLES[index])
        for i,nav in enumerate(self.nav):
            nav.configure(bg="#244469" if i==index else "#101f35",fg="white" if i==index else "#cfdef0")
        for widget in self.actions.winfo_children():
            widget.destroy()
        def button(label,action):
            ttk.Button(self.actions,text=label,command=action).pack(side="left",padx=2)
        if index==0:
            button("Abrir diagnóstico",lambda:self.select(2))
            button("Analisar agora",self.analyze)
        elif index==1:
            for label in ("Correntes","FFT","Espectrograma"):
                button(label,lambda p=label:self.change_plot(p))
            button("Intervalo",self.set_interval)
            button("Redefinir zoom",lambda:self.change_interval(None))
            button("Exportar PNG",self.save_png)
        elif index==2:
            button("Analisar arquivo",self.analyze)
            ttk.Combobox(self.actions,textvariable=self.mode,values=("Sem IA",*MODES),state="readonly",
                         width=17).pack(side="left",padx=5)
            button("Ativar modelo",self.activate_model)
            button("Inferir agora",self.infer_now)
        elif index==3:
            for label in ("Correlações","RMS móvel","Resíduo entre ciclos","Mudanças"):
                button(label,lambda p=label:self.change_plot(p))
            button("Intervalo",self.set_interval)
        else:
            button("Relatório PDF",self.save_pdf)
            button("CSV de análise",self.save_csv)
            button("PNG atual",self.save_png)
        self.plot_type={0:"Correntes",1:"Correntes",2:"Mudanças",3:"Correlações",4:"Correntes"}[index]
        self.render()

    def change_plot(self,p):
        self.plot_type=p
        self.render()

    def change_interval(self,interval):
        self.interval=interval
        self.render()

    def set_interval(self):
        if self.record is None:
            return
        from tkinter.simpledialog import askfloat
        first=askfloat("Intervalo","Início em segundos",initialvalue=float(self.record.time[0]))
        if first is None:return
        last=askfloat("Intervalo","Fim em segundos",initialvalue=float(self.record.time[self.visible()-1]) if self.visible() else first)
        if last is not None and first<last:
            self.change_interval((first,last))

    def zoom(self,factor,pan=0):
        if self.record is None or self.visible()<8:
            return
        t=self.record.time
        low,high=float(t[0]),float(t[self.visible()-1])
        a,b=self.interval or (low,high)
        span=max(8/self.record.fs,min((b-a)*factor,high-low))
        center=(a+b)/2+pan*(b-a)
        start=min(max(center-span/2,low),max(low,high-span))
        self.change_interval((start,start+span))

    def visible(self):
        return self.release if self.replay else len(self.record.time)

    def slice(self):
        if self.record is None:return None
        start,end=0,self.visible()
        if self.interval:
            start=np.searchsorted(self.record.time,self.interval[0],side="left")
            end=np.searchsorted(self.record.time,self.interval[1],side="right")
            end=min(end,self.visible())
        return self.record.cut(start,end) if end-start>=8 else None

    def render(self):
        self.view_generation+=1
        token=self.view_generation
        if self.record is None:
            self.info.set("Abra um registro CSV ou MAT, ou explore a demonstração sintética de 0,80 s.")
            self.chart.empty()
            return
        r=self.record
        cut=self.slice()
        end=self.visible()
        self.progress["value"]=100*end/len(r.time)
        self.status.set(f"{r.name}  ·  {r.fs:,.2f} Hz  ·  {'replay' if self.replay else 'arquivo'}  ·  {end:,}/{len(r.time):,} amostras")
        if cut is None:
            self.info.set("Aguardando amostras no intervalo selecionado.")
            for value in self.card_values:value.set("—")
            self.chart.original=None
            self.chart.empty("Aguardando as primeiras amostras")
            return
        q=quality(cut.current)
        jitter=f"Irregularidade máxima: {r.jitter:.2%}"
        valid=np.isfinite(cut.current).all()
        state=self.monitor.current if self.monitor else "Análise de diagnóstico ainda não executada"
        detail=f"Sensores A/B/C: {' / '.join(self.monitor.q if self.monitor and self.replay else q)} · {jitter}"
        detail+=f"\n{r.preparation}"
        card=["—"]*4+[str(len(self.monitor.events)) if self.monitor else "—"]
        if valid and r.jitter<=.05:
            s=stats(cut.current,r.fs)
            card[:4]=[*(f"{v:.2f} {r.units if r.units=='A' else ''}".strip() for v in s["rms"]),
                      f"{s['dispersao']:.2f}%"]
            if self.area==1:
                detail+="\nMédia / pico / integral quadrática: "+ "  |  ".join(
                    f"{k}: {s['media'][i]:.2f} / {s['pico'][i]:.2f} / {s['energia'][i]:.3f}" for i,k in enumerate("ABC"))
        if self.area in (0,2,4):
            events=len(self.monitor.events) if self.monitor else 0
            detail+=f"\n{state} · {events} {'episódio' if events==1 else 'episódios'}"
        if self.area==2:
            detail+="\nIA: "+(f"{self.net.name} · {self.net.version}" if self.net else "indisponível ou desativada")
            if self.net_value is not None:
                detail+=f" · escore pós {self.net_value:.3f} em {self.net_clock:.5f} s · proxy temporal"
                stale=float(r.time[end-1])-self.net_clock>max(.15,.15*float(self.speed.get().rstrip("×")))
                if self.monitor:
                    if stale or "inválida" in state or "insuficientes" in state or "Indeterminado" in state:
                        detail+="\nDecisão cooperativa: indeterminado."
                    elif (self.net_value>=.5)==self.monitor.active:
                        detail+="\nDecisão cooperativa: concordância entre detector e IA."
                    else:
                        detail+="\nDecisão cooperativa: indeterminado por divergência."
            if self.monitor and self.monitor.events:
                detail+="\nÚltimos eventos: "+", ".join(f"{t:.4f} s (z={z:.2f})" for t,_,z in self.monitor.events[-4:])
        self.info.set(detail)
        for value,new in zip(self.card_values,card):value.set(new)
        plot=self.plot_type
        if plot!="Correntes" and plot!="Mudanças" and (not valid or r.jitter>.05):
            hint="Dados não finitos no intervalo." if not valid else "Selecione uma grade regular para esta análise."
            self.chart.empty(f"{plot} indisponível. {hint}")
            self.info.set(detail+"\n"+hint)
            return
        if plot=="Mudanças" and not (self.monitor and len(self.monitor.history)>1):
            plot="Correntes"
        history=list(self.monitor.history) if self.monitor else []
        def calculate():
            x,t=cut.current,cut.time
            suffix=cut.units
            if plot=="FFT":
                f,a=spectrum(x,cut.fs)
                return render(f,a,"Espectro unilateral ABC","Frequência (Hz)",suffix),None
            if plot=="Espectrograma":
                f,t2,z=spectrogram(x[:,1],cut.fs)
                return heatmap(z,t2+t[0],f),None
            if plot=="Correlações":
                lag,a,period=advanced(x,cut.fs)
                text=(f"Período candidato B: {1000*period:.3f} ms" if period else "Sem período candidato.")
                text+=f" · pico B/A: {1000*lag[np.argmax(a[:,1])]:.3f} ms · pico C/A: {1000*lag[np.argmax(a[:,2])]:.3f} ms"
                return render(lag*1000,a,"Autocorrelação de B e correlação cruzada","Atraso (ms)","Coef.",
                              names=("B/B","B/A","C/A")),text
            if plot=="RMS móvel":
                n=max(4,round(cut.fs*.02616))
                if len(x)>=n:
                    cs=np.vstack((np.zeros(3),np.cumsum(x*x,axis=0)))
                    a=np.sqrt(np.maximum((cs[n:]-cs[:-n])/n,0))
                    return render(t[n-1:],a,"Evolução RMS ABC · janela causal 26,16 ms","Tempo (s)",suffix),None
            if plot=="Resíduo entre ciclos":
                _,_,period=advanced(x,cut.fs)
                n=round(period*cut.fs) if period else 0
                if 0<n<len(x):
                    return render(t[n:],x[n:]-x[:-n],"Resíduo ABC em relação ao ciclo anterior","Tempo (s)",suffix),None
            if plot=="Mudanças":
                a=np.array([[v["rms_B_A"] if v["rms_B_A"] is not None else np.nan] for v in history])
                t=np.array([v["tempo_s"] for v in history])
                return render(t,a,"RMS B · evolução da janela de diagnóstico","Tempo (s)",suffix,names=("RMS B",)),None
            return render(t,x,"Correntes trifásicas","Tempo (s)",suffix),None
        def done(result):
            if token==self.view_generation:
                image,note=result
                self.chart.show(image)
                self.figures[plot]=image
                if note:self.footer.set(note)
        if self.replay:
            if self.chart_busy:
                return
            self.chart_busy=True
            generation=self.generation
            def replay_done(result):
                self.chart_busy=False
                if generation==self.generation and self.replay and self.plot_type==plot_requested:
                    self.chart.show(result[0])
                    self.figures[plot]=result[0]
            def replay_failed(error):
                self.chart_busy=False
                self.footer.set(str(error))
            plot_requested=self.plot_type
            self.task(calculate,replay_done,replay_failed)
        else:
            self.task(calculate,done)

    def analyze(self):
        if self.record is None:return
        if self.replay:
            self.status.set("Use Arquivo inteiro para analisar o intervalo offline.")
            return
        r=self.slice()
        if r is None:return
        token=self.generation
        def work():
            m=Monitor(r.fs)
            m.process(r,len(r.time))
            return m
        def done(m):
            if self.generation==token and not self.replay:
                self.monitor=m
                self.render()
        self.task(work,done)

    def activate_model(self):
        mode=self.mode.get()
        self.net=None
        self.net_value=None
        self.net_pending=mode
        self.render()
        if mode=="Sem IA":return
        self.footer.set("Carregando checkpoint local…")
        def done(model):
            if self.mode.get()==mode and self.net_pending==mode:
                self.net=model
                self.footer.set("Modelo neural local ativo · uso experimental, proxy pré/pós.")
                self.render()
        def failed(error):
            self.footer.set("IA indisponível: "+str(error))
        self.task(lambda:Neural(self.model_root,mode),done,failed)

    def infer_now(self):
        self.infer(self.visible() if self.record else 0)

    def infer(self,end):
        if self.net is None or getattr(self,"ai_busy",False) or end<654:return
        rec=self.record
        if abs(rec.fs-25000)>.1 or rec.jitter>.001 or rec.units!="A":
            self.net_value=None
            self.footer.set("IA indisponível nesta grade. Use a grade canônica de 25 kHz, com correntes em A.")
            return
        model=self.net
        token=self.generation
        self.ai_busy=True
        sampled=rec.current[end-654:end].copy()
        def work():
            started=time.perf_counter()
            value=model.infer(sampled,rec.fs,rec.jitter,rec.units)
            return value, (time.perf_counter()-started)*1000
        def done(output):
            self.ai_busy=False
            if token==self.generation and model is self.net:
                self.net_value,self.net_cpu=output
                self.net_clock=float(rec.time[end-1])
                self.footer.set(f"Inferência local: {self.net_cpu:.2f} ms · resultado neural auxiliar")
                if not self.replay:self.render()
        def failed(error):
            self.ai_busy=False
            if token==self.generation:
                self.net_value=None
                self.footer.set(str(error))
        self.task(work,done,failed)

    def reanchor(self):
        if not self.record or not self.replay:return
        r=self.record
        self.anchor=time.perf_counter()
        self.anchor_t=float(r.time[max(0,self.release-1)])

    def play(self):
        if self.record is None:return
        if not self.replay or self.release>=len(self.record.time):
            self.reset()
        self.reanchor()
        self.playing=True

    def pause(self):
        self.playing=False
        if self.replay:self.footer.set("Reprodução pausada.")

    def reset(self):
        self.generation+=1
        self.replay=True
        self.release=0
        self.monitor=Monitor(self.record.fs)
        self.monitor_busy=False
        self.interval=None
        self.net_value=None
        self.net_clock=None
        self.net_wall=0.
        self.figures={}

    def stop(self):
        if self.record:
            self.playing=False
            self.reset()
            self.render()

    def restart(self):
        self.stop()
        self.play()

    def full(self):
        if self.record:
            self.generation+=1
            self.replay=self.playing=False
            self.release=len(self.record.time)
            self.interval=None
            self.monitor=None
            self.net_value=None
            self.net_clock=None
            self.figures={}
            self.render()

    def tick(self):
        while True:
            try: future,done,failed=self.events.get_nowait()
            except queue.Empty:break
            try: done(future.result())
            except Exception as error:
                logging.exception("Operação falhou")
                if failed:failed(error)
                else:self.footer.set(f"Operação interrompida: {error}")
        if self.replay and self.record:
            r=self.record
            if self.playing:
                elapsed=(time.perf_counter()-self.anchor)*float(self.speed.get().rstrip("×"))
                self.release=min(len(r.time),int(np.searchsorted(r.time,self.anchor_t+elapsed,side="right")))
                if self.release>=len(r.time): self.playing=False
                now=time.perf_counter()
                if self.release>=654 and now-self.net_wall>=.10:
                    self.net_wall=now
                    self.infer(self.release)
                self.render()
            if (not self.monitor_busy and self.monitor and self.release>=self.monitor.next
                    and (self.playing or self.release==len(r.time))):
                self.monitor_busy=True
                limit=self.release
                token=self.generation
                import copy
                m=copy.deepcopy(self.monitor)
                def work(r=r,limit=limit,m=m):
                    started=time.perf_counter()
                    m.process(r,limit)
                    return m,1000*(time.perf_counter()-started)
                def done(result,token=token,limit=limit):
                    if token==self.generation:
                        self.monitor,self.monitor_cpu=result
                        self.monitor_busy=False
                        self.footer.set(f"Tempo simulado: {self.record.time[limit-1]:.5f} s · processamento em lote: {self.monitor_cpu:.2f} ms")
                        if not self.playing:self.render()
                def failed(error,token=token):
                    if token==self.generation:
                        self.monitor_busy=False
                        self.playing=False
                        self.footer.set(str(error))
                self.task(work,done,failed)
        self.root.after(self.tick_ms,self.tick)

    def save_png(self):
        if self.chart.original is None:return
        path=filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("Imagem PNG","*.png")])
        if path:self.chart.original.save(path)

    def save_csv(self):
        r=self.slice()
        if r is None:return
        path=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV","*.csv")])
        if not path:return
        data=list(self.monitor.history) if self.monitor and self.monitor.history else []
        if not data:
            s=stats(r.current,r.fs)
            data=[dict(fase=k,rms=float(s["rms"][i]),media=float(s["media"][i]),
                       pico=float(s["pico"][i]),energia=float(s["energia"][i])) for i,k in enumerate("ABC")]
        export_csv(path,data)

    def save_pdf(self):
        if self.record is None or self.slice() is None:return
        target=filedialog.asksaveasfilename(defaultextension=".pdf",filetypes=[("PDF","*.pdf")])
        if not target:return
        r=self.slice()
        lines=[f"Máquina: {self.machine.get()}",f"Fonte: {self.record.name}",
               f"Faixa exibida: {r.time[0]:.6f} a {r.time[-1]:.6f} s",
               f"Taxa estimada: {r.fs:.6f} Hz · unidade: {r.units} · irregularidade: {r.jitter:.2%}",
               f"Preparação temporal: {r.preparation}",
               "Associação tempo / A / B / C: "+(" / ".join(r.mapping) if r.mapping else "demonstrativo interno"),
               *self.info.get().splitlines(),"Método: RMS causal, CUSUM/EWMA e confirmação de três decisões.",
               "Referência inicial: 125 ms do intervalo ou do replay. Presume regime inicial estável.",
               "IA opcional: janela ABC de 654 pontos a 25 kHz. Escore de proxy temporal pré/pós.",
               "Resultados experimentais. A validação em simulação tem poucas origens e depende do regime."]
        if self.net:
            lines.append(f"Modelo selecionado: {self.net.name} · checkpoint {self.net.version}")
            if self.net_value is not None:
                lines.append(f"Último escore pós: {self.net_value:.6f}, em {self.net_clock:.6f} s.")
        if self.monitor:
            lines+=["Episódios:",*(f"{t:.6f} s · {label} · z={z:.3f}" for t,label,z in self.monitor.events)]
        figures=list(self.figures.items())
        self.task(lambda:export_pdf(target,lines,figures),lambda _:self.footer.set("Relatório PDF salvo."))


def configure_tcl():
    tcl=Path(sys.base_prefix)/"tcl"
    if (tcl/"tcl8.6/init.tcl").is_file():
        os.environ.setdefault("TCL_LIBRARY",str(tcl/"tcl8.6"))
        os.environ.setdefault("TK_LIBRARY",str(tcl/"tk8.6"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--modelos",type=Path,default=MODEL_ROOT)
    parser.add_argument("--demo",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--log-dir",type=Path)
    args=parser.parse_args()
    log=args.log_dir or Path(__file__).resolve().parent/"logs"
    log.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=log/"aplicativo.log",level=logging.INFO,encoding="utf-8")
    configure_tcl()
    root=tk.Tk()
    app=App(root,args.modelos)
    if args.demo or args.smoke:app.load(demo())
    if args.smoke:
        def smoke():
            for i in range(5):app.select(i)
            app.play()
            root.after(220, finish)
        def finish():
            assert app.release>0
            app.pause()
            print("Interface e replay iniciados.",flush=True)
            app.close()
        root.after(20,smoke)
    root.mainloop()


if __name__=="__main__":
    main()
