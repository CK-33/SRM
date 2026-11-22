from dataclasses import dataclass
from pathlib import Path
import csv
import re
import numpy as np
from scipy import io, signal


@dataclass
class Record:
    time: np.ndarray
    current: np.ndarray
    name: str
    fs: float
    units: str
    jitter: float
    mapping: tuple = ()

    def cut(self, a, b):
        return Record(self.time[a:b], self.current[a:b], self.name, self.fs, self.units, self.jitter, self.mapping)


def checked(t, x, name, units="A"):
    t = np.asarray(t, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    if x.shape == (3, len(t)) and len(t) != 3:
        x = x.T
    if len(t) < 8 or x.shape != (len(t), 3):
        raise ValueError("É necessário um vetor de tempo e três correntes de mesmo comprimento.")
    dt = np.diff(t)
    if not np.isfinite(t).all() or np.any(dt <= 0):
        raise ValueError("Timestamps devem ser finitos e estritamente crescentes.")
    step = float(np.median(dt))
    return Record(t, x, name, 1/step, units, float(np.max(abs(dt-step))/step))


def catalog(path):
    path = Path(path)
    if re.search(r'(?i)(?:^|[^a-z0-9])(?:r0?6|r12|results?[_ -]?(?:0?6|12))(?:[^a-z0-9]|$)', path.stem):
        raise ValueError("R06 e R12 são reservados nesta campanha.")
    ext = path.suffix.lower()
    if ext == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            names = reader.fieldnames or []
            if len(names) != len(set(names)):
                raise ValueError("Há cabeçalhos CSV duplicados.")
            parse = lambda text: float(text.strip().replace(",", ".")) if text and text.strip() else np.nan
            columns = {key: [] for key in names}
            for row in reader:
                for key in names:
                    try:
                        columns[key].append(parse(row.get(key)))
                    except ValueError:
                        columns[key].append(np.nan)
            values = {k: np.array(v) for k,v in columns.items()}
            aliases = {k.lower().strip(): k for k in names}
            return values, tuple(aliases.get(k,"") for k in ("tempo_s","a","b","c")) if "tempo_s" in aliases else tuple(aliases.get(k,"") for k in ("time","a","b","c"))
    values = {}
    def add(key, raw):
        try:
            x = np.squeeze(np.asarray(raw, dtype=float))
        except (TypeError, ValueError):
            return
        if x.ndim == 1 and x.size >= 8:
            values[key] = x
        elif x.ndim == 2 and min(x.shape) <= 16:
            x = x if len(x) >= x.shape[1] else x.T
            for i,v in enumerate(x.T,1):
                values[f"{key}[{i}]"] = v

    def visit(raw,key=""):
        if isinstance(raw,dict):
            for k,v in raw.items():
                if not k.startswith("__"):
                    visit(v,f"{key}.{k}" if key else k)
        elif isinstance(raw,(list,tuple)) and raw and isinstance(raw[0],dict):
            for j,v in enumerate(raw,1):
                visit(v,f"{key}[{j}]")
        else:
            add(key,raw)
    if ext == ".mat":
        try:
            raw = io.loadmat(path,simplify_cells=True)
        except (NotImplementedError,ValueError):
            return hdf_catalog(path)
        visit(raw)
        if isinstance(raw.get("registro"),dict):
            r=raw["registro"]
            x=np.asarray(r.get("I_A"))
            if x.ndim==2 and "Fs_Hz" in r and "t0_s" in r:
                values["registro.tempo_s"]=float(r["t0_s"])+np.arange(max(x.shape))/float(r["Fs_Hz"])
                return values,("registro.tempo_s",*(f"registro.I_A[{i}]" for i in (1,2,3)))
        return values,("results.time",*(f"results.signals[3].values[{i}]" for i in (1,2,3)))
    if ext in (".h5",".hdf5"):
        return hdf_catalog(path)
    if ext == ".parquet":
        try:
            from pyarrow.parquet import read_table
        except ImportError as e:
            raise ValueError("Parquet exige PyArrow no ambiente local.") from e
        for k,v in read_table(path).to_pydict().items():
            add(k,v)
        return values,("tempo_s","A","B","C")
    raise ValueError("Formato não suportado.")


def hdf_catalog(path):
    try:
        import h5py
    except ImportError as e:
        raise ValueError("HDF5 e MAT v7.3 exigem h5py no ambiente local.") from e
    values={}
    with h5py.File(path,"r") as f:
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset) and not name.startswith("#") and obj.dtype.kind in "fiu":
                x=np.squeeze(obj[()])
                if x.ndim==1:
                    values[name]=x
                elif x.ndim==2 and min(x.shape)<=16:
                    if x.shape[0]<x.shape[1]:
                        x=x.T
                    for i,v in enumerate(x.T,1):
                        values[f"{name}[{i}]"]=v
        f.visititems(visit)
    return values,("tempo_s","A","B","C")


def demo():
    t=np.arange(20000)/25000
    phase=2*np.pi*76.45*t[:,None]+np.arange(3)[None,:]*2*np.pi/3
    x=10*np.sin(phase)+.9*np.sin(3*phase)
    x[:,1]*=1+2*(1-np.exp(-np.maximum(t-.37,0)/.012))
    return checked(t,x,"Demonstração sintética")


def stats(x,fs):
    rms=np.sqrt(np.mean(x*x,axis=0))
    peak=np.max(abs(x),axis=0)
    return dict(rms=rms,media=x.mean(axis=0),desvio=x.std(axis=0),pico=peak,
                crista=peak/np.maximum(rms,1e-12),energia=np.sum(x*x,axis=0)/fs,
                rms_total=float(np.sqrt(np.mean(rms*rms))),
                dispersao=float(100*np.std(rms,ddof=1)/max(rms.mean(),1e-12)))


def quality(x):
    result=[]
    for p in x.T:
        if not np.isfinite(p).all():
            result.append("Ausentes ou não finitos")
        elif np.ptp(p)<=1e-8:
            result.append("Canal imóvel")
        elif len(p)>=16 and max(np.mean(p==np.min(p)),np.mean(p==np.max(p)))>.08:
            result.append("Possível saturação")
        else:
            result.append("Sem alerta")
    return result


def spectrum(x,fs):
    w=np.hanning(len(x))
    a=np.abs(np.fft.rfft((x-x.mean(axis=0))*w[:,None],axis=0))/w.sum()
    a[1:-1 if len(x)%2==0 else None]*=2
    return np.fft.rfftfreq(len(x),1/fs),a


def spectrogram(x,fs):
    n=min(327,len(x))
    f,t,z=signal.stft(x,fs=fs,window="hann",nperseg=n,noverlap=n-min(41,n-1),
                     nfft=max(1024,n),boundary=None,padded=False)
    return f,t,20*np.log10(np.maximum(abs(z),1e-6))


def advanced(x,fs):
    u=x-x.mean(axis=0)
    n=len(x)
    cap=min(n-1,round(fs*.1))
    lags=signal.correlation_lags(n,n)
    select=abs(lags)<=cap
    ac=signal.correlate(u[:,1],u[:,1],method="fft")[select]
    ac=ac/max(np.dot(u[:,1],u[:,1]),1e-12)
    positive=lags[select]>0
    peak,_=signal.find_peaks(ac[positive],prominence=.08)
    period=float(lags[select][positive][peak[0]]/fs) if len(peak) else None
    cross=[]
    for phase in (1,2):
        y=signal.correlate(u[:,phase],u[:,0],method="fft")[select]
        cross.append(y/max(np.linalg.norm(u[:,phase])*np.linalg.norm(u[:,0]),1e-12))
    return lags[select]/fs,np.column_stack((ac,*cross)),period
