import argparse
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from scipy.io import savemat

from charts import render, heatmap
from diagnosis import Monitor
from neural import MODES, Neural
from reports import export_csv, export_pdf
from signals import advanced, catalog, checked, demo, prepare_srm, quality, spectrum, spectrogram, stats

MODEL_ROOT=None


class Campaign(unittest.TestCase):
    def test_prepare_srm_requires_complete_grid_and_exact_retention(self):
        x=demo().current[:100]
        retained=np.repeat(x,8,axis=0)
        t=np.arange(len(retained))/200000
        t_extra=np.insert(t,11,[t[10],t[10]+1e-6])
        extra=np.insert(retained,11,[retained[10],retained[10]],axis=0)
        regular,canonical=prepare_srm(checked(t_extra,extra,"sintético"))
        np.testing.assert_array_equal(regular.current,retained)
        np.testing.assert_array_equal(canonical.current,x)
        self.assertEqual(canonical.fs,25000)
        with self.assertRaisesRegex(ValueError,"incompleta"):
            prepare_srm(checked(np.delete(t,15),np.delete(retained,15,axis=0),"lacuna"))
        broken=retained.copy()
        broken[11,1]+=.01
        bounded,canonical=prepare_srm(checked(t,broken,"sem retenção"))
        self.assertEqual(bounded.fs,200000)
        self.assertIsNone(canonical)

    def test_io_processing_export(self):
        r=demo()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p=Path(tmp)
            x=np.column_stack((r.time,r.current))
            np.savetxt(p/"demo.csv",x,delimiter=",",header="tempo_s,A,B,C",comments="")
            savemat(p/"demo.mat",{"registro":{"I_A":r.current,"Fs_Hz":r.fs,"t0_s":r.time[0]}})
            structures=[{"values":r.time} for _ in range(6)]
            structures[2]={"values":r.current}
            savemat(p/"results_demo.mat",{"results":{"time":r.time,"signals":np.array(structures,dtype=object)}})
            for name in ("demo.csv","demo.mat","results_demo.mat"):
                values,keys=catalog(p/name)
                rr=checked(values[keys[0]],np.column_stack([values[k] for k in keys[1:]]),name)
                np.testing.assert_allclose(rr.current,r.current)
            with self.assertRaisesRegex(ValueError,"reservados"):
                catalog(p/"results_12.mat")
            s=stats(r.current[:6000],r.fs)
            self.assertLess(s["dispersao"],3)
            f,a=spectrum(r.current[:6000],r.fs)
            self.assertLess(abs(f[a[:,0].argmax()]-76.45),5)
            f,t,z=spectrogram(r.current[:6000,1],r.fs)
            self.assertEqual(z.shape,(513,len(t)))
            _,_,period=advanced(r.current[:6000],r.fs)
            self.assertAlmostEqual(period,.01308,delta=.001)
            img=render(r.time,r.current,"Correntes","Tempo (s)","A")
            img.save(p/"correntes.png")
            heatmap(z,t,f).save(p/"stft.png")
            export_pdf(p/"relatorio.pdf",["Teste de relatório","Uma análise completa"],[("Correntes",img)])
            export_csv(p/"metricas.csv",[{"RMS_B":float(s["rms"][1])}])
            self.assertTrue((p/"relatorio.pdf").read_bytes().startswith(b"%PDF"))
            self.assertIn("RMS_B",(p/"metricas.csv").read_text(encoding="utf-8-sig"))

    def test_replay_causal_and_quality(self):
        r=demo()
        altered=r.current.copy()
        altered[5000:]*=8
        a,b,c=Monitor(r.fs),Monitor(r.fs),Monitor(r.fs)
        a.process(r,5000)
        b.process(checked(r.time,altered,"alterado"),5000)
        self.assertEqual(a.history,b.history)
        for end in (6500,9000,12000,len(r.time)):
            a.process(r,end)
        c.process(r,len(r.time))
        self.assertEqual(a.history,c.history)
        self.assertTrue(c.events)
        self.assertTrue(all(.37<t<.50 for t,_,_ in c.events))
        stuck=r.current.copy()
        stuck[:,1]=0
        self.assertIn("imóvel",quality(stuck)[1])
        faulty=Monitor(r.fs)
        faulty.process(checked(r.time,stuck,"sensor preso"),len(stuck))
        self.assertIn("inválida",faulty.current)

    def test_neural_with_real_checkpoints_and_synthetic_windows(self):
        if MODEL_ROOT is None:self.skipTest("Informe --modelos para verificar checkpoints reais.")
        import torch
        from torch import nn
        import sys
        # Compara a implementação incorporada com funções do pipeline original, sem abrir sinais.
        sys.path.insert(0,str(MODEL_ROOT.parent/"codigo"))
        from G5_E09_AWGN import features
        from G5_E07_CNN1D import model_1d
        from torchvision.models import resnet18
        r=demo()
        for mode in MODES:
            neural=Neural(MODEL_ROOT,mode)
            ck=neural.checkpoint
            timings=[]
            with torch.inference_mode():
                for start in (0,5000,12000):
                    raw=r.current[start:start+654]
                    x=torch.from_numpy(np.ascontiguousarray(raw.T,dtype=np.float32))[None]
                    if mode=="CNN 1D":
                        original=model_1d().eval()
                        original.load_state_dict(ck["modelo"])
                        expected=original((x-ck["media"])/ck["desvio"]).sigmoid().item()
                    else:
                        encoder=resnet18(weights=None)
                        encoder.load_state_dict(torch.load(MODEL_ROOT/"pesos_pre_treinados/checkpoints/resnet18-f37072fd.pth",
                                                           map_location="cpu",weights_only=True))
                        encoder.fc=nn.Identity()
                        encoder.eval()
                        s,z=features(x,encoder)
                        z=(z-ck["cnn_media"])/ck["cnn_desvio"]
                        if mode=="Híbrido ABC":
                            z=torch.cat((z,(s-ck["stats_media"])/ck["stats_desvio"]),1)
                        head=nn.Linear(z.shape[1],1)
                        head.load_state_dict(ck["head"])
                        expected=head(z).sigmoid().item()
                    begin=time.perf_counter()
                    actual=neural.infer(raw,r.fs,r.jitter,r.units)
                    timings.append(1000*(time.perf_counter()-begin))
                    self.assertAlmostEqual(actual,expected,places=6)
            peak=""
            try:
                import ctypes
                # PROCESS_MEMORY_COUNTERS: 2 DWORD e 8 SIZE_T.
                class Counters(ctypes.Structure):
                    _fields_=[("cb",ctypes.c_ulong),("faults",ctypes.c_ulong)]+[
                        (f"p{i}",ctypes.c_size_t) for i in range(8)]
                pm=Counters()
                pm.cb=ctypes.sizeof(pm)
                ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
                handle=ctypes.windll.kernel32.GetCurrentProcess()
                ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
                if ctypes.windll.psapi.GetProcessMemoryInfo(handle,ctypes.byref(pm),pm.cb):
                    peak=f" | pico do processo de teste {pm.p0/2**20:.0f} MiB"
            except (AttributeError,OSError):
                pass
            print(f"{mode}: paridade em 3 janelas sintéticas | mediana {np.median(timings):.2f} ms{peak}",flush=True)
        with self.assertRaises(ValueError):
            neural.infer(r.current[:654],200000,0,"A")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--modelos",type=Path)
    args=parser.parse_args()
    MODEL_ROOT=args.modelos
    unittest.main(argv=["test_gate6.py"],verbosity=2)
