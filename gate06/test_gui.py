from pathlib import Path
import tempfile
import time
import tkinter as tk
from unittest.mock import patch

from app import App, configure_tcl
from signals import catalog


def wait(root, predicate, seconds=8):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        root.update()
        if predicate():
            return
        time.sleep(.012)
    raise AssertionError("A operação de interface não concluiu.")


def main():
    configure_tcl()
    root=tk.Tk()
    root.withdraw()
    app=App(root)
    try:
        path=Path(__file__).parent/"demonstracao/onda_srm.csv"
        app.map_dialog(path,catalog(path))
        dialog=next(w for w in root.winfo_children() if isinstance(w,tk.Toplevel))
        frame=dialog.winfo_children()[0]
        next(w for w in frame.winfo_children() if w.winfo_class()=="TButton").invoke()
        assert app.record.current.shape==(20000,3)
        wait(root,lambda:app.chart.original is not None)
        for area in range(5):
            app.select(area)
            root.update()
        app.select(1)
        for view in ("FFT","Espectrograma"):
            app.change_plot(view)
            wait(root,lambda:view in app.figures)
        app.change_plot("Correntes")
        app.change_interval((.15,.50))
        assert app.slice().time[0]>=.15 and app.slice().time[-1]<=.50
        app.change_interval(None)
        app.select(2)
        app.analyze()
        wait(root,lambda:app.monitor is not None and bool(app.monitor.events))
        assert app.net is None
        if (app.model_root/"checkpoints").is_dir():
            app.mode.set("CNN 1D")
            app.activate_model()
            wait(root,lambda:app.net is not None,20)
            app.infer_now()
            wait(root,lambda:app.net_value is not None)
            assert 0<=app.net_value<=1
            app.mode.set("Sem IA")
            app.activate_model()
        app.play()
        wait(root,lambda:app.release>=1400)
        app.pause()
        paused=app.release
        until=time.monotonic()+.14
        while time.monotonic()<until:
            root.update()
            time.sleep(.01)
        assert app.release==paused
        app.play()
        wait(root,lambda:app.release>paused)
        wait(root,lambda:app.release==20000 and not app.monitor_busy and app.monitor and bool(app.monitor.events))
        app.select(4)
        wait(root,lambda:app.chart.original is not None)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p=Path(tmp)
            for name,action in (("relatorio.pdf",app.save_pdf),("analise.csv",app.save_csv),("grafico.png",app.save_png)):
                with patch("app.filedialog.asksaveasfilename",return_value=str(p/name)):
                    action()
                wait(root,lambda:(p/name).is_file() and (p/name).stat().st_size>12)
            assert (p/"relatorio.pdf").read_bytes().startswith(b"%PDF")
        app.stop()
        assert not app.playing and app.release==0
        print("GUI OK: associação CSV, cinco áreas, FFT/STFT, intervalo, diagnóstico, pausa/retomada, PDF/CSV/PNG.")
    finally:
        app.close()


if __name__=="__main__":
    main()
