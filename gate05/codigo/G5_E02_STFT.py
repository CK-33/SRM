"""Verificação sintética da STFT causal. Requer NumPy."""
import json
from pathlib import Path

import numpy as np

FS = 25_000
NFFT = 1024
HOP = 41


def stft(signal, length, hop=HOP):
    """Recebe (..., amostras) e devolve tempos finais e amplitude (..., quadro, frequência)."""
    window = np.hanning(length)
    frames = np.lib.stride_tricks.sliding_window_view(signal, length, axis=-1)[..., ::hop, :]
    amplitude = np.abs(np.fft.rfft(frames * window, n=NFFT)) / window.sum()
    amplitude[..., 1:-1] *= 2
    ends = np.arange(frames.shape[-2]) * hop + length - 1
    return ends, amplitude


def main():
    t = np.arange(4096) / FS
    frequency = np.fft.rfftfreq(NFFT, 1 / FS)
    tone = np.sin(2 * np.pi * 1000 * t)
    abc = np.stack([tone, 2 * tone, 0.5 * tone])
    two_tones = tone + np.sin(2 * np.pi * 1200 * t)
    event = 2048
    changed = abc.copy()
    changed[1, event:] *= 2
    results = []
    for length in (164, 327, 654):
        ends, amplitude = stft(abc, length)
        _, modified = stft(changed, length)
        np.testing.assert_array_equal(amplitude[:, ends < event, :], modified[:, ends < event, :])
        peak = int(np.argmax(amplitude[0, 0]))
        assert abs(frequency[peak] - 1000) <= FS / NFFT
        np.testing.assert_allclose(amplitude[:, 0, peak] / amplitude[0, 0, peak], [1, 2, 0.5])
        db = 20 * np.log10(np.maximum(amplitude[:, 0, peak], 1e-12))
        np.testing.assert_allclose(db[1] - db[0], 20 * np.log10(2))
        _, pair = stft(two_tones, length)
        spectrum = np.mean(pair, axis=0)
        peaks = np.flatnonzero((spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])) + 1
        pair_detected = all(np.any(np.abs(frequency[peaks] - f) <= FS / NFFT) for f in (1000, 1200))
        first_full = ends[ends - length + 1 >= event][0]
        results.append(dict(janela_amostras=length, janela_ms=1000 * length / FS,
                            pico_Hz=float(frequency[peak]), dois_picos_1k_1p2k=bool(pair_detected),
                            ate_primeiro_quadro_inteiramente_pos_ms=float(1000 * (first_full-event) / FS),
                            ganho_B_dB=float(db[1] - db[0]), causalidade_verificada=True))
    result = dict(fs_Hz=FS, nfft=NFFT, salto_amostras=HOP, salto_ms=1000*HOP/FS,
                  espacamento_bins_Hz=FS/NFFT, numpy=np.__version__, resultados=results)
    report = json.dumps(result, indent=2, ensure_ascii=False)
    folder = Path(__file__).resolve().parent
    if folder.name == "codigo":
        folder = folder.parent / "resultados" / "metricas"
    (folder / "G5_E02_STFT_Sintetico.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
