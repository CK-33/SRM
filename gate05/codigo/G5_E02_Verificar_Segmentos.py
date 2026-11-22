"""Compara janelas e segmentos da STFT em um cenário sintético a 25 kHz."""
import json
from pathlib import Path

import numpy as np

from G5_E02_STFT import FS, HOP, NFFT, stft


def main():
    rows = []
    bins = np.fft.rfftfreq(NFFT, 1 / FS)
    band = (bins >= 850) & (bins <= 1150)
    for cycles, size in ((0.5, 164), (1, 327), (2, 654), (3, 981), (5, 1635)):
        event = size // 2
        t = np.arange(size) / FS
        phases = np.arange(3)[:, None] * (2 * np.pi / 3)
        before = np.sin(2 * np.pi * 1000 * t + phases)
        after = before.copy()
        after[1, event:] *= 2
        for length in (164, 327, 654):
            row = dict(ciclos=cycles, segmento_amostras=size, janela_amostras=length)
            if size < length:
                rows.append(dict(row, quadros=0))
                continue
            ends, reference = stft(before, length)
            _, changed = stft(after, length)
            starts = ends - length + 1
            pre = ends < event
            post = starts >= event
            mixed = ~(pre | post)
            assert changed.shape == (3, 1 + (size - length) // HOP, NFFT // 2 + 1)
            np.testing.assert_array_equal(reference[:, pre], changed[:, pre])
            np.testing.assert_array_equal(reference[[0, 2]], changed[[0, 2]])
            ratios = (np.linalg.norm(changed[1][:, band], axis=-1) /
                      np.linalg.norm(reference[1][:, band], axis=-1))
            np.testing.assert_allclose(ratios[post], 2)
            rows.append(dict(row, quadros=int(ends.size), pre=int(pre.sum()),
                             cruzam_evento=int(mixed.sum()), pos=int(post.sum()),
                             b_dB_por_quadro=np.round(20 * np.log10(ratios), 3).tolist(),
                             amostras_apos_ultimo_quadro=int(size - 1 - ends[-1])))
    report = json.dumps(dict(fs_Hz=FS, salto_amostras=HOP, nfft=NFFT,
                             cenario="Degrau de amplitude 2x em B no meio do segmento", resultados=rows),
                        indent=2, ensure_ascii=False)
    folder = Path(__file__).resolve().parent
    if folder.name == "codigo":
        folder = folder.parent / "resultados" / "metricas"
    (folder / "G5_E02_Verificacao_Segmentos.json").write_text(report + "\n", encoding="utf-8")
    for row in rows:
        print(row["ciclos"], row["janela_amostras"], row["quadros"],
              row.get("pre"), row.get("cruzam_evento"), row.get("pos"))


if __name__ == "__main__":
    main()
