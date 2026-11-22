"""Execute localmente: python G5_E02_Verificar_Recortes.py recortes.csv

CSV: origem,amostra,evento,A,B,C. Uma sequência contígua a 25 kHz por origem.
Use somente recortes de treino. 'evento' é o índice da primeira amostra pós-evento.
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from G5_E02_STFT import FS, stft

TRAIN = {"R01", "R02", "R04", "R05", "R07", "R08", "R10", "R11"}
CANDIDATES = ((654, 164), (981, 327))


def load(path):
    groups = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"origem", "amostra", "evento", "A", "B", "C"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("Cabeçalho esperado: origem,amostra,evento,A,B,C")
        for row in reader:
            origin = row["origem"].strip().upper()
            if origin not in TRAIN:
                raise ValueError("O CSV contém uma origem fora do treino.")
            groups[origin].append((int(row["amostra"]), int(row["evento"]),
                                   [float(row[p]) for p in "ABC"]))
    if not groups:
        raise ValueError("CSV sem recortes.")
    return groups


def evaluate(groups):
    results = []
    for origin, rows in sorted(groups.items()):
        indices = np.array([r[0] for r in rows])
        events = {r[1] for r in rows}
        signal = np.array([r[2] for r in rows]).T
        if len(events) != 1 or not np.all(np.diff(indices) == 1) or not np.isfinite(signal).all():
            raise ValueError(f"{origin}: índices, evento ou correntes inválidos.")
        event = events.pop() - int(indices[0])
        for size, length in CANDIDATES:
            for position in (0.25, 0.50, 0.75):
                offset = round(position * size)
                start = event - offset
                if start < 0 or start + size > signal.shape[1]:
                    raise ValueError(f"{origin}: forneça pelo menos 736 amostras de cada lado do evento.")
                ends, amplitude = stft(signal[:, start:start + size], length)
                pre = ends < offset
                post = ends - length + 1 >= offset
                strength = np.max(amplitude, axis=-1)
                def median(mask):
                    return np.median(strength[:, mask], axis=1).tolist() if mask.any() else None
                results.append(dict(origem=origin, segmento=size, janela=length,
                                    posicao_evento=position, quadros=len(ends),
                                    pre=int(pre.sum()), pos=int(post.sum()),
                                    cruzados=int((~(pre | post)).sum()),
                                    pico_espectral_mediano_pre_ABC=median(pre),
                                    pico_espectral_mediano_pos_ABC=median(post)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?")
    parser.add_argument("--sintetico", action="store_true")
    args = parser.parse_args()
    if args.sintetico == bool(args.csv):
        parser.error("Escolha um CSV local ou --sintetico.")
    if args.sintetico:
        time = np.arange(1800) / FS
        signal = np.sin(2 * np.pi * 1000 * time[:, None] + np.arange(3))
        signal[900:, 1] *= 2
        groups = {"SINTETICO": [(i, 900, v) for i, v in enumerate(signal)]}
    else:
        groups = load(args.csv)
    print(json.dumps(dict(fs_Hz=FS, sintetico=args.sintetico,
                          resultados=evaluate(groups)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
