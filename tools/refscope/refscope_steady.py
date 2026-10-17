"""Steady tones in a passage: the median spectrum over time keeps what is held
and drops what moves. Prints the peaks with note names, and the fundamentals
whose harmonic series best explain them.

  python tools/refscope/refscope_steady.py <audio> <stem|mix> <t0> <t1> [fmax]"""
import sys

import librosa
import numpy as np
from scipy import signal

from refscope import SR, load, separate
from refscope_analyze import NOTES, db

a = sys.argv
path, stem, t0, t1 = a[1], a[2], float(a[3]), float(a[4])
fmax = float(a[5]) if len(a) > 5 else 3000
x = load(path) if stem == 'mix' else separate(path)[stem]
y = x[:, int(t0 * SR):int(t1 * SR)].mean(0)
H, P = librosa.effects.hpss(y, margin=2.0)
S = np.abs(librosa.stft(H, n_fft=16384, hop_length=2048)) ** 2
f = librosa.fft_frequencies(sr=SR, n_fft=16384)
med = db(np.median(S, 1))
k = (f > 25) & (f < fmax)
base = signal.medfilt(med, 151)
pk, pr = signal.find_peaks(med[k] - base[k], prominence=8, distance=4)
fk, lv = f[k][pk], med[k][pk]
order = np.argsort(lv)[::-1][:24]
nm = lambda fr: (lambda mi: f"{NOTES[int(round(mi)) % 12]}{int(round(mi)) // 12 - 1}{'+' if mi - round(mi) > 0 else '-'}{abs(mi - round(mi)) * 100:.0f}c")(69 + 12 * np.log2(fr / 440))
print(f'{stem} {t0:.1f}-{t1:.1f} s: held peaks (dB re loudest; prominence over the local median)')
top = lv[order].max()
for i in sorted(order, key=lambda i: fk[i]):
    print(f'  {fk[i]:8.1f} Hz  {nm(fk[i]):10s} {lv[i] - top:6.1f} dB')
# fundamentals: score candidate f0 by summed prominence of peaks within 1.5% of its harmonics
cands = []
for c in fk[order]:
    for div in (1, 2, 3, 4):
        f0 = c / div
        if f0 < 25:
            continue
        hits = [lv[i] for i in range(len(fk)) if abs(fk[i] / f0 - round(fk[i] / f0)) < 0.015 and round(fk[i] / f0) <= 12]
        cands.append((len(hits), np.mean(hits) - top if hits else -99, f0))
cands.sort(key=lambda c: (-c[0], -c[1]))
seen = []
print('fundamentals explaining the most peaks:')
for n, l, f0 in cands:
    if any(abs(f0 / s - 1) < 0.02 for s in seen):
        continue
    seen.append(f0)
    print(f'  {f0:7.1f} Hz {nm(f0):10s} explains {n} peaks')
    if len(seen) == 6:
        break
