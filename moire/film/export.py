"""Export Moiré's control curves and events for its film, at 1 kHz.
Everything the picture moves by comes from here: the same score and stems as the sound.

  python3 moire/film/export.py   (after python3 moire/moire.py) -> out/moire_film/curves.npz, events.json
"""
import json
import os
import sys

import numpy as np
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'moire'))
import moire as P

OUT = os.path.join(ROOT, 'out', 'moire_film')
CR = 1000


def env(x, sr, hz=12.0):
    b, a = signal.butter(1, hz, fs=sr)
    e = np.sqrt(np.maximum(signal.lfilter(b, a, x.astype(np.float64) ** 2), 0))
    e = e[::sr // CR]
    ref = np.percentile(e[e > 1e-5], 95) if (e > 1e-5).any() else 1.0
    return (e / ref).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    st = np.load(os.path.join(ROOT, 'out', 'moire_stems.npz'))
    sr = int(st['sr'])
    sc = json.load(open(os.path.join(ROOT, 'out', 'moire_score.json')))
    dur = P.NBARS * P.BAR + P.TAIL
    n = int(dur * CR)
    tc = np.arange(n) / CR
    curves = {}
    for k, x in (('kick', st['kick']), ('lead', st['leadL'] + st['leadR']), ('bounce', st['bounceL'] + st['bounceR']),
                 ('ping', st['pingL'] + st['pingR']), ('bed', st['bedL'] + st['bedR']), ('pad', st['pad']),
                 ('deco', st['decoL'] + st['decoR'])):
        e = env(x, sr)
        curves[k] = np.pad(e, (0, max(0, n - len(e))))[:n]
    # the lead's pitch, as the sound has it (the same function the render uses)
    ev = P.lead_score()
    f0, gate = P.lead_curves(tc, ev)
    curves['lead_f0'] = f0.astype(np.float32)
    curves['lead_gate'] = gate.astype(np.float32)
    events = {
        'bpm': P.BPM, 'bar': P.BAR, 'six': P.SIX, 'nbars': P.NBARS, 'dur': dur,
        'grand': P.GRAND, 'align7': P.ALIGN7, 'align5': P.ALIGN5, 'takes': P.TAKES,
        'kicks': sc['kicks'], 'acc7': sc['acc7'], 'acc5': sc['acc5'],
        'lead_notes': [e for e in ev if e[3]],                     # phrase / figure / run notes: (t, midi, dur, True)
        'mel': [(P.tb(71 + 2 * k), m) for k, m in enumerate(P.MEL) if 71 + 2 * k <= 104],
        'sections': {s: [b for b in range(1, P.NBARS + 1) if P.sec(b) == s] for s in ('intro', 'drop1', 'break', 'drop2', 'outro')},
        'fig7': P.FIG7, 'callback': P.CALLBACK, 'hybrid': P.HYBRID, 'hinge': P.HINGE,
        'b_len': P.B_LEN, 'p_len': P.P_LEN,
    }
    np.savez_compressed(os.path.join(OUT, 'curves.npz'), tc=tc.astype(np.float32), **curves)
    json.dump(events, open(os.path.join(OUT, 'events.json'), 'w'))
    print('exported', n, 'samples;', {k: round(float(np.percentile(v, 99)), 2) for k, v in curves.items()})


if __name__ == '__main__':
    main()
