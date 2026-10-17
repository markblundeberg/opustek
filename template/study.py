"""A one-minute study to copy and change: a kick, off-beat hats, a clap, a pumping bass and a chord pad, mixed and
mastered to -10 LUFS. It is deliberately plain: the point is the plumbing (a score as data, instruments from the
kit, stems and a score file written beside the mix, a fault you can switch on so the checks can be shown failing).

  python3 template/study.py                    -> out/study.wav, out/study_stems.npz, out/study_score.json
                                                  (about 10 s and 1.6 GB of memory)
  python3 template/study.py --break NAME       the same with one fault (see BREAKS), for check_study.py --selftest

Form (124 bpm, E minor, 32 bars, about a minute): 1-8 kick and hats; 9-15 the bass and clap join; 16 the take-away
(no kick); 17-32 everything, with the pad.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'engine')]
import dsp                      # the kernel: saws, filters, delays, reverb, limiter, loudness
import song as S                # Engine's instruments and helpers (kick_hit, clap_hit, spaghetti, place, sidechain, ...)

SR, CR = 48000, 1000
BPM = 124.0
BEAT = 60 / BPM
BAR = 4 * BEAT
NBARS, TAIL = 32, 3.0
HZ = lambda m: 440.0 * 2 ** ((m - 69) / 12)
tb = lambda bar, beat=0.0: (bar - 1) * BAR + beat * BEAT
R = lambda a, b: list(range(a, b + 1))

# --- the score, as data
TAKE = 16
KICK = [b for b in R(1, NBARS) if b != TAKE]
CLAP = [b for b in R(9, NBARS) if b != TAKE]
BASS = [b for b in R(9, NBARS) if b != TAKE]
PAD = R(17, NBARS)
ROOTS = [40, 36, 43, 38]                       # E2 C2 G2 D2, one per bar of a 4-bar loop
CHORDS = [[64, 67, 71], [60, 64, 67], [62, 67, 71], [62, 66, 69]]   # Em C G D
BREAKS = {'kick-in-take-away': 'the kick keeps playing through bar 16', 'sub-fills-gaps': 'a held sub-bass under the bass, not ducked, filling the gaps between kicks'}


def render(brk=None):
    rng = np.random.default_rng(6)
    dur = NBARS * BAR + TAIL
    N = int(dur * SR)
    kick_bars = R(1, NBARS) if brk == 'kick-in-take-away' else KICK
    kicks = [tb(b, q) for b in kick_bars for q in range(4)]

    kick = np.zeros(N)
    S.place(kick, S.kick_hit(SR), [(t, 1.0) for t in kicks], SR, 0.8)
    clap = np.zeros(N)
    S.place(clap, S.clap_hit(SR), [(tb(b, q), 1.0) for b in CLAP for q in (1, 3)], SR, 0.35)
    hL, hR = np.zeros(N), np.zeros(N)
    for b in R(1, NBARS):
        for q in range(4):
            h = S.spaghetti(SR, rng) * rng.uniform(0.8, 1.0)
            p = 0.5 + rng.uniform(-0.3, 0.3)
            S.place(hL, h, [(tb(b, q + 0.5), 1 - p)], SR, 0.5)
            S.place(hR, h, [(tb(b, q + 0.5), p)], SR, 0.5)

    # the bass: off-beat eighths on the bar's root, a saw through a ladder whose cutoff is struck on each note
    t = np.arange(N) / SR
    f = np.full(N, HZ(ROOTS[0]))
    cut = np.full(N, 120.0)
    gate = np.zeros(N)
    for b in BASS:
        for q in range(4):
            t0 = tb(b, q + 0.5)
            d = 0.22
            i, j = int(t0 * SR), int((t0 + d) * SR)
            f[i:] = HZ(ROOTS[(b - 1) % 4])
            u = t[i:j] - t0
            gate[i:j] = np.minimum(1, u / 0.003) * np.clip((d - u) / 0.02, 0, 1)
            cut[i:j] = np.maximum(cut[i:j], 120 + 1400 * np.exp(-u / 0.06))
    bass = dsp.ladder(dsp.saw(f, SR), cut, SR, res=0.35, drive=1.5) * gate

    if brk == 'sub-fills-gaps':                             # the fault that cost a round on Moiré
        bass = bass + 0.5 * np.sin(2 * np.pi * np.cumsum(f / 2) / SR) * (t >= tb(BASS[0])) / 0.3
    # the pad: three detuned saws per chord note, a slow filter, held for the bar
    pad = np.zeros(N)
    for b in PAD:
        i, j = int(tb(b) * SR), int(tb(b + 1) * SR)
        for m in CHORDS[(b - 1) % 4]:
            for c in (-7, 0, 6):
                pad[i:j] += dsp.saw(np.full(j - i, HZ(m) * 2 ** (c / 1200)), SR, phase=rng.uniform())
    pad = dsp.ladder(pad, 900 + 500 * np.sin(2 * np.pi * t / (8 * BAR)), SR, res=0.1) * 0.09
    pad *= np.clip((t - tb(PAD[0])) / (2 * BAR), 0, 1)

    sc = S.sidechain(kicks, N, SR, 0.6, rel=0.15)
    pl, pr = dsp.reverb(pad * 0.6, pad * 0.6, SR, rt60=3.0, damp_hz=5000)
    L = kick + clap + hL * 0.5 + bass * 0.3 * sc + (pad + pl * 0.5) * sc
    Rr = kick + clap + hR * 0.5 + bass * 0.3 * sc + (pad + pr * 0.5) * sc
    mix = np.stack([L, Rr], 1)
    mix = np.stack([dsp.butter(mix[:, 0], 15, SR, 'high', 2), dsp.butter(mix[:, 1], 15, SR, 'high', 2)], 1)
    mix = mix + 10 ** (4 / 20) * np.stack([dsp.butter(mix[:, c], 4000, SR, 'high', 2) for c in range(2)], 1) - np.stack([dsp.butter(mix[:, c], 4000, SR, 'high', 2) for c in range(2)], 1)   # a +4 dB shelf above 4 kHz
    for _ in range(6):                                      # master: -10 LUFS through a true-peak limiter at -1.2 dBTP
        mix = dsp.limiter(mix * 10 ** ((-10.0 - dsp.lufs(mix, SR)) / 20), SR, ceiling_db=-1.2, rel_ms=60)
    mix *= np.clip((dur - t) / 1.5, 0, 1)[:, None]
    stems = {'kick': kick, 'clap': clap, 'hats': (hL + hR) / 2, 'bass': bass, 'pad': pad}
    score = {'bpm': BPM, 'bar': BAR, 'nbars': NBARS, 'take': TAKE, 'kicks': kicks, 'clap_bars': CLAP, 'bass_bars': BASS, 'pad_bars': PAD}
    return mix, stems, score


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--break', dest='brk', choices=list(BREAKS))
    a = ap.parse_args()
    name = 'study' + (f'_BROKEN-{a.brk}' if a.brk else '')
    os.makedirs('out', exist_ok=True)
    mix, stems, score = render(a.brk)
    wavfile.write(f'out/{name}.wav', SR, mix.astype(np.float32))
    np.savez_compressed(f'out/{name}_stems.npz', sr=SR, **stems)
    json.dump(score, open(f'out/{name}_score.json', 'w'))
    print(f'wrote out/{name}.wav: {len(mix) / SR:.1f} s, {dsp.lufs(mix, SR):.1f} LUFS, {dsp.true_peak_db(mix):.1f} dBTP')
