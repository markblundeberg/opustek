"""Checks for the template study, each against the score's intent or a measured range, and a self-test that renders
the study broken on purpose (one fault at a time) and shows the matching check fail. Copy the pattern: every check
you write gets a fault that should trip it, and a check that never fails hasn't been shown to work.

  python3 template/check_study.py              checks out/study.wav (+ stems and score)
  python3 template/check_study.py --selftest   each fault in study.BREAKS against its check
"""
import json
import os
import subprocess
import sys

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, 'tools')]
from measures import load, mono, db, Grid, kick_punch, band_share, phone_share, bed_fill, repetition, loudness

FAULT_TRIPS = {'kick-in-take-away': 'take-away', 'sub-fills-gaps': 'kick punch'}
# A fault may trip other checks too (the held sub also fills the take-away's low end and darkens the tone); what the
# self-test asks is that each fault trips the check written for it.


def run(path='out/study.wav', quiet=False):
    sr, x = load(path)
    sc = json.load(open(path.replace('.wav', '_score.json')))
    g = Grid(sc['bpm'], 0.0)
    y = mono(x)
    res = {}
    L, tp = loudness(x, sr)
    res['loudness'] = (abs(L + 10) <= 0.7 and tp <= -1.0, f'{L:.1f} LUFS (-10 +- 0.7), true peak {tp:.1f} dBTP (<= -1)')
    T = sc['take']
    low = signal.sosfilt(signal.butter(4, [40, 120], 'band', fs=sr, output='sos'), y) ** 2
    lw = lambda a, b: db(low[int(g.tb(a) * sr):int(g.tb(b) * sr)].mean())
    dip = lw(T, T + 1) - lw(T - 4, T)
    res['take-away'] = (dip <= -8, f'40-120 Hz in bar {T} {dip:+.1f} dB re the four bars before (<= -8: the kick stops)')
    kicks = [k for k in sc['kicks'] if g.tb(17) <= k < g.tb(33)]
    kp = kick_punch(y, sr, kicks)
    res['kick punch'] = (kp >= 11.5, f'bars 17-32 {kp:+.1f} dB (>= +11.5, the bottom of the references with bass, +11.5..+14)')
    pres, ph = band_share(y, sr, 3000, 8000, g.tb(17), g.tb(33)), phone_share(y, sr, g.tb(17), g.tb(33))
    res['tone'] = (-30 <= pres <= -18 and ph >= -16, f'3-8 kHz {pres:+.1f} dB of the whole (-30..-18), a phone keeps {ph:+.1f} (>= -16)')
    lit, held = bed_fill(y, sr, g, 17, 31)
    res['bed'] = (lit >= 0.3, f'bars 17-31 lit {lit:.2f} (>= 0.3), held {held:.2f}: the pad fills between the beats')
    s1, s4 = repetition(y, sr, g, 17, 31)
    res['repetition (report only)'] = (True, f'bars 17-31, with the bar 4 later {s4:.3f}: a 4-bar loop and nothing else, so expect it high; '
                                             f'drafts a listener called repetitive read 0.71-0.87')
    if not quiet:
        for k, (p, m) in res.items():
            print(f"  {'PASS' if p else 'FAIL'}  {k}: {m}")
    return res


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        ok = True
        for brk, check in FAULT_TRIPS.items():
            subprocess.run([sys.executable, os.path.join(HERE, 'study.py'), '--break', brk], check=True, stdout=subprocess.DEVNULL)
            r = run(f'out/study_BROKEN-{brk}.wav', quiet=True)
            caught = not r[check][0]
            others = [k for k, (p, _) in r.items() if not p and k != check]
            ok &= caught
            print(f"  {'ok ' if caught else 'BAD'} {brk}: '{check}' {'FAILS' if caught else 'passes'} ({r[check][1]})"
                  + (f'; also failing: {", ".join(others)}' if others else ''))
        print('  ->', 'each fault trips its check' if ok else 'a check is blind')
        sys.exit(0 if ok else 1)
    run(sys.argv[1] if len(sys.argv) > 1 else 'out/study.wav')
