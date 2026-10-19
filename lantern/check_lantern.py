"""Checks for Lantern, one per thing the piece intends, and a self-test that renders it broken on purpose (one fault
at a time, lantern.BREAKS) and shows the matching check fail.

  python3 lantern/check_lantern.py              checks out/lantern.wav (+ stems and score)
  python3 lantern/check_lantern.py --selftest   each fault against its check, several at once (--jobs N, --keep)
"""
import functools
import json
import os
import sys

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, 'tools'), HERE]
from measures import load, mono, db, Grid, kick_punch, band_share, phone_share, loudest_second, ending_steps, loudness
import lantern as P
import kit

MEM_GB = 2.5   # one broken render's peak memory (1.4 GB) plus its check's
FAULT_TRIPS = {'sharp-kalimba': 'tuning', 'wrong-note': 'key', 'kick-in-break': 'break',
               'buried-tune': 'tune audible', 'boom-at-end': 'ending'}

HZ = lambda m: 440.0 * 2 ** ((m - 69) / 12)


def note_cents(x, sr, t0, hz, win=0.3, skip=0.03):
    """Where the strongest partial within +-60 cents of hz actually sits, in cents re hz."""
    seg = x[int((t0 + skip) * sr):int((t0 + skip + win) * sr)]
    if len(seg) < int(0.1 * sr):
        return None
    nfft = 1 << 19
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), nfft))
    f = np.fft.rfftfreq(nfft, 1 / sr)
    lo, hi = np.searchsorted(f, [hz * 2 ** (-60 / 1200), hz * 2 ** (60 / 1200)])
    k = lo + int(np.argmax(S[lo:hi]))
    a, b, c = np.log(S[k - 1:k + 2] + 1e-20)
    kk = k + 0.5 * (a - c) / (a - 2 * b + c)
    return 1200 * np.log2(kk * sr / nfft / hz)


def band_power(s, sr, lo, hi, spans):
    y = signal.sosfilt(signal.butter(4, [lo, hi], 'band', fs=sr, output='sos'), s) ** 2
    return db(np.mean(np.concatenate([y[int(a * sr):int(b * sr)] for a, b in spans])))


def run(path='out/lantern.wav', quiet=False):
    sr, x = load(path)
    base = path[:-4]
    sc = json.load(open(base + '_score.json'))
    st = kit.load_stems(base + '_stems.npz')
    g = Grid(sc['bpm'], 0.0)
    y = mono(x)
    res = {}

    L, tp = loudness(x, sr)
    res['loudness'] = (abs(L + 12) <= 0.7 and tp <= -1.0, f'{L:.1f} LUFS (-12 +- 0.7: downtempo, not club-loud), true peak {tp:.1f} dBTP (<= -1)')

    # tuning: the kalimba stem (before the mix), each tune note held at least 0.3 s, its fundamental vs the score
    kal = st['kalimba_raw']
    errs = [note_cents(kal, sr, e['t'], HZ(e['midi'])) for e in sc['tune'] if e['dur'] >= 0.3]
    errs = np.array([e for e in errs if e is not None])
    med, worst = np.median(np.abs(errs)), np.max(np.abs(errs))
    res['tuning'] = (med <= 5 and worst <= 12, f'kalimba, {len(errs)} notes: median |error| {med:.1f} cents (<= 5), worst {worst:.1f} (<= 12)')

    # key: every pitch the composition plays, in D major
    pcs = set(sc['key_pcs'])
    played = [e['midi'] for e in sc['tune']] + [e['midi'] for e in sc['bass']] + [m for e in sc['epiano'] for m in e['notes']]
    out = sorted({m for m in played if m % 12 not in pcs})
    res['key'] = (not out, f'{len(played)} notes, out of D major: {out if out else "none"}')

    # the break: the kick and bass stop in bars 13-14, so the low end falls away
    b0, b1 = sc['break']
    low = signal.sosfilt(signal.butter(4, [40, 120], 'band', fs=sr, output='sos'), y) ** 2
    lw = lambda a, b: db(low[int(g.tb(a) * sr):int(g.tb(b) * sr)].mean())
    dip = lw(b0, b1 + 1) - lw(b0 - 4, b0)
    res['break'] = (dip <= -10, f'40-120 Hz in bars {b0}-{b1} {dip:+.1f} dB re bars {b0 - 4}-{b0 - 1} (<= -10: the floor drops out)')

    # the tune is the point: the kalimba must stand over the piano in the kalimba's own band while it plays
    spans = [(e['t'], e['t'] + max(e['dur'], 0.25)) for e in sc['tune'] if e['bar'] in range(5, 13) or e['bar'] in range(17, 21)]
    over = band_power(st['kalimba'], sr, 300, 1400, spans) - band_power(st['epiano'], sr, 300, 1400, spans)
    res['tune audible'] = (over >= 3, f'kalimba over the piano in 300-1400 Hz while it plays: {over:+.1f} dB (>= +3)')

    # the ending: after home (bar 21) the level only falls, no boom out of the fade
    lv, rise, fall = ending_steps(y, sr, g.tb(sc['home_bar']), skip=1)
    res['ending'] = (rise <= 4, f'after bar {sc["home_bar"]}: biggest half-second rise {rise:+.1f} dB (<= +4; the arpeggio adds a little), '
                                f'biggest fall while audible {fall:.1f} dB')

    # report-only measures: orientation, not targets from a reference in this genre
    main = (g.tb(5), g.tb(13))
    kp = kick_punch(y, sr, [k for k in sc['kicks'] if main[0] <= k < main[1]])
    ls, lt = loudest_second(y, sr)
    res['report: tone'] = (True, f'bars 5-12: 3-8 kHz {band_share(y, sr, 3000, 8000, *main):+.1f} dB of the whole, a phone keeps '
                                 f'{phone_share(y, sr, *main):+.1f}, kick punch {kp:+.1f} dB, loudest 2.5-4 kHz second {ls:.1f} dBFS at {lt:.1f} s')
    lv_sec = {f'{a}-{b}': db(np.mean(y[int(g.tb(a) * sr):int(g.tb(b + 1) * sr)] ** 2)) for a, b in [(1, 4), (5, 12), (13, 14), (15, 16), (17, 20)]}
    res['report: sections'] = (True, 'level by bars ' + ', '.join(f'{k} {v:.1f}' for k, v in lv_sec.items()) + ' dBFS')
    if not quiet:
        for k, (p, m) in res.items():
            print(f"  {'PASS' if p else 'FAIL'}  {k}: {m}")
    return res


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        jobs = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else None
        ok = kit.selftest(FAULT_TRIPS, lambda brk: [sys.executable, os.path.join(HERE, 'lantern.py'), '--break', brk],
                          lambda brk: f'out/lantern_BROKEN-{brk}.wav', functools.partial(run, quiet=True),
                          'out/lantern.wav', mem_gb=MEM_GB, jobs=jobs, keep='--keep' in sys.argv)
        sys.exit(0 if ok else 1)
    args = [a for a in sys.argv[1:] if not a.startswith('--') and not a.isdigit()]
    sys.exit(0 if all(p for p, _ in run(args[0] if args else 'out/lantern.wav').values()) else 1)
