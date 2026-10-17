"""Checks on song.py's render: each compares something measured on the audio
(often on one stem, so a neighbour can't pass it) with what the score intends.
Each was written before the render it judges; --selftest renders the song
broken on purpose, once per fault, and shows the matching check fail.

  python3 check_song.py              checks out/song.wav (+ its stems and score)
  python3 check_song.py --selftest   seven broken renders, ~1 min each, 5 GB each

Some ranges were measured on two commercial techno recordings, section by
section (they are not included): the low-mid share (120-300 Hz, -8.4..-12.3
dB of the total), the brightest upper-mid share (2.5-6 kHz, -22 dB), and the
balance of lows and highs against the mids."""
import json
import os
import subprocess
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the shared kernel, ../dsp.py
import dsp
import song as S
from song import BEAT, BAR, HZ, tb

LUFS_TARGET, LUFS_TOL, TP_MAX = -10.0, 0.6, -1.0


def lvl(x, sr, a, b):
    return 10 * np.log10((x[int(a * sr):int(b * sr)] ** 2).mean() + 1e-20)


def engine_early(st):
    """Mark: the engine should be prominent well before 1:10. By bars 19-20
    (0:35-0:39) it is within 6 dB of its level in the solo (bars 33-44)."""
    sr = int(st['sr'])
    x = (st['droneL'] + st['droneR']) / 2
    d = lvl(x, sr, tb(19), tb(21)) - lvl(x, sr, tb(33), tb(45))
    return d >= -6, f'bars 19-20 at {d:+.1f} dB of the solo (starts {tb(19):.0f} s)'


def seam(st):
    """Mark's seam: after the solo it goes quieter, then builds. Bars 49-50:
    the plucks alone (no doots, no engine; plucks present); bars 51-52: the
    doots arrive, still no engine; bars 53-55: the engine wakes."""
    sr = int(st['sr'])
    e = (st['droneL'] + st['droneR']) / 2
    dl = lambda a, b: lvl(st['doots'], sr, tb(a), tb(b))
    el = lambda a, b: lvl(e, sr, tb(a), tb(b))
    ref_d, ref_e = dl(57, 61), el(33, 45)
    ks = lvl(st['ks'], sr, tb(49), tb(51))
    ok = (dl(49, 51) < ref_d - 30 and el(49.3, 53) < ref_e - 30 and ks > -80
          and dl(51, 53) > ref_d - 25 and el(54, 56) > ref_e - 15)
    return ok, (f'49-50: doots {dl(49, 51) - ref_d:+.0f} dB, engine {el(49.3, 53) - ref_e:+.0f} dB, plucks {ks:.0f} dBFS; '
                f'51-52 doots {dl(51, 53) - ref_d:+.0f}; 54-55 engine {el(54, 56) - ref_e:+.0f} dB')


def no_wall(mix, sr):
    """Mark: the doots' return should start with one note, not a wall. The mix
    rises no more than 2 dB per beat over bar 51's first two beats, relative
    to the plucks alone (bars 49-50)."""
    m = mix.mean(1)
    lv = lambda a, b: 10 * np.log10((m[int(a * sr):int(b * sr)] ** 2).mean() + 1e-20)
    base = lv(tb(49), tb(51))
    r = [lv(tb(51, k), tb(51, k + 1)) - base for k in range(4)]
    return max(r[:2]) <= 2.0, f'bar 51 per beat vs plucks alone: {np.round(r, 1)} dB'


def build(st):
    """The mechanism under 'no wall': the doots placed in bars 51-52, counted
    per half-bar from the dry stem (an onset is a 16th step whose next 40 ms
    are 6 dB above its last 40 ms). Planned: 1, 1, 2, 3, then 6 in bar 53."""
    sr = int(st['sr'])
    x = st['doots'] ** 2
    e = lambda a, b: x[int(a * sr):int(b * sr)].mean() + 1e-20
    cnt = []
    for bar in (51, 52, 53):
        for half in (0, 1):
            n = 0
            for i in range(8 * half, 8 * half + 8):
                t = tb(bar) + i * BEAT / 4
                n += 10 * np.log10(e(t, t + 0.04) / e(t - 0.04, t)) > 6
            cnt.append(n)
    want = [1, 1, 2, 3]
    return cnt[:4] == want and sum(cnt[4:]) == 6, f'onsets per half-bar, bars 51-53: {cnt} (want {want} then 6)'


def chops(st):
    """The chopper gates: in every chop, the engine's closed parts sit >= 15 dB
    under its open parts (cells with < 12 ms open or closed skipped)."""
    sr = int(st['sr'])
    x = (st['droneL'] ** 2 + st['droneR'] ** 2) / 2
    ds = []
    for bar, beats, per, d0, d1, pat in S.CHOPS:
        a0, a1 = tb(bar), tb(bar) + beats * BEAT
        op, cl = [], []
        for a, o, e in S.chop_cells():
            if not (a0 <= a < a1):
                continue
            if o - a >= 0.012:
                op.append(x[int((a + 0.004) * sr):int((o - 0.004) * sr)].mean())
            if e - o >= 0.012:
                cl.append(x[int((o + 0.004) * sr):int((e - 0.004) * sr)].mean())
        ds.append(10 * np.log10(np.mean(op) / max(np.mean(cl), 1e-20)))
    return min(ds) >= 15, f'open/closed per chop {np.round(ds, 1)} dB'


def shares(mix, sr, a, b):
    m = mix.mean(1)[int(a * sr):int(b * sr)]
    f, Sx = signal.welch(m, sr, nperseg=8192)
    tot = Sx[(f > 30) & (f < 16000)].sum()
    sh = lambda lo, hi: 10 * np.log10(Sx[(f > lo) & (f < hi)].sum() / tot)
    return sh(120, 300), sh(2500, 6000)


def not_muddy(mix, sr):
    """The 120-300 Hz share of the total, in the groove and the climax, no more
    than the references' -8.4 dB."""
    a = shares(mix, sr, tb(9), tb(17))[0]
    b = shares(mix, sr, tb(61), tb(81))[0]
    return max(a, b) <= -8.4, f'120-300 Hz share: groove {a:.1f}, climax {b:.1f} dB'


def not_harsh(mix, sr):
    """The 2.5-6 kHz share in the climax within 2 dB of the references'
    brightest passage (-22 dB), i.e. <= -20."""
    h = shares(mix, sr, tb(61), tb(81))[1]
    return h <= -20, f'2.5-6 kHz share in the climax {h:.1f} dB'


def no_piercing(mix, sr):
    """The loudest second, not the average (Mark: the sung peak was "right at
    the edge of unbearable"; a reviewing model put it at 1:08): the 2.5-4 kHz
    band's level in any 1 s window stays within 1 dB of the references'
    loudest second (-28.2 dBFS), i.e. <= -27 dBFS. Without song.deharsh() it
    reaches -21.5 at 1:08, in the solo, where a section average hides it."""
    m = mix.mean(1)
    b = dsp.butter(m, (2500, 4000), sr, 'band', 4)
    lv = [(10 * np.log10((b[int(a * sr):int((a + 1) * sr)] ** 2).mean() + 1e-20), a)
          for a in np.arange(0, len(m) / sr - 1, 0.5)]
    top, at = max(lv)
    return top <= -27.0, f'loudest second in 2.5-4 kHz {top:.1f} dBFS at {int(at // 60)}:{at % 60:04.1f}'


def balance(mix, sr):
    """Lows (30-120 Hz) and highs (4-16 kHz) against the mids (200 Hz-4 kHz),
    within the references' section-by-section ranges."""
    out, ok = [], True
    for a, b in ((9, 17), (61, 81)):
        m = mix.mean(1)[int(tb(a) * sr):int(tb(b) * sr)]
        f, Sx = signal.welch(m, sr, nperseg=8192)
        lo = 10 * np.log10(Sx[(f > 30) & (f < 120)].sum() / Sx[(f > 200) & (f < 4000)].sum())
        hi = 10 * np.log10(Sx[(f > 4000) & (f < 16000)].sum() / Sx[(f > 200) & (f < 4000)].sum())
        ok &= 7.0 <= lo <= 14.5 and -16.0 <= hi <= -6.8
        out.append(f'bars {a}-{b - 1}: low/mid {lo:+.1f}, high/mid {hi:+.1f}')
    return ok, '; '.join(out)


def plucks(st):
    """Every metal pluck within 3 cents of its note (an integer delay line
    would put the high ones up to 12 cents off)."""
    sr, x = int(st['sr']), st['ks']
    errs = []
    for bar, beat, m in S.KS_NOTES:
        t = tb(bar, beat) + 0.05
        seg = x[int(t * sr):int((t + 0.5) * sr)]
        X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 20))
        f = np.fft.rfftfreq(1 << 20, 1 / sr)
        k = (f > HZ(m) * 2 ** -0.05) & (f < HZ(m) * 2 ** 0.05)
        errs.append(abs(1200 * np.log2(f[k][np.argmax(X[k])] / HZ(m))))
    return max(errs) <= 3, f'{len(errs)} plucks, worst {max(errs):.1f} c'


def clap_short(st, sc):
    """Mark: the clap was too long. 90% of each clap's energy arrives within
    80 ms of its onset."""
    sr = int(st['sr'])
    x = st['clap'] ** 2
    fr = []
    for t in sc['claps'][:40]:
        i = int(t * sr)
        seg = x[i:i + int(0.4 * sr)]
        c = np.cumsum(seg) / seg.sum()
        fr.append(np.argmax(c >= 0.9) / sr)
    t90 = float(np.median(fr))
    return t90 <= 0.08, f'90% of energy by {t90 * 1000:.0f} ms'


def on_grid(x, sr, times, what):
    """Every scheduled hit starts within 2 ms of its grid time."""
    x = np.abs(x)
    errs = []
    for t in times:
        i0, i1 = max(0, int((t - 0.02) * sr)), int((t + 0.04) * sr)
        seg = x[i0:i1]
        j = np.argmax(seg > 0.3 * seg.max())
        errs.append((i0 + j) / sr - t)
    errs = np.array(errs) * 1000
    return np.all(np.abs(errs) < 2.0), f'{len(errs)} {what}, onset error {errs.min():+.2f} to {errs.max():+.2f} ms'


def f0_at(x, sr, a, b, want):
    """Harmonic-sum pitch near `want` (+/-0.6 octave), on a short window."""
    seg = x[int(a * sr):int(b * sr)]
    y = signal.resample_poly(seg, 1, 4)
    nf = 1 << 17
    X = np.log(np.abs(np.fft.rfft(y * np.hanning(len(y)), nf)) + 1e-12)
    f = np.fft.rfftfreq(nf, 4 / sr)
    c = want * 2 ** np.linspace(-0.6, 0.6, 1441)
    H = [sum(X[min(len(X) - 1, int(round(cc * h / f[1])))] for h in (1, 2, 3)) for cc in c]
    return c[int(np.argmax(H))]


def bass_pitch(st):
    """The engine's pitch within 25 cents of each bass note, after its glide."""
    sr = int(st['sr'])
    x = (st['droneL'] + st['droneR']) / 2
    bad, n = [], 0
    for i, (bar, beat, m) in enumerate(S.BASS):
        a = tb(bar, beat) + 0.45
        b = tb(*S.BASS[i + 1][:2]) - 0.1 if i + 1 < len(S.BASS) else a + 1.5
        b = min(b, a + 0.6)
        if b - a < 0.3:
            continue
        seg = x[int(a * sr):int(b * sr)]
        if np.abs(seg).max() < 1e-3:
            continue
        c = 1200 * np.log2(f0_at(x, sr, a, b, HZ(m)) / HZ(m))
        n += 1
        if abs(c) > 25:
            bad.append(f'bar {bar}: {c:+.0f} c')
    return (not bad and n >= 15), f'{n} notes checked' + ('' if not bad else '; off: ' + ', '.join(bad[:4]))


def crest(x, sr):
    """Median peak-to-RMS (dB) over 1 s windows."""
    m = x.mean(1)
    return float(np.median([20 * np.log10(np.abs(m[i:i + sr]).max() / np.sqrt((m[i:i + sr] ** 2).mean()))
                            for i in range(0, len(m) - sr, sr) if np.abs(m[i:i + sr]).max() > 1e-3]))


def run(stem):
    sr, mix = wavfile.read(stem + '.wav')
    mix = mix.astype(np.float64)
    st = np.load(stem + '_stems.npz')
    sc = json.load(open(stem + '_score.json'))
    kp = np.concatenate([mix[int(tb(b) * sr):int((tb(b) + BAR) * sr)] for b in S.KICK])
    c = crest(kp, sr)
    L = dsp.lufs(mix, sr)
    tp = dsp.true_peak_db(mix)
    dc = float(np.abs(mix.mean(axis=0)).max())
    res = [('engine early', *engine_early(st)), ('the seam', *seam(st)), ('no wall', *no_wall(mix, sr)),
           ('the build', *build(st)), ('chopper gates', *chops(st)),
           ('not muddy', *not_muddy(mix, sr)), ('not harsh', *not_harsh(mix, sr)), ('no piercing second', *no_piercing(mix, sr)), ('balance', *balance(mix, sr)),
           ('plucks in tune', *plucks(st)), ('clap short', *clap_short(st, sc)),
           ('kick on the grid', *on_grid(st['kick'], int(st['sr']), sc['kicks'], 'kicks')),
           ('claps on 2 and 4', *on_grid(st['clap'], int(st['sr']), [tb(b, i) for b in S.CLAP for i in (1, 3)], 'claps')),
           ('bass line in tune', *bass_pitch(st)),
           ('punch (kick bars)', c >= 8, f'crest {c:.1f} dB'),
           ('loudness', abs(L - LUFS_TARGET) <= LUFS_TOL, f'{L:.2f} LUFS'),
           ('true peak', tp <= TP_MAX, f'{tp:.2f} dBTP (max {TP_MAX})'),
           ('finite, no DC', np.all(np.isfinite(mix)) and dc < 1e-3, f'dc {dc:.1e}')]
    for n, ok, msg in res:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}: {msg}")
    return {n: ok for n, ok, _ in res}


def render(brk):
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'song.py'), '--break', brk]
    try:                                       # cap the memory where systemd is available
        subprocess.check_call(['systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=10G', '-p', 'MemorySwapMax=0'] + cmd,
                              stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        for brk, which in (('no-chop', 'chopper gates'), ('cold-seam', 'the seam'), ('muddy', 'not muddy'),
                           ('harsh', 'not harsh'), ('wall', 'no wall'), ('old-build', 'the build'),
                           ('piercing', 'no piercing second')):
            render(brk)
            r = run(f'out/song_BROKEN-{brk}')
            print(f'== {brk}: {which}', 'caught' if not r[which] else 'NOT CAUGHT')
    else:
        r = run('out/song')
        print('ALL PASS' if all(r.values()) else 'SOME FAIL: ' + ', '.join(k for k, v in r.items() if not v))
