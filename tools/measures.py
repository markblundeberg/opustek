"""The measures: what an agent that can't hear can still know about a mix.

Each one answered a note from a human listener at some point (the guide, WORKFLOW.md, says which), and each was
tested on known cases before it was trusted: run `python3 tools/measures.py selftest` to see every one tell a
clean case from a broken one, and read a null input as nothing.

As a library (the tracks' check suites import it):

    from measures import load, db, Grid, band_share, phone_share, kick_punch, loudest_second, repetition, bed_fill,
                         sixteenth_levels, period_of, pitch_sim, side_ratio, crest, ending_steps, loudness

    load(path) -> (sr, x) with x shaped (channels, n); the kernel's dsp.lufs and dsp.true_peak_db want (n, 2), so use
    loudness(x, sr), or pass x.T. loudest_second -> (dBFS, time); repetition -> (next bar, 4 bars later);
    bed_fill -> (lit, held); ending_steps -> (levels, biggest rise, biggest fall while audible); period_of -> (lag, {lag: r}).

As a tool, a coarse profile of any finished mix, yours or a reference you own, to set targets:

    python3 tools/measures.py profile track.wav --bpm 124 --bar1 0.0 [--from 33 --to 64]

--bar1 is the time of the first downbeat in seconds (tools/refscope's analysis finds it, or read it off a spectrogram). The
profile prints numbers only; keep any reference audio, and anything you extract from it, out of version control.

Coarse ranges we measured on two commercial melodic-techno recordings (main sections), for orientation; your genre
and your references will differ, so measure your own:
    loudness -9 to -10 LUFS integrated, with a median 1 s crest of 9-10 dB
    3-8 kHz share of the whole: -22 to -26 dB;  share kept by a phone-sized speaker (300 Hz-10 kHz): -12 to -14 dB
    loudest 1 s of the 2.5-4 kHz band: about -28 dBFS (at those loudnesses)
    kick punch (50-160 Hz at the kick vs between kicks): +11.5 to +14 dB, +6.7 in one section under a busy bass
    bed fill (share of 200-4000 Hz lit between beats, and held over a beat): one reference's main sections 0.49-0.58
    lit, 0.14-0.18 held (its quieter stretches 0.21-0.41); a draft that was "missing something" 0.25 / 0.06
    repetition (bars 4 apart, rhythm only, by 32-bar span): one reference 0.53-0.61; our drafts a listener called
    repetitive 0.71-0.87; Moiré's finished drops 0.51-0.59
"""
import argparse
import json
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

db = lambda p: 10 * np.log10(np.maximum(p, 1e-15))


def load(path):
    """(sr, x) with x float64 of shape (channels, n); mono files come back with one channel."""
    sr, x = wavfile.read(path)
    if x.dtype.kind == 'i':
        x = x / float(np.iinfo(x.dtype).max)
    x = np.atleast_2d(x.astype(np.float64).T) if x.ndim == 2 else x.astype(np.float64)[None, :]
    return sr, x


def mono(x):
    return x.mean(0) if x.ndim == 2 else x


class Grid:
    """A rigid bar grid: tb(bar, beat) is the time of a beat, bars counted from 1."""

    def __init__(self, bpm, bar1=0.0):
        self.bpm, self.bar1 = float(bpm), float(bar1)
        self.beat = 60 / self.bpm
        self.bar = 4 * self.beat
        self.six = self.beat / 4

    def tb(self, bar, beat=0.0):
        return self.bar1 + (bar - 1) * self.bar + beat * self.beat


def _bandpass(y, sr, lo, hi, order=4):
    return signal.sosfilt(signal.butter(order, [lo, hi], 'band', fs=sr, output='sos'), y)


def _welch(y, sr):
    return signal.welch(y, sr, nperseg=8192)


def band_share(y, sr, lo, hi, t0=0.0, t1=None, whole=(30, 16000)):
    """dB of the power in [lo, hi] relative to the whole (30 Hz-16 kHz), over [t0, t1]. Tonal balance: compare it
    section by section with a reference's range for the same kind of section, never as a song average."""
    y = mono(y)[int(t0 * sr):None if t1 is None else int(t1 * sr)]
    f, P = _welch(y, sr)
    return float(db(P[(f >= lo) & (f < hi)].sum() / P[(f >= whole[0]) & (f < whole[1])].sum()))


def phone_share(y, sr, t0=0.0, t1=None):
    """What a phone-sized speaker keeps: the 300 Hz-10 kHz share. A mix can sound full on headphones and thin here."""
    return band_share(y, sr, 300, 10000, t0, t1)


def side_ratio(x, sr, lo, hi, t0=0.0, t1=None):
    """Stereo width in a band: side power over mid power, dB (-inf for mono). Measure it per stem, too: a wide
    neighbour can pass a mono lead."""
    a, b = int(t0 * sr), None if t1 is None else int(t1 * sr)
    L, R = _bandpass(x[0][a:b], sr, lo, hi), _bandpass(x[1][a:b], sr, lo, hi)
    return float(db(np.mean((L - R) ** 2)) - db(np.mean((L + R) ** 2)))


def crest(y, sr, win=1.0):
    """Median crest factor (peak over RMS, dB) of 1 s windows: how much the transients stand out."""
    y, w = mono(y), int(win * sr)
    c = [20 * np.log10(np.abs(y[i:i + w]).max() / (np.sqrt((y[i:i + w] ** 2).mean()) + 1e-12))
         for i in range(0, len(y) - w, w) if np.abs(y[i:i + w]).max() > 1e-3]
    return float(np.median(c)) if c else float('nan')


def loudest_second(y, sr, lo=2500, hi=4000, win=1.0, hop=0.5):
    """The loudest moment, not the average: the highest level of the [lo, hi] band in any 1 s window, dBFS (compare
    mixes at the same loudness), and when. Every section average passed while one second ran 7 dB hotter than
    anything in the references; a listener heard it ("right at the edge of unbearable")."""
    b = _bandpass(mono(y), sr, lo, hi) ** 2
    lv = [(float(db(b[int(a * sr):int((a + win) * sr)].mean())), float(a)) for a in np.arange(0, len(b) / sr - win, hop)]
    return max(lv)


def kick_punch(y, sr, kicks, lo=50, hi=160, after=0.03, between=0.5):
    """How far the kick stands out of the low end: mean power in [lo, hi] in the 30 ms after each kick against the
    power halfway to the next beat, dB. When the bass fills the gaps the kick gets 'swallowed'; the duck may have
    recovered long before halfway, so look at the bass's own level first."""
    e = _bandpass(mono(y), sr, lo, hi) ** 2
    ks = np.asarray(kicks)
    if len(ks) < 3:
        return float('nan')
    gap = np.median(np.diff(ks))
    on = [e[int(k * sr):int((k + after) * sr)].mean() for k in ks[:-1]]
    off = [e[int((k + between * gap - after / 2) * sr):int((k + between * gap + after / 2) * sr)].mean() for k in ks[:-1]]
    return float(db(np.mean(on)) - db(np.mean(off)))


def sixteenth_levels(y, sr, grid, lo, hi, b0, b1, win=0.06):
    """Level in a band at the start of each 16th from bar b0 up to bar b1 (exclusive), dB."""
    e = _bandpass(mono(y), sr, lo, hi) ** 2
    n = (b1 - b0) * 16
    t = [grid.tb(b0) + k * grid.six for k in range(n)]
    return np.array([db(e[int(a * sr):int((a + win) * sr)].mean()) for a in t])


def period_of(v, lags=range(2, 17)):
    """The figure length in a sequence of 16th-step levels: the lag with the highest autocorrelation, after
    removing the trend over a bar. Without the detrend a fade-in correlates with itself at every lag, and our check
    'the arpeggio is in 7' read a fade, not the 7, for four versions."""
    v = np.asarray(v, float)
    v = v - np.convolve(np.pad(v, 8, mode='edge'), np.ones(17) / 17, 'valid')
    ac = {L: float(np.corrcoef(v[:-L], v[L:])[0, 1]) for L in lags}
    return max(ac, key=ac.get), ac


def pitch_sim(y, sr, grid, b0, b1, lo=150, hi=900):
    """Pitch-class similarity of each 16th with the one L later, for L in 2..16. A figure with no rests has its
    period in its notes, not its levels."""
    y = mono(y)
    a, w = int(grid.tb(b0) * sr), int(0.1 * sr)
    f = np.fft.rfftfreq(8192, 1 / sr)
    m = (f > lo) & (f < hi)
    pc = np.round(12 * np.log2(f[m] / 440) + 69).astype(int) % 12
    C = []
    for k in range((b1 - b0) * 16):
        s = int(a + k * grid.six * sr)
        seg = y[s:s + w] * np.hanning(w)
        C.append(np.bincount(pc, weights=np.abs(np.fft.rfft(seg, 8192))[m] ** 2, minlength=12))
    C = np.array(C).T
    C = C / (np.linalg.norm(C, axis=0) + 1e-9)
    return {L: float(np.mean(np.sum(C[:, :-L] * C[:, L:], 0))) for L in range(2, 17)}


def _melspec(y, sr, n_fft, hop, n_mels, fmin=0.0, fmax=None):
    """A log-mel spectrogram in plain numpy/scipy (no librosa), power in dB."""
    fmax = fmax or sr / 2
    f, t, Z = signal.stft(y, sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    W = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        l, c, r = edges[i:i + 3]
        W[i] = np.clip(np.minimum((f - l) / (c - l + 1e-9), (r - f) / (r - c + 1e-9)), 0, None)
    return db(W @ P + 1e-10), sr / hop


def repetition(y, sr, grid, b0, b1, pattern=True):
    """How alike the bars are: cosine similarity of each bar's picture (log-mel, 64 bands x 16 sixteenths) with
    the next bar and with the bar 4 later; 1 is identical. Returns (next, 4 later).

    pattern=True (the default) takes each band's mean over the bar out first, so what is compared is the rhythm in
    each band, not the timbre. The first version (pattern=False, kept for comparison) took out only the overall mean,
    and the self-test showed it reading plain white noise as 0.97: it mostly measured how steady the spectrum was.
    Its calibration (ours "repetitive" at 0.975, the references 0.925-0.944) was real but measured sameness of sound
    as much as of rhythm."""
    y = mono(y)
    V = []
    for b in range(b0, b1 + 1):
        a = int(grid.tb(b) * sr)
        M, _ = _melspec(y[a:a + int(grid.bar * sr)], sr, 2048, 512, 64)
        idx = np.linspace(0, M.shape[1], 17).astype(int)
        P = np.stack([M[:, idx[i]:idx[i + 1]].mean(1) for i in range(16)], 1)
        P = P - (P.mean(1, keepdims=True) if pattern else P.mean())
        V.append(P.ravel() / (np.linalg.norm(P) + 1e-9))
    s1 = np.mean([V[i] @ V[i + 1] for i in range(len(V) - 1)])
    s4 = np.mean([V[i] @ V[i + 4] for i in range(len(V) - 4)]) if len(V) > 4 else float('nan')
    return float(s1), float(s4)


def bed_fill(y, sr, grid, b0, b1):
    """'Missing something': between the beats, the share of the 200-4000 Hz picture that is lit (within 35 dB of
    the span's loudest cell), and the share that is held (changes < 3 dB over a beat). Read mid-way between beats,
    away from the kick and clap. Returns (lit, held)."""
    y = mono(y)
    a, b = int(grid.tb(b0) * sr), int((grid.tb(b1 + 1) + grid.beat) * sr)
    M, fps = _melspec(y[a:b], sr, 4096, 512, 96, 200, 4000)
    fr = [int((k + 0.5) * grid.beat * fps) for k in range((b1 - b0) * 4)]
    fr = [i for i in fr if i + int(grid.beat * fps) < M.shape[1]]
    A, B = M[:, fr], M[:, [i + int(grid.beat * fps) for i in fr]]
    lit = A > M[:, fr[0]:fr[-1]].max() - 35
    return float(lit.mean()), float((lit & (np.abs(A - B) < 3)).mean())


def ending_steps(y, sr, t0, step=0.5, skip=0):
    """After t0, the level in half-second steps (dB re the first) and the biggest rise and fall between steps
    while still audible (above -30 dB re the first). A boom 20 dB up out of a fade hid here for six versions.
    If the ending starts on a hit, its own decay is a big first fall: skip that many steps before judging (Moiré's
    check skips 2, the final hit's first second)."""
    y = mono(y)
    lv = np.array([db(np.mean(y[int((t0 + k * step) * sr):int((t0 + (k + 1) * step) * sr)] ** 2))
                   for k in range(int((len(y) / sr - t0) / step))])
    lv = lv - lv[0]
    d = np.diff(lv)
    d[:skip] = 0
    aud = lv[1:] > -30
    return lv, float(d.max()), float(-d[aud].min()) if aud.any() else 0.0


def loudness(x, sr):
    """Integrated loudness (BS.1770, LUFS) and true peak (dBTP, 4x oversampled), from the kernel."""
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import dsp
    return float(dsp.lufs(x.T.copy() if x.shape[0] <= 2 else x, sr)), float(dsp.true_peak_db(x.T.copy()))


def find_kicks(y, sr, grid, b0, b1):
    """Kick times on the grid's beats from bar b0 to b1 where the 40-120 Hz band rises sharply."""
    e = _bandpass(mono(y), sr, 40, 120) ** 2
    out = []
    for b in range(b0, b1 + 1):
        for q in range(4):
            t = grid.tb(b, q)
            pre = e[max(0, int((t - 0.03) * sr)):int(t * sr)].mean()
            post = e[int(t * sr):int((t + 0.03) * sr)].mean()
            if post > 4 * pre and post > 1e-7:
                out.append(t)
    return out


def profile(path, bpm, bar1, b0=None, b1=None):
    sr, x = load(path)
    g = Grid(bpm, bar1)
    nb = int((x.shape[1] / sr - bar1) / g.bar)
    b0, b1 = b0 or 1, min(b1 or nb, nb)
    t0, t1 = g.tb(b0), g.tb(b1 + 1)
    out = {'file': path, 'bars': [b0, b1], 'seconds': [round(t0, 2), round(t1, 2)]}
    if x.shape[0] == 2:
        L, tp = loudness(x, sr)
        out.update(lufs=round(L, 1), true_peak=round(tp, 1))
        out['side_dB'] = {f'{lo}-{hi}': round(side_ratio(x, sr, lo, hi, t0, t1), 1) for lo, hi in ((30, 120), (120, 500), (500, 2000), (2000, 8000), (8000, 16000))}
    y = mono(x)
    out['share_3_8k'] = round(band_share(y, sr, 3000, 8000, t0, t1), 1)
    out['share_phone'] = round(phone_share(y, sr, t0, t1), 1)
    out['loudest_2.5_4k'] = [round(v, 1) for v in loudest_second(y[int(t0 * sr):int(t1 * sr)], sr)]
    out['crest'] = round(crest(y[int(t0 * sr):int(t1 * sr)], sr), 1)
    k = find_kicks(y, sr, g, b0, b1)
    out['kicks_found'] = len(k)
    out['kick_punch'] = round(kick_punch(y, sr, k), 1) if len(k) > 8 else None
    out['repetition'] = [round(v, 3) for v in repetition(y, sr, g, b0, min(b1, b0 + 31))]
    out['bed_fill'] = [round(v, 2) for v in bed_fill(y, sr, g, b0, min(b1, b0 + 31))]
    return out


# ---------------------------------------------------------------- self-test: each measure on known cases

def selftest():
    sr, rng = 48000, np.random.default_rng(0)
    g = Grid(124, 0.0)
    n = int(g.tb(17) * sr)
    t = np.arange(n) / sr
    ok = True

    def report(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'ok ' if good else 'BAD'} {name}: {detail}")

    def kick(at, amp=1.0):
        z = np.zeros(n)
        u = np.arange(int(0.25 * sr)) / sr
        k_ = amp * np.sin(2 * np.pi * np.cumsum(50 + 120 * np.exp(-u / 0.02)) / sr) * np.exp(-u / 0.12)
        for a in at:
            i = int(a * sr)
            z[i:i + len(k_)] += k_[:n - i]
        return z

    beats = [g.tb(b, q) for b in range(1, 17) for q in range(4)]
    # periodicity: a figure 7 sixteenths long reads 7; under a fade-in still 7; a fade alone and noise read no strong period
    def figure(L):
        z = np.zeros(n)
        for k in range(16 * 16):
            a = int(k * g.six * sr)
            amp = 1.0 if k % L == 0 else 0.3
            u = np.arange(int(0.08 * sr)) / sr
            z[a:a + len(u)] += amp * np.sin(2 * np.pi * 700 * u) * np.exp(-u / 0.03)
        return z
    fade = np.clip(t / t[-1], 0, 1)
    p7, _ = period_of(sixteenth_levels(figure(7), sr, g, 500, 1000, 2, 15))
    p7f, _ = period_of(sixteenth_levels(figure(7) * fade ** 3, sr, g, 500, 1000, 2, 15))
    pn, acn = period_of(sixteenth_levels(rng.standard_normal(n), sr, g, 500, 1000, 2, 15))
    report('period_of', p7 == 7 and p7f == 7 and max(acn.values()) < 0.3,
           f'figure in 7 -> {p7}; under a fade-in -> {p7f}; noise -> best r {max(acn.values()):.2f} (< 0.3)')
    # kick punch: kicks alone stand far out; with a sustained 60 Hz bass they stand much less
    kk = kick(beats)
    bass = 0.5 * np.sin(2 * np.pi * 61 * t)
    pa, pb = kick_punch(kk, sr, beats), kick_punch(kk + bass, sr, beats)
    report('kick_punch', pa > 15 and pb < pa - 8, f'kicks alone {pa:+.1f} dB; under a held bass {pb:+.1f} dB')
    # loudest second: a 1 s burst at 3 kHz in the middle of pink-ish noise is found, at the right time
    base = signal.lfilter([1], [1, -0.98], rng.standard_normal(n)) * 0.01
    hot = base.copy()
    i = int(10.0 * sr)
    hot[i:i + sr] += 0.02 * np.sin(2 * np.pi * 3100 * t[:sr])
    (va, _), (vb, at) = loudest_second(base, sr), loudest_second(hot, sr)
    report('loudest_second', vb > va + 6 and 9.0 <= at <= 10.5, f'clean {va:.1f} dBFS; with a hot second {vb:.1f} at {at:.1f} s (put at 10.0)')
    # repetition: the same bar over and over reads ~1; noise reads near 0
    bar = figure(4)[:int(g.bar * sr)]
    same = np.tile(bar, 17)[:n]
    r1, r4 = repetition(same, sr, g, 1, 12)
    q1, q4 = repetition(rng.standard_normal(n), sr, g, 1, 12)
    o1, o4 = repetition(rng.standard_normal(n), sr, g, 1, 12, pattern=False)
    report('repetition', r4 > 0.98 and q4 < 0.2, f'identical bars {r4:.3f}; noise {q4:.3f} (the first version read noise as {o4:.3f})')
    # bed fill: kicks and clicks alone light little between beats; a held chord lights a lot and is held
    chord = sum(0.1 * signal.sawtooth(2 * np.pi * f * t) for f in (330, 392, 494))
    l0, h0 = bed_fill(kk, sr, g, 2, 12)
    l1, h1 = bed_fill(kk + chord, sr, g, 2, 12)
    report('bed_fill', l0 < 0.1 and l1 > 0.3 and h1 > 0.1, f'kicks alone lit {l0:.2f} held {h0:.2f}; with a held chord lit {l1:.2f} held {h1:.2f}')
    # tonal balance: white noise's 3-8 kHz share is its bandwidth's share (about -5 dB); a low hum's is far below
    w = rng.standard_normal(n)
    s_w = band_share(w, sr, 3000, 8000)
    s_h = band_share(np.sin(2 * np.pi * 100 * t) + 0.001 * w, sr, 3000, 8000)
    report('band_share', abs(s_w - db(5000 / 15970)) < 0.5 and s_h < -40, f'white noise {s_w:.1f} dB (expect {db(5000 / 15970):.1f}); a 100 Hz hum {s_h:.1f}')
    # width: identical channels read mono; independent noise reads 0 dB (side = mid)
    s_m = side_ratio(np.stack([w, w]), sr, 200, 2000)
    s_d = side_ratio(np.stack([w, rng.standard_normal(n)]), sr, 200, 2000)
    report('side_ratio', s_m < -100 and abs(s_d) < 0.5, f'mono {s_m:.0f} dB; independent channels {s_d:.1f} dB')
    # endings: a smooth fade has no rise; a boom out of the fade is caught
    tail = np.exp(-t[:8 * sr] / 1.5) * rng.standard_normal(8 * sr) * 0.1
    boomed = tail.copy()
    boomed[5 * sr:5 * sr + sr // 4] += 0.3 * np.sin(2 * np.pi * 50 * t[:sr // 4])
    _, ra, _ = ending_steps(tail, sr, 0.0)
    _, rb, _ = ending_steps(boomed, sr, 0.0)
    report('ending_steps', ra < 3 and rb > 6, f'smooth fade biggest rise {ra:+.1f} dB; with a boom {rb:+.1f} dB')
    print('  ->', 'every measure tells its cases apart' if ok else 'a measure is blind')
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('profile', help='a coarse profile of a mix')
    p.add_argument('wav')
    p.add_argument('--bpm', type=float, required=True)
    p.add_argument('--bar1', type=float, default=0.0, help='time of the first downbeat, s')
    p.add_argument('--from', dest='b0', type=int)
    p.add_argument('--to', dest='b1', type=int)
    sub.add_parser('selftest', help='each measure against known cases')
    a = ap.parse_args()
    if a.cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    print(json.dumps(profile(a.wav, a.bpm, a.bar1, a.b0, a.b1), indent=1))
