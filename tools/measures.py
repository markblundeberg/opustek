"""The measures: what an agent that can't hear can still know about a mix.

Each one answered a note from a human listener at some point (the guide, WORKFLOW.md, says which), and each was
tested on known cases before it was trusted: run `python3 tools/measures.py selftest` to see every one tell a
clean case from a broken one, and read a null input as nothing.

As a library (the tracks' check suites import it):

    from measures import load, db, Grid, band_share, phone_share, kick_punch, loudest_second, repetition, bed_fill,
                         sixteenth_levels, period_of, pitch_sim, side_ratio, crest, ending_steps, loudness,
                         onsets, masking, kick_onset_contrast, master_cost, find_grid   (and Grid.ts for swung 16ths)

    load(path) -> (sr, x) with x shaped (channels, n); the kernel's dsp.lufs and dsp.true_peak_db want (n, 2), so use
    loudness(x, sr), or pass x.T. loudest_second -> (dBFS, time); repetition -> (next bar, 4 bars later);
    bed_fill -> (lit, held); ending_steps -> (levels, biggest rise, biggest fall while audible); period_of -> (lag, {lag: r});
    onsets -> one bool per 16th step; masking -> dB (part over other, in the part's band).

As a tool, a coarse profile of any finished mix, yours or a reference you own, to set targets:

    python3 tools/measures.py grid track.wav                  # tempo and first downbeat, numpy only
    python3 tools/measures.py profile track.wav --bpm 124 --bar1 0.0 [--from 33 --to 64]

--bar1 is the time of the first downbeat in seconds. Leave out --bpm and find_grid supplies both: its tempo is
reliable, its downbeat a good guess to confirm (its docstring says how good). The
profile prints numbers only; keep any reference audio, and anything you extract from it, out of version control.

Coarse ranges we measured on two commercial melodic-techno recordings (main sections), for orientation. They are for
that style only: a downtempo piece with a sustained bass reads far outside them and needn't be wrong. Measure your own:
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

    def ts(self, bar, step, swing=0.5):
        """The time of 16th step `step` (0-15, or beyond into later bars) of bar `bar`, with swing: each odd 16th lands
        at `swing` of its 8th (0.5 is straight; garage and lo-fi sit near 0.58-0.62). Pass swing explicitly where you
        call it: a default bound to a module constant is fixed when the function is defined, which once made a
        'straight' fault do nothing."""
        pair, odd = divmod(step, 2)
        return self.tb(bar) + pair * self.beat / 2 + odd * swing * self.beat / 2


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
    power halfway to that kick's next kick, dB (each kick's own gap, so uneven patterns work; with kicks close
    together, as in 2-step, halfway can still sit in the previous kick's tail: use kick_onset_contrast there). When
    the bass fills the gaps the kick gets 'swallowed'; the duck may have recovered long before halfway, so look at the
    bass's own level first. The other common cause is the master: a loud loudness target makes the limiter clamp the
    kick's first 100 ms (see master_cost). Lowering the kick then makes it worse."""
    e = _bandpass(mono(y), sr, lo, hi) ** 2
    ks = np.asarray(sorted(kicks))
    if len(ks) < 3:
        return float('nan')
    on, off = [], []
    for k, nxt in zip(ks[:-1], ks[1:]):
        m = k + between * (nxt - k)
        on.append(e[int(k * sr):int((k + after) * sr)].mean())
        off.append(e[int((m - after / 2) * sr):int((m + after / 2) * sr)].mean())
    return float(db(np.mean(on)) - db(np.mean(off)))


def kick_onset_contrast(y, sr, kicks, lo=40, hi=160):
    """The kick's first 40 ms against the 40 ms just before it, dB, in [lo, hi]. Needs no gap length, so it suits any
    pattern (2-step's kicks on steps 0, 7 and 10 read +1 dB on kick_punch where, by stem, the kick stood 6.7 dB over
    the sub). It reads onset contrast, not how much the gaps fill: a busy sub just before each kick lowers it, as it
    should. (From the first cold-test agent outside our setup, which wrote it for its garage track, Night Bus.)"""
    e = _bandpass(mono(y), sr, lo, hi) ** 2
    on = np.mean([e[int(k * sr):int((k + 0.04) * sr)].mean() for k in kicks])
    pre = np.mean([e[max(0, int((k - 0.045) * sr)):max(1, int((k - 0.005) * sr))].mean() for k in kicks])
    return float(db(on) - db(pre))


def master_cost(premix, master, sr, kicks, measure=None):
    """How much punch the master takes away: the punch measure on the sum of the stems minus the same on the mastered
    mix, dB. More than about 2 dB means beat-aligned peaks are feeding the limiter: saturate the clap, soft-clip the
    drum bus, or master a dB or two quieter. (One piece: +8.3 premix, +5.8 mastered at -11 LUFS, +4.1 at -10.)"""
    measure = measure or kick_punch
    return float(measure(premix, sr, kicks) - measure(master, sr, kicks))


def onsets(stem, sr, grid, b0, b1, rise_db=6.0, floor_db=-50.0, win=0.03, swing=0.5):
    """Which 16th steps of a stem have a note starting, from bar b0 up to b1 (exclusive): a step counts when any
    octave band (125 Hz-8 kHz) rises by rise_db from the 30 ms before it to the 30 ms after, above floor_db re the
    stem's loudest step. Octave bands, because a broadband level misses a high note right after a low one still
    ringing. Returns a boolean array, one per step. Count them per bar and compare with the score's intent, not with
    the render's own note list (that check would pass a missing part). Pass the piece's swing, or swung odd steps
    are looked for in the wrong place."""
    y = mono(stem)
    bands = [(125 * 2 ** k, 250 * 2 ** k) for k in range(6)]
    E = [_bandpass(y, sr, lo, hi) ** 2 for lo, hi in bands]
    steps = [grid.ts(b0, k, swing) for k in range((b1 - b0) * 16)]
    w = int(win * sr)
    post = np.array([[e[int(t * sr):int(t * sr) + w].mean() for e in E] for t in steps])
    pre = np.array([[e[max(0, int(t * sr) - w):int(t * sr)].mean() if int(t * sr) > 0 else 1e-20 for e in E] for t in steps])
    top = db(post.max())
    loud = db(post) > top + floor_db
    return ((db(post) - db(pre + 1e-20) >= rise_db) & loud).any(1)


def masking(part, other, sr, lo, hi, t0, t1):
    """How far a part stands over another in the part's own band: RMS of each in [lo, hi] over [t0, t1], dB (part
    minus other). Below about +3 dB, a listener said of an arpeggio "I can't hear it at all" (it read -8.9 there)."""
    a, b = int(t0 * sr), int(t1 * sr)
    pa = _bandpass(mono(part)[a:b], sr, lo, hi)
    po = _bandpass(mono(other)[a:b], sr, lo, hi)
    return float(db(np.mean(pa ** 2)) - db(np.mean(po ** 2)))


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


def _onset_env(y, sr, lo, hi, fps=400):
    """Half-wave-rectified rise of the log energy in a band, `fps` frames per second (10 ms smoothing)."""
    e = _bandpass(y, sr, lo, hi) ** 2
    h = sr // fps
    e = signal.lfilter([0.25] * 4, [1], e[:len(e) // h * h].reshape(-1, h).mean(1))
    le = db(e + 1e-12)
    d = np.maximum(np.diff(le, prepend=le[0]), 0)
    d[le < le.max() - 60] = 0
    return d


KICK_CUE = 0.5


def find_grid(y, sr, bpm_range=(70, 180), t0=0.0, t1=None):
    """Tempo, first downbeat and confidence of a mix with one rigid tempo (most music made on a computer):
    -> (bpm, bar1, info), bar1 the first downbeat at or after t0, ready for Grid(bpm, bar1).

    Tempo: an autocorrelation comb shortlists tempos, then each is refined by folding the onsets on one beat and
    keeping the tempo that piles the most into the narrowest slot (it uses every harmonic, so it is sharp).
    Downbeat: chords, bass notes and sections change on beat 1, so it compares half a bar of spectrum before and after
    every 8th, adds how much kick lands on that 8th, and picks the 8th of the bar that scores highest (8, not 4: a
    syncopated beat like 2-step can fold half a beat out).

    On our five tracks (90-132 bpm; four-on-the-floor, broken-beat, 2-step): tempo within 0.01 bpm over a whole track
    and 0.1 bpm over any 60 s; beats within 5 ms. The downbeat was right on every whole track but on only 19 of 25
    windows of 60 s, and the kick cue's weight was tuned on those same windows, so expect less on music unlike ours.
    info['downbeat_margin'] doesn't flag the wrong ones: confirm the downbeat (section changes should land on bar 1
    of 8-bar groups) before trusting bar numbers. A tempo off by a factor of 2 is possible on sparse music; narrow
    bpm_range if you know better. info['fold'] reads about 0.05 for noise and 0.12-0.43 for our tracks."""
    y = mono(y)
    t1 = len(y) / sr if t1 is None else t1
    y = y[int(t0 * sr):int(t1 * sr)]
    fps = 400
    low = _onset_env(y, sr, 40, 160, fps)
    rest = sum(_onset_env(y, sr, lo, hi, fps) for lo, hi in ((160, 600), (600, 2500), (2500, 10000)))
    on = low / (low.std() + 1e-12) + 0.5 * rest / (rest.std() + 1e-12)
    fr = np.arange(len(on))
    on0 = on - on.mean()
    ac = signal.fftconvolve(on0, on0[::-1], 'full')[len(on) - 1:]
    ac /= ac[0]
    cands = np.arange(bpm_range[0], bpm_range[1], 0.05)
    comb = np.array([np.mean([np.interp(k * 60 / b * fps, fr, ac) for k in (1, 2, 3, 4)]) for b in cands])

    def fold(b, nbin=96):
        ph = (fr / fps * b / 60) % 1
        h = np.bincount((ph * nbin).astype(int), on, nbin)
        hs = np.convolve(np.r_[h[-2:], h, h[:2]], np.ones(5), 'valid')
        i = int(hs.argmax())
        w = np.arange(i - 6, i + 7)
        ang = np.angle((h[w % nbin] * np.exp(2j * np.pi * (w + 0.5) / nbin)).sum())
        return hs[i] / on.sum(), (ang / (2 * np.pi)) % 1

    best = (-1, None)
    for b in cands[np.argsort(comb)[::-1][:5]]:
        fine = np.arange(b - 0.15, b + 0.15, 0.002)
        f_ = np.array([fold(f)[0] for f in fine])
        if f_.max() > best[0]:
            best = (f_.max(), fine[f_.argmax()])
    conc, bpm = best
    beat = 60 / bpm
    ph = fold(bpm)[1] * beat - 0.006                 # the onset envelope reads about 6 ms late on our tracks
    e8, W = beat / 2, 4                         # half a bar each side: one bar failed a track, two beats none
    n8 = int((len(y) / sr - ph) / e8) - 1
    f, _, S = signal.stft(y, sr, nperseg=2048, noverlap=2048 - 256)
    tS = np.arange(S.shape[1]) * 256 / sr
    P = np.abs(S[(f > 30) & (f < 2000)]) ** 2
    spec = np.array([P[:, (tS >= ph + k * e8) & (tS < ph + (k + 1) * e8)].mean(1) for k in range(n8)])
    cs = np.r_[np.zeros((1, spec.shape[1])), np.cumsum(spec, 0)]
    chg = np.array([np.linalg.norm(db((cs[j] - cs[j - W]) / W + 1e-12) - db((cs[j + W] - cs[j]) / W + 1e-12))
                    for j in range(W, n8 - W)])
    s8 = np.array([chg[(k - W) % 8::8].mean() for k in range(8)])
    # and the low band's onsets on each 8th of the bar: whatever the pattern, the kick is surest on beat 1
    hit = np.array([low[max(0, int((ph + j * e8 - 0.01) * fps)):int((ph + j * e8 + 0.03) * fps)].sum() for j in range(n8)])
    k8 = np.array([hit[k::8].mean() for k in range(8)])
    s8 = s8 / s8.mean() + KICK_CUE * k8 / (k8.mean() + 1e-12)
    k = int(s8.argmax())
    bar1 = t0 + (ph + k * e8 + 0.02) % (4 * beat) - 0.02
    return float(bpm), float(bar1), {'fold': round(float(conc), 3),
                                      'downbeat_margin': round(float(np.sort(s8)[-1] / np.sort(s8)[-2]), 3)}


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
    # onsets: plucks on known steps, low and high alternating with long rings, found exactly; noise finds almost none
    pl = np.zeros(n)
    want = np.zeros(4 * 16, bool)
    u = np.arange(int(0.4 * sr)) / sr
    for k in range(4 * 16):
        if k % 3 == 0 or k == 0:
            want[k] = True
            f0 = 220 if (k // 3) % 2 == 0 else 880
            i = int((g.tb(1) + k * g.six) * sr)
            pl[i:i + len(u)] += np.sign(np.sin(2 * np.pi * f0 * u)) * np.exp(-u / 0.15) * 0.3
    got = onsets(pl, sr, g, 1, 5)
    fn = onsets(rng.standard_normal(n) * 0.1, sr, g, 1, 5)
    report('onsets', np.array_equal(got, want) and fn.mean() < 0.05, f'{got.sum()} of {want.sum()} plucks found, {int((got & ~want).sum())} extra; on noise {fn.mean():.0%} of steps (< 5%)')
    # masking: a part 10 dB over another in its band reads about +10; equal reads about 0
    a_ = rng.standard_normal(n)
    m1, m0 = masking(a_, a_ * 10 ** (-10 / 20), sr, 600, 1400, 0, 10), masking(a_, rng.standard_normal(n), sr, 600, 1400, 0, 10)
    report('masking', abs(m1 - 10) < 0.2 and abs(m0) < 0.5, f'10 dB over: {m1:+.1f}; equal noise: {m0:+.1f}')
    # swing: Grid.ts places odd 16ths late; onsets on swung plucks need the swing passed in
    tsw = g.ts(1, 1, 0.6) - g.ts(1, 0, 0.6)
    report('Grid.ts', abs(tsw - 0.6 * g.beat / 2) < 1e-9 and abs(g.ts(1, 1) - g.ts(1, 0) - g.six) < 1e-9, f'a swung odd 16th lands {tsw * 1000:.0f} ms after its 8th (straight {g.six * 1000:.0f})')
    sw = np.zeros(n)
    for k in range(4 * 16):
        i = int(g.ts(1, k, 0.62) * sr)
        sw[i:i + len(u)] += np.sign(np.sin(2 * np.pi * 440 * u)) * np.exp(-u / 0.06) * 0.3
    got_sw, got_st = onsets(sw, sr, g, 1, 5, swing=0.62), onsets(sw, sr, g, 1, 5)
    report('onsets, swung', got_sw.all() and got_st.mean() < 0.8, f'with the swing passed {got_sw.mean():.0%} of steps found; on the straight grid {got_st.mean():.0%}')
    # 2-step: kicks on steps 0, 7 and 10 (swung); onset contrast reads the kick alone high and drops under a held sub
    tk = [g.ts(b, st, 0.6) for b in range(1, 13) for st in (0, 7, 10)]
    k2 = kick(tk)
    sub = 0.4 * np.sin(2 * np.pi * 44 * t)
    c0, c1 = kick_onset_contrast(k2, sr, tk), kick_onset_contrast(k2 + sub, sr, tk)
    report('kick_onset_contrast', c0 > 10 and c1 < c0 - 6, f'2-step kicks alone {c0:+.1f} dB; under a held sub {c1:+.1f} dB')
    # master cost: a hard clip on the kicks loses punch; no master loses none
    clipped = np.clip(kk * 6, -0.5, 0.5)
    mc, m0 = master_cost(kk, clipped, sr, beats), master_cost(kk, kk, sr, beats)
    report('master_cost', mc > 2 and abs(m0) < 1e-6, f'a hard limiter costs {mc:+.1f} dB of punch; no master {m0:+.1f}')
    # grid: a synthetic track at 117.3 bpm, first downbeat at 0.37 s (kick on beats, hat on offbeats, a chord per bar)
    # is found to 0.05 bpm and 15 ms with the right downbeat; noise folds no better than chance
    gg = Grid(117.3, 0.37)
    m = int(40 * sr)
    tt = np.arange(m) / sr
    syn = np.zeros(m)
    u = np.arange(int(0.25 * sr)) / sr
    kh = np.sin(2 * np.pi * np.cumsum(50 + 120 * np.exp(-u / 0.02)) / sr) * np.exp(-u / 0.12)
    hh = rng.standard_normal(len(u)) * np.exp(-u / 0.01) * 0.1
    for b in range(1, 20):
        for q in range(4):
            for hit, at in ((kh, gg.tb(b, q)), (hh, gg.tb(b, q + 0.5))):
                i = int(at * sr)
                if i + len(u) < m:
                    syn[i:i + len(u)] += hit
        i0, i1 = int(gg.tb(b) * sr), min(m, int(gg.tb(b + 1) * sr))
        root = (110, 147, 123, 165)[b % 4]
        syn[i0:i1] += sum(0.08 * np.sin(2 * np.pi * root * r * tt[i0:i1]) for r in (1, 1.2, 1.5))
    gb, g1, gi = find_grid(syn, sr)
    ge = ((g1 - 0.37 + gg.bar / 2) % gg.bar - gg.bar / 2) * 1000
    _, _, ni = find_grid(rng.standard_normal(m) * 0.1, sr)
    report('find_grid', abs(gb - 117.3) < 0.05 and abs(ge) < 15 and ni['fold'] < 0.08,
           f'{gb:.3f} bpm (made at 117.3), first downbeat {ge:+.0f} ms off; fold {gi["fold"]:.2f}, noise {ni["fold"]:.2f} (< 0.08)')
    print('  ->', 'every measure tells its cases apart' if ok else 'a measure is blind')
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('profile', help='a coarse profile of a mix')
    p.add_argument('wav')
    p.add_argument('--bpm', type=float, help='omit to find the grid (and check what it found)')
    p.add_argument('--bar1', type=float, default=0.0, help='time of the first downbeat, s')
    p.add_argument('--from', dest='b0', type=int)
    p.add_argument('--to', dest='b1', type=int)
    q = sub.add_parser('grid', help='find tempo and first downbeat')
    q.add_argument('wav')
    q.add_argument('--from-s', dest='t0', type=float, default=0.0)
    q.add_argument('--to-s', dest='t1', type=float)
    sub.add_parser('selftest', help='each measure against known cases')
    a = ap.parse_args()
    if a.cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    if a.cmd == 'grid':
        sr, x = load(a.wav)
        bpm, bar1, info = find_grid(x, sr, t0=a.t0, t1=a.t1)
        print(json.dumps({'bpm': round(bpm, 3), 'bar1': round(bar1, 4), **info}))
        sys.exit(0)
    if a.bpm is None:
        sr, x = load(a.wav)
        a.bpm, a.bar1, info = find_grid(x, sr)
        print(f'found {a.bpm:.3f} bpm, first downbeat {a.bar1:.3f} s {info} (a guess for the downbeat: check it)', file=sys.stderr)
    print(json.dumps(profile(a.wav, a.bpm, a.bar1, a.b0, a.b1), indent=1))
