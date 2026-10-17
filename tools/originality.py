"""Is our melody someone else's? Transcribe a 16th-note line from a finished mix and search it for ours, at every
alignment and transposition, against what random lines with the same rhythm score by the same search.

On a fixed-tempo grid (phase from the kick), each 16th step gets an onset strength (energy rise in 200-2500 Hz) and a
pitch (harmonic summation over MIDI 45-84 on a 70 ms window just after the step). Validated on our own track, whose
score we know: `selftest` transcribes Engine's intro and scores it against the score.

  python3 tools/originality.py selftest                                   (after python3 engine/song.py)
  python3 tools/originality.py transcribe track.wav T0 T1 --bpm 123       a section's 16th-step pitch classes
  python3 tools/originality.py compare track.wav T0 T1 --bpm 123 --ours 66,61,57,66,61,57,64,61,...
                                                                          our 16-step line(s) against it, and chance
  (--ours may be given more than once; MIDI numbers or note names like F#4)

What we found for Engine against the reference whose melody felt closest: our line matched 5 of 16 steps at best,
and a random 16-note line scores 5 by the same search. Use a section where the reference's line is exposed.
Keep the reference audio and its transcription out of version control.
"""
import argparse
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIDI = np.arange(45, 85)
HZ = lambda m: 440 * 2 ** ((m - 69) / 12)
NAMES = 'C C# D D# E F F# G G# A A# B'.split()


def load(p):
    sr, x = wavfile.read(p)
    x = x.astype(np.float64)
    if x.dtype.kind == 'i':
        x /= np.iinfo(x.dtype).max
    return sr, (x.mean(1) if x.ndim == 2 else x)


def _bp(x, sr, lo, hi):
    return signal.sosfilt(signal.butter(4, [lo, hi], 'band', fs=sr, output='sos'), x)


def kick_phase(m, sr, t0, t1, bpm):
    """Beat phase (s) from the low band's onset envelope folded on the beat."""
    lo = signal.sosfilt(signal.butter(4, 120, 'low', fs=sr, output='sos'), m[int(t0 * sr):int(t1 * sr)])
    env = signal.resample_poly(np.abs(signal.hilbert(lo)), 1, sr // 1000)    # 1 kHz
    on = np.maximum(np.diff(env), 0)
    beat = 60 / bpm
    ph = (np.arange(len(on)) / 1000) % beat
    hist, edges = np.histogram(ph, 200, (0, beat), weights=on)
    return t0 + edges[np.argmax(hist)]


def salience(seg, sr):
    n = 1 << 15
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n))
    f = np.fft.rfftfreq(n, 1 / sr)
    s = []
    for m in MIDI:
        v = 0.0
        for h, w in ((1, 1.0), (2, 0.8), (3, 0.6), (4, 0.45), (5, 0.35)):
            k = np.argmin(abs(f - h * HZ(m)))
            v += w * X[max(0, k - 2):k + 3].max()
        s.append(v)
    return np.array(s)


def transcribe(m, sr, t0, t1, bpm, phase=None):
    """[(time, onset dB, MIDI)] per 16th step in [t0, t1)."""
    step = 60 / bpm / 4
    ph = kick_phase(m, sr, t0, min(t1, t0 + 30), bpm) if phase is None else phase
    mid, band = _bp(m, sr, 200, 2500), _bp(m, sr, 150, 1400)
    t = ph + int(np.ceil((t0 - ph) / step)) * step
    out = []
    while t < t1 - 0.1:
        a, b = int((t - 0.03) * sr), int(t * sr)
        e0 = (mid[a:b] ** 2).mean() + 1e-20
        e1 = (mid[b + int(0.005 * sr):b + int(0.035 * sr)] ** 2).mean() + 1e-20
        sal = salience(band[b + int(0.012 * sr):b + int(0.082 * sr)], sr)
        out.append((t, 10 * np.log10(e1 / e0), int(MIDI[np.argmax(sal)])))
        t += step
    return out


def bars(tr, thresh=0.7):
    """16-step bars of pitch classes (None where no onset)."""
    return [[p % 12 if on > thresh else None for _, on, p in tr[i:i + 16]] for i in range(0, len(tr) - 15, 16)]


def best_match(rows, pat):
    """The most steps of a 16-step pitch-class line matched in any bar, at any rotation and transposition."""
    best = (0, 0, 0)
    for r in rows:
        for shift in range(16):
            rr = r[shift:] + r[:shift]
            for tr_ in range(12):
                same = sum(1 for a, b in zip(rr, pat) if a is not None and (a + tr_) % 12 == b)
                if same > best[0]:
                    best = (same, shift, tr_)
    return best


def chance(rows, n=300, seed=0, like=None):
    """What random lines score by the same search. With `like` (your line), each random line keeps its rhythm: the
    same note boundaries (a run of one pitch is one note), a random pitch per note. Use that: against free random
    lines a held or repeated note looks alarming (a bar of one held D scored 12/16 against a 95th percentile of 7)
    though any held note matches as well. Found by the second cold-test agent."""
    rng = np.random.default_rng(seed)
    if like is None:
        return [best_match(rows, list(rng.integers(0, 12, 16)))[0] for _ in range(n)]
    runs = np.r_[0, np.cumsum(np.diff(like) != 0)]
    return [best_match(rows, list(rng.integers(0, 12, runs[-1] + 1)[runs]))[0] for _ in range(n)]


def note(s):
    s = s.strip()
    if s.lstrip('-').isdigit():
        return int(s)
    name, octv = s[:-1].upper(), int(s[-1])
    return NAMES.index(name) + 12 * (octv + 1)


def selftest():
    """Engine's intro, bars 3-8 (the doots alone, then hats), against its score."""
    sys.path[:0] = [ROOT, os.path.join(ROOT, 'engine')]
    import song as P
    sr, m = load(os.path.join(ROOT, 'out', 'song.wav'))
    step = 60 / P.BPM / 4
    tr = transcribe(m, sr, P.tb(3), P.tb(9), P.BPM, phase=0.0)
    score = {round(P.tb(b) + x[0] * step, 3): x[1] for b in range(3, 9) for x in P.doot_pattern(b)}
    hit = pc = 0
    for t, on, p in tr:
        s = score.get(round(t, 3))
        if s is not None:
            hit += on > 0.7
            pc += (p % 12) == (s % 12)
    ph = kick_phase(m, sr, P.tb(9), P.tb(17), P.BPM)
    err = ((ph + 30 / P.BPM) % (60 / P.BPM) - 30 / P.BPM) * 1000
    rows = bars(tr)
    own = best_match(rows, [x % 12 for x in P.DOOT_A])[0]
    ch = chance(rows, 100)
    chr_ = chance(rows, 100, like=[x % 12 for x in P.DOOT_A])
    held = [2] * 16
    sh, chh = best_match(rows, held)[0], chance(rows, 100, like=held)
    ok = (pc >= 0.8 * len(score) and abs(err) < 15 and own >= 12 and own > np.percentile(chr_, 95) + 3
          and sh <= np.percentile(chh, 95))
    print(f'  {len(score)} scored notes: onsets found {hit}, pitch class right {pc} ({100 * pc / len(score):.0f}%, >= 80%)')
    print(f'  kick phase {err:+.1f} ms from the true downbeat (|err| < 15; it reads late by about 10 ms, the low band rising after the click)')
    print(f'  our own line found in our own intro: {own}/16 (>= 12); random lines {np.median(ch):.0f}, 95th pct {np.percentile(ch, 95):.0f}; '
          f'with its rhythm {np.median(chr_):.0f}, 95th pct {np.percentile(chr_, 95):.0f}')
    print(f'  a held note: {sh}/16; free random lines\' 95th pct {np.percentile(ch, 95):.0f}, held random notes\' '
          f'{np.percentile(chh, 95):.0f} (a held note must sit at chance against its own rhythm)')
    print('  ->', 'the transcription finds a known line, and only it' if ok else 'the transcription is blind here')
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['selftest', 'transcribe', 'compare'])
    ap.add_argument('wav', nargs='?')
    ap.add_argument('t0', nargs='?', type=float)
    ap.add_argument('t1', nargs='?', type=float)
    ap.add_argument('--bpm', type=float)
    ap.add_argument('--ours', action='append', default=[], help='a 16-step line, comma-separated')
    a = ap.parse_args()
    if a.cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    sr, m = load(a.wav)
    rows = bars(transcribe(m, sr, a.t0, a.t1, a.bpm))
    for r in rows:
        print('  ' + ' '.join(NAMES[p].ljust(2) if p is not None else '. ' for p in r))
    if a.cmd == 'compare':
        for line in a.ours:
            pat = [note(s) % 12 for s in line.split(',')]
            assert len(pat) == 16, 'a line is 16 steps'
            same, shift, tr_ = best_match(rows, pat)
            ch = chance(rows, like=pat)
            print(f'ours ({" ".join(NAMES[p] for p in pat)}): best match {same}/16 steps (rotate {shift}, transpose {tr_}); '
                  f'random lines with its rhythm score {np.median(ch):.0f}/16 (95th pct {np.percentile(ch, 95):.0f})')
