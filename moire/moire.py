"""Piece 6: two machines out of step.

Mark on piece 5: "ok now", but "pretty repetitive and generic". The idea that decides the rest:
the machines don't share a cycle.
  - the kick and the lead run in plain 4/4 bars (16 sixteenths);
  - the bouncer plays a figure 7 sixteenths long, so its head meets a downbeat every 7 bars;
  - the pinger (metal plucks) plays a figure 5 sixteenths long: every 5 bars;
  - all three agree every 35 bars (bars 36, 71, 106, 141): the drops, the breakdown, the end.
The lead plays its four-note phrases into the bars where the bouncer aligns (it speaks when the
machines agree), more often in the drops.

Mark on the sketch: "going somewhere good... makes me want to listen to more"; the drop stumbled
(fixed in v2); in the first minute he lost the beat, so the 0:40 and 1:08 entries seemed out of
time; the bouncer and pinger need long-term variation and a little less level. So: the grid comes
first (kick, an off-beat open hat, ticks; the bouncer enters at its first alignment, bar 8, the
pinger at its own, bar 16); a clap on 2 and 4 in the builds and drops; the figures change at
each alignment; the riff is 8 bars.

Form (124 bpm, E minor; 35-bar cycles):
  1-35    the grid, the bouncer (8), the pinger (16), the lead idling (22); take-away 35
  36-70   drop 1
  71-105  breakdown: the machines alone over a held chord; kick and lead back at 92; take-away 105
  106-140 drop 2: the lead plays in 7 (106-119), then calls back to its drop-1 self at the alignment (120)
  141-148 the last alignment: one hit, then the machines drift out

v4 (Mark on v3: "overall missing something"): a bed. Between the beats drop 1 lit a quarter of 200-4000 Hz
where the reference and Engine light half; a chord pad on the riff, pumped by the kick, held for the riff's first
four bars and answered with off-beat stabs in the last four; the breakdown carried by the same bed opening slowly,
its melody quieter.

  python3 moire/moire.py -> out/moire.wav, _stems.npz, _score.json
"""
import json
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'engine')]
import dsp                                  # the shared signal-processing kernel (../dsp.c, ../dsp.py)
import song as S                            # Engine, Learning to Sing: Moiré borrows its instruments and helpers

SR, CR = S.SR, S.CR
BPM = 124.0
BEAT = 60 / BPM
BAR = 4 * BEAT
SIX = BEAT / 4
NBARS, TAIL = 148, 5.0
HZ = S.HZ
tb = lambda bar, beat=0.0: (bar - 1) * BAR + beat * BEAT
R = lambda a, b: list(range(a, b + 1))


def kick_hit(sr, root=HZ(28)):
    """Engine's kick, retunable: it sweeps down into `root` (Hz; default E1). Engine's own is fixed at F#1."""
    t = np.arange(int(0.4 * sr)) / sr
    f = root + 150 * np.exp(-t / 0.011) + 22 * np.exp(-t / 0.06)
    env = np.exp(-np.clip(t - 0.03, 0, None) / 0.09) * np.clip(t / 0.0006, 0, 1)
    x = np.sin(2 * np.pi * np.cumsum(f) / sr) * env
    click = dsp.butter(np.random.default_rng(1).standard_normal(len(t)), (2000, 7000), sr, 'band') * np.exp(-t / 0.0025)
    y = np.tanh(0.8 * (x + 0.18 * click)) / np.tanh(0.8)
    return S.peq(y, sr, 180, -4.0, 0.8)


def tick(sr, rng):
    """A 17 ms narrow burst near 6.4 kHz."""
    n = int(0.017 * sr)
    x = rng.standard_normal(n) * np.hanning(n)
    return dsp.butter(x, (5600, 7200), sr, 'band', 2)


def bounce_note(sr, hz, accent, rng, vowel=(760, 1180)):
    """The bouncer's voice (Mark preferred it to a softer one): saw and pulse through two formant
    band-passes ("ah", 760 and 1180 Hz), with a 45-cent scoop up into the note."""
    t = np.arange(int(0.22 * sr)) / sr
    f = hz * 2 ** (-0.45 / 12 * np.exp(-t / 0.018))
    ph = np.cumsum(f) / sr
    x = 0.6 * (2 * (ph % 1) - 1) + 0.4 * np.sign(np.sin(2 * np.pi * ph) + 0.3)
    y = 0.55 * dsp.svf_bp(x, np.full(len(t), float(vowel[0])), sr, 3.5) + 0.45 * dsp.svf_bp(x, np.full(len(t), float(vowel[1])), sr, 4.0) + 0.12 * x
    env = np.clip(t / 0.003, 0, 1) * np.exp(-t / (0.07 + 0.05 * accent))
    return y * env


def lead_curves(tc, ev):
    """The lead's pitch (log-glided over 12 ms) and gate from its events (t, midi, dur, phrase)."""
    held = [e for e in ev if not e[3]]
    hook = [e for e in ev if e[3]]
    f0 = np.full(len(tc), HZ(28))
    for t0, m, d, _ in held:
        f0[tc >= t0] = HZ(m)
    gate = np.zeros(len(tc))
    for t0, m, d, _ in hook:
        on = (tc >= t0) & (tc < t0 + d)
        f0[on] = HZ(m)
        gate = np.maximum(gate, np.clip(np.minimum((tc - t0) / 0.004, (t0 + d - tc) / 0.012), 0, 1) * (tc >= t0) * (tc < t0 + d + 0.012))
    k = int(0.012 * CR)
    f0 = np.exp(np.convolve(np.pad(np.log(f0), (k, 0), mode='edge'), np.ones(k) / k, 'valid')[:len(tc)])
    return f0, gate

B_LEN, P_LEN = 7, 5                        # the bouncer's and the pinger's figures, in 16ths
GRAND = [36, 71, 106, 141]                 # 1 + k * lcm(16, 7, 5) / 16
TAKES = [35, 105]
ALIGN7 = [b for b in R(2, NBARS) if ((b - 1) * 16) % B_LEN == 0]
ALIGN5 = [b for b in R(2, NBARS) if ((b - 1) * 16) % P_LEN == 0]

# v2 (Mark on full v1: the bouncer and pinger vary too often and never leave, so they tire; drop 2
# is not exciting; the breakdown long; the lead "doesn't really go anywhere special"):
# v4: drop 1 gets a hinge, two bars without kick and clap (52-53) before its second half (Mark: "slightly repetitive")
HINGE = [52, 53]
KICK = [b for b in R(1, 34) + R(36, 70) + R(86, 104) + R(106, 140) if b not in HINGE]
OHAT = R(1, 34) + R(36, 70) + R(86, 104) + R(106, 140)
CLAP = [b for b in R(29, 34) + R(36, 70) + R(99, 104) + R(106, 140) if b not in HINGE]
# the machines take turns and sit out: the bouncer hands drop 1 to the pinger halfway, both have
# the breakdown, and in drop 2 the bouncer is gone because the lead plays in 7 itself
BOUNCE_ON = R(8, 34) + R(36, 45) + R(71, 78) + R(86, 91) + R(141, 148)     # v3: about half as much (Mark: still too much)
PING_ON = R(22, 34) + R(57, 64) + R(78, 85) + R(126, 136) + R(141, 148)
# v3: one figure per section, chosen (Mark: the variations "sound randomly chosen"): rising for the
# intro, drop 1 and the end; the mirror, falling, for the breakdown
B_SEC = {'intro': 0, 'drop1': 0, 'break': 1, 'drop2': 0, 'outro': 0}
P_SEC = {'intro': 0, 'drop1': 1, 'break': 0, 'drop2': 2, 'outro': 0}
# the breakdown's melody (one note per two bars from bar 71, over Em, Cmaj7, then the riff); the
# lead takes it up in drop 2's second half as the heads of its 7 (Mark: "pull in the melody it makes")
MEL = [71, 69, 67, 64, 67, 71, 72, 71, 69, 67, 66, 64, 67, 64, 66, 64, 67]
# the lead's arc: it learns the machines' meter. In drop 2 it plays a figure 7 sixteenths long
# (offsets from the bar's root, two octaves up; None = rest), its filter opening on each head
# v5 (Mark on v4: drop 2's lead "too weird / jazzy / dissonant. I'd redo it"; v4's figure, root fifth octave flat
# seventh, spelled a seventh chord on every root, and over C played B-flat and F): a cascade on the chord's own triad,
# falling from its head (indices from the head down; the heads carry the breakdown's melody from 113)
C7 = [0, -2, -1, -3, -2, -4, -3]
TICKS = R(1, 34) + R(36, 104) + R(106, 140)
DROPS = R(36, 70) + R(106, 140)
BOUNCE_FROM, PING_FROM = 8, 16
# v4 (Mark on v3: drop 2's lead "gets tired by ~3:50, needs something else. Maybe have it call back to the first
# lead section"): the 7 runs 14 bars, to the alignment at 120; from there the lead is its drop-1 self again
FIG7 = R(106, 119)
CALLBACK = R(120, 140)
HYBRID = R(128, 140)
LEAD_ON = R(22, 70) + R(86, 105) + CALLBACK     # held bass notes (the 7's bars are carried by a plain bass voice)
# v4, the bed (Mark on v3: "overall missing something"; between the beats drop 1 lit a quarter of 200-4000 Hz
# where the reference and Engine light half, and held a third as much): a chord pad on the riff, pumped by the kick
# v6 (Mark on v5: drop 2's solo "could drop the bed"): out under the 7 (106-118), swelling in through 119, back
# for the callback (120)
BED_ON = R(22, 34) + R(36, 104) + R(119, NBARS)

# the riff, 8 bars: Em Am D Bm | Em Am C D
BASS = [28, 33, 26, 35, 28, 33, 24, 26]
CHORDS = [[64, 67, 71, 76, 79], [64, 69, 72, 76, 81], [62, 66, 69, 74, 78], [62, 66, 71, 74, 78],
          [64, 67, 71, 76, 79], [64, 69, 72, 76, 81], [64, 67, 72, 76, 79], [62, 66, 69, 74, 78]]
PHRASES = [([43, 45, 47], 48), ([52, 50, 48], 45), ([42, 45, 47], 50), ([55, 54, 50], 52), ([47, 50, 52], 55), ([57, 55, 52], 50)]
# six phrases against an 8-bar riff: they line up every 24 bars, so the lead is out of step too
# the figures change at each of their alignments (long-term variation)
B_FIGS = [[0, 2, 4, 1, 3, 2, 4], [4, 2, 0, 3, 1, 2, 0], [0, 1, 2, 3, 4, 3, 2], [2, 4, 1, 3, 0, 4, 1]]
B_OCT = [0, 0, 0, 0]                      # no octave moves: down left its own band, up risked the chiptune
P_FIGS = [[83, 79, 86, 76, 81], [76, 83, 79, 88, 81], [86, 81, 83, 76, 79]]
import os
BED_KIND = os.environ.get('MOIRE_BED', 'glass')            # Mark chose it from the audition (v5)
G_GUIRO, G_DROPS = 0.55, 0.03          # the guiro's peaks about 24 dB under the kick's (at 0.05: 45, inaudible)
G_LEAD, G_BOUNCE, G_PING, G_BED = 0.32, 0.63, 0.37, 0.28        # v6: machines +3 dB (Mark: cut from most of the tune, they can be louder)


def riff_i(bar):
    return (bar - 1) % 8


def sec(bar):
    if bar < GRAND[0]:
        return 'intro'
    if bar < GRAND[1]:
        return 'drop1'
    if bar < GRAND[2]:
        return 'break'
    if bar < GRAND[3]:
        return 'drop2'
    return 'outro'


def tones(bar, lo, hi):
    """The bar's triad, every octave, lo..hi."""
    pcs = {m % 12 for m in CHORDS[riff_i(bar)][:3]}
    return [m for m in range(lo, hi + 1) if m % 12 in pcs]


def lead_score():
    ev = [(tb(b), BASS[riff_i(b)], BAR, False) for b in LEAD_ON]
    targets = set(b for b in ALIGN7 if b in LEAD_ON and b > LEAD_ON[0])
    targets |= set(b for b in R(37, 70) if b % 2 == 0)                 # drop 1: every other bar
    targets |= set(b for b in CALLBACK if b % 2 == 0 and b < HYBRID[0])  # the callback: as drop 1
    targets |= set(TAKES[i] + 1 for i in range(2)) | {71, 141}
    # the hybrid (v5; Mark: after the callback "don't just stay there, but start hybridizing"): drop 1's held bass,
    # and into the bar the engine revs up through the next chord in a 7-sixteenth run (drop 2's 7, now a pickup),
    # alternating with drop 1's phrases at first, then every bar
    runs = set(b for b in R(HYBRID[0], 141) if b > HYBRID[0] + 5 or b % 2 == 1)
    targets |= set(R(HYBRID[0], 141))
    for bar in sorted(runs):
        T = tones(bar, 28, 84)
        land = min(T, key=lambda m: abs(m - (64 if bar < 141 else 76)))
        i = T.index(land)
        for j, m in enumerate(T[i - 7:i]):
            ev.append((tb(bar - 1, (9 + j) / 4), m, 0.09, True))
        ev.append((tb(bar), land, 0.10, True))
    for k, bar in enumerate(sorted(targets - runs)):
        notes, land = PHRASES[k % len(PHRASES)]
        up8 = 12 if (sec(bar) == 'drop2' or 54 <= bar <= 70) and k % 2 else 0      # v4: drop 1's second half climbs too (Mark: drop 1 "slightly repetitive")
        for j, m in enumerate(notes):
            ev.append((tb(bar - 1, 3.25 + 0.25 * j), m + up8, 0.09, True))
        ev.append((tb(bar), land + up8, 0.10, True))
    for k in range(len(FIG7) * 16):                                    # drop 2: the lead in 7
        t = tb(106) + k * SIX
        if k < 2 or t >= tb(FIG7[-1], 3.0):
            continue                                                   # the landings into 106 and (pickups) into 120
        bar = int(t // BAR) + 1
        step = (105 * 16 + k) % B_LEN                                  # in phase with the bouncer's count
        T = tones(bar, 43, 80)
        want = 67 if bar < 113 else MEL[((bar - 113) // 2) % len(MEL)]     # the head: G4 first, then the breakdown's melody
        h = min(range(4, len(T)), key=lambda i: abs(T[i] - want))
        ev.append((t, T[h + C7[step % len(C7)]], 0.10 if step == 0 else 0.07, True))
    return sorted(ev)


def presence(tc, bars):
    """1 inside the bars, fading in over 3 bars at each start and out over 2 at each end
    (v3; Mark: the pinger's return at 3:58 "suddenly sounds weird")."""
    env = np.zeros(len(tc))
    bs = sorted(bars)
    segs = np.split(np.array(bs), np.nonzero(np.diff(bs) > 1)[0] + 1)
    for sg in segs:
        s0, s1 = sg[0], sg[-1]
        x = np.clip((tc - tb(s0)) / (4 * BAR), 0, 1)
        up_ = 10 ** ((-24 + 24 * x) / 20) * (x < 1) + (x >= 1)           # even in dB: 6 dB a bar from -24
        dn = np.clip((tb(s1 + 1) - tc) / (2 * BAR), 0, 1)
        env = np.maximum(env, np.minimum(up_, dn) * (tc >= tb(s0)) * (tc < tb(s1 + 1)))
    return env


def lead_level(tc):
    pts = [(0, 0), (tb(22) - 0.01, 0), (tb(22), 0.0), (tb(24), 0.62), (tb(35), 0.62), (tb(35) + 0.05, 0.15)]
    for T in TAKES:
        if T == 35:
            pts += [(tb(T, 3.25) - 0.01, 0.15), (tb(T, 3.25), 0.45), (tb(T, 3.5), 0.7), (tb(T, 3.75), 0.9), (tb(T + 1), 1.0)]
    pts += [(tb(71), 1.0), (tb(73), 0.0), (tb(86) - 0.01, 0.0), (tb(86), 0.0), (tb(88), 0.62), (tb(105), 0.62), (tb(105) + 0.05, 0.15),
            (tb(105, 3.25) - 0.01, 0.15), (tb(105, 3.25), 0.45), (tb(105, 3.5), 0.7), (tb(105, 3.75), 0.9), (tb(106), 1.0),
            (tb(141), 1.0), (tb(144), 0.0), (tb(NBARS + 3), 0.0)]
    t, v = zip(*pts)
    return np.interp(tc, t, v)


def chorus(x, sr, seed, ms=7.0, depth_ms=2.2, hz=0.43):
    """A modulated short delay (Juno-ish), mixed with the dry."""
    n = np.arange(len(x))
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi)
    d = (ms + depth_ms * np.sin(2 * np.pi * hz * n / sr + ph)) * sr / 1000
    return 0.6 * x + 0.55 * np.interp(n - d, n, x)


def bed_voice(kind, fqs, fcu, since, N, rng):
    """The bed's instrument (v5 audition; Mark on v4: the bed "helps a lot... sounds full now", but "pretty
    basic synth... needs something more fancy"). 'basic' is v4's; Mark chose 'glass' of the four.

    kind: 'basic' | 'lush' | 'analog' | 'glass'.  fqs: one per-sample frequency array (Hz, length N) per chord note.
    fcu: per-sample filter cutoff (Hz, length N).  since: per-sample seconds since the current chord began (the bells
    strike at 0).  N: samples at SR (48 kHz).  rng: a numpy Generator.  Returns (L, R), unscaled: set the level yourself.
    Reusable: `import moire as M; L, R = M.bed_voice('glass', fqs, fc, since, N, np.random.default_rng(1))`."""
    L, R = np.zeros(N), np.zeros(N)
    if kind == 'basic':
        for fq in fqs:
            for c, dets in ((L, (-8, 0, 7)), (R, (-5, 2, 9))):
                for d in dets:
                    c += dsp.saw(fq * 2 ** (d / 1200), SR, phase=rng.uniform())
        L = dsp.butter(dsp.ladder(L / 12, fcu, SR, res=0.25, drive=1.2), 110, SR, 'high', 2)
        R = dsp.butter(dsp.ladder(R / 12, fcu, SR, res=0.25, drive=1.2), 110, SR, 'high', 2)
        return L, R
    if kind in ('lush', 'glass'):
        # seven unison saws a note, spread over the stereo field, each wandering a few cents; a slow phaser; chorus
        spread = np.linspace(-18, 18, 7)
        pans = np.linspace(0.08, 0.92, 7)
        for j, fq in enumerate(fqs):
            for i, (d, p) in enumerate(zip(spread, pans[rng.permutation(7)])):
                w = S.drift(N // 480 + 2, SR / 480, 4.0, 100 + 7 * j + i, tau=3.0)
                v = dsp.saw(fq * 2 ** ((d + np.repeat(w, 480)[:N]) / 1200), SR, phase=rng.uniform())
                L += (1 - p) * v
                R += p * v
        L = dsp.ladder(L / 12, fcu, SR, res=0.3, drive=1.1)
        R = dsp.ladder(R / 12, fcu, SR, res=0.3, drive=1.1)
        L = 0.5 * L + 0.5 * dsp.phaser(L, SR, rate=0.045, fmin=300, fmax=2600, stages=6, fb=0.35, phase=0.0)
        R = 0.5 * R + 0.5 * dsp.phaser(R, SR, rate=0.045, fmin=300, fmax=2600, stages=6, fb=0.35, phase=1.6)
        L, R = chorus(L, SR, 1), chorus(R, SR, 2, ms=9.0)
        if kind == 'glass':
            # a glass layer an octave over the top two notes: FM bells struck at each chord, into a long reverb,
            # and a breath of air (5-10 kHz noise)
            gl = np.zeros(N)
            idx = 0.35 + 1.6 * np.exp(-since / 0.5)
            for fq in fqs[2:]:
                ph = 2 * np.pi * np.cumsum(2 * fq) / SR
                gl += np.sin(ph + idx * np.sin(ph * 1.0 + 0.3)) * (0.35 + 0.65 * np.exp(-since / 1.2))
            gL, gR = dsp.reverb(gl * 0.12, gl * 0.12, SR, rt60=5.0, damp_hz=6000)
            air = dsp.butter(dsp.butter(np.random.default_rng(31).standard_normal(N), 5000, SR, 'high', 2), 10000, SR, 'low', 2)
            L, R = 0.75 * L + gL + 0.05 * air, 0.75 * R + gR + 0.05 * np.roll(air, 977)
        return dsp.butter(L, 110, SR, 'high', 2), dsp.butter(R, 110, SR, 'high', 2)
    if kind == 'analog':
        # a poly-synth's breath: three saws a note, a resonant filter struck open at each chord (and each stab), drive
        for fq in fqs:
            for c, dets in ((L, (-9, 0, 6)), (R, (-6, 3, 10))):
                for d in dets:
                    c += dsp.saw(fq * 2 ** (d / 1200), SR, phase=rng.uniform())
        fe = fcu * (1 + 1.1 * np.exp(-since / 0.45))
        L = 0.7 * dsp.ladder(L / 12, fe, SR, res=0.5, drive=1.7)
        R = 0.7 * dsp.ladder(R / 12, fe, SR, res=0.5, drive=1.7)      # level-matched to the others for the audition
        L, R = chorus(L, SR, 1, ms=5.0, depth_ms=3.0, hz=0.5), chorus(R, SR, 2, ms=6.0, depth_ms=3.0, hz=0.5)
        return dsp.butter(L, 110, SR, 'high', 2), dsp.butter(R, 110, SR, 'high', 2)
    raise ValueError(kind)


def render():
    dur = NBARS * BAR + TAIL
    N = int(dur * SR)
    tc = np.arange(int(dur * CR)) / CR
    rng = np.random.default_rng(6)
    up = lambda c: S.up(c, tc, N, SR)
    fade = np.clip((dur - 1.0 - tc) / 3.0, 0, 1)
    dsec = lambda name: np.array([sec(int(t // BAR) + 1) == name for t in tc], float)
    in_drop2 = dsec('drop2')
    in_fig = np.array([int(t // BAR) + 1 in FIG7 for t in tc], float)
    in_cb = np.array([int(t // BAR) + 1 in CALLBACK for t in tc], float)

    # --- the lead
    ev = lead_score()
    f0, gate = lead_curves(tc, ev)
    fh = np.full(len(tc), HZ(BASS[0]))
    for b in R(1, NBARS):
        fh[tc >= tb(b)] = HZ(BASS[riff_i(b)])
    k_ = int(0.012 * CR)
    fh = np.exp(np.convolve(np.pad(np.log(fh), (k_, 0), mode='edge'), np.ones(k_) / k_, 'valid')[:len(tc)])
    cut = np.where(in_fig > 0, 600 + 5.0 * f0, (200 + 2.5 * fh) * (1 + 0.2 * in_cb))
    cut = np.where((in_cb > 0) & (gate > 0), np.maximum(cut, 500 + 4.0 * f0), cut)     # v5: the hybrid's runs open with their notes     # v3: drop 2's filter follows its notes (it was set from the bass: muffled)
    for t0 in [e[0] for e in ev if e[3] and e[2] >= 0.1 and tb(106) < e[0] < tb(FIG7[-1] + 1)]:     # drop 2: open on each figure head
        cut = cut + (5000 - cut) * S.analog_env(tc, t0, 0.01, 0.08, 0.22) * (tc >= t0)
    for t0 in [e[0] for e in ev if e[3] and abs(((e[0] / SIX) % 16) - 13) < 0.01]:
        cut = cut + (2400 * (1 + 0.25 * (sec(int(t0 // BAR) + 2) == 'drop2')) - cut) * S.analog_env(tc, t0, 0.02, 0.36, 0.25) * (tc >= t0)
    lvl = lead_level(tc) * fade
    # drop 2: the figure is articulated (each note sounds and decays, the head louder, rests silent),
    # or the engine voice just slides between pitches and the 7 isn't heard
    art = np.zeros(len(tc))
    for t0, m, d, ph in ev:
        if ph and tb(106) + 2 * SIX <= t0 < tb(FIG7[-1], 3.0):
            g = 1.0 if d >= 0.1 else 0.55
            art = np.maximum(art, g * np.exp(-np.clip(tc - t0, 0, None) / 0.12) * np.clip((tc - t0) / 0.003, 0, 1) * (tc >= t0))
    d2 = (tc >= tb(106) + 2 * SIX) & (tc < tb(FIG7[-1], 3.0))
    lvl = np.where(d2, 1.4 * lvl * (0.3 + 0.7 * art), lvl)          # and +3 dB: drop 2 is the lead's
    dc = {k: np.zeros(len(tc)) for k in S.KINDS}
    dc['drive'] = 0.18 * dsec('drop1') + 0.15 * in_fig + 0.22 * in_cb              # v3: less drive and grit in drop 2 ("strangled")
    for bar, beat, kind, amt in [(44, 2, 'growl', 0.45), (60, 2, 'fold', 0.4), (114, 2, 'growl', 0.5), (124, 0, 'fold', 0.45), (132, 2, 'growl', 0.5)]:
        dc[kind] = np.maximum(dc[kind], amt * S.analog_env(tc, tb(bar, beat), 0.25, 1.0, 1.0))
    S.BASS = [(1, 0, BASS[0])]
    S.glide_curve = lambda tc_, events, glide: fh
    grit = 0.06 + 0.06 * dsec('drop1') + 0.04 * in_fig + 0.07 * in_cb
    eL, eR = S.cloog(tc, f0, cut, lvl, dc, np.clip(3.0 + 2.0 * gate, 1, 22), 0.18 * np.ones(len(tc)), grit)
    eL, eR = (np.pad(x, (0, max(0, N - len(x))))[:N] for x in (eL, eR))
    # drop 2: the plain bass carries the bassline, so the engine's own low path and sub (which follow
    # it) only muddy the figure: high-pass the lead there
    hp = up(np.convolve(in_fig, np.ones(400) / 400, 'same'))
    eL = eL * (1 - hp) + dsp.butter(eL, 120, SR, 'high', 4) * hp
    eR = eR * (1 - hp) + dsp.butter(eR, 120, SR, 'high', 4) * hp
    land = np.zeros(len(tc))
    for t0, m, d, ph in ev:
        if ph and d >= 0.1:
            land = np.maximum(land, ((tc >= t0) & (tc < t0 + d)).astype(float))
    send = dsp.butter((eL + eR) / 2, 200, SR, 'high', 2) * up(np.convolve(land, np.ones(8) / 8, 'same'))
    xL = dsp.tapedelay(send, SR, BEAT, wow_ms=1.2, wow_hz=0.5, fb=0.42, damp_hz=3200, phase=0.0)
    xR = dsp.tapedelay(send, SR, BEAT + 0.004, wow_ms=1.4, wow_hz=0.43, fb=0.42, damp_hz=3000, phase=1.3)

    # --- the bouncer: 7 steps, free of the bar; its figure and octave change at each alignment
    bL, bR = np.zeros(N), np.zeros(N)
    acc7 = []
    for k in range((NBARS - BOUNCE_FROM + 1) * 16):
        t = tb(BOUNCE_FROM) + k * SIX
        bar = int(t // BAR) + 1
        if bar in TAKES or bar > NBARS or bar not in BOUNCE_ON:
            continue
        step = ((BOUNCE_FROM - 1) * 16 + k) % B_LEN
        n_al = sum(1 for a in ALIGN7 if BOUNCE_FROM <= a <= bar)
        fig = B_FIGS[B_SEC[sec(bar)]]
        m = CHORDS[riff_i(bar)][fig[step % len(fig)]] + B_OCT[(n_al - 1) % len(B_OCT)]
        acc = 1.0 if step == 0 else 0.4
        if step == 0:
            acc7.append(t)
        x = bounce_note(SR, HZ(m), acc * (1.6 if step == 0 else 1.0), rng) * acc
        pan = 0.42 * np.sin(2 * np.pi * (k * 5 / 16 + bar * 0.17))
        S.place(bL, x, [(t, 0.5 - pan)], SR)
        S.place(bR, x, [(t, 0.5 + pan)], SR)
    bshape = presence(tc, BOUNCE_ON) * np.interp(tc, [tb(36), tb(71), tb(86), tb(141)], [0.7, 1.0, 1.0, 0.75])
    bL, bR = bL * up(bshape * fade), bR * up(bshape * fade)
    # v4: less echo in the breakdown, where the machines are alone: its dotted 8th (3 sixteenths) blurred the 7 there
    esend = up(1 - 0.65 * np.convolve(dsec('break'), np.ones(500) / 500, 'same'))
    pL, pR = dsp.pingpong((bL + bR) * 0.8 * esend, (bL + bR) * 0.2 * esend, 0.75 * BEAT, SR, fb=0.45, damp_hz=4200)

    # --- the pinger: 5 steps; figure changes at each of its alignments; sparse in alternate cycles
    gL, gR = np.zeros(N), np.zeros(N)
    acc5 = []
    kc = {}
    for k in range((NBARS - PING_FROM + 1) * 16):
        t = tb(PING_FROM) + k * SIX
        bar = int(t // BAR) + 1
        if bar in TAKES or bar > NBARS or bar not in PING_ON:
            continue
        step = ((PING_FROM - 1) * 16 + k) % P_LEN
        n_al = sum(1 for a in ALIGN5 if PING_FROM <= a <= bar)
        fig = P_FIGS[P_SEC[sec(bar)]]
        if (n_al % 4 == 3) and step % 2 == 1:
            continue                                                     # a sparse cycle now and then
        m = fig[step % len(fig)]
        if m not in kc:
            kc[m] = S.ks_note(SR, HZ(m))
        acc = 1.0 if step == 0 else 0.35
        if step == 0:
            acc5.append(t)
        p = 0.5 + 0.3 * np.cos(2 * np.pi * k / 11)
        S.place(gL, kc[m] * acc, [(t, 1 - p)], SR)
        S.place(gR, kc[m] * acc, [(t, p)], SR)
    gshape = presence(tc, PING_ON)
    gL, gR = gL * up(gshape * fade), gR * up(gshape * fade)

    # --- the grid: kick, the off-beat open hat, the clap on 2 and 4, ticks (quarters in the take-aways)
    kicks = [tb(b, i) for b in KICK for i in range(4)] + [tb(141)]
    kick = np.zeros(N)
    S.place(kick, kick_hit(SR), [(k, 1.0) for k in kicks], SR, 0.70)
    hL, hR = np.zeros(N), np.zeros(N)
    for b in OHAT:
        for i in range(4):
            h = S.spaghetti(SR, rng)
            h = np.concatenate([h, np.zeros(len(h))]) * np.exp(-np.arange(2 * len(h)) / SR / 0.11)
            p = 0.5 + (0.12 if i % 2 else -0.12)
            S.place(hL, h, [(tb(b, i + 0.5), 1 - p)], SR)
            S.place(hR, h, [(tb(b, i + 0.5), p)], SR)
    clap = np.zeros(N)
    S.place(clap, S.clap_hit(SR), [(tb(b, i), 1.0) for b in CLAP for i in (1, 3)], SR)
    tL, tR = np.zeros(N), np.zeros(N)
    for b in TICKS + TAKES:
        for i in range(16):
            if b in TAKES and i % 4:
                continue
            if sec(b) == 'break' and i % 2:
                continue
            p = 0.5 + rng.uniform(-0.3, 0.3)
            g = (0.35 if b in TAKES else (0.5, 0.25, 0.4, 0.25)[i % 4]) * rng.uniform(0.8, 1.0)
            h = tick(SR, rng)
            S.place(tL, h, [(tb(b, i / 4), g * (1 - p))], SR)
            S.place(tR, h, [(tb(b, i / 4), g * p)], SR)
    boom = np.zeros(N)
    t_ = np.arange(int(1.2 * SR)) / SR
    hit = np.sin(2 * np.pi * np.cumsum(HZ(28) * (1 + 0.6 * np.exp(-t_ / 0.04))) / SR) * np.exp(-t_ / 0.45)
    S.place(boom, hit, [(tb(b), 1.0 if b in GRAND else 0.45) for b in sorted(set([a for a in ALIGN7 if 8 <= a <= GRAND[3]] + GRAND))], SR)      # v7: none after 141 (148 is an alignment: a boom 20 dB out of the fade)
    nz = np.random.default_rng(9).standard_normal(N)
    rum = dsp.butter(dsp.butter(nz, 70, SR, 'low', 4), 28, SR, 'high', 2) * (0.6 + 0.4 * np.cos(2 * np.pi * np.arange(N) / SR / BEAT) ** 2)
    rum *= up(np.array([sec(int(t // BAR) + 1) in ('drop1', 'drop2') and int(t // BAR) + 1 not in HINGE for t in tc], float) * fade)

    # --- the bed (v4): the riff's chords an octave under the bouncer's, detuned saws through a ladder, pumped by
    # the kick. It enters dark under the lead's idling (22), takes the drops, and carries the breakdown, opening
    # from 380 Hz to 2.2 kHz across it (v3's breakdown was sine chords: Mark, "a bit stale to be that loud")
    def bed_chord(bar):
        if 71 <= bar < 78:
            return [52, 55, 59, 64]                                      # Em
        if 78 <= bar < 86:
            return [48, 52, 55, 59]                                      # Cmaj7
        if bar >= GRAND[3]:
            return [m - 12 for m in CHORDS[0][1:5]]                      # v7: the ending, Em held
        lo = 1 if 54 <= bar <= 70 or bar >= 120 else 0                  # the voicing steps up for drop 1's second half and the callback
        return [m - 12 for m in CHORDS[riff_i(bar)][lo:lo + 4]]
    barc = np.minimum((tc // BAR).astype(int) + 1, NBARS)
    lg = lambda pts: np.interp(tc, *zip(*pts))
    bed_db = lg([(tb(22), -30), (tb(35), -9), (tb(36), 2), (tb(70), 3), (tb(71), -9), (tb(104), -3), (tb(106), -3), (tb(119), -1), (tb(120), 1.5), (tb(141), 1.5), (tb(144), -1), (tb(NBARS) + 2.5, -40)])
    bed_fc = np.exp(lg([(tb(22), np.log(250)), (tb(35), np.log(700)), (tb(36), np.log(1500)), (tb(52), np.log(1700)), (tb(54), np.log(2000)), (tb(70), np.log(2500)),
                        (tb(71), np.log(380)), (tb(104), np.log(2200)), (tb(106), np.log(1500)), (tb(140), np.log(1900)), (tb(141), np.log(2200)), (tb(NBARS) + 2.5, np.log(350))]))
    bed_fc *= 1 + 0.2 * np.sin(2 * np.pi * tc / (6 * BAR))           # 6 bars: out of step with the riff's halves
    bed_on = np.convolve(np.isin(barc, BED_ON).astype(float), np.ones(30) / 30, 'same')
    sm = lambda v: np.exp(np.convolve(np.pad(np.log(v), (20, 0), mode='edge'), np.ones(20) / 20, 'valid')[:len(tc)])
    fqs = [up(sm(np.array([HZ(bed_chord(b)[j]) for b in barc]))) for j in range(4)]
    since = up(np.where(barc >= GRAND[3], tc - tb(GRAND[3]), tc - tb(barc)))     # seconds since the bar began (the ending: since 141)
    bedL, bedR = bed_voice(BED_KIND, fqs, up(bed_fc), since, N, rng)
    # call and answer inside the riff (v4; drop 1 read 0.968 on the repetition measure with a held bed, the riff's halves
    # sharing three chords): held for its first four bars, off-beat stabs for the last four, in drop 1 and the callback
    gate = np.ones(len(tc))
    for b in R(36, 70) + CALLBACK:
        if riff_i(b) >= 4 and b not in HINGE:
            m_ = (tc >= tb(b)) & (tc < tb(b + 1))
            st = np.zeros(m_.sum())
            for i in range(4):
                u = tc[m_] - tb(b, i + 0.5)
                st = np.maximum(st, 1.3 * np.exp(-np.clip(u, 0, None) / 0.13) * np.clip(u / 0.004, 0, 1) * (u >= 0))
            gate[m_] = st
    benv = up(10 ** (bed_db / 20) * bed_on * fade * gate)
    pump = S.sidechain(kicks, N, SR, 0.7, rel=0.22)
    swell = np.where((tc >= tb(119)) & (tc < tb(120)), np.clip((tc - tb(119)) / BAR, 0, 1) ** 2, 1.0)
    benv = benv * up(swell)
    bedL, bedR = bedL * benv * pump, bedR * benv * pump
    # v6: the bed steps aside in the bouncer's band (600-1400 Hz, -12 dB; -7 left it 1-2 dB under in drop 1 and at 2:44) while it plays: the glass bed sat 2-9 dB over
    # it there (Mark, the breakdown at 2:44: "I can't hear it at all")
    duck = up(np.convolve(presence(tc, BOUNCE_ON), np.ones(100) / 100, 'same')) * (1 - 10 ** (-12 / 20))
    sos_b = signal.butter(2, [600, 1400], 'band', fs=SR, output='sos')
    bedL = bedL - duck * signal.sosfiltfilt(sos_b, bedL)
    bedR = bedR - duck * signal.sosfiltfilt(sos_b, bedR)
    # the breakdown's melody, a note every two bars: a soft saw voice (v3's sine was the stale hum), swelling, quieter
    pad = np.zeros(N)
    for k, m in enumerate(MEL):
        a, b_ = int(tb(71 + 2 * k) * SR), int(tb(73 + 2 * k) * SR)
        if a >= N:
            break
        u = np.arange(b_ - a) / SR
        sw = np.clip(u / 0.8, 0, 1) * np.clip((u[-1] - u) / 1.2, 0, 1)
        f_ = HZ(m) * (1 + 0.004 * np.sin(2 * np.pi * 4.8 * u) * np.clip((u - 0.6) / 1.0, 0, 1))
        v = dsp.saw(f_, SR, phase=rng.uniform()) + dsp.saw(f_ * 2 ** (7 / 1200), SR, phase=rng.uniform())
        pad[a:b_] += 0.5 * sw * dsp.ladder(v, HZ(m) * (2.0 + 2.5 * sw), SR, res=0.2)
    penv = np.interp(tc, [tb(71), tb(75), tb(100), tb(105)], [0, 1, 1, 0]) * (tc >= tb(71))
    pad *= up(np.convolve(penv, np.ones(200) / 200, 'same')) * 0.032      # v4: -4 dB (Mark: "stale to be that loud"; it masked the machines' figures)
    padL, padR = dsp.reverb(pad, pad, SR, rt60=4.0, damp_hz=3000)

    # --- the build into drop 2: a noise riser; crashes at its alignments; a plain bass under the lead's 7
    tt2 = np.arange(N) / SR
    rz = np.random.default_rng(11).standard_normal(N)
    a, b_ = int(tb(101) * SR), int(tb(106) * SR)
    riser = np.zeros(N)
    fc = np.geomspace(300, 6000, b_ - a)
    seg = rz[a:b_]
    riser[a:b_] = dsp.svf_bp(seg, fc, SR, 1.5)
    riser[a:b_] *= np.linspace(0, 1, b_ - a) ** 2
    crash = np.zeros(N)
    cz = dsp.butter(np.random.default_rng(12).standard_normal(int(2.0 * SR)), 4000, SR, 'high', 2) * np.exp(-np.arange(int(2.0 * SR)) / SR / 0.7)
    S.place(crash, cz, [(tb(b), 1.0 if b in (36, 106) else 0.6) for b in [36, 54, 106] + [a_ for a_ in ALIGN7 if 106 < a_ < 141]], SR)
    # --- v7, the detail pass: transitions and decorations, each placed for its section (Mark on the reference: little
    # decorations "add some extra texture")
    # a softer riser into drop 1 (32-35; drop 2 had the only one)
    a, b_ = int(tb(32) * SR), int(tb(36) * SR)
    riser[a:b_] = 0.7 * dsp.svf_bp(rz[a:b_], np.geomspace(400, 5000, b_ - a), SR, 1.5) * np.linspace(0, 1, b_ - a) ** 3
    # reverse swells (a crash backwards, one bar) into the hinge's end (54) and the callback (120)
    sw_ = cz[:int(BAR * SR)][::-1] * np.linspace(0, 1, int(BAR * SR)) ** 2
    for b in (54, 120):
        S.place(crash, sw_ * 0.6, [(tb(b) - BAR, 1.0)], SR)
    # a fall into the breakdown: noise sweeping down from 5 kHz to 300 Hz over its first two bars
    a, b_ = int(tb(71) * SR), int(tb(73) * SR)
    riser[a:b_] += 0.8 * dsp.svf_bp(rz[a:b_], np.geomspace(5000, 300, b_ - a), SR, 1.5) * np.linspace(1, 0, b_ - a) ** 2
    # a guiro: a scrape of 16-22 ticks through two resonances, into beat 2 and a short one on 4's "and"; the intro after
    # the grid's first 8 bars, drop 1's first half, the callback and hybrid, the ending; silent in the breakdown and the solo
    GUIRO_ON = R(9, 34) + R(36, 51) + R(120, 140) + R(141, 145)
    def guiro(n, length):
        out = np.zeros(int((length + 0.03) * SR))
        tk = np.cumsum(np.linspace(1.15, 0.85, n)); tk = tk / tk[-1] * length
        click = np.random.default_rng(41).standard_normal(int(0.0015 * SR)) * np.exp(-np.arange(int(0.0015 * SR)) / SR / 0.0004)
        for i, t_ in enumerate(tk):
            k = int(t_ * SR)
            out[k:k + len(click)] += click * (0.6 + 0.4 * np.sin(np.pi * i / n))
        return 0.6 * dsp.svf_bp(out, np.full(len(out), 3600.0), SR, 5.0) + 0.4 * dsp.svf_bp(out, np.full(len(out), 2100.0), SR, 4.0)
    gu_long, gu_short = guiro(20, 0.13), guiro(9, 0.055)
    guL, guR = np.zeros(N), np.zeros(N)
    for b in GUIRO_ON:
        p = 0.5 + (0.25 if b % 2 else -0.25)
        g_ = 0.6 if b in R(141, 145) else 1.0
        S.place(guL, gu_long * g_, [(tb(b, 1.0) - 0.13, 1 - p)], SR)
        S.place(guR, gu_long * g_, [(tb(b, 1.0) - 0.13, p)], SR)
        S.place(guL, gu_short * g_, [(tb(b, 3.5), p)], SR)
        S.place(guR, gu_short * g_, [(tb(b, 3.5), 1 - p)], SR)
    gshape_ = up(presence(tc, GUIRO_ON) * fade)
    guL, guR = guL * gshape_, guR * gshape_
    # the breakdown's melody, echoed: a glass drop two octaves over each of its notes, into the reverb
    drops = np.zeros(N)
    u_ = np.arange(int(1.5 * SR)) / SR
    for k, m in enumerate(MEL):
        if 71 + 2 * k > 104:
            break
        f_ = HZ(m + 24)
        S.place(drops, np.sin(2 * np.pi * f_ * u_ + 1.2 * np.exp(-u_ / 0.15) * np.sin(2 * np.pi * 2 * f_ * u_)) * np.exp(-u_ / 0.35) * np.clip(u_ / 0.002, 0, 1),
                [(tb(71 + 2 * k, 0.5), 1.0)], SR)
    dL, dR = dsp.reverb(drops, np.roll(drops, int(0.011 * SR)), SR, rt60=4.5, damp_hz=7000)
    bass_amp = up(np.convolve(in_fig, np.ones(60) / 60, 'same') * fade)       # v4: hands the bassline back to the engine at 120
    bv = dsp.ladder(dsp.saw(up(fh), SR), np.full(N, 180.0), SR, res=0.3, drive=1.4) * bass_amp

    # --- mix (MOIRE_SOLO=name,name keeps only those groups, for diagnosis: lead bounce ping grid bed pad fx)
    import os
    solo = [g for g in os.environ.get('MOIRE_SOLO', '').split(',') if g]
    if solo:
        keep = lambda g: 1.0 if g in solo else 0.0
        eL, eR, xL, xR = eL * keep('lead'), eR * keep('lead'), xL * keep('lead'), xR * keep('lead')
        bL, bR, pL, pR = bL * keep('bounce'), bR * keep('bounce'), pL * keep('bounce'), pR * keep('bounce')
        gL, gR = gL * keep('ping'), gR * keep('ping')
        kick, boom, rum, tL, tR, hL, hR, clap = (v * keep('grid') for v in (kick, boom, rum, tL, tR, hL, hR, clap))
        bedL, bedR, bv = bedL * keep('bed'), bedR * keep('bed'), bv * keep('bed')
        padL, padR = padL * keep('pad'), padR * keep('pad')
        riser, crash = riser * keep('fx'), crash * keep('fx')
        guL, guR, drops, dL, dR = (v * keep('deco') for v in (guL, guR, drops, dL, dR))
    sc = S.sidechain(kicks, N, SR, 0.45, rel=0.16)
    # v8, the finishing pass: the kick's low end lost 6.5 dB of contrast wherever the engine's bass played (a blind review by
    # another lab's model heard it swallowed from 1:08; measured +16.5 dB in the intro, +10 under the bass; Engine +12.4). Duck the low end
    # alone hard (below 120 Hz, -14 dB), and leave the lead's audible duck as it was (-5 dB, as the reference's lead)
    # (first try, ducking alone, moved nothing: by 0.3 s after a kick the duck has recovered, and the lead's sub was what
    # filled the gap, -19.5 dB against the kick's -4 there. So the low end also comes down 3 dB, its duck a little longer)
    scb = S.sidechain(kicks, N, SR, 0.8, rel=0.22)
    sos_lo = signal.butter(4, 120, 'low', fs=SR, output='sos')
    split = lambda e: (lambda lo: 0.7 * lo * scb + (e - lo) * sc)(signal.sosfiltfilt(sos_lo, e))
    eLd, eRd = split(eL), split(eR)
    L = (kick + boom * 0.5 + rum * 0.8 * scb + eLd * G_LEAD + xL * 0.15 * sc + tL * 0.10 + hL * 0.16 + clap * 0.10
         + (bL * 0.5 + pL * 0.45) * G_BOUNCE + gL * G_PING + padL * 0.8 + bedL * G_BED + riser * 0.05 + crash * 0.05 + bv * 0.21 * scb + guL * G_GUIRO + (drops * 0.4 + dL) * G_DROPS)
    Rr = (kick + boom * 0.5 + np.roll(rum, int(0.0015 * SR)) * 0.8 * scb + eRd * G_LEAD + xR * 0.15 * sc + tR * 0.10 + hR * 0.16 + clap * 0.10
          + (bR * 0.5 + pR * 0.45) * G_BOUNCE + gR * G_PING + padR * 0.8 + bedR * G_BED + riser * 0.05 + crash * 0.05 + bv * 0.21 * scb + guR * G_GUIRO + (drops * 0.4 + dR) * G_DROPS)
    rl, rr = dsp.reverb(xL * 0.3 + pL * 0.25 + gL * 0.5 + clap * 0.15 + bedL * G_BED * 0.4, xR * 0.3 + pR * 0.25 + gR * 0.5 + clap * 0.15 + bedR * G_BED * 0.4, SR, rt60=3.0, damp_hz=4000)
    mix = np.stack([L + rl * 0.35, Rr + rr * 0.35], 1)
    mix = np.stack([dsp.butter(mix[:, 0], 15, SR, 'high', 2), dsp.butter(mix[:, 1], 15, SR, 'high', 2)], 1)     # v7: was mix -= mean, which left the silent tail at a -41 dB offset
    # v8: mastering EQ to the references' tilt (3-8 kHz sat 6-8 dB darker than two references and Engine; on a phone
    # speaker drop 1 kept -15.6 dB of its power where they keep -11 to -14): +5 dB above 4 kHz, -2 dB below 50 Hz
    shelf_hi, shelf_lo = signal.butter(2, 4000, 'high', fs=SR, output='sos'), signal.butter(2, 50, 'low', fs=SR, output='sos')
    for c in range(2):
        mix[:, c] = mix[:, c] + (10 ** (5 / 20) - 1) * signal.sosfiltfilt(shelf_hi, mix[:, c]) - (1 - 10 ** (-2 / 20)) * signal.sosfiltfilt(shelf_lo, mix[:, c])
    for _ in range(0 if solo else 6):
        g = 10 ** ((-10.0 - dsp.lufs(mix, SR)) / 20)
        mix = dsp.limiter(mix * g, SR, ceiling_db=-1.2, rel_ms=60)
    stems = dict(kick=kick, leadL=eLd * G_LEAD, leadR=eRd * G_LEAD, bounceL=bL * G_BOUNCE, bounceR=bR * G_BOUNCE,
                 pingL=gL * G_PING, pingR=gR * G_PING, bedL=bedL * G_BED, bedR=bedR * G_BED, pad=pad,
                 decoL=guL * G_GUIRO + (drops * 0.4 + dL) * G_DROPS, decoR=guR * G_GUIRO + (drops * 0.4 + dR) * G_DROPS)
    score = dict(bpm=BPM, events=ev, kicks=kicks, acc7=acc7, acc5=acc5, align7=ALIGN7, align5=ALIGN5, grand=GRAND, takes=TAKES,
                 take=TAKES[0], arrive=GRAND[0])
    return mix, stems, score


def main():
    mix, stems, score = render()
    import os, sys
    if os.environ.get('MOIRE_AUDITION'):
        a, b = (int(v) for v in os.environ['MOIRE_AUDITION'].split('-'))
        wavfile.write(f"out/moire_bed_{BED_KIND}.wav", SR, mix[int(tb(a) * SR):int(tb(b) * SR)].astype(np.float32))
        return
    if os.environ.get('MOIRE_SOLO'):
        wavfile.write(f"out/moire_solo_{os.environ['MOIRE_SOLO'].replace(',', '+')}.wav", SR, mix.astype(np.float32))
        return
    wavfile.write('out/moire.wav', SR, mix.astype(np.float32))
    np.savez_compressed('out/moire_stems.npz', sr=SR, **stems)
    json.dump(score, open('out/moire_score.json', 'w'))
    print('wrote out/moire.wav', f'{len(mix) / SR / 60:.2f} min, {dsp.lufs(mix, SR):.1f} LUFS, {dsp.true_peak_db(mix):.1f} dBTP')


if __name__ == '__main__':
    main()
