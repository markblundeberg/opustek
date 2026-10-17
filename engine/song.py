"""Engine, Learning to Sing (Opustek, 2026): the whole track, from the score to
the master, in one file. Needs dsp.py (and dsp.c, which it compiles).

A drone engine (detuned saws through a ladder filter) learns to sing: a second
saw hard-synced to it puts a resonant peak on one of its harmonics, and the
ratio walks, like an overtone singer's throat. Around it: plucked "doots" with
tape echoes, a kick and clap, hats made of grain bundles, metal plucks
(Karplus-Strong), and a chopper that gates the engine.

F# Dorian, 123 bpm, 96 bars (3:14 with the tail).

  python3 song.py                 -> out/song.wav, out/song_stems.npz, out/song_score.json
  python3 song.py --break NAME    a deliberately broken render, for check_song.py's self-tests
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the shared kernel, ../dsp.py
import dsp

SR, OS, CR = 48000, 2, 1000            # audio rate, oversampling for the engine, control rate
BPM = 123.0
BEAT = 60 / BPM
BAR = 4 * BEAT
NBARS, TAIL = 96, 7.0
HZ = lambda m: 440.0 * 2 ** ((m - 69) / 12)
tb = lambda bar, beat=0.0: (bar - 1) * BAR + beat * BEAT
R = lambda a, b: list(range(a, b + 1))


# --- the score -------------------------------------------------------------------
# intro 1-8 | groove 9-16 (the engine idling) | the engine wakes 17-32 | its solo
# 33-48 | the seam 49-50 (only the metal plucks) | the doots return 51-52 | the
# engine wakes again 53-55 | a bar's pause 56 | climax 57-80 (an octave climb at
# 73) | outro 81-96, and the engine switches off.

KICK = R(9, 48) + R(57, 88)
CLAP = R(9, 15) + R(17, 31) + R(33, 47) + R(57, 79) + R(81, 84)
HATS = R(5, 32) + R(41, 48) + R(57, 88)
HAT16 = R(25, 32) + R(65, 80)                  # ghost sixteenths
DOOTS = R(1, 32) + R(51, 55) + R(57, 92)
MECH_BARS = R(13, 16) + R(33, 48) + R(65, 80)
SECTION_ENDS = (8, 16, 32, 48, 80)

# the engine's bass line: (bar, beat, MIDI note); it glides to each note
BASS = [(9, 0, 30), (17, 0, 30), (21, 0, 33), (23, 2, 35), (25, 0, 30), (29, 0, 37), (31, 2, 33), (33, 0, 30),
        (35, 3, 37), (37, 0, 40), (39, 0, 35), (41, 0, 30), (43, 2, 33), (45, 0, 37), (47, 3, 30), (50, 0, 30),
        (53, 0, 33), (55, 0, 30), (57, 0, 30), (59, 2, 35), (61, 0, 37), (63, 0, 39), (65, 0, 40), (67, 2, 42),
        (69, 0, 30), (71, 0, 33), (73, 0, 47), (75, 0, 37), (76, 2, 39), (77, 0, 40), (79, 0, 42), (80, 3, 30),
        (81, 0, 30), (89, 0, 33), (91, 0, 30)]
GLIDE = 0.28                                   # seconds

# decorations on the engine: (bar, beat, kind, attack, hold, decay, amount)
DECOR = [(20, 0, 'drive', 0.3, 0.8, 1.0, 0.8), (24, 2, 'growl', 0.3, 1.0, 1.0, 0.7), (28, 1, 'squelch', 0.2, 0.5, 0.9, 0.8),
         (31, 0, 'fold', 0.35, 0.8, 1.2, 0.7), (34, 0, 'growl', 0.25, 1.4, 1.0, 1.0), (36, 2, 'drive', 0.15, 0.5, 0.7, 1.0),
         (38, 1, 'fold', 0.3, 1.0, 1.0, 0.9), (42, 0, 'growl', 0.4, 1.8, 1.2, 0.9), (44, 2, 'drive', 0.2, 0.6, 0.9, 0.9),
         (46, 0, 'fold', 0.25, 1.2, 1.0, 1.0), (47, 2, 'squelch', 0.2, 0.4, 1.0, 0.9), (60, 2, 'growl', 0.3, 1.0, 0.9, 0.8),
         (63, 0, 'drive', 0.2, 0.7, 1.0, 1.0), (66, 1, 'squelch', 0.15, 0.4, 0.8, 1.0), (69, 0, 'fold', 0.3, 1.0, 1.2, 0.9),
         (71, 2, 'growl', 0.25, 1.6, 1.0, 1.0), (73, 0, 'drive', 0.15, 0.5, 0.6, 1.0), (77, 0, 'fold', 0.25, 0.9, 0.8, 1.0),
         (78, 2, 'growl', 0.3, 1.2, 1.2, 0.9), (79, 3, 'drive', 0.2, 0.7, 1.0, 0.9)]
KINDS = ('drive', 'growl', 'squelch', 'fold')

# filter sweeps: (bar, bars up, bars down, octaves)
SWEEPS = [(27, 2, 2, 1.2), (35, 1, 3, 1.5), (39, 2, 2, 1.0), (43, 1, 1, 1.4), (45, 2, 2, 1.6),
          (64, 2, 2, 1.0), (72, 1, 3, 1.5), (76, 2, 2, 1.3)]

# metal plucks: (bar, beat, MIDI note)
KS_NOTES = [(9, 0, 85), (9, 1.5, 81), (9, 3, 83), (11, 0.5, 88), (11, 2, 85), (13, 0, 90), (13, 1, 88), (13, 2.5, 85),
            (13, 3.5, 81), (15, 1, 83), (15, 3, 78), (49, 0, 85), (49, 1.5, 81), (49, 2.5, 83), (49, 3.5, 88),
            (50, 0.5, 85), (50, 2, 90), (50, 3, 88), (51, 0.5, 88), (51, 2, 85), (53, 0, 90), (53, 1, 88),
            (53, 2.5, 85), (53, 3.5, 81), (55, 1, 83), (55, 3, 78), (83, 0.5, 88), (83, 2, 85), (85, 0, 90),
            (85, 1, 88), (85, 2.5, 85), (85, 3.5, 81), (87, 1, 83), (87, 3, 78), (89, 0, 85), (89, 1.5, 81),
            (89, 3, 83), (91, 0.5, 88), (91, 2, 85)]

# the chopper, each one different: (bar, beats, cells per beat, duty at start,
# duty at end, pattern of open cells or None for all)
CHOPS = [(40, 4, 4, 0.55, 0.55, [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0]),   # 16ths, 3+3+2
         (44, 2, 8, 0.5, 0.5, None),            # 32nds, square
         (55, 4, 8, 0.8, 0.15, None),           # 32nds closing down: the engine choked into the pause
         (72, 4, 2, 0.3, 0.95, None),           # eighths opening up into the octave climb
         (79, 2, 3, 0.6, 0.35, None)]           # triplets, thinning into the outro

# the doots: two figures of 16 sixteenths, F# C# A arpeggios in 3+3+2
DOOT_A = [66, 61, 57, 66, 61, 57, 64, 61, 66, 61, 57, 66, 61, 57, 69, 64]
DOOT_B = [66, 61, 57, 66, 61, 57, 64, 61, 66, 61, 57, 66, 71, 69, 64, 61]
ACCENT = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
TAILS = [[71, 69, 64, 61], [73, 71, 69, 66], [69, 66, 64, 61], [76, 73, 71, 69]]
# after the seam the doots come back a note at a time, filling in bar 53's
# figure half a bar at a time: 1, 1, 2, 3, then all 6
BUILD = {51: {1, 9}, 52: {1, 6, 9, 12, 14}}

GAIN = dict(kick=0.70, engine=0.46, doots=0.135, hats=0.36, clap=0.11)
SYNC_PEAK_MAX = 1800.0                 # Hz: the sung peak stays out of the ear's 2-4 kHz zone
SQ_RES, SQ_LIFT = 0.78, 0.6            # the squelch: ladder resonance, cutoff lift


def figure(bar):
    """The doots' figure for a bar: [(step, MIDI, accent, velocity)]. It
    evolves by n, the bar's place in the arrangement the figure was written
    for (bars past the intro count 16 ahead): the two figures alternate, the
    accents rotate every 4 bars, every 8th bar thins its second half, 65-71
    keep only the accents, and from 73 the tails change and some accents lift
    an octave. Velocities are jittered with a fixed seed per bar."""
    n = bar if bar <= 8 else bar + 16
    pat = DOOT_B if n % 4 == 0 else DOOT_A
    rng = np.random.default_rng(1000 + n)
    rot = (n // 4) % 3
    acc = ACCENT[rot:] + ACCENT[:rot]
    out = []
    for i, m in enumerate(pat):
        if n >= 73:
            if i >= 12:
                m = TAILS[(n // 2) % 4][i - 12]
            if acc[i] and (n // 4) % 2 == 1:
                m += 12
        if n % 8 == 0 and i >= 8 and i % 2 == 1:
            continue
        if 65 <= n <= 71 and not acc[i]:
            continue
        out.append((i, m, acc[i], (0.72 + 0.28 * acc[i]) * rng.uniform(0.85, 1.0)))
    return out


def doot_pattern(bar, brk=None):
    if brk == 'old-build' and bar in (51, 52):           # filter each bar's own figure: steps it doesn't have
        return [x for x in figure(bar) if x[0] in {51: {0}, 52: {0, 6, 11}}[bar]]
    if bar in BUILD and brk != 'wall':
        return [x for x in figure(53) if x[0] in BUILD[bar]]
    return figure(bar)


# --- helpers -----------------------------------------------------------------------

def ramp(tc, a, b, lo=0.0, hi=1.0):
    """Linear from lo at time a to hi at time b, held outside."""
    return lo + (hi - lo) * np.clip((tc - a) / max(b - a, 1e-9), 0, 1)


def mask(tc, bars, ramp_s=0.03):
    """1 inside the given bars, 0 outside, with soft edges."""
    m = np.zeros(len(tc))
    for b in bars:
        m[(tc >= tb(b)) & (tc < tb(b + 1))] = 1
    k = max(1, int(ramp_s * CR))
    return np.convolve(m, np.ones(k) / k, mode='same')


def up(ctrl, tc, n, sr):
    """A control curve at audio rate."""
    return np.interp(np.arange(n) / sr, tc, ctrl)


def place(dst, hit, times, sr, gain=1.0, shift=0.0):
    for t, g in times:
        i = int(round((t + shift) * sr))
        j = min(len(dst), i + len(hit))
        if 0 <= i < len(dst):
            dst[i:j] += gain * g * hit[:j - i]


def sidechain(times, n, sr, depth, rel=0.16):
    """Gain that ducks by `depth` at each kick and recovers."""
    g = np.ones(n)
    t = np.arange(n) / sr
    for tk in times:
        i = int(tk * sr)
        j = min(n, i + int(6 * rel * sr))
        dt = t[i:j] - tk
        g[i:j] = np.minimum(g[i:j], 1 - depth * np.exp(-dt / rel) * np.clip(dt / 0.004, 0, 1))
    return g


def drift(n, rate, cents, seed, tau=2.0):
    """Slow analog wander, in cents."""
    r = np.random.default_rng(seed).standard_normal(n)
    b, a = signal.butter(1, 1 / (2 * np.pi * tau), fs=rate)
    w = signal.filtfilt(b, a, np.cumsum(r) / np.sqrt(rate))
    w -= w.mean()
    return cents * w / (np.abs(w).max() + 1e-9)


def peq(x, sr, f0, gain_db, q):
    """RBJ peaking EQ."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]
    a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x)


def glide_curve(tc, events, glide):
    """f0 following (bar, beat, note) events, with smooth glides in log pitch."""
    y = np.full(len(tc), np.log(HZ(events[0][2])))
    cur = y[0]
    for bar, beat, m in events:
        ts = tb(bar, beat)
        tgt = np.log(HZ(m))
        u = np.clip((tc - ts) / glide, 0, 1)
        e = u * u * (3 - 2 * u)
        sel = tc >= ts
        y[sel] = cur + (tgt - cur) * e[sel]
        cur = tgt
    return np.exp(y)


def analog_env(tc, t0, att, hold, dec):
    """Rises like a charging capacitor, holds, discharges: no hard edges."""
    a = 1 - np.exp(-np.clip(tc - t0, 0, None) / (att / 3))
    a *= tc >= t0
    t1 = t0 + att + hold
    d = np.exp(-np.clip(tc - t1, 0, None) / (dec / 3))
    return a * np.where(tc >= t1, d, 1.0)


def womp_depth(tc):
    """The womp's depth in dB, wandering between 0 and ~8 over phrases: it
    comes and goes. A smooth random curve, so no two phrases match."""
    r = np.random.default_rng(11).standard_normal(int(len(tc) / CR / BAR) + 3)
    knots = np.arange(len(r)) * BAR * 1.5
    d = np.interp(tc, knots, r)
    d = signal.savgol_filter(d, 2001, 3) if len(d) > 2001 else d
    d = (d - d.min()) / (d.max() - d.min() + 1e-9)
    return 8.0 * np.clip(d * 1.4 - 0.25, 0, 1)


def womp_curve(tc, depth_db):
    """The dotted womp: an amplitude swell every dotted eighth."""
    cyc = 0.75 * BEAT
    u = (tc / cyc) % 1
    shape = np.sin(np.pi * np.clip(u / 0.85, 0, 1)) ** 1.3
    floor = 10 ** (-depth_db / 20)
    return floor + (1 - floor) * shape


def tube(x, d):
    """Asymmetric soft clip, like a valve stage: the bias clips one side first,
    adding even harmonics. d in 0..1 turns it from gentle to hot and lopsided."""
    k = 1.3 * (1 + 3.0 * d)
    bias = 0.25 * d
    y = (np.tanh(k * (x + bias)) - np.tanh(k * bias)) / np.tanh(1.3)
    return y / (1 + 1.2 * d)


# --- instruments -------------------------------------------------------------------

def kick_hit(sr, muddy=False):
    """A sine sweep into F#1 with a click. Its sweep through the low mids is
    quick, its saturation light and a dip at 180 Hz keeps it out of the
    120-300 Hz band (muddy=True: the slower, hotter kick that muddied it)."""
    t = np.arange(int(0.4 * sr)) / sr
    if muddy:
        f = HZ(30) + 150 * np.exp(-t / 0.018) + 28 * np.exp(-t / 0.08)
    else:
        f = HZ(30) + 150 * np.exp(-t / 0.011) + 22 * np.exp(-t / 0.06)
    env = np.exp(-np.clip(t - 0.03, 0, None) / 0.09) * np.clip(t / 0.0006, 0, 1)
    x = np.sin(2 * np.pi * np.cumsum(f) / sr) * env
    click = dsp.butter(np.random.default_rng(1).standard_normal(len(t)), (2000, 7000), sr, 'band') * np.exp(-t / 0.0025)
    if muddy:
        return np.tanh(1.2 * (x + 0.15 * click)) / np.tanh(1.2)
    y = np.tanh(0.8 * (x + 0.18 * click)) / np.tanh(0.8)
    return peq(y, sr, 180, -4.0, 0.8)


def clap_hit(sr):
    """Three quick bursts of band-passed noise and a 50 ms tail."""
    t = np.arange(int(0.2 * sr)) / sr
    nz = np.random.default_rng(4).standard_normal(len(t))
    env = np.zeros(len(t))
    for k, o in enumerate((0.0, 0.009, 0.018)):
        d = np.clip(t - o, 0, None)
        env += (t >= o) * np.exp(-d / (0.005 if k < 2 else 0.05))
    x = dsp.butter(nz, (900, 2800), sr, 'band') * env
    return x / np.abs(x).max()


def doot(sr, hz, open_, accent):
    """A round pluck: saw and square through a ladder whose cutoff opens with
    `open_` (0 muffled .. 1 clear)."""
    t = np.arange(int(0.5 * sr)) / sr
    x = 0.7 * dsp.saw(np.full(len(t), hz), sr) + 0.3 * np.sign(np.sin(np.pi * hz * t))
    cut = 260 + open_ * 1900 + (500 + 1800 * open_) * np.exp(-t / 0.04) * (0.6 + 0.4 * accent)
    y = dsp.ladder(x, cut, sr, res=0.22 + 0.15 * open_, drive=1.3)
    return y * np.exp(-np.clip(t - 0.005, 0, None) / 0.12) * np.clip(t / 0.002, 0, 1)


def spaghetti(sr, rng):
    """One hat hit as a bundle: 4-7 grains a few ms apart, then a noisy tail."""
    t = np.arange(int(0.35 * sr)) / sr
    x = np.zeros(len(t))
    g = rng.integers(4, 8)
    offs = np.cumsum(rng.uniform(0.0015, 0.006, g)) - 0.0015
    for o in offs:
        i = int(o * sr)
        burst = rng.standard_normal(int(0.0012 * sr)) * rng.uniform(0.5, 1.0)
        x[i:i + len(burst)] += burst
    x += rng.standard_normal(len(t)) * 0.25 * np.exp(-np.clip(t - offs[-1], 0, None) / 0.07) * (t > offs[-1])
    return dsp.butter(dsp.butter(x, (5500, 15000), sr, 'band', 2), 4500, sr, 'high', 2)


def ks_note(sr, hz):
    """A struck metal bar: a band-passed click into a fractional-delay
    Karplus-Strong loop with little averaging (bright, long ring)."""
    n = int(2.5 * sr)
    t = np.arange(n) / sr
    click = dsp.butter(np.random.default_rng(int(hz)).standard_normal(n), (1500, 6000), sr, 'band') * np.exp(-t / 0.002)
    y = dsp.ks(click, sr, hz, fb=0.9975, bright=0.35)
    return y * np.exp(-t / 1.2)


def mech(sr, kind, rng):
    """Machine noises: a pneumatic valve's sigh, or a gear's click."""
    t = np.arange(int(0.35 * sr)) / sr
    if kind == 'valve':
        nz = rng.standard_normal(len(t))
        fc = 1500 + 4500 * np.exp(-t / 0.09)
        y = dsp.svf_bp(nz, fc, sr, 2.5)
        env = (1 - np.exp(-t / 0.025)) * np.exp(-t / 0.12)
        return y * env
    return (np.sin(2 * np.pi * 1840 * t) + 0.7 * np.sin(2 * np.pi * 2910 * t)) * np.exp(-t / 0.008)


# --- the engine ----------------------------------------------------------------------

def engine(tc):
    """The engine's control curves at CR: f0, filter cutoff, level, the
    decorations, the sync ratio (which harmonic it sings), the sync voice's
    blend, and the distortion send."""
    f_line = glide_curve(tc, BASS, GLIDE)
    f0 = f_line.copy()
    idle = mask(tc, R(9, 16), 1.0) * 0.6
    awake = mask(tc, R(17, 48) + R(57, 80), 0.5) * ramp(tc, tb(17), tb(19))
    wake2 = mask(tc, R(53, 55), 0.5) * ramp(tc, tb(53), tb(55), 0.0, 0.9)
    outro = mask(tc, R(81, 96), 0.5) * ramp(tc, tb(81), tb(93), 1.0, 0.75)
    lvl = np.maximum.reduce([idle, awake, wake2, outro])
    base = (220 + 180 * mask(tc, R(9, 16), 1) + 300 * mask(tc, R(17, 32), 2) + 800 * mask(tc, R(33, 48), 2) + 300 * mask(tc, R(53, 55), 2)
            * ramp(tc, tb(53), tb(55)) + 2600 * mask(tc, R(57, 80), 1) * ramp(tc, tb(57), tb(73), 0.4, 1.0)
            + 400 * mask(tc, R(81, 96), 2) * ramp(tc, tb(81), tb(93), 1, 0))
    base *= 1 + 0.2 * np.sin(2 * np.pi * tc / (16 * BAR) + 1.3)
    sw = np.zeros(len(tc))
    for bar, up_, dn, octv in SWEEPS:
        a, p, z = tb(bar), tb(bar + up_), tb(bar + up_ + dn)
        sw += octv * np.where(tc < p, ramp(tc, a, p), 1 - ramp(tc, p, z)) * (tc >= a) * (tc < z)
    depth = womp_depth(tc) * mask(tc, R(17, 80), 4)
    w = womp_curve(tc, depth)
    cut = base * 2 ** sw * (0.6 + 0.4 * w) * (np.minimum(f_line, HZ(42)) / HZ(30)) ** 0.8
    tend = tb(NBARS - 3)                        # the switch-off: pitch sinks to 24 Hz, level fades
    off = tc >= tend
    dk = np.exp(-np.clip(tc - tend, 0, None) / 1.6)
    f0 = np.where(off, f0 * dk + 24 * (1 - dk), f0)
    lvl = lvl * np.where(off, np.clip(1 - (tc - tend) / 5.0, 0, 1), 1)
    dc = {k: np.zeros(len(tc)) for k in KINDS}
    for bar, beat, kind, att, hold, dec, amt in DECOR:
        dc[kind] = np.maximum(dc[kind], amt * analog_env(tc, tb(bar, beat), att, hold, dec))
    dc['drive'] = np.maximum(dc['drive'], 0.25 * mask(tc, R(61, 80), 4))
    # the sync ratio: it learns (rises) over the wake and the climax, leaps when
    # the bass revs up, breathes over 8 bars, and the sung peak is capped
    learn = ramp(tc, tb(17), tb(33), 1.0, 5.0) + ramp(tc, tb(57), tb(69), 0.0, 3.0)
    rise = np.maximum(np.gradient(np.log(f_line)) * CR, 0)
    k = np.exp(-np.arange(int(3 * CR)) / (1.6 * CR))
    rev = np.convolve(rise, k)[:len(tc)] / CR
    rev = 9.0 * rev / max(rev.max(), 1e-9)
    breath = 3.0 * (0.5 - 0.5 * np.cos(2 * np.pi * tc / (8 * BAR)))
    ratio = np.clip(learn + rev + breath * (tc >= tb(17)), 1.0, 22.0)
    ratio = np.minimum(ratio, np.maximum(1.0, SYNC_PEAK_MAX / f0))
    blend = (0.12 * mask(tc, R(17, 32), 2) + 0.32 * mask(tc, R(33, 48), 1) + 0.22 * mask(tc, R(53, 55), 1)
             + 0.42 * mask(tc, R(57, 80), 1) + 0.2 * mask(tc, R(81, 88), 2))
    dist = 0.08 * mask(tc, R(17, 32), 2) + 0.2 * mask(tc, R(33, 48), 1) + 0.2 * mask(tc, R(61, 80), 2)
    return f0, cut, lvl * w, dc, ratio, blend, dist


def cloog(tc, f0, cut, amp, dc, ratio, blend, dist, grit_band=(300, 2500)):
    """The engine's voice, oversampled 2x: three detuned saws and the hard-synced
    saw through a ladder (with the squelch's resonant second ladder), the
    growl's amplitude modulation, the tube, the fold, and a parallel grit send
    band-passed to grit_band. Below 150 Hz it is mono, with a sub on the bass
    line; above, a phaser on each side."""
    nos, sro = int(len(tc) / CR * SR) * OS, SR * OS
    n = nos // OS
    U = lambda c: up(c, tc, nos, sro)
    vs = []
    for v, dct in enumerate([-7.0, 0.0, 6.0]):
        cents = dct + drift(len(tc), CR, 3.5, seed=10 + v)
        vs.append(dsp.saw(U(f0 * 2 ** (cents / 1200)), sro, phase=0.31 * v))
    sync = dsp.syncsaw(U(f0 * 2 ** (drift(len(tc), CR, 2.5, seed=31) / 1200)), U(ratio), sro, 0.13)
    bl = U(blend)
    sq = dc['squelch']
    cut_os = U(cut * (1 + SQ_LIFT * sq))
    sqo = U(sq)
    drive, fold, growl = U(dc['drive']), U(dc['fold']), U(0.10 + 0.75 * dc['growl'])
    fold_gain = 1.5 + 4.5 * fold
    ph = 2 * np.pi * np.cumsum(U(f0) / 2) / sro
    ds = U(dist)
    out = []
    for w in ((0.55, 0.3, 0.15), (0.15, 0.3, 0.55)):
        x = sum(a * v for a, v in zip(w, vs)) * (1 - 0.5 * bl) + sync * bl * 1.1
        b = dsp.ladder(x, cut_os, sro, res=0.42, drive=1.6)
        if sq.max() > 0:
            b = b * (1 - sqo) + dsp.ladder(x, cut_os, sro, res=SQ_RES, drive=1.6) * sqo
        b = b * (1 + growl * np.sin(ph)) / (1 + 0.5 * growl)
        y = tube(b, drive)
        y = y * (1 - fold) + np.sin(fold_gain * b) * fold
        grit = np.tanh(6.0 * dsp.butter(y, grit_band, sro, 'band', 2)) * 0.5
        y = y + ds * grit
        out.append(signal.resample_poly(y, 1, OS)[:n])
    L, Rr = out
    low = dsp.butter((L + Rr) / 2, 150, SR, 'low', 4)
    f_line = glide_curve(tc, BASS, GLIDE)
    sub = np.sin(2 * np.pi * np.cumsum(up(np.minimum(f0, f_line) / 2, tc, n, SR)) / SR)
    low = low + 0.35 * dsp.butter(sub, 120, SR, 'low', 2)
    hiL = dsp.phaser(dsp.butter(L, 150, SR, 'high', 4), SR, rate=0.045, fmin=250, fmax=2600, stages=6, fb=0.5, phase=0.0)
    hiR = dsp.phaser(dsp.butter(Rr, 150, SR, 'high', 4), SR, rate=0.045, fmin=250, fmax=2600, stages=6, fb=0.5, phase=1.9)
    a = up(amp, tc, n, SR)
    return (low + hiL) * a, (low + hiR) * a


def leveler(L, Rr, sr, ratio=3.0, att=0.03, rel=0.35):
    """Keeps the engine from jumping out: an RMS compressor on the engine
    alone, its threshold 3 dB over the engine's typical awake level."""
    p = (L ** 2 + Rr ** 2) / 2
    b, a = signal.butter(1, 1 / (2 * np.pi * 0.15), fs=sr)
    env = np.sqrt(np.maximum(signal.lfilter(b, a, p), 1e-20))
    awake = env[env > 0.05 * env.max()]
    thr = np.median(awake) * 10 ** (3 / 20)
    g = np.minimum(1.0, (thr / env) ** (1 - 1 / ratio))
    gd = 20 * np.log10(g)
    s = 0.0
    gd_ds = gd[::48]
    sm = np.empty_like(gd_ds)
    ka, kr = 1 - np.exp(-48 / (att * sr)), 1 - np.exp(-48 / (rel * sr))
    for i, v in enumerate(gd_ds):
        s += (ka if v < s else kr) * (v - s)
        sm[i] = s
    gs = 10 ** (np.interp(np.arange(len(gd)), np.arange(len(sm)) * 48, sm) / 20)
    return L * gs, Rr * gs


def deharsh(L, Rr, sr, thr=-24.0, span=6.0):
    """A dynamic dip in the upper mids, wherever the sung peak runs hot: the
    2.5-4 kHz band's level (30 ms RMS) is watched, and from `thr` dB up to
    thr + span the engine crossfades into a copy dipped 10 dB at 3.2 kHz
    (attack 10 ms, release 200 ms). Below the threshold, nothing changes."""
    det = dsp.butter((L + Rr) / 2, (2500, 4000), sr, 'band', 2)
    b, a = signal.butter(1, 1 / (2 * np.pi * 0.03), fs=sr)
    env = 10 * np.log10(np.maximum(signal.lfilter(b, a, det ** 2), 1e-20))
    k = np.clip((env - thr) / span, 0, 1)[::48]
    s, ka, kr = 0.0, 1 - np.exp(-1 / (0.010 * sr / 48)), 1 - np.exp(-1 / (0.200 * sr / 48))
    sm = np.empty_like(k)
    for i, v in enumerate(k):
        s += (ka if v > s else kr) * (v - s)
        sm[i] = s
    k = np.interp(np.arange(len(L)), np.arange(len(sm)) * 48, sm)
    dip = lambda z: peq(peq(z, sr, 3200, -10.0, 0.9), sr, 2500, -4.0, 1.2)
    return L * (1 - k) + dip(L) * k, Rr * (1 - k) + dip(Rr) * k


def chop_cells():
    """Every gate cell: (open start, open end, cell end) in seconds."""
    out = []
    for bar, beats, per, d0, d1, pat in CHOPS:
        n = int(beats * per)
        cell = BEAT / per
        for k in range(n):
            if pat is not None and not pat[k % len(pat)]:
                out.append((tb(bar) + k * cell, tb(bar) + k * cell, tb(bar) + (k + 1) * cell))
                continue
            duty = d0 + (d1 - d0) * k / max(n - 1, 1)
            a = tb(bar) + k * cell
            out.append((a, a + duty * cell, a + cell))
    return out


def chopper(n, sr, brk=None):
    """Gain at audio rate: 1, except in the chop cells, where it drops to
    -30 dB for the closed part of each cell, with 2 ms ramps so edges don't click."""
    g = np.ones(n)
    if brk == 'no-chop':
        return g
    r = int(0.002 * sr)
    for a, o, e in chop_cells():
        i, j = int(o * sr), int(e * sr)
        if j - i < 3 * r:
            continue
        g[i:j] = 0.03
        g[i:i + r] = np.linspace(1, 0.03, r)
        g[j - r:j] = np.linspace(0.03, 1, r)
    return g


# --- the mix -------------------------------------------------------------------------

def render(brk=None):
    """The whole track. brk names a deliberate fault for the checks' self-tests:
    no-chop, cold-seam (doots through the seam), muddy, harsh, wall (the doots
    return all at once), old-build (the return asks for steps its bars lack),
    piercing (no dynamic dip in the upper mids)."""
    t0 = time.time()
    dur = NBARS * BAR + TAIL
    N = int(dur * SR)
    tc = np.arange(int(dur * CR)) / CR
    pad = lambda z: np.pad(z, (0, max(0, N - len(z))))[:N]
    rng = np.random.default_rng(5)

    # the engine
    f0, cut, amp, dc, ratio, blend, dist = engine(tc)
    grit = (300, 4000) if brk == 'harsh' else (300, 2500)        # keep the grit out of 2.5-6 kHz
    eL, eR = (pad(z) for z in cloog(tc, f0, cut, amp, dc, ratio, blend, dist, grit))
    eL, eR = leveler(eL, eR, SR)
    if brk != 'muddy':
        eL, eR = peq(eL, SR, 210, -3.5, 0.9), peq(eR, SR, 210, -3.5, 0.9)
    if brk != 'harsh':
        # de-harsh the climax: dips at 3.5 and 2.6 kHz and a gentle top shelf, faded in over the climax only
        cx = up(mask(tc, R(57, 80), 2), tc, N, SR)
        tame = lambda z: peq(peq(peq(z, SR, 3500, -7.0, 0.7), SR, 2600, -3.0, 1.0), SR, 9000, -2.5, 0.5)
        eL = eL * (1 - cx) + tame(eL) * cx
        eR = eR * (1 - cx) + tame(eR) * cx
    if brk != 'piercing':
        eL, eR = deharsh(eL, eR, SR)
    chop = chopper(N, SR, brk)
    eL, eR = eL * chop, eR * chop

    # the doots: muffled at first, clearing over the intro; dark tape echoes turning bright
    cl = np.clip(ramp(tc, tb(2), tb(12)) * 0.85 + ramp(tc, tb(57), tb(61)) * 0.15 - ramp(tc, tb(81), tb(93)) * 0.9, 0, 1)
    doots = np.zeros(N)
    cache = {}
    bars = DOOTS + ([47, 48, 49, 50] if brk == 'cold-seam' else [])
    for bar in bars:
        for i, m, acc, v in doot_pattern(bar, brk):
            te = tb(bar) + i * BEAT / 4
            o = round(float(np.interp(te, tc, cl)), 2)
            key = (m, o, acc)
            if key not in cache:
                cache[key] = doot(SR, HZ(m), o, acc)
            place(doots, cache[key], [(te, v)], SR)
    lift = 1 + 0.45 * mask(tc, R(9, 16), 1.0)
    doots *= up(ramp(tc, 0, tb(2), 0.3, 1.0) * ramp(tc, tb(89), tb(93), 1.0, 0.0) * lift, tc, N, SR) * GAIN['doots']
    if brk != 'muddy':
        doots = peq(doots, SR, 220, -3.0, 1.0)
    bright = up(np.clip(ramp(tc, tb(1), tb(73)) - ramp(tc, tb(81), tb(93)), 0, 1), tc, N, SR)
    ech = {}
    for tag, damp in (('d', 1200), ('b', 4500)):
        ech[tag] = (dsp.tapedelay(doots, SR, 0.5 * BEAT, wow_ms=1.5, wow_hz=0.4, fb=0.5, damp_hz=damp, phase=0.0),
                    dsp.tapedelay(doots, SR, 0.5 * BEAT + 0.011, wow_ms=1.8, wow_hz=0.47, fb=0.5, damp_hz=damp, phase=1.7))
    eL8 = ech['d'][0] * (1 - bright) + ech['b'][0] * bright
    eR8 = ech['d'][1] * (1 - bright) + ech['b'][1] * bright
    wl = dsp.tapedelay(doots, SR, 0.75 * BEAT, wow_ms=3.5, wow_hz=0.55, fb=0.6, damp_hz=2200, phase=0.4)
    wr = dsp.tapedelay(doots, SR, 1.0 * BEAT, wow_ms=4.0, wow_hz=0.71, fb=0.58, damp_hz=2000, phase=2.3)

    # a shimmer under the engine's second wake: the doots an octave and a fifth up, all reverb
    shim = np.zeros(N)
    for bar in R(53, 55):
        for i, m, acc, v in doot_pattern(bar):
            te = tb(bar) + i * BEAT / 4
            for mult, g in ((2.0, 0.6), (1.5, 0.4)):
                place(shim, doot(SR, HZ(m) * mult, 0.6, 1), [(te, v * g)], SR)
    shim = dsp.butter(shim, 2800, SR, 'low', 2) * GAIN['doots'] * 0.8 * up(ramp(tc, tb(53), tb(55)), tc, N, SR)

    # metal plucks, louder at the seam
    ksig = np.zeros(N)
    kc = {}
    for bar, beat, m in KS_NOTES:
        if m not in kc:
            kc[m] = ks_note(SR, HZ(m))
        place(ksig, kc[m], [(tb(bar, beat), rng.uniform(0.7, 1.0))], SR)
    ksig *= 0.05 * up(1 + 0.7 * np.where(tc < tb(51), mask(tc, R(49, 50), 0.3), ramp(tc, tb(51), tb(53), 1.0, 0.0)), tc, N, SR)
    kL, kR = dsp.pingpong(ksig * 0.7, ksig * 0.3, 0.75 * BEAT, SR, fb=0.45, damp_hz=5000)

    # machine noises, three against four
    mv, mg = np.zeros(N), np.zeros(N)
    for b in MECH_BARS:
        for e in range(8):
            g = (b - 1) * 8 + e
            t = tb(b, e / 2)
            if g % 3 == 0 and not (b % 4 == 0 and e >= 6):
                place(mv, mech(SR, 'valve', rng), [(t, rng.uniform(0.7, 1.0))], SR)
            if g % 3 == 1 and b >= 33:
                place(mg, mech(SR, 'gear', rng), [(t, rng.uniform(0.6, 1.0))], SR)
    mv *= 0.045
    mg *= 0.03
    gL, gR = dsp.pingpong(mg, mg * 0.2, 0.25 * BEAT, SR, fb=0.35, damp_hz=7000)

    # drums
    kicks = [tb(b, i) for b in KICK for i in range(4)]
    kick = np.zeros(N)
    place(kick, kick_hit(SR, brk == 'muddy'), [(k, 1.0) for k in kicks], SR, GAIN['kick'])
    claps = [tb(b, i) for b in CLAP for i in (1, 3)]
    clap = np.zeros(N)
    place(clap, clap_hit(SR), [(c, 1.0) for c in claps], SR, GAIN['clap'])
    place(clap, clap_hit(SR), [(tb(b, 3.75), 0.35) for b in CLAP if b % 4 == 2], SR, GAIN['clap'])
    hL, hR = np.zeros(N), np.zeros(N)
    for b in HATS:
        lv = 0.6 if b < 9 else 1.0
        for i in range(8):
            if b in SECTION_ENDS and i >= 6:
                continue
            h = spaghetti(SR, rng)
            if b % 2 == 0 and i == 7:
                h = np.concatenate([h, np.zeros(len(h))])
                h = h * np.exp(-np.arange(len(h)) / SR / 0.25) * 2.2
            g = (0.5, 1.0)[i % 2] * rng.uniform(0.8, 1.0) * lv
            p = 0.5 + rng.uniform(-0.25, 0.25)
            place(hL, h, [(tb(b, i / 2), g * (1 - p))], SR)
            place(hR, h, [(tb(b, i / 2), g * p)], SR)
    for b in HAT16:
        for i in range(16):
            if i % 2 == 1 and rng.uniform() < 0.7:
                h = spaghetti(SR, rng) * 0.3
                p = 0.5 + rng.uniform(-0.4, 0.4)
                place(hL, h, [(tb(b, i / 4) + 0.006, 1 - p)], SR)
                place(hR, h, [(tb(b, i / 4) + 0.006, p)], SR)

    # the mix: the kick ducks the engine, the doots and the echoes
    sc = sidechain(kicks, N, SR, 0.55, rel=0.18)
    eL, eR = eL * sc * GAIN['engine'], eR * sc * GAIN['engine']
    dt = doots * sidechain(kicks, N, SR, 0.25)
    echo = sidechain(kicks, N, SR, 0.35)
    rl, rr = dsp.reverb(wl * 0.8 + doots * 0.2 + kL * 0.5 + eL * 0.05, wr * 0.8 + doots * 0.2 + kR * 0.5 + eR * 0.05,
                        SR, rt60=6.5, damp_hz=3500)
    sl, sr_ = dsp.reverb(shim, shim, SR, rt60=8.0, damp_hz=5000, mod=6.0)
    hg = GAIN['hats'] * up(1 - 0.5 * mask(tc, R(57, 80), 2), tc, N, SR)
    L = (kick + eL + dt * 0.9 + (eL8 * 0.45 + wl * 0.3 + rl * 0.5) * echo + hL * hg + clap
         + ksig * 0.5 + kL + sl * 0.9 + mv * 0.8 + gL)
    Rr = (kick + eR + dt * 1.1 + (eR8 * 0.45 + wr * 0.3 + rr * 0.5) * echo + hR * hg + clap
          + ksig * 0.5 + kR + sr_ * 0.9 + mv * 1.2 + gR)
    cl_, cr_ = dsp.reverb(clap * 0.25 + hL * 0.06 + mv * 0.3, clap * 0.25 + hR * 0.06 + mv * 0.3, SR, rt60=1.5, damp_hz=7000)
    mix = np.stack([L + cl_, Rr + cr_], 1)
    mix -= mix.mean(0)
    # master: -10 LUFS through a true-peak limiter at -1.2 dBTP
    for _ in range(6):
        g = 10 ** ((-10.0 - dsp.lufs(mix, SR)) / 20)
        mix = dsp.limiter(mix * g, SR, ceiling_db=-1.2, rel_ms=60)
    mix *= up(np.clip((dur - tc) / 1.5, 0, 1), tc, N, SR)[:, None]
    print(f'render {time.time() - t0:.1f} s')
    stems = {'kick': kick, 'clap': clap, 'droneL': eL, 'droneR': eR, 'doots': doots, 'ks': ksig, 'chop': chop[::48]}
    return mix, stems, kicks, claps, tc, f0, ratio, blend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--break', dest='brk', choices=['no-chop', 'cold-seam', 'muddy', 'harsh', 'wall', 'old-build', 'piercing'])
    a = ap.parse_args()
    mix, stems, kicks, claps, tc, f0, ratio, blend = render(a.brk)
    name = 'song' + (f'_BROKEN-{a.brk}' if a.brk else '')
    wavfile.write(f'out/{name}.wav', SR, mix.astype(np.float32))
    np.savez_compressed(f'out/{name}_stems.npz', sr=SR, tc=tc, f0=f0, ratio=ratio, blend=blend, **stems)
    with open(f'out/{name}_score.json', 'w') as fh:
        json.dump({'kicks': kicks, 'claps': claps}, fh)
    print('wrote', f'out/{name}.wav')


if __name__ == '__main__':
    main()
