"""Lantern: a one-minute downtempo piece. A kalimba sings a short tune over a warm electric piano, a round bass and a
soft swung kit, with a little tape hiss. Every instrument here is built on the kernel (dsp.c / dsp.py) and numpy; none
is borrowed from engine/ or moire/.

  python3 lantern/lantern.py                -> out/lantern.wav, out/lantern_stems.npz, out/lantern_score.json
  python3 lantern/lantern.py --break NAME   the same with one fault (see BREAKS), for check_lantern.py --selftest

Form (90 bpm, D major, IV-iii-ii-V loop: Gmaj9 F#m11 Em9 A11-A9, 22 bars, about a minute):
  1-4   intro: the piano alone through a closed filter that opens; a shaker from bar 3
  5-12  the kit, the bass and the tune (two four-bar phrases, a call and its answer)
  13-14 a break: no kick, no bass; the piano and a quiet echo of the tune
  15-16 the kit comes back in steps, a fill into
  17-20 the tune again, the kalimba playing in sixths, turning toward home
  21-22 home: Dmaj9 with a rising kalimba arpeggio, the kit stops, the tail rings
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]
import dsp  # the kernel only

SR = 48000
BPM = 90.0
BEAT = 60 / BPM
BAR = 4 * BEAT
NBARS, TAIL = 22, 4.0
SWING = 0.58                     # a swung 16th lands at 58% of its 8th, not 50%
HZ = lambda m: 440.0 * 2 ** ((m - 69) / 12)
tb = lambda bar, beat=0.0: (bar - 1) * BAR + beat * BEAT
R = lambda a, b: list(range(a, b + 1))
KEY_PCS = {2, 4, 6, 7, 9, 11, 1}  # D major

# --- the score, as data -------------------------------------------------------------------------------------------
# chord loop (one per bar): bass root (MIDI) and the piano's voicing; bar 4 of the loop moves A11 -> A9 on beat 3
LOOP = [(43, [66, 69, 71, 74]),          # Gmaj9:  F#3 A3 B3 D4 over G2
        (42, [64, 69, 71, 73]),          # F#m11:  E3 A3 B3 C#4 over F#2
        (40, [62, 66, 67, 71]),          # Em9:    D3 F#3 G3 B3 over E2
        (45, [67, 71, 74, 76])]          # A11:    G3 B3 D4 E4 over A2
A9 = [67, 71, 73, 76]                    # G3 B3 C#4 E4: the last two beats of the loop lean home
HOME = (38, [66, 69, 73, 76])            # Dmaj9: F#3 A3 C#4 E4 over D2

# the tune: (bar within the phrase 1-4, beat, MIDI, beats)
CALL = [(1, 0, 78, 1.0), (1, 1.5, 76, 0.5), (1, 2, 74, 1.0), (1, 3, 71, 1.0),
        (2, 0.5, 73, 1.5), (2, 2, 69, 1.0), (2, 3, 71, 0.5), (2, 3.5, 73, 0.5),
        (3, 0, 74, 1.5), (3, 1.5, 71, 0.5), (3, 2, 76, 1.0), (3, 3, 78, 1.0),
        (4, 0, 76, 2.5), (4, 3, 73, 0.5), (4, 3.5, 74, 0.5)]
ANSWER = [(1, 0, 78, 1.5), (1, 1.5, 81, 1.0), (1, 2.5, 79, 0.5), (1, 3, 78, 1.0),
          (2, 0, 76, 1.5), (2, 1.5, 73, 1.0), (2, 2.5, 69, 1.5),
          (3, 0, 71, 0.5), (3, 0.5, 74, 0.5), (3, 1, 76, 1.0), (3, 2, 79, 1.0), (3, 3, 78, 0.5), (3, 3.5, 76, 0.5),
          (4, 0, 69, 3.0), (4, 3.5, 73, 0.5)]
ECHO = [(1, 0.5, 78, 1.0), (1, 2, 76, 1.0), (1, 3, 74, 1.0), (2, 0.5, 73, 2.0), (2, 3, 71, 1.0)]   # the break
HOMEWARD = [(1, 0, 78, 1.5), (1, 1.5, 81, 1.0), (1, 2.5, 79, 0.5), (1, 3, 78, 1.0),
            (2, 0, 76, 1.5), (2, 1.5, 73, 1.0), (2, 2.5, 69, 1.5),
            (3, 0, 71, 0.5), (3, 0.5, 74, 0.5), (3, 1, 76, 1.0), (3, 2, 79, 1.0), (3, 3, 78, 0.5), (3, 3.5, 76, 0.5),
            (4, 0, 76, 1.0), (4, 1, 73, 1.0), (4, 2, 71, 0.5), (4, 2.5, 69, 0.5), (4, 3, 73, 0.5), (4, 3.5, 76, 0.5)]
ARPEGGIO = [(1, 0, 74, 4.0), (1, 1, 78, 3.0), (1, 2, 81, 2.0), (1, 3, 85, 1.0), (2, 0, 86, 4.0)]
# a sixth below in A' (diatonic, so it stays in key): the step down to the note a sixth under
SIXTH_BELOW = {81: 73, 79: 71, 78: 69, 76: 67, 74: 66, 73: 64, 71: 62, 69: 61}

TUNE_SECTIONS = [(5, CALL), (9, ANSWER), (13, ECHO), (17, HOMEWARD), (21, ARPEGGIO)]
SIXTHS = R(17, 20)
KICK_BARS = R(5, 12) + R(16, 21)
SNARE_BARS = R(5, 12) + R(15, 20)
HAT_BARS = R(5, 20)
SHAKER_BARS = R(3, 20)
BASS_BARS = R(5, 12) + R(17, 21)
BREAK = (13, 14)
KICK_BEATS = [0, 1.75, 2.5]              # boom ... b-boom
BASS_FIG = [(0, 0, 1.6), (1.75, 0, 0.25), (2.5, 7, 0.9), (3.5, 12, 0.4)]   # (beat, semitones over root, beats)

BREAKS = {
    'sharp-kalimba': 'the kalimba tuned 30 cents sharp',
    'wrong-note': 'a C natural in the tune, out of D major',
    'kick-in-break': 'the kick keeps playing through the break',
    'buried-tune': 'the kalimba 14 dB down, under the piano',
    'boom-at-end': 'a stray kick in the last bar, out of the fade',
}


def tune_events(brk=None):
    ev = []
    for start, phrase in TUNE_SECTIONS:
        for (b, q, m, d) in phrase:
            bar = start + b - 1
            vel = 0.75 if start == 13 else 1.0
            ev.append({'t': tb(bar, q), 'midi': m, 'dur': d * BEAT, 'vel': vel, 'bar': bar})
            if bar in SIXTHS and m in SIXTH_BELOW:
                ev.append({'t': tb(bar, q) + 0.012, 'midi': SIXTH_BELOW[m], 'dur': d * BEAT, 'vel': 0.55, 'bar': bar})
    if brk == 'wrong-note':
        ev.append({'t': tb(10, 3.5), 'midi': 72, 'dur': 0.5 * BEAT, 'vel': 1.0, 'bar': 10})
    return sorted(ev, key=lambda e: e['t'])


def chord_at(bar, beat=0.0):
    if bar >= 21:
        return HOME
    root, v = LOOP[(bar - 1) % 4]
    if (bar - 1) % 4 == 3 and beat >= 2:
        v = A9
    return root, v


# --- helpers ------------------------------------------------------------------------------------------------------

def add(dst, x, t, gain=1.0):
    i = int(round(t * SR))
    if i >= len(dst) or i + len(x) <= 0:
        return
    j = min(len(dst), i + len(x))
    dst[max(i, 0):j] += gain * x[max(0, -i):j - i]


def env_ar(n, att, rel_at, rel, dec):
    """Attack (s), then an exponential decay with time constant dec, released over rel seconds from rel_at."""
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(att, 1e-4)) * np.exp(-t / dec)
    e *= np.clip(1 - (t - rel_at) / rel, 0, 1)
    return e


def duck(times, n, depth, rel):
    """A gain curve that dips by depth at each kick and recovers with time constant rel."""
    g = np.ones(n)
    L = int(6 * rel * SR)
    shape = 1 - depth * np.exp(-np.arange(L) / (rel * SR)) * np.minimum(1, np.arange(L) / (0.006 * SR))
    for t in times:
        i = int(t * SR)
        j = min(n, i + L)
        g[i:j] = np.minimum(g[i:j], shape[:j - i])
    return g


def swing(beat):
    """Swing the off 16ths: a 16th at k + 0.25 or k + 0.75 moves later."""
    k, f = divmod(beat, 0.5)
    return k * 0.5 + (SWING * 0.5 if abs(f - 0.25) < 1e-9 else f)


# --- the instruments (all built here) -----------------------------------------------------------------------------

def kalimba(hz, dur, vel, rng):
    """A plucked steel tine: modal synthesis with the cantilever's own partials (1, 6.27, 17.55 times the
    fundamental), a weak octave from the bridge, a thumb thump of band-passed noise and a hollow box resonance.
    The partials decay faster the higher they sit, as a real tine's do."""
    n = int((max(dur, 0.4) + 2.2) * SR)
    t = np.arange(n) / SR
    ring = 1.9 * (440 / hz) ** 0.35
    y = np.zeros(n)
    for ratio, amp, tau in [(1.0, 1.0, ring), (2.0, 0.07, ring * 0.4), (6.27, 0.22 * vel, 0.18), (17.55, 0.06 * vel, 0.05)]:
        f = hz * ratio
        if f > 18000:
            continue
        y += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 0.3)) * np.exp(-t / tau)
    y *= np.minimum(1, t / 0.0015)
    thump = rng.standard_normal(int(0.012 * SR)) * np.exp(-np.arange(int(0.012 * SR)) / (0.003 * SR))
    th = np.zeros(n)
    th[:len(thump)] = thump
    y += 0.25 * vel * dsp.svf_bp(th, hz * 2.2, SR, 1.5)
    y += 0.5 * dsp.svf_bp(y, 330.0, SR, 3.0)                                  # the box
    # a damping finger lands well after the written length: let it ring, but not forever
    stop = max(dur, 0.4) + 1.4
    y *= np.clip(1 - (t - stop) / 0.6, 0, 1)
    return vel * y


def epiano(midi_notes, t0, t1, vel, N, rng, cents=0.0):
    """A tine electric piano: two-operator FM per note (modulator at the carrier's frequency, index decaying from a
    bright bark to a soft core), a quick metallic tine partial on the strike, held to t1 then released."""
    n0, n1 = int(t0 * SR), min(N, int((t1 + 0.6) * SR))
    t = np.arange(n1 - n0) / SR
    out = np.zeros(n1 - n0)
    for m in midi_notes:
        f = HZ(m) * 2 ** ((cents + rng.uniform(-3, 3)) / 1200)
        v = vel * rng.uniform(0.85, 1.0)
        idx = (1.1 * v) * np.exp(-t / 0.35) + 0.35
        amp = np.minimum(1, t / 0.004) * (0.55 * np.exp(-t / 0.9) + 0.45 * np.exp(-t / 4.0))
        amp *= np.clip(1 - (t - (t1 - t0)) / 0.25, 0, 1)
        ph = 2 * np.pi * f * t
        tone = np.sin(ph + idx * np.sin(ph))
        tine = 0.12 * v * np.sin(2 * np.pi * min(f * 14.0, 15000) * t) * np.exp(-t / 0.018)
        out += v * amp * (tone + tine) * (220 / f) ** 0.15
    return n0, out


def kick(rng, vel=1.0):
    """A soft round kick: a sine sweeping 150 -> 50 Hz, a short body, a tiny click, lightly saturated."""
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 50 + 100 * np.exp(-t / 0.028)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * (np.exp(-t / 0.16) * np.minimum(1, t / 0.0015))
    click = dsp.butter(rng.standard_normal(n) * np.exp(-t / 0.0015), 3000, SR, 'low')
    y = np.tanh(1.6 * (body + 0.25 * click)) / np.tanh(1.6)
    return vel * dsp.butter(y, 6000, SR, 'low')


def snare(rng, vel=1.0):
    """A dusty snare: two drum-head tones and band-passed noise with a slightly longer, darker tail."""
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    head = 0.6 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.045) + 0.3 * np.sin(2 * np.pi * 330 * t) * np.exp(-t / 0.03)
    nz = rng.standard_normal(n)
    wires = dsp.svf_bp(nz, 2200, SR, 0.7) * (np.exp(-t / 0.07) * np.minimum(1, t / 0.001))
    crack = dsp.svf_bp(nz, 5000, SR, 1.0) * np.exp(-t / 0.012)
    y = head + 1.1 * wires + 0.5 * crack
    return vel * dsp.butter(y, 9000, SR, 'low')


def hat(rng, open_=False, vel=1.0):
    """A closed (or slightly open) hat: high-passed noise plus a few metallic square-ish partials, short."""
    n = int((0.25 if open_ else 0.08) * SR)
    t = np.arange(n) / SR
    metal = sum(np.sign(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))) for f in (3150, 4430, 5870, 7260)) * 0.15
    y = dsp.butter(rng.standard_normal(n) + metal, 7000, SR, 'high', 4)
    y *= np.exp(-t / (0.09 if open_ else 0.018)) * np.minimum(1, t / 0.0008)
    return vel * y


def shaker(rng, vel=1.0):
    """A shaker grain: band-passed noise with a soft 8 ms attack (beads, not a stick)."""
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    y = dsp.svf_bp(rng.standard_normal(n), 6500, SR, 1.2)
    return vel * y * np.minimum(1, t / 0.008) * np.exp(-np.maximum(t - 0.008, 0) / 0.022)


def bass_line(N, bars):
    """A round bass: a sine for the floor plus a saw through a ladder whose cutoff is struck at each note, with a
    short glide between notes. Returns the line and its note events."""
    f = np.full(N, HZ(43))
    gate = np.zeros(N)
    cut = np.full(N, 180.0)
    ev = []
    t = np.arange(N) / SR
    for b in bars:
        for (q, iv, d) in BASS_FIG if b < 21 else [(0, 0, 6.0)]:
            root, _ = chord_at(b, q)
            m = root + iv
            if iv == 12 and b < 21:
                m = root + (12 if (b % 2) else 10 if root == 45 else 12)   # the octave, a 7th on alternate V bars
            t0 = tb(b, swing(q))
            i, j = int(t0 * SR), min(N, int((t0 + d * BEAT) * SR))
            g = int(0.025 * SR)
            f[i:i + g] = np.linspace(f[i - 1] if i > 0 else HZ(m), HZ(m), g)[:len(f[i:i + g])]
            f[i + g:] = HZ(m)
            u = t[i:j] - t0
            gate[i:j] = np.minimum(1, u / 0.004) * np.clip((d * BEAT - u) / 0.05, 0, 1)
            cut[i:j] = np.maximum(cut[i:j], 180 + 700 * np.exp(-u / 0.12))
            ev.append({'t': t0, 'midi': int(m), 'dur': d * BEAT})
    ph = 2 * np.pi * np.cumsum(f) / SR
    sub = np.sin(ph)
    body = dsp.ladder(dsp.saw(f, SR), cut, SR, res=0.2, drive=1.2)
    y = (0.6 * sub + 0.6 * body) * gate
    return np.tanh(1.3 * y) / np.tanh(1.3), ev


def hiss(N, rng):
    """Tape air and the odd dust crackle, very low."""
    nz = dsp.butter(dsp.butter(rng.standard_normal(N), 9000, SR, 'low'), 1500, SR, 'high')
    c = np.zeros(N)
    pos = rng.integers(0, N, int(N / SR * 7))
    c[pos] = rng.uniform(-1, 1, len(pos)) ** 3
    c = dsp.svf_bp(c, 2500, SR, 0.8)
    return 0.006 * nz + 0.12 * c


def chorus(x, phase):
    return dsp.tapedelay(x, SR, 0.009, wow_ms=1.8, wow_hz=0.55, fb=0.0, damp_hz=12000, phase=phase)


# --- the render ---------------------------------------------------------------------------------------------------

def render(brk=None):
    rng = np.random.default_rng(11)
    dur = NBARS * BAR + TAIL
    N = int(dur * SR)
    t = np.arange(N) / SR

    # drums
    kick_bars = sorted(set(KICK_BARS) | (set(BREAK) if brk == 'kick-in-break' else set()))
    kicks = []
    for b in kick_bars:
        beats = [0] if b == 21 else ([0, 2.5] if b == 16 else KICK_BEATS)
        for q in beats:
            kicks.append(tb(b, swing(q)))
    if brk == 'boom-at-end':
        kicks.append(tb(22, 2))
    K = np.zeros(N)
    for k in kicks:
        add(K, kick(rng, rng.uniform(0.92, 1.0)), k)
    Sn = np.zeros(N)
    snares = []
    for b in SNARE_BARS:
        for q in (1, 3):
            if b == 15 and q == 1:
                continue
            snares.append(tb(b, q))
            add(Sn, snare(rng, rng.uniform(0.9, 1.0)), tb(b, q) + rng.uniform(0.004, 0.012))   # a little late: laid back
        if b in (12, 16, 20):                                   # ghost notes into the next section
            for q, v in ((3.5, 0.3), (3.75, 0.45)):
                add(Sn, snare(rng, v), tb(b, swing(q)))
    hL, hR = np.zeros(N), np.zeros(N)
    for b in HAT_BARS:
        if b in BREAK:
            steps = [0.5, 1.5, 2.5, 3.5]
        else:
            steps = [q * 0.5 for q in range(8)] + ([1.75, 3.75] if b % 2 == 0 else [3.25])
        for q in steps:
            op = (b % 4 == 0 and q == 3.5)
            v = (0.55 if q % 1 == 0.5 else 0.8 if q % 1 == 0 else 0.35) * rng.uniform(0.8, 1.0)
            h = hat(rng, op, v)
            p = 0.62 + rng.uniform(-0.05, 0.05)
            add(hL, h, tb(b, swing(q)), 1 - p)
            add(hR, h, tb(b, swing(q)), p)
    shL, shR = np.zeros(N), np.zeros(N)
    for b in SHAKER_BARS:
        for k in range(16):
            q = k * 0.25
            v = (0.7 if k % 4 == 2 else 0.45 if k % 2 == 0 else 0.3) * rng.uniform(0.8, 1.0) * (0.6 if b < 5 else 1.0)
            s = shaker(rng, v)
            add(shL, s, tb(b, swing(q)), 0.62)
            add(shR, s, tb(b, swing(q)), 0.38)

    # bass
    bass, bass_ev = bass_line(N, BASS_BARS)

    # piano: each bar's chord, struck on beat 1 and again, softer, on the and of 2 (and at the A9 change)
    epL, epR = np.zeros(N), np.zeros(N)
    ep = np.zeros(N)
    ep_ev = []
    for b in R(1, 21):
        root, v = chord_at(b)
        hits = [(0, 2.5, 0.8), (2.5, 4.0, 0.5)]
        if (b - 1) % 4 == 3 and b < 21:
            hits = [(0, 2.0, 0.8), (2.0, 4.0, 0.6)]
        if b == 21:
            hits = [(0, 8.0, 0.75)]
        for (q0, q1, vel) in hits:
            _, vv = chord_at(b, q0)
            t0, t1 = tb(b, swing(q0)), tb(b, q1)
            i, y = epiano(vv, t0, t1, vel, N, rng)
            ep[i:i + len(y)] += y
            ep_ev.append({'t': t0, 'notes': vv, 'dur': t1 - t0})
    # filter: closed in the intro and opening over it, dipped in the break
    fc = np.interp(t, [0, tb(4.5), tb(5), tb(13), tb(13.2), tb(15), tb(16.5), tb(21), tb(21.5), dur],
                   [900, 2400, 1900, 1900, 1300, 1500, 1900, 1900, 3000, 2400])
    # and a zero-phase dip where the tune lives (600-1300 Hz), 6 dB deep while it plays, so the piano sits under it
    band = signal.sosfiltfilt(signal.butter(2, [600, 1300], 'band', fs=SR, output='sos'), ep)
    tune_on = np.zeros(N)
    for b0, b1 in ((5, 13), (17, 21)):
        tune_on[int(tb(b0) * SR):int(tb(b1) * SR)] = 1
    tune_on = np.convolve(tune_on, np.ones(int(0.3 * SR)) / int(0.3 * SR), mode='same')
    ep = ep - 0.5 * band * tune_on
    ep = dsp.ladder(ep, fc, SR, res=0.1, drive=0.8)
    trem = 0.5 + 0.5 * np.sin(2 * np.pi * (BPM / 60 / 2) * t)                # autopan at a half-note rate
    epL = ep * (0.65 + 0.35 * trem) + 0.4 * chorus(ep, 0.0)
    epR = ep * (0.65 + 0.35 * (1 - trem)) + 0.4 * chorus(ep, 2.0)

    # kalimba
    tev = tune_events(brk)
    kal = np.zeros(N)
    cents = 30.0 if brk == 'sharp-kalimba' else 0.0
    for e in tev:
        hz = HZ(e['midi']) * 2 ** (cents / 1200)
        add(kal, kalimba(hz, e['dur'], e['vel'], rng), e['t'] + rng.uniform(0, 0.008))
    kal *= 10 ** (-14 / 20) if brk == 'buried-tune' else 1.0
    kpan = 0.5 + 0.12 * np.sin(2 * np.pi * t / (2 * BAR))
    kL, kR = kal * (1 - kpan) * 1.2, kal * kpan * 1.2
    dL, dR = dsp.pingpong(kL * 0.35, kR * 0.35, 0.75 * BEAT, SR, fb=0.35, damp_hz=3000)

    air = hiss(N, rng)
    air *= np.clip(t / 1.5, 0, 1)

    # mix (gains set by measured level in the main section, not by ear: see ../notes/log.md)
    G = {'kick': 1.0, 'snare': 0.45, 'hats': 0.30, 'shaker': 0.25, 'bass': 0.30, 'epiano': 0.50, 'kalimba': 0.42}
    sc = duck(kicks, N, 0.6, 0.12)
    drumsL = K * G['kick'] + Sn * G['snare'] + hL * G['hats'] + shL * G['shaker']
    drumsR = K * G['kick'] + Sn * G['snare'] + hR * G['hats'] + shR * G['shaker']
    bass_m = bass * G['bass'] * sc
    epL, epR = epL * G['epiano'] * (0.7 + 0.3 * sc), epR * G['epiano'] * (0.7 + 0.3 * sc)
    kL, kR, dL, dR = (v * G['kalimba'] / 0.3 for v in (kL, kR, dL, dR))
    rvL, rvR = dsp.reverb(Sn * 0.04 + epL * 0.10 + kL * 0.30 + dL * 0.3, Sn * 0.04 + epR * 0.10 + kR * 0.30 + dR * 0.3,
                          SR, rt60=2.4, damp_hz=4500, mod=8)
    L = drumsL + bass_m + epL + (kL + dL) * 0.30 + rvL * 0.6 + air
    Rr = drumsR + bass_m + epR + (kR + dR) * 0.30 + rvR * 0.6 + air
    mix = np.stack([L, Rr], 1)
    mix *= 0.6 / np.percentile(np.abs(mix), 99.9)          # stage the level so the saturation below stays gentle
    # master: 20 Hz high-pass, a gentle warmth (a little tape-ish saturation and a soft top), -12 LUFS, -1.2 dBTP
    mix = np.stack([dsp.butter(mix[:, c], 20, SR, 'high', 2) for c in range(2)], 1)
    mix = np.tanh(1.2 * mix) / 1.2
    top = np.stack([dsp.butter(mix[:, c], 11000, SR, 'high', 2) for c in range(2)], 1)
    mix = mix - 0.3 * top
    for _ in range(6):
        mix = dsp.limiter(mix * 10 ** ((-12.0 - dsp.lufs(mix, SR)) / 20), SR, ceiling_db=-1.2, rel_ms=80)
    mix *= np.clip((dur - t) / 1.0, 0, 1)[:, None]

    # stems as they sit in the mix (after gains and ducking, before the master), so levels can be compared
    kal_dry = (kL + kR) / 2 * 0.30
    stems = {'kick': K * G['kick'], 'snare': Sn * G['snare'], 'hats': (hL + hR) / 2 * G['hats'],
             'shaker': (shL + shR) / 2 * G['shaker'], 'bass': bass_m, 'epiano': (epL + epR) / 2, 'kalimba': kal_dry,
             'kalimba_raw': kal, 'air': air}
    score = {'bpm': BPM, 'bar': BAR, 'nbars': NBARS, 'break': BREAK, 'kicks': kicks, 'snares': snares,
             'tune': tev, 'bass': bass_ev, 'epiano': ep_ev, 'home_bar': 21, 'tune_bars': sorted({e['bar'] for e in tev}),
             'key_pcs': sorted(KEY_PCS)}
    return mix, stems, score


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--break', dest='brk', choices=list(BREAKS))
    a = ap.parse_args()
    name = 'lantern' + (f'_BROKEN-{a.brk}' if a.brk else '')
    os.makedirs('out', exist_ok=True)
    mix, stems, score = render(a.brk)
    wavfile.write(f'out/{name}.wav', SR, mix.astype(np.float32))
    np.savez_compressed(f'out/{name}_stems.npz', sr=SR, **stems)
    json.dump(score, open(f'out/{name}_score.json', 'w'))
    print(f'wrote out/{name}.wav: {len(mix) / SR:.1f} s, {dsp.lufs(mix, SR):.1f} LUFS, {dsp.true_peak_db(mix):.1f} dBTP')
