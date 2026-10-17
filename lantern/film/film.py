"""Lantern's film: how it was made, told over the piece itself. Every moving thing is driven by the sound it stands
for (the stems' envelopes and the score's events), and the words follow what the agent that made it actually did.

  0:00  the brief                    the piano's chords glow as its filter opens; shaker grains
  0:11  building                     each instrument drawn as what it's made of, from the bar it first plays
  0:32  checking                     the checks, the one that failed and its fix, the five planted faults
  0:43  sent                         the listener's verdict
  0:53  the title card

  python3 lantern/lantern.py                       (writes out/lantern.wav, stems and score)
  python3 lantern/film/film.py --stills 5 12 20 35 47 57
  python3 lantern/film/film.py --render [--from T0 --to T1] [--out PATH] [--workers 4]
"""
import json
import math
import os
import subprocess
import sys

import cairo
import numpy as np
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'out', 'lantern_film')
W, H, FPS = 1920, 1080, 30
CR = 1000

SC = json.load(open(os.path.join(ROOT, 'out', 'lantern_score.json')))
BAR = SC['bar']
BEAT = BAR / 4
tb = lambda bar, beat=0.0: (bar - 1) * BAR + beat * BEAT
_z = np.load(os.path.join(ROOT, 'out', 'lantern_stems.npz'))
SR = int(_z['sr'])
DUR = len(_z['kick']) / SR


def _env(x, hz=14.0):
    b, a = signal.butter(1, hz, fs=SR)
    e = np.sqrt(np.maximum(signal.lfilter(b, a, x.astype(np.float64) ** 2), 0))[::SR // CR]
    ref = np.percentile(e[e > 1e-6], 97) if (e > 1e-6).any() else 1.0
    return np.clip(e / ref, 0, 1.5).astype(np.float32)


ENV = {k: _env(_z[k]) for k in ('kick', 'snare', 'hats', 'shaker', 'bass', 'epiano', 'kalimba')}
del _z


def env(k, t):
    a = ENV[k]
    return float(a[min(len(a) - 1, max(0, int(t * CR)))])


TUNE = sorted(SC['tune'], key=lambda e: e['t'])
TUNE_T = np.array([e['t'] for e in TUNE])
KICKS = np.array(SC['kicks'])
SNARES = np.array(SC['snares'])
EP = sorted(SC['epiano'], key=lambda e: e['t'])
EP_T = np.array([e['t'] for e in EP])
BASS = sorted(SC['bass'], key=lambda e: e['t'])
BASS_T = np.array([e['t'] for e in BASS])
CHORDS = ['Gmaj9', 'F#m11', 'Em9', 'A11']

INK = (0.93, 0.90, 0.84)
DIM = (0.60, 0.58, 0.54)
COL = {'kalimba': (1.00, 0.76, 0.36), 'epiano': (0.86, 0.56, 0.74), 'bass': (0.45, 0.62, 1.00), 'kick': (0.95, 0.93, 0.90),
       'snare': (0.80, 0.80, 0.78), 'hats': (0.62, 0.84, 0.80), 'shaker': (0.70, 0.78, 0.55), 'ok': (0.55, 0.85, 0.55),
       'bad': (1.00, 0.45, 0.40)}
FIRST = {'epiano': 0.0, 'shaker': tb(3), 'kick': tb(5), 'snare': tb(5), 'hats': tb(5), 'bass': tb(5), 'kalimba': tb(5)}


def ramp(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a)))


def since_last(times, t):
    i = int(np.searchsorted(times, t, side='right')) - 1
    return (t - times[i], i) if i >= 0 else (1e9, -1)


def font(ctx, size, weight=cairo.FONT_WEIGHT_NORMAL, slant=cairo.FONT_SLANT_NORMAL):
    ctx.select_font_face('Lato', slant, weight)
    ctx.set_font_size(size)


def text(ctx, s, x, y, size, rgb=INK, a=1.0, weight=cairo.FONT_WEIGHT_NORMAL, slant=cairo.FONT_SLANT_NORMAL, align='left'):
    if a <= 0.005:
        return 0
    font(ctx, size, weight, slant)
    w = ctx.text_extents(s).x_advance
    x0 = x - w / 2 if align == 'center' else x - w if align == 'right' else x
    ctx.set_source_rgba(*rgb, a)
    ctx.move_to(x0, y)
    ctx.show_text(s)
    return w


def mark(ctx, kind, x, y, size, rgb, a):
    """A drawn check or cross (the text face has neither)."""
    if a <= 0.005:
        return
    ctx.new_path()
    ctx.set_source_rgba(*rgb, a)
    ctx.set_line_width(size * 0.14)
    if kind == 'ok':
        ctx.move_to(x, y - size * 0.35)
        ctx.line_to(x + size * 0.3, y - size * 0.05)
        ctx.line_to(x + size * 0.8, y - size * 0.75)
    else:
        ctx.move_to(x + size * 0.1, y - size * 0.7)
        ctx.line_to(x + size * 0.7, y - size * 0.1)
        ctx.move_to(x + size * 0.7, y - size * 0.7)
        ctx.line_to(x + size * 0.1, y - size * 0.1)
    ctx.stroke()


def shown(t, t0, t1, fade=0.5):
    """Opacity of a caption on screen from t0 to t1, fading in and out."""
    return min(ramp(t, t0, t0 + fade), 1 - ramp(t, t1 - fade, t1))


# ------------------------------------------------------------------ the instrument cards

CARDS = [  # (name, label, recipe, column, row)
    ('epiano', 'electric piano', 'two FM operators, 1 : 1', 0, 0),
    ('kalimba', 'kalimba', 'a tine clamped at one end', 1, 0),
    ('bass', 'bass', 'sine + saw, ladder filter', 2, 0),
    ('kick', 'kick', 'a sine falling 150 → 50 Hz', 0, 1),
    ('snare', 'snare', 'two head tones + noise', 1, 1),
    ('hats', 'hats & shaker', 'noise and metal, swung 58%', 2, 1),
]
CW, CH = 520, 236
CX0, CY0 = (W - 3 * CW - 2 * 40) / 2, 250


def card_box(col, row):
    return CX0 + col * (CW + 40), CY0 + row * (CH + 30)


def draw_epiano(ctx, x, y, t, lv):
    """The FM pair: carrier and modulator circles, and one cycle of sin(p + I sin p) with the index decaying since
    the last chord was struck (as the instrument computes it)."""
    dt, _ = since_last(EP_T, t)
    idx = 1.1 * math.exp(-dt / 0.35) + 0.35 if dt < 1e8 else 0.35
    c = COL['epiano']
    for k, (cx, lab) in enumerate(((x + 60, 'mod'), (x + 150, 'carrier'))):
        ctx.set_source_rgba(*c, 0.25 + 0.6 * lv)
        ctx.set_line_width(2)
        ctx.new_path()
        ctx.arc(cx, y + 92, 26, 0, 2 * math.pi)
        ctx.stroke()
        text(ctx, lab, cx, y + 140, 17, DIM, 0.9, align='center')
    ctx.set_source_rgba(*c, 0.25 + 0.6 * lv)
    ctx.move_to(x + 86, y + 92)
    ctx.line_to(x + 124, y + 92)
    ctx.stroke()
    ctx.new_path()
    ctx.set_line_width(3)
    ctx.set_source_rgba(*c, 0.35 + 0.65 * lv)
    for i in range(241):
        p = 2 * math.pi * i / 240 * 2
        v = math.sin(p + idx * math.sin(p))
        px, py = x + 210 + i * 1.15, y + 92 - 44 * v
        ctx.line_to(px, py) if i else ctx.move_to(px, py)
    ctx.stroke()
    text(ctx, f'index {idx:.2f}', x + 350, y + 160, 17, DIM, 0.9, align='center')


_beta = [1.8751, 4.6941, 7.8548]
_xs = np.linspace(0, 1, 120)


def _mode(b):
    s = (math.cosh(b) + math.cos(b)) / (math.sinh(b) + math.sin(b))
    m = np.cosh(b * _xs) - np.cos(b * _xs) - s * (np.sinh(b * _xs) - np.sin(b * _xs))
    return m / np.abs(m).max()


MODES = [_mode(b) for b in _beta]


def draw_kalimba(ctx, x, y, t, lv):
    """A cantilever: clamped at the left, its first three modes (frequency ratios 1, 6.27, 17.55) excited by each
    note and decaying at the instrument's own rates (the higher, the faster). Shown in slow motion."""
    dt, i = since_last(TUNE_T, t)
    c = COL['kalimba']
    ctx.set_source_rgba(*DIM, 0.8)
    ctx.set_line_width(2)
    for k in range(7):                                   # the clamp
        ctx.move_to(x + 40, y + 50 + k * 12)
        ctx.line_to(x + 28, y + 62 + k * 12)
    ctx.stroke()
    ctx.move_to(x + 40, y + 46)
    ctx.line_to(x + 40, y + 140)
    ctx.stroke()
    amps = [0.0, 0.0, 0.0]
    if dt < 4:
        vel = TUNE[i]['vel']
        amps = [math.exp(-dt / 1.6), 0.6 * vel * math.exp(-dt / 0.18), 0.35 * vel * math.exp(-dt / 0.05)]
    slow = 0.55
    disp = sum(a * MODES[k] * math.cos(2 * math.pi * slow * r * t) for k, (a, r) in enumerate(zip(amps, (1.0, 6.27, 17.55))))
    ctx.set_source_rgba(*c, 0.35 + 0.65 * lv)
    ctx.set_line_width(5)
    for j, xx in enumerate(_xs):
        px, py = x + 40 + xx * 300, y + 93 - 36 * disp[j]
        ctx.line_to(px, py) if j else ctx.move_to(px, py)
    ctx.stroke()
    for k, (r, a) in enumerate(zip(('1', '6.27', '17.55'), amps)):     # the partials, lit as they ring
        bx = x + 380 + k * 42
        hgt = 10 + 90 * min(1, a)
        ctx.set_source_rgba(*c, 0.25 + 0.75 * min(1, a))
        ctx.rectangle(bx, y + 140 - hgt, 18, hgt)
        ctx.fill()
        text(ctx, r, bx + 9, y + 162, 15, DIM, 0.9, align='center')


def draw_bass(ctx, x, y, t, lv):
    """One cycle of sine + saw, rounded off by the ladder filter as the note's envelope closes."""
    dt, i = since_last(BASS_T, t)
    bright = math.exp(-dt / 0.12) if dt < 5 else 0.0
    c = COL['bass']
    n = 12 if bright > 0.05 else 3
    ctx.set_source_rgba(*c, 0.35 + 0.65 * lv)
    ctx.set_line_width(3)
    for j in range(241):
        p = 2 * math.pi * j / 240 * 2
        saw = sum((-1) ** (k + 1) * math.sin(k * p) / k * (1 if k == 1 else bright ** (0.25 * k)) for k in range(1, n + 1)) * 2 / math.pi
        v = 0.55 * math.sin(p) + 0.45 * saw
        px, py = x + 40 + j * 1.8, y + 92 - 50 * v
        ctx.line_to(px, py) if j else ctx.move_to(px, py)
    ctx.stroke()


def draw_kick(ctx, x, y, t, lv):
    """The pitch sweep f = 50 + 100 exp(-t / 28 ms), with a dot running down it on each kick."""
    c = COL['kick']
    ctx.set_source_rgba(*DIM, 0.6)
    ctx.set_line_width(2)
    pts = [(x + 40 + 3.6 * k, y + 150 - (50 + 100 * math.exp(-k * 0.0025 / 0.028)) * 0.85) for k in range(120)]
    for j, (px, py) in enumerate(pts):
        ctx.line_to(px, py) if j else ctx.move_to(px, py)
    ctx.stroke()
    dt, _ = since_last(KICKS, t)
    if dt < 0.3:
        k = min(119, int(dt / 0.0025))
        px, py = pts[k]
        ctx.set_source_rgba(*c, 0.95 * (1 - dt / 0.3) + 0.05)
        ctx.arc(px, py, 9, 0, 2 * math.pi)
        ctx.fill()
    ctx.set_source_rgba(*c, 0.2 + 0.8 * lv)
    ctx.arc(x + 470, y + 92, 14 + 20 * lv, 0, 2 * math.pi)
    ctx.fill()


def draw_snare(ctx, x, y, t, lv):
    """Two head tones and a spray of noise, on each hit."""
    dt, _ = since_last(SNARES, t)
    a = math.exp(-dt / 0.12) if dt < 2 else 0.0
    c = COL['snare']
    for k, fy in enumerate((62, 124)):                    # soft: two bright parallel lines flaring together read as stripes
        ctx.set_source_rgba(*c, 0.22 + 0.4 * a)
        ctx.set_line_width(4)
        ctx.move_to(x + 40, y + fy)
        ctx.line_to(x + 40 + 140 * (0.4 + 0.6 * a), y + fy)
        ctx.stroke()
    rng = np.random.default_rng(int(t * 20))
    for k in range(int(40 * a)):
        ctx.set_source_rgba(*c, 0.6 * a)
        ctx.arc(x + 230 + rng.uniform(0, 250), y + 40 + rng.uniform(0, 110), 2.2, 0, 2 * math.pi)
        ctx.fill()


def draw_hats(ctx, x, y, t, lv):
    """The swung 16ths of the bar, each tick lit as it plays; shaker grains below."""
    c, cs = COL['hats'], COL['shaker']
    bar0 = math.floor(t / BAR) * BAR
    for q in range(16):
        k, f = divmod(q * 0.25, 0.5)
        sw = k * 0.5 + (0.58 * 0.5 if abs(f - 0.25) < 1e-9 else f)
        tq = bar0 + sw * BEAT
        a = math.exp(-(t - tq) / 0.15) if 0 <= t - tq < 1 else 0.0
        px = x + 40 + sw / 4 * 430
        ctx.set_source_rgba(*c, 0.18 + 0.8 * a * (env('hats', t) > 0.02))
        ctx.rectangle(px, y + 50, 6, 50 if q % 4 == 0 else 30)
        ctx.fill()
    s = env('shaker', t)
    rng = np.random.default_rng(int(t * 30))
    for k in range(int(30 * min(1, s))):
        ctx.set_source_rgba(*cs, 0.55)
        ctx.arc(x + 40 + rng.uniform(0, 440), y + 135 + rng.uniform(0, 26), 2.0, 0, 2 * math.pi)
        ctx.fill()


DRAW = {'epiano': draw_epiano, 'kalimba': draw_kalimba, 'bass': draw_bass, 'kick': draw_kick, 'snare': draw_snare, 'hats': draw_hats}


def draw_cards(ctx, t, alpha):
    for name, label, recipe, col, row in CARDS:
        first = min(FIRST[name], FIRST['shaker']) if name == 'hats' else FIRST[name]
        a = alpha * ramp(t, first, first + 0.8)
        if a <= 0.01:
            continue
        x, y = card_box(col, row)
        ctx.push_group()
        ctx.set_source_rgba(1, 1, 1, 0.035)
        ctx.rectangle(x, y, CW, CH)
        ctx.fill()
        lv = min(1.0, env('hats' if name == 'hats' else name, t))
        if name == 'hats':
            lv = max(lv, min(1.0, env('shaker', t)))
        DRAW[name](ctx, x, y + 10, t, lv)
        text(ctx, label, x + 18, y + CH - 40, 24, COL[name], 1.0, cairo.FONT_WEIGHT_BOLD)
        text(ctx, recipe, x + 18, y + CH - 14, 19, DIM, 1.0)
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(a)


# ------------------------------------------------------------------ the score strip

def draw_roll(ctx, t, alpha):
    """The whole tune as data, left to right over the 22 bars, a playhead sweeping it; notes light as they sound."""
    if alpha <= 0.01:
        return
    x0, x1, y0, y1 = 120, W - 120, 860, 1010
    lo, hi = 60, 87
    span = 22 * BAR
    X = lambda tt: x0 + (x1 - x0) * tt / span
    Y = lambda m: y1 - (y1 - y0) * (m - lo) / (hi - lo)
    ctx.set_source_rgba(*DIM, 0.25 * alpha)
    ctx.set_line_width(1)
    for b in range(1, 23):
        ctx.move_to(X(tb(b)), y0 - 8)
        ctx.line_to(X(tb(b)), y1 + 6)
    ctx.stroke()
    for b in range(1, 21):
        text(ctx, CHORDS[(b - 1) % 4], X(tb(b)) + 4, y0 - 14, 14, DIM, 0.55 * alpha)
    text(ctx, 'Dmaj9', X(tb(21)) + 4, y0 - 14, 14, DIM, 0.55 * alpha)
    c = COL['kalimba']
    for e in TUNE:
        age = t - e['t']
        lit = 1.0 if 0 <= age < e['dur'] else (math.exp(-(age - e['dur']) / 0.6) if age >= e['dur'] else 0.0)
        base = 0.22 if age < 0 else 0.42
        ctx.set_source_rgba(*c, alpha * (base + (1 - base) * lit))
        ctx.rectangle(X(e['t']), Y(e['midi']) - 3, max(3, X(e['t'] + e['dur']) - X(e['t']) - 2), 6)
        ctx.fill()
    ctx.set_source_rgba(*INK, 0.5 * alpha)
    ctx.set_line_width(2)
    ctx.move_to(X(t), y0 - 8)
    ctx.line_to(X(t), y1 + 6)
    ctx.stroke()


# ------------------------------------------------------------------ the acts

def draw_brief(ctx, t):
    a1 = shown(t, 0.6, tb(5) - 0.3)
    text(ctx, 'Someone asked a fresh Claude agent:', W / 2, 330, 40, DIM, a1, align='center')
    a2 = shown(t, 1.6, tb(5) - 0.3)
    text(ctx, '“make me some music, about a minute long.”', W / 2, 420, 58, INK, a2, slant=cairo.FONT_SLANT_ITALIC, align='center')
    a3 = shown(t, tb(3), tb(5) - 0.3)
    text(ctx, 'One rule: no borrowed sounds. Build every instrument from scratch.', W / 2, 540, 36, COL['kalimba'], a3, align='center')
    # the piano's chords, glowing with its sound as its filter opens
    if t < tb(5) + 0.5:
        a = 1 - ramp(t, tb(5) - 0.5, tb(5) + 0.5)
        dt, i = since_last(EP_T, t)
        if i >= 0:
            notes = EP[i]['notes']
            lv = min(1.0, env('epiano', t))
            open_ = ramp(t, 0, tb(5))
            for k, m in enumerate(notes):
                cx = W / 2 + (k - 1.5) * 150
                ctx.set_source_rgba(*COL['epiano'], a * (0.15 + 0.6 * lv) * (0.4 + 0.6 * open_))
                ctx.rectangle(cx - 55, 760 - (m - 60) * 10, 110, 14)
                ctx.fill()
            text(ctx, 'the piano, opening', W / 2, 860, 22, DIM, a * 0.8 * ramp(t, 0.8, 2.0), align='center')
        s = env('shaker', t)
        rng = np.random.default_rng(int(t * 30))
        for k in range(int(40 * min(1, s))):
            ctx.set_source_rgba(*COL['shaker'], 0.5 * a)
            ctx.arc(W / 2 - 300 + rng.uniform(0, 600), 900 + rng.uniform(0, 24), 2.4, 0, 2 * math.pi)
            ctx.fill()


def draw_header(ctx, t, label, sub, t0, t1):
    a = shown(t, t0, t1)
    text(ctx, label, 120, 120, 30, COL['kalimba'], a, cairo.FONT_WEIGHT_BOLD)
    text(ctx, sub, 120, 180, 40, INK, a)


CHECKS = [  # (time, text, state)  state: ok / bad / note
    (tb(13, 0.2), 'in tune: 78 kalimba notes against the score, median error 0.0 cents', 'ok'),
    (tb(13, 1.4), 'in key: 291 notes, none outside D major', 'ok'),
    (tb(13, 2.6), 'tune heard over the piano?  −6.3 dB', 'bad'),
    (tb(14, 0.6), '   piano dipped at 600–1300 Hz, kalimba +2 dB  →  +5.4 dB', 'ok'),
]
FAULTS = ['kalimba 30 cents sharp', 'a wrong note', 'kick left in the break', 'tune buried 14 dB', 'a boom after the ending']


def draw_checks(ctx, t):
    a = shown(t, tb(13), tb(17) - 0.2)
    if a <= 0.01:
        return
    y = 300
    for k, (t0, s, st) in enumerate(CHECKS):
        aa = a * ramp(t, t0, t0 + 0.4)
        mark(ctx, st, 200, y + k * 64, 40, COL[st], aa)
        text(ctx, s, 260, y + k * 64, 36, INK if st != 'bad' else COL['bad'], aa)
    a2 = a * ramp(t, tb(15), tb(15) + 0.5)
    text(ctx, 'Then it broke its own song five ways, to prove each check could fail:', 200, 600, 34, DIM, a2)
    for k, f in enumerate(FAULTS):
        tk = tb(15, 1.0 + k * 1.1)
        aa = a * ramp(t, tk, tk + 0.3)
        cx = 200 + k * 310
        text(ctx, f, cx, 670, 26, INK, aa)
        caught = ramp(t, tk + 0.45, tk + 0.6)
        text(ctx, 'caught', cx, 712, 26, COL['ok'], aa * caught, cairo.FONT_WEIGHT_BOLD)
        mark(ctx, 'ok', cx + 90, 712, 26, COL['ok'], aa * caught)


def draw_sent(ctx, t):
    a1 = shown(t, tb(17, 0.2), tb(21) - 0.2)
    text(ctx, 'It sent the song with timestamps and four questions.', W / 2, 330, 38, DIM, a1, align='center')
    a2 = shown(t, tb(18), tb(21) - 0.2)
    text(ctx, '“nice and original, nice lofi BGM', W / 2, 470, 60, INK, a2, slant=cairo.FONT_SLANT_ITALIC, align='center')
    text(ctx, 'that could go on for hours.”', W / 2, 550, 60, INK, a2, slant=cairo.FONT_SLANT_ITALIC, align='center')
    a3 = shown(t, tb(19), tb(21) - 0.2)
    text(ctx, '— Mark, the first person to hear it', W / 2, 640, 34, COL['kalimba'], a3, align='center')


def draw_title(ctx, t):
    a = ramp(t, tb(21) + 0.3, tb(21) + 1.6) * min(1.0, max(0.0, (DUR - 0.3 - t) / 1.5))
    text(ctx, 'Lantern', W / 2, 470, 120, INK, a, align='center')
    a2 = a * ramp(t, tb(21) + 1.2, tb(21) + 2.2)
    text(ctx, 'made by a fresh Claude agent, on its first try, every sound built from scratch', W / 2, 570, 34, DIM, a2, align='center')
    a3 = a * ramp(t, tb(21) + 2.2, tb(21) + 3.2)
    text(ctx, 'the toolkit: github.com/markblundeberg/opustek', W / 2, 650, 34, COL['kalimba'], a3, align='center')


def frame(t):
    s = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    ctx = cairo.Context(s)
    ctx.set_source_rgb(0.043, 0.039, 0.047)
    ctx.paint()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    draw_brief(ctx, t)
    draw_header(ctx, t, 'BUILDING', 'It can’t hear. It builds each sound from its physics, then measures.', tb(5), tb(13) - 0.2)
    draw_header(ctx, t, 'CHECKING', 'Every intention gets a measurement.', tb(13), tb(17) - 0.2)
    draw_header(ctx, t, 'SENT', 'Then a person listens.', tb(17), tb(21) - 0.2)
    cards_a = shown(t, tb(5), tb(13) - 0.1, fade=0.6)
    draw_cards(ctx, t, cards_a)
    draw_checks(ctx, t)
    draw_sent(ctx, t)
    draw_roll(ctx, t, shown(t, tb(5) - 0.5, tb(21) + 0.5, fade=0.8))
    draw_title(ctx, t)
    s.flush()
    buf = np.frombuffer(s.get_data(), np.uint8).reshape(H, s.get_stride() // 4, 4)[:, :W]
    fade = min(1.0, t / 1.0, max(0.0, (DUR - t) / 0.4))
    return (buf[..., [2, 1, 0]].astype(np.float32) * fade).astype(np.uint8)


def _frame_bytes(k):
    return np.ascontiguousarray(frame(k / FPS)).tobytes()


def render(t0=0.0, t1=None, path=None, workers=4):
    from multiprocessing import Pool
    t1 = DUR if t1 is None else t1
    path = path or os.path.join(OUT, 'lantern_film.mp4')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    k0, k1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    wav = os.path.join(ROOT, 'out', 'lantern.wav')
    ff = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                           '-i', '-', '-ss', f'{t0:.3f}', '-t', f'{t1 - t0:.3f}', '-i', wav,
                           '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
                           '-c:a', 'aac', '-b:a', '256k', '-shortest', path], stdin=subprocess.PIPE)
    with Pool(workers) as p:
        for b in p.imap(_frame_bytes, range(k0, k1), chunksize=8):
            ff.stdin.write(b)
    ff.stdin.close()
    return ff.wait()


if __name__ == '__main__':
    a = sys.argv
    if '--stills' in a:
        from PIL import Image
        os.makedirs(OUT, exist_ok=True)
        for x in a[a.index('--stills') + 1:]:
            Image.fromarray(frame(float(x))).save(os.path.join(OUT, f'still_{float(x):05.1f}.png'))
    elif '--render' in a:
        t0 = float(a[a.index('--from') + 1]) if '--from' in a else 0.0
        t1 = float(a[a.index('--to') + 1]) if '--to' in a else None
        out = a[a.index('--out') + 1] if '--out' in a else None
        sys.exit(render(t0, t1, out, int(a[a.index('--workers') + 1]) if '--workers' in a else 4))
