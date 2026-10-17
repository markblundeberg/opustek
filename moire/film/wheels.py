"""Moiré, take 2: three wheels. One sound, one visible thing, at the same moment, in the same place.

Mark on take 1 (fringes of invisible gratings): "just a bunch of shakey lines that bounce with the beat... even
knowing that I'm supposed to look for something, I'm not seeing it." The fringes moved by magnified sub-pixel
shifts, so nothing on screen answered to a sound one for one.

  the bar:      the outer ring, 16 marks, one lit per sixteenth; the kick on the four big ones
  the bouncer:  the middle ring, 7 marks, the next lit by each note (its first note brightest); each first note
                leaves a dot just outside the bar's ring where it fell, kept for seven bars, so the dots walk round
                the bar until one lands at 12 o'clock (an alignment), where a beam rises through the rings
  the pinger:   the inner ring, 5 marks, cyan, the same
  the grand alignments (all three agree, every 35 bars): a white beam and a ring of light running out
  the lead:     its melody drawn round the outside like a clock hand's trail: round is time through the bar, out is
                pitch. In drop 2's solo it plays in 7, which against the bar's 16 draws a gear, and lights the
                bouncer's ring in gold
  the record:   each finished bar's trail, and each machine's first-note dot, drifts outward 14 px a bar
  the bed:      a soft halo between the rings, coloured by its chord, breathing with the kick's pump
  the melody:   in the breakdown, a figure in the middle, a vertex a note (round = its place in a 16-bar turn, out =
                pitch), its glass drop rippling from the vertex; drop 2's 7-figure heads, which carry the same
                melody, relight its first vertices in gold (bars 113-119)

v2 (Mark on v1: the solo "doesn't paint the outer orange line"): the trail through the solo, the record, the ring
of light. v3 (Mark: "the plain honest aesthetic suits the music"): the melody's figure.

  python3 moire/film/wheels.py --stills 50 67.5 68.2 ...      -> out/moire_film/wheel_<t>.png
  python3 moire/film/wheels.py --render --from 50 --to 72 --out out/moire_film/wheels_test.mp4
(system python3: it needs pycairo)
"""
import json
import math
import os
import subprocess
import sys

import cairo
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'out', 'moire_film')
W, H, FPS = 1920, 1080, 30
CX, CY = W / 2, H / 2

EV = json.load(open(os.path.join(OUT, 'events.json')))
_C = np.load(os.path.join(OUT, 'curves.npz'))
CUR = {k: _C[k] for k in _C.files}
CR = 1000
BAR, SIX, DUR = EV['bar'], EV['six'], EV['dur']
GRAND = EV['grand']
NOTES = sorted(EV['lead_notes'])
NT = np.array([n[0] for n in NOTES])

RINGS = {'bar': (330, 372, 16), 'bounce': (240, 292, 7), 'ping': (150, 202, 5)}
COL = {'bar': (0.85, 0.88, 0.95), 'bounce': (1.0, 0.62, 0.22), 'ping': (0.30, 0.85, 1.0), 'lead': (1.0, 0.84, 0.35),
       'bed': (0.85, 0.35, 0.65), 'mel': (0.75, 0.88, 1.0)}
CHORD_COL = [(0.55, 0.30, 0.75), (0.75, 0.30, 0.55), (0.35, 0.45, 0.80), (0.55, 0.35, 0.65),
             (0.55, 0.30, 0.75), (0.75, 0.30, 0.55), (0.80, 0.45, 0.35), (0.35, 0.45, 0.80)]


def cur(name, t, smooth=0.0):
    a = CUR[name]
    if smooth > 0:
        n = int(smooth * CR)
        i0, i1 = max(0, int(t * CR) - n), min(len(a), int(t * CR) + 1)
        return float(a[i0:i1].mean()) if i1 > i0 else 0.0
    return float(a[min(len(a) - 1, max(0, int(t * CR)))])


def tb(bar, beat=0.0):
    return (bar - 1) * BAR + beat * BAR / 4


def ang(frac):
    """Clockwise from 12 o'clock; frac in turns."""
    return -math.pi / 2 + 2 * math.pi * frac


def spoke(ctx, a, r0, r1, width, rgb, alpha):
    ctx.set_source_rgba(*rgb, alpha)
    ctx.set_line_width(width)
    ctx.move_to(CX + r0 * math.cos(a), CY + r0 * math.sin(a))
    ctx.line_to(CX + r1 * math.cos(a), CY + r1 * math.sin(a))
    ctx.stroke()


def glow_spoke(ctx, a, r0, r1, rgb, level):
    if level < 0.01:
        return
    spoke(ctx, a, r0, r1, 14, rgb, 0.10 * level)
    spoke(ctx, a, r0, r1, 6, rgb, 0.25 * level)
    spoke(ctx, a, r0, r1, 2.5, rgb, 0.9 * level)


def seg(ctx, name, i, n, rgb, alpha, glow=False, pad=0):
    """Mark i of n on a ring: a solid arc segment across the ring's band."""
    r0, r1, _ = RINGS[name]
    rm, wd = (r0 + r1) / 2, (r1 - r0) + pad
    gap = 0.16 if n > 8 else 0.10
    a0, a1 = ang((i - 0.5 + gap) / n), ang((i + 0.5 - gap) / n)
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    if glow:
        for extra, al in ((22, 0.10), (10, 0.22)):
            ctx.set_source_rgba(*rgb, al * alpha)
            ctx.set_line_width(wd + extra)
            ctx.arc(CX, CY, rm, a0 - extra / rm / 2, a1 + extra / rm / 2)
            ctx.stroke()
    ctx.set_source_rgba(*rgb, alpha)
    ctx.set_line_width(wd)
    ctx.arc(CX, CY, rm, a0, a1)
    ctx.stroke()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)


MELT = np.array([x[0] for x in EV['mel']])
MELM = [x[1] for x in EV['mel']]
MEL_LVL = [min(1.0, 0.25 + float(CUR['pad'][int((tm + 2.0) * CR)])) for tm in MELT]   # a vertex as bright as its note sounds


def mel_xy(k, m):
    a = ang((k % 8) / 8)
    r = 20 + (m - 63) * 13.0                    # E4 33 px ... C5 137 px, inside the pinger's ring
    return CX + r * math.cos(a), CY + r * math.sin(a)


def _reprise():
    """The heads of drop 2's 7-figure from bar 113, which take the breakdown's melody, note k for bars 113 + 2k."""
    out = []
    for tn, m, _, _ in NOTES:
        b = int(tn // BAR) + 1
        if 113 <= b <= EV['fig7'][-1] and round((tn - tb(106)) / SIX) % 7 == (-(105 * 16)) % 7:
            out.append((tn, m, (b - 113) // 2))
    return out


REPRISE = _reprise()

HIST, DR = 52, 14.0                 # bars of history kept, px of drift a bar
HIST_ON = os.environ.get('MOIRE_HIST', '1') == '1'
HIST_ECHO = os.environ.get('MOIRE_ECHO', '1') == '1'
ECHO_N = 24
HIST_LEAD = os.environ.get('MOIRE_HIST_LEAD', '0') == '1'

FIRST = {'bar': 0.0, 'bounce': tb(8), 'ping': tb(22)}      # each machine's ring is drawn in as it first plays


def ring_in(name, t):
    return min(1.0, max(0.0, (t - FIRST[name]) / (2 * BAR)))


def text(ctx, s_, x, y, size, alpha, spacing=0.0):
    ctx.select_font_face('Lato', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)
    ctx.set_source_rgba(0.88, 0.90, 0.95, alpha)
    for ch in s_:
        ctx.move_to(x, y)
        ctx.show_text(ch)
        x += ctx.text_extents(ch).x_advance + spacing


def ring_level(name, t):
    return {'bar': 1.0, 'bounce': min(1.0, cur('bounce', t, 0.15) * 1.2), 'ping': min(1.0, cur('ping', t, 0.15) * 1.2)}[name]


def frame(t):
    s = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    ctx = cairo.Context(s)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    fade = min(1.0, t / 2.0, max(0.0, (DUR - 0.5 - t) / 3.0))
    ctx.set_source_rgb(0.012, 0.012, 0.022)
    ctx.paint()
    bar1 = t / BAR + 1
    n_now = t / SIX                                    # sixteenths since the start
    kick = cur('kick', t)
    bed = min(1.2, cur('bed', t, 0.1))
    lead = min(1.2, cur('lead', t, 0.08))
    in_solo = EV['fig7'][0] <= bar1 < EV['fig7'][-1] + 1

    # the record: every first note of the bouncer and pinger (and every lead note) leaves a dot where it fell in the
    # bar, which then drifts outward, DR px a bar, for HIST bars: a polar record of the piece, where the 7 and 5
    # draw their own spirals and meet at 12 o'clock at the alignments
    if HIST_ON:
        for name, L in (('bounce', 7), ('ping', 5)):
            k_last = math.floor(n_now / L) * L
            for c in range(0, int(HIST * 16 / L) + 2):
                k = k_last - c * L
                age = (t - k * SIX) / BAR
                if age < 0 or k < 0:
                    continue
                if age > HIST:
                    break
                if ring_level(name, k * SIX + 0.01) <= 0.04:
                    continue
                rd = 392 + DR * age
                a = 0.55 * min(1.0, max(0.0, (rd - 420) / 120)) * (1 - age / HIST) ** 0.7
                if a < 0.01:
                    continue
                aa = ang((k % 16) / 16)
                ctx.set_source_rgba(*COL[name], a)
                ctx.arc(CX + rd * math.cos(aa), CY + rd * math.sin(aa), 3.2 + 0.006 * rd, 0, 2 * math.pi)
                ctx.fill()
        if HIST_LEAD:
            i1 = int(np.searchsorted(NT, t, side='right'))
            i0 = int(np.searchsorted(NT, t - HIST * BAR, side='left'))
            for i in range(i0, i1):
                tn = NOTES[i][0]
                age = (t - tn) / BAR
                rd = 392 + DR * age
                a = 0.40 * min(1.0, max(0.0, (rd - 560) / 80)) * (1 - age / HIST) ** 0.7
                if a < 0.01:
                    continue
                aa = ang((tn % BAR) / BAR)
                ctx.set_source_rgba(*COL['lead'], a)
                ctx.arc(CX + rd * math.cos(aa), CY + rd * math.sin(aa), 1.8 + 0.003 * rd, 0, 2 * math.pi)
                ctx.fill()

    # echoes of the lead's trail: each finished bar's trail drifts outward with the record
    if HIST_ECHO:
        bnow = math.floor(t / BAR)
        for bb in range(max(0, bnow - ECHO_N), bnow):
            b_start = bb * BAR
            age = (t - (b_start + BAR)) / BAR            # bars since this trail finished
            off = DR * age
            ts = np.arange(b_start, b_start + BAR, 1 / 120)
            idx = np.minimum(len(CUR['lead_f0']) - 1, (ts * CR).astype(int))
            f = CUR['lead_f0'][idx]
            lv = CUR['lead'][idx]
            if lv.max() < 0.05:
                continue
            rad = 420 + 36 * np.log2(np.maximum(f, 20) / 41.2) + off
            a = 0.30 * (1 - age / ECHO_N) ** 1.5
            ctx.set_source_rgba(*COL['lead'], a)
            ctx.set_line_width(1.4)
            on = False
            for j in range(len(ts)):
                if lv[j] < 0.08:
                    on = False
                    continue
                p = ang((ts[j] - b_start) / BAR)
                x, y = CX + rad[j] * math.cos(p), CY + rad[j] * math.sin(p)
                if on:
                    ctx.line_to(x, y)
                else:
                    ctx.move_to(x, y)
                    on = True
            ctx.stroke()

    # the bed: a halo between the inner and outer rings, coloured by its chord, breathing with the pump
    if bed > 0.02:
        bi = int(t // BAR) % 8
        r, g, b = CHORD_COL[bi]
        pat = cairo.RadialGradient(CX, CY, 100, CX, CY, 380)
        a = 0.16 * bed * (1 - 0.35 * kick)
        pat.add_color_stop_rgba(0.0, r, g, b, 0.0)
        pat.add_color_stop_rgba(0.55, r, g, b, a)
        pat.add_color_stop_rgba(1.0, r, g, b, 0.0)
        ctx.set_source(pat)
        ctx.arc(CX, CY, 380, 0, 2 * math.pi)
        ctx.fill()

    # the rings' marks, dim; the lit ones by recent notes
    for name, (r0, r1, n) in RINGS.items():
        rin = ring_in(name, t)
        if rin <= 0:
            continue
        lvl = ring_level(name, t)
        base = 0.035 + 0.035 * lvl if name != 'bar' else 0.07
        nn = 16 if name == 'bar' else n
        for i in range(nn):
            seg(ctx, name, i, nn, COL[name], rin * base * (1.6 if (name == 'bar' and i % 4 == 0) or (name != 'bar' and i == 0) else 1.0))
        # recent notes on this ring: the last few sixteenths, decaying
        play = name == 'bar' or lvl > 0.04
        for back in range(0, 6):
            k = math.floor(n_now) - back
            age = t - k * SIX
            if age < 0:
                continue
            step = k % (16 if name == 'bar' else n)
            head = step == 0
            if name == 'bar':
                q = (k % 4 == 0)
                lit = (0.55 + 0.45 * (kick if q else 0)) * (1.0 if q else 0.45)
            elif in_solo and name == 'bounce':
                continue
            else:
                lit = lvl * (1.0 if head else 0.55) if play else 0.0
            lv_ = lit * math.exp(-age / 0.22)
            if lv_ > 0.02:
                seg(ctx, name, step, 16 if name == 'bar' else n, COL[name], min(1.0, 0.25 + 0.75 * lv_), glow=lv_ > 0.3)

    # the lead in 7, in the solo: gold on the bouncer's ring
    if in_solo:
        r0, r1, n = RINGS['bounce']
        for back in range(0, 4):
            k = math.floor(n_now) - back
            age = t - k * SIX
            i = int(np.searchsorted(NT, k * SIX + 1e-4, side='right')) - 1
            if i < 0 or abs(NOTES[i][0] - k * SIX) > 0.01:
                continue
            step = k % 7
            lv_ = min(1, lead) * (1.0 if step == 0 else 0.55) * math.exp(-age / 0.16)
            if lv_ > 0.02:
                seg(ctx, 'bounce', step, 7, COL['lead'], min(1.0, 0.25 + 0.75 * lv_), glow=lv_ > 0.3, pad=8)

    # heads: where each first note of the bouncer and pinger fell in the bar, kept for a cycle (7 or 5 bars)
    for name, L, rr in (('bounce', 7, 380), ('ping', 5, 380)):
        lvl_now = ring_level(name, t)
        k_last = math.floor(n_now / L) * L
        for c in range(0, 16 * L // L * L // L + 16):           # enough heads to cover L bars (16 per bar / L... )
            k = k_last - c * L
            age = t - k * SIX
            if age < 0:
                continue
            if age > L * BAR:
                break
            pos = (k % 16) / 16
            sounding = ring_level(name, k * SIX + 0.01) > 0.04
            a = (0.85 if sounding else 0.25) * (1 - age / (L * BAR)) ** 1.5 * ring_in(name, k * SIX)
            aligned = k % 16 == 0
            rd = rr + (8 if name == 'bounce' else 22)
            ctx.set_source_rgba(*COL[name], a)
            ctx.arc(CX + rd * math.cos(ang(pos)), CY + rd * math.sin(ang(pos)), 7 if aligned else 5, 0, 2 * math.pi)
            ctx.fill()

    # the beam: when a cycle's first note lands on the downbeat, light rises through the rings at 12 o'clock
    for name, L in (('bounce', 7), ('ping', 5)):
        k_al = math.floor(n_now / (16 * L)) * 16 * L          # the last downbeat where this cycle agreed with the bar
        age = t - k_al * SIX
        if 0 <= age < 2 * BAR and k_al > 0:
            lv = math.exp(-age / (0.7 * BAR)) * (0.5 + 0.5 * ring_level(name, k_al * SIX + 0.01))
            r0 = RINGS[name][0]
            glow_spoke(ctx, ang(0), r0 - 10, 405, COL[name], 0.8 * lv)
    for g in GRAND:
        age = t - tb(g)
        if 0 <= age < 3 * BAR:
            lv = math.exp(-age / BAR)
            glow_spoke(ctx, ang(0), 60, 540, (1, 1, 1), 0.7 * lv)
            rw = 372 + 420 * age                       # and a ring of light runs out through the record
            if rw < 1150:
                ctx.set_source_rgba(1, 0.95, 0.85, 0.45 * math.exp(-age / (1.2 * BAR)))
                ctx.set_line_width(3.0)
                ctx.arc(CX, CY, rw, 0, 2 * math.pi)
                ctx.stroke()
            pat = cairo.RadialGradient(CX, CY, 0, CX, CY, 420)
            pat.add_color_stop_rgba(0, 1, 0.95, 0.85, 0.10 * lv)
            pat.add_color_stop_rgba(1, 1, 0.95, 0.85, 0.0)
            ctx.set_source(pat)
            ctx.paint()

    # the lead's trail round the outside: round = time through this bar, out = pitch
    if lead > 0.02:
        b0 = math.floor(t / BAR) * BAR
        for b_start, alpha in (((b0 - BAR, 0.25), (b0, 1.0)) if not HIST_ECHO else ((b0, 1.0),)):
            ts = np.arange(max(b_start, 0), min(b_start + BAR, t), 1 / 240)
            if len(ts) < 2:
                continue
            f = np.array([cur('lead_f0', x) for x in ts])
            rad = 420 + 36 * np.log2(np.maximum(f, 20) / 41.2)
            lv = np.array([cur('lead', x, 0.03) for x in ts])
            for width, al in ((10, 0.10), (4, 0.30), (1.8, 0.95)):
                ctx.set_line_width(width)
                for j in range(1, len(ts)):
                    a_ = alpha * al * min(1.0, lv[j]) * (0.35 + 0.65 * math.exp(-(t - ts[j]) / 1.2))
                    if a_ < 0.01:
                        continue
                    ctx.set_source_rgba(*COL['lead'], a_)
                    p0 = ang((ts[j - 1] - b_start) / BAR)
                    p1 = ang((ts[j] - b_start) / BAR)
                    ctx.move_to(CX + rad[j - 1] * math.cos(p0), CY + rad[j - 1] * math.sin(p0))
                    ctx.line_to(CX + rad[j] * math.cos(p1), CY + rad[j] * math.sin(p1))
                    ctx.stroke()

    # the breakdown's melody, a note every two bars, drawn as a figure in the middle: round is the note's place in a
    # 16-bar turn (eight notes), out is its pitch; each note's edge grows as it swells in, and its glass drop (half a
    # beat later) ripples from its vertex. In drop 2 (113-119) the lead's 7-figure heads carry the same melody: their
    # notes relight the figure's first vertices in gold
    fig_env = min(1.0, max(0.0, (tb(108) - t) / (3 * BAR)))
    if fig_env > 0 and t >= MELT[0]:
        kn = int(np.searchsorted(MELT, t, side='right'))           # notes begun so far
        for k in range(kn):
            age = t - MELT[k]
            grow = min(1.0, age / 0.8)
            lvl = MEL_LVL[k]
            x1, y1 = mel_xy(k, MELM[k])
            if k > 0:
                x0, y0 = mel_xy(k - 1, MELM[k - 1])
                a = fig_env * lvl * (0.32 + 0.55 * math.exp(-age / (2 * BAR)))
                for width, al in ((7, 0.12), (1.6, 1.0)):
                    ctx.set_source_rgba(*COL['mel'], a * al)
                    ctx.set_line_width(width)
                    ctx.move_to(x0, y0)
                    ctx.line_to(x0 + (x1 - x0) * grow, y0 + (y1 - y0) * grow)
                    ctx.stroke()
            held = k == kn - 1 and age < 2 * BAR
            glow = fig_env * lvl * (0.35 + (0.65 * min(1.0, cur('pad', t, 0.1)) if held else 0.0))
            ctx.set_source_rgba(*COL['mel'], glow)
            ctx.arc(x1, y1, 4.5 if held else 3.0, 0, 2 * math.pi)
            ctx.fill()
            if held:
                ctx.set_source_rgba(*COL['mel'], 0.12 * glow)
                ctx.arc(x1, y1, 14, 0, 2 * math.pi)
                ctx.fill()
            ad = age - BAR / 8                                       # the glass drop
            if 0 <= ad < 2.5:
                ctx.set_source_rgba(*COL['mel'], 0.45 * lvl * math.exp(-ad / 0.8))
                ctx.set_line_width(1.6)
                ctx.arc(x1, y1, 6 + 60 * ad, 0, 2 * math.pi)
                ctx.stroke()
    for tn, m, vertex in REPRISE:                                   # drop 2's heads on the melody
        age = t - tn
        if 0 <= age < 4 * BAR:
            x, y = mel_xy(vertex, m)
            a = min(1.0, lead) * math.exp(-age / (0.9 * BAR))
            ctx.set_source_rgba(*COL['lead'], 0.9 * a)
            ctx.arc(x, y, 4.0, 0, 2 * math.pi)
            ctx.fill()
            ctx.set_source_rgba(*COL['lead'], 0.15 * a)
            ctx.arc(x, y, 16, 0, 2 * math.pi)
            ctx.fill()
            if age < 1.5:
                ctx.set_source_rgba(*COL['lead'], 0.35 * math.exp(-age / 0.5))
                ctx.set_line_width(1.4)
                ctx.arc(x, y, 5 + 50 * age, 0, 2 * math.pi)
                ctx.stroke()

    # titles
    a_t = min(1.0, max(0.0, (t - 1.5) / 1.5)) * min(1.0, max(0.0, (tb(7) - t) / 2.0))
    if a_t > 0:
        text(ctx, 'OPUSTEK', 96, H - 132, 22, 0.55 * a_t, spacing=6)
        text(ctx, 'Moiré', 96, H - 84, 44, 0.85 * a_t, spacing=2)
    a_e = min(1.0, max(0.0, (t - tb(144)) / 2.0)) * min(1.0, max(0.0, (DUR - 1.0 - t) / 2.0))
    if a_e > 0:
        text(ctx, 'OPUSTEK', 96, H - 96, 22, 0.55 * a_e, spacing=6)

    s.flush()
    buf = np.frombuffer(s.get_data(), np.uint8).reshape(H, s.get_stride() // 4, 4)[:, :W]
    rgb = buf[..., [2, 1, 0]].astype(np.float32) * fade
    return rgb.astype(np.uint8)


def _frame_bytes(k):
    return np.ascontiguousarray(frame(k / FPS)).tobytes()


def render(t0=0.0, t1=None, path=None, workers=4):
    from multiprocessing import Pool
    t1 = DUR if t1 is None else t1
    path = path or os.path.join(OUT, 'wheels.mp4')
    k0, k1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    wav = os.path.join(ROOT, 'out', 'moire.wav')
    ff = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                           '-i', '-', '-ss', f'{t0:.3f}', '-t', f'{t1 - t0:.3f}', '-i', wav,
                           '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p',
                           '-c:a', 'aac', '-b:a', '256k', '-shortest', path], stdin=subprocess.PIPE)
    with Pool(workers) as p:
        for i, b in enumerate(p.imap(_frame_bytes, range(k0, k1), chunksize=8)):
            ff.stdin.write(b)
    ff.stdin.close()
    return ff.wait()


if __name__ == '__main__':
    a = sys.argv
    if '--stills' in a:
        from PIL import Image
        for x in a[a.index('--stills') + 1:]:
            Image.fromarray(frame(float(x))).save(os.path.join(OUT, f'wheel_{float(x):06.1f}.png'))
    elif '--render' in a:
        t0 = float(a[a.index('--from') + 1]) if '--from' in a else 0.0
        t1 = float(a[a.index('--to') + 1]) if '--to' in a else None
        out = a[a.index('--out') + 1] if '--out' in a else None
        sys.exit(render(t0, t1, out, int(a[a.index('--workers') + 1]) if '--workers' in a else 4))
