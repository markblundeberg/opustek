"""Engine, Learning to Sing: the film. An 1890s engineering plate whose
machine runs on the track: everything that moves is driven by the song's own
score and control curves (engine/film/export.py writes them from song.py).

Ink on aged paper, shade lines on lower-right edges, section hatching on cut
walls only, italic reference numerals on leader lines; one spot colour, the
hot bulb's, which is a blackbody's (Planck through the CIE 1931 colour-matching
functions). Five figures: Fig. 1 the engine on its bed; Fig. 2 the resonator;
Fig. 3 the commutator disc face-on; Fig. 4 the drip-pan in plan; Fig. 5 the
gauge. Sheet units: 3200 x 1800; the drawing's scale is 0.3 mm a unit.

The engine is a crankcase-scavenged two-stroke with a hot bulb, fired once a
revolution, so its crank turns at the note (2,800-7,400 rpm). Running, a
frame can't resolve it: the camera is drawn as Marey's chronophotographic one,
five exposures a frame, in phantom lines. Each opening of the music's gate
fires a spark gap (a spark stroboscope), which freezes the crank solid at that
instant. The engine stands on springs and rocks against its own torque.

  python3 engine/film/film.py [--from T0 --to T1] [--name N] [--workers 4]
  FILM_BREAK=flat-heat|blade-phase|open-gate|no-strobe|blackout|no-coast|short-skirt|
             misplaced-numeral|late-thump|gated-thump|no-rock|flat-speed|all-phantom|loose-engine
             (the checks' self-tests)
"""
import argparse
import json
import math
import os
import subprocess
from multiprocessing import Pool

import cairo
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'out', 'engine_film')
W, H, FPS = 1920, 1080, 30
CR = 1000

INK = (0.13, 0.11, 0.10)
PAPER = (0.95, 0.925, 0.855)

C = dict(np.load(os.path.join(OUT, 'curves.npz')))     # read fully before forking
E = json.load(open(os.path.join(OUT, 'events.json')))
TC = C['tc']
BAR, BEAT, DUR = E['bar'], E['beat'], E['dur']
tb = lambda bar, beat=0.0: (bar - 1) * BAR + beat * BEAT
BRK = os.environ.get('FILM_BREAK', '')


def at(name, t):
    return float(np.interp(t, TC, C[name]))


# --- physics -----------------------------------------------------------------

def _g(l, mu, s1, s2):
    s = np.where(l < mu, s1, s2)
    return np.exp(-0.5 * ((l - mu) / s) ** 2)


LAM = np.linspace(380, 780, 401)
CMF = (1.056 * _g(LAM, 599.8, 37.9, 31.0) + 0.362 * _g(LAM, 442.0, 16.0, 26.7) - 0.065 * _g(LAM, 501.1, 20.4, 26.2),
       0.821 * _g(LAM, 568.8, 46.9, 40.5) + 0.286 * _g(LAM, 530.9, 16.3, 31.1),
       1.217 * _g(LAM, 437.0, 11.8, 36.0) + 0.681 * _g(LAM, 459.0, 26.0, 13.8))   # CIE 1931 (Wyman et al. 2013)


def blackbody(T):
    """sRGB colour (max channel 1) and luminance Y of a blackbody at T kelvin."""
    l = LAM * 1e-9
    B = 1 / (l ** 5 * (np.exp(1.4388e-2 / (l * T)) - 1))
    X, Y, Z = (np.trapz(B * c, LAM) for c in CMF)
    rgb = np.clip([3.2406 * X - 1.5372 * Y - 0.4986 * Z, -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
                   0.0557 * X - 0.2040 * Y + 1.0570 * Z], 0, None)
    rgb = rgb / rgb.max()
    gam = lambda c: 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return tuple(gam(c) for c in rgb), Y


T_MAX = 1150.0                         # cherry red: brighter than a working hot bulb's dull red, for the screen
Y_DRAPER, Y_HOT = blackbody(798)[1], blackbody(T_MAX)[1]


def glow(T):
    """Spot colour and strength 0..1: none below the Draper point (~800 K),
    strength the log of visible luminance. Dull heat is a deep, dark red, so
    the colour is darkened as the strength falls (see fig1)."""
    if BRK == 'flat-heat':
        return (1.0, 0.45, 0.1), 0.8
    if T < 798:
        return (0, 0, 0), 0.0
    col, Y = blackbody(T)
    b = float(np.clip((math.log10(Y) - math.log10(Y_DRAPER)) / (math.log10(Y_HOT) - math.log10(Y_DRAPER)), 0, 1))
    return col, b


def heat_curve():
    """The head's temperature: dT/dt = q*HOT - a(T^4 - Ta^4) - b(T - Ta), with
    radiation and convection to a 300 K room. One physical law gives a fast
    rise and a slow dull-red tail. Time-compressed some tens of times (the
    35 mm bulb's own tau at 1000 K is a minute or two); a, b give tau ~ 3 s.
    q is set so the hottest moment reaches T_MAX."""
    hot = np.clip(C['env'], 0, 1.5) * (0.35 + 0.8 * C['dc_drive'] + 1.5 * C['dist'] + 0.6 * C['dc_fold']
                                        + 0.3 * C['dc_growl'] + 0.4 * C['dc_squelch'])
    a, b, Ta, dt = 7.7e-11, 0.026, 300.0, 1 / CR

    def run(q):
        T = np.empty_like(hot)
        y = Ta
        for i, h in enumerate(hot):
            y += dt * (q * h - a * (y ** 4 - Ta ** 4) - b * (y - Ta))
            T[i] = y
        return T
    lo, hi = 10.0, 2000.0
    for _ in range(30):
        q = 0.5 * (lo + hi)
        if run(q).max() > T_MAX:
            hi = q
        else:
            lo = q
    return run(lo)


def engine_speed():
    """rev/s of the crank. A two-stroke fires once a revolution, so its note is
    its speed: running, the crank follows f0 (46-123 rev/s, 2,800-7,400 rpm)
    with a light engine's 50 ms lag, but no faster than its torque allows:
    at most 350 rev/s^2 up (the octave climb's own need), and no harder down
    than it coasts, since a crankcase two-stroke has almost no engine braking.
    Off, it coasts on friction and pumping: dw/dt = -w/0.35 - 4 rev/s^2, so
    from idle it stops in about 1.2 s. The chop at bar 55 chokes it, and it
    coasts through the bar's pause (56) before the drop restarts it."""
    on = np.convolve((C['env'] > 0.03).astype(float), np.ones(50) / 50, mode='same') > 0.5
    if BRK != 'no-coast':
        on &= ~((TC >= tb(56)) & (TC < tb(57)))
    f, w, dt = C['f0'], np.zeros(len(C['f0'])), 1 / CR
    y = 0.0
    for i in range(len(f)):
        coast = -(y / 0.35 + 4.0) if y > 0 else 0.0
        if on[i] and BRK != 'flat-speed':
            y += dt * float(np.clip((f[i] - y) / 0.05, coast, 350.0))
        else:
            y = max(0.0, y + dt * coast)
        w[i] = y
    return w


TEMP = heat_curve()
SPEED = engine_speed()                                 # rev/s
THETA = 2 * np.pi * np.cumsum(SPEED) / CR
E_SPARKS = [a for a, o, e in E['chop_cells'] if o - a > 0.004]   # every opening of the gate
SPARKS = [] if BRK == 'no-strobe' else E_SPARKS


def theta_at(t):
    return float(np.interp(t, TC, THETA))


def spark_in_frame(t):
    """The spark caught by the frame at t (exposure [t, t + 1/FPS)), or None.
    The commutator fires the gap as each open cell of the gate begins."""
    for ts in SPARKS:
        if t <= ts < t + 1 / FPS:
            return ts
    return None


def running(t):
    """True when the crank sweeps more than 0.3 rad in a frame's exposure: a
    camera can't resolve it, and the drawing shows the moving parts in phantom
    (their extreme positions and paths, the draftsman's convention)."""
    return 2 * np.pi * float(np.interp(t, TC, SPEED)) / FPS > 0.3 or BRK == 'all-phantom'


# --- exposures: the ghosts and the spark's freeze, composited in linear light ------------
# Mark: "have the freeze frame punch in at full opacity and then decay, while
# the running piston has complementary opacity; in true light intensity".

GHOST_W = 0.45                         # the ghosts' strength with no spark in view
FREEZE_TAU = 0.12                      # s: the freeze's decay (about the gap between sparks at 8 a second)
LAYERS, LAYER_W = {}, {}


def freeze_state(t):
    """(spark time, strength k) for the most recent spark that is still showing:
    k = 1 in the frame that caught it, then exp(-dt / FREEZE_TAU)."""
    best = None
    for ts in SPARKS:
        if ts < t + 1 / FPS:
            best = ts
        else:
            break
    if best is None:
        return None, 0.0
    k = 1.0 if best >= t else math.exp(-(t - best) / FREEZE_TAU)
    return (best, k) if k > 0.03 else (None, 0.0)


def layer(name, cx):
    """A transparent layer the size of the frame, drawn with cx's transform."""
    if name not in LAYERS:
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        LAYERS[name] = (surf, cairo.Context(surf))
    c = LAYERS[name][1]
    m = cx.get_matrix()
    cx.identity_matrix()                     # cx's clip (the reveal's wipe), in device space
    rects = cx.copy_clip_rectangle_list()
    cx.set_matrix(m)
    c.reset_clip()
    c.identity_matrix()
    c.new_path()
    for rc in rects:
        c.rectangle(*rc)
    c.clip()
    c.set_matrix(m)
    return c


def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def composite(s):
    """Lay the exposures onto the frame in linear light: out = out (1 - wA) + col wA."""
    if not LAYERS:
        return
    s.flush()
    buf = np.ndarray((H, s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :W]
    out = _lin(buf[..., :3].astype(np.float32) / 255)
    for name in ('ghost', 'freeze'):
        if name not in LAYERS or LAYER_W.get(name, 0) <= 0:
            continue
        surf = LAYERS[name][0]
        surf.flush()
        L = np.ndarray((H, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :W].astype(np.float32) / 255
        A = L[..., 3:4]
        col = _lin(np.where(A > 0, L[..., :3] / np.maximum(A, 1e-6), 0))
        wA = LAYER_W[name] * A
        out = out * (1 - wA) + col * wA
    buf[..., :3] = (_srgb(out) * 255 + 0.5).astype(np.uint8)
    s.mark_dirty()
    LAYERS.clear()
    LAYER_W.clear()


# --- the pause: the engine is choked in bar 56 and coasts ------------------------

PAUSE = (tb(56), tb(57))


# --- camera ------------------------------------------------------------------

FIG = {'sheet': (1600, 900, 0.6), 'f1': (1060, 800, 0.97), 'f1close': (1180, 790, 1.3),
       'head': (1400, 770, 1.5), 'flywheel': (640, 860, 1.4), 'f2': (2580, 500, 1.45),
       'f3pair': (2250, 1020, 1.4), 'strobe': (1290, 860, 0.9), 'strobeclose': (1090, 845, 1.35), 'f4': (760, 1500, 1.7), 'f5': (1580, 1480, 1.9), 'f1x': (1180, 800, 0.8)}
# (bar, beats, target). Beats 0 is a cut, on the beat (big jumps between
# figures); smooth moves only where they can stay under ~30 px a frame.
CAM = [(1, 0, 'sheet'),
       (7, 8, 'f1'),               # push in over the last two intro bars: the engine catches as we arrive
       (13, 0, 'flywheel'),        # the groove: the flywheel bucking on the kick
       (17, 0, 'f1close'),         # the engine wakes: barrel, head, valve, resonator
       (25, 0, 'f2'),              # the sync voice starts learning (two bars)
       (27, 0, 'head'),            # the bulb starts to glow
       (31, 0, 'f1'),
       (33, 0, 'flywheel'),        # the solo: two-bar shots
       (35, 0, 'head'),
       (37, 8, 'f1close'),         # a slow push
       (39.75, 0, 'strobe'),       # a beat before the tresillo chop: the sparks and the crank they freeze
       (41, 0, 'f1'),
       (43.75, 0, 'strobeclose'),  # the 32nd chop, on the crank
       (44.5, 0, 'f3pair'),        # then the disc that did it
       (46, 0, 'head'),            # the engine's last bars, then it coasts and ticks
       (52, 0, 'f4'),              # the first doot back: one ring in the pan
       (53, 0, 'f1'),              # the engine wakes
       (54.75, 0, 'strobeclose'),  # the chop that chokes it: the crank crawls back in the sparks, then coasts through the pause
       (57, 0, 'sheet'),           # the drop: everything at once, for one bar
       (58, 0, 'flywheel'), (60, 0, 'head'), (61, 0, 'f2'), (63, 0, 'f1close'), (65, 0, 'f5'), (67, 0, 'f1'),
       (69, 0, 'head'),
       (71.75, 0, 'strobe'),       # the chop that prises it open
       (73, 0, 'flywheel'),        # the octave climb
       (75, 4, 'f1'),
       (77, 0, 'head'),            # the bulb near its peak
       (78.75, 0, 'strobeclose'),  # the triplet chop: the crank creeping forward a few degrees a spark
       (81, 0, 'f5'),
       (83, 0, 'head'),            # the outro's tinks
       (87, 8, 'sheet'), (91, 6, 'f1close'),   # the switch-off, close: the phantoms resolve into a turning crank
       (96, 8, 'f1x')]             # pull back while it comes apart


def smoother(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def camera(t):
    cur = FIG['sheet']
    for bar, beats, name in CAM:
        s = tb(bar)
        d = max(beats * BEAT, 1e-3)
        tgt = FIG[name]
        if t >= s:
            e = smoother((t - s) / d) if beats else 1.0
            cur = tuple(a + (b - a) * e for a, b in zip(cur, tgt))
    x, y, z = cur
    x += 14 * math.sin(t * 0.21)
    y += 9 * math.sin(t * 0.17 + 1)
    sx = sy = 0.0
    # no camera shake: it shook the parts that aren't on the engine too (Mark); the engine bucks instead
    return x, y, z, sx, sy


def to_screen(t, X, Y):
    x, y, z, sx, sy = camera(t)
    return (X - x) * z + W / 2 + sx, (Y - y) * z + H / 2 + sy


# --- ink -----------------------------------------------------------------------

def ink(cx, w=2.2, a=1.0, col=INK):
    cx.set_source_rgba(*col, a)
    cx.set_line_width(w)
    cx.set_line_cap(cairo.LINE_CAP_ROUND)
    cx.set_line_join(cairo.LINE_JOIN_ROUND)


def line(cx, x0, y0, x1, y1, w=2.2, a=1.0):
    cx.new_path()
    ink(cx, w, a)
    cx.move_to(x0, y0)
    cx.line_to(x1, y1)
    cx.stroke()


def rect(cx, x, y, w, h, lw=2.2, shade=True, fill=False):
    """Light from the upper left: the lower and right edges carry a heavier
    shade line."""
    cx.new_path()
    if fill:
        cx.rectangle(x, y, w, h)
        cx.set_source_rgba(*PAPER, 1)
        cx.fill()
    ink(cx, lw)
    cx.move_to(x + w, y)
    cx.line_to(x, y)
    cx.line_to(x, y + h)
    cx.stroke()
    ink(cx, lw * (2.1 if shade else 1.0))
    cx.move_to(x, y + h)
    cx.line_to(x + w, y + h)
    cx.line_to(x + w, y)
    cx.stroke()


def circle(cx, x, y, r, lw=2.2, shade=True, fill=False):
    cx.new_path()
    if fill:
        cx.arc(x, y, r, 0, 2 * math.pi)
        cx.set_source_rgba(*PAPER, 1)
        cx.fill()
    ink(cx, lw)
    cx.arc(x, y, r, 0, 2 * math.pi)
    cx.stroke()
    if shade:
        ink(cx, lw * 2.1)
        cx.arc(x, y, r, -math.pi / 4, 3 * math.pi / 4)
        cx.stroke()


def poly(cx, pts, lw=2.2, fill=True, close=True):
    cx.new_path()
    cx.move_to(*pts[0])
    for p in pts[1:]:
        cx.line_to(*p)
    if close:
        cx.close_path()
    if fill:
        cx.set_source_rgba(*PAPER, 1)
        cx.fill_preserve()
    ink(cx, lw)
    cx.stroke()


def hatch(cx, path_fn, step=10, lw=1.1, ang=45, a=0.9):
    """Section hatching, only on cut surfaces, clipped to them."""
    cx.save()
    cx.new_path()
    path_fn()
    cx.clip()
    x0, y0, x1, y1 = cx.clip_extents()
    ink(cx, lw, a)
    d = (x1 - x0) + (y1 - y0)
    k = math.tan(math.radians(ang))
    for o in np.arange(-d, d, step):
        cx.move_to(x0 + o, y1)
        cx.line_to(x0 + o + (y1 - y0) / k, y0)
    cx.stroke()
    cx.restore()


def pipe(cx, pts, r=5, lw=1.4):
    """A pipe drawn double (axis-aligned runs), with a fitting at each bend."""
    for sgn in (-1, 1):
        cx.new_path()
        for i, (x, y) in enumerate(pts):
            ox = oy = 0.0
            for j in (i - 1, i):
                if 0 <= j < len(pts) - 1:
                    dx, dy = pts[j + 1][0] - pts[j][0], pts[j + 1][1] - pts[j][1]
                    n = math.hypot(dx, dy) or 1
                    if abs(dx) > abs(dy):
                        oy = sgn * r
                    else:
                        ox = sgn * r
            (cx.move_to if i == 0 else cx.line_to)(x + ox, y + oy)
        ink(cx, lw)
        cx.stroke()
    for (x, y) in pts[1:-1]:
        rect(cx, x - r - 3, y - r - 3, 2 * r + 6, 2 * r + 6, 1.4, shade=False, fill=True)


def font(cx, size, italic=False, bold=False, face='FreeSerif'):
    cx.select_font_face(face, cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    cx.set_font_size(size)


def text(cx, s, x, y, size, italic=False, bold=False, center=False, face='FreeSerif', a=1.0):
    cx.new_path()
    font(cx, size, italic, bold, face)
    cx.set_source_rgba(*INK, a)
    if center:
        e = cx.text_extents(s)
        x -= e.width / 2 + e.x_bearing
    cx.move_to(x, y)
    cx.show_text(s)
    cx.new_path()


def numeral(cx, n, x, y, tx, ty):
    """A reference numeral on a fine curved leader that touches its part."""
    cx.new_path()
    ink(cx, 1.2)
    mx, my = (x + tx) / 2 + (ty - y) * 0.12, (y + ty) / 2 - (tx - x) * 0.12
    cx.move_to(tx, ty)
    cx.curve_to(mx, my, mx, my, x, y)
    cx.stroke()
    text(cx, str(n), tx + (8 if tx >= x else -26), ty + 8, 30, italic=True)


def fig_label(cx, s, x, y, size=46):
    text(cx, s, x, y, size, italic=True, bold=True)


# --- the sheet -------------------------------------------------------------------

def header(cx):
    """The plate's heading, as an engineering journal printed it: the plate's
    number and the date; the title goes at the foot (see signatures)."""
    text(cx, 'PLATE IV.', 150, 200, 54, bold=True)
    font(cx, 34)
    w = cx.text_extents('October 2026.').x_advance
    text(cx, 'October 2026.', 3050 - w, 200, 34)


def signatures(cx):
    """The title at the foot, and a plain ruled box: who drew it, who checked
    it, and a scale bar for Fig. 1 (honest at any print size)."""
    text(cx, 'ENGINE, LEARNING TO SING.', 1560, 1712, 40, center=True)
    x0, y0, x1, y1 = 2380, 1560, 3090, 1722
    rect(cx, x0, y0, x1 - x0, y1 - y0, 1.8, shade=False)
    text(cx, 'Drawn by', x0 + 22, y0 + 40, 26, italic=True)
    text(cx, 'Opustek.', x0 + 150, y0 + 42, 36, face='Z003')
    text(cx, 'Checked by', x0 + 22, y0 + 88, 26, italic=True)
    text(cx, 'a spectrogram;  M. Lundeberg.', x0 + 180, y0 + 90, 32, face='Z003')     # the ears
    u = 10 / MM_PER_UNIT                     # 0 to 50 mm at the drawing's scale, in 10 mm cells
    sx, sy = x0 + 22, y0 + 122
    for k in range(5):
        cx.new_path()
        cx.rectangle(sx + k * u, sy, u, 9)
        cx.set_source_rgba(*(INK if k % 2 == 0 else PAPER), 1)
        cx.fill_preserve()
        ink(cx, 1.2)
        cx.stroke()
    for k, lab in ((0, '0'), (5, '50 mm')):
        text(cx, lab, sx + k * u - (4 if k == 0 else 18), sy + 30, 20)
    text(cx, 'Scale of Fig. 1.  Figs. 2-5 enlarged.', sx + 5 * u + 50, sy + 12, 22, italic=True)


def make_paper():
    """The sheet's own paper, fixed to the sheet (it moves with the camera):
    toned from its four edges, a little foxing, fibre grain that also dithers
    the tone so it cannot band."""
    w, h = 1600, 900                                   # half resolution, scaled 2x
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy]).astype(float)
    tone = np.clip(1 - edge / 140, 0, 1) ** 1.6 * 0.10
    fox = np.zeros((h, w))
    for _ in range(26):
        cxp = rng.uniform(0, w)
        cyp = rng.uniform(0, 120) if rng.uniform() < 0.5 else rng.uniform(h - 120, h)
        r = rng.uniform(1.5, 4)
        fox += 0.035 * np.exp(-((xx - cxp) ** 2 + (yy - cyp) ** 2) / (2 * r * r))
    grain = rng.normal(0, 0.012, (h, w))
    dark = np.clip(tone + fox + grain, -0.03, 0.25)
    base = np.array(PAPER)
    rgb = np.clip(base[None, None, :] * (1 - dark[..., None]) - (dark + 2 * fox)[..., None] * np.array([0.0, 0.05, 0.12]), 0, 1)
    buf = np.zeros((h, w, 4), np.uint8)
    buf[..., 0] = (rgb[..., 2] * 255).astype(np.uint8)
    buf[..., 1] = (rgb[..., 1] * 255).astype(np.uint8)
    buf[..., 2] = (rgb[..., 0] * 255).astype(np.uint8)
    buf[..., 3] = 255
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_ARGB32, w, h, w * 4)
    return surf, buf


PAPER_SURF, _PAPER_BUF = make_paper()


def sheet_paper(cx):
    cx.save()
    cx.scale(2, 2)
    cx.set_source_surface(PAPER_SURF, 0, 0)
    cx.get_source().set_filter(cairo.FILTER_BILINEAR)
    cx.rectangle(0, 0, 1600, 900)
    cx.fill()
    cx.restore()


def sheet_border(cx):
    rect(cx, 60, 60, 3080, 1680, 2.4, shade=False)


# --- Fig. 1 geometry (sheet units) ---------------------------------------------------
# A crankcase-scavenged two-stroke (Joseph Day's 1889 layout, as on the later
# Bolinder and Lanz hot-bulb engines), fired once a revolution, so the crank
# turns at the note itself. Drawn at 0.3 mm a unit: bore 52 mm, stroke 84 mm,
# a 150 mm steel flywheel inside the crankcase (rim speed 58 m/s at 7,400 rpm).
# The piston is as long as the stroke so its skirt seals the exhaust port at
# TDC; a deflector on its crown turns the charge from the transfer port up.

FW = (640, 860, 250)                 # crank axis and the flywheel's radius (inside the case)
CASE = (282, 26)                     # crankcase inner radius and wall
RC, LROD = 140, 566                  # crank radius, rod length pin to pin: stroke 280, L/r 4
PIN_OFF, PW = (46, 150) if BRK == 'short-skirt' else (178, 282)   # wrist pin from the skirt end; piston length (its skirt seals the exhaust at TDC)
CYL = (920, 740, 680, 240)           # barrel, from the crankcase to the head; axis y = 860
WALL = 34
HEAD_X, HEAD = 1600, (1600, 700, 110, 320)   # the head: TDC crown 1450, clearance 150, CR ~2.9
GROUND = 1308
BASE = (300, 1222, 1940, 40)         # the bed
SEAT = 1172                          # spring seats: the engine stands on springs, 50 units (15 mm)
TUBE1 = (1060, 470, 1820, 34)        # resonator, on its feed pipe
CH1 = (1960, 935, 150)               # commutator disc (edge-on in Fig. 1); the brush is at y = 827
COIL = (2110, 1080, 120, 60)         # the induction coil, on the bed
GAP = (2170, 880, 18)                # the spark gap: centre, half-gap to each ball's surface
GAUGE1 = (1470, 598, 42)             # the gauge, on its own port
PAN = (1385, 1177, 95, 45)           # drip pan on the bed
DRIP = (1430, 1034, 1192)            # drain cock x, spout y, water surface y
MM_PER_UNIT = 0.3                    # the drawing's scale
JACKET = (1000, 1440, 24, 16)        # jacket x0, x1, water space, outer wall
HOPPER = (1060, 1400, 600, 14)       # open-topped hopper over the jacket: x0, x1, top y, wall
PORTS = {'gauge': 1470, 'valve': 1530, 'feed': 1560}   # on the barrel, in the clearance volume (crown TDC 1450)
TRANSFER = (1185, 1222)              # in the top wall: opens second, as the crown passes 1222
EXHAUST = (1180, 1280)               # in the bottom wall: opens first, as the crown passes 1280 (95 deg after TDC)
SILENCER = (1125, 1068, 140, 64)
FEET = (1030, 1300)                  # the barrel's feet
FUEL_Y = 420                         # the fuel line's run, over the resonator
BULB = (1785, 860, 58)               # the hot bulb on the head: the one thing that glows
BOLTS = [(1580, 760), (1580, 960), (1622, 730), (1622, 990), (1690, 730), (1690, 990)]
PHANTOM = [26, 6, 5, 6, 5, 6]        # long dash, two short: alternate positions
GHOSTS = 5                           # exposures a frame, running (Marey)

# the exploded view, as the engine is stripped: the head and the bed-mounted electrics
# stay; the crankcase comes off to the left; the barrel, with its piston and rod,
# slides off the head; its fittings lift off their ports
EXPLODE = {'case': (-190, 0), 'barrel': (-60, 0), 'tube': (0, -170), 'gauge': (0, -110), 'valve': (0, -110)}


def explode_k(t):
    return smoother((t - tb(96)) / 4.0)


def off(part, t):
    k = explode_k(t)
    dx, dy = EXPLODE.get(part, (0, 0))
    return dx * k, dy * k


PIVOT = (950, SEAT)                  # the springs' elastic centre: the engine rocks about it


def mount(drive, wn=2 * np.pi * 5, z=0.5):
    """A damped 5 Hz mount driven by `drive` (1/s^2 units, at CR)."""
    x = v = 0.0
    dt = 1 / CR
    out = np.empty_like(drive)
    for i, a in enumerate(drive):
        v += dt * (-2 * z * wn * v - wn * wn * x + a)
        x += dt * v
        out[i] = x
    return out


THUMP_LEAD = float(os.environ.get('FILM_THUMP_LEAD_MS', '30')) / 1000
# the thump peaks 30 ms ahead of the kick: Mark picked it blind from leads of
# -30, 0, 30, 60 and 90 ms (a sound usually seems simultaneous with a sight
# when it arrives a little after it)


def rocking():
    """The engine stands on springs and bucks: (1) against its own torque,
    reacting to every change of speed (-I dw/dt), as a motorbike's frame twists
    when the throttle is blipped; and (2) on each kick, which the film takes
    as the engine's heavy stroke (a conceit), harder the harder the engine is
    driven: its drive before the music's gate, so the thumps carry on through
    the chops (Mark felt the freezes break the rocking's rhythm: scaled by the
    gated sound, they dropped to a tenth there).
    The torque rings a damped 5 Hz mount (zeta 0.5); the kick is one heavily
    damped thump (4 Hz, zeta 0.7) timed to peak on the beat. The sum is soft-limited at 2.5 degrees
    (2.5 tanh(x / 2.5)), scaled so a kick bucks about 1 degree at the groove's
    level and the octave climb's glides reach about 2. Returns (angle, bounce)."""
    acc = np.gradient(SPEED) * CR                    # rev/s^2, now within -170..350
    torque = mount(-acc)
    torque = torque / (np.percentile(np.abs(torque), 99.9) + 1e-12) * 2.0        # degrees
    imp = np.zeros_like(acc)
    lvl = np.clip((C['env'] if BRK == 'gated-thump' else 0.9 * C['amp']) / 0.75, 0, 1.4)   # amp ~ env / 0.9 outside the chops
    for k in E['kicks']:
        i = int((k - (0.0 if BRK == 'late-thump' else 0.054 + THUMP_LEAD)) * CR)   # early by the mount's rise, so the thump peaks on the kick
        if 0 <= i < len(imp) - 20:
            imp[i:i + 20] += lvl[i + 54] / 0.020              # a 20 ms shove
    kick = mount(imp, wn=2 * np.pi * (5 if BRK == 'late-thump' else 4), z=(0.25 if BRK == 'late-thump' else 0.7))              # heavily damped: one thump and settle (Mark: the 5 Hz ring hid the beat)
    kick = kick / (np.percentile(np.abs(kick[np.abs(kick) > 0]), 99) + 1e-12) * 2.0
    ang = 2.5 * np.tanh((torque + kick) / 2.5)
    bounce = np.clip(kick, 0, None) * 2.0                     # units: the kick also presses it down
    return np.radians(ang), bounce


ROCK, BOUNCE = rocking()


def rock_at(t):
    return 0.0 if BRK == 'no-rock' else float(np.interp(t, TC, ROCK)) * (1 - explode_k(t))


def bounce_at(t):
    return 0.0 if BRK == 'no-rock' else float(np.interp(t, TC, BOUNCE)) * (1 - explode_k(t))


def engine_point(t, X, Y):
    """Where a point on the engine is drawn at t: rocked about the springs'
    elastic centre, and pressed down on the kick."""
    p, dyb = rock_at(t), bounce_at(t)
    dx, dy = X - PIVOT[0], Y - PIVOT[1]
    return PIVOT[0] + dx * math.cos(p) - dy * math.sin(p), PIVOT[1] + dx * math.sin(p) + dy * math.cos(p) + dyb


def crank(th):
    """Crank pin and wrist pin x (true slider-crank, axes aligned)."""
    cpx, cpy = FW[0] + RC * math.cos(th), FW[1] + RC * math.sin(th)
    wx = FW[0] + RC * math.cos(th) + math.sqrt(LROD ** 2 - (RC * math.sin(th)) ** 2)
    return cpx, cpy, wx


def crown(th):
    return crank(th)[2] - PIN_OFF + PW


ECC, INJ = 10, math.radians(-25)      # the pump's eccentric throw; its stroke peaks 25 deg before TDC


def cam_lift(th):
    """The fuel pump's plunger lift (0..2 ECC), from an eccentric on the
    crankshaft: a sinusoid, so its follower never leaves it."""
    return ECC * (1 + math.cos(th - INJ))


def reed_open(th):
    """The intake flap opens once the transfer port has closed and the piston,
    moving toward the head, has drawn the crankcase below atmosphere: from
    about 245 deg to 15 deg after TDC."""
    a = math.degrees(th) % 360
    return a >= 245 or a <= 15


def phantom(cx, w=1.3):
    """Phantom lines (long dash, two short), never thinner than 1.2 px on
    screen, so the wide shots don't turn them into grey hairlines."""
    m = cx.get_matrix()
    sc = math.hypot(m.xx, m.yx) or 1.0
    cx.set_dash(PHANTOM)
    ink(cx, max(w, 1.2 / sc))


# --- the moving parts -----------------------------------------------------------------

def draw_flywheel(cx, th, X, Y, mode):
    """The full-disc flywheel inside the case: lightening holes and the crank
    pin's boss. 'solid' at angle th; or a list of angles,
    the frame's exposures, for the holes in phantom and the pin's path."""
    R = FW[2]
    circle(cx, X, Y, R, 2.4, fill=True)
    circle(cx, X, Y, R - 22, 1.0, shade=False)
    if mode == 'solid':
        for k in range(4):
            a = th + math.pi / 4 + k * math.pi / 2
            circle(cx, X + 190 * math.cos(a), Y + 190 * math.sin(a), 32, 1.8, shade=False)
    else:                                    # running: the holes at each of the frame's exposures, in phantom
        g = layer('ghost', cx)
        for tj in mode:
            for k in range(4):
                a = tj + math.pi / 4 + k * math.pi / 2
                g.new_path()
                phantom(g, 1.0)
                g.arc(X + 190 * math.cos(a), Y + 190 * math.sin(a), 32, 0, 2 * math.pi)
                g.stroke()
        g.set_dash([])
        cx.save()
        cx.new_path()
        phantom(cx, 1.1)
        cx.arc(X, Y, RC, 0, 2 * math.pi)     # the crank pin's path
        cx.stroke()
        cx.restore()
    circle(cx, X, Y, 44, 2.0, fill=True)
    circle(cx, X, Y, 18, 1.6, shade=False)


def draw_rod_piston(cx, th, X, mode, w=1.0):
    """Rod and piston at crank angle th, the crank pin on a case at X. The rod
    is rigid and leaves with the piston (in the barrel group) in the exploded
    view; the pin stays on the case."""
    cpx, cpy, wx = crank(th)
    rbx, rby = cpx, cpy
    cpx += X - FW[0]
    by, bh = CYL[1], CYL[3]
    if mode == 'phantom':
        cx.save()
        cx.new_path()
        phantom(cx, 1.3 * w)
        ra = math.atan2(860 - rby, wx - rbx)
        nx, ny = -math.sin(ra), math.cos(ra)
        for sg in (1, -1):
            cx.move_to(rbx + sg * 15 * nx, rby + sg * 15 * ny)
            cx.line_to(wx + sg * 10 * nx, 860 + sg * 10 * ny)
        cx.stroke()
        cx.new_path()
        cx.arc(rbx, rby, 22, ra + math.pi / 2, ra + 3 * math.pi / 2)     # the big end's eye
        cx.stroke()
        px = wx - PIN_OFF
        cx.rectangle(px, by + WALL + 3, PW, bh - 2 * WALL - 6)
        cx.stroke()
        cx.restore()
        return
    ra = math.atan2(860 - rby, wx - rbx)
    nx, ny = -math.sin(ra), math.cos(ra)
    poly(cx, [(rbx + 15 * nx, rby + 15 * ny), (wx + 10 * nx, 860 + 10 * ny),
              (wx - 10 * nx, 860 - 10 * ny), (rbx - 15 * nx, rby - 15 * ny)], 2.0)
    circle(cx, cpx, cpy, 22, 2.0, fill=True)
    circle(cx, cpx, cpy, 8, 1.4, shade=False)
    if abs(rbx - cpx) > 1:                   # apart: the big end's own eye, empty, on a centre line from its pin
        circle(cx, rbx, rby, 22, 2.0, fill=True)
        circle(cx, rbx, rby, 9, 1.4, shade=False)
        cx.save()
        cx.new_path()
        ink(cx, 1.2, 0.8 * min(1.0, abs(rbx - cpx) / 100))
        cx.set_dash([26, 7, 4, 7])
        cx.move_to(cpx - 40, cpy)
        cx.line_to(rbx + 40, rby)
        cx.stroke()
        cx.restore()
    px = wx - PIN_OFF
    rect(cx, px, by + WALL + 3, PW, bh - 2 * WALL - 6, 2.2, fill=True)
    cr, top = px + PW, by + WALL + 3
    poly(cx, [(cr, top), (cr + 22, top), (cr + 22, top + 30), (cr, top + 66)], 2.0)    # the deflector
    for k in (26, 42, 58):
        line(cx, px + PW - k, by + WALL + 3, px + PW - k, by + bh - WALL - 3, 1.2)
    circle(cx, wx, 860, 14, 1.8, shade=False, fill=True)


def moving_parts(cx, t, th, X, run):
    """Running, the crank turns 1.5 to 4 times in a frame. The camera is drawn
    as Marey's chronophotographic one (1882): five short exposures on each
    frame, so the rod and piston appear in phantom at five instants a fifth of
    a frame apart, swirling with the pitch. A spark freezes them solid at its
    own instant, at full strength in the frame that caught it, then fading as
    the ghosts come back (both composited in linear light). Slow or stopped,
    the parts are drawn solid where they are."""
    if run:
        ts, k = freeze_state(t)
        g = layer('ghost', cx)
        for j in range(GHOSTS):
            draw_rod_piston(g, theta_at(t + j / (GHOSTS * FPS)), X, 'phantom')
        LAYER_W['ghost'] = GHOST_W * (1 - k)
        if ts is not None:
            draw_rod_piston(layer('freeze', cx), theta_at(ts), X, 'solid')
            LAYER_W['freeze'] = k
    else:
        draw_rod_piston(cx, th, X, 'solid')


def fuel_pump(cx, th, X, run, t):
    """On the case top: an eccentric on the crankshaft works the plunger
    through a strap and rod; the pump feeds the line to the bulb's sprayer.
    Running: the rod in phantom at the stroke's two ends, and the spark's
    freeze solid."""
    fy = FW[1]
    top = fy - CASE[0] - CASE[1]
    rect(cx, X - 22, top - 80, 44, 80, 2.2, fill=True)
    line(cx, X - 22, top - 56, X + 22, top - 56, 1.0)
    ths = [(INJ, 'phantom', cx), (INJ + math.pi, 'phantom', cx)] if run else [(th, 'solid', cx)]
    ts = freeze_state(t)[0] if run else None
    if ts is not None:
        ths.append((theta_at(ts), 'solid', layer('freeze', cx)))
    for a_, mode, c in ths:
        ex, ey = X + ECC * math.sin(a_ - INJ), fy - ECC * math.cos(a_ - INJ)     # the sheave's centre
        c.save()
        c.new_path()
        if mode == 'phantom':
            phantom(c, 1.2)
        else:
            ink(c, 1.8)
        c.arc(ex, ey, 62, 0, 2 * math.pi)           # the strap
        c.stroke()
        c.new_path()
        c.move_to(ex - 6, ey - 62)
        c.line_to(X - 6, top - 50 - cam_lift(a_))
        c.line_to(X + 6, top - 50 - cam_lift(a_))
        c.line_to(ex + 6, ey - 62)
        c.stroke()
        c.restore()


def intake(cx, th, X, run, t):
    """Air in through a flap valve on the case's left side."""
    x0, y = X - CASE[0] - CASE[1], FW[1]
    poly(cx, [(x0 + 4, y - 26), (x0 - 50, y - 26), (x0 - 72, y - 44), (x0 - 72, y + 44), (x0 - 50, y + 26), (x0 + 4, y + 26)], 2.2)
    states = [(True, 'phantom', cx), (False, 'phantom', cx)] if run else [(reed_open(th), 'solid', cx)]
    ts = freeze_state(t)[0] if run else None
    if ts is not None:
        states.append((reed_open(theta_at(ts)), 'solid', layer('freeze', cx)))
    for st, mode, c in states:
        hx, hy = x0 - 50, y - 26                # hinged at the top of the housing's mouth
        a = math.radians(40 if st else 0)
        c.save()
        c.new_path()
        if mode == 'phantom':
            phantom(c, 1.2)
        else:
            ink(c, 2.4)
        c.move_to(hx, hy)
        c.line_to(hx + 50 * math.sin(a), hy + 50 * math.cos(a))
        c.stroke()
        c.restore()


def springs(cx, t, pts):
    """Coil springs from the engine's feet (rocking) down to the bed, drawn as
    the period drew them: the coil's two outlines and its slanted turns."""
    for x, y in pts:
        h = BASE[1] - y
        n = 6
        line(cx, x - 13, y, x - 13, BASE[1], 1.2)
        line(cx, x + 13, y, x + 13, BASE[1], 1.2)
        for j in range(n):
            y0 = y + h * j / n
            line(cx, x - 13, y0 + h / n * 0.15, x + 13, y0 + h / n * 0.85, 1.6)
        line(cx, x - 18, y, x + 18, y, 2.0)


def fig1(cx, t):
    T = float(np.interp(t, TC, TEMP))
    col, gb = glow(T)
    th = theta_at(t)
    run = running(t) and explode_k(t) < 0.01
    spark = spark_in_frame(t) if run else None
    fx, fy, R = FW
    bx, by, bw, bh = CYL
    cox_ = off('case', t)[0]
    X = fx + cox_
    box = off('barrel', t)[0]
    jx0, jx1, ws, jw = JACKET
    # ground and bed (fixed)
    line(cx, 320, GROUND, 2260, GROUND, 2.2)
    cx.new_path()
    ink(cx, 1.0)
    for k in range(320, 2246, 26):
        cx.move_to(k, GROUND + 8)
        cx.line_to(k + 14, GROUND + 20)
    cx.stroke()
    x, y, w, h = BASE
    hatch(cx, lambda: cx.rectangle(x, y, w, h), 12)
    rect(cx, x, y, w, h)
    for fxp in (x + 40, x + w - 120):
        rect(cx, fxp, y + h, 80, GROUND - (y + h), fill=True)
    # drip pan on the bed, the drop falling into it
    tx, ty, tw, th_ = PAN
    drip(cx, t, lambda tt: engine_point(tt, DRIP[0] + off('barrel', tt)[0], DRIP[1]), DRIP[2])
    rect(cx, tx, ty, tw, th_, 2.2, fill=True)     # drawn over the drop: it vanishes at the rim
    cx.save()
    cx.set_dash([12, 6])
    line(cx, tx + 6, DRIP[2], tx + tw - 6, DRIP[2], 1.2)
    cx.restore()
    # the springs, from the rocking feet to the bed
    feet = [engine_point(t, X - 80, SEAT), engine_point(t, X + 80, SEAT)] + \
           [engine_point(t, f + 30 + box, SEAT) for f in FEET]
    springs(cx, t, feet)
    # --- the engine, which rocks on its springs about PIVOT and bounces on the kick
    cx.save()
    p = rock_at(t)
    cx.translate(0, bounce_at(t))
    cx.translate(*PIVOT)
    cx.rotate(p)
    cx.translate(-PIVOT[0], -PIVOT[1])
    # crankcase: a cut ring (hatched), mouths for the bore and the intake, feet to the seats
    ro, ri = CASE[0] + CASE[1], CASE[0]
    surf = lambda xx: fy + math.sqrt(max(0.0, ro ** 2 - (xx - X) ** 2)) - 8
    for fxp in (X - 80, X + 80):             # the case's feet: part of the cut casting
        pts = [(fxp - 30, surf(fxp - 30)), (fxp + 30, surf(fxp + 30)), (fxp + 36, SEAT), (fxp - 36, SEAT)]
        poly(cx, pts, 2.2)
        hatch(cx, lambda pts=pts: (cx.move_to(*pts[0]), [cx.line_to(*q) for q in pts[1:]], cx.close_path()), 10)

    def ring():
        cx.arc(X, fy, ro, 0, 2 * math.pi)
        cx.new_sub_path()
        cx.arc_negative(X, fy, ri, 2 * math.pi, 0)
    hatch(cx, ring, 10)
    cx.new_path()
    cx.arc(X, fy, ri, 0, 2 * math.pi)
    cx.set_source_rgba(*PAPER, 1)
    cx.fill()
    circle(cx, X, fy, ro, 2.6)
    circle(cx, X, fy, ri, 1.8, shade=False)
    cx.new_path()
    cx.rectangle(X + ri - 8, by + WALL, ro - ri + 40, bh - 2 * WALL)      # the mouth into the bore
    cx.rectangle(X - ro - 4, fy - 26, ro - ri + 8, 52)                     # the intake's mouth
    cx.rectangle(X - 6, fy - ro - 4, 12, ro - ri + 8)                      # the follower's guide
    cx.set_source_rgba(*PAPER, 1)
    cx.fill()
    draw_flywheel(cx, th, X, fy, [theta_at(t + j / (GHOSTS * FPS)) for j in range(GHOSTS)] if run else 'solid')
    fts, _ = freeze_state(t) if run else (None, 0.0)
    if fts is not None:                      # the spark's freeze of the holes, in the freeze layer
        f = layer('freeze', cx)
        tz = theta_at(fts)
        for k in range(4):
            a = tz + math.pi / 4 + k * math.pi / 2
            circle(f, X + 190 * math.cos(a), fy + 190 * math.sin(a), 32, 2.0, shade=False)
    fuel_pump(cx, th, X, run, t)
    intake(cx, th, X, run, t)
    # --- the barrel group: it slides off the head in the exploded view
    cx.save()
    cx.translate(box, 0)
    # the barrel's feet, on the jacket's outer wall, down to the spring seats
    for fxp in FEET:
        poly(cx, [(fxp, by + bh + ws + jw), (fxp + 60, by + bh + ws + jw), (fxp + 76, SEAT), (fxp - 16, SEAT)], 2.2)
    # exhaust: through the bottom wall and the jacket, down into the silencer
    sx_, sy_, sw_, sh_ = SILENCER
    ex0, ex1 = EXHAUST
    rect(cx, ex0 + 15, by + bh + ws + jw, ex1 - ex0 - 30, sy_ - (by + bh + ws + jw), 2.0, fill=True)
    rect(cx, sx_, sy_, sw_, sh_, 2.4, fill=True)
    for k in (0.25, 0.5, 0.75):
        line(cx, sx_ + sw_ * k, sy_ + 4, sx_ + sw_ * k, sy_ + sh_ - 4, 1.0)
    circle(cx, sx_ + sw_ - 26, sy_ + sh_ / 2, 12, 2.0, fill=True)           # the tailpipe, end-on
    circle(cx, sx_ + sw_ - 26, sy_ + sh_ / 2, 6, 1.2, shade=False)
    exhaust_puffs(cx, t, sx_ + sw_ - 26, sy_ + sh_ / 2 - 14)
    # water jacket around the barrel: outer wall cut (hatched), water space open; the
    # hopper rises from it, open-topped (the hit-and-miss engine's arrangement)
    for (yy, hh) in ((by - ws - jw, jw), (by + bh + ws, jw)):
        hatch(cx, lambda yy=yy, hh=hh: cx.rectangle(jx0, yy, jx1 - jx0, hh), 10)
        rect(cx, jx0, yy, jx1 - jx0, hh, 2.0)
    for xe in (jx0, jx1 - jw):
        for (yy, hh) in ((by - ws - jw, ws + jw), (by + bh, ws + jw)):
            hatch(cx, lambda xe=xe, yy=yy, hh=hh: cx.rectangle(xe, yy, jw, hh), 10)
            rect(cx, xe, yy, jw, hh, 1.6, shade=False)
    # the exhaust passage crossing the water space (its own cast walls)
    cx.new_path()
    cx.rectangle(ex0 + 15, by + bh, ex1 - ex0 - 30, ws + jw)
    cx.set_source_rgba(*PAPER, 1)
    cx.fill()
    for xx in (ex0 + 15, ex1 - 23):          # its walls, cut where they cross the water space
        hatch(cx, lambda xx=xx: cx.rectangle(xx, by + bh, 8, ws + jw), 6)
        rect(cx, xx, by + bh, 8, ws + jw, 1.4, shade=False)
    hx0, hx1, htop, hwl = HOPPER
    jtop = by - ws - jw
    cx.new_path()
    cx.rectangle(hx0 + hwl, htop, hx1 - hx0 - 2 * hwl, jtop - htop + jw)
    cx.set_source_rgba(*PAPER, 1)
    cx.fill()
    for xe in (hx0, hx1 - hwl):
        hatch(cx, lambda xe=xe: cx.rectangle(xe, htop, hwl, jtop - htop), 10)
        rect(cx, xe, htop, hwl, jtop - htop, 2.0)
    cx.save()
    cx.set_dash([14, 7])
    line(cx, hx0 + hwl + 4, htop + 24, hx1 - hwl - 4, htop + 24, 1.2)       # water surface
    cx.restore()
    steam(cx, t, T, (hx0 + hx1) / 2, htop)
    # drain cock under the jacket
    rect(cx, DRIP[0] - 8, by + bh + ws + jw, 16, DRIP[1] - (by + bh + ws + jw), 1.6, fill=True)
    line(cx, DRIP[0] - 18, DRIP[1] - 8, DRIP[0] + 18, DRIP[1] - 12, 2.0)
    # the barrel in section: walls cut (hatched), bore open, the ports cut through
    for (yy, hh) in ((by, WALL), (by + bh - WALL, WALL)):
        hatch(cx, lambda yy=yy, hh=hh: cx.rectangle(bx, yy, bw, hh), 10)
    line(cx, bx, by, bx + bw, by, 2.6)
    line(cx, bx, by + bh, bx + bw, by + bh, 5.4)
    line(cx, bx + bw, by, bx + bw, by + bh, 5.4)
    for yy in (by, by + bh - WALL):          # the crank end: only the walls' ends, the bore stays open
        line(cx, bx, yy, bx, yy + WALL, 2.6)
    line(cx, bx, by + WALL, bx + bw, by + WALL, 1.8)
    line(cx, bx, by + bh - WALL, bx + bw, by + bh - WALL, 1.8)
    t0_, t1_ = TRANSFER                      # the transfer passage, in the top wall, from the case
    ps = fx + ri - 30 if explode_k(t) < 0.01 else bx
    cx.new_path()
    cx.rectangle(ps, by + 9, t1_ - ps, 17)
    cx.rectangle(t0_, by + 9, t1_ - t0_, WALL - 9 + 1)
    cx.set_source_rgba(*PAPER, 1)
    cx.fill()
    line(cx, ps, by + 9, t1_, by + 9, 1.4)
    line(cx, ps, by + 26, t0_, by + 26, 1.4)
    line(cx, t0_, by + 26, t0_, by + WALL, 1.4)
    line(cx, t1_, by + 9, t1_, by + WALL, 1.4)
    cx.new_path()
    cx.rectangle(ex0, by + bh - WALL - 1, ex1 - ex0, WALL + 2)             # the exhaust port
    cx.set_source_rgba(*PAPER, 1)
    cx.fill()
    line(cx, ex0, by + bh - WALL, ex0, by + bh, 1.6)
    line(cx, ex1, by + bh - WALL, ex1, by + bh, 1.6)
    for px_ in PORTS.values():
        cx.new_path()
        cx.rectangle(px_ - 7, by - 1, 14, WALL + 2)
        cx.set_source_rgba(*PAPER, 1)
        cx.fill()
        line(cx, px_ - 7, by, px_ - 7, by + WALL, 1.6)
        line(cx, px_ + 7, by, px_ + 7, by + WALL, 1.6)
    # the moving parts: rod and piston
    moving_parts(cx, t, th, X - box, run)     # the crank pin stays on the case
    for bxp, byp in BOLTS:
        if bxp < 1590:
            circle(cx, bxp, byp, 7, 1.6, fill=True)
    # gauge on its own port
    goy = off('gauge', t)[1]
    gx, gy, gr = GAUGE1
    rect(cx, PORTS['gauge'] - 5, gy + gr + goy, 10, by - (gy + gr + goy), 1.6, fill=True)
    gauge(cx, t, gx, gy + goy, gr)
    # relief valve on its port
    voy = off('valve', t)[1]
    vx, vy = PORTS['valve'], by + voy
    rect(cx, vx - 5, vy - 26, 10, 26, 1.6, fill=True)
    rect(cx, vx - 16, vy - 66, 32, 40, 2.0, fill=True)
    line(cx, vx - 28, vy - 66, vx + 28, vy - 66, 3.0)
    puffs(cx, t, vx, vy - 72)
    # resonator on its feed
    tox_, toy_ = off('tube', t)
    pipe(cx, [(PORTS['feed'], by), (PORTS['feed'], TUBE1[1] + TUBE1[3])], 4)          # stays on its port
    resonator(cx, t, TUBE1[0] + tox_, TUBE1[1] + toy_, TUBE1[2] + tox_, TUBE1[3])
    cx.restore()                             # end of the barrel group
    # the head, cut, and the hot bulb on it: the bulb carries the heat, the one spot colour
    hx, hy, hw, hh = HEAD
    hatch(cx, lambda: cx.rectangle(hx, hy, 34, hh), 10)
    rect(cx, hx, hy, hw, hh, 2.6)
    for k in range(6):
        line(cx, hx + 46 + k * 11, hy, hx + 46 + k * 11, hy + hh, 1.6)
    bxc, byc, br = BULB
    rect(cx, hx + hw, byc - 22, bxc - br - (hx + hw) + 6, 44, 2.0, fill=True)    # the neck
    circle(cx, bxc, byc, br, 2.6, fill=True)
    if gb > 0:
        cx.save()
        cx.set_operator(cairo.OPERATOR_MULTIPLY)
        dark = 0.55 + 0.45 * gb                     # dull heat is a deep, dark red
        cx.set_source_rgba(col[0] * dark, col[1] * dark, col[2] * dark, 0.35 + 0.6 * gb)
        cx.arc(bxc, byc, br - 2, 0, 2 * math.pi)
        cx.fill()
        cx.rectangle(hx + hw, byc - 21, bxc - br - (hx + hw) + 6, 42)
        cx.fill()
        cx.restore()
        circle(cx, bxc, byc, br, 2.6)
    rect(cx, bxc + br - 2, byc - 7, 20, 14, 1.6, fill=True)                  # the sprayer
    for bxp, byp in BOLTS:
        if bxp > 1590:
            circle(cx, bxp, byp, 7, 1.6, fill=True)
    tinks(cx, t)
    # the fuel line: pump to sprayer, over the resonator (it leaves with the head)
    top = fy - CASE[0] - CASE[1] - 80
    if explode_k(t) < 0.01:
        pipe(cx, [(X, top), (X, FUEL_Y), (1880, FUEL_Y), (1880, byc), (bxc + br + 18, byc)], 3)
    else:                                    # apart at the union over the pump: the run stays with the head
        pipe(cx, [(X, top), (X, FUEL_Y + 30)], 3)
        pipe(cx, [(fx + 40, FUEL_Y), (1880, FUEL_Y), (1880, byc), (bxc + br + 18, byc)], 3)
        cx.save()
        cx.new_path()
        ink(cx, 1.2, 0.8 * explode_k(t))
        cx.set_dash([26, 7, 4, 7])
        cx.move_to(X, FUEL_Y + 30)
        cx.line_to(X, FUEL_Y)
        cx.line_to(fx + 40, FUEL_Y)
        cx.stroke()
        cx.restore()
    if BRK != 'loose-engine':                # the fixed-parts check's self-test: the bug Mark saw in v7
        cx.restore()                         # end of the engine (it rocks; what follows stands on the bed)
    # --- fixed on the bed: the commutator, the coil and the spark gap
    chx, chy, chr_ = CH1
    chopper_edge(cx, t, chx, chy, chr_, spark)
    for n, x0_, y0_, x1_, y1_ in numeral_marks(t):
        if n in ON_ENGINE:                   # the engine's numerals ride with it, so their leaders stay on their parts
            (x0_, y0_), (x1_, y1_) = engine_point(t, x0_, y0_), engine_point(t, x1_, y1_)
        numeral(cx, n, x0_, y0_, x1_, y1_)
    if explode_k(t) > 0.01:
        exploded_lines(cx, t)
    fig_label(cx, 'Fig. 1.' if explode_k(t) < 0.5 else 'Fig. 1, exploded.', 1060, 1385)
    return col, gb


# what each reference numeral names, and the part's box (sheet units, assembled):
# check_film.py's numeral check asks that every leader lands inside its part
LEGEND = {1: ('flywheel', (390, 610, 890, 1110)), 2: ('water jacket', (1000, 700, 1440, 1020)),
          3: ('resonator', (1060, 436, 1820, 504)), 4: ('commutator disc', (1954, 785, 1966, 1085)),
          5: ('gauge', (1428, 556, 1512, 640)), 6: ('relief valve', (1514, 674, 1546, 740)),
          7: ('drip pan', (1385, 1177, 1480, 1222)), 8: ('hot bulb', (1727, 802, 1843, 918)),
          9: ('spark gap', (2140, 866, 2200, 894)), 10: ('hopper', (1060, 600, 1074, 700)),   # its near wall (the top is open)
          11: ('silencer', (1125, 1068, 1265, 1132)), 12: ('fuel pump', (618, 472, 662, 552)),
          13: ('intake', (260, 816, 336, 904)), 14: ('induction coil', (2110, 1080, 2230, 1140)),
          15: ('brush', (1934, 818, 1954, 836))}


ON_ENGINE = {1, 2, 3, 5, 6, 8, 10, 11, 12, 13}


def numeral_marks(t):
    """Fig. 1's numerals: (n, leader end on the part, text place), following
    their parts in the exploded view."""
    fx, fy, R = FW
    bx, by, bw, bh = CYL
    jx0, jx1, ws, jw = JACKET
    hx0, hx1, htop, hwl = HOPPER
    sx_, sy_, sw_, sh_ = SILENCER
    tx, ty, tw, th_ = PAN
    chx, chy, chr_ = CH1
    gx, gy, gr = GAUGE1
    vx = PORTS['valve']
    bxc, byc, br = BULB
    cox_, box = off('case', t)[0], off('barrel', t)[0]
    X, top = fx + cox_, fy - CASE[0] - CASE[1] - 80
    bg = lambda part=None: (box + (off(part, t)[0] if part else 0), off(part, t)[1] if part else 0)
    cxo, zero = (cox_, 0), (0, 0)
    g5 = (gx - 30, gy - 30) if BRK != 'misplaced-numeral' else (1340, 480)
    marks = [(1, cxo, fx - 175, fy + 175, fx - 300, fy + 330),
             (2, bg(), 1150, by + bh + ws + jw, 1080, by + bh + 110),
             (3, bg('tube'), 1700, TUBE1[1] + 20, 1730, TUBE1[1] + 130),
             (4, zero, chx - 6, chy + 110, chx - 90, chy + 210),
             (5, bg('gauge'), g5[0], g5[1], gx - 85, gy - 50),
             (6, bg('valve'), vx + 16, by - 50, vx + 90, by - 130),
             (7, zero, tx + tw, ty + 10, tx + tw + 60, ty - 50),
             (8, zero, bxc + 10, byc - br + 2, bxc + 20, byc - br - 75),
             (9, zero, GAP[0] + 22, GAP[1] + 12, GAP[0] + 80, GAP[1] + 70),
             (10, bg(), hx0 + hwl / 2, htop + 20, hx0 - 40, htop - 70),
             (11, bg(), sx_ + 70, sy_ + sh_, sx_ + 55, sy_ + sh_ + 60),
             (12, cxo, fx - 22, top + 30, fx - 110, top - 40),
             (13, cxo, fx - CASE[0] - 90, fy + 40, fx - CASE[0] - 160, fy + 130),
             (14, zero, COIL[0] + COIL[2], COIL[1] + 20, COIL[0] + COIL[2] + 40, COIL[1] - 30),
             (15, zero, chx - 26, chy - chr_ * 0.72 - 9, chx - 90, chy - chr_ * 0.72 - 90)]
    return [(n, x0 + dx, y0 + dy, x1 + dx, y1 + dy) for n, (dx, dy), x0, y0, x1, y1 in marks]


def exhaust_puffs(cx, t, x, y):
    """The exhaust, toward us out of the tailpipe: while the engine runs, a
    stream of small curls, more of them the harder it works."""
    if float(np.interp(t, TC, SPEED)) < 5:
        return
    for k in E['kicks']:                     # the bark: a cloud of curls out of the pipe on each kick
        d = t - k
        if 0 <= d < 0.35:
            r = 14 + 160 * d
            for j in range(6):
                ang = -math.pi / 2 + (j - 2.5) * 0.35
                cx.new_path()
                ink(cx, 1.4 * (1 - d / 0.35) + 0.4)
                cx.arc(x + r * math.cos(ang), y + r * math.sin(ang) - 10, 8 + 20 * d, ang, ang + 2.4)
                cx.stroke()
    n = int(np.clip(at('env', t) * 6, 1, 5))
    for k in range(n):
        ph = (t * 3.1 + k / n) % 1.0
        x0 = x - 6 + 12 * math.sin(k * 2.3)
        cx.new_path()
        ink(cx, 1.3)
        cx.move_to(x0, y - 50 * ph)
        for j in range(1, 7):
            u = j / 6
            cx.line_to(x0 + 6 * math.sin(5 * u + k + t * 7), y - 50 * ph - 22 * u * (1 - ph))
        cx.stroke()


def steam(cx, t, T, x, y):
    """Over a hot engine the open hopper steams: a few ink curls, full ink,
    more of them the hotter the bulb (a hopper-cooled engine boils)."""
    n = int(np.clip((T - 900) / 80, 0, 4))
    for k in range(n):
        ph = (t * 0.6 + k * 0.37) % 1.0
        x0 = x - 90 + k * 60
        cx.new_path()
        ink(cx, 1.4)
        cx.move_to(x0, y - 6)
        for j in range(1, 13):
            u = j / 12
            cx.line_to(x0 + 10 * math.sin(6 * u + k + t * 2), y - 6 - u * 70 * (0.4 + ph))
        cx.stroke()


def tinks(cx, t):
    """The metal plucks: the hot end ticking as it contracts. A sound mark at a
    bolt: two or three small arcs opening outward, thinning as it fades."""
    for tk, m in E['ks']:
        d = t - tk
        if 0 <= d < 0.4:
            bx, by_ = BOLTS[(m * 7 + int(tk * 10)) % len(BOLTS)]
            a = math.exp(-d / 0.12)
            if a < 0.08:
                continue
            side = 1 if bx > 1600 else -1
            for j in range(3):
                r = 16 + 12 * j + 30 * (1 - a)
                cx.new_path()
                ink(cx, 2.0 * a + 0.5)
                c0 = -0.6 if side > 0 else math.pi - 0.6
                cx.arc(bx, by_, r, c0, c0 + 1.2)
                cx.stroke()


def exploded_lines(cx, t):
    """Dash-dot centre lines from each part's home to where it has moved."""
    k = explode_k(t)
    homes = {'case': FW[:2], 'barrel': (1260, 860), 'tube': ((TUBE1[0] + TUBE1[2]) / 2, TUBE1[1]),
             'gauge': GAUGE1[:2], 'valve': (PORTS['valve'], CYL[1] - 46)}
    cx.save()
    cx.new_path()
    ink(cx, 1.2, 0.8 * k)
    cx.set_dash([26, 7, 4, 7])
    for part, (hx, hy) in homes.items():
        dx, dy = off(part, t)
        if part in ('tube', 'gauge', 'valve'):    # they ride on the barrel: their centre line moves with it
            hx += off('barrel', t)[0]
        if abs(dx) + abs(dy) < 2:
            continue
        cx.move_to(hx - 0.15 * dx, hy - 0.15 * dy)
        cx.line_to(hx + 1.15 * dx, hy + 1.15 * dy)
        cx.stroke()
    cx.restore()


def puffs(cx, t, x, y):
    """The relief valve lifts on the clap: ink curls that shorten as they go."""
    for c in E['claps']:
        d = t - c
        if 0 <= d < 0.4:
            a = 1 - d / 0.4
            for s in (-1, 0, 1):
                cx.new_path()
                ink(cx, 1.6)
                cx.move_to(x + s * 10, y - 140 * d)
                cx.curve_to(x + s * 18, y - 140 * d - 20 * a, x + s * 4, y - 140 * d - 40 * a, x + s * 14, y - 140 * d - 60 * a)
                cx.stroke()


def gauge_reading(t):
    """The gauge reads the engine's roar: its level, pushed by drive and the
    grit, with a nudge on the kick, and a needle that flutters harder the
    harder the engine works (an unsnubbed gauge on a single at 7,000 rpm can't
    hold still). Full scale is 1; the danger zone starts at 0.85."""
    roar = np.clip(at('env', t) * (0.75 + 0.6 * at('dc_drive', t) + 1.2 * at('dist', t)), 0, 1.25)
    if float(np.interp(t, TC, SPEED)) < 5:
        roar = 0.0
    kick = sum(math.exp(-(t - tk) / 0.08) for tk in E['kicks'] if 0 <= t - tk < 0.5)
    n = round(t * FPS)
    jit = (math.sin(n * 12.9898) * 43758.5453) % 1.0 * 2 - 1     # one draw per frame
    return float(np.clip(0.06 + 0.78 * roar + 0.07 * kick * roar + 0.07 * roar ** 2 * jit, 0, 1.08))


def gauge(cx, t, x, y, r):
    v = gauge_reading(t)
    a = math.radians(-215 + 250 * v)
    circle(cx, x, y, r, 2.4, fill=True)
    circle(cx, x, y, r - 10, 1.0, shade=False)
    cx.new_path()                            # the danger zone: a heavy band from 0.85 to the stop
    ink(cx, max(3.0, r * 0.09))
    cx.set_line_cap(cairo.LINE_CAP_BUTT)
    cx.arc(x, y, r - 17, math.radians(-215 + 250 * 0.85), math.radians(-215 + 250 * 1.0))
    cx.stroke()
    for j in range(11):
        aa = math.radians(-215 + 25 * j)
        line(cx, x + (r - 22) * math.cos(aa), y + (r - 22) * math.sin(aa), x + (r - 12) * math.cos(aa), y + (r - 12) * math.sin(aa), 1.4)
    line(cx, x, y, x + (r - 24) * math.cos(a), y + (r - 24) * math.sin(a), 3.0)
    circle(cx, x, y, 6, 1.6, shade=False)


def resonator(cx, t, x0, y, x1, hh, detail=False):
    """Glass, drawn in ink (reflection dashes are the glass convention). The
    wave crests on every beat: its true rate is audio, so any visible rate is
    an alias, and the beat is the honest one to choose."""
    lvl = float(np.clip(at('blend', t) * at('env', t) * 2.4, 0, 1))
    n = int(round(np.clip(at('ratio', t), 1, 22)))
    line(cx, x0, y - hh, x1, y - hh, 2.0)
    line(cx, x0, y + hh, x1, y + hh, 4.0)
    for ex in (x0, x1):
        cx.save()
        cx.new_path()
        cx.translate(ex, y)
        cx.scale(12, hh)
        cx.arc(0, 0, 1, 0, 2 * math.pi)
        cx.restore()
        ink(cx, 2.0)
        cx.stroke()
    for k in (0.3, 0.42):
        line(cx, x0 + 30 + (x1 - x0) * k, y - hh + 7, x0 + 70 + (x1 - x0) * k, y - hh + 7, 1.0)
    if lvl > 0.01:
        A = (hh - 6) * lvl
        ph = math.cos(2 * math.pi * t / BEAT)
        for sgn in (1, -1):
            cx.new_path()
            cx.move_to(x0, y)
            for k in range(0, 401):
                u = k / 400
                cx.line_to(x0 + u * (x1 - x0), y + sgn * A * ph * math.sin(n * math.pi * u))
            ink(cx, 2.0 if detail else 1.5)
            cx.stroke()
    for k in range(n + 1):
        cx.new_path()
        cx.arc(x0 + k / n * (x1 - x0), y, 5 if detail else 3, 0, 2 * math.pi)
        cx.set_source_rgba(*INK, 1)
        cx.fill()
    return n


def current_chop(t):
    for i, (bar, beats, per, d0, d1, pat) in enumerate(E['chops']):
        t0 = (bar - 1) * BAR
        if t0 - 1.0 <= t < t0 + beats * BEAT + 0.6:
            return i, t0, beats, per
    return None, None, None, None


def chop_state(t):
    """Disc phase (revolutions), the 16 sectors' open fractions, how live the
    chopper is (fading in and out around a chop), the gate now, and cells (sectors) per second."""
    i, t0, beats, per = current_chop(t)
    N = 16
    if i is None:
        return 0.03 * t, [1.0] * N, 0.0, 0.0, 0.0
    cell = BEAT / per
    p = (t - t0) / cell
    mine = [c for c in E['chop_cells'] if t0 - 1e-6 <= c[0] < t0 + beats * BEAT - 1e-6]
    rev = int(max(p, 0) // N)
    sec = []
    for k in range(N):
        j = rev * N + k
        if p < 0:
            sec.append(1.0)                          # before the chop: no cell has started, all open
        elif j < len(mine):
            a, o, e = mine[j]
            sec.append((o - a) / (e - a))
        else:
            sec.append(1.0)
    live = float(np.clip((t - (t0 - 0.6)) / 0.4, 0, 1) * np.clip((t0 + beats * BEAT + 0.6 - t) / 0.4, 0, 1))
    gate = at('chop', t)
    if BRK == 'open-gate':                       # the geometry check's self-test
        gate = 1.0
    return p / N, sec, live, gate, per / BEAT


def sector_angle(rot, k, N=16):
    """Start angle of sector k. The aperture is at the top (-pi/2), and sector
    k sits under it when the phase says cell k (FILM_BREAK=blade-phase puts it
    half a sector off, for the geometry check's self-test)."""
    a0 = rot - math.pi / 2 + k * 2 * math.pi / N
    return a0 - (math.pi / N if BRK == 'blade-phase' else 0.0)


def open_under_aperture(phase, sec, N=16):
    """Is the sector under the aperture open where the aperture is? (phase and
    sec from chop_state; None if no sector is found, at a seam.)"""
    rot = -2 * math.pi * phase
    for k in range(N):
        u = (-math.pi / 2 - sector_angle(rot, k, N)) % (2 * math.pi)
        if u < 2 * math.pi / N:
            return u < sec[k] * 2 * math.pi / N
    return None


def spark_draw(cx, x, y, g, ts, t):
    """The spark: a jagged stroke across the gap, and its light drawn the
    engraver's way, as fine solid strokes tapering outward (capped at r 70).
    Only in the frame that caught it."""
    rng = np.random.default_rng(int(ts * 1000))
    cx.new_path()
    ink(cx, 2.6)
    cx.move_to(x - g, y)
    for j in range(1, 6):
        cx.line_to(x - g + 2 * g * j / 6, y + rng.uniform(-9, 9))
    cx.line_to(x + g, y)
    cx.stroke()
    for k in range(12):
        a = k * 2 * math.pi / 12 + rng.uniform(-0.12, 0.12)
        r0, r1 = 26, 26 + rng.uniform(28, 44)
        ux, uy = math.cos(a), math.sin(a)
        cx.new_path()
        cx.move_to(x + r0 * ux - 1.6 * uy, y + r0 * uy + 1.6 * ux)
        cx.line_to(x + r1 * ux, y + r1 * uy)
        cx.line_to(x + r0 * ux + 1.6 * uy, y + r0 * uy - 1.6 * ux)
        cx.close_path()
        cx.set_source_rgba(*INK, 1)
        cx.fill()


def chopper_edge(cx, t, x, y, R, spark):
    """Fig. 1: the commutator disc edge-on on its motor. A brush rides its face;
    the circuit through it is broken as each cell of the gate opens, and the
    induction coil's collapsing field (a Ruhmkorff coil fires on the break)
    charges a Leyden jar across the spark gap, whose discharge lasts a
    microsecond: a spark stroboscope, as in Mach's spark photographs of the
    1880s. A cell (battery) on the floor drives the primary; the wires run
    along the bed."""
    by = y - R * 0.72
    rect(cx, x - 6, y - R, 12, 2 * R, 2.2, fill=True)
    rect(cx, x + 6, y - 22, 26, 44, 2.0, fill=True)
    rect(cx, x + 32, y - 55, 90, 110, 2.4, fill=True)
    for k in range(1, 5):
        line(cx, x + 32 + k * 18, y - 55, x + 32 + k * 18, y + 55, 1.0)
    poly(cx, [(x + 40, y + 55), (x + 114, y + 55), (x + 134, BASE[1]), (x + 20, BASE[1])], 2.2)
    # the brush, on an insulated post standing on the bed
    rect(cx, x - 26, by - 9, 20, 18, 1.8, fill=True)
    rect(cx, x - 50, by - 4, 24, 8, 1.4, fill=True)
    rect(cx, x - 58, by - 12, 12, BASE[1] - by + 12, 1.8, fill=True)
    # the coil: a wound cylinder on two blocks; its secondary's rods stand on insulating pillars
    cx_, cy_, cw_, ch_ = COIL
    for bxp in (cx_ + 12, cx_ + cw_ - 36):
        rect(cx, bxp, cy_ + ch_, 24, BASE[1] - cy_ - ch_, 1.8, fill=True)
    rect(cx, cx_, cy_, cw_, ch_, 2.4, fill=True)
    for k in range(1, 12):
        line(cx, cx_ + k * cw_ / 12, cy_ + 3, cx_ + k * cw_ / 12, cy_ + ch_ - 3, 0.9)
    gx, gy, g = GAP
    for sgn, rx in ((-1, cx_ + 14), (1, cx_ + cw_ - 14)):
        rect(cx, rx - 8, cy_ - 46, 16, 46, 1.8, fill=True)            # the ebonite pillar
        for k in range(1, 4):
            line(cx, rx - 8, cy_ - 46 + k * 11.5, rx + 8, cy_ - 46 + k * 11.5, 0.9)
        line(cx, rx, cy_ - 46, rx, gy, 2.0)
        line(cx, rx, gy, gx + sgn * (g + 12), gy, 2.0)
        circle(cx, gx + sgn * (g + 12), gy, 12, 2.0, fill=True)
    # the Leyden jar, across the gap: on the coil, between the pillars, wired to both rods
    jx, jy0, jy1 = gx, cy_ - 104, cy_ - 4
    rect(cx, jx - 20, jy0, 40, jy1 - jy0, 2.0, fill=True)
    rect(cx, jx - 20, jy0 + 40, 40, jy1 - jy0 - 40, 1.4, shade=False)  # the outer foil
    line(cx, jx, jy0, jx, jy0 - 24, 1.6)
    circle(cx, jx, jy0 - 28, 5, 1.4, fill=True)
    line(cx, jx, jy0 - 28, cx_ + 14, jy0 - 28, 1.0)                    # its knob to one rod
    line(cx, jx + 20, jy1 - 20, cx_ + cw_ - 14, jy1 - 20, 1.0)          # its foil to the other
    # the cell on the floor, and the primary's wires along the bed
    cl0, clw, clh = BASE[0] + BASE[2] + 18, 56, 84
    rect(cx, cl0, GROUND - clh, clw, clh, 2.2, fill=True)
    rect(cx, cl0 + 6, GROUND - clh - 10, clw - 12, 10, 1.4, fill=True)
    for k in (14, 42):
        line(cx, cl0 + k, GROUND - clh - 10, cl0 + k, GROUND - clh - 22, 1.6)
    cx.save()
    cx.new_path()
    ink(cx, 1.0)
    yb = BASE[1] - 6
    cx.move_to(x - 52, BASE[1])                                       # brush post foot ...
    cx.line_to(x - 52, yb)
    cx.line_to(cx_ - 8, yb)                                           # ... along the bed to the coil
    cx.line_to(cx_ - 8, cy_ + ch_ / 2)
    cx.line_to(cx_, cy_ + ch_ / 2)
    cx.move_to(cx_ + cw_, cy_ + ch_ / 2)                              # the coil's other end ...
    cx.line_to(cl0 + 14, cy_ + ch_ / 2)
    cx.line_to(cl0 + 14, GROUND - clh - 22)                           # ... to the cell
    cx.move_to(cl0 + 42, GROUND - clh - 22)                           # the cell's return, to the bed (the frame)
    cx.line_to(cl0 + 42, GROUND - clh - 40)
    cx.line_to(BASE[0] + BASE[2] - 4, GROUND - clh - 40)
    cx.line_to(BASE[0] + BASE[2] - 4, BASE[1])
    cx.stroke()
    cx.restore()
    if spark is not None:
        spark_draw(cx, gx, gy, g, spark, t)
    return by


def chopper_face(cx, t, x, y, R):
    """Fig. 3: the commutator disc face-on, from the head's side. The brush on
    its post rides the face; a blade (conducting) is a closed cell of the gate,
    an open sector insulated, so the circuit breaks as each cell opens.
    Blades carry surface lines (not cut). Cells faster than 12 a second are
    drawn spun, in rings: sectors that fast would strobe at 30 fps."""
    phase, sec, live, gate, rate = chop_state(t)
    N = 16
    rot = -2 * math.pi * phase
    by = y - R * 0.72
    circle(cx, x, y, R, 2.6, fill=True)
    if rate > 12:
        cx.new_path()
        ink(cx, 0.8)
        for rr in np.arange(52, R - 4, 6):
            cx.new_sub_path()
            cx.arc(x, y, rr, 0, 2 * math.pi)
        cx.stroke()
    else:
        for k in range(N):
            a0 = sector_angle(rot, k, N)
            a_open = a0 + sec[k] * 2 * math.pi / N
            a1 = a0 + 2 * math.pi / N
            if sec[k] < 0.999:
                mid = 0.5 * (a_open + a1)
                cx.save()
                cx.new_path()
                cx.move_to(x + 42 * math.cos(a_open), y + 42 * math.sin(a_open))
                cx.arc(x, y, R, a_open, a1)
                cx.arc_negative(x, y, 42, a1, a_open)
                cx.close_path()
                cx.clip()
                cx.new_path()
                ink(cx, 0.8)
                ux, uy = math.cos(mid), math.sin(mid)
                for o in np.arange(-R, R, 7):
                    cx.move_to(x - uy * o, y + ux * o)
                    cx.line_to(x - uy * o + ux * R, y + ux * o + uy * R)
                cx.stroke()
                cx.restore()
                line(cx, x + 42 * math.cos(a_open), y + 42 * math.sin(a_open), x + R * math.cos(a_open), y + R * math.sin(a_open), 1.6)
            line(cx, x + 42 * math.cos(a0), y + 42 * math.sin(a0), x + R * math.cos(a0), y + R * math.sin(a0), 1.0)
    circle(cx, x, y, R, 2.6)
    circle(cx, x, y, 42, 2.0, fill=True)
    circle(cx, x, y, 12, 1.6, shade=False)
    # hidden lines, drawn over the disc: the motor behind it and the stand up to the hub
    cx.save()
    cx.set_dash([10, 6])
    rect(cx, x - 55, y - 55, 110, 110, 1.3, shade=False)
    for sxp in (-1, 1):
        line(cx, x + sxp * 26, y + 42, x + sxp * (26 + 20 * (R - 42) / (R + 95 - 42)), y + R, 1.4)
    cx.restore()
    # the stand below the disc, widening to its foot, ended by a break line
    ws_ = lambda yy: 26 + 20 * (yy - (y + 42)) / (R + 95 - 42)
    yb_ = y + R + 95
    poly(cx, [(x - ws_(y + R), y + R), (x + ws_(y + R), y + R), (x + ws_(yb_), yb_), (x - ws_(yb_), yb_)], 2.2)
    cx.new_path()
    ink(cx, 1.4)
    cx.move_to(x - ws_(yb_) - 6, yb_ + 4)
    cx.curve_to(x - 10, yb_ - 8, x + 10, yb_ + 14, x + ws_(yb_) + 6, yb_ + 2)
    cx.stroke()
    # the brush, on its post standing in front of the disc
    rect(cx, x - 7, by + 14, 14, yb_ - by - 14, 1.8, fill=True)
    rect(cx, x - 11, by - 14, 22, 28, 1.8, fill=True)
    return by


def drip(cx, t, spout, y1):
    """A drop leaves the drain cock from rest and falls freely at g (at the
    drawing's 0.3 mm a unit: 41 mm in 91 ms), landing on each doot. The cock
    rocks with the engine, so the drop falls from where it was at release
    (spout(t) gives that point on the sheet)."""
    fall_m = (y1 - DRIP[1]) * MM_PER_UNIT / 1000
    T = math.sqrt(2 * fall_m / 9.81)
    for td, m, v in E['doots']:
        d = td - t
        if 0 < d <= T:
            x, y0 = spout(td - T)
            yy = y1 - (y1 - y0) * (1 - (1 - d / T) ** 2)
            cx.new_path()
            cx.move_to(x, yy - 14)
            cx.curve_to(x + 2, yy - 8, x + 6, yy - 5, x + 6, yy - 1)
            cx.arc(x, yy - 1, 6, 0, math.pi)
            cx.curve_to(x - 6, yy - 5, x - 2, yy - 8, x, yy - 14)
            cx.set_source_rgba(*INK, 1)
            cx.fill()


# --- the detail figures ----------------------------------------------------------------



def fig2(cx, t):
    x0, y, x1, hh = 2150, 520, 3040, 40                # Fig. 1's tube enlarged 1.17 times, in proportion
    n = resonator(cx, t, x0, y, x1, hh, detail=True)
    fig_label(cx, 'Fig. 2.', 2150, 410)
    text(cx, f'n = {n}', 2900, 400, 34, italic=True)
    numeral(cx, 3, 2500, y - hh, 2560, y - hh - 70)


def fig3(cx, t):
    x, y, R = 2720, 1010, 230
    chopper_face(cx, t, x, y, R)
    fig_label(cx, 'Fig. 3.', 2650, 1420)
    numeral(cx, 4, x + 163, y + 163, x + 230, y + 220)


def fig4(cx, t):
    """The drip pan in plan: each drop rings out from under the cock; rings
    thin as they fade (full ink: a plate has one ink)."""
    x, y, r = 760, 1500, 165
    circle(cx, x, y, r, 2.6, fill=True)
    circle(cx, x, y, r - 18, 1.4, shade=False)
    circle(cx, x, y, 7, 1.4, shade=False)
    cx.save()
    cx.new_path()
    cx.arc(x, y, r - 18, 0, 2 * math.pi)
    cx.clip()
    for td, m, v in E['doots']:
        if v < 0.85 and not tb(49) <= td < tb(57):
            continue
        for dt, g in ((0, 1.0), (0.5 * BEAT, 0.6), (BEAT, 0.35)):
            d = t - td - dt
            if 0 <= d < 1.6:
                lw = 1.8 * v * g * (1 - d / 1.6)
                if lw < 0.4:
                    continue
                cx.new_path()
                ink(cx, lw)
                cx.arc(x, y, 10 + 110 * d, 0, 2 * math.pi)
                cx.stroke()
    cx.restore()
    fig_label(cx, 'Fig. 4.', 440, 1400)
    numeral(cx, 7, x + 150, y - 70, x + 240, y - 90)


def fig5(cx, t):
    x, y, r = 1580, 1480, 120
    gauge(cx, t, x, y, r)
    text(cx, 'P', x - 10, y + 60, 30, italic=True)
    fig_label(cx, 'Fig. 5.', 1730, 1600)
    numeral(cx, 5, x + 102, y - 63, x + 180, y - 80)


# --- the intro: the sheet is inked part by part ------------------------------------------

REVEAL = ['header', 'border', 'fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'sign']




def reveal_times():
    acc = [td for td, m, v in E['doots'] if td < tb(8.5) and v > 0.85]
    acc = acc[::max(1, len(acc) // len(REVEAL))][:len(REVEAL)]
    while len(acc) < len(REVEAL):
        acc.append(tb(8))
    return dict(zip(REVEAL, acc))


RT = reveal_times()


def revealed(name, t, cx, bbox):
    u = (t - RT[name]) / 0.9
    if u <= 0:
        return False
    if u < 1:
        x0, y0, x1, y1 = bbox
        cx.rectangle(x0, y0, (x1 - x0) * smoother(u), y1 - y0)
        cx.clip()
    return True


# --- a frame -----------------------------------------------------------------------

def draw(t, path):
    LAYERS.clear()                           # nothing left over from a frame that failed
    LAYER_W.clear()
    s = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    cx = cairo.Context(s)
    cx.set_source_rgb(*[c * 0.9 for c in PAPER])        # the desk beyond the sheet's edge
    cx.paint()
    x, y, z, sx, sy = camera(t)
    cx.save()
    cx.translate(W / 2 + sx, H / 2 + sy)
    cx.scale(z, z)
    cx.translate(-x, -y)
    sheet_paper(cx)
    parts = [('header', (0, 60, 3200, 230), header), ('border', (0, 0, 3200, 1800), sheet_border),
             ('fig1', (60, 300, 2350, 1420), lambda c: fig1(c, t)), ('fig2', (2100, 330, 3100, 650), lambda c: fig2(c, t)),
             ('fig3', (2100, 760, 3100, 1580), lambda c: fig3(c, t)), ('fig4', (400, 1340, 1300, 1700), lambda c: fig4(c, t)),
             ('fig5', (1250, 1340, 1900, 1700), lambda c: fig5(c, t)), ('sign', (1250, 1530, 3110, 1735), signatures)]
    for name, bbox, fn in parts:
        cx.new_path()
        cx.save()
        if revealed(name, t, cx, bbox):
            fn(cx)
        cx.restore()
    cx.restore()
    composite(s)
    if BRK == 'blackout' and current_chop(t)[0] is not None and spark_in_frame(t) is None:
        cx.set_source_rgba(0, 0, 0, 0.85)       # the room dark between sparks: what the flash check forbids
        cx.paint()
    a = max(0.0, 1 - t / 0.6) + max(0.0, (t - (DUR - 1.5)) / 1.5)
    if a > 0:
        cx.set_source_rgba(*PAPER, min(a, 1))
        cx.paint()
    s.write_to_png(path)


def solid_frame(theta, path, t=60.0, full=True):
    """For reviewers: the frame at t, but with the engine stopped at crank
    angle theta (radians, 0 = TDC), drawn solid; full=True shows the whole
    sheet at 3200 x 1800."""
    global theta_at, running, rock_at, bounce_at, W, H, camera
    saved = theta_at, running, rock_at, bounce_at, W, H, camera
    theta_at = lambda tt: theta
    running = lambda tt: False
    rock_at = bounce_at = lambda tt: 0.0
    if full:
        W, H = 3200, 1800
        camera = lambda tt: (1600, 900, 1.0, 0, 0)
    try:
        draw(t, path)
    finally:
        theta_at, running, rock_at, bounce_at, W, H, camera = saved


def job(args):
    k, t0, fr = args
    draw(t0 + k / FPS, os.path.join(fr, f'{k:05d}.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='t0', type=float, default=0.0)
    ap.add_argument('--to', dest='t1', type=float, default=DUR)
    ap.add_argument('--name', default='film')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    fr = os.path.join(OUT, a.name + '_frames')
    os.makedirs(fr, exist_ok=True)
    for f in os.listdir(fr):
        os.remove(os.path.join(fr, f))
    n = int(round((a.t1 - a.t0) * FPS))
    with Pool(a.workers) as p:
        p.map(job, [(k, a.t0, fr) for k in range(n)], chunksize=16)
    wav = os.path.join(OUT, '..', 'song.wav')
    mp4 = os.path.join(OUT, a.name + '.mp4')
    subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-framerate', str(FPS), '-i', os.path.join(fr, '%05d.png'),
                           '-ss', str(a.t0), '-t', str(a.t1 - a.t0), '-i', wav, '-map', '0:v', '-map', '1:a',
                           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow',
                           '-c:a', 'aac', '-b:a', '256k', '-shortest', mp4])
    print('wrote', mp4, n, 'frames')


if __name__ == '__main__':
    main()
