"""First Light, the film: a room with one window, at dawn; everything that moves in it is read from the score.

    python3 firstlight/film.py                     -> out/firstlight_film.mp4 (1920x1080, 30 fps, with the music)
    python3 firstlight/film.py --height 540        -> a quick one at 960x540
    python3 firstlight/film.py --still 87 120      -> frames at those times, out/firstlight_film_87.png, ...
    python3 firstlight/film.py --span 100 140      -> only those seconds (out/firstlight_film_span.mp4)

What drives what (one sound, one visible thing):
  - a red light on a radio mast blinks the hook's rhythm (two short, one long) in every bar, from bar 1
  - each fragment of the hook lights a piece of the city: windows whose heights draw its notes, the crane, the lamps
  - the drums are the wind: with no drums (the intro, the breakdown) the water goes still and mirrors the city;
    with them, ripples, and the sun's glitter spreads into a path
  - the sun's rim clears the hills on the hook's top note, first heard in the first drop; the cat lifts its head
  - the counter-line is a ferry, crossing while it plays, its cabin lights brightening on its notes
  - the camera drifts back from the window over the whole film. The sun through the window lands where it must
    (the room is ray cast; see room.py); in the bridge, the string's answers flare the sun's reflection off the
    water, flickering on the wall
  - the bongos are the gulls' wingbeats; the descant is a plane drawing its contour as a contrail, which drifts
    and dissolves
  - the outro turns the city's lights off in the reverse order they came on; the room's lamp, on from the first
    frame, goes off last, on the final note; the mast light stops blinking and stays on
  - the second drop: a flock lifts off the roofs and wheels away across the sky, its wingbeats the bongos
"""
import argparse
import gc
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import firstlight as fl  # noqa: E402
import room as R  # noqa: E402
import city as CITYMOD  # noqa: E402

ROOT = fl.ROOT
F = fl.FULL
BAR, BEAT, tb = fl.BAR, fl.BEAT, fl.tb
DUR = F.nbars * BAR + fl.TAIL
FPS = 30

# --- the score, as the film reads it --------------------------------------------------------------------------
LEAD = fl.lead_events(F=F)
HOOKN = [e for e in LEAD if e['statement'] is not None]
ANSWERS = [e for e in LEAD if e['statement'] is None]
COUNTER = fl.counter_events(F=F)
DESCANT = fl.descant_events(F=F)
T_TOP = min(e['t'] for e in HOOKN if e['midi'] == max(h[2] for h in fl.HOOK))
BONGO_HITS = sorted(tb(b, k * 0.25) for b in F.bongos for k, _ in fl.BONGO[(b - 1) % 4])
DROP_NOTES = [e for e in HOOKN if e['statement'] in (6, 7, 8, 9)]
BRIDGE = (tb(F.sections['bridge'][0]), tb(F.sections['bridge'][-1] + 1))
LAMP_OFF = None                          # set below: the lamp, the first light on, goes off last, on the final note
DAWN = [(0, 0.0), (tb(9), 0.04), (tb(25), 0.15), (tb(29), 0.2), (tb(33), 0.32), (T_TOP, 0.45), (tb(49), 0.56),
        (tb(65), 0.64), (tb(81), 0.7), (DUR, 0.74)]           # it stays early morning: warm to the end


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


lin = R.lin


def palette(stops, L):
    ks = [k for k, _ in stops]
    cols = np.array([c for _, c in stops], np.float32) ** 2.2
    return np.array([np.interp(L, ks, cols[:, i]) for i in range(3)], np.float32)


SKY_TOP = [(0, (.03, .04, .11)), (.2, (.06, .08, .2)), (.4, (.2, .24, .45)), (.6, (.34, .48, .74)), (1, (.4, .6, .86))]
SKY_HOR = [(0, (.05, .06, .14)), (.2, (.32, .22, .32)), (.4, (.95, .55, .35)), (.6, (1, .76, .56)), (1, (.84, .87, .9))]
HILL_FAR = [(0, (.03, .035, .08)), (.2, (.1, .08, .14)), (.4, (.32, .22, .28)), (.6, (.48, .44, .5)), (1, (.55, .6, .66))]
HILL = [(0, (.015, .02, .04)), (.2, (.05, .045, .08)), (.4, (.16, .12, .16)), (.6, (.26, .27, .29)), (1, (.32, .42, .38))]
CITY = [(0, (.01, .012, .025)), (.2, (.03, .03, .06)), (.4, (.09, .075, .11)), (.6, (.19, .18, .22)), (1, (.36, .37, .42))]
WATER = [(0, (.01, .015, .04)), (.2, (.03, .04, .09)), (.4, (.1, .1, .18)), (.6, (.18, .25, .36)), (1, (.2, .34, .46))]
WARM = lin(1.0, .82, .5)
SUN = lin(1.0, .86, .62)


def dawn(t):
    return float(np.interp(t, [k for k, _ in DAWN], [v for _, v in DAWN]))


def sunlit(t):
    return float(smooth((t - T_TOP + 0.4) / 6))


SUN_R = 20.0
HC = 30.0                                  # the eye's height above the water, metres


def ridge_far(x):
    return 258 + 9 * np.sin(x / 83 + 1) + 6 * np.sin(x / 31) + 2 * np.sin(x / 9.7)


def ridge_near(x):
    return 300 - 34 * np.exp(-((x - 840) / 120) ** 2) - 10 * np.exp(-((x - 700) / 60) ** 2) + 2 * np.sin(x / 13)


def sun_pos(t):
    """The sun's centre in the picture outside: its rim at the ridge on the hook's top note, then rising, slowing."""
    x0 = 700.0
    rim_y = ridge_far(x0) + SUN_R
    dt = t - T_TOP
    rise = 150 * (1 - np.exp(-dt / 55)) if dt > 0 else 9 * dt
    y = rim_y - rise
    return x0 + 0.3 * (rim_y - y), y


def sun_dir(t):
    """The direction toward the sun in the room's world, from where it stands in the picture outside."""
    sx, sy = sun_pos(t)
    u = R.U0 + (sx - OPEN[0]) * R.ENV_K
    v = R.V0 + (sy - OPEN[2]) * R.ENV_K
    d = np.array([(u - R.CX) / R.F, (R.CY - v) / R.F, -1.0])
    return d / np.linalg.norm(d)


# the picture outside, in the 2D layout's coordinates (base units, a frame 540 high): the opening at x 40-920,
# reaching down to y 574 for the room's taller window (more water below the old sill at 410)
OPEN = (40, 920, 24, 574)
YH = 300                                   # the waterline
from city import BUILDINGS, CITY_BASE, windows_of  # noqa: E402
from shapes import seg_mask, poly_mask, ellipse_mask, rect_mask  # noqa: E402
MAST = (132, 177)                          # the radio mast's light, on the far ridge
FRAGMENT_TARGET = {0: 5, 1: 3, 2: 8, 3: 10}      # statement -> building whose windows draw its notes
CRANE_LIGHTS = [(600, y) for y in (288, 276, 264, 252, 240, 228, 214)] + [(x, 197) for x in (614, 634, 654, 674)]
LAMPS_X = np.linspace(52, 556, 40)
LAMPS = None                               # chosen below, where the camera can see them


def seen(ox, oy, margin=0.03, cams=(R.Z_CLOSE,)):
    """Whether a point of the picture outside is visible through the glass from the camera(s): not behind the
    frame or the mullion."""
    u = R.U0 + (ox - OPEN[0]) * R.ENV_K
    v = R.V0 + (oy - OPEN[2]) * R.ENV_K
    dx, dy = (u - R.CX) / R.F, (R.CY - v) / R.F
    for zc in cams:
        tg = zc + R.DEPTH
        gx, gy = dx * tg, R.EYE_Y + dy * tg
        W = R.WIN
        if not (W[0] + R.FRAME + margin < gx < W[1] - R.FRAME - margin and W[2] + R.FRAME + margin < gy < W[3] - R.FRAME - margin):
            return False
        if R.MULL[0] - margin < gx < R.MULL[1] + margin or R.TRANSOM[0] - margin < gy < R.TRANSOM[1] + margin:
            return False
    return True


def lights_from_score():
    """Every light the hook's fragments turn on, in order: (x, y, kind, t_on, vel); and when each goes off."""
    lights, used = [], set()
    for si, (start, bars, vel) in enumerate(F.statements[:6]):
        notes = [e for e in HOOKN if e['statement'] == si]
        t0, span = tb(start), len(bars) * BAR
        for k, e in enumerate(notes):
            if si in FRAGMENT_TARGET:
                cols, rows = windows_of(FRAGMENT_TARGET[si])
                r = rows[int(round((e['midi'] - 68) / 16 * (len(rows) - 1)))]
                vis = [c for c in cols if seen(c, r)]                # never behind the mullion or the frame
                want = vis[min(len(vis) - 1, int((e['t'] - t0) / span * len(vis)))]
                free = [c for c in vis if (c, r) not in used] or vis  # and never a window already lit
                c = min(free, key=lambda c: abs(c - want))
                used.add((c, r))
                lights.append((c, r, 'win', e['t'], e['vel']))
            elif si == 4:
                lights.append(CRANE_LIGHTS[k] + ('crane', e['t'], e['vel']))
            else:
                lights.append(LAMPS[k] + ('lamp', e['t'], e['vel']))
    off_notes = [e['t'] for e in HOOKN if e['statement'] >= 10]
    t_off = [np.inf] * len(lights)
    order = list(range(len(lights)))[::-1]                      # the newest goes off first, the first window last
    groups = [order[round(k * len(order) / len(off_notes)):round((k + 1) * len(order) / len(off_notes))]
              for k in range(len(off_notes))]
    vis_end = [seen(x, y, cams=(R.Z_MID,)) for x, y, *_ in lights]
    for k, g in enumerate(groups):            # every note must visibly turn something off: trade with a neighbour
        if not any(vis_end[i] for i in g):
            for nb in (k + 1, k - 1, k + 2, k - 2):
                if 0 <= nb < len(groups) and sum(vis_end[i] for i in groups[nb]) >= 2:
                    give = next(i for i in groups[nb] if vis_end[i])
                    groups[nb][groups[nb].index(give)], g[0] = g[0], give
                    break
    for k, t in enumerate(off_notes):
        for i in groups[k]:
            t_off[i] = t
    return lights, t_off


_lx = [x for x in LAMPS_X if seen(x, 287)]
LAMPS = [(_lx[int(round(k))], 287) for k in np.linspace(0, len(_lx) - 1, 14)]
LIGHTS, LIGHTS_OFF = lights_from_score()
LAMP_OFF = max(e['t'] for e in HOOKN if e['statement'] >= 10)


def env_of(times, t, tau, vel=None):
    v = 0.0
    for i, t0 in enumerate(times):
        if t0 <= t < t0 + 6 * tau:
            v += (1 if vel is None else vel[i]) * np.exp(-(t - t0) / tau)
    return v


def mast_blink(t):
    """The hook's rhythm, two short and one long, every bar until the last one; then held on."""
    if t >= tb(F.nbars):
        return 1.0
    q = (t % BAR) / BEAT
    v = 0.0
    for q0, d in ((0, 0.35), (0.75, 0.35), (1.5, 1.8)):
        if q0 <= q < q0 + d + 0.4:
            a = min(1, (q - q0) / 0.05)
            r = 1 if q < q0 + d else np.exp(-(q - q0 - d) / 0.12)
            v = max(v, a * r)
    return v


def cam_z(t):
    """The camera drifts back from the window over the whole film, so slowly it is felt more than seen."""
    return R.Z_CLOSE + (R.Z_MID - R.Z_CLOSE) * smooth(t / DUR)


def swell(times, t, vel, rise=0.45, fall=1.4):
    """An envelope that swells in over rise seconds after each onset and fades over fall: light, not a switch."""
    v = 0.0
    for t0, w in zip(times, vel):
        if t0 <= t < t0 + 6 * fall:
            v += w * (1 - np.exp(-(t - t0) / rise)) * np.exp(-(t - t0) / fall)
    return v


def lamp_on(t):
    return float(np.clip(1 - (t - LAMP_OFF) / 0.12, 0, 1))


def cat_lift(t):
    """The cat lifts its head as the sun clears the hills, looks for two bars, and settles back."""
    return float(smooth((t - T_TOP + 0.3) / 1.6) * (1 - smooth((t - T_TOP - 2 * BAR) / 3.0)))


R.CATMOD.lift = cat_lift
NFR = int(np.ceil(DUR * FPS))
TF = np.arange(NFR) / FPS


def _wind():
    bars = (TF // BAR).astype(int) + 1
    raw = np.array([(0.5 if b in F.sections['bridge'] else 1.0) if b in F.kick else 0.06 for b in bars])
    out, v, a = np.zeros(NFR), raw[0], 1 - np.exp(-1 / (FPS * 1.4))
    for i, r in enumerate(raw):
        v += a * (r - v)
        out[i] = v
    return out


WIND = _wind()
FERRY = [(COUNTER[:31], 30, 940), ([e for e in COUNTER if e['bar'] >= 65], 940, 30)]


# --- drawing helpers (base units; C.S px per unit) --------------------------------------------------------------
class Canvas:
    def __init__(self, scale, base_w, base_h):
        self.S = scale
        self.W, self.H = int(round(base_w * scale)), int(round(base_h * scale))
        self.x = ((np.arange(self.W) + 0.5) / scale).astype(np.float32)
        self.y = ((np.arange(self.H) + 0.5) / scale).astype(np.float32)
        self.px = 1 / scale

    def ix(self, v):
        return int(round(v * self.S))


def glow(img, S, x, y, col, amp, r):
    """A soft light at (x, y) base units, radius r, added into img (S px per unit)."""
    R_ = max(2, int(3 * r * S) + 1)
    cx, cy = x * S, y * S
    x0, x1 = max(0, int(cx) - R_), min(img.shape[1], int(cx) + R_ + 1)
    y0, y1 = max(0, int(cy) - R_), min(img.shape[0], int(cy) + R_ + 1)
    if x0 >= x1 or y0 >= y1 or amp <= 0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    g = np.exp(-((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2) / (2 * (r * S) ** 2)).astype(np.float32)
    img[y0:y1, x0:x1] += (amp * g)[..., None] * col


def over(img, S, box, mask_fn, col):
    """Composite colour col through a mask computed only inside box (base units: x0, y0, x1, y1)."""
    x0, y0, x1, y1 = (int(np.floor(box[0] * S)), int(np.floor(box[1] * S)), int(np.ceil(box[2] * S)),
                      int(np.ceil(box[3] * S)))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img.shape[1], x1), min(img.shape[0], y1)
    if x0 >= x1 or y0 >= y1:
        return
    gx, gy = np.meshgrid((np.arange(x0, x1) + 0.5) / S, (np.arange(y0, y1) + 0.5) / S)
    m = mask_fn(gx, gy)
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - m[..., None]) + m[..., None] * col


# --- the picture outside ----------------------------------------------------------------------------------------
class Outside:
    """Its static layers, built once."""

    def __init__(self, C):
        self.C = C
        px = C.px
        ox0, ox1, oy0, oy1 = OPEN
        self.ox, self.oy = slice(C.ix(ox0), C.ix(ox1)), slice(C.ix(oy0), C.ix(oy1))
        X, Y = np.meshgrid(C.x[self.ox], C.y[self.oy])
        self.X, self.Y = X, Y
        self.yh_px = C.ix(YH) - C.ix(oy0)
        self.far = np.clip((Y - ridge_far(X)) / px + 0.5, 0, 1) * (Y < YH)
        self.near = np.clip((Y - ridge_near(X)) / px + 0.5, 0, 1) * (Y < YH) * (X > 600)
        self.cityobj = CITYMOD.City(X, Y, px)                 # the buildings: city.py
        cov = getattr(self.cityobj, 'coverage', None)         # how much of each pixel the city covers
        if cov is None:
            cov = np.zeros_like(X)
            P_ = getattr(self.cityobj, 'P', None)
            if P_ is not None and hasattr(P_, 'A'):
                cov[P_.sl] = P_.A
        self.city_cover = cov.astype(np.float32)
        crane = np.zeros_like(X)
        for p0, p1, w in [((600, 296), (600, 182), 2.2), ((570, 198), (692, 198), 2.0), ((600, 182), (692, 197), 0.8),
                          ((600, 182), (570, 197), 0.8), ((586, 190), (596, 190), 4), ((572, 200), (580, 200), 5)]:
            crane = np.maximum(crane, seg_mask(X, Y, p0, p1, w, px))
        for x, y in LAMPS:
            crane = np.maximum(crane, seg_mask(X, Y, (x, y), (x, 294), 0.9, px))
        mx = MAST[0]
        mast = seg_mask(X, Y, (mx, ridge_far(mx)), (mx, 178), 1.6, px)
        for k in range(1, 4):
            mast = np.maximum(mast, seg_mask(X, Y, (mx - 4 + k, ridge_far(mx)), (mx, 178 + 20 * k), 0.6, px))
        self.mast_top = MAST
        self.crane, self.mast = crane, mast
        rng = np.random.default_rng(7)
        n = 160
        self.stars = (rng.uniform(ox0, ox1, n), rng.uniform(oy0, 245, n), rng.uniform(0.2, 1, n) ** 2,
                      rng.uniform(0.5, 3, n), rng.uniform(0, 6.3, n))
        self.flock = [(rng.normal(0, 1), rng.normal(0, 1), rng.uniform(0, 6.3), rng.uniform(2.6, 4.2)) for _ in range(28)]
        self.clouds = [(rng.uniform(0, 1100), rng.uniform(55, 175), rng.uniform(0.7, 1.4), rng.uniform(1.2, 2.6),
                        [(rng.uniform(-40, 40), rng.uniform(-6, 6), rng.uniform(18, 34), rng.uniform(6, 11))
                         for _ in range(6)]) for _ in range(7)]
        # the water's waves, in metres-ish on a ground plane: direction, wavenumber, amplitude, speed
        # the water's waves on its plane (metres): wavevector, steepness (slope amplitude), deep-water frequency
        self.waves = []
        for k in np.geomspace(0.8, 9, 14):
            a = rng.normal(np.pi / 2, 1.2)                     # running every way, most toward the shore
            self.waves.append(((k * np.cos(a), k * np.sin(a)), 0.03, np.sqrt(9.8 * k), rng.uniform(0, 6.3)))


def outside(C, Z, t, i):
    """The view through the window at time t: the sky, the city and its lights, the water, the ferry, birds, plane."""
    S, px = C.S, C.px
    L, sl = dawn(t), sunlit(t)
    wind = WIND[i]
    ext = np.zeros((C.H, C.W, 3), np.float32)
    X, Y = Z.X, Z.Y
    E = ext[Z.oy, Z.ox]

    top, hor = palette(SKY_TOP, L), palette(SKY_HOR, L)
    f = np.clip((Y - OPEN[2]) / (YH - OPEN[2]), 0, 1) ** 1.6
    E[:] = top * (1 - f[..., None]) + hor * f[..., None]
    sx, sy = sun_pos(t)
    d = np.hypot(X - sx, (Y - sy) * 1.4)
    gl = 1.1 * smooth((L - 0.08) / 0.4) * (1 - 0.2 * smooth((L - 0.55) / 0.45)) + 0.5 * sunlit(t)
    E += (gl * (0.5 * np.exp(-d / 70) + 0.35 * np.exp(-d / 240)))[..., None] * lin(1, .6, .35) * (Y < YH)[..., None]
    if sy < YH + SUN_R:
        disc = ellipse_mask(X, Y, sx, sy, SUN_R, SUN_R, px) * (Y < YH)
        E[:] = E * (1 - disc[..., None]) + disc[..., None] * SUN * 5.0
    # the sun's bloom: the brightest thing in the film glares a little
    E += (sunlit(t) * (0.9 * np.exp(-np.hypot(X - sx, Y - sy) / 26) + 0.3 * np.exp(-np.hypot(X - sx, Y - sy) / 80)))[..., None] * SUN * (Y < YH + 30)[..., None]
    xs, ys, b, w, ph = Z.stars
    vis = max(0.0, 1 - L / 0.2)
    if vis > 0:
        v = b * vis * (0.75 + 0.25 * np.sin(w * t + ph)) * 0.8 * smooth((250 - ys) / 90)  # the city's glow drowns the low ones
        ext[(ys * S).astype(int), (xs * S).astype(int)] += v[:, None] * lin(.9, .92, 1)
    # clouds, drifting: lit underneath at dawn, white by day
    ca = 0.55 * smooth((L - 0.12) / 0.3)
    if ca > 0:
        lit = palette(SKY_HOR, L) * (0.75 + 0.6 * sl) + SUN * 0.25 * sl
        shade = top * 0.85 + hor * 0.2
        for cx0, cy, sc, speed, blobs in Z.clouds:
            cx = OPEN[0] - 120 + (cx0 + speed * t) % (OPEN[1] - OPEN[0] + 240)
            for bx, by, ra, rb in blobs:
                x, y, a_, b_ = cx + bx * sc, cy + by * sc, ra * sc, rb * sc
                col = shade * 0.6 + lit * 0.4

                def m(gx, gy, x=x, y=y, a_=a_, b_=b_):
                    return ca * 0.55 * np.exp(-(((gx - x) / a_) ** 2 + ((gy - y) / b_) ** 2) * 1.6)
                over(ext, S, (x - 2 * a_, y - 2 * b_, x + 2 * a_, y + 2 * b_), m, col)
                over(ext, S, (x - 1.5 * a_, y - 1.5 * b_ - 2, x + 1.5 * a_, y + 2 * b_),
                     lambda gx, gy, x=x, y=y, a_=a_, b_=b_: ca * 0.35 * np.exp(-(((gx - x) / a_) ** 2 + ((gy - y - 0.3 * b_) / (0.6 * b_)) ** 2) * 2), lit)
    # hills, the mast, the city (in the haze by day), the crane
    haze = 0.42 * smooth((L - 0.3) / 0.45)
    for m, pal, hz in ((Z.far, HILL_FAR, 0), (Z.near, HILL, 0), (Z.mast, HILL, 0)):
        col = palette(pal, L) * (1 - hz) + hor * hz
        E[:] = E * (1 - m[..., None]) + m[..., None] * col
    look = dict(t=t, L=L, sunlit=sl, haze=haze, horizon=hor, sky_top=top, city=palette(CITY, L), sun=SUN,
                sun_xy=sun_pos(t))
    Z.cityobj.draw(E, look)
    m = Z.crane
    col = palette(CITY, L) * (1 - haze) + hor * haze
    E[:] = E * (1 - m[..., None]) + m[..., None] * col
    Z.cityobj.draw_after(E, look)
    # the lights the fragments turn on
    day = 1 - 0.72 * L
    for (x, y, kind, t_on, vel), t_off in zip(LIGHTS, LIGHTS_OFF):
        if t < t_on:
            continue
        a = min(1, (t - t_on) / 0.04) * (1 + 1.6 * np.exp(-(t - t_on) / 0.35))
        if t > t_off:
            a *= np.exp(-(t - t_off) / 0.5)
        a *= day * (0.75 + 0.25 * vel)
        if a < 0.003:
            continue
        if kind == 'win':
            sl_ = (slice(C.ix(y - 2), C.ix(y + 2)), slice(C.ix(x - 1.3), C.ix(x + 1.3)))
            ext[sl_] += a * 1.1 * WARM
            glow(ext, S, x, y, WARM, 0.12 * a, 4)
        elif kind == 'crane':
            glow(ext, S, x, y, lin(1, .3, .25), 0.9 * a, 0.9)
            glow(ext, S, x, y, lin(1, .3, .25), 0.15 * a, 4)
        else:
            glow(ext, S, x, y, lin(1, .65, .3), 0.8 * a, 0.8)
            glow(ext, S, x, y, lin(1, .65, .3), 0.06 * a, 3.5)
    mb = mast_blink(t)
    glow(ext, S, *Z.mast_top, lin(1, .12, .08), 1.4 * mb, 1.1)
    glow(ext, S, *Z.mast_top, lin(1, .12, .08), 0.25 * mb, 6)

    # the gulls (from the first drop's second half): each bongo hit a wingbeat
    if t > tb(41):
        flap = env_of(BONGO_HITS, t, 0.09)
        for k in range(4):
            x = OPEN[0] + ((140 + 230 * k + (14 + 3 * k) * (t - tb(41))) % (OPEN[1] - OPEN[0] + 60)) - 30
            y = 150 + 32 * k % 70 + 6 * np.sin(0.5 * t + k)
            s = 5 + k % 2
            lift = np.clip(0.25 + 0.9 * flap * (1 if k % 2 == 0 else 0.7), 0, 1.2)

            def gull(gx, gy, x=x, y=y, s=s, lift=lift):
                m = np.zeros_like(gx)
                for sgn in (-1, 1):
                    el = (x + sgn * s * 0.5, y - s * lift * 0.6)
                    m = np.maximum(m, seg_mask(gx, gy, (x, y), el, 0.9, px))
                    m = np.maximum(m, seg_mask(gx, gy, el, (x + sgn * s, y - s * (lift - 0.4) * 0.5), 0.8, px))
                return m * smooth((t - tb(41)) / 4)
            over(ext, S, (x - 12, y - 12, x + 12, y + 8), gull, palette(HILL, L) * 0.8)

    # the second drop: a flock lifts off the roofs and wheels away over the city, its wingbeats the bongos
    tau = t - tb(65)
    if -0.2 < tau < 18:
        flap_all = env_of(BONGO_HITS, t, 0.09)
        rise = smooth(tau / 1.2)
        cx = 455 - 14 * tau - 30 * np.sin(0.35 * tau)
        cy = 236 - 70 * (1 - np.exp(-max(tau, 0) / 2.2)) - 5.5 * max(tau, 0) + 9 * np.sin(0.6 * tau)
        spread = 4 + 26 * smooth(tau / 3)
        for k, (ox, oy, ph, sz) in enumerate(Z.flock):
            a_ = 0.4 * tau * (0.6 + 0.4 * (k % 3)) + ph
            x = cx + spread * (ox * np.cos(a_) - 0.4 * oy * np.sin(a_))
            y = cy + 0.5 * spread * (oy * np.cos(a_) + 0.4 * ox * np.sin(a_)) + (1 - rise) * 6 * ox
            if not (OPEN[0] - 10 < x < OPEN[1] + 10 and OPEN[2] - 10 < y < YH):
                continue
            lift = np.clip(0.3 + 0.8 * abs(np.sin(9 * t + ph)) * (0.5 + flap_all), 0, 1.3)

            def bird(gx, gy, x=x, y=y, s=sz, lift=lift):
                m = np.zeros_like(gx)
                for sgn in (-1, 1):
                    el = (x + sgn * s * 0.5, y - s * lift * 0.6)
                    m = np.maximum(m, seg_mask(gx, gy, (x, y), el, 0.8, px))
                    m = np.maximum(m, seg_mask(gx, gy, el, (x + sgn * s, y - s * (lift - 0.4) * 0.5), 0.7, px))
                return m * smooth((tau + 0.2) / 0.6) * (1 - smooth((tau - 15) / 3))
            over(ext, S, (x - 8, y - 8, x + 8, y + 6), bird, palette(HILL, L) * 0.7)

    # the plane, drawing the descant's contour as a contrail that drifts, spreads and dissolves
    t0, t1 = DESCANT[0]['t'], DESCANT[-1]['t'] + DESCANT[-1]['dur']
    if t > t0:
        def plane(tt):
            m = DESCANT[0]['midi']
            for e in DESCANT:
                if e['t'] <= tt:
                    m = m + (e['midi'] - m) * smooth((tt - e['t']) / 3.5)      # a plane glides; it doesn't turn corners
            return OPEN[0] + 20 + (OPEN[1] - OPEN[0]) * (tt - t0) / (t1 - t0) * 0.98, 150 - (m - 77) * 7
        ts = np.arange(t0, min(t, t1 + 8), 0.25)
        if len(ts) > 1:
            pts = [plane(tt) for tt in ts] + [plane(min(t, t1 + 8))]
            bx0, bx1 = C.ix(OPEN[0]), C.ix(OPEN[1])
            by0, by1 = C.ix(40), C.ix(200)
            gx, gy = np.meshgrid(C.x[bx0:bx1], C.y[by0:by1])
            m = np.zeros_like(gx)
            for k in range(len(pts) - 1):
                age = t - ts[k]
                a = 0.7 * np.exp(-age / 13) * min(1, age / 0.6 + 0.3)
                if a < 0.01:
                    continue
                wd = 0.9 + 0.22 * age
                (xa, ya), (xb, yb) = pts[k], pts[k + 1]
                xa, xb = xa - 0.35 * age, xb - 0.35 * (age - 0.25)
                ya, yb = ya + 0.08 * age, yb + 0.08 * age
                lo, hi = C.ix(min(xa, xb) - wd - 2) - bx0, C.ix(max(xa, xb) + wd + 2) - bx0
                if hi <= 0 or lo >= gx.shape[1]:
                    continue
                lo, hi = max(lo, 0), min(hi, gx.shape[1])
                m[:, lo:hi] = np.maximum(m[:, lo:hi], a * np.clip(1 - np.hypot(
                    *(lambda u: (gx[:, lo:hi] - xa - u * (xb - xa), gy[:, lo:hi] - ya - u * (yb - ya)))(
                        np.clip(((gx[:, lo:hi] - xa) * (xb - xa) + (gy[:, lo:hi] - ya) * (yb - ya))
                                / ((xb - xa) ** 2 + (yb - ya) ** 2 + 1e-9), 0, 1)))
                    / (wd / 2 + px), 0, 1) ** 1.5)
            ext[by0:by1, bx0:bx1] += m[..., None] * (lin(1, 1, 1) * 0.5 - ext[by0:by1, bx0:bx1] * 0.4)
            if t < t1 + 8:
                glow(ext, S, *pts[-1], lin(1, .95, .9), 0.9, 1.0)

    # the water: the reflection of everything above, rippled by the wind (the drums)
    E = ext[Z.oy, Z.ox]
    yh = Z.yh_px
    rows = np.arange(yh, E.shape[0])
    dd = ((rows - yh) / max(1, E.shape[0] - yh)).astype(np.float32)[:, None]
    ph = 18 * np.log1p((rows - yh) / (2 * S))[:, None]
    xx = (np.arange(E.shape[1])[None, :] + 0.5) / S
    amp = (0.12 + 1.1 * wind) * (0.4 + 5.0 * dd)
    shift = amp * (np.sin(ph + 0.045 * xx + 1.4 * t) + 0.6 * np.sin(1.7 * ph - 0.027 * xx - 2.2 * t)) * S
    # the waves also tilt the reflection up and down, differently along each row: that is what breaks a
    # horizontal line (a roofline, the contrail) into a wavering one
    vy = (np.sin(1.3 * ph + 0.9 * t) + 0.8 * np.sin(0.9 * ph + 0.11 * xx - 1.6 * t)
          + 0.5 * np.sin(1.9 * ph - 0.23 * xx + 2.3 * t))
    src_y = np.clip(2 * yh - 1 - rows[:, None] + amp * 0.6 * S * vy, 0, yh - 1).astype(int)
    src_x = np.clip(np.arange(E.shape[1])[None, :] + shift, 0, E.shape[1] - 1).astype(int)
    refl = E[src_y, src_x]
    land = np.maximum(np.maximum(Z.far, Z.near), Z.city_cover)
    land_seen = land[src_y, src_x]                 # where the still water mirrors land, not sky
    wcol = palette(WATER, L)
    E[yh:] = refl * (0.68 - 0.3 * dd[..., None]) + wcol * (0.35 + 0.5 * dd[..., None])
    if sl > 0:
        # glints: the sun mirrored by the waves. Each sample of the water is a point on a plane HC below the eye,
        # where the ray through it lands; the waves' slopes there tilt its normal, and it glints where the view ray,
        # reflected, points at the sun. Waves too fine for a sample's footprint are averaged away, and their slope
        # widens the highlight instead, so far water glitters as a band and near water as single sparks. Calm
        # water (no drums) mirrors the sun in one place; wind spreads it into a path. The hook's notes in the
        # drops make it brighter.
        lenv = min(1.5, env_of([e['t'] for e in DROP_NOTES], t, 0.6, [e['vel'] for e in DROP_NOTES]))
        oy_ = (rows[:, None] + 0.5) / S + OPEN[2]
        u = R.U0 + xx * R.ENV_K                                       # (xx already counts from the opening's edge)
        v = R.V0 + (oy_ - OPEN[2]) * R.ENV_K
        dx_, dy_ = (u - R.CX) / R.F, (R.CY - v) / R.F
        dy_ = np.minimum(dy_, -1e-4)
        lam = HC / -dy_
        wx, wz = lam * dx_, -lam
        step = R.ENV_K / S / R.F                                     # a sample's angular size
        fx_, fz_ = lam * step, lam * lam / HC * step                 # its footprint across and along
        hx = np.zeros(np.broadcast(wx, wz).shape, np.float32)
        hz = np.zeros_like(hx)
        lost = np.zeros_like(hx)
        a_w = 0.3 + 0.9 * wind
        for (kx, kz), steep, om, ph0 in Z.waves:
            att = np.exp(-0.5 * ((kx * fx_) ** 2 + (kz * fz_) ** 2))
            c = np.cos(kx * wx + kz * wz - om * t + ph0) * steep * a_w * att
            hx += c * kx / np.hypot(kx, kz)
            hz += c * kz / np.hypot(kx, kz)
            lost += 0.5 * (steep * a_w) ** 2 * (1 - att ** 2)
        # the tilt that would mirror the sun into the eye: the surface normal halfway between the sun and the eye
        dn = np.sqrt(dx_ * dx_ + dy_ * dy_ + 1)
        sd = sun_dir(t)
        hx_, hy_, hz_ = sd[0] - dx_ / dn, sd[1] - dy_ / dn, sd[2] + 1 / dn
        need_x, need_z = -hx_ / hy_, -hz_ / hy_                  # as slopes of the water (dh/dx, dh/dz)
        s0 = 0.006                                                # the sun's own size, as a slope
        sig2 = s0 ** 2 + 0.35 * lost                              # plus the waves too fine to see, in slope space
        g = np.exp(-((hx - need_x) ** 2 + (hz - need_z) ** 2) / (2 * sig2)) * (s0 ** 2 / sig2)
        g *= 1 - land_seen                         # the land between hides the sun from that water: no glints there
        E[yh:] += np.minimum(g * sl * 260 * (0.9 + 1.0 * lenv), 2.5)[..., None] * SUN      # the sun is far brighter than the sky

    # the ferry: the counter-line; a steady crossing, its cabin lights brightening on its notes
    for notes, x0, x1 in FERRY:
        ta, tz = notes[0]['t'] - 1, notes[-1]['t'] + notes[-1]['dur'] + 2
        if not notes or not (ta <= t <= tz):
            continue
        p = (t - ta) / (tz - ta)
        fx, fy = x0 + (x1 - x0) * p, 352
        lit = 0.6 + 0.8 * min(1, env_of([e['t'] for e in notes], t, 0.4))
        sgn = 1 if x1 > x0 else -1
        body = lambda gx, gy, fx=fx, fy=fy, sgn=sgn: np.maximum(np.maximum(
            poly_mask(gx, gy, [(fx - 24, fy - 4), (fx + 24, fy - 4), (fx + 20, fy + 2), (fx - 20, fy + 2)], px),
            rect_mask(gx, gy, fx - 13 - 3 * sgn, fx + 11 - 3 * sgn, fy - 10, fy - 4, px)),
            rect_mask(gx, gy, fx - 3 * sgn - 1.5, fx - 3 * sgn + 1.5, fy - 15, fy - 10, px))
        over(ext, S, (fx - 30, fy - 16, fx + 30, fy + 6), body, palette(CITY, L) * 0.9)
        for k in range(5):
            wx = fx - 10 - 3 * sgn + 4.6 * k
            sl_ = (slice(C.ix(fy - 8.5), C.ix(fy - 6)), slice(C.ix(wx - 1), C.ix(wx + 1)))
            ext[sl_] += lit * (1 - 0.6 * L) * WARM
        glow(ext, S, fx, fy + 4, WARM, 0.05 * lit * (1 - L), 6)
    return ext


# --- the room, and the frame ------------------------------------------------------------------------------------
def caustic(px, py, pz, t):
    """The sun's reflection off rippling water: a net of bright lines that wobbles."""
    wz = pz + 0.06 * np.sin(7 * py + 1.3 * t) + 0.03 * np.sin(17 * pz - 2.1 * t)
    wy = py + 0.05 * np.sin(9 * pz - 1.1 * t) + 0.025 * np.sin(23 * py + 1.7 * t)
    c = 0
    for fz, fy, w, amp in ((23, 9, 1.4, 1.0), (-14, 21, -1.1, 0.8), (37, -27, 2.3, 0.5)):
        c = c + amp * (1 - np.abs(np.sin(fz * wz + fy * wy + w * t))) ** 4       # soft, wide threads: water light
    return c * 0.8 + 0.12


Z_STEP = 4e-4
SUN_IN = 2.0                                # direct sun in the room, against the sky's light
CAT_SS = 4                                  # samples a pixel, each way, in the cat's box


class Film:
    def __init__(self, height, ss):
        self.S = height / 540
        self.H, self.W = height, int(round(960 * self.S))
        self.ss = ss
        self.C = Canvas(self.S * R.ENV_K * 1.25, 960, 600)
        self.Z = Outside(self.C)
        self.cache_z = None
        Y, X = np.mgrid[0:self.H, 0:self.W]
        r = np.hypot((X - self.W / 2) / self.W, (Y - self.H / 2) / self.H)
        self.vig = (1 - 0.3 * r ** 2.2)[..., None].astype(np.float32)

    def geometry(self, zc, t):
        """The rays' hits (the static room) for camera z; cached while the camera stands still."""
        if self.cache_z is None or abs(zc - self.cache_z) > 1e-7:
            if self.cache_z is not None:                 # the old hits hold a closure on themselves: free them now
                self.G.shadow = None
                del self.G
                gc.collect()
            h, w = self.H * self.ss, self.W * self.ss
            G = R.cast(zc, t, h, w, self.S * self.ss, dynamic=False)
            G.n = list(R.MATS.bump(G.m, G.p, G.n))      # the surfaces' relief, once per camera position
            G.A = R.albedo(G, t)
            G.sky = R.sky_factor_coarse(G.p, G.n, self.ss)
            G.lamp = R.lamp_factor(G.p, G.n)
            self.G, self.cache_z = G, zc
        return self.G

    def frame(self, i):
        t = TF[i]
        zc = round(cam_z(t) / Z_STEP) * Z_STEP          # recast every 0.4 mm of the drift: steps of < 0.1 px
        G = self.geometry(zc, t)
        H = R.with_cat(G, t, self.S * self.ss)          # the cat breathes: re-cast where it is
        self._ext = None
        col = self.shade(H, t, i)
        # steam from the cup, in the second drop
        sky = (palette(SKY_TOP, dawn(t)) + palette(SKY_HOR, dawn(t))) * 0.5
        st = smooth((t - tb(65)) / (2 * BAR)) - smooth((t - tb(84)) / (2 * BAR))
        if st > 0:
            sc = self.S * self.ss
            for k in range(4):
                for j in range(26):
                    f_ = j / 26 + ((t * 0.07 + k / 4) % 1) / 26
                    hgt = f_ * 0.17
                    sway = hgt * (0.25 + 0.6 * f_)
                    x = R.CUP_C[0] + sway * np.sin(9 * hgt + 1.3 * t + 2.4 * k) + 0.012 * (k - 1.5)
                    z = R.CUP_C[1] + sway * 0.6 * np.cos(7 * hgt + 0.9 * t + k)
                    y = R.WIN[2] + 0.1 + hgt
                    dz_ = zc - z
                    u, v = R.CX + R.F * x / dz_, R.CY - R.F * (y - R.EYE_Y) / dz_
                    rr = R.F * (0.01 + hgt * 0.12) / dz_
                    glow(col, sc, u, v, sky * 1.2 + 0.03, 0.022 * st * (1 - f_) ** 1.5, rr)
        ss = self.ss
        out = col.reshape(self.H, ss, self.W, ss, 3).mean(axis=(1, 3))
        # the cat and the plant again, at CAT_SS x CAT_SS samples a pixel: their rim light, fur and leaf edges are
        # finer than 2 x 2
        boxes = [R.cat_box(t, zc, self.H, self.W, self.S)]
        for nm in ('PLANT', 'PHONESP'):                 # and the headphones: their cable is ~2 px wide
            if hasattr(R.PROPS, nm):
                boxes.append(R.sphere_box(*getattr(R.PROPS, nm).bound(), zc, self.H, self.W, self.S))
        for box in boxes:
            if box is None or CAT_SS <= ss:
                continue
            k = CAT_SS
            (ys, xs) = box
            Hc = R.cast(zc, t, (ys.stop - ys.start) * k, (xs.stop - xs.start) * k, self.S * k, dynamic=True,
                        org=(ys.start * k, xs.start * k))
            Hc.n = list(R.MATS.bump(Hc.m, Hc.p, Hc.n))
            Hc.A = R.albedo(Hc, t)
            Hc.sky = R.sky_factor_coarse(Hc.p, Hc.n, k)
            Hc.lamp = R.lamp_factor(Hc.p, Hc.n)
            R.attach_shadow(Hc, t)
            cc = self.shade(Hc, t, i)
            out[box] = cc.reshape(ys.stop - ys.start, k, xs.stop - xs.start, k, 3).mean(axis=(1, 3))
            Hc.shadow = None
        # the finish: fade in and out, a vignette, a little grain
        out = out / (1 + out / 6)
        fade = smooth(t / 2.5) * smooth((DUR - t) / 4.5)
        out *= fade * self.vig
        out = np.clip(out, 0, 1) ** (1 / 2.2)
        out += self.rng.normal(0, 0.006, out.shape[:2]).astype(np.float32)[..., None]
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)

    def shade(self, H, t, i):
        """The colour of every sample of a set of hits: the room's light, the props' glow, the view outside."""
        L, sl = dawn(t), sunlit(t)
        s = sun_dir(t)
        sr = (s[0], -s[1], s[2])

        # light: the sky through the window, the bounce, the lamp, the sun, and the sun off the water
        sky = (palette(SKY_TOP, L) + palette(SKY_HOR, L)) * 0.5
        lamp = lamp_on(t)
        E = (H.sky[..., None] * sky * 3.2 + (sky * 0.1 + lin(1, .8, .62) * 0.09 * sl)                 # bounce: warm once the sun is in
             + H.lamp[..., None] * lin(1, .78, .5) * 0.13 * lamp + lin(1, .8, .6) * 0.02 * lamp)
        if sl > 0:
            ns = np.clip(H.n[0] * s[0] + H.n[1] * s[1] + H.n[2] * s[2], 0, 1)
            ap = R.aperture(H.p, s, H.n)
            vis = ap * ns
            unshaded = 1 - H.shadow(s, ap)
            vis = vis * unshaded
            E += (vis * sl * SUN_IN)[..., None] * SUN
            # rim light: the sun is in front of the camera, so the things on the seat are backlit; their edges,
            # turned away from us and toward the window, catch it
            obj = np.isin(H.m, (R.CAT, R.CUP, R.POT, R.LEAF) + tuple(getattr(R.PROPS, n) for n in ('PHONES', 'BOOK', 'TOMES') if hasattr(R.PROPS, n)))
            if obj.any():
                dn = np.sqrt(H.d[0] ** 2 + H.d[1] ** 2 + 1)
                facing = np.clip(-(H.n[0] * H.d[0] + H.n[1] * H.d[1] + H.n[2] * H.d[2]) / dn, 0, 1)
                rim = obj * ap * unshaded * (1 - facing) ** 3 * np.clip(ns + 0.35, 0, 1)
                E += (rim * sl * 2.2)[..., None] * SUN
            # light through thin things: a leaf, the lamp's shade, the cat's ears glow on the side facing us when
            # the sun is behind them
            thin = np.isin(H.m, (R.LEAF, R.SHADE, R.CAT))
            if thin.any():
                idx = np.nonzero(thin)
                tr = np.zeros(len(idx[0]), np.float32)
                mm = H.m[idx]
                tr[mm == R.LEAF] = 0.45
                tr[mm == R.SHADE] = 0.3
                cs_ = mm == R.CAT
                if cs_.any():
                    tr[cs_] = R.CATMOD.translucency(tuple(a[idx][cs_] for a in H.p), tuple(a[idx][cs_] for a in H.n), t)
                pn = tuple(a[idx] for a in H.p)
                nb = tuple(-a[idx] for a in H.n)
                back = np.clip(nb[0] * s[0] + nb[1] * s[1] + nb[2] * s[2], 0, 1)
                through = tr * back * R.aperture(pn, s, nb) * unshaded[idx]
                E[idx] += (through * sl * 1.3)[:, None] * SUN
            nr = np.clip(H.n[0] * sr[0] + H.n[1] * sr[1] + H.n[2] * sr[2], 0, 1)
            if s[1] > 0.02:
                ans = min(1.5, swell([e['t'] for e in ANSWERS], t, [e['vel'] * 2 for e in ANSWERS]))
                shim = R.aperture(H.p, sr, H.n) * nr * caustic(*H.p, t) * (0.3 + 0.7 * ans) * smooth(WIND[i] + 0.3)   # always there while the water moves; the answers make it flare
                E += (shim * sl)[..., None] * SUN
        col = H.A * E
        for pr in R.PROPS.PROPS:                       # what the props give off themselves (the lamp's shade)
            sel = np.isin(H.m, pr.mats)
            if lamp > 0 and sel.any():
                col[sel] += pr.emission(tuple(a[sel] for a in H.p), tuple(a[sel] for a in H.n), H.m[sel]) * lamp
        # the view outside, looked up by each ray's direction
        env = H.m == R.ENV
        if env.any():
            if self._ext is None:
                self._ext = outside(self.C, self.Z, t, i)
            dx, dy = H.d[0][env], H.d[1][env]
            ox = OPEN[0] + (R.CX + R.F * dx - R.U0) / R.ENV_K
            oy = OPEN[2] + (R.CY - R.F * dy - R.V0) / R.ENV_K
            col[env] = bilinear(self._ext, ox * self.C.S - 0.5, oy * self.C.S - 0.5) * 0.94
            if lamp > 0:
                col[env] += glass_reflection(H, env, lamp, L, t)
        return col


def glass_reflection(H, env, lamp, L, t):
    """At night the glass mirrors the room: a faint warm wash of the lamp-lit walls, and the props and the cat
    themselves (the lamp's glowing shade above all). A plane mirror shows what a camera mirrored through it
    sees, so each window sample's ray is traced again from the camera's image behind the glass."""
    k = 0.06 * lamp * (1 - smooth(L / 0.35))                   # plain glass reflects ~4-8%; day outshines it
    if k <= 0:
        return 0.0
    idx = np.nonzero(env)
    out = np.broadcast_to(lin(1, .78, .5) * 0.05 * k, (len(idx[0]), 3)).copy()
    om = (0.0, R.EYE_Y, -2 * R.DEPTH - H.zc)                   # the camera's image in the glass
    dx, dy = H.d[0][idx], H.d[1][idx]
    u, v = R.CX + R.F * dx, R.CY - R.F * dy
    dr = (dx, dy, np.ones_like(dx))
    things = [(pr.bound(), pr) for pr in R.PROPS.PROPS] + [(R.CATMOD.bound(t), None)]
    for (c, r), pr in things:
        dist = c[2] - om[2]                                      # seen from the mirrored camera, looking +z
        cu, cv = R.CX + R.F * c[0] / dist, R.CY - R.F * (c[1] - R.EYE_Y) / dist
        rr = R.F * r / max(dist - r, 1e-3) * 1.1 + 2
        sel = (np.abs(u - cu) < rr) & (np.abs(v - cv) < rr)
        if not sel.any():
            continue
        d_ = tuple(a[sel] for a in dr)
        if pr is None:
            tt, nx, ny, nz = R.CATMOD.hit(om, d_, t)
        else:
            tt, nx, ny, nz, mm = pr.hit(om, d_)
        got = np.isfinite(tt)
        if not got.any():
            continue
        q = tuple(o_ + tt[got] * dd[got] for o_, dd in zip(om, d_))
        nn = (nx[got], ny[got], nz[got])
        lit = (R.lamp_factor(q, nn) * 0.13 + 0.02)[:, None] * lin(1, .78, .5) * lamp
        if pr is None:
            glow_ = R.CATMOD.albedo(q, nn, t) * lit
        else:
            glow_ = pr.emission(q, nn, mm[got]) * lamp + pr.albedo(q, nn, mm[got]) * lit
        out[np.nonzero(sel)[0][got]] += glow_ * k
    return out


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
            + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def worker(job):
    height, ss, i0, i1, path = job
    fm = Film(height, ss)
    fm.rng = np.random.default_rng(i0)
    if path.endswith('.png'):
        from PIL import Image
        Image.fromarray(fm.frame(i0)).save(path)
        fm.G.shadow = None                       # the hits hold a closure on themselves: free them before the next
        del fm
        gc.collect()
        return path
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{fm.W}x{fm.H}',
                           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '16',
                           '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
    for i in range(i0, i1):
        ff.stdin.write(fm.frame(i).tobytes())
    ff.stdin.close()
    ff.wait()
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--ss', type=int, default=2, help='samples per pixel, each way')
    ap.add_argument('--still', type=float, nargs='*', help='write frames at these times (s) instead')
    ap.add_argument('--span', type=float, nargs=2, help='render only from..to (s)')
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--outdir', help='where stills go (default out/)')
    a = ap.parse_args()
    out = os.path.join(ROOT, 'out')
    if a.still and a.outdir:
        out = a.outdir
        os.makedirs(out, exist_ok=True)
    if a.still:
        jobs = [(a.height, a.ss, int(round(s * FPS)), 0, os.path.join(out, f'firstlight_film_{s:g}.png'))
                for s in a.still]
        with Pool(min(a.jobs, len(jobs))) as p:
            print('\n'.join(p.map(worker, jobs)))
        sys.exit()
    i0, i1 = (int(a.span[0] * FPS), int(a.span[1] * FPS)) if a.span else (0, NFR)
    tmp = os.path.join(out, 'film_parts')
    os.makedirs(tmp, exist_ok=True)
    cuts = np.linspace(i0, i1, a.jobs + 1).astype(int)
    jobs = [(a.height, a.ss, cuts[k], cuts[k + 1], os.path.join(tmp, f'part{k:02d}.mp4')) for k in range(a.jobs)]
    with Pool(a.jobs) as p:
        parts = p.map(worker, jobs)
    with open(os.path.join(tmp, 'list.txt'), 'w') as f:
        f.writelines(f"file '{q}'\n" for q in parts)
    name = 'firstlight_film' + (f'_{a.height}' if a.height != 1080 else '') + ('_span' if a.span else '')
    dst = os.path.join(out, name + '.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', os.path.join(tmp, 'list.txt'),
                    '-ss', f'{i0 / FPS}', '-i', os.path.join(out, 'firstlight.wav'), '-map', '0:v', '-map', '1:a',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-shortest', dst], check=True)
    for q in parts:
        os.remove(q)
    os.remove(os.path.join(tmp, 'list.txt'))
    os.rmdir(tmp)
    print(f'wrote {dst}')
