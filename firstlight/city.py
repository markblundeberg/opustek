"""The city across the water in First Light's film: the buildings, their windows, the rows behind in the haze.

The film (film.py) asks the city three things:
  City(X, Y, px)       build once: X, Y are the picture's coordinates (2D arrays, base units of the old 2D
                       layout: x 40-920 across the window, y 24-574 down it; the waterline at y = 300), px the
                       size of one picture pixel in those units (masks are antialiased to it)
  draw(E, look)        paint the buildings over the sky and hills already in E (an RGB array, linear, same shape
                       as X), for this moment; look holds: t (s), L (dawn 0 night .. 0.74 morning), sunlit (0-1),
                       haze (0-0.4), horizon and sky_top (sky colours), city (the buildings' base colour for this
                       light), sun (its colour), sun_xy (where the sun is, picture units)
  draw_after(E, look)  anything to lay over the crane and lamp posts (the film draws those in between)
  coverage             (set by City()) a float32 array the shape of X: how much of each pixel the city covers,
                       0-1. The film reads it to keep the sun's glints off water that mirrors land, not sky.
Fixed, because the score's lights live in them: BUILDINGS (each x0, x1, top), CITY_BASE, and windows_of(b), the
grid of window centres of building b. The film lights single windows at those centres (2.6 x 4 units each)
and turns them off again; the mast, the crane and the lamps along the shore are the film's, not the city's.
The sun is behind the city, low, to the right (x about 700-760): the faces toward the camera are in shade.

How it is drawn. Everything is static but the light, so City() paints the whole city once, back to front, into
a few maps over the city's region (its coverage, and what each part of a pixel's colour is made of); draw() only
mixes them with this moment's colours:
    colour = city x CT  -  haze x city x CK  +  haze colour x (H0 + haze x HK)  +  sky_top x RF  +  rim light x SN
  CT, CK   each surface's tint (its material, its shading: floor lines, piers, sills, the glass), less its haze
           ("city" here is the light on the faces toward us: the film's city colour and the sky over the camera,
           cool, and darker than the sky behind them, more so in the morning)
  H0, HK   how much haze stands in front of it: four rows deep (the street of BUILDINGS, then three rows behind,
           each farther and paler), and a morning mist low over the water. The film's haze is taken in part
           (DAY_HAZE, thinner again in the morning: LATE), and thickens toward the sun (SUN_HAZE). Its colour is
           the horizon's at dawn, the sky's own at that height later, warmed where the sun's glow is
  RF       the glass, reflecting the western sky behind the camera
  SN       what the low sun behind the city lights: a warm rim along each shape's edges toward it (right and
           top, never the far side), and the side walls of the buildings left of the view's centre, which face
           it at a graze (in shade below a taller neighbour on the sun's side)
The masks are analytic and antialiased to the pixel (thin lines keep their area below a pixel), farther rows
softer. No window lights itself: every lit window is a note of the score.
"""
import numpy as np
from scipy.ndimage import map_coordinates

from shapes import rect_mask, seg_mask, poly_mask, ellipse_mask

BUILDINGS = [(48, 78, 250), (80, 104, 232), (108, 150, 262), (150, 188, 206), (192, 228, 244), (232, 270, 168),
             (274, 300, 226), (304, 328, 252), (330, 392, 214), (396, 416, 240), (420, 468, 196), (472, 500, 236),
             (504, 540, 256), (544, 566, 270)]
CITY_BASE = 296
YH = 300


def windows_of(b):
    x0, x1, top = BUILDINGS[b]
    return np.arange(x0 + 4, x1 - 3, 6.0), np.arange(CITY_BASE - 8, top + 5, -8.0)


REGION = (36.0, 668.0, 100.0, 302.0)        # x0, x1, y0, y1: everything the city paints lies inside
VP_X = 480.0                                # the view's centre: side walls show on buildings left of it
HAZE_MAX = 0.42                             # the film's haze, at its most
SUN_HAZE = (0.25, 0.9, 170.0)               # more toward the sun: glare (times sunlit), haze (the film's), reach
DAY_HAZE = (0.16, 0.2)                      # the film's haze as the city takes it: in full to 0.16, a fifth above
AMB = (0.85, 0.07, 0.015)                   # the faces' light: the film's city colour, the sky's, the horizon's
# in the morning (from L 0.56, over 0.14): the faces darker by, the rows behind clearer by, the haze thinner by
LATE = (0.56, 0.14, 0.35, 0.45, 0.45)
RIM_SUN, RIM_HAZE = 1.6, 1.3                # the sun's rim, and how much haze dims it
GLOW = np.array((1, .6, .35), np.float32) ** 2.2
# the rows, nearest first: haze h = h0 + haze * k; the rim light's strength; edge softness (pixels)
LAYER_H0 = (0.0, 0.14, 0.32, 0.46)
LAYER_K = (0.7, 0.9, 1.15, 1.2)
LAYER_RIM = (1.0, 0.55, 0.3, 0.15)
LAYER_SOFT = (1.0, 1.25, 1.7, 2.3)
MIST_H0 = (0.05, 0.14, 0.2, 0.22)           # the mist over the water: at the waterline, falling off upward
MIST_K = (0.35, 0.4, 0.4, 0.3)
MIST_SCALE = (9.0, 12.0, 15.0, 18.0)

MAT = {                                     # tints on the film's city colour (which carries the light)
    'brick': (1.32, 0.9, 0.78), 'brick2': (1.22, 0.86, 0.8), 'darkbrick': (1.0, 0.76, 0.72),
    'stone': (1.24, 1.12, 0.98), 'sand': (1.34, 1.1, 0.88), 'terracotta': (1.36, 0.98, 0.82),
    'concrete': (1.06, 1.04, 1.04), 'stucco': (1.3, 1.12, 1.0), 'glass': (0.74, 0.88, 1.08),
    'glass2': (0.86, 0.92, 1.06), 'slate': (0.72, 0.74, 0.84), 'steel': (0.86, 0.86, 0.9),
    'copper': (0.78, 1.08, 0.98), 'leaf': (0.62, 0.9, 0.6), 'leaf2': (0.72, 0.92, 0.56),
    'wood': (1.2, 0.92, 0.75), 'far': (1.08, 1.0, 1.0),
}

AWNINGS = ((1.5, 0.62, 0.52), (0.72, 1.0, 0.78), (1.4, 1.2, 0.9), (0.8, 0.82, 1.15))
# how much each building's sunward edges catch the light: unevenly, so they don't line up into a comb
EDGE = (0.9, 0.55, 0.75, 1.0, 0.5, 0.95, 0.65, 0.55, 0.85, 0.45, 0.8, 0.6, 0.75, 0.55)
# each building of the street its own: material, how its front is built, its tone
STYLE = {
    0: dict(mat='brick', kind='punched', sc=3, cross=True, tone=0.98, bays=(1, 3)),
    1: dict(mat='stone', kind='piers', tone=1.0),
    2: dict(mat='brick2', kind='punched', sc=1, cross=False, tone=0.92),
    3: dict(mat='terracotta', kind='piers', tone=0.95),
    4: dict(mat='concrete', kind='bands', tone=1.02, balconies=True),
    5: dict(mat='stone', kind='piers', tone=1.08),
    6: dict(mat='glass', kind='curtain', tone=1.0),
    7: dict(mat='brick', kind='punched', sc=2, cross=True, tone=1.02, bays=(1,)),
    8: dict(mat='sand', kind='bands', tone=0.97),
    9: dict(mat='darkbrick', kind='punched', sc=1, cross=False, tone=1.0),
    10: dict(mat='glass2', kind='curtain', tone=1.02),
    11: dict(mat='sand', kind='piers', tone=1.04),
    12: dict(mat='brick', kind='punched', sc=2, cross=False, tone=0.94),
    13: dict(mat='stucco', kind='punched', sc=4, cross=True, tone=1.0),
}


def _line(X, Y, p0, p1, w, px):
    """A line w wide; thinner than a pixel, it keeps its area (fainter, not wider)."""
    we = max(w, px)
    return seg_mask(X, Y, p0, p1, we, px) * (w / we)


def _band(v, a, b, px):
    """Antialiased indicator of a <= v <= b."""
    return np.clip((v - a) / px + 0.5, 0, 1) * np.clip((b - v) / px + 0.5, 0, 1)


def _stripe(v, period, w, px):
    """Antialiased lines w wide, centred on the multiples of period."""
    return _band((v + period / 2) % period - period / 2, -w / 2, w / 2, px)


def _shift(m, dx, dy):
    """m read at (row + dy, col + dx), zero outside."""
    out = np.zeros_like(m)
    H, W = m.shape
    if abs(dx) >= W or abs(dy) >= H:
        return out
    ys_o, ys_i = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs_o, xs_i = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[ys_o, xs_o] = m[ys_i, xs_i]
    return out


class _Paint:
    """The city's region and its maps; shapes are painted over them, back to front."""

    def __init__(self, X, Y, px):
        r = np.searchsorted(Y[:, 0], [REGION[2], REGION[3]])
        c = np.searchsorted(X[0, :], [REGION[0], REGION[1]])
        self.sl = (slice(int(r[0]), int(r[1])), slice(int(c[0]), int(c[1])))
        self.x, self.y = X[self.sl].astype(np.float32), Y[self.sl].astype(np.float32)
        self.xs, self.ys = self.x[0], self.y[:, 0]
        self.px = float(px)
        sh = self.x.shape
        z = lambda *e: np.zeros(sh + e, np.float32)
        self.A, self.H0, self.HK, self.RF = z(), z(), z(), z()
        self.A0 = z()                                       # the street's own coverage (row 0)
        self.CT, self.CK, self.SN = z(3), z(3), z(3)
        rng = np.random.default_rng(23)
        self.n1 = self._noise(rng, 7.0)                    # weathering, broad
        self.n2 = self._noise(rng, 2.2)                    # and finer
        self.rimk = max(2, int(round(1.8 / self.px)))      # the rim's reach, pixels

    def _noise(self, rng, scale):
        gw = int((REGION[1] - REGION[0]) / scale) + 4
        gh = int((REGION[3] - REGION[2]) / scale) + 4
        g = rng.normal(0, 1, (gh, gw)).astype(np.float32)
        yy = (self.y - REGION[2]) / scale + 1
        xx = (self.x - REGION[0]) / scale + 1
        n = map_coordinates(g, [yy.ravel(), xx.ravel()], order=3, mode='nearest').reshape(self.x.shape)
        return (n / (n.std() + 1e-6)).astype(np.float32)

    def box(self, x0, x1, y0, y1):
        r = np.searchsorted(self.ys, [y0, y1])
        c = np.searchsorted(self.xs, [x0, x1])
        s = (slice(int(r[0]), int(r[1])), slice(int(c[0]), int(c[1])))
        return s, self.x[s], self.y[s]

    def rim(self, m):
        """A soft rim inside a shape's edges toward the light (right, up-right, up)."""
        out = np.zeros_like(m)
        K = self.rimk
        for dx, dy, wd in ((1, 0, 1.0), (1, -1, 0.85), (0, -1, 0.7)):
            for k in range(1, K + 1):
                w = wd * np.exp(-2.6 * (k - 1) / K)
                out = np.maximum(out, w * (1 - _shift(m, k * dx, k * dy)))
        return m * out

    def paint(self, s, m, T, lay, rf=0.0, sn=0.0, rim=1.0):
        """Lay a surface over what is there: coverage m, tint T (RGB), in row lay; rf its reflection, sn how
        much its own face catches the sun (a side wall), rim its edges' light."""
        if m.size == 0:
            return
        bx, by = self.x[s], self.y[s]
        mist = np.exp(-np.clip(CITY_BASE - by, 0, None) / MIST_SCALE[lay])
        h0 = LAYER_H0[lay] + MIST_H0[lay] * mist
        k = np.minimum(LAYER_K[lay] + MIST_K[lay] * mist, (1 - h0) / HAZE_MAX)
        T = np.broadcast_to(np.asarray(T, np.float32), m.shape + (3,))
        near = 0.4 + 0.6 * np.exp(-(730 - bx) / 300)      # nearer the sun, brighter rims
        light = (np.asarray(sn, np.float32)[..., None] * T if np.ndim(sn) else sn * T) * (LAYER_RIM[lay] ** 0.5)
        if rim:
            r = self.rim(m) * (rim * LAYER_RIM[lay])
            light = light + (r * near)[..., None] * np.array((1.0, 0.84, 0.66), np.float32)
        light = light * (1 - h0)[..., None]
        m3 = m[..., None]
        for M, v in ((self.CT, T * (1 - h0)[..., None]), (self.CK, T * k[..., None]), (self.SN, light)):
            M[s] = M[s] * (1 - m3) + m3 * v
        for M, v in ((self.A, 1.0), (self.H0, h0), (self.HK, k), (self.RF, np.asarray(rf) * (1 - h0))):
            M[s] = M[s] * (1 - m) + m * v
        if lay == 0:
            self.A0[s] = self.A0[s] * (1 - m) + m


# --- the street: BUILDINGS, their fronts ------------------------------------------------------------------------
def _facade(P, b, rng):
    """Building b's front: its material, floors, windows (the score's, where windows_of puts them)."""
    x0, x1, top = BUILDINGS[b]
    st = STYLE[b]
    px = P.px
    s, bx, by = P.box(x0 - 1.5, x1 + 1.5, top - 1.5, CITY_BASE + 1.5)
    m = rect_mask(bx, by, x0, x1, top, CITY_BASE + 1, px)
    mat = np.array(MAT[st['mat']], np.float32) * st['tone'] * 0.88
    hgt = CITY_BASE - top
    g = 1 + 0.045 * P.n1[s] + 0.02 * P.n2[s]
    g *= 0.88 + 0.16 * np.clip((CITY_BASE - by) / hgt, 0, 1)            # more sky toward the top
    g *= 0.95 + 0.08 * np.clip((bx - x0) / (x1 - x0), 0, 1)             # and toward the sun's side
    cols, rows = windows_of(b)
    ci = np.clip(np.round((bx - cols[0]) / 6), 0, len(cols) - 1).astype(int)
    ri = np.clip(np.round((rows[0] - by) / 8), 0, len(rows) - 1).astype(int)
    u = bx - (cols[0] + 6 * ci)
    v = by - (rows[0] - 8 * ri)
    win = _band(u, -1.3, 1.3, px) * _band(v, -2, 2, px)
    nr, nc = len(rows), len(cols)
    # each window its own: the glass, a blind drawn, a curtain; and each floor a little different
    floor = rng.uniform(0.92, 1.08, nr)[:, None]
    gw = rng.uniform(0.34, 0.54, (nr, nc)) * floor
    blind = rng.random((nr, nc)) < 0.16
    gw = np.where(blind, rng.uniform(0.62, 0.78, (nr, nc)), gw)
    rfw = rng.uniform(0.04, 0.13, (nr, nc)) * np.where(blind, 0.4, 1.0)
    kind = st['kind']
    rf = np.zeros_like(bx)
    if kind == 'punched':
        g *= 1 + 0.16 * _band(u, -1.75, 1.75, px) * _band(v, 2.0, 2.65, px)       # the sills
        g *= 1 - 0.1 * _band(u, -1.6, 1.6, px) * _band(v, -2.7, -2.0, px)        # the lintels' shadow
        fl = (ri % st['sc'] == 0)
        g *= 1 + 0.09 * fl * _band(v, 2.9, 3.7, px)                              # string courses
        g *= 1 - 0.06 * fl * np.maximum(_band(v, 3.7, 4.01, px), _band(v, -4.01, -3.6, px))
    elif kind == 'piers':
        pier = 1 - _band(u, -2.1, 2.1, px)
        g *= 1 + 0.08 * pier
        g *= 1 - 0.12 * _band(u, -1.5, 1.5, px) * _band(np.abs(v), 2.0, 4.01, px)   # the recessed spandrels
        g *= 1 + 0.08 * _band(u, -2.1, -1.65, px)                                # light into the recess
    elif kind == 'bands':
        sp = _band(np.abs(v), 2.35, 4.01, px)
        g *= (1 + 0.1 * sp) * (1 - 0.12 * (1 - sp))
        g *= 1 - 0.07 * _band(v, -2.35, -1.9, px)                               # each band's shadow
        g *= 1 + 0.06 * _band(v, 2.35, 2.8, px)                                 # its top edge
    elif kind == 'curtain':
        g *= 0.9 + 0.06 * np.clip((by - top) / hgt, 0, 1)
        rf = rf + 0.05 + 0.04 * np.clip((CITY_BASE - by) / hgt, 0, 1)            # the whole face is glass
        mull = np.maximum(_band(np.abs(u), 2.8, 3.01, px), _band(np.abs(v), 3.8, 4.01, px))
        g *= 1 + 0.12 * _band(np.abs(v), 2.0, 4.01, px) * (1 - mull)             # spandrel panels
        g *= 1 + 0.22 * mull
        rf = rf * (1 - 0.6 * mull)
    sn = np.zeros_like(bx)
    span = _band(by, rows[-1] - 3.2, rows[0] + 3.0, px)
    for cb in st.get('bays', ()):                  # bay windows: a column of them stands out from the wall
        cu = bx - cols[cb]
        bay = _band(cu, -2.5, 2.5, px) * span
        g *= 1 + 0.24 * bay - 0.26 * _band(cu, -3.4, -2.5, px) * span
        sn = sn + 0.08 * _band(cu, 1.9, 2.5, px) * bay
    if st.get('balconies'):                        # balcony slabs under alternate windows
        slab = _band(u, -2.3, 2.3, px) * _band(v, 2.0, 2.6, px) * ((ci + ri) % 2 == 0) * (ri > 0)
        g *= 1 + 0.2 * slab - 0.2 * _band(u, -2.3, 2.3, px) * _band(v, 2.6, 3.6, px) * ((ci + ri) % 2 == 0) * (ri > 0)
    if st.get('cross') and kind == 'punched':
        cross = np.maximum(_band(u, -0.18, 0.18, px), _band(v, -0.75, -0.4, px)) * win
        gw_px = gw[ri, ci] * (1 + 0.35 * cross)
    else:
        gw_px = gw[ri, ci]
    # the glass: darker under the reveal at its top, reflecting the sky below that
    pane = gw_px * (0.82 + 0.18 * np.clip((v + 2) / 1.2, 0, 1))
    glass = np.array((0.84, 0.9, 1.02), np.float32)
    T = mat * g[..., None]
    Tw = (mat * 0.55 + glass * 0.45) * (pane * (0.88 + 0.12 * g))[..., None]
    T = T * (1 - win[..., None]) + Tw * win[..., None]
    rf = rf * (1 - win) + win * rfw[ri, ci] * np.clip(0.6 + 0.4 * (v + 2) / 4, 0, 1)
    # the shade under the cornice; the ground floor, shopfronts and a plinth
    T *= (1 - 0.2 * _band(by, top + 0.8, top + 2.2, px))[..., None]
    T *= (1 + 0.12 * _band(by, 290.45, 291.1, px))[..., None]
    shop = _band(by, 291.1, 296.6, px)
    sx = np.round((bx - x0 - 1.5) / 7.0)
    door = _band(bx - (x0 + 1.5 + 7 * sx), -0.4, 0.4, px)
    T *= (1 - shop * (0.24 - 0.1 * door))[..., None]
    P.paint(s, m, T, 0, rf=rf, sn=sn, rim=0.8 * EDGE[b])
    # awnings over some of the shops
    aw = np.zeros_like(bx)
    awc = np.zeros(bx.shape + (3,), np.float32)
    for k in range(int((x1 - x0 - 3) // 7)):
        if rng.random() < 0.45:
            xa = x0 + 1.5 + 7 * k + 0.9
            a = poly_mask(bx, by, [(xa, 291.3), (xa + 5.2, 291.3), (xa + 5.6, 292.9), (xa - 0.4, 292.9)], px)
            col = np.array(AWNINGS[rng.integers(len(AWNINGS))], np.float32)
            aw = np.maximum(aw, a)
            awc = awc * (1 - a[..., None]) + a[..., None] * col
    if aw.any():
        P.paint(s, aw, awc * (1 - 0.15 * _band(by, 292.3, 293, px))[..., None], 0, rim=0.6)
    return mat


def _side(P, b, lay=0, mat=None, x1=None, top=None, base=CITY_BASE + 1, depth=0.013):
    """The right-hand wall of a building left of the view's centre: it faces the low sun at a graze."""
    if x1 is None:
        _, x1, top = BUILDINGS[b]
    w = (VP_X - x1) * depth
    if w < 0.4:
        return
    drop = (YH - top) * depth                     # its top runs toward the vanishing point, on the horizon
    s, bx, by = P.box(x1 - 1, x1 + w + 1.5, top - 1.5, base + 1)
    m = poly_mask(bx, by, [(x1 - 0.3, top), (x1 + w, top + drop), (x1 + w, base), (x1 - 0.3, base)],
                  P.px * LAYER_SOFT[lay])
    f = np.clip((bx - x1) / w, 0, 1)
    g = 0.85 + 0.05 * P.n1[s]
    # its floors, converging
    yy = by - drop * f
    fl = 1 - 0.1 * _stripe(CITY_BASE - yy - 4, 8.0, 1.0, P.px)
    e = EDGE[b] if b is not None else 0.8
    sn = (0.6 - 0.25 * f) * e * np.clip(0.55 + 0.45 * (base - by) / max(base - top, 1), 0, 1) * fl
    rim = 0.6 * e
    if b is not None and b + 1 < len(BUILDINGS) and BUILDINGS[b + 1][0] - x1 < 10:
        ntop = BUILDINGS[b + 1][2]                 # the neighbour on the sun's side shades the wall below its top
        sn = sn * (0.12 + 0.88 * np.clip((ntop + 2 - by) / 3, 0, 1))
        if ntop < top + 4:
            rim *= 0.2
    P.paint(s, m, mat * (g * fl)[..., None], lay, sn=sn, rim=rim)


def _cornice(P, x0, x1, y, mat, lay=0, h=2.0, over=0.7):
    s, bx, by = P.box(x0 - over - 1, x1 + over + 1, y - h - 1, y + 2)
    m = rect_mask(bx, by, x0 - over, x1 + over, y - h * 0.6, y + h * 0.4, P.px * LAYER_SOFT[lay])
    g = 1.12 - 0.12 * _band(by, y + 0.1, y + h * 0.4, P.px)
    P.paint(s, m, mat * g[..., None], lay, rim=1.3)


def _block(P, x0, x1, y0, y1, mat, lay=0, wins=None, rim=1.0, rng=None, gain=1.0):
    """A plain box (a tier, a plant room, a building behind), with small windows of its own, if wanted:
    wins = (spacing x, spacing y, w, h, contrast)."""
    px = P.px * LAYER_SOFT[lay]
    s, bx, by = P.box(x0 - 1.5, x1 + 1.5, y0 - 1.5, y1 + 1.5)
    m = rect_mask(bx, by, x0, x1, y0, y1, px)
    g = (1 + 0.04 * P.n1[s]) * (0.9 + 0.14 * np.clip((y1 - by) / max(y1 - y0, 1), 0, 1)) * gain
    g *= 0.95 + 0.08 * np.clip((bx - x0) / max(x1 - x0, 1), 0, 1)
    rf = 0.0
    if wins is not None:
        sx_, sy_, ww, wh, ctr = wins
        nx = max(1, int((x1 - x0 - 2) // sx_))
        ox = x0 + (x1 - x0 - (nx - 1) * sx_) / 2
        ny = max(1, int((y1 - y0 - 3) // sy_))
        ci = np.clip(np.round((bx - ox) / sx_), 0, nx - 1)
        ri = np.clip(np.round((y0 + 2.5 + wh / 2 - by) / -sy_), 0, ny - 1)
        u = bx - (ox + sx_ * ci)
        v = by - (y0 + 2.5 + wh / 2 + sy_ * ri)
        w_ = _band(u, -ww / 2, ww / 2, px) * _band(v, -wh / 2, wh / 2, px) * (by < y1 - 1.5)
        rnd = rng.uniform(0.75, 1.15, (ny, nx))[ri.astype(int), ci.astype(int)] if rng is not None else 1
        g = g * (1 - ctr * rnd * w_)
        rf = 0.04 * w_
    P.paint(s, m, mat * g[..., None], lay, rf=rf, rim=rim)


def _water_tower(P, cx, base, mat, lay=0, r=4.0, h=7.0):
    px = P.px * LAYER_SOFT[lay]
    s, bx, by = P.box(cx - r - 2, cx + r + 2, base - h - 7, base + 1)
    legs = np.zeros_like(bx)
    for lx in (cx - r * 0.75, cx + r * 0.75, cx):
        legs = np.maximum(legs, _line(bx, by, (lx, base - 4.8), (lx, base + 0.5), 0.55, px))
    legs = np.maximum(legs, _line(bx, by, (cx - r * 0.75, base - 1), (cx + r * 0.75, base - 4.2), 0.3, px))
    legs = np.maximum(legs, _line(bx, by, (cx + r * 0.75, base - 1), (cx - r * 0.75, base - 4.2), 0.3, px))
    P.paint(s, legs, np.array(MAT['steel']) * 0.8, lay, rim=0.5)
    tank = rect_mask(bx, by, cx - r, cx + r, base - 4.5 - h, base - 4.5, px)
    f = np.clip((bx - cx) / r, -1, 1)
    g = 0.85 + 0.25 * np.sqrt(np.clip(1 - f * f, 0, 1)) * (0.6 + 0.4 * f)          # a barrel, lit to the right
    hoops = 1 - 0.18 * np.maximum(_band(by, base - 6.6, base - 6.2, px), _band(by, base - 9.6, base - 9.2, px))
    P.paint(s, tank, mat * (g * hoops)[..., None], lay, rim=1.1)
    roof = poly_mask(bx, by, [(cx - r - 0.5, base - 4.5 - h + 0.3), (cx + r + 0.5, base - 4.5 - h + 0.3),
                              (cx, base - 4.5 - h - 4.2)], px)
    P.paint(s, roof, np.array(MAT['slate']) * (0.95 + 0.15 * f)[..., None], lay, rim=1.3)


def _antenna(P, x, y0, y1, lay=0, w=0.45, bars=2):
    px = P.px * LAYER_SOFT[lay]
    s, bx, by = P.box(x - 3, x + 3, y1 - 1, y0 + 1)
    m = _line(bx, by, (x, y0), (x, y1), w, px)
    for k in range(bars):
        yb = y1 + (y0 - y1) * (0.2 + 0.25 * k)
        hw = 1.6 - 0.4 * k
        m = np.maximum(m, _line(bx, by, (x - hw, yb), (x + hw, yb), 0.3, px))
    P.paint(s, m, np.array(MAT['steel']) * 0.7, lay, rim=0.6)


def _dish(P, cx, base, r=2.0, lay=0):
    px = P.px * LAYER_SOFT[lay]
    s, bx, by = P.box(cx - r - 2, cx + r + 2, base - 2 * r - 2, base + 1)
    m = np.maximum(ellipse_mask(bx, by, cx, base - r - 0.6, r * 0.55, r, px) * (bx < cx + 0.2 * r),
                   _line(bx, by, (cx, base - r * 0.6), (cx - 0.3, base), 0.5, px))
    P.paint(s, m, np.array(MAT['concrete']) * 1.15, lay, rim=1.2)


def _tree(P, cx, base, r, rng, lay=0, mat='leaf'):
    """A small tree on a roof: a cluster of soft round leaves' masses, lit at the edges."""
    px = P.px * LAYER_SOFT[lay]
    s, bx, by = P.box(cx - 2.2 * r, cx + 2.2 * r, base - 3.2 * r, base + 1)
    trunk = _line(bx, by, (cx, base + 0.5), (cx + 0.2, base - r * 1.2), 0.55, px)
    P.paint(s, trunk, np.array(MAT['wood']) * 0.6, lay, rim=0.3)
    m = np.zeros_like(bx)
    for k in range(5):
        a = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(0.2, 0.75) * r
        ex, ey = cx + d * np.cos(a), base - 1.6 * r + 0.75 * d * np.sin(a)
        rr = r * rng.uniform(0.55, 0.85)
        m = np.maximum(m, ellipse_mask(bx, by, ex, ey, rr, rr * 0.9, px))
    g = 0.85 + 0.12 * P.n2[s] + 0.1 * np.clip((base - by) / (3 * r), 0, 1)
    P.paint(s, m, np.array(MAT[mat]) * g[..., None], lay, rim=1.4)


def _shrubs(P, x0, x1, base, rng, lay=0, h=1.8):
    px = P.px * LAYER_SOFT[lay]
    s, bx, by = P.box(x0 - 2, x1 + 2, base - 2.5 * h, base + 1)
    m = rect_mask(bx, by, x0, x1, base - 0.6 * h, base + 0.5, px)
    x = x0
    while x < x1:
        rr = rng.uniform(0.7, 1.2) * h
        m = np.maximum(m, ellipse_mask(bx, by, x + rr * 0.6, base - 0.5 * h, rr, rr * 0.75, px))
        x += rr * rng.uniform(0.9, 1.5)
    m = m * (bx > x0 - 0.3) * (bx < x1 + 0.3)
    P.paint(s, m, np.array(MAT['leaf2']) * (0.9 + 0.1 * P.n2[s])[..., None], lay, rim=1.2)


def _railing(P, x0, x1, y, lay=0, h=1.5, step=3.0):
    px = P.px * LAYER_SOFT[lay]
    s, bx, by = P.box(x0 - 1, x1 + 1, y - h - 1, y + 1)
    m = _line(bx, by, (x0, y - h), (x1, y - h), 0.35, px)
    for x in np.arange(x0, x1 + 0.01, step):
        m = np.maximum(m, _line(bx, by, (x, y - h), (x, y), 0.28, px))
    P.paint(s, m, np.array(MAT['steel']) * 0.8, lay, rim=0.8)


def _roofs(P, rng):
    """The street's roofs: crowns, setbacks, a spire, water towers, plant rooms, gardens, antennas."""
    B = BUILDINGS
    M = lambda k: np.array(MAT[k], np.float32)
    # 0: a brick block with a plant room, an antenna on it, and a garden of three small trees
    x0, x1, top = B[0]
    _block(P, 64.5, 73.5, top - 6.5, top + 0.5, M('concrete') * 0.95, wins=None)
    _louvres(P, 65.5, 72.5, top - 5.5, top - 1)
    _antenna(P, 71, top - 6.5, top - 17, bars=2)
    _railing(P, x0 + 0.5, 63.5, top, step=2.6)
    for cx, r in ((51.5, 1.9), (55.5, 2.4), (60, 1.7)):
        _tree(P, cx, top, r, rng)
    # 1: stone, a stepped crown and a pyramid with a finial
    x0, x1, top = B[1]
    st = M('stone') * 1.0
    _block(P, x0 + 2.5, x1 - 2.5, top - 6, top + 0.5, st * 1.02, wins=(4.0, 6, 1.4, 3, 0.3), rng=rng)
    _cornice(P, x0 + 2.5, x1 - 2.5, top - 6, st, h=1.2, over=0.4)
    _block(P, x0 + 6, x1 - 6, top - 11, top - 5.5, st * 1.04)
    _cornice(P, x0 + 6, x1 - 6, top - 11, st, h=1.0, over=0.4)
    s, bx, by = P.box(x0 + 5, x1 - 5, top - 24, top - 10)
    cx = (x0 + x1) / 2
    m = poly_mask(bx, by, [(x0 + 7.5, top - 10.6), (x1 - 7.5, top - 10.6), (cx, top - 19)], P.px)
    P.paint(s, m, M('copper') * (0.8 + 0.3 * np.clip((bx - cx) / 5 + 0.5, 0, 1))[..., None], 0, rim=1.3)
    _antenna(P, cx, top - 18.5, top - 24, w=0.4, bars=0)
    # 2: the old warehouse: a water tower, skylights, shrubs on its right
    x0, x1, top = B[2]
    _water_tower(P, x0 + 0.3 * (x1 - x0), top, M('wood'))
    for k in range(3):
        xa = x0 + 21 + 4.4 * k
        s, bx, by = P.box(xa - 1, xa + 4, top - 4, top + 1)
        P.paint(s, poly_mask(bx, by, [(xa, top + 0.3), (xa + 3.4, top + 0.3), (xa + 3.4, top - 2.6)], P.px),
                M('glass') * 0.9, 0, rf=0.12, rim=1.2)
    _billboard(P, 135.5, 148.5, top)
    # 3: terracotta, setbacks with windows, a flagpole
    x0, x1, top = B[3]
    _tiers(P, 3, M('terracotta') * 0.95, rng)
    cx = (x0 + x1) / 2
    _antenna(P, cx + 2, top - 20, top - 30, w=0.35, bars=0)
    s, bx, by = P.box(cx + 1, cx + 8, top - 32, top - 24)
    fl = poly_mask(bx, by, [(cx + 2.2, top - 29.7), (cx + 4.6, top - 29.1), (cx + 7, top - 29.5), (cx + 7, top - 26.7),
                            (cx + 4.6, top - 26.2), (cx + 2.2, top - 26.8)], P.px)
    P.paint(s, fl, np.array(AWNINGS[0], np.float32) * 0.95, 0, rim=1.0)
    # 4: concrete, a roof garden: trees, a pergola, a railing; a small plant room
    x0, x1, top = B[4]
    _railing(P, x0 + 0.6, x1 - 0.6, top, step=3.0)
    _block(P, x1 - 9, x1 - 2.5, top - 5, top + 0.5, M('concrete') * 0.92)
    _louvres(P, x1 - 8.2, x1 - 3.3, top - 4.2, top - 1)
    for cx, r in ((x0 + 4, 2.2), (x0 + 9.5, 2.8), (x0 + 15, 2.0)):
        _tree(P, cx, top, r, rng, mat='leaf2')
    s, bx, by = P.box(x0 + 17, x1 - 9, top - 5, top + 1)
    m = _line(bx, by, (x0 + 18, top - 3.6), (x1 - 10.5, top - 3.6), 0.6, P.px)
    for x in np.linspace(x0 + 18.4, x1 - 10.9, 3):
        m = np.maximum(m, _line(bx, by, (x, top - 3.6), (x, top), 0.45, P.px))
    P.paint(s, m, M('wood') * 0.8, 0, rim=1.0)
    # 5: the tallest: stone, setbacks, a spire
    x0, x1, top = B[5]
    _tiers(P, 5, M('stone') * 1.04, rng)
    cx = (x0 + x1) / 2
    cy_ = top - 15.6                                    # a clock on the top tier
    s, bx, by = P.box(cx - 4, cx + 4, cy_ - 4, cy_ + 4)
    P.paint(s, ellipse_mask(bx, by, cx, cy_, 2.9, 2.9, P.px), M('copper') * 0.85, 0, rim=1.0)
    hands = np.maximum(_line(bx, by, (cx, cy_), (cx + 0.25, cy_ - 1.8), 0.4, P.px),
                       _line(bx, by, (cx, cy_), (cx + 1.25, cy_ + 0.5), 0.4, P.px))
    P.paint(s, ellipse_mask(bx, by, cx, cy_, 2.25, 2.25, P.px), M('stone') * 1.3 * (1 - 0.55 * hands)[..., None], 0,
            rim=0)
    s, bx, by = P.box(cx - 5, cx + 5, top - 54, top - 18)
    m = poly_mask(bx, by, [(cx - 2.4, top - 19.5), (cx + 2.4, top - 19.5), (cx + 0.45, top - 44),
                           (cx - 0.45, top - 44)], P.px)
    f = np.clip((bx - cx) / 2.4, -1, 1)
    rib = 1 - 0.12 * np.maximum(_band(by, top - 26.4, top - 25.6, P.px), _band(by, top - 32.4, top - 31.8, P.px))
    P.paint(s, m, M('stone') * ((0.85 + 0.3 * (f > 0.1)) * rib)[..., None], 0, rim=1.3)
    P.paint(s, ellipse_mask(bx, by, cx, top - 44.3, 0.9, 0.9, P.px), M('copper') * 1.1, 0, rim=1.2)
    _antenna(P, cx, top - 45, top - 51.5, w=0.45, bars=0)
    # 6: the glass tower's slanted crown: a lattice over it
    x0, x1, top = B[6]
    s, bx, by = P.box(x0 - 1, x1 + 1, top - 11, top + 2)
    m = poly_mask(bx, by, [(x0, top + 1), (x1, top + 1), (x1, top - 9)], P.px)
    yy = top + 1 - (bx - x0) / (x1 - x0) * 10
    lat = np.maximum(_stripe(bx - x0, 4.33, 0.4, P.px), _stripe(by - yy, 3.0, 0.4, P.px))
    P.paint(s, m, M('glass') * (0.85 + 0.25 * lat)[..., None], 0, rf=0.1 * (1 - lat), rim=1.4)
    # 7: brick, a mansard with dormers, chimneys
    x0, x1, top = B[7]
    s, bx, by = P.box(x0 - 1, x1 + 1, top - 12, top + 1)
    m = poly_mask(bx, by, [(x0 + 0.5, top + 0.5), (x0 + 3, top - 7), (x1 - 3, top - 7), (x1 - 0.5, top + 0.5)], P.px)
    g = 0.9 + 0.1 * np.clip((by - top + 7) / 7, 0, 1)
    P.paint(s, m, M('slate') * g[..., None], 0, rim=1.2)
    for dx in (x0 + 6.5, x0 + 12, x0 + 17.5):
        s, bx, by = P.box(dx - 2.5, dx + 2.5, top - 8, top + 1)
        dm = np.maximum(rect_mask(bx, by, dx - 1.6, dx + 1.6, top - 4.6, top - 0.8, P.px),
                        poly_mask(bx, by, [(dx - 2, top - 4.4), (dx + 2, top - 4.4), (dx, top - 6.4)], P.px))
        P.paint(s, dm, M('stucco') * 0.95, 0, rim=1.0)
        s, bx, by = P.box(dx - 1.5, dx + 1.5, top - 4.5, top)
        P.paint(s, rect_mask(bx, by, dx - 0.8, dx + 0.8, top - 3.8, top - 1.2, P.px), M('glass') * 0.55, 0,
                rf=0.06, rim=0)
    for cxh in (x0 + 4, x1 - 4.5):
        _block(P, cxh - 1.1, cxh + 1.1, top - 11, top - 6, M('brick') * 0.95)
    # 8: the wide slab: a water tower, a plant room, antennas, a dish
    x0, x1, top = B[8]
    _water_tower(P, x0 + 0.3 * (x1 - x0), top, M('wood') * 0.95, r=4.5)
    _block(P, x0 + 30, x0 + 46, top - 7, top + 0.5, M('concrete'), wins=(4.0, 6, 2.2, 1.5, 0.15), rng=rng)
    _louvres(P, x0 + 31, x0 + 45, top - 5, top - 1.5)
    _antenna(P, x0 + 42, top - 7, top - 20, bars=3)
    _antenna(P, x0 + 34, top - 7, top - 13, w=0.35, bars=1)
    _dish(P, x0 + 52, top, r=2.0)
    _railing(P, x0 + 47, x1 - 0.8, top, step=2.8)
    # 9: narrow, dark brick: a lift room and two masts
    x0, x1, top = B[9]
    _block(P, x0 + 0.55 * (x1 - x0), x0 + 0.85 * (x1 - x0), top - 6, top + 0.5, M('darkbrick') * 1.05)
    _antenna(P, x0 + 4, top, top - 14, w=0.45, bars=2)
    _antenna(P, x0 + 8, top, top - 8, w=0.35, bars=1)
    # 10: the glass tower: setbacks, a crown frame, a mast
    x0, x1, top = B[10]
    _tiers(P, 10, M('glass2') * 0.95, rng, glass=True)
    cx = (x0 + x1) / 2
    s, bx, by = P.box(cx - 8, cx + 8, top - 30, top - 18)
    m = np.zeros_like(bx)
    for xx in np.linspace(cx - 6.2, cx + 6.2, 5):
        m = np.maximum(m, _line(bx, by, (xx, top - 20), (xx, top - 26), 0.45, P.px))
    m = np.maximum(m, _line(bx, by, (cx - 6.5, top - 26), (cx + 6.5, top - 26), 0.6, P.px))
    P.paint(s, m, M('steel') * 0.9, 0, rim=1.2)
    _antenna(P, cx - 2, top - 26, top - 36, w=0.5, bars=2)
    # 11: sandstone, its sloped roof copper, a dormer row of light
    x0, x1, top = B[11]
    s, bx, by = P.box(x0 - 1, x1 + 1, top - 11, top + 2)
    m = poly_mask(bx, by, [(x0, top + 1), (x1, top + 1), (x1, top - 9)], P.px)
    seams = _stripe(bx - x0, 3.0, 0.4, P.px)
    P.paint(s, m, M('copper') * (0.82 + 0.12 * seams + 0.1 * np.clip((bx - x0) / (x1 - x0), 0, 1))[..., None], 0,
            rim=1.4)
    # 12: brick, a water tower, a garden with a tree, a dish
    x0, x1, top = B[12]
    _water_tower(P, x0 + 0.3 * (x1 - x0), top, M('wood') * 1.05, r=3.6, h=6)
    _shrubs(P, x0 + 17, x1 - 5, top, rng, h=1.5)
    _tree(P, x1 - 4, top, 2.3, rng)
    _dish(P, x0 + 3.5, top, r=1.5)
    # 13: stucco: a lift room, a garden
    x0, x1, top = B[13]
    _block(P, x0 + 0.55 * (x1 - x0), x0 + 0.85 * (x1 - x0), top - 6, top + 0.5, M('stucco') * 0.95)
    _tree(P, x0 + 4, top, 2.0, rng)
    _shrubs(P, x0 + 6.5, x0 + 11.5, top, rng, h=1.3)
    _antenna(P, x0 + 0.7 * (x1 - x0), top - 6, top - 12, w=0.35, bars=1)


def _billboard(P, x0, x1, base):
    """An old sign frame on a roof: its legs, a catwalk, a pale board."""
    s, bx, by = P.box(x0 - 1, x1 + 1, base - 13, base + 1)
    m = np.zeros_like(bx)
    for x in np.linspace(x0 + 1.5, x1 - 1.5, 3):
        m = np.maximum(m, _line(bx, by, (x, base + 0.3), (x, base - 4), 0.5, P.px))
    m = np.maximum(m, _line(bx, by, (x0 + 1.5, base - 0.4), (x1 - 1.5, base - 3.6), 0.3, P.px))
    m = np.maximum(m, _line(bx, by, (x0, base - 3.6), (x1, base - 3.6), 0.45, P.px))
    P.paint(s, m, np.array(MAT['steel']) * 0.75, 0, rim=0.7)
    board = rect_mask(bx, by, x0, x1, base - 12, base - 4.4, P.px)
    g = 1.25 - 0.25 * np.maximum(_band(by, base - 12, base - 11.4, P.px), _band(by, base - 5, base - 4.4, P.px))
    g -= 0.3 * _band(by, base - 9.6, base - 7.2, P.px) * _band(bx, x0 + 1.5, x1 - 4.5, P.px)
    P.paint(s, board, np.array(MAT['stucco']) * g[..., None], 0, rim=1.2)


def _landmarks(P, rng):
    """Two landmarks among the buildings behind: a church tower and its spire, a dome with its lantern."""
    M = lambda k: np.array(MAT[k], np.float32)
    _block(P, 309, 316, 223, CITY_BASE + 1, M('stone') * 0.95, lay=1)
    s, bx, by = P.box(305, 320, 190, 233)
    op = np.zeros_like(bx)
    for xa in (310.2, 313.1):
        op = np.maximum(op, np.maximum(rect_mask(bx, by, xa, xa + 1.7, 226.6, 231, P.px),
                                       ellipse_mask(bx, by, xa + 0.85, 226.6, 0.85, 0.85, P.px)))
    P.paint(s, op, M('slate') * 0.45, 1, rim=0)
    sp = poly_mask(bx, by, [(308.5, 223.4), (316.5, 223.4), (312.5, 200)], P.px * LAYER_SOFT[1])
    P.paint(s, sp, M('slate') * (0.9 + 0.2 * (bx > 312.5))[..., None], 1, rim=1.3)
    cr = np.maximum(_line(bx, by, (312.5, 200.5), (312.5, 195), 0.45, P.px),
                    _line(bx, by, (311.2, 197), (313.8, 197), 0.4, P.px))
    P.paint(s, cr, M('steel'), 1, rim=0.8)
    _block(P, 398, 418, 231, CITY_BASE + 1, M('stone'), lay=1, wins=(4.0, 6.0, 1.6, 2.6, 0.3), rng=rng)
    _cornice(P, 398, 418, 231, M('stone'), lay=1, h=1.4, over=0.5)
    _block(P, 401.5, 414.5, 224, 231, M('stone') * 1.05, lay=1, wins=(3.2, 6.0, 1.2, 3.0, 0.4), rng=rng)
    s, bx, by = P.box(399, 417, 208, 225)
    d = ellipse_mask(bx, by, 408, 224.2, 6.8, 7.4, P.px * LAYER_SOFT[1]) * (by < 224.4)
    ribs = 1 + 0.1 * np.maximum(_band(bx, 404.6, 405.0, P.px), _band(bx, 411.0, 411.4, P.px))
    P.paint(s, d, M('copper') * ((0.85 + 0.25 * np.clip((bx - 404) / 8, 0, 1)) * ribs)[..., None], 1, rim=1.4)
    _block(P, 406.9, 409.1, 213.2, 217.4, M('stone') * 1.1, lay=1)
    s, bx, by = P.box(405, 411, 208, 214)
    P.paint(s, poly_mask(bx, by, [(406.5, 213.4), (409.5, 213.4), (408, 210.6)], P.px), M('copper'), 1, rim=1.2)


def _boats(P):
    """Boats tied up along the quay (their lower halves are the water's: the film mirrors above the waterline)."""
    M = lambda k: np.array(MAT[k], np.float32)
    for x, w, hull, cab in ((150, 8, (1.45, 1.4, 1.36), 'stucco'), (258, 10, (1.0, 0.55, 0.5), 'stucco'),
                            (454, 7, (0.7, 0.85, 1.1), 'concrete')):
        s, bx, by = P.box(x - w - 2, x + w + 2, 290, YH + 2)
        h = poly_mask(bx, by, [(x - w, 297.2), (x + w + 1.5, 296.6), (x + w - 0.8, 301), (x - w + 1, 301)], P.px)
        g = 1 - 0.3 * _band(by, 298.6, 301, P.px)
        P.paint(s, h, np.array(hull, np.float32) * 0.8 * g[..., None], 0, rim=1.1)
        c = rect_mask(bx, by, x - 0.45 * w, x + 0.25 * w, 293.6, 297.3, P.px)
        win = rect_mask(bx, by, x - 0.38 * w, x + 0.18 * w, 294.4, 295.6, P.px)
        P.paint(s, c, M(cab) * (1 - 0.45 * win)[..., None], 0, rf=0.05 * win, rim=1.0)


def _louvres(P, x0, x1, y0, y1, lay=0):
    s, bx, by = P.box(x0, x1, y0, y1)
    m = rect_mask(bx, by, x0, x1, y0, y1, P.px) * _stripe(by - y0 - 0.25, 1.0, 0.45, P.px)
    P.paint(s, m * 0.6, np.array(MAT['concrete']) * 0.75, lay, rim=0)


def _tiers(P, b, mat, rng, glass=False):
    """The setbacks above a tower's top (where the old city put them), each with windows and a cornice."""
    x0, x1, top = BUILDINGS[b]
    w = x1 - x0
    ga, gb = (x0 + 0.2 * w, x1 - 0.2 * w), (x0 + 0.36 * w, x1 - 0.36 * w)
    wins = (4.2, 6, 1.6, 2.8, 0.42 if not glass else 0.3)
    _block(P, ga[0], ga[1], top - 12, top + 0.5, mat, wins=wins, rng=rng)
    _cornice(P, ga[0], ga[1], top - 12, mat, h=1.4, over=0.5)
    _block(P, gb[0], gb[1], top - 20, top - 11, mat * 1.03, wins=(3.6, 6, 1.4, 2.6, wins[4]), rng=rng)
    _cornice(P, gb[0], gb[1], top - 20, mat, h=1.2, over=0.45)


# --- the rows behind --------------------------------------------------------------------------------------------
# steam from vents behind the street: x, y (its vent), how far it rises and drifts, its puffs' size, rate, opacity
STEAM = [(66.0, 222.0, 26.0, 9.0, 2.0, 0.045, 0.3), (364.0, 192.5, 24.0, 8.0, 1.8, 0.05, 0.27),
         (547.0, 243.0, 22.0, 7.0, 1.9, 0.04, 0.28)]
# the row just behind the street, placed by hand: in its gaps and over its lower roofs (x0, x1, top, a tower)
NEAR_ROW = [(40, 57, 236, 0), (60, 75, 224, 1), (93, 109, 212, 1), (111, 124, 246, 0), (123, 140, 255, 0),
            (139, 152, 242, 0), (185, 199, 220, 0), (204, 224, 230, 0), (262, 279, 204, 1), (289, 305, 236, 0),
            (318, 333, 230, 0), (352, 370, 198, 1), (384, 398, 224, 0), (414, 425, 238, 0), (461, 476, 212, 1),
            (483, 499, 222, 0), (499, 515, 236, 0),
            (519, 535, 216, 0), (539, 555, 244, 0), (555, 569, 226, 0)]


def _row(P, lay, rng, x0, x1, wr, tops, mats, wins, features, gap=(-2.5, 1.5), placed=None):
    """A row of buildings at one depth: widths wr, tops drawn from tops (lo, hi), with occasional towers; or
    the buildings given in placed."""
    out = [(a, b_, top, bool(tw)) for a, b_, top, tw in placed] if placed else []
    x = x1 if placed else x0
    while x < x1:
        w = rng.uniform(*wr)
        top = rng.uniform(*tops)
        tower = rng.random() < features.get('tower', 0)
        if tower:
            top -= rng.uniform(15, 35)
            w *= 0.75
        if 122 < x + w / 2 < 142 and top < 258:            # keep the radio mast clear
            top, tower = 258 + rng.uniform(0, 8), False
        if x > 624:                                        # and step down to the hills at the east end
            top, tower = max(top, 256 + 0.8 * (x - 624)), False
        out.append((x, x + w, top, tower))
        x += w + rng.uniform(*gap)
    for (a, b_, top, tower) in out:
        mat = np.array(MAT[mats[rng.integers(len(mats))]], np.float32) * rng.uniform(0.88, 1.1)
        if lay <= 2:
            _side(P, None, lay=lay, mat=mat * 0.95, x1=b_, top=top, depth=0.009)
        _block(P, a, b_, top, CITY_BASE + 1, mat, lay=lay, wins=wins, rng=rng)
        r = rng.random()
        if r < features.get('cornice', 0):
            _cornice(P, a, b_, top, mat, lay=lay, h=1.4, over=0.5)
        r = rng.random()
        wd = b_ - a
        if tower:                                          # a tower's crown: a setback or two, a mast
            ta, tb_ = a + 0.18 * wd, b_ - 0.18 * wd
            h1 = rng.uniform(3, 6)
            _block(P, ta, tb_, top - h1, top + 0.5, mat * 1.03, lay=lay)
            if rng.random() < 0.6:
                tc, td = a + 0.34 * wd, b_ - 0.34 * wd
                _block(P, tc, td, top - h1 - rng.uniform(2.5, 4.5), top - h1 + 0.5, mat * 1.05, lay=lay)
            if rng.random() < 0.5:
                _antenna(P, (a + b_) / 2, top - h1, top - h1 - rng.uniform(6, 12), lay=lay, bars=1)
        elif r < features.get('box', 0):
            ba = a + rng.uniform(0.15, 0.5) * wd
            _block(P, ba, ba + rng.uniform(0.2, 0.35) * wd, top - rng.uniform(3, 6), top + 0.5, mat * 0.95, lay=lay)
        elif r < features.get('box', 0) + features.get('tank', 0):
            _water_tower(P, a + rng.uniform(0.25, 0.75) * wd, top, np.array(MAT['wood']), lay=lay, r=3.0, h=5)
        elif r < features.get('box', 0) + features.get('tank', 0) + features.get('dome', 0):
            cx = (a + b_) / 2
            s, bx, by = P.box(a, b_, top - wd, top + 1)
            m = ellipse_mask(bx, by, cx, top, wd * 0.32, wd * 0.3, P.px * LAYER_SOFT[lay]) * (by < top + 0.5)
            P.paint(s, m, np.array(MAT['copper']) * 0.95, lay, rim=1.2)
        elif r < features.get('box', 0) + features.get('tank', 0) + features.get('dome', 0) + features.get('gable', 0):
            s, bx, by = P.box(a - 1, b_ + 1, top - 0.45 * wd - 1, top + 1)
            m = poly_mask(bx, by, [(a, top + 0.5), (b_, top + 0.5), ((a + b_) / 2, top - 0.4 * wd)],
                          P.px * LAYER_SOFT[lay])
            P.paint(s, m, np.array(MAT['slate']), lay, rim=1.2)
        if rng.random() < features.get('antenna', 0):
            _antenna(P, a + rng.uniform(0.3, 0.8) * wd, top, top - rng.uniform(5, 12), lay=lay, bars=1)
    return out


def _east(P, rng):
    """Past the street, behind the crane, stepping down to the hills: low blocks, a shed with a sawtooth roof."""
    M = lambda k: np.array(MAT[k], np.float32)
    for (a, b_, top, mat, wins, lay) in ((646, 664, 280, 'stucco', (4.4, 6.0, 1.8, 3.0, 0.3), 2),
                                         (626, 649, 270, 'sand', (4.4, 6.0, 1.8, 3.0, 0.3), 2),
                                         (604, 630, 260, 'glass2', (3.6, 5.2, 2.2, 3.2, 0.4), 2),
                                         (586, 607, 254, 'brick', (4.0, 6.0, 1.8, 3.0, 0.42), 1),
                                         (566, 588, 246, 'concrete', (4.4, 6.0, 2.4, 2.4, 0.4), 1)):
        _block(P, a, b_, top, CITY_BASE + 1, M(mat) * 0.92, lay=lay, wins=wins, rng=rng)
        _cornice(P, a, b_, top, M(mat) * 0.92, lay=lay, h=1.3, over=0.5)
    _antenna(P, 620, 260, 249, lay=2, bars=2)
    _water_tower(P, 594, 254, M('wood'), lay=1, r=3.2, h=5)
    _tree(P, 571, 246, 2.0, rng, lay=1)
    _tree(P, 575.5, 246, 1.6, rng, lay=1)
    # the terminal shed on the quay, low, in front of them
    s, bx, by = P.box(564, 650, 276, CITY_BASE + 2)
    m = rect_mask(bx, by, 566, 646, 283, CITY_BASE + 1, P.px)
    for k in range(8):
        xa = 566 + 10 * k
        m = np.maximum(m, poly_mask(bx, by, [(xa, 283.5), (xa + 10, 283.5), (xa + 10, 277.5)], P.px))
    g = 1 - 0.14 * _stripe(bx - 566 - 9.75, 10.0, 0.5, P.px) * (by < 283.5)
    g *= 1 - 0.18 * _band(by, 288, 295, P.px) * _band((bx - 566) % 10, 2, 8, P.px)      # its doors
    P.paint(s, m, M('steel') * 1.05 * g[..., None], 0, rim=1.1)


def _quay(P):
    s, bx, by = P.box(REGION[0], REGION[1], 292, REGION[3])
    m = rect_mask(bx, by, REGION[0], 652, 295.6, YH + 1, P.px)
    g = 0.78 + 0.35 * _band(by, 295.6, 296.4, P.px) + 0.04 * P.n2[s]
    P.paint(s, m, np.array(MAT['concrete'], np.float32) * g[..., None], 0, rim=0)


class City:
    def __init__(self, X, Y, px):
        P = _Paint(X, Y, px)
        rng = np.random.default_rng(11)
        # the farthest row: plain shapes in the haze, a few slender towers
        _row(P, 3, np.random.default_rng(31), 40, 666, (5, 15), (252, 288), ('far',), None,
             dict(tower=0.1, antenna=0.15, dome=0.05, gable=0.1))
        # the middle distance: blocks with faint floors, domes, gables, tanks
        _row(P, 2, np.random.default_rng(32), 40, 666, (8, 20), (232, 278), ('far', 'stone', 'brick2', 'concrete'),
             (3.6, 5.5, 1.6, 2.4, 0.22), dict(tower=0.12, box=0.2, tank=0.12, dome=0.1, gable=0.12, antenna=0.2,
                                              cornice=0.5), gap=(-2, 4))
        # just behind the street: they fill its gaps and rise over its lower roofs
        _row(P, 1, np.random.default_rng(33), 40, 566, None, None,
             ('brick', 'stone', 'concrete', 'sand', 'glass2', 'stucco'),
             (4.0, 6.0, 1.8, 2.8, 0.34), dict(box=0.3, tank=0.15, gable=0.08, antenna=0.25, cornice=0.7),
             placed=NEAR_ROW)
        _landmarks(P, rng)
        _east(P, rng)
        # the street: the side walls first (each behind its neighbours), then the fronts, their roofs
        mats = {}
        for b in range(len(BUILDINGS)):
            mats[b] = np.array(MAT[STYLE[b]['mat']], np.float32) * STYLE[b]['tone']
            _side(P, b, mat=mats[b] * 0.95)
        for b in range(len(BUILDINGS)):
            _facade(P, b, rng)
            x0, x1, top = BUILDINGS[b]
            if b not in (6, 11):
                _cornice(P, x0, x1, top, mats[b], h=2.2 if STYLE[b]['kind'] != 'curtain' else 1.2,
                         over=0.8 if STYLE[b]['kind'] != 'curtain' else 0.3)
        _roofs(P, rng)
        _quay(P)
        _boats(P)
        self.P = P
        self.F = (np.clip((P.y - 24.0) / (YH - 24.0), 0, 1) ** 1.6).astype(np.float32)    # as film.py's sky
        self.coverage = np.zeros(X.shape, np.float32)          # how much of each pixel the city covers (0-1)
        self.coverage[P.sl] = P.A

    def draw(self, E, look):
        P = self.P
        e = E[P.sl]
        L, sl = look['L'], look['sunlit']
        hz = min(look['haze'], DAY_HAZE[0]) + DAY_HAZE[1] * max(look['haze'] - DAY_HAZE[0], 0.0)
        sx, sy = look['sun_xy']
        sm = lambda x: float(np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1)))
        # the haze's colour: the horizon's, and the sun's glow as the film paints it on the sky
        gl = 1.1 * sm((L - 0.08) / 0.4) * (1 - 0.2 * sm((L - 0.55) / 0.45)) + 0.5 * sl
        d = np.hypot(P.x - sx, (P.y - sy) * 1.4)
        late = sm((L - LATE[0]) / LATE[1])                 # the morning, once the sun is well up
        hz *= 1 - LATE[4] * late
        w = P.H0 * (1 - LATE[3] * late) + hz * P.HK
        g0 = gl * (0.5 * np.exp(-d / 70) + 0.35 * np.exp(-d / 240))
        glow = (g0 * w)[..., None] * GLOW
        rim = (look['sun'] * (RIM_SUN * sl) * (1 - RIM_HAZE * hz) + look['horizon'] * (0.6 * sm((L - 0.08) / 0.3))
               + look['sky_top'] * (0.8 * (1 - sl)))
        # the faces toward us are lit by the sky over the camera, not by the sun: cool, and darker than the sky
        city = (look['city'] * AMB[0] + AMB[1] * look['sky_top'] + AMB[2] * look['horizon']) * (1 - LATE[2] * late)
        # toward the sun the haze thickens: the glare of a low sun through morning air
        x_ = np.minimum(0.9, SUN_HAZE[0] * sl + SUN_HAZE[1] * look['haze']) * np.exp(-d / SUN_HAZE[2]) * P.A
        surf = city * (P.CT - hz * P.CK)
        surf += look['sky_top'] * P.RF[..., None]
        surf += rim * P.SN
        surf *= (1 - x_)[..., None]
        e *= (1 - P.A)[..., None]
        e += surf
        # the haze: the horizon's colour at dawn; later the sky's own at its height (the film's gradient), cooler
        wx = w + x_
        F = 1 - late * (1 - self.F)
        e += look['horizon'] * (wx * F)[..., None] + look['sky_top'] * (wx * (1 - F))[..., None]
        e += glow + (g0 * x_)[..., None] * GLOW
        self._steam(e, look)

    def _steam(self, e, look):
        """Steam from vents on the roofs behind the street, rising and drifting slowly; behind the street's
        buildings, so it never crosses their windows."""
        P, t = self.P, look['t']
        sl, hz = look['sunlit'], look['haze']
        lit = look['horizon'] * 0.55 + look['sky_top'] * 0.35 + look['sun'] * (0.35 * sl)
        shade = look['horizon'] * 0.4 + look['sky_top'] * 0.35
        for x0, y0, rise, drift, r0, rate, op in STEAM:
            s, gx, gy = P.box(x0 - 6, x0 + drift + 3 * r0 + 6, y0 - rise - 3 * r0, y0 + 2)
            if gx.size == 0:
                continue
            keep = np.ones_like(gx)
            light = np.zeros_like(gx)
            n = 9
            for k in range(n):
                a = (t * rate + k / n) % 1.0
                cx = x0 + drift * a ** 1.4 + 0.8 * np.sin(2.1 * t * rate * 6 + k)
                cy = y0 - rise * a
                r = r0 * (0.6 + 1.6 * a)
                o = op * np.sin(np.pi * a) ** 1.5
                g = np.exp(-((gx - cx) ** 2 + ((gy - cy) * 1.15) ** 2) / (2 * r * r))
                keep *= 1 - o * g
                light += o * g * np.clip(0.5 + (gx - cx) / (2 * r), 0, 1)
            al = (1 - keep) * (1 - P.A0[s])
            f = np.clip(light / np.maximum(1 - keep, 1e-4), 0, 1)[..., None]
            col = (shade * (1 - f) + lit * f) * (1 - 0.5 * hz) + look['horizon'] * (0.5 * hz)
            e[s] = e[s] * (1 - al[..., None]) + al[..., None] * col

    def draw_after(self, E, look):
        pass
