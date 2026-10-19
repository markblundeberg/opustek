"""The surfaces of the room in First Light's film: walls, ceiling, floor, the window's paint, the seat's cushion,
the side table's wood, the bookshelf, the picture on the left wall, the rug.

The room (room.py) asks for two things, for every sample of the frame that lands on one of these surfaces:
  albedo(M, p, n)   the colour (linear RGB, shape M.shape + (3,)) at points p (a tuple of 3 arrays) with unit
                    normals n (a tuple of 3 arrays), where M holds each sample's material id (ids.py). Fill only
                    the room's own materials (WALL, CEIL, FLOOR, WOOD, CUSHION, BOOKS, PAINT); the cat and the
                    props colour their own samples afterwards.
  bump(M, p, n)     the normals to light by: n itself, or n tilted by the surface's relief (grain, weave,
                    plaster); a tuple of 3 arrays, unit length. It is applied once per camera position.
Where things are (metres; x right, y up, z out of the window wall into the room; see room.py for the numbers):
the window wall is z = 0, the left wall x = -1.35, the right wall x = 2.05, the floor y = 0, the ceiling y = 2.5;
the window opening x -1.2..1.2, y 0.55..2.05, its reveal and frame and mullion and transom are PAINT; the seat
(PAINT) and its cushion (CUSHION, top y 0.55) run under the window; the side table (WOOD) is at x 1.29..1.75,
z 0.3..0.75; the bookshelf (BOOKS) is on the right wall, x 1.72..2.05, z 1.0..2.25, y 0..1.85.

What the camera sees of them (it stands at z 2.4-3.0, eye height 1.3, and sees about 34 degrees each side): the
window wall, the left wall from the corner to z ~1.0, the window's paint, the cushion's top and front, the seat's
front, the table's top, left side and front. Never the floor, the ceiling, the right wall or the bookshelf. The
picture hangs where it can be seen: on the left wall at PICTURE, in the shade beside the sun's patch.

How the surfaces are made. Every texture is band-limited to what a pixel can hold: noise is baked once (at the
first call) into periodic tiles by filtering white noise in frequency space, with each surface's tile cut off at
about three pixels of the surface as the camera sees it (finer along a surface seen edge-on, such as the left wall
and the cushion's top), and read with bilinear lookups. Relief comes in as the height field's gradient, baked with
the tile. Plaster: a fine trowelled stipple, a broad mottle of colour, soft shading into the room's corners. The
window's paint: smooth, with faint brush strokes along each member, rounded edges (the frame's members are moulded
toward the glass, the reveal's edge is a bullnose whose paint wraps onto the wall, worn in places), and shading
into its inside corners. The cushion: a slate linen, its threads running front to back, rounded and piped at the
edge, and pressed down where the cat lies; the cat and the props darken it around them (an occlusion map, baked
from the cat's and the props' own occludes() over the hemisphere above the cushion). The table: oak-like grain,
a drawer with a brass knob. The picture: a framed print of hills at dawn, baked as an image.

bump() works out both the normals and the colours and keeps the colours for the albedo() call that follows it on
the same samples; albedo() on part of a frame it has already coloured (room.with_cat's box) reads them back.
"""
import weakref

import numpy as np

from ids import WALL, CEIL, FLOOR, WOOD, CUSHION, BOOKS, PAINT, lin

f32 = np.float32
ALBEDO = {WALL: lin(.84, .76, .66), CEIL: lin(.86, .84, .8), FLOOR: lin(.56, .4, .28), WOOD: lin(.66, .5, .36),
          CUSHION: lin(.46, .52, .6), BOOKS: lin(.6, .5, .45), PAINT: lin(.86, .82, .74)}     # each surface's mean
ROOM_MATS = (WALL, CEIL, FLOOR, WOOD, CUSHION, BOOKS, PAINT)
PICTURE = (0.08, 0.48, 1.30, 1.62)          # the picture's frame on the left wall: z0, z1, y0, y1
PIC_FRAME, PIC_MAT = 0.02, 0.03             # its moulding's width, and the mat's
SKIRT = 0.12                                # the skirting board's height


# --- textures ---------------------------------------------------------------------------------------------------
class Tile:
    """A periodic texture of C channels, ny x nx texels of (tx, ty) metres, read bilinearly at points (a, b)."""

    def __init__(self, data, tx, ty):
        self.ny, self.nx, self.c = data.shape
        self.flat = np.ascontiguousarray(data.reshape(-1, self.c), f32)
        self.ix, self.iy = f32(1 / tx), f32(1 / ty)

    def __call__(self, a, b):
        u = a * self.ix
        v = b * self.iy
        iu = np.floor(u)
        iv = np.floor(v)
        fu = (u - iu)[:, None]
        fv = (v - iv)[:, None]
        mx, my = self.nx - 1, self.ny - 1
        i0 = iu.astype(np.int32) & mx
        i1 = (i0 + 1) & mx
        j0 = iv.astype(np.int32) & my
        j1 = (j0 + 1) & my
        j0 *= self.nx
        j1 *= self.nx
        t = self.flat
        a0 = np.take(t, j0 + i0, axis=0)
        a1 = np.take(t, j0 + i1, axis=0)
        a1 -= a0
        a1 *= fu
        a0 += a1
        b0 = np.take(t, j1 + i0, axis=0)
        b1 = np.take(t, j1 + i1, axis=0)
        b1 -= b0
        b1 *= fu
        b0 += b1
        b0 -= a0
        b0 *= fv
        a0 += b0
        return a0


class Grid:
    """A texture over one rectangle (not periodic): values clamped at its border."""

    def __init__(self, data, a0, b0, step):
        self.ny, self.nx, self.c = data.shape
        self.flat = np.ascontiguousarray(data.reshape(-1, self.c), f32)
        self.a0, self.b0, self.inv = f32(a0), f32(b0), f32(1 / step)

    def __call__(self, a, b):
        u = np.clip((a - self.a0) * self.inv, 0, self.nx - 1.001)
        v = np.clip((b - self.b0) * self.inv, 0, self.ny - 1.001)
        i0 = u.astype(np.int32)
        j0 = v.astype(np.int32)
        fu = (u - i0)[:, None]
        fv = (v - j0)[:, None]
        k = j0 * self.nx + i0
        t = self.flat
        a_ = np.take(t, k, axis=0)
        b_ = np.take(t, k + 1, axis=0)
        c_ = np.take(t, k + self.nx, axis=0)
        d_ = np.take(t, k + self.nx + 1, axis=0)
        top = a_ + (b_ - a_) * fu
        return top + ((c_ + (d_ - c_) * fu) - top) * fv


def _spectrum(ny, nx, tx, ty, lam_x, lam_y, lam_max, beta):
    """A band-pass envelope: power-law amplitude f^-beta, a soft floor at lam_max, Gaussian roll-off at lam_x, lam_y
    (the amplitude is 1/e at a wavelength of lam along each axis)."""
    fy = np.fft.fftfreq(ny, d=ty)[:, None]
    fx = np.fft.rfftfreq(nx, d=tx)[None, :]
    f = np.sqrt(fx * fx + fy * fy) + 1e-9
    env = np.exp(-(fx * lam_x) ** 2 - (fy * lam_y) ** 2) * (1 - np.exp(-(f * lam_max) ** 2)) * f ** -beta
    env[0, 0] = 0
    return env, fx, fy


def _noise(ny, nx, tx, ty, seed, lam_x, lam_y, lam_max, beta=1.0, shape=None):
    """Band-limited periodic noise: its value (unit std) and its gradient (scaled to unit RMS slope together)."""
    rng = np.random.default_rng(seed)
    env, fx, fy = _spectrum(ny, nx, tx, ty, lam_x, lam_y, lam_max, beta)
    F = np.fft.rfft2(rng.standard_normal((ny, nx))) * env
    if shape is not None:                      # a nonlinearity (plateaus, islands), then band-limited again
        h = np.fft.irfft2(F, s=(ny, nx))
        h = shape(h / h.std())
        F = np.fft.rfft2(h - h.mean()) * np.exp(-(fx * lam_x) ** 2 - (fy * lam_y) ** 2)
    h = np.fft.irfft2(F, s=(ny, nx))
    gx = np.fft.irfft2(F * (2j * np.pi * fx), s=(ny, nx))
    gy = np.fft.irfft2(F * (2j * np.pi * fy), s=(ny, nx))
    g = np.sqrt((gx ** 2 + gy ** 2).mean())
    return (h / h.std()).astype(f32), (gx / g).astype(f32), (gy / g).astype(f32)


_T = {}


def _bake_tiles():
    """The tiles, each holding just the channels its surfaces read, so that each sample costs one lookup."""
    if _T:
        return _T
    tanh = lambda v: np.tanh(1.3 * v)                      # noqa: E731  trowelled: soft plateaus
    abs_ = lambda v: np.abs(v)                             # noqa: E731  sweeps: creased where they meet
    for name, (nx, ny, tx, ty, lx, ly, seed) in {
            # the window wall, seen face on (about 2 mm a pixel): 4.1 m x 2 m, features from 8 mm
            'wall': (2048, 1024, 0.002, 0.002, 0.008, 0.008, 1),
            # the left wall, seen edge on: along z (a) a pixel spans ~5 mm, so its features start at 2 cm
            'wall_side': (1024, 1024, 0.005, 0.002, 0.02, 0.008, 11)}.items():
        _, fgx, fgy = _noise(ny, nx, tx, ty, seed, lx, ly, 0.12, 0.6, shape=tanh)
        _, bgx, bgy = _noise(ny, nx, tx, ty, seed + 1, 0.07, 0.07, 0.4, 1.0, shape=abs_)
        fc, _, _ = _noise(ny, nx, tx, ty, seed + 2, lx * 1.1, ly * 1.1, 0.06, 0.8)
        mott, _, _ = _noise(ny, nx, tx, ty, seed + 3, 0.12, 0.12, 1.5, 1.2)
        hue, _, _ = _noise(ny, nx, tx, ty, seed + 4, 0.2, 0.2, 2.0, 1.2)
        _T[name] = Tile(np.stack([0.018 * fgx + 0.032 * bgx, 0.018 * fgy + 0.032 * bgy,
                                  0.075 * mott + 0.02 * fc, 0.03 * hue], -1), tx, ty)
    # the cushion's linen (a: x, 2.5 mm; b: z on the top, then on down the front): threads long front to back,
    # slubs, a little lumpiness. Seen at ~15 degrees, along b a pixel spans 6-8 mm
    th, tgx, tgy = _noise(256, 1024, 0.0025, 0.004, 21, 0.008, 0.2, 0.1, 0.7)
    sl, _, _ = _noise(256, 1024, 0.0025, 0.004, 22, 0.012, 0.07, 0.2, 0.8)
    he, _, _ = _noise(256, 1024, 0.0025, 0.004, 24, 0.009, 0.03, 0.05, 0.5)
    m, mgx, mgy = _noise(256, 1024, 0.0025, 0.004, 23, 0.06, 0.06, 0.5, 1.2)
    _T['cushion'] = Tile(np.stack([0.045 * th + 0.035 * sl + 0.025 * he + 0.014 * m, 0.03 * m + 0.02 * sl,
                                   0.03 * tgx + 0.018 * mgx, 0.01 * tgy + 0.018 * mgy], -1), 0.0025, 0.004)
    # streaks: brush strokes, the wood's fine lines; long along b, fine across a (2 mm x 8 mm texels)
    s1, _, _ = _noise(1024, 1024, 0.002, 0.008, 9, 0.007, 0.05, 0.08, 0.7)
    s2, _, _ = _noise(1024, 1024, 0.002, 0.008, 10, 0.01, 0.08, 0.2, 0.8)
    _T['streak'] = Tile(np.stack([s1, s2], -1), 0.002, 0.008)
    # broad, slow: the wood's wander, the wobble of a hand-cut paint line (16 mm texels)
    w, _, _ = _noise(256, 256, 0.016, 0.016, 8, 0.07, 0.07, 0.6, 1.0)
    _T['broad'] = Tile(w[..., None], 0.016, 0.016)
    # chips: blobs of 8 mm and up, for worn paint
    c, _, _ = _noise(256, 256, 0.002, 0.002, 12, 0.008, 0.008, 0.05, 0.5)
    _T['chips'] = Tile(c[..., None], 0.002, 0.002)
    return _T


# --- the cushion's occlusion and the cat's dip, and the lamp's on the table -----------------------------------------
def _hemi(nd, top=0.94):
    """nd directions over the upper hemisphere, cosine weighted (a spiral), as unit vectors (x, y, z), none less
    than 20 degrees above the horizon (lower ones cast far, and would be cut off at the edge of the probes)."""
    k = np.arange(nd) + 0.5
    r = np.sqrt(k / nd) * top
    a = k * np.pi * (3 - np.sqrt(5))
    x, z = r * np.cos(a), r * np.sin(a)
    return np.stack([x, np.sqrt(1 - r * r), z], 1)


def _occluders():
    import cat as CATMOD
    import props as PROPS
    out = [(CATMOD.bound(0.0), lambda P, s: CATMOD.occludes(P, tuple(s), 0.0), 1.0)]
    for pr in PROPS.PROPS:
        out.append((pr.bound(), pr.occludes, 0.0))
    return out


def _contact_map(x0, x1, z0, z1, top, step=0.005, nd=32, dip=True):
    """Over the rectangle (x, z) of a top surface at height top: the share of the sky above it that the cat and the
    props leave open (cosine weighted), and the dip the cat's weight presses into it (its height's gradient)."""
    from scipy.ndimage import gaussian_filter
    xs = np.arange(x0, x1 + step / 2, step)
    zs = np.arange(z0, z1 + step / 2, step)
    X, Z = np.meshgrid(xs, zs)
    occ = np.zeros(X.shape)
    foot = np.zeros(X.shape)
    dirs = _hemi(nd)
    for (c, r), occludes, heavy in _occluders():
        if not dip:
            heavy = 0
        near = np.hypot(X - c[0], Z - c[2]) < r + 0.2
        if not near.any() or c[1] - r > top + 0.6 or c[1] + r < top - 0.02:
            continue
        P = np.stack([X[near], np.full(near.sum(), top - 0.008), Z[near]], 1).astype(f32)
        acc = np.zeros(near.sum())
        for s in dirs:
            acc += occludes(P, s).astype(float) / nd
        occ[near] = np.maximum(occ[near], acc)
        if heavy:
            f = occludes(P, np.array([0.0, 1.0, 0.0])).astype(float)
            foot[near] = np.maximum(foot[near], f * heavy)
    vis = 1 - gaussian_filter(occ, 0.012 / step)
    h = -gaussian_filter(foot, 0.022 / step) * 0.025               # metres: about 1 cm at the cat's outline
    if foot.any():                                                 # and soft creases, drawn in toward it
        from scipy.ndimage import distance_transform_edt
        d = distance_transform_edt(foot < 0.5) * step
        w = foot.sum()
        cx, cz = (X * foot).sum() / w, (Z * foot).sum() / w
        th = np.arctan2((Z - cz) * 1.6, X - cx)
        crease = np.cos(11 * th + 1.7 * np.sin(3 * th + 1.0)) * np.exp(-d / 0.05) * (1 - np.exp(-d / 0.012))
        h += gaussian_filter(crease, 1.5) * 0.0025
    gz, gx = np.gradient(h, step)
    return Grid(np.stack([vis, gx, gz], -1), x0, z0, step)


# --- the picture: hills at dawn, a print in a mat and a dark frame ---------------------------------------------------
def _picture():
    """The picture as a grid on the left wall (a: z, b: y, 1 mm): RGB inside its frame, its coverage, the shadow it
    casts on the wall around it, and its frame's relief (d/dz, d/dy)."""
    from scipy.ndimage import gaussian_filter
    z0, z1, y0, y1 = PICTURE
    mg = 0.04
    step = 0.001
    zs = np.arange(z0 - mg, z1 + mg, step)
    ys = np.arange(y0 - mg, y1 + mg, step)
    Z, Y = np.meshgrid(zs, ys)
    # the frame and the mat, by distance in from the outer edge
    e = np.minimum(np.minimum(Z - z0, z1 - Z), np.minimum(Y - y0, y1 - Y))
    inside = e > 0
    rgb = np.zeros(Z.shape + (3,))
    rgb[:] = lin(.86, .83, .76)                                              # the mat
    wood = lin(.25, .16, .11)
    fr = inside & (e < PIC_FRAME)
    rgb[fr] = wood
    lip = inside & (e >= PIC_FRAME - 0.004) & (e < PIC_FRAME)
    rgb[lip] = lin(.62, .5, .32)                                             # a gilt inner lip
    a0 = PIC_FRAME + PIC_MAT
    art = e > a0
    rgb[(e > a0 - 0.002) & ~art] = lin(.97, .95, .9)                          # the mat's cut, its white core
    # the print, in its own coordinates: u left to right (z falling), v bottom to top
    u = (z1 - a0 - Z) / (z1 - z0 - 2 * a0)
    v = (Y - y0 - a0) / (y1 - y0 - 2 * a0)
    t = np.clip((v - 0.3) / 0.7, 0, 1)[..., None]
    sky = lin(.95, .76, .6) * (1 - t) + lin(.62, .68, .8) * t
    img = sky.copy()
    sd = np.hypot((u - 0.66) * 1.45, v - 0.6)
    img = np.where((sd < 0.13)[..., None], lin(.97, .66, .42), img)
    img = np.where(((sd >= 0.13) & (sd < 0.2))[..., None], img * 0.85 + lin(.98, .8, .6) * 0.15, img)
    ridges = [(0.5, 0.09, lin(.62, .6, .7), 1.3), (0.38, 0.08, lin(.42, .47, .58), 2.1),       # far to near hills
              (0.27, 0.06, lin(.26, .35, .38), 3.3)]
    for base, amp, col, fq in ridges:
        top = base + amp * (np.sin(u * 5.1 * fq + fq) * 0.6 + np.sin(u * 11.3 * fq + 2 * fq) * 0.25
                            + np.sin(u * 2.3 + fq) * 0.5)
        img = np.where((v < top)[..., None], col, img)
    lake = v < 0.16
    lk = lin(.56, .62, .72) * (0.85 + 0.15 * v[..., None] / 0.16)
    glint = np.exp(-((u - 0.66) / 0.05) ** 2) * (0.5 + 0.5 * np.cos(v * 90)) * (v < 0.14)
    lk = lk + (lin(.98, .78, .55) - lk) * (0.8 * glint)[..., None]
    img = np.where(lake[..., None], lk, img)
    shore = (v >= 0.16) & (v < 0.172)
    img = np.where(shore[..., None], lin(.2, .26, .28), img)
    rgb = np.where(art[..., None], img, rgb)
    A = inside.astype(float)
    # relief: the moulding rises in a round from both its edges; the mat sits a little below the frame
    ef = np.clip(e / PIC_FRAME, 0, 1)
    hgt = np.where(inside & (e < PIC_FRAME), 0.008 * np.sin(np.pi * ef) ** 0.7, 0.0)
    # the shadow the frame throws on the wall, away from the window and down
    sh = gaussian_filter(np.roll(np.roll(inside.astype(float), -7, 0), 9, 1), 7)
    shadow = 1 - 0.3 * sh
    # band-limit to the wall as seen: about 5 mm a pixel along z, 2 mm along y
    sig = (1.0, 2.2)
    rgb = np.stack([gaussian_filter(rgb[..., k], sig) for k in range(3)], -1)
    A = gaussian_filter(A, sig)
    hgt = gaussian_filter(hgt, sig)
    gy, gz = np.gradient(hgt, step)
    data = np.concatenate([rgb, A[..., None], shadow[..., None], gz[..., None], gy[..., None]], -1)
    return Grid(data, z0 - mg, y0 - mg, step)


# --- shapes -----------------------------------------------------------------------------------------------------
def _ss(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _round(e, r, smax):
    """The slope of a rounded edge at distance e in from it (radius r): smax at the edge, 0 from r on."""
    t = np.clip(1 - e * f32(1 / r), 0, 1)
    return smax * t * t


def _corner(e, r, k):
    """Shading into an inside corner at distance e from it: 1 - k at the corner, easing out over about r."""
    return 1 - k / (1 + np.maximum(e, 0) * f32(1 / r)) ** 2


_G = {}


def _geom():
    if not _G:
        import room as R
        _G.update(WIN=R.WIN, DEPTH=R.DEPTH, FRAME=R.FRAME, MULL=R.MULL, TRANSOM=R.TRANSOM, SEAT=R.SEAT, CUSH=R.CUSH,
                  TABLE=R.TABLE, X0=R.X0, X1=R.X1, Y1=R.Y1)
    return _G


_MAPS = {}


def _maps():
    if not _MAPS:
        g = _geom()
        (cx0, _, cz0), (cx1, ctop, cz1) = g['CUSH']
        _MAPS['cushion'] = _contact_map(cx0, cx1, cz0, cz1, ctop)
        (tx0, _, tz0), (tx1, ttop, tz1) = g['TABLE']
        _MAPS['table'] = _contact_map(tx0, tx1, tz0, tz1, ttop, dip=False)
        _MAPS['picture'] = _picture()
    return _MAPS


# --- the surfaces -----------------------------------------------------------------------------------------------
class Out:
    """One group of samples: their points and normals, and what the surface makes of them."""

    def __init__(self, idx, p, n):
        self.idx = idx
        self.x, self.y, self.z = (np.take(a, idx) for a in p)
        self.nx, self.ny, self.nz = (np.take(a, idx) for a in n)
        self.tx = np.zeros(len(idx), f32)          # the tilt added to the normal
        self.ty = np.zeros(len(idx), f32)
        self.tz = np.zeros(len(idx), f32)
        self.A = None


def _plaster(o, a, b, side, base):
    """Plaster on a wall: (a, b) the wall's own axes. Its colour, and its relief as slopes along a and b."""
    f = _bake_tiles()['wall_side' if side else 'wall'](a, b)
    mott = 1 + f[:, 2]
    A = np.empty((len(a), 3), f32)
    A[:, 0] = base[0] * mott * (1 + f[:, 3])
    A[:, 1] = base[1] * mott
    A[:, 2] = base[2] * mott * (1 - f[:, 3])
    return A, f[:, 0].copy(), f[:, 1].copy()


def _walls(o, kind):
    """The window wall (kind 'back', axes x, y), the left wall ('left', axes z, y), the right wall ('right')."""
    g = _geom()
    W = g['WIN']
    base = ALBEDO[WALL]
    if kind == 'back':
        a, b = o.x, o.y
    else:
        a, b = o.z, o.y
    A, sa, sb = _plaster(o, a, b, kind != 'back', base)
    # soft shading into the corners: the other walls, the ceiling, the floor (the room's light falls off there)
    sh = _corner(g['Y1'] - o.y, 0.09, 0.16) * _corner(o.y, 0.09, 0.1)
    if kind == 'back':
        sh *= _corner(o.x - g['X0'], 0.09, 0.16) * _corner(g['X1'] - o.x, 0.09, 0.16)
    else:
        sh *= _corner(o.z, 0.09, 0.16)
    # the window's opening: the reveal's paint wraps round a bullnose onto the wall
    if kind == 'back':
        cx = np.clip(o.x, W[0], W[1]) - o.x
        cy = np.clip(o.y, W[2], W[3]) - o.y
        e = np.sqrt(cx * cx + cy * cy) + f32(1e-6)
        near = e < 0.03
        if near.any():
            k = np.nonzero(near)[0]
            ek = e[k]
            s = _round(ek, 0.011, 2.2) / ek
            sa[k] -= s * cx[k]                                            # the normal leans toward the opening
            sb[k] -= s * cy[k]
            T = _bake_tiles()
            jit = T['broad'](o.x[k] * 3.0, o.y[k] * 3.0)[:, 0]
            edge = f32(0.0125) + f32(0.0015) * jit                       # where the paint stops: a hand-cut line
            cov = _ss((edge - ek) / f32(0.0025) + 0.5)
            along = np.where(np.abs(cy[k]) > np.abs(cx[k]), o.x[k], o.y[k])
            paint = ALBEDO[PAINT][None, :] * _wear(along, ek * 0, ek)[:, None]
            A[k] = A[k] * (1 - cov[:, None]) + paint * cov[:, None]
            # the bullnose's crest catches a little more of the room's light: a soft sheen
            A[k] *= (1 + f32(0.05) * np.exp(-((ek - 0.004) / 0.003) ** 2))[:, None]
    # a skirting board
    sk = o.y < SKIRT
    if sk.any():
        k = np.nonzero(sk)[0]
        A[k] = ALBEDO[PAINT] * _corner(SKIRT - o.y[k], 0.004, 0.25)[:, None]
        sb[k] -= _round(SKIRT - o.y[k], 0.008, 1.0)
    if kind == 'left':
        _hang_picture(o, A, sa, sb)
    A *= sh[:, None]
    o.A = A
    if kind == 'back':
        o.tx, o.ty = -sa, -sb
    else:
        o.tz, o.ty = -sa, -sb


def _hang_picture(o, A, sa, sb):
    z0, z1, y0, y1 = PICTURE
    m = 0.04
    sel = (o.z > z0 - m) & (o.z < z1 + m) & (o.y > y0 - m) & (o.y < y1 + m)
    if not sel.any():
        return
    k = np.nonzero(sel)[0]
    P = _maps()['picture'](o.z[k], o.y[k])
    cov = np.clip(P[:, 3], 0, 1)[:, None]
    A[k] = A[k] * P[:, 4:5] * (1 - cov) + P[:, :3] * cov
    sa[k] = sa[k] * (1 - cov[:, 0]) + P[:, 5]
    sb[k] = sb[k] * (1 - cov[:, 0]) + P[:, 6]


def _paint_colour(a, b):
    """The window's cream paint, with faint brush strokes along b (a across them)."""
    T = _bake_tiles()
    s = T['streak'](a + 7.0, b + 3.0)
    v = 1 + f32(0.018) * s[:, 0] + f32(0.012) * s[:, 1]
    return ALBEDO[PAINT][None, :] * v[:, None]


def _wear(a, b, e):
    """Worn paint at an edge's crest (e from the corner): here and there a soft chip, the undercoat showing."""
    T = _bake_tiles()
    s = T['chips'](a * 0.7 + 11.0, b * 0.7 + 5.0)[:, 0]
    chip = _ss((s - 1.9) / 0.6) * np.exp(-(e / 0.007) ** 2)
    return 1 - f32(0.22) * chip


def _paint(o):
    """The window's paint: the frame at the glass (with mullion and transom), the reveals, the seat."""
    g = _geom()
    W, D, FR, MU, TR = g['WIN'], g['DEPTH'], g['FRAME'], g['MULL'], g['TRANSOM']
    A = np.empty((len(o.idx), 3), f32)
    sh = np.ones(len(o.idx), f32)
    glass = o.z < -D + 2e-4
    seat = o.z > 0.05
    rev = ~glass & ~seat
    # --- the frame, mullion and transom, at the glass: each member rounds down toward the glass
    k = np.nonzero(glass)[0]
    if len(k):
        x, y = o.x[k], o.y[k]
        gx0, gx1, gy0, gy1 = W[0] + FR, W[1] - FR, W[2] + FR, W[3] - FR
        mc, mw = (MU[0] + MU[1]) / 2, (MU[1] - MU[0]) / 2
        tc, tw = (TR[0] + TR[1]) / 2, (TR[1] - TR[0]) / 2
        cand = [(gx0 - x, -1, 0), (x - gx1, 1, 0), (gy0 - y, 0, -1), (y - gy1, 0, 1),
                (mw - np.abs(x - mc), 'mx', 0), (tw - np.abs(y - tc), 0, 'ty')]
        e = np.full(len(k), -1.0, f32)
        dx = np.zeros(len(k), f32)
        dy = np.zeros(len(k), f32)
        for ee, ax, ay in cand:
            take = ee > e
            e = np.where(take, ee, e)
            if ax == 'mx':
                dx = np.where(take, np.sign(x - mc), dx)
                dy = np.where(take, 0, dy)
            elif ay == 'ty':
                dx = np.where(take, 0, dx)
                dy = np.where(take, np.sign(y - tc), dy)
            else:
                dx = np.where(take, -ax, dx)
                dy = np.where(take, -ay, dy)
        e = np.maximum(e, 0)
        s = _round(e, 0.012, 1.6)
        o.tx[k] = s * dx
        o.ty[k] = s * dy
        vert = np.abs(dx) > 0.5
        A[k] = _paint_colour(np.where(vert, x, y), np.where(vert, y, x))
        out_e = np.minimum(np.minimum(x - W[0], W[1] - x), np.minimum(y - W[2], W[3] - y))   # to the reveals
        shade = _corner(out_e, 0.006, 0.3) * (1 - f32(0.2) * (s / 1.6))        # the round turns from the room
        shade *= 1 + f32(0.05) * np.exp(-((e - 0.013) / 0.004) ** 2)           # its shoulder: a soft sheen
        putty = 1 - f32(0.12) * _ss(1 - e / f32(0.0035))                      # the glazing's line at the glass
        sh[k] = shade * putty
    # --- the reveals: a bullnose where they meet the wall, shading into their corners
    k = np.nonzero(rev)[0]
    if len(k):
        x, y, z = o.x[k], o.y[k], o.z[k]
        side = np.abs(np.abs(x) - W[1]) < np.minimum(np.abs(y - W[2]), np.abs(y - W[3]))
        along = np.where(side, y, x)
        ez = -z                                                    # in from the wall's face
        s = _round(ez, 0.011, 2.2)
        o.tz[k] = s
        A[k] = _paint_colour(ez * 0.4, along) * _wear(along, ez * 0, ez)[:, None]
        e_end = np.where(side, np.minimum(y - W[2], W[3] - y), np.minimum(x - W[0], W[1] - x))
        sh[k] = _corner(z + D, 0.008, 0.3) * _corner(e_end, 0.01, 0.22)
        sh[k] *= 1 + f32(0.05) * np.exp(-((ez - 0.004) / 0.003) ** 2)
    # --- the seat: a panelled front, its top's edge rounded, shading in under the cushion
    k = np.nonzero(seat)[0]
    if len(k):
        _seat(o, k, A, sh)
    o.A = A * sh[:, None]


def _seat(o, k, A, sh):
    g = _geom()
    (sx0, _, _), (sx1, stop, sz1) = g['SEAT']
    cz1 = g['CUSH'][1][2]
    x, y, z = o.x[k], o.y[k], o.z[k]
    front = z > sz1 - 2e-4
    top = ~front & (y > stop - 2e-4)
    A[k] = _paint_colour(np.where(front, y, z), x)
    f = np.nonzero(front)[0]
    if len(f):
        kf = k[f]
        o.ty[kf] = _round(stop - y[f], 0.01, 2.0)            # the top's edge
        sh[kf] = _seat_front(x[f], y[f], sx0, sx1, stop)
    t = np.nonzero(top)[0]
    if len(t):
        kt = k[t]
        zt = z[t]
        o.tz[kt] = _round(sz1 - zt, 0.01, 2.0)
        cx1 = g['CUSH'][1][0]
        e = np.maximum(zt - cz1, np.abs(x[t]) - cx1)
        # in under the cushion's edge; and the cushion hides the window from the strip in front of it, which the
        # room's light (sky and sun, through the opening) doesn't know
        sh[kt] = _corner(e, 0.006, 0.45) * (1 - f32(0.32) * np.exp(-np.maximum(e, 0) * f32(1 / 0.03)))


def _gauss(v, c, w):
    return np.exp(-((v - c) * f32(1 / w)) ** 2)


def _seat_front(x, y, sx0, sx1, stop):
    """The seat's front, as the light paints it: it faces the room, lit only by the room's bounce, so its relief is
    drawn into the colour, lit from above. From the top: a nosing (the seat's top, its lip rounded under), the
    shadow it casts, a bead, the top rail, then four sunk panels in moulded frames between stiles. The camera sees
    only its top 10-13 cm, so that is where the detail is. Every line is at least 2 mm soft: it holds still."""
    d = stop - y                                           # down from the top
    # the nosing's face, brighter toward its top; its rounded underside turning away; the shadow it throws
    v = 1 + f32(0.1) * _ss((0.02 - d) * f32(1 / 0.006) + 0.5) * (1 - d * f32(1 / 0.026))
    v *= 1 - f32(0.18) * _gauss(d, 0.021, 0.003)
    under = np.maximum(d - 0.0245, 0)
    v *= 1 - f32(0.55) * _ss((d - 0.0245) * f32(1 / 0.0032) + 0.5) * np.exp(-under * f32(1 / 0.009))
    # a bead under it: lit above its crown, shaded below
    v *= (1 + f32(0.1) * _gauss(d, 0.036, 0.0032)) * (1 - f32(0.18) * _gauss(d, 0.0425, 0.0032))
    # the panels: four between stiles (wider at the ends), their tops 6 cm down, their bottoms 8 cm up
    nps, stile, end = 4, 0.08, 0.1
    wpan = (sx1 - sx0 - 2 * end - (nps - 1) * stile) / nps
    u = np.clip(x - sx0 - end, -end, None)
    uu = u % (wpan + stile)
    ex = np.where(uu <= wpan, np.minimum(uu, wpan - uu), -np.minimum(uu - wpan, wpan + stile - uu))
    ex = np.where(u < 0, u, ex)                            # in from the panel's sides (< 0 on a stile)
    ex = np.where(x > sx1 - end, np.minimum(ex, sx1 - end - x), ex)
    et = stop - 0.06 - y                                   # in from its top
    eb = y - 0.08                                          # and its bottom
    ep = np.minimum(ex, np.minimum(et, eb))
    topw = _ss((np.minimum(ex, eb) - et) * f32(1 / 0.003) + 0.5)          # the top edge (lit from above) or a side
    inside = _ss(ep * f32(1 / 0.0032) + 0.5)
    deep = np.maximum(ep, 0)
    v *= 1 + f32(0.15) * _gauss(ep, -0.0035, 0.003)                        # the frame's arris catching the light
    v *= 1 - (f32(0.42) + f32(0.2) * topw) * inside * np.exp(-deep * f32(1 / 0.008))     # the step's shadow
    v *= 1 + f32(0.16) * topw * _gauss(ep, 0.017, 0.004)                 # the moulding's face, turned up
    v *= 1 - f32(0.2) * _gauss(ep, 0.027, 0.003)                          # its fillet, where the panel begins
    v *= 1 - f32(0.14) * _ss((ep - 0.027) * f32(1 / 0.004) + 0.5)         # the panel, set back in shade
    v *= _corner(x - sx0, 0.03, 0.15)                     # in toward the room's corner
    return v


def _cushion(o):
    """A slate linen cushion: threads front to back, a rounded and piped edge, pressed down where the cat lies."""
    g = _geom()
    (x0, y0, z0), (x1, y1, z1) = g['CUSH']
    T = _bake_tiles()
    top = (y1 - o.y) < (z1 - o.z)
    front = ~top & (o.z > z1 - 2e-4)
    A = np.empty((len(o.idx), 3), f32)
    a = o.x
    b = np.where(top, o.z, o.y * 4.0)                       # the threads run front to back, then down the front
    c = T['cushion'](a + 0.37, b + 1.3)
    yarn = 1 + c[:, 0]
    tint = c[:, 1]
    base = ALBEDO[CUSHION]
    A[:, 0] = base[0] * yarn * (1 - tint)
    A[:, 1] = base[1] * yarn * (1 + 0.3 * tint)
    A[:, 2] = base[2] * yarn * (1 + tint)
    sa, sb = c[:, 2], c[:, 3]
    sh = np.ones(len(o.idx), f32)
    k = np.nonzero(top)[0]
    if len(k):
        x, z = o.x[k], o.z[k]
        R = 0.04
        o.tz[k] = _round(z1 - z, R, 1.6) - _round(z - z0, R, 1.6) - sb[k]
        o.tx[k] = _round(x1 - x, R, 1.6) - _round(x - x0, R, 1.6) - sa[k]
        m = _maps()['cushion'](x, z)
        o.tx[k] -= m[:, 1]
        o.tz[k] -= m[:, 2]
        vis = np.clip(m[:, 0], 0, 1)
        sh[k] = 0.35 + 0.65 * vis ** 1.5
        # the piping along the front edge: a cord, a crease inside it
        ep = z1 - z
        sh[k] *= 1 - f32(0.16) * np.exp(-((ep - 0.009) / 0.0025) ** 2)
        sh[k] *= 1 + f32(0.05) * np.exp(-((ep - 0.004) / 0.0025) ** 2)
    k = np.nonzero(front)[0]
    if len(k):
        x, y = o.x[k], o.y[k]
        v = (y - (y0 + y1) / 2) / ((y1 - y0) / 2)
        o.ty[k] = f32(1.4) * v * np.abs(v)
        o.tx[k] = _round(x1 - x, 0.03, 1.5) - _round(x - x0, 0.03, 1.5) - sa[k]
        sh[k] = _corner(y - y0, 0.006, 0.35) * (1 + f32(0.06) * np.exp(-((y1 - y - 0.004) / 0.003) ** 2))
    o.A = A * sh[:, None]


def _wood(o):
    """The side table, oak-like: its top's grain runs front to back, its sides' up and down; a drawer, a knob."""
    g = _geom()
    (x0, y0, z0), (x1, y1, z1) = g['TABLE']
    T = _bake_tiles()
    dt, dl, df = y1 - o.y, o.x - x0, z1 - o.z            # how far from each face's plane
    top = (dt <= dl) & (dt <= df)
    left = ~top & (dl <= df)
    front = ~top & ~left
    # across the grain (a) and along it (b), for each face; the left side is seen edge on, so its grain is broader
    a = np.where(top, o.x, np.where(left, o.z, o.y))
    b = np.where(top, o.z, np.where(left, o.y, o.x))
    spacing = np.where(top, f32(0.012), np.where(left, f32(0.02), f32(0.01)))
    fine = np.where(left, f32(0.5), f32(1.0))
    w = T['broad'](a * 0.6 + 2.0, b * 0.15 + 4.0)[:, 0]
    st = T['streak'](a * fine + 0.71, b * 0.5 + 0.3)
    phase = a / spacing + f32(0.8) * w + f32(0.12) * st[:, 1]
    ring = f32(0.5) + f32(0.5) * np.cos(f32(2 * np.pi) * phase)
    tone = (1 - f32(0.14) * (ring - 0.5)) * (1 + f32(0.045) * st[:, 0]) * (1 + f32(0.04) * w)
    base = ALBEDO[WOOD]
    A = np.empty((len(o.idx), 3), f32)
    A[:, 0] = base[0] * tone
    A[:, 1] = base[1] * tone * (1 - f32(0.03) * (ring - 0.5))
    A[:, 2] = base[2] * tone * (1 - f32(0.06) * (ring - 0.5))
    sl = -f32(0.03) * np.sin(f32(2 * np.pi) * phase)          # the late wood sits a hair lower: a faint relief
    sh = np.ones(len(o.idx), f32)
    k = np.nonzero(top)[0]
    if len(k):
        x, z = o.x[k], o.z[k]
        o.tx[k] = _round(x1 - x, 0.006, 1.6) - _round(x - x0, 0.006, 1.6) - sl[k]
        o.tz[k] = _round(z1 - z, 0.006, 1.6) - _round(z - z0, 0.006, 1.6)
        m = _maps()['table'](x, z)
        sh[k] = 0.45 + 0.55 * np.clip(m[:, 0], 0, 1) ** 1.5
    for face in (left, front):
        k = np.nonzero(face)[0]
        if not len(k):
            continue
        y = o.y[k]
        o.ty[k] = _round(y1 - y, 0.006, 1.6)
        joint = 1 - f32(0.3) * np.exp(-((y - (y1 - 0.028)) / 0.0018) ** 2)    # the top meets the carcass
        sh[k] = joint * _corner(y, 0.01, 0.2)
        if face is left:
            o.tz[k] = -sl[k]
        else:
            o.ty[k] -= sl[k]
    k = np.nonzero(front)[0]
    if len(k):
        x, y = o.x[k], o.y[k]
        dx0, dx1, dy0, dy1 = x0 + 0.03, x1 - 0.03, y1 - 0.2, y1 - 0.045       # the drawer's front
        ed = np.minimum(np.minimum(x - dx0, dx1 - x), np.minimum(y - dy0, dy1 - y))
        sh[k] *= 1 - f32(0.45) * np.exp(-(ed / 0.0016) ** 2)
        kx, ky, kr = (dx0 + dx1) / 2, (dy0 + dy1) / 2, 0.013
        rr = np.hypot(x - kx, y - ky)
        knob = _ss((kr - rr) / 0.0025 + 0.5)
        brass = lin(.74, .56, .3)
        A[k] = A[k] * (1 - knob[:, None]) + brass * knob[:, None]
        q = np.clip(rr / kr, 0, 0.95)
        dome = q / np.sqrt(1 - q * q) * knob
        o.tx[k] += dome * (x - kx) / (rr + 1e-6)
        o.ty[k] += dome * (y - ky) / (rr + 1e-6)
        sh[k] *= 1 - f32(0.25) * np.exp(-((rr - kr - 0.002) / 0.003) ** 2) * (y < ky)
    o.A = A * sh[:, None]


def _books(o):
    """The bookshelf (out of the camera's view; kept simple): rows of spines, each its own height and colour."""
    py, pz = o.y, o.z
    A = np.empty((len(o.idx), 3), f32)
    A[:] = ALBEDO[BOOKS]
    face = o.nx < -0.5
    if face.any():
        row = np.floor(py / 0.37)
        spine = np.floor((pz - 1.1) / 0.045 + 7 * row)
        hue = (np.sin(spine * 12.9898) * 43758.5453) % 1
        cols = np.array([lin(.6, .38, .33), lin(.42, .48, .56), lin(.74, .66, .5), lin(.46, .54, .46),
                         lin(.58, .52, .6)])
        books = 0.65 * cols[(hue * 5).astype(int) % 5] + 0.35 * ALBEDO[WALL]
        tall = 0.22 + 0.1 * ((np.sin(spine * 78.233) * 12543.1) % 1)
        empty = ((np.sin(np.floor((pz - 1.0) / 0.3 + 5 * row) * 4.7) * 9871.3) % 1) < 0.3
        band = np.abs((py % 0.37) - tall * 0.82) < 0.008
        gap = ((py % 0.37) > tall) | empty
        bk = np.where(band[:, None], books * 0.8, books)
        A[face] = np.where(gap[face][:, None], ALBEDO[WOOD] * 0.7, bk[face])
    o.A = A


def _floor(o):
    px, pz = o.x, o.z
    A = np.empty((len(o.idx), 3), f32)
    A[:] = ALBEDO[FLOOR]
    rug = (px > -0.9) & (px < 1.05) & (pz > 0.75) & (pz < 2.8)
    A[rug] = lin(.62, .48, .42)
    A[rug & ((np.abs(px + 0.9) < 0.06) | (np.abs(px - 1.05) < 0.06) | (np.abs(pz - 0.75) < 0.06))] = lin(.5, .36, .33)
    planks = ~rug & (np.abs(((px + 3) % 0.16) - 0.08) > 0.075)
    A[planks] *= 0.92
    o.A = A


def _ceiling(o):
    g = _geom()
    A = np.empty((len(o.idx), 3), f32)
    A[:] = ALBEDO[CEIL]
    sh = _corner(o.z, 0.1, 0.15) * _corner(o.x - g['X0'], 0.1, 0.15) * _corner(g['X1'] - o.x, 0.1, 0.15)
    o.A = A * sh[:, None]


def _surfaces(M, p, n, want_normals=True):
    """The colours of the room's own samples (others left 0) and, if asked, the normals tilted by relief. Faces
    are told apart by where the samples are, not by their normals, which bump() has tilted."""
    g = _geom()
    shape = M.shape
    Mf = M.reshape(-1)
    pf = tuple(np.asarray(a).reshape(-1) for a in p)
    nf = tuple(np.asarray(a, f32).reshape(-1) for a in n)
    A = np.zeros((Mf.size, 3), f32)
    out_n = [a.copy() for a in nf] if want_normals else None
    order = np.argsort(Mf, kind='stable')
    ends = np.cumsum(np.bincount(Mf.astype(np.intp), minlength=max(ROOM_MATS) + 1))
    for m in ROOM_MATS:
        lo, hi = (ends[m - 1] if m else 0), ends[m]
        if hi <= lo:
            continue
        idx = order[lo:hi]
        if m == WALL:
            x = np.take(pf[0], idx)
            left = x < g['X0'] + 2e-4
            right = x > g['X1'] - 2e-4
            for gi, kind in ((idx[~left & ~right], 'back'), (idx[left], 'left'), (idx[right], 'right')):
                if len(gi):
                    o = Out(gi, pf, nf)
                    _walls(o, kind)
                    _store(o, A, out_n)
            continue
        o = Out(idx, pf, nf)
        {PAINT: _paint, CUSHION: _cushion, WOOD: _wood, BOOKS: _books, FLOOR: _floor, CEIL: _ceiling}[m](o)
        _store(o, A, out_n)
    A = A.reshape(shape + (3,))
    if want_normals:
        return tuple(a.reshape(shape) for a in out_n), A
    return None, A


def _store(o, A, out_n):
    A[o.idx] = o.A
    if out_n is None:
        return
    nx, ny, nz = o.nx + o.tx, o.ny + o.ty, o.nz + o.tz
    r = 1 / np.sqrt(nx * nx + ny * ny + nz * nz)
    out_n[0][o.idx] = nx * r
    out_n[1][o.idx] = ny * r
    out_n[2][o.idx] = nz * r


# --- the interface, and what it keeps between calls ---------------------------------------------------------------
_PENDING = []          # (M, p[0], A) from the last bump(), for the albedo() call on the same samples
_SEEN = []             # weak references to whole frames already coloured: (M, p[0], A)


def bump(M, p, n):
    nn, A = _surfaces(M, p, n)
    _PENDING[:] = [(M, p[0], A)]
    return nn


def albedo(M, p, n):
    """Colour by material and place: plaster, paint, linen, wood, the books, a picture, a rug."""
    if _PENDING:
        m0, p0, A = _PENDING.pop()
        if m0 is M and p0 is p[0]:
            live = [s for s in _SEEN if s[0]() is not None][-2:]
            _SEEN[:] = live + [(weakref.ref(M), weakref.ref(p[0]), weakref.ref(A))]
            return A
    A = _from_seen(M, p)
    if A is not None:
        return A
    return _surfaces(M, p, n, want_normals=False)[1]


def _from_seen(M, p):
    """The colours of a box of a frame already coloured (the samples there are a view of it), or None."""
    base = M.base
    if base is None or M.ndim != 2:
        return None
    for wm, wp, wa in _SEEN:
        bm, bp, ba = wm(), wp(), wa()
        if bm is None or bp is None or ba is None or bm is not base or base.ndim != 2:
            continue
        off = M.__array_interface__['data'][0] - base.__array_interface__['data'][0]
        r0, c0 = divmod(off, base.strides[0])
        c0 //= base.strides[1]
        h, w = M.shape
        if r0 < 0 or c0 < 0 or r0 + h > base.shape[0] or c0 + w > base.shape[1]:
            continue
        box = (slice(r0, r0 + h), slice(c0, c0 + w))
        # the points must be those the colours were made for, where the samples are still the room's
        room = np.isin(M, ROOM_MATS)
        ii = np.nonzero(room.reshape(-1))[0]
        if len(ii):
            probe = ii[np.linspace(0, len(ii) - 1, min(64, len(ii))).astype(int)]
            if not np.array_equal(np.asarray(p[0]).reshape(-1)[probe], bp[box].reshape(-1)[probe]):
                continue
        return ba[box].copy()
    return None
