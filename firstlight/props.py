"""The things on and by the window seat in First Light's film: the mug, the plant, the lamp.

The room (room.py) asks each prop five things, and nothing else about it:
  bound()            a sphere (centre, radius) that holds the prop
  hit(o, d)          where rays o + s d (o a point, d a tuple of arrays, not unit) first meet it:
                     (s, nx, ny, nz, mat), s = inf where they miss, n the unit outward normal, mat a material id
                     from ids.py per ray (int array; any id the prop uses, e.g. CUP, POT, LEAF, SHADE, DARK)
  occludes(P, s)     for points P (N x 3) and the unit direction s toward the sun: how much the prop is in the way
                     (0-1; here soft over a texel of its map as the sun sees it, so its shadows are antialiased)
  albedo(p, n, mat)  its colour (linear RGB, N x 3) at points p on it, normals n, materials mat
  emission(p, n, mat)  light it gives off itself (linear RGB, N x 3), times how much the lamp is on
  mats               the material ids it uses: no two props share one (the room tells them apart by these)
The world is in metres: x right, y up, z out of the window wall into the room; the camera looks along -z from
z = 2.4-3.0 at eye height 1.3. The seat's cushion top is at y = SEAT_TOP.

How they are made. Everything turned (the mug's body, the pot and its saucer, the lamp's base and stem, its shade
and bulb) is a solid of revolution: its half cross-section is drawn as a polygon with rounded corners, and its
signed distance in that plane is baked once, at import, into a fine 2D grid (0.2-0.4 mm), with its gradient. The
3D distance is then exact, a lookup at (distance from the axis, height), and rays are sphere-traced through it;
normals are the baked gradient, interpolated, so the shading is smooth. The mug's handle is a strap swept along a
curve (its distance to the curve, baked in the handle's plane), joined to the body by a smooth minimum, so it
grows out of the wall with a fillet. The plant's leaves are heart-shaped pothos leaves, each a thin sheet bent
over a quadric (arched along its midrib, cupped and folded across it, a little twisted), cut by a baked outline;
rays meet a sheet in closed form (a quadratic per half), and its margin's normals turn out across the edge, as a
curled leaf's do, so the rim light finds it. Stems and vines are chains of capsules. Each part is cast only on the
rays its bounds can reach. For shadows each prop is cast once more, along parallel rays from the sun's side (1.5
mm apart; remade when the sun has moved 1e-4 rad), and a point is shaded where the first thing the sun meets on
its line lies more than BIAS beyond it.

What can't be geometry is painted, band-limited to the pixel (a pixel is 1.5-2 mm at the props): the glaze's
drips, speckle and the sheen of the window in it; the coffee's crema and the window mirrored in it; the pots'
mottle and the soil; the leaves' marbling, midribs and veins; the shade's slubbed linen and its seam. The lamp's
own light in the glaze and on the leaves comes through emission (it is the lamp's light, so it goes with it).
"""
import numpy as np
from matplotlib.path import Path

from ids import CUP, POT, LEAF, SHADE, DARK, PHONES, BOOK, TOMES, lin

f32 = np.float32
SEAT_TOP = 0.55
TABLE_TOP = 0.62
MUG_C = (0.14, 0.22)                     # (x, z) on the seat
POT_C = (0.6, 0.29)
LAMP = (1.07, SEAT_TOP - 0.004, 0.3)     # the lamp's base, at the right end of the seat (sunk a little in the cushion)
BULB_Y = 0.38                            # the bulb's centre, above the lamp's base
LAMP_LIGHT = (LAMP[0], LAMP[1] + BULB_Y, LAMP[2])    # where its light comes from (the room lights by this)
EYE = np.array([0.0, 1.3, 2.7])          # about where the camera stands (it drifts z 2.4-3.0): for sheens
GLASS = (-1.15, 1.15, 0.6, 2.0, -0.18)   # the window's glass (x0, x1, y0, y1, z), as a glaze mirrors it


# --- small helpers -------------------------------------------------------------------------------------------
def _norm3(x, y, z):
    r = np.sqrt(x * x + y * y + z * z) + f32(1e-12)
    return x / r, y / r, z / r


def _at(a, i):
    return a[i] if np.ndim(a) else a


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _hash(i, j, k):
    return (np.sin(i * 12.9898 + j * 78.233 + k * 37.719) * 43758.5453) % 1


def noise(x, y, z):
    """Value noise, smooth, about -1..1 over a unit lattice."""
    ix, iy, iz = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - ix, y - iy, z - iz
    fx, fy, fz = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy), fz * fz * (3 - 2 * fz)
    v = 0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = (fx if a else 1 - fx) * (fy if b else 1 - fy) * (fz if c else 1 - fz)
                v = v + w * _hash(ix + a, iy + b, iz + c)
    return 2 * v - 1


def fbm(x, y, z, n=3):
    v, a, s = 0, 0.5, 1.0
    for k in range(n):
        v = v + a * noise(x * s + 13.1 * k, y * s + 7.7 * k, z * s + 3.3 * k)
        a, s = a * 0.5, s * 2.03
    return v / 0.875


def spline(pts, n):
    """A Catmull-Rom curve through pts, n points a span."""
    P = np.asarray(pts, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        for u in np.linspace(0, 1, n, endpoint=False):
            p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
            out.append(0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                              + (3 * p1 - p0 - 3 * p2 + p3) * u ** 3))
    out.append(P[-2])
    return np.array(out)


def rounded(pts, radii, n=10):
    """A polyline through pts whose corners are rounded with the given radii (0: left sharp)."""
    P = np.asarray(pts, float)
    out = [P[0]]
    for i in range(1, len(P) - 1):
        r = radii[i]
        a, b = P[i - 1] - P[i], P[i + 1] - P[i]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        a, b = a / la, b / lb
        cosang = np.clip(a @ b, -1, 1)
        ang = np.arccos(cosang)
        if r <= 0 or ang > np.pi - 1e-3:
            out.append(P[i])
            continue
        t = min(r / np.tan(ang / 2), 0.49 * la, 0.49 * lb)
        r = t * np.tan(ang / 2)
        t1, t2 = P[i] + a * t, P[i] + b * t
        bis = (a + b) / np.linalg.norm(a + b)
        c = P[i] + bis * r / np.sin(ang / 2)
        a1, a2 = np.arctan2(*(t1 - c)[::-1]), np.arctan2(*(t2 - c)[::-1])
        da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        for u in np.linspace(0, 1, n):
            out.append(c + r * np.array([np.cos(a1 + da * u), np.sin(a1 + da * u)]))
    out.append(P[-1])
    return np.array(out)


def mirror(half):
    """A half cross-section (from the axis, round, back to the axis) closed by its mirror image across it."""
    h = np.asarray(half, float)
    m = h[::-1].copy()
    m[:, 0] *= -1
    return np.vstack([h, m[1:-1]])


# --- distance fields in a plane, baked ----------------------------------------------------------------------
class Grid2:
    """A 2D distance field sampled on a grid, with its gradient, looked up bilinearly (and outside it, the
    distance to its edge added)."""

    def __init__(self, D, lo, h):
        self.n = D.shape
        self.lo, self.h = (f32(lo[0]), f32(lo[1])), f32(h)
        self.D = D.astype(f32).ravel()
        gx, gy = np.gradient(D.astype(np.float64), h)
        r = np.hypot(gx, gy) + 1e-12
        self.GX, self.GY = (gx / r).astype(f32).ravel(), (gy / r).astype(f32).ravel()

    def _cell(self, x, y):
        inv = 1 / self.h
        u, v = (x - self.lo[0]) * inv, (y - self.lo[1]) * inv
        uc, vc = np.clip(u, 0, self.n[0] - 1.001), np.clip(v, 0, self.n[1] - 1.001)
        out = np.sqrt((u - uc) ** 2 + (v - vc) ** 2) * self.h
        i, j = uc.astype(np.intp), vc.astype(np.intp)
        return i * self.n[1] + j, (uc - i).astype(f32), (vc - j).astype(f32), out

    def _lerp(self, A, idx, fx, fy):
        ny = self.n[1]
        a, b, c, d = A[idx], A[idx + 1], A[idx + ny], A[idx + ny + 1]
        return (a + (b - a) * fy) * (1 - fx) + (c + (d - c) * fy) * fx

    def dist(self, x, y):
        idx, fx, fy, out = self._cell(x, y)
        return self._lerp(self.D, idx, fx, fy) + out

    def grad(self, x, y):
        idx, fx, fy, _ = self._cell(x, y)
        return self._lerp(self.GX, idx, fx, fy), self._lerp(self.GY, idx, fx, fy)


def _grid_axes(lo, hi, h):
    nx, ny = int(np.ceil((hi[0] - lo[0]) / h)) + 1, int(np.ceil((hi[1] - lo[1]) / h)) + 1
    return nx, ny, lo[0] + h * np.arange(nx), lo[1] + h * np.arange(ny)


def _seg_dist2(D2, xs, ys, lo, h, band, a, b):
    """Fold one segment's squared distance into D2, only where it is within band of it."""
    nx, ny = D2.shape
    i0, i1 = max(0, int((min(a[0], b[0]) - band - lo[0]) / h)), min(nx, int((max(a[0], b[0]) + band - lo[0]) / h) + 2)
    j0, j1 = max(0, int((min(a[1], b[1]) - band - lo[1]) / h)), min(ny, int((max(a[1], b[1]) + band - lo[1]) / h) + 2)
    if i0 >= i1 or j0 >= j1:
        return
    x, y = xs[i0:i1, None], ys[None, j0:j1]
    e = b - a
    t = np.clip(((x - a[0]) * e[0] + (y - a[1]) * e[1]) / (e @ e + 1e-30), 0, 1)
    dd = (x - a[0] - e[0] * t) ** 2 + (y - a[1] - e[1] * t) ** 2
    np.minimum(D2[i0:i1, j0:j1], dd, out=D2[i0:i1, j0:j1])


def bake_polys(polys, lo, hi, h, band=0.012):
    """The signed distance (negative inside) to a union of closed polygons, on a grid, clamped at band."""
    nx, ny, xs, ys = _grid_axes(lo, hi, h)
    D = np.full((nx, ny), band)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    for P in polys:
        P = np.asarray(P, float)
        D2 = np.full((nx, ny), band * band)
        for i in range(len(P)):
            _seg_dist2(D2, xs, ys, lo, h, band, P[i], P[(i + 1) % len(P)])
        ins = Path(P).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(nx, ny)
        D = np.minimum(D, np.sqrt(D2) * np.where(ins, -1, 1))
    return Grid2(D, lo, h)


def bake_curve(C, lo, hi, h, band=0.012):
    """The distance to an open polyline, on a grid, clamped at band."""
    nx, ny, xs, ys = _grid_axes(lo, hi, h)
    D2 = np.full((nx, ny), band * band)
    for i in range(len(C) - 1):
        _seg_dist2(D2, xs, ys, lo, h, band, C[i], C[i + 1])
    return Grid2(np.sqrt(D2), lo, h)


# --- casting: rays against fields, sheets and tubes ----------------------------------------------------------
def _slab(O, U, lo, hi):
    t0, t1 = -np.inf, np.inf
    with np.errstate(divide='ignore', invalid='ignore'):
        for k in range(3):
            inv = 1 / U[k]
            a, b = (lo[k] - O[k]) * inv, (hi[k] - O[k]) * inv
            t0, t1 = np.maximum(t0, np.minimum(a, b)), np.minimum(t1, np.maximum(a, b))
    return np.nan_to_num(t0, nan=np.inf, neginf=-1e9), np.nan_to_num(t1, nan=-np.inf)


def _march(field, O, U, s, s1, eps, steps=80):
    """Sphere-trace rays O + s U (unit U) from s until the field is under eps or s passes s1."""
    s = s.copy()
    act = np.arange(len(s))
    done = np.zeros(len(s), bool)
    for _ in range(steps):
        if not len(act):
            break
        sa = s[act]
        f = field(tuple(_at(O[k], act) + U[k][act] * sa for k in range(3)))
        hitnow = f < eps[act]
        done[act[hitnow]] = True
        s[act] = sa + np.where(hitnow, 0, np.maximum(f * f32(0.9), eps[act] * 0.5))
        act = act[~hitnow & (s[act] < s1[act])]
    return s, done


class Solid:
    """Something with a distance field: field(p), normal(p), an axis-aligned box (lo, hi) and a sphere (c, r)."""
    mat = None

    def cylinder(self, O, U):
        return None

    def normal(self, p, h=f32(3e-4)):
        """The field's gradient by central differences."""
        g = []
        for k in range(3):
            a = [q for q in p]
            b = [q for q in p]
            a[k], b[k] = p[k] + h, p[k] - h
            g.append(self.field(tuple(a)) - self.field(tuple(b)))
        return _norm3(*g)

    def sphere(self):
        lo, hi = self.box
        return (lo + hi) / 2, float(np.linalg.norm(hi - lo) / 2)

    def cast(self, O, U, smin_):
        """Rays O + s U (unit U): (s, nx, ny, nz), s = inf where they miss."""
        n = len(U[0])
        S = np.full(n, np.inf, f32)
        N = [np.zeros(n, f32) for _ in range(3)]
        t0, t1 = _slab(O, U, *self.box)
        t0 = np.maximum(t0, smin_)
        cyl = self.cylinder(O, U)
        if cyl is not None:
            t0, t1 = np.maximum(t0, cyl[0]), np.minimum(t1, cyl[1])
        sel = np.nonzero(t1 > t0)[0]
        if not len(sel):
            return S, N
        o_, u_ = [_at(a, sel) for a in O], [a[sel] for a in U]
        eps = np.maximum(f32(1.5e-4) * t0[sel], f32(1e-4)).astype(f32)
        s, ok = _march(self.field, o_, u_, t0[sel].astype(f32), t1[sel].astype(f32), eps)
        idx = np.nonzero(ok)[0]
        if len(idx):
            p = tuple(_at(o_[k], idx) + u_[k][idx] * s[idx] for k in range(3))
            s2 = s[idx] + self.field(p)                              # one more step, onto the surface
            p = tuple(_at(o_[k], idx) + u_[k][idx] * s2 for k in range(3))
            S[sel[idx]] = s2
            nn = self.normal(p)
            for k in range(3):
                N[k][sel[idx]] = nn[k]
        return S, N


def _cylinder(O, U, cx, cz, r):
    """Where rays enter and leave a vertical cylinder (axis through (cx, cz), radius r): (t0, t1), t1 < t0 if missed."""
    ox, oz = O[0] - cx, O[2] - cz
    a = U[0] * U[0] + U[2] * U[2]
    b = ox * U[0] + oz * U[2]
    c = ox * ox + oz * oz - r * r
    disc = b * b - a * c
    with np.errstate(invalid='ignore', divide='ignore'):
        sq = np.sqrt(np.maximum(disc, 0))
        t0, t1 = (-b - sq) / a, (-b + sq) / a
    ok = (disc > 0) & (a > 1e-9)
    inside = c < 0
    return (np.where(ok, t0, np.where(inside, -np.inf, np.inf)).astype(f32),
            np.where(ok, t1, np.where(inside, np.inf, -np.inf)).astype(f32))


class Lathe(Solid):
    """A solid of revolution about a vertical axis through (cx, cz), its cross-sections (closed polygons in
    (r, y), y from y0 up; mirror() closes one drawn from the axis round and back) baked at spacing h."""

    def __init__(self, cx, y0, cz, polys, h, mat=None):
        allp = np.vstack(polys)
        self.rmax, self.ylo, self.yhi = allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max()
        band = 0.012
        rlo = max(-4 * h, allp[allp[:, 0] >= 0, 0].min() - band)
        self.G = bake_polys(polys, (rlo, self.ylo - band), (self.rmax + band, self.yhi + band), h, band)
        self.cx, self.y0, self.cz = f32(cx), f32(y0), f32(cz)
        self.box = (np.array([cx - self.rmax, y0 + self.ylo, cz - self.rmax]) - 1e-3,
                    np.array([cx + self.rmax, y0 + self.yhi, cz + self.rmax]) + 1e-3)
        self.mat = mat

    def local(self, p):
        x, z = p[0] - self.cx, p[2] - self.cz
        return np.sqrt(x * x + z * z), p[1] - self.y0, x, z

    def cylinder(self, O, U):
        return _cylinder(O, U, self.cx, self.cz, f32(self.rmax + 1e-3))

    def field(self, p):
        r, y, _, _ = self.local(p)
        return self.G.dist(r, y)

    def normal(self, p):
        r, y, x, z = self.local(p)
        gr, gy = self.G.grad(r, y)
        inv = 1 / np.maximum(r, f32(1e-6))
        return _norm3(gr * x * inv, gy, gr * z * inv)


def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k * 0.25


class Sheet:
    """A leaf: a thin sheet over a quadric in its own frame (u across, v along the midrib from where the petiole
    meets it, w up from its upper face), w = cup u^2 + fold |u| - arch v^2 + twist u v, cut by a heart outline of
    length L and half-widths (left, right)."""
    mat = LEAF

    def __init__(self, B, R, L, wl, wr, arch, cup, fold, twist, look):
        self.B, self.R = np.asarray(B, f32), np.asarray(R, f32)
        self.L, self.wl, self.wr = f32(L), f32(wl), f32(wr)
        self.arch, self.cup, self.fold, self.twist = f32(arch), f32(cup), f32(fold), f32(twist)
        self.look = look                                          # its colouring: (variegation, youth, seed)
        corners = []
        for u in (-wl * 1.05, 0, wr * 1.05):
            for v in (-0.2 * L, 0.5 * L, L * 1.02):
                corners.append(self.to_world(u, v, self.height(u, v)))
        corners = np.array(corners)
        self.c = corners.mean(0)
        self.r = float(np.linalg.norm(corners - self.c, axis=1).max()) + 0.004
        uu, vv = np.meshgrid(np.linspace(-wl * 1.05, wr * 1.05, 7), np.linspace(-0.32 * L, 1.06 * L, 9))
        pts = np.array([self.to_world(a, b, self.height(a, b)) for a, b in zip(uu.ravel(), vv.ravel())])
        self.box = (pts.min(0) - 0.003, pts.max(0) + 0.003)            # (the sheet bulges a little between them)

    def height(self, u, v):
        return self.cup * u * u + self.fold * np.abs(u) - self.arch * v * v + self.twist * u * v

    def to_world(self, u, v, w):
        return self.B + self.R @ np.array([u, v, w], f32)

    def sphere(self):
        return self.c, self.r

    def to_local(self, p):
        x, y, z = p[0] - self.B[0], p[1] - self.B[1], p[2] - self.B[2]
        R = self.R
        return (R[0, 0] * x + R[1, 0] * y + R[2, 0] * z, R[0, 1] * x + R[1, 1] * y + R[2, 1] * z,
                R[0, 2] * x + R[1, 2] * y + R[2, 2] * z)

    def outline(self, u, v):
        """Signed distance to the outline, in units of the leaf's width (negative inside)."""
        w = np.where(u < 0, self.wl, self.wr)
        return OUTLINE.dist(np.abs(u) / w, v / self.L)

    def cast(self, O, U, smin_):
        n = len(U[0])
        best = np.full(n, np.inf, f32)
        N = [np.zeros(n, f32) for _ in range(3)]
        # only the rays that pass within the leaf's sphere
        w = [self.c[k] - O[k] for k in range(3)]
        proj = w[0] * U[0] + w[1] * U[1] + w[2] * U[2]
        near = np.nonzero((w[0] * w[0] + w[1] * w[1] + w[2] * w[2] - proj * proj < self.r * self.r)
                          & (proj > -self.r))[0]
        if not len(near):
            return best, N
        Un = [a[near] for a in U]
        u0, v0, w0 = (np.broadcast_to(a, (len(near),)) if np.ndim(a) == 0 else a
                      for a in self.to_local([_at(a, near) for a in O]))
        R = self.R
        du = R[0, 0] * Un[0] + R[1, 0] * Un[1] + R[2, 0] * Un[2]
        dv = R[0, 1] * Un[0] + R[1, 1] * Un[1] + R[2, 1] * Un[2]
        dw = R[0, 2] * Un[0] + R[1, 2] * Un[1] + R[2, 2] * Un[2]
        c, a, T = self.cup, self.arch, self.twist
        bn = np.full(len(near), np.inf, f32)
        al = -(c * du * du - a * dv * dv + T * du * dv)
        for sg in (-1, 1):
            K = self.fold * sg
            be = dw - (2 * c * u0 * du + K * du - 2 * a * v0 * dv + T * (u0 * dv + v0 * du))
            ga = w0 - (c * u0 * u0 + K * u0 - a * v0 * v0 + T * u0 * v0)
            with np.errstate(invalid='ignore', divide='ignore'):
                disc = be * be - 4 * al * ga
                sq = np.sqrt(np.maximum(disc, 0))
                lin_ = np.abs(al) < 1e-7
                q = -0.5 * (be + np.where(be >= 0, sq, -sq))             # the stable pair of roots
                r1 = np.where(lin_, -ga / be, q / al)
                r2 = np.where(lin_, np.inf, ga / q)
            for r in (r1, r2):
                cand = np.nonzero((disc >= 0) & (r > smin_) & (r < bn))[0]
                if not len(cand):
                    continue
                rc = r[cand]
                uu, vv = u0[cand] + rc * du[cand], v0[cand] + rc * dv[cand]
                ok = (uu * sg >= 0) & (vv > -0.3 * self.L) & (vv < 1.05 * self.L)
                cand, rc, uu, vv = cand[ok], rc[ok], uu[ok], vv[ok]
                ok = self.outline(uu, vv) < 0
                bn[cand[ok]] = rc[ok]
        hit_ = np.nonzero(np.isfinite(bn))[0]
        if len(hit_):
            s = bn[hit_]
            uu, vv = u0[hit_] + s * du[hit_], v0[hit_] + s * dv[hit_]
            nl = self.normal_local(uu, vv)
            nw = [R[k, 0] * nl[0] + R[k, 1] * nl[1] + R[k, 2] * nl[2] for k in range(3)]
            face = nw[0] * Un[0][hit_] + nw[1] * Un[1][hit_] + nw[2] * Un[2][hit_]
            sgn = np.where(face > 0, -1, 1).astype(f32)                    # two-sided: face the ray
            # the margin curls a little: within EDGE of the outline the normal turns out across it, so the
            # leaf's edge catches the rim light as a rounded thing's would
            wd = np.where(uu < 0, self.wl, self.wr)
            xa, ta = np.abs(uu) / wd, vv / self.L
            dn = OUTLINE.dist(xa, ta)
            gx, gt = OUTLINE.grad(xa, ta)
            eu, ev = gx * np.sign(uu) / wd, gt / self.L
            m = np.sqrt(eu * eu + ev * ev) + f32(1e-6)
            dist = -dn / m
            th = (1 - smooth(dist / f32(EDGE))) * f32(0.95)
            eu, ev = eu / m, ev / m
            ow = [R[k, 0] * eu + R[k, 1] * ev for k in range(3)]
            ct, st = np.cos(th), np.sin(th)
            g = near[hit_]
            best[g] = s
            nf = _norm3(*[nw[k] * sgn * ct + ow[k] * st for k in range(3)])
            for k in range(3):
                N[k][g] = nf[k]
        return best, N

    def normal_local(self, u, v):
        """The sheet's normal, the midrib's crease softened, and the veins pressed in a little."""
        hu = 2 * self.cup * u + self.fold * np.tanh(u / 0.0018) + self.twist * v
        hv = -2 * self.arch * v + self.twist * u
        w = np.where(u < 0, self.wl, self.wr)
        x, t = np.abs(u) / w, v / self.L
        # the lateral veins: from the midrib out and forward toward the tip, a groove along each
        ph = (t - 0.62 * x ** 1.3) * 5.0 + 0.3 * noise(t * 4 + self.look[2], x * 3, self.look[2])
        g = (np.sin(2 * np.pi * ph) * np.clip(x * 4, 0, 1) * np.clip((1 - x) * 3, 0, 1) * np.clip(t * 4 + 0.6, 0, 1)
             * (0.6 + 0.4 * noise(ph * 2, x * 5, 3.3 + self.look[2])))
        hv = hv + 0.03 * g
        hu = hu - 0.03 * g * 0.8 * np.sign(u)
        return _norm3(-hu, -hv, np.ones_like(u))


class Tube:
    """A chain of capsules (a stem, a vine) through points C, radii r along it."""
    mat = LEAF

    def __init__(self, C, r):
        self.C, self.r = np.asarray(C, f32), np.broadcast_to(np.asarray(r, f32), (len(C),)).copy()
        lo, hi = self.C.min(0) - self.r.max(), self.C.max(0) + self.r.max()
        self.box = (lo - 1e-3, hi + 1e-3)
        self.c = (lo + hi) / 2
        self.rad = float(np.linalg.norm(hi - lo) / 2) + 0.002

    def sphere(self):
        return self.c, self.rad

    def cast(self, O, U, smin_):
        n = len(U[0])
        best = np.full(n, np.inf, f32)
        N = [np.zeros(n, f32) for _ in range(3)]
        w = [self.c[k] - O[k] for k in range(3)]                       # only the rays that pass near it
        proj = w[0] * U[0] + w[1] * U[1] + w[2] * U[2]
        near = np.nonzero((w[0] * w[0] + w[1] * w[1] + w[2] * w[2] - proj * proj < self.rad * self.rad)
                          & (proj > -self.rad))[0]
        if not len(near):
            return best, N
        if len(near) < n:
            s, nn = self.cast([_at(a, near) for a in O], [a[near] for a in U], smin_)
            best[near] = s
            for k in range(3):
                N[k][near] = nn[k]
            return best, N
        O = [np.broadcast_to(np.asarray(a, f32), (n,)) for a in O]
        for i in range(len(self.C) - 1):
            pa, pb = self.C[i], self.C[i + 1]
            r = (self.r[i] + self.r[i + 1]) / 2
            ba = pb - pa
            oa = [O[k] - pa[k] for k in range(3)]
            baba = float(ba @ ba)
            bard = ba[0] * U[0] + ba[1] * U[1] + ba[2] * U[2]
            baoa = ba[0] * oa[0] + ba[1] * oa[1] + ba[2] * oa[2]
            rdoa = U[0] * oa[0] + U[1] * oa[1] + U[2] * oa[2]
            oaoa = oa[0] * oa[0] + oa[1] * oa[1] + oa[2] * oa[2]
            a = baba - bard * bard
            b = baba * rdoa - baoa * bard
            c = baba * oaoa - baoa * baoa - r * r * baba
            h = b * b - a * c
            with np.errstate(invalid='ignore', divide='ignore'):
                t = (-b - np.sqrt(np.maximum(h, 0))) / a
                y = baoa + t * bard
                body = (h >= 0) & (y > 0) & (y < baba)
                ocs = [np.where(y <= 0, oa[k], O[k] - pb[k]) for k in range(3)]
                b2 = U[0] * ocs[0] + U[1] * ocs[1] + U[2] * ocs[2]
                c2 = ocs[0] * ocs[0] + ocs[1] * ocs[1] + ocs[2] * ocs[2] - r * r
                h2 = b2 * b2 - c2
                tc = -b2 - np.sqrt(np.maximum(h2, 0))
            t = np.where(body, t, np.where((h >= 0) & (h2 > 0), tc, np.inf))
            ok = (t > smin_) & (t < best)
            if not ok.any():
                continue
            best = np.where(ok, t, best)
            k_ = np.nonzero(ok)[0]
            p = [O[k][k_] + U[k][k_] * t[k_] for k in range(3)]
            hh = np.clip(((p[0] - pa[0]) * ba[0] + (p[1] - pa[1]) * ba[1] + (p[2] - pa[2]) * ba[2]) / baba, 0, 1)
            nn = _norm3(*[p[k] - pa[k] - ba[k] * hh for k in range(3)])
            for k in range(3):
                N[k][k_] = nn[k]
        return best, N


# --- the heart outline of a pothos leaf, in units of its half-width (x) and length (t) ----------------------
_HEART = spline([(0.0, 1.0), (0.06, 0.95), (0.16, 0.9), (0.36, 0.8), (0.6, 0.66), (0.83, 0.48), (0.97, 0.29),
                 (1.0, 0.12), (0.95, -0.01), (0.8, -0.1), (0.58, -0.14), (0.34, -0.11), (0.14, -0.04), (0.0, 0.025)], 8)
OUTLINE = bake_polys([mirror(_HEART)], (-0.02, -0.3), (1.25, 1.12), 0.004, band=0.2)


# --- the props -----------------------------------------------------------------------------------------------
SUNMAP = 0.0025                          # the spacing of the props' maps as the sun sees them (m): its penumbra
SKYMAP = 0.006                           # and for steep directions (never the sun's; asked for soft contact shade)
BIAS = 0.003                             # how far behind the first thing the sun meets a point is shaded by it
SUN_Q = 1e-4                             # the sun maps are made for its direction to this (rad): 0.05 mm of shadow
EDGE = 0.003                             # how far in from a leaf's edge its margin curls


def _ray_box(d, o, c, r, box=None):
    """For a 2D grid of rays from o along d (d[2] = -1, as the room casts them): the rows and columns whose rays
    can meet the sphere (c, r), or the box (lo, hi) if given, as slices; None if none can; (slice(None),) * 2 if
    the grid isn't one."""
    full = slice(None), slice(None)
    if np.ndim(d[0]) != 2 or np.ndim(o[0]) != 0 or d[0].shape[0] < 2 or d[0].shape[1] < 2:
        return full
    if np.ndim(d[2]) == 2 and (abs(float(d[2][0, 0]) + 1) > 1e-6 or abs(float(d[2][-1, -1]) + 1) > 1e-6):
        return full
    if not (d[0][0, 1] > d[0][0, 0] and d[1][1, 0] < d[1][0, 0]):
        return full
    D = o[2] - c[2]
    if D <= r * 1.5:
        return full
    if box is not None:
        lo, hi = box
        X = np.array([lo[0], hi[0]])[:, None, None] - o[0]
        Y = np.array([lo[1], hi[1]])[None, :, None] - o[1]
        Z = o[2] - np.array([lo[2], hi[2]])[None, None, :]
        px, py = (X / Z).ravel(), (Y / Z).ravel()
        m = 2e-4
        x0, x1, y0, y1 = px.min() - m, px.max() + m, py.min() - m, py.max() + m
    else:
        half = r / (D - r) * 1.15 + 1e-4
        cx, cy = (c[0] - o[0]) / D, (c[1] - o[1]) / D
        x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    xs, ys = d[0][0, :], d[1][:, 0]
    c0 = int(np.searchsorted(xs, x0))
    c1 = int(np.searchsorted(xs, x1)) + 1
    ry = -ys                                                       # rows run downward
    r0 = int(np.searchsorted(ry, -y1))
    r1 = int(np.searchsorted(ry, -y0)) + 1
    if c0 >= c1 or r0 >= r1 or c1 <= 0 or r1 <= 0 or c0 >= len(xs) or r0 >= len(ys):
        return None
    return slice(max(r0, 0), r1), slice(max(c0, 0), c1)


class Prop:
    """A prop made of parts (Solids, Sheets, Tubes), each with its material."""

    def __init__(self, parts):
        self.parts = parts
        self.mats = tuple(sorted({p.mat for p in parts}))
        cs = [p.sphere() for p in parts]
        c = (np.min([a - b for a, b in cs], 0) + np.max([a + b for a, b in cs], 0)) / 2
        self.c = c
        self.r = max(float(np.linalg.norm(a - c)) + b for a, b in cs)

    def bound(self):
        return self.c, self.r

    def hit(self, o, d):
        shape = np.shape(d[0])
        if len(shape) == 2 and all(np.ndim(a) == 0 for a in o):
            # a grid of rays from one point (the room's): each part on its own block of the grid
            T = np.full(shape, np.inf, f32)
            N = [np.zeros(shape, f32) for _ in range(3)]
            M = np.zeros(shape, np.int8)
            O = [f32(a) for a in o]
            for part in self.parts:
                box = _ray_box(d, o, *part.sphere(), getattr(part, 'box', None))
                if box is None:
                    continue
                dd = [np.asarray(np.broadcast_to(a, shape)[box], f32) for a in d]
                sub = dd[0].shape
                if not dd[0].size:
                    continue
                L = np.sqrt(dd[0] ** 2 + dd[1] ** 2 + dd[2] ** 2).ravel()
                s, nn = part.cast(O, [(a.ravel() / L) for a in dd], f32(1e-4))
                ok = np.isfinite(s)
                if not ok.any():
                    continue
                tt = (s / L).reshape(sub)
                Tb = T[box]
                closer = tt < Tb
                Tb[closer] = tt[closer]
                M[box][closer] = part.mat
                for k in range(3):
                    N[k][box][closer] = nn[k].reshape(sub)[closer]
            return T, N[0], N[1], N[2], M
        dd = [np.ravel(np.broadcast_to(np.asarray(a, f32), shape)) for a in d]
        L = np.sqrt(dd[0] ** 2 + dd[1] ** 2 + dd[2] ** 2)
        U = [a / L for a in dd]
        O = [np.asarray(a, f32) if np.ndim(a) == 0 else np.ravel(np.broadcast_to(np.asarray(a, f32), shape)) for a in o]
        S, N, M = self._cast(O, U, shape, [slice(None)] * len(self.parts))
        T = (S / L).reshape(shape)
        return T, N[0].reshape(shape), N[1].reshape(shape), N[2].reshape(shape), M.reshape(shape)

    def _cast(self, O, U, shape, boxes):
        """Cast the parts on rays (flat arrays O, U, unit U; a grid of shape), each part only on its box of the
        grid (a pair of slices, or None): the first hit's distance, normal and material."""
        n = int(np.prod(shape))
        S = np.full(n, np.inf, f32)
        N = [np.zeros(n, f32) for _ in range(3)]
        M = np.zeros(n, np.int8)
        idx_all = np.arange(n).reshape(shape)
        for part, box in zip(self.parts, boxes):
            if box is None:
                continue
            idx = idx_all[box].ravel()
            if not len(idx):
                continue
            s, nn = part.cast([_at(a, idx) for a in O], [a[idx] for a in U], f32(1e-4))
            closer = s < S[idx]
            k = idx[closer]
            S[k] = s[closer]
            M[k] = part.mat
            for j in range(3):
                N[j][k] = nn[j][closer]
        return S, N, M

    def _sunmap(self, s):
        """The prop seen from the sun: on a grid of parallel rays across its light, how far toward the sun (from
        its centre, along s) the first thing each meets is (-inf where it meets nothing)."""
        s = np.asarray(s, float) / np.linalg.norm(s)
        az, el = np.arctan2(s[0], -s[2]), np.arcsin(np.clip(s[1], -1, 1))
        key = (int(round(az / SUN_Q)), int(round(el / SUN_Q)))       # the sun's way, to SUN_Q: one map for many frames
        if getattr(self, '_sm', None) is not None and self._sm[0] == key:
            return self._sm[1]
        az, el = key[0] * SUN_Q, key[1] * SUN_Q
        s = np.array([np.cos(el) * np.sin(az), np.sin(el), -np.cos(el) * np.cos(az)])
        e1 = np.cross(s, (0.0, 1.0, 0.0) if abs(s[1]) < 0.9 else (1.0, 0.0, 0.0))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(s, e1)
        c, r = self.bound()
        h = SUNMAP if abs(el) < 0.5 else SKYMAP
        sph = [part.sphere() for part in self.parts]
        pu = [((pc - c) @ e1, (pc - c) @ e2, pr_) for pc, pr_ in sph]
        u0 = min(a - rr for a, _, rr in pu) - 2 * h
        v0 = min(b - rr for _, b, rr in pu) - 2 * h
        nu = int(np.ceil((max(a + rr for a, _, rr in pu) - u0) / h)) + 3
        nv = int(np.ceil((max(b + rr for _, b, rr in pu) - v0) / h)) + 3
        v, u = np.mgrid[0:nv, 0:nu].astype(f32)
        u, v = (u0 + u * h).ravel(), (v0 + v * h).ravel()
        top = r + 0.02
        O = [(c[k] + u * e1[k] + v * e2[k] + top * s[k]).astype(f32) for k in range(3)]
        U = [np.full(len(u), f32(-s[k])) for k in range(3)]
        boxes = []
        for a, b, rr in pu:
            i0, i1 = int((b - rr - v0) / h) - 1, int((b + rr - v0) / h) + 2
            j0, j1 = int((a - rr - u0) / h) - 1, int((a + rr - u0) / h) + 2
            boxes.append((slice(max(i0, 0), max(i1, 0)), slice(max(j0, 0), max(j1, 0))))
        S, _, _ = self._cast(O, U, (nv, nu), boxes)
        W = np.where(np.isfinite(S), top - S, -np.inf).astype(f32).reshape(nv, nu)
        sm = (np.stack([e1 / h, e2 / h, s], 1).astype(f32), c.astype(f32), f32(u0 / h), f32(v0 / h), nu, nv, W)
        self._sm = (key, sm)
        return sm

    def occludes(self, P, s):
        """How much the prop stands between each point and the sun (0-1, soft over a texel of its sun map). A point
        on the prop's own surface is not shaded by that surface (it lies within BIAS of the first thing the sun
        meets)."""
        if not len(P):
            return np.zeros(0, f32)
        M, c, u0, v0, nu, nv, W = self._sunmap(s)
        uvw = (np.asarray(P, f32) - c) @ M
        u, v, w = uvw[:, 0] - u0, uvw[:, 1] - v0, uvw[:, 2]
        i = np.clip(np.floor(v).astype(np.intp), 0, nv - 2)
        j = np.clip(np.floor(u).astype(np.intp), 0, nu - 2)
        fv, fu = np.clip(v - i, 0, 1), np.clip(u - j, 0, 1)
        lim = w + f32(BIAS)
        o = [(W[i + a, j + b] > lim).astype(f32) for a in (0, 1) for b in (0, 1)]
        cov = (o[0] * (1 - fu) + o[1] * fu) * (1 - fv) + (o[2] * (1 - fu) + o[3] * fu) * fv
        inmap = (u >= 0) & (u < nu - 1) & (v >= 0) & (v < nv - 1)
        return np.where(inmap, cov, 0).astype(f32)

    def albedo(self, p, n, mat):
        raise NotImplementedError

    def emission(self, p, n, mat):
        return np.zeros((len(mat), 3), f32)


def _view(p):
    """Unit vectors from points p toward the camera (about where it stands)."""
    return _norm3(EYE[0] - p[0], EYE[1] - p[1], EYE[2] - p[2])


def _window_seen(p, r, e=0.12):
    """How much of the window's glass a mirror at p sees along r (0-1, soft-edged over e)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = (GLASS[4] - p[2]) / r[2]
    x, y = p[0] + lam * r[0], p[1] + lam * r[1]
    m = (smooth((x - GLASS[0]) / e) * smooth((GLASS[1] - x) / e) * smooth((y - GLASS[2]) / e)
         * smooth((GLASS[3] - y) / e))
    return np.where((r[2] < -0.05) & np.isfinite(lam), m, 0.0)


def _reflect(n, v):
    k = 2 * (n[0] * v[0] + n[1] * v[1] + n[2] * v[2])
    return k * n[0] - v[0], k * n[1] - v[1], k * n[2] - v[2]


def _lamp_glint(p, n, power, k=1.0, stretch=1.0):
    """The lamp's highlight off a glossy surface (Blinn), as the camera about sees it; stretch < 1 draws it out
    up and down (a turned surface's streak)."""
    v = _view(p)
    lx, ly, lz = _norm3(LAMP_LIGHT[0] - p[0], LAMP_LIGHT[1] - p[1], LAMP_LIGHT[2] - p[2])
    hx, hy, hz = _norm3(lx + v[0], ly + v[1], lz + v[2])
    if stretch != 1.0:
        hx, hy, hz = _norm3(hx, hy * stretch, hz)
        n = _norm3(n[0], n[1] * stretch, n[2])
    nh = np.clip(n[0] * hx + n[1] * hy + n[2] * hz, 0, 1)
    nl = np.clip(n[0] * lx + n[1] * ly + n[2] * lz, 0, 1)
    d2 = (LAMP_LIGHT[0] - p[0]) ** 2 + (LAMP_LIGHT[1] - p[1]) ** 2 + (LAMP_LIGHT[2] - p[2]) ** 2
    return k * nh ** power * np.sqrt(nl) / (d2 + 0.05)


# --- the mug: stoneware, an ivory glaze dipped over a toasted clay foot, a strap handle, coffee nearly to the brim
MUG_Y0 = SEAT_TOP - 0.001                # it sinks a hair into the cushion
MUG_RIM = 0.0975                         # the rim's top, above its foot
COFFEE = 0.092                           # the coffee's level
_MUG_HALF = rounded([(0, 0.0022), (0.029, 0.0022), (0.0315, 0.0), (0.0352, 0.0), (0.0372, 0.006), (0.0402, 0.03),
                     (0.0411, 0.07), (0.0405, MUG_RIM), (0.0363, MUG_RIM), (0.0367, COFFEE), (0.0, COFFEE)],
                    [0, 0.0012, 0.0008, 0.0018, 0.004, 0.03, 0.03, 0.0019, 0.0019, 0.0016, 0])
HANDLE_AZ = np.radians(28)               # the handle's plane: from +x, turned this far toward the camera
_HANDLE = spline([(0.0372, 0.0835), (0.0485, 0.0858), (0.0595, 0.082), (0.0662, 0.0712), (0.0678, 0.0572),
                  (0.0642, 0.0425), (0.0555, 0.0318), (0.0455, 0.0265), (0.0372, 0.0245)], 10)
H_RA, H_RC, H_K = 0.0041, 0.0063, 0.0035     # the strap's half-thickness, half-width, and its fillet


class MugShape(Lathe):
    def __init__(self):
        super().__init__(MUG_C[0], MUG_Y0, MUG_C[1], [mirror(_MUG_HALF)], 0.0002, CUP)
        self.HG = bake_curve(_HANDLE, (0.025, 0.012), (0.08, 0.098), 0.0002)
        self.ea = np.array([np.cos(HANDLE_AZ), 0, np.sin(HANDLE_AZ)], f32)
        self.ec = np.array([-np.sin(HANDLE_AZ), 0, np.cos(HANDLE_AZ)], f32)
        lo, hi = self.box
        hx = [self.cx + self.ea[0] * a + self.ec[0] * c for a in (0.0, 0.075) for c in (-0.008, 0.008)]
        hz = [self.cz + self.ea[2] * a + self.ec[2] * c for a in (0.0, 0.075) for c in (-0.008, 0.008)]
        self.box = (np.array([min(lo[0], min(hx)), lo[1], min(lo[2], min(hz))]),
                    np.array([max(hi[0], max(hx)), hi[1], max(hi[2], max(hz))]))

    def cylinder(self, O, U):
        return _cylinder(O, U, self.cx, self.cz, f32(0.0735))           # the handle reaches out to here

    def parts(self, p):
        r, y, x, z = self.local(p)
        a = x * self.ea[0] + z * self.ea[2]
        c = x * self.ec[0] + z * self.ec[2]
        dl = self.G.dist(r, y)
        dc = self.HG.dist(a, y)
        k = f32(H_RA / H_RC)
        q = np.sqrt(dc * dc + (k * c) ** 2)
        dh = np.maximum(q - f32(H_RA), f32(0.0386) - a)
        return dl, dh, (r, y, x, z, a, c, dc, q)

    def field(self, p):
        dl, dh, _ = self.parts(p)
        return smin(dl, dh, f32(H_K))

    def normal(self, p):
        dl, dh, (r, y, x, z, a, c, dc, q) = self.parts(p)
        gr, gy = self.G.grad(r, y)
        inv = 1 / np.maximum(r, f32(1e-6))
        nl = (gr * x * inv, gy, gr * z * inv)
        ga, gb = self.HG.grad(a, y)
        k2 = f32((H_RA / H_RC) ** 2)
        iq = 1 / np.maximum(q, f32(1e-6))
        gA, gC, gY = dc * ga * iq, k2 * c * iq, dc * gb * iq        # in the handle's plane (a, c) and up
        cut = (f32(0.0386) - a) > (q - f32(H_RA))
        gA, gC, gY = np.where(cut, -1, gA), np.where(cut, 0, gC), np.where(cut, 0, gY)
        nh = (gA * self.ea[0] + gC * self.ec[0], gY, gA * self.ea[2] + gC * self.ec[2])
        w = np.clip(0.5 + 0.5 * (dl - dh) / f32(H_K), 0, 1)
        return _norm3(*[nl[k] * (1 - w) + nh[k] * w for k in range(3)])


GLAZE, GLAZE_DEEP = lin(.92, .87, .78), lin(.78, .7, .58)
OATMEAL, CLAY, BREAK = lin(.68, .52, .38), lin(.6, .45, .33), lin(.7, .53, .32)
COFFEE_C, CREMA = lin(.12, .065, .035), lin(.42, .27, .15)


class Mug(Prop):
    def __init__(self):
        self.shape = MugShape()
        super().__init__([self.shape])

    def _coords(self, p):
        S = self.shape
        dl, dh, (r, y, x, z, a, c, dc, q) = S.parts(p)
        handle = np.clip(0.5 + (dl - dh) / 0.003, 0, 1)
        ang = np.arctan2(z, x)
        return r, y, ang, handle, a, dc

    def albedo(self, p, n, mat):
        p = tuple(np.asarray(a, f32) for a in p)
        n = tuple(np.asarray(a, f32) for a in n)
        r, y, ang, handle, a, dc = self._coords(p)
        # where the dip ends: an uneven line round the body, with drips running down from it
        u = np.cos(ang), np.sin(ang)
        wob = noise(u[0] * 2.2 + 3, u[1] * 2.2, 0.5) * 0.006 + noise(u[0] * 5, u[1] * 5, 1.7) * 0.002
        dn = noise(u[0] * 5.5 + 1, u[1] * 5.5, 4.1)
        drip = smooth((dn - 0.45) / 0.3) * (0.004 + 0.007 * smooth(noise(u[0] * 3, u[1] * 3, 8.8) + 0.5))
        line = 0.021 + wob - drip
        glazed = smooth((y - line) / 0.0025)
        glazed = np.maximum(glazed, handle)                           # the handle went in with the dip
        inside = (r < 0.0372) & (y > 0.08)
        glazed = np.where(inside, 1.0, glazed)
        pool = smooth((y - line) / 0.0025) * (1 - smooth((y - line - 0.0045) / 0.003))   # the glaze pools at its edge
        sp = noise(p[0] * 230, p[1] * 230, p[2] * 230)
        speck = smooth((sp - 0.5) / 0.25)                             # dark specks in the clay, 2-3 px
        body = OATMEAL * (1 - 0.22 * speck)[:, None]
        foot = 1 - smooth((y - 0.004) / 0.003)
        body = body * (1 - foot[:, None]) + CLAY * foot[:, None]
        tone = 0.5 + 0.5 * fbm(p[0] * 60, p[1] * 60, p[2] * 60)
        g = GLAZE * (1 - 0.35 * tone)[:, None] + GLAZE_DEEP * (0.35 * tone)[:, None]
        g = g * (1 - 0.5 * pool[:, None]) + GLAZE_DEEP * (0.5 * pool)[:, None]
        # where the glaze runs thin, over the rim's edge and the handle's ridges, it breaks to amber
        rimy = np.clip(1 - np.abs(y - (MUG_RIM - 0.0012)) / 0.0022, 0, 1) * (r > 0.0355)
        g = g * (1 - 0.8 * rimy[:, None]) + BREAK * (0.8 * rimy)[:, None]
        g = g * (1 - 0.12 * speck)[:, None]                            # the clay's specks show through the glaze
        A = body * (1 - glazed[:, None]) + g * glazed[:, None]
        # the coffee, its crema at the wall
        cof = (y > COFFEE - 0.0012) & (r < 0.0362)
        if cof.any():
            rr = r[cof]
            cr = np.clip(1 - (0.0362 - rr) / 0.0025, 0, 1) ** 2 * (0.45 + 0.25 * noise(p[0][cof] * 300, 0, p[2][cof] * 300))
            A[cof] = COFFEE_C * (1 - cr[:, None]) + CREMA * cr[:, None]
        # the glaze and the coffee mirror the window (the camera stands about at EYE)
        v = _view(p)
        rf = _reflect(n, v)
        cos = np.clip(n[0] * v[0] + n[1] * v[1] + n[2] * v[2], 0, 1)
        fres = 0.04 + 0.96 * (1 - cos) ** 5
        win = _window_seen(p, rf, 0.06)
        gloss = np.where(cof, 0.04, glazed * 0.9 + 0.06)
        A = A + (win * fres * gloss * 2.4)[:, None] * lin(.95, .88, .8)
        # its own shade: inside the mug, under the handle, at the foot
        ao = 1 - 0.45 * np.clip(1 - (y - 0.0) / 0.012, 0, 1) * (r > 0.03)
        ao = ao * (1 - 0.55 * (r < 0.0368) * (y > COFFEE + 0.0004) * np.clip((MUG_RIM - y) / 0.005, 0, 1))
        A = A * ao[:, None]
        return A.astype(f32)

    def emission(self, p, n, mat):
        p = tuple(np.asarray(a, f32) for a in p)
        n = tuple(np.asarray(a, f32) for a in n)
        g = (np.maximum(_lamp_glint(p, n, 90, 0.3), _lamp_glint(p, n, 50, 0.4, stretch=0.3))
             + _lamp_glint(p, n, 14, 0.03))
        return (g[:, None] * lin(1, .8, .58)).astype(f32)


MUG = Mug()


# --- the plant: a golden pothos in a terracotta pot, on its saucer, two vines over the lip --------------------
POT_Y0 = SEAT_TOP
SOIL = 0.114
_SAUCER = rounded([(0, -0.001), (0.066, -0.001), (0.083, 0.0155), (0.0795, 0.0168), (0.0655, 0.0058), (0, 0.0058)],
                  [0, 0.002, 0.0022, 0.0016, 0.002, 0])
_POT = rounded([(0, 0.0058), (0.056, 0.0058), (0.0703, 0.092), (0.0782, 0.0935), (0.0808, 0.128), (0.0716, 0.1305),
                (0.0676, SOIL + 0.002), (0.05, SOIL + 0.0025), (0.0, SOIL + 0.005)],
               [0, 0.004, 0.0025, 0.0015, 0.003, 0.0026, 0.002, 0.01, 0])
TERRA, TERRA_D, BLOOM = lin(.74, .42, .28), lin(.58, .3, .2), lin(.84, .74, .64)
SOIL_C, SOIL_L = lin(.2, .14, .1), lin(.36, .27, .2)
LEAF_G, LEAF_D, LEAF_Y, LEAF_U = lin(.23, .44, .15), lin(.1, .25, .08), lin(.74, .72, .4), lin(.42, .52, .3)
STEM_C = lin(.36, .46, .2)


def _frame(az, el, roll):
    ev = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    eu = np.cross(ev, [0.0, 1.0, 0.0])
    eu /= np.linalg.norm(eu)
    ew = np.cross(eu, ev)
    eu, ew = np.cos(roll) * eu + np.sin(roll) * ew, -np.sin(roll) * eu + np.cos(roll) * ew
    return np.stack([eu, ev, ew], 1)


def _bezier(a, b, c, n=7):
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 2 * a + 2 * u * (1 - u) * b + u * u * c


def _pothos():
    rng = np.random.default_rng(5)
    cx, cz = POT_C
    y0 = POT_Y0
    leaves, tubes = [], []

    def leaf(B, az, el, roll, L, wratio=0.37, asym=0.0, droop=0.14, cupv=0.08, fold=0.22, twist=0.0, look=None,
             R=None):
        R = _frame(az, el, roll) if R is None else R
        W = L * wratio
        sh = Sheet(B, R, L, W * (1 - asym), W * (1 + asym), droop / L, cupv / W, fold, twist / W,
                   look if look is not None else (rng.uniform(0, 1), 0.0, rng.uniform(0, 100)))
        leaves.append(sh)
        return sh

    def blade(out, tau, yaw, up_leaf=False):
        """A leaf's frame from its outward direction: the blade tilted tau from flat (its face out and up, its tip
        out and down), or for a young upright leaf, the tip up and out by tau from vertical, the face out; then
        turned yaw about its face."""
        upv = np.array([0.0, 1.0, 0.0])
        if up_leaf:
            ev, ew = upv * np.cos(tau) + out * np.sin(tau), out * np.cos(tau) - upv * np.sin(tau)
        else:
            ev, ew = out * np.cos(tau) - upv * np.sin(tau), out * np.sin(tau) + upv * np.cos(tau)
        ev = ev * np.cos(yaw) + np.cross(ew, ev) * np.sin(yaw)
        return np.stack([np.cross(ev, ew), ev, ew], 1)

    soil = y0 + SOIL + 0.004
    # the crown: a mound of leaves on petioles from the soil, laid over each other like shingles; the top ones
    # young and upright, the outer ones older and larger, hanging from their petioles over the rim
    cc = np.array([cx, soil + 0.035, cz])
    n = 28
    for i in range(n):
        f = (i + 0.5) / n
        pel = np.radians(84 - 100 * f ** 0.9) + rng.uniform(-0.06, 0.06)       # where it sits on the mound
        az = np.radians(i * 137.5 + 40) + rng.uniform(-0.2, 0.2)
        reach = 0.044 + 0.06 * f + rng.uniform(-0.006, 0.006)
        B = cc + reach * np.array([np.cos(pel) * np.cos(az), np.sin(pel) * 1.35, np.cos(pel) * np.sin(az)])
        youth = float(np.clip(1.25 - 2.6 * f, 0, 1))
        L = 0.058 + 0.044 * f ** 0.6 + rng.uniform(-0.005, 0.006)
        laz = az + rng.uniform(-0.25, 0.25)
        out = np.array([np.cos(laz), 0.0, np.sin(laz)])
        if youth > 0.5:
            R = blade(out, rng.uniform(0.25, 0.7), rng.uniform(-0.4, 0.4), up_leaf=True)
        else:
            R = blade(out, np.radians(18 + 50 * f) + rng.uniform(-0.2, 0.2), rng.uniform(-0.6, 0.6))
        sh = leaf(B, 0, 0, 0, L, rng.uniform(0.4, 0.47), rng.uniform(-0.12, 0.12),
                  rng.uniform(0.07, 0.15) * (1 - 0.5 * youth), rng.uniform(0.04, 0.08),
                  rng.uniform(0.1, 0.18) + 0.22 * youth, rng.uniform(-0.06, 0.06),
                  (rng.uniform(0, 1), youth, rng.uniform(0, 100)), R=R)
        S = np.array([cx, soil - 0.006, cz]) + 0.35 * np.array([B[0] - cx, 0, B[2] - cz]) + rng.uniform(-0.01, 0.01, 3) * [1, 0, 1]
        ev = sh.R[:, 1]
        C = B - ev * 0.25 * np.linalg.norm(B - S) + np.array([0, 0.015, 0])
        C[1] = max(C[1], S[1] + 0.02)
        tubes.append(Tube(_bezier(S, C, B - ev * 0.002), np.linspace(0.002, 0.0016, 7)))
    # two vines over the pot's lip, the longer down onto the cushion, small leaves along them
    for az0, path, nleaf in ((np.radians(112), [(0.045, SOIL + 0.004), (0.072, 0.136), (0.087, 0.127), (0.091, 0.1),
                                                  (0.087, 0.065), (0.091, 0.035), (0.103, 0.012), (0.122, 0.0035),
                                                  (0.15, 0.0035), (0.17, 0.0045)], 4),
                             (np.radians(18), [(0.04, SOIL + 0.004), (0.072, 0.137), (0.088, 0.127), (0.092, 0.1),
                                               (0.089, 0.068), (0.093, 0.045)], 2)):
        P = []
        for k, (rho, hy) in enumerate(path):
            a = az0 + 0.12 * k / len(path) + 0.04 * np.sin(k * 1.7)
            P.append((cx + rho * np.cos(a), y0 + hy, cz + rho * np.sin(a)))
        C = spline(P, 4)
        tubes.append(Tube(C, np.linspace(0.0019, 0.0014, len(C))))
        for j in range(nleaf):
            q = int(len(C) * (0.3 + 0.62 * j / max(nleaf - 1, 1)))
            q = min(q, len(C) - 2)
            p0 = C[q]
            out = np.array([p0[0] - cx, 0, p0[2] - cz])
            out /= np.linalg.norm(out)
            side = np.cross(out, [0, 1, 0]) * (1 if j % 2 else -1)
            on_cushion = p0[1] < y0 + 0.01
            tip = p0 + out * 0.012 + side * 0.008 + np.array([0, 0.004 if on_cushion else 0.002, 0])
            tubes.append(Tube(_bezier(p0, p0 + out * 0.006 + np.array([0, 0.004, 0]), tip, 4), 0.0013))
            laz = np.arctan2(out[2], out[0]) + (0.5 if j % 2 else -0.5)
            el = 0.05 if on_cushion else rng.uniform(-1.2, -0.85)
            leaf(tip, laz, el, rng.uniform(-0.3, 0.3), rng.uniform(0.04, 0.052), rng.uniform(0.35, 0.4),
                 rng.uniform(-0.1, 0.1), 0.02 if on_cushion else 0.1, 0.06, 0.2, 0.0,
                 (rng.uniform(0, 1), 0.2, rng.uniform(0, 100)))
    return leaves, tubes


class Plant(Prop):
    def __init__(self):
        self.pot = Lathe(POT_C[0], POT_Y0, POT_C[1], [mirror(_SAUCER), mirror(_POT)], 0.00025, POT)
        self.leaves, tubes = _pothos()
        self.tubes = [Tube(t.C[i:i + 3], t.r[i:i + 3]) for t in tubes for i in range(0, len(t.C) - 1, 2)]
        super().__init__([self.pot] + self.leaves + self.tubes)

    def albedo(self, p, n, mat):
        p = tuple(np.asarray(a, f32) for a in p)
        n = tuple(np.asarray(a, f32) for a in n)
        A = np.zeros((len(mat), 3), f32)
        pot = mat == POT
        if pot.any():
            A[pot] = self._pot(tuple(a[pot] for a in p), tuple(a[pot] for a in n))
        lf = mat == LEAF
        if lf.any():
            A[lf] = self._leaf(tuple(a[lf] for a in p), tuple(a[lf] for a in n))
        return A

    def _pot(self, p, n):
        r, y, x, z = self.pot.local(p)
        m = fbm(p[0] * 40, p[1] * 40, p[2] * 40)
        A = TERRA * (1 + 0.07 * m)[:, None]
        A = A * (1 - 0.25 * smooth((m - 0.2) / 0.5))[:, None] + TERRA_D * (0.25 * smooth((m - 0.2) / 0.5))[:, None]
        bl = smooth((fbm(p[0] * 25 + 5, p[1] * 18, p[2] * 25) - 0.25) / 0.4) * (0.35 + 0.65 * smooth((0.05 - y) / 0.04))
        A = A * (1 - 0.35 * bl[:, None]) + BLOOM * (0.35 * bl)[:, None]
        saucer = y < 0.017
        A[saucer] = A[saucer] * 0.93
        # the shade under the rim band, and at the foot in the saucer
        ao = 1 - 0.4 * np.clip(1 - (0.0925 - y) / 0.012, 0, 1) * (y < 0.0925) * (r > 0.06)
        ao = ao * (1 - 0.35 * np.clip(1 - (y - 0.006) / 0.012, 0, 1) * (y > 0.005) * (r < 0.068))
        A = A * ao[:, None]
        soil = (y > SOIL - 0.002) & (r < 0.0682)
        if soil.any():
            q = tuple(a[soil] for a in p)
            g = fbm(q[0] * 150, q[1] * 150, q[2] * 150, 2)
            crumbs = smooth((g - 0.15) / 0.35)
            S = SOIL_C * (1 - crumbs[:, None]) + SOIL_L * crumbs[:, None]
            S = S * (0.7 + 0.3 * smooth((0.066 - r[soil]) / 0.01))[:, None]
            A[soil] = S
        return A

    def _leaf(self, p, n):
        npt = len(p[0])
        best = np.full(npt, 0.004, f32)
        who = np.full(npt, -1)
        loc = [np.zeros(npt, f32) for _ in range(3)]
        for i, sh in enumerate(self.leaves):
            c, r = sh.sphere()
            near = np.nonzero((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2 < r * r)[0]
            if not len(near):
                continue
            u, v, w = sh.to_local(tuple(a[near] for a in p))
            e = np.abs(w - sh.height(u, v))
            ok = (e < best[near]) & (sh.outline(u, v) < 0.05)
            k = near[ok]
            best[k], who[k] = e[ok], i
            for j, a in enumerate((u, v, w)):
                loc[j][k] = a[ok]
        A = np.tile(STEM_C, (npt, 1))
        # stems: lighter green, browner toward the soil
        st = who < 0
        if st.any():
            yy = p[1][st] - (POT_Y0 + SOIL)
            rs = np.sqrt((p[0][st] - POT_C[0]) ** 2 + (p[2][st] - POT_C[1]) ** 2)
            A[st] = STEM_C * ((0.75 + 0.25 * smooth(yy / 0.05)) * (0.45 + 0.55 * smooth((rs - 0.02) / 0.06)))[:, None]
        for i, sh in enumerate(self.leaves):
            k = np.nonzero(who == i)[0]
            if not len(k):
                continue
            u, v = loc[0][k], loc[1][k]
            var, youth, seed = sh.look
            wd = np.where(u < 0, sh.wl, sh.wr)
            x, t = np.abs(u) / wd, v / sh.L
            # the upper face or the paler underside, as the normal says
            ew = sh.R[:, 2]
            up = (n[0][k] * ew[0] + n[1][k] * ew[1] + n[2][k] * ew[2]) > 0
            base = LEAF_G * (1 - 0.4 * youth) + lin(.42, .62, .2) * (0.4 * youth)
            mott = fbm(u * 70 + seed, v * 70, seed * 0.37)
            col = base[None, :] * (1 + 0.12 * mott)[:, None]
            # golden marbling: streaks along the lateral veins, from the midrib out
            ph = (t - 0.62 * x ** 1.3) * 6
            st_ = noise(ph * 1.6 + seed, x * 3.2, seed * 1.3) + 0.5 * noise(ph * 3.3, x * 6 + seed, 2.2)
            gold = smooth((st_ - (1.02 - 0.8 * var)) / 0.3) * np.clip(x * 5, 0, 1) * np.clip((1 - x) * 4, 0, 1)
            col = col * (1 - 0.5 * gold[:, None]) + LEAF_Y * (0.5 * gold)[:, None]
            # the midrib, a little lighter; the margin darker; the underside paler
            mid = np.exp(-(u / 0.0017) ** 2) * np.clip(1 - t, 0, 1)
            col = col * (1 - 0.25 * mid[:, None]) + lin(.6, .66, .34) * (0.25 * mid)[:, None]
            edge = np.clip(1 + sh.outline(u, v) / 0.04, 0, 1)
            col = col * (1 - 0.15 * edge)[:, None]
            col = np.where(up[:, None], col, col * 0.5 + LEAF_U * 0.5)
            A[k] = col
        # the plant's own shade: deep in the crown, near the soil
        cx, cz = POT_C
        rr = np.sqrt((p[0] - cx) ** 2 + (p[2] - cz) ** 2)
        hy = p[1] - (POT_Y0 + SOIL)
        ao = 0.55 + 0.45 * smooth(rr / 0.09) * 0.6 + 0.45 * 0.4 * smooth(hy / 0.16)
        A = A * np.clip(ao, 0, 1)[:, None]
        # the glossy leaves mirror the window a little
        v = _view(p)
        rf = _reflect(n, v)
        cos = np.clip(n[0] * v[0] + n[1] * v[1] + n[2] * v[2], 0, 1)
        fres = 0.04 + 0.96 * (1 - cos) ** 5
        A = A + (_window_seen(p, rf) * fres * 0.5 * (who >= 0))[:, None] * lin(.9, .9, .85)
        return A.astype(f32)

    def emission(self, p, n, mat):
        p = tuple(np.asarray(a, f32) for a in p)
        n = tuple(np.asarray(a, f32) for a in n)
        g = _lamp_glint(p, n, 40, 0.05) * (mat == LEAF)
        return (g[:, None] * lin(1, .82, .6)).astype(f32)


PLANT = Plant()


# --- the lamp: a turned walnut base, a brass stem, a linen shade (lit, it glows), the bulb in it -------------
_BASE = rounded([(0, 0.0), (0.066, 0.0), (0.0685, 0.006), (0.064, 0.012), (0.05, 0.018), (0.046, 0.026),
                 (0.05, 0.034), (0.044, 0.048), (0.048, 0.07), (0.03, 0.098), (0.013, 0.112), (0.0175, 0.117),
                 (0.0175, 0.123), (0.0068, 0.1265), (0.0068, 0.312), (0.016, 0.316), (0.016, 0.354), (0.01, 0.36),
                 (0.0, 0.36)],
                [0, 0.002, 0.003, 0.003, 0.004, 0.003, 0.004, 0.01, 0.02, 0.012, 0.004, 0.002, 0.002, 0.001, 0.002,
                 0.002, 0.003, 0.002, 0])
_BULB = spline([(0.0, 0.353), (0.0105, 0.354), (0.0195, 0.361), (0.0275, 0.376), (0.0272, 0.39), (0.018, 0.403),
                (0.0, 0.4085)], 6)
SH_B, SH_T, SH_RB, SH_RT, SH_TH, SH_BEAD = 0.3, 0.52, 0.146, 0.106, 0.0011, 0.0024


def _shade_polys():
    a, b = np.array([SH_RB, SH_B]), np.array([SH_RT, SH_T])
    e = (b - a) / np.linalg.norm(b - a)
    nrm = np.array([e[1], -e[0]])                                   # outward
    band = [a + nrm * SH_TH, b + nrm * SH_TH, b - nrm * SH_TH, a - nrm * SH_TH]
    ring = lambda c: [c + SH_BEAD * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 40, endpoint=False)]
    return [np.array(band), np.array(ring(a + e * 0.0015)), np.array(ring(b - e * 0.0015))]


WALNUT, WALNUT_L, BRASS = lin(.4, .25, .15), lin(.56, .38, .24), lin(.82, .64, .34)
LINEN = lin(.9, .84, .72)
GLOW = lin(1, .8, .56)


class Lamp(Prop):
    def __init__(self):
        x, y, z = LAMP
        self.base = Lathe(x, y, z, [mirror(_BASE)], 0.0002, DARK)
        self.shade = Lathe(x, y, z, _shade_polys() + [mirror(_BULB)], 0.0003, SHADE)
        super().__init__([self.base, self.shade])

    def _fabric(self, p, y, ang):
        """The linen's slubs (streaks round the shade) and its seam, as 0-1 darkenings."""
        s = noise(ang * 4.0, y * 160, 0.3) * 0.6 + noise(ang * 9.0 + 4, y * 230, 1.1) * 0.4
        seam = np.exp(-(((ang - 2.9 + np.pi) % (2 * np.pi) - np.pi) * 0.13 / 0.0022) ** 2)   # toward the window
        return s, seam

    def albedo(self, p, n, mat):
        p = tuple(np.asarray(a, f32) for a in p)
        n = tuple(np.asarray(a, f32) for a in n)
        A = np.zeros((len(mat), 3), f32)
        r, y, x, z = self.base.local(p)
        ang = np.arctan2(z, x)
        dk = mat == DARK
        if dk.any():
            yy, rr = y[dk], r[dk]
            wood = yy < 0.124
            # turned walnut: the grain's figure, rings where the tool cut across it
            g = noise(p[0][dk] * 30, yy * 90 + 3 * noise(p[0][dk] * 40, yy * 40, p[2][dk] * 40), p[2][dk] * 30)
            W = WALNUT * (1 + 0.18 * g)[:, None]
            W = W * (1 - 0.3 * smooth((g - 0.3) / 0.4))[:, None] + WALNUT_L * (0.3 * smooth((g - 0.3) / 0.4))[:, None]
            # brass: what it mirrors (the window, bright; the room, dark), soft
            nn = tuple(a[dk] for a in n)
            v = _view(tuple(a[dk] for a in p))
            rf = _reflect(nn, v)
            win = _window_seen(tuple(a[dk] for a in p), rf)
            env = 0.25 + 0.35 * smooth(rf[1] * 2 + 0.3) + 2.2 * win
            Bm = BRASS * env[:, None]
            # the wax on the walnut catches the window too
            cos = np.clip(nn[0] * v[0] + nn[1] * v[1] + nn[2] * v[2], 0, 1)
            W = W + (win * (0.04 + 0.96 * (1 - cos) ** 5) * 0.6)[:, None] * lin(.9, .88, .82)
            A[dk] = np.where(wood[:, None], W, Bm)
        sh = mat == SHADE
        if sh.any():
            yy, rr, aa = y[sh], r[sh], ang[sh]
            s, seam = self._fabric(None, yy, aa)
            L_ = LINEN * (1 + 0.05 * s)[:, None] * (1 - 0.12 * seam)[:, None]
            bead = (np.abs(yy - SH_B - 0.0015) < 0.0035) | (np.abs(yy - SH_T + 0.0015) < 0.0035)
            L_[bead] *= 0.85
            bulb = rr < 0.03
            L_[bulb] = lin(.95, .93, .9)
            A[sh] = L_
        return A

    def emission(self, p, n, mat):
        p = tuple(np.asarray(a, f32) for a in p)
        n = tuple(np.asarray(a, f32) for a in n)
        E = np.zeros((len(mat), 3), f32)
        sh = mat == SHADE
        if sh.any():
            r, y, x, z = self.shade.local(tuple(a[sh] for a in p))
            ang = np.arctan2(z, x)
            outward = (n[0][sh] * x + n[2][sh] * z) / np.maximum(r, 1e-6) > 0
            # the bulb lights the shade most nearly level with it, falling off toward the edges
            d2 = r * r + (y - BULB_Y) ** 2
            fall = 0.024 / d2 * (r / np.sqrt(d2))
            s, seam = self._fabric(None, y, ang)
            out = fall * (1 + 0.06 * s) * (1 - 0.45 * seam) * 0.9
            inner = fall * 2.4
            e = np.where(outward, out, inner)
            bead = (np.abs(y - SH_B - 0.0015) < 0.0035) | (np.abs(y - SH_T + 0.0015) < 0.0035)
            e = np.where(bead, e * 0.3, e)
            bulb = r < 0.03
            e = np.where(bulb, 9.0, e)
            E[sh] = e[:, None] * GLOW
        dk = mat == DARK
        if dk.any():
            q, nn = tuple(a[dk] for a in p), tuple(a[dk] for a in n)
            E[dk] = (_lamp_glint(q, nn, 30, 0.01)[:, None] * GLOW).astype(f32)
        return E


LAMPP = Lamp()
# --- the listener's things, at the front of the seat: headphones, their cable over the edge, a paperback ------
# They lie where the camera's pull-back reaches last: the cushion's front edge, and the cable down the seat's face.
CUSH_FRONT, SEAT_FRONT, LEDGE_Y = 0.58, 0.62, 0.5      # the cushion's front face, the seat's, and the ledge between
PHONES_C, PHONES_YAW = (0.35, 0.49), np.radians(-14)   # the headband's centre (x, z), and how they are turned
BOOK_C, BOOK_YAW = (-0.09, 0.47), np.radians(36)


def _yaw(a):
    c, s = np.cos(a), np.sin(a)
    return f32(c), f32(s)


class _Local:
    """A frame on the cushion: origin (x, SEAT_TOP, z), turned yaw about the vertical."""

    def __init__(self, cx, cz, yaw, y0=SEAT_TOP):
        self.cx, self.y0, self.cz = f32(cx), f32(y0), f32(cz)
        self.c, self.s = _yaw(yaw)

    def to_local(self, p):
        dx, dz = p[0] - self.cx, p[2] - self.cz
        return self.c * dx - self.s * dz, p[1] - self.y0, self.s * dx + self.c * dz

    def to_world(self, x, y, z):
        return (self.cx + self.c * x + self.s * z, self.y0 + y, self.cz - self.s * x + self.c * z)

    def box(self, lo, hi):
        pts = np.array([self.to_world(a, b, c) for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
        return pts.min(0) - 1e-3, pts.max(0) + 1e-3


# an earcup lying pad-down: its cross-section from the axis (r, y), its oval squashed across (EX)
_CUP = rounded([(0.0, 0.012), (0.0185, 0.012), (0.0205, 0.0), (0.0405, 0.0), (0.0425, 0.0175), (0.0395, 0.0197),
                (0.0405, 0.0225), (0.0378, 0.034), (0.02, 0.0398), (0.0, 0.0408)],
               [0, 0.002, 0.004, 0.0085, 0.0045, 0.0012, 0.002, 0.009, 0.02, 0])
CUP_G = bake_polys([mirror(_CUP)], (-0.001, -0.012), (0.054, 0.05), 0.0002)
EX = 0.8
CUP_X = 0.093                            # the cups' centres, either side of the band's centre
CUP_Z = 0.03
BAND_R, BAND_T, BAND_H = 0.088, 0.0047, 0.0148   # the headband: its radius, half its thickness, half its height
BAND_ENDS = (np.radians(191), np.radians(349))   # it runs round the back, from cup to cup


class EarCup(Solid):
    mat = PHONES

    def __init__(self, frame, x, z):
        self.F, self.x, self.z = frame, f32(x), f32(z)
        r = 0.042
        self.box = frame.box((x - r, -0.001, z - r), (x + r, 0.037, z + r))

    def field(self, p):
        x, y, z = self.F.to_local(p)
        xs, zs = (x - self.x) / f32(EX), z - self.z
        return CUP_G.dist(np.sqrt(xs * xs + zs * zs), y) * f32(EX)

    def normal(self, p):
        x, y, z = self.F.to_local(p)
        xs, zs = (x - self.x) / f32(EX), z - self.z
        rs = np.sqrt(xs * xs + zs * zs)
        gr, gy = CUP_G.grad(rs, y)
        inv = 1 / np.maximum(rs, f32(1e-6))
        nx, ny, nz = gr * xs * inv / f32(EX), gy, gr * zs * inv
        F = self.F
        return _norm3(F.c * nx + F.s * nz, ny, -F.s * nx + F.c * nz)


class HeadBand(Solid):
    """A ribbon standing on its edge, round the back from cup to cup, its ends in the cups' yokes."""
    mat = PHONES

    def __init__(self, frame):
        self.F = frame
        r = BAND_R + 0.012
        self.box = frame.box((-r, 0.0, -r), (r, 2 * BAND_H + 0.006, 0.02))
        self.ends = [np.array([BAND_R * np.cos(a), BAND_R * np.sin(a)], f32) for a in BAND_ENDS]

    def field(self, p):
        x, y, z = self.F.to_local(p)
        rho = np.sqrt(x * x + z * z)
        ang = np.arctan2(z, x) % (2 * np.pi)
        on = (ang > BAND_ENDS[0]) & (ang < BAND_ENDS[1])
        dq = rho - f32(BAND_R)
        de = np.minimum(np.hypot(x - self.ends[0][0], z - self.ends[0][1]),
                        np.hypot(x - self.ends[1][0], z - self.ends[1][1]))
        q = np.where(on, dq, de)
        yy = y - f32(BAND_H + 0.004)
        rr = f32(0.0035)
        ax, ay = np.abs(q) - (f32(BAND_T) - rr), np.abs(yy) - (f32(BAND_H) - rr)
        return np.sqrt(np.maximum(ax, 0) ** 2 + np.maximum(ay, 0) ** 2) + np.minimum(np.maximum(ax, ay), 0) - rr


class Paperback(Solid):
    """An open paperback lying face-down: two halves sloping from the spine to the cushion, a rounded spine."""
    mat = BOOK
    LEN, WID, TH, TILT = 0.089, 0.108, 0.0085, np.radians(16)       # half its length; a half's width, thickness
    BOW = 1.1                                                       # how its covers bow (1/m)

    def __init__(self, frame):
        self.F = frame
        a = self.TILT
        self.ys = f32(self.WID * np.sin(a) + 0.5 * self.TH * np.cos(a))  # the spine's height
        self.halves = []
        for k in (-1, 1):
            es = np.array([0.0, -np.sin(a), k * np.cos(a)], f32)    # down the slope from the spine
            et = np.array([0.0, np.cos(a), k * np.sin(a)], f32)     # out of the cover
            c = np.array([0.0, self.ys, 0.0], f32) + es * f32(self.WID / 2)
            self.halves.append((c, es, et))
        L = self.LEN + 0.003
        self.box = frame.box((-L, -0.001, -self.WID - 0.004), (L, self.ys + 0.008, self.WID + 0.004))

    def half_local(self, q, k):
        c, es, et = self.halves[k]
        x, y, z = q[0] - c[0], q[1] - c[1], q[2] - c[2]
        return x, y * es[1] + z * es[2], y * et[1] + z * et[2]

    def field(self, p):
        q = self.F.to_local(p)
        d = None
        rr = f32(0.0016)
        for k in (0, 1):
            x, s_, t_ = self.half_local(q, k)
            t_ = t_ - f32(self.BOW) * ((self.WID / 2) ** 2 - s_ * s_) * (np.abs(s_) < self.WID / 2)
            ax = np.abs(x) - f32(self.LEN - 0.0016)
            ay = np.abs(s_) - f32(self.WID / 2 - 0.0016)
            az = np.abs(t_) - f32(self.TH / 2 - 0.0016)
            m = np.maximum(np.maximum(ax, ay), az)
            e = np.sqrt(np.maximum(ax, 0) ** 2 + np.maximum(ay, 0) ** 2 + np.maximum(az, 0) ** 2) + np.minimum(m, 0) - rr
            e = e * f32(0.88)                                           # the bow bends space: a safe bound
            d = e if d is None else np.minimum(d, e)
        # the spine, rounded over the top where the halves meet
        sx = np.abs(q[0]) - f32(self.LEN - 0.002)
        sr = np.sqrt((q[1] - (self.ys - f32(0.0045))) ** 2 + q[2] ** 2) - f32(0.005)
        sp = np.sqrt(np.maximum(sx, 0) ** 2 + np.maximum(sr, 0) ** 2) + np.minimum(np.maximum(sx, sr), 0) - f32(0.0015)
        return smin(d, sp, f32(0.004))


def _cable():
    """The headphones' cable: out of the left cup, a loose loop on the cushion, over its front edge, across the
    ledge and down the seat's face, its plug hanging at the end."""
    F = _Local(PHONES_C[0], PHONES_C[1], PHONES_YAW)
    start = [F.to_world(-CUP_X - 0.03, 0.009, CUP_Z + 0.012), F.to_world(-CUP_X - 0.05, 0.0035, CUP_Z + 0.03),
             F.to_world(-CUP_X - 0.06, 0.002, CUP_Z + 0.06)]
    x0 = start[-1][0]
    r = 0.0017
    pts = start + [(x0 + 0.01, SEAT_TOP + r, CUSH_FRONT - 0.03), (x0 + 0.02, SEAT_TOP + r * 0.6, CUSH_FRONT - 0.004),
                   (x0 + 0.023, SEAT_TOP - 0.006, CUSH_FRONT + r), (x0 + 0.025, SEAT_TOP - 0.03, CUSH_FRONT + r),
                   (x0 + 0.027, LEDGE_Y + r + 0.002, CUSH_FRONT + 0.006), (x0 + 0.029, LEDGE_Y + r, SEAT_FRONT - 0.012),
                   (x0 + 0.03, LEDGE_Y - 0.002, SEAT_FRONT + r), (x0 + 0.031, LEDGE_Y - 0.04, SEAT_FRONT + r),
                   (x0 + 0.032, LEDGE_Y - 0.075, SEAT_FRONT + r)]
    C = spline(pts, 5)
    plug = [C[-1], C[-1] + np.array([0.0, -0.016, 0.0]), C[-1] + np.array([0.0, -0.028, 0.0])]
    return C, plug


PH_SHELL, PH_PAD, PH_BAND, PH_CABLE, PH_METAL = lin(.7, .62, .52), lin(.34, .26, .2), lin(.66, .58, .48), \
    lin(.3, .26, .24), lin(.78, .7, .5)
COVER, COVER_D, PAGES = lin(.78, .58, .26), lin(.55, .36, .16), lin(.9, .86, .76)


class Headphones(Prop):
    def __init__(self):
        self.Fp = _Local(PHONES_C[0], PHONES_C[1], PHONES_YAW)
        self.cups = [EarCup(self.Fp, -CUP_X, CUP_Z), EarCup(self.Fp, CUP_X, CUP_Z)]
        self.band = HeadBand(self.Fp)
        C, plug = _cable()
        self.cable = [Tube(C[i:i + 3], 0.0017) for i in range(0, len(C) - 1, 2)]
        self.plug = [Tube(plug[:2], 0.0029), Tube(plug[1:], 0.0014)]
        for t in self.cable + self.plug:
            t.mat = PHONES
        super().__init__(self.cups + [self.band] + self.cable + self.plug)

    def albedo(self, p, n, mat):
        return self._phones(tuple(np.asarray(a, f32) for a in p), tuple(np.asarray(a, f32) for a in n))

    def _phones(self, p, n):
        x, y, z = self.Fp.to_local(p)
        dc = np.minimum(self.cups[0].field(p), self.cups[1].field(p))
        db = self.band.field(p)
        A = np.tile(PH_CABLE, (len(x), 1))
        cup = (dc < 0.0015) & (dc <= db)
        band = (db < 0.0015) & (db < dc)
        if cup.any():
            pad = y[cup] < 0.0188
            g = noise(x[cup] * 260, y[cup] * 260, z[cup] * 260)              # the leather's grain, ~2 px, faint
            col = np.where(pad[:, None], PH_PAD * (1 + 0.06 * g)[:, None], PH_SHELL * (1 + 0.025 * g)[:, None])
            # the maker's mark: a small round boss in the middle of each shell, a shade darker
            cx_ = np.where(x[cup] < 0, -CUP_X, CUP_X)
            boss = smooth((0.0085 - np.hypot((x[cup] - cx_) / EX, z[cup] - CUP_Z)) / 0.0025) * (y[cup] > 0.03)
            col = col * (1 - 0.14 * boss)[:, None]
            A[cup] = col
        if band.any():
            rho = np.hypot(x[band], z[band])
            inner = smooth((BAND_R - rho) / 0.0015)                        # the cushioned inside, darker
            A[band] = PH_BAND * (1 - inner)[:, None] + (PH_PAD * 0.4 + PH_BAND * 0.6) * inner[:, None]
        plug = (~cup & ~band) & (p[1] < LEDGE_Y - 0.072)
        A[plug] = PH_METAL
        # contact shade on the cushion
        ao = 1 - 0.35 * np.clip(1 - y / 0.008, 0, 1) * (p[1] > SEAT_TOP - 0.002)
        return (A * ao[:, None]).astype(f32)

class Book(Prop):
    def __init__(self):
        self.book = Paperback(_Local(BOOK_C[0], BOOK_C[1], BOOK_YAW))
        super().__init__([self.book])

    def albedo(self, p, n, mat):
        p = tuple(np.asarray(a, f32) for a in p)
        B = self.book
        B = self.book
        q = B.F.to_local(p)
        A = np.tile(PAGES, (len(p[0]), 1))
        best = np.full(len(p[0]), np.inf, f32)
        for k in (0, 1):
            x, s_, t_ = B.half_local(q, k)
            t_ = t_ - B.BOW * ((B.WID / 2) ** 2 - s_ * s_) * (np.abs(s_) < B.WID / 2)
            cover = smooth((t_ - (B.TH / 2 - 0.0009)) / 0.0004)              # the outer 0.5 mm: the cover
            # the back cover: mustard, a cream panel for the blurb, a darker band at the spine
            sp = s_ + B.WID / 2                                              # from the spine, along the half
            panel = (smooth((0.062 - np.abs(x + 0.004)) / 0.002) * smooth((sp - 0.026) / 0.002)
                     * smooth((0.088 - sp) / 0.002)) * (k == 1)
            spine = smooth((0.011 - sp) / 0.002)
            col = COVER * (1 - 0.45 * panel)[:, None] + PAGES * (0.45 * panel)[:, None]
            col = col * (1 - spine)[:, None] + COVER_D * spine[:, None]
            # the page block: cream, its leaves' edges a soft, uneven grain
            leaves = PAGES * (1 + 0.03 * noise(x * 40, t_ * 900, s_ * 40))[:, None]
            c = leaves * (1 - cover)[:, None] + col * cover[:, None]
            dist = np.abs(np.maximum(np.maximum(np.abs(x) - B.LEN, np.abs(s_) - B.WID / 2), np.abs(t_) - B.TH / 2))
            near = dist < best
            best = np.where(near, dist, best)
            A[near] = c[near]
        # the spine's ridge
        ridge = smooth((0.0065 - np.hypot(q[1] - (B.ys - 0.0045), q[2])) / 0.002)
        A = A * (1 - ridge)[:, None] + COVER_D * ridge[:, None]
        ao = 1 - 0.3 * np.clip(1 - q[1] / 0.006, 0, 1)
        return (A * ao[:, None]).astype(f32)


# --- two hardbacks on the side table, coming into view from the right as the camera backs away ---------------
TOMES_AT = [((1.47, 0.47), np.radians(8), (0.083, 0.0145, 0.118), lin(.34, .42, .36)),     # (x, z), yaw, half sizes, cloth
            ((1.465, 0.475), np.radians(-5), (0.074, 0.012, 0.107), lin(.52, .24, .2))]
BOARD = 0.0025                           # a cover board's thickness; boards stand 3 mm proud of the pages


class Hardback(Solid):
    mat = TOMES

    def __init__(self, xz, yaw, half, y0):
        self.F = _Local(xz[0], xz[1], yaw, y0)
        self.h = np.asarray(half, f32)
        self.box = self.F.box((-half[0] - 0.002, 0.0, -half[2] - 0.002), (half[0] + 0.002, 2 * half[1] + 0.001,
                                                                          half[2] + 0.002))

    def local(self, p):
        x, y, z = self.F.to_local(p)
        return x, y - self.h[1], z

    def field(self, p):
        x, y, z = self.local(p)
        r = f32(0.0012)
        ax, ay, az = np.abs(x) - (self.h[0] - r), np.abs(y) - (self.h[1] - r), np.abs(z) - (self.h[2] - r)
        box = (np.sqrt(np.maximum(ax, 0) ** 2 + np.maximum(ay, 0) ** 2 + np.maximum(az, 0) ** 2)
               + np.minimum(np.maximum(np.maximum(ax, ay), az), 0) - r)
        # the page block sits 3 mm in from the boards' edges, on the three sides away from the spine (-x)
        inner = np.maximum(np.abs(y) - (self.h[1] - f32(BOARD)), -(np.maximum(x - (self.h[0] - f32(0.003)),
                                                                               np.abs(z) - (self.h[2] - f32(0.003)))))
        return np.maximum(box, -np.maximum(inner, -(x + self.h[0] - f32(0.006))))


class Tomes(Prop):
    def __init__(self):
        y = TABLE_TOP
        books = []
        for xz, yaw, half, col in TOMES_AT:
            b = Hardback(xz, yaw, half, y)
            b.cloth = col
            books.append(b)
            y += 2 * half[1]
        super().__init__(books)

    def albedo(self, p, n, mat):
        p = tuple(np.asarray(a, f32) for a in p)
        A = np.zeros((len(p[0]), 3), f32)
        best = np.full(len(p[0]), np.inf, f32)
        for b in self.parts:
            d = np.abs(b.field(p))
            x, y, z = b.local(p)
            pages = (np.abs(y) < b.h[1] - BOARD + 2e-4) & (x > -b.h[0] + 0.004)
            weave = 1 + 0.04 * noise(x * 160, y * 160, z * 160)
            spine = smooth((-b.h[0] + 0.004 - x) / 0.002)
            cloth = b.cloth[None, :] * (weave * (1 - 0.25 * spine))[:, None]
            leaves = PAGES[None, :] * (1 + 0.03 * noise(x * 40, y * 900, z * 40))[:, None] * \
                (1 - 0.3 * smooth((np.abs(y) - (b.h[1] - BOARD - 0.0015)) / 0.0015))[:, None]
            c = np.where(pages[:, None], leaves, cloth)
            near = d < best
            best = np.where(near, d, best)
            A[near] = c[near]
        return A


PHONESP = Headphones()
BOOKP = Book()
TOMESP = Tomes()
PROPS = [MUG, PLANT, LAMPP, PHONESP, BOOKP, TOMESP]
ON_SEAT = [MUG, PLANT, PHONESP, BOOKP]   # what can shade the seat and the cat from the sun
MATERIALS = (CUP, POT, LEAF, SHADE, DARK, PHONES, BOOK, TOMES)
