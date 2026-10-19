"""The cat in First Light's film: asleep on the window seat, curled, breathing; it lifts its head once.

The room (room.py) asks the cat four things, and nothing else about it:
  bound(t)            a sphere (centre, radius) that holds the whole cat at time t
  hit(o, d, t)        where rays o + s d (o a point, d a tuple of arrays, not unit) first meet the cat:
                      (s, nx, ny, nz), s = inf where they miss; n the unit outward normal
  occludes(P, s, t)   for points P (N x 3) and the unit direction s toward the sun: True where the cat is in the way
  albedo(p, n, t)     the cat's colour (linear RGB, N x 3) at points p on it with normals n
  translucency(p, n, t)  how much sun from behind comes through to the side facing us (the ears), 0-1
The world is in metres: x right, y up, z out of the window wall into the room (the camera looks along -z, from
about z = 2.4-3.0 at eye height 1.3). The cushion's top is at y = CUSHION_TOP; the cat lies on it around CENTER.
lift(t) (0-1) is how far it has raised its head; the film sets it from the score.

How it is made: the cat is a signed-distance field, sculpted from smooth-blended ellipsoids, tapered tubes along
curves (its spine, legs and tail) and two cupped, rounded triangles (its ears). The field is costly to evaluate,
so it is baked once, at import, into two grids: the body (with the tail, legs and paws) and the head (with a neck
that dives into the shoulders). Each frame then marches rays through trilinear lookups of the grids: the body's
through a breathing warp (the ribs swell, the tail and paws keep still), the head's through the rigid motion of
its lift, the two joined by a smooth minimum. Normals come from the grids' baked gradients, interpolated, so the
shading has no facets; the coat (a ginger mackerel tabby) is painted in albedo, in each part's own rest frame, so
its stripes ride with the breath and the lift, and it carries the field's own ambient occlusion.

What keeps it quick (hit over the cat's box and occludes, about 0.2-0.4 s a frame at 1080 and ss 2): the room's rays
come as a 2D grid, so one cone a 4 x 4 tile finds where the cat can begin (tiles it misses are skipped); every other
ray each way is traced, and a ray between four traced ones that met one smooth, face-on piece is filled in from
them, so only the edges and creases are traced in full. Rays that pass within FRINGE of the coat, where tufts of
noise say the fur reaches, meet it too: a soft, backlit edge. occludes reads a map of the cat as the sun sees it,
made once a frame, instead of tracing each point's ray.
"""
import numpy as np

CUSHION_TOP = 0.55
CENTER = (-0.55, CUSHION_TOP + 0.085, 0.3)
ORIGIN = np.array([CENTER[0], CUSHION_TOP, CENTER[2]], np.float32)   # the body's frame: on the cushion, mid-cat
BREATH = 3.4                                                          # seconds a breath


def lift(t):
    return 0.0


def lin(*c):
    return np.array(c, np.float32) ** 2.2


f32 = np.float32


# --- distance primitives, vectorised: points as three float32 arrays ---------------------------------------------
def _len(x, y, z):
    return np.sqrt(x * x + y * y + z * z)


def ellipsoid(p, c, r):
    """An ellipsoid centred at c with semi-axes r (a near-exact bound, after Quilez)."""
    x, y, z = (p[0] - c[0]) / r[0], (p[1] - c[1]) / r[1], (p[2] - c[2]) / r[2]
    k0 = _len(x, y, z)
    k1 = _len(x / r[0], y / r[1], z / r[2])
    return k0 * (k0 - 1) / (k1 + 1e-9)


def tube(p, a, b, ra, rb):
    """A capsule from a to b whose radius runs from ra to rb."""
    ba = np.subtract(b, a)
    px, py, pz = p[0] - a[0], p[1] - a[1], p[2] - a[2]
    h = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / float(ba @ ba), 0, 1)
    return _len(px - ba[0] * h, py - ba[1] * h, pz - ba[2] * h) - (ra + (rb - ra) * h)


def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


def spline(pts, n):
    """A Catmull-Rom curve through pts, n points along it."""
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


def curve_tube(p, C, r0, r1, k):
    """A tapered tube along the polyline C, its segments blended."""
    d = None
    m = len(C) - 1
    for i in range(m):
        ra, rb = r0 + (r1 - r0) * i / m, r0 + (r1 - r0) * (i + 1) / m
        e = tube(p, C[i], C[i + 1], ra, rb)
        d = e if d is None else smin(d, e, k)
    return d


def frame(fwd, up):
    """A rotation (columns: side, up, forward) with forward along fwd and up as near up as it can be."""
    f = np.asarray(fwd, float)
    f /= np.linalg.norm(f)
    s = np.cross(np.asarray(up, float), f)
    s /= np.linalg.norm(s)
    u = np.cross(f, s)
    return np.stack([s, u, f], 1)


def rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def local(p, c, M):
    """Points p into the frame M (columns its axes) placed at c."""
    x, y, z = p[0] - c[0], p[1] - c[1], p[2] - c[2]
    return (M[0, 0] * x + M[1, 0] * y + M[2, 0] * z, M[0, 1] * x + M[1, 1] * y + M[2, 1] * z,
            M[0, 2] * x + M[1, 2] * y + M[2, 2] * z)


# --- the cat, sculpted: the body's frame has its origin on the cushion under the cat's middle -----------------------
# The spine curls round the back, toward the window (-z): the rump at the left, the shoulders at the right; the
# head rests at the right, in front, its nose toward the hind feet; the tail wraps round the front to meet it.
SPINE_C, SPINE_R = np.array([0.0, 0.0, 0.0]), (0.108, 0.08)
SPINE = [(a, rr) for a, rr in ((165, (0.06, 0.064, 0.066)), (198, (0.066, 0.07, 0.064)), (232, (0.066, 0.07, 0.06)),
                               (266, (0.064, 0.067, 0.057)), (300, (0.062, 0.066, 0.058)), (334, (0.06, 0.064, 0.06)),
                               (366, (0.056, 0.06, 0.058)))]
TAIL = spline([(-0.15, 0.05, -0.02), (-0.192, 0.024, 0.05), (-0.152, 0.018, 0.125), (-0.07, 0.016, 0.168),
               (0.01, 0.016, 0.182), (0.074, 0.016, 0.18), (0.126, 0.018, 0.166), (0.156, 0.022, 0.138)], 4)
PAWS = [((0.048, 0.016, 0.134), (0.021, 0.0155, 0.022)), ((0.098, 0.043, 0.177), (0.019, 0.012, 0.021))]
HEAD_C = np.array([0.08, 0.09, 0.085])                           # the head, at rest, in the body's frame
HEAD_M = frame((-0.36, -0.36, 0.86), (0.08, 1, -0.1))              # its axes: nose toward the hind feet, a little down
HEAD_SCALE = 1.08
PIVOT = np.array([0.12, 0.07, 0.03])                               # the neck turns here as the head lifts
KNECK = 0.009                                                      # how softly the head's grid meets the body's


def spine_at(a):
    a = np.radians(a)
    return SPINE_C + np.array([SPINE_R[0] * np.cos(a), 0, SPINE_R[1] * np.sin(a)])


def body_sdf(p):
    """The body: the curled torso, the haunch, the legs and paws, the tail."""
    d = None
    for a, (ra, rr, rv) in SPINE:                                  # the torso: ellipsoids along the spine, blended
        c = spine_at(a)
        c[1] = rv - 0.006
        M = frame(np.array([-np.sin(np.radians(a)) * SPINE_R[0], 0, np.cos(np.radians(a)) * SPINE_R[1]]), (0, 1, 0))
        e = ellipsoid(local(p, c, M), (0, 0, 0), (rr, rv, ra))
        d = e if d is None else smin(d, e, 0.05)
    d = smin(d, ellipsoid(p, (0.0, 0.03, 0.03), (0.085, 0.034, 0.065)), 0.04)     # the belly, in the curl
    d = smin(d, ellipsoid(p, (0.105, 0.066, 0.005), (0.048, 0.05, 0.05)), 0.03)   # the shoulder
    hM = frame((0.55, 0.1, 0.83), (0, 1, 0))
    d = smin(d, ellipsoid(local(p, (-0.078, 0.078, 0.045), hM), (0, 0, 0), (0.052, 0.058, 0.074)), 0.02)  # the haunch
    d = smin(d, tube(p, (-0.06, 0.045, 0.1), (0.0, 0.02, 0.128), 0.024, 0.016), 0.018)   # the hind leg, folded under
    d = smin(d, ellipsoid(p, (0.02, 0.016, 0.127), (0.032, 0.016, 0.019)), 0.01)        # its foot
    d = smin(d, tube(p, (0.105, 0.04, 0.05), (0.06, 0.018, 0.124), 0.022, 0.016), 0.018)   # a foreleg, tucked
    d = smin(d, tube(p, (0.132, 0.04, 0.06), (0.116, 0.036, 0.14), 0.022, 0.016), 0.018)   # and one laid
    d = smin(d, tube(p, (0.116, 0.036, 0.14), (0.102, 0.042, 0.17), 0.016, 0.015), 0.01)  # over the tail
    for c, r in PAWS:
        d = smin(d, ellipsoid(p, c, r), 0.008)                                          # the paws
    d = smin(d, curve_tube(p, TAIL, 0.019, 0.011, 0.006), 0.006)                       # the tail
    return np.maximum(d, -p[1] - 0.004)                            # it sinks a little into the cushion


EARS = [(s, np.array([0.026 * s, 0.028, -0.008]), frame((0.3 * s, -0.05, 1.0), (0.42 * s, 1, -0.12))) for s in (-1, 1)]


def ear(q):
    """An ear in its own frame: a rounded triangle (base on x, tip up y), thin in z, cupped forward, hollowed."""
    x, y, z = q
    w, h = 0.027, 0.05
    ax, ay = np.abs(x), h - y                                       # Quilez's isosceles triangle, apex up
    qx, qy = w, h
    t = np.clip((ax * qx + ay * qy) / (qx * qx + qy * qy), 0, 1)
    ax_, ay_ = ax - qx * t, ay - qy * t
    bx, by = ax - qx * np.clip(ax / qx, 0, 1), ay - qy
    s = -1.0
    da, sa = ax_ * ax_ + ay_ * ay_, s * (ax * qy - ay * qx)
    db, sb = bx * bx + by * by, s * (ay - qy)
    d2 = -np.sqrt(np.minimum(da, db)) * np.sign(np.minimum(sa, sb))
    d2 = d2 + 0.003
    zc = z - 8 * x * x + 0.004 * (y / h)                             # cupped: the front face is hollow
    th = 0.0055 * (1 - 0.55 * np.clip(y / h, 0, 1))
    dz = np.abs(zc) - th
    out = np.minimum(np.maximum(d2, dz), 0) + np.sqrt(np.maximum(d2, 0) ** 2 + np.maximum(dz, 0) ** 2) - 0.0025
    inner = np.maximum(d2 + 0.0065, -(zc - 0.0005))                  # the hollow of the ear, from the front
    return smax(out, -inner, 0.003) * 0.8                          # the cupping bends space: a safe bound


def head_sdf(q):
    """The head in its own frame (x side, y up, z forward, along the nose), with a neck behind it."""
    q = tuple(a / HEAD_SCALE for a in q)
    d = ellipsoid(q, (0, 0.008, -0.008), (0.045, 0.038, 0.043))                      # the cranium
    d = smin(d, ellipsoid(q, (0, -0.012, 0.006), (0.051, 0.029, 0.035)), 0.018)       # cheeks and jowls
    d = smin(d, ellipsoid(q, (0, -0.017, 0.032), (0.023, 0.017, 0.017)), 0.012)       # the muzzle
    for s in (-1, 1):
        d = smin(d, ellipsoid(q, (0.0105 * s, -0.02, 0.039), (0.013, 0.011, 0.0105)), 0.006)  # whisker pads
        d = smax(d, -ellipsoid(q, (0.019 * s, 0.004, 0.038), (0.011, 0.0035, 0.006)), 0.004)  # the closed eyes
    d = smin(d, ellipsoid(q, (0, -0.03, 0.027), (0.013, 0.008, 0.012)), 0.008)        # the chin
    d = smin(d, ellipsoid(q, (0, -0.006, 0.044), (0.007, 0.005, 0.006)), 0.006)       # the nose
    d = smin(d, tube(q, (0, -0.008, -0.02), (0, -0.022, -0.075), 0.036, 0.042), 0.03)  # the neck, into the shoulders
    for s, c, M in EARS:
        d = smin(d, ear(local(q, c, M)), 0.008)
    return np.maximum(d, -0.098 - q[2]) * HEAD_SCALE          # the neck ends inside the body


# --- the baked grids ---------------------------------------------------------------------------------------------
class Grid:
    """A distance field sampled on a box of voxels, with its gradient; looked up trilinearly."""

    def __init__(self, sdf, lo, hi, h):
        self.lo, self.h = np.asarray(lo, f32), f32(h)
        n = np.ceil((np.asarray(hi) - lo) / h).astype(int) + 1
        self.n = n
        xs = [lo[k] + h * np.arange(n[k], dtype=f32) for k in range(3)]
        D = np.empty(tuple(n), f32)
        Y, Z = np.meshgrid(xs[1], xs[2], indexing='ij')
        Y, Z = Y.astype(f32), Z.astype(f32)
        for i in range(n[0]):                                       # a slab at a time: memory stays small
            D[i] = sdf((np.full_like(Y, xs[0][i]), Y, Z))
        g = np.gradient(D, h)
        g = np.stack(g + [np.zeros_like(D)], -1)
        g[..., :3] /= np.linalg.norm(g[..., :3], axis=-1, keepdims=True) + 1e-9
        self.stride = np.array([n[1] * n[2], n[2], 1])
        H = D.astype(np.float16)                                     # each cell's eight corners, gathered as one
        C8 = np.stack([H[i:n[0] - 1 + i, j:n[1] - 1 + j, k:n[2] - 1 + k]
                       for i in (0, 1) for j in (0, 1) for k in (0, 1)], -1)
        C8 = np.concatenate([C8, np.zeros((n[0] - 1, n[1] - 1, 1, 8), np.float16)], 2)   # padded to the voxels' strides
        C8 = np.concatenate([C8, np.zeros((n[0] - 1, 1, n[2], 8), np.float16)], 1)
        self.C8 = np.ascontiguousarray(C8).reshape(-1, 8).view(np.complex128).ravel()
        self.G = g.astype(np.float16).reshape(-1, 4).view(np.int64).ravel()   # four halves gathered as one
        solid = [np.nonzero((D < 0).any(axis=tuple(j for j in range(3) if j != k)))[0] for k in range(3)]
        self.box = (np.array([s[0] for s in solid]) * h + self.lo - h,
                    np.array([s[-1] for s in solid]) * h + self.lo + h)

    def _cell(self, x, y, z):
        """Base index, fractions, and how far outside the box each point is."""
        idx = 0
        fr = []
        out = 0
        inv = 1 / self.h
        for k, a in enumerate((x, y, z)):
            u = (a - self.lo[k]) * inv
            uc = np.clip(u, 0, self.n[k] - 1.001)
            out = out + (u - uc) ** 2
            i = uc.astype(np.intp)
            fr.append(uc - i)
            idx = idx + i * self.stride[k]
        return idx, fr, np.sqrt(out) * self.h

    def dist(self, x, y, z):
        idx, (fx, fy, fz), out = self._cell(x, y, z)
        c = self.C8[idx].view(np.float16).reshape(-1, 8).T.astype(f32, order='C')   # rows: the eight corners
        a0, a1 = c[0] + (c[1] - c[0]) * fz, c[2] + (c[3] - c[2]) * fz
        a2, a3 = c[4] + (c[5] - c[4]) * fz, c[6] + (c[7] - c[6]) * fz
        a0 += (a1 - a0) * fy
        a2 += (a3 - a2) * fy
        a0 += (a2 - a0) * fx
        return a0 + out

    def grad(self, x, y, z):
        idx, (fx, fy, fz), _ = self._cell(x, y, z)
        sx, sy, _ = self.stride
        c = [self.G[idx + o].view(np.float16).reshape(-1, 4)[:, :3].astype(f32)
             for o in (0, 1, sy, sy + 1, sx, sx + 1, sx + sy, sx + sy + 1)]
        fx, fy, fz = fx[:, None], fy[:, None], fz[:, None]
        a0, a1 = c[0] + (c[1] - c[0]) * fz, c[2] + (c[3] - c[2]) * fz
        a2, a3 = c[4] + (c[5] - c[4]) * fz, c[6] + (c[7] - c[6]) * fz
        b0, b1 = a0 + (a1 - a0) * fy, a2 + (a3 - a2) * fy
        return b0 + (b1 - b0) * fx


def _build():
    body = Grid(body_sdf, (-0.25, -0.012, -0.18), (0.215, 0.175, 0.235), 0.0025)
    head = Grid(head_sdf, (-0.075, -0.06, -0.12), (0.075, 0.1, 0.07), 0.0016)
    return body, head


BODY, HEAD = _build()


# --- the cat at time t ---------------------------------------------------------------------------------------------
class Pose:
    """Where the parts are at time t: the breath's swell, and the head's frame in the world."""

    def __init__(self, t):
        self.t = t
        self.breath = f32(0.03 * np.sin(2 * np.pi * t / BREATH))
        lf = float(np.clip(lift(t), 0, 1.2))
        Rl = rot(HEAD_M[:, 0], -0.5 * lf) @ rot((0, 1, 0), -0.25 * lf)   # nose up, and turned toward the light
        self.hc = ORIGIN + PIVOT + Rl @ (HEAD_C - PIVOT) + np.array([0, 0.028 * lf, 0])
        self.hM = (Rl @ HEAD_M).astype(f32)
        self.hc = self.hc.astype(f32)
        lo, hi = HEAD.box
        self.hs = (self.hM @ ((lo + hi) / 2) + self.hc).astype(f32)      # the head's bounding sphere, in the world
        self.hr = f32(np.linalg.norm(hi - lo) / 2)
        self.sunmaps = {}

    def body_local(self, p):
        """World points into the body's rest frame, undoing the breath (the ribcage swells; the rest stays)."""
        x, y, z = p[0] - ORIGIN[0], p[1] - ORIGIN[1], p[2] - ORIGIN[2]
        w = np.exp(-((x + 0.01) ** 2 + (z + 0.02) ** 2) / f32(0.13 ** 2)) * self.breath
        s = 1 / (1 + w)
        return x * (1 + 0.5 * (s - 1)), y * s, (z + 0.02) * s - 0.02

    def head_local(self, p):
        return local(p, self.hc, self.hM)

    def field(self, p):
        """The distance to the cat, a lower bound off the grids, and the head's share of it. The head's grid is
        read only where the head could be nearer than the body is, or near enough to blend with it."""
        db = BODY.dist(*self.body_local(p))
        x, y, z = p[0] - self.hs[0], p[1] - self.hs[1], p[2] - self.hs[2]
        dh = np.sqrt(x * x + y * y + z * z) - self.hr                  # the head's bounding sphere
        near = np.nonzero(dh < db + KNECK)[0]
        if len(near):
            dh[near] = HEAD.dist(*self.head_local(tuple(a[near] for a in p)))
        return smin(db, dh, KNECK), db, dh

    def normal(self, p, db, dh):
        gb = BODY.grad(*self.body_local(p))
        gh = HEAD.grad(*self.head_local(p)) @ self.hM.T
        w = np.clip(0.5 + 0.5 * (db - dh) / KNECK, 0, 1)[:, None]   # the smooth minimum's blend
        g = gb * (1 - w) + gh * w
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
        return g

    def aabb(self):
        lo, hi = BODY.box
        lo, hi = lo + ORIGIN - 0.006, hi + ORIGIN + 0.006           # the breath moves it by less than this
        hlo, hhi = HEAD.box
        corners = np.array([[a, b, c] for a in (hlo[0], hhi[0]) for b in (hlo[1], hhi[1]) for c in (hlo[2], hhi[2])])
        wc = corners @ self.hM.T + self.hc
        return np.minimum(lo, wc.min(0)) - 0.002 - FRINGE, np.maximum(hi, wc.max(0)) + 0.002 + FRINGE

    def fur(self, p, db, dh):
        """How far out (0-1 of FRINGE) the fur reaches past the coat's surface at points p, in tufts."""
        q, hq = self.body_local(p), self.head_local(p)
        a = [np.where(dh < db, hq[k], q[k]) for k in range(3)]
        n1 = noise(a[0] * 230, a[1] * 230, a[2] * 230)
        return np.clip(0.5 + 0.8 * n1, 0, 1)


_POSE = {}


def pose(t):
    if t not in _POSE:
        if len(_POSE) > 4:
            _POSE.clear()
        _POSE[t] = Pose(t)
    return _POSE[t]


def bound(t):
    lo, hi = pose(t).aabb()
    return (lo + hi) / 2, float(np.linalg.norm(hi - lo) / 2)


def _slab(o, d, lo, hi):
    """Where rays enter and leave a box."""
    t0, t1 = -np.inf, np.inf
    with np.errstate(divide='ignore', invalid='ignore'):
        for k in range(3):
            inv = 1 / d[k]
            a, b = (lo[k] - o[k]) * inv, (hi[k] - o[k]) * inv
            t0, t1 = np.maximum(t0, np.minimum(a, b)), np.minimum(t1, np.maximum(a, b))
    return np.nan_to_num(t0, nan=np.inf, neginf=-1e9), np.nan_to_num(t1, nan=-np.inf)


def _at(a, i):
    """a[i], or a itself if it is one number (a ray origin all the rays share)."""
    return a[i] if np.ndim(a) else a


def march(P, o, d, s, t1, eps, cone=0.0, rc=0.0, steps=96, near=None):
    """Sphere-trace rays o + s d (unit d) on from s until the field is under eps (plus the radius rc + cone s, if
    this is a cone standing for a tile of rays) or s passes t1. Returns s, and whether each ray got there; given
    near (two arrays), keeps in them how close each ray came to the cat, and where."""
    s = s.copy()
    act = np.arange(len(s))
    done = np.zeros(len(s), bool)
    for _ in range(steps):
        if not len(act):
            break
        sa = s[act]
        p = tuple(_at(o[k], act) + d[k][act] * sa for k in range(3))
        f = P.field(p)[0]
        if near is not None:
            closer = f < near[0][act]
            near[0][act[closer]], near[1][act[closer]] = f[closer], sa[closer]
        r = rc + cone * sa
        hitnow = f < r + eps[act]
        done[act[hitnow]] = True
        s[act] = sa + np.where(hitnow, 0, np.maximum((f - r) * f32(0.95 / (1 + cone)), eps[act] * 0.5))
        act = act[~hitnow & (s[act] < t1[act])]
    return s, done


K = 4                                                       # rays a tile, each way, for the cone pass
FRINGE = 0.0025                                             # how far the fur's ends stand off its surface (m)


def _starts(P, O, U, t0, t1, eps, shape, ortho=0.0):
    """For rays on a 2D grid: march one cone a K x K tile, through its middle, to where it first touches the cat.
    Returns where each ray may start (inf: its tile's cone missed the cat, so does it). The rays spread from a
    point, or, given their spacing (ortho), run parallel."""
    H, W = shape
    ti, tj = np.arange(0, H, K), np.arange(0, W, K)
    ci, cj = np.minimum(ti + K // 2, H - 1), np.minimum(tj + K // 2, W - 1)
    cid = (ci[:, None] * W + cj[None, :]).ravel()
    j0 = cid[len(cid) // 2]
    jn = j0 + 1 if j0 % W < W - 1 else j0 - 1
    dang = np.sqrt(sum((U[k][j0] - U[k][jn]) ** 2 for k in range(3)))      # the angle between neighbouring rays
    cone = f32((K * 0.75 + 0.6) * np.sqrt(2) * (0 if ortho else dang))
    rc = f32((K * 0.75 + 0.6) * np.sqrt(2) * ortho)
    a0, a1 = t0[cid] - 0.01, t1[cid] + 0.01
    ok = a1 > np.maximum(a0, 0)
    start = np.full(len(cid), np.inf, f32)
    sel = np.nonzero(ok)[0]
    if len(sel):
        c = cid[sel]
        s, hit_ = march(P, [_at(O[k], c) for k in range(3)], [U[k][c] for k in range(3)], np.maximum(a0[sel], 0),
                        a1[sel], eps[c] + FRINGE, cone, rc)            # the fur's ends count as touching
        start[sel] = np.where(hit_, s, np.inf)
    tile = start.reshape(len(ti), len(tj))
    return np.repeat(np.repeat(tile, K, 0), K, 1)[:H, :W].ravel()


def _trace(P, O, U, start, t1, eps, sel):
    """March the rays sel: how far each meets the cat along its unit direction (inf: it misses), and the normals."""
    S = np.full(len(sel), np.inf, f32)
    N = np.zeros((len(sel), 3), f32)
    live = np.nonzero(t1[sel] > start[sel])[0]
    if len(live):
        r = sel[live]
        o_, u_ = [_at(a, r) for a in O], [a[r] for a in U]
        near = [np.full(len(r), np.inf, f32), np.zeros(len(r), f32)]
        s, ok = march(P, o_, u_, start[r], t1[r], eps[r], near=near)
        idx = np.nonzero(ok)[0]
        if len(idx):
            p = tuple(_at(o_[k], idx) + u_[k][idx] * s[idx] for k in range(3))
            f, db, dh = P.field(p)
            s = s[idx] + f                                          # one more step, onto the surface
            p = tuple(_at(o_[k], idx) + u_[k][idx] * s for k in range(3))
            S[live[idx]] = s
            N[live[idx]] = P.normal(p, db - f, dh - f)
        fr = np.nonzero(~ok & (near[0] < 1.5 * FRINGE))[0]          # rays that only just miss: the fur's edge
        if len(fr):
            s = near[1][fr]
            ofr, ufr = [_at(o_[k], fr) for k in range(3)], [u_[k][fr] for k in range(3)]
            h = f32(0.003)
            for _ in range(3):                  # the closest approach, found exactly, so the edge holds still
                fm, f0, fp = (P.field(tuple(ofr[k] + ufr[k] * (s + e) for k in range(3)))[0] for e in (-h, 0, h))
                curv = fm + fp - 2 * f0
                s = s + np.clip(np.where(curv > 1e-7, 0.5 * h * (fm - fp) / np.maximum(curv, 1e-7), 0), -h, h)
                h = h * f32(0.4)
            p = tuple(ofr[k] + ufr[k] * s for k in range(3))
            f, db, dh = P.field(p)
            tuft = f < FRINGE * P.fur(p, db, dh)
            fr, p, db, dh = fr[tuft], tuple(a[tuft] for a in p), db[tuft], dh[tuft]
            S[live[fr]] = s[tuft]
            N[live[fr]] = P.normal(p, db, dh)
    return S, N


def _fill(P, O, U, start, t1, eps, shape):
    """Rays on a 2D grid: trace every other ray each way, then fill in the rays between four traced ones that meet
    one smooth, face-on piece of the cat; trace the rest (the edges, the creases, the grazing ones)."""
    H, W = shape
    S = np.full(H * W, np.inf, f32)
    N = np.zeros((H * W, 3), f32)
    cand = np.nonzero(t1 > start)[0]
    ci, cj = cand // W, cand % W
    even = (ci % 2 == 0) & (cj % 2 == 0)
    ev = cand[even]
    S[ev], N[ev] = _trace(P, O, U, start, t1, eps, ev)
    od, i, j = cand[~even], ci[~even], cj[~even]
    i0, j0 = i - i % 2, j - j % 2
    i1, j1 = i0 + 2 * (i % 2), j0 + 2 * (j % 2)
    inside = (i1 < H) & (j1 < W)
    i1, j1 = np.minimum(i1, H - 1), np.minimum(j1, W - 1)
    nb = [i0 * W + j0, i0 * W + j1, i1 * W + j0, i1 * W + j1]
    s4 = [S[e] for e in nb]
    n4 = [N[e] for e in nb]
    ok = inside.copy()
    for e, sv, nv in zip(nb, s4, n4):                                # each must meet the cat, and squarely
        ok &= np.isfinite(sv) & (-(nv[:, 0] * U[0][e] + nv[:, 1] * U[1][e] + nv[:, 2] * U[2][e]) > 0.3)
    with np.errstate(invalid='ignore'):
        ok &= (np.maximum(np.maximum(s4[0], s4[1]), np.maximum(s4[2], s4[3]))
               - np.minimum(np.minimum(s4[0], s4[1]), np.minimum(s4[2], s4[3]))) < 0.003
    for a in range(4):
        for b in range(a + 1, 4):
            ok &= (n4[a] * n4[b]).sum(-1) > 0.85
    k = np.nonzero(ok)[0]
    if len(k):
        wi, wj = (i[k] % 2 * 0.5).astype(f32), (j[k] % 2 * 0.5).astype(f32)
        w4 = [(1 - wi) * (1 - wj), (1 - wi) * wj, wi * (1 - wj), wi * wj]
        g = od[k]
        S[g] = sum(w * sv[k] for w, sv in zip(w4, s4))
        nn = sum(w[:, None] * nv[k] for w, nv in zip(w4, n4))
        N[g] = nn / np.linalg.norm(nn, axis=1, keepdims=True)
    rest = od[~ok]
    S[rest], N[rest] = _trace(P, O, U, start, t1, eps, rest)
    return S, N


def hit(o, d, t):
    P = pose(t)
    shape = np.shape(d[0])
    D = [np.ravel(np.broadcast_to(a, shape)).astype(f32) for a in d]
    O = [np.ravel(np.broadcast_to(np.asarray(a, f32), shape)) if np.ndim(a) else np.asarray(a, f32) for a in o]
    L = np.sqrt(D[0] ** 2 + D[1] ** 2 + D[2] ** 2)
    U = [a / L for a in D]
    lo, hi = P.aabb()
    t0, t1 = _slab(O, U, lo, hi)
    eps = np.maximum(f32(3e-4), np.maximum(t0, 0) * f32(3e-4))      # about half a sample, at 1080 and ss 2
    start = np.maximum(t0, 0)
    if len(shape) == 2 and min(shape) >= 2 * K:
        start = np.maximum(start, _starts(P, O, U, t0, t1, eps, shape))
        S, N = _fill(P, O, U, start, t1, eps, shape)
    else:
        S, N = _trace(P, O, U, start, t1, eps, np.arange(len(L)))
    T = (S / L).reshape(shape)
    return T, N[:, 0].reshape(shape), N[:, 1].reshape(shape), N[:, 2].reshape(shape)


SUNMAP = 0.002                                              # the spacing of the sun's map of the cat (m)


def _sunmap(P, s):
    """The cat seen from the sun, on a grid of parallel rays across its light: along each, where the last piece of
    the cat it crosses on its way toward the sun begins (as a distance along s; -1 where it crosses none)."""
    key = tuple(np.round(s, 7))
    if key in P.sunmaps:
        return P.sunmaps[key]
    s = np.asarray(s, float) / np.linalg.norm(s)
    e1 = np.cross(s, (0.0, 1.0, 0.0) if abs(s[1]) < 0.9 else (1.0, 0.0, 0.0))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)
    lo, hi = P.aabb()
    box = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    uv = box @ np.stack([e1, e2, s], 1)
    u0, v0, wtop = uv[:, 0].min() - SUNMAP, uv[:, 1].min() - SUNMAP, uv[:, 2].max() + 0.01
    nu, nv = int(np.ceil((uv[:, 0].max() - u0) / SUNMAP)) + 2, int(np.ceil((uv[:, 1].max() - v0) / SUNMAP)) + 2
    v, u = np.mgrid[0:nv, 0:nu].astype(f32)
    u, v = (u0 + u * SUNMAP).ravel(), (v0 + v * SUNMAP).ravel()
    O = [(u * e1[k] + v * e2[k] + wtop * s[k]).astype(f32) for k in range(3)]
    U = [np.full(len(u), f32(-s[k])) for k in range(3)]
    t0, t1 = _slab(O, U, lo, hi)
    eps = np.full(len(u), f32(3e-4))
    start = np.maximum(np.maximum(t0, 0), _starts(P, O, U, t0, t1, eps, (nv, nu), ortho=SUNMAP))
    A = np.full(len(u), -1.0, f32)
    sel = np.nonzero(t1 > start)[0]
    if len(sel):
        o_, d_ = [a[sel] for a in O], [a[sel] for a in U]
        sx, ok = march(P, o_, d_, start[sel], t1[sel], eps[sel])     # down to the side the sun lights
        sel, sx = sel[ok], sx[ok] + 0.0005
        act = np.arange(len(sel))
        for _ in range(64):                                          # and on through the cat, out the far side
            if not len(act):
                break
            r = sel[act]
            f = P.field(tuple(O[k][r] + U[k][r] * sx[act] for k in range(3)))[0]
            out = f > -2e-4
            sx[act] += np.maximum(-f, 4e-4)
            act = act[~out]
        A[sel] = wtop - sx
    M = (e1.astype(f32), e2.astype(f32), s.astype(f32), f32(u0), f32(v0), nu, nv, A.reshape(nv, nu))
    P.sunmaps[key] = M
    return M


def occludes(P_, s, t):
    """Whether the cat stands between each point and the sun, read off the cat as the sun sees it (_sunmap). A ray
    that starts on the cat and goes into it is let through until it comes out (the fur glows along the edge
    turned from the sun, as the film's rim light wants); it is the cat farther along it that shades it."""
    n = len(P_)
    if not n:
        return np.zeros(0, bool)
    e1, e2, sv, u0, v0, nu, nv, A = _sunmap(pose(t), s)
    Q = np.asarray(P_, f32)
    u = (Q @ e1 - u0) / f32(SUNMAP)
    v = (Q @ e2 - v0) / f32(SUNMAP)
    w = Q @ sv
    inmap = (u >= 0) & (u < nu - 1) & (v >= 0) & (v < nv - 1)
    i, j = np.clip(v.astype(np.intp), 0, nv - 2), np.clip(u.astype(np.intp), 0, nu - 2)
    fv, fu = v - i, u - j
    a4 = A[i, j], A[i, j + 1], A[i + 1, j], A[i + 1, j + 1]
    a = (a4[0] * (1 - fu) + a4[1] * fu) * (1 - fv) + (a4[2] * (1 - fu) + a4[3] * fu) * fv
    cushion = np.abs(Q[:, 1] - CUSHION_TOP) < 5e-4
    lenient = np.minimum(np.minimum(a4[0], a4[1]), np.minimum(a4[2], a4[3])) - 0.0025   # a point on the cat itself
    return inmap & (w < np.where(cushion, a - 5e-4, lenient)) & (a > -0.5)


# --- the coat ----------------------------------------------------------------------------------------------------
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


GINGER, DARK, CREAM = lin(.9, .58, .31), lin(.62, .33, .15), lin(.96, .86, .72)
EAR_IN, NOSE, LID = lin(.92, .7, .62), lin(.8, .48, .44), lin(.32, .16, .08)


def stroke(v, w):
    """A soft-edged line: 1 on it (v = 0), fading out over w."""
    return np.clip(1.5 - np.abs(v) / w, 0, 1)


def tail_param(x, y, z):
    """How far along the tail (m) the nearest point of its axis is, and the distance to that axis."""
    seg = TAIL[1:] - TAIL[:-1]
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
    best = np.full(len(x), np.inf, f32)
    par = np.zeros(len(x), f32)
    for i in range(len(seg)):
        a, b = TAIL[i], seg[i]
        h = np.clip(((x - a[0]) * b[0] + (y - a[1]) * b[1] + (z - a[2]) * b[2]) / float(b @ b), 0, 1)
        dd = (x - a[0] - b[0] * h) ** 2 + (y - a[1] - b[1] * h) ** 2 + (z - a[2] - b[2] * h) ** 2
        c = dd < best
        best = np.where(c, dd, best)
        par = np.where(c, cum[i] + h * np.linalg.norm(b), par)
    return par, np.sqrt(best)


def body_coat(q):
    """The body's markings (0-1 dark), its cream (0-1), a fur grain and a slow mottle (about -1..1), in its rest
    frame."""
    x, y, z = q
    ang = np.arctan2((z - SPINE_C[2]) / SPINE_R[1], (x - SPINE_C[0]) / SPINE_R[0])
    ang = np.where(ang > 2.6, ang - 2 * np.pi, ang)               # one turn, unbroken over the curl's open side
    rad = np.sqrt(((x - SPINE_C[0]) / SPINE_R[0]) ** 2 + ((z - SPINE_C[2]) / SPINE_R[1]) ** 2)
    along = ang * 0.1                                               # metres along the spine, about
    wob = noise(x * 45, y * 45, z * 45)
    wob2 = noise(x * 120 + 7, y * 120, z * 120)
    # mackerel stripes: across the spine, down the flank, narrowing and breaking toward the belly
    grain = noise(along * 70, rad * 34 + y * 340, y * 60 + rad * 9)       # fur runs along the body
    ph = along / 0.024 + 0.35 * wob + 4 * y + 0.6 * rad
    sv = np.abs(((ph + 0.5) % 1) - 0.5) + 0.05 * grain                    # the stripes' edges, feathered by it
    width = 0.2 * (0.6 + 0.4 * np.clip(y / 0.1, 0, 1)) * (1 + 0.35 * wob2)
    stripes = np.clip((width - sv) / 0.07 + 0.5, 0, 1)
    flank = np.clip((rad - 0.35) / 0.4, 0, 1) * np.clip((y - 0.012) / 0.03, 0, 1)
    dorsal = np.clip(1 - np.abs(rad - 1.25) / 0.22, 0, 1) * np.clip((y - 0.05) / 0.03, 0, 1)
    mark = np.clip(stripes * flank * 0.85 + dorsal * 0.5, 0, 1)
    # the haunch: rings round the thigh
    hM = frame((0.55, 0.1, 0.83), (0, 1, 0))
    hx, hy, hz = local(q, (-0.078, 0.078, 0.045), hM)
    hr = np.sqrt((hx / 0.05) ** 2 + (hy / 0.058) ** 2 + (hz / 0.074) ** 2)
    thigh = np.clip((1.25 - hr) / 0.15, 0, 1) * np.clip((hy + 0.02) / 0.03, 0, 1)
    rings = np.clip((0.22 - np.abs((((hz / 0.074 - 0.3 * hy / 0.058) * 2.6 + 0.3 * wob) % 1) - 0.5)) / 0.07 + 0.5, 0, 1)
    mark = mark * (1 - thigh) + rings * thigh * 0.8
    # the tail: rings, darker toward its tip
    tp, td = tail_param(x, y, z)
    ontail = np.clip((0.03 - td) / 0.006, 0, 1) * np.clip((tp - 0.06) / 0.03, 0, 1)
    tr = np.clip((0.2 - np.abs(((tp / 0.03 + 0.2 * wob) % 1) - 0.5)) / 0.08 + 0.5, 0, 1) * (0.7 + 0.3 * tp / 0.4)
    mark = mark * (1 - ontail) + np.clip(tr + np.clip((tp - 0.36) / 0.03, 0, 1), 0, 1) * ontail
    # cream: the belly, the inner curl, the paws
    belly = np.clip((0.42 - rad) / 0.2, 0, 1) * np.clip((0.05 - y) / 0.02, 0, 1) + \
        np.clip((0.022 - y) / 0.012, 0, 1) * np.clip(1 - rad, 0, 1)
    paws = np.clip((0.03 - y) / 0.012, 0, 1) * np.clip((z - 0.11) / 0.012, 0, 1) * np.clip((x + 0.02) / 0.02, 0, 1)
    paws = paws * (1 - ontail)
    toes = 0
    for (c, r), toward in zip(PAWS, ((-0.35, 0, 1), (-0.45, 0, 1))):   # the paws, cream, the toes parted
        e = ((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2
        on = np.clip((1.45 - e) / 0.35, 0, 1)
        paws = np.maximum(paws, on)
        tw = np.array(toward) / np.linalg.norm(toward)
        side = (x - c[0]) * tw[2] - (z - c[2]) * tw[0]
        front = (x - c[0]) * tw[0] + (z - c[2]) * tw[2]
        part = np.maximum(stroke(np.abs(side) - 0.0085, 0.0011), stroke(side, 0.0011))
        toes = np.maximum(toes, part * on * np.clip(front / r[2] * 2.2 - 0.6, 0, 1))
    cream = np.clip(belly * (1 - ontail) + paws * 0.75, 0, 1)
    mark = np.maximum(mark * (1 - cream), toes * 0.6)
    return mark, cream, grain, wob


def head_coat(hq):
    """The head's markings, cream and grain, in its frame (unscaled)."""
    x, y, z = (a / HEAD_SCALE for a in hq)
    wob = noise(x * 150, y * 150, z * 150)
    grain = noise(x * 380, y * 380, z * 120)
    x = x + 0.0006 * grain                                          # the lines' edges, feathered by it
    ax = np.abs(x)
    # the forehead's lines, an M above the eyes, running back over the crown to the neck
    m = 0
    spread = 1 + 14 * np.clip(0.02 - y, 0, 0.02)                    # the outer lines bow out toward the eyes
    for x0, top in ((0.0, 0.02), (0.008, 0.012), (0.017, 0.006)):
        m = np.maximum(m, stroke(ax - x0 * spread, 0.0024) * np.clip((y - top + 0.004) / 0.004, 0, 1))
    m = m * np.clip((z + 0.06) / 0.02, 0, 1) * np.clip(1 - (ax - 0.022) / 0.006, 0, 1)
    # two lines back from each eye over the cheek
    cheek = np.maximum(stroke(y + 0.002 - 0.25 * (z - 0.026), 0.0022), stroke(y + 0.013 - 0.2 * (z - 0.02), 0.002))
    cheek = cheek * np.clip((ax - 0.028) / 0.004, 0, 1) * np.clip((z + 0.012) / 0.01, 0, 1) * (z < 0.034)
    mark = np.clip(np.maximum(m, cheek * 0.85) * (0.85 + 0.3 * wob), 0, 1)
    # cream: the muzzle, the chin and throat, above the eyes a little
    muz = 1 - (x ** 2 / 0.021 ** 2 + (y + 0.02) ** 2 / 0.016 ** 2 + (z - 0.04) ** 2 / 0.03 ** 2)
    chin = 1 - (x ** 2 / 0.03 ** 2 + (y + 0.034) ** 2 / 0.014 ** 2 + (z - 0.015) ** 2 / 0.04 ** 2)
    cream = np.clip(np.maximum(muz, chin) * 2.5, 0, 1)
    brow = stroke(y - 0.009 - 3 * (ax - 0.019) ** 2, 0.003) * np.clip((z - 0.026) / 0.006, 0, 1) * \
        np.clip(1 - np.abs(ax - 0.019) / 0.012, 0, 1)
    cream = np.clip(cream + brow * 0.5, 0, 1)
    mark = mark * (1 - cream)
    return mark, cream, grain, wob


def albedo(p, n, t):
    P = pose(t)
    p = tuple(np.asarray(a, f32) for a in p)
    f, db, dh = P.field(p)
    wh = np.clip(0.5 + (db - dh) / 0.008, 0, 1)                      # 1 on the head, 0 on the body
    hq = P.head_local(p)
    bm, bc, bg, bw = body_coat(P.body_local(p))
    hm, hc, hg, hw = head_coat(hq)
    mark = bm * (1 - wh) + hm * wh
    cream = bc * (1 - wh) + hc * wh
    grain = bg * (1 - wh) + hg * wh
    wob = bw * (1 - wh) + hw * wh
    A = GINGER * (1 + 0.05 * wob)[:, None]
    A = A * (1 - mark[:, None]) + DARK * mark[:, None]
    A = A * (1 - cream[:, None]) + CREAM * cream[:, None]
    # the face: closed eyes, a pink nose; inside the ears
    x, y, z = (a / HEAD_SCALE for a in hq)
    ax = np.abs(x)
    eye = stroke(y - 0.0035 + 50 * (ax - 0.019) ** 2, 0.0011) * np.clip(1 - np.abs(ax - 0.019) / 0.0095, 0, 1)
    eye = eye * (z > 0.024) * wh
    A = A * (1 - 0.9 * eye[:, None]) + LID * (0.9 * eye)[:, None]
    nose = np.clip(1.3 - (x ** 2 / 0.0055 ** 2 + (y + 0.005) ** 2 / 0.0035 ** 2), 0, 1) * (z > 0.042) * wh
    A = A * (1 - nose[:, None]) + NOSE * nose[:, None]
    for _, c, M in EARS:
        ex, ey, ez = local((x, y, z), c, M)
        zc = ez - 8 * ex * ex + 0.004 * ey / 0.05                     # in front of the ear's middle: its hollow
        inner = np.clip((zc + 0.0012) / 0.0012, 0, 1) * np.clip(ey / 0.006, 0, 1)
        inner = inner * np.clip((0.027 * (1 - ey / 0.05) - np.abs(ex) - 0.0015) / 0.002, 0, 1) * wh
        A = A * (1 - inner[:, None]) + EAR_IN * inner[:, None]
    # fur: a grain along the coat, and its own shade in the folds
    A = A * (1 + 0.12 * grain)[:, None]
    ao = np.ones(len(f), f32)
    nn = [np.asarray(a, f32) for a in n]
    for hgt, w in ((0.005, 0.45), (0.014, 0.35), (0.032, 0.25)):
        pp = tuple(p[k] + nn[k] * hgt for k in range(3))
        dd = np.minimum(P.field(pp)[0], pp[1] - CUSHION_TOP)
        ao -= w * np.clip((hgt - dd) / hgt, 0, 1)
    A = A * (0.3 + 0.7 * np.clip(ao, 0, 1))[:, None]
    return A.astype(np.float32)


def translucency(p, n, t):
    """How much sunlight comes through the cat to the side facing us, 0-1: the ears are thin; the rest isn't."""
    P = pose(t)
    p = tuple(np.asarray(a, f32) for a in p)
    f, db, dh = P.field(p)
    wh = np.clip(0.5 + (db - dh) / 0.008, 0, 1)
    x, y, z = (a / HEAD_SCALE for a in P.head_local(p))
    tr = np.zeros(len(x), f32)
    for _, c, M in EARS:
        ex, ey, ez = local((x, y, z), c, M)
        ear_ = (np.clip(ey / 0.006, 0, 1) * np.clip((0.03 * (1 - ey / 0.05) - np.abs(ex)) / 0.003, 0, 1)
                * np.clip(1 - np.abs(ez) / 0.009, 0, 1))
        tr = np.maximum(tr, 0.75 * ear_ * wh)
    return tr

