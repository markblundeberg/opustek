"""The room in First Light's film: a box with one window and one sun, ray cast.

Every patch of light in it is where the sun's rays land through the window. A surface point is lit when the ray
from it toward the sun passes through both faces of the window's opening (the wall is thick) and misses the
mullion; its brightness is the cosine of that ray with the surface. The shimmer is the same sun mirrored in the
water (its elevation negated), coming up through the window. The view outside is a picture at infinity, looked
up by each ray's direction, so it parallaxes as a window view does when the camera moves.

World units are metres: x to the right, y up, z out of the window wall into the room (the camera looks along -z).
"""
import numpy as np

WIN = (-1.2, 1.2, 0.55, 2.05)          # the opening in the wall's room face (z = 0): x0, x1, y0, y1
DEPTH = 0.18                           # the wall's thickness: the glass sits at z = -DEPTH
FRAME = 0.045                          # the window frame's border, at the glass
MULL = (-0.545, -0.5)                  # the mullion, at the glass (in a gap of the skyline, from the window)
TRANSOM = (1.9, 1.94)                  # the transom, at the glass: it crosses every patch of sun
X0, X1, Y1 = -1.35, 2.05, 2.5          # left wall, right wall, ceiling (floor y = 0, window wall z = 0)
EYE_Y = 1.3
F = 706.0                              # focal length, in base px (a frame 540 high)
CX, CY = 480.0, 264.6                  # principal point, base px
Z_CLOSE, Z_BACK, Z_MID = 2.4, 4.2, 3.0
# the picture outside is drawn in the old 2D layout's coordinates (the opening at x 40-920, y 24-574); the close
# camera sees the opening's room face at x 95-865, y 24-505 of the frame: one old unit is ENV_K frame units
ENV_K = F * WIN[1] / Z_CLOSE / 440
U0 = CX - F * WIN[1] / Z_CLOSE
V0 = CY - F * (WIN[3] - EYE_Y) / Z_CLOSE

from ids import ENV, WALL, CEIL, FLOOR, WOOD, CUSHION, CAT, CUP, POT, LEAF, SHADE, DARK, BOOKS, PAINT, lin  # noqa: E402,F401
SUN_ANG = 0.0093                       # the sun's angular diameter (rad): its penumbra


from prims import hit_box, hit_ellipsoid, hit_cyl, hit_tri, rot_y, rot_z  # noqa: E402,F401
import cat as CATMOD  # noqa: E402
import props as PROPS  # noqa: E402
import materials as MATS  # noqa: E402


# --- the furniture --------------------------------------------------------------------------------------------
SEAT = ((-1.28, 0.0, 0.0), (1.28, WIN[2] - 0.05, 0.62))   # a window seat; its cushion's top is level with the sill
CUSH = ((-1.22, WIN[2] - 0.05, 0.02), (1.22, WIN[2], 0.58))
TABLE = ((1.29, 0.0, 0.3), (1.75, 0.62, 0.75))
SHELF = ((1.72, 0.0, 1.0), (2.05, 1.85, 2.25))
LAMP_LIGHT = PROPS.LAMP_LIGHT
CAT_C = CATMOD.CENTER
CUP_C = PROPS.MUG_C


def static_objects():
    """The room's furniture (the props and the cat have their own modules): (material, kind, params)."""
    return [(PAINT, 'box', SEAT), (CUSHION, 'box', CUSH), (WOOD, 'box', TABLE), (BOOKS, 'box', SHELF)]


def hit_object(kind, p, o, d):
    if kind == 'box':
        return hit_box(p, o, d)
    if kind == 'ell':
        return hit_ellipsoid(p[0], p[1], o, d, p[2])
    if kind == 'cyl':
        return hit_cyl(p[0], p[1], p[2], p[3], o, d)
    return hit_tri(p[0], p[1], p[2], o, d)


def bound(kind, p):
    if kind == 'box':
        a, b = np.array(p[0]), np.array(p[1])
        return (a + b) / 2, np.linalg.norm(b - a) / 2
    if kind == 'ell':
        return np.array(p[0]), max(p[1])
    if kind == 'cyl':
        return np.array([p[0][0], (p[2] + p[3]) / 2, p[0][1]]), np.hypot(p[1], (p[3] - p[2]) / 2)
    pts = np.array(p)
    c = pts.mean(0)
    return c, np.linalg.norm(pts - c, axis=1).max()


# --- casting ---------------------------------------------------------------------------------------------------
class Hits:
    """What each sample of the frame sees: material, point, normal; and the ray, for the view outside."""


def cast(zc, t, h, w, scale, dynamic=True, org=(0, 0)):
    """Cast one ray per sample (h x w samples, scale samples per base px) from the camera at z = zc; org is the
    window's first sample (row, column) in the whole frame's samples at this scale."""
    v, u = np.mgrid[org[0]:org[0] + h, org[1]:org[1] + w].astype(np.float32)
    u, v = (u + 0.5) / scale, (v + 0.5) / scale
    dx, dy, dz = (u - CX) / F, -(v - CY) / F, -np.ones_like(u)
    o = (0.0, EYE_Y, zc)
    d = (dx, dy, dz)
    big = np.float32(1e9)
    with np.errstate(divide='ignore'):
        cand = [(np.full_like(u, zc), WALL, (0, 0, 1)),                                # the window wall
                (np.where(dy < 0, -EYE_Y / np.where(dy < 0, dy, -1), big), FLOOR, (0, 1, 0)),
                (np.where(dy > 0, (Y1 - EYE_Y) / np.where(dy > 0, dy, 1), big), CEIL, (0, -1, 0)),
                (np.where(dx < 0, (X0 - 0) / np.where(dx < 0, dx, -1), big), WALL, (1, 0, 0)),
                (np.where(dx > 0, (X1 - 0) / np.where(dx > 0, dx, 1), big), WALL, (-1, 0, 0))]
    T = np.full_like(u, np.inf)
    M = np.zeros(u.shape, np.int8)
    N = [np.zeros_like(u) for _ in range(3)]
    for tt, m, n in cand:
        closer = tt < T
        T = np.where(closer, tt, T)
        M = np.where(closer, m, M)
        for k in range(3):
            N[k] = np.where(closer, n[k], N[k])
    # through the opening: the reveal, the frame and mullion at the glass, or out into the view
    px, py = dx * T, EYE_Y + dy * T
    hole = (M == WALL) & (N[2] == 1) & (px > WIN[0]) & (px < WIN[1]) & (py > WIN[2]) & (py < WIN[3])
    tg = zc + DEPTH
    gx, gy = dx * tg, EYE_Y + dy * tg
    inner = (gx > WIN[0] + FRAME) & (gx < WIN[1] - FRAME) & (gy > WIN[2] + FRAME) & (gy < WIN[3] - FRAME)
    mull = ((gx > MULL[0]) & (gx < MULL[1])) | ((gy > TRANSOM[0]) & (gy < TRANSOM[1]))
    in_win = (gx > WIN[0]) & (gx < WIN[1]) & (gy > WIN[2]) & (gy < WIN[3])
    out = hole & inner & ~mull
    M = np.where(out, ENV, M)
    T = np.where(out, np.inf, T)
    glassframe = hole & in_win & ~(inner & ~mull)
    M = np.where(glassframe, PAINT, M)                     # the window is painted, like the seat
    T = np.where(glassframe, tg, T)
    N[0], N[1], N[2] = (np.where(glassframe, 0, N[0]), np.where(glassframe, 0, N[1]), np.where(glassframe, 1, N[2]))
    side = hole & ~in_win                                     # it leaves the opening through a reveal face
    if side.any():
        with np.errstate(divide='ignore', invalid='ignore'):
            ts = [(np.where(dx < 0, WIN[0] / dx, big), (1, 0, 0)), (np.where(dx > 0, WIN[1] / dx, big), (-1, 0, 0)),
                  (np.where(dy < 0, (WIN[2] - EYE_Y) / dy, big), (0, 1, 0)),
                  (np.where(dy > 0, (WIN[3] - EYE_Y) / dy, big), (0, -1, 0))]
        best = np.full_like(u, np.inf)
        bn = [np.zeros_like(u) for _ in range(3)]
        for tt, n in ts:
            ok = side & (tt > zc - 1e-6) & (tt < best)
            best = np.where(ok, tt, best)
            for k in range(3):
                bn[k] = np.where(ok, n[k], bn[k])
        M = np.where(side, PAINT, M)
        T = np.where(side, best, T)
        for k in range(3):
            N[k] = np.where(side, bn[k], N[k])
    H = Hits()
    H.m, H.t, H.n, H.d, H.zc, H.scale, H.org = M, T, N, d, zc, scale, org
    place(H, static_objects())
    place_props(H)
    if dynamic:
        place_cat(H, t)
    Tf = np.where(np.isfinite(H.t), H.t, 0)
    H.p = (dx * Tf, EYE_Y + dy * Tf, zc + dz * Tf)
    H.shadow = lambda s, vis: np.zeros_like(H.t)
    return H


def screen_box(obs, zc, h, w, scale, org=(0, 0)):
    """The samples a set of objects can cover: a slice pair, or None if off the frame."""
    x0 = y0 = np.inf
    x1 = y1 = -np.inf
    for m, kind, p in obs:
        c, r = bound(kind, p)
        dist = zc - c[2]
        if dist <= r:
            return slice(0, h), slice(0, w)
        cu, cv = CX + F * c[0] / dist, CY - F * (c[1] - EYE_Y) / dist
        rr = F * r / (dist - r) * 1.1 + 2
        x0, x1, y0, y1 = min(x0, cu - rr), max(x1, cu + rr), min(y0, cv - rr), max(y1, cv + rr)
    x0, x1 = int(max(0, x0 * scale - org[1])), int(min(w, x1 * scale + 1 - org[1]))
    y0, y1 = int(max(0, y0 * scale - org[0])), int(min(h, y1 * scale + 1 - org[0]))
    if x0 >= x1 or y0 >= y1:
        return None
    return slice(y0, y1), slice(x0, x1)


def place(H, obs):
    """Add objects to the hits, each tested only where its bounding sphere falls on the frame."""
    h, w = H.t.shape
    o = (0.0, EYE_Y, H.zc)
    for m, kind, p in obs:
        box = screen_box([(m, kind, p)], H.zc, h, w, H.scale)
        if box is None:
            continue
        ys, xs = box
        tt, nx, ny, nz = hit_object(kind, p, o, (H.d[0][ys, xs], H.d[1][ys, xs], H.d[2][ys, xs]))
        closer = tt < H.t[ys, xs]
        H.t[ys, xs] = np.where(closer, tt, H.t[ys, xs])
        H.m[ys, xs] = np.where(closer, m, H.m[ys, xs])
        for k, nn in enumerate((nx, ny, nz)):
            H.n[k][ys, xs] = np.where(closer, nn, H.n[k][ys, xs])


def cat_box(t, zc, h, w, scale, org=(0, 0)):
    return sphere_box(*CATMOD.bound(t), zc, h, w, scale, org)


def sphere_box(c, r, zc, h, w, scale, org=(0, 0)):
    """The samples a bounding sphere can cover: a slice pair (in a window starting at sample org), or None."""
    dist = zc - c[2]
    if dist <= r:
        return slice(0, h), slice(0, w)
    cu, cv = CX + F * c[0] / dist, CY - F * (c[1] - EYE_Y) / dist
    rr = F * r / (dist - r) * 1.1 + 2
    x0, x1 = int(max(0, (cu - rr) * scale - org[1])), int(min(w, (cu + rr) * scale + 1 - org[1]))
    y0, y1 = int(max(0, (cv - rr) * scale - org[0])), int(min(h, (cv + rr) * scale + 1 - org[0]))
    return None if x0 >= x1 or y0 >= y1 else (slice(y0, y1), slice(x0, x1))


def place_props(H):
    """Cast the props in, each where its bounding sphere falls on the frame."""
    h, w = H.t.shape
    for pr in PROPS.PROPS:
        box = sphere_box(*pr.bound(), H.zc, h, w, H.scale, H.org)
        if box is None:
            continue
        tt, nx, ny, nz, mm = pr.hit((0.0, EYE_Y, H.zc), tuple(a[box] for a in H.d))
        closer = tt < H.t[box]
        H.t[box] = np.where(closer, tt, H.t[box])
        H.m[box] = np.where(closer, mm, H.m[box])
        for k, nn in enumerate((nx, ny, nz)):
            H.n[k][box] = np.where(closer, nn, H.n[k][box])


def place_cat(H, t):
    """Cast the cat in, where its bounding sphere falls on the frame."""
    h, w = H.t.shape
    box = cat_box(t, H.zc, h, w, H.scale, H.org)
    if box is None:
        return None
    tt, nx, ny, nz = CATMOD.hit((0.0, EYE_Y, H.zc), tuple(a[box] for a in H.d), t)
    closer = tt < H.t[box]
    H.t[box] = np.where(closer, tt, H.t[box])
    H.m[box] = np.where(closer, CAT, H.m[box])
    for k, nn in enumerate((nx, ny, nz)):
        H.n[k][box] = np.where(closer, nn, H.n[k][box])
    return box


FIELDS = ('m', 't', 'A', 'sky', 'lamp')


def with_cat(G, t, scale):
    """The cached room with the cat cast in at time t (it breathes), patched in place where it can fall,
    and the shadows the things on the seat cast toward the sun."""
    h, w = G.t.shape
    if getattr(G, 'saved', None):                           # put back what the last frame's cat covered
        box, old = G.saved
        for k, v in old.items():
            if k.startswith('n'):
                G.n[int(k[1])][box] = v
            elif k.startswith('p'):
                pass
            else:
                getattr(G, k)[box] = v
        G.p = G.p_static
    box = cat_box(t, G.zc, h, w, scale)
    if not hasattr(G, 'p_static'):
        G.p_static = G.p
    if box is None:
        G.saved = None
    else:
        ys, xs = box
        old = {k: getattr(G, k)[box].copy() for k in FIELDS}
        old.update({f'n{k}': G.n[k][box].copy() for k in range(3)})
        G.saved = (box, old)
        place_cat(G, t)
        Tf = np.where(np.isfinite(G.t[box]), G.t[box], 0)
        p = tuple(a.copy() for a in G.p_static)
        p[0][box] = G.d[0][box] * Tf
        p[1][box] = EYE_Y + G.d[1][box] * Tf
        p[2][box] = G.zc + G.d[2][box] * Tf
        G.p = p
        sub = Hits()
        sub.m, sub.n, sub.p = G.m[box], [n[box] for n in G.n], tuple(a[box] for a in p)
        G.A[box] = albedo(sub, t)
        G.sky[box] = sky_factor(sub.p, sub.n)
        G.lamp[box] = lamp_factor(sub.p, sub.n)
    attach_shadow(G, t)
    return G


def attach_shadow(G, t):
    """Give the hits a shadow(s, vis) test: what on the seat (the cat, the mug, the plant) stands in the sun's way."""
    h, w = G.t.shape
    boxes = [screen_box([(CUSHION, 'box', CUSH)], G.zc, h, w, G.scale, G.org)] + [
        sphere_box(*pr.bound(), G.zc, h, w, G.scale, G.org) for pr in PROPS.ON_SEAT]
    boxes = [b for b in boxes if b is not None]
    seat_box = (slice(min(b[0].start for b in boxes), max(b[0].stop for b in boxes)),
                slice(min(b[1].start for b in boxes), max(b[1].stop for b in boxes))) if boxes else None

    def shadow(s, vis):
        out = np.zeros_like(G.t)
        if seat_box is not None:
            out[seat_box] = shadowed(G.p, s, t, seat_box, vis[seat_box] > 0)
        return out
    G.shadow = shadow


# --- light ------------------------------------------------------------------------------------------------------
def aperture(p, s, n=None):
    """How much of the sun (direction s, toward it) a point sees through the opening: 0-1, with a penumbra."""
    sx, sy, sz = s
    px, py, pz = p
    if n is not None:
        px, py, pz = px + 2e-3 * n[0], py + 2e-3 * n[1], pz + 2e-3 * n[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        lam0 = np.where(pz > 1e-4, -pz / sz, 0.0)
        lamg = (-DEPTH - pz) / sz
    def edge(a, lo, hi, pen):                               # signed coverage across one pair of edges
        return np.clip((np.minimum(a - lo, hi - a)) / pen + 0.5, 0, 1)
    pen0 = 0.004 + SUN_ANG * lam0
    peng = 0.004 + SUN_ANG * lamg
    qx, qy = px + lam0 * sx, py + lam0 * sy
    v0 = np.where(pz > 1e-4, edge(qx, WIN[0], WIN[1], pen0) * edge(qy, WIN[2], WIN[3], pen0), 1.0)
    gx, gy = px + lamg * sx, py + lamg * sy
    vg = edge(gx, WIN[0] + FRAME, WIN[1] - FRAME, peng) * edge(gy, WIN[2] + FRAME, WIN[3] - FRAME, peng)
    vm = (1 - edge(gx, MULL[0], MULL[1], peng)) * (1 - edge(gy, TRANSOM[0], TRANSOM[1], peng))
    return np.nan_to_num(v0 * vg * vm) * (lamg > 0)


def shadowed(p, s, t, region, cand):
    """1 where something on the seat (the cat, the cup, the plant) stands between the point and the sun. Only the
    samples the sun can reach (cand) are traced, and each group of parts is first tested against its bounding
    sphere along the ray."""
    ys, xs = region
    out = np.zeros(cand.shape, np.float32)
    idx = np.nonzero(cand)
    if not len(idx[0]):
        return out
    P = np.stack([a[ys, xs][idx] for a in p], 1)
    sh = np.zeros(len(P), np.float32)
    sv = np.asarray(s, np.float64)
    c, r = CATMOD.bound(t)                       # the cat, through its own test
    w = np.asarray(c) - P
    proj = w @ sv
    near = (proj > 0) & ((w * w).sum(1) - proj ** 2 < r * r)
    if near.any():
        sh[near] = np.maximum(sh[near], CATMOD.occludes(P[near], sv, t))
    for pr in PROPS.ON_SEAT:                     # the mug and the plant, through their own tests
        c, r = pr.bound()
        w = c - P
        proj = w @ sv
        near = (proj > 0) & ((w * w).sum(1) - proj ** 2 < r * r)
        if near.any():
            sh[near] = np.maximum(sh[near], pr.occludes(P[near], sv))
    out[idx] = sh
    return out


def sky_factor(p, n):
    """The window's view factor from each point (3 x 3 samples of the opening): how much sky light falls on it."""
    px, py, pz = p
    acc = np.zeros_like(px)
    xs = np.linspace(WIN[0], WIN[1], 4)
    ys = np.linspace(WIN[2], WIN[3], 4)
    dA = (xs[1] - xs[0]) * (ys[1] - ys[0])
    for i in range(3):
        for j in range(3):
            qx, qy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            rx, ry, rz = qx - px, qy - py, -pz
            r2 = rx * rx + ry * ry + rz * rz + 1e-3
            r = np.sqrt(r2)
            cp = np.clip((rx * n[0] + ry * n[1] + rz * n[2]) / r, 0, 1)
            cw = np.clip(pz / r, 0, 1)
            acc += cp * cw / r2 * dA / np.pi
    # inside the wall's thickness (the reveal), the sky comes through the glass instead: the opening's inner
    # edge at z = -DEPTH, from points behind z = 0. At 0.12: a quarter and more tripped the stripe test at 1:46
    tun = pz < -1e-4
    if tun.any():
        tx, ty, tz = px[tun], py[tun], pz[tun] + DEPTH
        tn = [a_[tun] for a_ in n]
        xs = np.linspace(WIN[0] + FRAME, WIN[1] - FRAME, 4)
        ys = np.linspace(WIN[2] + FRAME, WIN[3] - FRAME, 4)
        dA = (xs[1] - xs[0]) * (ys[1] - ys[0])
        t_acc = np.zeros_like(tx)
        for i in range(3):
            for j in range(3):
                qx, qy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
                rx, ry, rz = qx - tx, qy - ty, -tz
                r2 = rx * rx + ry * ry + rz * rz + 1e-3
                r = np.sqrt(r2)
                cp = np.clip((rx * tn[0] + ry * tn[1] + rz * tn[2]) / r, 0, 1)
                cw = np.clip(tz / r, 0, 1)
                t_acc += cp * cw / r2 * dA / np.pi
        acc[tun] = 0.12 * t_acc
    return acc


def sky_factor_coarse(p, n, k):
    """The view factor, worked out once per k x k samples (it varies slowly across a surface) and repeated."""
    if k == 1:
        return sky_factor(p, n)
    sub = sky_factor(tuple(a[::k, ::k] for a in p), tuple(a[::k, ::k] for a in n))
    return np.repeat(np.repeat(sub, k, 0), k, 1)[:p[0].shape[0], :p[0].shape[1]]


def lamp_factor(p, n):
    lx, ly, lz = LAMP_LIGHT
    rx, ry, rz = lx - p[0], ly - p[1], lz - p[2]
    r2 = rx * rx + ry * ry + rz * rz + 0.02
    r = np.sqrt(r2)
    return np.clip((rx * n[0] + ry * n[1] + rz * n[2]) / r, 0, 1) / r2


def albedo(H, t):
    """Colour by material and place: the room's surfaces (materials.py), the cat, the props."""
    M = H.m
    A = MATS.albedo(M, H.p, H.n)
    cat = M == CAT
    if cat.any():
        A[cat] = CATMOD.albedo(tuple(a[cat] for a in H.p), tuple(a[cat] for a in H.n), t)
    for pr in PROPS.PROPS:
        sel = np.isin(M, pr.mats)
        if sel.any():
            A[sel] = pr.albedo(tuple(a[sel] for a in H.p), tuple(a[sel] for a in H.n), M[sel])
    return A
