"""Ray-primitive intersections for First Light's room and its cat, vectorised over rays (numpy)."""
import numpy as np


# --- primitives: each returns (t, nx, ny, nz) for rays o + t d, t = inf where missed ---------------------------
def _norm(x, y, z):
    n = np.sqrt(x * x + y * y + z * z) + 1e-12
    return x / n, y / n, z / n


def hit_box(b, o, d):
    (x0, y0, z0), (x1, y1, z1) = b
    with np.errstate(divide='ignore', invalid='ignore'):
        ts = []
        for lo, hi, oo, dd in ((x0, x1, o[0], d[0]), (y0, y1, o[1], d[1]), (z0, z1, o[2], d[2])):
            a, c = (lo - oo) / dd, (hi - oo) / dd
            ts.append((np.minimum(a, c), np.maximum(a, c)))
    tmin = np.maximum(np.maximum(ts[0][0], ts[1][0]), ts[2][0])
    tmax = np.minimum(np.minimum(ts[0][1], ts[1][1]), ts[2][1])
    hit = (tmax >= tmin) & (tmin > 1e-4)
    t = np.where(hit, tmin, np.inf)
    n = [np.zeros_like(t) for _ in range(3)]
    for k in range(3):
        on = hit & (tmin == ts[k][0])
        n[k] = np.where(on, -np.sign(d[k]), n[k])
    return t, n[0], n[1], n[2]


def hit_ellipsoid(c, r, o, d, R=None):
    """An ellipsoid centred at c with semi-axes r, rotated by R (local -> world), if given."""
    ox, oy, oz = o[0] - c[0], o[1] - c[1], o[2] - c[2]
    dx, dy, dz = d
    if R is not None:
        ox, oy, oz = (R[0, 0] * ox + R[1, 0] * oy + R[2, 0] * oz, R[0, 1] * ox + R[1, 1] * oy + R[2, 1] * oz,
                      R[0, 2] * ox + R[1, 2] * oy + R[2, 2] * oz)
        dx, dy, dz = (R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz, R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz,
                      R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz)
    ox, oy, oz, dx, dy, dz = ox / r[0], oy / r[1], oz / r[2], dx / r[0], dy / r[1], dz / r[2]
    a = dx * dx + dy * dy + dz * dz
    b = ox * dx + oy * dy + oz * dz
    cc = ox * ox + oy * oy + oz * oz - 1
    disc = b * b - a * cc
    with np.errstate(invalid='ignore'):
        t = (-b - np.sqrt(disc)) / a
    t = np.where((disc > 0) & (t > 1e-4), t, np.inf)
    px, py, pz = (ox + t * dx) / r[0], (oy + t * dy) / r[1], (oz + t * dz) / r[2]
    if R is not None:
        px, py, pz = R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz, R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz, \
            R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz
    nx, ny, nz = _norm(np.nan_to_num(px), np.nan_to_num(py), np.nan_to_num(pz))
    return t, nx, ny, nz


def hit_cyl(c, r, y0, y1, o, d):
    """A closed vertical cylinder: axis through (c[0], ., c[1]), from y0 to y1."""
    ox, oz = o[0] - c[0], o[2] - c[1]
    dx, dy, dz = d
    a = dx * dx + dz * dz
    b = ox * dx + oz * dz
    cc = ox * ox + oz * oz - r * r
    disc = b * b - a * cc
    with np.errstate(invalid='ignore', divide='ignore'):
        ts = (-b - np.sqrt(disc)) / a
        ys = o[1] + ts * dy
        side = (disc > 0) & (ts > 1e-4) & (ys >= y0) & (ys <= y1)
        tt = (y1 - o[1]) / dy                                   # the top cap (seen from above)
        top = (tt > 1e-4) & ((ox + tt * dx) ** 2 + (oz + tt * dz) ** 2 <= r * r)
    t = np.where(side, ts, np.inf)
    t2 = np.where(top, tt, np.inf)
    use_top = t2 < t
    t = np.minimum(t, t2)
    nx = np.where(use_top, 0, (ox + ts * dx) / r)
    nz = np.where(use_top, 0, (oz + ts * dz) / r)
    ny = np.where(use_top, 1.0, 0.0)
    return t, np.nan_to_num(nx), ny, np.nan_to_num(nz)


def hit_tri(p0, p1, p2, o, d):
    e1, e2 = np.subtract(p1, p0), np.subtract(p2, p0)
    dx, dy, dz = d
    px, py, pz = dy * e2[2] - dz * e2[1], dz * e2[0] - dx * e2[2], dx * e2[1] - dy * e2[0]
    det = e1[0] * px + e1[1] * py + e1[2] * pz
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1 / det
        sx, sy, sz = o[0] - p0[0], o[1] - p0[1], o[2] - p0[2]
        u = (sx * px + sy * py + sz * pz) * inv
        qx, qy, qz = sy * e1[2] - sz * e1[1], sz * e1[0] - sx * e1[2], sx * e1[1] - sy * e1[0]
        v = (dx * qx + dy * qy + dz * qz) * inv
        t = (e2[0] * qx + e2[1] * qy + e2[2] * qz) * inv
    ok = (np.abs(det) > 1e-9) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4)
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n)
    t = np.where(ok, t, np.inf)
    s = np.where(n[0] * dx + n[1] * dy + n[2] * dz > 0, -1.0, 1.0)      # face the viewer
    return t, s * n[0], s * n[1], s * n[2]


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


