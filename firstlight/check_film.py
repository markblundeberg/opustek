"""Checks for First Light's film: that its light is where the geometry puts it, and its pictures land on the notes.

    python3 firstlight/check_film.py                      # the geometry, on the film's own code (seconds)
    python3 firstlight/check_film.py out/firstlight_film.mp4   # and the encoded film: sync, one-frame jumps
    python3 firstlight/check_film.py --selftest [film.mp4]   # each check, broken once, must fail

  patch     the sun's patches, worked out forward: the opening's corners (both faces of the wall, and the
            mullion) projected along the sun's rays onto the left wall and onto the seat's cushion, as polygons.
            The film's own test runs backward (a ray from each point toward the sun). On a grid of points away
            from the polygons' edges (farther than the penumbra), the two must agree everywhere.
  outro     every note of the outro turns off at least one light the camera, standing where it then stands, sees.
  sun       the room's light comes from where the sun is drawn: the direction the room is lit along, put
            through the camera at the window, lands within 1.5 px of the sun disc's centre in the picture; and
            from the encoded film, the bright disc found in the frames sits within 1.5 px of it too.
  mast      from the encoded film: the mast light's brightness, frame by frame, against the hook's rhythm the
            film means to blink: correlation >= 0.9, at a lag of 0 frames (of -3..3).
  lights    from the encoded film: each light a fragment of the hook turns on brightens within 1 frame of its note.
  glitter   from the encoded film: the sun's glitter on the water falls straight below the drawn sun (<= 4 px).
  jumps     from the encoded film: no frame whose change from the last is 4x its neighbours' and over 1.5 levels,
            except where the score puts one (the lamp going off, at the crash of bar 41).
"""
import subprocess
import sys
import os

import numpy as np
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import room as R  # noqa: E402
import film  # noqa: E402

TIMES = (90.0, 110.0, 135.0, 170.0)
# the room as built, kept here so that a break in room.py can't move the yardstick with it
TRUE_WIN, TRUE_DEPTH, TRUE_FRAME, TRUE_MULL, TRUE_TRANSOM = R.WIN, R.DEPTH, R.FRAME, R.MULL, R.TRANSOM


def project(pts, s, plane):
    """Points (x, y, z) carried along the sun's rays (away from the sun) onto a plane ('x', X0) or ('y', Y)."""
    out = []
    for x, y, z in pts:
        axis, c = plane
        lam = ((x - c) / s[0]) if axis == 'x' else ((y - c) / s[1])
        q = (x - lam * s[0], y - lam * s[1], z - lam * s[2])
        out.append((q[1], q[2]) if axis == 'x' else (q[0], q[2]))
    return Path(out)


def rect(x0, x1, y0, y1, z):
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]


def patch_check(aperture=R.aperture):
    bad, worst = 0, ''
    W, D, FR, MU, TR = TRUE_WIN, TRUE_DEPTH, TRUE_FRAME, TRUE_MULL, TRUE_TRANSOM
    for t in TIMES:
        s = film.sun_dir(t)
        for name, plane, grid in (('left wall', ('x', R.X0), (np.linspace(0, R.Y1, 160), np.linspace(0.02, 3.0, 240))),
                                  ('cushion', ('y', R.WIN[2]), (np.linspace(R.CUSH[0][0], R.CUSH[1][0], 240),
                                                                np.linspace(0.02, R.CUSH[1][2], 120)))):
            hole = project(rect(W[0], W[1], W[2], W[3], 0.0), s, plane)
            glass = project(rect(W[0] + FR, W[1] - FR, W[2] + FR, W[3] - FR, -D), s, plane)
            mull = project(rect(MU[0], MU[1], W[2] - 1, W[3] + 1, -D), s, plane)
            tran = project(rect(W[0] - 1, W[1] + 1, TR[0], TR[1], -D), s, plane)
            a, b = np.meshgrid(*grid, indexing='ij')
            pts = np.stack([a.ravel(), b.ravel()], 1)
            inside = (hole.contains_points(pts) & glass.contains_points(pts) & ~mull.contains_points(pts)
                      & ~tran.contains_points(pts))
            # points near an edge are not judged: within 4 cm, wider than the penumbra, or wider where the sun
            # grazes the surface, which stretches the penumbra by 1 / sin(the sun's angle to the surface)
            graze = abs(s[0]) if plane[0] == 'x' else abs(s[1])
            mg = 0.04 / max(graze, 0.15) if graze < 0.3 else 0.04
            near = np.zeros(len(pts), bool)
            for P in (hole, glass, mull, tran):
                near |= P.contains_points(pts, radius=mg) != P.contains_points(pts, radius=-mg)
            if plane[0] == 'x':
                p = (np.full(len(pts), R.X0), pts[:, 0], pts[:, 1])
                n = (np.ones(len(pts)), np.zeros(len(pts)), np.zeros(len(pts)))
            else:
                p = (pts[:, 0], np.full(len(pts), plane[1]), pts[:, 1])
                n = (np.zeros(len(pts)), np.ones(len(pts)), np.zeros(len(pts)))
            lit = aperture(p, s, n) > 0.5
            dis = int(((lit != inside) & ~near).sum())
            if dis:
                bad += dis
                worst = worst or f'{name} at {t:g} s: {dis} points'
            if t == TIMES[1] and name == 'left wall':
                lit_share = (inside & ~near).mean()
    ok = bad == 0 and lit_share > 0.05
    return ok, f'{bad} judged points disagree ({worst or "none"}); at {TIMES[1]:g} s {lit_share * 100:.0f}% of the left wall is lit'


def outro_check(lights=None, offs=None):
    """Every note of the outro turns off at least one light the camera can see from where it stands then."""
    lights, offs = lights or film.LIGHTS, offs or film.LIGHTS_OFF
    seen = R.cast(R.Z_MID, 0, 540, 960, 1.0).m == R.ENV
    blind = []
    for t in sorted(set(o for o in offs if np.isfinite(o))):
        grp = [j for j, o in enumerate(offs) if o == t]
        if not any(seen[int(screen_of(lights[j][0], lights[j][1], 0)[1]), int(screen_of(lights[j][0], lights[j][1], 0)[0])]
                   for j in grp):
            blind.append(round(t, 1))
    n = len(set(o for o in offs if np.isfinite(o)))
    return not blind and n > 0, f'{n} outro notes; {len(blind)} turn off nothing the camera sees' + (f' (at {blind} s)' if blind else '')


def sun_check(sun_dir=None):
    sun_dir = sun_dir or film.sun_dir
    errs = []
    for t in TIMES:
        s = sun_dir(t)
        u = R.CX + R.F * s[0] / -s[2]
        v = R.CY - R.F * s[1] / -s[2]
        sx, sy = film.sun_pos(t)                            # the disc's centre in the picture outside
        u2 = R.U0 + (sx - film.OPEN[0]) * R.ENV_K
        v2 = R.V0 + (sy - film.OPEN[2]) * R.ENV_K
        errs.append(np.hypot(u - u2, v - v2))
    return max(errs) <= 1.5, f'worst {max(errs):.2f} px (<= 1.5) between the light and the drawn sun'


def frames(path, scale_h=540):
    w = scale_h * 16 // 9
    p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', path, '-vf', f'scale={w}:{scale_h}:flags=area',
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    n = w * scale_h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(scale_h, w, 3)
    p.wait()


def screen_of(ox, oy, t, h=540):
    """Where a point of the picture outside lands on the frame (in px of a frame h high), from the camera at t."""
    u = R.U0 + (ox - film.OPEN[0]) * R.ENV_K
    v = R.V0 + (oy - film.OPEN[2]) * R.ENV_K
    dx, dy = (u - R.CX) / R.F, (R.CY - v) / R.F           # a direction: the same from wherever the camera stands
    return (R.CX + R.F * dx) * h / 540, (R.CY - R.F * dy) * h / 540


def video_checks(path, shift=0):
    mast_x, mast_y = screen_of(*film.MAST, 0)
    t_clear = next(t for t in np.arange(film.T_TOP, film.DUR, 0.1)
                   if film.sun_pos(t)[1] + film.SUN_R + 2 < film.ridge_far(film.sun_pos(t)[0]))
    lights = [(screen_of(x, y, 0), t_on) for (x, y, kind, t_on, vel) in film.LIGHTS]
    seen = R.cast(R.Z_CLOSE, 0, 540, 960, 1.0).m == R.ENV       # what the camera at the window sees of the outside
    hidden = sum(not seen[int(v), int(u)] for (u, v), _ in lights)
    mast, diffs, light_tr, sun_err, glit = [], [], {k: [] for k in range(len(lights))}, [], []
    prev = None
    for k, rgb in enumerate(frames(path)):
        t = k / film.FPS
        Y = rgb.astype(np.float32).mean(2)
        if t < film.tb(48):
            mast.append(Y[int(mast_y) - 1:int(mast_y) + 2, int(mast_x) - 1:int(mast_x) + 2].mean())
        for j, ((lx, ly), t_on) in enumerate(lights):
            if abs(t - t_on) < 0.5:
                light_tr[j].append((k, Y[int(ly) - 1:int(ly) + 2, int(lx) - 1:int(lx) + 2].max()))
        if film.T_TOP + 12 < t < film.tb(60) and k % 15 == 0:     # the glitter falls straight below the sun
            sky_ = Y[30:250, 100:850]
            yy_, xx_ = np.nonzero(sky_ > sky_.max() - 6)
            col_u = []
            wl = int(film.R.V0 + (film.YH - film.OPEN[2]) * film.R.ENV_K)       # the waterline: a direction, so
            for v0 in range(wl + 40, wl + 160, 30):                               # the same row from any camera z;
                band = Y[v0:v0 + 25, 100:850]                                     # below the water that mirrors the
                cols = np.nonzero(band >= np.percentile(band, 99.5))[1]           # headland (no glints there)
                if len(cols):
                    col_u.append(np.median(cols) + 100)
            if col_u and len(yy_):
                glit.append(abs(np.median(col_u) - (xx_.mean() + 100)))
        if t_clear < t < t_clear + 40 and k % 15 == 0:            # the whole disc clear of the hills
            sd = film.sun_dir(t)
            u, v = (R.CX + R.F * sd[0] / -sd[2]) * 540 / 540, (R.CY - R.F * sd[1] / -sd[2])
            box = Y[int(v) - 30:int(v) + 30, int(u) - 30:int(u) + 30]
            yy, xx = np.nonzero(box > 0.92 * box.max())
            if len(yy) > 20:
                sun_err.append(np.hypot(xx.mean() + int(u) - 30 + 0.5 - u, yy.mean() + int(v) - 30 + 0.5 - v))
        if prev is not None:
            diffs.append(np.abs(Y - prev).mean())
        prev = Y
    res = {}
    m = np.array(mast)
    m = m - np.array([np.median(m[max(0, k - 30):k + 30]) for k in range(len(m))])   # the sky brightens behind it
    want = np.array([film.mast_blink((k + shift) / film.FPS) for k in range(len(m))])
    best = max(range(-3, 4), key=lambda L: np.corrcoef(np.roll(m, L), want)[0, 1])
    r0 = np.corrcoef(m, want)[0, 1]
    res['mast'] = (r0 >= 0.9 and best == 0, f'correlation {r0:.2f} (>= 0.9), best at a lag of {best} frames (0)')
    errs = []
    for j, ((lx, ly), t_on) in enumerate(lights):
        tr = light_tr[j]
        if len(tr) < 4:
            continue
        ks, vs = np.array(tr).T
        jump = np.diff(vs)
        onset = ks[1:][np.argmax(jump)]
        errs.append(onset - round((t_on + shift / film.FPS) * film.FPS + 0.5))
    errs = np.abs(np.array(errs))
    res['lights'] = (errs.max() <= 1 and hidden == 0, f'{len(errs)} lights, {hidden} hidden behind the frame or mullion (0); '
                     f'onsets within {int(errs.max())} frames of their notes (<= 1)')
    gl = np.array(glit)
    res['glitter'] = (len(gl) > 5 and gl.max() <= 4, f'the glitter column within {gl.max():.1f} px of the sun disc\'s '
                      f'column in {len(gl)} frames (<= 4)')
    se = np.array(sun_err + [np.inf] * (not sun_err))
    res['sun on film'] = (len(se) > 5 and se.max() <= 1.5, f'the drawn sun disc, found in {len(se)} frames, sits within '
                          f'{se.max():.2f} px (<= 1.5, at 540 high) of the direction the room is lit from')
    d = np.array(diffs)
    med = np.array([np.median(d[max(0, i - 15):i + 15]) for i in range(len(d))])
    spikes = [i + 1 for i in range(len(d)) if d[i] > 4 * med[i] + 1e-6 and d[i] > 1.5]
    allowed = round(film.LAMP_OFF * film.FPS)
    extra = [k for k in spikes if abs(k - allowed) > 4]
    res['jumps'] = (not extra, f'{len(spikes)} one-frame jumps, {len(extra)} unexplained'
                    + (f' (frames {extra[:6]})' if extra else '') + '; largest frame change '
                    f'{d.max():.2f} levels at frame {int(np.argmax(d)) + 1}')
    return res


def report(res):
    for k, (ok, msg) in res.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}: {msg}")
    return all(ok for ok, _ in res.values())


def selftest():
    good = True
    print('broken: the sun mirrored left-right in the film\'s light test')
    ok, msg = patch_check(lambda p, s, n: R.aperture(p, (-s[0], s[1], s[2]), n))
    print(f"  {'missed' if ok else 'caught'}: {msg}")
    good &= not ok
    print('broken: the wall drawn without its thickness')
    save = R.DEPTH
    try:
        R.DEPTH = 0.0
        ok, msg = patch_check(lambda p, s, n: R.aperture(p, s, n))
    finally:
        R.DEPTH = save
    print(f"  {'missed' if ok else 'caught'}: {msg}")
    good &= not ok
    print('broken: the mullion moved 10 cm')
    save = R.MULL
    try:
        R.MULL = (save[0] + 0.1, save[1] + 0.1)
        ok, msg = patch_check(lambda p, s, n: R.aperture(p, s, n))
        R.MULL = save
    finally:
        R.MULL = save
    print(f"  {'missed' if ok else 'caught'}: {msg}")
    good &= not ok
    print('broken: the room lit from a sun 1 degree higher than the one drawn')
    def high(t):
        s = film.sun_dir(t)
        e = np.arcsin(s[1]) + np.radians(1)
        h = np.hypot(s[0], s[2])
        return np.array([s[0] / h * np.cos(e), np.sin(e), s[2] / h * np.cos(e)])
    ok, msg = sun_check(high)
    print(f"  {'missed' if ok else 'caught'}: {msg}")
    good &= not ok
    print('broken: one outro note\'s visible lights moved to a moment of their own')
    end = R.cast(R.Z_MID, 0, 540, 960, 1.0).m == R.ENV
    vis = [bool(end[int(screen_of(x, y, 0)[1]), int(screen_of(x, y, 0)[0])]) for x, y, *_ in film.LIGHTS]
    offs = film.LIGHTS_OFF
    t_mix = next(t for t in sorted(set(o for o in offs if np.isfinite(o)))
                 if any(o == t and not v for o, v in zip(offs, vis)))          # a note with a hidden light too
    later = min(o for o in offs if np.isfinite(o) and o > t_mix)
    moved = [later if o == t_mix and v else o for o, v in zip(offs, vis)]
    ok, msg = outro_check(film.LIGHTS, moved)
    print(f"  {'missed' if ok else 'caught'}: {msg}")
    good &= not ok
    if len(sys.argv) > 2:
        print('broken: the encoded film read against a score 3 frames late')
        res = video_checks(sys.argv[2], shift=3)
        for k in ('mast', 'lights'):
            print(f"  {'missed' if res[k][0] else 'caught'} ({k}): {res[k][1]}")
            good &= not res[k][0]
    print('clean:')
    good &= report({'patch': patch_check(), 'sun': sun_check(), 'outro': outro_check()})
    print('-> each break caught, the clean code passes' if good else '-> SELF-TEST FAILED')
    return good


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(0 if selftest() else 1)
    res = {'patch': patch_check(), 'sun': sun_check(), 'outro': outro_check()}
    if len(sys.argv) > 1:
        res.update(video_checks(sys.argv[1]))
    sys.exit(0 if report(res) else 1)
