"""Checks on the film. Frame checks read rendered pixels; the geometry check
compares the drawn mechanism with the audio itself (a check on what the
mechanism produces can pass a mechanism drawn wrong).

  python3 engine/film/check_film.py FRAMES_DIR T0        frame checks + geometry + camera
  python3 engine/film/check_film.py --selftest-camera | --selftest-geometry
  FILM_BREAK=<name> python3 engine/film/check_film.py ...   (see film.py's list)

  spark:    the spark is drawn at the gap in exactly the frames that catch a
            gate opening, and not in the frame after (>= 95%)
  strobe:   in those frames the piston is drawn solid at the crank angle of
            the spark's instant (its crown a continuous line, >= 85% of its
            length), and in the next frame that line is lighter (the freeze
            decays, >= 90%)
  flash:    no more than 3 general flashes a second (WCAG 2.3.1: a flash is a
            pair of opposing changes of relative luminance >= 0.1, the darker
            below 0.8, over >= 25% of any 640 x 360 window); the same for
            saturated red
  heat:     the hot bulb's most saturated pixel has a blackbody hue at the model
            temperature (median within 2 degrees) and its saturation follows
            the glow strength (r >= 0.6), when T > 950 K
  coast:    choked in the pause, the engine slows monotonically to a stop and
            is drawn solid at the end
  mechanism: port timing from the drawn geometry, and nothing collides
  numerals: every numeral is drawn, and its leader ends on its own part
  bucking:  the rocking's thump peaks on the kick (within 20 ms for >= 90%)
            and rings down before the next
  fixed parts: drawn with the engine rocked -2 and +2 degrees, nothing on the
            bed moves
  geometry: the drawn blade under the aperture agrees with the gate (>= 95%) in
            every chop the disc is drawn bladed (not spun)
  camera:   smooth moves under 30 px a frame; cuts on beats
"""
import colorsys
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import film

FPS = film.FPS


def frame(fr, t0, t):
    k = int(round((t - t0) * FPS))
    p = os.path.join(fr, f'{k:05d}.png')
    return np.asarray(Image.open(p).convert('RGB')).astype(float) / 255 if os.path.exists(p) else None


def onscreen(t, X, Y, m=8):
    x, y = film.to_screen(t, X, Y)
    return (x, y) if (m <= x < film.W - m and m <= y < film.H - m) else None


def darkness(im, x, y, r=1):
    px = im[int(y) - r:int(y) + r + 1, int(x) - r:int(x) + r + 1].mean(axis=(0, 1))
    return 1 - px.mean()


def engine_screen(t, X, Y):
    return onscreen(t, *film.engine_point(t, X, Y))


def spark(fr, t0, t1):
    gx, gy, g = film.GAP
    ok = n = 0
    marks = set(int(ts * FPS + 1e-9) for ts in film.E_SPARKS if t0 <= ts < t1)
    cands = [(k, True) for k in marks] + [(k + 1, False) for k in marks if k + 1 not in marks]
    for k, want in cands:
        tt = k / FPS
        im = frame(fr, t0, tt)
        if im is None:
            continue
        a, b = onscreen(tt, gx - 10, gy - 12), onscreen(tt, gx + 10, gy + 12)   # clear of the balls and their shade lines
        if not (a and b):
            continue
        # the darkest pixel anywhere between the balls (single samples can fall between a thin zigzag's strokes)
        d = 1 - im[int(a[1]):int(b[1]) + 1, int(a[0]):int(b[0]) + 1].mean(axis=2).min()
        n += 1
        ok += ((d > 0.35) == want)
    return n > 10 and ok >= 0.95 * n, f'{ok}/{n} frames: spark drawn exactly when one is caught'


def strobe(fr, t0, t1):
    ok = n = 0
    later = 0
    edges = tuple(round(film.crown(a) - d) for a in (0.0, math.pi) for d in (0, film.PW))   # the phantom piston's ends
    for ts in film.E_SPARKS:
        if not (t0 <= ts < t1):
            continue
        k = int(ts * FPS + 1e-9)
        tt = k / FPS
        if not film.running(tt):
            continue
        c = film.crown(film.theta_at(ts))
        if min(abs(c - e) for e in edges) < 25:
            continue
        pts = [engine_screen(tt, c, Y) for Y in range(800, 925, 5)]
        if not all(pts):
            continue
        im = frame(fr, t0, tt)
        im2 = frame(fr, t0, tt + 1 / FPS)
        if im is None or im2 is None:
            continue
        dk = [max(darkness(im, x + dx, y, r=0) for dx in (-2, -1, 0, 1, 2)) for x, y in pts]
        pts2 = [p for p in (engine_screen(tt + 1 / FPS, c, Y) for Y in range(800, 925, 5)) if p]
        dk2 = [max(darkness(im2, x + dx, y, r=0) for dx in (-2, -1, 0, 1, 2)) for x, y in pts2]
        n += 1
        ok += np.mean(np.array(dk) > 0.3) >= 0.85
        # the freeze decays: next frame, the same line is lighter (unless a new spark lands)
        later += (np.mean(dk2) < 0.95 * np.mean(dk) or film.spark_in_frame(tt + 1 / FPS) is not None)
    return n >= 10 and ok >= 0.9 * n and later >= 0.9 * n, f'{ok}/{n} spark frames show the crown solid at the spark angle; {later}/{n} next frames lighter (decaying)'


def _lum(path):
    im = np.asarray(Image.open(path).convert('RGB')).astype(float) / 255
    lin = np.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    s = im.sum(2) + 1e-9
    red = (im[..., 0] / s >= 0.8) & (im[..., 0] > 0.5)
    h, w = Y.shape
    B = 20
    Yb = Y[:h // B * B, :w // B * B].reshape(h // B, B, w // B, B).mean((1, 3))
    Rb = red[:h // B * B, :w // B * B].reshape(h // B, B, w // B, B).mean((1, 3)) > 0.5
    return Yb, Rb


def flash(fr, t0, t1, workers=4):
    from multiprocessing import Pool
    files = sorted(os.listdir(fr))
    with Pool(workers) as p:
        L = p.map(_lum, [os.path.join(fr, f) for f in files], chunksize=32)
    wy, wx = 18, 32                                    # 640 x 360 px window in 20 px blocks
    trans_g, trans_r = [], []
    for i in range(1, len(L)):
        (y0, r0), (y1, r1) = L[i - 1], L[i]
        dY = y1 - y0
        dark = np.minimum(y0, y1) < 0.8
        sg = sr = 0
        # an area changing one way: lines that merely move brighten some blocks and darken their
        # neighbours in the same frame (the engine rocking at close zoom), which is motion, not a flash
        for sign, m in ((1, (dY >= 0.1) & dark), (-1, (dY <= -0.1) & dark), (0, r0 != r1)):
            cs = np.pad(m.astype(float).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            win = cs[wy:, wx:] - cs[:-wy, wx:] - cs[wy:, :-wx] + cs[:-wy, :-wx]
            if (win >= 0.25 * wy * wx).any():
                if sign:
                    sg = sign
                else:
                    sr = 1
        trans_g.append(sg)
        trans_r.append(sr)

    def worst(tr):
        ev, last = [], 0
        for i, s_ in enumerate(tr):
            if s_ and s_ == -last:
                ev.append(i)                           # a flash: this transition opposes the last
            if s_:
                last = s_
        return max((sum(1 for e in ev if a <= e < a + FPS) for a in range(0, len(tr), 5)), default=0)
    g = worst(trans_g)
    r = max((sum(trans_r[a:a + FPS]) // 2 for a in range(0, len(trans_r), 5)), default=0)   # two red changes make a red flash
    return g <= 3 and r <= 3, f'most flashes in any second: general {g}, red {r} (limit 3)'


def heat(fr, t0, t1):
    errs, sats, strength, n = [], [], [], 0
    for tt in np.arange(t0 + 0.5, t1, 0.25):
        T = float(np.interp(tt, film.TC, film.TEMP))
        if T < 950:
            continue
        im = frame(fr, t0, tt)
        if im is None:
            continue
        p = engine_screen(tt, film.BULB[0] + 20, film.BULB[1])
        if not p:
            continue
        x, y = p
        px = im[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4].reshape(-1, 3)
        px = px[np.argmax(px.max(1) - px.min(1))]
        h = colorsys.rgb_to_hsv(*px)[0] * 360
        want = film.blackbody(T)[0]
        w = np.array(want) * np.array(film.PAPER)
        hw = colorsys.rgb_to_hsv(*w)[0] * 360
        errs.append(abs((h - hw + 180) % 360 - 180))
        sats.append(colorsys.rgb_to_hsv(*px)[1])
        strength.append(math.log10(film.blackbody(T)[1]) - math.log10(film.Y_DRAPER))
        n += 1
    if n < 5:
        return False, 'too few hot frames sampled'
    e = np.array(errs)
    r = np.corrcoef(sats, strength)[0, 1]
    return np.median(e) <= 2.0 and r >= 0.6, f'{n} hot frames, median hue error {np.median(e):.1f} deg; saturation follows temperature r = {r:.2f}'


def coast(fr, t0, t1):
    """The chop chokes the engine and it coasts through the bar's pause: its
    speed never rises within the pause, ends under a tenth of where it began,
    and the last frames of the pause draw the crank solid (slow enough to see)."""
    a, b = film.PAUSE
    sp = np.interp(np.arange(a + 0.05, b - 0.02, 0.01), film.TC, film.SPEED)
    mono = bool(np.all(np.diff(sp) <= 1e-9))
    end_ok = sp[-1] < 0.1 * max(sp[0], 1e-9)
    solid = not film.running(b - 0.05)
    return mono and end_ok and solid, f'pause: {sp[0]:.1f} -> {sp[-1]:.1f} rev/s, never rising: {mono}; drawn solid at its end: {solid}'


def mechanism():
    """The engine's timing and clearances, swept over a revolution: the skirt
    seals the exhaust port from the crankcase at every angle; the exhaust
    opens before the transfer; both are fully open at BDC; the intake flap is
    shut whenever the transfer port is open; the skirt clears the flywheel at
    BDC; the flywheel clears the case; the head studs clear the ports."""
    ths = np.linspace(0, 2 * np.pi, 3601)
    crowns = np.array([film.crown(t) for t in ths])
    skirt = crowns - film.PW
    ex0, ex1 = film.EXHAUST
    t0, t1 = film.TRANSFER
    res = {}
    res['skirt seals exhaust'] = skirt.max() <= ex0
    open_ex = crowns < ex1
    open_tr = crowns < t1
    first_ex = np.degrees(ths[np.argmax(open_ex)])
    first_tr = np.degrees(ths[np.argmax(open_tr)])
    res['exhaust before transfer'] = first_ex < first_tr
    res['ports open at BDC'] = film.crown(np.pi) <= min(ex0, t0)
    reed = np.array([film.reed_open(t) for t in ths])
    res['flap shut while transfer open'] = not np.any(reed & open_tr)
    sk = film.crown(np.pi) - film.PW
    half = film.CYL[3] / 2 - film.WALL - 3
    res['skirt clears flywheel'] = math.hypot(sk - film.FW[0], half) > film.FW[2] + 5
    res['flywheel clears case'] = film.FW[2] + 5 < film.CASE[0]
    res['studs clear ports'] = all(abs(bx - p) > 15 for bx, by in film.BOLTS for p in film.PORTS.values())
    exd = np.degrees(open_ex.sum() / len(ths) * 2 * np.pi)
    trd = np.degrees(open_tr.sum() / len(ths) * 2 * np.pi)
    bad = [k for k, v in res.items() if not v]
    return not bad, (f'exhaust opens {first_ex:.0f} deg ATDC ({exd:.0f} deg open), transfer {first_tr:.0f} ({trd:.0f}); '
                     + ('all clear' if not bad else 'FAILS: ' + ', '.join(bad)))


def bucking():
    """The engine bucks on the beat (Mark: the rhythm didn't read): for kicks
    in the running sections, the rocking's largest swing within 100 ms before
    to 200 ms after the kick lands within 20 ms of film.THUMP_LEAD before it, and one swing per beat
    dominates: after the thump's trough, nothing before the next thump rises
    past a quarter of it (median over kicks; a ringing mount swings two or
    three times a beat). And the thumps carry on through the chops: their
    median there is at least 0.6 of the median elsewhere."""
    R = np.abs(film.ROCK)
    lags, rings, peaks = [], [], {True: [], False: []}
    chopped = lambda k: any(film.tb(b) <= k < film.tb(b) + n * film.BEAT for b, n, *_ in film.E['chops'])
    for k in film.E['kicks']:
        i = int(k * 1000)
        if i < 200 or i + 500 > len(R) or float(np.interp(k, film.TC, film.SPEED)) < 20:
            continue
        seg = R[i - 100:i + 200]
        j = int(np.argmax(seg))
        lags.append(j - 100 + round(1000 * film.THUMP_LEAD))   # from the intended lead
        peaks[chopped(k)].append(seg[j])
        tail = R[i - 100 + j:i + 400]           # until the next thump starts to rise (it is timed 54 ms early)
        below = np.nonzero(tail < 0.5 * seg[j])[0]
        if not len(below):
            rings.append(1.0)
            continue
        rest = tail[below[0]:]
        trough = int(np.argmin(rest))           # a second swing has to rise again after the trough
        rings.append(rest[trough:].max() / max(seg[j], 1e-9))
    lags = np.array(lags)
    through = np.median(peaks[True]) / np.median(peaks[False])
    ok = np.mean(np.abs(lags) <= 20) >= 0.9 and np.median(rings) < 0.25 and through >= 0.6
    return ok, (f'{len(lags)} kicks: peak {np.median(lags):+.0f} ms from {1000 * film.THUMP_LEAD:.0f} ms before the kick (median), within 20 ms '
                f'for {100 * np.mean(np.abs(lags) <= 20):.0f}%; later swings at {np.median(rings):.2f} of the thump; '
                f'thumps in the chops at {through:.2f} of the rest')


def fixed_parts():
    """What stands on the bed stays put while the engine bucks: one frame drawn
    twice, with the engine rocked -2 and +2 degrees (and pressed down or not),
    must be pixel-identical at the drip pan, the commutator, the coil and the
    cell, and at the figure's label."""
    import tempfile
    t = film.tb(65) + 0.1
    saved = film.rock_at, film.bounce_at, film.camera
    film.camera = lambda tt: (1600, 900, 0.6, 0, 0)
    ims = []
    try:
        for ang, bo in ((-2.0, 0.0), (2.0, 4.0)):
            film.rock_at = lambda tt, a=ang: math.radians(a)
            film.bounce_at = lambda tt, b=bo: b
            with tempfile.NamedTemporaryFile(suffix='.png') as fh:
                film.draw(t, fh.name)
                ims.append(np.asarray(Image.open(fh.name).convert('L')).astype(int))
    finally:
        film.rock_at, film.bounce_at, film.camera = saved
    boxes = {'drip pan': (1380, 1170, 1490, 1226), 'commutator': (1910, 780, 2100, 1222),
             'coil and gap': (2100, 850, 2240, 1222), 'cell': (2250, 1190, 2330, 1308), 'Fig. 1. label': (1050, 1345, 1210, 1395)}
    moved = []
    for name, (x0, y0, x1, y1) in boxes.items():
        a = [int((v - c) * 0.6 + o) for v, c, o in ((x0, 1600, 960), (y0, 900, 540), (x1, 1600, 960), (y1, 900, 540))]
        d = np.abs(ims[0][a[1]:a[3], a[0]:a[2]] - ims[1][a[1]:a[3], a[0]:a[2]]).max()
        if d > 8:
            moved.append(name)
    return not moved, 'fixed parts still' + ('' if not moved else '; moving with the engine: ' + ', '.join(moved))


def numerals():
    """Every numeral's leader lands on its own part (film.LEGEND's boxes)."""
    bad = []
    marks = film.numeral_marks(10.0)
    for n, x0, y0, x1, y1 in marks:
        name, (a, b, c, d) = film.LEGEND[n]
        if not (a <= x0 <= c and b <= y0 <= d):
            bad.append(f'{n} ({name})')
    missing = sorted(set(film.LEGEND) - {m[0] for m in marks})
    if missing:
        bad.append('not drawn: ' + ', '.join(map(str, missing)))
    return not bad, f'{len(marks)} numerals of {len(film.LEGEND)}' + ('' if not bad else '; ' + ', '.join(bad))


def geometry():
    """For each chop drawn with blades: is the blade under the aperture (the
    top, -pi/2) open exactly when the gate is open?"""
    res = []
    for bar, beats, per, d0, d1, pat in film.E['chops']:
        t0 = film.tb(bar)
        agree, n = 0, 0
        for t in np.arange(t0 - 0.55, t0 + beats * film.BEAT - 0.005, 0.004):   # includes the lead-in
            phase, sec, live, gate, rate = film.chop_state(t)
            if rate > 12:
                break                                # drawn spun: no blades to be wrong
            drawn_open = film.open_under_aperture(phase, sec)
            if drawn_open is None:
                continue
            n += 1
            agree += (drawn_open == (gate > 0.5))
        if n:
            res.append(round(100 * agree / n))
    return all(r >= 95 for r in res) and len(res) > 0, f'blade under the aperture matches the gate: {res} % (bladed chops)'


def camera_smooth(cam_list=None):
    if cam_list is not None:
        film.CAM = cam_list
    ts = np.arange(0, film.DUR, 1 / FPS)
    cam = np.array([film.camera(t)[:3] for t in ts])
    d = np.hypot(np.diff(cam[:, 0]) * cam[1:, 2], np.diff(cam[:, 1]) * cam[1:, 2]) + np.abs(np.diff(cam[:, 2])) / cam[1:, 2] * 1000
    cuts = [film.tb(b) for b, bt, n in film.CAM if bt == 0]
    sm = [d[i] for i in range(len(d)) if min(abs(ts[i + 1] - c) for c in cuts) > 0.04]
    on_beat = all(abs((c / film.BEAT) - round(c / film.BEAT)) < 1e-6 for c in cuts)
    return max(sm) <= 30 and on_beat, f'fastest smooth move {max(sm):.1f} px/frame; {len(cuts)} cuts, all on beats: {on_beat}'


if __name__ == '__main__':
    if sys.argv[1] == '--selftest-camera':
        bad = [c if c[0] != 52 else (51.5, 2, 'f4') for c in film.CAM]
        r = camera_smooth(bad)
        print('whip pan:', 'caught' if not r[0] else 'NOT CAUGHT', r[1])
        sys.exit(0)
    if sys.argv[1] == '--selftest-geometry':
        r = geometry()
        print(f"blade-phase ({film.BRK or 'no break'}):", 'FAIL' if not r[0] else 'pass', r[1])
        sys.exit(0)
    fr, t0 = sys.argv[1], float(sys.argv[2])
    t1 = t0 + len(os.listdir(fr)) / FPS
    res = [('spark when the gate opens', *spark(fr, t0, t1)), ('strobe freezes the crank', *strobe(fr, t0, t1)),
           ('no flashing (WCAG 2.3.1)', *flash(fr, t0, t1)), ('heat is blackbody', *heat(fr, t0, t1)),
           ('engine coasts through the pause', *coast(fr, t0, t1)), ('mechanism', *mechanism()), ('numerals', *numerals()), ('bucking on the beat', *bucking()), ('fixed parts stay put', *fixed_parts()), ('geometry matches the audio', *geometry()),
           ('camera smooth', *camera_smooth())]
    for n, ok, msg in res:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}: {msg}")
    print('ALL PASS' if all(r[1] for r in res) else 'SOME FAIL')
