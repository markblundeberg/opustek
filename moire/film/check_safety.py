"""Photosensitivity checks for the Moiré film, read from the encoded video (what a viewer gets).

  python3 moire/check_safety.py out/moire/film.mp4
  python3 moire/check_safety.py --selftest

  flash:   no more than 3 general flashes in any second (WCAG 2.3.1: a flash is a pair of opposing changes of
           relative luminance >= 0.1, the darker below 0.8, over >= 25% of any 640 x 360 window, full-HD scale);
           the same for saturated red
  pattern (also screen-wide: broad stripes of five or more pairs across the width, at the same 0.08): regular stripes are a separate trigger (ITU-R BT.1702; Ofcom's guidance): more than five light-dark
           pairs, clearly discernible, over a large share of the screen, worst when they move. A film made of
           gratings must keep their luminance contrast low. Here: in every frame, no more than 3 of the 16 screen
           tiles (so under 25% of the screen) hold a periodic component (2.5 or more pairs a tile, period 6 px or
           more at full HD) whose peak-to-trough relative luminance reaches 0.08 (a margin under the 0.1 that
           defines a flash pair).
"""
import subprocess
import sys

import numpy as np

W, H = 960, 540                  # analysed at half HD
TILE_X, TILE_Y = W // 4, H // 4
LIM = 0.08


def rel_lum(rgb):
    c = rgb.astype(np.float32) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def red_sat(rgb):
    c = rgb.astype(np.float32) / 255
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    return (r / (r + g + b + 1e-6) >= 0.8)       # WCAG's saturated red: R / (R + G + B) >= 0.8


_win = np.outer(np.hanning(TILE_Y), np.hanning(TILE_X)).astype(np.float32)
fy = np.fft.fftfreq(TILE_Y)[:, None]
fx = np.fft.rfftfreq(TILE_X)[None, :]
fr = np.sqrt(fx ** 2 + fy ** 2)
# periods from 3 px (6 px at full HD) up to a fifth... of two tiles: >= 2.5 pairs in a tile
_band = (fr <= 1 / 3) & (fr >= 2.5 / min(TILE_X, TILE_Y))


def stripe_contrast(Y):
    """Peak-to-trough relative luminance of the strongest periodic component in each of the 16 tiles."""
    out = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            t = Y[i * TILE_Y:(i + 1) * TILE_Y, j * TILE_X:(j + 1) * TILE_X]
            F = np.abs(np.fft.rfft2((t - t.mean()) * _win))
            amp = 2 * F[_band].max() / _win.sum()          # a sinusoid's amplitude
            out[i, j] = 2 * amp                            # peak to trough
    return out


_wf = np.outer(np.hanning(H // 4), np.hanning(W // 4)).astype(np.float32)
_ffy = np.fft.fftfreq(H // 4)[:, None] / 4          # cycles per half-HD pixel
_ffx = np.fft.rfftfreq(W // 4)[None, :] / 4
_ffr = np.sqrt(_ffx ** 2 + _ffy ** 2)
# broad stripes across the whole screen: periods from the tiles' limit up to a fifth of the width (five pairs)
_fband = (_ffr <= 2.5 / min(TILE_X, TILE_Y)) & (_ffr >= 5.0 / W)


def broad_contrast(Y):
    """Peak-to-trough relative luminance of the strongest screen-wide stripe component (five or more pairs)."""
    y4 = Y[:H // 4 * 4, :W // 4 * 4].reshape(H // 4, 4, W // 4, 4).mean(axis=(1, 3))
    F = np.abs(np.fft.rfft2((y4 - y4.mean()) * _wf))
    return 2 * (2 * F[_fband].max() / _wf.sum())


def blocks(Y, b=10):
    return Y[:H // b * b, :W // b * b].reshape(H // b, b, W // b, b).mean(axis=(1, 3))


def frames_from(path):
    p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', path, '-vf', f'scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         stdout=subprocess.PIPE)
    n = W * H * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    p.wait()


def run(frames, fps=30, pattern_every=2, quiet=False):
    prevY = prevR = None
    trans_g, trans_r = [], []
    worst_pat, worst_k, n_bad = 0.0, -1, 0
    wy, wx = 18, 32                       # 180 x 320 at half HD in 10 px blocks = 640 x 360 at full HD
    for k, rgb in enumerate(frames):
        Y = rel_lum(rgb)
        Yb = blocks(Y)
        Rb = blocks(red_sat(rgb).astype(np.float32)) > 0.5
        if prevY is not None:
            dY = Yb - prevY
            dark = np.minimum(Yb, prevY) < 0.8
            sg = sr = 0
            for sign, m in ((1, (dY >= 0.1) & dark), (-1, (dY <= -0.1) & dark), (0, Rb != prevR)):
                cs = np.pad(m.astype(np.float32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                win = cs[wy:, wx:] - cs[:-wy, wx:] - cs[wy:, :-wx] + cs[:-wy, :-wx]
                if (win >= 0.25 * wy * wx).any():
                    if sign:
                        sg = sign
                    else:
                        sr = 1
            trans_g.append(sg)
            trans_r.append(sr)
        prevY, prevR = Yb, Rb
        if k % pattern_every == 0:
            c = stripe_contrast(Y)
            bad = int((c >= LIM).sum())
            bc = broad_contrast(Y)
            if max(c.max(), bc) > worst_pat:
                worst_pat, worst_k = float(max(c.max(), bc)), k
            n_bad += (bad > 3) or (bc >= LIM)
    # flashes: a flash is a pair of opposing transitions; count pairs in any 1 s window
    def worst(tr):
        ev, last = [], 0
        for i, s in enumerate(tr):
            if s and s != last:
                if last:
                    ev.append(i)
                last = s
        ev = np.array(ev)
        return int(max(((ev >= i) & (ev < i + fps)).sum() for i in range(len(tr)))) if len(ev) else 0
    fl_g = worst(trans_g)
    fl_r = int(max((np.convolve(trans_r, np.ones(fps), 'valid') // 2).max(), 0)) if len(trans_r) >= fps else 0
    res = {'flash': (fl_g <= 3 and fl_r <= 3, f'at most {fl_g} general and {fl_r} red flashes in any second (<= 3)'),
           'pattern': (n_bad == 0, f'{n_bad} frames with stripes >= {LIM} over more than 3 of 16 tiles; strongest stripe contrast {worst_pat:.3f} (frame {worst_k})')}
    if not quiet:
        for k_, (p_, m_) in res.items():
            print(f"  {'PASS' if p_ else 'FAIL'}  {k_}: {m_}")
    return res


def synth(kind, n=60):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for k in range(n):
        if kind == 'hi-grating':          # 0.0 / 0.6 luminance stripes, 6 px period at half HD, drifting
            v = 0.3 + 0.3 * np.cos(2 * np.pi * (xx + 2 * k) / 6)
        elif kind == 'lo-grating':        # the same, peak to trough 0.04
            v = 0.15 + 0.02 * np.cos(2 * np.pi * (xx + 2 * k) / 6)
        elif kind == 'hi-fringes':        # broad stripes, 150 px period at half HD (six pairs across), 0.25 contrast
            v = 0.2 + 0.125 * np.cos(2 * np.pi * (xx * 0.9 + yy * 0.3 + 3 * k) / 150)
        elif kind == 'line':              # one bright line on dark
            v = np.full((H, W), 0.02, np.float32)
            v[270 + int(20 * np.sin(k / 5)):274 + int(20 * np.sin(k / 5)), :] = 0.9
        elif kind == 'strobe':            # the whole screen 0.05 / 0.6 at 10 Hz (3 frames on, 3 off at 30 fps)
            v = np.full((H, W), 0.6 if (k // 3) % 2 else 0.05, np.float32)
        else:
            raise ValueError(kind)
        s = np.where(v <= 0.0031308, v * 12.92, 1.055 * np.clip(v, 0, 1) ** (1 / 2.4) - 0.055)
        g = (np.clip(s, 0, 1) * 255).astype(np.uint8)
        yield np.stack([g, g, g], -1)


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        expect = {'hi-grating': (True, False), 'hi-fringes': (True, False), 'lo-grating': (True, True), 'line': (True, True), 'strobe': (False, True)}
        ok = True
        for kind, (fl, pa) in expect.items():
            r = run(synth(kind), quiet=True)
            good = r['flash'][0] == fl and r['pattern'][0] == pa
            ok &= good
            print(f"  {'ok ' if good else 'BAD'} {kind}: flash {'pass' if r['flash'][0] else 'fail'} (expect {'pass' if fl else 'fail'}), "
                  f"pattern {'pass' if r['pattern'][0] else 'fail'} (expect {'pass' if pa else 'fail'}); {r['pattern'][1]}")
        print('  ->', 'the checks tell the cases apart' if ok else 'a check is blind')
    else:
        run(frames_from(sys.argv[1]))
