"""ctypes wrapper for dsp.c, plus the numpy-side helpers (envelopes, filters,
loudness). Rebuilds libdsp.so when dsp.c is newer."""
import ctypes
import os
import subprocess

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, LIB = os.path.join(HERE, 'dsp.c'), os.path.join(HERE, 'libdsp.so')
if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
    subprocess.check_call(['gcc', '-O2', '-shared', '-fPIC', '-o', LIB, SRC, '-lm'])
_lib = ctypes.CDLL(LIB)
_P = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
_I, _D = ctypes.c_int, ctypes.c_double
_lib.saw.argtypes = [_P, _P, _I, _D, _D, _I]
_lib.ladder.argtypes = [_P, _P, _I, _P, _D, _D, _D]
_lib.svf_bp.argtypes = [_P, _P, _I, _P, _D, _D]
_lib.pingpong.argtypes = [_P, _P, _P, _P, _I, _I, _D, _D]
_lib.fdn.argtypes = [_P, _P, _P, _P, _I, _D, _D, _D, _D]
_lib.release_smooth.argtypes = [_P, _P, _I, _D]
_lib.tapedelay.argtypes = [_P, _P, _I, _D, _D, _D, _D, _D, _D, _D]
_lib.syncsaw.argtypes = [_P, _P, _P, _I, _D, _D]
_lib.phaser.argtypes = [_P, _P, _I, _D, _D, _D, _D, _I, _D, _D]
_lib.ks.argtypes = [_P, _P, _I, _D, _D, _D, _D, _I]


def _c(x):
    return np.ascontiguousarray(x, dtype=np.float64)


def saw(freq, sr, phase=0.0, blep=True):
    freq = _c(freq)
    out = np.empty_like(freq)
    _lib.saw(freq, out, len(freq), sr, phase, int(blep))
    return out


def ladder(x, fc, sr, res=0.3, drive=1.0):
    x = _c(x)
    fc = _c(np.broadcast_to(fc, x.shape))
    out = np.empty_like(x)
    _lib.ladder(x, out, len(x), fc, res, drive, sr)
    return out


def svf_bp(x, fc, sr, q):
    x = _c(x)
    fc = _c(np.broadcast_to(fc, x.shape))
    out = np.empty_like(x)
    _lib.svf_bp(x, out, len(x), fc, q, sr)
    return out


def pingpong(L, R, delay_s, sr, fb=0.45, damp_hz=3500.0):
    L, R = _c(L), _c(R)
    oL, oR = np.empty_like(L), np.empty_like(R)
    damp = 1 - np.exp(-2 * np.pi * damp_hz / sr)
    _lib.pingpong(L, R, oL, oR, len(L), int(round(delay_s * sr)), fb, damp)
    return oL, oR


def reverb(L, R, sr, rt60=3.5, damp_hz=5000.0, mod=6.0):
    L, R = _c(L), _c(R)
    oL, oR = np.empty_like(L), np.empty_like(R)
    _lib.fdn(L, R, oL, oR, len(L), sr, rt60, damp_hz, mod)
    return oL, oR


def tapedelay(x, sr, delay_s, wow_ms=3.0, wow_hz=0.7, fb=0.55, damp_hz=3000.0, phase=0.0):
    x = _c(x)
    out = np.empty_like(x)
    _lib.tapedelay(x, out, len(x), sr, delay_s, wow_ms, wow_hz, fb, damp_hz, phase)
    return out


def syncsaw(freq, ratio, sr, phase=0.0):
    freq = _c(freq)
    ratio = _c(np.broadcast_to(ratio, freq.shape))
    out = np.empty_like(freq)
    _lib.syncsaw(freq, ratio, out, len(freq), sr, phase)
    return out


def phaser(x, sr, rate=0.08, fmin=200.0, fmax=3000.0, stages=6, fb=0.45, phase=0.0):
    x = _c(x)
    out = np.empty_like(x)
    _lib.phaser(x, out, len(x), sr, rate, fmin, fmax, stages, fb, phase)
    return out


def ks(exc, sr, freq, fb=0.996, bright=0.0, int_delay=False):
    exc = _c(exc)
    out = np.empty_like(exc)
    _lib.ks(exc, out, len(exc), sr, freq, fb, bright, int(int_delay))
    return out


def butter(x, fc, sr, kind, order=2):
    sos = signal.butter(order, fc, btype=kind, fs=sr, output='sos')
    return signal.sosfilt(sos, x)


def limiter(x, sr, ceiling_db=-1.5, look_ms=5.0, rel_ms=120.0):
    """Lookahead peak limiter on a stereo array (n, 2)."""
    c = 10 ** (ceiling_db / 20)
    # true-peak aware: detect on a 4x oversampled copy, fold back per sample
    up4 = signal.resample_poly(x, 4, 1, axis=0)[:len(x) * 4]
    peak = np.abs(up4).max(axis=1).reshape(len(x), 4).max(axis=1)
    g = np.minimum(1.0, c / np.maximum(peak, 1e-12))
    la = max(1, int(look_ms * 1e-3 * sr))
    g = _ndmin(g, la)
    gs = np.empty_like(g)
    _lib.release_smooth(_c(g), gs, len(g), 1 - np.exp(-1 / (rel_ms * 1e-3 * sr)))
    gs = np.convolve(gs, np.ones(la) / la, mode='same')  # soften the attack
    gs = np.minimum(gs, g)
    return x * gs[:, None]


def _ndmin(g, w):
    from scipy.ndimage import minimum_filter1d
    # window looks ahead: gain at i covers peaks in [i, i + w)
    return minimum_filter1d(g, size=w, origin=-(w // 2))


# --- loudness (ITU-R BS.1770-4) and true peak ------------------------------

def lufs(x, sr):
    """Integrated loudness of a stereo array (n, 2), gated."""
    assert sr == 48000, 'K-weighting coefficients below are for 48 kHz'
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=0), axis=0)
    blk, hop = int(0.4 * sr), int(0.1 * sr)
    z = []
    for s in range(0, len(y) - blk + 1, hop):
        z.append(np.mean(y[s:s + blk] ** 2, axis=0).sum())
    z = np.array(z)
    l = -0.691 + 10 * np.log10(np.maximum(z, 1e-20))
    z1 = z[l > -70]
    rel = -0.691 + 10 * np.log10(z1.mean()) - 10
    z2 = z[l > rel]
    return -0.691 + 10 * np.log10(z2.mean())


def true_peak_db(x):
    up = signal.resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.max(np.abs(up)) + 1e-12)
