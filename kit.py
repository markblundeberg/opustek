"""Genre-neutral plumbing for a piece: placing hits, envelopes, panning, ducking, mastering, and writing the mix with
its stems and score. No instruments here (those live in the tracks, and in your own code); just what every piece
needs around them. Everything is at the sample rate you pass, mono float64 unless it says otherwise.

    import kit
    buf = np.zeros(N); kit.place(buf, hit, [(t, gain), ...], SR)
    env = kit.adsr(N_note, SR, a=0.005, d=0.2, s=0.6, r=0.3, gate_s=0.5)
    L, R = kit.pan(x, 0.3)                       # -1 left .. +1 right, equal power
    g = kit.duck(kick_times, N, SR, depth=0.6)   # a gain curve: multiply what the kick should push down
    mix = kit.master(np.stack([L, R], 1), SR, lufs=-10.0)
    kit.write_outputs('mypiece', mix, {'kick': kick_as_mixed, ...}, score, SR)

Save stems as they sit in the mix (after their gains and ducking), so one stem can be measured against another; and
keep a raw copy of any part whose pitch or tuning you check (say 'lead_raw'), since gains and filters don't matter
there but effects do.
"""
import json
import os

import numpy as np

import dsp


def place(dst, hit, times, sr, gain=1.0):
    """Add `hit` into `dst` at each (time in s, gain) in `times`, clipped at the end of the buffer."""
    for t, g in times:
        i = int(round(t * sr))
        j = min(len(dst), i + len(hit))
        if 0 <= i < len(dst):
            dst[i:j] += gain * g * hit[:j - i]
    return dst


def adsr(n, sr, a=0.005, d=0.1, s=0.7, r=0.2, gate_s=None):
    """Attack, decay, sustain level, release (seconds; s a level), over n samples. The note is held for gate_s
    seconds (default: until the release must start to end at n)."""
    t = np.arange(n) / sr
    gate = (n / sr - r) if gate_s is None else gate_s
    up_ = np.clip(t / max(a, 1e-6), 0, 1)
    dec = s + (1 - s) * np.exp(-np.clip(t - a, 0, None) / max(d, 1e-6))
    env = np.where(t < a, up_, dec)
    held = env[min(n - 1, int(gate * sr))] if gate * sr < n else env[-1]
    rel = held * np.exp(-np.clip(t - gate, 0, None) / max(r / 4, 1e-6))
    return np.where(t < gate, env, rel)


def pan(x, p):
    """Equal-power pan of a mono signal: p from -1 (left) to +1 (right); p may be an array. Returns (L, R)."""
    th = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return x * np.cos(th), x * np.sin(th)


def duck(times, n, sr, depth=0.6, rel=0.16, att=0.004):
    """A gain curve that dips by `depth` at each time (a 4 ms ramp down, so it doesn't click) and recovers with time
    constant `rel`. Multiply the parts the kick should push down."""
    g = np.ones(n)
    t = np.arange(n) / sr
    for tk in times:
        i = int(tk * sr)
        j = min(n, i + int(6 * rel * sr))
        dt = t[i:j] - tk
        g[i:j] = np.minimum(g[i:j], 1 - depth * np.exp(-dt / rel) * np.clip(dt / att, 0, 1))
    return g


def up(ctrl, tc, n, sr):
    """A control curve sampled at times tc (say 1 kHz) brought to audio rate, n samples."""
    return np.interp(np.arange(n) / sr, tc, ctrl)


def master(mix, sr, lufs=-10.0, ceiling_db=-1.2, hp_hz=15.0):
    """High-pass (not mean subtraction, which leaves a DC offset in silences), then bring to `lufs` through the
    kernel's true-peak limiter at `ceiling_db`. mix is (n, 2). A loud target clamps transients: if your kick loses
    punch, try a lower target before anything else."""
    mix = np.stack([dsp.butter(mix[:, c], hp_hz, sr, 'high', 2) for c in range(2)], 1)
    for _ in range(6):
        mix = dsp.limiter(mix * 10 ** ((lufs - dsp.lufs(mix, sr)) / 20), sr, ceiling_db=ceiling_db, rel_ms=60)
    return mix


def write_outputs(name, mix, stems, score, sr, out='out'):
    """out/<name>.wav (float32), out/<name>_stems.npz (each stem as it sits in the mix), out/<name>_score.json."""
    from scipy.io import wavfile
    os.makedirs(out, exist_ok=True)
    wavfile.write(os.path.join(out, f'{name}.wav'), sr, mix.astype(np.float32))
    np.savez_compressed(os.path.join(out, f'{name}_stems.npz'), sr=sr, **stems)
    with open(os.path.join(out, f'{name}_score.json'), 'w') as fh:
        json.dump(score, fh)
    return os.path.join(out, f'{name}.wav')
