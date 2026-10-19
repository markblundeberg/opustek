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
    for i in range(6):                  # gain, limit, measure again: the limiter takes back some of the gain
        err = lufs - dsp.lufs(mix, sr)
        if i and abs(err) < 0.02:       # on target (and limited at least once): each pass costs seconds
            break
        mix = dsp.limiter(mix * 10 ** (err / 20), sr, ceiling_db=ceiling_db, rel_ms=60)
    return mix


def write_outputs(name, mix, stems, score, sr, out='out'):
    """out/<name>.wav (float32), out/<name>_stems.npz (each stem as it sits in the mix), out/<name>_score.json.
    Read the stems back with load_stems(), not np.load: an npz reads (and unzips) a stem again on every lookup."""
    from scipy.io import wavfile
    os.makedirs(out, exist_ok=True)
    wavfile.write(os.path.join(out, f'{name}.wav'), sr, mix.astype(np.float32))
    np.savez(os.path.join(out, f'{name}_stems.npz'), sr=sr, **stems)     # uncompressed: compressing took 15 s a render
    with open(os.path.join(out, f'{name}_score.json'), 'w') as fh:
        json.dump(score, fh)
    return os.path.join(out, f'{name}.wav')


def load_stems(path):
    """Every stem of an out/<name>_stems.npz, read once into a dict."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def smooth(x, k):
    """A centred moving average over k samples, the same as np.convolve(x, np.ones(k) / k, 'same') to about 1e-15.
    Use this for long windows: np.convolve takes k multiply-adds per sample (15 s for a 0.25 s window over a 3-minute
    song), and its long dot products run on several threads, so the last bits change with the number of cores."""
    from scipy import signal
    return signal.oaconvolve(np.asarray(x, float), np.ones(k) / k, mode='same')


def _available_gb():
    """Memory this process may still use: the lesser of the system's available memory and its cgroup's headroom
    (under systemd-run -p MemoryMax=..., the cap is what counts)."""
    avail = float('inf')
    try:
        with open('/proc/meminfo') as fh:
            avail = next(int(l.split()[1]) for l in fh if l.startswith('MemAvailable')) / 2 ** 20
        with open('/proc/self/cgroup') as fh:
            cg = '/sys/fs/cgroup' + fh.read().strip().split('::')[-1]
        while cg.startswith('/sys/fs/cgroup') and len(cg) > len('/sys/fs/cgroup'):
            try:
                cap = open(os.path.join(cg, 'memory.max')).read().strip()
                if cap != 'max':
                    used = int(open(os.path.join(cg, 'memory.current')).read())
                    avail = min(avail, (int(cap) - used) / 2 ** 30)
            except OSError:
                pass
            cg = os.path.dirname(cg)
    except (OSError, StopIteration):
        pass
    return avail


def _selftest_one(job):
    brk, check_name, argv, path, check, keep = job
    if argv is not None:
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')    # the jobs are the parallelism
        import subprocess
        r = subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
        if r.returncode:
            return brk, check_name, None, r.stderr.strip().splitlines()[-1:] or ['(no message)']
    try:
        res = check(path)
    finally:
        if argv is not None and not keep:
            for f in (path, path.replace('.wav', '_stems.npz'), path.replace('.wav', '_score.json')):
                if os.path.exists(f):
                    os.remove(f)
    return brk, check_name, res, None


def selftest(faults, render_argv, broken_path, check, clean_path, mem_gb, jobs=None, keep=False):
    """Show each check failing on the fault written for it, with the faults rendered and checked in parallel.

    faults: {fault: the check it should trip}; render_argv(fault): the command that renders it; broken_path(fault):
    the WAV that command writes; check(path): {check: (passed, message)} for a render; clean_path: the clean render,
    whose checks must all pass; mem_gb: one render's peak memory plus its check's, which sets how many run at once
    (jobs overrides). Broken renders are deleted once checked, unless keep. Prints one line per fault, returns ok."""
    import multiprocessing as mp
    if jobs is None:
        jobs = int(_available_gb() * 0.85 // mem_gb)
    jobs = max(1, min(jobs, os.cpu_count() or 1, len(faults) + 1))
    print(f'  {len(faults)} faults, {jobs} at a time ({mem_gb:g} GB each)', flush=True)
    work = [(None, None, None, clean_path, check, keep)]
    work += [(brk, c, render_argv(brk), broken_path(brk), check, keep) for brk, c in faults.items()]
    with mp.get_context('fork').Pool(jobs) as pool:
        done = {brk: (c, res, err) for brk, c, res, err in pool.imap_unordered(_selftest_one, work)}
    ok = True
    _, clean, _ = done[None]
    bad = [k for k, (p, _) in clean.items() if not p]
    print('  clean render:', 'all checks pass' if not bad else f'FAILING {bad}')
    ok &= not bad
    for brk in faults:
        c, r, err = done[brk]
        if r is None:
            print(f'  BAD {brk}: the broken render failed to run: {err[0]}'); ok = False; continue
        caught = not r[c][0]
        others = [k for k, (p, _) in r.items() if not p and k != c]
        ok &= caught
        print(f"  {'ok ' if caught else 'BAD'} {brk}: '{c}' {'FAILS' if caught else 'passes'} ({r[c][1]})"
              + (f'; also failing: {", ".join(others)}' if others else ''))
    print('  ->', 'each fault trips its check' if ok else 'a check is blind (or the clean render fails)')
    return ok
