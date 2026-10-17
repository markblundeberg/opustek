"""refscope: take a finished mix apart without its source.

Built to study reference tracks for the next piece, and calibrated on our own
first track, Engine, whose stems and score we have (see refscope_calibrate.py).
Needs the analysis environment: pip install -r tools/requirements-analysis.txt
(torch and torchaudio, CPU builds are fine; the separation model's weights
download from torchaudio on first use).

  python tools/refscope/refscope.py separate ref/x.wav    stems, cached
  python tools/refscope/refscope.py analyze ref/x.wav     grid, layers, echoes, pitch, sections

Everything lands in out/refscope/<name>/ (git-ignored). Reference audio and
anything derived from it stays out of git and out of release files."""
import json
import os
import sys

import numpy as np
import soundfile as sf

SR = 44100
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'out', 'refscope')


def name_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def outdir(path):
    d = os.path.join(OUT, name_of(path))
    os.makedirs(d, exist_ok=True)
    return d


def load(path):
    """Stereo float32 at 44.1 kHz, shape (2, n)."""
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.T
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR, res_type='soxr_hq')
    return np.ascontiguousarray(x[:2])


# ---------------------------------------------------------------- separation

STEMS = ['drums', 'bass', 'other', 'vocals']


def separate(path, threads=6):
    """Hybrid Demucs (torchaudio HDEMUCS_HIGH_MUSDB_PLUS), 10 s chunks with
    1 s linear crossfades, as in the torchaudio tutorial. Cached as float16."""
    d = outdir(path)
    cache = os.path.join(d, 'stems.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k].astype(np.float32) for k in STEMS}
    import torch
    import torchaudio
    from torchaudio.transforms import Fade
    torch.set_num_threads(threads)
    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().eval()
    assert bundle.sample_rate == SR and list(model.sources) == STEMS, model.sources

    x = torch.from_numpy(load(path))
    ref = x.mean(0)
    mu, sd = ref.mean(), ref.std()
    mix = ((x - mu) / sd)[None]
    n = mix.shape[-1]
    seg, ov = 10.0, 0.1
    chunk = int(SR * seg * (1 + ov))
    ovf = int(ov * SR * seg)
    fade = Fade(fade_in_len=0, fade_out_len=ovf, fade_shape='linear')
    out = torch.zeros(1, len(STEMS), 2, n)
    start, end = 0, chunk
    while start < n - ovf:
        with torch.no_grad():
            y = model(mix[:, :, start:end])
        y = fade(y)
        out[:, :, :, start:start + y.shape[-1]] += y
        if start == 0:
            fade.fade_in_len = ovf
            start += chunk - ovf
        else:
            start += chunk
        end += chunk
        if end >= n:
            fade.fade_out_len = 0
        print(f'  separated {min(start, n) / SR:6.1f} / {n / SR:.1f} s', flush=True)
    out = (out[0] * sd + mu).numpy()
    stems = {k: out[i] for i, k in enumerate(STEMS)}
    np.savez(cache, **{k: v.astype(np.float16) for k, v in stems.items()})
    # How much of the mix the stems account for: a separation sanity number.
    resid = load(path) - sum(stems.values())
    print(f'  residual {10 * np.log10((resid ** 2).mean() / (load(path) ** 2).mean()):.1f} dB re mix')
    return stems


if __name__ == '__main__':
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == 'separate':
        separate(path)
    elif cmd == 'analyze':
        import refscope_analyze
        refscope_analyze.run(path)
    else:
        sys.exit(__doc__)
