"""refscope analysis: grid, stems, layers (NMF), echoes, pitch, key, sections.

Called from refscope.py. Writes out/refscope/<name>/results.json, report.md,
overview.png and one audio file per layer (layers/<stem>-<k>.ogg) to audition."""
import json
import os
import subprocess

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

from refscope import SR, STEMS, load, outdir, separate

HOP = 512                      # 11.6 ms
NFFT = 2048
FPS = SR / HOP
NMF_K = {'drums': 10, 'bass': 5, 'other': 16, 'vocals': 8}   # over-split on purpose; voices() regroups
BANDS = [(20, 60, 'sub'), (60, 250, 'bass'), (250, 1000, 'low-mid'),
         (1000, 4000, 'mid'), (4000, 10000, 'high'), (10000, 20000, 'air')]
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def mmss(t):
    return f'{int(t // 60)}:{t % 60:04.1f}'


def db(p, floor=1e-12):
    return 10 * np.log10(np.maximum(p, floor))


# ---------------------------------------------------------------- grid

def grid(drums, mix):
    """A rigid beat grid: electronic music holds tempo, so fit one period and
    one phase to the whole track, then report how far real onsets stray."""
    env = librosa.onset.onset_strength(y=drums, sr=SR, hop_length=HOP, fmax=200, n_mels=32)
    if env.max() < 1e-6:
        env = librosa.onset.onset_strength(y=mix, sr=SR, hop_length=HOP)
    # Tempo: a comb over the onset autocorrelation. A rigid tempo puts peaks at
    # every multiple of the beat, so sum the autocorrelation at 1..32 beats and
    # take the best period between 90 and 160 bpm (librosa's single estimate
    # read our 123 bpm track as 127).
    e = env - env.mean()
    ac = signal.correlate(e, e, mode='full', method='fft')[len(e) - 1:]
    ac = ac / ac[0]
    lags = np.arange(len(ac))

    def comb(bpm):
        P = 60 / bpm * FPS
        return np.interp(P * np.arange(1, 33), lags, ac).sum()
    bpms = np.arange(90, 160, 0.02)
    sc = np.array([comb(b) for b in bpms])
    b0 = bpms[int(np.argmax(sc))]
    fine = np.arange(b0 - 0.05, b0 + 0.05, 0.001)
    period = 60 / fine[int(np.argmax([comb(b) for b in fine]))]
    # Phase: maximise onset energy summed on the grid (fine search, 1 ms).
    t_env = np.arange(len(env)) / FPS
    phases = np.arange(0, period, 0.001)
    score = []
    for ph in phases:
        bt = np.arange(ph, t_env[-1], period)
        score.append(np.interp(bt, t_env, env).sum())
    phase = phases[int(np.argmax(score))]
    # Polish both by least squares on the strong onsets near the grid: each
    # onset is assigned its nearest beat k, then t = phase + k * period.
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units='time',
                                    backtrack=False)
    strong = on[np.interp(on, t_env, env) > np.percentile(env, 90)]
    for tol in (0.25, 0.12, 0.06):
        k = np.round((strong - phase) / period)
        keep = np.abs(strong - phase - k * period) < tol * period
        if keep.sum() > 20:
            period, phase = np.polyfit(k[keep], strong[keep], 1)
    phase = phase % period
    n = len(mix) / SR
    beats = np.arange(phase, n, period)
    if phase > 0.5 * period:            # keep a beat before the first one, so the
        beats = np.concatenate([[phase - period], beats])   # intro is covered
    beats = beats[beats >= -period]
    # How rigid: distance from each strong drum onset to the grid.
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units='time')
    dev = (strong - phase + period / 2) % period - period / 2
    return dict(bpm=60 / period, period=period, phase=phase, beats=beats,
                onset_dev_ms=float(np.median(np.abs(dev[np.abs(dev) < 0.03])) * 1000),
                on_grid_share=float(np.mean(np.abs(dev) < 0.03)), env=env)


def downbeat(beats, feats):
    """Bar phase: section changes land on bar lines, so pick the beat-in-bar
    (0..3) where the beat-to-beat feature change is largest."""
    d = np.linalg.norm(np.diff(feats, axis=0), axis=1)
    d = np.concatenate([[0], d])
    score = [d[j::4].mean() for j in range(4)]
    return int(np.argmax(score)), score


# ---------------------------------------------------------------- per-beat features

def beat_index(n_frames, beats):
    t = np.arange(n_frames) / FPS
    return np.clip(np.searchsorted(beats, t, side='right') - 1, 0, len(beats) - 1)


def per_beat(x_frames, bidx, nb):
    """Mean of each beat's frames; x_frames is (..., n_frames)."""
    out = np.zeros(x_frames.shape[:-1] + (nb,))
    cnt = np.bincount(bidx, minlength=nb)
    flat = x_frames.reshape(-1, x_frames.shape[-1])
    o = np.stack([np.bincount(bidx, weights=row, minlength=nb) for row in flat])
    o = o / np.maximum(cnt, 1)
    return o.reshape(x_frames.shape[:-1] + (nb,))


def stem_features(st, bidx, nb, freqs):
    """Level (dB re full scale), side/mid, centroid and band levels per beat."""
    mid = 0.5 * (st[0] + st[1])
    side = 0.5 * (st[0] - st[1])
    M = np.abs(librosa.stft(mid, n_fft=NFFT, hop_length=HOP)) ** 2
    S = np.abs(librosa.stft(side, n_fft=NFFT, hop_length=HOP)) ** 2
    n = min(M.shape[1], len(bidx))
    M, S = M[:, :n], S[:, :n]
    tot = M.sum(0)
    lvl = per_beat(tot[None], bidx[:n], nb)[0]
    width = per_beat(S.sum(0)[None], bidx[:n], nb)[0] / np.maximum(lvl, 1e-12)
    cen = per_beat((M * freqs[:, None]).sum(0)[None], bidx[:n], nb)[0] / np.maximum(lvl, 1e-12)
    bands = np.stack([per_beat(M[(freqs >= lo) & (freqs < hi)].sum(0)[None], bidx[:n], nb)[0]
                      for lo, hi, _ in BANDS])
    norm = 2.0 / NFFT    # rough: so a full-scale sine reads near 0 dB
    return dict(level_db=db(lvl * norm ** 2), width_db=db(width), centroid=cen,
                bands_db=db(bands * norm ** 2))


# ---------------------------------------------------------------- layers

def nmf_layers(st, k, beats, bidx, nb, d, stem):
    """Split one stem into k layers with KL-NMF on its magnitude spectrogram;
    describe each, and write it out (Wiener mask) to audition."""
    from sklearn.decomposition import NMF
    X = [librosa.stft(st[c], n_fft=NFFT, hop_length=HOP) for c in (0, 1)]
    n = min(X[0].shape[1], len(bidx))
    X = [x[:, :n] for x in X]
    V = 0.5 * (np.abs(X[0]) + np.abs(X[1])) + 1e-9
    model = NMF(n_components=k, beta_loss='kullback-leibler', solver='mu', init='nndsvda',
                max_iter=250, random_state=0, tol=1e-4)
    W = model.fit_transform(V)          # (freq, k)
    H = model.components_               # (k, frames)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    order = np.argsort([np.sum(W[:, i] * freqs) / W[:, i].sum() for i in range(k)])
    W, H = W[:, order], H[order]
    WH = W @ H + 1e-9
    os.makedirs(os.path.join(d, 'layers'), exist_ok=True)
    layers = []
    for i in range(k):
        Vi = np.outer(W[:, i], H[i])
        mask = Vi / WH
        pw = (Vi ** 2).sum(0)                       # this layer's power per frame
        lvl = db(per_beat(pw[None], bidx[:n], nb)[0])
        # Audio to audition (mono mid, compressed).
        yL = librosa.istft(X[0] * mask, hop_length=HOP, length=st.shape[1])
        yR = librosa.istft(X[1] * mask, hop_length=HOP, length=st.shape[1])
        lab = f'{stem}-{i}'
        # libsndfile's Vorbis writer segfaults on long files: FLAC, then Opus via ffmpeg.
        fl = os.path.join(d, 'layers', lab + '.flac')
        sf.write(fl, np.clip(np.stack([yL, yR], 1), -1, 1).astype(np.float32), SR, subtype='PCM_16')
        r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', fl, '-c:a', 'libopus', '-b:a', '96k',
                            fl[:-5] + '.opus'])
        assert r.returncode == 0, r
        os.remove(fl)
        layers.append(dict(label=lab, stem=stem, W=W[:, i], H=H[i], level_db=lvl,
                           stereo=(yL, yR)))
    return layers, freqs


def describe(layer, freqs, beats, period, nb):
    """Numbers that say what a layer is: where it sits, how it moves, its
    rhythm within the beat, and when it plays."""
    W, H = layer['W'], layer['H']
    w = W / W.sum()
    cen = float((w * freqs).sum())
    pk = float(freqs[np.argmax(W)])
    cum = np.cumsum(w)
    lo, hi = float(freqs[np.searchsorted(cum, 0.1)]), float(freqs[np.searchsorted(cum, 0.9)])
    # Activity per bar, relative to the layer's own loudest bar.
    lv = layer['level_db']
    bars = lv[: (nb // 4) * 4].reshape(-1, 4)
    bar_db = db(np.mean(10 ** (bars / 10), axis=1))
    rel = bar_db - bar_db.max()
    # Rhythm: mean activation at each 16th of the beat (where in the beat it hits).
    t = np.arange(len(H)) / FPS
    pos = ((t - beats[0]) / period) % 1.0
    act = H > np.percentile(H, 60)
    prof = np.array([H[act & (((pos - j / 4 + 1 / 8) % 1) < 1 / 4)].mean() if np.any(act) else 0
                     for j in range(4)])
    prof = prof / prof.max() if prof.max() > 0 else prof
    # Periodicity of the activation, in beats.
    Hc = H[act.nonzero()[0].min():act.nonzero()[0].max()] if act.any() else H
    Hc = Hc - Hc.mean()
    lags_b = np.array([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4])
    ac0 = np.dot(Hc, Hc) + 1e-12
    per = {f'{l:g}': float(np.dot(Hc[:-int(round(l * period * FPS))], Hc[int(round(l * period * FPS)):]) / ac0)
           for l in lags_b}
    # Percussive or sustained: how spiky the activation is while active.
    a = H[H > np.percentile(H, 50)]
    spiky = float(np.percentile(a, 95) / (np.median(a) + 1e-12)) if len(a) else 0.0
    return dict(centroid=cen, peak=pk, band10=lo, band90=hi, bar_rel_db=rel, bar_db=bar_db,
                beat_profile=prof.tolist(), periodicity=per, spikiness=spiky)


def spans(rel_db, thresh=-15.0, min_bars=2):
    """Active bar spans (bar index start, end exclusive) from per-bar dB re max."""
    on = rel_db > thresh
    # close gaps of one bar, drop blips shorter than min_bars
    on = uniform_filter1d(on.astype(float), 3) > 0.34
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= min_bars:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


# ---------------------------------------------------------------- echoes

def echo_lags(yL, yR, period, active=None, hop=256):
    """Delay times. An echo repeats a sound's own spectrum later and quieter,
    so look for lags where onset spectra recur (cosine similarity of
    log-mel onset vectors), plus the lag between left and right (ping-pong)."""
    fps = SR / hop
    y = 0.5 * (yL + yR)
    Mm = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=hop, n_mels=64) + 1e-10)
    F = np.maximum(0, np.diff(Mm, axis=1))
    strength = F.sum(0)
    lags = np.arange(int(0.04 * fps), int(1.6 * fps))
    idx = np.nonzero(strength > np.percentile(strength, 97))[0]
    idx = idx[idx < F.shape[1] - lags[-1] - 1]
    if len(idx) < 20:
        return None
    A = F[:, idx]
    An = A / (np.linalg.norm(A, axis=0) + 1e-9)
    sim = np.empty(len(lags))
    ratio = np.empty(len(lags))
    for j, L in enumerate(lags):
        B = F[:, idx + L]
        Bn = B / (np.linalg.norm(B, axis=0) + 1e-9)
        sim[j] = np.mean((An * Bn).sum(0))
        ratio[j] = np.median(B.sum(0) / (A.sum(0) + 1e-9))
    base = signal.medfilt(sim, 31)
    excess = sim - base
    pk, _ = signal.find_peaks(excess, distance=int(0.03 * fps), prominence=0.01)
    pk = pk[np.argsort(excess[pk])[::-1][:4]]
    peaks = [dict(lag_s=float(lags[p] / fps), lag_beats=float(lags[p] / fps / period),
                  excess=float(excess[p]), sim=float(sim[p]), level_ratio_db=float(10 * np.log10(max(ratio[p], 1e-6))))
             for p in sorted(pk)]
    # Ping-pong: cross-correlate left and right onset envelopes.
    def env(ch):
        m = librosa.onset.onset_strength(y=ch, sr=SR, hop_length=hop)
        return (m - m.mean()) / (m.std() + 1e-9)
    eL, eR = env(yL), env(yR)
    maxl = int(1.2 * fps)
    xc = signal.correlate(eL, eR, mode='full', method='fft')
    mid = len(eR) - 1
    seg = xc[mid - maxl: mid + maxl + 1] / len(eL)
    lagx = np.arange(-maxl, maxl + 1) / fps
    # Repetition in the music is symmetric in lag; a ping-pong is not (one side
    # answers the other). So read the asymmetry seg(+t) - seg(-t). Positive:
    # the right channel leads (left answers it).
    pos = np.arange(int(0.04 * fps), maxl + 1)
    asym = seg[maxl + pos] - seg[maxl - pos]
    j = int(np.argmax(np.abs(asym)))
    return dict(peaks=peaks, curve_lag=(lags / fps).tolist(), curve_sim=sim.tolist(),
                pp_lag_s=float(pos[j] / fps), pp_lag_beats=float(pos[j] / fps / period),
                pp_asym=float(asym[j]), pp_leader='right' if asym[j] > 0 else 'left',
                lr_corr0=float(seg[maxl]), pp_asym_median=float(np.median(np.abs(asym))))


# ---------------------------------------------------------------- local sounds

def local_pingpong(stems, bars, period, win=8, hop=256):
    """Ping-pong per window and stem: the asymmetry of the L/R onset
    correlation, lags up to a beat and a bit, against that window's median."""
    fps = SR / hop
    out = []
    for st in ('other', 'vocals', 'bass'):
        x = stems[st]
        for b0 in range(0, len(bars) - 1, win):
            t0, t1 = bars[b0], bars[min(b0 + win, len(bars) - 1)]
            a, b = max(0, int(t0 * SR)), int(t1 * SR)
            if b - a < 4 * SR or np.mean(x[:, a:b] ** 2) < 1e-7:
                continue
            def env(ch):
                m = librosa.onset.onset_strength(y=ch, sr=SR, hop_length=hop)
                return (m - m.mean()) / (m.std() + 1e-9)
            eL, eR = env(x[0, a:b]), env(x[1, a:b])
            maxl = int(1.1 * period * fps)
            xc = signal.correlate(eL, eR, mode='full', method='fft') / len(eL)
            mid = len(eR) - 1
            pos = np.arange(int(0.04 * fps), maxl + 1)
            asym = xc[mid + pos] - xc[mid - pos]
            j = int(np.argmax(np.abs(asym)))
            out.append(dict(stem=st, bar0=b0 + 1, bar1=min(b0 + win, len(bars) - 1),
                            lag_s=float(pos[j] / fps), lag_beats=float(pos[j] / fps / period),
                            asym=float(asym[j]), leader='right' if asym[j] > 0 else 'left',
                            snr=float(abs(asym[j]) / (np.median(np.abs(asym)) + 1e-9)),
                            corr0=float(xc[mid])))
    return out


# ---------------------------------------------------------------- band events

THIRD_OCT = 1000 * 2 ** (np.arange(-15, 13) / 3)        # 31.5 Hz .. 16 kHz


def band_levels(y, bars, nfft=4096, hop=1024):
    """Per-bar power (dB) in third-octave bands, shape (bands, bars)."""
    P = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=nfft)
    t = np.arange(P.shape[1]) * hop / SR
    bi = np.clip(np.searchsorted(bars, t, side='right') - 1, 0, len(bars) - 1)
    out = np.zeros((len(THIRD_OCT), len(bars)))
    for i, fc in enumerate(THIRD_OCT):
        m = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        e = P[m].sum(0)
        out[i] = np.bincount(bi, weights=e, minlength=len(bars)) / np.maximum(np.bincount(bi, minlength=len(bars)), 1)
    return db(out)


def band_events(stems, mix, bars, rise=6.0, audible=-20.0):
    """What enters and leaves, and where in the spectrum: per stem and third-
    octave band, a bar whose level rises >= `rise` dB over both previous bars
    (or falls as much below both next bars), counted only where the stem is
    within `audible` dB of the whole mix in that band (a crude masking proxy).
    Neighbouring bands that move together are merged into one event."""
    M = band_levels(mix.mean(0), bars)
    events = []
    for st in STEMS:
        E = band_levels(stems[st].mean(0), bars)
        nb = E.shape[1]
        for b in range(1, nb):
            prev = E[:, max(0, b - 2):b].max(1)
            up = (E[:, b] - prev >= rise) & (E[:, b] - M[:, b] >= audible)
            nxt = E[:, b:min(nb, b + 2)].max(1)
            down = (E[:, b - 1] - nxt >= rise) & (E[:, b - 1] - M[:, b - 1] >= audible)
            for kind, mask, delta in (('enters', up, E[:, b] - prev), ('leaves', down, E[:, b - 1] - nxt)):
                idx = np.nonzero(mask)[0]
                if not len(idx):
                    continue
                groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
                for g in groups:
                    events.append(dict(bar=b + 1, kind=kind, stem=st, lo=float(THIRD_OCT[g[0]] / 2 ** (1 / 6)),
                                       hi=float(THIRD_OCT[g[-1]] * 2 ** (1 / 6)), nbands=len(g),
                                       change_db=float(delta[g].max()),
                                       re_mix_db=float((E[g, b] - M[g, b]).max() if kind == 'enters'
                                                       else (E[g, b - 1] - M[g, b - 1]).max())))
    return events, M


def tail_echo(y, t_end, period, dur=3.0, hop=128):
    """After the dry part stops, an echo keeps repeating at a fixed interval,
    decaying; a rhythm doesn't. Peaks of the onset envelope in the `dur`
    seconds after t_end, with their spacing and level."""
    a, b = int(t_end * SR), int((t_end + dur) * SR)
    seg = y[a:b]
    if len(seg) < SR:
        return None
    rms = librosa.feature.rms(y=seg, frame_length=1024, hop_length=hop)[0]
    e = db(rms ** 2)
    t = np.arange(len(e)) * hop / SR
    pk, pr = signal.find_peaks(e, prominence=3, distance=int(0.06 * SR / hop))
    if len(pk) < 2:
        return dict(peaks_s=t[pk].tolist(), peaks_db=e[pk].tolist())
    gaps = np.diff(t[pk])
    return dict(peaks_s=t[pk].tolist(), peaks_db=e[pk].tolist(), gaps_s=gaps.tolist(),
                gaps_beats=(gaps / period).tolist())


# ---------------------------------------------------------------- voices

def voices(layers, thresh=0.8):
    """NMF over-splits an instrument into several layers that rise and fall
    together, so join layers (within a stem) whose per-bar activity correlates
    above thresh (average linkage)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    groups = []
    for stem in STEMS:
        ls = [l for l in layers if l['stem'] == stem]
        if len(ls) == 1:
            groups.append([ls[0]['label']])
            continue
        A = np.array([np.maximum(l['bar_rel_db'], -30) for l in ls])
        C = np.corrcoef(A)
        D = 1 - C[np.triu_indices(len(ls), 1)]
        Z = linkage(np.maximum(D, 0), 'average')
        lab = fcluster(Z, 1 - thresh, 'distance')
        for g in sorted(set(lab)):
            groups.append([ls[i]['label'] for i in range(len(ls)) if lab[i] == g])
    return groups


# ---------------------------------------------------------------- pitch, key

def pitch_track(y, fmin, fmax):
    f0, vf, vp = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR, frame_length=4096, hop_length=HOP)
    return f0, vf, vp


KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


MODES = {'Ionian': [0, 2, 4, 5, 7, 9, 11], 'Dorian': [0, 2, 3, 5, 7, 9, 10], 'Phrygian': [0, 1, 3, 5, 7, 8, 10],
         'Lydian': [0, 2, 4, 6, 7, 9, 11], 'Mixolydian': [0, 2, 4, 5, 7, 9, 10], 'Aeolian': [0, 2, 3, 5, 7, 8, 10],
         'Locrian': [0, 1, 3, 5, 6, 8, 10]}


def mode_of(chroma_vec, tonic):
    """Given the tonic, score each church mode by the chroma mass inside its
    seven notes minus the mass outside."""
    c = np.roll(np.asarray(chroma_vec) / np.sum(chroma_vec), -tonic)
    sc = {m: c[ix].sum() - (1 - c[ix].sum()) for m, ix in MODES.items()}
    return sorted(((v, m) for m, v in sc.items()), reverse=True)[:3], c.tolist()


def key_of(chroma_vec):
    best = []
    for r in range(12):
        for prof, mode in ((KS_MAJ, 'major'), (KS_MIN, 'minor')):
            best.append((np.corrcoef(chroma_vec, np.roll(prof, r))[0, 1], f'{NOTES[r]} {mode}'))
    best.sort(reverse=True)
    return best[:3]


# ---------------------------------------------------------------- sections

def novelty(feats, kernel_beats=32):
    Z = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    S = Z @ Z.T
    L = kernel_beats // 2
    g = signal.windows.gaussian(2 * L, L / 2)
    G = np.outer(g, g)
    K = G * np.block([[np.ones((L, L)), -np.ones((L, L))], [-np.ones((L, L)), np.ones((L, L))]])
    n = len(S)
    nov = np.zeros(n)
    Sp = np.pad(S, L, mode='edge')
    for i in range(n):
        nov[i] = (Sp[i:i + 2 * L, i:i + 2 * L] * K).sum()
    return np.maximum(nov, 0), S


# ---------------------------------------------------------------- run

def run(path, marks=None):
    d = outdir(path)
    stems = separate(path)
    mix = load(path)
    n = mix.shape[1]
    nsec = n / SR
    mono = {k: v.mean(0) for k, v in stems.items()}
    g = grid(mono['drums'], mix.mean(0))
    beats, period = g['beats'], g['period']
    nfr = 1 + n // HOP
    bidx = beat_index(nfr, beats)
    nb = len(beats)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    print(f'grid: {g["bpm"]:.2f} bpm, phase {g["phase"]:.3f} s; {g["on_grid_share"]:.0%} of strong onsets within 30 ms '
          f'of the beat (median {g["onset_dev_ms"]:.1f} ms); the rest fall between beats')

    feats = {k: stem_features(v, bidx, nb, freqs) for k, v in list(stems.items()) + [('mix', mix)]}

    layers, descs = [], []
    for stem in STEMS:
        L, _ = nmf_layers(stems[stem], NMF_K[stem], beats, bidx, nb, d, stem)
        for lay in L:
            de = describe(lay, freqs, beats, period, nb)
            lay.update(de)
            layers.append(lay)
        print(f'layers: {stem} done')

    # Bar phase from all per-beat features together.
    F = np.column_stack([feats[s]['level_db'] for s in STEMS] + [l['level_db'] for l in layers])
    F = np.maximum(F, F.max(0) - 60)
    db0, dbs = downbeat(beats, F)
    bar0 = beats[db0] if beats[db0] >= 0 else beats[db0 + 4]
    srt = sorted(dbs)
    if srt[-1] / srt[-2] < 1.03:
        # Ambiguous (the reference: scores within 1%). Tracks made for DJs start on
        # a downbeat, so take the first strong kick-band onset as bar 1.
        kenv = librosa.onset.onset_strength(y=mono['drums'], sr=SR, hop_length=HOP, fmax=120, n_mels=16)
        kon = librosa.onset.onset_detect(onset_envelope=kenv, sr=SR, hop_length=HOP, units='time')
        kon = kon[np.interp(kon, np.arange(len(kenv)) / FPS, kenv) > np.percentile(kenv, 95)]
        k0 = kon[0]
        bar0 = beats[np.argmin(np.abs(beats - k0))]
        print(f'downbeat ambiguous; bar 1 from the first kick at {k0:.3f} s')
    # Bars from the first downbeat, extended back to cover any pickup.
    first = bar0 - 4 * period * np.floor(bar0 / (4 * period) + 1e-6)
    bars = np.arange(first, nsec, 4 * period)
    print(f'downbeat: beat {db0} of 4 (scores {np.round(dbs, 2).tolist()}), bar 1 at {first:.3f} s')
    # Re-derive per-bar activity on this bar grid for every layer and stem.
    beat_in_bar = np.round((beats - first) / period).astype(int)

    def bar_db(level_db):
        p = 10 ** (level_db / 10)
        nbar = len(bars)
        acc, cnt = np.zeros(nbar), np.zeros(nbar)
        bi = np.clip(beat_in_bar // 4, 0, nbar - 1)
        np.add.at(acc, bi, p)
        np.add.at(cnt, bi, 1)
        return db(acc / np.maximum(cnt, 1))

    for lay in layers:
        lay['bar_db'] = bar_db(lay['level_db'])
        lay['bar_rel_db'] = lay['bar_db'] - lay['bar_db'].max()
        lay['spans'] = spans(lay['bar_rel_db'])
    stem_bar = {s: bar_db(feats[s]['level_db']) for s in STEMS + ['mix']}

    # Echoes, per layer that carries real energy.
    for lay in layers:
        lay['echo'] = echo_lags(lay['stereo'][0], lay['stereo'][1], period)
    for s in STEMS:
        feats[s]['echo'] = echo_lags(stems[s][0], stems[s][1], period)
    print('echoes done')

    # Pitch: bass stem and the 'other' stem (the lead usually lives there).
    f0b, vb, _ = pitch_track(mono['bass'], 25, 700)
    f0o, vo, po = pitch_track(mono['other'], 70, 1500)
    f0v, vv, pv = pitch_track(mono['vocals'], 70, 1500)
    print('pitch done')

    # Key: chroma of the pitched stems, whole track.
    harm = mono['bass'] + mono['other'] + mono['vocals']
    C = librosa.feature.chroma_cqt(y=harm, sr=SR, hop_length=HOP)[:, :nfr]
    keys = key_of(C.mean(1))
    tonic = NOTES.index(keys[0][1].split()[0])
    modes, rel_chroma = mode_of(C.mean(1), tonic)
    Cb = per_beat(C, bidx[:C.shape[1]], nb)

    # Sections: novelty over per-beat stem levels, bands, layer levels and chroma.
    feat_mat = np.column_stack([F, feats['mix']['bands_db'].T, Cb.T])
    nov, SSM = novelty(feat_mat, 32)
    pk, _ = signal.find_peaks(nov, distance=16, prominence=np.percentile(nov, 75) * 0.5)
    bounds = sorted(set([0] + [int(np.floor((beats[p] - first) / (4 * period) + 0.5)) for p in pk]))

    res = dict(file=path, seconds=nsec, bpm=g['bpm'], period=period, phase=g['phase'],
               onset_dev_ms=g['onset_dev_ms'], on_grid_share=g['on_grid_share'], bar1=first, bar_seconds=4 * period, n_bars=len(bars),
               keys=[(float(c), k) for c, k in keys], chroma=C.mean(1).tolist(),
               modes=[(float(v), m) for v, m in modes], chroma_from_tonic=rel_chroma,
               section_bars=bounds, novelty_peaks_s=[float(beats[p]) for p in pk],
               stems={s: dict(bar_db=stem_bar[s].tolist(),
                              width_db=float(np.median(feats[s]['width_db'])) if s != 'mix' else None,
                              echo=feats[s].get('echo')) for s in STEMS + ['mix']},
               layers=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in lay.items()
                        if k not in ('W', 'H', 'stereo', 'level_db')} for lay in layers])
    res['pitch'] = dict(
        bass=dict(f0=np.nan_to_num(f0b).tolist(), voiced=vb.tolist()),
        other=dict(f0=np.nan_to_num(f0o).tolist(), voiced=vo.tolist(), prob=po.tolist()),
        vocals=dict(f0=np.nan_to_num(f0v).tolist(), voiced=vv.tolist(), prob=pv.tolist()))
    res['mix_bands_bar'] = {name: bar_db(feats['mix']['bands_db'][i]).tolist() for i, (_, _, name) in enumerate(BANDS)}
    res['stem_centroid_bar'] = {s: [float(x) for x in np.array(
        [np.median(feats[s]['centroid'][(beat_in_bar // 4) == b]) if np.any((beat_in_bar // 4) == b) else 0
         for b in range(len(bars))])] for s in STEMS}
    gW = {st: [l['W'] for l in layers if l['stem'] == st] for st in STEMS}
    res['local_pingpong'] = local_pingpong(stems, bars, period)
    ev, M = band_events(stems, mix, bars)
    res['band_events'] = ev
    res['mix_thirdoct_bar_db'] = M.tolist()
    print(f"band events: {len(ev)}")
    # Voices and their per-bar levels (power sum of their layers).
    vg = voices(layers)
    byl = {l['label']: l for l in layers}
    res['voices'] = []
    for gi, g in enumerate(vg):
        p = sum(10 ** (np.asarray(byl[x]['bar_db']) / 10) for x in g)
        bdb = db(p)
        res['voices'].append(dict(id=gi, layers=g, stem=byl[g[0]]['stem'], bar_db=bdb.tolist(),
                                  spans=spans(bdb - bdb.max())))
    with open(os.path.join(d, 'results.json'), 'w') as f:
        json.dump(res, f)
    np.savez(os.path.join(d, 'layers_WH.npz'),
             **{lay['label'] + '_W': lay['W'] for lay in layers},
             **{lay['label'] + '_H': lay['H'] for lay in layers})
    print('wrote', os.path.join(d, 'results.json'))
    return res
