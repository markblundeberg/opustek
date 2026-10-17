"""refscope probes: close measurements of one part over one stretch.

  python tools/refscope/refscope_probe.py <audio> <stem> <t0> <t1> [fmin fmax]

Prints pitch (notes, glides, excursions, loop length), amplitude modulation,
timbre (brightness vs pitch, spectral slope, odd/even harmonics, vibrato) and
left/right timing. Uses the cached stems from refscope.py separate."""
import json
import os
import sys

import librosa
import numpy as np
from scipy import signal

from refscope import SR, outdir, separate
from refscope_analyze import NOTES, db, mmss


def note(m):
    m = int(round(m))
    return f'{NOTES[m % 12]}{m // 12 - 1}'


def probe(path, stem, t0, t1, fmin=25, fmax=800):
    d = outdir(path)
    res = json.load(open(os.path.join(d, 'results.json')))
    beat = res['period']
    x = separate(path)[stem][:, int(t0 * SR):int(t1 * SR)]
    y = x.mean(0)
    hop = 256
    fps = SR / hop
    out = {}
    print(f'== {stem} {mmss(t0)}-{mmss(t1)} ({(t1 - t0) / beat:.0f} beats at {60 / beat:.2f} bpm)')

    # --- pitch
    # Calibrated on Engine's engine voice: frames of 8192 and probability > 0.1 pitch
    # 42% of frames with the right pitch class on all of them (median 5 cents),
    # but pYIN chose the subharmonic on 95%; the octave check below fixes that.
    f0, vf, vp = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR, frame_length=8192, hop_length=hop)
    ok = vf & (vp > 0.1) & np.isfinite(f0)
    S8 = np.abs(librosa.stft(y, n_fft=8192, hop_length=hop)) ** 2
    fr8 = librosa.fft_frequencies(sr=SR, n_fft=8192)
    idx = np.nonzero(ok[:S8.shape[1]])[0]
    if len(idx) > 20:
        bins = lambda f: np.clip(np.round(f / (SR / 8192)).astype(int), 0, len(fr8) - 1)
        a = S8[bins(f0[idx]), idx]
        b = S8[bins(2 * f0[idx]), idx]
        oc = np.median(10 * np.log10((a + 1e-12) / (b + 1e-12)))
        print(f'octave check: energy at the estimated f0 is {oc:+.1f} dB re its octave', end='')
        if oc < -6:
            f0 = f0 * 2
            print(' -> estimate was a subharmonic; moved up an octave')
        else:
            print()
    m = np.where(ok, 69 + 12 * np.log2(np.where(ok, f0, 440) / 440), np.nan)
    print(f'pitched {ok.mean():.0%} of frames')
    if ok.mean() > 0.2:
        mm = m[ok]
        q = np.percentile(mm, [1, 5, 50, 95, 99])
        print(f'range: {note(q[1])}..{note(q[3])} (1-99%: {note(q[0])}..{note(q[4])}); median {note(q[2])}')
        hist = np.bincount(np.round(mm).astype(int) - 12, minlength=128)
        top = np.argsort(hist)[::-1][:8]
        print('most held notes:', ', '.join(f'{note(i + 12)} {hist[i] / hist.sum():.0%}' for i in top if hist[i] > 0.02 * hist.sum()))
        pcs = np.bincount(np.round(mm).astype(int) % 12, minlength=12)
        print('pitch classes:', ' '.join(f'{NOTES[i]}:{pcs[i] / pcs.sum():.2f}' for i in np.argsort(pcs)[::-1] if pcs[i] > 0.01 * pcs.sum()))
        # glides: slope over 30 ms where pitched
        sm = np.copy(m)
        k = int(0.03 * fps)
        slope = (sm[k:] - sm[:-k]) / 0.03
        gl = np.abs(slope[np.isfinite(slope)])
        print(f'pitch slope while pitched: median {np.median(gl):.1f} st/s; 90th pct {np.percentile(gl, 90):.0f}; '
              f'99th {np.percentile(gl, 99):.0f} st/s; {np.mean(gl > 20):.0%} of frames gliding faster than 20 st/s')
        # excursions ("revs"): runs more than 7 semitones above the local floor
        floor = np.nanpercentile(mm, 20)
        up = ok & (m > floor + 7)
        runs = np.split(np.nonzero(up)[0], np.nonzero(np.diff(np.nonzero(up)[0]) > int(0.08 * fps))[0] + 1)
        runs = [r for r in runs if len(r) > int(0.05 * fps)]
        if runs:
            peaks = [np.nanmax(m[r]) for r in runs]
            durs = [len(r) / fps for r in runs]
            starts = np.array([r[0] / fps for r in runs])
            print(f'excursions >7 st above the floor ({note(floor)}): {len(runs)}, i.e. one per {(t1 - t0) / len(runs) / beat:.1f} beats; '
                  f'peak median {note(np.median(peaks))}, max {note(max(peaks))}; duration median {np.median(durs) * 1000:.0f} ms')
            # where in the bar they start
            pos = ((t0 + starts - res['bar1']) / beat) % 4
            print('  excursion starts by beat-in-bar (16ths):', np.bincount((pos * 4).astype(int) % 16, minlength=16).tolist())
            out['excursions'] = dict(n=len(runs), peaks=[float(p) for p in peaks], starts=(t0 + starts).tolist(), durs=durs)
        # loop length: autocorrelation of the pitch contour (filled) at bar multiples
        mf = np.where(ok, m, np.nanmedian(mm))
        mf = mf - mf.mean()
        best = []
        for bars in (1, 2, 4, 8, 16):
            L = int(round(bars * 4 * beat * fps))
            if L < len(mf) // 2:
                best.append((float(np.corrcoef(mf[:-L], mf[L:])[0, 1]), bars))
        print('pitch-contour self-similarity at a lag of N bars:', ', '.join(f'{b}: {c:.2f}' for c, b in best))
    out['f0'] = np.nan_to_num(f0).tolist()

    # --- amplitude modulation
    env = np.abs(signal.hilbert(signal.sosfilt(signal.butter(4, [30, 3000], 'band', fs=SR, output='sos'), y)))
    envd = signal.decimate(signal.decimate(env, 10), 10)       # 441 Hz
    fs2 = SR / 100
    e = np.log(np.maximum(np.abs(envd), 1e-6))
    e = e - uniform(e, int(fs2 * 0.5))                           # remove the slow (note-level) envelope
    F, Pxx = signal.welch(e, fs=fs2, nperseg=int(fs2 * 4))
    band = (F > 2) & (F < 20)      # above 20 Hz the envelope beats with low notes' own f0
    i = np.argmax(Pxx[band])
    fpk = F[band][i]
    prom = 10 * np.log10(Pxx[band][i] / np.median(Pxx[band]))
    # depth: peak-to-trough of the envelope band-passed around the peak
    sos = signal.butter(2, [fpk * 0.8, fpk * 1.25], 'band', fs=fs2, output='sos')
    am = signal.sosfiltfilt(sos, e)
    depth_db = 20 / np.log(10) * 2 * np.sqrt(2) * np.std(am)
    print(f'amplitude modulation: strongest at {fpk:.2f} Hz ({fpk * beat:.2f} per beat; 16ths = 4, 32nds = 8), '
          f'{prom:.0f} dB over the 2-20 Hz median; depth about {depth_db:.1f} dB peak-to-trough')
    top3 = np.argsort(Pxx[band])[::-1]
    picks = []
    for j in top3:
        if all(abs(F[band][j] - p) > 0.6 for p in picks):
            picks.append(F[band][j])
        if len(picks) == 4:
            break
    print('  next peaks:', ', '.join(f'{p:.2f} Hz ({p * beat:.2f}/beat)' for p in picks))
    out['am'] = dict(freq=float(fpk), per_beat=float(fpk * beat), prom_db=float(prom), depth_db=float(depth_db))

    # --- timbre: brightness vs pitch, slope, odd/even, vibrato
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=hop)) ** 2
    fr = librosa.fft_frequencies(sr=SR, n_fft=4096)
    n = min(S.shape[1], len(f0))
    cen = (S[:, :n] * fr[:, None]).sum(0) / (S[:, :n].sum(0) + 1e-12)
    okn = ok[:n]
    if okn.mean() > 0.2:
        r = np.corrcoef(np.log(cen[okn]), np.log(f0[:n][okn]))[0, 1]
        ratio = cen[okn] / f0[:n][okn]
        print(f'brightness: centroid median {np.median(cen[okn]):.0f} Hz = {np.median(ratio):.1f}x f0; '
              f'log-centroid vs log-f0 correlation {r:+.2f} (1 = the filter tracks the pitch)')
        # brightness over time independent of pitch: ratio's spread
        print(f'  centroid/f0 ratio 10-90%: {np.percentile(ratio, 10):.1f}..{np.percentile(ratio, 90):.1f}')
        # harmonic amplitudes at stable frames
        stab = okn.copy()
        stab[1:] &= np.abs(np.diff(np.nan_to_num(m[:n]))) < 0.05
        idx = np.nonzero(stab)[0]
        if len(idx) > 50:
            H = []
            for t in idx[::max(1, len(idx) // 400)]:
                h = []
                for k in range(1, 17):
                    fk = k * f0[t]
                    if fk > SR / 2 - 100:
                        break
                    b = int(round(fk / (SR / 4096)))
                    h.append(S[max(0, b - 2):b + 3, t].max())
                if len(h) == 16:
                    H.append(db(np.array(h)) - db(np.array(h[0])))
            if H:
                H = np.median(np.array(H), 0)
                kk = np.arange(1, 17)
                slope = np.polyfit(np.log2(kk), H, 1)[0]
                odd = H[0::2][1:8].mean()
                even = H[1::2][:7].mean()
                print(f'harmonics re the fundamental (dB, 1..16): {np.round(H, 0).astype(int).tolist()}')
                print(f'  slope {slope:.1f} dB/octave (saw -6, square -6 on odd only, triangle -12); '
                      f'odd 3..15 mean {odd:.1f} dB vs even 2..14 {even:.1f} dB')
                out['harmonics'] = H.tolist()
        # vibrato: periodic pitch wobble within held notes only (runs >= 300 ms that
        # stay within a semitone of their own median), so note changes don't count.
        mm_ = m[:n]
        held = []
        i = 0
        L = int(0.3 * fps)
        while i < n - L:
            w = mm_[i:i + L]
            if np.all(np.isfinite(w)) and np.ptp(w) < 1.0:
                j = i + L
                while j < n and np.isfinite(mm_[j]) and abs(mm_[j] - np.median(w)) < 0.5:
                    j += 1
                held.append((i, j))
                i = j
            else:
                i += int(0.05 * fps)
        if held:
            devs, specs = [], []
            for i0, j0 in held:
                w = mm_[i0:j0] - np.polyfit(np.arange(j0 - i0), mm_[i0:j0], 1)[0] * np.arange(j0 - i0)
                w = w - w.mean()
                devs.append(w)
                if len(w) >= int(0.5 * fps):
                    f_, p_ = signal.periodogram(w * np.hanning(len(w)), fs=fps, nfft=int(fps * 2))
                    specs.append(np.interp(np.arange(0.5, 15, 0.25), f_, p_))
            allw = np.concatenate(devs)
            dep = 2 * np.sqrt(2) * np.std(allw) * 100
            txt = f'vibrato: {len(held)} held notes ({sum(j - i for i, j in held) / fps:.1f} s); wobble depth about {dep:.0f} cents peak-to-peak'
            if specs:
                P = np.mean(specs, 0)
                fa = np.arange(0.5, 15, 0.25)
                band = fa >= 2
                k = np.argmax(P[band])
                txt += f', strongest at {fa[band][k]:.2f} Hz ({10 * np.log10(P[band][k] / np.median(P[band])):.0f} dB over median)'
            print(txt)

    # --- does the envelope's fast ripple follow the note? (sub-oscillator vs LFO)
    if ok.mean() > 0.2:
        rows = []
        mmx = np.round(np.nan_to_num(m)).astype(int)
        for nt in sorted(set(mmx[ok])):
            sel = ok & (mmx == nt)
            if sel.sum() < fps * 1.0:
                continue
            # envelope of this note's frames (hop-rate) -> stitched spectrum
            fr_idx = np.nonzero(sel)[0]
            segs = np.split(fr_idx, np.nonzero(np.diff(fr_idx) > 1)[0] + 1)
            P = []
            for sg in segs:
                a0, a1 = sg[0] * hop, (sg[-1] + 1) * hop
                if a1 - a0 < int(0.25 * SR):
                    continue
                ev = np.abs(signal.hilbert(y[a0:a1]))
                ev = signal.decimate(ev, 20)                     # 2205 Hz
                ev = np.log(np.maximum(np.abs(ev), 1e-6))
                ev = ev - uniform(ev, 220)
                f_, p_ = signal.periodogram(ev * np.hanning(len(ev)), fs=SR / 20, nfft=4410)
                P.append(np.interp(np.arange(4, 120, 0.5), f_, p_))
            if len(P) < 2:
                continue
            P = np.mean(P, 0)
            fa = np.arange(4, 120, 0.5)
            k = np.argmax(P)
            f0n = 440 * 2 ** ((nt - 69) / 12)
            rows.append((note(nt), f0n, fa[k], 10 * np.log10(P[k] / np.median(P))))
        for nm, f0n, fk, pr in rows:
            print(f'  envelope ripple on {nm} (f0 {f0n:.1f} Hz): strongest at {fk:.1f} Hz = {fk / f0n:.2f} x f0 ({pr:.0f} dB)')

    # --- left/right
    def oenv(ch):
        o = librosa.onset.onset_strength(y=ch, sr=SR, hop_length=hop)
        return (o - o.mean()) / (o.std() + 1e-9)
    eL, eR = oenv(x[0]), oenv(x[1])
    xc = signal.correlate(eL, eR, mode='full', method='fft') / len(eL)
    mid = len(eR) - 1
    pos = np.arange(int(0.04 * fps), int(1.1 * beat * fps))
    asym = xc[mid + pos] - xc[mid - pos]
    j = int(np.argmax(np.abs(asym)))
    snr = abs(asym[j]) / (np.median(np.abs(asym)) + 1e-9)
    side = db(np.mean((x[0] - x[1]) ** 2)) - db(np.mean((x[0] + x[1]) ** 2))
    print(f'stereo: side/mid {side:.1f} dB; L/R asymmetry peak at {pos[j] / fps * 1000:.0f} ms = {pos[j] / fps / beat:.2f} beat, '
          f'{"right" if asym[j] > 0 else "left"} leads, {snr:.1f}x median')
    return out


def uniform(x, n):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x, max(1, n))


if __name__ == '__main__':
    a = sys.argv
    probe(a[1], a[2], float(a[3]), float(a[4]), *(float(v) for v in a[5:7]))
