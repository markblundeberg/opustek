"""refscope report: results.json -> report.md, overview.png; and zooms.

  python tools/refscope/refscope_report.py out/refscope/<name>        report + overview
  python tools/refscope/refscope_report.py out/refscope/<name> zoom <t0> <t1> [stem] [fmax]

marks.json beside results.json (optional): [["m:ss", "what the listener hears"], ...];
they are drawn on the overview and each is answered in the report."""
import json
import os
import sys

import librosa
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from refscope import SR, STEMS, load, separate
from refscope_analyze import NOTES, db, mmss

EV_STRENGTH = 30          # event bars at or above this are listed (on a draft of Engine: 11/24 of the top on known boundaries, against 4 by chance)


def secs(s):
    m, x = s.split(':')
    return 60 * int(m) + float(x)


def register(hz):
    for lim, name in ((60, 'sub'), (250, 'bass'), (700, 'low-mid'), (2000, 'mid'), (5000, 'upper-mid'), (10000, 'high')):
        if hz < lim:
            return name
    return 'air'


def rhythm_words(prof, per, spiky):
    prof = np.asarray(prof)
    pos = ['on the beat', 'on the 2nd 16th', 'on the offbeat 8th', 'on the 4th 16th']
    if spiky < 1.6:
        tex = 'sustained'
    elif spiky < 3:
        tex = 'pulsing'
    else:
        tex = 'percussive'
    if prof.max() - prof.min() < 0.25:
        where = 'evenly across the 16ths'
    else:
        top = np.argsort(prof)[::-1]
        where = pos[top[0]] + (f' and {pos[top[1]]}' if prof[top[1]] > 0.8 * prof[top[0]] else '')
    p = max(per, key=lambda k: per[k])
    return f'{tex}, {where}, repeating every {p} beat{"s" if p != "1" else ""}'


def note(hz):
    if not hz or hz <= 0 or not np.isfinite(hz):
        return '-'
    m = 69 + 12 * np.log2(hz / 440)
    return f'{NOTES[int(round(m)) % 12]}{int(round(m)) // 12 - 1}'


def ev_strength(e):
    return e['change_db'] + 3 * np.log2(e['nbands']) + 0.5 * min(e['re_mix_db'], 0)


def load_res(d):
    r = json.load(open(os.path.join(d, 'results.json')))
    mk = os.path.join(d, 'marks.json')
    marks = [(secs(t), w) for t, w in json.load(open(mk))] if os.path.exists(mk) else []
    return r, marks


def bar_time(r, b):          # 1-based bar -> seconds
    return r['bar1'] + (b - 1) * r['bar_seconds']


def voice_desc(r, v):
    lay = {l['label']: l for l in r['layers']}
    L = [lay[x] for x in v['layers']]
    w = np.array([10 ** (np.max(l['bar_db']) / 10) for l in L])
    w = w / w.sum()
    cen = float(np.sum([l['centroid'] * wi for l, wi in zip(L, w)]))
    lo = min(l['band10'] for l in L)
    hi = max(l['band90'] for l in L)
    main = L[int(np.argmax(w))]
    return dict(centroid=cen, lo=lo, hi=hi, words=rhythm_words(main['beat_profile'], main['periodicity'], main['spikiness']),
                reg=register(cen), peak=max(np.max(l['bar_db']) for l in L))


def report(d):
    r, marks = load_res(d)
    name = os.path.basename(d.rstrip('/'))
    nb = r['n_bars']
    out = [f'# refscope: {name}', '']
    out.append(f"{mmss(r['seconds'])} long. **{r['bpm']:.2f} bpm**, one fixed grid ({r.get('on_grid_share', 0):.0%} of strong drum "
               f"onsets within 30 ms of a beat, median {r['onset_dev_ms']:.1f} ms; the rest fall between beats); bar 1 starts at {r['bar1']:.3f} s; "
               f"{nb} bars of {r['bar_seconds']:.3f} s.")
    k = r['keys']
    out.append(f"Key (Krumhansl profiles on the pitched stems): **{k[0][1]}** (r={k[0][0]:.2f}; next {k[1][1]} {k[1][0]:.2f}). "
               f"Mode from that tonic: {', '.join(f'{m} {v:.2f}' for v, m in r['modes'])}.")
    rel = np.array(r['chroma_from_tonic'])
    out.append('Pitch-class weight from the tonic (semitones 0-11): ' +
               ' '.join(f'{i}:{x:.2f}' for i, x in enumerate(rel / rel.max())))
    out.append('')

    # Sections
    out.append('## Sections (novelty boundaries, snapped to bars)')
    out.append('')
    out.append('| from | bar | bars | mix level | sub / bass / mid / high (dB) | drums / bass / other / vocals (dB) |')
    out.append('|---|---|---|---|---|---|')
    bounds = [b for b in r['section_bars'] if 0 <= b < nb] + [nb]
    bounds = sorted(set(bounds))
    mb = r['mix_bands_bar']
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        sl = slice(a, b)
        def m(x):
            return float(db(np.mean(10 ** (np.asarray(x)[sl] / 10))))
        bands = ' / '.join(f"{m(mb[n]):.0f}" for n in ('sub', 'bass', 'mid', 'high'))
        stems = ' / '.join(f"{m(r['stems'][s]['bar_db']):.0f}" for s in STEMS)
        out.append(f"| {mmss(bar_time(r, a + 1))} | {a + 1} | {b - a} | {m(r['stems']['mix']['bar_db']):.1f} | {bands} | {stems} |")
    out.append('')

    # Voices
    out.append('## Voices (layers that rise and fall together, per stem)')
    out.append('')
    vd = []
    for v in r['voices']:
        if not v['spans']:
            continue
        de = voice_desc(r, v)
        vd.append((v, de))
    for v, de in sorted(vd, key=lambda x: bar_time(r, x[0]['spans'][0][0] + 1)):
        sp = ', '.join(f"{mmss(bar_time(r, a + 1))}-{mmss(bar_time(r, b + 1))}" for a, b in v['spans'])
        out.append(f"- **V{v['id']}** ({v['stem']}; {', '.join(v['layers'])}): {de['reg']}, centroid {de['centroid']:.0f} Hz, "
                   f"80% of its energy in {de['lo']:.0f}-{de['hi']:.0f} Hz; {de['words']}. Plays {sp}.")
    out.append('')

    # Events
    out.append(f'## What changes, bar by bar (band events of strength >= {EV_STRENGTH})')
    out.append('')
    byb = {}
    for e in r['band_events']:
        byb.setdefault(e['bar'], []).append(e)
    for b in sorted(byb):
        es = sorted(byb[b], key=ev_strength, reverse=True)
        if ev_strength(es[0]) < EV_STRENGTH:
            continue
        txt = '; '.join(f"{e['stem']} {e['kind']} {e['lo']:.0f}-{e['hi']:.0f} Hz ({e['change_db']:+.0f} dB, {e['re_mix_db']:+.0f} re mix)"
                        for e in es[:4] if ev_strength(e) >= EV_STRENGTH - 10)
        out.append(f"- **{mmss(bar_time(r, b))}** (bar {b}): {txt}")
    out.append('')

    # Pitch
    out.append('## Pitch by section (pYIN; bass stem, and the other stem where voiced)')
    out.append('')
    hop = 512 / SR
    pb = np.array(r['pitch']['bass']['f0'])
    vb = np.array(r['pitch']['bass']['voiced'], bool)
    po = np.array(r['pitch']['other']['f0'])
    vo = np.array(r['pitch']['other']['voiced'], bool) & (np.array(r['pitch']['other']['prob']) > 0.5)
    t = np.arange(len(pb)) * hop
    for a, b in zip(bounds[:-1], bounds[1:]):
        t0, t1 = bar_time(r, a + 1), bar_time(r, b + 1)
        m = (t >= t0) & (t < t1)
        def summ(f, v):
            x = f[m & v]
            if len(x) < 20:
                return 'none tracked'
            q = np.percentile(x, [5, 50, 95])
            notes = 69 + 12 * np.log2(x / 440)
            hist = np.bincount(np.round(notes).astype(int) % 12, minlength=12)
            topn = ' '.join(NOTES[i] for i in np.argsort(hist)[::-1][:4] if hist[i] > 0.05 * hist.sum())
            return f'{note(q[0])}-{note(q[2])} (median {note(q[1])}; most: {topn}; voiced {np.mean(v[m]):.0%})'
        out.append(f"- {mmss(t0)}: bass {summ(pb, vb)}; other {summ(po, vo)}")
    out.append('')

    # Ping-pong
    out.append('## Ping-pong (L/R asymmetry per 8 bars; only windows >= 8x their median)')
    out.append('')
    for p in sorted(r['local_pingpong'], key=lambda p: (p['bar0'], p['stem'])):
        if p['snr'] >= 8:
            out.append(f"- {mmss(bar_time(r, p['bar0']))}-{mmss(bar_time(r, p['bar1'] + 1))} {p['stem']}: "
                       f"{p['lag_beats']:.2f} beat ({p['lag_s'] * 1000:.0f} ms), {p['leader']} leads ({p['snr']:.0f}x)")
    out.append('')

    # Marks
    if marks:
        out.append("## The listener's marks, answered")
        out.append('')
        for tm, w in marks:
            b = int(np.floor((tm - r['bar1']) / r['bar_seconds'])) + 1
            near = [e for bb in range(b - 1, b + 3) for e in byb.get(bb, [])]
            near = sorted(near, key=ev_strength, reverse=True)[:5]
            txt = '; '.join(f"bar {e['bar']} ({mmss(bar_time(r, e['bar']))}) {e['stem']} {e['kind']} {e['lo']:.0f}-{e['hi']:.0f} Hz "
                            f"({e['change_db']:+.0f} dB)" for e in near)
            starting = [f"V{v['id']} {de['reg']} {v['stem']}" for v, de in vd
                        if any(abs(bar_time(r, a + 1) - tm) < 2.5 * r['bar_seconds'] for a, _ in v['spans'])]
            out.append(f"- **{mmss(tm)}** (bar {b}) \"{w}\"")
            out.append(f"  - strongest changes nearby: {txt or 'none'}")
            out.append(f"  - voices starting within ~2 bars: {', '.join(starting) or 'none'}")
        out.append('')
    open(os.path.join(d, 'report.md'), 'w').write('\n'.join(out))
    print('wrote', os.path.join(d, 'report.md'))
    overview(d, r, marks, vd, bounds)


def overview(d, r, marks, vd, bounds):
    name = os.path.basename(d.rstrip('/'))
    path = r['file']
    x = load(path).mean(0)
    Smel = librosa.power_to_db(librosa.feature.melspectrogram(y=x, sr=SR, n_fft=4096, hop_length=2048, n_mels=160, fmax=16000), ref=np.max)
    nb = r['n_bars']
    tbar = np.array([bar_time(r, b + 1) for b in range(nb + 1)])
    rows = [v for v, _ in sorted(vd, key=lambda x: (STEMS.index(x[0]['stem']), x[1]['centroid']))]
    fig, ax = plt.subplots(4, 1, figsize=(22, 16), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1.1, 0.4 + 0.22 * len(rows), 1.6]))
    tm = np.arange(Smel.shape[1]) * 2048 / SR
    ax[0].pcolormesh(tm, librosa.mel_frequencies(n_mels=160, fmax=16000), Smel, shading='auto', cmap='magma', vmin=-80)
    ax[0].set_yscale('symlog', linthresh=200)
    ax[0].set_ylim(30, 16000)
    ax[0].set_ylabel('Hz (mix)')
    A = np.array([np.asarray(r['stems'][s]['bar_db']) - np.max(r['stems'][s]['bar_db']) for s in STEMS])
    ax[1].pcolormesh(tbar, np.arange(len(STEMS) + 1), np.maximum(A, -40), cmap='viridis', vmin=-40, vmax=0)
    ax[1].set_yticks(np.arange(len(STEMS)) + 0.5, STEMS)
    V = np.array([np.asarray(v['bar_db']) - np.max(v['bar_db']) for v in rows])
    ax[2].pcolormesh(tbar, np.arange(len(rows) + 1), np.maximum(V, -30), cmap='viridis', vmin=-30, vmax=0)
    lab = []
    dd = {v['id']: de for v, de in vd}
    for v in rows:
        de = dd[v['id']]
        lab.append(f"V{v['id']} {v['stem']} {de['centroid']:.0f} Hz")
    ax[2].set_yticks(np.arange(len(rows)) + 0.5, lab, fontsize=7)
    hop = 512 / SR
    for key, c, lbl in (('bass', 'tab:blue', 'bass stem f0'), ('other', 'tab:orange', 'other stem f0')):
        f = np.array(r['pitch'][key]['f0'])
        v = np.array(r['pitch'][key]['voiced'], bool)
        if key != 'bass':
            v &= np.array(r['pitch'][key]['prob']) > 0.5
        t = np.arange(len(f)) * hop
        ax[3].scatter(t[v], 69 + 12 * np.log2(f[v] / 440), s=0.3, c=c, label=lbl)
    ax[3].set_ylabel('MIDI note')
    ax[3].legend(loc='upper left', markerscale=20)
    for a in ax:
        for b in bounds:
            if b < nb:
                a.axvline(bar_time(r, b + 1), color='w' if a is ax[0] else 'k', lw=0.6, alpha=0.5)
        for tmk, w in marks:
            a.axvline(tmk, color='red', ls='--', lw=1)
    for tmk, w in marks:
        ax[0].text(tmk + 1, 12000, mmss(tmk), color='red', fontsize=9, va='top')
    ticks = np.arange(0, r['seconds'], 30)
    ax[3].set_xticks(ticks, [mmss(t)[:-2] for t in ticks])
    ax[3].set_xlim(0, r['seconds'])
    fig.suptitle(f'{name}: mix spectrogram, stems, voices, pitch (grey: section boundaries; red: listener marks)')
    fig.tight_layout()
    fig.savefig(os.path.join(d, 'overview.png'), dpi=90)
    print('wrote', os.path.join(d, 'overview.png'))


def zoom(d, t0, t1, stem='mix', fmax=8000):
    """A close spectrogram with the beat grid, to read echoes and timbre by eye."""
    r, marks = load_res(d)
    if stem == 'mix':
        y = load(r['file'])
    else:
        y = separate(r['file'])[stem]
    a, b = int(t0 * SR), int(t1 * SR)
    fig, ax = plt.subplots(3, 1, figsize=(18, 11), sharex=True, gridspec_kw=dict(height_ratios=[3, 1, 1]))
    m = y[:, a:b].mean(0)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(m, n_fft=2048, hop_length=128)), ref=np.max)
    t = t0 + np.arange(S.shape[1]) * 128 / SR
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    k = f <= fmax
    ax[0].pcolormesh(t, f[k], S[k], shading='auto', cmap='magma', vmin=-70)
    ax[0].set_ylabel(f'Hz ({stem})')
    for ch, c in ((0, 'tab:blue'), (1, 'tab:red')):
        e = librosa.feature.rms(y=y[ch, a:b], frame_length=1024, hop_length=128)[0]
        ax[1].plot(t0 + np.arange(len(e)) * 128 / SR, db(e ** 2), c=c, lw=0.8, label='LR'[ch])
    ax[1].legend(loc='upper right')
    ax[1].set_ylabel('RMS dB')
    on = librosa.onset.onset_strength(y=m, sr=SR, hop_length=128)
    ax[2].plot(t0 + np.arange(len(on)) * 128 / SR, on, lw=0.7)
    ax[2].set_ylabel('onset')
    beat = r['period']
    gb = np.arange(np.ceil((t0 - r['phase']) / beat), np.floor((t1 - r['phase']) / beat) + 1) * beat + r['phase']
    for a_ in ax:
        for g in gb:
            bar1 = abs(((g - r['bar1']) / r['bar_seconds']) - round((g - r['bar1']) / r['bar_seconds'])) < 1e-3
            a_.axvline(g, color='c' if bar1 else 'w' if a_ is ax[0] else '0.7', lw=1.2 if bar1 else 0.5, alpha=0.8)
    fn = os.path.join(d, f'zoom_{stem}_{t0:.1f}-{t1:.1f}.png')
    ax[2].set_xlabel('s')
    fig.tight_layout()
    fig.savefig(fn, dpi=80)
    print('wrote', fn)


if __name__ == '__main__':
    d = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == 'zoom':
        zoom(d, float(sys.argv[3]), float(sys.argv[4]), *(sys.argv[5:6] or ['mix']),
             *([float(sys.argv[6])] if len(sys.argv) > 6 else []))
    else:
        report(d)
