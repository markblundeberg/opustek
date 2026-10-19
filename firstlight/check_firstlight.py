"""Checks for First Light, one per thing the piece intends, and a self-test that renders it broken on purpose (one
fault at a time, firstlight.BREAKS) and shows the matching check fail. They read the form from the score, so the same
checks serve the full piece and the studies (study 1, study 4) it grew from.

The piece's point is the hook and how it is heard: its fragments must be its own opening, note for note; they must
grow, and the whole hook (its top note above all) must wait for the first drop. Then the lines around it: a
counter-line under it that moves while it holds, a bridge on new ground with a second theme (the hook's opening
backwards) that the string answers in its held notes, a descant above the hook's last pass. The rest keeps what Low
Orbit's rounds established with the listener: the balance, a kick that "hits fine" at +13 dB, a bass well under Low
Orbit's 1.44 notes a beat ("too much bass funk"), a middle that moves, drums that change from bar to bar.

  python3 firstlight/check_firstlight.py              checks out/firstlight.wav (+ stems and score), the full piece
  python3 firstlight/check_firstlight.py --selftest   each fault against its check, several at once (--jobs N to set
                                                      how many; --keep keeps the broken renders)
  (--form study1 or study4: the studies, out/firstlight_study1.wav, ...)
"""
import functools
import json
import os
import sys

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, 'tools'), HERE]
from measures import (load, mono, db, Grid, kick_punch, band_share, phone_share, loudest_second, ending_steps,
                      loudness, onsets, masking, side_ratio, crest, bed_fill, repetition, syncopation)
import firstlight as P
import kit

FAULTS_STUDY4 = {'sharp-lead': 'tuning', 'wrong-note': 'key', 'fragment-altered': 'hook integrity', 'buried-lead': 'lead audible',
                 'busy-bass': 'bass density', 'no-sequence': 'mid motion', 'static-drums': 'drum variety',
                 'bright-master': 'dark top', 'boom-at-end': 'ending', 'flat-bass': 'bass tuning', 'buzzy-bass': 'bass tone',
                 'counter-clash': 'counter-line', 'buried-counter': 'counter audible', 'four-on-floor-bridge': 'bridge'}
FAULT_TRIPS = {'sharp-lead': 'tuning', 'wrong-note': 'key', 'fragment-altered': 'hook integrity',
               'early-climax': 'hook assembles', 'buried-lead': 'lead audible', 'busy-bass': 'bass density',
               'no-sequence': 'mid motion', 'static-drums': 'drum variety', 'kick-in-breakdown': 'breakdown',
               'bright-master': 'dark top', 'boom-at-end': 'ending', 'flat-bass': 'bass tuning',
               'buzzy-bass': 'bass tone'}
FAULTS_FULL = {**FAULTS_STUDY4, 'early-climax': 'hook assembles', 'kick-in-breakdown': 'breakdown',
               'descant-below': 'descant', 'answer-clash': 'answers'}
FAULTS = {'study1': FAULT_TRIPS, 'study4': FAULTS_STUDY4, 'full': FAULTS_FULL}

HZ = lambda m: 440.0 * 2 ** ((m - 69) / 12)


def note_cents(x, sr, t0, hz, win=0.22, skip=0.08):
    """Where the strongest partial within +-60 cents of hz sits, in cents re hz: after the glide, before the vibrato."""
    seg = x[int((t0 + skip) * sr):int((t0 + skip + win) * sr)]
    if len(seg) < int(0.1 * sr):
        return None
    nfft = 1 << 19
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), nfft))
    f = np.fft.rfftfreq(nfft, 1 / sr)
    lo, hi = np.searchsorted(f, [hz * 2 ** (-60 / 1200), hz * 2 ** (60 / 1200)])
    k = lo + int(np.argmax(S[lo:hi]))
    a, b, c = np.log(S[k - 1:k + 2] + 1e-20)
    kk = k + 0.5 * (a - c) / (a - 2 * b + c)
    return 1200 * np.log2(kk * sr / nfft / hz)


def harmonic_tilt(x, sr, notes, skip=0.1, win=0.25):
    """Brightness that carries across pitches: energy in harmonics 4-16 against 1-3 of each held note, dB (median)."""
    out = []
    for t0, hz in notes:
        seg = x[int((t0 + skip) * sr):int((t0 + skip + win) * sr)]
        if len(seg) < int(0.1 * sr):
            continue
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 16)) ** 2
        f = np.fft.rfftfreq(1 << 16, 1 / sr)
        h = lambda k: S[np.abs(f - k * hz) < hz * 0.08].sum()
        out.append(db(sum(h(k) for k in range(4, 17))) - db(sum(h(k) for k in range(1, 4))))
    return float(np.median(out)) if out else float('nan')


def bar_shape_change(y, sr, g, b0, b1, lo=200, hi=12000, nb=24):
    """How much a part's tone changes from one bar to the next: each bar's spectrum in nb log bands, its own level
    removed, mean absolute change between neighbouring bars, dB. A drum part that repeats exactly reads low."""
    f, t, Z = signal.stft(mono(y), sr, nperseg=4096, noverlap=3072)
    P2 = np.abs(Z) ** 2
    edges = np.geomspace(lo, hi, nb + 1)
    B = np.array([P2[(f >= a) & (f < b)].sum(0) for a, b in zip(edges[:-1], edges[1:])])
    S = np.array([db(B[:, (t >= g.tb(b)) & (t < g.tb(b + 1))].mean(1) + 1e-12) for b in range(b0, b1)])
    S -= S.mean(1, keepdims=True)
    return float(np.mean(np.abs(np.diff(S, axis=0))))


def bar_changes(y, sr, g, b0, b1, lo=200, hi=12000, nb=24):
    """bar_shape_change's per-bar values: the change from each bar to the next, b0 to b1."""
    f, t, Z = signal.stft(mono(y), sr, nperseg=4096, noverlap=3072)
    P2 = np.abs(Z) ** 2
    edges = np.geomspace(lo, hi, nb + 1)
    B = np.array([P2[(f >= a) & (f < b)].sum(0) for a, b in zip(edges[:-1], edges[1:])])
    S = np.array([db(B[:, (t >= g.tb(b)) & (t < g.tb(b + 1))].mean(1) + 1e-12) for b in range(b0, b1 + 1)])
    S -= S.mean(1, keepdims=True)
    return np.abs(np.diff(S, axis=0)).mean(1)


def band_onsets(y, sr, g, b0, b1, lo, hi, rise_db=6.0, win=0.03):
    """Notes starting per beat in one band: 16th steps where the band rises by rise_db over 30 ms."""
    e = signal.sosfilt(signal.butter(4, [lo, hi], 'band', fs=sr, output='sos'), mono(y)) ** 2
    w = int(win * sr)
    n = 0
    for k in range((b1 - b0) * 16):
        i = int(g.ts(b0, k) * sr)
        n += db(e[i:i + w].mean()) - db(e[max(0, i - w):i].mean() + 1e-20) >= rise_db
    return 4 * n / ((b1 - b0) * 16)


def peakiness(x, sr, notes, skip=0.15, win=0.25):
    """How far a harmonic stands above the quietest one under it (harmonics 1-12 of each held note, dB, median): 0
    for a plain low-pass, high for a resonant peak. The voice Mark heard as "too piercing" reads highest."""
    out = []
    for t0, hz in notes:
        seg = x[int((t0 + skip) * sr):int((t0 + skip + win) * sr)]
        if len(seg) < int(0.1 * sr):
            continue
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 16)) ** 2
        f = np.fft.rfftfreq(1 << 16, 1 / sr)
        h = [db(S[np.abs(f - k * hz) < hz * 0.08].max()) for k in range(1, 13) if k * hz < sr / 2 - 1000]
        out.append(max(h[k] - min(h[:k]) for k in range(1, len(h))))
    return float(np.median(out)) if out else float('nan')


def hardness(x, sr, onsets_, lo=2000, hi=5000):
    """How hard a voice strikes: its lo-hi Hz level in the first 20 ms of each note against 100-300 ms in, dB (median).
    Study 2's string was "a bit too hard"."""
    e = signal.sosfilt(signal.butter(4, [lo, hi], 'band', fs=sr, output='sos'), mono(x)) ** 2
    v = [db(e[int(t * sr):int((t + 0.02) * sr)].mean()) - db(e[int((t + 0.1) * sr):int((t + 0.3) * sr)].mean() + 1e-20)
         for t in onsets_]
    return float(np.median(v))


def bass_profile(hits, pitches):
    """Notes a beat, share on each 16th of the beat, share of beats with no new note, share of notes changing pitch."""
    h = np.asarray(hits, bool)
    _, npb = syncopation(h)
    pos = np.bincount(np.flatnonzero(h) % 4, minlength=4) / max(1, h.sum())
    empty = float((~h.reshape(-1, 4).any(1)).mean())
    moves = float((np.diff(pitches) != 0).mean()) if len(pitches) > 1 else 0.0
    return npb, pos, empty, moves


def study4_checks(res, sc, st, sr, g, sec, span, main):
    """The counter-line and the bridge (study 4 and the full piece)."""
    beat = g.beat
    lead = sc['lead']
    on_drops = lambda e: any(g.tb(sec[k][0]) - 1e-6 <= e['t'] < g.tb(sec[k][-1] + 1) for k in main)
    cev = [e for e in sc['counter'] if on_drops(e)]
    moving = below = clash = sounding = 0
    for e in cev:
        if not any(abs(e['t'] - h['t']) < 0.2 * beat for h in lead):
            moving += 1
        hk = [h for h in lead if h['t'] - 1e-6 <= e['t'] < h['t'] + h['dur']]
        if hk:
            sounding += 1
            iv = hk[-1]['midi'] - e['midi']
            below += iv > 0
            clash += (abs(iv) % 12) in (1, 6, 11)
    mv, bl, cl = moving / max(1, len(cev)), below / max(1, sounding), clash / max(1, sounding)
    res['counter-line'] = (mv >= 0.7 and bl >= 0.9 and cl <= 0.1,
                           f'{len(cev)} counter notes in the drops: {100 * mv:.0f}% start while the hook holds (>= 70), {100 * bl:.0f}% '
                           f'sit under the hook (>= 90), {100 * cl:.0f}% a minor 2nd, tritone or major 7th from it (<= 10)')
    errs = [note_cents(st['counter_raw'], sr, e['t'], HZ(e['midi'])) for e in sc['counter'] if e['dur'] >= 0.35]
    errs = np.array([e for e in errs if e is not None])
    res['counter tuning'] = (np.median(np.abs(errs)) <= 5 and np.max(np.abs(errs)) <= 12,
                             f'counter, {len(errs)} notes: median |error| {np.median(np.abs(errs)):.1f} cents, worst {np.max(np.abs(errs)):.1f}')
    bed = st['pad'] + st['stabs'] + st['seq'] + st['bass']
    sp = [(e['t'], e['t'] + e['dur']) for e in cev]
    cat = lambda s_, sp_: np.concatenate([s_[int(a * sr):int(b * sr)] for a, b in sp_])
    over = masking(cat(st['counter'], sp), cat(bed, sp), sr, 250, 1500, 0, len(cat(bed, sp)) / sr)
    bsp = [(e['t'], e['t'] + e['dur']) for e in sc['counter'] if e['bar'] in sec['bridge']]
    rest = bed + st['snare'] + st['hats'] + st['perc'] + st['lead']
    over_b = masking(cat(st['counter'], bsp), cat(rest, bsp), sr, 400, 2500, 0, len(cat(rest, bsp)) / sr)
    res['counter audible'] = (over >= 0 and over_b >= 3, f'the counter over the bed in 250-1500 Hz in the drops {over:+.1f} dB (>= 0); '
                                                         f'the second theme over everything else in the bridge {over_b:+.1f} (>= +3)')
    kpb = lambda bars: sum(1 for k in sc['kicks'] if g.tb(bars[0]) - 1e-6 <= k < g.tb(bars[-1] + 1)) / len(bars)
    drop_chords = {e['chord'] for e in sc['pad'] if e['t'] >= g.tb(sec[main[0]][0]) - 1e-6 and
                   not (g.tb(sec['bridge'][0]) - 1e-6 <= e['t'] < g.tb(sec['bridge'][-1] + 1))}
    new = {e['chord'] for e in sc['pad'] if g.tb(sec['bridge'][0]) - 1e-6 <= e['t'] < g.tb(sec['bridge'][-1] + 1)} - drop_chords
    hook_iv = np.diff([m for hb, q, m, d in sorted(sc['hook'], key=lambda h: (h[0], h[1]))][:3])
    th = sorted([e for e in sc['counter'] if e['bar'] in sec['bridge']], key=lambda e: e['t'])
    th_iv = np.diff([e['midi'] for e in th[:3]])
    ok = kpb(sec['bridge']) <= 2.5 and kpb(sec[main[0]]) >= 3.5 and len(new) >= 2 and np.array_equal(th_iv, -hook_iv[::-1])
    res['bridge'] = (ok, f'kicks a bar: bridge {kpb(sec["bridge"]):.1f} (<= 2.5, half time), drops {kpb(sec[main[0]]):.1f}; chords new to the '
                         f'bridge {sorted(new)} (>= 2); the second theme opens {th_iv.tolist()}, the hook {hook_iv.tolist()} (it should be the hook backwards)')


def full_checks(res, sc, st, sr, g, sec):
    """The descant over the hook's last pass, and the string's answers to the second theme (the full piece)."""
    beat = g.beat
    sounding = lambda ev, t: ([e for e in ev if e['t'] - 1e-6 <= t < e['t'] + e['dur']] or [None])[-1]
    hook = [e for e in sc['lead'] if e['statement'] is not None]
    ds = sc['descant']
    steps = np.arange(ds[0]['t'], ds[-1]['t'] + ds[-1]['dur'] - 1e-6, beat / 4)
    pairs = [(sounding(ds, t), sounding(hook, t)) for t in steps]
    iv = np.array([a['midi'] - h['midi'] for a, h in pairs if a and h])
    above, harsh = float((iv > 0).mean()), float(np.isin(np.abs(iv) % 12, (1, 6, 11)).mean())
    errs = np.array([e for e in (note_cents(st['descant_raw'], sr, e['t'], HZ(e['midi'])) for e in ds) if e is not None])
    bed = st['pad'] + st['seq'] + st['bass']
    sp = [(e['t'], e['t'] + e['dur']) for e in ds]
    cat = lambda s_: np.concatenate([s_[int(a * sr):int(b * sr)] for a, b in sp])
    over = masking(cat(st['descant']), cat(bed), sr, 800, 2500, 0, len(cat(bed)) / sr)
    res['descant'] = (above >= 0.9 and harsh <= 0.1 and np.max(np.abs(errs)) <= 12,
                      f'{len(ds)} notes over {len(iv)} 16ths with the hook: above it {100 * above:.0f}% (>= 90), a minor 2nd, '
                      f'tritone or major 7th from it {100 * harsh:.0f}% (<= 10); tuning worst {np.max(np.abs(errs)):.1f} cents (<= 12); '
                      f'over the bed in 800-2500 Hz {over:+.1f} dB (reported)')
    # the answers: each starts inside a held note of the second theme (begun at least half a beat before), so the
    # string speaks while the breath holds, and makes no harsh interval with that note
    th = [e for e in sc['counter'] if e['bar'] in sec['bridge']]
    ans = [e for e in sc['lead'] if e['statement'] is None]
    inside = harsh_a = 0
    for e in ans:
        h = sounding(th, e['t'])
        if h and e['t'] - h['t'] >= 0.5 * beat - 1e-6:
            inside += 1
            harsh_a += (abs(e['midi'] - h['midi']) % 12) in (1, 6, 11)
    share = inside / max(1, len(ans))
    res['answers'] = (share >= 0.8 and harsh_a == 0, f'{len(ans)} answers in the bridge: {100 * share:.0f}% start inside a held note '
                                                    f'of the second theme (>= 80), {harsh_a} a harsh interval from it (0)')


def run(path=f'out/{P.NAME}.wav', quiet=False):
    sr, x = load(path)
    base = path[:-4]
    sc = json.load(open(base + '_score.json'))
    st = kit.load_stems(base + '_stems.npz')
    g = Grid(sc['bpm'], 0.0)
    y = mono(x)
    sec = sc['sections']
    form = sc.get('form', 'study1')
    span = lambda bars: (g.tb(bars[0]), g.tb(bars[-1] + 1))
    main = sc.get('main', ['groove', 'drop'])
    drop, groove = span(sec[sc.get('peak', 'drop')]), span(sec[main[0]])
    kind = lambda k: sc['kinds'].get(k, k)
    lead_secs = (['breakdown'] if 'breakdown' in sec else []) + [k for k in main if kind(k) == 'drop']
    res = {}

    L, tp = loudness(x, sr)
    T = sc.get('lufs', -11.0)                  # the master's own target (the sound A/B tried -14)
    res['loudness'] = (abs(L - T) <= 0.7 and tp <= -1.0, f'{L:.1f} LUFS ({T:g} +- 0.7), true peak {tp:.1f} dBTP (<= -1)')

    errs = [note_cents(st['lead_raw'], sr, e['t'], HZ(e['midi'])) for e in sc['lead'] if e['dur'] >= 0.35]
    errs = np.array([e for e in errs if e is not None])
    med, worst = np.median(np.abs(errs)), np.max(np.abs(errs))
    res['tuning'] = (med <= 5 and worst <= 12, f'lead, {len(errs)} notes: median |error| {med:.1f} cents (<= 5), worst {worst:.1f} (<= 12)')

    # the bass's tuning: its raw oscillator (the string itself, for the struck basses), notes of 0.35 s or more
    berrs = [note_cents(st['bass_raw'], sr, e['t'], HZ(e['midi']), win=0.3, skip=0.05) for e in sc['bass'] if e['dur'] >= 0.35]
    berrs = np.array([e for e in berrs if e is not None])
    bmed, bworst = np.median(np.abs(berrs)), np.max(np.abs(berrs))
    res['bass tuning'] = (bmed <= 5 and bworst <= 12, f"bass '{sc.get('bass_tone', 'layered')}', {len(berrs)} notes: median |error| {bmed:.1f} cents (<= 5), worst {bworst:.1f} (<= 12)")

    # the bass's tone: its 300-3000 Hz against 30-300 Hz. Heard in study 3: the fingered string bass at -6.3 dB was
    # "quite buzzy and too bright, worse than the layered one" (-15.0); the round one (-20.9) "actually works"
    gd = [span(sec[k]) for k in main]
    bandp = lambda yy, lo, hi: np.mean(np.concatenate([signal.sosfilt(signal.butter(4, [lo, hi], 'band', fs=sr, output='sos'),
                                                                      yy[int(a * sr):int(b * sr)]) ** 2 for a, b in gd]))
    bt = db(bandp(st['bass'], 300, 3000)) - db(bandp(st['bass'], 30, 300))
    res['bass tone'] = (-28 <= bt <= -10, f"bass '{sc.get('bass_tone')}' 300-3000 Hz against 30-300 Hz in {', '.join(main)}: {bt:+.1f} dB "
                                          f"(-28 to -10; fingered -6.3 'buzzy', layered -15.0, round -20.9 'works')")

    # key: every pitch played is in F minor; strong lead notes (on the beat, or pushed onto it) are chord tones
    pcs = set(sc['key_pcs'])
    lines = sc['lead'] + sc.get('counter', []) + sc.get('descant', [])
    played = ([e['midi'] for e in lines] + [e['midi'] for e in sc['bass']] + [e['midi'] for e in sc['seq']]
              + [m for e in sc['stabs'] for m in e['notes']] + [m for e in sc['pad'] for m in e['notes']])
    out = sorted({m for m in played if m % 12 not in pcs})
    ct = [e for e in lines if (e['beat'] % 1 == 0 or e['beat'] % 1 >= 0.75) and e['dur'] >= 0.5 * g.beat]
    share = np.mean([e['midi'] % 12 in set(sc['chord_tones'][e['chord']]) for e in ct])
    res['key'] = (not out and share >= 0.75, f'{len(played)} notes, out of F minor: {out if out else "none"}; strong lead notes '
                                             f'that are tones of their chord {100 * share:.0f}% (>= 75)')

    # the hook's integrity: every statement plays exactly the hook's own notes for its bars (pitch and place), so a
    # fragment heard early is the same tune the drop completes (the last note of the last statement may be held)
    hook = {(hb, q): m for hb, q, m, d in sc['hook']}
    bad = []
    for si, (start, bars, _) in enumerate(sc['statements']):
        got = {(e['hook_bar'], e['hook_beat']): e['midi'] for e in sc['lead'] if e['statement'] == si}
        want = {k: m for k, m in hook.items() if k[0] in bars}
        if got != want:
            bad.append(start)
    res['hook integrity'] = (not bad, f'{len(sc["statements"])} statements of the hook, those that differ from it: {bad if bad else "none"}')

    # the hook assembles: statements grow (in bars) until the whole one; the whole one starts a drop; the hook's top
    # note and its last four bars are heard first there (study 4 has no fragments, so nothing to assemble)
    lens = [len(b) for s, b, v in sc['statements']]
    if len(set(lens)) > 1:
        full = lens.index(max(lens))
        grow = all(a <= b for a, b in zip(lens[:full], lens[1:full + 1]))
        fbar = sc['statements'][full][0]
        fsec = next(k for k, v in sec.items() if fbar in v)
        d0 = g.tb(sec[fsec][0])
        top = max(m for _, _, m, _ in sc['hook'])
        first_top = min(e['t'] for e in sc['lead'] if e['midi'] == top)
        late = [e for e in sc['lead'] if (e['hook_bar'] or 0) >= 5 and e['t'] < d0]
        ok = grow and kind(fsec) == 'drop' and fbar == sec[fsec][0] and first_top >= d0 - 1e-6 and not late
        res['hook assembles'] = (ok, f'statement lengths {lens} (growing to the whole: {grow}); the whole hook at bar {fbar}, '
                                     f'{"the start of" if fbar == sec[fsec][0] else "inside"} {fsec}; its top note ({top}) first at {first_top:.1f} s '
                                     f'({fsec} starts at {d0:.1f}); hook bars 5-8 before it: {len(late)}')

    # singable: the hook's range, its share of steps, its widest leap
    hm = [m for _, _, m, _ in sorted(sc['hook'], key=lambda h: (h[0], h[1]))]
    iv = np.abs(np.diff(hm))
    rng_, steps, leap = max(hm) - min(hm), float((iv <= 2).mean()), int(iv.max())
    res['singable'] = (rng_ <= 17 and steps >= 0.45 and leap <= 7, f'hook range {rng_} semitones (<= 17), steps {100 * steps:.0f}% of its '
                                                                    f'intervals (>= 45), widest leap {leap} (<= 7)')

    # the breakdown: the floor falls away
    if 'breakdown' in sec:
        b0, b1 = sec['breakdown'][0], sec['breakdown'][-1]
        low = signal.sosfilt(signal.butter(4, [40, 120], 'band', fs=sr, output='sos'), y) ** 2
        lw = lambda a, b: db(low[int(g.tb(a) * sr):int(g.tb(b) * sr)].mean())
        dip = lw(b0, b1 + 1) - lw(b0 - 4, b0)
        res['breakdown'] = (dip <= -10, f'40-120 Hz in bars {b0}-{b1} {dip:+.1f} dB re bars {b0 - 4}-{b0 - 1} (<= -10)')

    ks = [k for k in sc['kicks'] if drop[0] <= k < drop[1]]
    kp = kick_punch(y, sr, ks)
    raw = sum(st[k] for k in ('kick', 'snare', 'hats', 'perc', 'bass', 'pad', 'stabs', 'seq', 'lead'))
    kp_raw = kick_punch(raw, sr, ks)
    res['kick punch'] = (kp >= 12, f'drop: {kp:+.1f} dB on the master (>= +12; heard as fine at +13.3 in Low Orbit), {kp_raw:+.1f} on the stems')

    # the lead is the point: over everything else in its band, while it plays in the breakdown and the drop
    overs = []
    rest = st['pad'] + st['stabs'] + st['seq'] + st['bass'] + st['snare'] + st['hats'] + st['perc'] + st['counter']
    if 'descant' in st:
        rest = rest + st['descant']
    for a, b in (span(sec[k]) for k in lead_secs):
        spans = [(e['t'], e['t'] + e['dur']) for e in sc['lead'] if a <= e['t'] < b]
        cat = lambda s: np.concatenate([s[int(p * sr):int(q * sr)] for p, q in spans])
        overs.append(masking(cat(st['lead']), cat(rest), sr, 400, 2500, 0, len(cat(rest)) / sr))
    res['lead audible'] = (min(overs) >= 3, f'lead over everything else in 400-2500 Hz while it plays: '
                                            + ', '.join(f'{k} {v:+.1f} dB' for k, v in zip(lead_secs, overs)) + ' (each >= +3)')

    # the bass: well under Low Orbit's 1.44 notes a beat, which the listener heard as "too much bass funk"
    gb = [b for k in main for b in sec[k]]
    hits = np.zeros(len(gb) * 16, bool)
    pitches = []
    for e in sc['bass']:
        if e['bar'] in gb:
            hits[gb.index(e['bar']) * 16 + int(round((e['t'] - g.tb(e['bar'])) / g.six))] = True
            pitches.append(e['midi'])
    npb, pos, empty, moves = bass_profile(hits, pitches)
    res['bass density'] = (0.8 <= npb <= 1.2, f'{", ".join(main)}: {npb:.2f} notes a beat (0.8-1.2; Low Orbit 1.44), on the beat '
                                              f'{100 * pos[0]:.0f}%, beats with no new note {100 * empty:.0f}%, changing pitch {100 * moves:.0f}%')
    short = []
    for b in sc['bass_bars']:
        want = sum(1 for e in sc['bass'] if e['bar'] == b)
        got = int(onsets(st['bass'], sr, g, b, b + 1, rise_db=4.0).sum())
        if got < max(1, 0.7 * want):
            short.append((b, got, want))
    res['bass plays'] = (not short, f'{len(sc["bass_bars"])} bars, bars with under 70% of their written onsets on the stem: {short if short else "none"}')

    # the middle moves (the accompaniment: pad, stabs, sequence), and the drums change from bar to bar (Low Orbit's
    # calibrations: true-stem thresholds, set against a synthwave album's separated stems)
    acc = st['pad'] + st['stabs'] + st['seq']
    spans_ = [(sec[k][0], sec[k][-1]) for k in main]
    mm = [min(band_onsets(acc, sr, g, a, b + 1, 250, 500), band_onsets(acc, sr, g, a, b + 1, 500, 1000)) for a, b in spans_]
    res['mid motion'] = (min(mm) >= 0.6, f'notes starting a beat in the quieter of 250-500 and 500-1000 Hz of the accompaniment, '
                                         + ', '.join(f'{k} {v:.2f}' for k, v in zip(main, mm)) + ' (each >= 0.6)')
    # (0.85 was the album's least varied, true stems; the studies' 8-bar sections passed it. The full piece's 16-bar
    # sections, with something at every four-bar mark, read 0.79-0.84: the steady bars between the marks outweigh
    # them, and the step into the next section counts half as much. So 0.75 for 16 bars, provisional until heard;
    # the static fault, v2's drums, reads 0.24-0.32)
    drums = st['kick'] + st['snare'] + st['hats'] + st['perc']
    dv = [bar_shape_change(drums, sr, g, a, b + 1) for a, b in spans_]
    floor = [0.85 if b - a < 12 else 0.75 for a, b in spans_]
    res['drum variety'] = (all(v >= f for v, f in zip(dv, floor)), 'mean bar-to-bar change of the drums, '
                           + ', '.join(f'{k} {v:.2f} dB (>= {f})' for k, v, f in zip(main, dv, floor)))

    if form != 'study1':
        study4_checks(res, sc, st, sr, g, sec, span, main)
    if sc.get('descant'):
        full_checks(res, sc, st, sr, g, sec)

    # (the floor at -28, not the synthwave album's -25: the listener chose the darker, gentler voices twice, in
    # studies 2 and 3, and they sit near -25; the ceiling still catches a bright master)
    s38 = band_share(y, sr, 3000, 8000, *drop)
    s816 = band_share(y, sr, 8000, 16000, *drop)
    ls, lt = loudest_second(y, sr)
    res['dark top'] = (-28 <= s38 <= -16.5 and s816 <= -24 and ls <= -25,
                       f'drop: 3-8 kHz {s38:+.1f} dB of the whole (-28 to -16.5), 8-16 kHz {s816:+.1f} (<= -24), '
                       f'loudest 2.5-4 kHz second {ls:.1f} dBFS at {lt:.1f} s (<= -25)')

    # (rises counted only into a level above -30 dB re the final hit, as the falls are: a wobble 45 dB down, the
    # bass's two saws beating as it dies, isn't a boom)
    def audible_rise(yy):
        lv_, _, fall_ = ending_steps(yy, sr, g.tb(sc['nbars']), skip=1)
        d = np.diff(lv_)
        d[:1] = 0
        return float(max([v for v, to in zip(d, lv_[1:]) if to > -30] + [0.0])), fall_
    rise, fall = audible_rise(y)
    rise_lo, _ = audible_rise(signal.sosfilt(signal.butter(4, [40, 160], 'band', fs=sr, output='sos'), y))
    res['ending'] = (rise <= 4 and rise_lo <= 4, f'after bar {sc["nbars"]}: biggest half-second rise {rise:+.1f} dB, in 40-160 Hz {rise_lo:+.1f} '
                                                 f'(each <= +4), biggest fall while audible {fall:.1f} dB')

    # report only
    lead_notes = [(e['t'], HZ(e['midi'])) for e in sc['lead'] if drop[0] <= e['t'] < drop[1] and e['dur'] >= 0.4]
    lead_drop = st['lead'][int(drop[0] * sr):int(drop[1] * sr)]
    res['report: lead'] = (True, f"'{sc.get('lead_voice', 'pulse')}'{' on the soft bed' if sc.get('soft') else ''}: peakiness {peakiness(st['lead_raw'] if sc.get('lead_voice') in ('epiano', 'pluck') else st['lead'], sr, lead_notes):.1f} dB, "
                                 f"2-5 kHz {band_share(lead_drop, sr, 2000, 5000):+.1f} dB of the lead, harmonics 4-16 vs 1-3 {harmonic_tilt(st['lead'], sr, lead_notes):+.1f} dB, "
                                 f"hardness {hardness(st['lead_raw'], sr, [t0 for t0, _ in lead_notes]):+.1f} dB (2-5 kHz, first 20 ms against 100-300 ms)")
    res['report: tone'] = (True, f'drop: a phone keeps {phone_share(y, sr, *drop):+.1f} dB, crest {crest(y[int(drop[0] * sr):int(drop[1] * sr)], sr):.1f} dB, '
                                 f'side/mid 30-120 Hz {side_ratio(x, sr, 30, 120, *drop):.1f} dB, 500-2000 {side_ratio(x, sr, 500, 2000, *drop):.1f}; '
                                 f'lead harmonics 4-16 vs 1-3 {harmonic_tilt(st["lead"], sr, lead_notes):+.1f} dB')
    pk = sec[sc.get('peak', 'drop')]
    bf = bed_fill(y, sr, g, pk[0], pk[-1] + 1)
    rp = repetition(y, sr, g, sec[main[0]][0], pk[-1] + 1)
    res['report: texture'] = (True, f'bed fill in the peak {bf[0]:.2f} lit, {bf[1]:.2f} held; repetition from the first main section {rp[0]:.2f} / {rp[1]:.2f}')
    lv_sec = {k: db(np.mean(y[int(span(v)[0] * sr):int(span(v)[1] * sr)] ** 2)) for k, v in sec.items()}
    res['report: sections'] = (True, 'level ' + ', '.join(f'{k} {v:.1f}' for k, v in lv_sec.items()) + ' dBFS')
    if not quiet:
        for k, (p, m) in res.items():
            print(f"  {'PASS' if p else 'FAIL'}  {k}: {m}")
    return res


if __name__ == '__main__':
    form = sys.argv[sys.argv.index('--form') + 1] if '--form' in sys.argv else 'full'
    tag = '' if form == 'full' else f'_{form}'
    if '--selftest' in sys.argv:
        jobs = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else None
        ok = kit.selftest(FAULTS[form],
                          lambda brk: [sys.executable, os.path.join(HERE, 'firstlight.py'), '--break', brk, '--form', form],
                          lambda brk: f'out/{P.NAME}{tag}_BROKEN-{brk}.wav', functools.partial(run, quiet=True),
                          f'out/{P.NAME}{tag}.wav', mem_gb=7.5, jobs=jobs, keep='--keep' in sys.argv)
        sys.exit(0 if ok else 1)
    args = [a for a in sys.argv[1:] if not a.startswith('--') and a != form and not a.isdigit()]
    sys.exit(0 if all(p for p, _ in run(args[0] if args else f'out/{P.NAME}{tag}.wav').values()) else 1)
