"""Moiré's checks, on the mix and its stems: the machines run in 7 and 5, out of step with the bar, and agree where
the score says; and one check for each note the listener gave (the comments say which, by version).

  python3 moire/check_moire.py [out/moire.wav]
  python3 moire/check_moire.py --selftest   (a render with both figures 8 long: the period checks must fail)

Some checks still FAIL on the released version (the bed in drop 1, the bouncer over the bed in two spots, the kick
punch in the hybrid by 0.2 dB, the breakdown bouncer's 7): each was weighed against the listener's ear and left.
"""
import json
import subprocess
import sys

import numpy as np
from scipy import signal

import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), 'tools')]
import moire as P
from measures import db, Grid, bed_fill, repetition

SR, BEAT, BAR, tb = P.SR, P.BEAT, P.BAR, P.tb
GRID = Grid(P.BPM, 0.0)


def load(path):
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    assert sr == SR
    return x.T.astype(np.float64)

SIX = BEAT / 4


def sixteenth_levels(y, lo, hi, b0, b1):
    sos = signal.butter(4, [lo, hi], 'band', fs=SR, output='sos')
    e = signal.sosfilt(sos, y) ** 2
    n = (b1 - b0) * 16
    return np.array([db(e[int((tb(b0) + k * SIX) * SR):int((tb(b0) + k * SIX + 0.06) * SR)].mean()) for k in range(n)])


def pitch_sim(y, b0, b1, lo=150, hi=900):
    """Pitch-class similarity of each 16th with the one L later (v5): the lead's cascade has no rests, so its 7 is in
    its notes, not its levels (level r7 0.06 on the mix; pitch 0.65 at 7 against 0.31 at 8)."""
    a, w = int(tb(b0) * SR), int(0.1 * SR)
    f = np.fft.rfftfreq(8192, 1 / SR)
    m = (f > lo) & (f < hi)
    pc = np.round(12 * np.log2(f[m] / 440) + 69).astype(int) % 12
    C = []
    for k in range((b1 - b0) * 16):
        seg = y[int(a + k * SIX * SR):int(a + k * SIX * SR) + w] * np.hanning(w)
        C.append(np.bincount(pc, weights=np.abs(np.fft.rfft(seg, 8192))[m] ** 2, minlength=12))
    C = np.array(C).T
    C = C / (np.linalg.norm(C, axis=0) + 1e-9)
    return {L: float(np.mean(np.sum(C[:, :-L] * C[:, L:], 0))) for L in range(2, 17)}


def period_of(v, lags=range(2, 17)):
    # detrended over a bar (v4): with v3's loud breakdown hum gone, the bouncer's 4-bar fade-in became the biggest thing
    # in its band's levels, and a trend correlates at every lag (r 0.77-0.87 at 2-8)
    v = v - np.convolve(np.pad(v, 8, mode='edge'), np.ones(17) / 17, 'valid')
    ac = {L: float(np.corrcoef(v[:-L], v[L:])[0, 1]) for L in lags}
    best = max(ac, key=ac.get)
    return best, ac


def run(wav='out/moire.wav'):
    x = load(wav)
    y = x.mean(0)
    sc = json.load(open(wav.replace('.wav', '_score.json')))
    res = {}
    for lab, b0, b1, c0, c1 in (('first cycle', 17, 28, 17, 28), ('breakdown', 72, 78, 79, 85)):     # v3: they take turns in the breakdown
        pb, acb = period_of(sixteenth_levels(y, 600, 1400, b0, b1))
        pp, acp = period_of(sixteenth_levels(y, 2000, 5000, c0, c1))
        # present, not dominant: Mark lost the beat in the sketch, so the 4/4 grid was made to match the 7
        res[f'bouncer in 7, {lab}'] = (acb[7] >= 0.3 and acb[7] > acb[8] + 0.1, f'bars {b0}-{b1}: r at 7 16ths {acb[7]:.2f} (>= 0.3, above 8: {acb[8]:.2f}); 6: {acb[6]:.2f}, 4: {acb[4]:.2f}, 2: {acb[2]:.2f}; strongest {pb}')
        res[f'pinger in 5, {lab}'] = (pp in (5, 10, 15), f'bars {c0}-{c1}: strongest repeat {pp} 16ths (r {acp[pp]:.2f}; 4: {acp[4]:.2f}, 8: {acp[8]:.2f})')
    # the lead's arc: in drop 2 it plays in 7 (band 150-1200 Hz, where its figure sits)
    if 'grand' in sc and len(sc['grand']) > 2:
        # in the lead's register (its figure notes 130-590 Hz): 150-1200 Hz took in the clap and the bass (r7 -0.01 on the mix,
        # while the lead's own stem read 0.79)
        ps = pitch_sim(y, 107, 118)
        res['lead in 7, drop 2'] = (ps[7] >= 0.5 and ps[7] > ps[8] + 0.15 and ps[7] == max(ps.values()),
                                    f'bars 107-118, pitch-class similarity at 7 16ths {ps[7]:.2f} (8: {ps[8]:.2f}, 4: {ps[4]:.2f}; v4 0.67 / 0.42)')
        try:
            with np.load(wav.replace('.wav', '_stems.npz')) as f:     # each stem once: an npz unzips it on every lookup
                z = {k: f[k] for k in f.files}
        except FileNotFoundError:
            z = None
        pw = lambda k, a, b: db(np.mean(((z[k + 'L'] + z[k + 'R'])[int(tb(a) * SR):int(tb(b) * SR)]) ** 2))
        off = {} if z is None else {'bouncer, drop 1 second half': pw('bounce', 48, 70) - pw('bounce', 17, 34), 'bouncer, drop 2': pw('bounce', 107, 140) - pw('bounce', 17, 34),
               'pinger, drop 1 first half': pw('ping', 37, 55) - pw('ping', 17, 34), 'pinger, drop 2 first half': pw('ping', 107, 124) - pw('ping', 17, 34)}
        if off:
            res['machines sit out'] = (all(v <= -30 for v in off.values()), '; '.join(f'{k} {v:+.0f} dB' for k, v in off.items()) + ' re the first cycle (<= -30)')
    if 'grand' in sc and len(sc['grand']) > 2 and z is not None:
        # re-entries fade in (Mark: the pinger's return at 3:58 "suddenly sounds weird"): bar-to-bar rise <= 6 dB
        # after the first bar of each segment, and that first bar >= 6 dB under the segment's full level
        rises = []
        for k in ('bounce', 'ping'):
            e = ((z[k + 'L'] + z[k + 'R']) / 2) ** 2
            bl = np.array([db(e[int(tb(b) * SR):int(tb(b + 1) * SR)].mean()) for b in range(1, 149)])
            on = bl > bl.max() - 40
            starts = [b for b in range(2, 148) if on[b - 1] and not on[b - 2] and b > 8]
            for st in starts:
                if st + 3 <= 147:
                    rises.append((k, st + 1, bl[st + 2] - bl[st - 1] if st + 2 < 148 else 0, bl[st] - bl[st - 1], bl[st + 2] - bl[st - 1]))
        # the fade is 6 dB a bar by design (-24 dB to full over 4 bars), and v3's check sat its threshold exactly
        # there (v3 read 6.3 and failed; Mark on v3: "the fade-ins are good"): 7.5 a bar
        worst = max((r[3] for r in rises), default=0)
        res['re-entries fade in'] = (worst <= 7.5 and all(r[4] - 0 >= 6 for r in rises),
                                     '; '.join(f'{k} bar {b}: +{r1:.1f} then +{r2 - r1:.0f} over 2 bars' for k, b, _, r1, r2 in rises[:6]) + ' (second bar over the first <= 7.5 dB; design 6)')
        share = np.mean([max(db(np.mean(((z['bounceL'] + z['bounceR']) / 2)[int(tb(b) * SR):int(tb(b + 1) * SR)] ** 2)),
                             db(np.mean(((z['pingL'] + z['pingR']) / 2)[int(tb(b) * SR):int(tb(b + 1) * SR)] ** 2))) > -57 for b in range(1, 148)])     # v6: -57, as both machines rose 3 dB (at -60 their fade tails counted: 55%)
        res['machines play less'] = (share <= 0.55, f'bouncer or pinger sounding in {share:.0%} of bars (v2: about 72%; v3 first try 58%)')
        a2, b2 = int(tb(107) * SR), int(tb(119) * SR)     # v4: the 7's bars (the callback's held bass is dark by design)
        ld = ((z['leadL'] + z['leadR'])[a2:b2] / 2).astype(np.float32)
        f_, S_ = signal.welch(ld, SR, nperseg=4096)
        hi = 10 * np.log10(S_[f_ > 1500].sum() / S_.sum())
        res['lead clear in drop 2'] = (hi >= -10, f'lead energy above 1.5 kHz {hi:.1f} dB of its total in drop 2 (>= -10; v2, "muffled and strangled": -15.4)')
        # v4, Mark on v3: drop 2's lead "gets tired by ~3:50... maybe have it call back to the first lead section"
        lo = lambda a, b: db(np.mean(signal.sosfilt(signal.butter(4, [40, 150], 'band', fs=SR, output='sos'), (z['leadL'] + z['leadR'])[int(tb(a) * SR):int(tb(b) * SR)]) ** 2))
        cb = lo(121, 140) - lo(37, 70)
        res['lead calls back'] = (cb >= -6, f"the lead's held bass (40-150 Hz) in bars 121-139 {cb:+.1f} dB re drop 1 (>= -6; v3, the 7 to the end: about -20)")
    # v5, Mark on v4's drop-2 lead: "too weird / jazzy / dissonant" (its figure spelled a seventh chord on every root
    # and played B-flat and F over C). The written notes of drop 2 and the end: all in E minor, and the 7's all chord tones
    bar_ = lambda t: int(t // P.BAR) + 1
    ph = [e for e in sc['events'] if e[3] and P.tb(106) <= e[0] < P.tb(142)]
    outk = [e for e in ph if e[1] % 12 not in {4, 6, 7, 9, 11, 0, 2}]
    f7 = [e for e in ph if bar_(e[0]) in P.FIG7 and e[0] < P.tb(P.FIG7[-1], 3.0) and e[0] >= P.tb(106) + 2 * SIX]
    inch = np.mean([e[1] % 12 in {m % 12 for m in P.CHORDS[P.riff_i(bar_(e[0]))][:3]} for e in f7]) if f7 else 0
    res['lead in key, drop 2'] = (not outk and inch == 1.0, f'{len(outk)} of {len(ph)} notes out of E minor; the 7: {inch:.0%} triad tones (v4: 8 out of key, 53% chord tones)')
    # the bed (v4): on this measure (tools/measures.py bed_fill) one reference's main sections read lit 0.49-0.58 / held
    # 0.14-0.18, Engine 0.50 / 0.16;
    # v3's drop 1 0.23 / 0.05, drop 2 0.38 / 0.12; the grid alone (bars 2-7) 0 / 0 (the null)
    if z is not None and 'bedL' in z:
        # v6: the machines heard over the bed in their bands (v5: the bouncer 2-9 dB under the glass bed; Mark: "can't hear it")
        sb = signal.butter(4, [600, 1400], 'band', fs=SR, output='sos')
        bl_ = lambda k, a, b: db(np.mean(signal.sosfilt(sb, ((z[k + 'L'] + z[k + 'R']) / 2)[int(tb(a) * SR):int(tb(b) * SR)]) ** 2))
        ov = {f'{a}-{b - 1}': bl_('bounce', a, b) - bl_('bed', a, b) for a, b in ((37, 46), (75, 79), (87, 92))}
        res['bouncer over the bed'] = (all(v >= 3 for v in ov.values()), ', '.join(f'bars {k} {v:+.1f}' for k, v in ov.items()) + ' dB in 600-1400 Hz (>= +3; v5 -8.9 / -2.2 / -8.0)')
    if z is not None and 'decoL' in z:
        # v7, the detail pass: decorations faint but there (the reference's scrape: Mark found it "so faint"), set before
        # the render at -32..-18 dB re the mix in their sections
        mixp = lambda a, b: db(np.mean(y[int(tb(a) * SR):int(tb(b) * SR)] ** 2))
        dp = lambda a, b: db(np.mean(((z['decoL'] + z['decoR']) / 2)[int(tb(a) * SR):int(tb(b) * SR)] ** 2))
        # the stems are before the master gain: put them on the mix's scale by the kick, whose stem and mix level are known
        kg = db(np.mean(y[int(tb(2) * SR):int(tb(8) * SR)] ** 2)) - db(np.mean(z['kick'][int(tb(2) * SR):int(tb(8) * SR)] ** 2))
        ds = {'72-104 (drops, RMS re mix)': dp(72, 105) - mixp(72, 105) + kg}
        # the guiro is a sparse scrape: an RMS over bars reads it 35 dB lower than it sounds, so its peaks, re the kick's
        dpk = lambda a, b: 20 * np.log10(np.abs(((z['decoL'] + z['decoR']) / 2)[int(tb(a) * SR):int(tb(b) * SR)]).max() / np.abs(z['kick']).max())
        gs = {f'{a}-{b - 1} (guiro, peak re kick)': dpk(a, b) for a, b in ((12, 35), (37, 52), (121, 141))}
        ok = -32 <= ds['72-104 (drops, RMS re mix)'] <= -18 and all(-30 <= v <= -18 for v in gs.values())
        res['decorations faint'] = (ok, ', '.join(f'bars {k} {v:+.1f}' for k, v in {**ds, **gs}.items()) + ' dB (drops -32..-18; guiro -30..-18; v7 first try: guiro -45, inaudible)')
    # v7: the ending fades over a held chord, with no step (<= 10 dB in any half second) down to -40 dB re bar 141
    e5 = np.array([db(np.mean(y[int((tb(141) + 0.5 * i) * SR):int((tb(141) + 0.5 * (i + 1)) * SR)] ** 2)) for i in range(int((len(y) / SR - tb(141)) / 0.5) - 1)])
    e5 -= e5[0]
    stop = int(np.argmax(e5 < -30)) if (e5 < -30).any() else len(e5)     # falls judged while audible (v8: the fade's last step, -39 to -50, read 10.9)
    step = float(np.max(-np.diff(e5[2:stop + 1]))) if stop > 3 else 99     # after the final hit's own first second (v8: 10.7 there)
    rise = float(np.max(np.diff(e5[2:stop + 1]))) if stop > 3 else 0     # and nothing comes back up (v6: a boom at 148, +20 dB)
    res['ending fades'] = (step <= 10 and rise <= 6 and e5[-1] <= -35, f'largest half-second fall {step:.1f} dB above -30 (<= 10), rise {rise:+.1f} (<= 6); last half second {e5[-1]:+.0f} dB re bar 141 (<= -35)')
    # v8, the finishing pass. The kick's low end at the kick vs between kicks (50-160 Hz): v7 +16.5 in the intro, about
    # +10 under the bass; Engine +16.3 / +12.4, a reference +14.0 / +6.7 / +11.5
    def punch(bars):
        e = signal.sosfiltfilt(signal.butter(4, [50, 160], 'band', fs=SR, output='sos'), y) ** 2
        on = [e[int(tb(b, i) * SR):int((tb(b, i) + 0.05) * SR)].mean() for b in bars for i in range(4)]
        off = [e[int((tb(b, i) + 0.30) * SR):int((tb(b, i) + 0.45) * SR)].mean() for b in bars for i in range(4)]
        return db(np.mean(on)) - db(np.mean(off))
    pk = {k: punch(v) for k, v in {'36-40': range(36, 41), '57-67': range(57, 68), '107-118': range(107, 119), '121-137': range(121, 138)}.items()}
    res['kick punch'] = (all(v >= 12 for v in pk.values()), ', '.join(f'bars {k} {v:+.1f}' for k, v in pk.items()) + ' dB (>= 12, Engine with bass; v7 about +10)')
    # tone: 3-8 kHz re the whole, and what a phone-sized speaker (300 Hz-10 kHz) keeps, in drop 1 (41-69)
    a1, b1_ = int(tb(41) * SR), int(tb(70) * SR)
    M = y[a1:b1_]
    pres = db(np.mean(signal.sosfiltfilt(signal.butter(4, [3000, 8000], 'band', fs=SR, output='sos'), M) ** 2)) - db(np.mean(M ** 2))
    phone = db(np.mean(signal.sosfiltfilt(signal.butter(4, [300, 10000], 'band', fs=SR, output='sos'), M) ** 2)) - db(np.mean(M ** 2))
    res['tone'] = (-26 <= pres <= -20 and phone >= -14, f'drop 1: 3-8 kHz {pres:+.1f} dB of the whole (-26..-20; refs -22..-26, Engine -23, v7 -31); a phone keeps {phone:+.1f} (>= -14; refs -12.3..-13.8, Engine -11.1, v7 -15.6)')
    for b0, b1 in ((37, 69), (120, 139)):     # v6: the bed rests under drop 2's solo (106-118)
        lit, held = bed_fill(y, SR, GRID, b0, b1)
        res[f'bed {b0}-{b1}'] = (lit >= 0.40 and held >= 0.08, f"lit {lit:.2f} (>= 0.40), held {held:.2f} (>= 0.08) of 200-4000 Hz between beats")
    # the breakdown's hum (Mark on v3: "a bit stale to be that loud"): v3 sat 1.5 dB under drop 1 in 150-1000 Hz
    e3 = signal.sosfilt(signal.butter(4, [150, 1000], 'band', fs=SR, output='sos'), y) ** 2
    l3 = lambda a, b: db(e3[int(tb(a) * SR):int(tb(b) * SR)].mean())
    res['breakdown quieter'] = (l3(72, 86) - l3(37, 70) <= -6, f'bars 72-85 {l3(72, 86) - l3(37, 70):+.1f} dB re drop 1 in 150-1000 Hz (<= -6; v3 -1.5)')
    lv = lambda a, b: db(np.mean(y[int(tb(a) * SR):int(tb(b) * SR)] ** 2))
    sos2 = signal.butter(4, [150, 3000], 'band', fs=SR, output='sos')
    e2 = signal.sosfilt(sos2, y) ** 2
    lv16 = lambda t: db(e2[int(t * SR):int((t + 0.06) * SR)].mean())
    for T in sc.get('takes', [sc['take']]):
        A = T + 1
        take = lv(T, T + 0.75) - lv(T - 4, T)
        arrive = lv(A, A + 1) - lv(T, T + 0.75)
        floor = lv16(tb(T, 2.0))
        pk = [lv16(tb(T, 3.25 + 0.25 * j)) for j in range(3)]
        steps = np.diff([floor] + pk)
        res[f'drop at bar {A}'] = (take <= -5 and arrive >= 10 and pk[0] - floor >= 6 and max(steps[1:]) <= 6,
                                   f'take-away {take:+.1f} dB, arrival {arrive:+.1f}; pickups {pk[0] - floor:+.1f} / {pk[1] - floor:+.1f} / {pk[2] - floor:+.1f} over the take-away')
    for b0, b1 in ((9, 34), (37, 69), (107, 139)):
        s1, s4 = repetition(y, SR, GRID, b0, b1)
        res[f'repetition {b0}-{b1}'] = (s4 < 0.70, f'rhythm similarity with the bar 4 later {s4:.3f} (< 0.70; drafts heard as repetitive 0.71-0.87, a reference 0.53-0.61)')
    for k, (p, msg) in res.items():
        print(f"  {'PASS' if p else 'FAIL'}  {k}: {msg}")
    return res


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        subprocess.run([sys.executable, '-c', f'import sys; sys.path.insert(0, {HERE!r}); import moire as P; P.B_LEN = 8; P.P_LEN = 8; '
                        'mix, st, sc = P.render(); import numpy as np, json; from scipy.io import wavfile; '
                        'wavfile.write("out/moire_BROKEN-in-step.wav", P.SR, mix.astype(np.float32)); json.dump(sc, open("out/moire_BROKEN-in-step_score.json", "w"))'],
                       check=True, capture_output=True)
        print('self-test in-step (both figures 8 long):')
        r = run('out/moire_BROKEN-in-step.wav')
        print('  ->', 'the period checks FAIL as they should' if not any(r[k][0] for k in r if ' in 7' in k or ' in 5' in k) else 'a period check is blind')
    else:
        run(sys.argv[1] if len(sys.argv) > 1 else 'out/moire.wav')
