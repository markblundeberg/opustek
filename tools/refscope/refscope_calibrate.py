"""Score refscope against Engine, where every answer is known.

  python3 engine/song.py                                   (writes out/song.wav and its dry stems)
  python tools/refscope/refscope.py analyze out/song.wav    (separates it as if it were a stranger's mix)
  python tools/refscope/refscope_calibrate.py [results.json]

Truth: out/song_stems.npz (dry stems at 48 kHz; the engine's f0 at 1 kHz), engine/song.py (123 bpm, bar 1 at
0:00, F# Dorian; doots echo at 1/2, 3/4 and 1 beat; plucks ping-pong at 3/4 beat)."""
import json
import os
import sys

import numpy as np

from refscope_analyze import spans, db, mmss

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BPM, BAR1 = 123.0, 0.0
BAR = 4 * 60 / BPM
res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'out', 'refscope', 'song', 'results.json')))
z = np.load(os.path.join(ROOT, 'out', 'song_stems.npz'))
sr = int(z['sr'])
truth = {'kick': z['kick'], 'clap': z['clap'], 'engine': z['droneL'] + z['droneR'],
         'doots': z['doots'], 'plucks': z['ks']}
nbar = int(np.ceil(len(z['kick']) / sr / BAR))
ok = True


def check(name, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}: {msg}")


print('Grid')
check('tempo', abs(res['bpm'] - BPM) < 0.05, f"{res['bpm']:.3f} bpm (truth {BPM})")
off = (res['bar1'] - BAR1 + BAR / 2) % BAR - BAR / 2
check('bar line', abs(off) < 0.03, f"bar 1 at {res['bar1']:.3f} s, {off * 1000:+.0f} ms from truth")
shift = int(round((res['bar1'] - BAR1) / BAR))   # their bar i == our bar i + shift

print('Key')
check('key', any(k in ('F# minor', 'E major', 'A major') for _, k in res['keys'][:2]),
      f"{res['keys']} (truth F# Dorian = the E major scale on F#)")
check('mode', res['modes'][0][1] == 'Dorian', f"{res['modes']}")

print('Layers: does each instrument appear as a layer or stem, and when?')
cands = [(f"voice {v['id']} {v['layers']}", np.array(v['bar_db'])) for v in res['voices']] + \
        [(l['label'], np.array(l['bar_db'])) for l in res['layers']] + \
        [(s, np.array(res['stems'][s]['bar_db'])) for s in ('drums', 'bass', 'other', 'vocals')]
for name, x in truth.items():
    seg = x[: nbar * int(BAR * sr)]
    tb = db(np.array([np.mean(seg[int(i * BAR * sr):int((i + 1) * BAR * sr)] ** 2) for i in range(nbar)]))
    trel = tb - tb.max()
    tsp = spans(trel, -20)
    best = None
    for lab, cb in cands:
        o = -shift       # their bar o + i is our bar i
        n = min(len(cb) - o, nbar)
        a, b = trel[:n], cb[o:o + n]
        b = np.maximum(b, b.max() - 40)
        r = np.corrcoef(np.maximum(a, -40), b)[0, 1]
        if best is None or r > best[0]:
            best = (r, lab, b - b.max())
    r, lab, crel = best
    csp = spans(crel, -15)
    # boundary error: each truth boundary to the nearest detected one
    tb_ = sorted({e for sp in tsp for e in sp})
    cb_ = sorted({e for sp in csp for e in sp})
    errs = [min(abs(t - c) for c in cb_) if cb_ else 99 for t in tb_]
    within1 = np.mean([e <= 1 for e in errs]) if errs else 0
    check(name, r > 0.7 and within1 >= 0.75,
          f"best match {lab} (r={r:.2f}); truth spans {tsp}; found {csp}; boundaries within 1 bar: {within1:.0%}")

print('Echoes (ping-pong: asymmetry of the L/R onset correlation)')
lay = {l['label']: l for l in res['layers']}
rows = []
for lab in list(lay) + ['other', 'vocals', 'drums', 'bass']:
    e = lay[lab]['echo'] if lab in lay else res['stems'][lab]['echo']
    if not e:
        continue
    snr = abs(e['pp_asym']) / (e['pp_asym_median'] + 1e-9)
    rows.append((snr, lab, e))
rows.sort(reverse=True, key=lambda r: r[0])
for snr, lab, e in rows[:10]:
    rec = ', '.join(f"{p['lag_beats']:.2f}" for p in e['peaks'])
    print(f"    {lab:10s} ping-pong {e['pp_lag_beats']:.2f} b, {e['pp_leader']} leads, asym {e['pp_asym']:+.3f} "
          f"({snr:.1f}x median); recurrences at {rec} b")
top = rows[0]
check('plucks ping-pong', abs(top[2]['pp_lag_beats'] - 0.75) < 0.03 and top[2]['pp_leader'] == 'left',
      f"strongest asymmetric echo: {top[1]} {top[2]['pp_lag_beats']:.2f} b, {top[2]['pp_leader']} leads (truth 0.75 b, left leads)")
print('Ping-pong per 8-bar window (top 6 by asymmetry over median)')
pp = sorted(res['local_pingpong'], key=lambda r: -r['snr'])
for r in pp[:6]:
    print(f"    {r['stem']:6s} bars {r['bar0']:3d}-{r['bar1']:3d}  {r['lag_beats']:.2f} b ({r['lag_s'] * 1000:.0f} ms), "
          f"{r['leader']} leads, asym {r['asym']:+.3f}, {r['snr']:.1f}x median")
w = [r for r in res['local_pingpong'] if r['stem'] == 'other' and r['bar0'] == 49]
check('plucks ping-pong (window 49-56)', w and abs(w[0]['lag_beats'] - 0.75) < 0.03 and w[0]['leader'] == 'left',
      f"{w[0]['lag_beats']:.2f} b, {w[0]['leader']} leads, {w[0]['snr']:.1f}x median" if w else 'no window')
print('PASS' if ok else 'SOME FAIL')
print('Known limits (the same on a draft of Engine when these tools were built): four quiet bars of plucks 8 dB under the')
print("engine are never isolated as a layer, and neither echo measure (whole track, or the 8-bar window over the plucks)")
print("finds their 3/4-beat ping-pong; a person reading a zoom did (refscope_report.py zoom). So the three plucks checks")
print('FAIL by design: they mark what this toolset cannot see, and a change that made them pass would be worth a look.')
