"""Side by side: the same measures of form and arrival on two analysed tracks.

  python tools/refscope/refscope_compare.py out/refscope/<a> out/refscope/<b>"""
import json
import sys

import numpy as np

from refscope_analyze import mmss
from refscope_report import ev_strength

P = lambda x: 10 ** (np.asarray(x) / 10)
dbm = lambda x: 10 * np.log10(np.mean(P(x)))


def measures(d):
    r = json.load(open(f'{d}/results.json'))
    bar = r['bar_seconds']
    t = lambda b: r['bar1'] + b * bar
    mix = np.array(r['stems']['mix']['bar_db'])
    n = len(mix)
    body = slice(2, n - 4)                    # skip the edges (fade-in, tail)
    low = 10 * np.log10(P(r['mix_bands_bar']['sub']) + P(r['mix_bands_bar']['bass']))
    out = dict(name=d.rstrip('/').split('/')[-1], minutes=r['seconds'] / 60, bpm=r['bpm'], bars=n)
    # Arrivals: the biggest one-bar rises of the whole mix, and of the lows, over the two bars before.
    rise = np.array([mix[b] - dbm(mix[max(0, b - 2):b]) if b >= 2 else 0 for b in range(n)])
    lrise = np.array([low[b] - dbm(low[max(0, b - 2):b]) if b >= 2 else 0 for b in range(n)])
    rise[n - 4:] = lrise[n - 4:] = 0
    arr = []
    for b in np.argsort(rise)[::-1]:
        if rise[b] < 3 or len(arr) == 5:
            break
        if all(abs(b - a) > 2 for a, *_ in arr):
            before = dbm(mix[max(0, b - 4):b])            # the 4 bars before
            earlier = dbm(mix[max(0, b - 12):max(1, b - 4)])  # the 8 before those
            arr.append((b, rise[b], lrise[b], before - earlier))
    out['arrivals'] = [(mmss(t(b)), round(x, 1), round(y, 1), round(z, 1)) for b, x, y, z in sorted(arr)]
    out['form_range_db'] = float(np.percentile(mix[body], 95) - np.percentile(mix[body], 5))
    out['low_range_db'] = float(np.percentile(low[body], 95) - np.percentile(low[body], 5))
    # How long the track sits near its top level, and how much of it is "reduced" (6+ dB under).
    top = np.percentile(mix[body], 90)
    out['share_within_3db_of_top'] = float(np.mean(mix[body] > top - 3))
    out['share_6db_down'] = float(np.mean(mix[body] < top - 6))
    # Kick drop-outs: drums low band falls 15+ dB for 1-2 bars and comes back.
    dr = np.array(r['stems']['drums']['bar_db'])
    drops = [b for b in range(2, n - 3) if dr[b] < dr[b - 1] - 12 and max(dr[b + 1], dr[b + 2]) > dr[b - 1] - 4]
    out['kick_dropouts'] = len(drops)
    # New things: strong band events per minute (strength >= 30, one per bar).
    bars_ev = {e['bar'] for e in r['band_events'] if ev_strength(e) >= 30}
    out['strong_event_bars_per_min'] = len(bars_ev) / (r['seconds'] / 60)
    # First entry of the lead: first bar the bass stem is within 6 dB of its loudest.
    bs = np.array(r['stems']['bass']['bar_db'])
    out['lead_in_bar'] = int(np.argmax(bs > bs.max() - 6)) + 1
    out['lead_in_share'] = out['lead_in_bar'] / n
    out['width_db'] = {s: round(r['stems'][s]['width_db'], 1) for s in ('drums', 'bass', 'other', 'vocals')}
    sec = [(b0, b1) for b0, b1 in zip(r['section_bars'], r['section_bars'][1:] + [n]) if b1 > b0]
    out['sections'] = len(sec)
    out['median_section_bars'] = float(np.median([b1 - b0 for b0, b1 in sec]))
    return out


a, b = (measures(d) for d in sys.argv[1:3])
for k in a:
    if k == 'arrivals':
        continue
    print(f'{k:28s} {str(a[k]):>34s}   {str(b[k]):>34s}')
for m in (a, b):
    print(f"\n{m['name']}: biggest arrivals (time, mix rise dB, low-band rise dB, the 4 bars before vs the 8 before them dB)")
    for x in m['arrivals']:
        print('   ', x)
