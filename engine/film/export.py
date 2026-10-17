"""Export the song's control curves and events for the film, at 1 kHz.
Everything the drawing moves by comes from here: the same score as the sound."""
import json
import os
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import song as P
from song import BEAT, BAR, CR, tb

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'out', 'engine_film')


def main():
    dur = P.NBARS * BAR + P.TAIL
    tc = np.arange(int(dur * CR)) / CR
    f0, cut, amp, dc, ratio, blend, dist = P.engine(tc)
    st = np.load(os.path.join(OUT, '..', 'song_stems.npz'))
    sr = int(st['sr'])
    e = (st['droneL'] ** 2 + st['droneR'] ** 2) / 2
    b, a = signal.butter(1, 12, fs=sr)
    env = np.sqrt(np.maximum(signal.lfilter(b, a, e), 0))[::sr // CR][:len(tc)]
    env = env / np.percentile(env[env > 1e-4], 95)
    chop = st['chop'][:len(tc)]
    ev = {'kicks': [tb(b, i) for b in P.KICK for i in range(4)],
          'claps': [tb(b, i) for b in P.CLAP for i in (1, 3)],
          'ks': [(tb(b, bt), m) for b, bt, m in P.KS_NOTES],
          'doots': [(tb(b) + i * BEAT / 4, m, v) for b in P.DOOTS for i, m, acc, v in P.doot_pattern(b)],
          'chops': [list(c[:5]) + [c[5]] for c in P.CHOPS],
          'chop_cells': P.chop_cells(),
          'bar': BAR, 'beat': BEAT, 'dur': dur}
    np.savez_compressed(os.path.join(OUT, 'curves.npz'), tc=tc, f0=f0, cut=cut, amp=amp, ratio=ratio, blend=blend,
                        dist=dist, env=env, chop=chop, **{'dc_' + k: v for k, v in dc.items()})
    json.dump(ev, open(os.path.join(OUT, 'events.json'), 'w'))
    print('exported', len(tc), 'samples;', {k: len(v) for k, v in ev.items() if isinstance(v, list)})


if __name__ == '__main__':
    main()
