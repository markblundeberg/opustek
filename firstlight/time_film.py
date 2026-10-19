"""Time real frames of First Light's film, and their memory: python3 firstlight/time_film.py [--height 1080] [t ...]

Renders the frames at those times (seconds; default 30 100 160) in one process, after one warm-up frame, and
prints each frame's time and the process's peak memory. Run it inside a memory cap, at low priority:
    systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 python3 -W ignore firstlight/time_film.py
"""
import argparse
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import film  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--height', type=int, default=1080)
ap.add_argument('times', type=float, nargs='*', default=[30.0, 100.0, 160.0])
a = ap.parse_args()
fm = film.Film(a.height, 2)
fm.rng = np.random.default_rng(0)
t0 = time.time()
fm.frame(int(a.times[0] * film.FPS) - 1)
print(f'warm-up frame (builds the outside, casts the room): {time.time() - t0:.2f} s')
for t in a.times:
    i = int(round(t * film.FPS))
    t0 = time.time()
    fm.frame(i)
    print(f'frame at {t:g} s: {time.time() - t0:.2f} s')
print(f'peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB')
