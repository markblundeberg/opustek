"""Can a listener keep the beat? A beat tracker on the mix, scored against the true grid: the share
of its beats within 40 ms of a true beat, and the share on the true downbeat phase.

  python tools/refscope/pulse.py <wav> <bpm> <bar1 s> <t0> <t1>"""
import sys

import librosa
import numpy as np
import soundfile as sf


def pulse(path, bpm, bar1, t0, t1):
    y, sr = sf.read(path, dtype='float32', always_2d=True)
    y = y.mean(1)[int(t0 * sr):int(t1 * sr)]
    tempo, bt = librosa.beat.beat_track(y=y, sr=sr, hop_length=256, start_bpm=bpm, units='time', tightness=100)
    bt = bt + t0
    P = 60 / bpm
    dev = (bt - bar1 + P / 2) % P - P / 2
    on_beat = np.mean(np.abs(dev) < 0.04)
    # onset-envelope periodicity: the beat (4 sixteenths) against the bouncer's 7 and the pinger's 5
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    fps = sr / 256
    six = P / 4
    ac = lambda k: float(np.corrcoef(env[:-int(round(k * six * fps))], env[int(round(k * six * fps)):])[0, 1])
    return float(np.atleast_1d(tempo)[0]), on_beat, ac(4), ac(7), ac(5)


if __name__ == '__main__':
    a = sys.argv
    tmp, ob, a4, a7, a5 = pulse(a[1], float(a[2]), float(a[3]), float(a[4]), float(a[5]))
    print(f'tracker {tmp:.1f} bpm, {ob:.0%} of its beats on the true grid; onset periodicity at 4 / 7 / 5 sixteenths: {a4:.2f} / {a7:.2f} / {a5:.2f}')
