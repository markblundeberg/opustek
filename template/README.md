# Your first piece

1. **Copy the template.** `cp -r template mypiece`, rename `study.py`, and change the score at the top (tempo, key,
   bars, which instruments play where). It renders in about 10 s with 1.6 GB of memory; its checks take a few
   seconds. Run everything from the repository root (outputs go to `out/`).
2. **Borrow instruments** (below) before writing new ones. Each was chosen by ear by a person, which you can't do yet.
3. **Write a check per thing you intend**, and a fault that should trip it (add it to `BREAKS` and `FAULT_TRIPS`).
   Run `--selftest` and watch each fault fail its own check before you trust a pass.
4. **Make an MP3 and send it**, with timestamps for where to listen and three or four short questions:

       ffmpeg -loglevel error -y -i out/mypiece.wav -b:a 224k out/mypiece_v1.mp3

**If your person gave you no seed** (no genre, no reference, no idea), don't guess a whole track: send a 30-60 s
study of one idea and ask. "Is this the right world? What would you put on to get in the mood?" One sentence back
from them settles more than an hour of your guessing. The checks' target ranges here come from two melodic-techno
recordings; your person's taste may want others (`tools/measures.py profile` on a track they like gives you theirs).

## Instruments you can borrow

Import them with the template's two path lines (`sys.path[:0] = [ROOT, os.path.join(ROOT, 'engine')]`, plus
`os.path.join(ROOT, 'moire')` for Moiré's). Importing a track runs no render and costs almost no memory (the 5 and 14 GB figures are for rendering Engine and Moiré whole). Everything is at SR = 48 kHz and returns
a mono float64 array unless it says otherwise.

From `engine/song.py` (`import song as S`):

| Call | What it is |
|---|---|
| `S.kick_hit(sr)` | A sine-sweep kick into F#1 with a click, quick through the low mids, a dip at 180 Hz. Fixed pitch: for another key use Moiré's. |
| `S.clap_hit(sr)` | Three quick bursts of band-passed noise and a 50 ms tail (long claps read as amateur). |
| `S.spaghetti(sr, rng)` | One hat hit: 4-7 noise grains a few ms apart, then a noisy tail. Vary level and pan per hit. |
| `S.ks_note(sr, hz)` | A struck metal bar: Karplus-Strong with a fractional delay, bright and long. |
| `S.doot(sr, hz, open_, accent)` | A round pluck, saw and square through a ladder; `open_` 0 (muffled) to 1 (clear). |
| `S.cloog(tc, f0, cut, amp, dc, ratio, blend, dist)` | The singing engine: detuned saws plus a hard-synced saw whose resonant peak walks the harmonics (`ratio`). Control-rate arrays at 1 kHz (`tc`); read `render()` for how it is driven. |
| `S.place(dst, hit, [(t, gain), ...], sr, gain)` | Add a hit into a buffer at times in seconds. |
| `S.sidechain(times, n, sr, depth, rel)` | A gain curve that ducks at each kick time and recovers. |
| `S.analog_env(tc, t0, att, hold, dec)`, `S.drift(n, rate, cents, seed)`, `S.peq(x, sr, f0, gain_db, q)` | An envelope, slow pitch drift, a peaking EQ. |

From `moire/moire.py` (`import moire as M`):

| Call | What it is |
|---|---|
| `M.kick_hit(sr, root=hz)` | Engine's kick, retunable to any root. |
| `M.tick(sr, rng)` | A 17 ms burst near 6.4 kHz, for 16th ticks. |
| `M.bounce_note(sr, hz, accent, rng, vowel=(760, 1180))` | The bouncer: saw and pulse through two formant band-passes ("ah"), with a 45-cent scoop up into each note. |
| `M.bed_voice('glass', fqs, fc, since, N, rng)` | The chord bed the listener chose: seven drifting unison saws per note, phaser, chorus, FM bells into a long reverb, air. Returns (L, R). `fqs` is a list of per-sample frequency arrays (one per chord note), `fc` a per-sample cutoff, `since` per-sample seconds since the chord began. Build them at 1 kHz and upsample with `S.up(curve, tc, N, SR)`; smooth chord changes in log frequency first (see `sm` in `moire.py`'s `render()`), or every change clicks. Also `'basic'`, `'lush'`, `'analog'`. |
| `M.chorus(x, sr, seed)` | A short modulated delay mixed with the dry (Juno-ish). |

From the kernel (`import dsp`): `dsp.saw(freq_array, sr)`, `dsp.syncsaw(freq, ratio, sr)`, `dsp.ladder(x, fc, sr,
res, drive)`, `dsp.svf_bp(x, fc, sr, q)`, `dsp.phaser(x, sr)`, `dsp.tapedelay(x, sr, delay_s)`,
`dsp.pingpong(L, R, delay_s, sr)`, `dsp.reverb(L, R, sr, rt60)`, `dsp.ks(exc, sr, freq)`, `dsp.butter(x, fc, sr,
kind)`, `dsp.limiter(x_n2, sr, ceiling_db)`, `dsp.lufs(x_n2, sr)`, `dsp.true_peak_db(x_n2)`. Frequencies and cutoffs
are per-sample arrays (a constant can be broadcast); stereo arrays for the limiter and loudness are shaped (n, 2).
