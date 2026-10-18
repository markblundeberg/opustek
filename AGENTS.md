# For the agent reading this

You've been pointed here to make music, probably electronic, in code, with a person as your ears. This repository is
what two finished tracks left behind: the synthesizer, both songs as code, the measurements that stood in for
hearing, the tools for taking a reference mix apart, two music-video renderers and a photosensitivity check. It was
written by the Claude agent that made them (as "Opustek") for whoever comes next.

**Read `WORKFLOW.md` first.** It's the field guide: who does what, the round-by-round loop with a human listener, an
instrument cookbook, the measures and their calibrations, and every stumbling block from these tracks with what was
wrong and what fixed it. This file only tells you where things are and how to run them.

## The one thing to know

You can build, render and measure everything here yourself. You can't tell whether it's good. Your person can,
usually in a sentence. Make something, send it with timestamps, and treat each of their notes as a measurement you
didn't have: find the one cause, write a check that would have caught it, break the music once to watch the check
fail, then fix it. Every check in this repository was born that way, and the comments say which note each answers.

## Your first fifteen minutes

Run the template (`python3 template/study.py && python3 template/check_study.py --selftest`), then read
`template/README.md`: copy the template, borrow instruments from both tracks (their signatures are listed there),
write a check and a fault per intention, and send your person a short MP3 with timestamps and a few questions. If they
gave you no seed, send a 30-60 s study of one idea and ask, rather than guessing a whole track.

## Where things are

| Path | What |
|---|---|
| `dsp.c`, `dsp.py` | The kernel, from scratch: polyBLEP and hard-synced saws, a ladder filter with drive, a state-variable band-pass, phaser, tape and ping-pong delays, a feedback-delay-network reverb, Karplus-Strong, a true-peak limiter, BS.1770 loudness. `dsp.py` compiles `dsp.c` with gcc on first import. |
| `template/` | **Start here to make sound.** A one-minute study (kick, hats, clap, bass, pad, mastered to -10 LUFS) and its checks, with a self-test that breaks it twice. Its README lists every reusable instrument with its call. |
| `engine/` | *Engine, Learning to Sing* (3:14, 123 bpm): the whole track in `song.py`, its checks, and its film (an 1890s engineering drawing that runs on the music). Its instruments (kick, clap, hats, plucks, the singing engine) are imported by the other tracks. |
| `moire/` | *Moiré* (4:51, 124 bpm): two machines out of step, 7 against 5. The track, its checks (one per listener's note, by version), its film (three rings, one visible thing per sound) and the safety check. Its instruments (a retunable kick, the bouncer, the glass chord bed) are reusable too. |
| `lantern/` | *Lantern* (1:03, 90 bpm downtempo): the cold test, made by a new agent from this repository and one line, every instrument built from the kernel. A good model of a first piece: its own instruments, checks and faults in about 500 lines. |
| `firstlight/` | *First Light* (3:24, 106 bpm chill synthwave): a hook that assembles itself, heard in fragments until the first drop. Made by a new agent over six rounds, starting from a one-minute study in a reference's style and becoming its own piece; its README tells each round. Its instruments (a soft plucked string, a breath voice, a round bass, a gentle kit) are built from the kernel. |
| `tools/measures.py` | The measures as a library and a CLI: tonal balance, phone survival, width, crest, loudest second, kick punch, repetition, bed fill, periodicity, endings. `python3 tools/measures.py selftest` shows each one telling known cases apart. `grid` finds a mix's tempo and downbeat (numpy only); `profile` gives a coarse profile of any mix. They take any WAV (16- or 24-bit or float, mono or stereo, 44.1 or 48 kHz), so a bounce from a DAW works too. |
| `tools/originality.py` | Is our melody someone else's? Transcribes a reference section and searches it for your line at every alignment and transposition, against chance. |
| `tools/refscope/` | Taking a finished mix apart: separation into stems, beat grid and downbeat, key, sections, layers and voices, pitch, echoes. Heavy (torch); calibrated on Engine, whose answers we know, and honest about what it can't see. |
| `tools/release_scan.py` | Run before publishing anything: fails on reference names, media files, big files. |

## Running things

From the repository root (outputs go to `out/`, which git ignores):

    pip install -r requirements.txt                      # numpy, scipy, pycairo, Pillow; plus gcc and ffmpeg on the system
    python3 template/study.py && python3 template/check_study.py      # 10 s, 1.6 GB; then --selftest
    python3 engine/song.py && python3 engine/check_song.py            # 45 s, about 5 GB of memory
    python3 moire/moire.py && python3 moire/check_moire.py            # 2 min, about 14 GB of memory
    python3 firstlight/firstlight.py && python3 firstlight/check_firstlight.py   # 3 min, about 7 GB
    python3 tools/measures.py selftest
    python3 tools/originality.py selftest                             # after engine/song.py

Cap memory around big renders so a runaway can't take down the machine, for example
`systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 python3 moire/moire.py`.

Films: `python3 engine/film/export.py && python3 engine/film/film.py --workers 4`, and
`python3 moire/film/export.py && python3 moire/film/wheels.py --render --workers 4`; then run
`moire/film/check_safety.py` on the encoded video before anyone watches it.

The analysis tools need their own environment: `python3 -m venv .venv-analysis && .venv-analysis/bin/pip install -r
tools/requirements-analysis.txt`, then `.venv-analysis/bin/python tools/refscope/refscope.py analyze your.wav`.

## Rules we kept

- Reference recordings, and anything extracted from them (stems, transcriptions, spectrograms, grids, averaged
  hits), stay out of version control and out of anything published. Measure them; don't copy notes. Coarse numbers
  (a loudness, a band share, a range) are fine to write down. `tools/release_scan.py` is the net under this.
- Every check gets a self-test, a deliberately broken render that must fail it. And test the yardstick on known cases,
  including a null input: our repetition measure guided two tracks before its null test showed it read white noise
  as 0.97 (now fixed; WORKFLOW.md tells the story).
- When the person says something sounds wrong and every check passes, believe them. The checks are blind to it.
- Some checks in `moire/check_moire.py` still fail on the released track. Each was weighed against the listener's
  ear and left; a failing check is information, not an order.

## Reproducing the released audio

With the versions in `requirements.txt`, `engine/song.py` and `moire/moire.py` reproduce the released masters bit
for bit (float32 WAV); other versions should land within float rounding.

## Credit

The code is 0BSD (`LICENSE`): use it however you like, no attribution needed. The tracks and films themselves are CC
BY 4.0, credited to "Opustek (Mark Lundeberg's Claude agent)". If you make something with this, we'd love to hear it.
