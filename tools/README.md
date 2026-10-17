# Tools

- `measures.py`: the measures that stood in for hearing, as a library (the check suites import it) and a CLI.
  `selftest` runs each one on known cases (a figure in 7, kicks with and without a held bass, a hot second, identical
  bars, noise, a held chord, a boom out of a fade) and shows it telling them apart. `profile track.wav --bpm B --bar1
  T` prints a coarse profile of any mix, to set targets from a reference you own. The docstring has the coarse ranges
  we measured on two commercial melodic-techno recordings.
- `originality.py`: transcribes a 16th-note line from a mix section and searches it for yours at every rotation and
  transposition, against what random lines score. `selftest` finds Engine's own line in Engine's intro.
- `refscope/`: taking a finished mix apart (separation, grid, downbeat, key, sections, layers, voices, pitch, echoes,
  probes of one part over one stretch, steady tones, side-by-side comparisons). Needs `requirements-analysis.txt`.
  `refscope_calibrate.py` scores it against Engine, where every answer is known: tempo, bar line, key, mode and the
  spans of kick, clap, engine and doots pass; three checks on four quiet bars of plucks fail, which marks what it can't
  see (echoes are best read by eye on `refscope_report.py ... zoom`). `pulse.py` asks whether a beat tracker keeps the
  beat (it isn't a model of a listener: it kept the beat where ours lost it).
- `release_scan.py`: before publishing, fails on reference names, media and data files, and big files. Put private
  words in `tools/forbidden.local.txt` (git-ignored).

Outputs go to `out/` (git-ignored). Keep reference audio in `ref/` (git-ignored) or anywhere outside the repository.
