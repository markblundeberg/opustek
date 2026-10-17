# Engine, Learning to Sing

A techno track (3:14, 123 bpm, F# Dorian) and its film, made in October 2026. The idea: an engine whose note is its
speed, a hard-synced saw whose resonant peak walks the harmonics, so the machine seems to learn to sing over the
track. The film is an 1890s-style engineering plate whose two-stroke hot-bulb engine runs on the music, crank turning
at the bass line's pitch, every part driven by the score. The page, with the plate described as a Victorian
engineering journal would and a playable engine: https://marklundeberg.com/opustek/engine/

- `song.py`: the whole track in one file: the score as data, the instruments, the engine and the mix, down to the
  mastered WAV. Its instruments (kick, clap, hats, plucks, the engine voice `cloog`) and helpers (`place`,
  `sidechain`, `analog_env`, ...) are reused by `moire/` and `template/`.
- `check_song.py`: the checks. Each measures something on the audio, mostly on a single stem so a neighbour can't
  pass it for the part, against what the score intends or against ranges measured on two commercial techno
  recordings (not included). `--selftest` renders the song broken on purpose, once per fault (seven faults), and
  shows the matching check fail.
- `film/export.py`, `film/film.py`: the film. `film/check_film.py` checks the rendered frames: sparks against the
  gate, the strobe freezing the crank at the right angle, no flashing beyond WCAG 2.3.1's limit, blackbody colour,
  the pause, camera speed, and the commutator's blades against the audio.

From the repository root:

    python3 engine/song.py                  # -> out/song.wav (+ stems, score), about 45 s and 5 GB
    python3 engine/check_song.py            # the checks
    python3 engine/check_song.py --selftest
    python3 engine/film/export.py           # the score's curves and events, for the film
    python3 engine/film/film.py --workers 4 # -> out/engine_film/film.mp4 (about half an hour; fonts FreeSerif and Z003)
    python3 engine/film/check_film.py out/engine_film/film_frames 0
