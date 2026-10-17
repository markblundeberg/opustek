# Moiré

A techno track (4:51, 124 bpm, E minor) and its film, made in October 2026. Two machines play out of step over a
steady four-on-the-floor beat: the bouncer (a saw-and-pulse "ah" arpeggio) repeats every 7 sixteenths, the pinger
(metal plucks) every 5. Against the bar of 16 they drift, and all three agree only every 35 bars, at bars 36, 71,
106 and 141 (1:08, 2:15, 3:23, 4:31). The lead's arc: in the second drop it learns the machines' meter and plays in 7
itself, then calls back to its first section and hybridizes the two. The page, with the round-by-round notes and how
much of the listener's time it took: https://marklundeberg.com/opustek/moire/

- `moire.py`: the track. It borrows Engine's instruments (`engine/song.py`) and adds its own: the bouncer, the glass
  chord bed (seven drifting unison saws per note, phaser, chorus, FM bells into a long reverb, a breath of air), the
  acid-style lead, the transitions. The comments record each version's change and the note that asked for it.
- `check_moire.py`: the checks, one or more per listener's note. Some still FAIL on the released track (the bed in
  drop 1 at 0.38 against 0.40, the bouncer over the bed in two spots, the kick punch in the hybrid by 0.2 dB, the
  breakdown bouncer's 7): each was weighed against the ear and left. `--selftest` renders the piece with both
  machines' figures 8 long, and the period checks must fail.
- `film/export.py`, `film/wheels.py`: the film. Three rings (the bar's 16 steps, the bouncer's 7, the pinger's 5),
  each note lighting its step; the lead traced round the outside, which against the bar draws a gear when it plays in
  7; a drifting record of past bars; the breakdown's melody as a figure in the middle. One sound, one visible thing,
  at the same moment and in the same place. (A first film, drawing the polymeter as interference fringes, passed
  every check and showed the listener "just a bunch of shakey lines": nothing on screen answered to a sound.)
- `film/check_safety.py`: photosensitivity checks on the encoded video: WCAG 2.3.1 flashes, and regular stripe
  patterns (our own margin on ITU-R BT.1702 and Ofcom's guidance, not a certification), self-tested on synthetic
  strobes and gratings.

From the repository root:

    python3 moire/moire.py                  # -> out/moire.wav (+ stems, score), about 2 min and 14 GB
    python3 moire/check_moire.py            # about 2.5 min
    python3 moire/film/export.py            # -> out/moire_film/curves.npz, events.json
    python3 moire/film/wheels.py --stills 67.7 203.3
    python3 moire/film/wheels.py --render --workers 4      # -> out/moire_film/wheels.mp4, about 3 min
    python3 moire/film/check_safety.py out/moire_film/wheels.mp4
    python3 moire/film/check_safety.py --selftest

Environment variables for listening to parts: `MOIRE_SOLO=lead,bounce` renders only those groups (lead, bounce,
ping, grid, bed, pad, fx, deco); `MOIRE_BED=basic|lush|analog|glass` swaps the bed voice (the listener chose glass
from these four, level-matched).
