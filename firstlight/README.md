# First Light

A chill synthwave piece (3:24, 106 bpm, F minor) built around one idea: a hook that assembles itself. Its melody is
heard first in fragments (its opening three notes, then two bars, three, four), each over chords that change what
the same notes mean, and only whole at the first drop. The hook's rhythm (two dotted 8ths, then a held note) is the
rhythm the sequence pulses from bar 1: the tune grows out of the accompaniment. Around it, a breathy counter-line, a
bridge with a second theme that is the hook's opening backwards, a descant over its last pass; the outro takes it
apart again. A soft plucked string carries the hook, over a round bass and a gentle kit.

It is the quiet counterpart to *Engine*, made with the same kit: where Engine's sync lead screams, everything here
was chosen by ear to stay out of the way. The listener, after it ran on repeat for ten minutes unnoticed: "pretty
good, very chill BGM ... very 'music to relax and study to' vibe."

A Claude agent made it over six rounds with one listener, who isn't a musician and chose between versions by ear.
It started as a one-minute study in the style of a synthwave album (*Low Orbit*, below) and became its own piece
when the listener suggested composing something original with those instruments. Every instrument is built from the
kernel.

    python3 firstlight/firstlight.py && python3 firstlight/check_firstlight.py      # about 3 min and 7 GB
    python3 firstlight/check_firstlight.py --selftest     # eighteen faults, each tripping its check (about 40 min)
    python3 firstlight/firstlight.py --form study1        # the studies it grew from (below): study1, study4

## The piece

| bars | time | what | the lines |
|---|---|---|---|
| 1-8 | 0:00 | intro: the pad, the sequence, the bass rising in | the hook's first bar, over Dbmaj9, then Fm9 |
| 9-24 | 0:18 | the groove; bongos in its second eight | its first two bars, twice, over other chords; then three, the counter-line's first three under them |
| 25-32 | 0:54 | the breakdown (no drums), then a build on Eb7sus | its first four bars, exposed |
| 33-48 | 1:12 | the first drop | the whole hook, its top note heard for the first time; again, with the counter-line |
| 49-64 | 1:49 | the bridge: half time, Cm7 and Eb (the first major chord) | the second theme on the breath, twice; the second time the string answers in its held notes |
| 65-80 | 2:25 | the second drop, a walking bass | hook and counter-line, twice; the last pass with a descant above |
| 81-88 | 3:01 | outro: the drums leave after four bars; no last hit | the hook's first two bars, its first, its last, the F held into the final chord |

The lines, and what the checks hold them to (each check has a planted fault it must catch):

- **The hook**: every fragment is its own opening, note for note; the statements grow, and the whole hook (its top
  note above all) waits for the first drop. Singable: a range of 16 semitones, 59% of its steps a tone or less.
- **The counter-line** (the breath voice): under the hook, moving while it holds and holding while it moves (88% of
  its notes start while the hook holds), never a minor second, tritone or major seventh from it.
- **The bridge**: half the kicks, two chords the drops never use, and a second theme that opens with the hook's
  opening backwards. In its second pass the string answers in each of the theme's held notes, in the hook's own
  rhythm: the roles of the drops reversed.
- **The descant**: long notes above the hook's last pass, rising by step (Ab F Bb Ab C Bb) to an Eb over the hook's
  top note, then down to the third; above the hook in every 16th they share.

Playing the second theme against the hook, which would have been neat, doesn't work: they cross and meet on seconds.

The new lines were tested against the listener's three favourite tracks of the album that started it
(`tools/originality.py`, every line bar by bar against every minute of each track, scored against random lines with
the same rhythm): of 600 comparisons, none beat the random lines' 95th percentile by two steps or more, and 13 by one
step, where chance alone gives about 30.

## How it got here

**Low Orbit**, a one-minute cold test (not in this repository; its instruments live on here): asked for "about a minute in the style of" a mid-tempo synthwave
album, the agent measured the album (tempo, loudness, kick punch, tonal balance, and its parts on separated stems)
and made it in three rounds. v1 was "more generic synthwave", where the album "has a lot of funk to it"; v2's pushed
bass had "good funk", and of three leads "pulse is best"; v3 answered "reminds me somewhat of indie game music" with
held seventh and sus chords, a layered bass, a sequence inside the pad's register and drums that move ("less like
game music, more complex and interesting now", but "too much bass funk"). Then the listener suggested breaking from
the reference: keep the instruments, compose something of our own.

**Study 1** (`--form study1`): the hook assembling itself in one minute, on Low Orbit's instruments, with
a bass of one note a beat. "The 0:36 arrival does work, as does the dawn mood", but the lead "too piercing and
synthy for this relaxing mood", "every instrument a bit too synth", "seems pretty basic".

**Study 2**: the whole hook on four lead voices, matched by loudness, on a gentler bed. "The string and flute (breath)
most promising. String is a bit too hard or loud, but the echo effect is nice. The gentler bed is nice. Bass
definitely needs to be less synth too."

**Study 3**: the string softened (a thumb-dark burst, its first 8 ms eased in, 2 dB down) and two less synthetic
basses. "Fingered one is not good, quite buzzy and too bright. 'Round' actually works though. The softer lead is
nice!" The measure that would have caught the buzz had been dropped from Low Orbit's checks: the bass's 300-3000 Hz
against its 30-300 Hz read -6.3 dB for the fingered bass, -20.9 for the round one. It's back, with the fingered bass
as its fault.

**Study 4** (`--form study4`): for "basic", composition, not sound: the counter-line, the bridge and its second
theme. "A good enough study. I think ready to make the full song."

## What the checks found along the way

- The sparser bass held its notes across the point halfway between kicks, which cost the kick its punch; a slower
  pump on the bass and a master at -11 LUFS brought it back (+13 dB, where Low Orbit's "hits fine").
- The last bass note stopped dead, dropping the whole mix 30 dB in half a second; the ending check had looked only
  for rises. In the full piece, the bass held from bar 5 stepped the intro up 11 dB in one bar; it now rises in.
- The drums' bar-to-bar change depends on section length: the step into the next section counts once in however many
  bars. The 16-bar sections read 0.54-0.64 dB with a change every eight bars; small fills and take-aways at every
  four-bar mark and Low Orbit's bongo cycle brought them to 0.79-0.84, under the 0.85 the 8-bar studies passed. The
  floor is 0.75 for 16-bar sections; static drums read 0.24-0.32.
- "Mid motion" (notes starting a beat in the accompaniment's middle) moved from 0.69 to 0.59 when only the random
  stream shifted, the sequence unchanged: it is noisy by about 0.1 near its 0.6 floor.
- The top-end check's floor moved from the synthwave album's -25 dB to -28: the voices the listener chose are darker.
- Refactors were proven by bit-identical renders: studies 1 and 4 still render bit for bit from the code that renders
  the full piece.

## The film

One unbroken shot, 3:24: a room with one window at dawn, the camera drifting slowly back from the glass. Everything
that moves in it is read from the score, one sound to one visible thing:

- a red light on a radio mast blinks the hook's rhythm (two short, one long) in every bar, from bar 1;
- each fragment of the hook lights a piece of the city: windows whose heights draw its notes, the crane, the lamps;
- the drums are the wind: with none (the intro, the breakdown) the water goes still and mirrors the city;
- the sun's rim clears the hills on the hook's top note (1:27), and the cat lifts its head;
- the counter-line is a ferry crossing, the bongos are wingbeats, the descant is a plane drawing its contour as a
  contrail;
- in the bridge, the string's answers flare the sun's light off the water onto the wall;
- the outro turns the city's lights off in the reverse order they came on, and the room's lamp, on from the first
  frame, goes off last, on the final note.

The room is ray cast, so the sun through the window lands only where it can. The cat is made of blended
signed-distance shapes. Everything is numpy, with no image or video models, and ffmpeg encodes it.

    python3 firstlight/firstlight.py                     # the music first: the film muxes out/firstlight.wav
    python3 firstlight/film.py --height 540 --jobs 5     # a quick cut at 960x540 -> out/firstlight_film_540.mp4
    python3 firstlight/film.py --jobs 5                  # the release, 1920x1080 -> out/firstlight_film.mp4
    python3 firstlight/film.py --still 87 130            # single frames -> out/firstlight_film_87.png, ...
    python3 firstlight/film.py --span 100 140            # only those seconds
    python3 firstlight/check_film.py out/firstlight_film.mp4      # its checks (below)
    python3 firstlight/check_film.py --selftest out/firstlight_film.mp4   # each check, broken once, must fail
    python3 firstlight/time_film.py --height 1080 87     # one frame's time and memory, before a long render

It is slow, because every pixel is ray cast on the CPU. The 1080p release (6112 frames) took 5 h 38 min on a 6-core
Ryzen 5 7600X at `--jobs 5`, about 3.3 s per frame per worker. Each worker holds about 2.1 GB, and its ffmpeg encoder
another 0.4 GB, so `--jobs 5` peaked at about 14 GB. Cap it, because an uncapped render that runs out of memory can
take the whole machine down:

    systemd-run --user --scope -p MemoryMax=17G -p MemorySwapMax=0 nice -n 19 python3 firstlight/film.py --jobs 5

A first try at `--jobs 6` under a 15 GB cap was stopped by the cap after 27 minutes: the encoders' memory hadn't
been counted.

The files:

| file | what |
|---|---|
| `film.py` | the score-to-picture map, the view outside, the camera, the render and encode |
| `room.py` | the room: one window, one sun, ray cast |
| `cat.py` | the cat: asleep on the window seat, curled, breathing; it lifts its head once |
| `props.py` | the mug, the plant, the lamp, the headphones and the paperback |
| `materials.py` | the surfaces: walls, ceiling, floor, the window's paint, the seat's cushion, the wood |
| `city.py` | the city across the water: buildings, their windows, the rows behind in the haze |
| `prims.py`, `shapes.py`, `ids.py` | ray-primitive intersections, 2D masks, material ids |
| `check_film.py` | the film's checks |
| `time_film.py` | time and memory for single frames |

### The film's checks

`check_film.py` runs three checks on the code and five on the encoded film. Its self-test breaks each one once
and must see it fail, including the whole film read against a score 3 frames late.

- **patch**: the sun's patches on the wall and the cushion, worked out forward as polygons, agree with the film's
  own backward test.
- **sun**: the room is lit from where the sun is drawn, within 1.5 px, in the code and in the encoded frames.
- **outro**: every outro note turns off at least one light the camera can see.
- **mast**: the mast light, frame by frame, correlates with the hook's rhythm (at least 0.9, at a lag of 0 frames).
- **lights**: each of the 45 window lights brightens within 1 frame of its note.
- **glitter**: the glitter on the water falls straight below the sun (within 4 px).
- **jumps**: no one-frame jump the score doesn't put there.

For photosensitivity, run the kit's `moire/film/check_safety.py` on the encoded film (flashes, and regular stripes
in 16 tiles, after ITU-R BT.1702). The release has no flashes, and no frame with stripes in more than 3 of the 16
tiles. Its "pattern" line still prints FAIL, from the broad-stripe test, which reads a layered landscape (sky,
hills, city, water) as bands across the screen. As that checker's docstring says, trust the tile test on a picture.
The stripe tiles were the binding constraint on the film: the window paint's texture, the reveal's sky light and the
direct sun's strength were each set by them.

### How the film got here

Mark asked what a film for the piece would be. The pitch was one shot of a window at dawn, easy to leave on, with
every event read from the score. A quick mockup in flat shapes drew four notes: the ferry "speeds up and slows
down" (it moved only on its notes; now it crosses at a steady speed and its lights carry them); white-noise
sparkle on the water was "too basic" (now glints where a ripple's slope sends the sun to the camera); the contrail
should fade (it does now); and the sunlight on the indoor wall didn't "make sense geometrically". That last note
made the room a ray-cast one, with the camera pulling back from the window and the cat asleep beneath it.

A helper agent sculpted the cat. Mark: "holy smokes that cat looks good", then "You know what the problem is now
though right?" The cat outclassed the room, so three more helpers brought up the props, the surfaces and the city,
each behind its own interface in its own file. A fresh agent with an open brief then reviewed the film. Its notes
gave the flock at the second drop, glints that stop where the water mirrors land, the lamp going off on the final
note, and the cat in the night glass; its one wrong claim (lights still on at 3:20) was checked against the frames
and dropped. Mark's last note, that the light off the water showed on the wall only "briefly at 2:22", made it
stay there whenever the sun is up and the water moves. Mark on the release: "no notes".
