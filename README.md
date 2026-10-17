# Opustek: techno made in code by an agent that can't hear

Two techno tracks and their music videos, made entirely in code by Opustek, Mark Lundeberg's Claude agent (Claude
Opus 5.5, an AI model made by Anthropic), with Mark listening to every version and sending notes. There are no
samples, plugins, music apps, or music or video models: the sound is numpy and scipy with a few hundred lines of C,
and the films are pycairo.

- *Engine, Learning to Sing* (3:14): a hard-synced engine that learns to sing over techno; its film is an 1890s
  engineering drawing of the machine, running on the music.
- *Moiré* (4:51): two machines play out of step, 7 against 5, over a steady beat, and meet only every 35 bars.

Listen and watch, with the story of how each was made, at **https://marklundeberg.com/opustek/**.

This repository is for doing it yourself, or more likely for pointing your own agent at:

- `WORKFLOW.md`, *Making music you can't hear*: a field guide written for an agent. What it can do alone, what needs
  a person, the tools that worked, and every place we stumbled.
- `AGENTS.md`: where everything is and how to run it.
- Both tracks as code that reproduces the released audio bit for bit, with their check suites and film renderers.
- The measures that stood in for hearing (each self-tested on known cases), a melody-originality test, and a toolset
  for taking a finished mix apart (stems, grid, key, sections, voices, pitch), calibrated on our own track.
- A one-minute template study to start from.

No reference recordings, or anything extracted from them, are included; the tools let you measure your own.

Expect it to take rounds, not one prompt: Mark spent about three hours of attention on *Moiré*, over ten versions.
The listener's ear is the part that can't be automated, and the guide is mostly about how to use it well.

Code: 0BSD. The tracks and films: CC BY 4.0, credited to "Opustek (Mark Lundeberg's Claude agent)". Mark set the
briefs and answers for the result: corrections go to him.
