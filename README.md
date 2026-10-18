# Opustek: making music with an agent that can't hear

A coding agent can write, render and measure music, but it can't hear it. You can, usually in a sentence: "quite
piercing at 1:08", "too much bass funk", "the saw lead is still plain". This repository is a kit for working that
way. It helps an agent turn a listener's few words into a measured cause, a check that would have caught it, and a
fix, and then keep the check so the fault can't quietly come back.

It was built by Opustek, Mark Lundeberg's Claude agent (Claude Opus 5.5, an AI model made by Anthropic), while
making two techno tracks with Mark listening to every version and sending notes. What's here:

- **`WORKFLOW.md`, *Making music you can't hear*:** the core. It's a field guide written for an agent: what it can
  do alone and what needs a person, how to run rounds with a listener, and tables of what the listener said, what
  the cause turned out to be, and what fixed it.
- **`tools/measures.py`:** the measures that stood in for hearing. They work on any WAV, so you can bounce a mix from
  whatever you make music with and have your agent measure it. They cover tempo and downbeat, tonal balance against
  a reference, phone survival, width, crest, the loudest second, kick punch, groove (notes per beat, where they land,
  swing), repetition and endings. Each one is self-tested on known cases, including noise, which is how we caught one
  that read white noise as a strong repeating pattern.
- **`tools/refscope/`:** for taking a reference mix apart into stems, grid, key, sections, voices and pitch, so you
  can measure a reference's bass on its own instead of guessing from the mix. It's calibrated on our own track,
  whose answers we know.
- **`tools/originality.py`:** checks whether your melody is someone else's, against what chance alone would give.
- **A from-scratch synthesizer** (numpy, scipy and a few hundred lines of C) and a one-minute template study. With
  these the whole loop runs in code, with no samples, plugins, music apps, or music or video models. That's how our
  tracks were made, but you don't need it to use the rest.

`AGENTS.md` says where everything is and how to run it. Point your agent there.

## The music it came from

- *Engine, Learning to Sing* (3:14): a hard-synced engine that learns to sing over techno; its film is an 1890s
  engineering drawing of the machine, running on the music.
- *Moiré* (4:51): two machines play out of step, 7 against 5, over a steady beat, and meet only every 35 bars.
- *Lantern* (1:03): a downtempo piece made from this repository by a new Claude agent with no memory of the other
  tracks, given one line and no reference. It's a good model of a first piece.

All three are here as code, each with its checks; the first two reproduce the released audio bit for bit and come
with their film renderers. Listen and watch, with the story of how each was made, at
**https://marklundeberg.com/opustek/**.

## What to expect

It takes rounds, not one prompt: Mark spent about three hours of attention on *Moiré*, over ten versions. The
listener's ear is the part that can't be automated, and the guide is mostly about how to use it well.

So far it has been tested only by new Claude agents on Mark's machine, with Mark as the listener. There have been
three cold tests, each by a new agent that had never seen our tracks, *Lantern* among them, and each one's debrief
went back into the kit. That means its calibrations come from one listener and mostly from electronic music between
90 and 132 bpm. A different ear or genre will need its own, and the guide shows how to build them. We'd love to hear
how it goes.

No reference recordings, or anything extracted from them, are included; the tools let you measure your own.

## A neighbour

[earshot](https://github.com/Leonxlnx/earshot) gives coding agents learned ears: pretrained audio models that can
say what a sound is ("a whoosh", "a keypress") and how close it is to good examples, aimed mainly at sound effects for
videos. This kit works the other way round: it can't name a sound, but it can find the cause behind a listener's
note, in music. The two fit together.

## Licence and credit

Code: 0BSD. The tracks and films: CC BY 4.0, credited to "Opustek (Mark Lundeberg's Claude agent)". Mark set the
briefs and answers for the result: corrections go to him.
