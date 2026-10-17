# Lantern

A one-minute downtempo piece (90 bpm, D major): a kalimba tune over an FM electric piano, a round bass and a soft
swung kit. It is this repository's cold test. A new Claude agent with no memory of the other tracks was given the
repository and one line, "make me some music, about a minute long", plus one rule: build every instrument from the
kernel, no borrowed sounds. It chose the genre, built every instrument (the kalimba by modal synthesis on the
partials of a beam clamped at one end, 1 : 6.27 : 17.55), wrote its checks and five planted faults, fixed what the
checks found, and sent it, on its first try. Nobody heard it until it was done. The first listener: "nice and
original, nice lofi BGM that could go on for hours."

It ran in its maker's setup, so it had his general working notes for Claude agents (how to work with him, not about
music); asked afterwards, it judged they shaped its manner more than its music.

    python3 lantern/lantern.py && python3 lantern/check_lantern.py      # about 12 s and 1.3 GB
    python3 lantern/check_lantern.py --selftest
    python3 lantern/film/film.py --render --workers 4                  # the explainer film, about 30 s

To run your own cold test, give a new agent a fresh clone and a one-line brief, and nothing else.
