"""First Light: a hook that assembles itself. Mid-tempo synthwave, 106 bpm, F minor, 3:24. Its melody is heard first
as fragments (its opening three notes, then two bars, three, four) and only whole at the first drop, while the
harmony under each fragment changes what it means. The hook's rhythm (two dotted 8ths, then a held note) is the rhythm
the sequence has been pulsing since bar 1: the tune grows out of the accompaniment. Then the lines around it: a
counter-line under it, a bridge with a second theme (the hook's opening backwards) that the string answers, a descant
over its last pass; and the outro takes it apart again.

On the instruments from Low Orbit (a cold-test study made from this repository), brought in here so this file stands
alone, and the voices the listener chose in four studies: a soft plucked string for the hook, a breathy near-sine for
the counter-line and the descant, a round bass, a gentler bed. Every instrument is built on the kernel (dsp.c /
dsp.py) and numpy.

  python3 firstlight/firstlight.py                -> out/firstlight.wav, _stems.npz, _score.json (about 2 min, 7 GB)
  python3 firstlight/firstlight.py --break NAME   the same with one fault (see BREAKS), for the self-test
  python3 firstlight/firstlight.py --form study1|study4   the studies, -> out/firstlight_study1.wav, ...
  python3 firstlight/firstlight.py --lead felt --drums brush --space dark --tape light --lufs -14
                                                  the sound's alternatives (the score unchanged), -> a wav named for them

The full form (88 bars):
  1-8    intro: the pad on Dbmaj9 and Fm9, the sequence alone, the bass rising in; the hook's first bar, twice
  9-24   the groove: its first two bars, twice, over other chords; then three, over its own, the counter-line's first
         three bars under them; bongos in the second eight
  25-32  the breakdown (no drums, the hook's first four bars exposed) and a build on Eb7sus with a roll
  33-48  the first drop: the whole hook, its top note (C6) and its way home heard for the first time; then again with
         the counter-line
  49-64  the bridge, half time, on new ground (Cm7, Eb): the breath's second theme, twice; the second time the string
         answers in its held notes
  65-80  the second drop, over a walking bass: hook and counter-line, twice; the last pass with the descant over it
  81-88  outro: the hook's first two bars, its first, its last, the F held into the final chord; no last hit
"""
import argparse
import os
import sys

import numpy as np
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]
import dsp                      # the kernel only
import kit                      # plumbing: place, pan, duck, master, write_outputs

NAME = 'firstlight'
SR = 48000
BPM = 106.0
BEAT = 60 / BPM
BAR = 4 * BEAT
NBARS, TAIL = 28, 4.5
HZ = lambda m: 440.0 * 2 ** ((m - 69) / 12)
tb = lambda bar, beat=0.0: (bar - 1) * BAR + beat * BEAT
R = lambda a, b: list(range(a, b + 1))
KEY_PCS = {5, 7, 8, 10, 0, 1, 3}          # F natural minor

# --- the score, as data -------------------------------------------------------------------------------------------
# chords: bass root (MIDI) and the pad's voicing. Dbmaj9, Fm9 and Ab6 share most of their upper notes (F Ab C Eb):
# the bass under them is what changes, and with it what the same melody means.
CH = {'Fm9': (41, [56, 60, 63, 67]), 'Dbmaj9': (37, [53, 56, 60, 63]), 'Ab6': (44, [56, 60, 63, 65]),
      'Bbm9': (34, [56, 60, 61, 65]), 'Eb7sus': (39, [56, 58, 61, 63]),
      'Cm7': (36, [55, 58, 60, 63]), 'Eb': (39, [55, 58, 63, 67])}     # the bridge's new ground (Eb: the first major)
CHORD_TONES = {'Fm9': {5, 8, 0, 3, 7}, 'Dbmaj9': {1, 5, 8, 0, 3}, 'Ab6': {8, 0, 3, 5}, 'Bbm9': {10, 1, 5, 8, 0},
               'Eb7sus': {3, 8, 10, 1}, 'Cm7': {0, 3, 7, 10}, 'Eb': {3, 7, 10}}
HOOK_CHORDS = ['Fm9', 'Dbmaj9', 'Fm9', 'Eb7sus', 'Dbmaj9', 'Bbm9', 'Ab6', 'Fm9']
CHORDS = (['Dbmaj9', 'Dbmaj9', 'Fm9', 'Fm9']                                               # 1-4
          + ['Dbmaj9', 'Dbmaj9', 'Ab6', 'Bbm9', 'Dbmaj9', 'Dbmaj9', 'Bbm9', 'Eb7sus']     # 5-12
          + HOOK_CHORDS[:4] + HOOK_CHORDS                                                  # 13-16, 17-24
          + ['Dbmaj9', 'Dbmaj9', 'Fm9', 'Fm9'])                                            # 25-28

INTRO, GROOVE, BREAKDOWN, DROP, HOME = R(1, 4), R(5, 12), R(13, 16), R(17, 24), R(25, 28)
KICK_BARS = GROOVE + DROP + [25, 26]
SNARE_BARS = GROOVE + DROP + [25, 26]
HAT_BARS = GROOVE + DROP + [25, 26]
OPEN_HAT_BARS = R(9, 12) + DROP
SHAKER_BARS = R(9, 12) + DROP + [25, 26]
CLAP_BARS = DROP
STAB_BARS = DROP
BASS_BARS = R(3, 28)
CRASHES = [(5, 1.0), (17, 1.0), (21, 0.6), (28, 0.9)]
RISERS = [4, 16]
PUNCT = [(8, 0.5), (20, 0.6)]             # a take-away at each section's four-bar mark: toms, a reversed cymbal, a crash

# the hook: (bar of the hook, beat, MIDI, beats). Each bar opens with the sequence's own rhythm (0, 0.75, 1.5): two
# dotted 8ths and a held note. Bars 1-4 rise and settle on C, open; bars 5-8 answer a third higher, reach C6 once,
# and come home to F.
HOOK = [(1, 0, 72, 0.75), (1, 0.75, 75, 0.75), (1, 1.5, 77, 2.5),
        (2, 0.5, 75, 0.5), (2, 1.0, 73, 0.5), (2, 1.5, 72, 1.0), (2, 2.75, 68, 1.25),
        (3, 0, 72, 0.75), (3, 0.75, 75, 0.75), (3, 1.5, 79, 1.5), (3, 3.0, 77, 1.0),
        (4, 0, 75, 2.0), (4, 2.0, 73, 1.0), (4, 3.0, 72, 1.0),
        (5, 0, 75, 0.75), (5, 0.75, 79, 0.75), (5, 1.5, 80, 2.5),
        (6, 0.5, 79, 0.5), (6, 1.0, 77, 0.5), (6, 1.5, 75, 1.0), (6, 2.75, 72, 1.25),
        (7, 0, 77, 0.75), (7, 0.75, 80, 0.75), (7, 1.5, 84, 2.0), (7, 3.5, 82, 0.5),
        (8, 0, 80, 1.0), (8, 1.0, 79, 1.0), (8, 2.0, 77, 2.0)]
# where the hook is heard: (first bar, which bars of the hook, level). Its statements grow (1, 1, 2, 2, 4, 8 bars);
# at home the opening once more and the last bar, its F held into the final chord.
STATEMENTS = [(2, [1], 0.55), (4, [1], 0.65), (7, [1, 2], 0.85), (11, [1, 2], 0.85), (13, [1, 2, 3, 4], 0.8),
              (17, list(range(1, 9)), 1.0), (25, [1], 0.7), (27, [8], 0.85)]
DOUBLED = DROP                             # the lead's octave-down layer while the whole hook plays

# the bass: sparser than Low Orbit's (whose 1.44 notes a beat read as "too much bass funk"): two one-bar figures,
# (16th step, interval, length in 16ths), 4 notes a bar, held notes, one push onto an 'a' a bar
BASS_FIGS = [[(0, 'R', 6), (7, 'R', 3), (10, '5', 2), (14, 'O', 2)],
             [(0, 'R', 3), (3, 'R', 4), (8, 'R', 5), (14, 'ap', 2)]]
BASS_WALK = [[(0, 'R', 4), (6, '5', 2), (8, 'O', 3), (12, '5', 2), (14, 'ap', 2)],    # the return: it walks a little
             [(0, 'R', 3), (3, 'R', 3), (8, '5', 2), (10, 'O', 4), (14, 'ap', 2)]]
BASS_BUSY = [[(0, 'R', 3), (3, 'R', 1), (6, 'O', 2), (10, '5', 1), (11, '5', 4)],          # Low Orbit's, for the fault
             [(0, 'R', 2), (5, 'R', 1), (6, 'O', 2), (9, 'O', 1), (11, 'R', 3), (15, 'ap', 1)]]
SEQ_STEP = 3                               # the sequence: every third 16th, three against four, from bar 1
SEQ_FIG = [0, 2, 1, 3, 2, 1]               # indices into the pad's voicing
STAB_FIG = [(3, 0.9), (7, 0.7), (11, 1.0)]  # the stabs: three a bar, on the 'a', in the drop only
# Low Orbit's bongos, high (1) and low (0): a four-bar cycle that thickens toward its end, never on the beat (16th, high)
BONGO = [[(3, 1), (6, 0), (10, 1), (14, 0)],
         [(2, 1), (7, 1), (11, 0), (13, 1), (15, 0)],
         [(3, 1), (6, 0), (9, 1), (10, 1), (14, 0)],
         [(2, 1), (3, 1), (7, 0), (11, 1), (13, 0), (14, 1), (15, 0)]]

LEAD_VOICES = {'pulse': "study 1's lead, Low Orbit's pulse: a resonant ladder struck on each note (heard as 'too piercing')",
               'epiano': 'a tine electric piano: two-operator FM, its index falling from a bark to a soft core',
               'pluck': 'a soft plucked string: Karplus-Strong from a dark noise burst, fully averaged, ringing out',
               'breath': 'a breathy, flute-like voice: a sine with a little 2nd and 3rd, breath noise, a slow attack',
               'felt': 'a felt piano: the stretched partials of two strings a cent apart, a soft felt hammer, a damper'}
LEAD_LUFS = -9.2                           # every voice matched to the pulse lead's own loudness (BS.1770, dry)
LEAD_TRIM = {'pluck': -2.0}                # then the string, chosen in study 2, 2 dB down: "a bit too hard or loud"
LEAD_TRIM.update(felt=-2.0, epiano=-2.0)   # (the sound A/B: the other struck voices at the string's own level)
COUNTER_TRIM = -4.0                        # the counter-line (the breath voice) sits this far under the lead's loudness
DESCANT_TRIM = -5.0                        # the descant (the breath again, above the hook), further under
BASS_VOICES = {'layered': "Low Orbit's: a ducked sine sub, two saws through a struck resonant ladder, grit on the accents",
               'fingered': 'a plucked low string: Karplus-Strong from a thumb-soft burst, lightly driven, over a sine sub',
               'round': 'a mellow near-sine (a little 2nd and 3rd) with a plucked envelope and no resonance'}
BASS_LUFS = -16.7                          # every bass matched to the layered one's loudness (body and sub at their gains)
BASS_TRIM = {'round': -0.7}                # then the round one, chosen in study 3 ("'round' actually works"), a touch down
                                           # so the kick keeps its punch (its weight sits where the kick is)
# the sound A/B (the score unchanged): the alternatives to each part of the sound, the first of each the piece's own
DRUM_KITS = {'soft': 'the soft kit: the punchy kick, the 80s snare dry, without its gated room',
             'brush': 'the punchy kick and a brushed snare: a slap of grainy noise, a little head, the bristles settling',
             'brush-round': 'the brushed snare and a rounder kick: a lower, slower sweep, no click, a longer body'}
SPACES = {'room': (3.8, 4000, 0.5), 'dark': (4.8, 2400, 0.6), 'close': (2.6, 3800, 0.38)}  # (rt60, damping Hz, return)
TAPES = {'none': None,                     # (wow %, flutter %, drive, low-pass Hz, head bump dB, hiss dBFS)
         'light': (0.08, 0.03, 0.5, 14000, 1.0, -68), 'medium': (0.2, 0.05, 0.9, 10000, 1.5, -61)}
KICK_GAIN, LUFS_TARGET = 1.0, -11.0           # 0.5 dB under Low Orbit: the held bass costs the kick some punch
BRUSH_GAIN = 1.56                          # the brushed snare matched to the dry snare's loudness (BS.1770, 8 hits)
CLIP = 0.6                                 # the drum bus clips above this fraction of the kick's peak

BREAKS = {
    'sharp-lead': 'the lead tuned 30 cents sharp',
    'wrong-note': 'an E natural in the hook, out of F minor',
    'fragment-altered': "the groove's second statement changes a note of the hook",
    'early-climax': "the hook's top note, C6, played in the groove",
    'buried-lead': 'the lead 14 dB down, under the pad and the sequence',
    'busy-bass': "Low Orbit's bass figures, 1.44 notes a beat",
    'no-sequence': 'the sequence muted',
    'static-drums': 'no percussion, no fills beyond the section ends, every hit at one level',
    'kick-in-breakdown': 'the kick keeps playing through the breakdown',
    'bright-master': 'a +9 dB shelf above 5 kHz on the master',
    'boom-at-end': 'a stray kick in the tail, out of the fade',
    'flat-bass': 'the bass tuned 30 cents flat',
    'buzzy-bass': "the fingered string bass, heard as 'quite buzzy and too bright, worse than the layered one'",
    'counter-clash': "study 4: the counter-line shadows the hook an octave down, note for note",
    'buried-counter': 'study 4: the counter-line 14 dB down',
    'four-on-floor-bridge': "study 4: the bridge keeps the drops' kick and snare",
    'descant-below': 'the full piece: the descant an octave down, through the hook instead of over it',
    'answer-clash': "the full piece: the string's answers two beats early, on the second theme's moving notes",
}


class Form:
    """A song form: what plays in each bar. Every bar-dependent choice in render() reads one of these."""

    def __init__(self, **kw):
        self.answers, self.descant, self.bongos, self.bass_swell = [], [], [], None
        self.__dict__.update(kw)

    def kind(self, bar):
        """The kind of section a bar is in: intro, groove, breakdown, drop, bridge or home."""
        for name, bars in self.sections.items():
            if bar in bars:
                return self.kinds.get(name, name)
        return 'home'


STUDY1 = Form(
    name='study1', nbars=NBARS, chords=CHORDS,
    sections={'intro': INTRO, 'groove': GROOVE, 'breakdown': BREAKDOWN, 'drop': DROP, 'home': HOME}, kinds={},
    main=['groove', 'drop'], peak='drop',
    kick={b: [0, 1, 2, 3] for b in KICK_BARS}, snare={b: [1, 3] for b in SNARE_BARS},
    hats=HAT_BARS, open_hats=OPEN_HAT_BARS, shaker=SHAKER_BARS, clap=CLAP_BARS, stabs=STAB_BARS,
    bass={b: ('held' if b in BREAKDOWN or b in (27, 28) else 'fig') for b in BASS_BARS},
    statements=STATEMENTS, last_hold=6.0, counter=[], doubled=[(DOUBLED[0], DOUBLED[-1])], final_hit=True,
    crashes=CRASHES, risers=RISERS, punct=PUNCT,
    fills=[(12, [3, 3.25, 3.5, 3.75], [0.35, 0.5, 0.65, 0.85]), (24, [3.5, 3.75], [0.5, 0.75]),
           (16, [k * 0.5 for k in range(4)] + [2 + k * 0.25 for k in range(8)], list(np.linspace(0.15, 0.9, 12)))],
    rolls=[16], toms=[(12, [(2.0, 50), (2.5, 47), (3.0, 45)]), (24, [(2.5, 52), (3.0, 48), (3.25, 45)])],
    seq_stop=27, dips=[(13, 16)],
    pad_fc=[(1, 600), (5, 2800), (13, 2600), (13.5, 3200), (16.5, 3800), (17, 3000), (25, 3000), (28, 1800), (None, 600)],
    fault_statement={'fragment-altered': 3, 'early-climax': 3, 'wrong-note': 5})


# study 4: the new material for the full piece. The whole hook with a counter-line (the breath voice) under it; a
# bridge on new ground (Cm7, and Eb, the first major chord) in half time, where the breath carries a second theme,
# the hook's opening backwards; then the hook and the counter-line together over a bass that walks a little.
# The counter-line, (bar of the hook, beat, MIDI, beats): below the hook, moving while it holds, holding while it
# moves, climbing to meet its top note, and ending a sixth under it.
COUNTER = [(0, 1.75, 63, 0.75), (0, 2.5, 65, 0.5), (0, 3.0, 67, 1.0),
           (1, 0, 65, 2.5), (1, 3.0, 63, 1.0),
           (2, 1.75, 60, 0.75), (2, 2.5, 63, 0.5), (2, 3.25, 65, 0.75),
           (3, 0.5, 61, 0.5), (3, 1.0, 63, 0.5), (3, 1.5, 68, 1.5), (3, 3.5, 63, 0.5),
           (4, 1.75, 65, 0.75), (4, 2.5, 68, 0.5), (4, 3.0, 72, 1.0),
           (5, 0, 70, 2.5), (5, 3.0, 68, 1.0),
           (6, 1.75, 72, 0.75), (6, 2.5, 75, 0.5), (6, 3.0, 77, 1.0),
           (7, 0, 72, 1.0), (7, 1.0, 70, 1.0), (7, 2.0, 68, 2.0)]
# the second theme: the hook's opening (C Eb F, rising) backwards (F Eb C, falling), in the same rhythm
THEME_B = [(0, 0, 77, 0.75), (0, 0.75, 75, 0.75), (0, 1.5, 72, 2.5),
           (1, 0.5, 73, 0.5), (1, 1.0, 75, 0.5), (1, 1.5, 77, 1.5), (1, 3.0, 80, 1.0),
           (2, 0, 79, 0.75), (2, 0.75, 77, 0.75), (2, 1.5, 75, 2.5),
           (3, 0.5, 72, 0.5), (3, 1.0, 75, 0.5), (3, 1.5, 79, 1.5), (3, 3.0, 82, 1.0),
           (4, 0, 80, 0.75), (4, 0.75, 77, 0.75), (4, 1.5, 73, 2.5),
           (5, 0.5, 75, 0.5), (5, 1.0, 77, 0.5), (5, 1.5, 80, 1.5), (5, 3.0, 77, 1.0),
           (6, 0, 79, 0.75), (6, 0.75, 75, 0.75), (6, 1.5, 82, 2.5),
           (7, 0, 80, 1.0), (7, 1.0, 79, 1.0), (7, 2.0, 75, 2.0)]
_D1, _BR, _D2 = R(3, 10), R(11, 18), R(19, 26)
STUDY4 = Form(
    name='study4', nbars=27,
    chords=['Dbmaj9', 'Dbmaj9'] + HOOK_CHORDS + ['Bbm9', 'Bbm9', 'Cm7', 'Cm7', 'Dbmaj9', 'Dbmaj9', 'Eb', 'Eb']
           + HOOK_CHORDS + ['Fm9'],
    sections={'intro': R(1, 2), 'drop1': _D1, 'bridge': _BR, 'drop2': _D2, 'home': [27]},
    kinds={'drop1': 'drop', 'drop2': 'drop'}, main=['drop1', 'drop2'], peak='drop2',
    kick={**{b: [0, 1, 2, 3] for b in _D1 + _D2}, **{b: [0, 2.5] for b in _BR}},       # the bridge in half time
    snare={**{b: [1, 3] for b in _D1 + _D2}, **{b: [2] for b in _BR}},
    hats=_D1 + _BR + _D2, open_hats=_D2, shaker=R(7, 10) + _D2, clap=_D2, stabs=_D2,
    bass={**{b: 'fig' for b in _D1}, **{b: 'held' for b in _BR}, **{b: 'walk' for b in _D2}, 27: 'held'},
    statements=[(3, list(range(1, 9)), 0.95), (19, list(range(1, 9)), 1.0)], last_hold=6.0,
    counter=[(3 + hb, q, m, d, 0.8) for hb, q, m, d in COUNTER] + [(11 + hb, q, m, d, 1.0) for hb, q, m, d in THEME_B]
            + [(19 + hb, q, m, d, 0.95) for hb, q, m, d in COUNTER],
    doubled=[], final_hit=True, crashes=[(3, 1.0), (11, 0.5), (19, 1.0), (23, 0.6), (27, 0.9)], risers=[2, 18],
    punct=[(6, 0.5), (22, 0.6)],
    fills=[(10, [3, 3.25, 3.5, 3.75], [0.35, 0.5, 0.65, 0.85]), (26, [3.5, 3.75], [0.5, 0.75]),
           (18, [k * 0.5 for k in range(4)] + [2 + k * 0.25 for k in range(8)], list(np.linspace(0.15, 0.9, 12)))],
    rolls=[18], toms=[(10, [(2.0, 50), (2.5, 47), (3.0, 45)]), (26, [(2.5, 52), (3.0, 48), (3.25, 45)])],
    seq_stop=27, dips=[],
    pad_fc=[(1, 900), (3, 3000), (11, 2400), (18.5, 3600), (19, 3000), (27, 1800), (None, 600)],
    fault_statement={'fragment-altered': 1, 'early-climax': None, 'wrong-note': 1})
# the full piece. Its first half is study 1's arc, stretched (the hook assembles: one bar, two, three, four, whole);
# its second half is study 4's (the counter-line, the bridge), each heard twice and changed the second time; the
# outro takes the hook apart again. Two new lines:
# the string's answers in the bridge's second pass, (bar of the theme, beat, MIDI, beats): in each of the second
# theme's held notes, the hook's rhythm (two dotted 8ths and a note) rising, so the roles of the drops are reversed
ANSWERS = [(0, 2.0, 77, 0.75), (0, 2.75, 80, 0.75), (0, 3.5, 82, 0.5),
           (2, 2.0, 79, 0.75), (2, 2.75, 82, 0.75), (2, 3.5, 84, 0.5),
           (4, 2.0, 77, 0.75), (4, 2.75, 80, 0.75), (4, 3.5, 82, 0.5),
           (6, 2.0, 70, 0.75), (6, 2.75, 75, 0.75), (6, 3.5, 79, 0.5)]
# the descant over the last pass of the hook, (bar of the hook, beat, MIDI, beats): long notes above it, rising by
# step (Ab F Bb Ab C Bb) to an Eb over the hook's own top note, then down to the third
DESCANT = [(0, 0, 80, 5.5), (1, 1.5, 77, 2.5), (2, 0, 82, 4.0), (3, 0, 80, 4.0), (4, 0, 84, 5.5), (5, 1.5, 82, 2.5),
           (6, 0, 87, 4.0), (7, 0, 84, 2.0), (7, 2.0, 80, 2.0)]
BRIDGE_CHORDS = ['Bbm9', 'Bbm9', 'Cm7', 'Cm7', 'Dbmaj9', 'Dbmaj9', 'Eb', 'Eb']
_G, _BD, _BU, _F1, _BG, _F2, _OU = R(9, 24), R(25, 28), R(29, 32), R(33, 48), R(49, 64), R(65, 80), R(81, 88)
FULL = Form(
    name='full', nbars=88,
    chords=(['Dbmaj9', 'Dbmaj9', 'Fm9', 'Fm9'] * 2                                                     # 1-8
            + ['Dbmaj9', 'Dbmaj9', 'Ab6', 'Bbm9', 'Dbmaj9', 'Dbmaj9', 'Bbm9', 'Eb7sus']                  # 9-16
            + ['Fm9', 'Dbmaj9', 'Fm9', 'Eb7sus', 'Dbmaj9', 'Dbmaj9', 'Bbm9', 'Eb7sus']                   # 17-24
            + HOOK_CHORDS[:4] + ['Dbmaj9', 'Bbm9', 'Eb7sus', 'Eb7sus']                                   # 25-32
            + HOOK_CHORDS * 2 + BRIDGE_CHORDS * 2 + HOOK_CHORDS * 2                                      # 33-80
            + ['Dbmaj9', 'Dbmaj9', 'Fm9', 'Fm9'] * 2),                                                   # 81-88
    sections={'intro': R(1, 8), 'groove': _G, 'breakdown': _BD, 'build': _BU, 'drop1': _F1, 'bridge': _BG,
              'drop2': _F2, 'outro': _OU},
    kinds={'drop1': 'drop', 'drop2': 'drop', 'outro': 'home'}, main=['groove', 'drop1', 'drop2'], peak='drop2',
    kick={**{b: [0, 1, 2, 3] for b in _G + [31, 32] + _F1 + _F2 + R(81, 84)}, **{b: [0, 2.5] for b in _BG}},
    snare={**{b: [1, 3] for b in _G + [31] + _F1 + _F2 + R(81, 84)}, **{b: [2] for b in _BG}},
    hats=_G + [31, 32] + _F1 + _BG + _F2 + R(81, 84), open_hats=R(17, 24) + _F1 + _F2,
    shaker=R(17, 24) + R(41, 48) + R(57, 64) + _F2, clap=_F2, stabs=_F2,
    bongos=[b for b in R(17, 24) + R(41, 48) + _F2 if b not in (24, 48, 80)],     # Low Orbit's, in each second pass
    bass={**{b: 'held' for b in R(5, 8) + _BD + [29, 30] + _BG + _OU}, **{b: 'fig' for b in _G + [31, 32] + _F1},
          **{b: 'walk' for b in _F2}},
    statements=[(6, [1], 0.55), (8, [1], 0.65), (11, [1, 2], 0.85), (15, [1, 2], 0.85), (17, [1, 2, 3], 0.9),
                (25, [1, 2, 3, 4], 0.8), (33, list(range(1, 9)), 0.95), (41, list(range(1, 9)), 1.0),
                (65, list(range(1, 9)), 1.0), (73, list(range(1, 9)), 1.0),
                (81, [1, 2], 0.75), (85, [1], 0.65), (87, [8], 0.8)], last_hold=6.0,
    counter=([(17 + hb, q, m, d, 0.7) for hb, q, m, d in COUNTER if hb < 3]                  # a first glimpse of it
             + [(41 + hb, q, m, d, 0.8) for hb, q, m, d in COUNTER]
             + [(st + hb, q, m, d, 1.0) for st in (49, 57) for hb, q, m, d in THEME_B]
             + [(st + hb, q, m, d, 0.95) for st in (65, 73) for hb, q, m, d in COUNTER]),
    answers=[(57 + hb, q, m, d, 0.6) for hb, q, m, d in ANSWERS],
    descant=[(73 + hb, q, m, d, 0.8) for hb, q, m, d in DESCANT],
    doubled=[], final_hit=False,           # it ends softly: no last kick after four bars without drums
    crashes=[(9, 0.8), (25, 0.6), (33, 1.0), (41, 0.6), (49, 0.5), (65, 1.0), (73, 0.8), (81, 0.6)],
    risers=[8, 32, 64], punct=[(12, 0.5), (36, 0.6), (44, 0.5), (68, 0.6), (76, 0.6)],
    # something at every four-bar mark of the long sections (the album measured had a change every four bars or so;
    # with them every eight, the drums' bar-to-bar change read 0.54-0.64 dB, under the 0.85 that study 1 passed)
    fills=[(24, [3, 3.25, 3.5, 3.75], [0.35, 0.5, 0.65, 0.85]), (48, [3.5, 3.75], [0.5, 0.75]),
           (80, [3.5, 3.75], [0.5, 0.75])]
          + [(b, [3.5, 3.75], [0.4, 0.6]) for b in (16, 40, 72)]
          + [(b, [2.75, 3.25, 3.5, 3.75], [0.3, 0.4, 0.55, 0.7]) for b in (20, 56)]
          + [(b, [k * 0.5 for k in range(4)] + [2 + k * 0.25 for k in range(8)], list(np.linspace(0.15, 0.9, 12)))
             for b in (32, 64)],
    rolls=[32, 64], toms=[(24, [(2.0, 50), (2.5, 47), (3.0, 45)]), (48, [(2.5, 52), (3.0, 48), (3.25, 45)]),
                          (80, [(2.5, 52), (3.0, 48), (3.25, 45)])],
    seq_stop=88, dips=[(25, 28)], bass_swell=(5, 9),
    pad_fc=[(1, 600), (9, 2800), (17, 3000), (25, 2600), (28.5, 3200), (32.5, 3800), (33, 3000), (49, 2400),
            (64.5, 3600), (65, 3000), (81, 2600), (88, 1800), (None, 600)],
    fault_statement={'fragment-altered': 3, 'early-climax': 3, 'wrong-note': 7})
FORMS = {'study1': STUDY1, 'study4': STUDY4, 'full': FULL}


def chord(bar, F=STUDY1):
    name = F.chords[bar - 1]
    return (name,) + CH[name]


def lead_events(brk=None, F=STUDY1):
    """Every note the lead plays, with the statement it belongs to and its place in the hook."""
    ev = []
    fs = F.fault_statement
    for si, (start, bars, vel) in enumerate(F.statements):
        for (hb, q, m, d) in HOOK:
            if hb not in bars:
                continue
            bar = start + bars.index(hb)
            if si == len(F.statements) - 1 and hb == 8 and q == 2.0 and F.last_hold:
                d = F.last_hold                            # home: the F held into the final chord
            if brk == 'fragment-altered' and si == fs.get(brk) and hb == 2 and q == 1.5:
                m = 70
            if brk == 'early-climax' and si == fs.get(brk) and hb == 1 and q == 1.5:
                m = 84
            if brk == 'wrong-note' and si == fs.get(brk) and hb == 6 and q == 1.5:
                m = 76
            anchor = np.ceil(q) if q % 1 >= 0.75 else q
            ev.append({'t': tb(bar, q), 'midi': m, 'dur': d * BEAT, 'vel': vel, 'bar': bar, 'beat': float(q),
                       'anchor_bar': bar + int(anchor // 4), 'statement': si, 'hook_bar': hb, 'hook_beat': q,
                       'chord': F.chords[min(F.nbars, bar + int(anchor // 4)) - 1]})
    for e in line_events(F.answers, F):
        if brk == 'answer-clash':      # the fault: each answer moved onto the theme's moving notes, two beats early
            e = dict(e, t=e['t'] - 2 * BEAT, beat=e['beat'] - 2)
        ev.append(dict(e, statement=None, hook_bar=None, hook_beat=None, anchor_bar=e['bar']))
    return sorted(ev, key=lambda e: e['t'])


def line_events(line, F):
    """A line's notes from the form, (bar, beat, MIDI, beats, level), as events with the chord each one belongs to."""
    ev = []
    for (bar, q, m, d, vel) in line:
        anchor = np.ceil(q) if q % 1 >= 0.75 else q
        ab = min(F.nbars, bar + int(anchor // 4))
        ev.append({'t': tb(bar, q), 'midi': m, 'dur': d * BEAT, 'vel': vel, 'bar': bar, 'beat': float(q),
                   'chord': F.chords[ab - 1]})
    return sorted(ev, key=lambda e: e['t'])


def counter_events(F=STUDY1, brk=None):
    """The counter-line's notes (the breath voice), from the form: (bar, beat, MIDI, beats, level)."""
    counter = F.counter
    if brk == 'counter-clash':      # the fault: in the drops, the hook itself an octave down instead
        drops = {b for k, bars in F.sections.items() if F.kinds.get(k, k) == 'drop' for b in bars}
        counter = [c for c in counter if c[0] not in drops] + [
            (start + bars.index(hb), q, m - 12, d, 0.8) for start, bars, _ in F.statements for hb, q, m, d in HOOK if hb in bars]
    return line_events(counter, F)


def descant_events(F=STUDY1, brk=None):
    """The descant's notes (the breath again, above the hook's last pass)."""
    return line_events([(b, q, m - 12 if brk == 'descant-below' else m, d, v) for b, q, m, d, v in F.descant], F)


def bass_events(brk=None, F=STUDY1):
    """The bass's notes, (time, MIDI, seconds, accent, bar)."""
    ev = []
    for b in sorted(F.bass):
        name, root, _ = chord(b, F)
        if F.bass[b] == 'held':
            ev.append((tb(b), root, (4.0 if b < F.nbars else 7.0) * BEAT, 0.8, b))
            continue
        nxt = chord(min(b + 1, F.nbars), F)[1]
        iv = {'R': 0, 'O': 12, '5': 7}
        figs = BASS_BUSY if brk == 'busy-bass' else BASS_WALK if F.bass[b] == 'walk' else BASS_FIGS
        for (k, what, n16) in figs[(b - 1) % 2]:
            if what == 'ap':     # a scale step below the next root
                m = nxt - 2 if (nxt - 2) % 12 in KEY_PCS else nxt - 1
                m += 12 * round((root - m) / 12)
            else:
                m = root + iv[what]
            ev.append((tb(b, k * 0.25), m, n16 * 0.25 * BEAT, 1.0 if k % 4 == 3 else 0.85, b))
    return ev


# --- the instruments (from Low Orbit, kept as they were heard there unless noted) ----------------------------------

def end_fade(y, s=0.03):
    """A hit cut off mid-decay clicks: fade its last s seconds to zero."""
    n = min(len(y), int(s * SR))
    y = y.copy()
    y[-n:] *= np.linspace(1, 0, n)
    return y


def kick_hit(rng):
    """A punchy kick: a sine sweeping from 230 Hz down to 50 Hz in a few tens of ms, half its level by about 60 ms
    and a tenth by about 200 ms, gone by 0.32 s (its tail, not the bass, was what filled the gap between kicks), a
    short filtered click, saturated a little for weight."""
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 50 + 140 * np.exp(-t / 0.025) + 40 * np.exp(-t / 0.004)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.085) * np.minimum(1, t / 0.0008)
    click = dsp.butter(rng.standard_normal(n) * np.exp(-t / 0.0018), 4000, SR, 'low')
    y = np.tanh(1.8 * (body + 0.2 * click)) / np.tanh(1.8)
    y *= np.clip((0.32 - t) / 0.14, 0, 1)
    return dsp.butter(y, 9000, SR, 'low')


def snare_hit(rng, vel=1.0, gated=True):
    """An 80s snare: two head tones with a small pitch drop, band-passed wires, and a gated reverb (a big room cut
    off at about a quarter of a second). Returns (L, R)."""
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    head = (0.7 * np.sin(2 * np.pi * np.cumsum(180 + 40 * np.exp(-t / 0.01)) / SR) * np.exp(-t / 0.05)
            + 0.3 * np.sin(2 * np.pi * 330 * t) * np.exp(-t / 0.03))
    nz = rng.standard_normal(n)
    wires = dsp.butter(dsp.butter(nz, 1200, SR, 'high', 2), 7500, SR, 'low', 2) * np.exp(-t / 0.075)
    dry = (head + 0.8 * wires) * np.minimum(1, t / 0.0006)
    if not gated:
        return vel * dry, vel * dry
    rl, rr = dsp.reverb(dry * 0.6, dry * 0.6, SR, rt60=2.2, damp_hz=5500, mod=4)
    gate = np.clip((0.26 - t) / 0.035, 0, 1)
    return vel * (dry + 1.3 * rl * gate), vel * (dry + 1.3 * rr * gate)


def hat_hit(rng, open_=False, vel=1.0):
    """A dark hat: high-passed noise with a few square partials, low-passed so it stays out of the air band."""
    n = int((0.35 if open_ else 0.07) * SR)
    t = np.arange(n) / SR
    metal = sum(np.sign(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))) for f in (2960, 4180, 5520, 7040)) * 0.15
    y = dsp.butter(rng.standard_normal(n) * 0.7 + metal, 6000, SR, 'high', 4)
    y = dsp.butter(y, 10500, SR, 'low', 2)
    y *= np.exp(-t / (0.11 if open_ else 0.016)) * np.minimum(1, t / 0.0005)
    return vel * end_fade(y)


def crash_hit(rng):
    """A dark crash: two decorrelated noise washes with metal partials, a long decay. Returns (L, R)."""
    n = int(2.6 * SR)
    t = np.arange(n) / SR
    out = []
    for _ in range(2):
        metal = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) for f in rng.uniform(3000, 8000, 12)) * 0.08
        y = dsp.butter(rng.standard_normal(n) + metal, 3500, SR, 'high', 2)
        y = dsp.butter(y, 9000, SR, 'low', 2)
        out.append(end_fade(y * np.exp(-t / 0.7) * np.minimum(1, t / 0.002), 0.3))
    return out[0], out[1]


def riser(n, rng):
    """One bar of noise through a band-pass sweeping 400 Hz -> 6 kHz, rising in level."""
    t = np.arange(n) / SR
    u = t / (n / SR)
    fc = 400 * (6000 / 400) ** u
    y = dsp.svf_bp(rng.standard_normal(n), fc, SR, 2.5)
    return y * u ** 2 / np.percentile(np.abs(y), 99)


def seq_note(hz, vel, cutoff, rng, res=0.45):
    """One step of the sequence: a saw and a 25% pulse through a resonant ladder struck from `cutoff` and falling,
    about 150 ms long: a pluck in the pad's own register, so the middle of the mix moves."""
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    f = np.full(n, hz)
    ph = rng.uniform()
    x = 0.6 * dsp.saw(f, SR, phase=ph) + 0.4 * (dsp.saw(f, SR, phase=ph) - dsp.saw(f, SR, phase=(ph + 0.25) % 1))
    y = dsp.ladder(x, 180 + cutoff * np.exp(-t / 0.06), SR, res=res, drive=1.3)
    return vel * end_fade(y * np.minimum(1, t / 0.0015) * np.exp(-t / 0.07))


def clap_hit(rng):
    """A clap: three quick bursts of band-passed noise 9 ms apart and a 60 ms tail."""
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    env = sum(np.exp(-np.clip(t - d, 0, None) / 0.004) * (t >= d) for d in (0.0, 0.009, 0.018)) + 0.6 * np.exp(-np.clip(t - 0.027, 0, None) / 0.06) * (t >= 0.027)
    y = dsp.butter(dsp.butter(rng.standard_normal(n), 900, SR, 'high', 2), 6000, SR, 'low', 2) * env
    return end_fade(y / np.abs(y).max())


def shaker(rng, vel=1.0):
    """A shaker grain: band-passed noise with a soft 6 ms attack."""
    n = int(0.08 * SR)
    t = np.arange(n) / SR
    y = dsp.butter(dsp.butter(rng.standard_normal(n), 5000, SR, 'high', 2), 9500, SR, 'low', 2)
    return vel * end_fade(y * np.minimum(1, t / 0.006) * np.exp(-np.maximum(t - 0.006, 0) / 0.02))


def tom_hit(hz, rng, vel=1.0):
    """A synth tom: a sine falling a fifth onto hz, a short noise skin, 0.35 s."""
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    f = hz * (1 + 0.5 * np.exp(-t / 0.03))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.13) + 0.15 * dsp.butter(rng.standard_normal(n), 3000, SR, 'low') * np.exp(-t / 0.01)
    return vel * end_fade(np.tanh(1.5 * y) / np.tanh(1.5))


def stab(notes, vel, bright, rng):
    """A short chord stab: a 30% pulse per note through one resonant ladder whose cutoff falls fast, gone in about
    150 ms, so it ticks against the beat rather than filling it."""
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for m in notes:
        f = np.full(n, HZ(m) * 2 ** (rng.uniform(-4, 4) / 1200))
        ph = rng.uniform()
        y += dsp.saw(f, SR, phase=ph) - dsp.saw(f, SR, phase=(ph + 0.3) % 1)
    fc = 350 + bright * np.exp(-t / 0.045)
    y = dsp.ladder(y / len(notes), fc, SR, res=0.45, drive=1.2)
    return vel * end_fade(y * np.minimum(1, t / 0.0015) * np.exp(-t / 0.06))


def lead_voice(N, ev, cents=0.0, tone='pulse'):
    """The lead, gliding between notes (a 60 ms ramp in log frequency) with a vibrato that grows in after 0.3 s on
    held notes. The voice Mark chose in Low Orbit's A/B ("pulse is best"): a pulse whose width drifts, through a
    resonant ladder that follows the lead's own notes and is struck on each one, a quack, not a swell. Scaled to one
    RMS over its notes.
    Returns (filtered lead, one undetuned saw on the same pitch path for the tuning check: two detuned saws beat,
    which moves a short window's spectral peak by up to 15 cents; an octave-down square layer)."""
    t = np.arange(N) / SR
    target = np.full(N, np.log(HZ(ev[0]['midi'])))
    amp = np.zeros(N)
    fenv = np.zeros(N)       # slow and brassy: up in 40 ms, settling to 55%
    strike = np.zeros(N)     # quick: struck at each note, gone in about 120 ms
    since = np.zeros(N)      # seconds into the current note
    vib = np.zeros(N)
    on = np.zeros(N, bool)
    for k, e in enumerate(ev):
        i = int(e['t'] * SR)
        target[i:] = np.log(HZ(e['midi']) * 2 ** (cents / 1200))
        n = min(N - i, int((e['dur'] + 0.8) * SR))
        u = np.arange(n) / SR
        a = np.minimum(1, u / (0.008 if tone == 'pulse' else 0.07)) * np.where(u < e['dur'], 1.0, np.exp(-np.clip(u - e['dur'], 0, None) / 0.09))
        amp[i:i + n] = np.maximum(amp[i:i + n], a * e['vel'])
        fe = (1 - np.exp(-u / 0.04)) * (0.55 + 0.45 * np.exp(-u / 0.35)) * (u < e['dur'] + 0.3)
        fenv[i:i + n] = np.maximum(fenv[i:i + n], fe)
        strike[i:i + n] = np.maximum(strike[i:i + n], np.exp(-u / 0.12) * (0.7 + 0.3 * e.get('pushed', False)))
        since[i:] = np.arange(N - i) / SR
        v = np.clip((u - 0.3) / 0.4, 0, 1) * (u < e['dur'])
        vib[i:i + n] = np.maximum(vib[i:i + n], v)
        on[i:i + int(e['dur'] * SR)] = True
    w = int(0.06 * SR)                    # a fixed-time glide: a 60 ms ramp in log frequency, whatever the leap
    logf = signal.oaconvolve(np.concatenate([np.full(w, target[0]), target]), np.ones(w) / w, mode='valid')[1:N + 1]
    f = np.exp(logf) * 2 ** (14 * vib * np.sin(2 * np.pi * 5.3 * t) / 1200)
    ph = np.cumsum(f) / SR
    if tone == 'breath':     # a sine with a little 2nd and 3rd, and breath: noise around the note, rising with it
        tone_ = np.sin(2 * np.pi * ph) + 0.22 * np.sin(4 * np.pi * ph) + 0.07 * np.sin(6 * np.pi * ph)
        nz = dsp.svf_bp(np.random.default_rng(7).standard_normal(N), np.clip(f * 2.0, 200, 6000), SR, 3.0)
        lead = dsp.ladder(tone_ + 0.06 * nz / (np.std(nz) + 1e-12) * (0.6 + 0.4 * fenv), np.clip(f * 3.0, 300, 5000), SR, res=0.0, drive=1.0)
        lead *= amp
        lead *= 0.25 / (np.sqrt(np.mean(lead[on] ** 2)) + 1e-12)
    else:
        width = 0.3 + 0.12 * np.sin(2 * np.pi * 0.23 * t)
        s = dsp.saw(f, SR)
        raw = s - np.interp(ph - width, ph, s)      # a pulse: the saw minus itself shifted by `width` of a cycle
        lead = dsp.ladder(raw, np.clip(f * (2.2 + 7.0 * strike), 200, 8000), SR, res=0.55, drive=1.5)
        lead *= amp
        lead *= 0.25 / (np.sqrt(np.mean(lead[on] ** 2)) + 1e-12)
    sq = dsp.saw(f / 2, SR) - dsp.saw(f / 2, SR, phase=0.5)
    low = dsp.ladder(sq, np.clip(f * 2.5, 150, 3000), SR, res=0.1, drive=1.0) * amp
    return lead, dsp.saw(f, SR) * amp, low


def note_voice(N, ev, tone, rng, cents=0.0):
    """The struck voices, one note at a time (no glide: each note is its own strike). 'epiano': two-operator FM, the
    modulator at the carrier's frequency, its index falling from a bright bark to a soft core, a quick tine partial
    (after Lantern's piano). 'pluck': Karplus-Strong (the kernel's fractional delay keeps it in tune) from a dark,
    short noise burst, fully averaged, so it rings warm and dies on its own; a damping finger after the note.
    Returns the voice (it is its own tuning stem: these are the oscillators the ear hears)."""
    out = np.zeros(N)
    for e in ev:
        f = HZ(e['midi']) * 2 ** ((cents + rng.uniform(-2, 2)) / 1200)
        i = int(e['t'] * SR)
        n = min(N - i, int((e['dur'] + {'epiano': 0.6, 'felt': 0.5}.get(tone, 0.9)) * SR))
        t = np.arange(n) / SR
        v = e['vel'] * rng.uniform(0.9, 1.0)
        if tone == 'epiano':
            idx = 1.0 * v * np.exp(-t / 0.3) + 0.3
            amp = np.minimum(1, t / 0.004) * (0.55 * np.exp(-t / 0.9) + 0.45 * np.exp(-t / 4.0))
            amp *= np.clip(1 - (t - e['dur']) / 0.25, 0, 1)
            ph = 2 * np.pi * f * t
            y = np.sin(ph + idx * np.sin(ph)) + 0.08 * v * np.sin(2 * np.pi * min(f * 14.0, 15000) * t) * np.exp(-t / 0.015)
            y *= amp
        elif tone == 'felt':
            y = felt_note(f, v, e['dur'], t, rng)
        else:
            m = int(0.014 * SR)                        # a thumb, not a pick ("a bit too hard")
            exc = np.zeros(n)
            exc[:m] = dsp.butter(rng.standard_normal(m), 900, SR, 'low', 2) * np.hanning(m)
            y = dsp.ks(exc, SR, f, fb=0.997, bright=0.0)
            y = dsp.butter(y, 3500, SR, 'low', 2) * np.clip(1 - (t - e['dur'] - 0.4) / 0.4, 0, 1)
            y *= np.minimum(1, t / 0.008)              # and the first 8 ms eased in
        out[i:i + n] += v * end_fade(y)
    return out


FELT_B = 4e-4                              # the string's stiffness: partial k sits at k f sqrt(1 + B k^2)


def felt_note(f, v, dur, t, rng):
    """One felt-piano note: two strings a cent or so apart (so held notes beat a little), each with the stretched
    partials of a stiff string; the felt hammer leaves few upper partials (fewer when soft), and each partial dies
    faster the higher it is, quickly at first, then slowly; a short thump of the felt on the string; the damper
    falls 50 ms after the note ends."""
    y = np.zeros(len(t))
    tau1 = 1.6 * (440 / f) ** 0.4
    for k in range(1, 13):
        fk = k * f * np.sqrt(1 + FELT_B * k * k)
        if fk > 12000:
            break
        a = np.exp(-(k - 1) / (1.2 + 2.5 * v)) / k
        tk = tau1 / (1 + 0.35 * (k - 1))
        env = 0.6 * np.exp(-t / (0.3 * tk)) + 0.4 * np.exp(-t / (1.5 * tk))
        for c in (-0.6, 0.6):
            fs = fk * 2 ** (c * rng.uniform(0.5, 1.2) / 1200)
            y += a * env * np.sin(2 * np.pi * fs * t + rng.uniform(0, 2 * np.pi))
    m = int(0.03 * SR)
    th = np.zeros(len(t))
    th[:m] = dsp.butter(rng.standard_normal(m), 1000, SR, 'low', 2)[:m] * np.exp(-t[:m] / 0.008)
    y = y + 0.15 * v * th
    y *= np.minimum(1, t / 0.002) * np.exp(-np.clip(t - dur - 0.05, 0, None) / 0.08)
    return y


def brush_hit(rng, vel=1.0):
    """A brushed snare: a slap of grainy noise (the bristles, noise under a fast-flickering envelope), band-passed,
    a little of the head's tone, then the bristles settling for a quarter of a second. Returns (L, R), each side its
    own bristles in part, as wide as a snare in a room."""
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    grain = 0.55 + 0.45 * np.abs(dsp.butter(rng.standard_normal(n), 700, SR, 'low', 2)) / 0.35
    env = np.minimum(1, t / 0.003) * (np.exp(-t / 0.07) + 0.18 * np.exp(-t / 0.25))
    head = 0.3 * np.sin(2 * np.pi * np.cumsum(190 + 25 * np.exp(-t / 0.01)) / SR) * np.exp(-t / 0.035)
    common = rng.standard_normal(n)
    out = []
    for _ in range(2):
        nz = 0.7 * common + 0.3 * rng.standard_normal(n)
        y = dsp.butter(dsp.butter(nz * grain, 900, SR, 'high', 2), 8000, SR, 'low', 2) * env
        out.append(vel * end_fade(0.9 * y + head * np.minimum(1, t / 0.001), 0.05))
    return out[0], out[1]


def round_kick(rng):
    """A rounder kick: a sine from 150 Hz down to 46 Hz, slower than the punchy one's sweep, no click, its body a
    little longer (still gone by 0.36 s), saturated less."""
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 46 + 104 * np.exp(-t / 0.035)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.11) * np.minimum(1, t / 0.0015)
    y = np.tanh(1.3 * body) / np.tanh(1.3)
    y *= np.clip((0.36 - t) / 0.14, 0, 1)
    return dsp.butter(y, 5000, SR, 'low')


def tape(mix, spec, rng):
    """The mix through a tape machine, before the master: wow and flutter (the mix read around its own time, so
    nothing moves on average), a soft saturation set against the mix's own peaks, a head bump in the lows, the top
    rolled off. Returns the mix and the hiss to add after the master (dBFS RMS)."""
    wow, flutter, drive, lp, bump, hiss = spec
    t = np.arange(len(mix)) / SR
    drift = dsp.butter(rng.standard_normal(len(t) // 480 + 2), 0.02, 100, 'low', 2)
    drift = np.interp(t, np.arange(len(drift)) / 100, drift / (np.std(drift) + 1e-12))
    # a delay d(t) shifts pitch by -d'(t): A sin(w t) gives a peak deviation of A w
    dev = (wow / 100 / (2 * np.pi * 0.55) * (0.8 * np.sin(2 * np.pi * 0.55 * t + rng.uniform(0, 6.3)) + 0.2 * drift)
           + flutter / 100 / (2 * np.pi * 6.5) * np.sin(2 * np.pi * 6.5 * t + rng.uniform(0, 6.3)))
    out = np.stack([np.interp(t + dev, t, mix[:, c]) for c in range(2)], 1)
    ref = np.percentile(np.abs(out), 99.9)
    out = np.tanh(drive * out / ref) / drive * ref
    for c in range(2):
        x = out[:, c]
        x = x + (10 ** (bump / 20) - 1) * signal.sosfiltfilt(signal.butter(1, [45, 110], 'band', fs=SR, output='sos'), x)
        out[:, c] = dsp.butter(x, lp, SR, 'low', 2)
    return out, hiss


def bass_struck(N, notes, tone, rng):
    """The less synthetic basses, one note at a time (notes as for bass_voice). 'fingered': a low Karplus-Strong string
    from a short, dark burst (a thumb), lightly driven like an electric bass into an amp, over a sine sub that follows
    each note's envelope. 'round': a near-sine (a little 2nd and 3rd harmonic) with a plucked envelope settling to a
    hold, low-passed, no resonance, no separate sub. Returns (body, sub, raw): raw is the oscillator the tuning check
    reads (the string itself, or the round tone)."""
    body, sub = np.zeros(N), np.zeros(N)
    for (t0, m, dur, acc, base, amt) in notes:
        f = HZ(m)
        i = int(t0 * SR)
        n = min(N - i, int((dur + 0.12) * SR))
        t = np.arange(n) / SR
        rel = np.clip(1 - (t - dur) / 0.07, 0, 1)
        if tone == 'fingered':
            k = int(0.008 * SR)
            exc = np.zeros(n)
            exc[:k] = dsp.butter(rng.standard_normal(k), 900, SR, 'low', 2) * np.hanning(k) * (0.7 + 0.3 * acc)
            y = dsp.ks(exc, SR, f, fb=0.996, bright=0.0) * rel
            s_ = np.sin(2 * np.pi * f * t) * np.minimum(1, t / 0.004) * (0.65 + 0.35 * np.exp(-t / 0.25)) * rel
        else:
            ph = 2 * np.pi * f * t
            y = (np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)) * np.minimum(1, t / 0.006)
            y *= (0.6 + 0.4 * np.exp(-t / 0.15)) * (0.85 + 0.15 * acc) * rel
            s_ = np.zeros(n)
        body[i:i + n] += end_fade(y, 0.01)
        sub[i:i + n] += end_fade(s_, 0.01)
    raw = body.copy()
    if tone == 'fingered':
        body = dsp.butter(np.tanh(1.6 * body / (np.abs(body).max() + 1e-12)) / np.tanh(1.6), 1600, SR, 'low', 2)   # its tone knob down, out of the lead's way
    else:
        body = dsp.butter(body, 900, SR, 'low', 2)
    on = np.abs(raw) > 1e-4
    body *= 0.3 / (np.sqrt(np.mean(body[on] ** 2)) + 1e-12)
    return body, sub, raw


def bass_voice(N, notes, beat):
    """Low Orbit's layered bass, for any notes: (time, MIDI, seconds, accent, filter base Hz, filter strike Hz). A sine
    sub (returned apart, so the mix can duck it hard under the kick), two saws through a ladder struck on each note,
    and a saturated mid layer on the accents. A note longer than `beat` is held: a slow filter fall and a long
    release; a 12 ms slide between notes that touch. Less resonance and grit than in Low Orbit, where the listener
    heard "a slight acid feel". Returns (body, sub)."""
    f = np.full(N, HZ(notes[0][1]) if notes else HZ(38))
    gate = np.zeros(N)
    cut = np.full(N, 120.0)
    t = np.arange(N) / SR
    prev_end, prev_hz = -1.0, HZ(38)
    for (t0, m, dur, acc, base, amt) in notes:
        i, j = int(t0 * SR), min(N, int((t0 + dur) * SR))
        g = int(0.012 * SR) if abs(t0 - prev_end) < 0.01 else 0
        f[i:] = HZ(m)
        if g:
            f[i:i + g] = np.exp(np.linspace(np.log(prev_hz), np.log(HZ(m)), g))
        u = t[i:j] - t0
        held = dur > beat
        rel = 0.35 if held else 0.02
        gate[i:j] = np.minimum(1, u / 0.002) * np.clip(((dur if held else dur - 0.012) - u) / rel, 0, 1)
        tau = 0.5 if held else 0.07
        cut[i:j] = np.maximum(cut[i:j], base + amt * acc * np.exp(-u / tau))
        prev_end, prev_hz = t0 + dur, HZ(m)
    strike = np.clip((cut - 120) / 1400, 0, 1)            # 0..1: how far each note's filter is struck open
    osc = 0.5 * dsp.saw(f, SR) + 0.5 * dsp.saw(f * 2 ** (7 / 1200), SR, phase=0.37)
    body = np.tanh(1.2 * dsp.ladder(osc, cut, SR, res=0.35, drive=1.4) * gate) / np.tanh(1.2)
    grit = dsp.butter(np.tanh(4.0 * dsp.butter(body, 250, SR, 'high', 2)), 2500, SR, 'low', 2)
    body = body + 0.12 * grit * strike
    on = gate > 0.5
    body *= 0.3 / (np.sqrt(np.mean(body[on] ** 2)) + 1e-12)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * gate
    return body, sub


def clip_knee(x, th, w):
    """Linear below th; above it, the excess squeezed by tanh into at most w more."""
    ax = np.abs(x)
    return np.where(ax > th, np.sign(x) * (th + w * np.tanh((ax - th) / w)), x)


def bandsplit_dip(x, on, lo, hi, depth):
    """A zero-phase dip in [lo, hi] wherever `on` (0..1 per sample) is up: subtract a filtfilt band-passed copy."""
    band = signal.sosfiltfilt(signal.butter(2, [lo, hi], 'band', fs=SR, output='sos'), x)
    return x - depth * band * on


def pad(N, rng, F=STUDY1):
    """The chord bed: per chord note, three detuned saws on each side (so the two sides differ), a slow attack, a
    release overlapping the next chord. Returns (L, R), dry (filtered later) and the chord events."""
    L, Rr = np.zeros(N), np.zeros(N)
    ev = []
    for b in R(1, F.nbars):
        name, _, voicing = chord(b, F)
        t0 = tb(b)
        hold = BAR if b < F.nbars else BAR + 2.5
        n = min(N - int(t0 * SR), int((hold + 0.9) * SR))
        t = np.arange(n) / SR
        env = np.minimum(1, t / (1.6 if b == 1 else 0.18)) * np.clip(1 - (t - hold) / 0.9, 0, 1) ** 2
        if b == F.nbars:
            env *= np.exp(-np.clip(t - 1.0, 0, None) / 1.4)
        i = int(t0 * SR)
        for m in voicing:
            for side, cents in ((L, (-11, -3, 6)), (Rr, (-6, 4, 12))):
                for c in cents:
                    fr = HZ(m) * 2 ** ((c + rng.uniform(-1.5, 1.5)) / 1200)
                    side[i:i + n] += env * dsp.saw(np.full(n, fr), SR, phase=rng.uniform())
        ev.append({'t': t0, 'chord': name, 'notes': voicing, 'dur': hold})
    return L / 12, Rr / 12, ev


def bass_line(N, brk=None, tone='layered', F=STUDY1):
    """The bass: its notes through bass_voice (layered) or bass_struck (fingered, round), the filter opening over the
    intro, held and low in the breakdown. Returns (body, sub, events, raw for the tuning check)."""
    notes, ev = [], []
    for (t0, m, dur, acc, b) in bass_events(brk, F):
        if F.kind(b) == 'intro':
            base, amt = 90 + 140 * (b - 3), 250 + 300 * (b - 3)
        elif F.bass[b] == 'held':
            base, amt = 160, 250
        else:
            base, amt = 160, 900 if F.kind(b) == 'drop' else 750
        notes.append((t0, m, dur, acc, base, amt))
        ev.append({'t': t0, 'midi': int(m), 'dur': dur, 'bar': b})
    if tone == 'layered':
        body, sub = bass_voice(N, notes, BEAT)
        return body, sub, ev, sub.copy()
    if brk == 'flat-bass':
        notes = [(a, m - 0.3, d, c, b_, am) for a, m, d, c, b_, am in notes]
    body, sub, raw = bass_struck(N, notes, tone, np.random.default_rng(9))
    return body, sub, ev, raw


# --- the render ---------------------------------------------------------------------------------------------------

def render(brk=None, voice='pluck', soft=True, bass_tone='round', F=STUDY1, drums='soft', space='room', tape_='none',
           lufs=LUFS_TARGET):
    rng = np.random.default_rng(31)
    brng = np.random.default_rng(41)       # the A/B's own drums: the main stream is drawn from exactly as without them
    NB = F.nbars
    dur = NB * BAR + TAIL
    N = int(dur * SR)
    t = np.arange(N) / SR
    static = brk == 'static-drums'
    vary = (lambda lo, hi: 1.0) if static else (lambda lo, hi: rng.uniform(lo, hi))

    # drums
    kick_pat, snare_pat = dict(F.kick), dict(F.snare)
    if brk == 'four-on-floor-bridge':
        kick_pat.update({b: [0, 1, 2, 3] for b in F.sections.get('bridge', [])})
        snare_pat.update({b: [1, 3] for b in F.sections.get('bridge', [])})
    if brk == 'kick-in-breakdown':
        kick_pat.update({b: [0, 1, 2, 3] for b in F.sections.get('breakdown', [])})
    kick_bars = sorted(kick_pat)
    away = {b for b, _ in F.punct} if not static else set()        # the take-away: no kick on beat 4, no hats after 2
    kicks = [tb(b, q) for b in kick_bars for q in kick_pat[b] if not (b in away and q == 3)] + ([tb(NB)] if F.final_hit else [])
    if brk == 'boom-at-end':
        kicks.append(tb(NB) + 3.5)
    K = np.zeros(N)
    kh = kick_hit(rng)
    kh = round_kick(brng) if drums == 'brush-round' else kh
    kit.place(K, kh, [(k, vary(0.94, 1.0)) for k in kicks], SR, KICK_GAIN)

    snL, snR = np.zeros(N), np.zeros(N)
    takes = [snare_hit(rng, gated=not soft) for _ in range(1 if static else 3)]     # soft: dry, no gated room
    brush = drums.startswith('brush')
    if brush:
        takes = [tuple(BRUSH_GAIN * x for x in brush_hit(brng)) for _ in takes]
    snares = []
    for i_, (b, q) in enumerate((b, q) for b in sorted(snare_pat) for q in snare_pat[b]):
        snares.append(tb(b, q))
        sl, sr_ = takes[i_ % len(takes)]
        v = vary(0.88, 1.0)
        kit.place(snL, sl, [(tb(b, q), v * (0.75 if soft else 1.0))], SR)
        kit.place(snR, sr_, [(tb(b, q), v * (0.75 if soft else 1.0))], SR)
    fill_bars = [b for b, _, _ in F.fills]
    ghosts = [] if static else [(b, 2.75, 0.13) for b in sorted(F.snare) if b % 2 == 0 and b not in fill_bars
                                and F.snare[b] == [1, 3]]
    for b, qs, vs in F.fills + [(b, [q], [v]) for b, q, v in ghosts]:
        for q, v in zip(qs, vs):
            dL, dR = snare_hit(rng, v, gated=(b not in F.rolls and v > 0.3))
            if brush:
                dL, dR = (BRUSH_GAIN * x for x in brush_hit(brng, v))
            kit.place(snL, dL, [(tb(b, q), 1.0)], SR)
            kit.place(snR, dR, [(tb(b, q), 1.0)], SR)

    pcL, pcR = np.zeros(N), np.zeros(N)                     # percussion: clap, shaker, toms, the take-aways
    if not static:
        cl = clap_hit(rng)
        for b in ([] if soft else F.clap):
            for q in (1, 3):
                l, r = kit.pan(cl, rng.uniform(-0.2, 0.2))
                kit.place(pcL, l, [(tb(b, q) + 0.004, 0.5 * vary(0.85, 1.0))], SR)
                kit.place(pcR, r, [(tb(b, q) + 0.004, 0.5 * vary(0.85, 1.0))], SR)
        for b in F.shaker:
            for k in range(16):
                v = (0.7 if k % 4 == 2 else 0.45 if k % 2 == 0 else 0.3) * vary(0.75, 1.0)
                l, r = kit.pan(shaker(rng, v), 0.45)
                kit.place(pcL, l, [(tb(b, k * 0.25), 1.0)], SR)
                kit.place(pcR, r, [(tb(b, k * 0.25), 1.0)], SR)
        for b in F.bongos:
            for k, hi_ in BONGO[(b - 1) % 4]:
                x = tom_hit(HZ(66 if hi_ else 61), rng, 0.35 * vary(0.55, 1.0))[:int(0.15 * SR)]
                l, r = kit.pan(end_fade(x, 0.02), -0.5 if hi_ else -0.3)
                kit.place(pcL, l, [(tb(b, k * 0.25), 1.0)], SR)
                kit.place(pcR, r, [(tb(b, k * 0.25), 1.0)], SR)
        for b, v in F.punct:
            cl_, cr_ = crash_hit(rng)
            n_ = int(BEAT * SR)
            for side, cy in ((pcL, cl_), (pcR, cr_)):
                kit.place(side, cy[:n_][::-1] * np.linspace(0, 1, n_) ** 2, [(tb(b, 3), 1.2 * v)], SR)
                kit.place(side, cy, [(tb(b + 1), 1.6 * v)], SR)
            for q, m in ((2.5, 52), (3.0, 50), (3.25, 50), (3.5, 47), (3.75, 45)):
                l, r = kit.pan(tom_hit(HZ(m), rng, 0.9), (m - 48) / 8)
                kit.place(pcL, l, [(tb(b, q), 1.0)], SR)
                kit.place(pcR, r, [(tb(b, q), 1.0)], SR)
        for b, steps in F.toms:
            for q, m in steps:
                l, r = kit.pan(tom_hit(HZ(m), rng, 0.8), (m - 48) / 8)
                kit.place(pcL, l, [(tb(b, q), 1.0)], SR)
                kit.place(pcR, r, [(tb(b, q), 1.0)], SR)

    hL, hR = np.zeros(N), np.zeros(N)
    for b in F.hats:
        swell = 1.0 if static else 0.85 + 0.15 * ((b - 1) % 4) / 3
        for k in range(16):
            if b in away and k >= 8:
                continue
            open_ = b in F.open_hats and (k % 4 == 2 if (static or b % 4) else k in (2, 6, 10, 15))
            if open_:
                h, v = hat_hit(rng, True, 0.55), 1.0
            else:
                h, v = hat_hit(rng, False), (0.75 if k % 4 == 2 else 0.5 if k % 2 == 0 else 0.32)
            p = 0.15 * np.sin(k * 1.3) + rng.uniform(-0.05, 0.05)
            l, r = kit.pan(h, p)
            g = v * swell * (1.0 if static else rng.uniform(0.85, 1.0))
            kit.place(hL, l, [(tb(b, k * 0.25), g)], SR)
            kit.place(hR, r, [(tb(b, k * 0.25), g)], SR)

    crL, crR = np.zeros(N), np.zeros(N)
    for b, v in F.crashes:
        cl_, cr_ = crash_hit(rng)
        kit.place(crL, cl_, [(tb(b), v)], SR)
        kit.place(crR, cr_, [(tb(b), v)], SR)
    rsL, rsR = np.zeros(N), np.zeros(N)
    for b in F.risers:
        n = int(BAR * SR)
        rsL[int(tb(b) * SR):int(tb(b) * SR) + n] += riser(n, rng)
        rsR[int(tb(b) * SR):int(tb(b) * SR) + n] += riser(n, rng)

    # bass
    bass, sub, bass_ev, bass_raw = bass_line(N, brk, bass_tone, F)
    floor = np.ones(N)
    for d0, d1 in F.dips:                                      # the floor falls away in the breakdown
        bd0, bd1 = tb(d0), tb(d1 + 1)
        floor *= np.interp(t, [0, bd0, bd0 + 0.05, bd1 - 0.05, bd1, dur], [1, 1, 0.4, 0.4, 1, 1])
    if F.bass_swell:                                        # the bass comes in low and rises: held at full level, it
        a_, b_ = F.bass_swell                               # stepped the intro up 11 dB in one bar
        floor *= np.interp(t, [tb(a_), tb(b_)], [0.3, 1.0])
    # and dies away under the last chord: a held note that stopped dead 4 s in dropped the whole mix 30 dB at once
    floor *= np.exp(-np.clip(t - tb(NB) - 0.3, 0, None) / 1.0)
    bass, sub = bass * floor, sub * floor ** 1.5

    # pad: opens over the intro, sits lower under the groove, opens in the breakdown, closes at home
    pL, pR, pad_ev = pad(N, rng, F)
    fc = np.interp(t, [dur if b is None else tb(b) for b, _ in F.pad_fc], [v for _, v in F.pad_fc])
    fc = fc * (0.65 if soft else 1.0)                       # soft: a darker, warmer bed
    lev = dsp.ladder(pL, fc, SR, res=0.15, drive=1.0), dsp.ladder(pR, fc, SR, res=0.15, drive=1.0)
    pL = lev[0] + 0.5 * dsp.tapedelay(lev[0], SR, 0.011, wow_ms=2.0, wow_hz=0.4, fb=0.0, damp_hz=9000)
    pR = lev[1] + 0.5 * dsp.tapedelay(lev[1], SR, 0.013, wow_ms=2.0, wow_hz=0.5, fb=0.0, damp_hz=9000, phase=1.7)

    # the sequence: a pluck every third 16th from bar 1, through the chord's voicing; its filter breathes over eight
    # bars, low in the intro and the breakdown, open in the drop
    qL, qR = np.zeros(N), np.zeros(N)
    seq_ev = []
    if brk != 'no-sequence':
        for j, k in enumerate(range(0, NB * 16 - 16, SEQ_STEP)):
            t0 = k * BEAT / 4
            b = int(t0 // BAR) + 1
            if (b in F.rolls and t0 > tb(b, 2)) or b >= F.seq_stop:
                continue                                   # it steps aside for the roll, and for the last chord
            _, _, voicing = chord(b, F)
            kind = F.kind(b)
            m = voicing[SEQ_FIG[j % len(SEQ_FIG)]]
            breath = 0.5 - 0.5 * np.cos(2 * np.pi * t0 / (8 * BAR))
            if kind == 'intro':
                cutoff, v = 350 + 220 * (b - 1) + 300 * breath, 0.5 + 0.1 * (b - 1)
            elif kind in ('breakdown', 'bridge'):
                cutoff, v = 500 + 400 * breath, 0.55
            elif kind == 'build':                          # opening and rising toward the drop
                u = (t0 - tb(F.sections['build'][0])) / (len(F.sections['build']) * BAR)
                cutoff, v = 700 + 1600 * u, 0.6 + 0.3 * u
            elif kind == 'home':
                cutoff, v = 700 + 500 * breath, 0.7
            else:
                cutoff, v = (1100 if kind == 'groove' else 1300) + 1300 * breath, 1.05 if kind == 'groove' else 0.85
            x = seq_note(HZ(m), v * (1.0 if j % 2 == 0 else 0.8), cutoff * (0.6 if soft else 1.0), rng, res=0.2 if soft else 0.45)
            l, r = kit.pan(x, 0.25 if j % 2 else -0.25)
            kit.place(qL, l, [(t0, 1.0)], SR)
            kit.place(qR, r, [(t0, 1.0)], SR)
            seq_ev.append({'t': t0, 'midi': int(m), 'bar': b})
    sdL, sdR = dsp.pingpong(qL * 0.3, qR * 0.3, 0.5 * BEAT, SR, fb=0.25, damp_hz=2500)

    # stabs: three a bar on the 'a', the chord's top notes an octave up, in the drop only
    aL, aR = np.zeros(N), np.zeros(N)
    stab_ev = []
    for b in ([] if soft else F.stabs):
        _, _, voicing = chord(b, F)
        notes = [m + 12 for m in voicing[1:]]
        for i_, (k, v) in enumerate(STAB_FIG):
            x = stab(notes, v, 2600, rng)
            l, r = kit.pan(x, 0.4 if i_ % 2 else -0.4)
            kit.place(aL, l, [(tb(b, k * 0.25), 1.0)], SR)
            kit.place(aR, r, [(tb(b, k * 0.25), 1.0)], SR)
            stab_ev.append({'t': tb(b, k * 0.25), 'notes': notes, 'bar': b, 'step': k})
    eL, eR = dsp.pingpong(aL * 0.4, aR * 0.4, 0.75 * BEAT, SR, fb=0.3, damp_hz=3000)

    # lead
    lev_ = lead_events(brk, F)
    cents = 30.0 if brk == 'sharp-lead' else 0.0
    if voice in ('epiano', 'pluck'):
        lead = note_voice(N, lev_, voice, np.random.default_rng(5), cents)
        lead_raw, low = lead.copy(), np.zeros(N)
    else:
        lead, lead_raw, low = lead_voice(N, lev_, cents, tone=voice)
        if voice != 'pulse':
            low = np.zeros(N)                              # the octave-down square is the pulse's alone
    if voice != 'pulse':                                    # matched to the pulse by loudness, not RMS
        lead *= 10 ** ((LEAD_LUFS + LEAD_TRIM.get(voice, 0.0) - dsp.lufs(np.stack([lead, lead], 1), SR)) / 20)
    dbl = np.zeros(N)
    for d0, d1 in F.doubled:
        dbl += np.clip((t - tb(d0) + 0.05) / 0.1, 0, 1) * np.clip((tb(d1 + 1) + 0.3 - t) / 0.3, 0, 1)
    low *= dbl
    lead = lead + 0.45 * low
    lead *= 10 ** (-14 / 20) if brk == 'buried-lead' else 1.0
    ldL, ldR = dsp.pingpong(lead * 0.35, lead * 0.25, 0.75 * BEAT, SR, fb=0.35, damp_hz=2800)
    chL = dsp.tapedelay(lead, SR, 0.009, wow_ms=1.5, wow_hz=0.6, fb=0.0, damp_hz=8000)
    chR = dsp.tapedelay(lead, SR, 0.014, wow_ms=1.5, wow_hz=0.45, fb=0.0, damp_hz=8000, phase=2.1)
    ldL, ldR = ldL + 0.4 * chL - 0.15 * chR, ldR + 0.4 * chR - 0.15 * chL

    # the counter-line: the breath voice (the listener's other pick in study 2), under the hook, a little to the right
    cev = counter_events(F, brk)
    if cev:
        cn, cn_raw, _ = lead_voice(N, cev, cents, tone='breath')
        cn *= 10 ** ((LEAD_LUFS + COUNTER_TRIM - dsp.lufs(np.stack([cn, cn], 1), SR)) / 20)
        cn *= 10 ** (-14 / 20) if brk == 'buried-counter' else 1.0
        cnL, cnR = kit.pan(cn, 0.3)
        e1, e2 = dsp.pingpong(cnL * 0.25, cnR * 0.25, 0.75 * BEAT, SR, fb=0.3, damp_hz=2500)
        cnL, cnR = cnL + e1, cnR + e2
    else:
        cn_raw, cnL, cnR = np.zeros(N), np.zeros(N), np.zeros(N)
    dev = descant_events(F, brk)                           # the descant: the breath above the hook, a little to the left
    if dev:
        ds, ds_raw, _ = lead_voice(N, dev, cents, tone='breath')
        ds *= 10 ** ((LEAD_LUFS + DESCANT_TRIM - dsp.lufs(np.stack([ds, ds], 1), SR)) / 20)
        dsL, dsR = kit.pan(ds, -0.3)
        e1, e2 = dsp.pingpong(dsL * 0.25, dsR * 0.25, 0.75 * BEAT, SR, fb=0.3, damp_hz=2500)
        dsL, dsR = dsL + e1, dsR + e2
    else:
        ds_raw, dsL, dsR = np.zeros(N), np.zeros(N), np.zeros(N)

    # the pad steps aside in the lead's band while the lead plays
    on = np.zeros(N)
    for e in lev_ + cev + dev:
        on[int(e['t'] * SR):int((e['t'] + e['dur'] + 0.2) * SR)] = 1
    on = kit.smooth(on, int(0.25 * SR))
    pL, pR = bandsplit_dip(pL, on, 500, 1800, 0.55), bandsplit_dip(pR, on, 500, 1800, 0.55)

    # mix
    G = {'kick': 1.0, 'snare': 0.30, 'hats': 0.22, 'perc': 0.24, 'crash': 0.05, 'riser': 0.05, 'bass': 0.27,
         'sub': 0.24, 'pad': 0.30, 'stab': 0.14, 'seq': 0.36, 'lead': 0.30}
    if soft:
        G.update(hats=0.15, seq=0.34, perc=0.18)
    # held notes: the body and the sub pump back slowly, still 3-4 dB down halfway to the next kick, so it stands out
    sc_bass = kit.duck(kicks, N, SR, depth=0.85, rel=0.26)
    sc_sub = kit.duck(kicks, N, SR, depth=0.97, rel=0.3)
    sc = kit.duck(kicks, N, SR, depth=0.2 if soft else 0.45, rel=0.14)        # soft: less pump on the bed
    sc_lead = kit.duck(kicks, N, SR, depth=0.2, rel=0.12)
    if bass_tone != 'layered':                              # matched to the layered bass by loudness
        x_ = bass * G['bass'] + sub * G['sub']
        gain = 10 ** ((BASS_LUFS + BASS_TRIM.get(bass_tone, 0.0) - dsp.lufs(np.stack([x_, x_], 1), SR)) / 20)
        bass, sub = bass * gain, sub * gain
    # (the round bass carries its fundamental in its body, so the body takes the sub's deeper, slower duck)
    bass_m = bass * G['bass'] * (sc_sub if bass_tone == 'round' else sc_bass) + sub * G['sub'] * sc_sub
    padL, padR = pL * G['pad'] * sc, pR * G['pad'] * sc
    stL, stR = (aL + eL) * G['stab'] * (0.6 + 0.4 * sc), (aR + eR) * G['stab'] * (0.6 + 0.4 * sc)
    seqL, seqR = (qL + sdL) * G['seq'] * (0.55 + 0.45 * sc), (qR + sdR) * G['seq'] * (0.55 + 0.45 * sc)
    leadL, leadR = (lead + ldL) * G['lead'] * sc_lead, (lead + ldR) * G['lead'] * sc_lead
    ctL, ctR = cnL * G['lead'] * sc_lead, cnR * G['lead'] * sc_lead
    dcL, dcR = dsL * G['lead'] * sc_lead, dsR * G['lead'] * sc_lead
    drL = K * G['kick'] + snL * G['snare'] + hL * G['hats'] + pcL * G['perc'] + crL * G['crash'] + rsL * G['riser']
    drR = K * G['kick'] + snR * G['snare'] + hR * G['hats'] + pcR * G['perc'] + crR * G['crash'] + rsR * G['riser']
    rt60, damp, ret = SPACES[space]
    rvL, rvR = dsp.reverb(padL * 0.25 + stL * 0.3 + leadL * 0.35 + seqL * 0.3 + (ctL + dcL) * 0.4,
                          padR * 0.25 + stR * 0.3 + leadR * 0.35 + seqR * 0.3 + (ctR + dcR) * 0.4, SR, rt60=rt60, damp_hz=damp, mod=8)
    L = drL + bass_m + padL + stL + seqL + leadL + ctL + dcL + ret * rvL
    Rr = drR + bass_m + padR + stR + seqR + leadR + ctR + dcR + ret * rvR
    mix = np.stack([L, Rr], 1)
    dr = np.stack([drL, drR], 1)
    mix = mix - dr + clip_knee(dr, CLIP * np.abs(K).max(), 0.3 * np.abs(K).max())

    # master: the low end mono below 120 Hz, then the loudness target through the true-peak limiter
    lo = np.stack([dsp.butter(mix[:, c], 120, SR, 'low', 2) for c in range(2)], 1)
    mix = mix - lo + lo.mean(1, keepdims=True)
    if brk == 'bright-master':
        hi = np.stack([dsp.butter(mix[:, c], 5000, SR, 'high', 2) for c in range(2)], 1)
        mix = mix + (10 ** (9 / 20) - 1) * hi
    hiss = None
    if TAPES[tape_]:
        mix, hiss = tape(mix, TAPES[tape_], np.random.default_rng(43))
    mix = kit.master(mix, SR, lufs=lufs)
    if hiss is not None:                                   # the tape's own hiss, after the master: it isn't limited
        h = dsp.butter(np.random.default_rng(44).standard_normal((N, 2)), 9000, SR, 'low', 2)
        mix = mix + h * 10 ** (hiss / 20) / np.std(h)
    mix *= np.clip((dur - t) / 1.5, 0, 1)[:, None]

    stems = {'kick': K * G['kick'], 'snare': (snL + snR) / 2 * G['snare'], 'hats': (hL + hR) / 2 * G['hats'],
             'perc': (pcL + pcR) / 2 * G['perc'], 'sub': sub * G['sub'] * sc_sub, 'bass': bass_m,
             'pad': (padL + padR) / 2, 'stabs': (stL + stR) / 2, 'seq': (seqL + seqR) / 2, 'lead': (leadL + leadR) / 2,
             'lead_raw': lead_raw, 'bass_raw': bass_raw, 'counter': (ctL + ctR) / 2, 'counter_raw': cn_raw,
             'descant': (dcL + dcR) / 2, 'descant_raw': ds_raw, 'fx': (crL + crR + rsL + rsR) / 2 * 0.05}
    score = {'bpm': BPM, 'bar': BAR, 'nbars': NB, 'kicks': kicks, 'snares': snares, 'kick_bars': kick_bars,
             'lead': lev_, 'bass': bass_ev, 'pad': pad_ev, 'stabs': stab_ev, 'seq': seq_ev, 'key_pcs': sorted(KEY_PCS),
             'hook': [list(h) for h in HOOK], 'statements': [[s, b, v] for s, b, v in F.statements],
             'sections': F.sections, 'kinds': {k: F.kinds.get(k, k) for k in F.sections}, 'main': F.main,
             'peak': F.peak, 'form': F.name, 'counter': cev, 'descant': dev,
             'bass_bars': [b for b in sorted(F.bass) if F.bass[b] != 'held'],
             'chord_tones': {k: sorted(v) for k, v in CHORD_TONES.items()}, 'lead_voice': voice, 'soft': soft, 'bass_tone': bass_tone,
             'drums': drums, 'space': space, 'tape': tape_, 'lufs': lufs}
    return mix, stems, score


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--break', dest='brk', choices=list(BREAKS))
    ap.add_argument('--lead', choices=list(LEAD_VOICES), default='pluck')
    ap.add_argument('--hard', action='store_true', help="study 1's bed: the gated snare, the clap and stabs, more pump")
    ap.add_argument('--bass', choices=list(BASS_VOICES), default='round')
    ap.add_argument('--form', choices=list(FORMS), default='full')
    ap.add_argument('--drums', choices=list(DRUM_KITS), default='soft')
    ap.add_argument('--space', choices=list(SPACES), default='room')
    ap.add_argument('--tape', choices=list(TAPES), default='none')
    ap.add_argument('--lufs', type=float, default=LUFS_TARGET)
    a = ap.parse_args()
    name = (NAME + (f'_{a.form}' if a.form != 'full' else '') + (f'_BROKEN-{a.brk}' if a.brk else '')
            + (f'_lead-{a.lead}' if a.lead != 'pluck' else '')
            + ('_hard' if a.hard else '') + (f'_bass-{a.bass}' if a.bass != 'round' else '')
            + (f'_drums-{a.drums}' if a.drums != 'soft' else '') + (f'_space-{a.space}' if a.space != 'room' else '')
            + (f'_tape-{a.tape}' if a.tape != 'none' else '') + (f'_lufs{a.lufs:g}' if a.lufs != LUFS_TARGET else ''))
    mix, stems, score = render(a.brk, a.lead, not a.hard, 'fingered' if a.brk == 'buzzy-bass' else a.bass, FORMS[a.form],
                               a.drums, a.space, a.tape, a.lufs)
    kit.write_outputs(name, mix, stems, score, SR)
    print(f'wrote out/{name}.wav: {len(mix) / SR:.1f} s, {dsp.lufs(mix, SR):.1f} LUFS, {dsp.true_peak_db(mix):.1f} dBTP')
