"""Before anything is published: scan the tracked files for what must not ship.

  python3 tools/release_scan.py          scan every file git tracks (and any untracked, unignored ones)
  python3 tools/release_scan.py --break  plant a forbidden name in a temporary file: the scan must fail

It fails on: the words in tools/forbidden.local.txt (your references' names, brands, private words; git ignores the
file, so the list itself never publishes them); audio, video, stems or analysis arrays of any kind; any
file over 2 MB; and a total over 10 MB. Reference audio and anything extracted from it (stems, transcriptions,
spectrograms, grids) stay out of version control entirely; this is the net under that rule, not the rule.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The words to block (your references' titles and artists, brands, private names, other projects) go one per line in
# tools/forbidden.local.txt, which git ignores, so the list itself is never published. Regular expressions are fine.
WORDS = []
LOCAL = os.path.join(ROOT, 'tools', 'forbidden.local.txt')
if os.path.exists(LOCAL):
    WORDS += [w.strip() for w in open(LOCAL) if w.strip() and not w.startswith('#')]
TEST_WORD = 'zzforbiddenzz'          # planted by --break, so the self-test needs no real word
FORBIDDEN = re.compile('|'.join(WORDS + [TEST_WORD]), re.I)
BAD_EXT = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac', '.webm', '.mp4', '.mkv', '.mov', '.npz', '.npy', '.pt', '.pth', '.so'}
MAX_FILE, MAX_TOTAL = 2 << 20, 10 << 20
SELF = os.path.relpath(os.path.abspath(__file__), ROOT)


def files():
    out = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines() if os.path.isfile(os.path.join(ROOT, f))]


def scan(paths):
    bad, total = [], 0
    for f in paths:
        p = os.path.join(ROOT, f)
        size = os.path.getsize(p)
        total += size
        if os.path.splitext(f)[1].lower() in BAD_EXT:
            bad.append(f'{f}: a media or data file')
        if size > MAX_FILE:
            bad.append(f'{f}: {size / 1e6:.1f} MB (> {MAX_FILE / 1e6:.0f} MB)')
        if f == SELF:
            continue
        for k, line in enumerate(open(p, errors='replace'), 1):
            if FORBIDDEN.search(line):
                bad.append(f'{f}:{k}: {line.strip()[:120]}')
    if total > MAX_TOTAL:
        bad.append(f'total {total / 1e6:.1f} MB (> {MAX_TOTAL / 1e6:.0f} MB)')
    return bad, total


if __name__ == '__main__':
    paths = files()
    planted = None
    if '--break' in sys.argv:
        planted = os.path.join(ROOT, 'PLANTED_FOR_SCAN_TEST.txt')
        open(planted, 'w').write(f'our lead, after {TEST_WORD}\n')
        paths.append(os.path.relpath(planted, ROOT))
    try:
        bad, total = scan(paths)
    finally:
        if planted:
            os.remove(planted)
    if not WORDS:
        print('note: tools/forbidden.local.txt is missing or empty, so no names are being blocked')
    if bad:
        print('NOT CLEAN:\n  ' + '\n  '.join(bad))
        sys.exit(1)
    print(f'clean: {len(paths)} files, {total / 1e3:.0f} KB')
