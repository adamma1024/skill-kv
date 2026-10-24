#!/usr/bin/env bash
# Build the anonymized supplementary-material zip for double-blind review (e.g. TMLR on OpenReview).
#   scripts/build_supplement.sh [out.zip]
# Copies code, data, logs and the paper sources, strips author identity (names, accounts, URLs, local paths, the
# lab notebook, the arXiv metadata) and fails if any identifier survives. Venue limit: TMLR allows ZIP or PDF up to 100 MB.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$(cd "$(dirname "${1:-$root/paper/supplement.zip}")" && pwd)/$(basename "${1:-supplement.zip}")"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
dst="$work/supplement"
mkdir -p "$dst"

rsync -a "$root"/ "$dst"/ \
  --exclude .git --exclude .venv --exclude packs --exclude logs --exclude docs \
  --exclude 'paper/arxiv' --exclude 'paper/arxiv.tar.gz' --exclude 'paper/arxiv_metadata.md' \
  --exclude 'paper/supplement.zip' --exclude 'paper/*.pdf' --exclude 'paper/tmlr_submission.md' \
  --exclude .anon-identifiers \
  --exclude .DS_Store --exclude __pycache__ --exclude .pytest_cache --exclude '*.pyc'

# author identity -> anonymous placeholders
python3 - "$dst" <<'PY'
import re, sys
from pathlib import Path
d = Path(sys.argv[1])
def sub(path, pairs):
    p = d / path
    s = p.read_text()
    for a, b in pairs:
        s = re.sub(a, b, s)
    p.write_text(s)
sub("LICENSE", [(r"Copyright \(c\) (\d{4}) .*", r"Copyright (c) \1 Anonymous Authors")])
sub("README.md", [(r"author\s*=\s*\{[^}]*\}", "author = {Anonymous}"),
                  (r"https?://github\.com/[\w.-]+/[\w.-]+", "(anonymized)"),
                  (r"\(github\.com/[\w.-]+/[\w.-]+\)", "")])
sub("paper/main.tex", [(r"\\author\{\\name [^\\]+ \\email [^ ]+ \\\\\s*\n\s*\\addr [^}]*\}",
                        r"\\author{\\name Anonymous Authors \\email anonymous@example.com \\\\\n      \\addr Anonymous affiliation}"),
                       (r"\\url\{https://github\.com/[^}]*\}", r"(anonymized)")])
PY

# fail on any surviving identifier. The identifiers themselves are private, so they live in a git-ignored file
# (.anon-identifiers at the repo root: one case-insensitive regex alternative per line, e.g. your name, usernames,
# employer, local folder names). Benchmark fixtures such as john.doe@gmail.com are fictional and allowed.
idfile="$root/.anon-identifiers"
[[ -s "$idfile" ]] || { echo "missing $idfile: list the strings that identify you, one per line" >&2; exit 1; }
names="$(grep -v '^[[:space:]]*\(#\|$\)' "$idfile" | paste -sd'|' -)"
paths='/Users/[a-z]|/home/[a-z]'   # case-sensitive: model outputs legitimately contain api.github.com/users/...
hits="$( { rg -n -i "$names" "$dst"; rg -n "$paths" "$dst"; } 2>/dev/null || true)"
if [[ -n "$hits" ]]; then echo "identifiers remain:" >&2; echo "$hits" | cut -c1-200 | head -20 >&2; exit 1; fi

rm -f "$out"
( cd "$work" && zip -qrX "$out" supplement )
size=$(stat -f %z "$out" 2>/dev/null || stat -c %s "$out")
[[ $size -lt 100000000 ]] || { echo "zip is $size bytes; TMLR allows 100 MB" >&2; exit 1; }
echo "built $out ($(du -h "$out" | cut -f1)); no identifiers found"
