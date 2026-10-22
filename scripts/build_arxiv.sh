#!/usr/bin/env bash
# Build the arXiv preprint (named authors, code link) and its source tarball.
#   scripts/build_arxiv.sh   ->  paper/arxiv/  (sources + main.pdf)  and  paper/arxiv.tar.gz
# arXiv compiles with pdflatex and does not run BibTeX reliably, so the .bbl is shipped with the sources.
set -euo pipefail
cd "$(dirname "$0")/../paper"
rm -rf arxiv arxiv.tar.gz
mkdir -p arxiv/tables arxiv/figures
{ echo '\def\preprintmode{1}'; cat main.tex; } > arxiv/main.tex
cp tmlr.sty tmlr.bst fancyhdr.sty numbers.tex appendix_prompts.tex refs.bib arxiv/
cp tables/*.tex arxiv/tables/
# only the figures the paper includes
for f in $(grep -o 'figures/[A-Za-z0-9_]*\.pdf' main.tex | sort -u); do cp "$f" arxiv/figures/; done
( cd arxiv && tectonic -X compile --keep-intermediates main.tex >/dev/null )
( cd arxiv && rm -f main.aux main.log main.out main.blg )  # ship main.bbl (arXiv reads it) and refs.bib
grep -q 'bibitem' arxiv/main.bbl || { echo "main.bbl has no entries" >&2; exit 1; }
( cd arxiv && tar czf ../arxiv.tar.gz --exclude main.pdf --exclude '*.aux' --exclude '*.log' --exclude '*.out' . )
echo "built paper/arxiv/main.pdf and paper/arxiv.tar.gz ($(du -h arxiv.tar.gz | cut -f1))"
