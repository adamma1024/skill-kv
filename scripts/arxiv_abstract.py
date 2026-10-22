"""Print the paper's abstract as plain ASCII text for the arXiv metadata form.

Expands the \\newcommand macros from numbers.tex (and simple ones defined in main.tex), strips formatting, and fails if the
result is not ASCII or exceeds arXiv's 1920-character limit, so the form always matches the PDF.

  uv run python scripts/arxiv_abstract.py paper/main.tex
"""

import re
import sys
from pathlib import Path

LIMIT = 1920


def macros(*files: Path) -> dict[str, str]:
    out = {}
    for f in files:
        for name, body in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", f.read_text(), re.M):
            out[name] = body
    return out


def plain(tex: str, defs: dict[str, str]) -> str:
    for _ in range(3):  # macros may expand to other macros
        tex = re.sub(r"\\(\w+)\\? ?", lambda m: defs.get(m.group(1), m.group(0)) + (" " if m.group(0).endswith(" ") and m.group(1) in defs else ""), tex)
    tex = re.sub(r"\\(emph|textit|textbf|cond|texttt)\{([^{}]*)\}", r"\2", tex)
    repl = {"---": " - ", "--": "-", "$-$": "-", "\\%": "%", "~": " ", "``": '"', "''": '"', "\\,": " ", "{,}": ","}
    for a, b in repl.items():
        tex = tex.replace(a, b)
    tex = re.sub(r"\$([^$]*)\$", r"\1", tex)
    tex = tex.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", tex).strip()


def main() -> None:
    main_tex = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex")
    src = main_tex.read_text()
    body = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S)
    if not body:
        raise SystemExit("no abstract environment in " + str(main_tex))
    text = plain(body.group(1), macros(main_tex.with_name("numbers.tex"), main_tex))
    if "\\" in text:
        raise SystemExit(f"unexpanded LaTeX left in the abstract: {re.findall(r'\\\\\\w+', text)}")
    if not text.isascii():
        raise SystemExit("abstract has non-ASCII characters: " + "".join(sorted({c for c in text if ord(c) > 127})))
    if len(text) > LIMIT:
        raise SystemExit(f"abstract is {len(text)} characters; arXiv allows {LIMIT}")
    print(text)


if __name__ == "__main__":
    main()
