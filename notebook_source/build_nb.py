#!/usr/bin/env python3
"""Build the Colab notebook from percent-format part files (no execution).

Part files live in ``build/`` next to this script and are concatenated in
sorted file-name order.  Each part file is a sequence of cells:

    # %% [markdown]
    # markdown line 1
    #
    # markdown line 3

    # %%
    code line 1
    code line 2

Format (exact and reversible):
  * a cell starts at a line that is exactly ``# %% [markdown]`` or ``# %%``;
  * markdown body lines are ``'# ' + line`` (an empty markdown line is ``#``);
  * code body lines are verbatim;
  * cells are separated by one blank line and the file ends with a newline.
    The parser removes exactly that one separator line, so blank lines that
    belong to a cell are preserved.

Usage
  python build_nb.py                       # build notebook + concatenated .py
  python build_nb.py --out X.ipynb --py X.py
  python build_nb.py --exclude 'part21h*' --verify-against REF.ipynb --strict
  python build_nb.py --verify-against REF.ipynb --expect-modified 5
      # every REF cell kept unchanged and in order (except the listed REF
      # cell indices); added cells are listed
  python build_nb.py extract REF.ipynb --bounds 0:p01,4:p02,...  (re-extract)

The notebook is never executed.  Outputs are empty and execution counts null.
"""
from __future__ import annotations

import argparse
import difflib
import fnmatch
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "build"
DEFAULT_OUT = HERE / "Sepsis_Offline_RL_v4.ipynb"
DEFAULT_PY = HERE / "Sepsis_Offline_RL_v4.py"

MD_HEADER = "# %% [markdown]"
CODE_HEADER = "# %%"
HEADERS = (MD_HEADER, CODE_HEADER)

NOTEBOOK_METADATA = {
    "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
}

Cell = Tuple[str, str]  # (cell_type, source text)


# ----------------------------------------------------------------------------
# Encoding / decoding
# ----------------------------------------------------------------------------
def encode_cell(cell_type: str, source: str) -> str:
    lines = source.split("\n")
    if cell_type == "markdown":
        body = "\n".join(("# " + l) if l != "" else "#" for l in lines)
        return MD_HEADER + "\n" + body
    if cell_type == "code":
        for l in lines:
            if l in HEADERS:
                raise ValueError("code cell contains a line equal to a cell "
                                 "header; the percent format cannot encode it")
        return CODE_HEADER + "\n" + source
    raise ValueError(f"unsupported cell type {cell_type!r}")


def encode_cells(cells: Sequence[Cell]) -> str:
    return "\n\n".join(encode_cell(t, s) for t, s in cells) + "\n"


def parse_part(text: str, name: str = "<part>") -> List[Cell]:
    if not text.endswith("\n"):
        raise ValueError(f"{name}: file must end with a newline")
    lines = text.split("\n")
    heads = [i for i, l in enumerate(lines) if l in HEADERS]
    if not heads:
        raise ValueError(f"{name}: no cell header found")
    pre = [l for l in lines[:heads[0]] if l.strip()]
    if pre:
        raise ValueError(f"{name}: text before the first cell header")
    cells: List[Cell] = []
    for k, h in enumerate(heads):
        end = heads[k + 1] if k + 1 < len(heads) else len(lines)
        body = lines[h + 1:end]
        if not body or body[-1] != "":
            raise ValueError(f"{name}: cell at line {h + 1} is not followed "
                             "by the blank separator line")
        body = body[:-1]
        if lines[h] == MD_HEADER:
            out = []
            for j, l in enumerate(body):
                if l == "#":
                    out.append("")
                elif l.startswith("# "):
                    out.append(l[2:])
                else:
                    raise ValueError(f"{name}: markdown line {h + 2 + j} does "
                                     "not start with '# '")
            cells.append(("markdown", "\n".join(out)))
        else:
            cells.append(("code", "\n".join(body)))
    return cells


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
def part_files(build_dir: Path, exclude: Sequence[str] = ()) -> List[Path]:
    files = sorted(p for p in build_dir.glob("part*.py") if p.is_file())
    return [p for p in files
            if not any(fnmatch.fnmatch(p.name, pat) for pat in exclude)]


def load_cells(files: Sequence[Path]) -> Tuple[List[Cell], List[str]]:
    cells: List[Cell] = []
    origin: List[str] = []
    for p in files:
        cs = parse_part(p.read_text(encoding="utf-8"), p.name)
        cells.extend(cs)
        origin.extend([p.name] * len(cs))
    return cells, origin


def to_notebook(cells: Sequence[Cell]) -> Dict:
    nb_cells = []
    for t, s in cells:
        src = s.splitlines(True)
        if t == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {},
                             "source": src})
        else:
            nb_cells.append({"cell_type": "code", "execution_count": None,
                             "metadata": {}, "outputs": [], "source": src})
    return {"nbformat": 4, "nbformat_minor": 0,
            "metadata": json.loads(json.dumps(NOTEBOOK_METADATA)),
            "cells": nb_cells}


def notebook_cells(path: Path) -> List[Cell]:
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        src = c["source"]
        out.append((c["cell_type"], "".join(src) if isinstance(src, list)
                    else src))
    return out


def verify(cells: Sequence[Cell], origin: Sequence[str], ref: Path,
           strict: bool, expect_modified: Sequence[int] = ()) -> bool:
    rcells = notebook_cells(ref)
    if strict:
        ok = len(cells) == len(rcells) and all(
            a == b for a, b in zip(cells, rcells))
        if ok:
            print(f"VERIFY strict: {len(cells)} cells identical to {ref.name}")
            return True
        n = min(len(cells), len(rcells))
        first = next((i for i in range(n) if cells[i] != rcells[i]), n)
        print(f"VERIFY strict FAILED: built {len(cells)} cells, ref "
              f"{len(rcells)}; first difference at cell {first}")
        return False
    # alignment: every reference cell is kept unchanged (or is an expected
    # modification), in order; added cells are listed
    sm = difflib.SequenceMatcher(None, rcells, list(cells), autojunk=False)
    equal, added, modified, deleted = 0, [], [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            equal += i2 - i1
            continue
        n_pair = min(i2 - i1, j2 - j1) if tag == "replace" else 0
        modified += [(i1 + k, j1 + k) for k in range(n_pair)]
        deleted += list(range(i1 + n_pair, i2))
        added += list(range(j1 + n_pair, j2))
    bad_mod = [(i, j) for i, j in modified if i not in set(expect_modified)]
    ok = not deleted and not bad_mod
    print(f"VERIFY alignment: {equal}/{len(rcells)} reference cells identical; "
          f"{len(modified)} modified; {len(deleted)} deleted; "
          f"{len(added)} added")
    for i, j in modified:
        flag = "expected" if i in set(expect_modified) else "UNEXPECTED"
        print(f"   ~ ref cell {i} -> cell {j} ({origin[j]}), {flag}:")
        diff = difflib.unified_diff(rcells[i][1].split("\n"),
                                    cells[j][1].split("\n"), lineterm="", n=0)
        for line in list(diff)[2:]:
            print("       " + line[:150])
    for i in deleted:
        print(f"   - ref cell {i} deleted")
    for j in added:
        first = cells[j][1].split("\n", 1)[0][:90]
        print(f"   + cell {j:3d} [{cells[j][0][:4]}] {origin[j]}: {first}")
    return ok


def extract(ref: Path, bounds: List[Tuple[int, str]], build_dir: Path) -> None:
    rcells = notebook_cells(ref)
    bounds = sorted(bounds)
    if bounds[0][0] != 0:
        raise ValueError("first bound must start at cell 0")
    build_dir.mkdir(parents=True, exist_ok=True)
    for k, (start, name) in enumerate(bounds):
        end = bounds[k + 1][0] if k + 1 < len(bounds) else len(rcells)
        text = encode_cells(rcells[start:end])
        (build_dir / f"{name}.py").write_text(text, encoding="utf-8")
        print(f"wrote {name}.py  cells {start}..{end - 1}")


def main(argv: Sequence[str] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "extract":
        ap = argparse.ArgumentParser(prog="build_nb.py extract")
        ap.add_argument("ref", type=Path)
        ap.add_argument("--bounds", required=True,
                        help="comma list start:name, e.g. 0:part01,4:part02")
        ap.add_argument("--build-dir", type=Path, default=BUILD_DIR)
        a = ap.parse_args(argv[1:])
        bounds = [(int(s.split(":")[0]), s.split(":")[1])
                  for s in a.bounds.split(",")]
        extract(a.ref, bounds, a.build_dir)
        return 0
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--py", type=Path, default=DEFAULT_PY)
    ap.add_argument("--exclude", action="append", default=[],
                    help="fnmatch pattern of part files to skip (repeatable)")
    ap.add_argument("--verify-against", type=Path, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="with --verify-against: require identical cell lists")
    ap.add_argument("--expect-modified", type=int, action="append", default=[],
                    help="reference cell index allowed to differ (repeatable)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    files = part_files(a.build_dir, a.exclude)
    cells, origin = load_cells(files)
    print(f"{len(files)} part files, {len(cells)} cells "
          f"({sum(t == 'code' for t, _ in cells)} code)")
    ok = True
    if a.verify_against is not None:
        ok = verify(cells, origin, a.verify_against, a.strict,
                    a.expect_modified)
    if not a.no_write:
        nb = to_notebook(cells)
        a.out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
        a.py.write_text(encode_cells(cells), encoding="utf-8")
        print(f"wrote {a.out}\nwrote {a.py}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
