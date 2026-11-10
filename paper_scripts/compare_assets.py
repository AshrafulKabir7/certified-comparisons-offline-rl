"""Compare regenerated paper assets with the files used in the paper.

Usage: python -I compare_assets.py <regenerated_dir> <paper_assets_expected_dir> <results/v3_tables>
Tables (.tex) and the tables extracted from the final-run record must be identical. Facts files are compared value by value; floats
may differ by 1e-9 (relative) across library versions. Figures are only listed. Exit code 1 on any mismatch.
"""
import json
import math
import sys
from pathlib import Path

NEW, EXP, V3 = (Path(a) for a in sys.argv[1:4])
REL = 1e-9
problems = []


def same(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                problems.append(f"{path}.{k}: present in only one file")
            else:
                same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{path}: list length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            problems.append(f"{path}: {a!r} vs {b!r}")
        elif not (a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= REL * max(1.0, abs(a), abs(b))):
            problems.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        problems.append(f"{path}: {a!r} vs {b!r}")


n_tex = 0
for f in sorted((EXP / "tables").glob("*.tex")):
    g = NEW / "tables" / f.name
    n_tex += 1
    if not g.exists():
        problems.append(f"tables/{f.name}: not regenerated")
    elif g.read_text() != f.read_text():
        problems.append(f"tables/{f.name}: differs")
n_v3 = 0
for f in sorted(V3.iterdir()):
    g = NEW / "v3_tables" / f.name
    n_v3 += 1
    if not g.exists():
        problems.append(f"v3_tables/{f.name}: not regenerated")
    elif f.suffix == ".json":
        same(json.loads(g.read_text()), json.loads(f.read_text()), f"v3_tables/{f.name}")
    elif g.read_text() != f.read_text():
        problems.append(f"v3_tables/{f.name}: differs")
FACTS = ("facts.json", "facts_v3.json", "facts_robust.json", "facts_v4.json", "facts_repro.json", "theory_verification.json")
for name in FACTS:
    g = NEW / name
    if not g.exists():
        problems.append(f"{name}: not regenerated")
    else:
        same(json.loads(g.read_text()), json.loads((EXP / name).read_text()), name)
figs = sorted(p.name for p in (NEW / "figures").glob("*.pdf"))
print(f"compared {n_tex} tables, {n_v3} tables extracted from the final-run record and {len(FACTS)} facts files; figures written: {figs}")
if problems:
    print(f"{len(problems)} mismatch(es):")
    print("\n".join(problems[:50]))
    sys.exit(1)
print("all tables and facts match the paper")
