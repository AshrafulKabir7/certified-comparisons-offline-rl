"""Extract the results of the additional experiments (II.8) and the re-run Part I outputs from the executed v3 notebook.

Usage: python -I extract_v3.py <executed_notebook.ipynb> <out_dir>
Reads only the stored cell outputs (nothing is executed): the HTML tables the notebook displays are saved as CSV, and
the summary lines it prints (theorem checks, bootstrap and driver summaries, gate, compute) go to text_facts.json.
The tables are displayed rounded to 4 decimals (II.8i rounds before display), so rates are known to 1e-4.
"""
import ast
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

NB, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
cells = json.loads(NB.read_text())["cells"]


def text_of(cell):
    parts = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append("".join(o.get("text", "")))
        elif "text/plain" in o.get("data", {}):
            parts.append("".join(o["data"]["text/plain"]))
    return "".join(parts)


def html_tables(cell):
    out = []
    for o in cell.get("outputs", []):
        html = o.get("data", {}).get("text/html")
        if html:
            for df in pd.read_html(io.StringIO("".join(html))):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [b if not str(b).startswith("Unnamed") else a for a, b in df.columns]
                out.append(df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore"))
    return out


def find_cell(marker):
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and marker in text_of(c)]
    assert len(hits) == 1, f"{marker!r}: {hits}"
    return hits[0]


def json_after(text, marker):
    """The first top-level JSON object printed after `marker`."""
    i = text.index(marker)
    j = text.index("{", i)
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return json.loads(text[j:k + 1])
    raise ValueError(marker)


# ---- HTML tables, located by the section banner each cell prints
TABLES = {"=== II.6 Exact-tabular cross-check ===": ["cross_check"],
          "II.8b physician report": ["physicians_diag"],
          "II.8 environment suite built": ["aggregated_diag"],
          "=== II.8 Revision experiments: ICU-Sepsis harm curves ===": ["icu_harm"],
          "=== II.8 Revision analyses ===": ["bracket_v3", "certified_v3", "bootstrap_v3"]}
for marker, names in TABLES.items():
    dfs = html_tables(cells[find_cell(marker)])
    assert len(dfs) == len(names), (marker, len(dfs))
    for nm, df in zip(names, dfs):
        df.to_csv(OUT / f"{nm}.csv", index=False)

# ---- printed summaries
F = {"notebook": NB.name}
t = text_of(cells[find_cell("II.8i theorem checks")])
m = re.search(r"II\.8i theorem checks: (\{.*?\}) \| passes: (\w+)", t)
F["theorem_checks"] = {**ast.literal_eval(m.group(1)), "passes": m.group(2) == "True"}

t = text_of(cells[find_cell("II.8 driver complete")])
F["v3_driver"] = json_after(t, "II.8 driver: ")  # summary JSON printed after the progress lines and the table
# per-cell true values of the reference rows and projected IQL (mean, std over draws), printed as a text table
rows, cell = [], None
for line in t[t.index("mean     std  count"):t.index("{")].splitlines()[2:]:
    m = re.match(r"^(\S+)?\s+(IQL \[projected\]|REF [^\d-]+?)\s+(-?[\d.]+)\s+([\d.]+)\s+(\d+)$", line)
    if m:
        cell = m.group(1) or cell
        rows.append({"cell": cell, "policy": m.group(2).strip(), "mean": float(m.group(3)), "std": float(m.group(4)),
                     "draws": int(m.group(5))})
pd.DataFrame(rows).to_csv(OUT / "v3_driver_refs.csv", index=False)
F["v3_code_hash"] = re.search(r"II\.8 code hash: (\w+)", text_of(cells[find_cell("II.8 code hash")])).group(1)
F["part2_code_hash"] = re.search(r"Part II code hash: (\w+)", text_of(cells[find_cell("II.8 code hash")])).group(1)

t = text_of(cells[find_cell("II.8g bootstrap:")])
F["bootstrap_summary"] = json_after(t, "II.8g bootstrap:")

t = text_of(cells[find_cell("II.8h harm curves:")])
F["harm_curves_status"] = ast.literal_eval(re.search(r"II\.8h harm curves: (\{.*?\})", t).group(1))

t = text_of(cells[find_cell("=== 26. Selection protocol (M04) ===")])
F["selection_m04"] = {"evaluations_skipped": t.count("evaluation skipped"),
                      "skipped_mean": t.count("evaluation skipped (mean)"),
                      "skipped_terminate": t.count("evaluation skipped (terminate)"),
                      "skip_reason": sorted(set(re.findall(r"evaluation skipped \(\w+\): (.*)", t))),
                      "selected_tables_empty": t.count("Empty DataFrame")}

t = text_of(cells[find_cell("=== 34. Final integrity gate ===")])
F["revision_gate"] = json.loads(re.search(r"Revision gate \(II\.8\): (\{.*\})", t).group(1))
F["part2_gate"] = json_after(t, "Part II improper rows")
m = re.search(r"icu\s+fqe_dev\s+(\d+)\s+(\d+)", t)
F["icu_fqe_dev_improper"] = {"rows": int(m.group(1)), "improper": int(m.group(2))}
F["all_structural_checks_passed"] = "All structural integrity checks passed" in t

t = text_of(cells[find_cell("Logged compute:")])
m = re.search(r"Logged compute: ([\d.]+) h over (\d+) session.*?Part I': ([\d.]+), 'Part II': ([\d.]+).*?II\.8 draws ([\d.]+) h, "
              r"II\.8 bootstrap draws ([\d.]+) h", t)
F["compute"] = {"total_h": float(m.group(1)), "sessions": int(m.group(2)), "part1_h": float(m.group(3)),
                "part2_h": float(m.group(4)), "v3_draws_h": float(m.group(5)), "v3_bootstrap_h": float(m.group(6))}

t = text_of(cells[find_cell("=== 35. Paper hand-off ===")])
F["handoff_missing_evidence"] = "Missing evidence files: none" in t
F["prereg_status"] = re.findall(r"PREREG(?:_PART2)? status: '([^']*)'", t)

(OUT / "text_facts.json").write_text(json.dumps(F, indent=1))
print("written:", sorted(p.name for p in OUT.iterdir()))
