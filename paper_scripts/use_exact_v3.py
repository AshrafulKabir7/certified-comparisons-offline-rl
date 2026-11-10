"""Replace the rounded revision tables (v3_tables/, read from the notebook's displayed outputs) by the notebook's own
full-precision CSV files, which also carry the integer counts.

Usage: python -I use_exact_v3.py <paper_dir> [<dir holding the v3_*.csv files>]
Without the second argument the script searches the folders in $V3_SEARCH (os.pathsep-separated) and ~/Downloads
for v3_certified.csv.
Inputs (notebook Section II.8, written to tables/xenv/ on Drive): v3_bracketing.csv, v3_certified.csv,
v3_bootstrap_coverage.csv, v3_icu_harm_summary.csv.

Every exact rate is checked against the rate the notebook displayed (|exact - displayed| <= 5e-5), so a stale or
foreign file is refused. The exact tables keep the column names that make_v3_assets.py and make_assets.py read, add
the count columns, and v3_tables/exact_sources.json records the files used (SHA-256) and the checks. The rounded
tables are kept as *_displayed.csv.

After the swap, rerun make_v3_assets.py and make_assets.py, then replace the text that worked around rounding:
  10.95% (pooled bootstrap point error: abstract, Sections 5.3, 5.5, 7) -> facts_v3.json bootstrap_v3_headline
  "at most 0.02%" / "<=0.02%" / "up to 0.02%" (abstract, intro, Section 5.1) -> certified_v3 true_rows counts
  "about 12--13%" (Section 5.3, aggregated, proj.) -> certified_v3 bench point error rates
  F2_revision.tex precision sentence and the tab:boot/tab_certified comments.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1])
V3 = OUT / "v3_tables"
NEEDED = ["v3_bracketing.csv", "v3_certified.csv", "v3_bootstrap_coverage.csv", "v3_icu_harm_summary.csv"]
SEARCH = [Path(s) for s in os.environ.get("V3_SEARCH", "").split(os.pathsep) if s] + [Path.home() / "Downloads"]


def locate():
    if len(sys.argv) > 2:
        d = Path(sys.argv[2])
        missing = [f for f in NEEDED if not (d / f).exists()]
        assert not missing, f"missing in {d}: {missing}"
        return d
    hits = sorted({p.parent for root in SEARCH if root.exists() for p in root.rglob("v3_certified.csv")},
                  key=lambda d: (d / "v3_certified.csv").stat().st_mtime, reverse=True)
    for d in hits:
        if all((d / f).exists() for f in NEEDED):
            return d
    raise SystemExit(f"v3 CSV files not found under {[str(s) for s in SEARCH]} (need {NEEDED})")


SRC = locate()
sha = {f: hashlib.sha256((SRC / f).read_bytes()).hexdigest() for f in NEEDED}
print("using", SRC)


def displayed(nm):
    """The rounded table extracted from the notebook output (kept once as *_displayed.csv)."""
    keep = V3 / f"{nm}_displayed.csv"
    if not keep.exists():
        (V3 / f"{nm}.csv").rename(keep)
    return pd.read_csv(keep)


def short(c):
    return c[:-3] if isinstance(c, str) and c.endswith("...") else c


def match(exact, shown, keys, cols, label):
    """Join the displayed rows to the exact rows (displayed cell names may be truncated with '...') and check every
    displayed rate. Returns the exact rows in the displayed order, restricted to the displayed rows."""
    out, worst = [], 0.0
    for _, r in shown.iterrows():
        m = exact
        for k in keys:
            v = r[k]
            if k == "cell" and str(v).endswith("..."):          # truncated in the display
                m = m[m[k].astype(str).str.startswith(short(v))]
            else:
                m = m[m[k] == v]
        assert len(m) == 1, (label, dict(r[keys]), len(m))
        e = m.iloc[0]
        for c in cols:
            if pd.isna(r[c]) and pd.isna(e[c]):
                continue
            dev = abs(float(e[c]) - float(r[c]))
            worst = max(worst, dev)
            assert dev <= 5e-5 + 1e-12, (label, dict(r[keys]), c, float(e[c]), float(r[c]))
        out.append(e)
    print(f"{label}: {len(out)} rows match the displayed table (largest deviation {worst:.2e})")
    return pd.DataFrame(out).reset_index(drop=True), worst


checks = {}
ex = {f: pd.read_csv(SRC / f) for f in NEEDED}

# bracketing: semantics-only and benchmark-level coverage per cell and pool
d = displayed("bracket_v3")
rate_cols = ["sem_upper_coverage_violating", "bench_coverage_violating"]
e, checks["bracket_v3"] = match(ex["v3_bracketing.csv"], d, ["cell"], rate_cols + ["sem_lower_violations"], "bracketing")
e.to_csv(V3 / "bracket_v3.csv", index=False)

# certified comparisons (true_rows and bench targets)
d = displayed("certified_v3")
rate_cols = ["certified_share", "certified_error_rate", "point_mean_error_rate", "point_terminate_error_rate"]
e, checks["certified_v3"] = match(ex["v3_certified.csv"], d, ["cell", "targets"], rate_cols, "certified")
e.to_csv(V3 / "certified_v3.csv", index=False)

# bootstrap coverage (point vs bootstrap intervals)
d = displayed("bootstrap_v3")
rate_cols = ["coverage_violating", "mean_width_violating", "cc_certified_share", "cc_certified_error_rate"]
e, checks["bootstrap_v3"] = match(ex["v3_bootstrap_coverage.csv"], d, ["cell", "interval"], rate_cols, "bootstrap")
e.to_csv(V3 / "bootstrap_v3.csv", index=False)

# ICU-Sepsis harm curves (shares are count / pairs; tau_b is displayed to 4 decimals)
d = displayed("icu_harm")
h = ex["v3_icu_harm_summary.csv"].copy()
h["lambda"] = h["lambda"].round(10)
d["lambda"] = d["lambda"].round(10)
e, checks["icu_harm"] = match(h, d, ["source", "policy_set", "lambda"],
                              ["kendall_tau_b_vs_lambda0", "certified_share", "lambda_robust_share"], "harm curves")
assert (e["reversals_vs_lambda0"].to_numpy() == d["reversals_vs_lambda0"].to_numpy()).all(), "reversal counts differ"
e.to_csv(V3 / "icu_harm.csv", index=False)

(V3 / "exact_sources.json").write_text(json.dumps({"source_dir": "tables/xenv (notebook v3, Section II.8)",
                                                   "files": sha, "largest_deviation_from_displayed": checks}, indent=1))
print("exact tables written; now run make_v3_assets.py and make_assets.py")
