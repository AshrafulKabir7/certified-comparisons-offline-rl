"""Facts and tables of the confidence-set certificates (notebook Section II.9c) and the sensitivity cells (II.9e).

Usage: python -I make_v4_assets.py <run paper/tables dir> <paper dir>
Reads <tables>/xenv/v4_cs_summary.csv, v4_cs_summary.json, v4_sens_summary.csv, v4_config.json and writes
<paper>/facts_v4.json, tables/tab_v4_cs.tex and tables/tab_v4_sens.tex. Nothing is recomputed: every number is the
notebook's, only selected and formatted.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

SRC, OUT = Path(sys.argv[1]) / "xenv", Path(sys.argv[2])
TAB = OUT / "tables"
RANGE = {"gumbel": 2.0, "garnet": 1.0}          # v_max - v_min (Gumbel returns in [-1, 1], Garnet in [0, 1])


def num(x):
    """JSON-safe float: NaN and inf become None."""
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


def pct(x, d=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{100 * x:.{d}f}"


# ---------------------------------------------------------------- II.9c confidence sets
cs = pd.read_csv(SRC / "v4_cs_summary.csv")
cs_json = json.loads((SRC / "v4_cs_summary.json").read_text())
cfg = json.loads((SRC / "v4_config.json").read_text())
assert cs_json["status"] == "run" and cs_json["checks"]["passes"] and cs_json["selftest"]["passes"]
assert cs_json["draws_complete"]


def cs_row(unit, test, scale=None):
    q = cs[(cs.unit == unit) & (cs.test == test)]
    if scale is not None:
        q = q[q.scale == scale]
    assert len(q) == 1, (unit, test, scale, len(q))
    return q.iloc[0]


CS_UNITS = [u for u in dict.fromkeys(cs.unit) if not u.startswith("pooled:")] + ["pooled:gumbel", "pooled:garnet",
                                                                                 "pooled:all"]
facts_cs = {"delta": cfg["cs_delta"], "scales": cfg["cs_scales"], "valid_scale": cs_json["config"]["valid_scale"],
            "selftest": cs_json["selftest"], "checks": cs_json["checks"], "draws": cs_json["draws"], "units": {}}
for u in CS_UNITS:
    fam = "garnet" if "garnet" in u else ("gumbel" if "gumbel" in u else "all")
    d = {}
    for sc in cfg["cs_scales"]:
        for t in ("free", "nhb", "paired_free", "paired_nhb"):
            r = cs_row(u, t, sc)
            d[f"{t}@{sc}"] = {"claims": int(r.claims), "errors": int(r.errors), "pairs": int(r.pairs),
                              "coverage": num(r.coverage), "mean_width": num(r.mean_width),
                              "share_unrestricted": num(r.share_unrestricted),
                              "radius_median_finite": num(r.radius_median_finite), "r0_mean": num(r.r0_mean)}
    for t in ("point_nhb", "point_free"):
        r = cs_row(u, t)
        d[t] = {"claims": int(r.claims), "errors": int(r.errors), "pairs": int(r.pairs), "share": num(r.share),
                "error_rate": num(r.error_rate), "coverage": num(r.coverage), "mean_width": num(r.mean_width)}
    d["value_range"] = RANGE.get(fam)
    facts_cs["units"][u] = d
facts_cs["claims_any_set_test"] = int(cs[cs.set_kind.isin(["valid", "heuristic"])].claims.sum())
facts_cs["min_coverage_set_tests"] = num(cs[cs.set_kind.isin(["valid", "heuristic"])].coverage.min())

CS_LABEL = {"gumbel|best|n2000|tau5|eps0.05": r"Gumbel 2k, $\tau{=}5$",
            "gumbel|best|n10000|tau20|eps0.05": r"Gumbel 10k, $\tau{=}20$",
            "gumbel|best|n10000|tau20|eps0.05|phys-withhold-vaso": r"Gumbel 10k, withhold-vaso",
            "garnet|h-0.5|eps0.05": r"Garnet $h{=}{-}0.5$", "garnet|h0.0|eps0.05": r"Garnet $h{=}0$",
            "garnet|h0.5|eps0.05": r"Garnet $h{=}0.5$", "garnet|h1.0|eps0.05": r"Garnet $h{=}1$",
            "pooled:gumbel": "Gumbel, pooled", "pooled:garnet": "Garnet, pooled", "pooled:all": "all 7 cells"}
lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"& \multicolumn{5}{c}{confidence set (95\%), NHB test} & \multicolumn{3}{c}{point interval} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){7-9}",
         r"cell & cover. & width & unrestr. & unrestr.\ (0.1) & cert.\ (1/0.3/0.1) & cover. & cert. & err. \\",
         r"\midrule"]
for u in CS_UNITS:
    d = facts_cs["units"][u]
    v, h = d["nhb@1.0"], d["nhb@0.1"]
    c = "/".join(str(d[f"nhb@{s}"]["claims"]) for s in cfg["cs_scales"])
    w = v["mean_width"] / d["value_range"] if d["value_range"] else None
    p = d["point_nhb"]
    if u == "pooled:gumbel":
        lines.append(r"\midrule")
    lines.append(f"{CS_LABEL[u]} & {pct(v['coverage'], 0)} & {pct(w, 0)} & {pct(v['share_unrestricted'], 0)} & "
                 f"{pct(h['share_unrestricted'], 0)} & {c} & {pct(p['coverage'])} & {pct(p['share'])} & "
                 f"{pct(p['error_rate'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_v4_cs.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- II.9e sensitivity cells
ss = pd.read_csv(SRC / "v4_sens_summary.csv")
sj = json.loads((SRC / "v4_sens_summary.json").read_text())
assert sj["status"] == "run" and sj.get("draws_complete") in (True, None)


def side(unit, s):
    q = ss[(ss.cell == unit) & (ss.side == s)]
    assert len(q) == 1, (unit, s, len(q))
    return q.iloc[0]


FIELDS = {"n_draws": "n_draws", "single_action_share": "constr_single_admissible_fraction",
          "mean_supported_actions": "constr_mean_admissible_actions",
          "true_upper_coverage": "sem_upper_coverage_violating",
          "true_cert_share": "cc_true_rows_cert_nhb_share", "true_cert_errors": "cc_true_rows_cert_nhb_errors",
          "true_cert_claims": "cc_true_rows_cert_nhb_claims",
          "true_nhb_only_claims": "cc_true_rows_cert_nhb_only_claims",
          "true_nhb_only_errors": "cc_true_rows_cert_nhb_only_errors",
          "true_point_mean_error": "cc_true_rows_point_mean_error_rate",
          "true_point_term_error": "cc_true_rows_point_term_error_rate",
          "bench_coverage": "bench_coverage_violating",
          "bench_cert_share": "cc_bench_cert_nhb_share", "bench_cert_error": "cc_bench_cert_nhb_error_rate",
          "bench_cert_free_error": "cc_bench_cert_free_error_rate",
          "bench_point_mean_error": "cc_bench_point_mean_error_rate",
          "bench_point_term_error": "cc_bench_point_term_error_rate",
          "tau_mean": "tau_mean_mean", "tau_term": "tau_terminate_mean", "tau_diff": "tau_diff_mean",
          "tau_diff_ci_low": "tau_diff_ci_low", "tau_diff_ci_high": "tau_diff_ci_high"}
VARIANTS = [  # (key, unit in v4_sens_summary.csv, label)
    ("phys_eps0.3", "gumbel|best|n10000|tau20|eps0.05|v4eps0.3", r"physician $\epsilon{=}0.3$"),
    ("phys_eps0.5", "gumbel|best|n10000|tau20|eps0.05|v4eps0.5", r"physician $\epsilon{=}0.5$"),
    ("timeout_eta0.02_n10k", "gumbel|best|n10000|tau20|eps0.05|eta0.02", r"timeout $\eta{=}0.02$, 10k"),
    ("timeout_eta0.1_n10k", "gumbel|best|n10000|tau20|eps0.05|eta0.1", r"timeout $\eta{=}0.1$, 10k"),
    ("timeout_eta0.02_n2k", "gumbel|best|n2000|tau5|eps0.05|eta0.02", r"timeout $\eta{=}0.02$, 2k"),
    ("timeout_eta0.1_n2k", "gumbel|best|n2000|tau5|eps0.05|eta0.1", r"timeout $\eta{=}0.1$, 2k"),
    ("deadrule_gumbel", "pooled:deadrule|gumbel", r"dead states by argmax, Gumbel (6)"),
    ("deadrule_garnet", "pooled:deadrule|garnet", r"dead states by argmax, Garnet (2)"),
    ("agg_glucose", "gumbel|agg-glucose|n10000|tau20|eps0.05", r"aggregation: glucose hidden, diabetes shown"),
    ("agg_kmeans150", "gumbel|agg-kmeans150|n10000|tau20|eps0.05", r"aggregation: $k$-means 150 (diabetes hidden)"),
    ("agg_kmeans50", "gumbel|agg-kmeans50|n10000|tau20|eps0.05", r"aggregation: $k$-means 50 (diabetes hidden)"),
]
facts_sens = {"config": {k: cfg[k] for k in ("timeout_cells", "deadrule_cells", "support_base", "support_eps",
                                             "aggregation_base", "aggregations")}, "variants": {}}
for key, unit, _ in VARIANTS:
    facts_sens["variants"][key] = {"unit": unit, "reference": str(side(unit, "new")["reference"])}
    for s in ("new", "base"):
        r = side(unit, s)
        facts_sens["variants"][key][s] = {k: num(r[c]) for k, c in FIELDS.items()}


def cell(v, f, d=1, ref=True):
    a, b = v["new"][f], v["base"][f]
    return pct(a, d) + (f" ({pct(b, d)})" if ref else "")


def tau(v):
    a, b = v["new"]["tau_diff"], v["base"]["tau_diff"]
    return "--" if a is None else f"{a:+.2f} ({b:+.2f})".replace("+-", "-").replace("-", "$-$")


def nhb_only(v):
    a, e = v["new"]["true_nhb_only_claims"], v["new"]["true_nhb_only_errors"]
    return "--" if a is None else f"{int(e)} / {int(a):,}".replace(",", "{,}")


lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"& & \multicolumn{1}{c}{true rows} & \multicolumn{4}{c}{estimated rows} \\",
         r"\cmidrule(lr){3-3}\cmidrule(lr){4-7}",
         r"variant (reference) & one action & NHB-only wrong & certified & cert.\ err. & point err.\ (mean/term) & "
         r"$\tau_b$(mean) $-$ $\tau_b$(term) \\",
         r"\midrule"]
for i, (key, unit, label) in enumerate(VARIANTS):
    v = facts_sens["variants"][key]
    if i in (2, 6, 8):
        lines.append(r"\midrule")
    pm = f"{pct(v['new']['bench_point_mean_error'])}/{pct(v['new']['bench_point_term_error'])}"
    lines.append(f"{label} & {cell(v, 'single_action_share', 0)} & {nhb_only(v)} & {cell(v, 'bench_cert_share')} & "
                 f"{cell(v, 'bench_cert_error')} & {pm} & {tau(v)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_v4_sens.tex").write_text("\n".join(lines) + "\n")

(OUT / "facts_v4.json").write_text(json.dumps({"source": "notebook Sections II.9c and II.9e (final run)",
                                               "confidence_sets": facts_cs, "sensitivity": facts_sens},
                                              indent=1) + "\n")
print("wrote facts_v4.json, tables/tab_v4_cs.tex, tables/tab_v4_sens.tex")
