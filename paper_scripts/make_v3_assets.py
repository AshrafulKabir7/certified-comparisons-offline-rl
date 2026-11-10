"""Facts sheet and supplement tables of the additional experiments (notebook v3, Sections II.6b, II.8 and 26).

Usage: python -I make_v3_assets.py <paper_dir>
Reads <paper_dir>/v3_tables (written by extract_v3.py from the executed notebook) and writes facts_v3.json and
tables/tab_v3_*.tex, tables/tab_crosscheck.tex. Nothing is recomputed: rates are the notebook's, rounded to 1e-4.
The rows of the additional cells in tables/tab_certified.tex are written by make_assets.py from the same CSV files.
"""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1])
V3, TAB = OUT / "v3_tables", OUT / "tables"
rd = lambda nm: pd.read_csv(V3 / f"{nm}.csv")
TF = json.loads((V3 / "text_facts.json").read_text())
# After use_exact_v3.py, the bracketing, certified, bootstrap and harm tables are the notebook's full-precision CSV
# files (with counts); the physician, environment and cross-check tables stay as displayed (rounded).
EXACT = (V3 / "exact_sources.json").exists()

CELL_SHORT = {"gumbel|best|n10000|tau20|eps0.05|phys-confounded": "confounded",
              "gumbel|best|n10000|tau20|eps0.05|phys-withhold-abx": "withhold-abx",
              "gumbel|agg-obs|n10000|tau20|eps0.05": "agg-obs", "gumbel|agg-proj|n10000|tau20|eps0.05": "agg-proj",
              "garnet|hrand|eps0.05": "hrand-eps0.05", "garnet|hrand|eps0.3": "hrand-eps0.3"}


def full_cell(c):
    """The displayed tables truncate long cell names ('...'); map them back."""
    if c.endswith("..."):
        hits = [k for k in CELL_SHORT if k.startswith(c[:-3])]
        assert len(hits) == 1, c
        return hits[0]
    return c


def records(df, key="cell"):
    df = df.copy()
    if key in df:
        df[key] = df[key].map(full_cell)
    return {str(r[key]): {k: (None if pd.isna(v) else v) for k, v in r.items() if k != key} for r in df.to_dict("records")}


def pct(v, nd=1):
    if v is None or pd.isna(v):
        return "--"
    v = 100.0 * float(v)
    if v == 0.0:
        return "0"
    if v < 10 ** -nd:
        return r"$<$" + f"{10 ** -nd:g}"
    return f"{v:.{nd}f}"


def pct4(v):
    """A rate displayed by the notebook to 4 decimals, as a percentage with 2 decimals: exact at the source precision
    (printing 1 decimal would round twice)."""
    if v is None or pd.isna(v):
        return "--"
    return f"{100.0 * float(v):.2f}"


def exact_share(v, pairs):
    """Recover an integer count from a share displayed to 4 decimals when the count is unique (pairs < 5,000)."""
    cands = [c for c in range(int((v - 5e-5) * pairs) - 1, int((v + 5e-5) * pairs) + 2) if abs(c / pairs - v) <= 5e-5 + 1e-12]
    assert len(cands) == 1, (v, pairs, cands)
    return cands[0] / pairs


def num(v, nd=3):
    if v is None or pd.isna(v):
        return "--"
    s = f"{float(v):.{nd}f}"
    return ("$-$" + s[1:]) if s.startswith("-") else s


FACTS = {"source": TF["notebook"],
         "precision": ("bracketing, certified, bootstrap and harm tables: full precision (notebook CSV files, "
                       "exact_sources.json); physician, environment and cross-check tables: displayed values"
                       if EXACT else "rates as displayed by the notebook, rounded to 1e-4"),
         "exact_sources": json.loads((V3 / "exact_sources.json").read_text()) if EXACT else None,
         "text_facts": TF}

# ---- physicians and environment suite of the II.8 cells (draw 0)
phys = rd("physicians_diag").set_index("kind")
FACTS["physicians"] = {k: {c: float(v) for c, v in r.items()} for k, r in phys.iterrows()}
env = rd("aggregated_diag")
env["cell"] = env["cell"].map(full_cell)
FACTS["environment_suite_v3_draw0"] = records(env)
refs = rd("v3_driver_refs")
FACTS["v3_reference_values_true"] = {c: {p: {"mean": m, "std": s} for p, m, s in zip(g["policy"], g["mean"], g["std"])}
                                     for c, g in refs.groupby("cell")}

# ---- bracketing, certified comparisons, bootstrap intervals (II.8i)
FACTS["bracketing_v3"] = records(rd("bracket_v3"))
cert = rd("certified_v3")
FACTS["certified_v3"] = {t: records(g.drop(columns="targets")) for t, g in cert.groupby("targets")}
boot = rd("bootstrap_v3")
boot["cell"] = boot["cell"].map(lambda c: full_cell(c) if not c.endswith("...") else
                                "gumbel|best|n10000|tau20|eps0.05|phys-withhold-vaso")
FACTS["bootstrap_v3"] = {iv: records(g.drop(columns="interval")) for iv, g in boot.groupby("interval")}
_bp, _bb = FACTS["bootstrap_v3"]["point"], FACTS["bootstrap_v3"]["bootstrap"]
FACTS["bootstrap_v3_headline"] = {
    k: {"cert_error_point": _bp[k]["cc_certified_error_rate"], "cert_error_boot": _bb[k]["cc_certified_error_rate"],
        "cert_share_point": _bp[k]["cc_certified_share"], "cert_share_boot": _bb[k]["cc_certified_share"],
        "coverage_point": _bp[k]["coverage_violating"], "coverage_boot": _bb[k]["coverage_violating"],
        "width_point": _bp[k]["mean_width_violating"], "width_boot": _bb[k]["mean_width_violating"],
        **({"cert_point": _bp[k]["cc_certified"], "cert_errors_point": _bp[k]["cc_certified_errors"],
            "cert_boot": _bb[k]["cc_certified"], "cert_errors_boot": _bb[k]["cc_certified_errors"],
            "pairs": _bp[k]["cc_pairs"]} if EXACT else {})}
    for k in ("pooled:all", "pooled:gumbel", "pooled:garnet", "gumbel|best|n2000|tau5|eps0.05")}

# ---- ICU-Sepsis harm curves (II.8h)
harm = rd("icu_harm")
harm["pairs"] = harm["n_policies"] * (harm["n_policies"] - 1) // 2
harm["reversal_share"] = harm["reversals_vs_lambda0"] / harm["pairs"]
FACTS["icu_harm"] = {f"{r.source}|{r.policy_set}|lambda{r['lambda']:g}": {k: (v.item() if hasattr(v, "item") else v)
                     for k, v in r.items()} for _, r in harm.iterrows()}

# ---- exact-tabular vs SGD cross-check (II.6b) and the Part I selection protocol (Section 26)
cc = rd("cross_check")
FACTS["cross_check"] = {f"{r.algo}|{r.variant}": {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
                        for _, r in cc.iterrows()}
FACTS["selection_m04"] = {**TF["selection_m04"], "candidates": 27,
                          "candidates_definition": "3 tabular families (CQL, IQL, BCQ) x 3 configurations x 3 extractions",
                          "part2_icu_fqe_dev_improper": TF["icu_fqe_dev_improper"]}

(OUT / "facts_v3.json").write_text(json.dumps(FACTS, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

# =============================== tables ===============================
# ---- tab_v3_cells: the additional cells (draw 0 construction; values in truth, means over 30 draws)
LAB = {"confounded": "Gumbel, confounded physician", "withhold-abx": "Gumbel, withhold-abx physician",
       "agg-obs": "Gumbel, aggregated (diabetes hidden)", "agg-proj": "Gumbel, aggregated (projected)",
       "hrand-eps0.05": r"Garnet, random rows, $\epsilon{=}0.05$", "hrand-eps0.3": r"Garnet, random rows, $\epsilon{=}0.3$"}
lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r" & \multicolumn{5}{c}{benchmark of draw 0} & \multicolumn{3}{c}{true value (30 draws)} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){7-9}",
         r"Cell & live & mean $|\M|$ & $|\M|{=}1$ & opt.\ $\in\M$ & $V_{\mathrm{unsupp}}$ & $\pibhat$ & proj.\ IQL & $J^*_{\M}$ \\",
         r"\midrule"]
for c, short in CELL_SHORT.items():
    e, rf = FACTS["environment_suite_v3_draw0"][c], FACTS["v3_reference_values_true"][c]
    lines.append(f"{LAB[short]} & {int(e['live_states'])} & {e['mean_admissible_actions']:.2f} & "
                 f"{pct(e['construction_single_admissible_fraction'])} & {pct(e['construction_true_optimum_in_mask_fraction'])} & "
                 f"{num(e['value_of_unsupported_actions'], 4)} & {num(rf['REF behaviour']['mean'], 4)} & "
                 f"{num(rf['IQL [projected]']['mean'], 4)} & {num(rf['REF optimum (admissible)']['mean'], 4)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_v3_cells.tex").write_text("\n".join(lines) + "\n")

# ---- tab_v3_physicians: the four Gumbel physicians on the true system
PLAB = {"optimal": r"$\epsilon$-optimal (main cells)", "withhold-vaso": "withhold-vasopressor (control)",
        "confounded": "confounded (diabetes-blind)", "withhold-abx": "withhold-antibiotics"}
lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r" & & & \multicolumn{3}{c}{disagrees with optimum (\%)} & \multicolumn{3}{c}{treatment share (\%)} \\",
         r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
         r"Physician ($\epsilon{=}0.05$) & value & core & live & occup. & diabetic & abx & vent & vaso \\", r"\midrule"]
for k, lab_ in PLAB.items():
    r = FACTS["physicians"][k]
    lines.append(f"{lab_} & {num(r['value_true'], 4)} & {num(r['value_deterministic_core'], 4)} & {pct4(r['disagree_live_fraction'])} & "
                 f"{pct4(r['disagree_occupancy_weighted'])} & {pct4(r['disagree_diabetic_live_fraction'])} & "
                 f"{pct4(r['antibiotic_share_live'])} & {pct4(r['ventilation_share_live'])} & {pct4(r['vasopressor_share_live'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_v3_physicians.tex").write_text("\n".join(lines) + "\n")

# ---- tab_v3_bootstrap: point vs bootstrap-ensemble intervals
BLAB = {"gumbel|best|n2000|tau5|eps0.05": r"Gumbel 2k, $\tau{=}5$", "gumbel|best|n10000|tau20|eps0.05": r"Gumbel 10k, $\tau{=}20$",
        "gumbel|best|n10000|tau20|eps0.05|phys-withhold-vaso": r"Gumbel 10k, $\tau{=}20$, withhold-vaso",
        "garnet|h-0.5|eps0.05": r"Garnet $h{=}{-}0.5$, $\epsilon{=}0.05$", "garnet|h0.0|eps0.05": r"Garnet $h{=}0$, $\epsilon{=}0.05$",
        "garnet|h0.5|eps0.05": r"Garnet $h{=}0.5$, $\epsilon{=}0.05$", "garnet|h1.0|eps0.05": r"Garnet $h{=}1$, $\epsilon{=}0.05$",
        "pooled:gumbel": "pooled Gumbel (3 cells)", "pooled:garnet": "pooled Garnet (4 cells)", "pooled:all": "pooled (7 cells)"}
lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r" & & \multicolumn{2}{c}{coverage} & \multicolumn{2}{c}{mean width} & \multicolumn{2}{c}{certified share} & "
         r"\multicolumn{2}{c}{certified err.} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"Cell & $n_{p>0}$ & point & boot & point & boot & point & boot & point & boot \\", r"\midrule"]
for k, lab_ in BLAB.items():
    if k == "pooled:gumbel":
        lines.append(r"\midrule")
    p, b = _bp[k], _bb[k]
    rt = pct if EXACT else pct4          # exact rates: one decimal; displayed rates: two (no double rounding)
    lines.append(f"{lab_} & {int(p['n_violating']):,} & {rt(p['coverage_violating'])} & {rt(b['coverage_violating'])} & "
                 f"{num(p['mean_width_violating'], 4)} & {num(b['mean_width_violating'], 4)} & {rt(p['cc_certified_share'])} & "
                 f"{rt(b['cc_certified_share'])} & {rt(p['cc_certified_error_rate'])} & {rt(b['cc_certified_error_rate'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_v3_bootstrap.tex").write_text("\n".join(lines) + "\n")

# ---- tab_v3_harm: ICU-Sepsis harm curves
# The notebook's Part I "main_table" set (33 policies at seed 0) is not the 19-policy main table of the paper, so it is
# left out of the table (it stays in facts_v3.json).
HLAB = {("part1", "all"): "Part I, all policies (seed 0)", ("part2", "default"): "Part II, default policies (draw 0)"}
lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r" & & \multicolumn{2}{c}{$\tau_b$ vs.\ $\lambda{=}0$} & \multicolumn{2}{c}{reversed pairs (\%)} & & \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Policy set & $n$ & $\lambda{=}0.5$ & $\lambda{=}1$ & $\lambda{=}0.5$ & $\lambda{=}1$ & certified & $\lambda$-robust \\",
         r"\midrule"]
for (src, ps), lab_ in HLAB.items():
    h5, h1 = FACTS["icu_harm"][f"{src}|{ps}|lambda0.5"], FACTS["icu_harm"][f"{src}|{ps}|lambda1"]
    npairs = h1["pairs"]
    lines.append(f"{lab_} & {h1['n_policies']} & {h5['kendall_tau_b_vs_lambda0']:.4f} & {h1['kendall_tau_b_vs_lambda0']:.4f} & "
                 f"{pct(h5['reversal_share'])} & {pct(h1['reversal_share'])} & {pct(exact_share(h1['certified_share'], npairs))} & "
                 f"{pct(exact_share(h1['lambda_robust_share'], npairs))} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_v3_harm.tex").write_text("\n".join(lines) + "\n")

# ---- tab_crosscheck: exact tabular (Part II solvers) vs Part I minibatch-SGD learners, 10 seeds
lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r" & & \multicolumn{3}{c}{greedy-action agreement (\%)} & \multicolumn{2}{c}{$|\Delta J|$, mean rule} & "
         r"\multicolumn{2}{c}{$|\Delta J|$, terminate} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Method & policy & live & seen & occup. & mean & max & mean & max \\", r"\midrule"]
for algo in ("iql", "cql", "bcq"):
    for var in ("native", "masked"):
        r = FACTS["cross_check"][f"{algo}|{var}"]
        # agreement is printed by the notebook to 5 decimals; one decimal is unambiguous unless the 2nd and 3rd decimals
        # of the percentage are exactly 50 (asserted)
        for k_ in ("agreement_live", "agreement_seen", "agreement_occupancy"):
            assert round(1e5 * r[k_]) % 100 != 50, (algo, var, k_)
        lines.append(f"{algo.upper()} & {var} & {pct(r['agreement_live'])} & {pct(r['agreement_seen'])} & "
                     f"{pct(r['agreement_occupancy'])} & {r['mean_abs_diff_mean']:.4f} & {r['max_abs_diff_mean']:.4f} & "
                     f"{r['mean_abs_diff_terminate']:.4f} & {r['max_abs_diff_terminate']:.4f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_crosscheck.tex").write_text("\n".join(lines) + "\n")
print("v3 assets written:", sorted(p.name for p in TAB.glob("tab_v3_*.tex")) + ["tab_crosscheck.tex", "facts_v3.json"])
