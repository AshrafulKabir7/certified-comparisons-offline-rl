"""Agreement between the final run (the source of every number in the paper) and the earlier Colab run.

Usage: python -I make_repro_summary.py <paper dir>
Reads <paper>/facts*.json (final run) and <paper>/original_run/facts*.json (earlier run) and writes
<paper>/facts_repro.json and tables/tab_repro.tex. Both runs executed the same analysis code on different hardware
(earlier: Colab T4, five sessions with cached results; final: RTX 4070 Ti SUPER, one uninterrupted run from an empty
folder), so GPU-trained values can differ in the last digits; exact computations should agree.
"""
import json
import math
import sys
from pathlib import Path

P = Path(sys.argv[1])
TAB = P / "tables"
new = {n: json.loads((P / f"{n}.json").read_text()) for n in ("facts", "facts_v3", "facts_robust")}
old = {n: json.loads((P / "original_run" / f"{n}.json").read_text()) for n in ("facts", "facts_v3", "facts_robust")}


def leaves(d, path=""):
    """(path, value) of every numeric leaf."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, f"{path}.{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(d, (int, float)) and not isinstance(d, bool) and not (isinstance(d, float) and math.isnan(d)):
        yield path, float(d)


def get(d, path):
    for k in path:
        d = d[k]
    return d


def max_abs(a, b):
    la, lb = dict(leaves(a)), dict(leaves(b))
    common = sorted(set(la) & set(lb))
    diffs = [abs(la[k] - lb[k]) for k in common]
    return {"leaves": len(common), "identical": sum(d == 0 for d in diffs), "max_abs_diff": max(diffs) if diffs else 0.0}


def by_cell(section, prefix):
    """max |diff| over the cells of facts.json[section] (a dict keyed by cell, or a list with 'cell') starting with prefix."""
    a, b = new["facts"][section], old["facts"][section]
    if isinstance(a, list):
        a = {r["cell"] + "|" + str(r.get("completion", "")): r for r in a}
        b = {r["cell"] + "|" + str(r.get("completion", "")): r for r in b}
    keys = [k for k in a if k.startswith(prefix) and k in b]
    return max_abs({k: a[k] for k in keys}, {k: b[k] for k in keys})


rows = {}
# Part I (neural and tabular learners on ICU-Sepsis): GPU-trained, so small differences are expected
mt_new = {r["policy"]: r for r in new["facts"]["icu_main_table"]}
mt_old = {r["policy"]: r for r in old["facts"]["icu_main_table"]}
d = [abs(mt_new[k]["J_mean"] - mt_old[k]["J_mean"]) for k in mt_new]
rank = lambda m: [k for k, _ in sorted(m.items(), key=lambda kv: -kv[1]["J_mean"])]
rows["part1_main_table"] = {"policies": len(d), "max_abs_diff_J": max(d), "mean_abs_diff_J": sum(d) / len(d),
                            "same_ranking": rank(mt_new) == rank(mt_old),
                            "top_policy_new": rank(mt_new)[0], "top_policy_old": rank(mt_old)[0]}
pt_new, pt_old = new["facts"]["icu_primary_test"], old["facts"]["icu_primary_test"]
rows["primary_test"] = {k: {"new": pt_new[k], "old": pt_old[k]} for k in ("mean_difference", "p_value", "tost_p")}
ic_new, ic_old = new["facts"]["icu_certified"]["part1_main_table"], old["facts"]["icu_certified"]["part1_main_table"]
rows["icu_main_table_certified"] = {k: {"new": ic_new[k], "old": ic_old[k]} for k in ("certified", "certified_free")}
p2n, p2o = new["facts"]["icu_certified"]["part2_icu_draws"], old["facts"]["icu_certified"]["part2_icu_draws"]
rows["icu_part2_certified_share"] = {"new": p2n["certified_share"], "old": p2o["certified_share"]}
# Part II simulators (exact evaluation of tabular methods)
cert = lambda f, tgt, cell: f["facts"]["certified_comparisons"][tgt][cell]
for tgt in ("true_rows", "bench"):
    for pool in ("pooled:gumbel_estimated", "pooled:garnet_all"):
        a, b = cert(new, tgt, pool), cert(old, tgt, pool)
        rows[f"{tgt}|{pool}"] = {k: {"new": a[k], "old": b[k]} for k in ("certified", "certified_errors",
                                                                         "certified_error_rate")}
rows["gumbel_certified_comparisons_all_cells"] = by_cell("certified_comparisons", "")  # filled below per family
cc_new, cc_old = new["facts"]["certified_comparisons"], old["facts"]["certified_comparisons"]
fam = {}
for f in ("gumbel", "garnet"):
    a = {t: {c: v for c, v in cc_new[t].items() if c.startswith(f)} for t in ("true_rows", "bench")}
    b = {t: {c: v for c, v in cc_old[t].items() if c.startswith(f)} for t in ("true_rows", "bench")}
    fam[f] = max_abs(a, b)
rows["gumbel_certified_comparisons_all_cells"] = fam["gumbel"]
rows["garnet_certified_comparisons_all_cells"] = fam["garnet"]
for f in ("gumbel", "garnet"):   # per-cell rates (shares and error rates), both targets
    cells = [(t, c) for t in ("true_rows", "bench") for c in cc_new[t] if c.startswith(f) and c in cc_old[t]]
    rows[f"{f}_cell_rates"] = {k: max(abs(cc_new[t][c][k] - cc_old[t][c][k]) for t, c in cells)
                               for k in ("certified_share", "certified_error_rate")}
    rows[f"{f}_cell_rates"]["cells"] = len(cells)
def tau_b_change(prefix):
    """max |diff| of the mean Kendall tau_b over the rank-fidelity rows (cell, completion) of one family."""
    key = lambda r: r["cell"] + "|" + str(r.get("completion", ""))
    a = {key(r): r for r in new["facts"]["rank_fidelity"] if r["cell"].startswith(prefix)}
    b = {key(r): r for r in old["facts"]["rank_fidelity"] if r["cell"].startswith(prefix)}
    d = [abs(a[k]["kendall_tau_b_mean"] - b[k]["kendall_tau_b_mean"]) for k in a if k in b]
    return {"rows": len(d), "max_abs_diff": max(d)}


rows["gumbel_rank_fidelity"] = tau_b_change("gumbel")   # mean Kendall tau_b only
rows["garnet_rank_fidelity"] = tau_b_change("garnet")
rows["H1"] = max_abs(new["facts"]["H1"], old["facts"]["H1"])
rows["H3_rho"] = {"new": new["facts"]["H3"]["spearman_rho"], "old": old["facts"]["H3"]["spearman_rho"]}
rows["kappa_icu_median"] = {"new": new["facts_robust"]["kappa"]["pools"]["bench"]["icu"]["kappa_star_quantiles"]["0.5"],
                            "old": old["facts_robust"]["kappa"]["pools"]["bench"]["icu"]["kappa_star_quantiles"]["0.5"]}
bh_new, bh_old = new["facts_v3"]["bootstrap_v3_headline"]["pooled:all"], old["facts_v3"]["bootstrap_v3_headline"]["pooled:all"]
rows["bootstrap_pooled"] = {k: {"new": bh_new[k], "old": bh_old[k]} for k in ("cert_error_point", "cert_error_boot")}
whole = {n: max_abs(new[n], old[n]) for n in new}
rows["whole_files"] = whole

(P / "facts_repro.json").write_text(json.dumps({"definition": __doc__.split("\n\n")[0], "rows": rows}, indent=1) + "\n")

# ------------------------------------------------------------------ table
def f3(x):
    return (f"{x:.4f}" if isinstance(x, float) else f"{x:,}".replace(",", "{,}")).replace("-", "$-$")


def pair(r, k, fmt=f3):
    return f"{fmt(r[k]['new'])} & {fmt(r[k]['old'])}"


pct = lambda x: f"{100 * x:.2f}\\%"
t = rows
L = [r"\begin{tabular}{lrr}", r"\toprule", r"quantity & final run & earlier run \\", r"\midrule",
     r"\multicolumn{3}{l}{\emph{Part I, \icu{} (GPU-trained learners, 10 seeds)}} \\",
     f"main table: largest $|\\Delta J|$ over {t['part1_main_table']['policies']} policies & "
     f"\\multicolumn{{2}}{{c}}{{{t['part1_main_table']['max_abs_diff_J']:.4f}}} \\\\",
     f"primary difference (diffusion $-$ projected IQL) & {pair(t['primary_test'], 'mean_difference')} \\\\",
     f"\\quad two-sided $p$ & {pair(t['primary_test'], 'p_value')} \\\\",
     f"certified main-table pairs (NHB-free) & {t['icu_main_table_certified']['certified']['new']} "
     f"({t['icu_main_table_certified']['certified_free']['new']}) & {t['icu_main_table_certified']['certified']['old']} "
     f"({t['icu_main_table_certified']['certified_free']['old']}) \\\\",
     r"\multicolumn{3}{l}{\emph{Part II, simulators (exact evaluation)}} \\",
     f"Gumbel certified claims, true rows (wrong) & {f3(t['true_rows|pooled:gumbel_estimated']['certified']['new'])} "
     f"({t['true_rows|pooled:gumbel_estimated']['certified_errors']['new']}) & "
     f"{f3(t['true_rows|pooled:gumbel_estimated']['certified']['old'])} "
     f"({t['true_rows|pooled:gumbel_estimated']['certified_errors']['old']}) \\\\",
     f"Gumbel certified error, estimated rows & {pct(t['bench|pooled:gumbel_estimated']['certified_error_rate']['new'])} & "
     f"{pct(t['bench|pooled:gumbel_estimated']['certified_error_rate']['old'])} \\\\",
     f"Garnet certified error, estimated rows & {pct(t['bench|pooled:garnet_all']['certified_error_rate']['new'])} & "
     f"{pct(t['bench|pooled:garnet_all']['certified_error_rate']['old'])} \\\\",
     f"per cell, largest $|\\Delta|$: Gumbel certified share / error / mean $\\tau_b$ & \\multicolumn{{2}}{{c}}{{"
     f"{t['gumbel_cell_rates']['certified_share']:.4f} / {t['gumbel_cell_rates']['certified_error_rate']:.4f} / "
     f"{t['gumbel_rank_fidelity']['max_abs_diff']:.4f}}} \\\\",
     f"\\quad Garnet certified share / error / mean $\\tau_b$ & \\multicolumn{{2}}{{c}}{{"
     f"{t['garnet_cell_rates']['certified_share']:.4f} / {t['garnet_cell_rates']['certified_error_rate']:.4f} / "
     f"{t['garnet_rank_fidelity']['max_abs_diff']:.4f}}} \\\\",
     f"H1 (six Gumbel cells): identical numbers & \\multicolumn{{2}}{{c}}{{{t['H1']['identical']} of {t['H1']['leaves']}}} \\\\",
     f"bootstrap: certified error, point / bootstrap & {pct(t['bootstrap_pooled']['cert_error_point']['new'])} / "
     f"{pct(t['bootstrap_pooled']['cert_error_boot']['new'])} & {pct(t['bootstrap_pooled']['cert_error_point']['old'])} / "
     f"{pct(t['bootstrap_pooled']['cert_error_boot']['old'])} \\\\",
     f"\\icu{{}} median breakdown $\\kappa^*$ & {pair(t, 'kappa_icu_median')} \\\\",
     r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_repro.tex").write_text("\n".join(L) + "\n")
print(json.dumps({k: v for k, v in rows.items() if k != "whole_files"}, indent=1)[:4000])
print("whole files:", whole)
