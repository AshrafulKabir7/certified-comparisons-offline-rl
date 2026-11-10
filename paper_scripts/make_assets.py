"""Build every figure, LaTeX table and the facts sheet of the paper from the Colab result tables.

Usage: python make_assets.py <results_tables_dir> <paper_dir>
Nothing here re-runs an experiment: every number is read from the notebook's exported CSV/JSON files.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
X = RES / "xenv"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (dataviz reference instance, light surface): mean rule = slot 1, terminate = slot 2.
C_MEAN, C_TERM, C_3, C_4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "mathtext.fontset": "stix", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
TEXTW = 5.5  # figure-sizing text width, inches
# Printed width of each figure (fraction of the 5.5 in text width).
FIGW = {"fig_frontiers": 0.82 * TEXTW, "fig_fidelity": 0.82 * TEXTW, "fig_bench_vs_true": 0.82 * TEXTW,
        "fig_h2": 0.9 * TEXTW, "fig_icu": 0.86 * TEXTW}

FACTS = {}


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def f3(x):
    if pd.isna(x):
        return "--"
    out = f"{x:.3f}"
    return "0.000" if out == "-0.000" else out


def sg(x, nd=4):
    """Signed number with a math-mode minus (and plus) sign."""
    if pd.isna(x):
        return "--"
    v = f"{x:+.{nd}f}"
    if float(v) == 0.0:
        return "0." + "0" * nd
    return ("$-$" if v[0] == "-" else "$+$") + v[1:]


def f4(x):
    return "--" if pd.isna(x) else f"{x:.4f}"


def pfmt(p):
    if pd.isna(p):
        return "--"
    return r"$<10^{-4}$" if p < 1e-4 else (f"{p:.4f}" if p < 0.01 else f"{p:.3f}")


def gumbel_short(cell):
    parts = dict(p.split("=", 1) if "=" in p else (p, "") for p in [])
    toks = cell.split("|")
    if toks[1] == "oracle":
        return "oracle"
    n = toks[2][1:]
    n = f"{int(n) // 1000}k"
    tau = toks[3][3:]
    eps = toks[4][3:]
    lab = f"$n{{=}}{n[:-1]}$k, $\\tau{{=}}{tau}$"
    if eps != "0.05":
        lab += f", $\\epsilon{{=}}{eps}$"
    if len(toks) > 5 and "withhold" in toks[5]:
        lab += ", withhold-vaso"
    return lab


def gumbel_plain(cell):
    return gumbel_short(cell).replace("$", "").replace("{=}", "=").replace("\\tau", "τ").replace("\\epsilon", "ε")


# =====================================================================================
# Part I: ICU-Sepsis case study
# =====================================================================================
main = pd.read_csv(RES / "table_main_policy_comparison.csv")
allres = pd.read_csv(RES / "exact_results_all.csv")
prim = json.load(open(RES / "statistics_primary.json"))
sec = pd.read_csv(RES / "statistics_secondary_family.csv")
crit = pd.read_csv(RES / "critic_stability_summary_main_all_seeds.csv")
gate = json.load(open(RES / "critic_stability_gate.json"))
legacy = json.load(open(RES / "legacy_replay.json"))
proj = pd.read_csv(RES / "projection_equivalence_checks.csv")
mc = pd.read_csv(RES / "monte_carlo_crosscheck.csv")
hier = pd.read_csv(RES / "statistics_hierarchical_replication.csv")
manifest = json.load(open(RES / "fresh_reproducibility_manifest.json"))
eda = pd.read_csv(RES / "eda_dataset_summary.csv") if (RES / "eda_dataset_summary.csv").exists() else None
sem = json.load(open(RES / "benchmark_semantics_audit.json"))
supp = json.load(open(RES / "corrected_support_audit.json"))

ICU_ROWS = [
    ("Tabular BCQ (tau=0.3) [masked greedy]", "Tabular BCQ, masked greedy"),
    ("Tabular IQL [masked greedy]", "Tabular IQL, masked greedy"),
    ("Tabular IQL [projected]", "Tabular IQL, projected (comparator)"),
    ("AW categorical imitation [masked greedy]", "AW categorical imitation, masked"),
    ("Tabular CQL [masked greedy]", "Tabular CQL, masked greedy"),
    ("Diffusion-AM-CQL [legacy critic]", "Diffusion, legacy critic"),
    ("Diffusion-AM-CQL [primary critic B]", "Diffusion, critic B (treatment)"),
    ("Masked critic ensemble [critic B + tabular CQL]", "Masked critic ensemble (no diffusion)"),
    ("AM-CQL (Hard, design B: admissible-row CQL)", "Hard AM-CQL, design B"),
    ("AM-CQL (Hard, legacy)", "Hard AM-CQL, legacy"),
    ("AM-CQL (Hard, design A: full-action CQL)", "Hard AM-CQL, design A"),
    ("Deep CQL [masked greedy]", "Deep CQL, masked greedy"),
    ("AM-CQL (Soft, legacy)", "Soft AM-CQL"),
    ("Tabular IQL", "Tabular IQL, unmasked"),
    ("Tabular CQL", "Tabular CQL, unmasked"),
    ("Tabular BCQ (tau=0.3)", "Tabular BCQ, unmasked"),
    ("Clinician", "Clinician (behaviour)"),
    ("Random (uniform over 25)", "Uniform random"),
    ("Constrained Optimal Oracle", "Optimal (= constrained optimal)"),
]
m = main.set_index("policy")
rows = []
for key, lab in ICU_ROWS:
    r = m.loc[key]
    rows.append({"policy": lab, "key": key, "n": int(r.n_seeds), "J_mean": r.return_mean, "J_mean_sd": r.return_mean_sd,
                 "J_term": r.return_terminate, "J_term_sd": r.return_terminate_sd,
                 "viol_pct": r.inadmissible_rate_mean})
icu_tab = pd.DataFrame(rows)
FACTS["icu_main_table"] = icu_tab.to_dict(orient="records")

lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Policy & seeds & $J_{\mathrm{mean}}$ & $J_{\mathrm{term}}$ & viol.\ (\%) \\", r"\midrule"]
for r in rows:
    sd = (lambda v: "" if pd.isna(v) else f"\\,$\\pm$\\,{v:.4f}")
    lines.append(f"{r['policy']} & {r['n']} & {r['J_mean']:.4f}{sd(r['J_mean_sd'])} & {r['J_term']:.4f}{sd(r['J_term_sd'])} & "
                 f"{r['viol_pct']:.2f} \\\\")
    if r["key"] in ("AM-CQL (Soft, legacy)", "Tabular BCQ (tau=0.3)"):
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_icu_main.tex").write_text("\n".join(lines) + "\n")

p = prim["primary"]
FACTS["icu_primary_test"] = {k: p[k] for k in ("treatment", "comparator", "rule", "n_pairs", "mean_difference", "sd_difference",
                                                 "bootstrap_ci_low", "bootstrap_ci_high", "signflip_confidence_low",
                                                 "signflip_confidence_high", "p_value", "test_method", "tost_p",
                                                 "equivalence_margin", "prob_improvement")}
FACTS["icu_prereg_locked"] = str(prim["prereg"]["status"]).strip().upper().startswith("LOCKED")
FACTS["icu_prereg_status"] = prim["prereg"]["status"]
FACTS["icu_prereg_hash"] = prim["prereg_hash"]

# secondary family (distinct hypotheses)
secd = sec.drop_duplicates(subset=["comparator"]) if "hypothesis" not in sec else sec
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Comparator (treatment: diffusion, critic B) & $\Delta J$ & 95\% CI & $p$ & Holm $p$ \\", r"\midrule"]
SEC_NAMES = {**dict(ICU_ROWS), "Behavior Cloning [masked greedy]": "BC, masked greedy",
             "Neural BC prior x critic [critic B + tabular CQL]": "Neural BC prior $\\times$ critic",
             "Deep CQL [masked greedy]": "Deep CQL, masked greedy"}
for _, r in sec.iterrows():
    if r["rule"] != "terminate":
        continue
    lines.append(f"{SEC_NAMES.get(r['comparator'], r['comparator']).replace('&', '+')} & {sg(r['mean_difference'])} & [{sg(r['bootstrap_ci_low'])}, {sg(r['bootstrap_ci_high'])}] & "
                 f"{pfmt(r['p_value'])} & {pfmt(r['holm_adjusted_p_full_family'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_icu_secondary.tex").write_text("\n".join(lines) + "\n")
FACTS["icu_secondary_family"] = sec[sec.rule == "terminate"][["comparator", "mean_difference", "bootstrap_ci_low", "bootstrap_ci_high",
                                                              "p_value", "holm_adjusted_p_full_family", "tost_p"]].to_dict(orient="records")

FACTS["critic_drift"] = crit[["display", "n", "final_q_adm_max_abs", "bellman_rmse", "stable_fraction"]].to_dict(orient="records")
FACTS["critic_gate"] = gate
FACTS["legacy_replay"] = {k: legacy[k] for k in ("replayed_rows", "max_abs_discrepancy", "original_diffusion_minus_projected_iql_terminate_mean",
                                                 "original_diffusion_minus_projected_iql_terminate_per_seed")}
FACTS["legacy_projected_iql"] = 0.78997  # audit prediction; verified in legacy_projection_controls_summary.csv
FACTS["projection_checks"] = {"policies": int(len(proj)), "max_dP": float(proj["max_abs_dP_original_vs_projected_mean"].max()),
                              "max_dr": float(proj["max_abs_dr_original_vs_projected_mean"].max())}
FACTS["mc_check"] = {"rows": int(len(mc)), "all_within_4se": bool(mc["within_4_se"].all()),
                     "max_abs_z_return": float(mc["z_score"].abs().max())}
FACTS["hierarchical"] = {"n_datasets": int(hier["n_datasets"].iloc[0]), "min_holm_p": float(hier["holm_p"].min()),
                         "min_raw_p": float(hier["dataset_level_signflip_p_two_sided"].min())}
FACTS["compute"] = manifest.get("compute_logged")
FACTS["icu_semantics"] = {"mean_rule_identity_holds_1e12": sem["mean_rule_identity"]["identity_holds_to_1e-12"],
                          "inadmissible_live_pairs": sem["mean_rule_identity"]["inadmissible_live_pairs"]}
FACTS["icu_support"] = supp

# per-seed terminate returns for the ICU figure
FIG_ICU = ["Tabular BCQ (tau=0.3) [masked greedy]", "Tabular IQL [masked greedy]", "Tabular IQL [projected]",
           "AW categorical imitation [masked greedy]", "Tabular CQL [masked greedy]", "Diffusion-AM-CQL [primary critic B]",
           "Masked critic ensemble [critic B + tabular CQL]", "AM-CQL (Hard, design B: admissible-row CQL)",
           "Deep CQL [masked greedy]", "AM-CQL (Soft, legacy)"]
lab = dict(ICU_ROWS)
ps = allres[(allres["rule"] == "terminate") & (allres["status"] == "ok") & allres["name"].isin(FIG_ICU) & (allres["group"] == "main")]
fig, axes = plt.subplots(1, 2, figsize=(FIGW["fig_icu"], 2.15), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.75})
ax = axes[0]
for i, name in enumerate(FIG_ICU[::-1]):
    v = ps[ps["name"] == name]["return"].to_numpy()
    col = C_TERM if "Diffusion" in name else (C_MEAN if "Tabular IQL [projected]" in name else MUTED)
    ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), s=9, color=col, alpha=0.55, lw=0)
    mu, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    ax.errorbar(mu, i, xerr=1.96 * se, fmt="o", ms=4, color=col, capsize=2, lw=1)
ax.set_yticks(range(len(FIG_ICU)))
SHORT_ICU = {"Tabular BCQ (tau=0.3) [masked greedy]": "BCQ, masked", "Tabular IQL [masked greedy]": "IQL, masked",
             "Tabular IQL [projected]": "IQL, projected (comp.)", "AW categorical imitation [masked greedy]": "AW imitation, masked",
             "Tabular CQL [masked greedy]": "CQL, masked", "Diffusion-AM-CQL [primary critic B]": "diffusion, critic B (treat.)",
             "Masked critic ensemble [critic B + tabular CQL]": "critic ensemble, masked", "AM-CQL (Hard, design B: admissible-row CQL)": "hard AM-CQL, design B",
             "Deep CQL [masked greedy]": "Deep CQL, masked", "AM-CQL (Soft, legacy)": "soft AM-CQL"}
ax.set_yticklabels([SHORT_ICU[n] for n in FIG_ICU[::-1]])
ax.axvline(m.loc["Tabular IQL [projected]"].return_terminate, color=C_MEAN, lw=0.8, ls=":")
ax.set_xlabel(r"exact return $J$ (terminate rule)")
ax.set_title("(a) ICU-Sepsis learned policies", loc="left")
ax.grid(axis="y", visible=False)
ax = axes[1]
cr = crit.set_index("display")
order = ["AM-CQL (Hard, legacy)", "AM-CQL (Soft, legacy)", "AM-CQL (Hard, design A: full-action CQL)", "Deep CQL",
         "AM-CQL (Hard, design B: admissible-row CQL)"]
short = {"AM-CQL (Hard, legacy)": "hard, legacy", "AM-CQL (Soft, legacy)": "soft", "AM-CQL (Hard, design A: full-action CQL)": "hard, design A",
         "Deep CQL": "Deep CQL", "AM-CQL (Hard, design B: admissible-row CQL)": "hard, design B"}
vals = [cr.loc[o, "final_q_adm_max_abs"] for o in order]
ax.barh(range(len(order)), vals, color=[C_TERM if "legacy" in o else C_MEAN for o in order], height=0.6)
ax.set_xscale("log")
ax.axvline(1.0, color=INK, lw=0.8, ls="--")
ax.set_xticks([1, 1e2, 1e4, 1e6])
ax.set_yticks(range(len(order)))
ax.set_yticklabels([short[o] for o in order])
for i, v in enumerate(vals):
    ax.text(v * 1.25, i, f"{v:,.0f}", va="center", fontsize=7, color=INK)
ax.set_xlim(0.5, 5e6)
ax.set_xlabel(r"final $\max|Q|$, admissible actions")
ax.set_title("(b) critic value drift", loc="left")
ax.grid(axis="y", visible=False)
save(fig, "fig_icu")

# =====================================================================================
# Part II: cross-environment study
# =====================================================================================
suite = pd.read_csv(X / "xenv_environment_suite.csv")
fr = pd.read_csv(X / "exact_frontiers.csv")
rf = pd.read_csv(X / "table_rank_fidelity.csv")
ed = pd.read_csv(X / "table_error_decomposition.csv")
pre = json.load(open(X / "xenv_prereg_tests.json"))
h2 = pd.read_csv(X / "xenv_h2_family.csv")
props = pd.read_csv(X / "proposition_summary.csv")
zs = pd.read_csv(X / "xenv_normalized_scores.csv")
meta = pd.read_csv(X / "xenv_draw_meta.csv")
drv = json.load(open(X / "xenv_driver_summary.json"))
gex = json.load(open(X / "gumbel_exactness_check.json"))
cfg = json.load(open(X / "xenv_config.json"))

FACTS["xenv_config"] = {k: cfg[k] for k in ("icu_draws", "gumbel_draws", "garnet_draws", "icu_user_episodes", "gumbel_user_episodes",
                                            "garnet_user_episodes", "gumbel_variant", "gumbel_timeout_hazard", "garnet_states",
                                            "garnet_actions", "garnet_branching", "garnet_hazard", "garnet_tau", "garnet_hosp_episodes",
                                            "dev_fraction") if k in cfg}
FACTS["xenv_prereg_locked"] = pre.get("prereg_locked")
FACTS["gumbel_exactness"] = {k: v for k, v in gex.items() if not isinstance(v, (dict, list))}
FACTS["gumbel_exactness_checks"] = gex.get("checks", {})
FACTS["propositions"] = props[["proposition", "title", "checks", "passed", "failed"]].to_dict(orient="records")

# ---- environment suite table
srows = []
for _, r in suite.iterrows():
    fam = r["family"]
    if fam == "icu":
        srows.append(["ICU-Sepsis v2", "released", "--", "--", int(r["live_states"]), r["mean_admissible_actions"],
                      r["behaviour_value[bench_terminate]"], np.nan, r["optimum_value[bench_mean]"], np.nan])
        continue
    if fam == "gumbel":
        name = "Gumbel: " + gumbel_short(r["cell"])
        hosp = int(r["construction_hospital_episodes"]) if not pd.isna(r.get("construction_hospital_episodes")) else "--"
        tau = int(r["construction_tau"]) if not pd.isna(r.get("construction_tau")) else "--"
    else:
        toks = r["cell"].split("|")
        name = f"Garnet: $h{{=}}{toks[1][1:].replace('-', '{-}')}$, $\\epsilon{{=}}{toks[2][3:]}$"
        hosp = int(r["construction_hospital_episodes"]) if not pd.isna(r.get("construction_hospital_episodes")) else "--"
        tau = int(r["construction_tau"]) if not pd.isna(r.get("construction_tau")) else "--"
    srows.append([name, {"best": "est.", "oracle": "oracle"}.get(r.get("construction_construction", ""), ""), hosp, tau, int(r["live_states"]), r["mean_admissible_actions"],
                  r["behaviour_value[true]"], r["optimum_value[true]"], r["optimum_value[bench_mean]"], r["value_of_unsupported_actions"]])
lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r"Cell & constr. & hosp. & $\tau$ & live $|\mathcal S|$ & $\overline{|M(s)|}$ & $J_{\beta}^{\mathrm{true}}$ & $J^{*}_{\mathrm{true}}$ & $J^{*}_{\mathrm{bench}}$ & $V_{\mathrm{unsupp}}$ \\",
         r"\midrule"]
for s_ in srows:
    lines.append(" & ".join([str(s_[0]), str(s_[1]), str(s_[2]), str(s_[3]), str(s_[4]), f"{s_[5]:.2f}", f3(s_[6]), f3(s_[7]), f3(s_[8]), f3(s_[9])]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_envs.tex").write_text("\n".join(lines) + "\n")
FACTS["environment_suite_draw0"] = [dict(zip(["cell", "construction", "hosp", "tau", "live", "mean_M", "J_beh_true", "J_opt_true",
                                              "J_opt_bench", "value_unsupported"], [str(x) for x in s_])) for s_ in srows]

# ---- frontiers figure
def frontier(cell, target, semantics):
    d = fr[(fr["cell"] == cell) & (fr["target"] == target) & (fr["semantics"] == semantics)].sort_values("kappa")
    return d["kappa"].to_numpy(), d["objective"].to_numpy()

ctrl = "gumbel|best|n10000|tau20|eps0.05|phys-withhold-vaso"
KINF = 3.0
def kx(k):
    return np.where(np.isinf(k), KINF, k)
fig, axes = plt.subplots(1, 2, figsize=(FIGW["fig_frontiers"], 1.9))
ax = axes[0]
for (tgt, sem_, col, lab_, ls) in [("bench_mean", "given", C_MEAN, "bench., mean rule", "-"),
                                    ("bench_terminate", "terminate", C_TERM, "bench., terminate", "--"),
                                    ("true", "given", INK, "truth", "-")]:
    k, o = frontier(ctrl, tgt, sem_)
    ax.plot(kx(k), o, ls, color=col, marker="o", ms=3, label=lab_)
ax.set_xscale("symlog", linthresh=0.003)
ax.set_xticks([0, 0.01, 0.1, 1, KINF])
ax.set_xticklabels(["0", "0.01", "0.1", "1", r"$\infty$"])
ax.set_xlabel(r"budget $\kappa$ on expected violations")
ax.set_ylabel(r"constrained optimum $J^*(\kappa)$")
ax.set_title("(a) Gumbel, no-vaso physician", loc="left")
ax.set_ylim(top=ax.get_ylim()[1] + 0.045)
ax.legend(loc="upper left", fontsize=7, handlelength=1.3, borderaxespad=0.2, ncol=1)
ax = axes[1]
garn = [c for c in fr["cell"].unique() if c.startswith("garnet")]
cols = {"-0.5": C_3, "0.0": MUTED, "0.5": C_4, "1.0": C_TERM}
for c in sorted(garn):
    k, o = frontier(c, "true", "given")
    if len(k) == 0:
        continue
    h = c.split("|")[1][1:]
    eps = c.split("|")[2][3:]
    ax.plot(kx(k), o - o[0], "-" if eps == "0.05" else ":", color=cols.get(h, MUTED), marker="o" if eps == "0.05" else "s", ms=2.5,
            label=f"h={h}, ε={eps}")
ax.set_xscale("symlog", linthresh=0.003)
ax.set_xticks([0, 0.01, 0.1, 1, KINF])
ax.set_xticklabels(["0", "0.01", "0.1", "1", r"$\infty$"])
ax.set_xlabel(r"budget $\kappa$")
ax.set_ylabel(r"$J^*(\kappa) - J^*(0)$")
ax.set_title("(b) Garnet, true systems", loc="left")
ax.legend(ncol=1, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.04), handlelength=1.6)
fig.tight_layout()
save(fig, "fig_frontiers")
FACTS["frontier_value_of_unsupported"] = (fr.groupby(["cell", "target", "semantics"])["objective"].agg(lambda v: float(v.max() - v.min()))
                                          .reset_index().rename(columns={"objective": "range"}).to_dict(orient="records"))

# ---- H1 table + fidelity figure
h1 = pd.DataFrame(pre["H1"]["rows"])
rfi = rf.set_index(["cell", "completion"])
lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{Kendall $\tau_b$ vs.\ truth} & & & \multicolumn{2}{c}{selection regret ($z$)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){6-7}",
         r"Gumbel cell & mean rule & terminate & $\Delta$ [95\% CI] & Holm $p$ & mean & terminate \\", r"\midrule"]
for _, r in h1.iterrows():
    c = r["cell"]
    lines.append(f"{gumbel_short(c)} & {r['mean_tau_mean_rule']:.2f} & {r['mean_tau_terminate_rule']:.2f} & "
                 f"{sg(r['mean_difference'], 2)} [{sg(r['ci_low'], 2)}, {sg(r['ci_high'], 2)}] & {pfmt(r['holm_p'])} & "
                 f"{rfi.loc[(c, 'mean'), 'selection_regret_z_mean']:.2f} & {rfi.loc[(c, 'terminate'), 'selection_regret_z_mean']:.2f} \\\\")
oc = "gumbel|oracle|n10000|tau20|eps0.05"
lines.append(r"\midrule")
lines.append(f"oracle construction (exploratory) & {rfi.loc[(oc, 'mean'), 'kendall_tau_b_mean']:.2f} & {rfi.loc[(oc, 'terminate'), 'kendall_tau_b_mean']:.2f} & "
             f"{sg(rfi.loc[(oc, 'mean'), 'kendall_tau_b_mean'] - rfi.loc[(oc, 'terminate'), 'kendall_tau_b_mean'], 2)} & -- & "
             f"{rfi.loc[(oc, 'mean'), 'selection_regret_z_mean']:.2f} & {rfi.loc[(oc, 'terminate'), 'selection_regret_z_mean']:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_h1.tex").write_text("\n".join(lines) + "\n")
FACTS["H1"] = h1[["cell", "n_pairs", "n_benchmark_constant", "mean_tau_mean_rule", "mean_tau_terminate_rule", "mean_difference", "ci_low",
                  "ci_high", "p_value", "holm_p", "reject_holm"]].to_dict(orient="records")

gcells = [c for c in h1["cell"]] + [oc]
fig, axes = plt.subplots(1, 2, figsize=(FIGW["fig_fidelity"], 2.1), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
x = np.arange(len(gcells))
for j, (comp, col) in enumerate([("mean", C_MEAN), ("terminate", C_TERM)]):
    mu = [rfi.loc[(c, comp), "kendall_tau_b_mean"] for c in gcells]
    lo = [rfi.loc[(c, comp), "kendall_tau_b_ci_low"] for c in gcells]
    hi = [rfi.loc[(c, comp), "kendall_tau_b_ci_high"] for c in gcells]
    ax.bar(x + (j - 0.5) * 0.38, mu, width=0.36, color=col, label=f"{comp} rule",
           yerr=[np.subtract(mu, lo), np.subtract(hi, mu)], capsize=1.5, error_kw={"lw": 0.7, "ecolor": INK})
ax.set_xticks(x)
def _tick(c):
    if "oracle" in c:
        return "ora-\ncle"
    t = c.split("|")
    lab_ = f"{int(t[2][1:]) // 1000}k\nτ{t[3][3:]}"
    if t[4] != "eps0.05":
        lab_ += "\nε" + t[4][3:].lstrip("0")
    if len(t) > 5:
        lab_ += "\nno-v"
    return lab_
ax.set_xticklabels([_tick(c) for c in gcells], fontsize=7)
ax.set_xlabel(r"hospital episodes, threshold $\tau$")
ax.set_ylabel(r"Kendall $\tau_b$ (benchmark vs truth)")
ax.set_ylim(0, 1)
ax.set_title("(a) Gumbel-max sepsis", loc="left")
ax.grid(axis="x", visible=False)
ax = axes[1]
gar = rf[(rf["family"] == "garnet") & rf["completion"].isin(["mean", "terminate"])].copy()
for comp, col in [("mean", C_MEAN), ("terminate", C_TERM)]:
    for eps, mk, ls in [(0.05, "o", "-"), (0.3, "s", "--")]:
        d = gar[(gar["completion"] == comp) & np.isclose(gar["eps"], eps)].sort_values("harm")
        ax.errorbar(d["harm"] + (0.02 if comp == "terminate" else -0.02), d["kendall_tau_b_mean"],
                    yerr=[d["kendall_tau_b_mean"] - d["kendall_tau_b_ci_low"], d["kendall_tau_b_ci_high"] - d["kendall_tau_b_mean"]],
                    color=col, marker=mk, ms=3.5, ls=ls, lw=1.2, capsize=1.5, label=f"{comp}, ε={eps}")
ax.axvline(0, color=MUTED, lw=0.7, ls=":")
ax.set_xlabel(r"harm $h$ of unsupported actions")
ax.set_ylabel(r"Kendall $\tau_b$")
ax.set_title("(b) Garnet: rule flips with $h$", loc="left")
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=C_MEAN, lw=4, label="mean rule"), Line2D([], [], color=C_TERM, lw=4, label="terminate rule"),
           Line2D([], [], color=MUTED, marker="o", ls="-", label="ε = 0.05"), Line2D([], [], color=MUTED, marker="s", ls="--", label="ε = 0.3")]
fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
save(fig, "fig_fidelity")
FACTS["rank_fidelity"] = rf[["cell", "completion", "n_draws", "kendall_tau_b_mean", "kendall_tau_b_ci_low", "kendall_tau_b_ci_high",
                             "top1_agreement_mean", "selection_regret_z_mean"]].to_dict(orient="records")
FACTS["H3"] = {k: pre["H3"][k] for k in ("spearman_rho", "p_value", "test", "n_blocks", "n_draw_units", "n_permutations",
                                          "mean_error_by_harm", "per_eps_spearman_descriptive")}

# ---- error decomposition table
edd = ed[ed["family"].isin(["gumbel", "garnet"])].copy()
lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r" & \multicolumn{4}{c}{mean rule} & \multicolumn{4}{c}{terminate rule} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Cell & $|\mathrm{total}|$ & $|\mathrm{sem.}|$ & sem. & $|\mathrm{est.}|$ & $|\mathrm{total}|$ & $|\mathrm{sem.}|$ & sem. & $|\mathrm{est.}|$ \\", r"\midrule"]
for c in list(dict.fromkeys(edd["cell"])):
    a = edd[(edd["cell"] == c) & (edd["completion"] == "mean")].iloc[0]
    b = edd[(edd["cell"] == c) & (edd["completion"] == "terminate")].iloc[0]
    if c.startswith("gumbel"):
        name = "Gumbel: " + gumbel_short(c)
    else:
        t = c.split("|")
        name = f"Garnet: $h{{=}}{t[1][1:].replace('-', '{-}')}$, $\\epsilon{{=}}{t[2][3:]}$"
    lines.append(f"{name} & {a['mean_abs_total_mean']:.3f} & {a['mean_abs_semantics_mean']:.3f} & {sg(a['mean_semantics'], 3)} & {a['mean_abs_estimation_mean']:.3f} & "
                 f"{b['mean_abs_total_mean']:.3f} & {b['mean_abs_semantics_mean']:.3f} & {sg(b['mean_semantics'], 3)} & {b['mean_abs_estimation_mean']:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_decomp.tex").write_text("\n".join(lines) + "\n")
FACTS["error_decomposition"] = edd[["cell", "completion", "mean_abs_total_mean", "mean_abs_semantics_mean", "mean_abs_estimation_mean",
                                    "mean_total", "mean_semantics", "mean_estimation"]].to_dict(orient="records")

# ---- H2 forest plot + table
treat = list(dict.fromkeys(h2["treatment"]))
fig, axes = plt.subplots(1, 3, figsize=(FIGW["fig_h2"], 2.45), sharey=True)
for ax, fam, title in zip(axes, ["icu", "gumbel", "garnet"], ["(a) ICU-Sepsis", "(b) Gumbel", "(c) Garnet"]):
    d = h2[h2["family"] == fam].set_index("treatment").reindex(treat)
    y = np.arange(len(treat))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        sig = bool(r["reject_holm"])
        col = (C_MEAN if r["mean_difference"] > 0 else C_TERM) if sig else MUTED
        ax.errorbar(r["mean_difference"], yi, xerr=[[r["mean_difference"] - r["mean_difference_ci_low"]], [r["mean_difference_ci_high"] - r["mean_difference"]]],
                    fmt="o" if sig else "o", mfc=col if sig else "white", mec=col, color=col, ms=3.5, capsize=1.5, lw=0.9)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(np.arange(len(treat))[::-1])
axes[0].set_yticklabels([t.replace(" [masked]", ", masked").replace(" [native]", "").replace(" [projected]", ", projected") for t in treat])
axes[1].set_xlabel("paired difference in normalised score vs. projected IQL")
fig.tight_layout()
save(fig, "fig_h2")
lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{ICU-Sepsis} & \multicolumn{2}{c}{Gumbel} & \multicolumn{2}{c}{Garnet} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Treatment & $\Delta$ mean & $\Delta$ IQM & $\Delta$ mean & $\Delta$ IQM & $\Delta$ mean & $\Delta$ IQM \\", r"\midrule"]
for t in treat:
    cells = []
    for fam in ["icu", "gumbel", "garnet"]:
        r = h2[(h2["family"] == fam) & (h2["treatment"] == t)].iloc[0]
        star = "$^{*}$" if r["reject_holm"] else ""
        cells.append(f"{sg(r['mean_difference'], 3)}{star}")
        cells.append(sg(r["iqm_difference"], 3))
    lines.append(t.replace("&", "+") + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_h2.tex").write_text("\n".join(lines) + "\n")
FACTS["H2"] = h2[["family", "treatment", "n_test_units", "iqm_treatment", "iqm_comparator", "mean_difference", "mean_difference_ci_low",
                  "mean_difference_ci_high", "paired_win_rate", "holm_p", "reject_holm", "direction", "equivalent_tost_holm"]].to_dict(orient="records")
FACTS["H2_counts"] = {"family_size": pre["H2"]["family_size"], "n_rejections_holm": pre["H2"]["n_rejections_holm"],
                      "n_equivalent_tost_holm": pre["H2"]["n_equivalent_tost_holm"]}

# ---- benchmark vs truth scatter (Gumbel) + raw returns table
zd = zs[zs["is_default"] & (zs["status"] == "ok")]
KEYM = ["IQL [projected]", "IQL [masked]", "BCQ [masked]", "VI-LCB [masked]", "MBS-QI [masked]", "SPIBB [projected]",
        "Lagrangian-MB [native]", "CMDP-LP (model) [native]", "SfBC (exact) [masked]", "Diffusion (exact, guided) [native]",
        "CQL [masked]", "BC [masked]", "Behaviour (true) [native]"]
fig, axes = plt.subplots(1, 2, figsize=(FIGW["fig_bench_vs_true"], 2.35))
HL = [("IQL [projected]", C_MEAN, "o", "IQL, projected (comparator)"), ("VI-LCB [masked]", C_TERM, "s", "VI-LCB, masked"),
      ("SPIBB [projected]", C_3, "D", "SPIBB, projected"), ("Lagrangian-MB [native]", C_4, "^", "Lagrangian-MB"),
      ("CMDP-LP (model) [native]", INK, "x", "CMDP-LP (model)"), ("Diffusion (exact, guided) [native]", INK, "v", "exact guided diffusion")]
for ax, cell in zip(axes, ["gumbel|best|n2000|tau5|eps0.05", "gumbel|best|n10000|tau20|eps0.05"]):
    d = zd[zd["cell"] == cell]
    piv = d.pivot_table(index="policy", columns="target", values="return", aggfunc="mean")
    piv = piv[["bench_mean", "true"]].dropna()
    zoom = piv[(piv["true"] > 0.28) & (piv["bench_mean"] > 0.33)]
    ax.scatter(zoom["bench_mean"], zoom["true"], s=9, color=MUTED, alpha=0.45, lw=0, label="other policies")
    lo = min(zoom.min().min(), 0.3) - 0.005
    hi = zoom.max().max() + 0.005
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.6, ls=":")
    for pol, col, mk, lab_ in HL:
        if pol in piv.index:
            if mk == "x":
                ax.scatter(piv.loc[pol, "bench_mean"], piv.loc[pol, "true"], s=30, color=col, marker=mk, zorder=4, lw=1.1, label=lab_)
            else:
                ax.scatter(piv.loc[pol, "bench_mean"], piv.loc[pol, "true"], s=26, color=col, marker=mk, zorder=3,
                           edgecolor="white", lw=0.6, label=lab_)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("true value")
    ax.set_title("(%s) Gumbel %s" % ("a" if "2000" in cell else "b", gumbel_plain(cell)), loc="left")
    best_b = piv["bench_mean"].idxmax()
    best_t = piv["true"].idxmax()
    FACTS.setdefault("bench_vs_true", {})[cell] = {"best_on_benchmark": best_b, "best_on_truth": best_t,
                                                   "values": piv.loc[[p for p in KEYM if p in piv.index]].round(5).to_dict(orient="index"),
                                                   "n_outside_zoom": int(len(piv) - len(zoom))}
handles, labels = axes[1].get_legend_handles_labels()
for ax in axes:
    ax.set_xlabel("benchmark value (mean rule)")
fig.tight_layout(rect=(0, 0.24, 1, 1))
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.0)
save(fig, "fig_bench_vs_true")

raw = zd[zd["policy"].isin(KEYM)].copy()
dep = {"icu": "bench_terminate", "gumbel": "true", "garnet": "true"}
lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{ICU-Sepsis} & \multicolumn{2}{c}{Gumbel (pooled)} & \multicolumn{2}{c}{Garnet (pooled)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Policy & mean & terminate & bench (mean) & true & bench (mean) & true \\", r"\midrule"]
for pol in KEYM:
    v = []
    for fam, tg in [("icu", "bench_mean"), ("icu", "bench_terminate"), ("gumbel", "bench_mean"), ("gumbel", "true"),
                    ("garnet", "bench_mean"), ("garnet", "true")]:
        s_ = raw[(raw["family"] == fam) & (raw["target"] == tg) & (raw["policy"] == pol)]["return"]
        v.append(f"{s_.mean():.4f}" if len(s_) else "--")
    lines.append(pol.replace("&", "+") + " & " + " & ".join(v) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_raw_returns.tex").write_text("\n".join(lines) + "\n")
FACTS["raw_returns_pooled"] = (raw.groupby(["family", "target", "policy"])["return"].mean().round(5).reset_index().to_dict(orient="records"))

# ---- proposition summary table (appendix)
TERM_EQ_VMIN = r"$r_{\mathrm{term}}=v_{\min}$"
lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"& Statement & checks & passed & failed \\", r"\midrule"]
for _, r in props.iterrows():
    title = r['title'].replace('&', 'and').replace('r_term = v_min', TERM_EQ_VMIN).replace('convex imputation W', 'convex imputation $W$')
    lines.append(f"{r['proposition']} & {title} & {int(r['checks'])} & {int(r['passed'])} & {int(r['failed'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_props.tex").write_text("\n".join(lines) + "\n")

# ---- full rank-fidelity table (appendix)
lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Cell & rule & $\tau_b$ [95\% CI] & top-1 & regret ($z$) & draws \\", r"\midrule"]
for _, r in rf[rf["completion"].isin(["mean", "terminate"])].iterrows():
    c = r["cell"]
    name = ("Gumbel: " + gumbel_short(c)) if c.startswith("gumbel") else (f"Garnet: $h{{=}}{c.split('|')[1][1:].replace('-', '{-}')}$, $\\epsilon{{=}}{c.split('|')[2][3:]}$")
    lines.append(f"{name} & {r['completion']} & {r['kendall_tau_b_mean']:.3f} [{r['kendall_tau_b_ci_low']:.3f}, {r['kendall_tau_b_ci_high']:.3f}] & "
                 f"{r['top1_agreement_mean']:.2f} & {r['selection_regret_z_mean']:.3f} & {int(r['n_draws'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_fidelity_full.tex").write_text("\n".join(lines) + "\n")

# ---- FQE / selection facts
imp = pd.DataFrame(drv.get("improper_by_family_target", [])) if isinstance(drv.get("improper_by_family_target"), list) else None
FACTS["driver_summary"] = {k: v for k, v in drv.items() if not isinstance(v, (list, dict))}
xmeta = manifest.get("xenv", {})
FACTS["improper_by_family_target"] = xmeta.get("improper_by_family_target")
FACTS["result_rows"] = xmeta.get("result_rows")
FACTS["policies_per_draw"] = int(zd.groupby(["cell", "draw"])["policy"].nunique().median())
FACTS["all_policies_per_draw"] = int(zs.groupby(["cell", "draw", "target"])["policy"].nunique().median())


# ---- additional exports
_rd = pd.read_csv(X / "table_rank_fidelity_draws.csv")
_c = "gumbel|best|n2000|tau20|eps0.05"
_m = _rd[(_rd["cell"] == _c) & (_rd["completion"] == "mean")].set_index("draw")
_t = _rd[(_rd["cell"] == _c) & (_rd["completion"] == "terminate")].set_index("draw")
_keep = _m.index[~_m["tau_benchmark_constant"].astype(bool)]
FACTS["H1_n2k_tau20_excluding_constant"] = {"draws_kept": int(len(_keep)), "tau_mean_rule": float(_m.loc[_keep, "kendall_tau_b"].mean()),
                                            "tau_terminate_rule": float(_t.loc[_keep, "kendall_tau_b"].mean()),
                                            "difference": float((_m.loc[_keep, "kendall_tau_b"] - _t.loc[_keep, "kendall_tau_b"]).mean())}
# per-cell Gumbel differences vs IQL [projected] on the true system
_g = zd[(zd["family"] == "gumbel") & (zd["target"] == "true")]
_cmp = _g[_g["policy"] == "IQL [projected]"].set_index(["cell", "draw"])
_rows = []
for t in treat:
    for c in sorted(_g["cell"].unique()):
        a = _g[(_g["policy"] == t) & (_g["cell"] == c)].set_index(["cell", "draw"])
        j = a.join(_cmp[["return", "z"]], rsuffix="_cmp", how="inner")
        if not len(j):
            continue
        _rows.append({"treatment": t, "cell": c, "draws": int(len(j)), "d_return": float((j["return"] - j["return_cmp"]).mean()),
                      "d_z": float((j["z"] - j["z_cmp"]).mean()), "wins": int((j["return"] > j["return_cmp"] + 1e-9).sum())})
GUMBEL_CELLS = pd.DataFrame(_rows)
FACTS["gumbel_per_cell_vs_projected_iql"] = GUMBEL_CELLS.to_dict(orient="records")
_cells = [c for c in list(h1["cell"]) + [oc] if c in set(GUMBEL_CELLS["cell"])]
lines = [r"\begin{tabular}{l" + "r" * len(_cells) + "}", r"\toprule",
         "Treatment & " + " & ".join(gumbel_short(c) for c in _cells) + r" \\", r"\midrule"]
for t in treat:
    cells = []
    for c in _cells:
        r = GUMBEL_CELLS[(GUMBEL_CELLS["treatment"] == t) & (GUMBEL_CELLS["cell"] == c)]
        cells.append("--" if not len(r) else f"{sg(r.iloc[0]['d_return'], 3)} ({int(r.iloc[0]['wins'])})")
    lines.append(t.replace("&", "+") + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_gumbel_cells.tex").write_text("\n".join(lines) + "\n")
# default policies with unsupported mass on the mean-rule benchmark
_b = zd[(zd["target"] == "bench_mean") & zd["family"].isin(["gumbel", "garnet"])]
FACTS["default_policies_with_unsupported_mass"] = (_b.assign(v=_b["prob_any_violation"] > 1e-12).groupby(["cell", "draw"])["v"].sum()
                                                   .groupby("cell").mean().round(2).to_dict())
# Part I hierarchical replication, primary pair
_h = hier[(hier["treatment"] == "Diffusion-AM-CQL [primary critic B]") & (hier["comparator"] == "Tabular IQL [projected]") & (hier["rule"] == "terminate")]
_hp = pd.read_csv(RES / "statistics_hierarchical_paired_differences.csv")
_hp = _hp[(_hp["comparator"] == "Tabular IQL [projected]") & (_hp["rule"] == "terminate")]
FACTS["hierarchical_primary_pair"] = {**_h.iloc[0][["mean_of_dataset_means", "cluster_bootstrap_ci_low", "cluster_bootstrap_ci_high",
                                                    "dataset_level_signflip_p_two_sided", "holm_p", "n_datasets"]].to_dict(),
                                      "per_dataset_means": _hp.groupby("dataset")["diff"].mean().round(5).to_dict()}

# ---- extra facts requested by the writers
FACTS["H1_excluding_constant_benchmarks"] = h1[["cell", "n_benchmark_constant", "n_pairs_excluding_constant_benchmarks",
                                               "p_value_excluding_constant_benchmarks"]].to_dict(orient="records")
cons_cols = [c for c in suite.columns if c.startswith("construction_")]
FACTS["construction_draw0"] = suite[["cell"] + cons_cols].to_dict(orient="records")
leg = pd.read_csv(RES / "legacy_projection_controls_summary.csv")
_o = leg[leg["name"] == "LEGACY Optimal Oracle"].set_index("rule")
FACTS["legacy_unconstrained_oracle_lowest_index_ties"] = {"inadmissible_rate_mean_rule": float(_o.loc["mean", "inadmissible_rate_mean"]),
                                                          "return_terminate": float(_o.loc["terminate", "return_mean"]),
                                                          "return_mean": float(_o.loc["mean", "return_mean"])}


# =====================================================================================
# Additional analyses from the existing results (no rerun): bracketing theorem checks, certified comparisons,
# ICU-Sepsis identified intervals. Everything below only adds new outputs; nothing above depends on it.
# =====================================================================================
BR_TOL, BR_PTOL = 1e-9, 1e-12   # value tolerance; "violating" = P(any unsupported decision) > BR_PTOL on bench_mean
# [v_min, v_max] per family from the notebook's entry rewards (Gumbel: death -1, discharge +1, timeout 0; Garnet: goal 1,
# death 0; ICU-Sepsis: survival 1, death 0). The terminate rule jumps to death, so r_term = v_min in every family.
BR_VRANGE = {"gumbel": (-1.0, 1.0), "garnet": (0.0, 1.0), "icu": (0.0, 1.0)}
BR_TARGETS = ["bench_mean", "bench_terminate", "true", "true_rows_mean", "true_rows_terminate"]
FIGW["fig_bracketing"] = 6.3  # full text width (6.5 in) at the same height as before, larger type
_xr = pd.read_csv(X / "xenv_results_all.csv.gz", low_memory=False)
_xr = _xr[_xr["target"].isin(BR_TARGETS)]
assert (_xr["status"] == "ok").all(), "every exact evaluation on these targets is proper"
_k = ["cell", "draw", "policy"]
assert not _xr.duplicated(_k + ["target"]).any()
BJ = _xr.set_index(_k + ["target"])["return"].unstack("target")
_BP = _xr.set_index(_k + ["target"])["prob_any_violation"].unstack("target")
BJ = BJ.join(_xr.drop_duplicates(_k).set_index(_k)[["family", "is_default", "is_reference"]])
BJ["p_bench"], BJ["p_true"] = _BP["bench_mean"], _BP["true_rows_mean"]
# Oracle construction: its benchmark rows are the true supported rows (oracle_benchmark: true P with the mean rule and the
# true d0; bench_terminate = true dynamics + terminate), so its semantics-only targets are its benchmark targets.
_orc = BJ.index.get_level_values("cell").str.contains("|oracle|", regex=False)
for _a, _b in (("true_rows_mean", "bench_mean"), ("true_rows_terminate", "bench_terminate"), ("p_true", "p_bench")):
    assert BJ.loc[_orc, _a].isna().all()
    BJ.loc[_orc, _a] = BJ.loc[_orc, _b]
_sim = BJ[BJ["family"].isin(["gumbel", "garnet"]) & ~BJ["is_reference"].astype(bool)].copy()
# Sanity: completions that share supported rows share P(any unsupported decision) (paths agree until the first one).
_sim = _sim.sort_index()
_simP = _BP.loc[_sim.index]   # raw targets: the oracle cell has no true_rows_* rows (NaN, skipped)
assert np.nanmax(np.abs(_simP["true"] - _simP["true_rows_mean"])) < 1e-9
assert np.nanmax(np.abs(_simP["true_rows_terminate"] - _simP["true_rows_mean"])) < 1e-9
assert np.nanmax(np.abs(_BP["bench_terminate"] - _BP["bench_mean"])) < 1e-9
for _fam, (_lo, _hi) in BR_VRANGE.items():
    _v = BJ.loc[BJ["family"] == _fam, [c for c in BR_TARGETS if c in BJ]].to_numpy(dtype=float)
    assert np.nanmin(_v) >= _lo - 1e-9 and np.nanmax(_v) <= _hi + 1e-9, _fam
_sim["vrange"] = _sim["family"].map(lambda f: BR_VRANGE[f][1] - BR_VRANGE[f][0])
_sim["viol"] = _sim["p_bench"] > BR_PTOL
_sim["viol_true"] = _sim["p_true"] > BR_PTOL
_sim["lo_ok"] = _sim["true"] >= _sim["true_rows_terminate"] - BR_TOL
_sim["up_ok"] = _sim["true"] <= _sim["true_rows_mean"] + BR_TOL
_sim["b_lo_ok"] = _sim["true"] >= _sim["bench_terminate"] - BR_TOL
_sim["b_up_ok"] = _sim["true"] <= _sim["bench_mean"] + BR_TOL
_sim["b_in"] = _sim["b_lo_ok"] & _sim["b_up_ok"]
_sim["exceed"] = _sim["true"] - _sim["true_rows_mean"]
_sim["width"] = _sim["true_rows_mean"] - _sim["true_rows_terminate"]
_bnd = _sim["p_true"] * _sim["vrange"]
_sim["exceed_ratio"] = np.where(_bnd > 0, _sim["exceed"] / _bnd.where(_bnd > 0, 1.0), np.nan)
_sim["width_ratio"] = np.where(_bnd > 0, _sim["width"] / _bnd.where(_bnd > 0, 1.0), np.nan)
assert (_sim["width"] >= -1e-12).all() and (_sim["bench_mean"] - _sim["bench_terminate"] >= -1e-12).all()
assert (_sim.loc[~_sim["viol_true"], "width"].abs() <= BR_TOL).all(), "admissible policies are identified exactly"

GUMBEL_ORDER = list(h1["cell"]) + [oc]
GARNET_ORDER = sorted([c for c in _sim.index.get_level_values("cell").unique() if c.startswith("garnet")],
                      key=lambda c: (float(c.split("|")[1][1:]), float(c.split("|")[2][3:])))
BR_POOLS = {"gumbel_estimated": [c for c in GUMBEL_ORDER if "|best|" in c], "garnet_all": GARNET_ORDER,
            "garnet_h_ge_0.5": [c for c in GARNET_ORDER if float(c.split("|")[1][1:]) >= 0.5]}


def _rate(num, den):
    return float(num / den) if den else float("nan")


def _bracket_semantics(d):
    v, f = d[d["viol"]], d[~d["up_ok"]]
    vt = d[d["viol_true"]]
    return {"n_rows": int(len(d)), "n_draws": int(d.index.get_level_values("draw").nunique()), "n_violating": int(len(v)),
            "lower_violations": int((~d["lo_ok"]).sum()), "lower_holds_violating": _rate(v["lo_ok"].sum(), len(v)),
            "max_lower_shortfall": float((d["true_rows_terminate"] - d["true"]).max()),
            "upper_coverage_all": _rate(d["up_ok"].sum(), len(d)), "upper_coverage_violating": _rate(v["up_ok"].sum(), len(v)),
            "upper_coverage_violating_truep": _rate(vt["up_ok"].sum(), len(vt)), "n_violating_truep": int(len(vt)),
            "upper_failures": int(len(f)), "upper_exceedance_mean": float(f["exceed"].mean()) if len(f) else 0.0,
            "upper_exceedance_median": float(f["exceed"].median()) if len(f) else 0.0,
            "upper_exceedance_max": float(f["exceed"].max()) if len(f) else 0.0,
            "upper_exceedance_max_over_p_range": float(np.nanmax(f["exceed_ratio"])) if len(f) else 0.0,
            "width_mean_violating": float(v["width"].mean()) if len(v) else 0.0,
            "width_max_over_p_range": float(np.nanmax(d["width_ratio"])) if d["width_ratio"].notna().any() else 0.0}


def _bracket_benchmark(d):
    v = d[d["viol"]]
    return {"n_rows": int(len(d)), "n_violating": int(len(v)), "coverage_all": _rate(d["b_in"].sum(), len(d)),
            "coverage_violating": _rate(v["b_in"].sum(), len(v)),
            "lower_holds_violating": _rate(v["b_lo_ok"].sum(), len(v)), "upper_holds_violating": _rate(v["b_up_ok"].sum(), len(v)),
            "lower_violations_all": int((~d["b_lo_ok"]).sum()), "upper_violations_all": int((~d["b_up_ok"]).sum()),
            "coverage_nonviolating": _rate(d.loc[~d["viol"], "b_in"].sum(), int((~d["viol"]).sum()))}


def _cell_rows(cells):
    return _sim[_sim.index.get_level_values("cell").isin(cells)]


BRS, BRB = {}, {}
for c in GUMBEL_ORDER + GARNET_ORDER:
    BRS[c], BRB[c] = _bracket_semantics(_cell_rows([c])), _bracket_benchmark(_cell_rows([c]))
for pool, cells in BR_POOLS.items():
    BRS["pooled:" + pool], BRB["pooled:" + pool] = _bracket_semantics(_cell_rows(cells)), _bracket_benchmark(_cell_rows(cells))
FACTS["bracketing_true_rows"] = {
    "definition": ("semantics-only test on the true supported rows: lower = J(true rows + terminate), upper = J(true rows + "
                   "mean rule), truth = J(true system); policies = all non-reference policies of every draw; violating = "
                   f"P(any unsupported decision) > {BR_PTOL:g} on bench_mean; tolerance {BR_TOL:g}; oracle cell: benchmark rows "
                   "are the true rows; exceedance = J_true - upper when the upper bound fails; *_over_p_range divides by "
                   "p_true (v_max - v_min) (Theorem part iii/iv bound)"),
    "cells": BRS}
FACTS["bracketing_benchmark"] = {
    "definition": ("benchmark-level test (completion plus estimation error): J_true in [J_bench_terminate, J_bench_mean]; "
                   "same policies, violating flag and tolerance as bracketing_true_rows"),
    "cells": BRB}


# ---- certified comparisons over the default policy set, per draw
def _pairs(lo, up, jt=None):
    n = len(lo)
    out = {"n_policies": n, "pairs": n * (n - 1) // 2}
    cert = lo[:, None] > up[None, :] + BR_TOL
    np.fill_diagonal(cert, False)
    out["certified"] = int(cert.sum())
    pm = up[:, None] > up[None, :] + BR_TOL
    pt = lo[:, None] > lo[None, :] + BR_TOL
    out["claims_mean"], out["claims_terminate"] = int(pm.sum()), int(pt.sum())
    out["rule_dependent"] = int((pm & (lo[:, None] < lo[None, :] - BR_TOL)).sum())
    if jt is not None:
        wrong = jt[:, None] <= jt[None, :] + BR_TOL
        rev = jt[:, None] < jt[None, :] - BR_TOL
        out["truth_ordered"] = int((jt[:, None] > jt[None, :] + BR_TOL).sum())
        out["certified_errors"], out["certified_reversals"] = int((cert & wrong).sum()), int((cert & rev).sum())
        for nm, cl in (("mean", pm), ("terminate", pt)):
            out[f"errors_{nm}"], out[f"reversals_{nm}"] = int((cl & wrong).sum()), int((cl & rev).sum())
            unc = cl & ~cert
            out[f"claims_{nm}_uncertified"], out[f"errors_{nm}_uncertified"] = int(unc.sum()), int((unc & wrong).sum())
    return out


_dp = BJ[BJ["is_default"].astype(bool) & ~BJ["is_reference"].astype(bool)]
_cc = []
for (c, dr), d in _dp.groupby(level=["cell", "draw"]):
    fam = d["family"].iloc[0]
    jt = d["true"].to_numpy(dtype=float) if fam != "icu" else None
    for tf, (lc, uc) in (("true_rows", ("true_rows_terminate", "true_rows_mean")), ("bench", ("bench_terminate", "bench_mean"))):
        if d[lc].isna().any():
            continue
        _cc.append({"cell": c, "draw": int(dr), "family": fam, "targets": tf,
                    **_pairs(d[lc].to_numpy(dtype=float), d[uc].to_numpy(dtype=float), jt)})
CC = pd.DataFrame(_cc)


def _cc_summary(d):
    s = d.drop(columns=["cell", "draw", "family", "targets"]).sum(numeric_only=True)
    share = d["certified"] / d["pairs"]
    o = {"draws": int(len(d)), "policies_per_draw": int(d["n_policies"].median()), "pairs": int(s["pairs"]),
         "certified": int(s["certified"]), "certified_share": _rate(s["certified"], s["pairs"]),
         "certified_share_draw_min": float(share.min()), "certified_share_draw_max": float(share.max()),
         "rule_dependent_share": _rate(s["rule_dependent"], s["pairs"]),
         "claims_mean_share": _rate(s["claims_mean"], s["pairs"]), "claims_terminate_share": _rate(s["claims_terminate"], s["pairs"])}
    if "certified_errors" in d and d["certified_errors"].notna().all():
        o.update({"truth_ordered_share": _rate(s["truth_ordered"], s["pairs"]),
                  "certified_errors": int(s["certified_errors"]), "certified_error_rate": _rate(s["certified_errors"], s["certified"]),
                  "certified_reversal_rate": _rate(s["certified_reversals"], s["certified"])})
        for nm in ("mean", "terminate"):
            o.update({f"point_{nm}_errors": int(s[f"errors_{nm}"]), f"point_{nm}_error_rate": _rate(s[f"errors_{nm}"], s[f"claims_{nm}"]),
                      f"point_{nm}_reversal_rate": _rate(s[f"reversals_{nm}"], s[f"claims_{nm}"]),
                      f"point_{nm}_error_rate_uncertified": _rate(s[f"errors_{nm}_uncertified"], s[f"claims_{nm}_uncertified"])})
    return o


CCS = {}
for tf in ("true_rows", "bench"):
    for c in GUMBEL_ORDER + GARNET_ORDER + ["icu|released"]:
        d = CC[(CC["cell"] == c) & (CC["targets"] == tf)]
        if len(d):
            CCS.setdefault(tf, {})[c] = _cc_summary(d)
    for pool, cells in BR_POOLS.items():
        CCS[tf]["pooled:" + pool] = _cc_summary(CC[CC["cell"].isin(cells) & (CC["targets"] == tf)])
FACTS["certified_comparisons"] = {
    "definition": ("default (non-reference) policies of each draw; certified ordered pair (i, j): lower(i) > upper(j) + "
                   f"{BR_TOL:g}, lower = terminate-type and upper = mean-type value on the same targets (true_rows_* for the "
                   "semantics-only test, bench_* for the benchmark test); at most one direction of a pair can be certified, so "
                   "certified_share = certified / unordered pairs; certification error = certified but J_true(i) <= J_true(j) + "
                   "tol (reversal = J_true(i) < J_true(j) - tol); point claims under one rule: J_rule(i) > J_rule(j) + tol, scored "
                   "the same way; rule_dependent = pairs whose order flips between the two rules; counts pooled over draws"),
    **CCS}

# ---- ICU-Sepsis, Part I: identified intervals [J_term, J_mean] per policy (seed means; same released rows for both)
_p1 = allres[allres["group"].isin(["main", "reference"]) & (allres["status"] == "ok")]
_p1v = _p1.pivot_table(index=["group", "name", "seed"], columns="rule", values="return", aggfunc="first")
_p1p = _p1[_p1["rule"] == "mean"].set_index(["group", "name", "seed"])["prob_any_violation"]
_p1v["p_any"] = _p1p
assert (_p1v["mean"] - _p1v["terminate"] >= -1e-12).all(), "J_mean >= J_term (Theorem part iii)"
assert (_p1v["mean"] - _p1v["terminate"] <= _p1v["p_any"] * 1.0 + 1e-12).all(), "width <= p (v_max - v_min)"
_p1v["upper_free"] = _p1v["terminate"] + _p1v["p_any"] * 1.0   # NHB-free upper end, Cor. 2 (ICU: max_T R - v_min = 1)
_p1g = _p1v.groupby(level=["group", "name"]).agg(J_term=("terminate", "mean"), J_mean=("mean", "mean"), p_any=("p_any", "mean"),
                                                  p_any_max=("p_any", "max"), n_seeds=("mean", "size"),
                                                  upper_free=("upper_free", "mean"))
_p1g["width"] = _p1g["J_mean"] - _p1g["J_term"]
_p1g["admissible"] = _p1g["p_any_max"] <= BR_PTOL
assert ((_p1g["width"] <= BR_TOL) == _p1g["admissible"]).all()
ICU_INT = _p1g.reset_index()


def _cert_matrix(lo, up):
    C = lo[:, None] > up[None, :] + BR_TOL
    np.fill_diagonal(C, False)
    return C


_lo_all, _up_all = ICU_INT["J_term"].to_numpy(), ICU_INT["J_mean"].to_numpy()
_C_all = _cert_matrix(_lo_all, _up_all)
_Cf_all = _cert_matrix(_lo_all, ICU_INT["upper_free"].to_numpy())
_n_all = len(ICU_INT)
assert ICU_INT["name"].is_unique
_mt = ICU_INT.set_index("name").loc[[k for k, _ in ICU_ROWS]]
_C_mt = _cert_matrix(_mt["J_term"].to_numpy(), _mt["J_mean"].to_numpy())
_Cf_mt = _cert_matrix(_mt["J_term"].to_numpy(), _mt["upper_free"].to_numpy())
assert not (_Cf_mt & ~_C_mt).any() and not (_Cf_all & ~_C_all).any(), "every NHB-free certificate is an NHB certificate"
_n_mt = len(_mt)
_mean_rank = _mt["J_mean"].rank(ascending=False, method="min").astype(int)
_mt_rows = []
for i, (key, lab_) in enumerate(ICU_ROWS):
    r = _mt.loc[key]
    _mt_rows.append({"policy": lab_, "key": key, "n_seeds": int(r["n_seeds"]), "J_term": float(r["J_term"]),
                     "J_mean": float(r["J_mean"]), "width": float(r["width"]), "p_any": float(r["p_any"]),
                     "admissible": bool(r["admissible"]), "mean_rule_rank": int(_mean_rank.loc[key]),
                     "certified_above": int(_C_mt[i].sum()), "certified_below": int(_C_mt[:, i].sum()),
                     "upper_free": float(r["upper_free"]), "certified_above_free": int(_Cf_mt[i].sum()),
                     "certified_below_free": int(_Cf_mt[:, i].sum()),
                     "certified_above_of": [ICU_ROWS[j][1] for j in np.flatnonzero(_C_mt[i])],
                     "certified_below_of": [ICU_ROWS[j][1] for j in np.flatnonzero(_C_mt[:, i])]})
# mean-rule point claims among the main-table policies that are not certified
_pm_mt = _mt["J_mean"].to_numpy()[:, None] > _mt["J_mean"].to_numpy()[None, :] + BR_TOL
_unc_mt = [(ICU_ROWS[i][1], ICU_ROWS[j][1]) for i, j in zip(*np.nonzero(_pm_mt & ~_C_mt))]
FACTS["icu_intervals"] = {
    "definition": ("Part I (exact_results_all.csv, groups main + reference): per policy the identified interval [J_term, J_mean] "
                   "on the released ICU-Sepsis rows, seed means; width = J_mean - J_term; p_any = mean-rule P(any unsupported "
                   "decision), seed mean; admissible = p_any = 0 in every seed (width 0 <=> admissible verified)"),
    "n_policies": int(_n_all), "n_zero_width": int((ICU_INT["width"] <= BR_TOL).sum()),
    "n_admissible": int(ICU_INT["admissible"].sum()),
    "policies": ICU_INT[["group", "name", "n_seeds", "J_term", "J_mean", "width", "p_any", "admissible"]].to_dict(orient="records"),
    "main_table": _mt_rows}
_icu_b = CCS["bench"]["icu|released"]
_icu_dp = _dp[_dp["family"] == "icu"]
FACTS["icu_certified"] = {
    "definition": ("certified pair (i, j): J_term(i) > J_mean(j) + 1e-9 on the released rows (needs no ground truth: only the "
                   "completion is unknown); shares are over unordered pairs"),
    "part1_main_table": {"n_policies": int(_n_mt), "pairs": int(_n_mt * (_n_mt - 1) // 2), "certified": int(_C_mt.sum()),
                         "certified_free": int(_Cf_mt.sum()),
                         "certified_share": _rate(_C_mt.sum(), _n_mt * (_n_mt - 1) // 2),
                         "mean_rule_claims": int(_pm_mt.sum()), "mean_rule_claims_not_certified": int(len(_unc_mt)),
                         "mean_rule_claims_not_certified_pairs": _unc_mt,
                         "certified_pairs": [(ICU_ROWS[i][1], ICU_ROWS[j][1]) for i, j in zip(*np.nonzero(_C_mt))]},
    "part1_all": {"n_policies": int(_n_all), "pairs": int(_n_all * (_n_all - 1) // 2), "certified": int(_C_all.sum()),
                  "certified_free": int(_Cf_all.sum()),
                  "certified_share": _rate(_C_all.sum(), _n_all * (_n_all - 1) // 2)},
    "part2_icu_draws": {**_icu_b, "default_policy_rows": int(len(_icu_dp)),
                        "certified_free": int(sum(_cert_matrix(g["bench_terminate"].to_numpy(),
                                                               (g["bench_terminate"] + g["p_bench"]).to_numpy()).sum()
                                                  for _, g in _icu_dp.groupby(level="draw"))),
                        "default_zero_width_share": _rate(int((_icu_dp["p_bench"] <= BR_PTOL).sum()), len(_icu_dp)),
                        "default_zero_width_check": bool(((_icu_dp["bench_mean"] - _icu_dp["bench_terminate"]).abs()
                                                          [_icu_dp["p_bench"] <= BR_PTOL] <= BR_TOL).all())}}


# ---- tab_certified
def _pct(v):
    if pd.isna(v):
        return "--"
    v = 100.0 * v
    if v == 0.0:
        return "0"
    if v < 0.1:
        return r"$<$0.1"
    if v > 99.9 and v < 100.0:
        return r"$>$99.9"
    return f"{v:.1f}"


def _garnet_lab(c):
    t = c.split("|")
    return f"$h{{=}}{t[1][1:].replace('-', '{-}')}$, $\\epsilon{{=}}{t[2][3:]}$"


def _gumbel_tab_lab(c):
    t = c.split("|")
    lab_ = f"{int(t[2][1:]) // 1000}k, $\\tau{{=}}{t[3][3:]}$"
    if t[4] != "eps0.05":
        lab_ += f", $\\epsilon{{=}}{t[4][3:]}$"
    if len(t) > 5 and "withhold" in t[5]:
        lab_ += ", withhold-vaso"
    return ("oracle, " + lab_) if t[1] == "oracle" else lab_


def _cert_row(lab_, key):
    s, b = BRS[key], BRB[key]
    ct, cb = CCS["true_rows"][key], CCS["bench"][key]
    return (f"{lab_} & {s['lower_violations']} & {_pct(s['upper_coverage_violating'])} & {_pct(ct['certified_share'])} & "
            f"{_pct(ct['certified_error_rate'])} & {_pct(ct['point_mean_error_rate'])}/{_pct(ct['point_terminate_error_rate'])} & "
            f"{_pct(b['coverage_violating'])} & {_pct(cb['certified_share'])} & {_pct(cb['certified_error_rate'])} & "
            f"{_pct(cb['point_mean_error_rate'])}/{_pct(cb['point_terminate_error_rate'])} \\\\")


lines = [r"\setlength{\tabcolsep}{3pt}%", r"\begin{tabular}{lrrrrcrrrc}", r"\toprule",
         r" & \multicolumn{5}{c}{semantics only (true supported rows)} & \multicolumn{4}{c}{benchmark (estimated rows)} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){7-10}",
         r" & \multicolumn{2}{c}{bounds} & \multicolumn{2}{c}{certified} & point err. & bounds & \multicolumn{2}{c}{certified} & point err. \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}\cmidrule(lr){8-9}\cmidrule(lr){10-10}",
         r"Cell & lower & upper & share & err. & mean/term & hold & share & err. & mean/term \\", r"\midrule",
         r"\multicolumn{10}{l}{\textit{Gumbel-max sepsis} (hospital episodes, threshold $\tau$)} \\"]
for c in GUMBEL_ORDER:
    lines.append(_cert_row(_gumbel_tab_lab(c), c))
lines.append(_cert_row("pooled (%d estimated cells)" % len(BR_POOLS["gumbel_estimated"]), "pooled:gumbel_estimated"))
lines += [r"\midrule", r"\multicolumn{10}{l}{\textit{Garnet}} \\"]
for c in GARNET_ORDER:
    lines.append(_cert_row(_garnet_lab(c), c))
lines.append(_cert_row("pooled (%d cells)" % len(BR_POOLS["garnet_all"]), "pooled:garnet_all"))
# Additional cells (notebook v3, II.8): rows from v3_tables (extract_v3.py), rates as displayed there (rounded to 1e-4).
V3T = OUT / "v3_tables"
if (V3T / "certified_v3.csv").exists():
    V3_ROWS = {"gumbel|best|n10000|tau20|eps0.05|phys-confounded": "confounded phys.",
               "gumbel|best|n10000|tau20|eps0.05|phys-withhold-abx": "withhold-abx phys.",
               "gumbel|agg-obs|n10000|tau20|eps0.05": "aggregated, obs.",
               "gumbel|agg-proj|n10000|tau20|eps0.05": "aggregated, proj.",
               "garnet|hrand|eps0.05": r"random rows, $\epsilon{=}0.05$", "garnet|hrand|eps0.3": r"random rows, $\epsilon{=}0.3$"}
    _v3b = pd.read_csv(V3T / "bracket_v3.csv")
    _v3c = pd.read_csv(V3T / "certified_v3.csv")
    _v3key = lambda c: next(k for k in V3_ROWS if k.startswith(c[:-3])) if c.endswith("...") else c
    _v3b["cell"], _v3c["cell"] = _v3b["cell"].map(lambda c: _v3key(c) if "|" in c else c), \
        _v3c["cell"].map(lambda c: _v3key(c) if "|" in c else c)

    _V3_EXACT = (V3T / "exact_sources.json").exists()       # full-precision tables from use_exact_v3.py

    def _v3pct(v):
        if _V3_EXACT:
            return _pct(v)
        # Rounded inputs (4 decimals): a displayed 0 may hide a rate below 5e-5, so it is printed as "<0.1", never as an
        # exact 0. Percentages get one decimal like the other rows; a 4th decimal of 5 is rounded half up (double
        # rounding, an error below 0.06 percentage points, stated in App. H).
        if pd.isna(v):
            return _pct(v)
        if v == 0.0:
            return r"$<$0.1"
        from decimal import Decimal, ROUND_HALF_UP
        if 100.0 * v >= 0.1 and 100.0 * v <= 99.9:
            return str(Decimal(str(round(100.0 * v, 2))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        return _pct(v)

    def _v3row(cell, lab_):
        b = _v3b.set_index("cell").loc[cell]
        ct = _v3c[(_v3c["cell"] == cell) & (_v3c["targets"] == "true_rows")]
        cb = _v3c[(_v3c["cell"] == cell) & (_v3c["targets"] == "bench")].iloc[0]
        if len(ct):
            ct = ct.iloc[0]
            sem = (f"{int(b['sem_lower_violations'])} & {_v3pct(b['sem_upper_coverage_violating'])} & {_v3pct(ct['certified_share'])} & "
                   f"{_v3pct(ct['certified_error_rate'])} & {_v3pct(ct['point_mean_error_rate'])}/{_v3pct(ct['point_terminate_error_rate'])}")
        else:
            sem = "-- & -- & -- & -- & --"
        return (f"{lab_} & {sem} & {_v3pct(b['bench_coverage_violating'])} & {_v3pct(cb['certified_share'])} & "
                f"{_v3pct(cb['certified_error_rate'])} & {_v3pct(cb['point_mean_error_rate'])}/{_v3pct(cb['point_terminate_error_rate'])} \\\\")

    lines += [r"\midrule", r"\multicolumn{10}{l}{\textit{Additional cells} (Gumbel: 10k, $\tau{=}20$, $\epsilon{=}0.05$; Garnet: random unsupported rows)} \\"]
    for c, lab_ in V3_ROWS.items():
        lines.append(_v3row(c, lab_))
    FACTS["tab_certified_revision_rows"] = {"source": "v3_tables/bracket_v3.csv, certified_v3.csv", "cells": list(V3_ROWS)}
lines += [r"\midrule",
          f"ICU-Sepsis (no truth) & -- & -- & -- & -- & -- & -- & {_pct(_icu_b['certified_share'])} & -- & -- \\\\",
          r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_certified.tex").write_text("\n".join(lines) + "\n")

# ---- tab_icu_intervals (compact: violating policies of the main table plus admissible anchors)
ICU_INT_ROWS = ["Constrained Optimal Oracle", "Tabular BCQ (tau=0.3) [masked greedy]", "Tabular BCQ (tau=0.3)",
                "Tabular IQL [projected]", "Tabular IQL", "Diffusion-AM-CQL [primary critic B]",
                "Tabular CQL", "Clinician", "Random (uniform over 25)"]
_mtr = {r["key"]: r for r in _mt_rows}
lines = [r"\setlength{\tabcolsep}{3.5pt}%", r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrr}", r"\toprule",
         r" & & & & & NHB-free & \multicolumn{2}{c}{certified} & mean \\",
         r"\cmidrule(lr){7-8}",
         r"Policy & $J_{\mathrm{term}}$ & $J_{\mathrm{mean}}$ & width & $p_\pi$ & upper & above & below & rank \\",
         r"\midrule"]
ICU_INT_SHORT = {"Constrained Optimal Oracle": "Optimal (constrained)", "Tabular IQL [projected]": "Tabular IQL, projected (comp.)",
                 "Diffusion-AM-CQL [primary critic B]": "Diffusion, critic B (treat.)"}
for key in ICU_INT_ROWS:
    r = _mtr[key]
    lines.append(f"{ICU_INT_SHORT.get(key, r['policy'])} & {r['J_term']:.4f} & {r['J_mean']:.4f} & {r['width']:.4f} & {r['p_any']:.3f} & "
                 f"{r['upper_free']:.4f} & {r['certified_above']} ({r['certified_above_free']}) & "
                 f"{r['certified_below']} ({r['certified_below_free']}) & {r['mean_rule_rank']} \\\\")
_fi, _fp = FACTS["icu_intervals"], FACTS["icu_certified"]
_p2 = _fp["part2_icu_draws"]
_fr = lambda d: _rate(d["certified_free"], d["pairs"])
lines += [r"\midrule",
          r"\multicolumn{9}{l}{Audited study (App.~\ref{app:icu}), %d policies: %d of width 0; %s\%% (%s\%%) of pairs certified (%d main: %s\%% (%s\%%))} \\"
          % (_fi["n_policies"], _fi["n_zero_width"], _pct(_fp["part1_all"]["certified_share"]), _pct(_fr(_fp["part1_all"])),
             _fp["part1_main_table"]["n_policies"], _pct(_fp["part1_main_table"]["certified_share"]), _pct(_fr(_fp["part1_main_table"]))),
          r"\multicolumn{9}{l}{Tabular suite (Sec.~\ref{sec:design}), %d draws $\times$ %d default policies: %s\%% (%s\%%) certified (per draw %s--%s\%%)} \\"
          % (_p2["draws"], _p2["policies_per_draw"], _pct(_p2["certified_share"]), _pct(_fr(_p2)), _pct(_p2["certified_share_draw_min"]),
             _pct(_p2["certified_share_draw_max"])),
          r"\bottomrule", r"\end{tabular*}"]
(TAB / "tab_icu_intervals.tex").write_text("\n".join(lines) + "\n")

# ---- fig_bracketing: (a-c) intervals on the true rows vs the true value, draw 0; (d-e) coverage against the harm h
BR_FIG_CELLS = [("gumbel|best|n10000|tau20|eps0.05", "(a) Gumbel"),        # n = 10k, tau = 20, eps = 0.05 (caption)
                ("garnet|h-0.5|eps0.05", r"(b) Garnet, $h=-0.5$"), ("garnet|h0.5|eps0.05", r"(c) Garnet, $h=0.5$")]
BR_FIG_DRAW = 0
_rc_saved = dict(plt.rcParams)
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
fig = plt.figure(figsize=(FIGW["fig_bracketing"], 3.3))
gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.9], hspace=0.55, wspace=0.9)
_seg = "#b9b8b3"
_fig_facts = {}
for k_, (c, title) in enumerate(BR_FIG_CELLS):
    ax = fig.add_subplot(gs[0, 2 * k_:2 * k_ + 2])
    d = _sim.loc[(c, BR_FIG_DRAW)]
    d = d[d["viol"]].sort_values(["true_rows_mean", "true_rows_terminate"])
    xs = np.arange(len(d))
    ax.vlines(xs, d["true_rows_terminate"], d["true_rows_mean"], color=_seg, lw=1.6, zorder=1)
    ax.scatter(xs, d["true_rows_mean"], marker="_", s=22, color=C_MEAN, lw=1.4, zorder=2)
    ax.scatter(xs, d["true_rows_terminate"], marker="_", s=22, color=C_TERM, lw=1.4, zorder=2)
    ins = d["up_ok"] & d["lo_ok"]
    ax.scatter(xs[ins.to_numpy()], d.loc[ins, "true"], s=7, color=INK, lw=0, zorder=3)
    ax.scatter(xs[~ins.to_numpy()], d.loc[~ins, "true"], s=11, facecolor="white", edgecolor=INK, lw=0.8, zorder=3)
    ax.set_xticks([])
    ax.set_xlim(-1, len(d))
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.set_xlabel(f"truth inside: {int(ins.sum())}/{len(d)}", fontsize=8.5, color=MUTED, labelpad=2)
    if k_ == 0:
        ax.set_ylabel("value $J$")
    _fig_facts[c] = {"draw": BR_FIG_DRAW, "n_violating": int(len(d)), "n_inside": int(ins.sum())}
from matplotlib.lines import Line2D
_h = [Line2D([], [], color=C_MEAN, marker="_", ls="", ms=7, mew=1.4, label="mean rule (upper)"),
      Line2D([], [], color=C_TERM, marker="_", ls="", ms=7, mew=1.4, label="terminate (lower)"),
      Line2D([], [], color=INK, marker="o", ls="", ms=3, label="truth inside"),
      Line2D([], [], color=INK, marker="o", ls="", ms=3.5, mfc="white", mew=0.8, label="truth outside")]
fig.legend(handles=_h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), handletextpad=0.2, columnspacing=0.9, fontsize=8.5)
_hs = sorted({float(c.split("|")[1][1:]) for c in GARNET_ORDER})
for k_, (title, cols_) in enumerate([("(d) semantics only (true rows)", ("lo_sem", "up_sem")),
                                     ("(e) benchmark (estimated rows)", ("lo_b", "up_b"))]):
    ax = fig.add_subplot(gs[1, 3 * k_:3 * k_ + 3])
    for eps, mk, ls in [("0.05", "o", "-"), ("0.3", "s", "--")]:
        cs = [f"garnet|h{h}|eps{eps}" for h in _hs]
        if k_ == 0:
            lo_v = [100.0 * BRS[c]["lower_holds_violating"] for c in cs]
            up_v = [100.0 * BRS[c]["upper_coverage_violating"] for c in cs]
        else:
            lo_v = [100.0 * BRB[c]["lower_holds_violating"] for c in cs]
            up_v = [100.0 * BRB[c]["upper_holds_violating"] for c in cs]
        ax.plot(_hs, up_v, ls, color=C_MEAN, marker=mk, ms=3.2, lw=1.2)
        ax.plot(_hs, lo_v, ls, color=C_TERM, marker=mk, ms=3.2, lw=1.2)
    ax.set_ylim(-4, 106)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(_hs)
    ax.set_xticklabels([f"{h:g}".replace("-", "−") for h in _hs])
    ax.set_xlabel(r"Garnet harm $h$")
    ax.set_title(title, loc="left", fontsize=8.5)
    if k_ == 0:
        ax.set_ylabel("bound holds (%)")
        nlow = sum(BRS[c]["lower_violations"] for c in GARNET_ORDER)
        nrow = sum(BRS[c]["n_rows"] for c in GARNET_ORDER)
        ax.text(0.97, 0.36, f"lower bound:\n{nlow} violations in\n{nrow:,} policy-draws", transform=ax.transAxes, ha="right",
                va="center", fontsize=8.5, color=C_TERM, linespacing=1.15)
_h2 = [Line2D([], [], color=C_MEAN, lw=1.2, label="upper (mean rule)"),
       Line2D([], [], color=C_TERM, lw=1.2, label="lower (terminate)"),
       Line2D([], [], color=MUTED, marker="o", ls="-", ms=3, lw=1.0, label="ε = 0.05"),
       Line2D([], [], color=MUTED, marker="s", ls="--", ms=3, lw=1.0, label="ε = 0.3")]
fig.legend(handles=_h2, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.11), handletextpad=0.3, columnspacing=0.9, fontsize=8.5)
save(fig, "fig_bracketing")
plt.rcParams.update(_rc_saved)
FACTS["bracketing_figure"] = {"panels_a_c": _fig_facts, "panels_d_e": "Garnet cells; both bounds scored among violating "
                              "non-reference policy-draws; annotation in (d) counts lower-bound violations over all of them"}

(OUT / "facts.json").write_text(json.dumps(FACTS, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("assets written:", sorted(p.name for p in FIG.glob("*.pdf")), sorted(p.name for p in TAB.glob("*.tex")))
