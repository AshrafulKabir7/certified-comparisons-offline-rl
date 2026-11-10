# %% [markdown]
# ## 35. Paper hand-off
#
# This section lists which output files of the notebook's own contribution map exist after the run. The map
# has groups C1–C7 and, where this notebook registers them, the II.8 groups REV-1 to REV-3 and the II.9 groups
# REV-4 to REV-9. **Nothing in this section is a result**, and the list is not the set of inputs of the paper:
# it omits files the paper reads (for example `exact_results_all.csv` and `xenv/xenv_results_all.csv.gz`),
# and its figures are the notebook's own. In the final run II.9d, II.9f and II.9g were switched off, so their
# files are listed as missing, which is expected. The paper's tables, figures and numbers are rebuilt by
# `paper_scripts/` from the final run's exports in `results/` and the printed outputs of its record in
# `run_record/`; the App. B.5 counts come from `paper_scripts/verify_bracket.py`. The supplementary
# `TABLE_MAP.md` maps each table and figure to its script and input files.

# %%
# ============================================================
# 35. Paper hand-off: which evidence files exist
# ============================================================
EXECLOG.mark("35. Paper hand-off")
HANDOFF_EVIDENCE = {
    "C1 propositions": ["xenv/proposition_checks.csv", "xenv/proposition_summary.csv"],
    "C2 frontiers": ["xenv/exact_frontiers.csv", "figures/xenv/fig_xenv_frontiers.pdf"],
    "C3 ground-truth testbed": ["xenv/gumbel_exactness_check.json", "xenv/icu_adapter_check.json",
                                "xenv/xenv_environment_suite.csv"],
    "C4 benchmark validity": ["xenv/table_rank_fidelity.csv", "xenv/table_error_decomposition.csv",
                              "xenv/xenv_prereg_tests.json", "figures/xenv/fig_xenv_rank_fidelity.pdf",
                              "figures/xenv/fig_xenv_harm_phase.pdf", "figures/xenv/fig_xenv_error_decomposition.pdf"],
    "C5 methods study": ["xenv/table_xenv_main.csv", "xenv/xenv_h2_family.csv", "xenv/xenv_normalized_scores.csv",
                         "xenv/table_performance_profiles.csv", "xenv/table_hp_selection.csv", "xenv/xenv_method_catalog.csv",
                         "figures/xenv/fig_xenv_iqm.pdf", "figures/xenv/fig_xenv_profiles.pdf", "figures/xenv/fig_xenv_poi.pdf"],
    "C6 ICU-Sepsis case study": ["table_main_policy_comparison.csv", "statistics_primary.json",
                                 "statistics_secondary_family.csv", "statistics_hierarchical_replication.csv",
                                 "projection_equivalence_checks.csv", "critic_stability_summary.csv",
                                 "regularizer_gradient_tests.json"],
    "C7 reproducibility": ["table_to_command_map.csv", "audit_closure_checklist.csv", "fresh_reproducibility_manifest.json",
                           "input_sha256.json", "compute_summary.csv", "run_profile.json"],
    "REV-1 II.8 cells: bracketing, certified comparisons, rank fidelity": [
        "xenv/v3_environment_suite.csv", "xenv/v3_results_all.csv.gz", "xenv/v3_bracketing.csv", "xenv/v3_certified.csv",
        "xenv/v3_rank_fidelity.csv", "xenv/v3_stats_checks.json", "figures/xenv/fig_v3_bracketing.pdf"],
    "REV-2 bootstrap-ensemble intervals": ["xenv/v3_bootstrap_summary.json", "xenv/v3_bootstrap_intervals.csv",
                                           "xenv/v3_bootstrap_coverage.csv", "figures/xenv/fig_v3_bootstrap.pdf"],
    "REV-3 ICU-Sepsis harm curves": ["xenv/v3_icu_harm_curves.csv", "xenv/v3_icu_harm_summary.csv",
                                     "xenv/v3_icu_harm_checks.json", "figures/xenv/fig_v3_icu_harm.pdf"],
    "REV-4 certificates without NHB, pair types, kappa, II.8g over all policies": [
        "xenv/v4_robust_summary.csv", "xenv/v4_robust_kappa.csv", "xenv/v4_robust_gapbins.csv",
        "xenv/v4_v3boot_allpolicy.csv", "xenv/v4_robust_summary.json"],
    "REV-5 confidence-set certificates": ["xenv/v4_cs_summary.csv", "xenv/v4_cs_draws.csv", "xenv/v4_cs_summary.json",
                                          "figures/xenv/fig_v4_confidence.pdf"],
    "REV-6 bootstrap B = 200, re-estimated construction": ["xenv/v4_boot_coverage.csv", "xenv/v4_boot_draws.csv",
                                                           "xenv/v4_boot_summary.json",
                                                           "figures/xenv/fig_v4_bootstrap.pdf"],
    "REV-7 sensitivity cells": ["xenv/v4_sens_summary.csv", "xenv/v4_sens_certified_draws.csv",
                                "xenv/v4_sens_rank_fidelity.csv", "xenv/v4_sens_summary.json"],
    "REV-8 lift sensitivity": ["xenv/v4_lift_summary.csv", "xenv/v4_lift_results.csv.gz", "xenv/v4_lift_summary.json"],
    "REV-9 clean rerun": ["xenv/v4_clean_check.json", "xenv/v4_clean_diffs.csv"],
    "II.9 overview and checks": ["xenv/v4_overview.json", "xenv/v4_stats_checks.json"],
}


def handoff_path(rel: str) -> Path:
    return FIG_DIR / rel[len("figures/"):] if rel.startswith("figures/") else TABLE_DIR / rel


HANDOFF_STATUS = pd.DataFrame([{"contribution": c, "file": f, "exists": handoff_path(f).exists()}
                               for c, files in HANDOFF_EVIDENCE.items() for f in files])
display(HANDOFF_STATUS.groupby("contribution", sort=False)["exists"].agg(present="sum", expected="size").reset_index())
_handoff_missing = HANDOFF_STATUS.loc[~HANDOFF_STATUS["exists"], "file"].tolist()
print("Missing evidence files:", _handoff_missing or "none")
print(f"Paper scale (closure MT-5): {MT5_OK} | profile={COMPUTE_PROFILE}, mode={RUN_MODE} | "
      f"PREREG status: {PREREG['status'][:40]!r} | PREREG_PART2 status: {PREREG_PART2['status'][:40]!r}")
if not MT5_OK:
    print("NOTE: this run is not paper scale - do not report its numbers as the paper's results (see audit_closure_checklist.csv).")
