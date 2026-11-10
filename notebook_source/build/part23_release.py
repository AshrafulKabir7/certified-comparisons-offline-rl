# %% [markdown]
# ## 33. Release package: code, environment lock, manifests, table-to-command map, compute summary (audit C04)
#
# The inventory with sizes and SHA-256 and the release archive are written **after the Section 34 gate** (the cell
# at the end of Section 34), so that they contain this run's audit closure checklist and reproducibility manifest.
# That cell also re-exports the code snapshot (and its hashes in `input_sha256.json`), so the archived notebook and
# cell sources include the gate and closure output; the export below is kept as a fallback.
#
# The archive contains **the code** (the executed notebook is exported automatically on Colab, plus the
# executed cell sources), per-seed results, policy files, checkpoint metadata (and weights if
# `RELEASE_INCLUDE_WEIGHTS`), dataset manifests always and dataset arrays if `RELEASE_INCLUDE_DATASET` (check the
# benchmark's terms before redistribution; manifests and content hashes allow deterministic regeneration), the
# environment lock, a table-to-command map covering every generated table and figure (Part I and the Part II files
# under `tables/xenv/` and `figures/xenv/`), the Part II per-(cell, draw) result cache (`xenv_cache/`, so Part II
# can be replayed with `ALLOW_TRAINING = False`), and an inventory. `compute_summary.csv` sums the logged wall time
# of every executed cell per section, per session and per part, with each session's GPU (reproducibility-checklist item
# "compute resources"). The execution-log totals count only cells that finished (a cell interrupted by a disconnect,
# e.g. during the long II.6a driver cell, is not logged), so a "part II draws" row adds the per-draw driver compute
# stored with every draw (`XDRAW_META["timing.total_s"]`, computed in this or an earlier session); the additional
# experiments of Section II.8 add "part II v3 draws" (`XDRAW_META_V3`) and "part II v3 bootstrap draws"
# (`XV3_BOOT_META`) rows in the same way. Every II.8 output (`tables/xenv/v3_*`, `figures/xenv/fig_v3_*`) is in the
# table-to-command map, and the II.8 code hashes are in `input_sha256.json`. The further experiments of
# Section II.9 add "part II v4 ..." rows for their cached draws (confidence sets when their draw meta is in memory as
# `XV4_CS_META`, bootstrap `XV4_BOOT_META`, sensitivity cells `XDRAW_META_V4SENS`, lifts `XV4_LIFT["draw_meta"]`) and
# for the clean rerun (`XV4_CLEAN_DRAWS`); every II.9 output (`tables/xenv/v4_*`, `figures/xenv/fig_v4_*`) is in the
# map, and the II.9 code and configuration hashes are in `input_sha256.json`. 

# %%
# ============================================================
# 33. Release package
# ============================================================
EXECLOG.mark("33. Release package")


def compute_summary(log_dir: Path, current_gpu: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Logged cell wall time per (session, section), per section, per session, per part and in total.

    Sources: `execution_log.jsonl` (one record per executed cell, appended after the cell finishes, so the
    release cells themselves are not included) and the per-session `runtime_*.json` (GPU, compute profile).
    Every logged execution counts, including re-runs. Unparseable lines (a write cut by a disconnect) and cells
    without timing are counted and reported, never silently dropped."""
    recs, bad_lines = [], 0
    log_path = Path(log_dir) / "execution_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    log = pd.DataFrame(recs).reindex(columns=["session_start", "section", "success", "elapsed_s"])
    log["elapsed_s"] = pd.to_numeric(log["elapsed_s"], errors="coerce")
    sessions: Dict[str, dict] = {}
    for f in sorted(Path(log_dir).glob("runtime_*.json")):
        info = json.loads(f.read_text(encoding="utf-8"))
        sessions[str(info.get("session_start_utc"))] = info
    log["gpu"] = [sessions.get(str(s), {}).get("gpu") for s in log["session_start"]]
    log["compute_profile"] = [sessions.get(str(s), {}).get("compute_profile") for s in log["session_start"]]
    log["part"] = ["Part II" if str(sec).startswith("II.") else "Part I" for sec in log["section"]]
    rows = []

    def _row(level: str, sub: pd.DataFrame, **keys) -> None:
        e = sub["elapsed_s"]
        rows.append({"level": level, **keys, "cells": int(len(sub)), "failed_cells": int(sub["success"].eq(False).sum()),
                     "cells_without_timing": int(e.isna().sum()), "elapsed_s": float(e.sum()),
                     "elapsed_h": float(e.sum()) / 3600.0})

    for (sess, sec), sub in log.groupby(["session_start", "section"], sort=False):
        _row("session x section", sub, session_start=sess, section=sec, part=sub["part"].iloc[0],
             gpu=sub["gpu"].iloc[0], compute_profile=sub["compute_profile"].iloc[0])
    for sec, sub in log.groupby("section", sort=False):
        _row("section", sub, section=sec, part=sub["part"].iloc[0])
    for sess, sub in log.groupby("session_start", sort=False):
        _row("session", sub, session_start=sess, gpu=sub["gpu"].iloc[0], compute_profile=sub["compute_profile"].iloc[0])
    for part, sub in log.groupby("part", sort=False):
        _row("part", sub, part=part)
    gpus = sorted({str(g) for g in log["gpu"].dropna()})
    _row("total", log, gpu="; ".join(gpus) if gpus else None)
    out = pd.DataFrame(rows)
    summary = {"total_hours": float(log["elapsed_s"].sum()) / 3600.0, "sessions": int(log["session_start"].nunique()),
               "cells_logged": int(len(log)), "cells_without_timing": int(log["elapsed_s"].isna().sum()),
               "unparseable_log_lines": int(bad_lines), "gpus": gpus, "current_gpu": current_gpu,
               "hours_by_part": {p: float(g["elapsed_s"].sum()) / 3600.0 for p, g in log.groupby("part")},
               "scope": "cells logged before Section 33 in this ROOT (all sessions, re-runs included)"}
    return out, summary


COMPUTE_SUMMARY, COMPUTE_TOTALS = compute_summary(LOG_DIR, GPU_NAME)
# Part II driver compute per draw (stored in each draw's cached meta, so it survives interrupted cells and is
# re-emitted for draws loaded from the cache); the execution log counts finished cells only.
_xdm = globals().get("XDRAW_META")
_xdm = pd.DataFrame() if _xdm is None else _xdm
_xdraw_s = (pd.to_numeric(_xdm["timing.total_s"], errors="coerce") if "timing.total_s" in _xdm.columns
            else pd.Series(dtype=np.float64))
COMPUTE_SUMMARY = pd.concat([COMPUTE_SUMMARY, pd.DataFrame([{
    "level": "part II draws", "part": "Part II", "cells": int(len(_xdraw_s)), "failed_cells": 0,
    "cells_without_timing": int(_xdraw_s.isna().sum()), "elapsed_s": float(_xdraw_s.sum()),
    "elapsed_h": float(_xdraw_s.sum()) / 3600.0}])], ignore_index=True)
COMPUTE_TOTALS["part2_draw_compute_h"] = float(_xdraw_s.sum()) / 3600.0
COMPUTE_TOTALS["part2_draws_with_timing"] = int(_xdraw_s.notna().sum())
# Section II.8 (revision experiments): per-draw compute of the new cells and of the bootstrap ensembles.
for _lvl, _key, _name in (("part II v3 draws", "part2_v3", "XDRAW_META_V3"),
                          ("part II v3 bootstrap draws", "part2_v3_bootstrap", "XV3_BOOT_META")):
    _m = globals().get(_name)
    _m = pd.DataFrame() if _m is None else _m
    _ts = (pd.to_numeric(_m["timing.total_s"], errors="coerce") if "timing.total_s" in _m.columns
           else pd.Series(dtype=np.float64))
    COMPUTE_SUMMARY = pd.concat([COMPUTE_SUMMARY, pd.DataFrame([{
        "level": _lvl, "part": "Part II", "cells": int(len(_ts)), "failed_cells": 0,
        "cells_without_timing": int(_ts.isna().sum()), "elapsed_s": float(_ts.sum()),
        "elapsed_h": float(_ts.sum()) / 3600.0}])], ignore_index=True)
    COMPUTE_TOTALS[f"{_key}_draw_compute_h"] = float(_ts.sum()) / 3600.0
    COMPUTE_TOTALS[f"{_key}_draws_with_timing"] = int(_ts.notna().sum())
# Section II.9: per-draw compute of the cached draws of II.9c-II.9f (draw meta 'timing.total_s') and
# of the clean rerun of II.9g (rerun, reference and comparison seconds per draw).
_xv4_lift_meta = (globals().get("XV4_LIFT") or {}).get("draw_meta")
for _lvl, _key, _m, _cols in (
        ("part II v4 confidence-set draws", "part2_v4_confidence", globals().get("XV4_CS_META"), ("timing.total_s",)),
        ("part II v4 bootstrap draws", "part2_v4_bootstrap", globals().get("XV4_BOOT_META"), ("timing.total_s",)),
        ("part II v4 sensitivity draws", "part2_v4_sensitivity", globals().get("XDRAW_META_V4SENS"),
         ("timing.total_s",)),
        ("part II v4 lift draws", "part2_v4_lift", _xv4_lift_meta, ("timing.total_s",)),
        ("part II v4 clean-rerun draws", "part2_v4_clean", globals().get("XV4_CLEAN_DRAWS"),
         ("rerun_s", "reference_s", "compare_s"))):
    _m = pd.DataFrame() if _m is None else _m
    _ts = (sum(pd.to_numeric(_m[c], errors="coerce") for c in _cols) if set(_cols) <= set(_m.columns)
           else pd.Series(dtype=np.float64))
    COMPUTE_SUMMARY = pd.concat([COMPUTE_SUMMARY, pd.DataFrame([{
        "level": _lvl, "part": "Part II", "cells": int(len(_ts)), "failed_cells": 0,
        "cells_without_timing": int(_ts.isna().sum()), "elapsed_s": float(_ts.sum()),
        "elapsed_h": float(_ts.sum()) / 3600.0}])], ignore_index=True)
    COMPUTE_TOTALS[f"{_key}_draw_compute_h"] = float(_ts.sum()) / 3600.0
    COMPUTE_TOTALS[f"{_key}_draws_with_timing"] = int(_ts.notna().sum())
COMPUTE_SUMMARY.to_csv(TABLE_DIR / "compute_summary.csv", index=False)
_xv4_hours = {k[len("part2_v4_"):-len("_draw_compute_h")]: round(v, 2) for k, v in COMPUTE_TOTALS.items()
              if k.startswith("part2_v4_") and k.endswith("_draw_compute_h")}
print(f"Logged compute: {COMPUTE_TOTALS['total_hours']:.2f} h over {COMPUTE_TOTALS['sessions']} session(s); "
      f"by part {({k: round(v, 2) for k, v in COMPUTE_TOTALS['hours_by_part'].items()})}; GPUs {COMPUTE_TOTALS['gpus']}; "
      f"Part II driver compute over {COMPUTE_TOTALS['part2_draws_with_timing']} draws (any session): "
      f"{COMPUTE_TOTALS['part2_draw_compute_h']:.2f} h; II.8 draws {COMPUTE_TOTALS['part2_v3_draw_compute_h']:.2f} h, "
      f"II.8 bootstrap draws {COMPUTE_TOTALS['part2_v3_bootstrap_draw_compute_h']:.2f} h; II.9 draws (h) {_xv4_hours}")
if COMPUTE_TOTALS["unparseable_log_lines"] or COMPUTE_TOTALS["cells_without_timing"]:
    print(f"NOTE: {COMPUTE_TOTALS['unparseable_log_lines']} unparseable execution-log lines and "
          f"{COMPUTE_TOTALS['cells_without_timing']} cells without timing are excluded from the hour totals.")

TABLE_TO_COMMAND = [
    ("Table 1 dataset properties", "Sections 5-6", "validate_dataset, corrected support audit", "eda_dataset_summary.csv; corrected_support_audit.json; dataset_manifest_main.json; table1_reconciliation.csv"),
    ("Table 2 feature representation", "Sections 4, 7", "feature provenance", "feature_provenance_timing.csv; benchmark_construction_provenance.json"),
    ("Table 3 hyperparameters", "Section 1", "Config / apply_profile / PREREG", "fresh_config.json; preregistration.json; experiment_switches.json; run_profile.json"),
    ("Table 4 main comparison", "Sections 19-20", "evaluate_registered / wide_table", "table_main_policy_comparison.csv; exact_results_summary.csv; exact_results_all.csv"),
    ("Table 5 statistics", "Section 30", "compare / holm_adjust", "statistics_primary.json; statistics_secondary_family.csv; statistics_historical_table5_one_sided.csv; paired_differences.csv; replication_planning.csv"),
    ("Table 6 inference ablations", "Sections 18, 22", "run_diffusion_suite / ablation_table", "table6_inference_ablations_primary_critic.csv; table6_inference_ablations_legacy_critic.csv; post_training_controls_all.csv"),
    ("Table 7 guidance sweep", "Sections 18, 22", "run_diffusion_suite", "table7_guidance_sweep.csv"),
    ("Table 8 per-seed returns", "Section 20", "pivot of RESULTS", "table_per_seed_returns.csv"),
    ("C01 projection controls", "Sections 19, 21, 32", "add_projection_controls", "projection_equivalence_checks.csv; projection_return_differences.csv; legacy_projection_controls_summary.csv"),
    ("C02 critic stability / constraint placement", "Sections 16, 23", "run_critics(ABLATION_SPECS) / PLACEMENT_CELLS", "critic_stability_ablation.csv; critic_stability_summary.csv; critic_stability_summary_main_all_seeds.csv; critic_stability_gate.json; critic_diagnostics.csv; legacy_critic_diagnostics.csv; table_constraint_placement.csv"),
    ("C03 reproducibility", "Section 13", "reproducibility + regulariser tests", "reproducibility_tests.json; regularizer_gradient_tests.json"),
    ("M02 support estimation", "Section 25", "estimated_support_mask", "support_estimation_mask_comparison.csv; support_estimation_policy_results.csv"),
    ("M04 selection protocol", "Section 26", "empirical_mdp_evaluator", "selection_candidates_fqe_vs_exact.csv; selection_fqe_selected.csv; selection_locked_fresh_evaluation.csv; selection_protocol.json; tuning_budget.csv"),
    ("M05 soft constraint", "Section 24", "soft analysis", "soft_constraint_analysis.csv; greedy_margin_tie_report.csv"),
    ("M06 diffusion necessity and cost", "Sections 17-18, 22", "policy nets / prior x critic / cost", "table_diffusion_schedule_length.csv; cost_accounting.csv; cost_accounting_components.csv; cost_accounting_component_summary.csv; diffusion_policy_construction_cost.csv; diffusion_schedule_audit.json; diffusion_reverse_support_*.json"),
    ("M07 robustness", "Sections 27-28, 30", "replication / regimes / stress / subgroups", "statistics_hierarchical_replication.csv; statistics_hierarchical_paired_differences.csv; replication_results.csv; data_regime_datasets.csv; data_regime_results.csv; data_regime_size_summary.csv; stress_test_results.csv; stress_test_ranking_persistence.csv; subgroup_diagnostics.csv"),
    ("M08 representation", "Sections 4, 29", "feature provenance / ablation", "representation_ablation.csv"),
    ("M09 evaluator validation", "Sections 3, 8, 14, 20", "toy tests / env semantics / Monte Carlo", "numerical_audit.json; monte_carlo_crosscheck.csv; benchmark_reference_sanity.csv; benchmark_semantics_audit.json; benchmark_audit.json"),
    ("M10 episode semantics", "Section 5", "validate_dataset + short-cutoff test", "dataset_manifests_all.csv"),
    ("Replay", "Sections 21, 32", "artifact replay", "cross_session_replay.csv; policy_reevaluation.csv; checkpoint_replay_cpu.csv; legacy_policy_reevaluation.csv; legacy_replay.json"),
    ("EDA tables and figures", "Section 6", "EDA", "eda_*.csv; eda_transitions_enriched.csv.gz; figures/eda/*"),
    ("Benchmark support figure", "Section 4", "benchmark audit", "figures/benchmark_support_set_sizes.png"),
    ("Figures 2-3", "Section 6", "EDA", "figures/paper_fig2_dataset_distributions.*; figures/paper_fig3_severity.*"),
    ("Figures 7-10", "Section 31", "figures", "figures/paper_fig7_*; figures/paper_fig8_*; figures/paper_fig9_*; figures/paper_fig10_*"),
    ("New diagnostic figures", "Section 31", "figures", "figures/fig_c01_projection_controls.*; figures/fig_c02_critic_scale_drift.*; figures/fig_m03_hierarchical_replication.*; figures/fig_m04_fqe_vs_exact.*; figures/fig_m07_stress_ranking.*"),
    ("Gates, manifests and provenance", "Sections 1, 30, 33, 34", "support gate / closure / inventory", "fresh_custom_model_performance_gate.json; audit_closure_checklist.csv; fresh_reproducibility_manifest.json; input_sha256.json; table_to_command_map.csv; checkpoint_metadata.csv"),
    ("Compute resources (checklist)", "Section 33", "compute_summary over logs/execution_log.jsonl", "compute_summary.csv"),
    # ---- Part II: cross-environment exact study (Sections II.1-II.7); paths relative to tables/ and figures/ ----
    ("Part II configuration and preregistration", "Section II.1", "XConfig / apply_xrun_mode / PREREG_PART2", "xenv/xenv_config.json; xenv/xenv_preregistration.json"),
    ("Part II environment suite and adapter checks", "Section II.3", "x_icu_env / gumbel_true_env / gumbel_exactness_check / garnet_true_env / construct_benchmark / x_environment_suite_table", "xenv/icu_adapter_check.json; xenv/gumbel_exactness_check.json; xenv/gumbel_simulator_report.json; xenv/gumbel_state_decode.csv; xenv/xenv_environment_suite.csv; data/xenv_gumbel_true_*.npz"),
    ("Part II method catalogue and self-test", "Section II.4", "run_xmethods / X_METHOD_CATALOG / x_methods_selftest", "xenv/xenv_method_catalog.csv; xenv/xenv_methods_selftest.json"),
    ("Propositions P1-P7", "Section II.5", "proposition checks", "xenv/proposition_checks.csv; xenv/proposition_summary.csv"),
    ("Exact constrained frontiers", "Section II.5", "x_cmdp_lp over XCFG.frontier_kappas", "xenv/exact_frontiers.csv"),
    ("Part II raw results and draw metadata", "Section II.6", "run_cell_draw", "xenv/xenv_results_all.csv.gz; xenv/xenv_draw_meta.csv; xenv/xenv_driver_summary.json"),
    ("Exact tabular learners vs Part I SGD learners", "Section II.6", "tabular cross-check", "xenv/xenv_tabular_crosscheck.csv"),
    ("Main-track table: cross-environment performance", "Section II.7", "normalised scores / stratified-bootstrap IQM / profiles", "xenv/xenv_normalized_scores.csv; xenv/table_xenv_main.csv; xenv/table_xenv_main_long.csv; xenv/table_performance_profiles.csv; xenv/xenv_poi_matrix.csv; xenv/xenv_stats_checks.json"),
    ("Main-track table: benchmark fidelity", "Section II.7", "Kendall tau-b / top-1 / regret / error decomposition", "xenv/table_rank_fidelity.csv; xenv/table_rank_fidelity_draws.csv; xenv/table_error_decomposition.csv; xenv/table_error_decomposition_draws.csv"),
    ("Part II preregistered tests H1-H3", "Section II.7", "sign-flip / Holm / TOST / permutation Spearman", "xenv/xenv_prereg_tests.json; xenv/xenv_h2_family.csv"),
    ("Hyper-parameter selection (default / FQE / oracle)", "Section II.7", "hp selection", "xenv/table_hp_selection.csv"),
    ("Part II figures", "Sections II.5-II.7", "matplotlib figures", "figures/xenv/fig_xenv_frontiers.*; figures/xenv/fig_xenv_rank_fidelity.*; figures/xenv/fig_xenv_harm_phase.*; figures/xenv/fig_xenv_iqm.*; figures/xenv/fig_xenv_profiles.*; figures/xenv/fig_xenv_poi.*; figures/xenv/fig_xenv_error_decomposition.*"),
    # ---- Part II revision experiments (Section II.8, REV-1 ... REV-3) ----
    ("Revision II.8: configuration, physicians, environment suite", "Sections II.8a-II.8e", "XV3Config / gumbel_physician_report / x_v3_build_cells / x_v3_environment_suite_table", "xenv/v3_config.json; xenv/v3_physician_report.csv; xenv/v3_environment_suite.csv; xenv/v3_environment_checks.json"),
    ("Revision II.8: raw results of the new cells (REV-1)", "Section II.8f", "x_v3_run_cells / run_v3_cell_draw", "xenv/v3_results_all.csv.gz; xenv/v3_draw_meta.csv; xenv/v3_driver_summary.json"),
    ("Revision II.8: bootstrap-ensemble intervals, raw (REV-2)", "Section II.8g", "x_v3_boot_run / run_v3_boot_draw / x_boot_step_map / x_boot_model", "xenv/v3_bootstrap_config.json; xenv/v3_bootstrap_results.csv.gz; xenv/v3_bootstrap_meta.csv; xenv/v3_bootstrap_summary.json"),
    ("Revision II.8: ICU-Sepsis harm curves J(lambda) (REV-3)", "Section II.8h", "x_harm_env / x_harm_curves / x_harm_summary", "xenv/v3_icu_harm_curves.csv; xenv/v3_icu_harm_summary.csv; xenv/v3_icu_harm_checks.json"),
    ("Revision II.8: bracketing coverage, certified comparisons, rank fidelity, bootstrap coverage", "Section II.8i", "xv3_stats_run", "xenv/v3_bracketing.csv; xenv/v3_bracketing_draws.csv; xenv/v3_certified.csv; xenv/v3_certified_draws.csv; xenv/v3_rank_fidelity.csv; xenv/v3_rank_fidelity_draws.csv; xenv/v3_bootstrap_intervals.csv; xenv/v3_bootstrap_coverage.csv; xenv/v3_stats_checks.json"),
    ("Revision II.8 figures", "Sections II.8h-II.8i", "matplotlib figures", "figures/xenv/fig_v3_bracketing.*; figures/xenv/fig_v3_bootstrap.*; figures/xenv/fig_v3_icu_harm.*"),
    # ---- Part II second-revision experiments (Section II.9, REV-4 ... REV-9) ----
    ("Revision II.9: configuration", "Section II.9a", "XV4Config / apply_xv4_run_mode", "xenv/v4_config.json"),
    ("Revision II.9: certificates without NHB, pair types, abstention, hidden-benefit sensitivity kappa, II.8g intervals over all policies (REV-4)", "Section II.9b", "xv4r_run / x_v4_pair_claims / xv4r_kappa_claims / xv4r_boot_table", "xenv/v4_robust_pairs.csv.gz; xenv/v4_robust_summary.csv; xenv/v4_robust_gapbins.csv; xenv/v4_robust_kappa.csv; xenv/v4_v3boot_allpolicy.csv; xenv/v4_robust_summary.json"),
    ("Revision II.9: confidence-set certificates on the true state space (REV-5)", "Section II.9c", "run_v4_cs_draw / x_cs_robust_vector / x_cs_l1_extreme / x_cs_summary_table", "xenv/v4_cs_policies.csv.gz; xenv/v4_cs_draws.csv; xenv/v4_cs_summary.csv; xenv/v4_cs_summary.json"),
    ("Revision II.9: bootstrap B = 200, fixed mask and re-estimated construction (REV-6)", "Section II.9d", "run_v4_boot_draw / x_v4_boot_construct / x_v4_boot_intervals / x_v4_boot_coverage", "xenv/v4_boot_intervals.csv.gz; xenv/v4_boot_draws.csv; xenv/v4_boot_coverage.csv; xenv/v4_boot_meta.csv; xenv/v4_boot_summary.json"),
    ("Revision II.9: sensitivity cells - timeout hazard, dead-state rule, softer physicians, state abstractions (REV-7)", "Section II.9e", "build_v4_instance / run_v4_cell_draw / xv4_sens_analyses", "xenv/v4_sens_results_all.csv.gz; xenv/v4_sens_draw_meta.csv; xenv/v4_sens_summary.csv; xenv/v4_sens_certified_draws.csv; xenv/v4_sens_rank_fidelity.csv; xenv/v4_sens_rank_fidelity_draws.csv; xenv/v4_sens_summary.json"),
    ("Revision II.9: lift sensitivity and the range of J_true over all lifts (REV-8)", "Section II.9f", "run_v4_lift_draw / x_v4_lift_bounds / x_v4_lift_summary", "xenv/v4_lift_results.csv.gz; xenv/v4_lift_summary.csv; xenv/v4_lift_summary.json"),
    ("Revision II.9: clean rerun of Part II into an empty cache (REV-9)", "Section II.9g", "x_v4_clean_run / x_v4_clean_compare / x_v4_clean_certified", "xenv/v4_clean_check.json; xenv/v4_clean_diffs.csv"),
    ("Revision II.9: overview, headline numbers and checks", "Section II.9h", "xv4h_run", "xenv/v4_overview.json; xenv/v4_stats_checks.json"),
    ("Revision II.9 figures", "Section II.9h", "matplotlib figures", "figures/xenv/fig_v4_confidence.*; figures/xenv/fig_v4_bootstrap.*"),
]
TABLE_MAP = pd.DataFrame(TABLE_TO_COMMAND, columns=["manuscript_item", "notebook_section", "code", "output_files"])
TABLE_MAP.to_csv(TABLE_DIR / "table_to_command_map.csv", index=False)

# Dataset manifests for every generated dataset (M10 closure), independent of the release switch.
man_rows = []
for mf in sorted(DATA_DIR.glob("*.manifest.json")):
    d = json.loads(mf.read_text())
    man_rows.append({"file": d["file"], "spec_hash": d["spec_hash"], "content_sha256": d["content_sha256"],
                     "file_sha256": d.get("file_sha256"),
                     **{f"spec_{k}": v for k, v in d["spec"].items() if k not in ("benchmark",)}, **d["report"]})
DATASET_MANIFESTS = pd.DataFrame(man_rows)
DATASET_MANIFESTS.to_csv(TABLE_DIR / "dataset_manifests_all.csv", index=False)
display(DATASET_MANIFESTS[[c for c in ["file", "transitions", "episodes", "terminated_episodes", "censored_episodes",
                                       "inadmissible_logged_fraction"] if c in DATASET_MANIFESTS]])

# Every generated table / figure must be mapped to its section and code.
import fnmatch
_patterns = [p_.strip() for v in TABLE_MAP["output_files"] for p_ in v.split(";")]
_unmapped = []
for f in sorted(list(TABLE_DIR.rglob("*")) + list(FIG_DIR.rglob("*"))):
    if not f.is_file():
        continue
    rel_t = str(f.relative_to(TABLE_DIR)) if TABLE_DIR in f.parents else None
    rel_f = "figures/" + str(f.relative_to(FIG_DIR)) if FIG_DIR in f.parents else None
    if not any(fnmatch.fnmatch(x, pat) for pat in _patterns for x in (rel_t, rel_f) if x):
        _unmapped.append(str(f.relative_to(ROOT)))
pd.DataFrame({"unmapped_output": _unmapped}).to_csv(RELEASE_DIR / "unmapped_outputs.csv", index=False)
if _unmapped:
    print(f"NOTE: {len(_unmapped)} generated files are not listed in the table-to-command map (release/unmapped_outputs.csv).")

meta_rows = []
for f in CKPT_DIR.rglob("*.identity.json"):
    d = json.loads(f.read_text())
    pt_file = f.with_name(d["file"])
    meta_rows.append({"label": d["label"], "identity_hash": d["identity_hash"], "saved_utc": d["saved_utc"],
                      "trained_on": d.get("device_type"), "gpu": d.get("gpu"),
                      "written_this_session": d["identity_hash"] in STORE.session_hashes,
                      "loaded_from_earlier_session": d["identity_hash"] in STORE.loaded_hashes,
                      "file": str(pt_file.relative_to(ROOT)), "bytes": pt_file.stat().st_size if pt_file.exists() else None})
CHECKPOINT_META = pd.DataFrame(meta_rows)
CHECKPOINT_META.to_csv(TABLE_DIR / "checkpoint_metadata.csv", index=False)

def export_release_code(code_dir: Path) -> None:
    """Executed notebook (Colab export) and executed cell sources into code_dir; a failed export keeps older files."""
    code_dir.mkdir(parents=True, exist_ok=True)
    try:
        from google.colab import _message
        nb_json = _message.blocking_request("get_ipynb", timeout_sec=120)["ipynb"]
        (code_dir / "notebook_executed.ipynb").write_text(json.dumps(nb_json), encoding="utf-8")
        print("Executed notebook exported to the release.")
    except Exception as exc:
        print("Could not export the notebook automatically (download it via File -> Download):", type(exc).__name__)
    try:
        _cells = get_ipython().user_ns.get("In", [])[1:]  # noqa: F821
        (code_dir / "executed_cells.py").write_text("\n\n# ---- cell ----\n".join(_cells), encoding="utf-8")
    except Exception:
        pass


if EXPERIMENTS["release_package"]:
    code_dir = RELEASE_DIR / "code"
    export_release_code(code_dir)   # re-exported in 34b after the gate; this copy is the fallback
    lock = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (RELEASE_DIR / "environment_lock.txt").write_text(lock, encoding="utf-8")
    save_json(RUNTIME_INFO, RELEASE_DIR / "runtime.json")
    TABLE_MAP.to_csv(RELEASE_DIR / "table_to_command_map.csv", index=False)
    registry_frame().to_csv(RELEASE_DIR / "policy_registry.csv", index=False)
    pd.DataFrame(list(LATEST_FILES.values())).to_csv(RELEASE_DIR / "policy_files.csv", index=False)
    CHECKPOINT_META.to_csv(RELEASE_DIR / "checkpoint_metadata.csv", index=False)
    inputs = {
        "benchmark_assets_sha256": BENCH.asset_hashes,
        "benchmark_identity": BENCH.identity,
        "construction_scripts": PROVENANCE.get("fetched_sha256", PROVENANCE_REF["sha256"]),
        "datasets": {r["file"]: {"content_sha256": r["content_sha256"], "file_sha256": r["file_sha256"]} for r in man_rows},
        "config_hash": obj_hash(asdict(CFG)),
        "preregistration_hash": obj_hash(PREREG),
        "run_profile": RUN_PROFILE,
        "xenv_config_hash": obj_hash(asdict(XCFG)) if "XCFG" in globals() else None,
        "xenv_preregistration_hash": obj_hash(PREREG_PART2) if "PREREG_PART2" in globals() else None,
        "xenv_code_hash": globals().get("XENV_CODE_HASH"),
        "xenv_v3_code_hashes": {"cells": globals().get("XENV_V3_CODE_HASH"),
                                "bootstrap": globals().get("XV3_BOOT_CODE_HASH"),
                                "icu_harm": globals().get("XV3_HARM_CODE_HASH")},
        "xenv_v3_config_hashes": {"cells": obj_hash(asdict(XV3CFG)) if "XV3CFG" in globals() else None,
                                  "bootstrap": obj_hash(asdict(XV3BOOT)) if "XV3BOOT" in globals() else None},
        "xenv_v4_code_hashes": {"common": globals().get("XV4_COMMON_CODE_HASH"),
                                "robust": globals().get("XV4_ROBUST_CODE_HASH"),
                                "confidence": globals().get("XV4_CS_CODE_HASH"),
                                "bootstrap": globals().get("XV4_BOOT_CODE_HASH"),
                                "sensitivity": globals().get("XV4_SENS_CODE_HASH"),
                                "lift": globals().get("XV4_LIFT_CODE_HASH"),
                                "clean_check": globals().get("XV4_CLEAN_CODE_HASH"),
                                "stats": globals().get("XV4_STATS_CODE_HASH")},
        "xenv_v4_config_hashes": {"v4": obj_hash(asdict(XV4CFG)) if "XV4CFG" in globals() else None},
        "code_hashes": {"dataset": DATASET_CODE_HASH, "critics": CRITIC_CODE_HASH, "policy_nets": POLICYNET_CODE_HASH,
                        "diffusion": DIFFUSION_CODE_HASH,
                        "evaluator": code_hash(ExactMDPEvaluator, validate_policy, project_to_support)},
        "release_code_sha256": {p_.name: sha256_file(p_) for p_ in code_dir.glob("*") if p_.is_file()},
        "legacy_inputs": {str(p_.relative_to(LEGACY_ROOT)): sha256_file(p_) for p_ in
                          ([LEGACY_ROOT / "data" / "offline_expert_10000eps_seed2026.npz", LEGACY_ROOT / "tables" / "exact_results_all.csv"]
                           if LEGACY_ROOT.exists() else []) if p_.exists()},
    }
    save_json(inputs, TABLE_DIR / "input_sha256.json")
    save_json(inputs, RELEASE_DIR / "input_sha256.json")
    (RELEASE_DIR / "README_release.md").write_text(
        "# Release contents\n\n"
        f"Run mode: {RUN_MODE}; created {utc_now()}.\n\n"
        "* release/code/ — executed notebook (Colab export) and executed cell sources\n"
        "* tables/ — every CSV/JSON behind the manuscript tables (see table_to_command_map.csv)\n"
        "* figures/ — PNG + PDF figures\n* policies/ — every evaluated policy (npz) per group and seed\n"
        "* checkpoints/ — identity-addressed checkpoints (+ .identity.json metadata)\n"
        "* data/ — dataset manifests (content SHA-256) and, if included, the frozen datasets; Part II exact Gumbel "
        "simulator matrices (xenv_gumbel_true_*.npz)\n* logs/ — training and execution logs\n"
        "* tables/xenv/, figures/xenv/ — Part II (cross-environment exact study) tables and figures; v3_* and fig_v3_* are "
        "the revision experiments of Section II.8 (REV-1 ... REV-3), v4_* and fig_v4_* the second-revision experiments "
        "of Section II.9 (REV-4 ... REV-9; status and headline numbers in tables/xenv/v4_overview.json)\n"
        "* xenv_cache/ — identity-addressed Part II results per (cell, draw); replay with ALLOW_TRAINING=False\n"
        "* tables/compute_summary.csv — logged wall time per section / session / part, with the GPU\n"
        "* environment_lock.txt, runtime.json, artifact_inventory.csv, input_sha256.json\n\n"
        f"Compute profile: {COMPUTE_PROFILE} ({len(CFG.seeds)} main seeds; {len(CFG.replication_dataset_seeds)} x "
        f"{len(CFG.replication_train_seeds)} replication).\n\n"
        "Clean training: a fresh ROOT (or FORCE_RETRAIN=True; delete checkpoints/force_retrain_token.json to start a new forced run).\n"
        "Artifact replay: same ROOT, ALLOW_TRAINING=False. Checkpoints are device-independent in identity; CPU replay of GPU "
        "checkpoints is supported and labelled (near-tie greedy actions may differ).\n",
        encoding="utf-8")
    print("Release files written; the code re-export, the inventory and the archive follow the Section 34 gate.")
