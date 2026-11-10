# Where each table and figure of the paper comes from

Numbers refer to an earlier, longer version of the manuscript.

- **Scripts** are in `paper_scripts/`. They read three sources: the result files of the final run in `results/`,
  the printed outputs of `run_record/Sepsis_Offline_RL_v4_final_run.ipynb`, and, for Table 24 only, the earlier
  run's facts files in `paper_assets_expected/original_run/`. They run no experiment.
  `bash paper_scripts/reproduce_paper_assets.sh` runs all of them in order.
- **Notebook sections** are those of the supplied notebook. The notebook's own file-level map is
  `results/release_metadata/table_to_command_map.csv`, written by the final run. Its `manuscript_item` labels
  (Tables 1–8, Figures 2–3 and 7–10) refer to the report of the earlier study audited in App. F, not to this PDF.

Order used by the driver:

1. `extract_v3.py` writes `regenerated/v3_tables/` from the final-run record.
2. `use_exact_v3.py` replaces four of those rounded tables by the run's full-precision files in
   `results/tables/xenv/` (`v3_bracketing.csv`, `v3_certified.csv`, `v3_bootstrap_coverage.csv`,
   `v3_icu_harm_summary.csv`). The result is identical to `results/v3_tables/`.
3. `make_v3_assets.py` writes `facts_v3.json` and the App. H tables.
4. `make_assets.py` writes `facts.json`, figures and tables.
5. `make_robustness.py` reads `facts.json` and writes `facts_robust.json` and its tables.
6. `make_v4_assets.py` writes `facts_v4.json` and the App. J tables.
7. `make_repro_summary.py` compares the final run's facts files with those of the earlier run
   (`paper_assets_expected/original_run/`) and writes `facts_repro.json` and Table 24.
8. `verify_bracket.py` writes `theory_verification.json`.

In the tables below, inputs are paths under `results/`. `v3_tables/…` stands for the folder built in steps 1–2.

## Main text

| Item | Content | Script → output | Input | Notebook section |
|---|---|---|---|---|
| Figure 1 | Pipeline diagram | drawn in LaTeX (TikZ) | – | – |
| Figure 2 | Bracketing on true and estimated rows | `make_assets.py` → `fig_bracketing` | `tables/xenv/xenv_results_all.csv.gz` | II.6 |
| Table 1 | Certified comparisons, true vs estimated rows, all cells | `make_assets.py` → `tab_certified.tex` | `tables/xenv/xenv_results_all.csv.gz`; `v3_tables/certified_v3.csv`, `bracket_v3.csv` (additional cells) | II.6, II.8 |
| Table 2 | NHB-free vs conditional certificates; matched-coverage margins | `make_robustness.py` → `tab_robust.tex` | `tables/xenv/xenv_results_all.csv.gz` | II.6 |
| Table 3 | ICU-Sepsis identified intervals and certificates | `make_assets.py` → `tab_icu_intervals.tex` | `tables/exact_results_all.csv`; `tables/xenv/xenv_results_all.csv.gz` (last row) | Sections 19–20; II.6 |
| Table 4 | Hidden-benefit (κ) sensitivity | `make_robustness.py` → `tab_kappa.tex` | `tables/xenv/xenv_results_all.csv.gz`; `tables/exact_results_all.csv` (audited main-table row) | II.6; Sections 19–20 |
| Numbers in the text | All quoted counts and rates | `facts.json`, `facts_v3.json`, `facts_robust.json`, `facts_v4.json`, `facts_repro.json`, `theory_verification.json` | all of the above, plus the other `tables/` files that `make_assets.py` reads: `statistics_primary.json`, `critic_stability_gate.json`, `projection_equivalence_checks.csv`, `monte_carlo_crosscheck.csv`, `statistics_hierarchical_replication.csv`, `statistics_hierarchical_paired_differences.csv`, `fresh_reproducibility_manifest.json`, `eda_dataset_summary.csv`, `benchmark_semantics_audit.json`, `corrected_support_audit.json`, `legacy_replay.json`, `legacy_projection_controls_summary.csv` | – |

## Appendices

| Item | Content | Script → output | Input | Notebook section |
|---|---|---|---|---|
| App. B.5 counts | Numerical check of the bracketing theorem on random MDPs | `verify_bracket.py` → `theory_verification.json` | none (self-contained, about 100 s) | – |
| Table 5 | Method catalogue | written in LaTeX from `tables/xenv/xenv_method_catalog.csv` | `tables/xenv/xenv_method_catalog.csv` | II.4 |
| Figure 3 | Audited ICU-Sepsis study | `make_assets.py` → `fig_icu` | `tables/exact_results_all.csv`, `table_main_policy_comparison.csv`, `critic_stability_summary_main_all_seeds.csv` | Sections 16, 19–20, 23 |
| App. F.5 legacy replay and oracle tie | Replay of the original study's artefacts | `make_assets.py` → `facts.json` (`legacy_replay`, `legacy_unconstrained_oracle_lowest_index_ties`) | `tables/legacy_replay.json`, `tables/legacy_projection_controls_summary.csv` | Section 21 (needs the earlier study's run directory; present in the final run, not supplied) |
| Table 6 | ICU-Sepsis exact returns, all Part I policies | `make_assets.py` → `tab_icu_main.tex` | `tables/table_main_policy_comparison.csv` | Sections 19–20 |
| Table 7 | Part I secondary family | `make_assets.py` → `tab_icu_secondary.tex` | `tables/statistics_secondary_family.csv` | Section 30 |
| Figure 4 | Rank fidelity (Kendall τ_b) | `make_assets.py` → `fig_fidelity` | `tables/xenv/table_rank_fidelity*.csv` | II.7 |
| Figure 5 | Exact constrained frontiers | `make_assets.py` → `fig_frontiers` | `tables/xenv/exact_frontiers.csv` | II.5 |
| Figure 6 | H2 forest plot | `make_assets.py` → `fig_h2` | `tables/xenv/xenv_h2_family.csv` | II.7 |
| Table 8 | Environment cells | `make_assets.py` → `tab_envs.tex` | `tables/xenv/xenv_environment_suite.csv` | II.3 |
| Table 9 | H1 test | `make_assets.py` → `tab_h1.tex` | `tables/xenv/xenv_prereg_tests.json`, `table_rank_fidelity.csv` | II.7 |
| Table 10 | Exact error decomposition | `make_assets.py` → `tab_decomp.tex` | `tables/xenv/table_error_decomposition.csv` | II.7 |
| Figure 7 | Benchmark vs true values (Gumbel) | `make_assets.py` → `fig_bench_vs_true` | `tables/xenv/xenv_normalized_scores.csv` | II.7 |
| Table 11 | Proposition checks | `make_assets.py` → `tab_props.tex` | `tables/xenv/proposition_summary.csv` | II.5 |
| Table 12 | Rank fidelity, all cells | `make_assets.py` → `tab_fidelity_full.tex` | `tables/xenv/table_rank_fidelity.csv` | II.7 |
| Table 13 | H2 in full | `make_assets.py` → `tab_h2.tex` | `tables/xenv/xenv_h2_family.csv` | II.7 |
| Table 14 | Gumbel H2 treatments per cell | `make_assets.py` → `tab_gumbel_cells.tex` | `tables/xenv/xenv_normalized_scores.csv` | II.7 |
| Table 15 | Raw exact returns | `make_assets.py` → `tab_raw_returns.tex` | `tables/xenv/xenv_normalized_scores.csv` | II.7 |
| Table 16 | Exact-tabular vs SGD cross-check (corrected) | `make_v3_assets.py` → `tab_crosscheck.tex` | `v3_tables/cross_check.csv` | II.6 |
| Table 17 | The four Gumbel physicians | `make_v3_assets.py` → `tab_v3_physicians.tex` | `v3_tables/physicians_diag.csv` | II.8 |
| Table 18 | Additional cells | `make_v3_assets.py` → `tab_v3_cells.tex` | `v3_tables/aggregated_diag.csv`, `v3_driver_refs.csv` | II.8e–II.8f |
| Table 19 | Point vs bootstrap intervals (B = 10) | `make_v3_assets.py` → `tab_v3_bootstrap.tex` | `v3_tables/bootstrap_v3.csv` | II.8 |
| Table 20 | ICU-Sepsis harm curves | `make_v3_assets.py` → `tab_v3_harm.tex` | `v3_tables/icu_harm.csv` | II.8 |
| Table 21 | Certified comparisons by pair type (AA/AU/UU) | `make_robustness.py` → `tab_pairtypes.tex` | `tables/xenv/xenv_results_all.csv.gz` | II.6 |
| Table 22 | Confidence-set certificates (95% Weissman L1 sets) vs point intervals | `make_v4_assets.py` → `tab_v4_cs.tex` | `tables/xenv/v4_cs_summary.csv`, `v4_cs_summary.json`, `v4_config.json` | II.9c |
| Table 23 | Sensitivity cells: softer physicians, timeout, dead-state rule, abstractions | `make_v4_assets.py` → `tab_v4_sens.tex` | `tables/xenv/v4_sens_summary.csv`, `v4_sens_summary.json`, `v4_config.json` | II.9e |
| Table 24 | Final run vs earlier run | `make_repro_summary.py` → `tab_repro.tex` | `facts*.json` of both runs (`paper_assets_expected/original_run/`) | – |
| Table 25 | Reproducibility checklist | written in LaTeX | – | – |
