# %% [markdown]
# ## 34. Final integrity gate and audit closure checklist
#
# Structural invariants are hard assertions. Scientific checks are reported with pass/fail so that a
# legitimate negative finding never crashes the notebook. The checklist maps every audit ID to its evidence;
# items that concern only the paper text are marked "manuscript-only".
#
# **Part II gate** (applies to whichever `xenv_*` switches are on): every proposition check must pass
# (`PROPOSITIONS_OK` is not `False`), and so must every exact-frontier cross-check (`X_FRONTIERS_OK`, which also
# covers frontiers run without `xenv_theory`); the ICU-Sepsis adapter check (`X_ICU_ADAPTER_CHECK`, run whenever any
# `xenv_*` switch of Sections II.3–II.8 is on; Section II.9 does not use the adapter) and the Gumbel exactness check
# (`GUMBEL_EXACTNESS`) must pass when they ran; every Part II
# result row has status `ok` or `improper` (improper = non-absorbing under γ = 1, counted and reported per family and
# target, never hidden); `ok` rows satisfy |J| ≤ 1 (all entry rewards lie in [−1, 1]),
# a pooled violation rate in [0, 1], P(any violation) ≤ E[violations], and equality of the two under termination.
# Closure rows **MT-1 … MT-5** record the Part II additions: multi-environment ground truth, verified
# propositions, the extended baseline suite, benchmark fidelity (H1–H3), and statistics at scale (10 seeds, 8 × 2
# replication, the configured number of Part II draws per cell, a Part II design without deviations from
# `PREREG_PART2['design']`). MT-5 fails by design for the lite profile and smoke mode, which are not paper scale. The
# last cell of this section re-exports the code snapshot and writes the release inventory (sizes, SHA-256) and the
# archive, after the closure checklist and the reproducibility manifest exist.
#
# **II.8 gate (Section II.8)**, switch-aware: it never fails because a switch is off. When an II.8 cell switch is
# on, the II.8 environment checks must pass and `XRESULTS_V3` must satisfy the row invariants of the Part II gate.
# The theorem counts of II.8i must be zero: no lower-bound violation on the true rows, non-negative widths, zero
# width for admissible policies, width ≤ p·(v_max − v_min), and values inside [v_min, v_max]. With
# `xenv_v3_bootstrap`, the per-draw bootstrap invariants (Theorem part iii on every replicate) must hold. With
# `xenv_v3_icu_harm`, the harm-curve identities must hold: endpoints equal to the reported mean / terminate values,
# J_term ≤ J_λ ≤ J_mean, monotone in λ, and certified pairs ordered at every λ. Coverage and certification rates are
# scientific outcomes: they are reported, never gated. Closure rows **REV-1 … REV-3** record the additional
# experiments: REV-1 the II.8 cells (physicians that are not near-optimal, aggregated-state benchmarks with diabetes
# hidden, random-row Garnet) with their bracketing, certified-comparison and rank-fidelity analyses; REV-2 the
# bootstrap-ensemble intervals; REV-3 the ICU-Sepsis harm curves and certified comparisons.
#
# **II.9 gate (Section II.9)**, switch-aware in the same way. It gates only on invariants, on the completion
# of the enabled subsections and on the clean check's pass flag; coverage and error rates are never gated. Hard
# assertions: every invariant flag that II.9h collected is not false (II.9b theorem and identity counts, II.9c
# self-test of the inner problem and bound order, II.9d per-replicate invariants and unit weights, II.9e self-test and
# theorem counts, II.9f invariant slacks) and the orderings II.9h recounted are zero; with II.9c, L ≤ U_free; with
# II.9d, Theorem 1(iii) on every replicate; with II.9e, its draw and theorem checks. Closure rows **REV-4 … REV-9**
# record each subsection:
# switch on, status `run`, draws complete and invariants holding (REV-9 also needs the clean rerun to pass; it is not
# run in artifact-replay mode, `ALLOW_TRAINING = False`). REV-4 is the robustness analysis of the stored results
# (certificates without NHB, pair types, κ, II.8g intervals over all policies), REV-5 the confidence-set certificates,
# REV-6 the bootstrap with the construction re-estimated, REV-7 the sensitivity cells, REV-8 the lift sensitivity and
# REV-9 the clean rerun.

# %%
# ============================================================
# 34. Final integrity gate
# ============================================================
EXECLOG.mark("34. Final integrity gate")
refresh_results()
finalize_projection_checks(verbose=False)
for name, e in REGISTRY.items():
    for seed, pi in e.by_seed.items():
        validate_policy(pi, f"{name} seed={seed}")
STRUCTURAL_USES = {"extraction", "training+extraction", "projection", "construction", "inference (every reverse step)",
                   "inference (final step only)"}
support_failures = []
for name, e in REGISTRY.items():
    if "released benchmark mask" in e.mask_source and "dataset-estimated" not in e.mask_source and e.mask_use in STRUCTURAL_USES:
        worst = max(inadmissible_mass(pi).max() for pi in e.by_seed.values())
        if worst > 1e-12:
            support_failures.append((name, worst))
assert not support_failures, f"Masked / projected policies with inadmissible mass: {support_failures[:5]}"
core = RESULTS[RESULTS["group"].isin(["main", "reference"])]
improper_core = core[core["status"] != "ok"][["name", "seed", "rule", "status"]]
if RUN_MODE == "paper":
    assert improper_core.empty, improper_core
elif len(improper_core):
    print("Smoke mode: some barely-trained main policies are improper (expected with tiny budgets):\n", improper_core)
ok = RESULTS[RESULTS["status"] == "ok"]
assert ok["return"].between(-1e-10, 1 + 1e-10).all()
assert ok["inadmissible_rate"].between(-1e-12, 1 + 1e-12).all()
assert (ok["prob_any_violation"] <= ok["expected_violations"] + 1e-9).all(), "P(any violation) must not exceed E[count]."
term_ok = ok[ok["rule"] == "terminate"]
assert np.allclose(term_ok["prob_any_violation"], term_ok["expected_violations"], atol=1e-9), \
    "Under termination the expected violation count must equal the first-violation probability."
assert DATASET_REPORT["transitions"] == len(OFFLINE["state"]) and DATASET_REPORT["episodes"] == CFG.dataset_episodes
assert 0.0 <= SUPPORT_AUDIT["admissible_pair_coverage"] <= 1.0

# ---------- Part II gate (cross-environment exact study) ----------
XENV_SWITCH_STATE = {k: bool(v) for k, v in EXPERIMENTS.items() if k.startswith("xenv_")}
XENV_RUN_FAMILIES = [f for f in ("icu", "gumbel", "garnet") if XENV_SWITCH_STATE.get(f"xenv_{f}")]


def xenv_pass_flag(check: Optional[dict], name: str) -> Optional[bool]:
    """Boolean pass flag ('passes') of a Part II self-check dict; None when the check did not run (missing, or the
    placeholder {"status": "not run"} that Section II.3 leaves when its switch is off)."""
    if check is None:
        return None
    if "passes" not in check and "passed" not in check and str(check.get("status", "")).startswith("not run"):
        return None
    flag = check.get("passes", check.get("passed"))
    if not isinstance(flag, (bool, np.bool_)):
        raise KeyError(f"{name} has no boolean 'passes' flag; keys: {sorted(check)[:25]}")
    return bool(flag)


PROPOSITIONS_STATUS = globals().get("PROPOSITIONS_OK")
if XENV_SWITCH_STATE["xenv_theory"]:
    assert "PROPOSITIONS_OK" in globals(), "xenv_theory is on but Section II.5 did not run."
assert PROPOSITIONS_STATUS is not False, "A Part II proposition check failed - see xenv/proposition_checks.csv."
ICU_ADAPTER_OK = xenv_pass_flag(globals().get("X_ICU_ADAPTER_CHECK"), "X_ICU_ADAPTER_CHECK")
GUMBEL_EXACT_OK = xenv_pass_flag(globals().get("GUMBEL_EXACTNESS"), "GUMBEL_EXACTNESS")
assert ICU_ADAPTER_OK == globals().get("X_ICU_ADAPTER_OK"), "X_ICU_ADAPTER_CHECK and X_ICU_ADAPTER_OK disagree."
assert GUMBEL_EXACT_OK == globals().get("GUMBEL_EXACTNESS_OK"), "GUMBEL_EXACTNESS and GUMBEL_EXACTNESS_OK disagree."
assert ICU_ADAPTER_OK is not False, "ICU-Sepsis FiniteEnv adapter differs from the Part I evaluator (xenv/icu_adapter_check.json)."
assert GUMBEL_EXACT_OK is not False, "Enumerated Gumbel rows disagree with the simulator step (xenv/gumbel_exactness_check.json)."
if any(v for k, v in XENV_SWITCH_STATE.items() if not k.startswith("xenv_v4_")):   # II.9 never uses the adapter
    assert ICU_ADAPTER_OK is not None, "a Part II switch is on but the ICU-Sepsis adapter check (Section II.3) did not run."
if XENV_SWITCH_STATE["xenv_gumbel"]:
    assert GUMBEL_EXACT_OK is not None, "xenv_gumbel is on but the Gumbel exactness check did not run."
FRONTIERS_STATUS = globals().get("X_FRONTIERS_OK")
if XENV_SWITCH_STATE["xenv_frontiers"]:
    assert "X_FRONTIERS_OK" in globals(), "xenv_frontiers is on but Section II.5 did not run."
assert FRONTIERS_STATUS is not False, "An exact-frontier cross-check failed - see xenv/exact_frontiers.csv (column passes)."

XRES_GATE = globals().get("XRESULTS")
XRES_GATE = pd.DataFrame() if XRES_GATE is None else XRES_GATE
XIMPROPER = pd.DataFrame(columns=["family", "target", "rows", "improper"])
XDRAWS = pd.DataFrame(columns=["family", "cell", "draws", "configured_draws"])
if XENV_RUN_FAMILIES:
    assert len(XRES_GATE), f"Part II switches {XENV_RUN_FAMILIES} are on but XRESULTS is missing or empty."
    assert bool(globals().get("XENV_DONE", False)), "Part II statistics (Section II.7) did not complete (XENV_DONE)."
if len(XRES_GATE):
    _xneed = {"cell", "family", "draw", "policy", "method_family", "target", "semantics", "status", "return",
              "inadmissible_rate", "expected_violations", "prob_any_violation"}
    assert _xneed <= set(XRES_GATE.columns), f"XRESULTS lacks columns {sorted(_xneed - set(XRES_GATE.columns))}"
    _xbad = sorted(set(XRES_GATE["status"].astype(str)) - {"ok", "improper"})
    assert not _xbad, f"Part II rows with unexpected status: {_xbad}"
    _xok = XRES_GATE[XRES_GATE["status"] == "ok"]
    _xret = _xok["return"].to_numpy(dtype=float)
    assert np.isfinite(_xret).all() and (np.abs(_xret) <= 1 + 1e-9).all(), "Part II return outside [-1, 1]."
    assert _xok["inadmissible_rate"].between(-1e-12, 1 + 1e-12).all(), "Part II violation rate outside [0, 1]."
    assert (_xok["prob_any_violation"] <= _xok["expected_violations"] + 1e-9).all(), "Part II: P(any violation) > E[count]."
    _xterm = _xok[_xok["semantics"] == "terminate"]
    assert np.allclose(_xterm["prob_any_violation"], _xterm["expected_violations"], atol=1e-9), \
        "Part II: under termination the expected violation count must equal the first-violation probability."
    XIMPROPER = (XRES_GATE.assign(improper=XRES_GATE["status"].eq("improper"))
                 .groupby(["family", "target"])["improper"].agg(rows="size", improper="sum").reset_index())
    _xcfg_draws = {f: int(getattr(XCFG, f"{f}_draws")) for f in ("icu", "gumbel", "garnet")}
    XDRAWS = (XRES_GATE.groupby(["family", "cell"])["draw"].nunique().rename("draws").reset_index()
              .assign(configured_draws=lambda d: d["family"].map(_xcfg_draws)))
    print("Part II improper rows (family x target):")
    display(XIMPROPER)
XDRAWS_COMPLETE = (bool((XDRAWS["draws"] == XDRAWS["configured_draws"]).all()
                        and set(XENV_RUN_FAMILIES) <= set(XDRAWS["family"])) if XENV_RUN_FAMILIES else None)
GUMBEL_PROVENANCE_STATUS = (globals().get("GUMBEL_REPORT") or {}).get("provenance", {}).get("status")
XGATE = {"switches": XENV_SWITCH_STATE, "families_run": XENV_RUN_FAMILIES, "propositions_ok": PROPOSITIONS_STATUS,
         "frontiers_ok": FRONTIERS_STATUS, "icu_adapter_ok": ICU_ADAPTER_OK, "gumbel_exactness_ok": GUMBEL_EXACT_OK,
         "gumbel_provenance": GUMBEL_PROVENANCE_STATUS,
         "result_rows": int(len(XRES_GATE)), "improper_rows": int(XIMPROPER["improper"].sum()) if len(XIMPROPER) else 0,
         "improper_by_family_target": XIMPROPER.to_dict(orient="records"),
         "draws_per_cell": XDRAWS.to_dict(orient="records"), "draws_complete": XDRAWS_COMPLETE,
         "xenv_done": globals().get("XENV_DONE")}
print(json.dumps({k: v for k, v in XGATE.items() if k not in ("improper_by_family_target", "draws_per_cell")},
                 indent=2, default=_json_default))

# ---------- Revision gate (Section II.8; switch-aware, never fails because a switch is off) ----------
XV3_SWITCH_STATE = {k: bool(EXPERIMENTS.get(k, False)) for k in globals().get("XV3_ALL_SWITCHES", ())}
_xv3_cells_on = bool(globals().get("X_V3_CELLS")) and any(
    XV3_SWITCH_STATE.get(k, False) for k in ("xenv_v3_physicians", "xenv_v3_aggregated", "xenv_v3_garnet_random"))
_xv3_env_ok = xenv_pass_flag(globals().get("X_V3_ENV_CHECK"), "X_V3_ENV_CHECK")
XRES_V3_GATE = globals().get("XRESULTS_V3")
XRES_V3_GATE = pd.DataFrame() if XRES_V3_GATE is None else XRES_V3_GATE
if _xv3_cells_on:
    assert _xv3_env_ok is True, "II.8 environment checks did not pass (xenv/v3_environment_checks.json)."
    assert len(XRES_V3_GATE), "II.8 cell switches are on but XRESULTS_V3 is missing or empty (Section II.8f)."
if len(XRES_V3_GATE):
    _v3bad = sorted(set(XRES_V3_GATE["status"].astype(str)) - {"ok", "improper"})
    assert not _v3bad, f"II.8 rows with unexpected status: {_v3bad}"
    _v3ok = XRES_V3_GATE[XRES_V3_GATE["status"] == "ok"]
    _v3ret = _v3ok["return"].to_numpy(dtype=float)
    assert np.isfinite(_v3ret).all() and (np.abs(_v3ret) <= 1 + 1e-9).all(), "II.8 return outside [-1, 1]."
    assert _v3ok["inadmissible_rate"].between(-1e-12, 1 + 1e-12).all(), "II.8 violation rate outside [0, 1]."
    assert (_v3ok["prob_any_violation"] <= _v3ok["expected_violations"] + 1e-9).all(), "II.8: P(any violation) > E[count]."
    _v3term = _v3ok[_v3ok["semantics"] == "terminate"]
    assert np.allclose(_v3term["prob_any_violation"], _v3term["expected_violations"], atol=1e-9), \
        "II.8: under termination the expected violation count must equal the first-violation probability."
    del _v3bad, _v3ok, _v3ret, _v3term
_v3drv = globals().get("XV3_DRIVER_SUMMARY") or {}
_v3st = globals().get("XV3_STATS") or {}
_v3chk = _v3st.get("checks") or {}
_v3done = globals().get("XV3_STATS_DONE") or {}
_v3boot = globals().get("XV3_BOOT_SUMMARY") or {}
_v3harm = globals().get("XV3_HARM_CHECK") or {}
_v3boot_ran = _v3boot.get("status") == "run"
assert _v3chk.get("passes") is not False, f"II.8i theorem checks failed: {_v3chk.get('theorem_counts')}"
if _v3boot_ran:
    assert (_v3boot.get("max_width_minus_bound") or 0.0) <= 1e-9 and (_v3boot.get("max_rule_p_difference") or 0.0) <= 1e-9, \
        "II.8g bootstrap invariants (Theorem part iii on every replicate) failed - see xenv/v3_bootstrap_summary.json."
if XV3_SWITCH_STATE.get("xenv_v3_icu_harm"):
    assert _v3harm.get("passes") is True, "II.8h harm-curve identities failed (xenv/v3_icu_harm_checks.json)."
_v3pc = (_v3boot.get("point_check") or {}).get("match")
REV1_OK = (None if not _xv3_cells_on else
           bool(_xv3_env_ok and len(XRES_V3_GATE) and _v3drv.get("draws_complete") and _v3done.get("cells")
                and _v3chk.get("passes")))
REV2_OK = (None if not (XV3_SWITCH_STATE.get("xenv_v3_bootstrap") and _v3boot_ran) else
           bool(_v3boot.get("draws_complete") and _v3done.get("bootstrap") and _v3chk.get("passes")))
REV3_OK = None if not XV3_SWITCH_STATE.get("xenv_v3_icu_harm") else bool(_v3harm.get("passes") is True)
XV3GATE = {"switches": XV3_SWITCH_STATE, "environment_checks_ok": _xv3_env_ok, "result_rows": int(len(XRES_V3_GATE)),
           "code_hash": globals().get("XENV_V3_CODE_HASH"), "draws_complete": _v3drv.get("draws_complete"),
           "draws_per_cell": _v3drv.get("draws_per_cell"),
           "improper_rows": int(XRES_V3_GATE["status"].ne("ok").sum()) if len(XRES_V3_GATE) else 0,
           "stats_done": _v3done, "theorem_counts": _v3chk.get("theorem_counts"), "stats_passes": _v3chk.get("passes"),
           "bootstrap": {k: _v3boot.get(k) for k in ("status", "cells", "cells_skipped", "draws", "draws_complete",
                                                     "fallback_pairs_total", "draws_with_trap_replicate",
                                                     "max_width_minus_bound", "code_hash")},
           "bootstrap_policies_match_xresults": _v3pc,
           "icu_harm": {k: _v3harm.get(k) for k in ("status", "passes", "part1_policies", "part2_policies",
                                                    "endpoints_part1_vs_RESULTS", "endpoints_part2_vs_XRESULTS")},
           "rev_checks": {"REV-1": REV1_OK, "REV-2": REV2_OK, "REV-3": REV3_OK}}
print("Revision gate (II.8):", json.dumps({k: XV3GATE[k] for k in ("switches", "result_rows", "draws_complete",
                                                                     "stats_passes", "rev_checks")},
                                          default=_json_default))

# ---------- Second-revision gate (Section II.9): invariants, completion of enabled switches, clean-check pass ----------
XV4_SWITCH_STATE = {k: bool(EXPERIMENTS.get(k, False)) for k in globals().get("XV4_SWITCHES", ())}
# closure row -> (II.9 subsection, switch, summary global)
XV4_REV_ROWS = {"REV-4": ("II.9b", "xenv_v4_revision_robust", "XV4_ROBUST_SUMMARY"),
                "REV-5": ("II.9c", "xenv_v4_confidence", "XV4_CS_SUMMARY"),
                "REV-6": ("II.9d", "xenv_v4_bootstrap", "XV4_BOOT_SUMMARY"),
                "REV-7": ("II.9e", "xenv_v4_sensitivity", "XV4_SENS_SUMMARY"),
                "REV-8": ("II.9f", "xenv_v4_lift", "XV4_LIFT_SUMMARY"),
                "REV-9": ("II.9g", "xenv_v4_clean_check", "XV4_CLEAN")}
_v4st = globals().get("XV4_STATS") or {}
_v4sub = _v4st.get("subsections") or {}
_v4sum = {rev: globals().get(g) or {} for rev, (_, _, g) in XV4_REV_ROWS.items()}
_v4bad = {k: v.get("invariants") for k, v in _v4sub.items() if v.get("invariants_pass") is False}
assert not _v4bad, f"II.9 invariant checks failed: {_v4bad} - see xenv/v4_stats_checks.json"
assert not any((_v4st.get("checks") or {}).get("order_counts", {}).values()), \
    f"II.9h recounted orderings failed: {_v4st['checks']['order_counts']} - see xenv/v4_stats_checks.json"
if _v4sum["REV-5"].get("status") == "run":
    assert ((_v4sum["REV-5"].get("checks") or {}).get("max_lower_minus_upper") or 0.0) <= 1e-9, \
        "II.9c confidence-set bounds out of order (L > U_free) - see xenv/v4_cs_summary.json."
if _v4sum["REV-6"].get("status") == "run":
    assert (_v4sum["REV-6"].get("max_width_minus_bound") or 0.0) <= 1e-9 \
        and (_v4sum["REV-6"].get("max_rule_p_difference") or 0.0) <= 1e-9, \
        "II.9d bootstrap invariants (Theorem 1 part iii on every replicate) failed - see xenv/v4_boot_summary.json."
if _v4sum["REV-7"].get("status") == "run":
    assert (_v4sum["REV-7"].get("checks") or {}).get("passes") is not False, \
        "II.9e sensitivity-cell checks (self-test, theorem counts) failed - see xenv/v4_sens_summary.json."


def xv4_rev_flag(rev: str) -> Optional[bool]:
    """Closure flag of one II.9 subsection: None when its switch is off (II.9g also in artifact-replay mode);
    otherwise status 'run', draws complete and invariants not failed in II.9h, and for II.9g the clean-check pass
    flag. Coverage and error rates never enter."""
    sub, switch, _ = XV4_REV_ROWS[rev]
    if not XV4_SWITCH_STATE.get(switch) or (sub == "II.9g" and not ALLOW_TRAINING):
        return None
    s, info = _v4sum[rev], _v4sub.get(sub) or {}
    ok = (s.get("status") == "run" and s.get("draws_complete") is True and info.get("ran") is True
          and info.get("invariants_pass") is not False)
    return bool(ok and (sub != "II.9g" or s.get("pass") is True))


XV4_REV = {rev: xv4_rev_flag(rev) for rev in XV4_REV_ROWS}
XV4GATE = {"switches": XV4_SWITCH_STATE, "stats_status": (_v4st.get("overview") or {}).get("status"),
           "stats_passes": (_v4st.get("checks") or {}).get("passes"),
           "subsections": {sub: {k: (_v4sub.get(sub) or {}).get(k) for k in ("enabled", "status", "ran",
                                                                            "draws_complete", "invariants_pass",
                                                                            "code_hash")}
                           for sub, _, _ in XV4_REV_ROWS.values()},
           "clean_check": {k: _v4sum["REV-9"].get(k) for k in ("status", "pass", "bitwise_identical", "criteria",
                                                                "failing_draws_total")},
           "stats_code_hash": globals().get("XV4_STATS_CODE_HASH"), "rev_checks": XV4_REV}
print("Second-revision gate (II.9):", json.dumps({k: XV4GATE[k] for k in ("switches", "stats_passes", "rev_checks")},
                                                 default=_json_default))

legacy_ok = LEGACY.get("max_abs_discrepancy")
repro_ok = (REPRO_TESTS.get("critic_training", {}).get("passes_declared_tolerance") and
            REPRO_TESTS.get("diffusion_training", {}).get("passes_declared_tolerance")) if REPRO_TESTS else None


def status(flag, yes="check passed", no="check FAILED - inspect evidence", none="not run (experiment switched off)"):
    return none if flag is None else (yes if flag else no)


CLOSURE = [
    ("C01", "Mean-equivalent projection + masked BC/IQL + uniform-admissible", status(PROJECTION_IDENTITY_VERIFIED), "projection_equivalence_checks.csv; table_main_policy_comparison.csv; legacy_projection_controls_summary.csv"),
    ("C02", "Hard AM-CQL repaired designs, TD-row/loss/alpha/placement ablations, diagnostics",
     status(CRITIC_STABILITY_GATE.get("stable_on_all_seeds") if CRITIC_STABILITY_GATE else None,
            "primary critic empirically stable on all main seeds", "primary critic NOT stable on all seeds - see gate"),
     "critic_stability_gate.json; critic_stability_summary*.csv; table_constraint_placement.csv; critic_diagnostics.csv"),
    ("C03", "Seeding, RNG streams, run identity, resumable checkpoints, cache invalidation", status(repro_ok), "reproducibility_tests.json"),
    ("C04", "All tables generated here; replay; release with code and hashes",
     status(REPLAY_SUMMARY.get("cross_session_ok"),
            "cross-session replay matches the previous session",
            "cross-session differences (expected after FORCE_RETRAIN / code change) - inspect",
            none="no previous session in this ROOT: only the in-session file round-trip was checked (re-run with ALLOW_TRAINING=False for a replay)"),
     "cross_session_replay.csv; policy_reevaluation.csv; checkpoint_replay_cpu.csv; release/"),
    ("M01", "Corrected admissible support coverage", status(0.0 <= SUPPORT_AUDIT["admissible_pair_coverage"] <= 1.0), "corrected_support_audit.json"),
    ("M02", "Released vs estimated support; threshold verified", status(PROVENANCE["checks"].get("threshold_rule_count_le_threshold_removed") if isinstance(PROVENANCE["checks"].get("threshold_rule_count_le_threshold_removed"), bool) else None, none="offline: verify threshold manually"), "benchmark_construction_provenance.json; support_estimation_*.csv"),
    ("M03", "Prereg, two-sided tests, Holm over distinct hypotheses, hierarchical analysis", status("primary" in STATS, "analyses produced"), "statistics_*.csv/json; replication_planning.csv"),
    ("M04", "Selection protocol with FQE validation and locked fresh evaluation", status(bool(SELECTION) if EXPERIMENTS["selection_protocol"] else None, "protocol run"), "selection_*.csv; tuning_budget.csv"),
    ("M05", "Soft training vs extraction alignment", status(bool(len(SOFT_ANALYSIS)) if EXPERIMENTS["main_neural_critics"] else None, "analysis produced"), "soft_constraint_analysis.csv"),
    ("M06", "Direct categorical alternatives, placement ablations, cost", status(bool(len(TABLE6_PRIMARY)) if EXPERIMENTS["main_diffusion"] else None, "analysis produced"), "table6_*.csv; cost_accounting.csv"),
    ("M07", "Replication, regimes, stress tests, subgroups", status(bool(len(SUBGROUPS)), "analysis produced"), "replication_results.csv; statistics_hierarchical_replication.csv; data_regime_size_summary.csv; stress_*.csv; subgroup_diagnostics.csv"),
    ("M08", "Feature provenance/timing table and representation ablation", status(True, "table produced; treatment columns define the action and are clustering features"), "feature_provenance_timing.csv; representation_ablation.csv"),
    ("M09", "Strict evaluator, invariants, env semantics, Monte Carlo", status(bool(ENV_SEMANTICS.get("terminate_all_consistent", True)) and all(MC_CHECK["within_4_se"]) if EXPERIMENTS["monte_carlo_crosscheck"] and len(MC_ROWS) else bool(EVALUATOR_TOY_TESTS)), "numerical_audit.json; monte_carlo_crosscheck.csv"),
    ("M10", "Episode semantics + short-cutoff test", status(CUTOFF_TEST["cutoff_episodes"] > 0), "dataset_manifest_main.json; dataset_manifests_all.csv"),
    ("M11", "Manuscript rewrite and consistency pass", "manuscript-only (see Section 35)", "-"),
    ("M12", "Related work around support-constrained methods", "manuscript-only (tabular BCQ baseline added here)", "-"),
    ("m01-m03", "Citations, merged feature sections, notation/precision", "manuscript-only", "-"),
    ("m04", "Pearson vs Spearman labelled", "closed in code", "benchmark_audit.json; eda_sofa_admissibility_association.csv"),
    ("m05", "Pinned benchmark, recorded runtime, version-safe boxplot", "closed in code", "release/environment_lock.txt"),
    ("m06", "Benchmark-support terminology", "closed in code", "-"),
    ("m07", "Execution metadata and logging", "closed in code", "logs/execution_log.jsonl; logs/runtime_*.json"),
    ("m08", "Venue packaging / authorship", "manuscript-only", "-"),
    ("Legacy", "Original artifacts replayed",
     status(((legacy_ok is not None and legacy_ok <= 1e-10) if (EXPERIMENTS["legacy_artifact_replay"] and LEGACY.get("found")
             and (LEGACY_ROOT / "tables" / "exact_results_all.csv").exists()) else None),
            none="not run (switched off, original directory or its exact_results_all.csv not found)"), "legacy_replay.json"),
]

# ---------- Main-track closure rows (Part II and statistics at scale) ----------
_xfam = set(XRES_GATE["family"]) if len(XRES_GATE) else set()
_ground_truth_on = [f for f in ("gumbel", "garnet") if XENV_SWITCH_STATE.get(f"xenv_{f}")]
_gumbel_prov_bad = str(GUMBEL_PROVENANCE_STATUS or "").startswith(("MISMATCH", "MISSING"))
MT1_OK = (None if not _ground_truth_on else bool(set(_ground_truth_on) <= _xfam and ICU_ADAPTER_OK is not False
                                                  and (GUMBEL_EXACT_OK is True or "gumbel" not in _ground_truth_on)
                                                  and not _gumbel_prov_bad))
XENV_REQUIRED_METHOD_FAMILIES = {"behaviour", "value", "pessimistic", "constrained", "guided-prior", "diffusion"}
_xmf = set(XRES_GATE["method_family"].dropna()) if len(XRES_GATE) else set()
MT3_OK = None if not XENV_RUN_FAMILIES else bool(XENV_REQUIRED_METHOD_FAMILIES <= _xmf)
_fidelity_files = [XENV_TABLE_DIR / n for n in ("xenv_prereg_tests.json", "table_rank_fidelity.csv", "table_error_decomposition.csv")]
_xst = globals().get("XSTATS") or {}
_xpt = _xst.get("prereg_tests") or {}
_rank_fid = _xst.get("rank_fidelity")
MT4_OK = None if not _ground_truth_on else bool(
    all(f_.exists() for f_ in _fidelity_files)                      # written by this session's II.7 (xs_run)
    and (not XENV_SWITCH_STATE.get("xenv_gumbel") or (_xpt.get("H1") or {}).get("status") == "run")
    and (not XENV_SWITCH_STATE.get("xenv_garnet") or (_xpt.get("H3") or {}).get("status") == "run")
    and _rank_fid is not None and len(_rank_fid) > 0)
_primary_n = (STATS.get("primary") or {}).get("n_pairs")
_hier_n = int(HIERARCHICAL["n_datasets"].min()) if len(HIERARCHICAL) else None
MT5_CHECKS = {
    "paper_scale_design": bool(RUN_PROFILE["paper_scale"] and len(CFG.seeds) >= 10
                               and len(CFG.replication_dataset_seeds) >= 8 and len(CFG.replication_train_seeds) >= 2),
    "primary_pairs_ge_10": None if _primary_n is None else bool(_primary_n >= 10),
    "replication_datasets_ge_8": None if _hier_n is None else bool(_hier_n >= 8),
    "part2_draws_complete": XDRAWS_COMPLETE,
    "all_part2_families_run": {"icu", "gumbel", "garnet"} <= set(XENV_RUN_FAMILIES),
    "all_cells_present": bool(set(str(c_["cell"]) for c_ in (globals().get("X_CELLS") or []))
                              <= set(XDRAWS["cell"].astype(str))),
    "part2_design_as_preregistered": (("design_deviations" in _xpt and not _xpt["design_deviations"])
                                      if XENV_RUN_FAMILIES else None),   # a missing report does not establish it
}
MT5_OK = all(v is True for v in MT5_CHECKS.values())   # a component that did not run does not establish scale
_mt5_text = (f"{len(CFG.seeds)} main seeds (primary pairs: {_primary_n}); {len(CFG.replication_dataset_seeds)} x "
             f"{len(CFG.replication_train_seeds)} replication (estimation; datasets analysed: {_hier_n}); Part II draws "
             f"complete: {XDRAWS_COMPLETE}; all Part II families run: {MT5_CHECKS['all_part2_families_run']}; all cells "
             f"present: {MT5_CHECKS['all_cells_present']}; Part II design as preregistered: "
             f"{MT5_CHECKS['part2_design_as_preregistered']}; profile={COMPUTE_PROFILE}, mode={RUN_MODE}")
CLOSURE += [
    ("MT-1", "Multi-environment ground truth: exact Gumbel-max simulator + Garnet harm family + ICU-Sepsis adapter",
     status(MT1_OK, f"environments built, adapter and exactness checks passed; Gumbel provenance: {GUMBEL_PROVENANCE_STATUS}",
            f"missing environment, failed check or Gumbel provenance {GUMBEL_PROVENANCE_STATUS} - inspect"),
     "xenv/icu_adapter_check.json; xenv/gumbel_exactness_check.json; xenv/xenv_environment_suite.csv; data/xenv_gumbel_true_*.npz"),
    ("MT-2", "Propositions P1-P7 verified numerically (+ exact constrained frontiers)",
     status(PROPOSITIONS_STATUS if XENV_SWITCH_STATE["xenv_theory"] else None, "all proposition checks passed"),
     "xenv/proposition_checks.csv; xenv/proposition_summary.csv; xenv/exact_frontiers.csv"),
    ("MT-3", "Baseline suite: value, pessimistic (VI-LCB, MBS-QI, SPIBB), constrained (Lagrangian, CMDP-LP), guided-prior (SfBC, IDQL, FISOR-style), exact diffusion",
     status(MT3_OK, "all method families evaluated", f"method families missing: {sorted(XENV_REQUIRED_METHOD_FAMILIES - _xmf)}"),
     "xenv/xenv_method_catalog.csv; xenv/table_xenv_main.csv; xenv/xenv_h2_family.csv"),
    ("MT-4", "Benchmark fidelity H1-H3: rank fidelity, error decomposition, harm moderation",
     status(MT4_OK, "analyses produced (H1 / H3 run, rank-fidelity rows present)",
            "fidelity outputs missing or H1 / H3 not run - inspect Section II.7"),
     "xenv/table_rank_fidelity.csv; xenv/table_error_decomposition.csv; xenv/xenv_prereg_tests.json"),
    ("MT-5", "Statistics at scale: 10 main seeds, 8 x 2 replication (estimation, not testing), configured Part II draws",
     status(MT5_OK, "paper scale: " + _mt5_text, "NOT paper scale: " + _mt5_text),
     "run_profile.json; statistics_primary.json; statistics_hierarchical_replication.csv; xenv/xenv_draw_meta.csv"),
]
# ---------- Revision closure rows (Section II.8) ----------
_rev2_none = ("not run (xenv_v3_bootstrap off)" if not XV3_SWITCH_STATE.get("xenv_v3_bootstrap")
              else f"not run: {_v3boot.get('status', 'no summary')}")
CLOSURE += [
    ("REV-1", "Revision cells (II.8): confounded and antibiotic-withholding Gumbel physicians, aggregated-state "
              "benchmarks with diabetes hidden, Garnet with random unsupported rows; bracketing coverage, certified "
              "comparisons and rank fidelity on them",
     status(REV1_OK, f"cells run, draws complete, environment and theorem checks passed "
                     f"({XV3GATE['result_rows']} rows)",
            "II.8 draws incomplete, analyses missing or a check failed - inspect xenv/v3_driver_summary.json and "
            "xenv/v3_stats_checks.json",
            none="not run (II.8 cell switches off)"),
     "xenv/v3_environment_suite.csv; xenv/v3_results_all.csv.gz; xenv/v3_bracketing.csv; xenv/v3_certified.csv; "
     "xenv/v3_rank_fidelity.csv; figures/xenv/fig_v3_bracketing.*"),
    ("REV-2", "Estimation-aware intervals: bootstrap ensembles of the hospital data (B = "
              f"{XV3BOOT.replicates if 'XV3BOOT' in globals() else '?'}) on II.6 cells; coverage and certified comparisons",
     status(REV2_OK, f"bootstrap run on {len(_v3boot.get('cells') or [])} cells, draws complete, invariants hold; "
                     f"regenerated policies match XRESULTS: {_v3pc}",
            "bootstrap draws incomplete or analyses missing - inspect xenv/v3_bootstrap_summary.json",
            none=_rev2_none),
     "xenv/v3_bootstrap_results.csv.gz; xenv/v3_bootstrap_summary.json; xenv/v3_bootstrap_coverage.csv; "
     "xenv/v3_bootstrap_intervals.csv; figures/xenv/fig_v3_bootstrap.*"),
    ("REV-3", "ICU-Sepsis harm curves J(lambda) on the released rows (Part I main + reference policies, seed 0; Part II "
              "ICU draw-0 default policies): ranking switches and certified share",
     status(REV3_OK, "curves produced; endpoint, bracketing, monotonicity and certified-order identities hold",
            "a harm-curve identity failed - inspect xenv/v3_icu_harm_checks.json",
            none="not run (xenv_v3_icu_harm off)"),
     "xenv/v3_icu_harm_curves.csv; xenv/v3_icu_harm_summary.csv; xenv/v3_icu_harm_checks.json; figures/xenv/fig_v3_icu_harm.*"),
]
# ---------- Second-revision closure rows (Section II.9) ----------


def _xv4_none(rev: str) -> str:
    sub, switch, _ = XV4_REV_ROWS[rev]
    if sub == "II.9g" and XV4_SWITCH_STATE.get(switch) and not ALLOW_TRAINING:
        return "not run (artifact replay: ALLOW_TRAINING is False)"
    return f"not run ({switch} off)"


def _xv4_fail(rev: str, file: str) -> str:
    sub = XV4_REV_ROWS[rev][0]
    st = (_v4sub.get(sub) or {}).get("status", _v4sum[rev].get("status", "no summary"))
    return (f"{sub} status '{st}', draws incomplete or a check failed - inspect xenv/{file} and "
            "xenv/v4_stats_checks.json")


CLOSURE += [
    ("REV-4", "Certificates without NHB (Corollary 2), pair types, distinct policies, abstention at matched coverage, "
              "hidden-benefit sensitivity kappa and the completion-sensitive headline on every Part II cell; II.8g "
              "intervals over all default policies",
     status(XV4_REV["REV-4"], "analysis run on every stored draw; theorem and identity checks passed",
            _xv4_fail("REV-4", "v4_robust_summary.json"), none=_xv4_none("REV-4")),
     "xenv/v4_robust_summary.csv; xenv/v4_robust_kappa.csv; xenv/v4_robust_gapbins.csv; xenv/v4_v3boot_allpolicy.csv; "
     "xenv/v4_robust_summary.json; xenv/v4_overview.json"),
    ("REV-5", "Confidence-set certificates on the true state space: L1 sets for the true rows and d0 from the raw "
              "hospital counts, valid for all claims of a draw at once at scale 1",
     status(XV4_REV["REV-5"], "draws complete; inner-problem self-test and bound order passed",
            _xv4_fail("REV-5", "v4_cs_summary.json"), none=_xv4_none("REV-5")),
     "xenv/v4_cs_summary.csv; xenv/v4_cs_draws.csv; xenv/v4_cs_policies.csv.gz; xenv/v4_cs_summary.json; "
     "figures/xenv/fig_v4_confidence.*"),
    ("REV-6", "Bootstrap with B = 200 that re-estimates the whole construction in every replicate, beside the fixed "
              "mask; coverage over all default policies",
     status(XV4_REV["REV-6"], "draws complete; per-replicate invariants and unit-weight identities hold",
            _xv4_fail("REV-6", "v4_boot_summary.json"), none=_xv4_none("REV-6")),
     "xenv/v4_boot_coverage.csv; xenv/v4_boot_draws.csv; xenv/v4_boot_intervals.csv.gz; xenv/v4_boot_summary.json; "
     "figures/xenv/fig_v4_bootstrap.*"),
    ("REV-7", "Construction sensitivity: Gumbel timeout hazard, the argmax dead-state rule on the same hospital data, "
              "softer physicians (wider support) and other state abstractions, each paired with its base cell",
     status(XV4_REV["REV-7"], "cells run, draws complete; self-test, draw checks and theorem counts passed",
            _xv4_fail("REV-7", "v4_sens_summary.json"), none=_xv4_none("REV-7")),
     "xenv/v4_sens_summary.csv; xenv/v4_sens_certified_draws.csv; xenv/v4_sens_rank_fidelity.csv; "
     "xenv/v4_sens_results_all.csv.gz; xenv/v4_sens_summary.json"),
    ("REV-8", "Lift sensitivity: physician and uniform lifts beside the fallback lift, and the range of J_true over "
              "all lifts",
     status(XV4_REV["REV-8"], "draws complete; Theorem 1 and range invariants hold under every lift",
            _xv4_fail("REV-8", "v4_lift_summary.json"), none=_xv4_none("REV-8")),
     "xenv/v4_lift_summary.csv; xenv/v4_lift_results.csv.gz; xenv/v4_lift_summary.json"),
    ("REV-9", "Clean rerun of the Part II draws into an empty cache, compared row by row with the cached results",
     status(XV4_REV["REV-9"], "rerun matches the cache (max |diff| <= clean_tol, identical statuses and certified "
                              f"counts, no missing row); bitwise identical: {_v4sum['REV-9'].get('bitwise_identical')}",
            _xv4_fail("REV-9", "v4_clean_check.json") + f" (pass flag: {_v4sum['REV-9'].get('pass')})",
            none=_xv4_none("REV-9")),
     "xenv/v4_clean_check.json; xenv/v4_clean_diffs.csv"),
]
CLOSURE_DF = pd.DataFrame(CLOSURE, columns=["audit_id", "item", "status", "evidence"])
CLOSURE_DF.to_csv(TABLE_DIR / "audit_closure_checklist.csv", index=False)
display(CLOSURE_DF)

manifest = {
    "implementation": "revised-audit-complete",
    "run_mode": RUN_MODE,
    "root": str(ROOT),
    "benchmark": BENCH.identity,
    "dataset": OFFLINE_MANIFEST["file"],
    "dataset_content_sha256": OFFLINE_MANIFEST["content_sha256"],
    "policies_registered": len(REGISTRY),
    "policy_seed_pairs_evaluated": int(RESULTS[["name", "seed"]].drop_duplicates().shape[0]),
    "seeds": list(CFG.seeds),
    "compute_profile": COMPUTE_PROFILE,
    "run_profile": RUN_PROFILE,
    "compute_logged": COMPUTE_TOTALS,
    "runtime": RUNTIME_INFO,
    "experiments": EXPERIMENTS,
    "prereg_hash": obj_hash(PREREG),
    "checkpoints_written_this_session": int(CHECKPOINT_META["written_this_session"].sum()) if len(CHECKPOINT_META) else 0,
    "checkpoints_loaded_from_earlier_sessions": int(CHECKPOINT_META["loaded_from_earlier_session"].sum()) if len(CHECKPOINT_META) else 0,
    "xenv": {
        **XGATE,
        "config": asdict(XCFG) if "XCFG" in globals() else None,
        "prereg_hash": obj_hash(PREREG_PART2) if "PREREG_PART2" in globals() else None,
        "code_hash": globals().get("XENV_CODE_HASH"),
        "cells": int(XRES_GATE["cell"].nunique()) if len(XRES_GATE) else 0,
        "cell_draws_evaluated": int(XRES_GATE[["cell", "draw"]].drop_duplicates().shape[0]) if len(XRES_GATE) else 0,
        "xcache_written_this_session": len(XCACHE.written) if "XCACHE" in globals() else 0,
        "xcache_loaded_from_earlier_sessions": len(XCACHE.loaded) if "XCACHE" in globals() else 0,
        "mt_checks": {"MT-1": MT1_OK, "MT-2": PROPOSITIONS_STATUS, "MT-3": MT3_OK, "MT-4": MT4_OK, "MT-5": MT5_OK,
                      "MT-5 components": MT5_CHECKS},
        "v3": XV3GATE,
        "v4": XV4GATE,
    },
    "finished_utc": utc_now(),
}
save_json(manifest, TABLE_DIR / "fresh_reproducibility_manifest.json")
print("All structural integrity checks passed. Results root:", ROOT)

# %%
# ============================================================
# 34b. Code re-export, release inventory and archive (after the gate, so the closure checklist, the manifest and
#     the executed gate cells are included)
# ============================================================
if EXPERIMENTS["release_package"]:
    # Re-export the code snapshot now that the gate and closure cells have run, and re-hash it.
    export_release_code(code_dir)
    inputs["release_code_sha256"] = {p_.name: sha256_file(p_) for p_ in code_dir.glob("*") if p_.is_file()}
    save_json(inputs, TABLE_DIR / "input_sha256.json")
    save_json(inputs, RELEASE_DIR / "input_sha256.json")
    inv_path = RELEASE_DIR / "artifact_inventory.csv"
    for old_zip in RELEASE_DIR.glob("release_*.zip"):
        old_zip.unlink()
    inventory = []
    for sub in ["tables", "figures", "logs", "policies", "release", "data", "checkpoints", "xenv_cache"]:
        if not (ROOT / sub).exists():
            continue
        for f in sorted((ROOT / sub).rglob("*")):
            if not f.is_file() or f == inv_path or f.name.endswith(".zip") or f.name.startswith("resume_") or f.name.endswith(".tmp"):
                continue
            if sub == "checkpoints" and not RELEASE_INCLUDE_WEIGHTS and f.suffix == ".pt":
                continue
            if sub == "data" and not RELEASE_INCLUDE_DATASET and f.name.startswith("offline_") and f.suffix == ".npz":
                continue
            inventory.append({"path": str(f.relative_to(ROOT)), "bytes": f.stat().st_size, "sha256": sha256_file(f)})
    INVENTORY = pd.DataFrame(inventory)
    INVENTORY.to_csv(inv_path, index=False)
    zip_path = RELEASE_DIR / f"release_{RUN_MODE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for rec in inventory:
            zf.write(ROOT / rec["path"], arcname=rec["path"])
        zf.write(inv_path, arcname="release/artifact_inventory.csv")
    print(f"Release archive: {zip_path} ({zip_path.stat().st_size / 2 ** 20:.1f} MB, {len(inventory)} files)")
