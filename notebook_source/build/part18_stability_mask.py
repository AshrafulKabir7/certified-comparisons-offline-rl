# %% [markdown]
# ## 23. Hard-critic stability study (audit C02)
#
# Equal training budget (same steps, optimiser, batch, minibatch stream and initial weights) on the ablation
# seeds. Compared: legacy objective (failure-mode control), design A, design B (both normalisations), retaining
# vs excluding inadmissible logged transitions from TD regression, squared vs Huber TD, conservative weight
# α ∈ {0.3, 1, 3}, a separately labelled TD-target clipping ablation of the legacy objective (stabilisation only,
# not a repair), and — only if enabled — a separately labelled discounted (γ = 0.99) estimand.
#
# The "empirically stable" flag is a declared diagnostic criterion (final max |Q| over admissible actions ≤ 5,
# growth of that quantity over the last third of training ≤ 1.5×, Bellman RMSE ≤ 1 on the rows of the TD
# objective — all logged rows, or admissible-logged rows only for the TD-admissible ablations; the full-buffer RMSE
# and the admissible/inadmissible-row split are reported alongside); it is an
# empirical statement about these runs, not a convergence guarantee.

# %%
# ============================================================
# 23. Critic stability ablations
# ============================================================
EXECLOG.mark("23. Critic stability ablations (C02)")
R = dataclasses.replace
ABLATION_SPECS = [
    R(SPEC_HARD_B, key="hard_admrows_batchnorm", display="AM-CQL (Hard, B) [batch-normalised regulariser]", reg_normalization="batch", status="ablation"),
    R(SPEC_HARD_B, key="hard_admrows_tdadm", display="AM-CQL (Hard, B) [TD on admissible-logged rows only]", td_rows="admissible_only", status="ablation"),
    R(SPEC_HARD_A, key="hard_fullcql_tdadm", display="AM-CQL (Hard, A) [TD on admissible-logged rows only]", td_rows="admissible_only", status="ablation"),
    R(SPEC_HARD_B, key="hard_admrows_mse", display="AM-CQL (Hard, B) [squared TD]", td_loss="mse", status="ablation"),
    R(SPEC_HARD_A, key="hard_fullcql_mse", display="AM-CQL (Hard, A) [squared TD]", td_loss="mse", status="ablation"),
    R(SPEC_HARD_B, key="hard_admrows_alpha0.3", display="AM-CQL (Hard, B) [alpha=0.3]", alpha=0.3, status="ablation"),
    R(SPEC_HARD_B, key="hard_admrows_alpha3", display="AM-CQL (Hard, B) [alpha=3]", alpha=3.0, status="ablation"),
    R(SPEC_HARD_LEGACY, key="hard_legacy_mse", display="AM-CQL (Hard, legacy) [squared TD]", td_loss="mse", status="failure-mode ablation"),
    R(SPEC_HARD_LEGACY, key="hard_legacy_tdadm", display="AM-CQL (Hard, legacy) [TD on admissible-logged rows only]", td_rows="admissible_only", status="failure-mode ablation"),
    R(SPEC_HARD_LEGACY, key="hard_legacy_clip01", display="AM-CQL (Hard, legacy) [TD target clipped to [0,1]: stabilisation ablation, not a repair]", target_clip=(0.0, 1.0), status="stabilisation ablation"),
    R(SPEC_HARD_B, key="hard_admrows_regonly", display="AM-CQL (Hard, B-regulariser only: unmasked backup)", masked_backup=False, masked_extraction=False, status="placement ablation"),
]
if EXPERIMENTS["include_discounted_ablation"]:
    ABLATION_SPECS.append(R(SPEC_HARD_B, key="hard_admrows_gamma0.99",
                            display="AM-CQL (Hard, B) [gamma=0.99: discounted-survival estimand]", gamma=0.99,
                            status="different estimand"))

Q_STABLE_BOUND = 5.0


def stability_row(res: Dict[str, Any]) -> dict:
    diag, log = res["diag"], res["log"]
    final = diag.iloc[-1]
    steps = res["spec"].steps
    early = diag[diag["step"] <= int(round(2 * steps / 3))]
    ref = float(early["q_adm_max_abs"].iloc[-1]) if len(early) else float(diag["q_adm_max_abs"].iloc[0])
    growth = float(final["q_adm_max_abs"]) / max(ref, 1e-12)
    row = {"variant": res["spec"].key, "display": res["spec"].display, "status": res["spec"].status,
           "seed": res["seed"], "seed_set": "main" if res["spec"].key in SPEC_BY_KEY else "ablation",
           "final_q_adm_max_abs": float(final["q_adm_max_abs"]),
           "final_q_adm_q01": float(final["q_adm_q01"]), "final_q_adm_q99": float(final["q_adm_q99"]),
           "growth_last_third": growth,
           "final_bellman_rmse": float(final.get("bellman_residual_rmse_td_objective_rows", final["bellman_residual_rmse_full_buffer"])),
           "final_bellman_rmse_full_buffer": float(final["bellman_residual_rmse_full_buffer"]),
           "final_bellman_rmse_adm_logged_rows": float(final.get("bellman_residual_rmse_adm_logged_rows", np.nan)),
           "final_bellman_rmse_inadm_logged_rows": float(final.get("bellman_residual_rmse_inadm_logged_rows", np.nan)),
           "final_mean_state_spread": float(final["mean_state_spread"]),
           "final_offset_to_spread_ratio": float(final["median_offset_to_spread_ratio"]),
           "states_spread_within_64_ulp": int(final["states_spread_within_64_ulp_of_offset"]),
           "final_grad_norm_mean_last5": float(log["grad_norm"].tail(5).mean()),
           "final_td_loss": float(log["td_loss"].iloc[-1]), "final_cql_loss": float(log["cql_loss"].iloc[-1])}
    row["empirically_stable"] = bool(row["final_q_adm_max_abs"] <= Q_STABLE_BOUND and growth <= 1.5
                                     and row["final_bellman_rmse"] <= 1.0)
    return row


if EXPERIMENTS["critic_stability_ablations"] and EXPERIMENTS["main_neural_critics"]:
    run_critics(ABLATION_SPECS, MAIN, CFG.ablation_seeds, group="critic_ablation")
    refresh_results()
_abl_keys = {x.key for x in ABLATION_SPECS}
stab_rows = [stability_row(r) for (k, b, s), r in CRITICS.items() if b == "main" and len(r["diag"])
             and ((k in SPEC_BY_KEY and s in CFG.seeds) or (k in _abl_keys and s in CFG.ablation_seeds))]
CRITIC_STABILITY = pd.DataFrame(stab_rows)
if len(CRITIC_STABILITY):
    ret = RESULTS[(RESULTS["rule"] == "terminate") & (RESULTS["status"] == "ok")][["name", "seed", "return", "inadmissible_rate"]]
    CRITIC_STABILITY = CRITIC_STABILITY.merge(ret.rename(columns={"name": "display", "return": "terminate_return_native",
                                                                  "inadmissible_rate": "native_inadmissible_rate"}),
                                              on=["display", "seed"], how="left")
    CRITIC_STABILITY.to_csv(TABLE_DIR / "critic_stability_ablation.csv", index=False)
    STAB_MAIN_ALL = CRITIC_STABILITY[CRITIC_STABILITY["seed_set"] == "main"].groupby(["variant", "display"]).agg(
        n=("seed", "nunique"), final_q_adm_max_abs=("final_q_adm_max_abs", "median"),
        bellman_rmse=("final_bellman_rmse", "median"), stable_fraction=("empirically_stable", "mean")).reset_index()
    STAB_MAIN_ALL.to_csv(TABLE_DIR / "critic_stability_summary_main_all_seeds.csv", index=False)
    _prim = CRITIC_STABILITY[CRITIC_STABILITY["variant"] == PREREG["primary_hard_critic"]]
    CRITIC_STABILITY_GATE = {"primary_critic": PREREG["primary_hard_critic"], "seeds": sorted(_prim["seed"].tolist()),
                             "stable_on_all_seeds": bool(len(_prim) and _prim["empirically_stable"].all()),
                             "criterion": f"max|Q_adm| <= {Q_STABLE_BOUND}, growth <= 1.5x, TD-objective Bellman RMSE <= 1"}
    save_json(CRITIC_STABILITY_GATE, TABLE_DIR / "critic_stability_gate.json")
    print("Primary critic stability gate:", CRITIC_STABILITY_GATE)
    STAB_SUMMARY = CRITIC_STABILITY[CRITIC_STABILITY["seed"].isin(CFG.ablation_seeds)].groupby(["variant", "display", "status"]).agg(
        n=("seed", "nunique"), final_q_adm_max_abs=("final_q_adm_max_abs", "median"),
        growth_last_third=("growth_last_third", "median"), bellman_rmse=("final_bellman_rmse", "median"),
        stable_fraction=("empirically_stable", "mean"), terminate_return=("terminate_return_native", "mean"),
        terminate_return_sd=("terminate_return_native", "std")).reset_index()
    STAB_SUMMARY.to_csv(TABLE_DIR / "critic_stability_summary.csv", index=False)
    display(STAB_SUMMARY.round(4))
else:
    CRITIC_STABILITY_GATE = {}

# Constraint-placement table (audit §7): extraction, backup and regulariser placement under one architecture.
PLACEMENT_CELLS = [
    (False, False, "full", SPEC_DEEP.display),
    (True, False, "full", SPEC_DEEP.display + " [masked greedy]"),
    (False, True, "full", SPEC_HARD_A.display + " [unmasked greedy]"),
    (True, True, "full", SPEC_HARD_A.display),
    (False, False, "masked_admissible_rows", "AM-CQL (Hard, B-regulariser only: unmasked backup)"),
    (True, False, "masked_admissible_rows", "AM-CQL (Hard, B-regulariser only: unmasked backup) [masked greedy]"),
    (False, True, "masked_admissible_rows", SPEC_HARD_B.display + " [unmasked greedy]"),
    (True, True, "masked_admissible_rows", SPEC_HARD_B.display),
]
_okr = RESULTS[(RESULTS["status"] == "ok") & RESULTS["seed"].isin(CFG.ablation_seeds)]
_pl_rows = []
for ext, bk, reg, nm in PLACEMENT_CELLS:
    sub = _okr[_okr["name"] == nm]
    if sub.empty:
        continue
    row = {"extraction_masked": ext, "backup_masked": bk, "regulariser_domain": reg, "td_rows": "all", "policy": nm,
           "n_seeds": int(sub["seed"].nunique())}
    for rule in RULES:
        r = sub[sub["rule"] == rule]
        row[f"return_{rule}_mean"] = float(r["return"].mean())
        row[f"return_{rule}_sd"] = float(r["return"].std(ddof=1)) if len(r) > 1 else np.nan
        row[f"inadmissible_rate_{rule}"] = float(r["inadmissible_rate"].mean())
    _pl_rows.append(row)
CONSTRAINT_PLACEMENT = pd.DataFrame(_pl_rows)
CONSTRAINT_PLACEMENT.to_csv(TABLE_DIR / "table_constraint_placement.csv", index=False)
if len(CONSTRAINT_PLACEMENT):
    display(CONSTRAINT_PLACEMENT.round(5))

all_diag = [r["diag"].assign(variant=k, seed=s, dataset=b) for (k, b, s), r in CRITICS.items() if len(r["diag"])]
if all_diag:
    pd.concat(all_diag, ignore_index=True).to_csv(TABLE_DIR / "critic_diagnostics.csv", index=False)

# %% [markdown]
# ## 24. Soft AM-CQL: training constraint versus exported policy (audit M05)
#
# The soft penalty acts on a **mean-critic softmax at temperature T_p, under the logged-state minibatch
# distribution**; the legacy export is an **unmasked minimum-critic argmax**. For finite logits the softmax
# always assigns positive probability to inadmissible actions (numerical underflow is not structural zero
# support), and a finite penalty does not guarantee that inadmissible actions rank below admissible ones.
# The table reports, per extraction: surrogate mass under the training distribution, evaluated occupancy rate,
# expected violations and first-violation probability, the primal residual (full logged-distribution surrogate − ε), complementary slackness λ·(g − ε), the dual residual (max |Δλ| over the last 10% of logged steps) and the final λ, the per-state
# distribution of softmax inadmissible mass, mean/min critic disagreement, and greedy margins/ties.
# The original study's statement that the soft variant's zero violations were "achieved by construction" is not supported by these runs.

# %%
# ============================================================
# 24. Soft-constraint alignment analysis
# ============================================================
EXECLOG.mark("24. Soft constraint analysis (M05)")
SOFT_SENS_SPECS = [
    R(SPEC_SOFT, key="soft_eps0.01", display="AM-CQL (Soft) [tolerance eps=0.01]", soft_tolerance=0.01, status="sensitivity"),
    R(SPEC_SOFT, key="soft_T0.5", display="AM-CQL (Soft) [T_p=0.5]", soft_temperature=0.5, status="sensitivity"),
    R(SPEC_SOFT, key="soft_T2", display="AM-CQL (Soft) [T_p=2]", soft_temperature=2.0, status="sensitivity"),
]
if EXPERIMENTS["soft_sensitivity"] and EXPERIMENTS["main_neural_critics"]:
    run_critics(SOFT_SENS_SPECS, MAIN, CFG.ablation_seeds, group="soft_sensitivity")
    add_projection_controls(["soft_sensitivity", "critic_ablation"])
    finalize_projection_checks()
    refresh_results()

soft_rows = []
logged_states = MAIN.data["state"]
for (k, b, seed), res in CRITICS.items():
    spec = res["spec"]
    if b != "main" or not spec.soft:
        continue
    q1, q2 = res["q1"], res["q2"]
    logits = soft_policy_logits(torch.from_numpy(q1), torch.from_numpy(q2), spec.soft_aggregation).numpy() / spec.soft_temperature
    logits -= logits.max(axis=1, keepdims=True)
    sm = np.exp(logits)
    sm /= sm.sum(axis=1, keepdims=True)
    bmass = inadmissible_mass(sm)
    surrogate = float(bmass[logged_states].mean())
    log = res["log"]
    base = dict(variant=k, display=spec.display, seed=seed, tolerance=spec.soft_tolerance, T_p=spec.soft_temperature,
                surrogate_mass_training_distribution=surrogate,
                final_minibatch_surrogate=float(log["inadmissible_mass_surrogate"].iloc[-1]),
                dual_lambda_final=float(res["meta"]["final_lambda"]),
                dual_at_cap=bool(res["meta"]["final_lambda"] >= CFG.soft_lambda_max - 1e-9),
                softmax_state_mass_q50=float(np.quantile(bmass[LIVE], 0.5)),
                softmax_state_mass_q90=float(np.quantile(bmass[LIVE], 0.9)),
                softmax_state_mass_max=float(bmass[LIVE].max()),
                states_softmax_mass_above_1e_minus_6=int((bmass[LIVE] > 1e-6).sum()),
                mean_vs_min_greedy_disagreement=float(np.mean((q1[LIVE] + q2[LIVE]).argmax(1)
                                                              != np.minimum(q1, q2)[LIVE].argmax(1))))
    lam_hist = log["lambda"].to_numpy(dtype=float)
    tail = lam_hist[-max(2, int(math.ceil(0.1 * len(lam_hist)))):]
    base.update(primal_residual_final=surrogate - spec.soft_tolerance,
                primal_residual_last_minibatch=float(log["inadmissible_mass_surrogate"].iloc[-1]) - spec.soft_tolerance,
                complementary_slackness=float(res["meta"]["final_lambda"]) * (surrogate - spec.soft_tolerance),
                dual_residual_max_abs_dlambda_last10pct=float(np.abs(np.diff(tail)).max()) if len(tail) > 1 else np.nan)
    for suffix in ["", "[masked greedy]", "[aligned greedy: mean critic]", "[aligned stochastic: training softmax]",
                   "[aligned stochastic: training softmax] [projected]", "[aligned stochastic: training softmax] [mask-renormalized]"]:
        nm = f"{spec.display}{(' ' + suffix) if suffix else ''}"
        for rule in RULES:
            hit = RESULTS[(RESULTS["name"] == nm) & (RESULTS["seed"] == seed) & (RESULTS["rule"] == rule)]
            if len(hit):
                h = hit.iloc[0]
                soft_rows.append({**base, "extraction": suffix or "legacy: unmasked min-critic greedy", "rule": rule,
                                  "return": h["return"], "evaluated_occupancy_rate": h["inadmissible_rate"],
                                  "expected_violations": h["expected_violations"],
                                  "prob_any_violation": h["prob_any_violation"]})
SOFT_ANALYSIS = pd.DataFrame(soft_rows)
SOFT_ANALYSIS.to_csv(TABLE_DIR / "soft_constraint_analysis.csv", index=False)
MARGINS = pd.DataFrame(MARGIN_ROWS)
MARGINS.to_csv(TABLE_DIR / "greedy_margin_tie_report.csv", index=False)
if len(SOFT_ANALYSIS):
    display(SOFT_ANALYSIS.groupby(["display", "extraction", "rule"]).agg(
        n=("seed", "nunique"), surrogate=("surrogate_mass_training_distribution", "mean"),
        occupancy=("evaluated_occupancy_rate", "mean"), p_any=("prob_any_violation", "mean"),
        ret=("return", "mean"), lam=("dual_lambda_final", "mean")).reset_index().round(5))
if len(MARGINS):
    display(MARGINS.groupby("policy").agg(min_margin=("min_margin_admissible_minus_inadmissible", "min"),
                                          inadm_first=("states_inadmissible_ranked_first", "mean"),
                                          ties=("states_tied_admissible_vs_inadmissible", "mean")).reset_index())

# %% [markdown]
# ## 25. Released mask versus support estimated from the simulated training data (audit M02)
#
# Two different objects: the released mask encodes source-data transition support (count > 20 in the
# original MIMIC-derived table); counts in 10,000 *simulated* clinician episodes answer a different question,
# so an action can be frequent here yet unsupported in the source data (the 11.87% logged inadmissible rate is
# expected, not a paradox or an action-code bug). This experiment gives learners **only** an estimated mask
# (no released-mask access) and evaluates violations against the released mask. The matched-information
# comparison in which *every* policy receives the released mask is the projection / masked-greedy set of
# Section 19. Estimated masks do not reconstruct the original clinical evidence.

# %%
# ============================================================
# 25. Support estimation experiment
# ============================================================
EXECLOG.mark("25. Support estimation (M02)")
EST_MASK_RULES = [("count_at_least", 1), ("count_at_least", 5), ("count_at_least", 21),
                  ("behavior_ratio_at_least", 0.1), ("behavior_ratio_at_least", 0.3)]
EST_REPORTS = []
EST_MASKS = {}
for rule, param in EST_MASK_RULES:
    est, rep = estimated_support_mask(MAIN.counts, rule, param)
    EST_MASKS[(rule, param)] = est
    EST_REPORTS.append(rep)
SUPPORT_ESTIMATION = pd.DataFrame(EST_REPORTS)
SUPPORT_ESTIMATION.to_csv(TABLE_DIR / "support_estimation_mask_comparison.csv", index=False)
display(SUPPORT_ESTIMATION)

if EXPERIMENTS["support_estimation"] and EXPERIMENTS["main_tabular"]:
    for (rule, param), est in EST_MASKS.items():
        tag = f"est:{rule}>={param}"
        src = f"dataset-estimated ({rule} >= {param})"
        for seed in CFG.seeds:
            cql, iql = get_tab("cql", MAIN, seed), get_tab("iql", MAIN, seed)
            register_policy(f"Tabular CQL [masked greedy, {tag}]", seed, one_hot_policy(masked_argmax(cql["q"], est)),
                            family="Tabular CQL", extraction="masked greedy on an estimated mask", stochastic=False,
                            mask_source=src, mask_use="extraction", information="logged transitions only",
                            group="support_estimation")
            w = np.where(est, iql["aw_scores"], -np.inf)
            act = w.argmax(axis=1)
            none = ~(np.where(est, iql["aw_scores"], 0.0) > 0).any(axis=1)
            act[none] = masked_argmax(np.broadcast_to(MAIN.counts.sum(axis=0).astype(float), iql["q"].shape), est)[none]
            register_policy(f"Tabular IQL [masked greedy, {tag}]", seed, one_hot_policy(act), family="Tabular IQL",
                            extraction="AW-score argmax on an estimated mask", stochastic=False, mask_source=src,
                            mask_use="extraction", information="logged transitions only", group="support_estimation")
            register_policy(f"Tabular IQL [projected onto {tag}]", seed,
                            project_to_support(REGISTRY["Tabular IQL"].by_seed[seed], est), family="Tabular IQL",
                            extraction="uniform redistribution onto an estimated mask (not mean-equivalent)",
                            stochastic=True, mask_source=src, mask_use="projection", information="logged transitions only",
                            group="support_estimation")
        register_policy(f"Behavior Cloning [masked greedy, {tag}]", FIXED, one_hot_policy(masked_argmax(MAIN.counts.astype(float), est)),
                        family="BC", extraction="count argmax on an estimated mask", stochastic=False, mask_source=src,
                        mask_use="extraction", information="logged counts only", group="support_estimation")
    if EXPERIMENTS["main_diffusion"]:
        for seed in CFG.seeds:
            prior = train_diffusion_prior(MAIN, seed)
            cache = ReverseCache(prior["model"], prior["diffusion"], MAIN.all_x)
            for key in [("count_at_least", 21), ("behavior_ratio_at_least", 0.3)]:
                est = EST_MASKS[key]
                G = critic_ensemble(get_tab("cql", MAIN, seed)["q"], mask=est)
                pi, tr = guided_reverse_policy(cache, G, G_PRIMARY, mask=est)
                register_diffusion(f"Diffusion [tabular CQL critic, est:{key[0]}>={key[1]}]", seed, pi, None,
                                   "support_estimation", "guided reverse chain with an estimated mask",
                                   "tabular CQL", mask_source=f"dataset-estimated ({key[0]} >= {key[1]})")
            del cache
            gc.collect()
    refresh_results()
    SUPPORT_EST_RESULTS = SUMMARY[SUMMARY["group"] == "support_estimation"]
    SUPPORT_EST_RESULTS.to_csv(TABLE_DIR / "support_estimation_policy_results.csv", index=False)
    display(SUPPORT_EST_RESULTS[["name", "rule", "n_seeds", "return_mean", "inadmissible_rate_mean",
                                 "prob_any_violation_mean"]].round(5))
