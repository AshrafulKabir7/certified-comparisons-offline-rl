# %% [markdown]
# ## 26. Selection protocol: episode split, training-only normalisation, FQE selector validation, locked fresh evaluation (audit M04)
#
# * Disclosure: the main-study configurations are fixed a priori (Section 1; Table 3 of the original study); exact-evaluation results of the
#   previous run were visible during development, and the guidance sweep is exploratory. Every method in the
#   main study therefore has a tuning budget of **one** configuration (see `tuning_budget.csv`).
# * Offline selection demonstration (tabular families, equal budget of three configurations each; the
#   training-only normaliser fit is recorded for protocol completeness, the tabular candidates do not use it): complete
#   episodes are split 80/20 with a seeded stream (no trajectory is shared across partitions); the feature
#   normaliser is fitted on training episodes only; candidates are trained on the training episodes; a
#   **tabular model-based FQE** estimate on the development episodes selects one configuration per family.
#   FQE is a candidate selector, not a guaranteed oracle, so its absolute error, rank correlation with exact
#   returns and top-1 regret are reported. The selected configurations are then **locked** and retrained on a
#   fresh, independently generated dataset for a final untouched exact evaluation. Holding out trajectories
#   from the same MDP does not create an independent hospital.

# %%
# ============================================================
# 26. Selection protocol
# ============================================================
EXECLOG.mark("26. Selection protocol (M04)")
TUNING_BUDGET = pd.DataFrame([
    {"method": m, "configurations_tried_main_study": 1, "selection_criterion": "fixed a priori (Table 3 settings)",
     "exact_results_visible_during_development": "yes (previous run); disclose in paper"}
    for m in ["Behavior Cloning", "Tabular CQL", "Tabular IQL", "Tabular BCQ", "Deep CQL", "AM-CQL (Hard, legacy)",
              "AM-CQL (Hard, design A)", "AM-CQL (Hard, design B)", "AM-CQL (Soft)", "Neural BC", "AW imitation",
              "Diffusion prior + guidance (g=1.5 fixed; sweep exploratory)"]])
TUNING_BUDGET.to_csv(TABLE_DIR / "tuning_budget.csv", index=False)


def empirical_mdp_evaluator(dev: Dict[str, np.ndarray], fallback: Dict[str, np.ndarray]) -> Tuple[ExactMDPEvaluator, dict]:
    """Tabular model-based FQE on dev episodes. Each (s, a) row uses dev counts if observed in dev, else the
    training split's counts for that (s, a) if observed there, else the state's action-pooled dev distribution
    (training-split pooled if the state is unseen in dev); states unseen in both splits are pessimistic
    absorbing-death rows."""
    def counts_of(d):
        c = np.zeros((S_N, A_N, S_N), dtype=np.float64)
        np.add.at(c, (d["state"], d["action"], d["next_state"]), 1.0)
        return c
    c_dev, c_tr = counts_of(dev), counts_of(fallback)
    P = np.zeros((S_N, A_N, S_N))
    stats_ = {"rows_from_dev": 0, "rows_from_train": 0, "rows_state_pooled": 0, "states_unobserved": 0}
    for s in np.flatnonzero(LIVE):
        sa_d, sa_t = c_dev[s].sum(axis=1), c_tr[s].sum(axis=1)
        base = c_dev[s] if sa_d.sum() > 0 else c_tr[s]
        if base.sum() == 0:
            P[s, :, BENCH.death_state] = 1.0
            stats_["states_unobserved"] += 1
            continue
        pooled = base.sum(axis=0) / base.sum()
        P[s] = np.where(sa_d[:, None] > 0, c_dev[s] / np.maximum(sa_d[:, None], 1),
                        np.where(sa_t[:, None] > 0, c_tr[s] / np.maximum(sa_t[:, None], 1), pooled[None, :]))
        stats_["rows_from_dev"] += int((sa_d > 0).sum())
        stats_["rows_from_train"] += int(((sa_d == 0) & (sa_t > 0)).sum())
        stats_["rows_state_pooled"] += int(((sa_d == 0) & (sa_t == 0)).sum())
    for t in BENCH.terminal_states:
        P[t, :, t] = 1.0
    d0 = np.bincount(dev["state"][dev["timestep"] == 0], minlength=S_N).astype(np.float64)
    d0 /= d0.sum()
    r_sa = P[:, :, BENCH.survival_state].copy()   # reward 1 on entering survival (verified in Section 3)
    r_sa[~LIVE] = 0.0
    del c_dev, c_tr
    ev = ExactMDPEvaluator(P, None, d0, SUPPORT_MASK, BENCH.terminal_states, BENCH.death_state,
                           BENCH.survival_state, r_sa=r_sa)
    return ev, stats_


def reward_entry_only_on_support() -> Dict[str, Any]:
    """ICU-Sepsis pays rewards on entering a state: on every possible transition out of a live state the reward is
    the entry reward of the destination. Absorbing self-loops are excluded: their reward never enters gamma = 1
    evaluation and is not destination-only in the released r_mat, so the all-rows flag of Section 3 is too strict
    for the model-based FQE below (which estimates r(s, a) = P_hat(survival | s, a) from data only)."""
    R = np.asarray(BENCH.base.dynamics["r_mat"], dtype=np.float64)
    entry = R[int(np.flatnonzero(LIVE)[0]), 0].copy()
    holds = True
    for s in np.flatnonzero(LIVE):      # row by row: no full-size temporaries
        sup = EVALUATOR.P[s] > 0
        holds &= bool(np.all(R[s][sup] == np.broadcast_to(entry, R[s].shape)[sup]))
    return {"holds_on_support": bool(holds), "rewarding_destinations": np.flatnonzero(entry != 0).tolist()}


SELECTION = {}
if EXPERIMENTS["selection_protocol"]:
    REWARD_SUPPORT_CHECK = reward_entry_only_on_support()
    assert REWARD_SUPPORT_CHECK["holds_on_support"] and \
        REWARD_SUPPORT_CHECK["rewarding_destinations"] == [BENCH.survival_state], REWARD_SUPPORT_CHECK
    rng = np_rng("selection", "episode-split", MAIN.data_hash)
    eps = np.unique(OFFLINE["episode_id"])
    dev_eps = rng.choice(eps, size=int(round(CFG.dev_fraction * len(eps))), replace=False)
    dev_rows = np.isin(OFFLINE["episode_id"], dev_eps)
    TRAIN_SPLIT = make_bundle("split_train", OFFLINE, OFFLINE_MANIFEST, rows=np.flatnonzero(~dev_rows))
    DEV_DATA = subset_rows(OFFLINE, np.flatnonzero(dev_rows))
    assert not set(np.unique(TRAIN_SPLIT.data["episode_id"])) & set(np.unique(DEV_DATA["episode_id"]))
    FQE_EVAL, FQE_MODEL_STATS = empirical_mdp_evaluator(DEV_DATA, TRAIN_SPLIT.data)
    sel_seed = CFG.seeds[0]
    grid = ([("Tabular CQL", {"alpha": a}) for a in (0.3, 1.0, 3.0)]
            + [("Tabular IQL", {"expectile": e, "beta": b}) for e, b in ((0.7, 1.0), (0.7, 3.0), (0.9, 3.0))]
            + [("Tabular BCQ", {"tau": t}) for t in (0.1, 0.3, 0.5)])

    def train_candidate(method: str, hp: dict, bundle: DataBundle, seed: int) -> Dict[str, np.ndarray]:
        if method == "Tabular CQL":
            r = train_tabular_cql(bundle, seed, alpha=hp["alpha"], tag="select_cql")
            return {"native": greedy_policy(r["q"], False), "masked greedy": greedy_policy(r["q"], True)}
        if method == "Tabular IQL":
            r = train_tabular_iql(bundle, seed, expectile=hp["expectile"], beta=hp["beta"], tag="select_iql")
            return {"native": r["deterministic"], "masked greedy": r["masked_greedy"]}
        r = train_tabular_bcq(bundle, seed, tau=hp["tau"], tag="select_bcq")
        return {"native": r["deterministic"], "masked greedy": r["masked_greedy"]}

    cand_rows = []
    for method, hp in grid:
        pols = train_candidate(method, hp, TRAIN_SPLIT, sel_seed)
        pols["projected"] = project_to_support(pols["native"])
        for extraction, pi in pols.items():
            for rule in RULES:
                ex = safe_evaluate(pi, rule)
                exact = ex["return"] if ex is not None else np.nan
                fq = safe_evaluate(pi, rule, evaluator=FQE_EVAL)
                fqe = fq["return"] if fq is not None else np.nan
                cand_rows.append({"method": method, "hparams": json.dumps(hp), "extraction": extraction, "rule": rule,
                                  "fqe_dev_return": fqe, "exact_return": exact, "abs_error": abs(fqe - exact)})
    CANDIDATES = pd.DataFrame(cand_rows)
    CANDIDATES.to_csv(TABLE_DIR / "selection_candidates_fqe_vs_exact.csv", index=False)
    endpoint_rule = PREREG["primary_endpoint"]["rule"]
    sel_rows = []
    for method in CANDIDATES["method"].unique():
        sub = CANDIDATES[(CANDIDATES["method"] == method) & (CANDIDATES["rule"] == endpoint_rule)].dropna(subset=["fqe_dev_return"])
        if sub.empty:
            continue
        chosen = sub.loc[sub["fqe_dev_return"].idxmax()]
        best = sub.loc[sub["exact_return"].idxmax()]
        rho = float(stats.spearmanr(sub["fqe_dev_return"], sub["exact_return"])[0]) if len(sub) > 2 else np.nan
        sel_rows.append({"method": method, "selected_hparams": chosen["hparams"], "selected_extraction": chosen["extraction"],
                         "selected_exact_return_on_split_train": chosen["exact_return"],
                         "best_exact_return_in_grid": best["exact_return"], "top1_regret": best["exact_return"] - chosen["exact_return"],
                         "fqe_mean_abs_error": float(sub["abs_error"].mean()), "fqe_spearman_vs_exact": rho})
    SELECTED = pd.DataFrame(sel_rows)
    # Locked evaluation on a fresh, independently generated dataset.
    fresh_spec = DatasetSpec(n_episodes=CFG.dataset_episodes, seed=CFG.fresh_eval_dataset_seed, max_steps=CFG.max_episode_steps)
    fresh_data, fresh_manifest = load_or_build_dataset(fresh_spec)
    FRESH = make_bundle(f"fresh{CFG.fresh_eval_dataset_seed}", fresh_data, fresh_manifest)
    locked = []
    for _, row in SELECTED.iterrows():
        hp = json.loads(row["selected_hparams"])
        default_hp = {"Tabular CQL": {"alpha": CFG.tab_cql_alpha},
                      "Tabular IQL": {"expectile": CFG.iql_expectile, "beta": CFG.iql_beta},
                      "Tabular BCQ": {"tau": CFG.bcq_threshold}}[row["method"]]
        for label, h in (("selected", hp), ("main-study default", default_hp)):
            vals = []
            for seed in CFG.ablation_seeds:
                pols = train_candidate(row["method"], h, FRESH, seed)
                pols["projected"] = project_to_support(pols["native"])
                ev_ = safe_evaluate(pols[row["selected_extraction"]], endpoint_rule)
                vals.append(ev_["return"] if ev_ is not None else np.nan)
            locked.append({"method": row["method"], "config": label, "hparams": json.dumps(h),
                           "extraction": row["selected_extraction"], "rule": endpoint_rule,
                           "fresh_dataset_return_mean": float(np.mean(vals)),
                           "fresh_dataset_return_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                           "n_seeds": len(vals)})
    LOCKED = pd.DataFrame(locked)
    SELECTED.to_csv(TABLE_DIR / "selection_fqe_selected.csv", index=False)
    LOCKED.to_csv(TABLE_DIR / "selection_locked_fresh_evaluation.csv", index=False)
    SELECTION = {"dev_episodes": int(len(dev_eps)), "train_episodes": int(len(eps) - len(dev_eps)),
                 "normalizer_fit_on": "training episodes only", "normalizer_hash": TRAIN_SPLIT.normalizer.hash,
                 "fqe": ("tabular empirical-model evaluation on dev episodes; (s,a) unseen in dev use training-split "
                         "counts, else the state's pooled dev distribution; states unseen in both are absorbing-death rows"),
                 "fqe_model_rows": FQE_MODEL_STATS,
                 "n_candidate_evaluations_improper_in_fqe_model": int(CANDIDATES["fqe_dev_return"].isna().sum()),
                 "note": "the training-only normaliser fit is recorded for protocol completeness; tabular candidates do not use features",
                 "stages": ["episode split (seeded, 80/20, complete episodes)", "normaliser fit on training episodes",
                            "train 3 configurations per tabular family on the training split (equal budget)",
                            "tabular FQE on dev episodes selects one configuration + extraction per family",
                            "lock the selected configuration",
                            f"retrain on fresh dataset seed {CFG.fresh_eval_dataset_seed} and evaluate exactly (untouched)"],
                 "endpoint_rule": endpoint_rule, "fresh_dataset": fresh_manifest["file"]}
    save_json(SELECTION, TABLE_DIR / "selection_protocol.json")
    display(SELECTED.round(5))
    display(LOCKED.round(5))
    del FQE_EVAL
    gc.collect()

# %% [markdown]
# ## 27. Robustness: independent datasets × training seeds, smaller datasets, behaviour mixtures (audit M03, M07)
#
# Independent clinician datasets are generated with different dataset seeds; training seeds are nested within
# datasets. The full profile uses **8 datasets × 2 training seeds** (`CFG.replication_dataset_seeds`,
# `CFG.replication_train_seeds`): 16 trained runs per method, but only **8 independent units** — the dataset is the
# unit of the hierarchical analysis (Section 30), so a dataset-level two-sided sign-flip test can reach
# p = 2/2⁸ ≈ 0.008. The lite profile (5 × 2) cannot reject two-sided at 0.05 (minimum p = 2/2⁵ = 0.0625); it is a
# development setting. Spending the budget on datasets rather than on training seeds per dataset is deliberate:
# the between-dataset variance is the component that a single frozen dataset cannot reveal. The primary dataset
# reuses the main-study checkpoints automatically through identical run identities. All of this measures
# variability *within one simulator*; it does not measure uncertainty in the simulator itself (Part II addresses
# benchmark-construction error on simulators with known ground truth).

# %%
# ============================================================
# 27. Replication and data regimes
# ============================================================
EXECLOG.mark("27. Replication and data regimes (M03/M07)")
REPLICATION_BUNDLES: Dict[int, DataBundle] = {}
REPL_METHODS = ["Tabular CQL", "Tabular CQL [masked greedy]", "Tabular IQL", "Tabular IQL [masked greedy]",
                "Tabular IQL [projected]", f"Tabular BCQ (tau={CFG.bcq_threshold})", "Behavior Cloning [masked greedy]",
                "Masked critic ensemble [critic B + tabular CQL]", "AM-CQL (Hard, design B: admissible-row CQL)", PRIMARY_DIFFUSION]


def rep_prefix(ds: int) -> str:
    return "" if ds == CFG.dataset_seed else f"[D{ds}] "


if EXPERIMENTS["replication"]:
    for ds in CFG.replication_dataset_seeds:
        if ds == CFG.dataset_seed:
            bundle = MAIN
        else:
            spec = DatasetSpec(n_episodes=CFG.dataset_episodes, seed=ds, max_steps=CFG.max_episode_steps)
            data, manifest = load_or_build_dataset(spec)
            bundle = make_bundle(f"rep{ds}", data, manifest)
        REPLICATION_BUNDLES[ds] = bundle
        grp = "main" if ds == CFG.dataset_seed else "replication"
        pre = rep_prefix(ds)
        run_tabular_suite(bundle, CFG.replication_train_seeds, group=grp, prefix=pre, dataset_label=f"dataset {ds}")
        bcp = train_behavior_cloning(bundle)
        register_policy(f"{pre}Behavior Cloning [masked greedy]", FIXED, bcp["masked_greedy"], group=grp,
                        dataset=f"dataset {ds}", family="BC", extraction="argmax of counts over admissible actions",
                        stochastic=False, mask_source="released benchmark mask", mask_use="extraction",
                        information="logged counts + released mask")
        if EXPERIMENTS["replication_include_neural"]:
            run_critics([SPEC_HARD_B], bundle, CFG.replication_train_seeds, group=grp, prefix=pre,
                        dataset_label=f"dataset {ds}")
            for seed in CFG.replication_train_seeds:
                run_diffusion_suite(bundle, seed, group=grp, prefix=pre, full_ablations=False, dataset_label=f"dataset {ds}")
        if ds != CFG.dataset_seed:
            bundle.release()
    add_projection_controls(["replication", "main"])
    finalize_projection_checks(verbose=False)
    refresh_results()

REGIME_ROWS = []
if EXPERIMENTS["data_regimes"]:
    # Independent dataset draws per reduced size (not prefixes of the main dataset).
    regimes = [(f"n={n} draw{k}", DatasetSpec(n_episodes=n, seed=int(stream_seed("dataset", "regime", n, k) % (2 ** 31 - 1)),
                                              max_steps=CFG.max_episode_steps))
               for n in CFG.data_size_regimes for k in range(CFG.regime_dataset_draws)]
    regimes += [(f"behaviour={b}", DatasetSpec(n_episodes=CFG.dataset_episodes, seed=CFG.dataset_seed,
                                               max_steps=CFG.max_episode_steps, behavior=b)) for b in CFG.behavior_mixtures]
    for label, spec in regimes:
        data, manifest = load_or_build_dataset(spec)
        bundle = make_bundle(f"regime_{slug(label)}", data, manifest)
        run_tabular_suite(bundle, CFG.regime_train_seeds, group="data_regime", prefix=f"[{label}] ", dataset_label=label)
        bcp = train_behavior_cloning(bundle)
        register_policy(f"[{label}] Behavior Cloning [masked greedy]", FIXED, bcp["masked_greedy"], group="data_regime",
                        dataset=label, family="BC", extraction="argmax of counts over admissible actions", stochastic=False,
                        mask_source="released benchmark mask", mask_use="extraction", information="logged counts + released mask")
        REGIME_ROWS.append({"regime": label, "dataset_seed": spec.seed, "behavior": spec.behavior,
                            "n_episodes": spec.n_episodes, **validate_dataset(data, spec)})
        bundle.release()
    add_projection_controls(["data_regime"])
    finalize_projection_checks(verbose=False)
    refresh_results()
    REGIMES = pd.DataFrame(REGIME_ROWS)
    REGIMES.to_csv(TABLE_DIR / "data_regime_datasets.csv", index=False)
    reg_sum = SUMMARY[SUMMARY["group"] == "data_regime"]
    reg_sum.to_csv(TABLE_DIR / "data_regime_results.csv", index=False)
    # Per-size summary separating dataset-draw variance from training-seed variance.
    ok_r = RESULTS[(RESULTS["status"] == "ok") & (RESULTS["group"] == "data_regime")]
    size_rows = []
    for n in CFG.data_size_regimes:
        labs = [f"n={n} draw{k}" for k in range(CFG.regime_dataset_draws)]
        methods = sorted({nm.split("] ", 1)[1] for nm in ok_r["name"].unique() if any(nm.startswith(f"[{l}] ") for l in labs)})
        for method in methods:
            for rule in RULES:
                per = [ok_r[(ok_r["name"] == f"[{l}] {method}") & (ok_r["rule"] == rule)]["return"].to_numpy(dtype=float) for l in labs]
                per = [v for v in per if len(v)]
                if not per:
                    continue
                means = np.array([v.mean() for v in per])
                w = [v.var(ddof=1) for v in per if len(v) > 1]
                within = float(np.mean(w)) if w else np.nan
                vm = float(means.var(ddof=1)) if len(per) > 1 else np.nan
                k_h = len(per) / sum(1.0 / len(v) for v in per)
                size_rows.append({"n_episodes": n, "method": method, "rule": rule, "n_draws": len(per),
                                  "mean_of_draw_means": float(means.mean()), "variance_of_draw_means": vm,
                                  "mean_within_draw_variance": within,
                                  "between_draw_variance_component_mom": (max(0.0, vm - within / k_h)
                                                                          if np.isfinite(vm) and np.isfinite(within) else np.nan)})
    REGIME_SIZE_SUMMARY = pd.DataFrame(size_rows)
    REGIME_SIZE_SUMMARY.to_csv(TABLE_DIR / "data_regime_size_summary.csv", index=False)
    display(reg_sum[["name", "rule", "n_seeds", "return_mean", "return_sd", "inadmissible_rate_mean"]].round(5))

# %% [markdown]
# ## 28. MDP stress tests (locked policies) and severity / support subgroup diagnostics (audit M07)
#
# Policies trained on the original data are evaluated without retraining on perturbed MDPs:
# (i) log-normal perturbation of admissible transition rows with inadmissible rows recomputed as the mean of
# admissible rows (mean-rule semantics preserved); (ii) a SOFA-tilted initial-state distribution;
# (iii) changed admissible sets (removing or adding one action in a random subset of states). These are
# **stress tests, not confidence sets**: no calibrated source-count uncertainty is available. Ranking
# persistence is summarised by Kendall τ against the unperturbed ranking. Subgroup diagnostics report
# exact returns conditional on the initial state's SOFA quartile and support-set size; they are simulation
# analyses over state-centroid groups, not patient cohorts. Immediate termination is a benchmark stress rule,
# not a prognosis for an unsupported intervention. Under the mean rule, adding an admissible action whose row is the
# mean-imputed row leaves the dynamics unchanged, so `mask_add_one` is reported for the terminate rule only.

# %%
# ============================================================
# 28. Stress tests + subgroup diagnostics
# ============================================================
EXECLOG.mark("28. Stress tests and subgroups (M07)")
STRESS_POLICIES = [n for n in MAIN_TABLE_POLICIES if n in REGISTRY and "Oracle" not in n]


def recompute_mean_rows(P: np.ndarray, mask: np.ndarray) -> None:
    for s in np.flatnonzero(LIVE):
        m = mask[s]
        if (~m).any():
            P[s, ~m] = P[s, m].mean(axis=0)


def perturbed_evaluator(kind: str, param: float, seed: int) -> ExactMDPEvaluator:
    rng = np_rng("stress", kind, param, seed)
    P = EVALUATOR.P.copy()
    d0 = D0.copy()
    mask = SUPPORT_MASK.copy()
    if kind == "transition_lognormal":
        noise = np.exp(param * rng.standard_normal(P.shape))
        live_rows = LIVE[:, None] & mask
        P[live_rows] = P[live_rows] * noise[live_rows]
        P[live_rows] /= P[live_rows].sum(axis=1, keepdims=True)
        recompute_mean_rows(P, mask)
    elif kind == "d0_sofa_tilt":
        z = (BENCH.sofa_scores - BENCH.sofa_scores[LIVE].mean()) / BENCH.sofa_scores[LIVE].std()
        d0 = d0 * np.exp(param * z)
        d0[~LIVE] = 0.0
        d0 /= d0.sum()
    elif kind == "mask_remove_one":
        for s in np.flatnonzero(LIVE & (mask.sum(axis=1) >= 2)):
            if rng.random() < param:
                mask[s, rng.choice(np.flatnonzero(mask[s]))] = False
        recompute_mean_rows(P, mask)
    elif kind == "mask_add_one":
        for s in np.flatnonzero(LIVE & (mask.sum(axis=1) < A_N)):
            if rng.random() < param:
                mask[s, rng.choice(np.flatnonzero(~mask[s]))] = True   # its row stays the mean-imputed estimate
        recompute_mean_rows(P, mask)
    else:
        raise ValueError(kind)
    r_sa = P[:, :, BENCH.survival_state].copy()
    r_sa[~LIVE] = 0.0
    return ExactMDPEvaluator(P, None, d0, mask, BENCH.terminal_states, BENCH.death_state, BENCH.survival_state, r_sa=r_sa)


STRESS_ROWS = []
if EXPERIMENTS["mdp_stress_tests"] and STRESS_POLICIES:
    perturbations = ([("transition_lognormal", s) for s in (0.1, 0.3)] + [("d0_sofa_tilt", k) for k in (-0.5, 0.5)]
                     + [("mask_remove_one", 0.2), ("mask_add_one", 0.1)])
    for kind, param in perturbations:
        pseeds = CFG.stress_seeds if kind != "d0_sofa_tilt" else (0,)
        for ps in pseeds:
            ev = perturbed_evaluator(kind, param, ps)
            for name in STRESS_POLICIES:
                for seed, pi in REGISTRY[name].by_seed.items():
                    for rule in RULES:
                        if kind == "mask_add_one" and rule == "mean":
                            continue   # tautological: the added action keeps the mean-imputed row, so P is unchanged
                        try:
                            m = ev.evaluate(pi, rule)
                            STRESS_ROWS.append({"perturbation": kind, "param": param, "perturbation_seed": ps,
                                                "policy": name, "seed": seed, "rule": rule, "return": m["return"],
                                                "inadmissible_rate": m["inadmissible_rate"],
                                                "prob_any_violation": m["prob_any_violation"]})
                        except (ImproperPolicyError, EvaluationInvariantError) as exc:
                            STRESS_ROWS.append({"perturbation": kind, "param": param, "perturbation_seed": ps,
                                                "policy": name, "seed": seed, "rule": rule, "return": np.nan,
                                                "inadmissible_rate": np.nan, "prob_any_violation": np.nan,
                                                "error": str(exc)})
            del ev
            gc.collect()
    STRESS = pd.DataFrame(STRESS_ROWS)
    STRESS.to_csv(TABLE_DIR / "stress_test_results.csv", index=False)
    base = SUMMARY[SUMMARY["name"].isin(STRESS_POLICIES)].set_index(["name", "rule"])["return_mean"]
    rank_rows = []
    for (kind, param, ps, rule), g in STRESS.groupby(["perturbation", "param", "perturbation_seed", "rule"]):
        pm = g.groupby("policy")["return"].mean()
        common = [n for n in pm.index if (n, rule) in base.index and np.isfinite(pm[n])]
        if len(common) < 3:
            continue
        tau = float(stats.kendalltau([base[(n, rule)] for n in common], [pm[n] for n in common])[0])
        top_base = max(common, key=lambda n: base[(n, rule)])
        top_pert = max(common, key=lambda n: pm[n])
        rank_rows.append({"perturbation": kind, "param": param, "perturbation_seed": ps, "rule": rule,
                          "kendall_tau_vs_unperturbed": tau, "top1_unchanged": top_base == top_pert,
                          "top1_unperturbed": top_base, "top1_perturbed": top_pert, "n_policies": len(common)})
    STRESS_RANKING = pd.DataFrame(rank_rows)
    STRESS_RANKING.to_csv(TABLE_DIR / "stress_test_ranking_persistence.csv", index=False)
    display(STRESS_RANKING.groupby(["perturbation", "param", "rule"]).agg(
        kendall_tau=("kendall_tau_vs_unperturbed", "mean"), top1_unchanged=("top1_unchanged", "mean")).reset_index().round(3))

# Subgroup diagnostics on the unperturbed MDP.
SUB_ROWS = []
init_states = np.flatnonzero(D0 > 0)
sofa_q = pd.qcut(pd.Series(BENCH.sofa_scores[init_states]).rank(method="first"), q=4,
                 labels=["Q1 (lowest SOFA)", "Q2", "Q3", "Q4 (highest SOFA)"])
k_init = SUPPORT_MASK[init_states].sum(axis=1)
k_group = np.where(k_init == 1, "k=1", np.where(k_init <= 4, "k=2-4", "k>=5"))
for name in STRESS_POLICIES:
    for seed, pi in REGISTRY[name].by_seed.items():
        for rule in RULES:
            m = safe_evaluate(pi, rule, return_vectors=True)
            if m is None:
                continue
            v = m["value_vector"][init_states]
            w = D0[init_states]
            for gname, labels in (("initial SOFA quartile", np.asarray(sofa_q, dtype=object)), ("initial support size", k_group)):
                for lab in pd.unique(labels):
                    sel = labels == lab
                    SUB_ROWS.append({"policy": name, "seed": seed, "rule": rule, "grouping": gname, "group": lab,
                                     "conditional_return": float((w[sel] * v[sel]).sum() / w[sel].sum()),
                                     "initial_mass": float(w[sel].sum())})
SUBGROUPS = pd.DataFrame(SUB_ROWS)
SUBGROUPS.to_csv(TABLE_DIR / "subgroup_diagnostics.csv", index=False)
if len(SUBGROUPS):
    display(SUBGROUPS.groupby(["grouping", "group", "rule", "policy"])["conditional_return"].mean().unstack("rule").head(40).round(4))

# %% [markdown]
# ## 29. Representation ablation — removing treatment-related input features (audit M08)
#
# The benchmark construction script defines the logged action from `input_4hourly` and `max_dose_vaso` of the
# same 4-hour bloc **and** clusters states on all 47 features including those columns. Dropping neural input
# columns therefore cannot remove information already embedded in the state IDs and dynamics; this ablation
# only measures how much the neural models *rely on the explicit columns*. A clinically valid representation
# would need lagged treatment variables and a rebuilt state clustering. Treatment leakage is a risk requiring
# verification at patient level, not a confirmed patient-level leak.

# %%
# ============================================================
# 29. Representation ablation
# ============================================================
EXECLOG.mark("29. Representation ablation (M08)")
REPR_BUNDLES = {}
if EXPERIMENTS["representation_ablation"] and EXPERIMENTS["main_tabular"]:
    for tag_name, drop in (("drop treatment-direct features", TREATMENT_FEATURES_DIRECT),
                           ("drop treatment-related features", TREATMENT_FEATURES_RELATED)):
        bundle = make_bundle(f"main_{slug(tag_name)}", OFFLINE, OFFLINE_MANIFEST, feature_drop=drop)
        REPR_BUNDLES[tag_name] = bundle
        pre = f"[{tag_name}] "
        run_critics([SPEC_HARD_B], bundle, CFG.ablation_seeds, group="representation", prefix=pre, dataset_label="main")
        for seed in CFG.ablation_seeds:
            bc_net = train_policy_net(bundle, seed, mode="bc")
            PNETS[("bc", bundle.name, seed)] = bc_net
            register_policy_net(bc_net, f"{pre}Neural BC (matched capacity)", seed, group="representation")
            if EXPERIMENTS["main_diffusion"]:
                run_diffusion_suite(bundle, seed, group="representation", prefix=pre, full_ablations=False, tab_bundle=MAIN)
        bundle.release()
    add_projection_controls(["representation"])
    finalize_projection_checks(verbose=False)
    refresh_results()
    REPR_RESULTS = SUMMARY[(SUMMARY["group"] == "representation") | (SUMMARY["name"].isin(
        ["AM-CQL (Hard, design B: admissible-row CQL)", "Neural BC (matched capacity) [masked greedy]", PRIMARY_DIFFUSION]))]
    REPR_RESULTS.to_csv(TABLE_DIR / "representation_ablation.csv", index=False)
    display(REPR_RESULTS[["name", "rule", "n_seeds", "return_mean", "return_sd", "inadmissible_rate_mean"]].round(5))
