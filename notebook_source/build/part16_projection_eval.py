# %% [markdown]
# ## 19. Admissible projection and extraction controls — the decisive missing baseline (audit C01)
#
# Every registered policy that places any probability on an inadmissible action receives a
# **`[projected]`** counterpart built from the original policy and the *same released mask* available to the
# masked learners (so the projection's mask use is disclosed exactly like AM-CQL's). Stochastic natives also get
# a **`[mask-renormalized]`** counterpart, which conditions on admissibility and is *not* return-preserving.
#
# Closure checks, per policy and seed: row sums, zero inadmissible probability, and equality of the projected
# versus original **policy-induced transition matrix and expected-reward vector under the mean rule**, plus
# equality of the projected policy's induced quantities under the two rules, all to numerical tolerance.
# Note the projection of a deterministic export is a *stochastic* policy; deterministic masked variants are
# evaluated separately (`[masked greedy]`).

# %%
# ============================================================
# 19. Projection controls + equivalence verification
# ============================================================
EXECLOG.mark("19. Projection controls")
PROJECTION_ROWS: List[dict] = []


def add_projection_controls(groups: Sequence[str]) -> None:
    for name in [n for n, e in REGISTRY.items() if e.group in groups and not n.endswith("[projected]")
                 and not n.endswith("[mask-renormalized]")]:
        e = REGISTRY[name]
        if not any(inadmissible_mass(pi).max() > 0 for pi in e.by_seed.values()):
            continue
        done = REGISTRY.get(f"{name} [projected]")
        for seed, pi in e.by_seed.items():
            if done is not None and seed in done.by_seed:
                continue   # already projected (avoids redundant work when a group is revisited)
            proj = project_to_support(pi)
            register_policy(f"{name} [projected]", seed, proj, family=e.family,
                            extraction=f"mean-equivalent admissible projection of: {e.extraction}", stochastic=True,
                            mask_source="released benchmark mask", mask_use="projection",
                            information=e.information + " + released mask", parent=name, critic=e.critic,
                            group=e.group, dataset=e.dataset)
            P_o, r_o = EVALUATOR.induced(pi, "mean")
            P_pm, r_pm = EVALUATOR.induced(proj, "mean")
            P_pt, r_pt = EVALUATOR.induced(proj, "terminate")
            PROJECTION_ROWS.append({
                "policy": name, "seed": seed, "group": e.group,
                "max_inadmissible_mass_original": float(inadmissible_mass(pi).max()),
                "max_inadmissible_mass_projected": float(inadmissible_mass(proj).max()),
                "max_row_sum_error_projected": float(np.abs(proj.sum(axis=1) - 1).max()),
                "max_abs_dP_original_vs_projected_mean": float(np.abs(P_o - P_pm)[LIVE].max()),
                "max_abs_dr_original_vs_projected_mean": float(np.abs(r_o - r_pm)[LIVE].max()),
                "max_abs_dP_projected_mean_vs_terminate": float(np.abs(P_pm - P_pt)[LIVE].max()),
                "max_abs_dr_projected_mean_vs_terminate": float(np.abs(r_pm - r_pt)[LIVE].max()),
            })
            if e.stochastic:
                ren, n_fb = renormalize_on_support(pi)
                register_policy(f"{name} [mask-renormalized]", seed, ren, family=e.family,
                                extraction=f"renormalised on admissible actions: {e.extraction}", stochastic=True,
                                mask_source="released benchmark mask", mask_use="extraction",
                                information=e.information + " + released mask", parent=name, critic=e.critic,
                                group=e.group, dataset=e.dataset, notes=f"{n_fb} zero-mass rows -> uniform admissible")


def finalize_projection_checks(verbose: bool = True) -> bool:
    """Write and assert ALL projection-equivalence rows accumulated so far (called after every projection batch)."""
    global PROJECTION_EQUIVALENCE, PROJECTION_IDENTITY_VERIFIED
    PROJECTION_EQUIVALENCE = (pd.DataFrame(PROJECTION_ROWS).drop_duplicates(["policy", "seed"], keep="last")
                              if PROJECTION_ROWS else pd.DataFrame())
    PROJECTION_EQUIVALENCE.to_csv(TABLE_DIR / "projection_equivalence_checks.csv", index=False)
    if not len(PROJECTION_EQUIVALENCE):
        PROJECTION_IDENTITY_VERIFIED = False
        return False
    assert PROJECTION_EQUIVALENCE["max_inadmissible_mass_projected"].max() == 0.0
    assert PROJECTION_EQUIVALENCE["max_row_sum_error_projected"].max() <= 1e-10
    worst = PROJECTION_EQUIVALENCE[[c for c in PROJECTION_EQUIVALENCE.columns if c.startswith("max_abs")]].max()
    if verbose:
        print(f"Projection equivalence over {len(PROJECTION_EQUIVALENCE)} policy/seed pairs; worst deviations:\n", worst)
    PROJECTION_IDENTITY_VERIFIED = bool(worst.max() <= 1e-10)
    if MEAN_RULE_IDENTITY_HOLDS:
        assert PROJECTION_IDENTITY_VERIFIED, "Projection did not preserve mean-rule dynamics although the identity holds."
    return PROJECTION_IDENTITY_VERIFIED


add_projection_controls(["main", "reference", "inference_ablation"])
finalize_projection_checks()

# %% [markdown]
# ## 20. Exact evaluation of every registered policy under `mean` and `terminate`, plus a Monte-Carlo cross-check (audit M09)
#
# Every results row carries the policy's family, extraction rule, stochastic/deterministic label,
# **mask source and mask use** (audit M02 closure: every table identifies the information each policy used).

# %%
# ============================================================
# 20. Exact evaluation
# ============================================================
EXECLOG.mark("20. Exact evaluation")
RULES = ("mean", "terminate")
RESULT_ROWS: List[dict] = []
EVALUATED: set = set()
METRICS = ["return", "avg_episode_length", "inadmissible_rate", "expected_violations", "prob_any_violation",
           "states_with_inadmissible_mass", "max_state_inadmissible_mass", "mean_state_inadmissible_mass",
           "absorb_death", "absorb_survival"]


def evaluate_registered(names: Optional[Iterable[str]] = None, evaluator: Optional[ExactMDPEvaluator] = None,
                        condition: bool = False) -> pd.DataFrame:
    ev = EVALUATOR if evaluator is None else evaluator
    for name in (list(REGISTRY) if names is None else list(names)):
        e = REGISTRY[name]
        for seed, pi in sorted(e.by_seed.items()):
            if (name, seed) in EVALUATED:
                continue
            for rule in RULES:
                try:
                    m = ev.evaluate(pi, rule, condition=condition and e.group in ("main", "reference"))
                    status = "ok"
                except ImproperPolicyError as exc:
                    m, status = {}, f"improper: {exc}"
                RESULT_ROWS.append({**e.meta(), "seed": int(seed), "rule": rule, "status": status,
                                    **{k: m.get(k, np.nan) for k in METRICS + ["condition_number"]}})
            EVALUATED.add((name, seed))
    return pd.DataFrame(RESULT_ROWS)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    agg = {f"{m}_mean": (m, "mean") for m in METRICS}
    agg.update({f"{m}_sd": (m, "std") for m in METRICS})
    out = ok.groupby(["name", "rule"], sort=False).agg(n_seeds=("seed", "nunique"), **agg).reset_index()
    meta_cols = ["name", "family", "extraction", "stochastic", "mask_source", "mask_use", "information", "group",
                 "dataset", "parent", "critic"]
    meta = df[meta_cols].drop_duplicates("name")
    return out.merge(meta, on="name", how="left")


RESULTS = evaluate_registered(condition=True)
SUMMARY = summarize_results(RESULTS)
RESULTS.to_csv(TABLE_DIR / "exact_results_all.csv", index=False)
SUMMARY.to_csv(TABLE_DIR / "exact_results_summary.csv", index=False)
n_bad = int((RESULTS["status"] != "ok").sum())
print(f"Evaluated {RESULTS[['name', 'seed']].drop_duplicates().shape[0]} policy/seed pairs; improper: {n_bad}")

MAIN_TABLE_POLICIES = [
    "Random (uniform over 25)", "Random (uniform admissible)", "Clinician", "Clinician [projected]",
    "Behavior Cloning", "Behavior Cloning [masked greedy]", "Behavior Cloning [projected]",
    "Tabular CQL", "Tabular CQL [masked greedy]", "Tabular CQL [projected]",
    "Tabular IQL", "Tabular IQL [masked greedy]", "Tabular IQL [projected]", "Tabular IQL [stochastic AW] [projected]",
    f"Tabular BCQ (tau={CFG.bcq_threshold})", f"Tabular BCQ (tau={CFG.bcq_threshold}) [masked greedy]",
    "Deep CQL", "Deep CQL [masked greedy]", "Deep CQL [projected]",
    "AM-CQL (Hard, legacy)", "AM-CQL (Hard, design A: full-action CQL)", "AM-CQL (Hard, design B: admissible-row CQL)",
    "AM-CQL (Soft, legacy)", "AM-CQL (Soft, legacy) [masked greedy]", "AM-CQL (Soft, legacy) [projected]",
    "Neural BC (matched capacity) [masked greedy]", "AW categorical imitation [masked greedy]",
    "Masked critic ensemble [critic B + tabular CQL]", "Neural BC prior x critic [critic B + tabular CQL]",
    "Diffusion-AM-CQL [legacy critic]", "Diffusion-AM-CQL [critic A]", PRIMARY_DIFFUSION,
    "Constrained Optimal Oracle", "Optimal Oracle (mean-optimal)",
]


def mask_columns(mask_use: str) -> Dict[str, bool]:
    """Mask availability at training and at inference/extraction (audit M02: every table states it)."""
    mu = str(mask_use)
    return {"mask_at_training": "training" in mu,
            "mask_at_inference": any(t in mu for t in ("extraction", "projection", "inference", "construction"))}


def wide_table(summary: pd.DataFrame, names: Sequence[str], metrics=("return", "inadmissible_rate")) -> pd.DataFrame:
    rows = []
    for n in names:
        sub = summary[summary["name"] == n]
        if sub.empty:
            continue
        row = {"policy": n, "n_seeds": int(sub["n_seeds"].max()), "mask_source": sub["mask_source"].iloc[0],
               "mask_use": sub["mask_use"].iloc[0], **mask_columns(sub["mask_use"].iloc[0]),
               "information": sub["information"].iloc[0], "stochastic": bool(sub["stochastic"].iloc[0])}
        for rule in RULES:
            r = sub[sub["rule"] == rule]
            if r.empty:
                continue
            for m in metrics:
                mean, sd = float(r[f"{m}_mean"].iloc[0]), r[f"{m}_sd"].iloc[0]
                scale = 100.0 if m == "inadmissible_rate" else 1.0
                row[f"{m}_{rule}"] = mean * scale
                row[f"{m}_{rule}_sd"] = (float(sd) * scale) if pd.notna(sd) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


MAIN_TABLE = wide_table(SUMMARY, MAIN_TABLE_POLICIES, metrics=("return", "inadmissible_rate", "prob_any_violation",
                                                               "avg_episode_length"))
MAIN_TABLE.to_csv(TABLE_DIR / "table_main_policy_comparison.csv", index=False)
display(MAIN_TABLE[["policy", "n_seeds", "mask_use", "return_mean", "return_mean_sd", "return_terminate",
                    "return_terminate_sd", "inadmissible_rate_mean", "prob_any_violation_mean"]].round(5))

per_seed = RESULTS[(RESULTS["group"] == "main") & (RESULTS["status"] == "ok")].pivot_table(
    index=["name", "rule"], columns="seed", values="return").reset_index()
per_seed.to_csv(TABLE_DIR / "table_per_seed_returns.csv", index=False)

# ---------- Monte-Carlo cross-check through the real environment ----------
MC_ROWS = []
if EXPERIMENTS["monte_carlo_crosscheck"]:
    mc_targets = [("Clinician", FIXED), ("Random (uniform over 25)", FIXED), ("Tabular IQL", CFG.seeds[0]),
                  ("Tabular IQL [projected]", CFG.seeds[0]), (PRIMARY_DIFFUSION, CFG.seeds[0])]
    for name, seed in mc_targets:
        if name not in REGISTRY or seed not in REGISTRY[name].by_seed:
            continue
        pi = REGISTRY[name].by_seed[seed]
        for rule in RULES:
            exact = safe_evaluate(pi, rule)
            if exact is None:
                continue
            mc = monte_carlo_estimate(pi, rule, CFG.mc_episodes, f"{name}|{seed}")
            z = (mc["mc_return"] - exact["return"]) / max(mc["mc_return_se"], 1e-12)
            zL = (mc["mc_length"] - exact["avg_episode_length"]) / max(mc["mc_length_se"], 1e-12)
            # Poisson-type floor on the SE so that an all-zero Monte-Carlo sample is judged against the exact rate.
            se_v = max(mc["mc_violations_se"], math.sqrt(max(exact["expected_violations"], 1e-12) / CFG.mc_episodes))
            zV = (mc["mc_violations"] - exact["expected_violations"]) / se_v
            MC_ROWS.append({"policy": name, "seed": seed, "rule": rule, "exact_return": exact["return"], **mc,
                            "z_score": z, "exact_length": exact["avg_episode_length"], "z_length": zL,
                            "exact_expected_violations": exact["expected_violations"], "z_violations": zV,
                            "within_4_se": bool(max(abs(z), abs(zL), abs(zV)) <= 4.0)})
    MC_CHECK = pd.DataFrame(MC_ROWS)
    MC_CHECK.to_csv(TABLE_DIR / "monte_carlo_crosscheck.csv", index=False)
    display(MC_CHECK.round(5))
    if len(MC_CHECK) and not MC_CHECK["within_4_se"].all():
        print("WARNING: a Monte-Carlo estimate deviates from exact evaluation by more than 4 standard errors.")
