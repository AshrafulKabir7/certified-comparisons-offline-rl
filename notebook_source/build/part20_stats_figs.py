# %% [markdown]
# ## 30. Statistics redesigned around the experimental unit (audit M03)
#
# * **Primary analysis** (as preregistered): one comparison, one endpoint, two-sided exact sign-flip test on
#   the mean paired difference, with the effect size, a percentile bootstrap interval (flagged as fragile for
#   few pairs), the confidence set obtained by **inverting the sign-flip test**, a TOST equivalence test at the
#   declared margin, and the probability of improvement. The sign-flip test assumes sign-exchangeability
#   (symmetry) of the paired differences under the null; equal means alone do not guarantee it.
# * **Secondary family:** Holm over the distinct hypotheses (comparator × rule). When the treatment and a
#   comparator both have zero inadmissible mass, their mean- and terminate-rule paired differences coincide exactly,
#   so that comparison is counted once (`rule_collapsed_comparators`, `family_size_distinct_hypotheses`).
# * **Attainability:** with n pairs the smallest two-sided p is 2/2ⁿ (one-sided 1/2ⁿ). The full profile therefore
#   uses 10 main seeds: the two-sided minimum is 2/2¹⁰ ≈ 0.002, so the primary test can reject at 0.05 and a first
#   Holm rejection is attainable in families of up to 25 distinct hypotheses (2/2¹⁰ ≤ 0.05/25); the 8 replication
#   datasets give a dataset-level minimum of 2/2⁸ ≈ 0.008. With 5 seeds (the lite profile and the original study) the two-sided minimum is 0.0625, so no two-sided test can reject at 0.05 and the inverted confidence
#   set is unbounded — reported as such, never as "no difference". Seven pairs make a one-sided 5-comparison Holm
#   rejection attainable and eight the two-sided analogue; that is a resolution fact, not a power analysis
#   (`family_attainability` and `replication_planning.csv` report it for the realised design).
# * **Hierarchical analysis** of the replication study: dataset-level paired summaries (training seeds nested
#   within datasets), a cluster bootstrap over datasets, and a method-of-moments split into a between-dataset
#   component (truncated at 0) and the within-dataset (training-seed) variance; per-method replication summaries.
#   This is **estimation, not testing**: with 8 datasets the smallest two-sided p is 2/2⁸ ≈ 0.008, which exceeds
#   0.05/7, so no Holm-adjusted rejection is attainable over the (comparator × rule) rows. Holm-adjusted p-values,
#   the family size and attainability are reported for completeness, and the 8-cluster bootstrap is flagged as
#   fragile (`cluster_bootstrap_fragile_small_n`).
# * **Replication planning** from pilot variability of paired differences and the declared practical effect.
# * The original study's one-sided analysis is reproduced for the legacy-critic diffusion policy only
#   as a historical record: on the original five-seed artifacts, and on fresh retraining with the current main seeds.

# %%
# ============================================================
# 30. Statistics
# ============================================================
EXECLOG.mark("30. Statistics (M03)")
refresh_results()
ALPHA = float(PREREG["alpha"])


_SIGN_CACHE: Dict[int, np.ndarray] = {}


def sign_patterns(n: int) -> np.ndarray:
    if n not in _SIGN_CACHE:
        codes = np.arange(2 ** n, dtype=np.int64)[:, None]
        _SIGN_CACHE[n] = 1.0 - 2.0 * ((codes >> np.arange(n)) & 1).astype(np.float64)
    return _SIGN_CACHE[n]


def sign_flip_test(d: np.ndarray, alternative: str = "two-sided", key: str = "", exact_max_n: int = 16,
                   mc_draws: int = 200_000) -> Dict[str, Any]:
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if n < 2:
        return {"p_value": np.nan, "n_pairs": n, "method": "insufficient pairs"}
    if n <= exact_max_n:
        signs, method = sign_patterns(n), "exact enumeration"
    else:
        signs = np_rng("evaluation", "sign-flip", key, n).choice([-1.0, 1.0], size=(mc_draws, n))
        method = f"Monte Carlo ({mc_draws} draws)"
    null = signs @ d / n
    obs = d.mean()
    tol = 1e-12 * max(1.0, abs(obs))
    if alternative == "greater":
        cond = null >= obs - tol
    elif alternative == "less":
        cond = null <= obs + tol
    else:
        cond = np.abs(null) >= abs(obs) - tol
    # Exact enumeration includes the observed sign pattern; the Monte-Carlo estimate uses the +1 correction.
    p = float(cond.mean()) if n <= exact_max_n else float((1 + cond.sum()) / (1 + len(cond)))
    return {"p_value": p, "n_pairs": n, "method": method}


def min_attainable_p(n: int, alternative: str) -> float:
    return (1.0 if alternative != "two-sided" else 2.0) / (2 ** n)


def holm_adjust(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down over the finite p-values; non-finite p-values are excluded (reported as NaN)."""
    valid = {k: float(v) for k, v in p_values.items() if v is not None and np.isfinite(v)}
    items = sorted(valid.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for rank, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[name] = running
    for k in p_values:
        out.setdefault(k, np.nan)
    return out


def bootstrap_ci(d: np.ndarray, key: str) -> Tuple[float, float]:
    d = np.asarray(d, dtype=np.float64)
    if len(d) < 2:
        return (np.nan, np.nan)
    idx = np_rng("evaluation", "bootstrap", key).integers(0, len(d), size=(CFG.bootstrap_samples, len(d)))
    return tuple(np.quantile(d[idx].mean(axis=1), [ALPHA / 2, 1 - ALPHA / 2]).tolist())


def sign_flip_confidence_set(d: np.ndarray, key: str) -> Tuple[float, float]:
    d = np.asarray(d, dtype=np.float64)
    if len(d) < 2 or min_attainable_p(len(d), "two-sided") > ALPHA:
        return (-np.inf, np.inf)
    span = max(d.max() - d.min(), 1e-6)
    grid = np.linspace(d.min() - 2 * span, d.max() + 2 * span, 601)
    accept = lambda g: sign_flip_test(d - g, "two-sided", key, mc_draws=20_000)["p_value"] > ALPHA
    keep = [g for g in grid if accept(g)]
    if not keep:
        return (np.nan, np.nan)
    lo, hi = min(keep), max(keep)

    def _bisect(rejected, accepted):
        for _ in range(50):
            mid = 0.5 * (rejected + accepted)
            rejected, accepted = (rejected, mid) if accept(mid) else (mid, accepted)
        return accepted

    i_lo, i_hi = int(np.searchsorted(grid, lo)), int(np.searchsorted(grid, hi))
    lo = _bisect(grid[i_lo - 1], lo) if i_lo > 0 else -np.inf
    hi = _bisect(grid[i_hi + 1], hi) if i_hi < len(grid) - 1 else np.inf
    return (lo, hi)


def tost(d: np.ndarray, margin: float, key: str) -> float:
    d = np.asarray(d, dtype=np.float64)
    if len(d) < 2:
        return np.nan
    p_lower = sign_flip_test(d + margin, "greater", key + "|lo")["p_value"]
    p_upper = sign_flip_test(d - margin, "less", key + "|hi")["p_value"]
    return float(max(p_lower, p_upper))


def probability_of_improvement(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x), np.asarray(y)
    return float(np.mean(x[:, None] > y[None, :]) + 0.5 * np.mean(x[:, None] == y[None, :]))


def compare(a: str, b: str, rule: str, alternative: str = "two-sided", metric: str = "return",
            df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    pf = paired_frame(a, b, rule, metric, df)
    d = pf["diff"].to_numpy()
    key = f"{a}|{b}|{rule}|{metric}"
    t = sign_flip_test(d, alternative, key)
    ci = bootstrap_ci(d, key)
    cs = sign_flip_confidence_set(d, key)
    n = len(d)
    return {"treatment": a, "comparator": b, "rule": rule, "metric": metric, "alternative": alternative,
            "treatment_mask_source": REGISTRY[a].mask_source, "treatment_mask_use": REGISTRY[a].mask_use,
            "comparator_mask_source": REGISTRY[b].mask_source, "comparator_mask_use": REGISTRY[b].mask_use,
            "signflip_confidence_unbounded": bool(not (np.isfinite(cs[0]) and np.isfinite(cs[1]))),
            "n_pairs": n, "seeds": pf["seed"].tolist(), "mean_difference": float(d.mean()) if n else np.nan,
            "sd_difference": float(d.std(ddof=1)) if n > 1 else np.nan,
            "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1], "bootstrap_fragile_small_n": bool(n < 10),
            "signflip_confidence_low": cs[0], "signflip_confidence_high": cs[1],
            "p_value": t["p_value"], "test_method": t.get("method"), "min_attainable_p": min_attainable_p(n, alternative) if n else np.nan,
            "rejection_attainable_at_alpha": bool(n and min_attainable_p(n, alternative) <= ALPHA),
            "tost_p": tost(d, PREREG["equivalence_margin"], key), "equivalence_margin": PREREG["equivalence_margin"],
            "prob_improvement": probability_of_improvement(pf["a"].to_numpy(), pf["b"].to_numpy()) if n else np.nan}


STATS = {"prereg_hash": obj_hash(PREREG), "prereg": PREREG}
pt, pc = PREREG["primary_treatment"], PREREG["primary_comparator"]
if pt in REGISTRY and pc in REGISTRY:
    STATS["primary"] = compare(pt, pc, PREREG["primary_endpoint"]["rule"], PREREG["alternative"])
    print(json.dumps(STATS["primary"], indent=2, default=_json_default))

fam_rows = []
for comp in PREREG["secondary_comparators"]:
    if comp in REGISTRY and pt in REGISTRY:
        for rule in PREREG["secondary_rules"]:
            fam_rows.append(compare(pt, comp, rule, PREREG["alternative"]))
FAMILY = pd.DataFrame(fam_rows)
if len(FAMILY):
    def _same_across_rules(c):
        a_ = paired_frame(pt, c, "mean")["diff"].to_numpy()
        b_ = paired_frame(pt, c, "terminate")["diff"].to_numpy()
        return len(a_) == len(b_) and np.array_equal(a_, b_)
    same = {c: _same_across_rules(c) for c in FAMILY["comparator"].unique()}
    FAMILY["hypothesis"] = [r.comparator if same[r.comparator] else f"{r.comparator}|{r.rule}" for r in FAMILY.itertuples()]
    uniq = FAMILY.drop_duplicates("hypothesis")
    adj = holm_adjust(dict(zip(uniq["hypothesis"], uniq["p_value"])))
    FAMILY["holm_adjusted_p_full_family"] = FAMILY["hypothesis"].map(adj)
    m = int(uniq["p_value"].notna().sum())
    FAMILY["family_size_distinct_hypotheses"] = m
    FAMILY["holm_excluded_nonfinite"] = int(uniq["p_value"].isna().sum())
    if FAMILY["holm_excluded_nonfinite"].iloc[0]:
        print("WARNING: some secondary comparisons have non-finite p-values (too few pairs) and were excluded from Holm.")
    n_needed = next(n for n in range(2, 64) if min_attainable_p(n, PREREG["alternative"]) <= ALPHA / max(m, 1))
    STATS["family_attainability"] = {"family_size": m, "pairs_needed_for_first_holm_rejection": n_needed,
                                     "pairs_available": int(FAMILY["n_pairs"].min()),
                                     "rule_collapsed_comparators": [c for c, v in same.items() if v]}
FAMILY.to_csv(TABLE_DIR / "statistics_secondary_family.csv", index=False)
display(FAMILY[["comparator", "rule", "hypothesis", "n_pairs", "mean_difference", "sd_difference", "bootstrap_ci_low",
                "bootstrap_ci_high", "p_value", "holm_adjusted_p_full_family", "tost_p", "prob_improvement"]].round(5)
        if len(FAMILY) else FAMILY)

# Historical reproduction of manuscript Table 5 (one-sided, Holm within each rule).
legacy_cmp = ["AM-CQL (Hard, legacy)", "AM-CQL (Soft, legacy)", "Tabular CQL", "Tabular IQL", "Deep CQL"]
hist_rows = []
for source, treat, comps, df in [
        ("fresh retraining", "Diffusion-AM-CQL [legacy critic]", legacy_cmp, None),
        ("original artifacts", "LEGACY Diffusion-AM-CQL", [f"LEGACY {c}" for c in
                                                          ["AM-CQL (Hard)", "AM-CQL (Soft)", "Tabular CQL", "Tabular IQL", "Deep CQL"]], None)]:
    if treat not in REGISTRY:
        continue
    for rule in RULES:
        rows = [compare(treat, c, rule, "greater", df=df) for c in comps if c in REGISTRY]
        if rows:
            adj = holm_adjust({r["comparator"]: r["p_value"] for r in rows})
            for r in rows:
                r["holm_within_rule"] = adj[r["comparator"]]
                r["source"] = source
            hist_rows.extend(rows)
HISTORICAL_TABLE5 = pd.DataFrame(hist_rows)
HISTORICAL_TABLE5.to_csv(TABLE_DIR / "statistics_historical_table5_one_sided.csv", index=False)

# Paired differences for every family comparison (publication requirement).
pd.concat([paired_frame(pt, c, rule).assign(treatment=pt, comparator=c, rule=rule)
           for c in PREREG["secondary_comparators"] + [pc] if c in REGISTRY and pt in REGISTRY
           for rule in RULES] or [pd.DataFrame()]).to_csv(TABLE_DIR / "paired_differences.csv", index=False)

# Hierarchical analysis over independent datasets (training seeds nested within datasets).
HIER_ROWS, NESTED_ROWS, REPL_ROWS = [], [], []
HIER_REF = None
if EXPERIMENTS["replication"] and REPLICATION_BUNDLES:
    n_diff = sum((rep_prefix(ds) + PRIMARY_DIFFUSION) in REGISTRY for ds in REPLICATION_BUNDLES)
    HIER_REF = PRIMARY_DIFFUSION if n_diff >= 2 else PREREG["primary_comparator"]
    for method in REPL_METHODS:
        if method == HIER_REF:
            continue
        for rule in RULES:
            per_ds, nested = [], []
            for ds in REPLICATION_BUNDLES:
                a, b = f"{rep_prefix(ds)}{HIER_REF}", f"{rep_prefix(ds)}{method}"
                if a not in REGISTRY or b not in REGISTRY:
                    continue
                pf = paired_frame(a, b, rule)
                pf = pf[pf["seed"].isin(CFG.replication_train_seeds)]
                if len(pf):
                    per_ds.append(float(pf["diff"].mean()))
                    nested.append(pf["diff"].to_numpy())
                    NESTED_ROWS.append(pf.assign(treatment=HIER_REF, comparator=method, rule=rule, dataset=ds))
            if len(per_ds) < 2:
                continue
            d = np.asarray(per_ds)
            t = sign_flip_test(d, "two-sided", f"hier|{method}|{rule}")
            # Cluster bootstrap: datasets resampled, training seeds kept intact within each dataset.
            boots = d[np_rng("evaluation", "hier-bootstrap", method, rule).integers(0, len(d), size=(CFG.bootstrap_samples, len(d)))].mean(axis=1)
            within_list = [x.var(ddof=1) for x in nested if len(x) > 1]
            within = float(np.mean(within_list)) if within_list else np.nan
            k_h = len(nested) / sum(1.0 / len(x) for x in nested)
            var_means = float(d.var(ddof=1))
            between = max(0.0, var_means - within / k_h) if np.isfinite(within) else np.nan
            HIER_ROWS.append({"treatment": HIER_REF, "comparator": method, "rule": rule,
                              "n_datasets": len(per_ds), "training_seeds_per_dataset": len(CFG.replication_train_seeds),
                              "mean_of_dataset_means": float(d.mean()), "sd_dataset_means": float(d.std(ddof=1)),
                              "cluster_bootstrap_ci_low": float(np.quantile(boots, ALPHA / 2)),
                              "cluster_bootstrap_ci_high": float(np.quantile(boots, 1 - ALPHA / 2)),
                              "dataset_level_signflip_p_two_sided": t["p_value"],
                              "min_attainable_p": min_attainable_p(len(per_ds), "two-sided"),
                              "variance_of_dataset_means": var_means, "mean_within_dataset_variance": within,
                              "between_dataset_variance_component_mom": between,
                              "icc_dataset": (between / (between + within) if np.isfinite(between) and (between + within) > 0 else np.nan)})
    # Per-method dataset / training-seed variance (independent of any reference policy).
    ok_ = RESULTS[RESULTS["status"] == "ok"]
    for method in REPL_METHODS:
        for rule in RULES:
            per = []
            for ds in REPLICATION_BUNDLES:
                s_ = ok_[(ok_["name"] == rep_prefix(ds) + method) & (ok_["rule"] == rule)
                         & ok_["seed"].isin(list(CFG.replication_train_seeds) + [FIXED])]
                if len(s_):
                    v = s_["return"].to_numpy(dtype=float)
                    per.append(v)
                    REPL_ROWS.append({"method": method, "rule": rule, "dataset_seed": ds, "n_seeds": len(v),
                                      "return_mean": float(v.mean()), "return_sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan})
            if len(per) >= 2:
                means = np.array([v.mean() for v in per])
                w = [v.var(ddof=1) for v in per if len(v) > 1]
                within = float(np.mean(w)) if w else np.nan
                vm = float(means.var(ddof=1))
                k_h = len(per) / sum(1.0 / len(v) for v in per)
                REPL_ROWS.append({"method": method, "rule": rule, "dataset_seed": "pooled", "n_datasets": len(per),
                                  "mean_of_dataset_means": float(means.mean()), "variance_of_dataset_means": vm,
                                  "mean_within_dataset_variance": within,
                                  "between_dataset_variance_component_mom": (max(0.0, vm - within / k_h) if np.isfinite(within) else np.nan)})
HIERARCHICAL = pd.DataFrame(HIER_ROWS)
if len(HIERARCHICAL):
    # Estimation, not testing: with 8 datasets the smallest two-sided p is 2/2^8 = 0.0078 > 0.05/7, so no Holm-adjusted
    # rejection is attainable over these rows, and a cluster bootstrap over fewer than 10 datasets is fragile.
    _hkey = HIERARCHICAL["comparator"].astype(str) + "|" + HIERARCHICAL["rule"].astype(str)
    _hp = HIERARCHICAL["dataset_level_signflip_p_two_sided"]
    HIERARCHICAL["holm_p"] = _hkey.map(holm_adjust(dict(zip(_hkey, _hp))))
    _hm = int(np.isfinite(_hp.to_numpy(dtype=np.float64)).sum())
    HIERARCHICAL["family_size"] = _hm
    HIERARCHICAL["holm_first_step_attainable"] = HIERARCHICAL["min_attainable_p"] <= ALPHA / max(_hm, 1)
    HIERARCHICAL["cluster_bootstrap_fragile_small_n"] = HIERARCHICAL["n_datasets"] < 10
    HIERARCHICAL["analysis_role"] = "estimation (effect sizes with intervals; not a confirmatory test)"
HIERARCHICAL.to_csv(TABLE_DIR / "statistics_hierarchical_replication.csv", index=False)
(pd.concat(NESTED_ROWS, ignore_index=True) if NESTED_ROWS else pd.DataFrame()).to_csv(
    TABLE_DIR / "statistics_hierarchical_paired_differences.csv", index=False)
REPLICATION_RESULTS = pd.DataFrame(REPL_ROWS)
REPLICATION_RESULTS.to_csv(TABLE_DIR / "replication_results.csv", index=False)
if len(HIERARCHICAL):
    display(HIERARCHICAL.round(5))
if len(REPLICATION_RESULTS):
    display(REPLICATION_RESULTS[REPLICATION_RESULTS["dataset_seed"] == "pooled"].round(6))

# Replication planning (planning aid, not a universal seed rule).
plan_rows = []
z_b = stats.norm.ppf(0.8)
m_fam = int(STATS.get("family_attainability", {}).get("family_size", 1) or 1)


def _plan(r, label, z_a):
    sd = r["sd_difference"]
    if np.isfinite(sd) and sd > 0:
        plan_rows.append({"analysis": label, "comparator": r["comparator"], "rule": r["rule"],
                          "pilot_sd_paired_difference": sd, "practical_effect": PREREG["practical_effect"],
                          "alpha_used": float(2 * (1 - stats.norm.cdf(z_a))),
                          "approx_pairs_for_80pct_power_normal": int(math.ceil(((z_a + z_b) * sd / PREREG["practical_effect"]) ** 2)),
                          "min_pairs_attainable_single_two_sided": next(n for n in range(2, 64) if 2 / 2 ** n <= ALPHA),
                          "min_pairs_attainable_holm_family": STATS.get("family_attainability", {}).get("pairs_needed_for_first_holm_rejection"),
                          "unit": "training seeds on the frozen main dataset"})


if "primary" in STATS:
    _plan(STATS["primary"], "primary (unadjusted, two-sided)", stats.norm.ppf(1 - ALPHA / 2))
for r in fam_rows:
    _plan(r, "secondary (Bonferroni bound on the first Holm step)", stats.norm.ppf(1 - ALPHA / (2 * m_fam)))
REPLICATION_PLAN = pd.DataFrame(plan_rows)
REPLICATION_PLAN.to_csv(TABLE_DIR / "replication_planning.csv", index=False)
STATS["hierarchical_available"] = bool(len(HIERARCHICAL))
save_json(STATS, TABLE_DIR / "statistics_primary.json")

# Structural support gate (renamed from the earlier "safety" gate; benchmark-support terminology).
support_gate = {"primary_policy": PRIMARY_DIFFUSION, "seeds": list(CFG.seeds)}
for nm in [PRIMARY_DIFFUSION, "Diffusion-AM-CQL [legacy critic]", "AM-CQL (Hard, design B: admissible-row CQL)",
           "AM-CQL (Hard, legacy)", "Tabular IQL [projected]"]:
    sub = RESULTS[(RESULTS["name"] == nm) & (RESULTS["status"] == "ok")]
    if len(sub):
        support_gate[f"max_exact_inadmissible_rate::{nm}"] = float(sub["inadmissible_rate"].max())
support_gate["max_reverse_step_inadmissible_mass_primary"] = float(max(
    (max(v) for v in DIFFUSION_TRACES.get(PRIMARY_DIFFUSION, {}).values()), default=0.0))
support_gate["passes_structural_support_check"] = bool(all(
    v <= 1e-12 for k, v in support_gate.items() if k.startswith("max_")))
save_json({"support_gate": support_gate, "primary": STATS.get("primary"),
           "family": FAMILY.to_dict(orient="records") if len(FAMILY) else []},
          TABLE_DIR / "fresh_custom_model_performance_gate.json")
print(json.dumps(support_gate, indent=2))

# %% [markdown]
# ## 31. Figures (original-study figures regenerated from code, plus new diagnostic figures)

# %%
# ============================================================
# 31. Figures
# ============================================================
EXECLOG.mark("31. Figures")


def summary_value(name: str, rule: str, metric: str = "return"):
    r = SUMMARY[(SUMMARY["name"] == name) & (SUMMARY["rule"] == rule)]
    if r.empty:
        return np.nan, np.nan
    sd = r[f"{metric}_sd"].iloc[0]
    return float(r[f"{metric}_mean"].iloc[0]), (float(sd) if pd.notna(sd) else 0.0)


fig7_policies = [n for n in ["Behavior Cloning", "Tabular CQL", "Tabular IQL", f"Tabular BCQ (tau={CFG.bcq_threshold})",
                             "Deep CQL", "AM-CQL (Hard, legacy)", "AM-CQL (Hard, design A: full-action CQL)",
                             "AM-CQL (Hard, design B: admissible-row CQL)", "AM-CQL (Soft, legacy)",
                             "Diffusion-AM-CQL [legacy critic]", PRIMARY_DIFFUSION] if n in REGISTRY]
if fig7_policies:
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(fig7_policies) + 2))
    y = np.arange(len(fig7_policies))
    for j, rule in enumerate(RULES):
        vals = [summary_value(n, rule) for n in fig7_policies]
        axes[0].errorbar([v[0] for v in vals], y + (0.15 if j else -0.15), xerr=[v[1] for v in vals], fmt="o",
                         capsize=3, label=rule)
        axes[1].barh(y + (0.2 if j else -0.2), [100 * summary_value(n, rule, "inadmissible_rate")[0] for n in fig7_policies],
                     height=0.4, label=rule)
    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(fig7_policies, fontsize=8)
        ax.invert_yaxis()
        ax.legend()
    axes[0].set_xlabel("Exact expected return (mean ± SD over seeds)")
    axes[0].set_title("(a) Return under two handling rules")
    axes[1].set_xlabel("Inadmissible decisions (% of occupancy)")
    axes[1].set_title("(b) Benchmark-support violations")
    save_figure(fig, FIG_DIR / "paper_fig7_main_policies")

# C01 projection figure: native vs projected vs masked greedy, both rules.
c01_families = [("Behavior Cloning", "Behavior Cloning [projected]", "Behavior Cloning [masked greedy]"),
                ("Tabular CQL", "Tabular CQL [projected]", "Tabular CQL [masked greedy]"),
                ("Tabular IQL", "Tabular IQL [projected]", "Tabular IQL [masked greedy]"),
                ("Deep CQL", "Deep CQL [projected]", "Deep CQL [masked greedy]"),
                ("Clinician", "Clinician [projected]", None)]
c01_families = [f for f in c01_families if f[0] in REGISTRY]
if c01_families:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for j, rule in enumerate(RULES):
        ax = axes[j]
        x = np.arange(len(c01_families))
        for k, (lab, idx) in enumerate((("native", 0), ("projected", 1), ("masked greedy", 2))):
            vals = [summary_value(f[idx], rule) if f[idx] and f[idx] in REGISTRY else (np.nan, 0) for f in c01_families]
            ax.bar(x + (k - 1) * 0.27, [v[0] for v in vals], 0.27, yerr=[v[1] for v in vals], capsize=2, label=lab)
        d_val = summary_value(PRIMARY_DIFFUSION, rule)[0]
        if np.isfinite(d_val):
            ax.axhline(d_val, linestyle="--", color="k", linewidth=1, label="primary diffusion")
        ax.set_xticks(x)
        ax.set_xticklabels([f[0] for f in c01_families], rotation=20, ha="right")
        ax.set_title(f"Rule: {rule}")
        ax.set_ylim(0.70 if rule == "mean" else 0.35, 0.80)
    axes[0].set_ylabel("Exact expected return")
    axes[1].legend(fontsize=8)
    fig.suptitle("Admissible projection and masked extraction of baseline policies (audit C01)")
    save_figure(fig, FIG_DIR / "fig_c01_projection_controls")

if len(TABLE7):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for critic_label, g in TABLE7.groupby("critic"):
        axes[0].errorbar(g["guidance_g"], g["return_mean"], yerr=g["return_sd"], marker="o", capsize=3, label=critic_label)
    axes[0].set_xlabel("Guidance scale g")
    axes[0].set_ylabel("Exact expected return (terminate)")
    axes[0].set_title("(a) Post-training guidance sensitivity")
    axes[0].legend()
    t6 = TABLE6_PRIMARY.dropna(subset=["return_terminate"]) if "return_terminate" in TABLE6_PRIMARY else TABLE6_PRIMARY.iloc[0:0]
    axes[1].errorbar(t6["return_terminate"], np.arange(len(t6)), xerr=t6["return_terminate_sd"].fillna(0), fmt="o", capsize=3)
    axes[1].set_yticks(np.arange(len(t6)))
    axes[1].set_yticklabels(t6["control"], fontsize=7)
    axes[1].invert_yaxis()
    ref = summary_value(PRIMARY_DIFFUSION, "terminate")[0]
    axes[1].axvline(ref, linestyle="--", color="k", linewidth=1)
    axes[1].set_xlabel("Exact expected return (terminate)")
    axes[1].set_title("(b) Inference components and direct alternatives")
    save_figure(fig, FIG_DIR / "paper_fig8_guidance_and_components")

panels = [("Tabular CQL", "td_loss", "(a) Tabular CQL: squared TD"), ("Tabular IQL", "q_loss", "(b) Tabular IQL: squared Q"),
          ("Deep CQL", "td_loss", "(c) Deep CQL: twin Huber"), ("AM-CQL (Hard, legacy)", "td_loss", "(d) Hard legacy: twin Huber"),
          ("AM-CQL (Hard, design A: full-action CQL)", "td_loss", "(e) Hard design A"),
          ("AM-CQL (Hard, design B: admissible-row CQL)", "td_loss", "(f) Hard design B"),
          ("AM-CQL (Soft, legacy)", "td_loss", "(g) Soft: twin Huber"), ("Diffusion prior", "loss", "(h) Diffusion: denoising CE")]
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for ax, (stem, col, title) in zip(axes.ravel(), panels):
    for seed in CFG.seeds:
        f = LOG_DIR / f"{slug(stem)}__seed{seed}.csv"
        if f.exists():
            df = pd.read_csv(f)
            if col in df:
                ax.plot(df["step"] / 1000, df[col], linewidth=0.8, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Updates (thousands)")
    ax.set_ylabel("Minibatch loss")
save_figure(fig, FIG_DIR / "paper_fig9_training_diagnostics")

soft_logs = [(seed, LOG_DIR / f"{slug('AM-CQL (Soft, legacy)')}__seed{seed}.csv") for seed in CFG.seeds]
if any(f.exists() for _, f in soft_logs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for seed, f in soft_logs:
        if f.exists():
            df = pd.read_csv(f)
            axes[0].plot(df["step"] / 1000, df["lambda"], label=f"seed {seed}")
            axes[1].plot(df["step"] / 1000, 100 * df["inadmissible_mass_surrogate"], label=f"seed {seed}")
    axes[0].set_title("(a) Soft constraint multiplier λ")
    axes[1].set_title("(b) Minibatch softmax inadmissible mass (%)")
    for ax in axes:
        ax.set_xlabel("Updates (thousands)")
        ax.legend(fontsize=7)
    save_figure(fig, FIG_DIR / "paper_fig10_soft_diagnostics")

diag_path = TABLE_DIR / "critic_diagnostics.csv"
if diag_path.exists():
    dg = pd.read_csv(diag_path)
    dg = dg[dg["dataset"] == "main"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for key in ["hard_legacy", "hard_fullcql", "hard_admrows", "deep", "soft_legacy", "hard_legacy_clip01", "hard_admrows_tdadm"]:
        g = dg[dg["variant"] == key]
        if g.empty:
            continue
        m = g.groupby("step")[["q_adm_max_abs", "bellman_residual_rmse_full_buffer"]].median().reset_index()
        axes[0].plot(m["step"] / 1000, m["q_adm_max_abs"], marker="o", label=key)
        axes[1].plot(m["step"] / 1000, m["bellman_residual_rmse_full_buffer"], marker="o", label=key)
    for ax, t in zip(axes, ["(a) max |Q| over admissible actions (median over seeds)", "(b) Full-buffer Bellman RMSE"]):
        ax.set_yscale("log")
        ax.set_xlabel("Updates (thousands)")
        ax.set_title(t)
        ax.legend(fontsize=7)
    save_figure(fig, FIG_DIR / "fig_c02_critic_scale_drift")

if len(HIERARCHICAL):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    hh = HIERARCHICAL[HIERARCHICAL["rule"] == "terminate"]
    ax.errorbar(hh["mean_of_dataset_means"], np.arange(len(hh)),
                xerr=[(hh["mean_of_dataset_means"] - hh["cluster_bootstrap_ci_low"]).clip(lower=0),
                      (hh["cluster_bootstrap_ci_high"] - hh["mean_of_dataset_means"]).clip(lower=0)],
                fmt="o", capsize=3)
    ax.axvline(0, color="k", linewidth=1)
    ax.set_yticks(np.arange(len(hh)))
    ax.set_yticklabels(hh["comparator"], fontsize=8)
    ax.set_xlabel(f"{HIER_REF} minus comparator (terminate); cluster-bootstrap interval over datasets")
    ax.set_title("Replication across independent datasets (training seeds nested)")
    save_figure(fig, FIG_DIR / "fig_m03_hierarchical_replication")

if EXPERIMENTS["selection_protocol"] and "CANDIDATES" in globals():
    fig, ax = plt.subplots(figsize=(6, 5))
    for method, g in CANDIDATES[CANDIDATES["rule"] == PREREG["primary_endpoint"]["rule"]].groupby("method"):
        ax.scatter(g["exact_return"], g["fqe_dev_return"], label=method)
    lim = [np.nanmin(CANDIDATES[["exact_return", "fqe_dev_return"]].to_numpy()), np.nanmax(CANDIDATES[["exact_return", "fqe_dev_return"]].to_numpy())]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_xlabel("Exact return (true MDP)")
    ax.set_ylabel("Tabular FQE estimate (dev episodes)")
    ax.set_title("OPE selector validation (audit M04)")
    ax.legend(fontsize=8)
    save_figure(fig, FIG_DIR / "fig_m04_fqe_vs_exact")

if EXPERIMENTS["mdp_stress_tests"] and "STRESS_RANKING" in globals() and len(STRESS_RANKING):
    fig, ax = plt.subplots(figsize=(9, 4))
    sr = STRESS_RANKING.groupby(["perturbation", "param", "rule"])["kendall_tau_vs_unperturbed"].mean().reset_index()
    labels = [f"{r.perturbation}({r.param})\n{r.rule}" for r in sr.itertuples()]
    ax.bar(np.arange(len(sr)), sr["kendall_tau_vs_unperturbed"])
    ax.set_xticks(np.arange(len(sr)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Kendall τ vs unperturbed ranking")
    ax.set_title("Ranking persistence under MDP stress tests (not confidence sets)")
    save_figure(fig, FIG_DIR / "fig_m07_stress_ranking")
print("Figures written to", FIG_DIR)
