# %% [markdown]
# ### II.8g Estimation-aware intervals: bootstrap ensembles of the hospital data
#
# The bracketing theorem holds for the **true** supported rows: J_term ≤ J* for every truth, and J* ≤ J_mean under no
# hidden benefit. A benchmark has only **estimated** rows, so [J_bench_term, J_bench_mean] also carries estimation
# error. This section measures how much of the coverage gap that error explains, using a nonparametric bootstrap of
# the hospital data behind each benchmark.
#
# **Cells** (`XV3BootConfig.cells`): Gumbel `best` with n = 2 000, τ = 5 and with n = 10 000, τ = 20; the
# withhold-vaso physician cell; and Garnet h ∈ {−0.5, 0, 0.5, 1} at ε = 0.05. Every draw of each cell is used. A
# configured cell that is not in `X_CELLS`, because its family switch is off, is skipped and listed in the summary.
#
# **Per (cell, draw).**
# 1. Rebuild the II.6 instance (`build_cell_instance`). Regenerate the **default** policies by running `run_xmethods`
#    on the same training split under the same key as II.6. Part II learners are deterministic functions of their
#    data, so these are the II.6 policies. The behaviour policy and the projections are added by `x_collect_records`.
# 2. Regenerate the hospital episodes with the same stream and call as II.3d (`x_boot_hospital`). Rebuild the
#    benchmark with `x_benchmark_from_data`; its identity must equal the instance's benchmark (asserted).
# 3. `x_boot_step_map` records where each logged hospital step enters the count tensor of that benchmark. It
#    replicates the censoring of `x_benchmark_from_data`: informative (non-timeout) steps, dead-state removal (drop),
#    trap censoring and dangling redirects. It also records the benchmark index of every initial state.
#    `x_boot_model` applied to unit weights must reproduce the benchmark's P and d0 (asserted to 1e-14).
# 4. For b = 0, …, B − 1, resample the hospital **episodes** with replacement (stream
#    `("xenv", "bootstrap", cell, draw, b)`). `x_boot_model` then re-estimates the rows of the supported pairs from
#    the resampled steps, with the original decision states, mask, censoring and redirects; for Gumbel it uses the
#    same known-horizon form h·e_timeout + (1 − h)·C/N. Unsupported rows follow the mean rule. A supported pair with
#    no resampled step keeps its original row (counted as `fallback_pairs`). With `resample_d0` (default) the
#    empirical d0 is re-estimated from the resampled episodes as well, since it is part of the estimation error.
# 5. Evaluate every default policy exactly under both rules on each replicate, and on the original benchmark and the
#    true system (`true`, lifted). A replicate model can contain a trap (rows estimated from few steps may miss every
#    absorbing outcome); a policy that does not absorb there is reported as `improper`, or `near-singular` when the
#    evaluator's residual checks fail, and that replicate is left out of its interval.
#
# **Invariants asserted per draw** (re-run on cached tables): the original benchmark and the truth are proper for
# every policy; on every replicate the two rules share P(any unsupported decision) (1e-9); and
# 0 ≤ J_mean^b − J_term^b ≤ p^b·(v_max − v_min) (Theorem part iii, tolerance 1e-9).
#
# The interval of a policy is [q_2.5%(J_term^b), q_97.5%(J_mean^b)] over its proper replicates (II.8i). With B = 10
# these quantiles are close to the replicate minimum and maximum (linear interpolation). II.8i compares the coverage
# and the certified comparisons of these intervals with those of the point interval [J_bench_term, J_bench_mean].
#
# Results are cached in `XCACHE` under `v3boot/<cell>/draw<k>` (and `/meta`). The identity is the II.6 draw identity
# (`x_draw_identity`, including `XENV_CODE_HASH`) plus the code hash of this section, B and `resample_d0`. The
# quantile levels configure the analysis only. As a check that the regenerated policies are the II.6 policies,
# `x_boot_point_check` compares the replicate −1 values with the `XRESULTS` rows of the same draws. It reports and
# does not assert: on a different machine, BLAS rounding can break a near-tie differently.
#
# | Output | Content |
# |---|---|
# | `xenv/v3_bootstrap_config.json` | `XV3BootConfig` |
# | `xenv/v3_bootstrap_results.csv.gz` | one row per draw × default policy × replicate × target (replicate −1: original benchmark and truth) |
# | `xenv/v3_bootstrap_meta.csv` | per draw: fallback pairs, re-estimated d0 support, trap states and improper counts per replicate, reproduction check, timings |
# | `xenv/v3_bootstrap_summary.json` | cells run and skipped, cache use, invariant checks, the point check against `XRESULTS` |
#
# **Expected cost on Colab.** These are estimates. Local harness timings were scaled by the Colab / local ratios of
# II.8f. Each draw is an instance rebuild, one `run_xmethods` fit, and 2B + 3 exact evaluations of the 50 default
# policies (the benchmark models are small). That is about 6–12 s per Gumbel draw and 4–5 s per Garnet draw: roughly
# 20–25 min for 7 cells × 30 draws. A cached draw reloads in about 3 s, so a re-run costs about 10 min.

# %%
# ============================================================
# II.8g Bootstrap-ensemble intervals (estimation-aware brackets)
# ============================================================
EXECLOG.mark("II.8 Revision experiments: bootstrap intervals")


@dataclass
class XV3BootConfig:
    cells: Tuple[str, ...] = ("gumbel|best|n2000|tau5|eps0.05", "gumbel|best|n10000|tau20|eps0.05",
                              "gumbel|best|n10000|tau20|eps0.05|phys-withhold-vaso", "garnet|h-0.5|eps0.05",
                              "garnet|h0.0|eps0.05", "garnet|h0.5|eps0.05", "garnet|h1.0|eps0.05")
    replicates: int = 10                                  # B bootstrap resamples of the hospital episodes
    quantiles: Tuple[float, float] = (0.025, 0.975)       # interval = [q_lo(J_term^b), q_hi(J_mean^b)] (analysis only)
    resample_d0: bool = True                              # re-estimate the empirical d0 from the resampled episodes


def apply_xv3_boot_run_mode(c: XV3BootConfig, mode: str) -> XV3BootConfig:
    if mode == "smoke":
        c.cells = ("gumbel|best|n300|tau5|eps0.05", "garnet|h0.0|eps0.05", "garnet|h1.0|eps0.05")
        c.replicates = 3
    return c


XV3BOOT = apply_xv3_boot_run_mode(XV3BootConfig(), RUN_MODE)
save_json(asdict(XV3BOOT), XENV_TABLE_DIR / "v3_bootstrap_config.json")
XV3_BOOT_POINT = -1                     # replicate label of the original benchmark and of the true system
XV3_EVAL_STATUSES = ("ok", "improper", "near-singular")
XV3_BOOT_COLUMNS = ["cell", "family", "draw", "policy", "method", "variant", "hp", "is_default", "method_family",
                    "replicate", "target", "semantics", "status", "return", "prob_any_violation",
                    "expected_violations", "avg_episode_length"]
XV3_TOL = 1e-9                          # value tolerance of the bracketing / width checks (as in II.8i)


def _x_v3_eval(env: FiniteEnv, pi: np.ndarray, semantics: str) -> Dict[str, Any]:
    """Exact evaluation that never raises on a model-specific failure: ok / improper / near-singular."""
    try:
        out = env.evaluator().evaluate(pi, _X_STRATEGY[semantics])
    except ImproperPolicyError as exc:
        return {"status": "improper", "error": str(exc)[:200]}
    except EvaluationInvariantError as exc:
        return {"status": "near-singular", "error": str(exc)[:200]}
    return {"status": "ok", "return": float(out["return"]), "prob_any_violation": float(out["prob_any_violation"]),
            "expected_violations": float(out["expected_violations"]),
            "avg_episode_length": float(out["avg_episode_length"])}


def x_boot_hospital(spec: dict, draw: int) -> Tuple[FiniteEnv, Dict[str, np.ndarray]]:
    """True system and hospital episodes of a draw of a II.3d `best` cell (same stream and call as construct_benchmark)."""
    cell, family = spec["cell"], spec["family"]
    rng_h = np_rng("xenv", "hospital", cell, int(draw))
    if family == "gumbel":
        assert spec["construction"] == "best", f"{cell}: bootstrap needs the estimated ('best') construction"
        true_env = _x_gumbel_env(spec["variant"], spec["timeout_hazard"])
        beh = _x_gumbel_physician(spec["variant"], spec["timeout_hazard"], spec["eps"], spec.get("physician", "optimal"))
    else:
        assert family == "garnet", family
        true_env, beh = _x_garnet_env(spec, draw)
    hosp = x_sample_dataset(true_env, beh, int(spec["n_hosp"]), XCFG.max_steps, rng_h)
    return true_env, hosp


def x_boot_step_map(true_env: FiniteEnv, hosp: Dict[str, np.ndarray], tau: int, B: Dict[str, Any]) -> Dict[str, Any]:
    """Where each logged hospital step enters the count tensor of benchmark B (x_benchmark_from_data on these steps).

    Returns the flat index (bench state, action, bench next state) of every step, or -1 if the estimate does not use it,
    and the benchmark index of every episode's initial state (-1 if censored). Replicates the censoring steps of
    x_benchmark_from_data: informative steps, dead-state removal (rule 'icu'), trap censoring and redirects."""
    A = true_env.A
    rep = B["report"]
    rule = str(rep["empty_rule"])
    assert rule in ("icu", "argmax"), rule
    s, a, ns = hosp["state"], hosp["action"], hosp["next_state"]
    hazard = true_env.meta["censoring_hazard"]
    informative = (np.ones(len(s), dtype=bool) if hazard is None else ns != int(true_env.meta["timeout_state"]))
    N_inf = _x_hospital_counts(true_env, {"state": s[informative], "action": a[informative]})
    mask_count, _ = x_count_mask(N_inf, tau, true_env.live, fallback=rule == "argmax")
    redirect = _x_dangling_target(true_env)
    logged = np.unique(s)
    V = logged[N_inf[logged].sum(axis=1) > 0]
    dead = np.zeros(0, dtype=np.int64)
    rounds = 0
    if rule == "icu":
        est = _x_estimate_on(true_env, hosp, V, mask_count, redirect, informative, drop_into=dead)
        while est["dead"].any():
            dead = np.sort(np.concatenate([dead, V[est["dead"]]]))
            V = V[~est["dead"]]
            rounds += 1
            est = _x_estimate_on(true_env, hosp, V, mask_count, redirect, informative, drop_into=dead)
        del est
    assert (len(dead), rounds) == (rep["dead_states_removed"], rep["dead_state_rounds"]), \
        f"dead-state removal differs from the benchmark report: {(len(dead), rounds)}"
    V_final = np.asarray(B["decision_states_true"], dtype=np.int64)
    trap = np.setdiff1d(V, V_final)
    assert np.isin(V_final, V).all() and len(trap) == rep["trap_states_censored"], "trap censoring differs"
    index = np.asarray(B["index"], dtype=np.int64)
    nV, Sb = len(V_final), len(index)
    assert np.array_equal(index[:nV], V_final), "benchmark index must start with the decision states"
    to_bench = np.full(true_env.S, -1, dtype=np.int64)
    to_bench[index] = np.arange(Sb)
    keep = to_bench[s] >= 0
    use = keep & informative
    if rule == "icu":
        use &= ~np.isin(ns, dead)
    ns_b = to_bench[ns]
    ns_b[ns_b < 0] = to_bench[redirect]
    flat = np.full(len(s), -1, dtype=np.int64)
    flat[use] = (to_bench[s[use]] * A + a[use]) * Sb + ns_b[use]
    first = hosp["timestep"] == 0
    init = np.full(len(s), -1, dtype=np.int64)
    init[first & keep] = to_bench[s[first & keep]]
    return {"flat": flat, "init": init, "episode": np.asarray(hosp["episode_id"], dtype=np.int64), "Sb": int(Sb),
            "nV": int(nV), "hazard": hazard,
            "timeout_bench": None if hazard is None else int(to_bench[int(true_env.meta["timeout_state"])]),
            "dead_states": int(len(dead)), "trap_states": int(len(trap)), "steps": int(len(s)),
            "steps_used": int(use.sum())}


def x_boot_weights(n_episodes: int, rng: np.random.Generator) -> np.ndarray:
    """Multiplicity of each episode in a resample of n_episodes episodes with replacement."""
    return np.bincount(rng.integers(0, int(n_episodes), size=int(n_episodes)), minlength=int(n_episodes)).astype(np.float64)


def x_boot_model(bench: FiniteEnv, smap: Dict[str, Any], w_steps: np.ndarray,
                 resample_d0: bool) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """(P, d0, stats) of `bench` with the supported rows re-estimated from weighted hospital steps.

    Mask, decision states, censoring and redirects are those of `bench`; with a known censoring hazard h the rows are
    h * e_timeout + (1 - h) * C / N (as in _x_estimate_on); unsupported rows follow the mean rule; a supported pair
    without any weighted step keeps its row of `bench`. Unit weights reproduce bench.P and bench.d0."""
    Sb, A = bench.S, bench.A
    assert smap["Sb"] == Sb and len(w_steps) == len(smap["flat"])
    w = np.asarray(w_steps, dtype=np.float64)
    used = smap["flat"] >= 0
    C = np.bincount(smap["flat"][used], weights=w[used], minlength=Sb * A * Sb).reshape(Sb, A, Sb)
    N = C.sum(axis=2)
    adm = bench.mask & bench.live[:, None]
    have = adm & (N > 0)
    P = np.zeros((Sb, A, Sb))
    P[have] = C[have] / N[have][:, None]
    del C
    if smap["hazard"] is not None:
        c = int(smap["timeout_bench"])
        P[have] *= 1.0 - float(smap["hazard"])
        P[have, c] += float(smap["hazard"])
    miss = adm & ~have
    P[miss] = bench.P[miss]
    P = x_impute_inadmissible(P, bench.mask, bench.live)
    for t in bench.terminals:
        P[t, :, t] = 1.0
    if resample_d0:
        ini = smap["init"] >= 0
        d0 = np.bincount(smap["init"][ini], weights=w[ini], minlength=Sb).astype(np.float64)
        assert d0.sum() > 0 and np.all(d0[~bench.live] == 0)
        d0 /= d0.sum()
    else:
        d0 = bench.d0.copy()
    stats_ = {"admissible_pairs": int(adm.sum()), "fallback_pairs": int(miss.sum()),
              "d0_support_states": int((d0 > 0).sum()), "steps_weight": float(w[used].sum())}
    return P, d0, stats_


def x_boot_identity(spec: dict, inst: CellInstance) -> dict:
    return {"kind": "xenv_v3_bootstrap", "draw": x_draw_identity(spec, inst), "code": XV3_BOOT_CODE_HASH,
            "replicates": int(XV3BOOT.replicates), "resample_d0": bool(XV3BOOT.resample_d0),
            "rng": ["xenv", "bootstrap", "<cell>", "<draw>", "<b>"], "columns": XV3_BOOT_COLUMNS,
            "policies": "default (is_default) records of x_collect_records"}


def x_boot_check_rows(df: pd.DataFrame, v_range: float) -> Dict[str, Any]:
    """Invariants of one draw's bootstrap rows (see markdown); asserted, and a small report is returned."""
    bad = sorted(set(df["status"].astype(str)) - set(XV3_EVAL_STATUSES))
    assert not bad, f"unexpected bootstrap status values {bad}"
    pt = df[df["replicate"] == XV3_BOOT_POINT]
    assert (pt["status"] == "ok").all(), "a default policy is not proper on the original benchmark or the truth"
    dup = df.duplicated(["policy", "replicate", "target"])
    assert not dup.any(), f"{int(dup.sum())} duplicated (policy, replicate, target) bootstrap rows"
    sub = df[df["target"].isin(["bench_mean", "bench_terminate"])].set_index(["policy", "replicate", "target"])
    J, Pv = sub["return"].unstack("target"), sub["prob_any_violation"].unstack("target")
    St = sub["status"].unstack("target")
    assert set(St.columns) == {"bench_mean", "bench_terminate"} and St.notna().all().all(), "missing bootstrap rows"
    both = ((St["bench_mean"] == "ok") & (St["bench_terminate"] == "ok")).to_numpy()
    jm, jt = J["bench_mean"].to_numpy(float)[both], J["bench_terminate"].to_numpy(float)[both]
    pm, ptm = Pv["bench_mean"].to_numpy(float)[both], Pv["bench_terminate"].to_numpy(float)[both]
    width = jm - jt
    dp = float(np.abs(pm - ptm).max(initial=0.0))
    slack_lo = float((-width).max(initial=-np.inf))
    slack_hi = float((width - ptm * float(v_range)).max(initial=-np.inf))
    assert dp <= XV3_TOL, f"the two rules disagree on P(any unsupported decision) by {dp:.3g}"
    assert slack_lo <= XV3_TOL, f"J_mean < J_term on a bootstrap model by {slack_lo:.3g} (Theorem part iii)"
    assert slack_hi <= XV3_TOL, f"J_mean - J_term exceeds p (v_max - v_min) by {slack_hi:.3g} (Theorem part iii)"
    nonok = df[(df["status"] != "ok")].groupby("target").size()
    return {"rows": int(len(df)), "pairs_checked": int(both.sum()), "nonok_rows": int((df["status"] != "ok").sum()),
            "nonok_by_target": {str(k): int(v) for k, v in nonok.items()}, "max_rule_p_difference": dp,
            "max_negative_width": slack_lo, "max_width_minus_bound": slack_hi}


def run_v3_boot_draw(spec: dict, draw: int) -> Tuple[pd.DataFrame, dict]:
    """Bootstrap-ensemble evaluation of the default policies of one draw of a II.3d cell (cached under v3boot/)."""
    cell, family, draw = spec["cell"], spec["family"], int(draw)
    assert family in ("gumbel", "garnet"), family
    t0 = time.perf_counter()
    inst = build_cell_instance(spec, draw)
    identity = x_boot_identity(spec, inst)
    key, ident_hash = "v3boot/" + x_cache_key(cell, draw), obj_hash(identity)[:20]
    build_s = time.perf_counter() - t0
    hit, meta_hit = XCACHE.get(key, identity), XCACHE.get(key + "/meta", identity)
    if hit is not None and meta_hit is not None:
        del inst
        assert list(hit.columns) == XV3_BOOT_COLUMNS, f"{key}: cached columns differ from XV3_BOOT_COLUMNS"
        meta = json.loads(str(meta_hit["meta_json"].iloc[0]))
        meta.update(cache="loaded", session_build_s=build_s, checks=x_boot_check_rows(hit, meta["v_range"]))
        gc.collect()
        return hit, meta
    if (hit is None) != (meta_hit is None):
        print(f"  {key}: incomplete cache entry (results or meta missing) - recomputing this draw")
    XCACHE.require_allowed(key)
    learn = inst.learn_env
    t1 = time.perf_counter()
    raw = run_xmethods(LearnerView.from_env(learn), inst.train, f"{cell}|draw{draw}")
    records = [r for r in x_collect_records(raw, inst.behavior, learn) if r["is_default"]]
    del raw
    keys = [_x_pi_key(r["pi"]) for r in records]
    t2 = time.perf_counter()
    true_env, hosp = x_boot_hospital(spec, draw)
    B0 = x_benchmark_from_data(true_env, hosp, spec["tau"], learn.name)
    assert obj_hash(B0["bench"].identity()) == obj_hash(learn.identity()), \
        f"{cell} draw {draw}: regenerated hospital data do not rebuild the II.6 benchmark"
    smap = x_boot_step_map(true_env, hosp, spec["tau"], B0)
    eps_id = smap["episode"]
    n_ep = int(eps_id.max()) + 1
    assert np.array_equal(np.unique(eps_id), np.arange(n_ep)), "hospital episode ids must be 0..n-1"
    P0, d00, _ = x_boot_model(learn, smap, np.ones(len(eps_id)), XV3BOOT.resample_d0)
    repro = {"max_abs_diff_P": float(np.abs(P0 - learn.P).max()), "max_abs_diff_d0": float(np.abs(d00 - learn.d0).max())}
    assert max(repro.values()) <= 1e-14, f"{cell} draw {draw}: step map does not reproduce the benchmark {repro}"
    del P0, d00, B0, hosp
    lo, hi = learn.value_bounds
    base = {"cell": cell, "family": family, "draw": draw}

    def rows_for(env: FiniteEnv, target: str, semantics: str, replicate: int,
                 lift: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> List[dict]:
        memo: Dict[bytes, dict] = {}
        out = []
        for r, k in zip(records, keys):
            if k not in memo:
                memo[k] = _x_v3_eval(env, r["pi"] if lift is None else lift(r["pi"]), semantics)
            ev = memo[k]
            d = r["desc"]
            out.append({**base, "policy": d["policy"], "method": d["method"], "variant": d["variant"], "hp": d["hp"],
                        "is_default": True, "method_family": d["method_family"], "replicate": int(replicate),
                        "target": target, "semantics": semantics, "status": ev["status"],
                        **{c: ev.get(c, np.nan) for c in ("return", "prob_any_violation", "expected_violations",
                                                         "avg_episode_length")}})
        return out

    rows = rows_for(learn, "bench_mean", "given", XV3_BOOT_POINT) + rows_for(learn, "bench_terminate", "terminate",
                                                                             XV3_BOOT_POINT)
    tgt = {lab: (env, sem, lift) for lab, env, sem, lift in inst.targets}
    rows += rows_for(tgt["true"][0], "true", tgt["true"][1], XV3_BOOT_POINT, tgt["true"][2])
    t3 = time.perf_counter()
    rep_stats = []
    for b in range(int(XV3BOOT.replicates)):
        mult = x_boot_weights(n_ep, np_rng("xenv", "bootstrap", cell, draw, b))
        P_b, d0_b, st_b = x_boot_model(learn, smap, mult[eps_id], XV3BOOT.resample_d0)
        env_b = learn.variant(f"{learn.name}|boot{b}", P=P_b, d0=d0_b, kind="benchmark", bootstrap_replicate=b)
        del P_b
        st_b["trap_states_given"] = x_trap_states(env_b.P, env_b.live, np.ones((env_b.S, env_b.A), dtype=bool))
        st_b["trap_states_terminate"] = x_trap_states(env_b.P, env_b.live, env_b.mask)
        rb = rows_for(env_b, "bench_mean", "given", b) + rows_for(env_b, "bench_terminate", "terminate", b)
        st_b["nonok_given"] = int(sum(r["status"] != "ok" for r in rb if r["semantics"] == "given"))
        st_b["nonok_terminate"] = int(sum(r["status"] != "ok" for r in rb if r["semantics"] == "terminate"))
        rows += rb
        rep_stats.append({"replicate": b, **st_b})
        del env_b, rb
    t4 = time.perf_counter()
    df = pd.DataFrame(rows, columns=XV3_BOOT_COLUMNS)
    checks = x_boot_check_rows(df, hi - lo)
    meta = _x_compact({
        "cell": cell, "family": family, "draw": draw, "identity_hash": ident_hash, "code_hash": XV3_BOOT_CODE_HASH[:16],
        "cache": "computed", "replicates": int(XV3BOOT.replicates), "resample_d0": bool(XV3BOOT.resample_d0),
        "v_range": float(hi - lo), "n_policies": len(records), "unique_policies": len(set(keys)),
        "hospital_episodes": n_ep, "step_map": {k: v for k, v in smap.items() if not isinstance(v, np.ndarray)},
        "reproduction": repro,
        **{f"replicate_{k}": [r[k] for r in rep_stats] for k in ("fallback_pairs", "d0_support_states",
                                                                 "trap_states_given", "trap_states_terminate",
                                                                 "nonok_given", "nonok_terminate")},
        "admissible_pairs": int(rep_stats[0]["admissible_pairs"]) if rep_stats else None,
        "fallback_pairs_total": int(sum(r["fallback_pairs"] for r in rep_stats)),
        "trap_states_given_max": int(max([r["trap_states_given"] for r in rep_stats], default=0)),
        "checks": checks,
        "timing": {"build_s": build_s, "methods_s": t2 - t1, "point_eval_s": t3 - t2, "bootstrap_s": t4 - t3,
                   "total_s": time.perf_counter() - t0}})
    XCACHE.put(key + "/meta", identity, pd.DataFrame({"meta_json": [json.dumps(meta, default=_json_default)]}))
    XCACHE.put(key, identity, df)
    back = XCACHE.get(key, identity)
    assert back is not None and list(back.columns) == XV3_BOOT_COLUMNS and len(back) == len(df), f"{key}: cache write"
    for c in ("return", "prob_any_violation", "expected_violations", "avg_episode_length"):
        assert np.allclose(back[c].to_numpy(dtype=float), df[c].to_numpy(dtype=float), rtol=1e-14, atol=0.0,
                           equal_nan=True), f"{key}: cache round trip changed column {c}"
    del inst, rows, df, smap
    gc.collect()
    return back, meta


def x_v3_boot_run(cells: Sequence[dict]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Every draw of every bootstrap cell (run_v3_boot_draw): stacked rows, flattened draw meta, summary."""
    frames: List[pd.DataFrame] = []
    meta_rows: List[dict] = []
    times: Dict[str, List[float]] = {}
    status: Counter = Counter()
    t_all = time.perf_counter()
    print(f"II.8g bootstrap: {len(cells)} cells, {sum(int(s['draws']) for s in cells)} draws, "
          f"B = {XV3BOOT.replicates}")
    for ci, spec in enumerate(cells, 1):
        t_cell, n_loaded, nd = time.perf_counter(), 0, int(spec["draws"])
        bar = trange(nd, desc=f"II.8g [{ci}/{len(cells)}] {spec['cell']}", leave=False, disable=not SHOW_PROGRESS)
        for draw in bar:
            t0 = time.perf_counter()
            df, meta = run_v3_boot_draw(spec, draw)
            dt = time.perf_counter() - t0
            frames.append(df)
            meta_rows.append(_x_flatten({**meta, "session_wall_s": dt}))
            status[meta["cache"]] += 1
            if meta["cache"] == "loaded":
                n_loaded += 1
            else:
                times.setdefault(spec["family"], []).append(dt)
            if SHOW_PROGRESS:
                bar.set_postfix_str(f"{dt:.1f}s {meta['cache']}")
            elif (draw + 1) % max(1, nd // 5) == 0 or draw + 1 == nd:
                print(f"    {spec['cell']} draw {draw + 1}/{nd} ({meta['cache']}, {dt:.1f} s); "
                      f"cell ETA {_x_fmt_s(x_eta_seconds([(spec['family'], nd - draw - 1)], times))}")
        eta = x_eta_seconds([(s["family"], int(s["draws"])) for s in cells[ci:]], times)
        print(f"  [{ci}/{len(cells)}] {spec['cell']}: {nd} draws ({n_loaded} cached) in "
              f"{_x_fmt_s(time.perf_counter() - t_cell)} | total elapsed {_x_fmt_s(time.perf_counter() - t_all)} | "
              f"ETA remaining cells {_x_fmt_s(eta) if ci < len(cells) else '0 s'}")
    res = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=XV3_BOOT_COLUMNS)
    meta_df = pd.DataFrame(meta_rows)
    summary = {"code_hash": XV3_BOOT_CODE_HASH, "cells": [s["cell"] for s in cells], "draws": int(sum(status.values())),
               "draws_computed": int(status["computed"]), "draws_loaded": int(status["loaded"]),
               "elapsed_s": time.perf_counter() - t_all, "rows": int(len(res)),
               "nonok_rows_by_target": {str(k): int(v) for k, v in
                                        res[res["status"] != "ok"].groupby("target").size().items()},
               "fallback_pairs_total": (int(meta_df["fallback_pairs_total"].sum())
                                        if "fallback_pairs_total" in meta_df else 0),
               "draws_with_trap_replicate": (int((meta_df["trap_states_given_max"] > 0).sum())
                                             if "trap_states_given_max" in meta_df else 0),
               "max_width_minus_bound": (float(meta_df["checks.max_width_minus_bound"].max())
                                         if "checks.max_width_minus_bound" in meta_df else None),
               "max_rule_p_difference": (float(meta_df["checks.max_rule_p_difference"].max())
                                         if "checks.max_rule_p_difference" in meta_df else None),
               "draws_per_cell": {s["cell"]: int((meta_df["cell"] == s["cell"]).sum()) if len(meta_df) else 0
                                  for s in cells},
               "draws_complete": bool(cells) and all(int((meta_df["cell"] == s["cell"]).sum()) == int(s["draws"])
                                                     for s in cells)}
    return res, meta_df, summary


def x_boot_point_check(boot: pd.DataFrame, xresults: Optional[pd.DataFrame], tol: float = XV3_TOL) -> Dict[str, Any]:
    """Replicate -1 values (regenerated default policies) against the XRESULTS rows of the same draws."""
    if xresults is None or not len(xresults) or not len(boot):
        return {"status": "not compared (XRESULTS or bootstrap rows missing)", "match": None}
    k = ["cell", "draw", "policy", "target"]
    pt = boot[boot["replicate"] == XV3_BOOT_POINT][k + ["return"]]
    xr = xresults[xresults["cell"].isin(pt["cell"].unique()) & xresults["target"].isin(pt["target"].unique())]
    m = pt.merge(xr[k + ["return"]], on=k, how="left", suffixes=("", "_xresults"), validate="one_to_one")
    d = (m["return"] - m["return_xresults"]).abs()
    out = {"status": "compared", "rows": int(len(m)), "missing_in_xresults": int(m["return_xresults"].isna().sum()),
           "max_abs_diff": float(d.max()) if d.notna().any() else None,
           "rows_above_tol": int((d > tol).sum()), "tolerance": tol,
           "max_abs_diff_by_target": {str(t): float(g.max()) for t, g in d.groupby(m["target"]) if g.notna().any()}}
    out["match"] = bool(out["missing_in_xresults"] == 0 and out["rows_above_tol"] == 0)
    return out


# Objects whose source changes what a bootstrap draw computes (the checks are re-run on cached tables).
XV3_BOOT_CODE = [_x_v3_eval, x_boot_hospital, x_boot_step_map, x_boot_weights, x_boot_model, x_boot_identity,
                 run_v3_boot_draw]
XV3_BOOT_CODE_HASH = obj_hash({"code": code_hash(*XV3_BOOT_CODE),
                               "constants": {"point": XV3_BOOT_POINT, "statuses": list(XV3_EVAL_STATUSES),
                                             "columns": XV3_BOOT_COLUMNS}})

XV3_BOOT_RESULTS = pd.DataFrame(columns=XV3_BOOT_COLUMNS)
XV3_BOOT_META = pd.DataFrame()
XV3_BOOT_CELLS: List[dict] = []
XV3_BOOT_SUMMARY: Dict[str, Any] = {"status": "not run (xenv_v3_bootstrap is off)", "draws_complete": None,
                                    "code_hash": XV3_BOOT_CODE_HASH}
if EXPERIMENTS["xenv_v3_bootstrap"]:
    _by_cell = {str(s["cell"]): s for s in X_CELLS}
    XV3_BOOT_CELLS = [_by_cell[c] for c in XV3BOOT.cells if c in _by_cell]
    _skipped = [c for c in XV3BOOT.cells if c not in _by_cell]
    if _skipped:
        print("NOTE: bootstrap cells not in X_CELLS (family switch off or not in the design), skipped:", _skipped)
    if XV3_BOOT_CELLS:
        XV3_BOOT_RESULTS, XV3_BOOT_META, _s = x_v3_boot_run(XV3_BOOT_CELLS)
        XV3_BOOT_SUMMARY = {"status": "run", **_s, "cells_skipped": _skipped,
                            "point_check": x_boot_point_check(XV3_BOOT_RESULTS, globals().get("XRESULTS"))}
        XV3_BOOT_RESULTS.to_csv(XENV_TABLE_DIR / "v3_bootstrap_results.csv.gz", index=False, compression="gzip")
        XV3_BOOT_META.to_csv(XENV_TABLE_DIR / "v3_bootstrap_meta.csv", index=False)
        _pc = XV3_BOOT_SUMMARY["point_check"]
        if _pc.get("match") is False:
            print("NOTE: regenerated default policies differ from XRESULTS on", _pc["rows_above_tol"], "rows (max",
                  _pc["max_abs_diff"], ") - see xenv/v3_bootstrap_summary.json; the intervals use the regenerated values.")
        del _s, _pc
    else:
        XV3_BOOT_SUMMARY = {"status": "not run: no configured bootstrap cell is enabled", "cells_skipped": _skipped,
                            "draws_complete": None, "code_hash": XV3_BOOT_CODE_HASH}
    del _by_cell, _skipped
save_json(XV3_BOOT_SUMMARY, XENV_TABLE_DIR / "v3_bootstrap_summary.json")
print(json.dumps({k: v for k, v in XV3_BOOT_SUMMARY.items() if k not in ("draws_per_cell",)}, indent=1,
                 default=_json_default))
