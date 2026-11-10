# %% [markdown]
# ### II.9f Lift sensitivity: other lifts of the benchmark policies, and the range of J_true over all lifts
#
# A benchmark policy is defined on the benchmark's decision states V only. To score it on the true system (targets
# `true`, `true_rows_mean`, `true_rows_terminate`), Section II.3d **lifts** it to the true state space: the policy's own
# rows on V, and the hospital's most frequent action (`fallback_action`) on every other true live state. That rule is
# a convention of the testbed, not part of the benchmark, and the results may depend on it.
# This section re-scores the default policies of every lift cell (`XV4CFG.lift_cells`: the six II.6 Gumbel `best`
# cells and the eight Garnet cells) under other lifts, and computes the range of J_true over **all** lifts.
#
# **Lifts** (`XV4CFG.lift_kinds`). On V every lift copies the benchmark policy; outside V (the true live states that
# the construction censored or that the hospital never logged) the rows are:
#
# | Lift | Rows outside V |
# |---|---|
# | `fallback` | the II.3d lift `B["lift"]` (the hospital's most frequent action); asserted equal to the instance's `true` lift on the first default policy |
# | `physician` | the true behaviour policy that generated the hospital data |
# | `uniform` | uniform over all actions |
# | `worst` / `best` | the lift that minimises / maximises J_true (`XV4CFG.lift_bounds`; target `true` only) |
#
# **Range over all lifts.** With the policy fixed on V, the lifts are exactly the policies of a variant of the
# `true` target in which every action at a state of V has the policy-mixed row Σ_a π(a|s) P(·|s, a) and every action
# is admissible (semantics `given`). The largest J_true over all lifts is the optimum of that variant, J_hi
# (`x_optimal(variant, "given")`; a deterministic stationary lift attains it), and the smallest is J_lo = −(optimum of
# the same variant with negated entry rewards). The negated variant is a valid `FiniteEnv`: rewards stay on absorbing
# states, the sink and goal labels are kept (they matter only for the terminate rule, which is not used), and the
# evaluator's value range follows the negated rewards. Value iteration needs every policy to absorb: the true Gumbel
# and Garnet systems have no trap (asserted in II.3), and a policy-mixed row has the union support of its actions'
# rows, so the variant has no trap either (asserted per policy). The greedy lift of each end is evaluated exactly on
# `true` itself and must reproduce the solver's exact value (1e-9). A lift copies the benchmark policy on V, so J
# under any lift depends only on the policy's rows on V; evaluations and range solves are memoised by those rows.
#
# **Per (cell, draw).**
# 1. Rebuild the instance (`x_v4_instance`), regenerate the default policies (`x_v4_default_records`), and rebuild the
#    hospital episodes and the benchmark (`x_v4_rebuild_benchmark`, benchmark identity asserted). V =
#    `B["decision_states_true"]`, the first |V| benchmark states (asserted).
# 2. Assert that the rebuilt `B["lift"]` equals the instance's `true` lift on the first default policy.
# 3. Evaluate every default policy exactly on `bench_mean` and `bench_terminate` (lift `benchmark`), and under each
#    lift on the three true-type targets of the instance (their environments and semantics).
# 4. With `lift_bounds`: J_lo and J_hi on `true` (above), stored as the lifts `worst` and `best`.
#
# **Invariants asserted per draw** (re-run on cached tables): every evaluation is proper (the benchmark and the true
# system have no trap); on the benchmark the two rules share p = P(any unsupported decision) and
# 0 ≤ J_mean − J_term ≤ p·(v_max − v_min); under every lift the three true-type targets share p, J_true ≥
# J_true_rows_terminate (Theorem 1(i)) and 0 ≤ J_true_rows_mean − J_true_rows_terminate ≤ p·(v_max − v_min)
# (Theorem 1(iii)); J_worst ≤ J_true(lift) ≤ J_best for every lift (tolerance 1e-8, the DP / exact-value gap that
# `x_optimal` allows). Coverage and certification rates under a lift are outcomes and are only reported.
#
# **Analyses** (default policies, tolerance 1e-9, ties count as errors), per cell, family pool and all cells, for
# each lift:
# * semantic bracket coverage J_true ≤ J_true_rows_mean among policies with p > 0 on the true rows (the lower end is
#   a theorem; its violations are counted), and benchmark bracket coverage J_true ∈ [J_bench_terminate,
#   J_bench_mean] among policies with p > 0 on the benchmark and overall;
# * certificates (`x_v4_pair_claims`) on the true-rows targets and on the benchmark, scored against J_true under the
#   lift: NHB and NHB-free claims and errors, overall and on completion-sensitive pairs (AU + UU), single-rule point
#   claims, draws with any wrong certificate. Benchmark claims do not change with the lift; their errors can;
# * rank fidelity: per-draw Kendall τ-b and top-1 agreement of `bench_mean` and `bench_terminate` against J_true
#   under the lift (`xs_rank_agreement`), and τ(mean) − τ(terminate) with a draw-stratified bootstrap interval;
# * the change of J_true against the `fallback` lift.
#
# One more row per unit (lift `range`) summarises the bounds: the width J_hi − J_lo, the gaps between the fallback
# lift and each end, the share of policies whose benchmark interval contains the whole range, and the benchmark NHB /
# NHB-free certified claims that hold under **every** lift (J_lo(i) > J_hi(j)) or fail under every lift
# (J_hi(i) ≤ J_lo(j), ties count as failures); the rest are lift-dependent. Both conditions are applied with a margin
# of twice the range tolerance (2·10⁻⁸) beyond the tie tolerance, so the solver slack of the two ends cannot move a
# claim into the wrong class. Each policy's range is taken over its own lifts, so, up to that margin, the two
# conditions are exact when the two policies may be lifted independently; for one lift rule shared by both
# policies, "holds" and "fails" stay sufficient conditions and some lift-dependent claims may in fact be robust.
#
# Results are cached in `XCACHE` under `v4lift/<cell>/draw<k>` (and `/meta`). The identity is the regenerated draw
# identity (`x_v4_draw_identity`), the code hash of this section (which folds in `XV4_COMMON_CODE_HASH`), the lift
# kinds and `lift_bounds`. As a check that the regenerated policies are the II.6 ones, the `fallback` and benchmark
# rows are compared with `XRESULTS` / `XRESULTS_V3` (`x_boot_point_check`; reported, not asserted).
#
# | Output | Content |
# |---|---|
# | `xenv/v4_lift_results.csv.gz` | one row per draw × default policy × lift × target: lifts `fallback` / `physician` / `uniform` on `true`, `true_rows_mean`, `true_rows_terminate`; `worst` / `best` on `true`; `benchmark` on `bench_mean` / `bench_terminate` |
# | `xenv/v4_lift_summary.csv` | per cell, family pool and all cells × lift: coverage, certificates, rank fidelity, change against `fallback`; `range` rows: width of the range and benchmark claims that hold or fail under every lift |
# | `xenv/v4_lift_summary.json` | status, code hash, cells run and skipped, cache use, invariant maxima, the comparison with II.6, headline numbers |
#
# **Expected cost on Colab** (estimates, scaled from local timings of the same array sizes). A Gumbel draw costs an
# instance rebuild and one `run_xmethods` fit (6–12 s, as in II.8g), about 400 exact evaluations on the 606-state
# true system (about 10 s), and, with `lift_bounds`, two value-iteration solves per distinct default policy (about 40)
# on the 609 × 8 × 609 variant, each about 500 sweeps at 2–3 ms: 1.5–2.5 min per draw, so 4.5–7.5 h for the 180
# Gumbel draws of the paper design. A Garnet draw takes 8–15 s (about 1 h for 240 draws). The `lite` profile (10
# draws per cell) takes about a third. `lift_bounds` is off by default in paper mode (a Gumbel draw then takes about
# 20–25 s: about 1 h for 180 Gumbel draws, 2 h in all); enable it for a subset of cells to get the range. A cached draw
# reloads in about 3–5 s (instance rebuild and the checks). The smoke run (two draws of one Gumbel and two Garnet
# cells, all three lifts and the bounds) still solves the full 609-state Gumbel variant: about 3–5 min.

# %%
# ============================================================
# II.9f Lift sensitivity
# ============================================================
EXECLOG.mark("II.9f Lift sensitivity")

XV4_LIFT_TARGETS = ("true", "true_rows_mean", "true_rows_terminate")   # the true-type targets of an estimated draw
XV4_LIFT_BENCH = "benchmark"         # lift label of the benchmark rows (bench_mean / bench_terminate need no lift)
XV4_LIFT_ENDS = ("worst", "best")    # lift labels of the range ends of J_true over all lifts (target 'true' only)
XV4_LIFT_METRICS = ("return", "prob_any_violation", "expected_violations", "avg_episode_length")
XV4_LIFT_COLUMNS = ["cell", "family", "draw", "policy", "method", "variant", "hp", "is_default", "method_family",
                    "lift", "target", "semantics", "status", *XV4_LIFT_METRICS]
XV4_LIFT_BOUND_TOL = 1e-8            # slack of J_worst <= J_lift <= J_best: the DP / exact gap x_optimal allows
XV4_LIFT_RANGE_CHECKS = ("lift_below_worst", "lift_above_best", "range_negative_width")   # checked at BOUND_TOL
XV4_LIFT_PAIR_TARGETS = {"true_rows": ("true_rows_terminate", "true_rows_mean", "p_true"),
                         "bench": ("bench_terminate", "bench_mean", "p_bench")}


def x_v4_lift_map(V: np.ndarray, Sb: int, rows: np.ndarray, live: np.ndarray,
                  name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Lift of benchmark policies ([Sb, A]) to the true state space: the policy's rows on the decision states V (the
    benchmark states 0..|V|-1), `rows` ([S, A]) on every other true live state, uniform rows on absorbing states (as
    the II.3d lift)."""
    V = np.asarray(V, dtype=np.int64)
    S, A = np.shape(rows)
    outside = np.asarray(live, dtype=bool).copy()
    outside[V] = False
    base = np.full((S, A), 1.0 / A)
    base[outside] = validate_policy(rows, f"{name} rows", shape=(S, A))[outside]

    def lift(pi_bench: np.ndarray) -> np.ndarray:
        pb = validate_policy(pi_bench, f"{name} policy", shape=(int(Sb), A))
        out = base.copy()
        out[V] = pb[:len(V)]
        return out

    return lift


def x_v4_lift_maps(B: Dict[str, Any], true_env: FiniteEnv, true_behavior: np.ndarray,
                   kinds: Sequence[str]) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """The lifts `kinds` of benchmark B: 'fallback' is B['lift']; 'physician' / 'uniform' copy the policy on V and use
    the true behaviour rows / uniform rows on every other true live state."""
    S, A, Sb = true_env.S, true_env.A, int(B["bench"].S)
    V = np.asarray(B["decision_states_true"], dtype=np.int64)
    name = B["bench"].name
    out: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for kind in kinds:
        if kind == "fallback":
            out[kind] = B["lift"]
        elif kind == "physician":
            out[kind] = x_v4_lift_map(V, Sb, true_behavior, true_env.live, f"{name}|lift-physician")
        elif kind == "uniform":
            out[kind] = x_v4_lift_map(V, Sb, np.full((S, A), 1.0 / A), true_env.live, f"{name}|lift-uniform")
        else:
            raise ValueError(f"unknown lift kind {kind!r}")
    return out


def x_v4_lift_bounds(env: FiniteEnv, pi_on_V: np.ndarray, V: np.ndarray,
                     name: str) -> Dict[str, Tuple[np.ndarray, float, Dict[str, Any]]]:
    """Range of J (semantics 'given') of `env` over all lifts of one policy: exact optimal control outside the decision
    states V with the policy's rows `pi_on_V` ([|V|, A]) fixed on V.

    In the variant every action at a state of V has the policy-mixed row sum_a pi(a|s) P(.|s, a) and every action is
    admissible, so its policies are the lifts. 'best': x_optimal of the variant; 'worst': minus the optimum of the
    variant with negated entry rewards. Returns per end (greedy action of every state, end value, solver report). Every
    state of V ties across actions by construction, so the report counts near ties outside V only."""
    V = np.asarray(V, dtype=np.int64)
    pin = validate_policy(pi_on_V, f"{name} rows on V", shape=(len(V), env.A))
    P = env.P.copy()
    P[V] = np.einsum("sa,san->sn", pin, env.P[V], optimize=True)[:, None, :]
    full = np.ones((env.S, env.A), dtype=bool)
    assert x_trap_states(P, env.live, full) == 0, f"{name}: a lift is improper, but the true system has no trap"
    out: Dict[str, Tuple[np.ndarray, float, Dict[str, Any]]] = {}
    for end, sign in (("best", 1.0), ("worst", -1.0)):
        var = FiniteEnv(name=f"{name}|lift-{end}", kind="variant", P=P,
                        entry_reward=np.where(env.live, 0.0, sign * env.entry_reward), d0=env.d0,
                        terminals=env.terminals, sink=env.sink, goal=env.goal, mask=full,
                        meta={"base": env.name, "lift_range_end": end})
        pi_opt, _, info = x_optimal(var, "given")
        ties_out = int(info["near_tie_live_states"]) - (len(V) if env.A > 1 else 0)
        out[end] = (pi_opt.argmax(axis=1), sign * float(info["exact_return_of_greedy"]),
                    {"iterations": int(info["iterations"]), "abs_gap": float(info["abs_gap"]),
                     "near_tie_outside_V": max(ties_out, 0)})
        del var, pi_opt
    del P
    return out


def x_v4_lift_identity(spec: dict, inst: CellInstance) -> dict:
    return {"kind": "xenv_v4_lift", "draw": x_v4_draw_identity(spec, inst), "code": XV4_LIFT_CODE_HASH,
            "lift_kinds": list(XV4CFG.lift_kinds), "lift_bounds": bool(XV4CFG.lift_bounds),
            "columns": XV4_LIFT_COLUMNS, "policies": "default (is_default) records of x_collect_records"}


def x_v4_lift_check_rows(df: pd.DataFrame, v_range: float) -> Dict[str, Any]:
    """Invariants of one draw's lift rows (see markdown); asserted, and a small report of the slacks is returned."""
    bad = sorted(set(df["status"].astype(str)) - {"ok"})
    assert not bad, f"statuses {bad}: the benchmark and the true system are proper for every policy"
    dup = df.duplicated(["policy", "lift", "target"])
    assert not dup.any(), f"{int(dup.sum())} duplicated (policy, lift, target) lift rows"
    w = df.set_index(["policy", "lift", "target"])
    J, Pv = w["return"].unstack("target"), w["prob_any_violation"].unstack("target")
    pols = list(pd.unique(df["policy"]))
    lifts = list(pd.unique(df["lift"]))
    kinds = [x for x in lifts if x not in (XV4_LIFT_BENCH, *XV4_LIFT_ENDS)]
    assert XV4_LIFT_BENCH in lifts and kinds, f"lift rows without benchmark rows or lifts: {lifts}"

    def block(lift: str, targets: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        j = J.xs(lift, level="lift").reindex(index=pols, columns=list(targets))
        p = Pv.xs(lift, level="lift").reindex(index=pols, columns=list(targets))
        assert j.notna().all().all(), f"lift {lift}: missing (policy, target) rows"
        return j, p

    def slacks(lo: pd.Series, up: pd.Series, p: pd.Series) -> Tuple[float, float]:
        width = (up - lo).to_numpy(dtype=float)
        over = width - p.to_numpy(dtype=float) * float(v_range)
        return float((-width).max(initial=-np.inf)), float(over.max(initial=-np.inf))

    jb, pb = block(XV4_LIFT_BENCH, ("bench_mean", "bench_terminate"))
    rep = {"rows": int(len(df)), "policies": len(pols), "lifts": kinds,
           "bench_rule_p_difference": float((pb["bench_mean"] - pb["bench_terminate"]).abs().max())}
    rep["bench_negative_width"], rep["bench_width_minus_bound"] = slacks(jb["bench_terminate"], jb["bench_mean"],
                                                                        pb["bench_terminate"])
    dp, lower, neg, over = [], [], [], []
    for kind in kinds:
        j, p = block(kind, XV4_LIFT_TARGETS)
        dp.append(float(max((p["true"] - p["true_rows_mean"]).abs().max(),
                            (p["true"] - p["true_rows_terminate"]).abs().max())))
        lower.append(float((j["true_rows_terminate"] - j["true"]).max()))
        n_, o_ = slacks(j["true_rows_terminate"], j["true_rows_mean"], p["true_rows_mean"])
        neg.append(n_)
        over.append(o_)
    rep.update({"target_p_difference": max(dp), "lower_bound_shortfall": max(lower), "rows_negative_width": max(neg),
                "rows_width_minus_bound": max(over)})
    for k in ("bench_rule_p_difference", "target_p_difference"):
        assert rep[k] <= XV4_TOL, f"{k} = {rep[k]:.3g}: the completions must share P(any unsupported decision)"
    for k in ("bench_negative_width", "bench_width_minus_bound", "lower_bound_shortfall", "rows_negative_width",
              "rows_width_minus_bound"):
        assert rep[k] <= XV4_TOL, f"{k} = {rep[k]:.3g} (Theorem 1 parts i / iii)"
    rep["has_range"] = all(e in lifts for e in XV4_LIFT_ENDS)
    if rep["has_range"]:
        lo = block("worst", ("true",))[0]["true"]
        hi = block("best", ("true",))[0]["true"]
        below = [float((lo - block(k, ("true",))[0]["true"]).max()) for k in kinds]
        above = [float((block(k, ("true",))[0]["true"] - hi).max()) for k in kinds]
        rep.update({"lift_below_worst": max(below), "lift_above_best": max(above),
                    "range_negative_width": float((lo - hi).max()), "range_width_max": float((hi - lo).max())})
        for k in XV4_LIFT_RANGE_CHECKS:
            assert rep[k] <= XV4_LIFT_BOUND_TOL, f"{k} = {rep[k]:.3g}: the range must contain every lift"
    return rep


def run_v4_lift_draw(spec: dict, draw: int) -> Tuple[pd.DataFrame, dict]:
    """Default policies of one draw under every lift of XV4CFG.lift_kinds on the true-type targets, the benchmark
    values, and the range of J_true over all lifts (cached under v4lift/)."""
    cell, family, draw = spec["cell"], spec["family"], int(draw)
    assert family in ("gumbel", "garnet"), family
    t0 = time.perf_counter()
    inst = x_v4_instance(spec, draw)
    identity = x_v4_lift_identity(spec, inst)
    key, ident_hash = "v4lift/" + x_cache_key(cell, draw), obj_hash(identity)[:20]
    build_s = time.perf_counter() - t0
    hit, meta_hit = XCACHE.get(key, identity), XCACHE.get(key + "/meta", identity)
    if hit is not None and meta_hit is not None:
        del inst
        assert list(hit.columns) == XV4_LIFT_COLUMNS, f"{key}: cached columns differ from XV4_LIFT_COLUMNS"
        meta = json.loads(str(meta_hit["meta_json"].iloc[0]))
        meta.update(cache="loaded", session_build_s=build_s, checks=x_v4_lift_check_rows(hit, meta["v_range"]))
        gc.collect()
        return hit, meta
    if (hit is None) != (meta_hit is None):
        print(f"  {key}: incomplete cache entry (results or meta missing) - recomputing this draw")
    XCACHE.require_allowed(key)
    learn = inst.learn_env
    t1 = time.perf_counter()
    records, keys = x_v4_default_records(inst)
    t2 = time.perf_counter()
    true_env, beh, hosp, B = x_v4_rebuild_benchmark(spec, draw, inst)
    del hosp
    V = np.asarray(B["decision_states_true"], dtype=np.int64)
    nV = len(V)
    assert np.array_equal(np.asarray(B["index"], dtype=np.int64)[:nV], V), "benchmark index must start with V"
    outside = true_env.live.copy()
    outside[V] = False
    tgt = {lab: (env, sem, lift) for lab, env, sem, lift in inst.targets}
    missing = [t for t in (*XV4_LIFT_TARGETS, "bench_mean", "bench_terminate") if t not in tgt]
    assert not missing, f"{cell}: lift sensitivity needs the benchmark and true-type targets; missing {missing}"
    assert all(tgt[t][2] is not None and tgt[t][0].S == true_env.S for t in XV4_LIFT_TARGETS), cell
    assert tgt["true"][1] == "given", tgt["true"][1]
    pi0 = records[0]["pi"]
    assert np.array_equal(B["lift"](pi0), tgt["true"][2](pi0)), \
        f"{cell} draw {draw}: the rebuilt fallback lift differs from the instance's 'true' lift"
    lifts = x_v4_lift_maps(B, true_env, beh, XV4CFG.lift_kinds)
    on_V = [_x_pi_key(np.ascontiguousarray(r["pi"][:nV])) for r in records]   # J under a lift depends on these rows
    lo, hi = learn.value_bounds
    base = {"cell": cell, "family": family, "draw": draw}
    rows: List[dict] = []

    def add_rows(lift: str, target: str, semantics: str, evs: Dict[bytes, dict], rkeys: Sequence[bytes]) -> None:
        for r, k in zip(records, rkeys):
            d, ev = r["desc"], evs[k]
            rows.append({**base, "policy": d["policy"], "method": d["method"], "variant": d["variant"], "hp": d["hp"],
                         "is_default": True, "method_family": d["method_family"], "lift": lift, "target": target,
                         "semantics": semantics, "status": ev["status"],
                         **{c: ev.get(c, np.nan) for c in XV4_LIFT_METRICS}})

    for target in ("bench_mean", "bench_terminate"):
        env, sem, _ = tgt[target]
        memo: Dict[bytes, dict] = {}
        for r, k in zip(records, keys):
            if k not in memo:
                memo[k] = x_v4_eval(env, r["pi"], sem)
        add_rows(XV4_LIFT_BENCH, target, sem, memo, keys)
    n_eval = 0
    for kind, lift in lifts.items():
        for target in XV4_LIFT_TARGETS:
            env, sem, _ = tgt[target]
            memo = {}
            for r, k in zip(records, on_V):
                if k not in memo:
                    memo[k] = x_v4_eval(env, lift(r["pi"]), sem)
                    n_eval += 1
            add_rows(kind, target, sem, memo, on_V)
    t3 = time.perf_counter()
    bound_stats: List[dict] = []
    if XV4CFG.lift_bounds:
        env, sem, _ = tgt["true"]
        ends: Dict[str, Dict[bytes, dict]] = {e: {} for e in XV4_LIFT_ENDS}
        for r, k in zip(records, on_V):
            if k in ends["best"]:
                continue
            res = x_v4_lift_bounds(env, r["pi"][:nV], V, f"{learn.name}|{r['desc']['policy']}")
            for end in XV4_LIFT_ENDS:
                act, value, info = res[end]
                lift_end = x_v4_lift_map(V, learn.S, one_hot_policy(act, env.A), env.live, f"{learn.name}|lift-{end}")
                ev = x_v4_eval(env, lift_end(r["pi"]), sem)
                gap = abs(float(ev.get("return", np.nan)) - value)
                assert ev["status"] == "ok" and gap <= XV4_TOL * max(1.0, abs(value)), \
                    f"{cell} draw {draw} {r['desc']['policy']}: the {end} lift on 'true' gives {ev} vs {value}"
                ends[end][k] = ev
                bound_stats.append({"end": end, "value_gap": gap, **info})
            del res
        for end in XV4_LIFT_ENDS:
            add_rows(end, "true", sem, ends[end], on_V)
    t4 = time.perf_counter()
    df = pd.DataFrame(rows, columns=XV4_LIFT_COLUMNS)
    checks = x_v4_lift_check_rows(df, hi - lo)
    bs = pd.DataFrame(bound_stats)
    meta = _x_compact({
        "cell": cell, "family": family, "draw": draw, "identity_hash": ident_hash, "code_hash": XV4_LIFT_CODE_HASH[:16],
        "cache": "computed", "v_range": float(hi - lo), "lift_kinds": list(lifts),
        "lift_bounds": bool(XV4CFG.lift_bounds), "n_policies": len(records), "unique_policies": len(set(keys)),
        "unique_rows_on_V": len(set(on_V)), "true_space_evaluations": n_eval, "decision_states": int(nV),
        "outside_live_states": int(outside.sum()), "true_d0_mass_outside": float(true_env.d0[outside].sum()),
        "fallback_action": int(B["fallback_action"]),
        "bounds": ({"solves": int(len(bs)), "vi_iterations_max": int(bs["iterations"].max()),
                    "vi_iterations_mean": float(bs["iterations"].mean()), "dp_gap_max": float(bs["abs_gap"].max()),
                    "value_gap_max": float(bs["value_gap"].max()),
                    "near_tie_outside_V_max": int(bs["near_tie_outside_V"].max())} if len(bs) else {"solves": 0}),
        "checks": checks,
        "timing": {"build_s": build_s, "methods_s": t2 - t1, "eval_s": t3 - t2, "bounds_s": t4 - t3,
                   "total_s": time.perf_counter() - t0}})
    XCACHE.put(key + "/meta", identity, pd.DataFrame({"meta_json": [json.dumps(meta, default=_json_default)]}))
    XCACHE.put(key, identity, df)
    back = XCACHE.get(key, identity)
    assert back is not None and list(back.columns) == XV4_LIFT_COLUMNS and len(back) == len(df), f"{key}: cache write"
    for c in XV4_LIFT_METRICS:
        assert np.allclose(back[c].to_numpy(dtype=float), df[c].to_numpy(dtype=float), rtol=1e-14, atol=0.0,
                           equal_nan=True), f"{key}: cache round trip changed column {c}"
    for c in ("status", "lift", "target", "policy"):
        assert (back[c].astype(str).to_numpy() == df[c].astype(str).to_numpy()).all(), f"{key}: cache round trip {c}"
    del inst, df, B, lifts
    gc.collect()
    return back, meta


# Objects whose source changes what a lift draw computes or stores (the checks are re-run on cached tables);
# _x_v3_eval (II.8g) is the evaluation wrapper behind x_v4_eval, whose keys are the stored metric columns.
XV4_LIFT_CODE = [x_v4_lift_map, x_v4_lift_maps, x_v4_lift_bounds, x_v4_lift_identity, run_v4_lift_draw, _x_v3_eval]
XV4_LIFT_CODE_HASH = obj_hash({"common": XV4_COMMON_CODE_HASH, "code": code_hash(*XV4_LIFT_CODE),
                               "constants": {"targets": list(XV4_LIFT_TARGETS), "bench": XV4_LIFT_BENCH,
                                             "ends": list(XV4_LIFT_ENDS), "metrics": list(XV4_LIFT_METRICS),
                                             "columns": XV4_LIFT_COLUMNS, "bound_tol": XV4_LIFT_BOUND_TOL}})


# ---------- analyses (pure pandas on the stacked rows) ----------
_XV4_LIFT_FLAGS = ["viol", "viol_true", "sem_lo_ok", "sem_up_ok", "b_lo_ok", "b_up_ok", "b_in", "width",
                   "bench_width", "bound_width", "bench_covers_range", "change_vs_fallback"]
XV4_LIFT_FRAME_COLUMNS = ["cell", "family", "draw", "policy", "lift", "v_range", *XV4_LIFT_TARGETS, "p_true",
                          "bench_mean", "bench_terminate", "p_bench", "j_lo", "j_hi", "true_fallback", *_XV4_LIFT_FLAGS]


def x_v4_lift_frame(res: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, draw, policy, lift): J_true and the true-rows bracket under the lift (p from true_rows_mean), the
    benchmark values (p from bench_mean), the range ends J_lo / J_hi over all lifts (NaN without bounds), J_true under
    the fallback lift, and the coverage flags."""
    if res is None or not len(res):
        return pd.DataFrame(columns=XV4_LIFT_FRAME_COLUMNS)
    r = res.copy()
    r["draw"] = pd.to_numeric(r["draw"]).astype(np.int64)
    k = ["cell", "draw", "policy"]
    w = r.set_index(k + ["lift", "target"])
    J, Pv = w["return"], w["prob_any_violation"]
    lifts = list(pd.unique(r["lift"]))
    kinds = [x for x in lifts if x not in (XV4_LIFT_BENCH, *XV4_LIFT_ENDS)]
    jb = J.xs(XV4_LIFT_BENCH, level="lift").unstack("target")
    side = pd.DataFrame({"bench_mean": jb["bench_mean"], "bench_terminate": jb["bench_terminate"],
                         "p_bench": Pv.xs(XV4_LIFT_BENCH, level="lift").unstack("target")["bench_mean"]})

    def true_of(lift: str) -> pd.Series:
        if lift not in lifts:
            return pd.Series(np.nan, index=side.index)
        return J.xs(lift, level="lift").xs("true", level="target").reindex(side.index)

    side["j_lo"], side["j_hi"], side["true_fallback"] = true_of("worst"), true_of("best"), true_of("fallback")
    side = side.join(r.drop_duplicates(k).set_index(k)[["family"]])
    parts = []
    for kind in kinds:
        jk = J.xs(kind, level="lift").unstack("target").reindex(side.index)
        pk = Pv.xs(kind, level="lift").unstack("target").reindex(side.index)
        parts.append(side.assign(lift=kind, **{t: jk[t] for t in XV4_LIFT_TARGETS}, p_true=pk["true_rows_mean"]))
    W = pd.concat(parts).reset_index()
    m = meta.assign(draw=pd.to_numeric(meta["draw"]).astype(np.int64)).set_index(["cell", "draw"])["v_range"]
    W["v_range"] = m.reindex(pd.MultiIndex.from_frame(W[["cell", "draw"]])).to_numpy(dtype=float)
    assert W["v_range"].notna().all(), "draw meta without v_range"
    W["viol"] = W["p_bench"] > XV4_PTOL
    W["viol_true"] = W["p_true"] > XV4_PTOL
    W["sem_lo_ok"] = W["true"] >= W["true_rows_terminate"] - XV4_TOL
    W["sem_up_ok"] = W["true"] <= W["true_rows_mean"] + XV4_TOL
    W["b_lo_ok"] = W["true"] >= W["bench_terminate"] - XV4_TOL
    W["b_up_ok"] = W["true"] <= W["bench_mean"] + XV4_TOL
    W["b_in"] = W["b_lo_ok"] & W["b_up_ok"]
    W["width"] = W["true_rows_mean"] - W["true_rows_terminate"]
    W["bench_width"] = W["bench_mean"] - W["bench_terminate"]
    W["bound_width"] = W["j_hi"] - W["j_lo"]
    W["bench_covers_range"] = (W["bench_terminate"] <= W["j_lo"] + XV4_TOL) & (W["j_hi"] <= W["bench_mean"] + XV4_TOL)
    W["change_vs_fallback"] = W["true"] - W["true_fallback"]
    return W[XV4_LIFT_FRAME_COLUMNS]


def x_v4_lift_draw_stats(W: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(pairs, ranks, range) per draw. pairs: x_v4_pair_claims per (cell, draw, lift, targets) with J_true under the
    lift as truth; ranks: Kendall tau-b / top-1 agreement of each benchmark rule against J_true under the lift; range:
    benchmark NHB / NHB-free claims that hold (J_lo(i) > J_hi(j)) or fail (J_hi(i) <= J_lo(j)) under every lift, both
    with a margin of 2 * XV4_LIFT_BOUND_TOL beyond the tie tolerance XV4_TOL."""
    pair_rows, rank_rows, range_rows = [], [], []
    for (c, dr, lift), d in W.groupby(["cell", "draw", "lift"], sort=False):
        unit = {"cell": c, "family": d["family"].iloc[0], "draw": int(dr), "lift": lift}
        vr, jt = float(d["v_range"].iloc[0]), d["true"].to_numpy(dtype=float)
        for tf, (lc, uc, pc) in XV4_LIFT_PAIR_TARGETS.items():
            for r in x_v4_pair_claims(d[lc].to_numpy(dtype=float), d[uc].to_numpy(dtype=float),
                                      d[pc].to_numpy(dtype=float), vr, jt):
                pair_rows.append({**unit, "targets": tf, **r})
        for rule in ("bench_mean", "bench_terminate"):
            ra = xs_rank_agreement(d[rule].to_numpy(dtype=float), jt, np.full(len(d), np.nan))
            rank_rows.append({**unit, "ranked_by": rule, "evaluated_on": "true",
                              **{m: ra[m] for m in ("n_policies", "kendall_tau_b", "top1_agreement",
                                                    "tau_benchmark_constant", "tau_undefined_constant_values")}})
    one = W.drop_duplicates(["cell", "draw", "policy"])
    for (c, dr), d in one.groupby(["cell", "draw"], sort=False):
        if d[["j_lo", "j_hi"]].isna().any().any():
            continue
        n, vr = len(d), float(d["v_range"].iloc[0])
        off = ~np.eye(n, dtype=bool)
        blo, bup, pb = (d[x].to_numpy(dtype=float) for x in ("bench_terminate", "bench_mean", "p_bench"))
        jlo, jhi = d["j_lo"].to_numpy(dtype=float), d["j_hi"].to_numpy(dtype=float)
        # each end is the exact value of an attained lift, within the solver slack XV4_LIFT_BOUND_TOL of the extreme:
        # a margin of 2 * BOUND_TOL around the tie tolerance keeps 'hold' (correct under every lift) and 'fail'
        # (wrong, ties included, under every lift) valid whatever that slack does; the rest are lift-dependent
        hold = jlo[:, None] - jhi[None, :] > XV4_TOL + 2.0 * XV4_LIFT_BOUND_TOL
        fail = jhi[:, None] - jlo[None, :] <= XV4_TOL - 2.0 * XV4_LIFT_BOUND_TOL
        u = pb > XV4_PTOL
        scopes = {"all": off, "cs": off & (u[:, None] | u[None, :])}
        r = {"cell": c, "family": d["family"].iloc[0], "draw": int(dr), "n_policies": n}
        for nm, C in (("cert_nhb", _x_v4_gt(blo, bup)), ("cert_free", _x_v4_gt(blo, blo + pb * vr))):
            for sc, msk in scopes.items():
                Cm = C & msk
                r[f"{nm}_{sc}_claims"] = int(Cm.sum())
                r[f"{nm}_{sc}_hold_every_lift"] = int((Cm & hold).sum())
                r[f"{nm}_{sc}_wrong_every_lift"] = int((Cm & fail).sum())
        range_rows.append(r)
    return pd.DataFrame(pair_rows), pd.DataFrame(rank_rows), pd.DataFrame(range_rows)


def x_v4_lift_summary(W: pd.DataFrame, D: pd.DataFrame, R: pd.DataFrame, G: pd.DataFrame, cells: Sequence[str],
                      n_boot: int) -> pd.DataFrame:
    """Per unit (cell, family pool, all cells) x lift: coverage, certificates, rank fidelity and the change against
    the fallback lift; plus one 'range' row per unit with the bounds statistics."""
    if W.empty:
        return pd.DataFrame()
    present = set(W["cell"])
    units = [(c, [c]) for c in cells if c in present]
    fams = sorted(set(W["family"]))
    units += [(f"pooled:{f}", [c for c in cells if c in set(W.loc[W["family"] == f, "cell"])]) for f in fams]
    if len(fams) > 1:
        units.append(("pooled:all", [c for c in cells if c in present]))
    kinds = list(pd.unique(W["lift"]))
    rows = []
    for name, cs in units:
        for lift in kinds:
            w = W[W["cell"].isin(cs) & (W["lift"] == lift)]
            if w.empty:
                continue
            v, vt = w[w["viol"]], w[w["viol_true"]]
            ch = w["change_vs_fallback"].dropna().abs()
            row = {"unit": name, "n_cells": len(cs), "lift": lift,
                   "n_draws": int(w[["cell", "draw"]].drop_duplicates().shape[0]), "n_policy_draws": int(len(w)),
                   "n_violating_bench": int(len(v)), "n_violating_true": int(len(vt)),
                   "sem_lower_violations": int((~w["sem_lo_ok"]).sum()),
                   "sem_upper_coverage_all": _xv3_rate(w["sem_up_ok"].sum(), len(w)),
                   "sem_upper_coverage_violating": _xv3_rate(vt["sem_up_ok"].sum(), len(vt)),
                   "sem_width_mean_violating": float(vt["width"].mean()) if len(vt) else np.nan,
                   "bench_coverage_all": _xv3_rate(w["b_in"].sum(), len(w)),
                   "bench_coverage_violating": _xv3_rate(v["b_in"].sum(), len(v)),
                   "bench_lower_holds_violating": _xv3_rate(v["b_lo_ok"].sum(), len(v)),
                   "bench_upper_holds_violating": _xv3_rate(v["b_up_ok"].sum(), len(v)),
                   "true_change_vs_fallback_mean_abs": float(ch.mean()) if len(ch) else np.nan,
                   "true_change_vs_fallback_max_abs": float(ch.max()) if len(ch) else np.nan,
                   "true_changed_vs_fallback_share": _xv3_rate((ch > XV4_TOL).sum(), len(ch))}
            d = D[D["cell"].isin(cs) & (D["lift"] == lift)]
            for tf in XV4_LIFT_PAIR_TARGETS:
                dt = d[d["targets"] == tf]
                a, s = dt[dt["type"] == "all"], dt[dt["type"].isin(["AU", "UU"])]
                pa, ps = int(a["pairs"].sum()), int(s["pairs"].sum())
                row[f"{tf}_pairs"], row[f"{tf}_cs_pairs"] = pa, ps
                for kind in ("cert_nhb", "cert_free", "point_mean", "point_term"):
                    n, e = int(a[f"{kind}_claims"].sum()), int(a[f"{kind}_errors"].sum())
                    row.update({f"{tf}_{kind}_claims": n, f"{tf}_{kind}_share": _xv3_rate(n, pa),
                                f"{tf}_{kind}_errors": e, f"{tf}_{kind}_error_rate": _xv3_rate(e, n)})
                for kind in ("cert_nhb", "cert_free"):
                    n, e = int(s[f"{kind}_claims"].sum()), int(s[f"{kind}_errors"].sum())
                    row.update({f"{tf}_{kind}_cs_claims": n, f"{tf}_{kind}_cs_share": _xv3_rate(n, ps),
                                f"{tf}_{kind}_cs_errors": e, f"{tf}_{kind}_cs_error_rate": _xv3_rate(e, n),
                                f"{tf}_draws_any_wrong_{kind}": int((a[f"{kind}_errors"] > 0).sum())})
            rk = R[R["cell"].isin(cs) & (R["lift"] == lift)]
            for rule in ("bench_mean", "bench_terminate"):
                t = rk[rk["ranked_by"] == rule]
                row[f"tau_{rule}_mean"] = float(t["kendall_tau_b"].mean()) if len(t) else np.nan
                row[f"top1_{rule}_mean"] = float(t["top1_agreement"].mean()) if len(t) else np.nan
            piv = rk.pivot_table(index=["cell", "draw"], columns="ranked_by", values="kendall_tau_b", aggfunc="first")
            dlt = ((piv["bench_mean"] - piv["bench_terminate"]).dropna()
                   if {"bench_mean", "bench_terminate"} <= set(piv.columns) else pd.Series(dtype=float))
            groups = ({c: g.to_numpy(dtype=np.float64)[:, None] for c, g in dlt.groupby(level="cell")}
                      if len(dlt) else {})
            point, reps = xs_stratified_bootstrap(groups, xs_stat_mean(), ("v4lift", lift, "tau difference"), n_boot)
            row["tau_diff_mean"] = point
            row["tau_diff_ci_low"], row["tau_diff_ci_high"] = xs_ci(reps)
            row["tau_diff_n"] = int(len(dlt))
            rows.append(row)
        one = W[W["cell"].isin(cs)].drop_duplicates(["cell", "draw", "policy"])
        one = one[one["j_lo"].notna() & one["j_hi"].notna()]
        if one.empty:
            continue
        v = one[one["viol"]]
        row = {"unit": name, "n_cells": len(cs), "lift": "range",
               "n_draws": int(one[["cell", "draw"]].drop_duplicates().shape[0]), "n_policy_draws": int(len(one)),
               "n_violating_bench": int(len(v)), "bound_width_mean": float(one["bound_width"].mean()),
               "bound_width_median": float(one["bound_width"].median()),
               "bound_width_max": float(one["bound_width"].max()),
               "bound_width_mean_violating": float(v["bound_width"].mean()) if len(v) else np.nan,
               "bound_width_zero_share": _xv3_rate((one["bound_width"] <= XV4_TOL).sum(), len(one)),
               "fallback_gap_to_best_mean": float((one["j_hi"] - one["true_fallback"]).mean()),
               "fallback_gap_to_worst_mean": float((one["true_fallback"] - one["j_lo"]).mean()),
               "bench_covers_range_share": _xv3_rate(one["bench_covers_range"].sum(), len(one)),
               "bench_covers_range_share_violating": _xv3_rate(v["bench_covers_range"].sum(), len(v))}
        g = G[G["cell"].isin(cs)] if len(G) else G
        for nm in ("cert_nhb", "cert_free"):
            for sc in ("all", "cs"):
                n = int(g[f"{nm}_{sc}_claims"].sum()) if len(g) else 0
                h = int(g[f"{nm}_{sc}_hold_every_lift"].sum()) if len(g) else 0
                f_ = int(g[f"{nm}_{sc}_wrong_every_lift"].sum()) if len(g) else 0
                row.update({f"bench_{nm}_{sc}_claims": n, f"bench_{nm}_{sc}_hold_every_lift": h,
                            f"bench_{nm}_{sc}_hold_every_lift_share": _xv3_rate(h, n),
                            f"bench_{nm}_{sc}_wrong_every_lift": f_,
                            f"bench_{nm}_{sc}_wrong_every_lift_share": _xv3_rate(f_, n),
                            f"bench_{nm}_{sc}_lift_dependent": n - h - f_})
            if len(g):   # draws with a certified benchmark claim that some lift could make wrong
                row[f"bench_draws_any_{nm}_not_robust"] = int(
                    (g[f"{nm}_all_claims"] > g[f"{nm}_all_hold_every_lift"]).sum())
        rows.append(row)
    return pd.DataFrame(rows)


XV4_LIFT_HEADLINE_COLUMNS = ("n_draws", "n_policy_draws", "sem_upper_coverage_violating", "bench_coverage_violating",
                             "true_rows_cert_nhb_error_rate", "true_rows_cert_free_error_rate",
                             "bench_cert_nhb_error_rate", "bench_cert_free_error_rate", "bench_cert_nhb_cs_error_rate",
                             "bench_cert_free_cs_error_rate", "tau_bench_mean_mean", "tau_bench_terminate_mean",
                             "tau_diff_mean", "tau_diff_ci_low", "tau_diff_ci_high",
                             "true_change_vs_fallback_mean_abs", "bound_width_mean", "bound_width_mean_violating",
                             "bench_covers_range_share", "bench_cert_nhb_all_hold_every_lift_share",
                             "bench_cert_free_all_hold_every_lift_share", "bench_cert_nhb_all_wrong_every_lift_share",
                             "bench_cert_free_all_wrong_every_lift_share")


def x_v4_lift_headline(summary: pd.DataFrame) -> Dict[str, Any]:
    """Headline numbers per lift of the widest pooled unit (pooled:all, else the family pool, else the only cell)."""
    if summary is None or summary.empty:
        return {}
    units = list(pd.unique(summary["unit"]))
    unit = "pooled:all" if "pooled:all" in units else next((u for u in units if u.startswith("pooled:")), units[0])
    s = summary[summary["unit"] == unit].set_index("lift")
    out: Dict[str, Any] = {"unit": unit}
    for lift, r in s.iterrows():
        out[str(lift)] = {c: r[c] for c in XV4_LIFT_HEADLINE_COLUMNS if c in r.index and pd.notna(r[c])}
    return out


def x_v4_lift_point_check(res: pd.DataFrame) -> Dict[str, Any]:
    """Fallback-lift and benchmark rows of the regenerated default policies against the II.6 / II.8 rows of the same
    draws (x_boot_point_check; reported, not asserted)."""
    pt = res[res["lift"].isin(["fallback", XV4_LIFT_BENCH])].assign(replicate=XV3_BOOT_POINT)
    cells = set(pt["cell"])
    xr = [x[x["cell"].isin(cells)] for x in (globals().get("XRESULTS"), globals().get("XRESULTS_V3"))
          if x is not None and len(x)]
    xr = [x for x in xr if len(x)]
    return x_boot_point_check(pt, pd.concat(xr, ignore_index=True) if xr else None)


XV4_LIFT_CHECK_KEYS = ("bench_rule_p_difference", "bench_negative_width", "bench_width_minus_bound",
                       "target_p_difference", "lower_bound_shortfall", "rows_negative_width", "rows_width_minus_bound",
                       "lift_below_worst", "lift_above_best", "range_negative_width", "range_width_max")


def x_v4_lift_checks(meta: pd.DataFrame) -> Dict[str, Any]:
    """Largest per-draw slack of every asserted invariant (each was asserted per draw), the solver statistics, and
    `passes` (every slack within its tolerance: XV4_TOL, or XV4_LIFT_BOUND_TOL for the range checks)."""
    out: Dict[str, Any] = {"draws_checked": int(len(meta))}
    for k in XV4_LIFT_CHECK_KEYS:
        col = f"checks.{k}"
        out[f"max_{k}"] = float(pd.to_numeric(meta[col], errors="coerce").max()) if col in meta else None
    for k in ("bounds.vi_iterations_max", "bounds.dp_gap_max", "bounds.value_gap_max",
              "bounds.near_tie_outside_V_max"):
        out[f"max_{k.split('.', 1)[1]}"] = float(pd.to_numeric(meta[k], errors="coerce").max()) if k in meta else None
    out["bound_solves"] = (int(pd.to_numeric(meta["bounds.solves"], errors="coerce").sum())
                           if "bounds.solves" in meta else 0)
    out["tolerances"] = {"value": XV4_TOL, "p": XV4_PTOL, "bounds": XV4_LIFT_BOUND_TOL}
    slack = {k: out[f"max_{k}"] for k in XV4_LIFT_CHECK_KEYS if k != "range_width_max"}
    out["passes"] = bool(len(meta)) and all(
        v is None or not np.isfinite(v) or v <= (XV4_LIFT_BOUND_TOL if k in XV4_LIFT_RANGE_CHECKS else XV4_TOL)
        for k, v in slack.items())
    return out


XV4_LIFT_ANALYSIS_CODE = [x_v4_lift_check_rows, x_v4_lift_frame, x_v4_lift_draw_stats, x_v4_lift_summary,
                          x_v4_lift_headline, x_v4_lift_point_check, x_v4_lift_checks]

XV4_LIFT_FRAMES = ("results", "draw_meta", "policies", "pairs", "ranks", "range", "summary")
XV4_LIFT: Dict[str, pd.DataFrame] = {k: pd.DataFrame() for k in XV4_LIFT_FRAMES}
XV4_LIFT["results"] = pd.DataFrame(columns=XV4_LIFT_COLUMNS)
XV4_LIFT_CELLS: List[dict] = []
XV4_LIFT_SUMMARY: Dict[str, Any] = {"status": "not run (xenv_v4_lift off)", "code_hash": XV4_LIFT_CODE_HASH,
                                    "checks": {}, "elapsed_s": 0.0, "draws_complete": None}
if EXPERIMENTS["xenv_v4_lift"]:
    _t0 = time.perf_counter()
    XV4_LIFT_CELLS, _skipped = x_v4_cells(XV4CFG.lift_cells)
    if _skipped:
        print("NOTE: lift cells not enabled (family switch off or not in the design), skipped:", _skipped)
    if XV4_LIFT_CELLS:
        _frames, _meta, _run = x_v4_run_draws(XV4_LIFT_CELLS, run_v4_lift_draw, "II.9f lift")
        _res = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame(columns=XV4_LIFT_COLUMNS)
        del _frames
        _cells = [str(s["cell"]) for s in XV4_LIFT_CELLS]
        _W = x_v4_lift_frame(_res, _meta)
        _D, _R, _G = x_v4_lift_draw_stats(_W)
        _summ = x_v4_lift_summary(_W, _D, _R, _G, _cells, int(XCFG.bootstrap))
        XV4_LIFT = {"results": _res, "draw_meta": _meta, "policies": _W, "pairs": _D, "ranks": _R, "range": _G,
                    "summary": _summ}
        _res.to_csv(XENV_TABLE_DIR / "v4_lift_results.csv.gz", index=False, compression="gzip")
        _summ.to_csv(XENV_TABLE_DIR / "v4_lift_summary.csv", index=False)
        _n_meta = _meta["cell"].value_counts() if len(_meta) else pd.Series(dtype=int)
        XV4_LIFT_SUMMARY = {
            "status": "run", "code_hash": XV4_LIFT_CODE_HASH, **_run, "cells_skipped": _skipped,
            "draws_complete": all(int(_n_meta.get(s["cell"], 0)) == x_v4_draws(s) for s in XV4_LIFT_CELLS),
            "lift_kinds": list(XV4CFG.lift_kinds), "lift_bounds": bool(XV4CFG.lift_bounds), "rows": int(len(_res)),
            "analysis_code_hash": code_hash(*XV4_LIFT_ANALYSIS_CODE)[:16], "checks": x_v4_lift_checks(_meta),
            "point_check": x_v4_lift_point_check(_res), "headline": x_v4_lift_headline(_summ),
            "outputs": ["xenv/v4_lift_results.csv.gz", "xenv/v4_lift_summary.csv", "xenv/v4_lift_summary.json"],
            "elapsed_s": time.perf_counter() - _t0}
        if XV4_LIFT_SUMMARY["point_check"].get("match") is False:
            print("NOTE: regenerated fallback / benchmark rows differ from XRESULTS on",
                  XV4_LIFT_SUMMARY["point_check"]["rows_above_tol"], "rows (max",
                  XV4_LIFT_SUMMARY["point_check"]["max_abs_diff"], ") - see xenv/v4_lift_summary.json.")
        del _res, _meta, _run, _cells, _W, _D, _R, _G, _summ, _n_meta
    else:
        XV4_LIFT_SUMMARY = {"status": "not run: no configured lift cell is enabled", "code_hash": XV4_LIFT_CODE_HASH,
                            "cells_skipped": _skipped, "checks": {}, "draws_complete": None,
                            "elapsed_s": time.perf_counter() - _t0}
    del _t0, _skipped
save_json(XV4_LIFT_SUMMARY, XENV_TABLE_DIR / "v4_lift_summary.json")
print(json.dumps({k: v for k, v in XV4_LIFT_SUMMARY.items() if k not in ("draws_per_cell", "headline")}, indent=1,
                 default=_json_default))
if len(XV4_LIFT["summary"]):
    _cols = [c for c in ("unit", "lift", "n_draws", "sem_upper_coverage_violating", "bench_coverage_violating",
                         "true_rows_cert_nhb_error_rate", "bench_cert_nhb_error_rate", "bench_cert_free_error_rate",
                         "tau_bench_mean_mean", "tau_bench_terminate_mean", "true_change_vs_fallback_mean_abs",
                         "bound_width_mean", "bench_cert_nhb_all_hold_every_lift_share",
                         "bench_cert_free_all_hold_every_lift_share") if c in XV4_LIFT["summary"].columns]
    display(XV4_LIFT["summary"][_cols].round(4))
    del _cols
