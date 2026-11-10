# %% [markdown]
# ### II.9e Sensitivity cells: timeout hazard, dead-state rule, physician support and state abstraction
#
# The II.6 benchmarks fix three construction choices: the Gumbel timeout hazard η = 0.05 (the evaluation horizon,
# 20 decisions on average), the ICU-Sepsis dead-state rule `icu` (a decision state without an action reaching τ is
# removed and its inflow dropped) and an ε-optimal physician whose count masks are near-singleton. II.8 added a
# benchmark on aggregated observation cells. This section varies these choices in new cells that run the **full II.6
# pipeline**: the same learners on user data of the same size, the same records (behaviour policy and the projection
# of every native record), exact evaluation on every target, the reference rows, the strict per-draw invariant checks
# (`x_check_draw_rows`) and the result columns `XRESULT_COLUMNS`. The lift of benchmark policies is varied in II.9f.
#
# **Cells** (`x_v4_sens_build_cells`, from `XV4CFG`). A base cell that is not enabled is skipped and listed; in
# smoke mode `x_v4_cells` substitutes the smoke cell of the same construction, and a cell that would then repeat
# another is skipped.
#
# * **timeout**: `<base>|eta<η>` for each (base, η) in `timeout_cells`, the base II.6 spec with `timeout_hazard` = η.
#   The exact Gumbel system is enumerated at η (`x_v4_true_system`; `_x_v3_gumbel_env` caches one system per
#   hazard, so a hazard never reuses the system of another), and the physician softens the optimum of that system.
#   The random streams are named by the new cell.
# * **deadrule**: `<base>|dead-argmax` for each base in `deadrule_cells` (Gumbel and Garnet), the base spec with
#   `empty_rule = "argmax"`: a decision state without an action reaching τ keeps its most frequent action instead of
#   being removed. The streams are named by the **base** cell (`stream_cell`), so the hospital episodes and the user
#   and split streams are those of the base cell. Only the rule differs (paired design).
# * **support**: `<support_base>|v4eps<ε>` for ε in `support_eps`, the base spec with physician ε. The physician is
#   `_x_v3_gumbel_physician(variant, η, ε, "optimal")`. A softer physician logs more actions per state, so more pairs
#   reach τ. The streams are named by the new cell.
# * **abstraction**: `gumbel|agg-<kind>|n<n>|tau<τ>|eps<ε>` for kind in `aggregations`, with (n, τ, ε) =
#   `aggregation_base`. This is the II.8c design (true full-state system, full-state physician, benchmark estimated
#   on cells φ(s)) with a new φ from `x_v4_gumbel_aggregation`:
#   * `glucose`: φ(s) = (hr, sbp, o2, abx, vaso, vent, diabetic). Glucose is hidden and diabetes visible
#     (3·3·2·2·2·2·2 = 288 cells, 224 of them with a live state). Glucose then confounds: the physician sees it,
#     and it changes death and discharge.
#   * `kmeans<K>`: K cells from k-means over the observation features of the 606 true live states: (hr, sbp, o2,
#     glucose, abx, vaso, vent), each level divided by (levels − 1), the scaling of the true environment's
#     features. Diabetes is excluded, so it is a hidden confounder as in `agg-obs`. A state and its diabetic twin
#     have the same features and always share a cell, so there are 303 distinct points and K ≤ 303. The clustering
#     is deterministic: k-means++ seeding from `np_rng("xenv", "v4", "kmeans", K)`, then Lloyd iterations until the
#     assignment repeats. Squared distances are computed elementwise (no BLAS, so the cells do not depend on the
#     machine), ties go to the lowest cluster index, an empty cluster is re-seeded at the farthest point, and the
#     cells are numbered by their first live state.
#
#   `construct_aggregated_benchmark_v4` repeats `construct_aggregated_benchmark` step by step, with φ given as an
#   argument. It reuses the II.8 functions unchanged: the population-limit model `x_aggregated_env`, the mapped
#   hospital data `x_aggregate_dataset`, the estimate `x_benchmark_from_data` and the lift π(a | φ(s)). The targets
#   are `bench_mean`, `bench_terminate` and `true` (lifted through φ), as in the II.8 aggregated cells. The reference
#   cell is the II.8 cell `gumbel|agg-obs|n<n>|tau<τ>|eps<ε>` when it is enabled. In smoke mode, whose II.8 cells
#   use other sizes, the abstraction cells take the sizes of the smoke `agg-obs` cell and pair with it.
#
# **Per (cell, draw).**
# 1. `build_v4_instance(spec, draw)` mirrors the `best` branch of `build_cell_instance`. The hospital, user and split
#    streams are `("xenv", purpose, spec.get("stream_cell", cell), draw)`. The hospital data are validated
#    (`x_validate_dataset`), the benchmark is `x_benchmark_from_data(..., empty_rule=spec.get("empty_rule"))`, and
#    the instance has the five targets, `true_optimum_in_mask_fraction` for Gumbel and the same meta. Abstraction
#    cells take the aggregated construction above, as `build_v3_instance` does.
# 2. `run_v4_cell_draw` mirrors `run_v3_cell_draw`: the identity `{"kind", "draw": x_v4_draw_identity(spec, inst),
#    "code": XV4_SENS_CODE_HASH}` is looked up in `XCACHE` under `v4sens/<cell>/draw<k>` (and `/meta`). On a miss,
#    if `ALLOW_TRAINING`, the draw runs `run_xmethods` on the training episodes, `x_collect_records`, the FQE
#    selection model, `x_evaluate_targets` and `x_check_draw_rows` (strict). The table returned is always the one
#    read back from the cache, with a round-trip check.
#
# `XV4_SENS_CODE_HASH` covers the source of every function of this section on the path of a draw and the module
# constants they read. It folds in `XV4_COMMON_CODE_HASH`, and through it the Part II and II.8 code hashes.
#
# **Self-test** (asserted; a few seconds).
# 1. Take the base spec of the first enabled sensitivity cell of each family, with `stream_cell` = its own name, the
#    hazard unchanged and the default dead-state rule (`XCFG.benchmark_empty_rule`, `icu`). `build_v4_instance(spec,
#    0)` must have the identity, behaviour policy and dev split of `build_cell_instance(spec, 0)`.
# 2. The factor aggregation must reproduce II.8's `obs` map, and `construct_aggregated_benchmark_v4` with that map
#    must reproduce `construct_aggregated_benchmark(…, "obs", …)`: benchmark identity, pulled-back mask, decision
#    states and lift.
# 3. Every abstraction of the full design (`glucose`, `kmeans50`, `kmeans150`) is computed twice; the two maps must
#    be identical, with non-empty cells. This also exercises the k-means path in smoke mode, where only the first
#    abstraction runs as a cell.

# %%
# ============================================================
# II.9e Sensitivity cells: cells, abstractions, instances, per-draw driver, self-test
# ============================================================
EXECLOG.mark("II.9e Sensitivity cells: definitions and self-test")

XV4_SENS_KINDS = ("timeout", "deadrule", "support", "abstraction")
XV4_AGG_FACTORS = {"glucose": ("hr", "sbp", "o2", "abx", "vaso", "vent", "diabetic")}   # factor maps (glucose hidden)
XV4_KMEANS_FEATURES = ("hr", "sbp", "o2", "glucose", "abx", "vaso", "vent")              # diabetes hidden
XV4_KMEANS_MAX_ITER = 1000                     # Lloyd iterations (+ empty-cluster re-seeds); convergence is asserted


def x_v4_sens_build_cells() -> Tuple[List[dict], Dict[str, Optional[str]], List[str]]:
    """(specs, reference cell of each spec, skipped names) of the II.9e cells (see markdown). The reference cell is
    used by the analyses only and never enters a spec, so a missing II.8 reference does not change any identity."""
    cells: List[dict] = []
    refs: Dict[str, Optional[str]] = {}
    skipped: List[str] = []

    def base_of(name: str, families: Sequence[str]) -> Optional[dict]:
        found, _ = x_v4_cells([name], families=families)
        b = found[0] if found else None
        return b if b is not None and "v3_kind" not in b and b.get("construction", "best") == "best" else None

    def add(spec: dict, ref: Optional[str], configured: str) -> None:
        if any(c["cell"] == spec["cell"] for c in cells):
            skipped.append(f"{configured} (would repeat {spec['cell']} after the smoke-mode substitution)")
            return
        assert spec["v4_kind"] in XV4_SENS_KINDS, spec["v4_kind"]
        cells.append(spec)
        refs[spec["cell"]] = ref

    for name, eta in XV4CFG.timeout_cells:
        assert 0.0 < float(eta) < 1.0, eta
        b = base_of(name, ("gumbel",))
        if b is None:
            skipped.append(f"{name}|eta{float(eta)}")
            continue
        add({**b, "cell": f"{b['cell']}|eta{float(eta)}", "timeout_hazard": float(eta), "v4_kind": "timeout"},
            str(b["cell"]), f"{name}|eta{float(eta)}")
    for name in XV4CFG.deadrule_cells:
        b = base_of(name, ("gumbel", "garnet"))
        if b is None or str(XCFG.benchmark_empty_rule) == "argmax":    # 'argmax' everywhere: the cell is its base
            skipped.append(f"{name}|dead-argmax")
            continue
        add({**b, "cell": f"{b['cell']}|dead-argmax", "empty_rule": "argmax", "stream_cell": str(b["cell"]),
             "v4_kind": "deadrule"}, str(b["cell"]), f"{name}|dead-argmax")
    b = base_of(XV4CFG.support_base, ("gumbel",))
    for eps in XV4CFG.support_eps:
        assert 0.0 < float(eps) < 1.0, eps
        if b is None:
            skipped.append(f"{XV4CFG.support_base}|v4eps{float(eps)}")
            continue
        add({**b, "cell": f"{b['cell']}|v4eps{float(eps)}", "eps": float(eps), "v4_kind": "support"}, str(b["cell"]),
            f"{XV4CFG.support_base}|v4eps{float(eps)}")
    n, tau, eps = XV4CFG.aggregation_base
    v3 = {str(s["cell"]): s for s in (globals().get("X_V3_CELLS") or [])}
    ref = f"gumbel|agg-obs|n{int(n)}|tau{int(tau)}|eps{float(eps)}"
    if ref not in v3:   # reference not enabled: none (listed as missing by the analyses)
        ref = None
        if RUN_MODE == "smoke":   # the smoke designs use other sizes: the agg-obs cell of the optimal physician
            ref = next((c for c, s in v3.items() if s.get("aggregation") == "obs" and s.get("physician") == "optimal"),
                       None)
            if ref is not None:   # and its sizes, as x_v4_cells substitutes the smoke cell of a construction
                n, tau, eps = v3[ref]["n_hosp"], v3[ref]["tau"], v3[ref]["eps"]
    for kind in XV4CFG.aggregations:
        kind = str(kind)
        assert kind in XV4_AGG_FACTORS or (kind.startswith("kmeans") and kind[6:].isdigit()), kind
        cell = f"gumbel|agg-{kind}|n{int(n)}|tau{int(tau)}|eps{float(eps)}"
        add({"cell": cell, "family": "gumbel", "v4_kind": "abstraction", "construction": f"agg-{kind}",
             "aggregation": kind, "n_hosp": int(n), "tau": int(tau), "eps": float(eps), "physician": "optimal",
             "draws": int(XCFG.gumbel_draws), "variant": XCFG.gumbel_variant,
             "timeout_hazard": float(XCFG.gumbel_timeout_hazard), "user_episodes": int(XCFG.gumbel_user_episodes)},
            ref, cell)
    names = [c["cell"] for c in cells]
    assert len(names) == len(set(names)), f"duplicate II.9e cell names: {names}"
    clash = sorted(set(names) & set(x_v4_all_specs()))
    assert not clash, f"II.9e cells must not reuse II.6 / II.8 cell names: {clash}"
    return cells, refs, skipped


# ---------- state abstractions of the exact Gumbel system ----------
def _x_v4_gumbel_live_layout(env: FiniteEnv) -> Tuple[Dict[str, Any], np.ndarray, int]:
    """(gumbel_state_space(), live full indices, number of live states) after the layout checks of II.8c."""
    space = gumbel_state_space()
    live_full = space["live_full"]
    n_live = int(env.meta["n_live"])
    assert env.kind == "gumbel" and n_live == len(live_full) and env.S == n_live + 3, env.name
    death, discharge, timeout = n_live, n_live + 1, n_live + 2
    assert (env.terminals, env.sink, env.goal, int(env.meta["timeout_state"])) == \
        ((death, discharge, timeout), death, discharge, timeout), "unexpected absorbing layout"
    return space, live_full, n_live


def _x_v4_factor_aggregation(env: FiniteEnv, label: str, factors: Sequence[str]) -> Dict[str, Any]:
    """x_gumbel_aggregation (II.8c) for any set of Gumbel factors (first factor most significant): phi maps a live
    state to the rank of its factor tuple among the tuples of live states; absorbing states follow in true order."""
    space, live_full, n_live = _x_v4_gumbel_live_layout(env)
    factors = tuple(str(f) for f in factors)
    assert factors and len(set(factors)) == len(factors) and set(factors) <= set(GumbelSim.FACTORS), factors
    fidx = [GumbelSim.FACTORS.index(f) for f in factors]
    radix = np.array([GumbelSim.LEVELS[i] for i in fidx], dtype=np.int64)
    strides = np.ones(len(radix), dtype=np.int64)
    for j in range(len(radix) - 2, -1, -1):
        strides[j] = strides[j + 1] * radix[j + 1]
    full = np.array([space["full"][f] for f in live_full], dtype=np.int64)
    cell = full[:, fidx] @ strides
    keys = np.unique(cell)
    n_agg = int(len(keys))
    phi = np.empty(env.S, dtype=np.int64)
    phi[:n_live] = np.searchsorted(keys, cell)
    phi[n_live:] = n_agg + np.arange(3)
    levels = (keys[:, None] // strides[None, :]) % radix[None, :]
    members = np.bincount(phi[:n_live], minlength=n_agg)
    assert members.min() >= 1 and int(members.sum()) == n_live
    return {"aggregation": str(label), "phi": phi, "n_live": n_agg, "S": n_agg + 3, "cells_total": int(radix.prod()),
            "factors": factors, "radix": radix, "keys": keys, "levels": levels, "members": members,
            "report": {"method": "factors", "factors": list(factors)}}


def x_v4_kmeans(X: np.ndarray, K: int, rng: np.random.Generator,
                max_iter: int = XV4_KMEANS_MAX_ITER) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Deterministic k-means of the rows of X into K non-empty clusters: (labels [n], centres [K, d], info).

    k-means++ seeding from `rng`, then Lloyd iterations until the assignment repeats. Squared Euclidean distances are
    computed elementwise (no BLAS, so the result does not depend on the machine); ties go to the lowest cluster index;
    an empty cluster is re-seeded at the point farthest from its centre among clusters with more than one member
    (lowest row index on ties), and the next assignment is always followed by a centre update, so the result is a
    fixed point of Lloyd's map (centres = member means). Clusters are numbered by their first row, so the labels do
    not depend on the seeding order."""
    X = np.asarray(X, dtype=np.float64)
    n, K = int(len(X)), int(K)
    n_distinct = int(len(np.unique(X, axis=0)))
    assert X.ndim == 2 and np.isfinite(X).all() and 1 <= K <= n_distinct, (X.shape, K, n_distinct)

    def sqdist(C: np.ndarray) -> np.ndarray:
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)

    C = np.empty((K, X.shape[1]))
    C[0] = X[int(rng.integers(0, n))]
    d2 = sqdist(C[:1])[:, 0]
    for k in range(1, K):                         # k-means++: P(row) proportional to its squared distance
        cum = np.cumsum(d2)
        assert cum[-1] > 0.0, "k-means++: fewer distinct rows than clusters"
        i = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
        i = min(i, int(np.flatnonzero(d2 > 0.0)[-1]))          # rounding at the top end never picks a chosen row
        C[k] = X[i]
        d2 = np.minimum(d2, sqdist(C[k:k + 1])[:, 0])
    labels = np.full(n, -1, dtype=np.int64)
    reseeds, converged, it = 0, False, 0
    for it in range(1, int(max_iter) + 1):
        D = sqdist(C)
        new = D.argmin(axis=1)                                   # ties -> lowest cluster index
        counts = np.bincount(new, minlength=K)
        if (counts == 0).any():
            own = np.where(counts[new] > 1, D[np.arange(n), new], -1.0)
            assert own.max() > 0.0, "k-means: no row can re-seed an empty cluster"
            C[int(np.flatnonzero(counts == 0)[0])] = X[int(own.argmax())]
            reseeds += 1
            labels = np.full(n, -1, dtype=np.int64)       # the next assignment is accepted and every centre updated
            continue
        if np.array_equal(new, labels):
            converged = True
            break
        labels = new
        for j in range(X.shape[1]):
            C[:, j] = np.bincount(labels, weights=X[:, j], minlength=K) / counts
    assert converged, f"k-means (K={K}) did not converge in {max_iter} iterations"
    first = np.array([int(np.flatnonzero(labels == k)[0]) for k in range(K)])
    order = np.argsort(first, kind="stable")
    rank = np.empty(K, dtype=np.int64)
    rank[order] = np.arange(K)
    labels, C = rank[labels], C[order]
    info = {"K": K, "rows": n, "distinct_rows": n_distinct, "iterations": int(it), "reseeds": int(reseeds),
            "converged": bool(converged), "inertia": float(((X - C[labels]) ** 2).sum())}
    return labels, C, info


def x_v4_gumbel_aggregation(env: FiniteEnv, kind: str) -> Dict[str, Any]:
    """Index map phi of the exact Gumbel env for an II.9e abstraction, in the dict layout of x_gumbel_aggregation.

    'glucose': the factor map XV4_AGG_FACTORS['glucose'] (glucose hidden, diabetes visible). 'kmeansK': x_v4_kmeans
    of the scaled observation features XV4_KMEANS_FEATURES of the true live states into K cells; the cell 'levels' are
    the cluster centres with radix 1, so x_aggregated_env uses the centres as the cell features."""
    kind = str(kind)
    if kind in XV4_AGG_FACTORS:
        return _x_v4_factor_aggregation(env, kind, XV4_AGG_FACTORS[kind])
    assert kind.startswith("kmeans") and kind[6:].isdigit(), f"unknown II.9e aggregation {kind!r}"
    K = int(kind[6:])
    space, live_full, n_live = _x_v4_gumbel_live_layout(env)
    fidx = [GumbelSim.FACTORS.index(f) for f in XV4_KMEANS_FEATURES]
    scale = np.array([GumbelSim.LEVELS[i] - 1 for i in fidx], dtype=np.float64)
    X = np.array([[space["full"][f][i] for i in fidx] for f in live_full], dtype=np.float64) / scale
    labels, centres, info = x_v4_kmeans(X, K, np_rng("xenv", "v4", "kmeans", K))
    phi = np.empty(env.S, dtype=np.int64)
    phi[:n_live] = labels
    phi[n_live:] = K + np.arange(3)
    members = np.bincount(labels, minlength=K)
    assert members.min() >= 1 and int(members.sum()) == n_live
    return {"aggregation": kind, "phi": phi, "n_live": K, "S": K + 3, "cells_total": K,
            "factors": XV4_KMEANS_FEATURES, "radix": np.ones(len(fidx), dtype=np.int64), "keys": np.arange(K),
            "levels": centres, "members": members,
            "report": {"method": "kmeans", "features": list(XV4_KMEANS_FEATURES),
                       "feature_scaling": "level / (levels - 1)", **info}}


_X_V4_AGG: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _x_v4_aggregation(env: FiniteEnv, kind: str) -> Dict[str, Any]:
    """x_v4_gumbel_aggregation, cached per (system, kind) within the session."""
    key = (str(env.name), str(kind))
    if key not in _X_V4_AGG:
        _X_V4_AGG[key] = x_v4_gumbel_aggregation(env, kind)
    return _X_V4_AGG[key]


def construct_aggregated_benchmark_v4(true_env: FiniteEnv, behavior: np.ndarray, agg: Dict[str, Any], n_hosp: int,
                                      tau: int, rng: np.random.Generator, name: str) -> Dict[str, Any]:
    """construct_aggregated_benchmark (II.8c) with the aggregation `agg` (x_v4_gumbel_aggregation or
    x_gumbel_aggregation) given instead of its name: the same steps, report keys and outputs (II.8 functions reused)."""
    kind = str(agg["aggregation"])
    phi = agg["phi"]
    agg_env, agg_check = x_aggregated_env(true_env, agg, behavior, f"{true_env.name}|agg-{kind}")
    hosp = x_sample_dataset(true_env, behavior, int(n_hosp), XCFG.max_steps, rng)
    hosp_report = x_validate_dataset(hosp, true_env)
    hosp_agg = x_aggregate_dataset(hosp, phi, agg_env)
    agg_report = x_validate_dataset(hosp_agg, agg_env)
    out = x_benchmark_from_data(agg_env, hosp_agg, tau, name)
    lift_cells = out["lift"]
    S, A = true_env.S, true_env.A

    def lift(pi_bench: np.ndarray) -> np.ndarray:
        """Benchmark policy -> true full-state policy pi(a | phi(s)) (absorbing rows uniform)."""
        return lift_cells(pi_bench)[phi]

    mask_true = out["mask_true"][phi]
    assert mask_true.shape == (S, A) and mask_true[~true_env.live].all()
    cells = out["decision_states_true"]                     # cell indices of the benchmark's decision states
    in_cells = true_env.live & np.isin(phi, cells)
    logged = np.zeros(S, dtype=bool)
    logged[np.unique(hosp["state"])] = True
    rep = out["report"]
    rep.update({"construction": f"agg-{kind}", "aggregation": kind,
                "aggregated_cells_total": int(agg["cells_total"]), "aggregated_live_cells": int(agg["n_live"]),
                "members_per_live_cell_max": int(agg["members"].max()),
                "full_live_states": int(true_env.live.sum()), "full_states_logged": int(logged.sum()),
                "full_live_states_in_decision_cells": int(in_cells.sum()),
                "full_logged_states_in_decision_cells": int((in_cells & logged).sum()),
                "full_d0_mass_in_decision_cells": float(true_env.d0[in_cells].sum()),
                **{f"aggregation_check_{k}": v for k, v in agg_check.items()},
                **{f"phi_{k}": v for k, v in (agg.get("report") or {}).items()},
                "hospital_validation": hosp_report, "aggregated_hospital_validation": agg_report})
    del hosp, hosp_agg
    return {"bench": out["bench"], "bench_behavior": out["bench_behavior"], "index": out["index"],
            "mask_true": mask_true, "lift": lift, "fallback_action": out["fallback_action"], "report": rep,
            "decision_states_true": np.flatnonzero(in_cells & logged), "decision_cells": cells, "phi": phi,
            "agg_env": agg_env}


# ---------- instances and the per-draw driver ----------
def build_v4_instance(spec: dict, draw: int) -> CellInstance:
    """Deterministic (cell, draw) instance of a Section II.9e cell: the 'best' branch of build_cell_instance with the
    streams named by spec['stream_cell'] (default: the cell), the dead-state rule spec['empty_rule'] (default:
    XCFG.benchmark_empty_rule) and the spec's own hazard and physician eps; abstraction cells ('aggregation' in the
    spec) take the aggregated construction of build_v3_instance with the II.9e phi."""
    t0 = time.perf_counter()
    cell, family, draw = spec["cell"], spec["family"], int(draw)
    stream = str(spec.get("stream_cell", cell))
    aggregated = "aggregation" in spec
    assert draw >= 0 and family in ("gumbel", "garnet") and "v3_kind" not in spec, (draw, family, spec.get("v3_kind"))
    assert aggregated or spec.get("construction", "best") == "best", spec.get("construction")
    rng_u = np_rng("xenv", "user", stream, draw)
    true_env, true_behavior = x_v4_true_system_draw(spec, draw)
    if family == "gumbel":
        assert float(true_env.meta["timeout_hazard"]) == float(spec["timeout_hazard"]), (true_env.name, spec["cell"])
    bname = f"{cell}|d{draw}|bench"
    if aggregated:
        assert family == "gumbel" and spec.get("empty_rule") is None, (family, spec.get("empty_rule"))
        B = construct_aggregated_benchmark_v4(true_env, true_behavior, _x_v4_aggregation(true_env, spec["aggregation"]),
                                              spec["n_hosp"], spec["tau"], np_rng("xenv", "hospital", stream, draw),
                                              bname)
    else:   # construct_benchmark with the stream cell and the dead-state rule of the spec
        hosp = x_v4_hospital(spec, draw, true_env, true_behavior, stream_cell=stream)
        hosp_report = x_validate_dataset(hosp, true_env)
        B = x_benchmark_from_data(true_env, hosp, spec["tau"], bname, empty_rule=spec.get("empty_rule"))
        B["report"]["hospital_validation"] = hosp_report
        del hosp
    learn_env, behavior, lift = B["bench"], B["bench_behavior"], B["lift"]
    data = x_sample_dataset(learn_env, behavior, spec["user_episodes"], XCFG.max_steps, rng_u)
    T_masked, T_meanimp = x_true_targets(true_env, B["mask_true"])
    targets = [("bench_mean", learn_env, "given", None), ("bench_terminate", learn_env, "terminate", None),
               ("true", T_masked, "given", lift)]
    if not aggregated:
        targets += [("true_rows_mean", T_meanimp, "given", lift), ("true_rows_terminate", T_masked, "terminate", lift)]
    del T_meanimp
    if family == "gumbel":   # is the true optimal action supported? (decision states of the construction)
        opt_act, Vt = _x_v3_gumbel_opt_action(spec["variant"], spec["timeout_hazard"]), B["decision_states_true"]
        B["report"]["true_optimum_in_mask_fraction"] = (float(B["mask_true"][Vt, opt_act[Vt]].mean()) if len(Vt)
                                                        else float("nan"))
    deploy = "true"
    construction = {k: v for k, v in B["report"].items() if k != "hospital_validation"}
    hosp_validation = B["report"]["hospital_validation"]
    source = f"x_sample_dataset({spec.get('construction', 'best')} benchmark, clinician estimate)"
    del B
    labels = [t[0] for t in targets]
    assert len(labels) == len(set(labels)) and deploy in labels, labels
    for lab, env, sem, lift_fn in targets:
        assert sem in X_SEMANTICS and (lift_fn is not None or env.S == learn_env.S), (lab, env.S, learn_env.S)
    user_validation = x_validate_dataset(data, learn_env)
    train, dev = x_split_episodes(data, XCFG.dev_fraction, np_rng("xenv", "split", stream, draw))
    meta = {"spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}, "user_data_source": source,
            "stream_cell": stream, "user_validation": user_validation, "hospital_validation": hosp_validation,
            "construction": construction, "train_transitions": int(len(train["state"])),
            "dev_transitions": int(len(dev["state"])), "build_s": time.perf_counter() - t0}
    return CellInstance(cell=cell, family=family, draw=draw, learn_env=learn_env, behavior=behavior, data=data,
                        train=train, dev=dev, targets=targets, deploy_target=deploy, meta=meta)


def x_v4_sens_identity(spec: dict, inst: CellInstance) -> dict:
    """Cache identity of one II.9e draw: the II.6 draw identity (instance, behaviour, dev split, XCFG, spec, driver
    constants) and this section's code hash."""
    return {"kind": "xenv_v4_sensitivity", "draw": x_v4_draw_identity(spec, inst), "code": XV4_SENS_CODE_HASH,
            "section": "II.9e"}


def run_v4_cell_draw(spec: dict, draw: int) -> Tuple[pd.DataFrame, dict]:
    """All results of one draw of one Section II.9e cell (the steps and checks of run_cell_draw; cached in v4sens/)."""
    cell, family = spec["cell"], spec["family"]
    assert family in ("gumbel", "garnet") and spec.get("v4_kind") in XV4_SENS_KINDS, (family, spec.get("v4_kind"))
    t0 = time.perf_counter()
    inst = build_v4_instance(spec, draw)
    assert (inst.cell, inst.family, int(inst.draw)) == (cell, family, int(draw)), (inst.cell, inst.family, inst.draw)
    identity = x_v4_sens_identity(spec, inst)
    key, ident_hash = "v4sens/" + x_cache_key(cell, draw), obj_hash(identity)[:20]
    strict = True
    build_s = time.perf_counter() - t0
    hit, meta_hit = XCACHE.get(key, identity), XCACHE.get(key + "/meta", identity)
    if hit is not None and meta_hit is not None:
        del inst
        assert list(hit.columns) == XRESULT_COLUMNS, f"{key}: cached columns differ from XRESULT_COLUMNS"
        meta = json.loads(str(meta_hit["meta_json"].iloc[0]))
        meta.update(cache="loaded", session_build_s=build_s, checks=x_check_draw_rows(hit, strict))
        gc.collect()
        return hit, meta
    if (hit is None) != (meta_hit is None):
        print(f"  {key}: incomplete cache entry (results or meta missing) - recomputing this draw")
    XCACHE.require_allowed(key)
    learn = inst.learn_env
    t1 = time.perf_counter()
    raw = run_xmethods(LearnerView.from_env(learn), inst.train, f"{cell}|draw{int(draw)}")
    records = x_collect_records(raw, inst.behavior, learn)
    del raw
    t2 = time.perf_counter()
    fqe = x_fqe_env(inst.dev, inst.train, learn, name=f"fqe|{cell}|draw{int(draw)}")
    t3 = time.perf_counter()
    rows, ev_stats = x_evaluate_targets(inst, records, fqe)
    df = pd.DataFrame(rows, columns=XRESULT_COLUMNS)
    t4 = time.perf_counter()
    checks = x_check_draw_rows(df, strict)
    t5 = time.perf_counter()
    nonconv = [r["label"] for r in records if r["desc"]["converged"] is False]
    lm = learn.mask[learn.live]
    meta = _x_compact({
        "cell": cell, "family": family, "v4_kind": spec["v4_kind"], "draw": int(draw),
        "stream_cell": str(spec.get("stream_cell", cell)), "deploy_target": inst.deploy_target,
        "identity_hash": ident_hash, "code_hash": XV4_SENS_CODE_HASH[:16], "cache": "computed",
        "learn_env": {"name": learn.name, "S": learn.S, "A": learn.A, "live": int(learn.live.sum()),
                      "admissible_pairs": int(lm.sum()), "mean_admissible_actions": float(lm.sum(axis=1).mean())},
        "data": {"train_transitions": int(len(inst.train["state"])),
                 "train_episodes": int(np.unique(inst.train["episode_id"]).size),
                 "dev_transitions": int(len(inst.dev["state"])),
                 "dev_episodes": int(np.unique(inst.dev["episode_id"]).size)},
        "targets": [t[0] for t in inst.targets],
        "records": {"total": len(records), **dict(Counter(r["variant"] for r in records)),
                    "nonconverged": len(nonconv), "nonconverged_labels": nonconv},
        "evaluation": ev_stats, "fqe_rows": fqe.meta.get("fqe_rows", {}), "checks": checks,
        "timing": {"build_s": build_s, "methods_s": t2 - t1, "fqe_build_s": t3 - t2, "eval_s": t4 - t3,
                   "check_s": t5 - t4, "total_s": t5 - t0},
        "instance": inst.meta})
    XCACHE.put(key + "/meta", identity, pd.DataFrame({"meta_json": [json.dumps(meta, default=_json_default)]}))
    XCACHE.put(key, identity, df)
    back = XCACHE.get(key, identity)
    assert back is not None and list(back.columns) == XRESULT_COLUMNS and len(back) == len(df), f"{key}: cache write"
    for c in _XMETRIC_SOURCE:
        assert np.allclose(back[c].to_numpy(dtype=float), df[c].to_numpy(dtype=float), rtol=1e-14, atol=0.0,
                           equal_nan=True), f"{key}: cache round trip changed column {c}"
    assert (back["status"].astype(str).to_numpy() == df["status"].to_numpy()).all(), f"{key}: cache round trip"
    del inst, fqe, records, rows, df
    gc.collect()
    return back, meta


# Objects whose source changes what an II.9e draw computes (the checks are re-run on cached tables instead).
XV4_SENS_CODE = [x_v4_sens_build_cells, _x_v4_gumbel_live_layout, _x_v4_factor_aggregation, x_v4_kmeans,
                 x_v4_gumbel_aggregation, _x_v4_aggregation, construct_aggregated_benchmark_v4, build_v4_instance,
                 x_v4_sens_identity, run_v4_cell_draw]
XV4_SENS_CODE_HASH = obj_hash({"common": XV4_COMMON_CODE_HASH, "code": code_hash(*XV4_SENS_CODE),
                               "constants": {"kinds": list(XV4_SENS_KINDS),
                                             "agg_factors": {k: list(v) for k, v in XV4_AGG_FACTORS.items()},
                                             "kmeans_features": list(XV4_KMEANS_FEATURES),
                                             "kmeans_max_iter": XV4_KMEANS_MAX_ITER, "columns": XRESULT_COLUMNS}})


def x_v4_sens_selftest(cells: Sequence[dict], refs: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """Identity self-test of II.9e (see markdown); every comparison is asserted and reported."""
    t0 = time.perf_counter()
    out: Dict[str, Any] = {"instances": {}, "aggregation": {}}
    by_cell = {str(s["cell"]): s for s in X_CELLS}
    for s in cells:
        base = by_cell.get(str(refs.get(str(s["cell"]))))
        if base is None or base["family"] in {v["family"] for v in out["instances"].values()}:
            continue
        s4 = {**base, "stream_cell": str(base["cell"]), "empty_rule": str(XCFG.benchmark_empty_rule)}
        a, b = build_v4_instance(s4, 0), build_cell_instance(base, 0)
        same = {"instance_identity": obj_hash(a.identity()) == obj_hash(b.identity()),
                "behaviour": bool(np.array_equal(a.behavior, b.behavior)),
                "dev_split": array_dict_hash(a.dev) == array_dict_hash(b.dev)}
        assert all(same.values()), f"build_v4_instance differs from build_cell_instance on {base['cell']}: {same}"
        out["instances"][str(base["cell"])] = {"family": base["family"], "for_cell": str(s["cell"]), **same}
        del a, b
    agg_cells = [s for s in cells if s["v4_kind"] == "abstraction"]
    if agg_cells:
        s = agg_cells[0]
        env, beh = x_v4_true_system(s)
        obs = x_gumbel_aggregation(env, "obs")
        mine = _x_v4_factor_aggregation(env, "obs", ("hr", "sbp", "o2", "glucose", "abx", "vaso", "vent"))
        same_map = (all(np.array_equal(obs[k], mine[k]) for k in ("phi", "keys", "levels", "radix", "members"))
                    and all(obs[k] == mine[k] for k in ("n_live", "S", "cells_total")))
        assert same_map, "the II.9e factor aggregation does not reproduce II.8's 'obs' map"
        n_hosp, tau = int(s["n_hosp"]), int(s["tau"])
        B3 = construct_aggregated_benchmark(env, beh, "obs", n_hosp, tau, np_rng("xenv", "v4", "selftest", "agg-obs"),
                                           "v4-selftest|agg-obs")
        B4 = construct_aggregated_benchmark_v4(env, beh, obs, n_hosp, tau, np_rng("xenv", "v4", "selftest", "agg-obs"),
                                               "v4-selftest|agg-obs")
        pi = x_uniform_admissible(B3["bench"].mask, B3["bench"].live)
        same = {"factor_map": bool(same_map),
                "bench_identity": obj_hash(B3["bench"].identity()) == obj_hash(B4["bench"].identity()),
                "mask_true": bool(np.array_equal(B3["mask_true"], B4["mask_true"])),
                "decision_states_true": bool(np.array_equal(B3["decision_states_true"], B4["decision_states_true"])),
                "lift": bool(np.array_equal(B3["lift"](pi), B4["lift"](pi)))}
        assert all(same.values()), f"construct_aggregated_benchmark_v4 does not reproduce II.8c on 'obs': {same}"
        out["aggregation"]["obs_reproduces_ii8c"] = {"n_hosp": n_hosp, "tau": tau, **same}
        del B3, B4
        for kind in dict.fromkeys([*XV4CFG.aggregations, *XV4Config().aggregations]):   # full design, also in smoke
            g1, g2 = x_v4_gumbel_aggregation(env, kind), x_v4_gumbel_aggregation(env, kind)
            assert np.array_equal(g1["phi"], g2["phi"]) and np.array_equal(g1["levels"], g2["levels"]), kind
            out["aggregation"][str(kind)] = {"cells": int(g1["n_live"]), "cells_total": int(g1["cells_total"]),
                                             "members_min": int(g1["members"].min()),
                                             "members_max": int(g1["members"].max()),
                                             **{k: v for k, v in g1["report"].items()
                                                if k in ("iterations", "reseeds", "inertia")}}
    gc.collect()
    out.update(passes=True, seconds=time.perf_counter() - t0)
    return out


XV4_SENS_CELLS: List[dict] = []
XV4_SENS_REFS: Dict[str, Optional[str]] = {}
XV4_SENS_SKIPPED: List[str] = []
XV4_SENS_SELFTEST: Dict[str, Any] = {"status": "not run (xenv_v4_sensitivity off)"}
if EXPERIMENTS["xenv_v4_sensitivity"]:
    XV4_SENS_CELLS, XV4_SENS_REFS, XV4_SENS_SKIPPED = x_v4_sens_build_cells()
    print("II.9e cells:", [(c["cell"], c["v4_kind"], x_v4_draws(c)) for c in XV4_SENS_CELLS])
    if XV4_SENS_SKIPPED:
        print("NOTE: II.9e cells skipped (base cell not enabled, or a repeat in smoke mode):", XV4_SENS_SKIPPED)
    XV4_SENS_SELFTEST = x_v4_sens_selftest(XV4_SENS_CELLS, XV4_SENS_REFS)
    print(f"II.9e self-test passed in {XV4_SENS_SELFTEST['seconds']:.1f} s:",
          json.dumps({k: v for k, v in XV4_SENS_SELFTEST.items() if k not in ("passes", "seconds")}, indent=1,
                     default=_json_default))
print("II.9e code hash:", XV4_SENS_CODE_HASH[:16])

# %% [markdown]
# #### II.9e Run and paired analyses
#
# Every draw of every II.9e cell runs through `run_v4_cell_draw` (`x_v4_run_draws`, with the progress and ETA of
# II.6a; at most `XV4CFG.max_draws` draws per cell). Each new cell is then analysed beside its **reference cell**: the
# base II.6 cell from `XRESULTS` for the timeout, dead-rule and support cells, and the II.8 `agg-obs` cell from
# `XRESULTS_V3`, when present, for the abstraction cells. The reference rows are restricted to the same draw indices.
# `XRESULTS` and `XRESULTS_V3` are used when they are in memory; otherwise `xenv_results_all.csv.gz` and
# `v3_results_all.csv.gz` (and the draw-meta files) are read from `XENV_TABLE_DIR`. Only the default policies enter
# the reported statistics, as in II.8g; the theorem counts below use every policy of the new cells.
#
# * **Bracket coverage** among the policy-draws with p > 0, where p = P(any unsupported decision) on `bench_mean`
#   > 1e-12, computed with the functions of II.8i. The semantic bracket (`sem_*`) is J(true rows + terminate) ≤
#   J_true ≤ J(true rows + mean rule), whose lower end is a theorem (so `sem_*` coverage is the upper end's); the
#   benchmark bracket (`bench_*`) is J_true ∈ [J_bench_term, J_bench_mean]. Abstraction cells have no true-rows
#   targets, so they have only `bench_*`. II.8i bracketed every policy, so its rates are not directly comparable
#   with these; the base rows here are recomputed on the default policies, so new and base are.
# * **Certificates** per draw (`x_v4_pair_claims`, v_range = v_max − v_min of the family), on two kinds of rows:
#   * the true supported rows (`cc_true_rows_*`): J_term = `true_rows_terminate`, J_W = `true_rows_mean`, p from
#     `true_rows_mean`;
#   * the benchmark (`cc_bench_*`).
#
#   The kinds are the NHB certificate J_term(i) > J_W(j), the NHB-free certificate J_term(i) > J_term(j) + p_j
#   v_range, the NHB-only claims and the single-rule point claims. Each kind reports its share of the pairs and of the
#   completion-sensitive pairs (types AU + UU), its errors against J_true (ties are errors), and the draws with at
#   least one wrong certificate.
# * **Rank fidelity**: the per-draw Kendall τ_b of `bench_mean` and of `bench_terminate` against `true`
#   (`xs_rank_fidelity_draws` of II.7), and τ(mean) − τ(terminate) with a draw-bootstrap interval. The change new −
#   reference also gets a bootstrap interval (`XCFG.bootstrap` resamples). It is paired by draw for the dead-rule
#   cells, which share the hospital data; otherwise the draws are resampled independently within each cell.
# * **Construction statistics** (`constr_*`: means over draws of the construction report):
#   * decision states, coverage of the true live states, admissible pairs;
#   * mean admissible actions and the single-admissible fraction (the support of the softer physicians);
#   * dead states removed and empty-row fallbacks (the dead-state rule);
#   * censored d0 mass, and the share of decision states whose true optimal action is supported;
#   * for the abstractions: cells, live cells, decision cells and the d0 mass on decision cells.
#
# `v4_sens_summary.csv` has three rows per new cell: `side` = `new`, `base` (the reference cell) and `new - base`
# (the numeric differences and the τ intervals); a cell whose reference is missing has the `new` row only. It has
# the same rows per pool of at least two cells of one kind and family (`pooled:<kind>|<family>`, no τ intervals).
#
# **Invariants asserted.**
# * Per draw, the II.6 checks (`x_check_draw_rows`, strict projection identity), on computed and on cached tables.
# * Across draws, `x_check_results`.
# * On the new cells, the bracketing theorem counts of II.8i: no lower-bound violation on the true rows, no negative
#   width, zero width without unsupported decisions, width ≤ p·v_range, and values in [v_min, v_max].
#
# Coverage, certificate errors and rank fidelity are reported, never asserted.
#
# | Output | Content |
# |---|---|
# | `XRESULTS_V4SENS` · `xenv/v4_sens_results_all.csv.gz` | one row per draw × policy × target (`XRESULT_COLUMNS`) |
# | `XDRAW_META_V4SENS` · `xenv/v4_sens_draw_meta.csv` | per draw: construction and validation reports, timings |
# | `XV4_SENS_TABLE` · `xenv/v4_sens_summary.csv` | per new cell and pool: new, reference and difference rows |
# | `XV4_SENS_CLAIMS` · `xenv/v4_sens_certified_draws.csv` | per cell × draw × targets × pair type: claims, errors |
# | `xenv/v4_sens_rank_fidelity.csv`, `XV4_SENS_RANK` · `xenv/v4_sens_rank_fidelity_draws.csv` | rank fidelity |
# | `XV4_SENS_SUMMARY` · `xenv/v4_sens_summary.json` | status, code hash, cells, self-test, checks, deltas, timing |
#
# **Expected cost on Colab (CPU-bound).** These are estimates, not measurements. On Colab, II.6 Gumbel `best` draws
# took 10–16 s and Garnet draws about 4 s (`xenv_draw_meta.csv`); II.8 aggregated draws were estimated at 10–12 s.
# The full design has 17 cells (4 timeout, 8 dead-rule of which 2 Garnet, 2 support, 3 abstraction) × 30 draws:
# * 360 Gumbel `best` draws, about 80 min; the softer physicians may take somewhat longer;
# * 60 Garnet draws, about 5 min;
# * 90 abstraction draws, about 17 min;
# * about a minute per new timeout hazard for the exact enumeration and its physician;
# * under a minute for the self-test and a few minutes for the analyses (the reference rows are read from disk
#   when they are not in memory).
#
# That is about **1.5–2 h** in all. The lite profile (10 draws per cell) takes about a third of that. A cached draw
# reloads in about 3 s, so a re-run costs about 25 min plus the analyses.

# %%
# ============================================================
# II.9e Sensitivity cells: run and paired analyses
# ============================================================
EXECLOG.mark("II.9e Sensitivity cells: run and analyses")

XV4_SENS_CLAIM_KINDS = ("cert_nhb", "cert_free", "cert_nhb_only", "point_mean", "point_term")
XV4_SENS_CONSTRUCTION_KEYS = ("decision_states", "coverage_true_live", "admissible_pairs", "mean_admissible_actions",
                              "single_admissible_fraction", "dead_states_removed", "empty_row_fallbacks",
                              "censored_d0_mass", "true_d0_mass_on_decision_states", "true_optimum_in_mask_fraction",
                              "inadmissible_logged_fraction", "aggregated_cells_total", "aggregated_live_cells",
                              "full_live_states_in_decision_cells", "full_d0_mass_in_decision_cells",
                              "phi_inertia")
XV4_SENS_HEADLINE = ("bench_coverage_violating", "sem_upper_coverage_violating", "cc_bench_cert_nhb_share",
                     "cc_bench_cert_nhb_error_rate", "cc_bench_cert_free_share", "cc_bench_cert_free_error_rate",
                     "cc_true_rows_cert_nhb_share", "cc_true_rows_cert_nhb_error_rate", "tau_mean_delta",
                     "tau_terminate_delta", "tau_diff_delta", "constr_mean_admissible_actions",
                     "constr_single_admissible_fraction")


def _xv4_sens_table(global_name: str, file_name: str, results: bool) -> Tuple[Optional[pd.DataFrame], str]:
    """A Part II table from memory (global) when non-empty, else from XENV_TABLE_DIR; (frame or None, source)."""
    df = globals().get(global_name)
    if df is not None and len(df):
        return df, global_name
    p = XENV_TABLE_DIR / file_name
    if not p.exists():
        return None, "missing"
    if results:
        return pd.read_csv(p, float_precision="round_trip",
                           dtype={"deploy_target": str, "target": str, "target_env": str}), str(p)
    return pd.read_csv(p), str(p)


def xv4_sens_reference(draws_of: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Result rows and draw meta of the reference cells (draws 0..draws_of[cell] - 1): II.6 (XRESULTS / XDRAW_META)
    and II.8 (XRESULTS_V3 / XDRAW_META_V3), from memory or from their files, and the source of each cell."""
    res, meta, source = [], [], {}
    for rg, mg, rf, mf in (("XRESULTS", "XDRAW_META", "xenv_results_all.csv.gz", "xenv_draw_meta.csv"),
                           ("XRESULTS_V3", "XDRAW_META_V3", "v3_results_all.csv.gz", "v3_draw_meta.csv")):
        if not draws_of:
            break
        r, src = _xv4_sens_table(rg, rf, True)
        if r is None:
            continue
        r = r[r["cell"].astype(str).isin(list(draws_of))]
        r = r[pd.to_numeric(r["draw"]).astype(np.int64) < r["cell"].astype(str).map(draws_of).astype(np.int64)]
        if not len(r):
            continue
        res.append(r)
        source.update({str(c): src for c in r["cell"].unique()})
        m, _ = _xv4_sens_table(mg, mf, False)
        if m is not None and len(m) and {"cell", "draw"} <= set(m.columns):
            m = m[m["cell"].astype(str).isin(list(draws_of))]
            meta.append(m[pd.to_numeric(m["draw"]).astype(np.int64)
                          < m["cell"].astype(str).map(draws_of).astype(np.int64)])
    return (pd.concat(res, ignore_index=True) if res else pd.DataFrame(columns=XRESULT_COLUMNS),
            pd.concat(meta, ignore_index=True) if meta else pd.DataFrame(), source)


def xv4_sens_claims_draws(Wd: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, draw, targets, pair type): certificates and point claims of the default policies of one draw
    (x_v4_pair_claims, v_range = v_max - v_min of the family), scored against J_true. 'true_rows': J_term / J_W on
    the true supported rows (p from true_rows_mean); 'bench': the estimated benchmark (p from bench_mean). A draw
    with an improper evaluation on the targets is left out (counted by the caller)."""
    rows = []
    for (c, dr), d in Wd.groupby(["cell", "draw"], sort=False):
        fam = str(d["family"].iloc[0])
        v_range = float(XV3_VRANGE[fam][1] - XV3_VRANGE[fam][0])
        for tf, (lc, uc, pc) in (("true_rows", ("true_rows_terminate", "true_rows_mean", "p_true")),
                                 ("bench", ("bench_terminate", "bench_mean", "p_bench"))):
            if d[[lc, uc, pc, "true"]].isna().any().any():
                continue
            for r in x_v4_pair_claims(d[lc].to_numpy(dtype=float), d[uc].to_numpy(dtype=float),
                                      d[pc].to_numpy(dtype=float), v_range, d["true"].to_numpy(dtype=float)):
                rows.append({"cell": c, "draw": int(dr), "family": fam, "kind": str(d["v3_kind"].iloc[0]),
                             "targets": tf, **r})
    return pd.DataFrame(rows)


def xv4_sens_claims_summary(C: pd.DataFrame) -> Dict[str, Any]:
    """Claims, shares (of all pairs and of completion-sensitive AU + UU pairs), errors, error rates and draws with a
    wrong certificate, per targets, pooled over the rows of C (xv4_sens_claims_draws)."""
    out: Dict[str, Any] = {}
    for tf in ("true_rows", "bench"):
        d = C[C["targets"] == tf] if len(C) else C
        if d.empty:
            continue
        a, cs = d[d["type"] == "all"], d[d["type"].isin(["AU", "UU"])]
        pairs, pairs_cs = int(a["pairs"].sum()), int(cs["pairs"].sum())
        p = f"cc_{tf}_"
        out.update({p + "draws": int(len(a)), p + "pairs": pairs, p + "pairs_completion_sensitive": pairs_cs})
        for k in XV4_SENS_CLAIM_KINDS:
            n, e = int(a[f"{k}_claims"].sum()), int(a[f"{k}_errors"].sum())
            out.update({p + f"{k}_claims": n, p + f"{k}_share": _xv3_rate(n, pairs), p + f"{k}_errors": e,
                        p + f"{k}_error_rate": _xv3_rate(e, n),
                        p + f"{k}_share_completion_sensitive": _xv3_rate(int(cs[f"{k}_claims"].sum()), pairs_cs)})
            if k.startswith("cert_"):
                out[p + f"draws_any_wrong_{k}"] = int((a[f"{k}_errors"] > 0).sum())
    return out


def xv4_sens_rank_summary(cells: Sequence[str], rank_tab: pd.DataFrame, rank_draws: pd.DataFrame) -> Dict[str, Any]:
    """Kendall tau-b of the ranking by bench_mean / bench_terminate against true, and their difference: the II.8i
    table (mean, bootstrap interval, n) for one cell; the mean over the draws of all cells for a pool."""
    labels = (("mean", "tau_mean"), ("terminate", "tau_terminate"), ("mean - terminate", "tau_diff"))
    out: Dict[str, Any] = {}
    if len(cells) == 1:
        t = rank_tab[rank_tab["cell"] == cells[0]].set_index("completion") if len(rank_tab) else rank_tab
        for comp, lab in labels:
            if len(t) and comp in t.index:
                for sfx in ("mean", "ci_low", "ci_high", "n"):
                    out[f"{lab}_{sfx}"] = float(t.loc[comp, f"kendall_tau_b_{sfx}"])
        return out
    d = rank_draws[rank_draws["cell"].isin(list(cells)) & rank_draws["completion"].isin(["mean", "terminate"])] \
        if len(rank_draws) else rank_draws
    if d.empty:
        return out
    t = d.pivot_table(index=["cell", "draw"], columns="completion", values="kendall_tau_b", aggfunc="first")
    if {"mean", "terminate"} <= set(t.columns):
        t["mean - terminate"] = t["mean"] - t["terminate"]
    for comp, lab in labels:
        if comp in t.columns:
            v = t[comp].dropna()
            out[f"{lab}_mean"], out[f"{lab}_n"] = (float(v.mean()) if len(v) else np.nan), float(len(v))
    return out


def _xv4_mean_gap(parts: List[np.ndarray]) -> np.ndarray:
    """Bootstrap statistic: mean of the first stratum minus mean of the second (independent draws)."""
    return parts[0][..., 0].mean(axis=1) - parts[1][..., 0].mean(axis=1)


def xv4_sens_tau_delta(rank_draws: pd.DataFrame, cell: str, ref: str, paired: bool, n_boot: int) -> Dict[str, Any]:
    """Kendall tau-b of `cell` minus that of its reference cell, per completion and for tau(mean) - tau(terminate),
    with a percentile bootstrap interval over draws: paired by draw when both cells share the hospital draws,
    otherwise draws resampled independently within each cell."""
    out: Dict[str, Any] = {"delta_paired": bool(paired)}
    d = rank_draws[rank_draws["cell"].isin([cell, ref]) & rank_draws["completion"].isin(["mean", "terminate"])] \
        if len(rank_draws) else rank_draws
    if d.empty:
        return out
    t = d.pivot_table(index=["cell", "draw"], columns="completion", values="kendall_tau_b", aggfunc="first")
    if not {cell, ref} <= set(t.index.get_level_values("cell")):
        return out
    if {"mean", "terminate"} <= set(t.columns):
        t["diff"] = t["mean"] - t["terminate"]
    for col, lab in (("mean", "tau_mean"), ("terminate", "tau_terminate"), ("diff", "tau_diff")):
        if col not in t.columns:
            continue
        a, b = t.xs(cell, level="cell")[col].dropna(), t.xs(ref, level="cell")[col].dropna()
        if paired:
            j = a.index.intersection(b.index)
            v = (a[j] - b[j]).to_numpy(dtype=np.float64)
            point, reps = xs_stratified_bootstrap({cell: v[:, None]} if len(v) else {}, xs_stat_mean(),
                                                  ("v4sens", cell, lab), n_boot)
            n = len(v)
        elif len(a) and len(b):
            point, reps = xs_stratified_bootstrap({"new": a.to_numpy(dtype=np.float64)[:, None],
                                                   "base": b.to_numpy(dtype=np.float64)[:, None]},
                                                  _xv4_mean_gap, ("v4sens", cell, lab), n_boot)
            n = min(len(a), len(b))
        else:
            continue
        lo, hi = xs_ci(reps)
        out.update({f"{lab}_delta": float(point), f"{lab}_delta_ci_low": lo, f"{lab}_delta_ci_high": hi,
                    f"{lab}_delta_n": int(n)})
    return out


def xv4_sens_construction(cells: Sequence[str], meta: pd.DataFrame) -> Dict[str, Any]:
    """Means over draws of the construction report (flattened draw meta 'instance.construction.<key>')."""
    if meta is None or not len(meta) or "cell" not in meta.columns:
        return {}
    m = meta[meta["cell"].astype(str).isin(list(cells))]
    out: Dict[str, Any] = {}
    for k in XV4_SENS_CONSTRUCTION_KEYS:
        col = f"instance.construction.{k}"
        if col in m.columns:
            v = pd.to_numeric(m[col], errors="coerce").dropna()
            if len(v):
                out[f"constr_{k}"] = float(v.mean())
    return out


def xv4_sens_unit(cells: Sequence[str], Wd: pd.DataFrame, C: pd.DataFrame, rank_tab: pd.DataFrame,
                  rank_draws: pd.DataFrame, meta: pd.DataFrame) -> Dict[str, Any]:
    """All analyses of one unit (one cell or a pool of cells): bracketing, certificates, rank fidelity, construction."""
    d = Wd[Wd["cell"].isin(list(cells))]
    out: Dict[str, Any] = {"n_cells": len(cells), "n_draws": int(d[["cell", "draw"]].drop_duplicates().shape[0]),
                           "default_policies_per_draw": (float(d.groupby(["cell", "draw"]).size().median())
                                                         if len(d) else np.nan)}
    out.update({f"sem_{k}": v for k, v in xv3_bracket_semantics(d).items()})
    out.update({f"bench_{k}": v for k, v in xv3_bracket_benchmark(d).items()})
    out.update(xv4_sens_claims_summary(C[C["cell"].isin(list(cells))] if len(C) else C))
    out.update(xv4_sens_rank_summary(list(cells), rank_tab, rank_draws))
    out.update(xv4_sens_construction(list(cells), meta))
    return out


def _xv4_is_num(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def xv4_sens_analyses(cells: Sequence[dict], refs: Dict[str, Optional[str]], res_new: pd.DataFrame,
                      meta_new: pd.DataFrame, n_boot: int
                      ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Section II.9e analyses: (summary table, per-draw frames, checks) for the new cells and their references."""
    names = [str(s["cell"]) for s in cells]
    draws_of: Dict[str, int] = {}
    for s in cells:
        r = refs.get(str(s["cell"]))
        if r:
            draws_of[r] = max(draws_of.get(r, 0), x_v4_draws(s))
    ref_res, ref_meta, source = xv4_sens_reference(draws_of)
    present = [c for c in draws_of if c in set(ref_res["cell"].astype(str))]
    kind_of = {**{str(s["cell"]): str(s["v4_kind"]) for s in cells}, **{c: "reference" for c in present}}
    _, frame_checks = xv3_bracket_frame(res_new, kind_of)
    all_res = pd.concat([res_new, ref_res], ignore_index=True) if len(ref_res) else res_new
    W, _ = xv3_bracket_frame(all_res, kind_of)
    Wd = W[W["is_default"]]
    C = xv4_sens_claims_draws(Wd)
    rank_draws, rank_tab = xv3_rank_fidelity(all_res, names + present, n_boot)
    meta_all = pd.concat([meta_new, ref_meta], ignore_index=True) if len(ref_meta) else meta_new
    del all_res, W
    rows: List[dict] = []

    def emit(head: dict, new_cells: List[str], ref_cells: List[str], tau: Optional[dict]) -> None:
        new = xv4_sens_unit(new_cells, Wd, C, rank_tab, rank_draws, meta_all)
        rows.append({**head, "side": "new", "unit": ";".join(new_cells), **new})
        if not ref_cells:
            return
        base = xv4_sens_unit(ref_cells, Wd, C, rank_tab, rank_draws, meta_all)
        rows.append({**head, "side": "base", "unit": ";".join(ref_cells), **base})
        delta = {k: new[k] - base[k] for k in new if k in base and k not in ("n_cells", "n_draws")
                 and not k.endswith(("_ci_low", "_ci_high")) and _xv4_is_num(new[k]) and _xv4_is_num(base[k])}
        rows.append({**head, "side": "new - base", "unit": f"{';'.join(new_cells)} - {';'.join(ref_cells)}", **delta,
                     **(tau or {})})

    for s in cells:
        c, r = str(s["cell"]), refs.get(str(s["cell"]))
        paired = bool(r) and str(s.get("stream_cell", c)) == r
        head = {"row_type": "cell", "cell": c, "v4_kind": s["v4_kind"], "family": s["family"], "reference": r or "",
                "reference_source": source.get(str(r), "missing") if r else "none", "paired": paired}
        ok = r in present
        emit(head, [c], [r] if ok else [], xv4_sens_tau_delta(rank_draws, c, r, paired, n_boot) if ok else None)
    pools: Dict[Tuple[str, str], List[str]] = {}
    for s in cells:
        pools.setdefault((str(s["v4_kind"]), str(s["family"])), []).append(str(s["cell"]))
    for (kind, fam), cs in pools.items():
        if len(cs) < 2:
            continue
        rc = sorted({str(refs[c]) for c in cs if refs.get(c) in present})
        head = {"row_type": "pool", "cell": f"pooled:{kind}|{fam}", "v4_kind": kind, "family": fam,
                "reference": ";".join(rc), "reference_source": ";".join(sorted({source[c] for c in rc})),
                "paired": all(spec.get("stream_cell") == refs.get(str(spec["cell"])) for spec in cells
                              if str(spec["cell"]) in cs)}
        emit(head, cs, rc, None)
    table = pd.DataFrame(rows)
    nd = Wd[Wd["cell"].isin(names)][["cell", "draw"]].drop_duplicates()
    have = (C[C["cell"].isin(names) & (C["type"] == "all")].groupby("targets").size() if len(C)   # one row per draw
            else pd.Series(dtype=int))
    tc = {k: int(frame_checks.get(k, 0)) for k in ("values_outside_value_range", "lower_bound_violations_true_rows",
                                                   "negative_width_true_rows", "negative_width_bench",
                                                   "nonzero_width_admissible", "width_above_bound")}
    checks = {"bracket_frame": frame_checks, "theorem_counts": tc, "theorem_passes": all(v == 0 for v in tc.values()),
              "reference_cells": present, "reference_cells_missing": [c for c in draws_of if c not in present],
              "reference_source": source, "new_cell_draws": int(len(nd)),
              "certificate_draws_by_targets": {str(k): int(v) for k, v in have.items()}}
    return table, {"claims_draws": C, "rank_draws": rank_draws, "rank_table": rank_tab}, checks


XRESULTS_V4SENS = pd.DataFrame(columns=XRESULT_COLUMNS)
XDRAW_META_V4SENS = pd.DataFrame()
XV4_SENS_TABLE = pd.DataFrame()
XV4_SENS_CLAIMS = pd.DataFrame()
XV4_SENS_RANK = pd.DataFrame()
XV4_SENS_SUMMARY: Dict[str, Any] = {"status": "not run (xenv_v4_sensitivity off)", "code_hash": XV4_SENS_CODE_HASH,
                                    "checks": {}, "elapsed_s": 0.0, "draws_complete": None}
if EXPERIMENTS["xenv_v4_sensitivity"] and not XV4_SENS_CELLS:
    XV4_SENS_SUMMARY = {"status": "not run: no configured II.9e cell is enabled", "code_hash": XV4_SENS_CODE_HASH,
                        "cells_skipped": XV4_SENS_SKIPPED, "checks": {"selftest": XV4_SENS_SELFTEST}, "elapsed_s": 0.0,
                        "draws_complete": None}
elif EXPERIMENTS["xenv_v4_sensitivity"]:
    _t0 = time.perf_counter()
    _frames, XDRAW_META_V4SENS, _run = x_v4_run_draws(XV4_SENS_CELLS, run_v4_cell_draw, "II.9e",
                                                      time_key=lambda s: f"{s['family']}|{s['v4_kind']}")
    XRESULTS_V4SENS = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame(columns=XRESULT_COLUMNS)
    del _frames
    gc.collect()
    assert list(XRESULTS_V4SENS.columns) == XRESULT_COLUMNS
    _cross = x_check_results(XRESULTS_V4SENS)
    XRESULTS_V4SENS.to_csv(XENV_TABLE_DIR / "v4_sens_results_all.csv.gz", index=False, compression="gzip")
    XDRAW_META_V4SENS.to_csv(XENV_TABLE_DIR / "v4_sens_draw_meta.csv", index=False)
    XV4_SENS_TABLE, _sframes, _schecks = xv4_sens_analyses(XV4_SENS_CELLS, XV4_SENS_REFS, XRESULTS_V4SENS,
                                                           XDRAW_META_V4SENS, int(XCFG.bootstrap))
    XV4_SENS_CLAIMS, XV4_SENS_RANK = _sframes["claims_draws"], _sframes["rank_draws"]
    _outputs = {"v4_sens_summary.csv": XV4_SENS_TABLE, "v4_sens_certified_draws.csv": XV4_SENS_CLAIMS,
                "v4_sens_rank_fidelity.csv": _sframes["rank_table"], "v4_sens_rank_fidelity_draws.csv": XV4_SENS_RANK}
    for _name, _df in _outputs.items():
        _df.to_csv(XENV_TABLE_DIR / _name, index=False)
    _have = XRESULTS_V4SENS.groupby("cell")["draw"].nunique().to_dict() if len(XRESULTS_V4SENS) else {}
    _per = {str(s["cell"]): {"v4_kind": s["v4_kind"], "draws": int(_have.get(s["cell"], 0)),
                             "configured_draws": x_v4_draws(s), "reference": XV4_SENS_REFS.get(str(s["cell"]))}
            for s in XV4_SENS_CELLS}
    _dm = {}
    for _c in ("checks.max_slack_vs_optimum", "checks.max_slack_vs_admissible_optimum",
               "checks.projection_identity_worst"):
        _dm[f"max_{_c.split('.', 1)[1]}"] = (float(XDRAW_META_V4SENS[_c].max()) if _c in XDRAW_META_V4SENS else None)
    _dm["nonconverged_records"] = (int(XDRAW_META_V4SENS["records.nonconverged"].sum())
                                   if "records.nonconverged" in XDRAW_META_V4SENS else 0)
    _dl = XV4_SENS_TABLE[(XV4_SENS_TABLE["side"] == "new - base") & (XV4_SENS_TABLE["row_type"] == "cell")] \
        if len(XV4_SENS_TABLE) else XV4_SENS_TABLE
    XV4_SENS_SUMMARY = {
        "status": "run", "code_hash": XV4_SENS_CODE_HASH, "cells": [s["cell"] for s in XV4_SENS_CELLS],
        "cells_skipped": XV4_SENS_SKIPPED, "draws_per_cell": _per,
        "draws_complete": bool(_per) and all(v["draws"] == v["configured_draws"] for v in _per.values()),
        "driver": {k: v for k, v in _run.items() if k != "draws_per_cell"},
        "checks": {"selftest": XV4_SENS_SELFTEST, "cross_draw": _cross, "draw_checks": _dm, **_schecks,
                   "passes": bool(XV4_SENS_SELFTEST.get("passes")) and bool(_schecks["theorem_passes"])},
        "deltas": {str(r["cell"]): {k: float(r[k]) for k in XV4_SENS_HEADLINE if k in r and pd.notna(r[k])}
                   for r in _dl.to_dict("records")},
        "outputs": ["xenv/v4_sens_results_all.csv.gz", "xenv/v4_sens_draw_meta.csv",
                    *[f"xenv/{k}" for k in _outputs], "xenv/v4_sens_summary.json"],
        "elapsed_s": time.perf_counter() - _t0}
    if len(XV4_SENS_TABLE):
        _cols = [c for c in ("cell", "side", "n_draws", "sem_upper_coverage_violating", "bench_coverage_violating",
                             "cc_bench_cert_nhb_share", "cc_bench_cert_nhb_error_rate", "cc_bench_cert_free_share",
                             "cc_bench_cert_free_error_rate", "tau_mean_mean", "tau_terminate_mean", "tau_diff_mean",
                             "constr_mean_admissible_actions", "constr_single_admissible_fraction")
                 if c in XV4_SENS_TABLE.columns]
        display(XV4_SENS_TABLE[_cols].round(4))
        del _cols
    del _t0, _run, _cross, _sframes, _schecks, _outputs, _name, _df, _have, _per, _dm, _c, _dl
    gc.collect()
save_json(XV4_SENS_SUMMARY, XENV_TABLE_DIR / "v4_sens_summary.json")
print(json.dumps({k: v for k, v in XV4_SENS_SUMMARY.items() if k not in ("draws_per_cell", "checks", "deltas")},
                 indent=1, default=_json_default))
print("II.9e complete:", len(XRESULTS_V4SENS), "result rows,", len(XDRAW_META_V4SENS), "draws.")
if XV4_SENS_SUMMARY["status"] == "run" and not XV4_SENS_SUMMARY["checks"]["passes"]:
    raise AssertionError(f"II.9e theorem checks failed: {XV4_SENS_SUMMARY['checks']['theorem_counts']} "
                         "(xenv/v4_sens_summary.json)")
