# %% [markdown]
# ### II.8c Benchmarks on aggregated states: state aggregation and hidden confounding
#
# ICU-Sepsis estimates its benchmark on k-means clusters of patient features (II.3d), while the clinicians in its
# source data acted on the patients, not on the clusters. The cells of II.3–II.6 had neither feature, because their
# benchmark was estimated on the true Markov states. The aggregated cells keep the exact Gumbel system as the truth and the
# physician of II.3b, who sees the full state, diabetes included. The benchmark, however, is built on an
# aggregation φ of the true states:
#
# * **`agg-obs`**: φ(s) is the observation without the diabetes indicator (full index mod 720, i.e. 3·3·2·5·2·2·2 =
#   720 observation cells). A live state and its diabetic twin share one cell, so the benchmark has 303 live cells.
#   Diabetes is a hidden confounder: it changes both the physician's action and the dynamics (vasopressors, glucose),
#   but the benchmark cannot see it.
# * **`agg-proj`**: glucose is dropped as well, φ(s) = (hr, sbp, o2, abx, vaso, vent) with 3·3·2·2·2·2 = 144 cells, of
#   which 112 contain a live state (at most two abnormal vitals among hr, sbp, o2). Death and discharge then also
#   depend on the unobserved glucose level.
#
# `x_gumbel_aggregation` returns φ as an index map. Live cells keep the increasing order of their cell index, and
# death, discharge and timeout follow in the true order. Absorbing states map to absorbing states, so φ preserves
# absorption, entry rewards and the timeout hazard h.
#
# **Construction** (`construct_aggregated_benchmark`). Hospital episodes are simulated on the true full-state
# system with the physician and validated there. They are then mapped through φ (`x_aggregate_dataset`) and
# validated again on the aggregated state space. The benchmark is then estimated by exactly the code of II.3d
# (`x_benchmark_from_data`): count mask with threshold τ, ICU-Sepsis dead-state removal, known-horizon
# maximum-likelihood rows, mean completion, empirical d0 and clinician estimate. Benchmark users and learners act on
# the cells. `x_benchmark_from_data` reads the state space, the absorbing structure, the rewards, the censoring hazard
# and d0 of its "true" environment; it never reads P. Here that environment is `x_aggregated_env`, the population
# limit of the count estimate. Its row for (x, a) is Σ_s μ_β(s) π_β(a|s) P(φ⁻¹(·)|s, a) / Σ_s μ_β(s) π_β(a|s) over
# the members s of x, where μ_β is the physician's exact occupancy (equal weights for members it never visits). Its
# d0 is the pushforward of the true d0. Under the aggregated physician π_β,agg(a|x) = Σ_s μ_β(s) π_β(a|s) / Σ_s μ_β(s),
# this model reproduces the true occupancy of every cell, and hence the true value of the physician; this is
# asserted to 1e-9.
#
# **Targets.** `bench_mean` and `bench_terminate` live on the benchmark. `true` is the true full-state system with
# the pulled-back mask M(φ(s)), evaluated with the lifted policy π(a | φ(s)). Cells outside the benchmark play the
# most frequent hospital action, as in II.3d. The `true_rows_*` targets of II.3d are not used, because they compare
# rows on one state space. `REF optimum` on `true` is the optimum over full-state policies, which see diabetes;
# a policy on the cells cannot always reach it.

# %%
# ============================================================
# II.8c Aggregated-state Gumbel benchmarks
# ============================================================
def x_gumbel_aggregation(env: FiniteEnv, aggregation: str) -> Dict[str, Any]:
    """Index map phi of the exact Gumbel env onto observation cells: 'obs' drops diabetes, 'proj' also glucose."""
    assert aggregation in ("obs", "proj"), aggregation
    space = gumbel_state_space()
    live_full = space["live_full"]
    n_live = int(env.meta["n_live"])
    assert env.kind == "gumbel" and n_live == len(live_full) and env.S == n_live + 3, env.name
    death, discharge, timeout = n_live, n_live + 1, n_live + 2
    assert (env.terminals, env.sink, env.goal, int(env.meta["timeout_state"])) == \
        ((death, discharge, timeout), death, discharge, timeout), "unexpected absorbing layout"
    factors = (("hr", "sbp", "o2", "glucose", "abx", "vaso", "vent") if aggregation == "obs"
               else ("hr", "sbp", "o2", "abx", "vaso", "vent"))
    fidx = [GumbelSim.FACTORS.index(f) for f in factors]
    radix = np.array([GumbelSim.LEVELS[i] for i in fidx], dtype=np.int64)
    strides = np.ones(len(radix), dtype=np.int64)
    for j in range(len(radix) - 2, -1, -1):
        strides[j] = strides[j + 1] * radix[j + 1]                      # first factor most significant
    full = np.array([space["full"][f] for f in live_full], dtype=np.int64)
    cell = full[:, fidx] @ strides
    if aggregation == "obs":
        assert np.array_equal(cell, live_full % GumbelSim.N_OBS), "obs cell must equal full index mod 720"
    keys = np.unique(cell)
    n_agg = int(len(keys))
    phi = np.empty(env.S, dtype=np.int64)
    phi[:n_live] = np.searchsorted(keys, cell)
    phi[n_live:] = n_agg + np.arange(3)
    levels = (keys[:, None] // strides[None, :]) % radix[None, :]
    members = np.bincount(phi[:n_live], minlength=n_agg)
    assert members.min() >= 1 and int(members.sum()) == n_live
    return {"aggregation": aggregation, "phi": phi, "n_live": n_agg, "S": n_agg + 3,
            "cells_total": int(radix.prod()), "factors": factors, "radix": radix, "keys": keys, "levels": levels,
            "members": members}


def x_aggregated_env(true_env: FiniteEnv, agg: Dict[str, Any], behavior: np.ndarray,
                     name: str) -> Tuple[FiniteEnv, Dict[str, Any]]:
    """Population limit of the benchmark's count estimate on the cells of `agg` under `behavior` (see markdown)."""
    phi, Sa, nA = agg["phi"], int(agg["S"]), int(agg["n_live"])
    S, A = true_env.S, true_env.A
    beh = validate_policy(behavior, f"{name} behaviour", shape=(S, A))
    G = np.zeros((S, Sa))
    G[np.arange(S), phi] = 1.0
    ev = x_evaluate(true_env, beh, "given", vectors=True)
    assert ev["status"] == "ok", ev
    mu = ev["occupancy_vector"]
    Pphi = (true_env.P.reshape(S * A, S) @ G).reshape(S, A, Sa)
    W = mu[:, None] * beh
    W[~true_env.live] = 0.0
    W_eq = np.where(true_env.live[:, None], 1.0, 0.0) * np.ones((1, A))
    num = (G.T @ (W[:, :, None] * Pphi).reshape(S, A * Sa)).reshape(Sa, A, Sa)
    den = G.T @ W
    num_eq = (G.T @ (W_eq[:, :, None] * Pphi).reshape(S, A * Sa)).reshape(Sa, A, Sa)
    den_eq = G.T @ W_eq
    P = np.zeros((Sa, A, Sa))
    seen = den[:nA] > 0
    P[:nA] = np.where(seen[:, :, None], num[:nA] / np.maximum(den[:nA], 1e-300)[:, :, None],
                      num_eq[:nA] / np.maximum(den_eq[:nA], 1e-300)[:, :, None])
    for t in range(nA, Sa):
        P[t, :, t] = 1.0
    entry = np.zeros(Sa)
    for t in true_env.terminals:
        entry[phi[t]] = true_env.entry_reward[t]
    d0 = G.T @ true_env.d0
    radix = np.asarray(agg["radix"], dtype=np.float64)
    feats = np.zeros((Sa, len(radix)), dtype=np.float32)
    feats[:nA] = agg["levels"] / np.maximum(radix - 1.0, 1.0)
    timeout = int(phi[int(true_env.meta["timeout_state"])])
    env = FiniteEnv(name=name, kind="aggregate", P=P, entry_reward=entry, d0=d0,
                    terminals=tuple(int(phi[t]) for t in true_env.terminals), sink=int(phi[true_env.sink]),
                    goal=int(phi[true_env.goal]), mask=np.ones((Sa, A), dtype=bool), features=feats,
                    meta={"aggregation": agg["aggregation"], "factors": list(agg["factors"]), "true_env": true_env.name,
                          "n_live": nA, "weights": "behaviour occupancy", "behavior": array_hash(beh),
                          "censoring_hazard": true_env.meta["censoring_hazard"], "timeout_state": timeout,
                          "dangling_target": timeout, "dangling_target_label": "timeout"})
    row_dev = float(np.abs(P.sum(axis=2) - 1.0).max())
    assert row_dev < 1e-12 and (P >= 0).all(), row_dev
    assert x_trap_states(env.P, env.live, np.ones((Sa, A), dtype=bool)) == 0, "aggregated model must absorb"
    occ_agg = G.T @ mu
    pi_agg = np.full((Sa, A), 1.0 / A)
    vis = occ_agg[:nA] > 0
    pi_agg[:nA][vis] = (G.T @ (mu[:, None] * beh))[:nA][vis] / occ_agg[:nA][vis, None]
    ev_agg = x_evaluate(env, pi_agg, "given", vectors=True)
    assert ev_agg["status"] == "ok", ev_agg
    check = {"behaviour_value_true": float(ev["return"]), "behaviour_value_aggregated_model": float(ev_agg["return"]),
             "behaviour_value_abs_diff": abs(float(ev["return"]) - float(ev_agg["return"])),
             "cell_occupancy_max_abs_diff": float(np.abs(ev_agg["occupancy_vector"] - occ_agg).max()),
             "cells_visited_by_behaviour": int(vis.sum()), "max_row_sum_deviation": row_dev}
    assert check["behaviour_value_abs_diff"] <= 1e-9 and check["cell_occupancy_max_abs_diff"] <= 1e-8 * max(1.0, occ_agg.max()), \
        f"{name}: the aggregated model does not reproduce the behaviour's occupancy ({check})"
    return env, check


def x_aggregate_dataset(data: Dict[str, np.ndarray], phi: np.ndarray, agg_env: FiniteEnv) -> Dict[str, np.ndarray]:
    """The same episodes with states and next states mapped to their cells (rows, order and rewards unchanged)."""
    out = {k: np.array(v, copy=True) for k, v in data.items()}
    out["state"] = phi[data["state"]].astype(np.int64)
    out["next_state"] = phi[data["next_state"]].astype(np.int64)
    out["logged_action_admissible"] = agg_env.mask[out["state"], out["action"]]
    return out


def construct_aggregated_benchmark(true_env: FiniteEnv, behavior: np.ndarray, aggregation: str, n_hosp: int,
                                   tau: int, rng: np.random.Generator, name: str) -> Dict[str, Any]:
    """Hospital data from the true full-state system -> ICU-style benchmark on the cells phi (Section II.8c)."""
    agg = x_gumbel_aggregation(true_env, aggregation)
    phi = agg["phi"]
    agg_env, agg_check = x_aggregated_env(true_env, agg, behavior, f"{true_env.name}|agg-{aggregation}")
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
    rep.update({"construction": f"agg-{aggregation}", "aggregation": aggregation,
                "aggregated_cells_total": agg["cells_total"], "aggregated_live_cells": agg["n_live"],
                "members_per_live_cell_max": int(agg["members"].max()),
                "full_live_states": int(true_env.live.sum()), "full_states_logged": int(logged.sum()),
                "full_live_states_in_decision_cells": int(in_cells.sum()),
                "full_logged_states_in_decision_cells": int((in_cells & logged).sum()),
                "full_d0_mass_in_decision_cells": float(true_env.d0[in_cells].sum()),
                **{f"aggregation_check_{k}": v for k, v in agg_check.items()},
                "hospital_validation": hosp_report, "aggregated_hospital_validation": agg_report})
    del hosp, hosp_agg
    return {"bench": out["bench"], "bench_behavior": out["bench_behavior"], "index": out["index"],
            "mask_true": mask_true, "lift": lift, "fallback_action": out["fallback_action"], "report": rep,
            "decision_states_true": np.flatnonzero(in_cells & logged), "decision_cells": cells, "phi": phi,
            "agg_env": agg_env}
