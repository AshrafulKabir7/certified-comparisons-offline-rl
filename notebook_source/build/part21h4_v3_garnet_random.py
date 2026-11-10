# %% [markdown]
# ### II.8d Garnet with independent random rows for unsupported actions
#
# In the harm family (II.3c), the true row of every non-preferred action is a mixture,
# (1 − |h|)·mean_{K(s)} P_base + |h|·e_death (or e_goal), so the sign of the hidden benefit of unsupported actions
# is fixed by h. The `hrand` cells leave that family. A draw keeps the base MDP, the preferred sets K(s) and the
# behaviour policy of the harm cells of the same draw (`garnet_true_env` with stream `("xenv", "garnet", "base",
# draw)`, so the cells share common random numbers). Every **non-preferred** pair (s, a ∉ K(s)) then gets an
# independent row from the Garnet row generator. The generator absorbs with hazard U(`garnet_hazard`), reaching the
# goal with probability U(0, 1) and death otherwise; the remaining mass moves to `branching` distinct uniformly random
# live successors with Dirichlet(1) weights. The rows come from their own stream,
# `("xenv", "garnet", "random-rows", "base", draw)`. They are drawn for all S·A pairs, so they do not depend on ε,
# and they are drawn after K(s) is fixed, so they are independent of the preference. Preferred rows stay the base
# rows; this is asserted.
#
# Whether an unsupported action is better or worse than the supported ones is therefore random per pair. The theorem's
# no-hidden-benefit condition can hold or fail, which tests the boundary of the bracketing result. Each draw reports,
# under the exact optimum of its true system, the share of non-preferred pairs whose action value exceeds the mean
# value of the preferred actions (`nonpreferred_beneficial_share`), and the mean, minimum and maximum of that
# advantage. Every row absorbs with probability at least min(`garnet_hazard`) > 0, so every policy is proper (asserted).
# The benchmark construction, user data and the five targets are those of the Garnet cells in II.3d.

# %%
# ============================================================
# II.8d Garnet family with independent random unsupported rows
# ============================================================
def garnet_random_rows_env(draw_key: tuple, n_states: int, n_actions: int, branching: int,
                           hazard_range: Tuple[float, float], eps: float) -> Tuple[FiniteEnv, np.ndarray, Dict[str, Any]]:
    """True Garnet CMDP of one draw whose non-preferred rows are independent random rows, its behaviour policy with
    softening `eps` (identical to the harm cells of the draw), and a report of the hidden benefit of those rows."""
    S, A, B = int(n_states), int(n_actions), int(branching)
    lo, hi = (float(x) for x in hazard_range)
    eps = float(eps)
    assert 0.0 < eps < (A - 3) / A, "preferred actions must keep a larger behaviour probability than the others"
    base0, pi = garnet_true_env(draw_key, S, A, B, (lo, hi), 0.0, eps)
    goal, death = S, S + 1
    assert (base0.goal, base0.sink) == (goal, death)
    pl = pi[:S]
    K = pl > pl.min(axis=1, keepdims=True)                               # preferred set K(s) of the behaviour
    k = K.sum(axis=1)
    assert np.isin(k, (1, 2, 3)).all() and np.allclose((pl * K).sum(axis=1), 1.0 - eps, atol=1e-12)
    rng = np_rng("xenv", "garnet", "random-rows", *draw_key)
    succ = np.argsort(rng.random((S * A, S)), axis=1)[:, :B]
    w = rng.dirichlet(np.ones(B), size=S * A)
    haz = rng.uniform(lo, hi, size=S * A)
    q_goal = rng.uniform(0.0, 1.0, size=S * A)
    rows = np.zeros((S * A, S + 2))
    rows[np.arange(S * A)[:, None], succ] = (1.0 - haz)[:, None] * w
    rows[:, goal] = haz * q_goal
    rows[:, death] = haz * (1.0 - q_goal)
    rows = rows.reshape(S, A, S + 2)
    P = base0.P.copy()
    PL = P[:S]
    PL[~K] = rows[~K]
    assert np.array_equal(P[:S][K], base0.P[:S][K]), "preferred rows must stay the base rows"
    tag = "|".join(str(x) for x in draw_key)
    meta = {key: val for key, val in base0.meta.items() if key != "harm"}
    meta.update(row_model="independent Garnet rows on non-preferred pairs",
                row_stream=["xenv", "garnet", "random-rows", *draw_key])
    env = FiniteEnv(name=f"garnet|hrand|eps{eps}|{tag}", kind="garnet", P=P, entry_reward=base0.entry_reward,
                    d0=base0.d0, terminals=base0.terminals, sink=death, goal=goal, mask=np.ones((S + 2, A), bool),
                    meta=meta)
    assert np.allclose(env.P.sum(axis=2), 1.0, atol=1e-12) and (env.P >= 0).all()
    assert x_trap_states(env.P, env.live, env.mask) == 0, "every Garnet policy must be proper"
    _, V, _ = x_optimal(env, "given")
    Q = env.evaluator().q_values(V, "mean")[:S]
    adv = Q - ((Q * K).sum(axis=1) / k)[:, None]
    a_np = adv[~K]
    report = {"nonpreferred_pairs": int((~K).sum()),
              "nonpreferred_beneficial_share": float((a_np > 1e-12).mean()),
              "nonpreferred_mean_advantage": float(a_np.mean()), "nonpreferred_min_advantage": float(a_np.min()),
              "nonpreferred_max_advantage": float(a_np.max()),
              "nonpreferred_mean_absorption": float(rows[~K][:, [goal, death]].sum(axis=1).mean())}
    return env, validate_policy(pi, f"Garnet behaviour {env.name}", shape=(S + 2, A)), report


_X_V3_GARNET_CACHE: "OrderedDict[tuple, Tuple[FiniteEnv, np.ndarray, Dict[str, Any]]]" = OrderedDict()


def _x_v3_garnet_env(spec: dict, draw: int, keep: int = 4) -> Tuple[FiniteEnv, np.ndarray, Dict[str, Any]]:
    key = (spec["cell"], int(draw))
    if key in _X_V3_GARNET_CACHE:
        _X_V3_GARNET_CACHE.move_to_end(key)
        return _X_V3_GARNET_CACHE[key]
    val = garnet_random_rows_env(("base", int(draw)), spec["n_states"], spec["n_actions"], spec["branching"],
                                 spec["hazard_range"], spec["eps"])
    _X_V3_GARNET_CACHE[key] = val
    while len(_X_V3_GARNET_CACHE) > keep:
        _X_V3_GARNET_CACHE.popitem(last=False)
    return val
