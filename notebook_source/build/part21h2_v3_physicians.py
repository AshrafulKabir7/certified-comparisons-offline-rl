# %% [markdown]
# ### II.8b Gumbel physicians that are not near-optimal
#
# `gumbel_physician_v3(env, ε, kind)` extends `gumbel_physician` (II.3b) with two kinds. Every kind puts 1 − ε on
# one action per state and ε/7 on each other action. The kinds `optimal` and `withhold-vaso` call
# `gumbel_physician` unchanged.
#
# * **`confounded`**: the optimal policy of the diabetes-blind model, here the non-diabetic dynamics, applied to
#   every patient. A diabetic patient gets the optimal action of the non-diabetic patient with the same observation
#   (hr, sbp, o2, glucose, treatment flags; `gumbel_diabetes_counterpart`). A step never changes the diabetes
#   indicator, so the live states of the two groups never reach each other; this is asserted on P. The true optimum
#   restricted to the non-diabetic states is therefore the optimum of the non-diabetic model. Non-diabetic patients
#   are treated exactly as by the `optimal` physician. Diabetic patients are treated as if they were not diabetic.
# * **`withhold-abx`**: the exact optimum among policies that never give antibiotics on live states (admissible
#   optimum under the mask 1[antibiotic bit of a = 0]; a = 4·abx + 2·vent + vaso). Antibiotic actions then get ε/7
#   each, so they are rarely supported in the hospital data, although they can be beneficial in truth.
#
# The benchmark construction, user data and targets of these cells are those of the `best` construction (II.3d),
# with the five targets `bench_mean`, `bench_terminate`, `true`, `true_rows_mean` and `true_rows_terminate`.
# `xenv/v3_physician_report.csv` describes each physician on the true system: its value, the value of its
# deterministic core, how often its action differs from the true optimum (over live states, weighted by its own
# occupancy, and by diabetes group), and how often it gives each treatment.

# %%
# ============================================================
# II.8b Gumbel physicians: confounded and antibiotic-withholding
# ============================================================
GUMBEL_PHYSICIAN_KINDS_V3 = GUMBEL_PHYSICIAN_KINDS + ("confounded", "withhold-abx")


def gumbel_diabetes_counterpart(env: FiniteEnv) -> np.ndarray:
    """[S] index map of the exact Gumbel env: a live diabetic state -> the live non-diabetic state with the same
    observation (full index - 720); non-diabetic and absorbing states map to themselves."""
    space = gumbel_state_space()
    live_full = space["live_full"]
    n_live = int(env.meta["n_live"])
    assert env.kind == "gumbel" and n_live == len(live_full) and env.S == n_live + 3, env.name
    twin = np.arange(env.S)
    for k, f in enumerate(live_full):
        diab, obs = divmod(int(f), GumbelSim.N_OBS)
        if diab == 1:
            assert obs in space["pos"], ("the non-diabetic twin of a live state must be live", int(f))
            twin[k] = space["pos"][obs]
    assert np.all(twin[twin] == twin), "twins of diabetic states must be non-diabetic"
    return twin


def gumbel_physician_v3(env: FiniteEnv, eps: float, kind: str = "optimal") -> np.ndarray:
    """eps-softened Gumbel physician (Section II.8b): 1 - eps on its action, eps/(A-1) on every other action.

    'optimal' / 'withhold-vaso': exactly gumbel_physician (II.3b). 'confounded': the optimum of the non-diabetic
    dynamics applied to every patient (diabetic states take the action of their non-diabetic twin). 'withhold-abx':
    the exact optimum among policies that never give antibiotics on live states."""
    if kind in GUMBEL_PHYSICIAN_KINDS:
        return gumbel_physician(env, eps, kind)
    assert 0.0 <= eps < 1.0, eps
    A = env.A
    if kind == "confounded":
        twin = gumbel_diabetes_counterpart(env)
        n_live = int(env.meta["n_live"])
        is_diab = np.zeros(env.S, dtype=bool)
        is_diab[:n_live] = gumbel_state_space()["live_full"] >= GumbelSim.N_OBS
        non_diab = env.live & ~is_diab
        assert env.P[non_diab][:, :, is_diab].max() == 0.0 and env.P[is_diab][:, :, non_diab].max() == 0.0, \
            "the diabetes groups must not mix during a step"
        pi_opt, _, _ = x_optimal(env, "given")
        act = pi_opt.argmax(axis=1)[twin]
    elif kind == "withhold-abx":
        no_abx = np.ones((env.S, A), dtype=bool)
        no_abx[env.live] = np.array([gumbel_action_bits(a)[0] == 0 for a in range(A)])[None, :]
        pi_opt, _, _ = x_optimal(env.variant(f"{env.name}|no-antibiotics", mask=no_abx), "given", constrained=True)
        act = pi_opt.argmax(axis=1)
        assert all(gumbel_action_bits(a)[0] == 0 for a in act[env.live]), "withhold-abx physician gives antibiotics"
    else:
        raise ValueError(f"unknown Gumbel physician kind {kind!r} (expected one of {GUMBEL_PHYSICIAN_KINDS_V3})")
    pi = np.full((env.S, A), eps / (A - 1))
    pi[np.arange(env.S), act] = 1.0 - eps
    return validate_policy(pi, f"Gumbel physician ({kind}) eps={eps}", shape=(env.S, A))


_X_V3_GUMBEL_ENVS: Dict[Tuple[str, float], FiniteEnv] = {}
_X_V3_GUMBEL_PHYS: Dict[tuple, np.ndarray] = {}
_X_V3_GUMBEL_OPT_ACTION: Dict[Tuple[str, float], np.ndarray] = {}


def _x_v3_gumbel_env(variant: str, hazard: float) -> FiniteEnv:
    """The exact Gumbel env of II.3b when it was built there; otherwise enumerated here by the same function."""
    env = X_GUMBEL_ENVS.get(variant)
    if env is not None and float(env.meta["timeout_hazard"]) == float(hazard):
        return env
    key = (str(variant), float(hazard))
    if key not in _X_V3_GUMBEL_ENVS:
        _X_V3_GUMBEL_ENVS[key] = gumbel_true_env(variant, hazard)
    return _X_V3_GUMBEL_ENVS[key]


def _x_v3_gumbel_physician(variant: str, hazard: float, eps: float, kind: str) -> np.ndarray:
    key = (str(variant), float(hazard), float(eps), str(kind))
    if key not in _X_V3_GUMBEL_PHYS:
        _X_V3_GUMBEL_PHYS[key] = gumbel_physician_v3(_x_v3_gumbel_env(variant, hazard), eps, kind)
    return _X_V3_GUMBEL_PHYS[key]


def _x_v3_gumbel_opt_action(variant: str, hazard: float) -> np.ndarray:
    """argmax of the exact optimum of the Gumbel system (cached per variant and hazard)."""
    key = (str(variant), float(hazard))
    if key not in _X_V3_GUMBEL_OPT_ACTION:
        pi_opt, _, _ = x_optimal(_x_v3_gumbel_env(variant, hazard), "given")
        _X_V3_GUMBEL_OPT_ACTION[key] = pi_opt.argmax(axis=1)
    return _X_V3_GUMBEL_OPT_ACTION[key]


def gumbel_physician_report(env: FiniteEnv, eps: float, kinds: Sequence[str] = GUMBEL_PHYSICIAN_KINDS_V3) -> pd.DataFrame:
    """One row per physician kind: exact values on the true system and disagreement with the true optimum."""
    _, _, opt_info = x_optimal(env, "given")
    opt = _x_v3_gumbel_opt_action(env.meta["variant"], env.meta["timeout_hazard"])
    n_live = int(env.meta["n_live"])
    is_diab = np.zeros(env.S, dtype=bool)
    is_diab[:n_live] = gumbel_state_space()["live_full"] >= GumbelSim.N_OBS
    L = env.live
    bits = np.array([gumbel_action_bits(a) for a in range(env.A)])      # (abx, vent, vaso) per action
    rows = []
    for kind in kinds:
        pi = gumbel_physician_v3(env, eps, kind)
        act = pi.argmax(axis=1)
        ev = x_evaluate(env, pi, "given", vectors=True)
        det = x_evaluate(env, one_hot_policy(act, env.A), "given")
        assert ev["status"] == "ok" and det["status"] == "ok", (kind, ev["status"], det["status"])
        occ = ev["occupancy_vector"]
        differ = (act != opt) & L
        rows.append({"kind": kind, "eps": float(eps), "value_true": float(ev["return"]),
                     "value_deterministic_core": float(det["return"]), "optimum_value": float(opt_info["dp_value"]),
                     "disagree_live_fraction": float(differ[L].mean()),
                     "disagree_occupancy_weighted": float((occ * differ).sum() / occ.sum()),
                     "disagree_diabetic_live_fraction": float(differ[L & is_diab].mean()),
                     "disagree_nondiabetic_live_fraction": float(differ[L & ~is_diab].mean()),
                     "antibiotic_share_live": float(bits[act[L], 0].mean()),
                     "ventilation_share_live": float(bits[act[L], 1].mean()),
                     "vasopressor_share_live": float(bits[act[L], 2].mean())})
    return pd.DataFrame(rows)


X_V3_PHYSICIAN_REPORT = pd.DataFrame()
if EXPERIMENTS["xenv_v3_physicians"] or EXPERIMENTS["xenv_v3_aggregated"]:
    _t0 = time.perf_counter()
    _env = _x_v3_gumbel_env(XCFG.gumbel_variant, XCFG.gumbel_timeout_hazard)
    _eps = sorted({float(c[3]) for c in (*XV3CFG.physician_cells, *XV3CFG.aggregated_cells)})
    X_V3_PHYSICIAN_REPORT = pd.concat([gumbel_physician_report(_env, e) for e in _eps], ignore_index=True)
    X_V3_PHYSICIAN_REPORT.to_csv(XENV_TABLE_DIR / "v3_physician_report.csv", index=False)
    display(X_V3_PHYSICIAN_REPORT.round(4))
    print(f"II.8b physician report in {time.perf_counter() - _t0:.1f} s.")
    del _env, _eps
