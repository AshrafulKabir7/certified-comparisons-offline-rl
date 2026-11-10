# %% [markdown]
# ## II.6 Driver: every method × every target × every dataset draw
#
# The unit of Part II is one **dataset draw** of one environment cell. For each draw the driver
#
# 1. builds the `CellInstance` (deterministic in (cell, draw)) and its identity: instance hashes (learner MDP, user
#    data, train / dev split, behaviour policy, every target environment), the Part II code hash, the `XCFG` fields
#    that can change a draw's result, the categorical-diffusion schedule, the cell parameters and the run mode;
# 2. loads the identity-matched result from `XCACHE`, or, if `ALLOW_TRAINING`, computes it. On a cache miss the full
#    method suite (`run_xmethods`) is run on the **training episodes only**; learners receive a `LearnerView` (sizes,
#    mask, absorbing states, reward range) and never see P or d0;
# 3. adds `Behaviour (true) [native]`, the policy that generated the user data, and the uniform admissible
#    projection `x_project(π, M, live)` of **every** native record, with the same method, hyper-parameters and
#    default flag. Every method then has a `[projected]` counterpart in every draw, as paired tests need;
# 4. evaluates every policy exactly on every target of the cell. A `lift` maps learner-space policies to the true
#    state space where needed. Each policy is also evaluated on `fqe_dev`, the tabular FQE model of the held-out
#    dev episodes (`x_fqe_env`, semantics "given"). `fqe_dev` is used only for offline selection (`target_role` =
#    "selection") and never for reporting;
# 5. adds three reference rows per target, computed on the target itself: `REF optimum`, `REF optimum (admissible)`
#    and `REF behaviour` (the lifted behaviour). They give the normalised score
#    z = (J − J_REF behaviour) / (J_REF optimum − J_REF behaviour);
# 6. checks the invariants below before caching, and again after loading. The table returned is always the one read
#    back from the cache, so an interrupted, resumed run and an uninterrupted run give identical tables.
#
# **Exact reference optimum.** `REF optimum` comes from value iteration (`x_optimal`). The model may be estimated:
# a constructed benchmark, the tabular FQE model, or ICU-Sepsis itself. In such a model a reward-free closed loop
# can make the greedy VI policy improper, because under γ = 1 value iteration also counts behaviour that is never
# absorbed (value 0). In that case the optimum over **proper** policies is solved exactly. The solver is the occupancy
# LP `x_cmdp_lp`, with full-support initial weights w = ½·d0 + ½·uniform(live) and κ = ∞ (or κ = 0 with
# cost 1[a ∉ M] for the admissible optimum). Every LP solution is proper: a closed recurrent class C would need
# w(C) = 0. A proper policy that attains the proper-policy optimum at every state exists (strict policy iteration
# from a proper policy stays proper), so the LP policy attains it from every state. Its exact value at d0 is
# therefore the proper optimum. The driver checks wᵀv = LP objective to 1e-7 and records which solver ran in the
# `solver` column.
#
# **Invariants asserted per draw.**
# * Every status is ok or improper. An improper (never-absorbing) policy is a reported outcome, never hidden.
# * Each target has exactly one row per reference policy, and each policy is evaluated once on every target.
# * Masked and projected policies have zero inadmissible mass on every target. This also checks that the lifted
#   masks agree with the learner mask.
# * No policy beats `REF optimum` on any target, and no zero-violation policy beats `REF optimum (admissible)`
#   (tolerance 1e-6).
# * Projection identity (Proposition P1, driver-level): on mean-imputed targets (`bench_mean`, `true_rows_mean`)
#   native and projected policies have identical return and length (1e-8). For ICU-Sepsis this is asserted only when
#   the released mean-rule identity holds (Section 3).
# * The cache round-trip is exact.
#
# **Identity design.** Some settings decide which cells and draws exist, or configure statistics and theory, but do
# not change what a draw computes: the draw counts, the cell lists, the Monte-Carlo check sizes, the frontier budgets
# and the bootstrap settings. They are left out of the identity, so a lite run can be extended to the full profile
# without recomputation. A cell's own parameters still enter through the spec and the instance hashes. The code hash
# covers source text only, so the values of module constants that hashed functions read (tie and leak tolerances,
# the semantics map) enter through `XDRIVER_CONSTANTS["numeric"]`. Section II.5
# code is not hashed, because no driver result depends on it. The per-draw checks are re-run on cached tables, so
# they are not hashed either.
#
# **Exact deduplication.** Byte-identical policies (for example a projection of a policy that is already admissible)
# are evaluated once per target.
#
# | `XRESULTS` column | meaning |
# |---|---|
# | cell, family, draw, deploy_target | experimental unit; the cell's pre-declared deployment target |
# | policy, method, variant, hp, is_default, method_family | `x_label`; native / masked / projected / reference; canonical-JSON hyper-parameters |
# | is_reference, solver | reference rows (`REF …`) and the solver that produced them (vi / lp_full_support / evaluation) |
# | converged | learner convergence flag from `run_xmethods` (empty if the method has none) |
# | target, target_role, is_deploy_target, semantics, target_env, lifted | evaluation target; "target" or "selection" (`fqe_dev`) |
# | status, error | ok / improper (with the evaluator's message) |
# | return … absorb_other_terminal | exact metrics; `absorb_goal` is the probability of absorption in the goal state |

# %%
# ============================================================
# II.6 Cross-environment driver
# ============================================================
EXECLOG.mark("II.6 Cross-environment driver")
from collections import Counter

XBEHAVIOUR_METHOD = "Behaviour (true)"
XMETHOD_FAMILIES = ("behaviour", "value", "pessimistic", "constrained", "guided-prior", "diffusion")
XFQE_TARGET = "fqe_dev"
XREF_POLICIES = ("REF optimum", "REF optimum (admissible)", "REF behaviour")
XMEAN_IMPUTED_TARGETS = ("bench_mean", "true_rows_mean")   # inadmissible rows = uniform mean of admissible rows
XPROJ_TOL, XDOM_TOL, XMASS_TOL, XLP_TOL = 1e-8, 1e-6, 1e-12, 1e-7
_XMETRIC_SOURCE = {"return": "return", "avg_episode_length": "avg_episode_length",
                   "inadmissible_rate": "inadmissible_rate", "expected_violations": "expected_violations",
                   "prob_any_violation": "prob_any_violation",
                   "states_with_inadmissible_mass": "states_with_inadmissible_mass",
                   "max_state_inadmissible_mass": "max_state_inadmissible_mass",
                   "mean_state_inadmissible_mass": "mean_state_inadmissible_mass",
                   "absorb_death": "absorb_death", "absorb_goal": "absorb_survival",
                   "absorb_other_terminal": "absorb_other_terminal"}
XRESULT_COLUMNS = ["cell", "family", "draw", "deploy_target", "policy", "method", "variant", "hp", "is_default",
                   "method_family", "is_reference", "solver", "converged", "target", "target_role",
                   "is_deploy_target", "semantics", "target_env", "lifted", "status", *_XMETRIC_SOURCE, "error"]
# _XCFG_NOT_IN_IDENTITY and x_result_xcfg (the XCFG fields that can change a draw) are defined in Section II.2;
# X_NUMERIC_CONSTANTS (module constants read by hashed functions, by value) in Section II.4.
XDRIVER_CONSTANTS = {"columns": XRESULT_COLUMNS, "metrics": _XMETRIC_SOURCE, "fqe_target": XFQE_TARGET,
                     "behaviour_method": XBEHAVIOUR_METHOD, "reference_policies": list(XREF_POLICIES),
                     "numeric": X_NUMERIC_CONSTANTS}


def x_cache_key(cell: str, draw: int) -> str:
    return cell.replace("|", "/") + f"/draw{int(draw)}"


def x_draw_identity(spec: dict, inst: CellInstance) -> dict:
    # behaviour and dev split are hashed explicitly: CellInstance.identity() covers data, train and targets only.
    return {"instance": inst.identity(), "behavior": array_hash(np.asarray(inst.behavior, dtype=np.float64)),
            "dev": array_dict_hash(inst.dev), "code": XENV_CODE_HASH, "xcfg": x_result_xcfg(),
            "diffusion": [CFG.diffusion_T, CFG.diffusion_beta_start, CFG.diffusion_beta_end], "run_mode": RUN_MODE,
            "spec": {k: v for k, v in spec.items() if k != "draws"}, "driver": XDRIVER_CONSTANTS}


def _x_pi_key(pi: np.ndarray) -> bytes:
    a = np.ascontiguousarray(pi, dtype=np.float64)
    return hashlib.blake2b(a.tobytes(), digest_size=16).digest() + str(a.shape).encode()


def _x_compact(obj, limit: int = 2000):
    """JSON-native copy of a meta object; sequences longer than `limit` JSON characters become a hashed summary."""
    if isinstance(obj, dict):
        return {str(k): _x_compact(v, limit) for k, v in obj.items()}
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return obj.item()
    if isinstance(obj, (list, tuple, np.ndarray)):
        txt = canonical_json(obj)
        if len(txt) <= limit:
            return json.loads(txt)
        return f"<{type(obj).__name__} of {len(obj)} items, sha256 {sha256_bytes(txt.encode())[:12]}>"
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return json.loads(canonical_json(obj))


def _x_flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_x_flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = canonical_json(v)
        else:
            out[key] = v
    return out


# ---------- records: validated learner outputs + data-generating behaviour + admissible projections ----------
def _x_record(r: dict, label: str, pi: np.ndarray) -> dict:
    c = r["info"].get("converged")
    desc = {"policy": label, "method": r["method"], "variant": r["variant"], "hp": canonical_json(r["hp"]),
            "is_default": bool(r["is_default"]), "method_family": r["family"], "is_reference": False,
            "solver": np.nan, "converged": np.nan if c is None else bool(c)}
    return {**r, "label": label, "pi": pi, "desc": desc}


def x_collect_records(raw: Sequence[dict], behavior: np.ndarray, env: FiniteEnv) -> List[dict]:
    """Validate run_xmethods records, add the behaviour policy and the projection of every native record."""
    need = {"method", "variant", "hp", "is_default", "family", "pi", "info"}
    base = []
    for r in raw:
        missing = need.difference(r)
        assert not missing, f"run_xmethods record {r.get('method')!r} lacks {sorted(missing)}"
        assert r["variant"] in ("native", "masked"), \
            f"{r['method']}: variant {r['variant']!r} (run_xmethods returns native / masked; the driver adds projected)"
        assert r["method"] != XBEHAVIOUR_METHOD, f"'{XBEHAVIOUR_METHOD}' is reserved for the driver"
        assert r["family"] in XMETHOD_FAMILIES, f"{r['method']}: family {r['family']!r} not in {XMETHOD_FAMILIES}"
        base.append({"method": str(r["method"]), "variant": r["variant"], "hp": dict(r["hp"] or {}),
                     "is_default": bool(r["is_default"]), "family": str(r["family"]), "pi": r["pi"],
                     "info": dict(r["info"] or {})})
    base.append({"method": XBEHAVIOUR_METHOD, "variant": "native", "hp": {}, "is_default": True, "family": "behaviour",
                 "pi": behavior, "info": {"source": "behaviour policy that generated the user data"}})
    out = []
    for r in base:
        label = x_label(r["method"], r["variant"], r["hp"], r["is_default"])
        pi = validate_policy(r["pi"], label, shape=(env.S, env.A))
        mass = float(x_inadmissible_mass(pi, env.mask, env.live).max())
        if r["variant"] == "masked":
            assert mass == 0.0, f"{label}: masked variant puts mass {mass:.3g} outside the learner mask"
        out.append(_x_record(r, label, pi))
        if r["variant"] == "native":
            info = {**r["info"], "projected_from": label, "native_max_inadmissible_mass": mass}
            out.append(_x_record({**r, "variant": "projected", "info": info},
                                 x_label(r["method"], "projected", r["hp"], r["is_default"]),
                                 x_project(pi, env.mask, env.live)))
    dup = sorted(k for k, n in Counter(r["label"] for r in out).items() if n > 1)
    assert not dup, f"duplicate policy labels: {dup[:5]}"
    n_def = Counter((r["method"], r["variant"]) for r in out if r["is_default"])
    no_def = sorted({(r["method"], r["variant"]) for r in out} - set(n_def))
    assert not no_def and max(n_def.values()) == 1, f"need exactly one default record per method / variant: {no_def[:5]}"
    defaults = {r["label"] for r in out if r["is_default"]}
    missing = [lab for lab in (XCOMPARATOR, *XTREATMENTS) if lab not in defaults]
    assert not missing, f"preregistered Part II policies missing from the default set: {missing}"
    return out


# ---------- exact evaluation, reference rows and row assembly ----------
def _x_eval_metrics(env: FiniteEnv, pi: np.ndarray, semantics: str) -> dict:
    out = x_evaluate(env, pi, semantics)
    if out["status"] == "ok":
        return {"status": "ok", **{src: float(out[src]) for src in _XMETRIC_SOURCE.values()}}
    assert out["status"] == "improper", out["status"]
    return {"status": "improper", "error": str(out.get("error", ""))[:200]}


def x_result_row(base: dict, desc: dict, tgt: dict, out: dict) -> dict:
    """One XRESULTS row: unit columns + policy description + target description + exact metrics."""
    status = out["status"]
    assert status in ("ok", "improper"), status
    ok = status == "ok"
    row = {**base, **desc, **tgt, "status": status,
           **{col: (float(out[src]) if ok else np.nan) for col, src in _XMETRIC_SOURCE.items()},
           "error": np.nan if ok else str(out.get("error", ""))[:200]}
    assert len(row) == len(XRESULT_COLUMNS) and set(row) == set(XRESULT_COLUMNS), \
        sorted(set(row) ^ set(XRESULT_COLUMNS))
    if ok:
        assert all(np.isfinite(row[c]) for c in _XMETRIC_SOURCE), f"non-finite metric for {desc['policy']}"
    return row


def _x_ref_desc(name: str, solver: str) -> dict:
    return {"policy": name, "method": name, "variant": "reference", "hp": "{}", "is_default": False,
            "method_family": "reference", "is_reference": True, "solver": solver, "converged": np.nan}


def x_reference_lp(env: FiniteEnv, semantics: str, constrained: bool) -> Tuple[Optional[np.ndarray], str]:
    """Optimum over proper policies: occupancy LP with full-support weights (see markdown); verified w'v = objective."""
    live = env.live
    P = x_semantic_P(env, semantics)
    r = np.einsum("san,n->sa", P, env.entry_reward, optimize=True)
    r[~live] = 0.0
    cost = (~env.mask & live[:, None]).astype(np.float64)
    w = 0.5 * env.d0 + 0.5 * live / live.sum()
    try:
        pi, info = x_cmdp_lp(P, r, w, live, cost, 0.0 if constrained else np.inf)
    except RuntimeError as exc:   # infeasible: some live state has no proper policy (every policy is improper)
        return None, f"lp_infeasible: {str(exc)[:120]}"
    if constrained:   # drop solver-tolerance flow on inadmissible pairs (the identity check below re-verifies)
        pi[live] = np.where(env.mask[live], pi[live], 0.0)
        pi[live] /= pi[live].sum(axis=1, keepdims=True)
    out = x_evaluate(env, pi, semantics, vectors=True)
    if out["status"] != "ok":
        raise EvaluationInvariantError(f"{env.name}: full-support LP policy is improper ({out.get('error')})")
    jw = float(w @ out["value_vector"])
    if abs(jw - info["objective"]) > XLP_TOL * max(1.0, abs(info["objective"])):
        raise EvaluationInvariantError(f"{env.name}: LP objective {info['objective']:.12g} != exact w'v {jw:.12g}")
    return pi, "lp_full_support"


_XREF_MEMO: Dict[Tuple[str, str, bool], Tuple[dict, str]] = {}


def x_reference_optimum(env: FiniteEnv, semantics: str, constrained: bool) -> Tuple[dict, str]:
    """Exact (admissible-)optimum metrics of a target; memoised by the environment identity (bounded)."""
    key = (obj_hash(env.identity()), semantics, bool(constrained))
    if key not in _XREF_MEMO:
        try:
            pi, _, _ = x_optimal(env, semantics, constrained=constrained)
            solver = "vi"
        except RuntimeError as exc:   # improper greedy VI policy / DP-evaluation gap / VI non-convergence
            pi, solver = x_reference_lp(env, semantics, constrained)
            solver = f"{solver} (vi: {type(exc).__name__})"
        out = _x_eval_metrics(env, pi, semantics) if pi is not None else {"status": "improper", "error": solver}
        if len(_XREF_MEMO) >= 256:
            _XREF_MEMO.pop(next(iter(_XREF_MEMO)))
        _XREF_MEMO[key] = (out, solver)
    return _XREF_MEMO[key]


def x_reference_rows(base: dict, tgt: dict, env: FiniteEnv, semantics: str,
                     behaviour_out: dict) -> Tuple[List[dict], List[str]]:
    rows, solvers = [], []
    for name, constrained in (("REF optimum", False), ("REF optimum (admissible)", True)):
        out, solver = x_reference_optimum(env, semantics, constrained)
        rows.append(x_result_row(base, _x_ref_desc(name, solver), tgt, out))
        solvers.append(solver)
    rows.append(x_result_row(base, _x_ref_desc("REF behaviour", "evaluation"), tgt, behaviour_out))
    return rows, solvers


def x_evaluate_targets(inst: CellInstance, records: List[dict], fqe: FiniteEnv) -> Tuple[List[dict], dict]:
    """Exact evaluation of every record on every target (lifted where needed) and on the FQE selection model."""
    learn = inst.learn_env
    base = {"cell": inst.cell, "family": inst.family, "draw": int(inst.draw), "deploy_target": inst.deploy_target}
    keys = [_x_pi_key(r["pi"]) for r in records]
    beh = [k for r, k in zip(records, keys) if r["method"] == XBEHAVIOUR_METHOD and r["variant"] == "native"]
    assert len(beh) == 1
    plan = [(lab, env, sem, lift, "target") for lab, env, sem, lift in inst.targets]
    plan.append((XFQE_TARGET, fqe, "given", None, "selection"))
    labels = [p[0] for p in plan]
    assert len(set(labels)) == len(labels), f"duplicate target labels {labels}"
    assert inst.deploy_target in labels[:-1], f"deploy target {inst.deploy_target!r} not among {labels[:-1]}"
    rows, stats_ = [], Counter()
    for lab, env, sem, lift, role in plan:
        assert sem in X_SEMANTICS and env.A == learn.A, (lab, sem, env.A, learn.A)
        if lift is None:
            assert env.S == learn.S, f"target {lab}: state space differs from the learner's but no lift is given"
        tgt = {"target": lab, "target_role": role, "is_deploy_target": lab == inst.deploy_target, "semantics": sem,
               "target_env": env.name, "lifted": lift is not None}
        memo: Dict[bytes, dict] = {}
        for r, k in zip(records, keys):
            if k not in memo:
                memo[k] = _x_eval_metrics(env, r["pi"] if lift is None else lift(r["pi"]), sem)
                stats_["evaluations_unique"] += 1
            stats_["evaluations_total"] += 1
            rows.append(x_result_row(base, r["desc"], tgt, memo[k]))
        if role == "target":
            ref_rows, solvers = x_reference_rows(base, tgt, env, sem, memo[beh[0]])
            rows += ref_rows
            for s in solvers:
                stats_[f"ref_solver.{s}"] += 1
        del memo
    return rows, dict(stats_)


# ---------- invariant checks (pure pandas; re-run on cached tables) ----------
def x_projection_identity(df: pd.DataFrame,
                          targets: Sequence[str] = XMEAN_IMPUTED_TARGETS) -> Dict[str, Dict[str, float]]:
    """Native vs projected return / length on mean-imputed targets (identical by Proposition P1)."""
    out = {}
    keys = ["method", "hp", "is_default"]
    for t in targets:
        sub = df[(df["target"] == t) & ~df["is_reference"].astype(bool)]
        if sub.empty:
            continue
        nat = sub[sub["variant"] == "native"].set_index(keys).sort_index()
        prj = sub[sub["variant"] == "projected"].set_index(keys).sort_index()
        assert nat.index.is_unique and nat.index.equals(prj.index), f"{t}: native and projected records do not pair up"
        s_n, s_p = nat["status"].to_numpy(), prj["status"].to_numpy()
        both = (s_n == "ok") & (s_p == "ok")
        jn, jp = nat["return"].to_numpy(dtype=float), prj["return"].to_numpy(dtype=float)
        ln, lp = nat["avg_episode_length"].to_numpy(dtype=float), prj["avg_episode_length"].to_numpy(dtype=float)
        dj = np.abs(jn - jp)[both]
        dl = (np.abs(ln - lp) / np.maximum(1.0, np.abs(ln)))[both]
        out[t] = {"pairs": int(len(nat)), "status_mismatches": int((s_n != s_p).sum()),
                  "max_abs_diff_return": float(dj.max(initial=0.0)), "max_rel_diff_length": float(dl.max(initial=0.0))}
    return out


def x_check_draw_rows(df: pd.DataFrame, strict_projection: bool) -> Dict[str, Any]:
    """Assert the invariants of one draw's rows (see markdown) and return a small report."""
    bad = sorted(set(df["status"].astype(str)) - {"ok", "improper"})
    assert not bad, f"unexpected status values {bad}"
    ok = df["status"] == "ok"
    ref = df["is_reference"].astype(bool)
    is_tgt = df["target_role"] == "target"
    assert not (ref & ~is_tgt).any(), "reference rows on a selection model"
    pol_sets = {t: frozenset(s) for t, s in df[~ref].groupby("target", sort=False)["policy"]}
    assert len(set(pol_sets.values())) == 1, "the policy set differs between targets"
    per = df[~ref].groupby("target", sort=False)["policy"].agg(lambda s: s.duplicated().any())
    assert not per.any(), f"a policy is evaluated twice on targets {list(per[per].index)}"
    adm = df["variant"].isin(["masked", "projected"]) & ok
    worst_mass = float(df.loc[adm, ["max_state_inadmissible_mass", "expected_violations"]].to_numpy().max(initial=0.0))
    assert worst_mass <= XMASS_TOL, f"masked / projected policy with inadmissible mass {worst_mass:.3g}"
    slack, slack_adm = -np.inf, -np.inf
    for t, sub in df[is_tgt].groupby("target", sort=False):
        got = Counter(sub.loc[sub["is_reference"].astype(bool), "policy"])
        assert dict(got) == {p: 1 for p in XREF_POLICIES}, f"target {t}: reference rows {dict(got)}"
        opt = sub[sub["policy"] == "REF optimum"].iloc[0]
        aopt = sub[sub["policy"] == "REF optimum (admissible)"].iloc[0]
        okt = sub[sub["status"] == "ok"]
        if opt["status"] == "ok":
            s = float((okt["return"] - opt["return"]).max())
            slack = max(slack, s)
            assert s <= XDOM_TOL * max(1.0, abs(opt["return"])), f"target {t}: a policy beats REF optimum by {s:.3g}"
        if aopt["status"] == "ok":
            assert aopt["expected_violations"] <= XMASS_TOL, f"target {t}: admissible optimum violates the mask"
            zero = okt[okt["expected_violations"] <= XMASS_TOL]
            s = float((zero["return"] - aopt["return"]).max())
            slack_adm = max(slack_adm, s)
            assert s <= XDOM_TOL * max(1.0, abs(aopt["return"])), \
                f"target {t}: a zero-violation policy beats REF optimum (admissible) by {s:.3g}"
        if opt["status"] == "ok" and aopt["status"] == "ok":
            assert aopt["return"] <= opt["return"] + XDOM_TOL * max(1.0, abs(opt["return"])), f"target {t}"
    pid = x_projection_identity(df)
    worst_p = max([max(v["max_abs_diff_return"], v["max_rel_diff_length"]) for v in pid.values()], default=0.0)
    mism = sum(v["status_mismatches"] for v in pid.values())
    if strict_projection:
        assert worst_p <= XPROJ_TOL and mism == 0, f"projection identity violated: {pid}"
    improper = df.loc[~ok].groupby("target")["policy"].size()
    return {"rows": int(len(df)), "improper_rows": int((~ok).sum()),
            "improper_by_target": {str(k): int(v) for k, v in improper.items()},
            "max_slack_vs_optimum": float(slack), "max_slack_vs_admissible_optimum": float(slack_adm),
            "masked_projected_max_inadmissible_mass": worst_mass, "projection_identity": pid,
            "projection_identity_worst": float(worst_p), "projection_identity_strict": bool(strict_projection)}


# ---------- one (cell, draw) ----------
def run_cell_draw(spec: dict, draw: int) -> Tuple[pd.DataFrame, dict]:
    """All Part II results of one dataset draw of one cell (identity-cached in XCACHE)."""
    cell, family = spec["cell"], spec["family"]
    assert family in ("icu", "gumbel", "garnet"), family
    t0 = time.perf_counter()
    inst = build_cell_instance(spec, draw)
    assert (inst.cell, inst.family, int(inst.draw)) == (cell, family, int(draw)), (inst.cell, inst.family, inst.draw)
    if family == "icu" and int(draw) == 0:
        assert all(np.array_equal(inst.data[k], OFFLINE[k]) for k in
                   ("state", "action", "reward", "next_state", "episode_id", "timestep")), \
            "ICU draw 0 must be the Part I dataset (OFFLINE)"
    identity = x_draw_identity(spec, inst)
    key, ident_hash = x_cache_key(cell, draw), obj_hash(identity)[:20]
    strict = family != "icu" or bool(MEAN_RULE_IDENTITY_HOLDS)
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
        "cell": cell, "family": family, "draw": int(draw), "deploy_target": inst.deploy_target,
        "identity_hash": ident_hash, "code_hash": XENV_CODE_HASH[:16], "cache": "computed",
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


def x_check_results(df: pd.DataFrame) -> Dict[str, Any]:
    """Cross-draw invariants: one status vocabulary and an identical policy set across the draws of a family."""
    if df.empty:
        return {"rows": 0}
    bad = sorted(set(df["status"].astype(str)) - {"ok", "improper"})
    assert not bad, f"unexpected status values {bad}"
    ref = df["is_reference"].astype(bool)
    sets = df[~ref].groupby(["family", "cell", "draw"])["policy"].agg(lambda s: frozenset(s))
    for fam, s in sets.groupby(level="family"):
        assert s.nunique() == 1, f"{fam}: the evaluated policy set differs across cells / draws ({s.nunique()} sets)"
    tr = (df[ref].set_index(["cell", "draw", "target", "policy"])["return"].unstack("policy")
          .reindex(columns=list(XREF_POLICIES)))
    denom = (tr["REF optimum"] - tr["REF behaviour"]).astype(float)
    solver = df[ref & (df["policy"] != "REF behaviour")]["solver"].astype(str).value_counts()
    imp = df[df["status"] != "ok"].groupby(["family", "target"]).size()
    return {"rows": int(len(df)), "cell_draws": int(df[["cell", "draw"]].drop_duplicates().shape[0]),
            "policies_per_family": {str(f): len(next(iter(s))) for f, s in sets.groupby(level="family")},
            "improper_rows": int(imp.sum()),
            "improper_by_family_target": {f"{a}|{b}": int(v) for (a, b), v in imp.items()},
            "reference_solver_counts": {str(k): int(v) for k, v in solver.items()},
            "normalisation_denominator_below_1e-9": int((denom.abs() < 1e-9).sum()),
            "normalisation_denominator_missing": int(denom.isna().sum())}


def x_eta_seconds(remaining: Sequence[Tuple[str, int]], times: Dict[str, List[float]]) -> Optional[float]:
    """Remaining time from the mean computed-draw time per family (pooled mean for unseen families)."""
    pooled = [t for v in times.values() for t in v]
    if not pooled:
        return None
    g = float(np.mean(pooled))
    return float(sum(n * (float(np.mean(times[f])) if times.get(f) else g) for f, n in remaining))


def _x_fmt_s(sec: Optional[float]) -> str:
    if sec is None or not np.isfinite(sec):
        return "n/a"
    return f"{sec:.1f} s" if sec < 120 else (f"{sec / 60:.1f} min" if sec < 7200 else f"{sec / 3600:.2f} h")


# Objects whose source changes what a draw computes or stores (the checks are re-run on cached tables instead).
_XDRIVER_HASHED = [x_result_xcfg, x_cache_key, x_draw_identity, _x_pi_key, _x_compact, _x_record, x_collect_records,
                   _x_eval_metrics, x_result_row, _x_ref_desc, x_reference_lp, x_reference_optimum, x_reference_rows,
                   x_evaluate_targets, run_cell_draw]
XENV_CODE_HASH = code_hash(*XENV_CORE_CODE, *XENV_ENV_CODE, *XENV_METHOD_CODE, *_XDRIVER_HASHED)
print("Part II code hash:", XENV_CODE_HASH[:16], "| cache:", XENV_CACHE_DIR)

# %% [markdown]
# ### II.6a Run all cells and draws
#
# The loop below goes over every enabled cell (`X_CELLS`, filtered by the family switches) and each of its draws. It
# shows a progress bar per cell (`SHOW_PROGRESS`; otherwise a line every ~20 % of the draws). After each cell it
# prints the elapsed time and an ETA for the remaining cells. The ETA uses the mean time of the **computed** draws
# of each family, so cached draws count as zero time and the ETA is an upper estimate when later draws are cached.
# The outputs are `XRESULTS` (`xenv/xenv_results_all.csv.gz`, one row per draw × policy × target) and `XDRAW_META`
# (`xenv/xenv_draw_meta.csv`: construction and validation reports, record counts, convergence, reference solvers,
# invariant-check slacks and timings per draw). The cross-draw check asserts that every draw of a family evaluates
# the same policy set, which the paired statistics in II.7 require.

# %%
# ============================================================
# II.6a Driver: all cells and draws
# ============================================================
EXECLOG.mark("II.6 Driver: all cells and draws")
_xcells = [s for s in X_CELLS if EXPERIMENTS[f"xenv_{s['family']}"]]
if len(_xcells) != len(X_CELLS):
    print("Skipping cells whose family switch is off:", [s["cell"] for s in X_CELLS if s not in _xcells])
_xframes: List[pd.DataFrame] = []
_xmeta_rows: List[dict] = []
_xtimes: Dict[str, List[float]] = {}
_xcache_status: Counter = Counter()
_xt_all = time.perf_counter()
print(f"Part II driver: {len(_xcells)} cells, {sum(int(s['draws']) for s in _xcells)} draws")
for _ci, _spec in enumerate(_xcells, 1):
    _t_cell, _n_loaded, _nd = time.perf_counter(), 0, int(_spec["draws"])
    _bar = trange(_nd, desc=f"II.6 [{_ci}/{len(_xcells)}] {_spec['cell']}", leave=False, disable=not SHOW_PROGRESS)
    for _draw in _bar:
        _t0 = time.perf_counter()
        _df, _meta = run_cell_draw(_spec, _draw)
        _dt = time.perf_counter() - _t0
        _xframes.append(_df)
        _xmeta_rows.append(_x_flatten({**_meta, "session_wall_s": _dt}))
        _xcache_status[_meta["cache"]] += 1
        if _meta["cache"] == "loaded":
            _n_loaded += 1
        else:
            _xtimes.setdefault(_spec["family"], []).append(_dt)
        if SHOW_PROGRESS:
            _bar.set_postfix_str(f"{_dt:.1f}s {_meta['cache']}")
        elif (_draw + 1) % max(1, _nd // 5) == 0 or _draw + 1 == _nd:
            _left = [(_spec["family"], _nd - _draw - 1)]
            print(f"    {_spec['cell']} draw {_draw + 1}/{_nd} ({_meta['cache']}, {_dt:.1f} s); "
                  f"cell ETA {_x_fmt_s(x_eta_seconds(_left, _xtimes))}")
    _eta = x_eta_seconds([(s["family"], int(s["draws"])) for s in _xcells[_ci:]], _xtimes)
    print(f"  [{_ci}/{len(_xcells)}] {_spec['cell']}: {_nd} draws ({_n_loaded} cached) in "
          f"{_x_fmt_s(time.perf_counter() - _t_cell)} | total elapsed {_x_fmt_s(time.perf_counter() - _xt_all)} | "
          f"ETA remaining cells {_x_fmt_s(_eta) if _ci < len(_xcells) else '0 s'}")

XRESULTS = (pd.concat(_xframes, ignore_index=True) if _xframes else pd.DataFrame(columns=XRESULT_COLUMNS))
XDRAW_META = pd.DataFrame(_xmeta_rows)
del _xframes, _xmeta_rows
gc.collect()
assert list(XRESULTS.columns) == XRESULT_COLUMNS
XRESULTS.to_csv(XENV_TABLE_DIR / "xenv_results_all.csv.gz", index=False, compression="gzip")
XDRAW_META.to_csv(XENV_TABLE_DIR / "xenv_draw_meta.csv", index=False)
XDRIVER_SUMMARY = {"code_hash": XENV_CODE_HASH, "cells": len(_xcells), "draws": int(sum(_xcache_status.values())),
                   "draws_computed": int(_xcache_status["computed"]), "draws_loaded": int(_xcache_status["loaded"]),
                   "elapsed_s": time.perf_counter() - _xt_all, "xcache_written": len(XCACHE.written),
                   "xcache_loaded": len(XCACHE.loaded), **x_check_results(XRESULTS)}
if len(XDRAW_META):
    for _c in ("checks.max_slack_vs_optimum", "checks.max_slack_vs_admissible_optimum", "checks.projection_identity_worst"):
        XDRIVER_SUMMARY[f"max_{_c.split('.', 1)[1]}"] = float(XDRAW_META[_c].max()) if _c in XDRAW_META else None
    XDRIVER_SUMMARY["nonconverged_records"] = (int(XDRAW_META["records.nonconverged"].sum())
                                               if "records.nonconverged" in XDRAW_META else 0)
save_json(XDRIVER_SUMMARY, XENV_TABLE_DIR / "xenv_driver_summary.json")
print(json.dumps({k: v for k, v in XDRIVER_SUMMARY.items() if k != "improper_by_family_target"}, indent=1,
                 default=_json_default))
if len(XRESULTS):
    _dep = XRESULTS[XRESULTS["is_deploy_target"].astype(bool)
                    & XRESULTS["policy"].isin([*XREF_POLICIES, XCOMPARATOR])]
    display(_dep.groupby(["family", "target", "policy"])["return"].agg(["mean", "std", "count"]).round(4))
else:
    print("Part II driver: no cell enabled (all family switches off) - XRESULTS is empty.")

# %% [markdown]
# ### II.6b Exact tabular learners vs Part I SGD learners
#
# Part I trains tabular IQL, discrete BCQ and CQL by minibatch Adam on the full ICU-Sepsis dataset (Sections 9 and
# 15), with a fixed step budget and, for CQL and BCQ, soft target updates. Part II solves the same objectives
# exactly: IQL is the in-sample expectile fixed point, BCQ is value iteration over the behaviour-thresholded action
# set, and CQL is full-batch Adam on the exact empirical objective. Here the Part II implementations are run on the
# same full `OFFLINE` data with the released mask. They are called through `run_xmethods`, so exactly the policies that enter the Part II tables are
# checked. For every main seed we report, for the native and the masked exports: greedy-action agreement over live
# states, over states visited in the data, and weighted by the exact occupancy of the Part I policy (mean rule); and
# the exact return of both policies and their difference under both rules (`EVALUATOR`). `hp_match` records whether
# the pre-declared Part II defaults equal the Part I settings (γ = 1 in both). This comparison is exploratory
# (`PREREG_PART2`). Differences are expected where counts are small: SGD with a finite budget does not reach the
# fixed point there, and unvisited states keep their initial values.

# %%
# ============================================================
# II.6b Exact-tabular vs Part I SGD cross-check
# ============================================================
EXECLOG.mark("II.6 Exact-tabular cross-check")
_X_PART1_NAMES = {"iql": ("Tabular IQL", "Tabular IQL [masked greedy]"),
                  "bcq": (f"Tabular BCQ (tau={CFG.bcq_threshold})", f"Tabular BCQ (tau={CFG.bcq_threshold}) [masked greedy]"),
                  "cql": ("Tabular CQL", "Tabular CQL [masked greedy]")}
_X_PART2_METHOD = {"iql": "IQL", "bcq": "BCQ", "cql": "CQL"}


def x_action_agreement(pi_a: np.ndarray, pi_b: np.ndarray, states: np.ndarray,
                       weights: Optional[np.ndarray] = None) -> float:
    """Fraction (optionally weighted) of the selected states where the argmax actions agree (ties: lowest index)."""
    st = np.asarray(states, dtype=bool)
    if not st.any():
        return float("nan")
    agree = (np.asarray(pi_a).argmax(axis=1) == np.asarray(pi_b).argmax(axis=1))[st].astype(np.float64)
    if weights is None:
        return float(agree.mean())
    w = np.asarray(weights, dtype=np.float64)[st]
    return float((w * agree).sum() / w.sum()) if w.sum() > 0 else float("nan")


def _x_is_deterministic(pi: np.ndarray, live: np.ndarray) -> bool:
    return bool(np.all(np.asarray(pi)[live].max(axis=1) == 1.0))


def _x_part1_tabular_policies(algo: str, res: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """The exports registered by Section 15 (`run_tabular_suite`) for one Part I tabular result."""
    if algo == "cql":
        return {"native": greedy_policy(res["q"], masked=False), "masked": greedy_policy(res["q"], masked=True)}
    return {"native": res["deterministic"], "masked": res["masked_greedy"]}


def _x_part1_hp(algo: str) -> dict:
    if algo == "iql":
        return {"expectile": CFG.iql_expectile, "beta": CFG.iql_beta, "max_weight": CFG.iql_max_weight,
                "gamma": CFG.gamma, "steps": CFG.iql_steps, "lr": CFG.iql_lr, "batch": CFG.batch_size}
    if algo == "bcq":
        return {"tau": CFG.bcq_threshold, "gamma": CFG.gamma, "steps": CFG.bcq_steps, "lr": CFG.bcq_lr,
                "batch": CFG.batch_size}
    return {"alpha": CFG.tab_cql_alpha, "temperature": CFG.cql_temperature, "gamma": CFG.gamma,
            "steps": CFG.tab_cql_steps, "lr": CFG.tab_cql_lr, "batch": CFG.batch_size}


def _x_hp_match(algo: str) -> bool:
    same = {"iql": [XCFG.iql_expectile_default == CFG.iql_expectile, XCFG.iql_beta == CFG.iql_beta,
                    XCFG.iql_max_weight == CFG.iql_max_weight],
            "bcq": [XCFG.bcq_tau_default == CFG.bcq_threshold],
            "cql": [XCFG.cql_alpha_default == CFG.tab_cql_alpha, CFG.cql_temperature == 1.0]}[algo]
    return bool(all(same) and CFG.gamma == 1.0)


def _x_icu_return(pi: np.ndarray, rule: str) -> Tuple[float, str]:
    try:
        return float(EVALUATOR.evaluate(pi, rule)["return"]), "ok"
    except ImproperPolicyError:
        return float("nan"), "improper"
    except EvaluationInvariantError:
        return float("nan"), "near-singular"


def x_tabular_crosscheck(seeds: Sequence[int]) -> pd.DataFrame:
    """Part II exact IQL / BCQ / CQL (defaults) on the full OFFLINE data vs the Part I SGD learners, per seed."""
    assert X_ICU.S == S_N and X_ICU.A == A_N
    part1: Dict[Tuple[str, str, int], np.ndarray] = {}
    for algo, names in _X_PART1_NAMES.items():
        for seed in seeds:
            pols = _x_part1_tabular_policies(algo, get_tab(algo, MAIN, seed))
            for variant, name in zip(("native", "masked"), names):
                pi = validate_policy(pols[variant], f"{name} seed={seed}")
                reg = REGISTRY.get(name)
                if reg is not None and int(seed) in reg.by_seed:
                    assert np.array_equal(reg.by_seed[int(seed)], pi), f"{name} seed={seed}: TAB and REGISTRY differ"
                part1[(algo, variant, int(seed))] = pi
    identity = {"kind": "xenv_tabular_crosscheck", "offline": MAIN.data_hash, "x_icu": X_ICU.identity(),
                "code": XENV_CODE_HASH, "crosscheck_code": code_hash(*_XCROSSCHECK_CODE), "xcfg": x_result_xcfg(),
                "diffusion": [CFG.diffusion_T, CFG.diffusion_beta_start, CFG.diffusion_beta_end], "run_mode": RUN_MODE,
                "part1": {f"{a}/{v}/{s}": array_hash(p) for (a, v, s), p in sorted(part1.items())},
                "part1_hp": {a: _x_part1_hp(a) for a in _X_PART1_NAMES}}
    key = "crosscheck/icu/offline"
    hit = XCACHE.get(key, identity)
    if hit is not None:
        return hit
    XCACHE.require_allowed(key)
    recs = run_xmethods(LearnerView.from_env(X_ICU), OFFLINE, "crosscheck|icu|offline")
    seen = LIVE & (MAIN.counts.sum(axis=1) > 0)
    rows = []
    for algo, method in _X_PART2_METHOD.items():
        for variant, name in zip(("native", "masked"), _X_PART1_NAMES[algo]):
            hits = [r for r in recs if r["method"] == method and r["variant"] == variant and r["is_default"]]
            assert len(hits) == 1, f"expected one default '{method} [{variant}]' record from run_xmethods, got {len(hits)}"
            r2 = hits[0]
            pi2 = validate_policy(r2["pi"], f"{method} [{variant}] (Part II)")
            j2 = {rule: _x_icu_return(pi2, rule) for rule in ("mean", "terminate")}
            for seed in seeds:
                pi1 = part1[(algo, variant, int(seed))]
                try:
                    occ = EVALUATOR.evaluate(pi1, "mean", return_vectors=True)["occupancy_vector"]
                except (ImproperPolicyError, EvaluationInvariantError):
                    occ = None
                j1 = {rule: _x_icu_return(pi1, rule) for rule in ("mean", "terminate")}
                row = {"algo": algo, "variant": variant, "seed": int(seed), "part2_policy": x_label(method, variant),
                       "part1_policy": name, "part2_hp": canonical_json(dict(r2["hp"] or {})),
                       "part1_hp": canonical_json(_x_part1_hp(algo)), "hp_match": _x_hp_match(algo),
                       "part2_converged": r2["info"].get("converged"),
                       "agreement_live": x_action_agreement(pi2, pi1, LIVE),
                       "agreement_seen": x_action_agreement(pi2, pi1, seen),
                       "agreement_occupancy_weighted": (float("nan") if occ is None
                                                        else x_action_agreement(pi2, pi1, LIVE, occ)),
                       "n_live": int(LIVE.sum()), "n_seen": int(seen.sum()),
                       "part2_deterministic": _x_is_deterministic(pi2, LIVE),
                       "part1_deterministic": _x_is_deterministic(pi1, LIVE)}
                for rule in ("mean", "terminate"):
                    row[f"return_part2_{rule}"], row[f"status_part2_{rule}"] = j2[rule]
                    row[f"return_part1_{rule}"], row[f"status_part1_{rule}"] = j1[rule]
                    row[f"diff_return_{rule}"] = j2[rule][0] - j1[rule][0]
                rows.append(row)
    del recs
    XCACHE.put(key, identity, pd.DataFrame(rows))
    return XCACHE.get(key, identity)


_XCROSSCHECK_CODE = [x_tabular_crosscheck, x_action_agreement, _x_is_deterministic, _x_part1_tabular_policies,
                     _x_part1_hp, _x_hp_match, _x_icu_return]
XTAB_CROSSCHECK = pd.DataFrame()
if EXPERIMENTS["xenv_tabular_crosscheck"] and EXPERIMENTS["main_tabular"]:
    XTAB_CROSSCHECK = x_tabular_crosscheck(CFG.seeds)
    XTAB_CROSSCHECK.to_csv(XENV_TABLE_DIR / "xenv_tabular_crosscheck.csv", index=False)
    if not XTAB_CROSSCHECK["hp_match"].astype(bool).all():
        print("NOTE: Part II default hyper-parameters differ from Part I for",
              sorted(XTAB_CROSSCHECK.loc[~XTAB_CROSSCHECK["hp_match"].astype(bool), "algo"].unique()),
              "- the cross-check is then descriptive of both changes.")
    _xc = XTAB_CROSSCHECK.assign(abs_diff_mean=XTAB_CROSSCHECK["diff_return_mean"].abs(),
                                 abs_diff_terminate=XTAB_CROSSCHECK["diff_return_terminate"].abs())
    display(_xc.groupby(["algo", "variant"]).agg(
        seeds=("seed", "nunique"), hp_match=("hp_match", "all"), agreement_live=("agreement_live", "mean"),
        agreement_seen=("agreement_seen", "mean"), agreement_occupancy=("agreement_occupancy_weighted", "mean"),
        mean_abs_diff_mean=("abs_diff_mean", "mean"), max_abs_diff_mean=("abs_diff_mean", "max"),
        mean_abs_diff_terminate=("abs_diff_terminate", "mean"),
        max_abs_diff_terminate=("abs_diff_terminate", "max")).round(5))
else:
    print("Exact-tabular cross-check not run (needs EXPERIMENTS['xenv_tabular_crosscheck'] and EXPERIMENTS['main_tabular']).")

XENV_DRIVER_CODE = [*_XDRIVER_HASHED, x_projection_identity, x_check_draw_rows, x_check_results, x_eta_seconds,
                    _x_fmt_s, _x_flatten, *_XCROSSCHECK_CODE]
print("Part II driver complete:", len(XRESULTS), "result rows,", len(XDRAW_META), "draws.")
