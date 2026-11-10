# %% [markdown]
# ### II.9d Bootstrap with the construction re-estimated in every replicate
#
# The certificates of Corollary 2 hold for the **given** supported rows. A benchmark's rows, mask, decision states and
# d0 are estimated from hospital data, so its brackets also carry estimation error. II.8g measured part of that error
# with B = 10 replicates, a **fixed** mask and fixed decision states, and scored coverage only over policies with
# p > 0. This subsection repeats the bootstrap with B = `XV4CFG.boot_replicates` (200) replicates, in two variants:
#
# * `fixed`: the II.8g model (`x_boot_model`: the supported rows and d0 re-estimated from the resampled steps, with the
#   original mask, decision states, censoring and redirects), on the same random streams as II.8g;
# * `reestimated`: the **whole construction** is rerun on every resampled hospital data set (`x_benchmark_from_data`:
#   informative steps, count mask at τ, dead-state removal, trap censoring, dangling redirects, empirical d0,
#   fallback action). The mask and the decision states can change from one replicate to the next.
#
# **Cells:** `XV4CFG.boot_cells` (resolved by `x_v4_cells`; a configured cell whose family switch is off is skipped and
# listed). The default set is Gumbel `best` with n = 10 000, τ = 20 and with n = 2 000, τ = 5, and Garnet h ∈ {0, 1}
# at ε = 0.05. At most `XV4CFG.max_draws` draws per cell; default policies only (`is_default`), as in II.8g.
#
# **Per (cell, draw).**
# 1. Rebuild the instance (`x_v4_instance`), regenerate the default policies (`x_v4_default_records`) and the hospital
#    episodes and benchmark B (`x_v4_rebuild_benchmark`, which asserts that B equals the instance's benchmark).
# 2. Step map `smap = x_boot_step_map(true_env, hosp, τ, B)` (II.8g). Unit weights must reproduce the benchmark:
#    `x_boot_model` with weights 1 equals `learn.P` and `learn.d0` (1e-14), and the resampled data set with every
#    multiplicity 1 is the hospital data set, whose construction equals B (identity, index, decision states and
#    fallback action). Both are asserted.
# 3. For b = 0, …, B − 1: episode multiplicities `mult = x_boot_weights(n_ep, np_rng("xenv", "bootstrap", cell, draw,
#    b))`, the II.8g stream, so the first 10 `fixed` replicates are the II.8g replicates (II.8g default
#    `resample_d0 = True`).
#    * `fixed`: `x_boot_model(learn, smap, mult[episode], True)`; every default policy is evaluated exactly under the
#      `given` (mean) rule and the `terminate` rule: J and p = P(any unsupported decision).
#    * `reestimated`: `x_v4_boot_dataset` repeats episode e `mult[e]` times with new consecutive `episode_id`s, keeping
#      the step order, `timestep`, `cutoff`, `terminated` and every other array field; the data set is validated with
#      `x_validate_dataset`. Then `Bb = x_benchmark_from_data(true_env, hosp_b, τ, f"{name}|boot{b}")` with the
#      default dead-state rule. A construction that fails (`AssertionError` / `ValueError`) is recorded as a `failed`
#      replicate with its message, and the draw continues. Policies are mapped through the true state space:
#      π_true = `B["lift"](π)`, π_b = π_true[`Bb["index"]`] (validated), and evaluated on `Bb["bench"]` under both
#      rules; p refers to the replicate's own mask.
# 4. Point values on B (replicate −1) and the truth: J_true on the `true` target (T_masked, `given`, lifted) and the
#    true-rows terminate value on `true_rows_terminate`.
#
# **Intervals** (per policy and variant, over the **proper** replicates: not failed and `ok` under both rules; linear
# quantiles at `XV4CFG.boot_quantiles` = (2.5 %, 97.5 %), which configure the analysis only):
#
# | Interval | Lower end | Upper end |
# |---|---|---|
# | `nhb` | q_lo(J_term^b) | q_hi(J_mean^b) |
# | `free` (no NHB) | q_lo(J_term^b) | q_hi(J_term^b + p^b (v_max − v_min)) |
# | `minmax` | min J_term^b | max J_mean^b |
# | `point` variant | J_bench_term (both types) | J_bench_mean (`nhb`); J_bench_term + p (v_max − v_min) (`free`) |
#
# The interval table also carries the number of replicates, failed replicates and proper replicates of each policy,
# the share of proper replicates with p^b > 0, the point values, p on the point benchmark and the truth.
#
# **Invariants asserted per draw** (re-run on cached tables): the point benchmark and the truth are proper for every
# policy; every (variant, replicate) holds the whole policy set once; failed replicates are whole replicates of the
# `reestimated` variant; on every benchmark model (point, fixed, re-estimated) the two rules share p (1e-9) and
# 0 ≤ J_mean − J_term ≤ p (v_max − v_min) (Theorem 1 part iii, 1e-9). **Counted per draw:** the true-rows bracket
# J_rows_term ≤ J_true ≤ J_rows_term + p (v_max − v_min) (Theorem 1 on the true rows); as for the theorem counts of
# II.8i and II.9b, a violation fails the summary checks and the analysis cell raises after writing its files.
# **Reported, not asserted:** the agreement of the first `fixed` replicates (and of replicate −1) with the II.8g rows of
# the same draw (`XV3_BOOT_RESULTS`, else `xenv/v3_bootstrap_results.csv.gz`): same streams and model, but BLAS
# rounding can differ across machines.
#
# **Cache.** Key `v4boot/<cell>/draw<k>` (and `/meta`). The identity is the draw identity (`x_v4_draw_identity`), the
# code hash of this subsection (its functions, the II.8g functions it calls, and `XV4_COMMON_CODE_HASH`), B, the
# variants, `resample_d0` and the stream names. The raw replicate rows live **only in the cache**, in compact columns
# (`variant`, `replicate`, `policy`, `status_given`, `status_term`, `J_given`, `J_term`, `p_given`, `p_term`; about
# 20 000 rows per draw). The table returned per draw is the interval table, computed from the rows read back.
#
# | In memory | Content |
# |---|---|
# | `XV4_BOOT_IV` | one row per draw × default policy × variant (`point`, `fixed`, `reestimated`; II.9d analyses add `v3boot`) |
# | `XV4_BOOT_META` | per draw: replicate counts, failures and messages, construction variability of the re-estimated replicates (decision states gained / lost, mask rows changed, dead and trap states, d0 support), fixed-model fallback pairs and traps, unit-weight checks, invariant slacks, II.8g agreement, timings |
#
# The files are written by the analysis cell below.
#
# **Expected cost on Colab.** These are estimates, not measurements. Per draw: an instance rebuild and one
# `run_xmethods` fit (4–16 s, as in II.6), then per replicate one fixed-mask model and one full construction, each
# followed by 2 × ~40 exact evaluations of the distinct default policies (≈ 32 000 evaluations per draw at B = 200).
# With the II.6 / II.8g Colab timings (about 3–8 ms per Gumbel benchmark evaluation, 1–3 ms on Garnet; a Gumbel
# construction from about 200 000 steps at n = 10 000 in about 0.2–0.5 s), that is about 3–6 min per Gumbel draw and
# 1–2 min per Garnet draw, i.e. roughly 4–8 h for the default 4 cells × 30 draws (lite profile: about 1.5–3 h; smoke: a
# few minutes). Every draw is cached, so a disconnect loses at most the draw in progress; a cached draw reloads in
# about 3–5 s.

# %%
# ============================================================
# II.9d Bootstrap with the construction re-estimated per replicate
# ============================================================
EXECLOG.mark("II.9 Second-revision experiments: bootstrap with re-estimated construction")
import traceback

XV4_BOOT_POINT = XV3_BOOT_POINT                  # replicate label of the point benchmark and of the truth (-1)
XV4_BOOT_VARIANTS = ("fixed", "reestimated")     # bootstrap variants this subsection can compute
XV4_BOOT_VARIANT_ORDER = ("point", "v3boot", "fixed", "reestimated")   # row order of the analyses
XV4_BOOT_INTERVALS = ("nhb", "free", "minmax")
XV4_BOOT_STATUSES = (*XV3_EVAL_STATUSES, "failed")   # 'failed': the re-estimated construction raised
XV4_BOOT_RAW_COLUMNS = ["variant", "replicate", "policy", "status_given", "status_term", "J_given", "J_term",
                        "p_given", "p_term"]
XV4_BOOT_IV_COLUMNS = ["cell", "family", "draw", "policy", "method", "policy_variant", "method_family",
                       "boot_variant", "replicates", "n_failed", "n_proper", "n_nonok", "true", "p_true",
                       "true_rows_term", "point_term", "point_mean", "p_bench", "p_boot_pos_share",
                       "lo_nhb", "up_nhb", "lo_free", "up_free", "lo_minmax", "up_minmax"]
assert XV4CFG.boot_variants and set(XV4CFG.boot_variants) <= set(XV4_BOOT_VARIANTS), XV4CFG.boot_variants
assert int(XV4CFG.boot_replicates) >= 1 and 0.0 <= XV4CFG.boot_quantiles[0] < XV4CFG.boot_quantiles[1] <= 1.0


# ---------- resampled hospital data and re-estimated construction ----------
def x_v4_boot_dataset(hosp: Dict[str, np.ndarray], mult: np.ndarray) -> Dict[str, np.ndarray]:
    """Hospital data set of one bootstrap resample: episode e is repeated mult[e] times (episodes in id order), with
    new consecutive episode ids. Step order, timestep, cutoff, terminated and every other array field are carried."""
    ep = np.asarray(hosp["episode_id"], dtype=np.int64)
    n = len(ep)
    assert n > 0 and all(isinstance(v, np.ndarray) and len(v) == n for v in hosp.values()), \
        "every hospital field must be an array with one entry per step"
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    n_ep = len(starts)
    assert np.array_equal(ep[starts], np.arange(n_ep)), "hospital episode ids must be 0..n-1, one block each, in order"
    m = np.asarray(mult, dtype=np.float64)
    assert m.shape == (n_ep,) and np.all(m >= 0) and np.all(m == np.round(m)), "multiplicities must be counts"
    lengths = np.diff(np.r_[starts, n])
    src = np.repeat(np.arange(n_ep), m.astype(np.int64))          # original episode of every new episode
    L = lengths[src]
    new_ep = np.repeat(np.arange(len(src)), L)
    offset = np.arange(int(L.sum())) - np.repeat(np.cumsum(L) - L, L)
    rows = starts[src][new_ep] + offset
    out = {k: v[rows] for k, v in hosp.items()}
    out["episode_id"] = new_ep.astype(hosp["episode_id"].dtype)
    return out


def x_v4_boot_construct(true_env: FiniteEnv, hosp_b: Dict[str, np.ndarray], tau: int, name: str,
                        empty_rule: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(x_benchmark_from_data on one resampled data set, None), or (None, message) when the construction fails with
    an AssertionError / ValueError. The benchmark name in the message is replaced by '<name>' so messages group;
    messages are cut to 150 characters so that the draw meta keeps them as a JSON list (_x_compact limit)."""
    try:
        return x_benchmark_from_data(true_env, hosp_b, tau, name, empty_rule=empty_rule), None
    except (AssertionError, ValueError) as exc:
        tb = traceback.extract_tb(exc.__traceback__)
        where = f" [{tb[-1].name}:{tb[-1].lineno}]" if tb else ""
        return None, f"{type(exc).__name__}{where}: {str(exc).replace(name, '<name>')}"[:150]


def _x_v4_boot_eval(env: FiniteEnv, pis: Dict[bytes, np.ndarray]) -> Dict[bytes, Tuple[dict, dict]]:
    """Exact evaluation of every distinct policy on `env` under the given (mean) and the terminate rule."""
    return {k: (x_v4_eval(env, pi, "given"), x_v4_eval(env, pi, "terminate")) for k, pi in pis.items()}


def _x_v4_boot_rows(variant: str, replicate: int, labels: Sequence[str], keys: Sequence[bytes],
                    ev: Optional[Dict[bytes, Tuple[dict, dict]]]) -> List[dict]:
    """Compact raw rows of one (variant, replicate); ev=None marks a failed replicate."""
    fail = ({"status": "failed"}, {"status": "failed"})
    out = []
    for lab, k in zip(labels, keys):
        g, t = fail if ev is None else ev[k]
        out.append({"variant": variant, "replicate": int(replicate), "policy": lab, "status_given": g["status"],
                    "status_term": t["status"], "J_given": float(g.get("return", np.nan)),
                    "J_term": float(t.get("return", np.nan)), "p_given": float(g.get("prob_any_violation", np.nan)),
                    "p_term": float(t.get("prob_any_violation", np.nan))})
    return out


def x_v4_boot_construction_stats(Bb: Dict[str, Any], B: Dict[str, Any]) -> Dict[str, float]:
    """How the re-estimated construction Bb differs from the benchmark B (decision states, mask rows, censoring)."""
    rep = Bb["report"]
    V, V0 = np.asarray(Bb["decision_states_true"]), np.asarray(B["decision_states_true"])
    common = np.intersect1d(V, V0)
    changed = ~(Bb["mask_true"][common] == B["mask_true"][common]).all(axis=1)
    return {"decision_states": int(rep["decision_states"]), "decision_states_new": int(len(np.setdiff1d(V, V0))),
            "decision_states_lost": int(len(np.setdiff1d(V0, V))), "mask_rows_changed": int(changed.sum()),
            "mask_rows_changed_share": float(changed.mean()) if len(common) else float("nan"),
            "admissible_pairs": int(rep["admissible_pairs"]),
            "mean_admissible_actions": float(rep["mean_admissible_actions"]),
            "dead_states_removed": int(rep["dead_states_removed"]),
            "trap_states_censored": int(rep["trap_states_censored"]),
            "censored_d0_mass": float(rep["censored_d0_mass"]), "d0_support_states": int((Bb["bench"].d0 > 0).sum()),
            "fallback_action_changed": int(int(rep["fallback_action"]) != int(B["fallback_action"]))}


def _x_v4_boot_dist(stats_: Sequence[dict]) -> Dict[str, Dict[str, Optional[float]]]:
    """min / median / mean / max over replicates of every statistic in `stats_` (None if no finite value)."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k in (stats_[0] if stats_ else {}):
        v = np.asarray([s[k] for s in stats_], dtype=np.float64)
        v = v[np.isfinite(v)]
        out[k] = ({"min": float(v.min()), "median": float(np.median(v)), "mean": float(v.mean()), "max": float(v.max())}
                  if len(v) else {"min": None, "median": None, "mean": None, "max": None})
    return out


def x_v4_boot_unit_check(true_env: FiniteEnv, hosp: Dict[str, np.ndarray], tau: int, B: Dict[str, Any],
                         learn: FiniteEnv, smap: Dict[str, Any]) -> Dict[str, Any]:
    """Unit weights reproduce the benchmark in both variants (asserted): x_boot_model with weights 1 gives learn.P and
    learn.d0, and the data set of unit multiplicities is `hosp`, whose construction equals B."""
    P1, d01, _ = x_boot_model(learn, smap, np.ones(len(smap["flat"])), True)
    out = {"max_abs_diff_P": float(np.abs(P1 - learn.P).max()), "max_abs_diff_d0": float(np.abs(d01 - learn.d0).max())}
    assert max(out.values()) <= 1e-14, f"{learn.name}: unit weights do not reproduce the benchmark {out}"
    del P1, d01
    n_ep = int(np.asarray(hosp["episode_id"]).max()) + 1
    h1 = x_v4_boot_dataset(hosp, np.ones(n_ep))
    assert set(h1) == set(hosp) and all(h1[k].dtype == hosp[k].dtype and np.array_equal(h1[k], hosp[k])
                                        for k in hosp), "unit multiplicities must give the hospital data set back"
    B1 = x_benchmark_from_data(true_env, h1, tau, learn.name)
    same = (obj_hash(B1["bench"].identity()) == obj_hash(B["bench"].identity())
            and np.array_equal(B1["index"], B["index"])
            and np.array_equal(B1["decision_states_true"], B["decision_states_true"])
            and int(B1["fallback_action"]) == int(B["fallback_action"]))
    assert same, f"{learn.name}: the construction on the unit-multiplicity data set differs from the benchmark"
    out["construction_identity"] = True
    return out


def x_v4_boot_identity(spec: dict, inst: CellInstance) -> dict:
    return {"kind": "xenv_v4_bootstrap", "draw": x_v4_draw_identity(spec, inst), "code": XV4_BOOT_CODE_HASH,
            "replicates": int(XV4CFG.boot_replicates), "variants": list(XV4CFG.boot_variants), "resample_d0": True,
            "rng": ["xenv", "bootstrap", "<cell>", "<draw>", "<b>"], "columns": XV4_BOOT_RAW_COLUMNS,
            "policies": "default (is_default) records of x_v4_default_records"}


# ---------- checks and intervals (pure pandas; re-run on cached tables) ----------
def x_v4_boot_check_rows(df: pd.DataFrame, v_range: float,
                         expect: Optional[Tuple[int, Sequence[str]]] = None) -> Dict[str, Any]:
    """Invariants of one draw's raw replicate rows (see markdown); asserted, and a small report is returned.
    expect = (B, variants): every variant holds exactly the replicates 0..B-1 (a complete table)."""
    sg, stt = df["status_given"].astype(str), df["status_term"].astype(str)
    var = df["variant"].astype(str)
    bad = sorted((set(sg) | set(stt)) - set(XV4_BOOT_STATUSES))
    assert not bad, f"unexpected bootstrap status values {bad}"
    assert set(var) <= {"point", "truth", *XV4_BOOT_VARIANTS}, sorted(set(var))
    dup = df.duplicated(["variant", "replicate", "policy"])
    assert not dup.any(), f"{int(dup.sum())} duplicated (variant, replicate, policy) rows"
    rep = df["replicate"].astype(np.int64)
    one = var.isin(["point", "truth"])
    assert (rep[one] == XV4_BOOT_POINT).all() and (rep[~one] >= 0).all(), "replicate labels"
    assert {"point", "truth"} <= set(var), "the point benchmark and truth rows (replicate -1) are missing"
    if expect is not None:
        n_rep, variants = int(expect[0]), [str(v) for v in expect[1]]
        assert set(var) - {"point", "truth"} == set(variants), (sorted(set(var)), variants)
        for v in variants:
            assert np.array_equal(np.sort(rep[var == v].unique()), np.arange(n_rep)), \
                f"variant {v}: the replicates are not exactly 0..{n_rep - 1}"
    assert ((sg[one] == "ok") & (stt[one] == "ok")).all(), \
        "a default policy is not proper on the point benchmark or on the truth"
    pols = set(df.loc[var == "point", "policy"])
    n_pol = df.groupby([var, rep])["policy"].nunique()
    assert pols and set(df["policy"]) == pols and (n_pol == len(pols)).all(), \
        "every (variant, replicate) must hold the whole default policy set"
    fg, ft = sg == "failed", stt == "failed"
    assert (fg == ft).all() and not (fg & (var != "reestimated")).any(), "failed rows outside re-estimated replicates"
    whole = fg.groupby([var, rep]).agg(["all", "any"])
    assert (whole["all"] == whole["any"]).all(), "a failed replicate must fail for every policy"
    bench = var.isin(["point", *XV4_BOOT_VARIANTS]) & (sg == "ok") & (stt == "ok")
    jm, jt = df.loc[bench, "J_given"].to_numpy(float), df.loc[bench, "J_term"].to_numpy(float)
    pm, ptm = df.loc[bench, "p_given"].to_numpy(float), df.loc[bench, "p_term"].to_numpy(float)
    width = jm - jt
    dp = float(np.abs(pm - ptm).max(initial=0.0))
    slack_lo = float((-width).max(initial=-np.inf))
    slack_hi = float((width - ptm * float(v_range)).max(initial=-np.inf))
    assert dp <= XV4_TOL, f"the two rules disagree on P(any unsupported decision) by {dp:.3g}"
    assert slack_lo <= XV4_TOL, f"J_mean < J_term on a bootstrap model by {slack_lo:.3g} (Theorem 1 part iii)"
    assert slack_hi <= XV4_TOL, f"J_mean - J_term exceeds p (v_max - v_min) by {slack_hi:.3g} (Theorem 1 part iii)"
    tr = df[var == "truth"]
    jtrue, jrt, ptr = (tr[c].to_numpy(float) for c in ("J_given", "J_term", "p_term"))
    bracket = (jrt > jtrue + XV4_TOL) | (jtrue > jrt + ptr * float(v_range) + XV4_TOL)
    nonok = df[~(fg | ((sg == "ok") & (stt == "ok")))].groupby("variant").size()
    return {"rows": int(len(df)), "policies": int(len(pols)), "pairs_checked": int(bench.sum()),
            "replicates": {str(v): int(rep[var == v].nunique()) for v in XV4_BOOT_VARIANTS if (var == v).any()},
            "failed_replicates": int(rep[fg].nunique()), "nonok_rows": int(nonok.sum()),
            "nonok_by_variant": {str(k): int(v) for k, v in nonok.items()}, "max_rule_p_difference": dp,
            "max_negative_width": slack_lo, "max_width_minus_bound": slack_hi,
            "truth_rows_bracket_violations": int(bracket.sum())}


def x_v4_boot_intervals(raw: pd.DataFrame, v_range: float, q: Tuple[float, float]) -> pd.DataFrame:
    """Per policy x variant of one draw: truth, point values, replicate counts and the nhb / free / minmax intervals
    over the proper replicates (not failed, 'ok' under both rules; linear quantiles as in II.8i). The 'point' rows hold
    the point intervals [J_term, J_mean] and [J_term, J_term + p v_range] (minmax undefined)."""
    var = raw["variant"].astype(str)
    pt = raw[var == "point"].set_index("policy")
    pol = pt.index
    assert len(pol) and pol.is_unique, "one point row per policy"
    tr = raw[var == "truth"].set_index("policy").reindex(pol)
    vr = float(v_range)
    base = pd.DataFrame({"policy": pol.to_numpy(), "true": tr["J_given"].to_numpy(float),
                         "p_true": tr["p_given"].to_numpy(float),
                         "true_rows_term": np.where(tr["status_term"].astype(str).to_numpy() == "ok",
                                                    tr["J_term"].to_numpy(float), np.nan),
                         "point_term": pt["J_term"].to_numpy(float), "point_mean": pt["J_given"].to_numpy(float),
                         "p_bench": pt["p_given"].to_numpy(float)})
    nan = np.full(len(pol), np.nan)
    frames = [base.assign(boot_variant="point", replicates=0, n_failed=0, n_proper=0, n_nonok=0, p_boot_pos_share=nan,
                          lo_nhb=base["point_term"], up_nhb=base["point_mean"], lo_free=base["point_term"],
                          up_free=base["point_term"] + base["p_bench"] * vr, lo_minmax=nan, up_minmax=nan)]
    others = [v for v in XV4_BOOT_VARIANT_ORDER if v != "point" and (var == v).any()]
    assert set(var) <= {"point", "truth", *others}, sorted(set(var))
    for v in others:
        sub = raw[var == v]
        reps = np.sort(sub["replicate"].astype(np.int64).unique())

        def mat(col: str) -> pd.DataFrame:
            return sub.pivot(index="policy", columns="replicate", values=col).reindex(index=pol, columns=reps)

        sg, stt = mat("status_given"), mat("status_term")
        ok = ((sg == "ok") & (stt == "ok")).to_numpy()
        n_failed = int((sg == "failed").all(axis=0).sum())
        Jg = np.where(ok, mat("J_given").to_numpy(float), np.nan)
        Jt = np.where(ok, mat("J_term").to_numpy(float), np.nan)
        Pg = np.where(ok, mat("p_given").to_numpy(float), np.nan)
        n_ok = ok.sum(axis=1)
        has = n_ok > 0
        res = {c: nan.copy() for c in ("lo_nhb", "up_nhb", "up_free", "lo_minmax", "up_minmax", "p_boot_pos_share")}
        if has.any():
            res["lo_nhb"][has] = np.nanquantile(Jt[has], float(q[0]), axis=1)
            res["up_nhb"][has] = np.nanquantile(Jg[has], float(q[1]), axis=1)
            res["up_free"][has] = np.nanquantile(Jt[has] + Pg[has] * vr, float(q[1]), axis=1)
            res["lo_minmax"][has] = np.nanmin(Jt[has], axis=1)
            res["up_minmax"][has] = np.nanmax(Jg[has], axis=1)
            res["p_boot_pos_share"][has] = (np.nan_to_num(Pg[has], nan=0.0) > XV4_PTOL).sum(axis=1) / n_ok[has]
        frames.append(base.assign(boot_variant=v, replicates=len(reps), n_failed=n_failed, n_proper=n_ok,
                                  n_nonok=len(reps) - n_failed - n_ok, lo_free=res["lo_nhb"], **res))
    return pd.concat(frames, ignore_index=True)


# ---------- II.8g rows of the same draws (comparison only; never cached) ----------
def x_v4_boot_from_v3(boot: pd.DataFrame) -> Dict[Tuple[str, int], pd.DataFrame]:
    """II.8g rows (XV3_BOOT_COLUMNS, one row per target) in this subsection's raw format, by (cell, draw): 'point'
    and 'truth' at replicate -1 (no true-rows terminate value: status 'absent') and 'v3boot' for the replicates."""
    if boot is None or not len(boot):
        return {}
    b = boot[boot["target"].astype(str).isin(["bench_mean", "bench_terminate", "true"])].copy()
    b["draw"] = pd.to_numeric(b["draw"]).astype(np.int64)
    b["replicate"] = pd.to_numeric(b["replicate"]).astype(np.int64)
    k = ["cell", "draw", "replicate", "policy"]
    cols = {"status": "status", "return": "J", "prob_any_violation": "p"}

    def part(target: str, sfx: str) -> pd.DataFrame:
        d = b.loc[b["target"] == target, k + list(cols)]
        return d.rename(columns={c: f"{n}_{sfx}" for c, n in cols.items()})

    bench = part("bench_mean", "given").merge(part("bench_terminate", "term"), on=k, how="outer", validate="one_to_one")
    bench["variant"] = np.where(bench["replicate"] == XV3_BOOT_POINT, "point", "v3boot")
    tr = part("true", "given").assign(variant="truth", status_term="absent", J_term=np.nan, p_term=np.nan)
    allr = pd.concat([bench, tr], ignore_index=True)
    for c in ("status_given", "status_term"):
        allr[c] = allr[c].fillna("missing").astype(str)
    return {(str(c), int(d)): g[XV4_BOOT_RAW_COLUMNS].reset_index(drop=True)
            for (c, d), g in allr.groupby(["cell", "draw"], sort=False)}


def x_v4_boot_v3_reference(cells: Sequence[str]) -> Tuple[Dict[Tuple[str, int], pd.DataFrame], str]:
    """II.8g rows of `cells` by (cell, draw): XV3_BOOT_RESULTS, or else the saved xenv/v3_bootstrap_results.csv.gz."""
    src = globals().get("XV3_BOOT_RESULTS")
    origin = "XV3_BOOT_RESULTS"
    if src is None or not len(src):
        path = XENV_TABLE_DIR / "v3_bootstrap_results.csv.gz"
        if not path.exists():
            return {}, "not available (II.8g not run in this session and no saved table)"
        src = pd.read_csv(path, float_precision="round_trip",
                          dtype={"cell": str, "policy": str, "target": str, "status": str})
        origin = f"xenv/{path.name}"
    return x_v4_boot_from_v3(src[src["cell"].astype(str).isin([str(c) for c in cells])]), origin


def x_v4_boot_v3_match(raw: pd.DataFrame, ref: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Replicate -1 and the first `fixed` replicates against the II.8g rows of the same draw (same streams and model;
    equal up to BLAS rounding). Reported, never asserted."""
    if ref is None or not len(ref):
        return {"status": "not compared (no II.8g rows for this draw)", "match": None}
    is_fixed = raw["variant"] == "fixed"
    n_fixed = int(raw.loc[is_fixed, "replicate"].max()) + 1 if is_fixed.any() else 0
    mine = raw[raw["variant"].isin(["point", "truth", "fixed"])].replace({"variant": {"fixed": "v3boot"}})
    r = ref[(ref["variant"] != "v3boot") | (ref["replicate"] < n_fixed)]
    m = r.merge(mine, on=["variant", "replicate", "policy"], how="left", suffixes=("_v3", ""), validate="one_to_one")
    have = m["status_given"].notna()
    with_term = have & (m["variant"] != "truth")
    st = have & ((m["status_given_v3"] != m["status_given"]) | (with_term & (m["status_term_v3"] != m["status_term"])))
    okg = have & (m["status_given_v3"] == "ok") & (m["status_given"] == "ok")
    okt = with_term & (m["status_term_v3"] == "ok") & (m["status_term"] == "ok")
    dJ = pd.concat([(m["J_given_v3"] - m["J_given"]).abs()[okg], (m["J_term_v3"] - m["J_term"]).abs()[okt]])
    dP = pd.concat([(m["p_given_v3"] - m["p_given"]).abs()[okg], (m["p_term_v3"] - m["p_term"]).abs()[okt]])
    var_j = np.concatenate([m.loc[okg, "variant"].to_numpy(), m.loc[okt, "variant"].to_numpy()])
    out = {"status": "compared", "rows": int(len(m)), "missing_in_v4": int((~have).sum()),
           "replicates_compared": int(r.loc[r["variant"] == "v3boot", "replicate"].nunique()),
           "status_mismatches": int(st.sum()), "max_abs_diff_return": float(dJ.max()) if len(dJ) else None,
           "max_abs_diff_p": float(dP.max()) if len(dP) else None,
           "max_abs_diff_return_by_variant": {str(v): float(dJ.to_numpy()[var_j == v].max())
                                              for v in sorted(set(var_j))}}
    out["match"] = bool(out["missing_in_v4"] == 0 and out["status_mismatches"] == 0
                        and (out["max_abs_diff_return"] or 0.0) <= XV4_TOL)
    return out


_XV4_BOOT_V3REF: Dict[Tuple[str, int], pd.DataFrame] = {}


def x_v4_boot_finish(spec: dict, draw: int, raw: pd.DataFrame, meta: dict) -> Tuple[pd.DataFrame, dict]:
    """Interval table of one draw (from the raw rows read back from the cache) and the session meta."""
    meta = dict(meta)
    desc = meta.pop("policy_desc")
    iv = x_v4_boot_intervals(raw, float(meta["v_range"]), XV4CFG.boot_quantiles)
    d = pd.DataFrame.from_dict(desc, orient="index").reindex(iv["policy"].to_numpy())
    assert d[["method", "policy_variant", "method_family"]].notna().all().all(), "policy descriptions missing"
    for c in ("method", "policy_variant", "method_family"):
        iv[c] = d[c].to_numpy()
    iv["cell"], iv["family"], iv["draw"] = str(spec["cell"]), str(spec["family"]), int(draw)
    meta["v3_match"] = x_v4_boot_v3_match(raw, _XV4_BOOT_V3REF.get((str(spec["cell"]), int(draw))))
    return iv[XV4_BOOT_IV_COLUMNS].reset_index(drop=True), meta


# ---------- one (cell, draw) ----------
def run_v4_boot_draw(spec: dict, draw: int) -> Tuple[pd.DataFrame, dict]:
    """Fixed-mask and re-estimated bootstrap of the default policies of one draw (raw rows cached under v4boot/);
    returns the per policy x variant intervals and the draw meta."""
    cell, family, draw = spec["cell"], spec["family"], int(draw)
    assert family in ("gumbel", "garnet"), family
    t0 = time.perf_counter()
    inst = x_v4_instance(spec, draw)
    assert (inst.cell, inst.family, int(inst.draw)) == (cell, family, draw), (inst.cell, inst.family, inst.draw)
    identity = x_v4_boot_identity(spec, inst)
    key, ident_hash = "v4boot/" + x_cache_key(cell, draw), obj_hash(identity)[:20]
    build_s = time.perf_counter() - t0
    hit, meta_hit = XCACHE.get(key, identity), XCACHE.get(key + "/meta", identity)
    if hit is not None and meta_hit is not None:
        del inst
        assert list(hit.columns) == XV4_BOOT_RAW_COLUMNS, f"{key}: cached columns differ from XV4_BOOT_RAW_COLUMNS"
        meta = json.loads(str(meta_hit["meta_json"].iloc[0]))
        meta.update(cache="loaded", session_build_s=build_s,
                    checks=x_v4_boot_check_rows(hit, meta["v_range"], (meta["replicates"], meta["variants"])))
        gc.collect()
        return x_v4_boot_finish(spec, draw, hit, meta)
    if (hit is None) != (meta_hit is None):
        print(f"  {key}: incomplete cache entry (results or meta missing) - recomputing this draw")
    XCACHE.require_allowed(key)
    learn = inst.learn_env
    t1 = time.perf_counter()
    records, keys = x_v4_default_records(inst)
    labels = [r["desc"]["policy"] for r in records]
    assert len(set(labels)) == len(labels), "duplicate default policy labels"
    pis = {k: r["pi"] for r, k in zip(records, keys)}
    t2 = time.perf_counter()
    true_env, _, hosp, B = x_v4_rebuild_benchmark(spec, draw, inst)
    smap = x_boot_step_map(true_env, hosp, spec["tau"], B)
    eps_id = smap["episode"]
    n_ep = int(eps_id.max()) + 1
    assert np.array_equal(np.unique(eps_id), np.arange(n_ep)), "hospital episode ids must be 0..n-1"
    repro = x_v4_boot_unit_check(true_env, hosp, spec["tau"], B, learn, smap)
    lifted = {k: B["lift"](pi) for k, pi in pis.items()}
    tgt = {lab: (env, sem, lift) for lab, env, sem, lift in inst.targets}
    assert {"true", "true_rows_terminate"} <= set(tgt) and tgt["true"][1] == "given" \
        and tgt["true_rows_terminate"][1] == "terminate", sorted(tgt)
    assert np.array_equal(tgt["true"][2](pis[keys[0]]), lifted[keys[0]]), "instance lift and rebuilt lift differ"
    lo, hi = learn.value_bounds
    t3 = time.perf_counter()
    rows = _x_v4_boot_rows("point", XV4_BOOT_POINT, labels, keys, _x_v4_boot_eval(learn, pis))
    (e_t, s_t, f_t), (e_r, s_r, f_r) = tgt["true"], tgt["true_rows_terminate"]
    ev_true = {k: (x_v4_eval(e_t, f_t(pi), s_t), x_v4_eval(e_r, f_r(pi), s_r)) for k, pi in pis.items()}
    rows += _x_v4_boot_rows("truth", XV4_BOOT_POINT, labels, keys, ev_true)
    del ev_true
    t4 = time.perf_counter()
    nB, variants = int(XV4CFG.boot_replicates), tuple(XV4CFG.boot_variants)
    fixed_stats: List[dict] = []
    re_stats: List[dict] = []
    failed: List[Tuple[int, str]] = []
    t_fixed = t_re = 0.0
    for b in trange(nB, desc=f"    {cell} draw {draw}: B = {nB}", leave=False, disable=not SHOW_PROGRESS):
        mult = x_boot_weights(n_ep, np_rng("xenv", "bootstrap", cell, draw, b))
        if "fixed" in variants:
            ta = time.perf_counter()
            P_b, d0_b, st_b = x_boot_model(learn, smap, mult[eps_id], True)
            env_b = learn.variant(f"{learn.name}|boot{b}", P=P_b, d0=d0_b, kind="benchmark", bootstrap_replicate=b)
            del P_b
            st_b["trap_states_given"] = x_trap_states(env_b.P, env_b.live, np.ones((env_b.S, env_b.A), dtype=bool))
            st_b["trap_states_terminate"] = x_trap_states(env_b.P, env_b.live, env_b.mask)
            rb = _x_v4_boot_rows("fixed", b, labels, keys, _x_v4_boot_eval(env_b, pis))
            st_b["nonok_rows"] = int(sum(r["status_given"] != "ok" or r["status_term"] != "ok" for r in rb))
            rows += rb
            fixed_stats.append(st_b)
            del env_b, rb
            t_fixed += time.perf_counter() - ta
        if "reestimated" in variants:
            ta = time.perf_counter()
            hosp_b = x_v4_boot_dataset(hosp, mult)
            x_validate_dataset(hosp_b, true_env)
            Bb, err = x_v4_boot_construct(true_env, hosp_b, spec["tau"], f"{learn.name}|boot{b}")
            del hosp_b
            if Bb is None:
                failed.append((b, str(err)))
                rows += _x_v4_boot_rows("reestimated", b, labels, keys, None)
            else:
                env_b, idx = Bb["bench"], np.asarray(Bb["index"], dtype=np.int64)
                pis_b = {k: validate_policy(p_true[idx], f"lifted policy on {env_b.name}", shape=(env_b.S, env_b.A))
                         for k, p_true in lifted.items()}
                rb = _x_v4_boot_rows("reestimated", b, labels, keys, _x_v4_boot_eval(env_b, pis_b))
                st_b = x_v4_boot_construction_stats(Bb, B)
                st_b["nonok_rows"] = int(sum(r["status_given"] != "ok" or r["status_term"] != "ok" for r in rb))
                re_stats.append(st_b)
                rows += rb
                del env_b, pis_b, rb
            del Bb
            t_re += time.perf_counter() - ta
    t5 = time.perf_counter()
    df = pd.DataFrame(rows, columns=XV4_BOOT_RAW_COLUMNS)
    checks = x_v4_boot_check_rows(df, hi - lo, (nB, variants))
    msgs = Counter(m for _, m in failed)
    meta = _x_compact({
        "cell": cell, "family": family, "draw": draw, "identity_hash": ident_hash, "code_hash": XV4_BOOT_CODE_HASH[:16],
        "cache": "computed", "replicates": nB, "variants": list(variants), "resample_d0": True,
        "v_range": float(hi - lo), "n_policies": len(records), "unique_policies": len(pis),
        "policy_desc": {r["desc"]["policy"]: {"method": r["desc"]["method"], "policy_variant": r["desc"]["variant"],
                                              "method_family": r["desc"]["method_family"]} for r in records},
        "hospital_episodes": n_ep, "step_map": {k: v for k, v in smap.items() if not isinstance(v, np.ndarray)},
        "benchmark": {"decision_states": int(len(B["decision_states_true"])),
                      "admissible_pairs": int(B["report"]["admissible_pairs"]),
                      "empty_rule": B["report"]["empty_rule"]},
        "reproduction": repro,
        "fixed": {"replicates": len(fixed_stats), **_x_v4_boot_dist(fixed_stats),
                  "fallback_pairs_total": int(sum(s["fallback_pairs"] for s in fixed_stats)),
                  "trap_states_given_max": int(max([s["trap_states_given"] for s in fixed_stats], default=0))},
        "reestimated": {"replicates": len(re_stats) + len(failed), "failed": len(failed), **_x_v4_boot_dist(re_stats)},
        "failed_replicates": [b for b, _ in failed],
        "failure_messages": [[m, int(n)] for m, n in msgs.most_common(5)],
        "checks": checks,
        "timing": {"build_s": build_s, "methods_s": t2 - t1, "rebuild_s": t3 - t2, "point_eval_s": t4 - t3,
                   "fixed_s": t_fixed, "reestimated_s": t_re, "bootstrap_s": t5 - t4,
                   "total_s": time.perf_counter() - t0}})
    XCACHE.put(key + "/meta", identity, pd.DataFrame({"meta_json": [json.dumps(meta, default=_json_default)]}))
    XCACHE.put(key, identity, df)
    back = XCACHE.get(key, identity)
    assert back is not None and list(back.columns) == XV4_BOOT_RAW_COLUMNS and len(back) == len(df), \
        f"{key}: cache write"
    for c in ("J_given", "J_term", "p_given", "p_term"):
        assert np.allclose(back[c].to_numpy(dtype=float), df[c].to_numpy(dtype=float), rtol=1e-14, atol=0.0,
                           equal_nan=True), f"{key}: cache round trip changed column {c}"
    for c in ("variant", "policy", "status_given", "status_term", "replicate"):
        assert (back[c].astype(str).to_numpy() == df[c].astype(str).to_numpy()).all(), f"{key}: cache round trip {c}"
    del inst, records, pis, lifted, rows, df, smap, hosp, B, true_env
    gc.collect()
    return x_v4_boot_finish(spec, draw, back, meta)


def x_v4_boot_selftest(spec: dict) -> Dict[str, Any]:
    """Cheap asserted checks on draw 0 of `spec`: a resampled data set is a valid hospital data set with the right
    multiplicities, a failing construction is caught and reported, and a failed replicate gives 'failed' rows."""
    inst = x_v4_instance(spec, 0)
    true_env, _, hosp, _ = x_v4_rebuild_benchmark(spec, 0, inst)
    del inst
    ep = np.asarray(hosp["episode_id"], dtype=np.int64)
    n_ep = int(ep.max()) + 1
    mult = x_boot_weights(n_ep, np_rng("xenv", "v4", "boot-selftest", str(spec["cell"])))
    hb = x_v4_boot_dataset(hosp, mult)
    val = x_validate_dataset(hb, true_env)
    S, A = true_env.S, true_env.A
    want = np.bincount(hosp["state"] * A + hosp["action"], weights=mult[ep], minlength=S * A)
    assert val["episodes"] == n_ep and val["transitions"] == int(mult[ep].sum()), (val["episodes"], n_ep)
    assert np.array_equal(np.bincount(hb["state"] * A + hb["action"], minlength=S * A), want), "pair multiplicities"
    T = int(hosp["timestep"].max()) + 1
    assert np.array_equal(np.bincount(hb["timestep"], minlength=T),
                          np.bincount(hosp["timestep"], weights=mult[ep], minlength=T)), "timestep multiplicities"
    Bf, err = x_v4_boot_construct(true_env, hosp, 10 ** 9, "v4-boot-selftest", empty_rule="icu")
    assert Bf is None and err, "a construction without any supported pair must fail and be caught"
    fr = _x_v4_boot_rows("reestimated", 0, ["p"], [b"k"], None)[0]
    assert fr["status_given"] == fr["status_term"] == "failed" and np.isnan(fr["J_given"]), fr
    # the checks and intervals of a draw with a failed replicate (toy rows: replicate 0 of 'reestimated' failed)
    ok = {"status": "ok", "return": 0.25, "prob_any_violation": 0.0}
    one = {b"k": (ok, ok)}
    toy = pd.DataFrame([r for v, b, e in (("point", XV4_BOOT_POINT, one), ("truth", XV4_BOOT_POINT, one),
                                          ("fixed", 0, one), ("fixed", 1, one), ("reestimated", 0, None),
                                          ("reestimated", 1, one))
                        for r in _x_v4_boot_rows(v, b, ["p"], [b"k"], e)], columns=XV4_BOOT_RAW_COLUMNS)
    chk = x_v4_boot_check_rows(toy, 1.0, (2, XV4_BOOT_VARIANTS))
    tiv = x_v4_boot_intervals(toy, 1.0, XV4CFG.boot_quantiles).set_index("boot_variant")
    assert chk["failed_replicates"] == 1 and chk["nonok_rows"] == 0, chk
    assert (tiv.loc["reestimated", "n_failed"], tiv.loc["reestimated", "n_proper"]) == (1, 1), tiv
    assert np.isclose(tiv.loc["reestimated", "lo_nhb"], 0.25) and tiv.loc["fixed", "n_failed"] == 0, tiv
    return {"cell": spec["cell"], "episodes": n_ep, "resampled_transitions": int(val["transitions"]),
            "hospital_transitions": int(len(ep)), "failure_path_message": err}


# Objects whose source changes what a draw computes or stores (the checks and intervals are re-run on cached rows).
XV4_BOOT_CODE = [x_v4_boot_dataset, x_v4_boot_construct, _x_v4_boot_eval, _x_v4_boot_rows,
                 x_v4_boot_construction_stats, _x_v4_boot_dist, x_v4_boot_unit_check, x_v4_boot_identity,
                 run_v4_boot_draw, _x_v3_eval, x_boot_step_map, x_boot_weights, x_boot_model]
XV4_BOOT_CODE_HASH = obj_hash({"common": XV4_COMMON_CODE_HASH, "code": code_hash(*XV4_BOOT_CODE),
                               "constants": {"point": XV4_BOOT_POINT, "statuses": list(XV4_BOOT_STATUSES),
                                             "variants": list(XV4_BOOT_VARIANTS), "columns": XV4_BOOT_RAW_COLUMNS}})

XV4_BOOT_IV = pd.DataFrame(columns=XV4_BOOT_IV_COLUMNS)
XV4_BOOT_META = pd.DataFrame()
XV4_BOOT_CELLS: List[dict] = []
XV4_BOOT_SUMMARY: Dict[str, Any] = {"status": "not run (xenv_v4_bootstrap off)", "code_hash": XV4_BOOT_CODE_HASH,
                                    "draws_complete": None}
if EXPERIMENTS["xenv_v4_bootstrap"]:
    _t_boot = time.perf_counter()
    XV4_BOOT_CELLS, _xv4b_skipped = x_v4_cells(XV4CFG.boot_cells)
    _xv4b_skipped += [s["cell"] for s in XV4_BOOT_CELLS if s.get("construction", "best") != "best"]
    XV4_BOOT_CELLS = [s for s in XV4_BOOT_CELLS if s.get("construction", "best") == "best"]
    if _xv4b_skipped:
        print("NOTE: II.9d bootstrap cells not enabled (family switch off or not in the design), skipped:",
              _xv4b_skipped)
    if XV4_BOOT_CELLS:
        _XV4_BOOT_V3REF, _xv4b_v3src = x_v4_boot_v3_reference([s["cell"] for s in XV4_BOOT_CELLS])
        print(f"II.9d: II.8g rows for the comparison: {_xv4b_v3src} ({len(_XV4_BOOT_V3REF)} draws)")
        _xv4b_self = x_v4_boot_selftest(XV4_BOOT_CELLS[0])
        print(f"II.9d bootstrap: B = {XV4CFG.boot_replicates}, variants {list(XV4CFG.boot_variants)}")
        _xv4b_frames, XV4_BOOT_META, _xv4b_run = x_v4_run_draws(XV4_BOOT_CELLS, run_v4_boot_draw, "II.9d bootstrap")
        XV4_BOOT_IV = (pd.concat(_xv4b_frames, ignore_index=True) if _xv4b_frames
                       else pd.DataFrame(columns=XV4_BOOT_IV_COLUMNS))
        _xv4b_counts = {s["cell"]: int((XV4_BOOT_META["cell"] == s["cell"]).sum()) if len(XV4_BOOT_META) else 0
                        for s in XV4_BOOT_CELLS}
        XV4_BOOT_SUMMARY = {"status": "run", "code_hash": XV4_BOOT_CODE_HASH, **_xv4b_run,
                            "run_elapsed_s": float(time.perf_counter() - _t_boot), "cells_skipped": _xv4b_skipped,
                            "replicates": int(XV4CFG.boot_replicates), "variants": list(XV4CFG.boot_variants),
                            "quantiles": list(XV4CFG.boot_quantiles), "selftest": _xv4b_self,
                            "v3_reference": _xv4b_v3src, "draws_per_cell_present": _xv4b_counts,
                            "draws_complete": all(_xv4b_counts[s["cell"]] == x_v4_draws(s) for s in XV4_BOOT_CELLS)}
        del _xv4b_frames, _xv4b_run, _xv4b_counts, _xv4b_self, _xv4b_v3src
    else:
        XV4_BOOT_SUMMARY = {"status": "not run: no configured II.9d bootstrap cell is enabled",
                            "cells_skipped": _xv4b_skipped, "draws_complete": None, "code_hash": XV4_BOOT_CODE_HASH}
    del _xv4b_skipped, _t_boot
    gc.collect()
print("II.9d bootstrap:", XV4_BOOT_SUMMARY["status"], "|", len(XV4_BOOT_IV), "interval rows | code hash",
      XV4_BOOT_CODE_HASH[:16])

# %% [markdown]
# ### II.9d Analyses: coverage over all default policies, certificates, family-wise errors, matched baselines
#
# The analyses use the interval table of the run above (`XV4_BOOT_IV`). For the comparison with II.8g, the II.8g rows
# of the same draws (B = 10, fixed mask; `XV3_BOOT_RESULTS` or its saved table) are turned into intervals with the
# **same definitions** (variant `v3boot`; proper replicates = `ok` under both rules) and added to the table. A policy
# enters `v3boot` only if its II.8g replicate −1 values (J_bench_term, J_bench_mean, J_true) equal this run's to 1e-9:
# otherwise II.8g regenerated another policy (a near-tie broken differently on another machine), and the policy is
# left out and counted (`v3boot_policies` in the summary). Variants:
# `point` (the point benchmark), `v3boot` (II.8g, where available), `fixed` and `reestimated` (B = 200); interval
# types `nhb`, `free` and `minmax` (the `point` variant has no `minmax`). Policies whose interval is undefined (no
# proper replicate) are counted and left out of that interval's statistics.
#
# * **Coverage** of J_true: lo − 1e-9 ≤ J_true ≤ up + 1e-9, over **all** default policies, and separately over the
#   policies with p > 1e-12 and p = 0 on the point benchmark; lower / upper end holding; mean width.
# * **Certificates** of an interval type: i is certified above j if lo_i > up_j + 1e-9. A claim is an error if
#   J_true(i) ≤ J_true(j) + 1e-9 (ties are errors) and a reversal if J_true(i) < J_true(j) − 1e-9. Shares are over
#   unordered pairs, overall and over completion-sensitive pairs (at least one policy with p > 0 on the point
#   benchmark: types AU + UU of II.9b).
# * **Family-wise errors per draw:** draws with at least one wrong certificate, and draws in which every policy is
#   covered.
# * **Matched-coverage margin baseline:** per draw, the k largest gaps of the point J_mean, J_term and midpoint values
#   (`x_v4_margin_claims`), with k = the number of certified pairs of that interval type in that draw.
# * Pools: per cell, per family (`pooled:gumbel`, `pooled:garnet`) and `pooled:all`.
#
# Coverage and error rates are results: reported, never asserted. The summary also reports the invariant slacks of the
# run, the unit-weight checks, the true-rows bracket count, the failed re-estimated replicates (count, draws, most
# frequent messages), the variability of the re-estimated construction, and the agreement of the fixed replicates with
# II.8g. If `checks.passes` is false (a theorem count or identity failed), the cell raises after writing its files.
#
# | Output | Content |
# |---|---|
# | `XV4_BOOT_IV` · `xenv/v4_boot_intervals.csv.gz` | one row per draw × default policy × variant: truth, point values, p, replicate counts, the three intervals |
# | `XV4_BOOT_DRAWS` · `xenv/v4_boot_draws.csv` | one row per draw × variant × interval type: coverage counts, certified pairs and errors (all and completion-sensitive), family-wise flags, margin baselines |
# | `XV4_BOOT_COVERAGE` · `xenv/v4_boot_coverage.csv` | per cell and pooled × variant × interval type: coverage (all, p > 0, p = 0), widths, certified share and error rate, draws with any wrong certificate / all covered, margin baselines |
# | `XV4_BOOT_META` · `xenv/v4_boot_meta.csv` | per draw meta of the run (flattened), timings |
# | `XV4_BOOT_SUMMARY` · `xenv/v4_boot_summary.json` | status, code hash, checks, timing, draws complete, failures, II.8g agreement, headline numbers, outputs |
#
# **Expected cost on Colab:** under a minute (pure pandas over at most a few hundred draws).

# %%
# ============================================================
# II.9d Bootstrap analyses
# ============================================================
EXECLOG.mark("II.9 Second-revision experiments: bootstrap analyses")


def x_v4_boot_pair_stats(lo: np.ndarray, up: np.ndarray, p: np.ndarray, jt: np.ndarray, pm: np.ndarray,
                         pt: np.ndarray) -> Dict[str, Any]:
    """Coverage, certified comparisons lo_i > up_j and the matched-coverage margin baselines (k = certified pairs,
    on the point J_mean pm / J_term pt / midpoint) of one interval type over the policies of one draw."""
    lo, up, p, jt, pm, pt = (np.asarray(x, dtype=float) for x in (lo, up, p, jt, pm, pt))
    n = len(lo)
    off = ~np.eye(n, dtype=bool)
    u = p > XV4_PTOL
    cs = off & (u[:, None] | u[None, :])
    C = _x_v4_gt(lo, up) & off
    ordered = _x_v4_gt(jt, jt)
    wrong, rev = ~ordered, ordered.T
    lo_ok, up_ok = jt >= lo - XV4_TOL, jt <= up + XV4_TOL
    cov = lo_ok & up_ok
    k = int(C.sum())
    out = {"policies": n, "pairs": n * (n - 1) // 2, "pairs_cs": int(cs.sum() // 2), "certified": k,
           "certified_cs": int((C & cs).sum()), "errors": int((C & wrong).sum()),
           "errors_cs": int((C & cs & wrong).sum()), "reversals": int((C & rev).sum()), "covered": int(cov.sum()),
           "lower_fail": int((~lo_ok).sum()), "upper_fail": int((~up_ok).sum()), "n_ppos": int(u.sum()),
           "covered_ppos": int((cov & u).sum()), "n_pzero": int((~u).sum()), "covered_pzero": int((cov & ~u).sum()),
           "width_sum": float((up - lo).sum()), "width_sum_ppos": float((up - lo)[u].sum()),
           "any_wrong": bool((C & wrong).any()), "all_covered": bool(cov.all())}
    for nm, v in (("mean", pm), ("term", pt), ("mid", (pm + pt) / 2.0)):
        M = x_v4_margin_claims(v, k)
        out[f"margin_{nm}_claims"], out[f"margin_{nm}_errors"] = int(M.sum()), int((M & wrong).sum())
        out[f"margin_{nm}_any_wrong"] = bool((M & wrong).any())
    return out


def x_v4_boot_draw_stats(iv: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, draw, variant, interval type) with x_v4_boot_pair_stats over the defined intervals."""
    rows = []
    for (c, dr, var), d in iv.groupby(["cell", "draw", "boot_variant"], sort=False):
        for it in XV4_BOOT_INTERVALS:
            lc, uc = f"lo_{it}", f"up_{it}"
            ok = d[[lc, uc, "true", "p_bench", "point_mean", "point_term"]].notna().all(axis=1)
            if not ok.any():
                continue
            dd = d[ok]
            rows.append({"cell": c, "family": d["family"].iloc[0], "draw": int(dr), "boot_variant": var,
                         "interval": it, "policies_without_interval": int((~ok).sum()),
                         "median_proper_replicates": float(dd["n_proper"].median()),
                         **x_v4_boot_pair_stats(*(dd[col].to_numpy(float) for col in
                                                  (lc, uc, "p_bench", "true", "point_mean", "point_term")))})
    return pd.DataFrame(rows)


def x_v4_boot_coverage(D: pd.DataFrame, cells: Sequence[str]) -> pd.DataFrame:
    """Per cell, per family pool and pooled, x variant x interval type: sums of the draw statistics."""
    if D.empty:
        return pd.DataFrame()
    fams = sorted(set(D["family"]))
    units = [(c, [c]) for c in cells if c in set(D["cell"])]
    units += [(f"pooled:{f}", sorted(set(D.loc[D["family"] == f, "cell"]))) for f in fams]
    if len(fams) > 1:
        units.append(("pooled:all", sorted(set(D["cell"]))))
    rows = []
    for name, cs in units:
        du = D[D["cell"].isin(cs)]
        for var in XV4_BOOT_VARIANT_ORDER:
            for it in XV4_BOOT_INTERVALS:
                d = du[(du["boot_variant"] == var) & (du["interval"] == it)]
                if d.empty:
                    continue
                s = d.drop(columns=["cell", "family", "draw", "boot_variant", "interval"]).sum(numeric_only=True)
                nd, aw, ac = len(d), int(d["any_wrong"].astype(bool).sum()), int(d["all_covered"].astype(bool).sum())
                r = {"cell": name, "n_cells": len(cs), "boot_variant": var, "interval": it, "draws": nd,
                     "policy_rows": int(s["policies"]),
                     "policies_without_interval": int(s["policies_without_interval"]),
                     "n_ppos": int(s["n_ppos"]), "n_pzero": int(s["n_pzero"]),
                     "coverage_all": _xv3_rate(s["covered"], s["policies"]),
                     "coverage_ppos": _xv3_rate(s["covered_ppos"], s["n_ppos"]),
                     "coverage_pzero": _xv3_rate(s["covered_pzero"], s["n_pzero"]),
                     "lower_holds": _xv3_rate(s["policies"] - s["lower_fail"], s["policies"]),
                     "upper_holds": _xv3_rate(s["policies"] - s["upper_fail"], s["policies"]),
                     "mean_width": _xv3_rate(s["width_sum"], s["policies"]),
                     "mean_width_ppos": _xv3_rate(s["width_sum_ppos"], s["n_ppos"]),
                     "pairs": int(s["pairs"]), "certified": int(s["certified"]),
                     "certified_share": _xv3_rate(s["certified"], s["pairs"]), "pairs_cs": int(s["pairs_cs"]),
                     "certified_cs": int(s["certified_cs"]),
                     "certified_share_cs": _xv3_rate(s["certified_cs"], s["pairs_cs"]),
                     "errors": int(s["errors"]), "error_rate": _xv3_rate(s["errors"], s["certified"]),
                     "errors_cs": int(s["errors_cs"]), "error_rate_cs": _xv3_rate(s["errors_cs"], s["certified_cs"]),
                     "reversals": int(s["reversals"]), "draws_any_wrong": aw,
                     "draws_any_wrong_share": _xv3_rate(aw, nd),
                     "draws_all_covered": ac, "draws_all_covered_share": _xv3_rate(ac, nd)}
                for nm in ("mean", "term", "mid"):
                    mw = int(d[f"margin_{nm}_any_wrong"].astype(bool).sum())
                    r.update({f"margin_{nm}_claims": int(s[f"margin_{nm}_claims"]),
                              f"margin_{nm}_errors": int(s[f"margin_{nm}_errors"]),
                              f"margin_{nm}_error_rate": _xv3_rate(s[f"margin_{nm}_errors"], s[f"margin_{nm}_claims"]),
                              f"margin_{nm}_draws_any_wrong": mw})
                rows.append(r)
    return pd.DataFrame(rows)


def x_v4_boot_v3_intervals(iv: pd.DataFrame, meta: pd.DataFrame, ref: Dict[Tuple[str, int], pd.DataFrame]
                           ) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """(II.8g intervals, counts): the fixed-mask B = XV3BOOT.replicates intervals of the draws in `iv`, with this
    subsection's definitions (variant 'v3boot'). The policy descriptions, the truth, the point values and p are those
    of this run's point rows of the same draw (so every variant of a draw is scored against identical values).

    A policy is kept only if its II.8g replicate -1 values (J_bench_term, J_bench_mean, J_true) equal this run's to
    XV4_TOL, i.e. II.8g regenerated the same policy: a near-tie broken differently on another machine (II.8g's
    point check) gives another policy, whose interval must not be scored against this run's truth. The other
    policies, and policies present on one side only, are left out and counted."""
    rep = {"draws": 0, "policies": 0, "policies_value_mismatch": 0, "policies_missing_in_v3": 0,
           "policies_not_in_v4": 0}
    if iv.empty or not ref or meta.empty:
        return pd.DataFrame(columns=XV4_BOOT_IV_COLUMNS), rep
    vr = {(str(c), int(d)): float(v) for c, d, v in zip(meta["cell"], meta["draw"], meta["v_range"])}
    same = ["true", "p_true", "true_rows_term", "point_term", "point_mean", "p_bench"]
    chk = ["point_term", "point_mean", "true"]
    frames = []
    for (c, dr), g in iv[iv["boot_variant"] == "point"].groupby(["cell", "draw"], sort=False):
        r = ref.get((str(c), int(dr)))
        if r is None or not (r["variant"] == "v3boot").any():
            continue
        one = x_v4_boot_intervals(r, vr[(str(c), int(dr))], XV4CFG.boot_quantiles)
        v3pt, mine = one[one["boot_variant"] == "point"].set_index("policy"), g.set_index("policy")
        both = mine.index.intersection(v3pt.index)
        dv = np.abs(v3pt.loc[both, chk].to_numpy(float) - mine.loc[both, chk].to_numpy(float))
        keep = both[(dv <= XV4_TOL).all(axis=1)]              # NaN (a missing II.8g value) is not kept
        rep["policies_missing_in_v3"] += int(len(mine.index.difference(v3pt.index)))
        rep["policies_not_in_v4"] += int(len(v3pt.index.difference(mine.index)))
        rep["policies_value_mismatch"] += int(len(both) - len(keep))
        one = one[(one["boot_variant"] == "v3boot") & one["policy"].isin(keep)].drop(columns=same).merge(
            g[["policy", "method", "policy_variant", "method_family", *same]], on="policy", how="inner",
            validate="one_to_one")
        if one.empty:
            continue
        one["cell"], one["family"], one["draw"] = c, g["family"].iloc[0], int(dr)
        frames.append(one[XV4_BOOT_IV_COLUMNS])
        rep["draws"] += 1
        rep["policies"] += int(len(one))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=XV4_BOOT_IV_COLUMNS)
    return out, rep


def _x_v4_boot_col(meta: pd.DataFrame, col: str, how: str):
    """max / sum / all of a flattened meta column (None when the column is absent)."""
    if meta.empty or col not in meta.columns:
        return None
    v = meta[col]
    if how == "all":
        return bool(v.astype(bool).all())
    v = pd.to_numeric(v, errors="coerce")
    return float(v.max()) if how == "max" else float(v.sum())


def x_v4_boot_run_checks(meta: pd.DataFrame) -> Dict[str, Any]:
    """Invariant slacks and unit-weight checks over the draws of the run (asserted per draw), and the true-rows bracket
    J_rows_term <= J_true <= J_rows_term + p (v_max - v_min) (Theorem 1 on the true rows; counted per draw, and a
    violation fails `passes`, as II.8i / II.9b count their theorem facts)."""
    out = {"max_rule_p_difference": _x_v4_boot_col(meta, "checks.max_rule_p_difference", "max"),
           "max_negative_width": _x_v4_boot_col(meta, "checks.max_negative_width", "max"),
           "max_width_minus_bound": _x_v4_boot_col(meta, "checks.max_width_minus_bound", "max"),
           "unit_weights_max_abs_diff_P": _x_v4_boot_col(meta, "reproduction.max_abs_diff_P", "max"),
           "unit_weights_max_abs_diff_d0": _x_v4_boot_col(meta, "reproduction.max_abs_diff_d0", "max"),
           "unit_weights_construction_identity": _x_v4_boot_col(meta, "reproduction.construction_identity", "all"),
           "truth_rows_bracket_violations": _x_v4_boot_col(meta, "checks.truth_rows_bracket_violations", "sum")}
    theorem = all((out[k] or 0.0) <= XV4_TOL
                  for k in ("max_rule_p_difference", "max_negative_width", "max_width_minus_bound"))
    unit = all((out[k] or 0.0) <= 1e-14 for k in ("unit_weights_max_abs_diff_P", "unit_weights_max_abs_diff_d0"))
    out["passes"] = (bool(len(meta)) and theorem and unit and out["unit_weights_construction_identity"] is not False
                     and not out["truth_rows_bracket_violations"])
    return out


def x_v4_boot_failures(meta: pd.DataFrame) -> Dict[str, Any]:
    """Failed re-estimated replicates: totals, draws affected, per cell and the most frequent messages."""
    if meta.empty or "reestimated.failed" not in meta.columns:
        return {"total": 0, "draws_with_failure": 0}
    f = pd.to_numeric(meta["reestimated.failed"], errors="coerce").fillna(0).astype(int)
    n = pd.to_numeric(meta["reestimated.replicates"], errors="coerce").fillna(0).astype(int)
    msgs: Counter = Counter()
    unparsed = 0
    for txt in meta.get("failure_messages", pd.Series(dtype=object)).dropna():
        try:      # a JSON list; _x_compact replaces a list longer than its limit by a hashed summary string
            pairs = json.loads(str(txt))
        except json.JSONDecodeError:
            unparsed += 1
            continue
        for m, c in pairs:
            msgs[str(m)] += int(c)
    return {"total": int(f.sum()), "replicates": int(n.sum()), "share": _xv3_rate(int(f.sum()), int(n.sum())),
            "draws_with_failure": int((f > 0).sum()),
            "by_cell": {str(c): int(v) for c, v in f.groupby(meta["cell"]).sum().items()},
            "messages": [[m, int(c)] for m, c in msgs.most_common(10)], "draws_messages_unparsed": unparsed}


def x_v4_boot_construction_summary(meta: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    """Per cell: the mean over draws of the per-draw median of each re-estimated construction statistic."""
    stats_ = ("decision_states_new", "decision_states_lost", "mask_rows_changed_share", "admissible_pairs",
              "dead_states_removed", "trap_states_censored", "fallback_action_changed")
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for c, g in (meta.groupby("cell", sort=False) if len(meta) and "cell" in meta.columns else []):
        out[str(c)] = {s: (float(pd.to_numeric(g[f"reestimated.{s}.median"], errors="coerce").mean())
                           if f"reestimated.{s}.median" in g.columns else None) for s in stats_}
    return out


def x_v4_boot_v3_summary(meta: pd.DataFrame) -> Dict[str, Any]:
    """Agreement of the fixed replicates (and replicate -1) with the II.8g rows, over the compared draws."""
    if meta.empty or "v3_match.status" not in meta.columns:
        return {"draws_compared": 0}
    m = meta[meta["v3_match.status"] == "compared"]
    if m.empty:
        return {"draws_compared": 0}
    return {"draws_compared": int(len(m)),
            "rows": int(pd.to_numeric(m["v3_match.rows"]).sum()),
            "missing_in_v4": int(pd.to_numeric(m["v3_match.missing_in_v4"]).sum()),
            "status_mismatches": int(pd.to_numeric(m["v3_match.status_mismatches"]).sum()),
            "max_abs_diff_return": _x_v4_boot_col(m, "v3_match.max_abs_diff_return", "max"),
            "max_abs_diff_p": _x_v4_boot_col(m, "v3_match.max_abs_diff_p", "max"),
            "draws_matching": int(m["v3_match.match"].astype(bool).sum()), "tolerance": XV4_TOL}


def x_v4_boot_headline(cov: pd.DataFrame) -> Dict[str, Any]:
    """Headline numbers of the widest pool, per variant x interval type."""
    if cov.empty:
        return {}
    pool = next((p for p in ("pooled:all", "pooled:gumbel", "pooled:garnet") if (cov["cell"] == p).any()),
                str(cov["cell"].iloc[0]))
    keep = ("coverage_all", "coverage_ppos", "coverage_pzero", "certified_share", "certified_share_cs", "error_rate",
            "error_rate_cs", "draws_any_wrong_share", "draws_all_covered_share", "margin_mid_error_rate")
    return {"pool": pool, **{f"{r['boot_variant']}|{r['interval']}": {k: r[k] for k in keep}
                             for _, r in cov[cov["cell"] == pool].iterrows()}}


XV4_BOOT_DRAWS = pd.DataFrame()
XV4_BOOT_COVERAGE = pd.DataFrame()
_xv4b_out = {"intervals": "xenv/v4_boot_intervals.csv.gz", "draws": "xenv/v4_boot_draws.csv",
             "coverage": "xenv/v4_boot_coverage.csv", "meta": "xenv/v4_boot_meta.csv",
             "summary": "xenv/v4_boot_summary.json"}
if XV4_BOOT_SUMMARY.get("status") == "run":
    _t_an = time.perf_counter()
    _xv4b_v3iv, _xv4b_v3rep = x_v4_boot_v3_intervals(XV4_BOOT_IV, XV4_BOOT_META, _XV4_BOOT_V3REF)
    if _xv4b_v3rep["policies_value_mismatch"] or _xv4b_v3rep["policies_missing_in_v3"]:
        print("NOTE: II.8g policies left out of the v3boot intervals (replicate -1 values differ from this run's, or "
              "missing):", _xv4b_v3rep)
    if len(_xv4b_v3iv):
        XV4_BOOT_IV = pd.concat([XV4_BOOT_IV[XV4_BOOT_IV["boot_variant"] != "v3boot"], _xv4b_v3iv], ignore_index=True)
    XV4_BOOT_DRAWS = x_v4_boot_draw_stats(XV4_BOOT_IV)
    XV4_BOOT_COVERAGE = x_v4_boot_coverage(XV4_BOOT_DRAWS, [s["cell"] for s in XV4_BOOT_CELLS])
    XV4_BOOT_IV.to_csv(XENV_TABLE_DIR / "v4_boot_intervals.csv.gz", index=False, compression="gzip")
    XV4_BOOT_DRAWS.to_csv(XENV_TABLE_DIR / "v4_boot_draws.csv", index=False)
    XV4_BOOT_COVERAGE.to_csv(XENV_TABLE_DIR / "v4_boot_coverage.csv", index=False)
    XV4_BOOT_META.to_csv(XENV_TABLE_DIR / "v4_boot_meta.csv", index=False)
    _xv4b_checks = x_v4_boot_run_checks(XV4_BOOT_META)
    XV4_BOOT_SUMMARY.update({
        "checks": _xv4b_checks, "max_rule_p_difference": _xv4b_checks["max_rule_p_difference"],
        "max_width_minus_bound": _xv4b_checks["max_width_minus_bound"],
        "failed_replicates": x_v4_boot_failures(XV4_BOOT_META),
        "construction_variability": x_v4_boot_construction_summary(XV4_BOOT_META),
        "v3_match": x_v4_boot_v3_summary(XV4_BOOT_META),
        "v3boot_draws": int(_xv4b_v3rep["draws"]), "v3boot_policies": _xv4b_v3rep,
        "headline": x_v4_boot_headline(XV4_BOOT_COVERAGE),
        "interval_rows": int(len(XV4_BOOT_IV)), "draw_rows": int(len(XV4_BOOT_DRAWS)),
        "analysis_elapsed_s": float(time.perf_counter() - _t_an), "outputs": _xv4b_out})
    XV4_BOOT_SUMMARY["elapsed_s"] = float(XV4_BOOT_SUMMARY["run_elapsed_s"] + XV4_BOOT_SUMMARY["analysis_elapsed_s"])
    if XV4_BOOT_SUMMARY["v3_match"].get("draws_compared") and \
            XV4_BOOT_SUMMARY["v3_match"].get("draws_matching") != XV4_BOOT_SUMMARY["v3_match"]["draws_compared"]:
        print("NOTE: fixed-mask replicates differ from the II.8g rows on some draws (reported, not asserted):",
              {k: XV4_BOOT_SUMMARY["v3_match"][k] for k in ("status_mismatches", "max_abs_diff_return")})
    if len(XV4_BOOT_COVERAGE):
        display(XV4_BOOT_COVERAGE[XV4_BOOT_COVERAGE["cell"].str.startswith("pooled:")][
            ["cell", "boot_variant", "interval", "draws", "coverage_all", "coverage_ppos", "coverage_pzero",
             "certified_share", "certified_share_cs", "error_rate", "draws_any_wrong_share",
             "draws_all_covered_share"]].round(4))
    del _t_an, _xv4b_v3iv, _xv4b_v3rep, _xv4b_checks
else:
    XV4_BOOT_SUMMARY.setdefault("checks", {})
    XV4_BOOT_SUMMARY.setdefault("elapsed_s", 0.0)
save_json(XV4_BOOT_SUMMARY, XENV_TABLE_DIR / "v4_boot_summary.json")
print(json.dumps({k: v for k, v in XV4_BOOT_SUMMARY.items()
                  if k not in ("draws_per_cell", "construction_variability", "headline")}, indent=1,
                 default=_json_default))
del _xv4b_out
gc.collect()
if XV4_BOOT_SUMMARY.get("checks", {}).get("passes") is False:
    raise AssertionError(f"II.9d checks failed: {XV4_BOOT_SUMMARY['checks']} (xenv/v4_boot_summary.json)")
