# %% [markdown]
# ### II.9c Confidence-set certificates on the true state space
#
# The certificates of Corollary 2 hold for the **given** supported rows, but a benchmark's rows are estimated from
# hospital data. II.8g measured this with a bootstrap. This section gives the frequentist alternative: an L1
# confidence set for every true row and for d0, built from the raw hospital counts on the **true** state space, and
# bounds on J_true that hold for every policy at once on one event E. At scale 1 the union bound gives
# P(E) ≥ 1 − δ (δ = `XV4CFG.cs_delta`), so with probability at least 1 − δ **no** NHB-free claim of a draw is
# wrong (family-wise), and no NHB claim is wrong when NHB also holds. Cells: `XV4CFG.cs_cells` through
# `x_v4_cells` (II.6 and II.8 `best` cells; no aggregated cells).
#
# **Per (cell, draw).**
# 1. **Policies and truth.** `x_v4_instance`, `x_v4_default_records` (the II.6 default policies) and
#    `x_v4_rebuild_benchmark` (hospital episodes and benchmark `B`, asserted equal to the instance's). True-space
#    policies are π↑ = `B["lift"](π)`; NHB policies are proj(π)↑ = `B["lift"](x_project(π, M, live))` (uniform
#    W, Proposition 3). J_true is exact on the `true` target (T_masked, semantics `given`, its lift), as in
#    II.8g; the point-benchmark values J_bench_mean, J_bench_term and p_bench are exact on the benchmark.
# 2. **Counts** on the true state space (`x_cs_counts`): raw counts C(s, a, s′) of the informative steps of
#    **every** logged live state, N(s, a) = Σ_s′ C, q̂ = C / N. Gumbel has a known censoring hazard h and timeout
#    state t, so the informative steps are those with s′ ≠ t; Garnet has no hazard and every step is
#    informative. No dead-state removal, no trap censoring, no redirect: this is the true system. The empirical
#    d0 comes from the first step of each of the n_ep episodes.
# 3. **Structure of a true row.** Gumbel: P*(·|s, a) = h·δ_t + (1 − h)·q(·|s, a) with q a distribution on the
#    states other than t (asserted on the simulator). Garnet: P*(·|s, a) = q(·|s, a) on all states.
# 4. **Radii** (`x_cs_radii`; Weissman et al. 2003: P(‖q̂_n − q‖₁ ≥ ε) ≤ (2^m − 2)·exp(−n ε²/2)). The k-th
#    informative step of a pair is an independent draw from q whatever the history, so a bound for each fixed n
#    with δ_rows / (K n (n + 1)) and a union over n ≥ 1 and the K = (true live states) × A pairs gives, with
#    δ_rows = δ_d0 = δ/2 and m the number of outcomes of q (S − 1 on Gumbel, S on Garnet),
#    r(s, a) = scale·√((2/N)(m ln 2 + ln(K N (N + 1) / δ_rows))) for N ≥ 1, and
#    r0 = scale·√((2/n_ep)(m0 ln 2 + ln(1/δ_d0))) for d0 (m0 = true live states; the initial states are
#    i.i.d.). r = ∞ (the whole simplex) when N = 0 or r ≥ 2. Only scale = 1 is covered by the union bound; the
#    smaller scales of `XV4CFG.cs_scales` are **heuristic** (`set_kind` = `heuristic` in every output).
# 5. **Inner problems** (`x_cs_l1_extreme`, exact greedy, vectorised over rows). min q·U over
#    B₁(q̂, r) ∩ Δ(support) moves δ = min(r/2, 1 − q̂[i*]) onto i* = argmin_support U and removes it from the
#    support states with the highest U first; max is symmetric; r = ∞ puts all mass on the extreme support state.
#    A self-test compares it with `scipy.optimize.linprog` (both senses, shared and per-row U and supports, tied
#    values, every kind of radius).
# 6. **Monotone robust iteration** (`x_cs_robust_vector`) for a fixed true-space policy π, over the pairs with
#    π(a|s) > 0 only: L_0 = v_min on live states (absorbing states hold their entry reward R),
#    L_{k+1}(s) = Σ_a π(a|s)[h·R_t + (1 − h)·min_q q·(R + L_k)] on Gumbel, without the h term on Garnet;
#    U_0 = v_max with max for the upper vector. On E the true P is a member, its Bellman operator is monotone
#    and v_min ≤ V_true ≤ v_max, so **every** iterate satisfies L_k ≤ V_true ≤ U_k. The iteration stops at a
#    sup-norm change ≤ `cs_tol` or after `cs_max_iter` steps (a cap only loosens the bound); iterations and the
#    converged flag are recorded. Unlogged rows (also outside the decision states V) are unrestricted.
# 7. **Ends and tests.** L(π) = min_{d∈D} d·L_vec(π↑), U_free(π) = max_{d∈D} d·U_vec(π↑),
#    U_proj(π) = max_{d∈D} d·U_vec(proj(π)↑), U_nhb(π) = min(U_free(π), U_proj(π)), D the d0 ball. Under NHB,
#    J_true(π↑) ≤ J_true(proj(π)↑) (the mean rule on the true rows, Proposition 3), so U_nhb is an upper bound
#    on E under NHB; a policy with L > U_proj (both in the policy table) is evidence against NHB for it on E.
#    Tests of an ordered pair (i, j): `free`: L(πi) > U_free(πj); `nhb`: L(πi) > U_nhb(πj); `paired_free`:
#    min_{d∈D} d·(L_vec(πi) − U_vec(πj↑)) > 0; `paired_nhb`: the same with the state-wise minimum of the two
#    upper vectors of πj (both are upper bounds on V_true under NHB). Margins use the tolerance `XV4_TOL`; a
#    claim is wrong if J_true(i) ≤ J_true(j) + `XV4_TOL` (ties are errors). The point-benchmark certificates
#    (`point_nhb`: J_term(i) > J_mean(j), `point_free`: J_term(i) > J_term(j) + p_j·v_range) on the same
#    policies are the reference; their counts are checked against `x_v4_pair_claims`.
# 8. State vectors are memoised by policy within a draw and scale (distinct policies are few) and kept in memory
#    only.
#
# **What to expect.** The sets are honest but wide. At scale 1 a Gumbel row has m ln 2 ≈ 421 (m = 608 outcomes),
# so it stays unrestricted below about 220 informative steps, and r ≈ 0.95 at N = 1 000 and 0.30 at N = 10 000
# (Garnet, m = 152: below about 60 steps; r ≈ 0.51 at N = 1 000). The d0 radius is r0 ≈ 0.65 for 2 000 and 0.29
# for 10 000 Gumbel episodes (0.21 for 5 000 Garnet episodes). Few certificates are therefore expected at the
# valid scale; the heuristic scales show how fast they appear as the sets shrink.
#
# **Invariants asserted per draw:** every raw robust step is monotone (the lower iterates non-decreasing, the
# upper non-increasing; slack 1e-12 plus the row-sum rounding of π); L ≤ U_free and U_nhb ≤ U_free for every
# policy; the paired tests are at least as strong as the unpaired ones (min_D d·(a − b) ≥ min_D d·a − max_D d·b);
# all values lie in [v_min, v_max]; the point claim counts equal `x_v4_pair_claims`. L ≤ J_true ≤ U_free is
# **not** asserted: it holds only on E. Coverage and error rates are reported, never asserted.
#
# | Output | Content |
# |---|---|
# | `xenv/v4_cs_policies.csv.gz` · `XV4_CS_POLICIES` | per draw × default policy × scale: L, U_free, U_proj, U_nhb, J_true, point values, p_bench, iterations, convergence, rows used |
# | `xenv/v4_cs_draws.csv` · `XV4_CS_DRAWS` | per draw × scale × test (four set tests + two point tests): pairs, claims and errors (all, AA, AU, UU, completion-sensitive), coverage of the test's interval, any-wrong / all-covered flags, radius statistics (median r over the used pairs, share unrestricted, r0) |
# | `xenv/v4_cs_summary.csv` · `XV4_CS_TABLE` | per cell, family pool and pooled × scale × test: claims, share, errors, error rate (also on completion-sensitive pairs), draws with a wrong claim / an uncovered policy, coverage, mean width |
# | `xenv/v4_cs_summary.json` · `XV4_CS_SUMMARY` | status, code hash, configuration, self-test, invariant checks, headline numbers, timing |
# | `XV4_CS_META` (memory only) | per draw meta (flattened): counts, radii, iterations, slacks, timings (`timing.total_s` enters the compute totals of the release) |
#
# Results are cached under `v4cs/<cell>/draw<k>` (and `/meta`); the identity is the draw identity
# (`x_v4_draw_identity`), this section's code hash, δ, the scales, `cs_tol` and `cs_max_iter`.
#
# **Expected cost on Colab** (an estimate: local timings of the robust iteration on Gumbel- and Garnet-sized
# synthetic count models, scaled by the Colab / local ratio of II.8f). Each draw rebuilds the instance and runs one
# `run_xmethods` fit, as an II.8g draw does (about 5–10 s for Gumbel, 3–4 s for Garnet), plus three exact
# evaluations per distinct default policy. One robust iteration takes about 0.4 ms for a deterministic Gumbel
# policy (606 used rows of 609 states) and 3 ms for a fully stochastic one (4 848 rows); Garnet (152 states, ≤ 6
# successors per row) about 0.1–0.3 ms. The adversary moves mass onto death (lower) or discharge / goal (upper), so
# the iterations stopped after 3–25 steps in those tests; on Gumbel the timeout hazard caps them at about 360
# (contraction 1 − h, h = 0.05). With 50–100 robust vectors per draw and scale this adds a few seconds per draw
# (under a minute in the worst case). In total roughly 10–20 s per Gumbel draw and 5 s per Garnet draw, about
# 30–60 min for 3 Gumbel and 4 Garnet cells × 30 draws; a cached draw reloads in about 3 s (instance rebuild).

# %%
# ============================================================
# II.9c Confidence-set certificates: definitions
# ============================================================
EXECLOG.mark("II.9c Confidence-set certificates: definitions")

XV4_CS_TESTS = ("free", "nhb", "paired_free", "paired_nhb")      # certificates from the confidence set
XV4_CS_POINT_TESTS = ("point_nhb", "point_free")                # point-benchmark reference (Corollary 2)
XV4_CS_INTERVALS = {"free": "[L, U_free]", "nhb": "[L, U_nhb]", "paired_free": "[L, U_free]",
                    "paired_nhb": "[L, U_nhb]", "point_nhb": "[J_bench_term, J_bench_mean]",
                    "point_free": "[J_bench_term, J_bench_term + p_bench v_range]"}
XV4_CS_MONO_TOL = 1e-12      # slack of the monotone-step assertion (plus the row-sum rounding of the policy)
XV4_CS_PAIR_TOL = 1e-10      # slack of 'paired >= unpaired' (both sides are sums of ~600 rounded products)
XV4_CS_COLUMNS = ["cell", "family", "draw", "policy", "method", "variant", "hp", "method_family", "scale", "set_kind",
                  "J_true", "J_bench_mean", "J_bench_term", "p_bench", "L", "U_free", "U_proj", "U_nhb",
                  "proj_is_same", "covered_free", "covered_nhb", "iter_lower", "iter_upper", "iter_upper_proj",
                  "converged_lower", "converged_upper", "converged_upper_proj", "rows_used", "rows_unrestricted"]
XV4_CS_FLOAT_COLUMNS = ["scale", "J_true", "J_bench_mean", "J_bench_term", "p_bench", "L", "U_free", "U_proj",
                        "U_nhb"]


# ---------- the inner problem: extreme of q.U over an L1 ball intersected with a simplex ----------
def x_cs_pack(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Packed rows of a non-negative [n, S] matrix: (vals [n, m], cols [n, m]) hold the positive entries of each row
    in column order, padded with zeros (column 0); m = max(1, largest row support)."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    assert q.ndim == 2 and (q >= 0).all(), "rows must be non-negative"
    nz = q > 0
    cnt = nz.sum(axis=1)
    m = max(1, int(cnt.max(initial=0)))
    rows, cols = np.nonzero(nz)
    pos = np.arange(len(rows)) - (np.cumsum(cnt) - cnt)[rows]
    vals_p, cols_p = np.zeros((q.shape[0], m)), np.zeros((q.shape[0], m), dtype=np.int64)
    vals_p[rows, pos] = q[rows, cols]
    cols_p[rows, pos] = cols
    return vals_p, cols_p


def _x_cs_extreme_packed(vals: np.ndarray, cols: np.ndarray, r: np.ndarray, U: np.ndarray, support: np.ndarray,
                         sense: str, check: bool = True) -> np.ndarray:
    """min ('min') or max ('max') of q.U over {q in simplex(support) : ||q - qhat||_1 <= r} for every packed row.

    vals / cols: [n, m] (or [1, m], shared by all rows) packed qhat (x_cs_pack), inside the support; r: [n] radii
    (inf = unrestricted); U: [S] (shared) or [n, S]; support: [S] or [n, S] boolean. Exact greedy: move
    delta = min(r/2, 1 - qhat[i*]) onto i*, the support state with the lowest U (highest for 'max'), and remove it
    from the other states with the highest U first (lowest for 'max'); r = inf puts all mass on i*."""
    assert sense in ("min", "max"), sense
    sign = 1.0 if sense == "min" else -1.0
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    n = len(r)
    vals = np.broadcast_to(np.asarray(vals, dtype=np.float64), (n, np.shape(vals)[-1]))
    cols = np.broadcast_to(np.asarray(cols, dtype=np.int64), (n, np.shape(cols)[-1]))
    W = sign * np.asarray(U, dtype=np.float64)
    sup = np.asarray(support, dtype=bool)
    if W.ndim == 1 and sup.ndim == 1:                      # one value vector and one support for every row
        Wm = np.where(sup, W, np.inf)
        istar = int(Wm.argmin())
        wstar = np.full(n, Wm[istar])
        Wc, hit = W[cols], cols == istar
        sup_c = sup[cols] if check else None
    else:
        W2 = np.broadcast_to(W, (n, W.shape[-1]))
        sup2 = np.broadcast_to(sup, (n, sup.shape[-1]))
        Wm = np.where(sup2, W2, np.inf)
        istar = Wm.argmin(axis=1)
        wstar = Wm[np.arange(n), istar]
        Wc, hit = np.take_along_axis(W2, cols, axis=1), cols == istar[:, None]
        sup_c = np.take_along_axis(sup2, cols, axis=1) if check else None
    assert np.isfinite(wstar).all(), "every row needs a non-empty support and finite values"
    fin = np.isfinite(r)
    if check:
        assert (r >= 0).all() and (vals >= 0).all(), "radii and qhat must be non-negative"
        assert not ((vals > 0) & ~sup_c).any(), "qhat must lie inside the support"
        assert np.allclose(vals[fin].sum(axis=1), 1.0, atol=1e-9), "a row with a finite radius needs a distribution"
    out = wstar.copy()                                     # unrestricted rows: all mass on i*
    if fin.any():
        f = slice(None) if fin.all() else fin
        v, w, h = vals[f], Wc[f], hit[f]
        qstar = np.where(h, v, 0.0).sum(axis=1)
        delta = np.clip(np.minimum(r[f] / 2.0, 1.0 - qstar), 0.0, None)
        rem = np.where(h, 0.0, v)
        order = np.argsort(-w, axis=1, kind="stable")     # highest (signed) value first
        rem_s, w_s = np.take_along_axis(rem, order, axis=1), np.take_along_axis(w, order, axis=1)
        take = np.clip(delta[:, None] - (np.cumsum(rem_s, axis=1) - rem_s), 0.0, rem_s)
        out[f] = ((rem_s - take) * w_s).sum(axis=1) + (qstar + delta) * wstar[f]
    return sign * out


def x_cs_l1_extreme(qhat: np.ndarray, r, U: np.ndarray, support_mask: np.ndarray, sense: str) -> np.ndarray:
    """min / max of q.U over B_1(qhat, r) ∩ simplex(support_mask), one value per row (vectorised greedy).

    qhat: [S] or [n, S] rows inside the support (a row with r = inf may be all zero: N = 0); r: scalar or [n]
    (inf = unrestricted); U: [S] or [n, S]; support_mask: [S] or [n, S]. A single qhat row is shared by the n rows of U.
    Used for the true rows and for d0."""
    q = np.atleast_2d(np.asarray(qhat, dtype=np.float64))
    U = np.asarray(U, dtype=np.float64)
    n = max(q.shape[0], U.shape[0] if U.ndim == 2 else 1, np.size(r),
            np.shape(support_mask)[0] if np.ndim(support_mask) == 2 else 1)
    assert q.shape[0] in (1, n) and np.size(r) in (1, n), (q.shape, np.shape(r), U.shape)
    vals, cols = x_cs_pack(q)
    rr = np.broadcast_to(np.asarray(r, dtype=np.float64).reshape(-1), (n,))
    return _x_cs_extreme_packed(vals, cols, rr, U, support_mask, sense)


# ---------- counts, radii and the monotone robust iteration ----------
def x_cs_counts(true_env: FiniteEnv, hosp: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Raw hospital counts on the true state space: packed empirical rows qhat of every (s, a) over the informative
    steps (every step without a known hazard; the steps that do not enter the timeout state with one), N(s, a), the
    empirical d0 of the episodes' first states, and the structure of a true row (support of q, hazard)."""
    S, A, live = true_env.S, true_env.A, np.asarray(true_env.live, dtype=bool)
    s, a, ns = (np.asarray(hosp[k], dtype=np.int64) for k in ("state", "action", "next_state"))
    assert len(s) > 0 and live[s].all(), "logged states must be live"
    h = true_env.meta["censoring_hazard"]
    support = np.ones(S, dtype=bool)                       # possible outcomes of q
    t, r_t = None, 0.0
    if h is not None:
        h, t = float(h), int(true_env.meta["timeout_state"])
        assert 0.0 < h < 1.0 and t in true_env.terminals, (h, t)
        assert np.all(true_env.P[live, :, t] == h), "true rows must be h e_timeout + (1 - h) q (II.3b)"
        support[t] = False
        r_t = float(true_env.entry_reward[t])
    inf = np.ones(len(s), dtype=bool) if h is None else ns != t
    key, cnt = np.unique((s[inf] * A + a[inf]) * S + ns[inf], return_counts=True)
    row, col = key // S, key % S
    assert support[col].all(), "an informative step left the support of q"
    N = np.bincount(row, weights=cnt, minlength=S * A).astype(np.float64)
    nnz = np.bincount(row, minlength=S * A)
    m = max(1, int(nnz.max(initial=0)))
    pos = np.arange(len(row)) - (np.cumsum(nnz) - nnz)[row]   # np.unique sorts by (row, col)
    vals, cols = np.zeros((S * A, m)), np.zeros((S * A, m), dtype=np.int64)
    vals[row, pos] = cnt / N[row]
    cols[row, pos] = col
    first = np.asarray(hosp["timestep"]) == 0
    n_ep = int(first.sum())
    assert n_ep == len(np.unique(hosp["episode_id"])), "every episode has exactly one first step"
    d0hat = np.bincount(s[first], minlength=S).astype(np.float64) / n_ep
    lo, hi = true_env.value_bounds
    n_live = int(live.sum())
    return {"S": int(S), "A": int(A), "live": live.copy(), "R": np.asarray(true_env.entry_reward, dtype=np.float64),
            "hazard": h, "timeout": t, "R_timeout": r_t, "support": support, "vals": vals, "cols": cols, "N": N,
            "d0hat": d0hat, "n_ep": n_ep, "v_min": float(lo), "v_max": float(hi), "K": n_live * int(A),
            "m": int(support.sum()), "m0": n_live, "steps": int(len(s)), "informative_steps": int(inf.sum()),
            "max_row_support": int(nnz.max(initial=0))}


def x_cs_radii(model: Dict[str, Any], scale: float, delta: float) -> Tuple[np.ndarray, float]:
    """Radii of the row sets of every (s, a) (Weissman L1 bound, anytime over N, union over the K live pairs, delta/2)
    and of the d0 set (delta/2), times `scale`; inf where N = 0 or where the radius reaches 2 (whole simplex)."""
    assert scale > 0.0 and 0.0 < delta < 1.0, (scale, delta)
    N = np.asarray(model["N"], dtype=np.float64)
    d_rows, d_d0 = delta / 2.0, delta / 2.0
    r = np.full(N.shape, np.inf)
    pos = N >= 1.0
    n = N[pos]
    r[pos] = scale * np.sqrt((2.0 / n) * (model["m"] * math.log(2.0) + np.log(model["K"] * n * (n + 1.0) / d_rows)))
    r[r >= 2.0] = np.inf
    n0 = int(model["n_ep"])
    r0 = (scale * math.sqrt((2.0 / n0) * (model["m0"] * math.log(2.0) + math.log(1.0 / d_d0))) if n0 >= 1
          else math.inf)
    return r, (r0 if r0 < 2.0 else math.inf)


def x_cs_robust_vector(model: Dict[str, Any], r: np.ndarray, pi: np.ndarray, sense: str, tol: float,
                       max_iter: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Monotone robust iteration of one true-space policy over the row sets with radii r (every (s, a)).

    'min': L_0 = v_min on live states, L_{k+1}(s) = sum_a pi(a|s) [h R_t + (1 - h) min_q q.(R + L_k)] (no h term
    without a known hazard); 'max': U_0 = v_max and max. Only the pairs with pi(a|s) > 0 enter. Returns the state
    vector (0 on absorbing states) and the iteration report. Every raw step must be monotone (asserted, slack
    XV4_CS_MONO_TOL plus the row-sum rounding of pi); the returned sequence is the exact monotone envelope."""
    assert sense in ("min", "max"), sense
    S, A, live = model["S"], model["A"], model["live"]
    pi = np.asarray(pi, dtype=np.float64)
    assert pi.shape == (S, A), pi.shape
    ss, aa = np.nonzero((pi > 0.0) & live[:, None])
    assert np.array_equal(np.unique(ss), np.flatnonzero(live)), "every live state needs an action with pi > 0"
    rr = np.asarray(r, dtype=np.float64)[ss * A + aa]
    o = np.argsort(~np.isfinite(rr), kind="stable")        # restricted rows first, then the unrestricted ones
    ss, aa, rr = ss[o], aa[o], rr[o]
    rows, w, nf = ss * A + aa, pi[ss, aa], int(np.isfinite(rr).sum())
    vals, cols = model["vals"][rows[:nf]], model["cols"][rows[:nf]]
    width = max(1, int((vals > 0).sum(axis=1).max(initial=0)))     # packed rows are left-aligned (x_cs_counts)
    vals, cols = vals[:, :width], cols[:, :width]
    pad_v, pad_c = np.zeros((1, 1)), np.zeros((1, 1), dtype=np.int64)  # unrestricted rows need no qhat
    R, sup, h = model["R"], model["support"], model["hazard"]
    _x_cs_extreme_packed(vals, cols, rr[:nf], R, sup, sense, check=True)     # validate the packed rows once
    sign = 1.0 if sense == "min" else -1.0
    V = np.where(live, model["v_min"] if sense == "min" else model["v_max"], 0.0)
    dev = float(np.abs(pi[live].sum(axis=1) - 1.0).max())
    mono_tol = XV4_CS_MONO_TOL + dev * max(abs(model["v_min"]), abs(model["v_max"]))
    worst, change, converged, it = 0.0, math.inf, False, 0
    for it in range(1, int(max_iter) + 1):
        X = R + V
        x = np.concatenate([_x_cs_extreme_packed(vals, cols, rr[:nf], X, sup, sense, check=False),
                            _x_cs_extreme_packed(pad_v, pad_c, rr[nf:], X, sup, sense, check=False)])
        if h is not None:
            x = h * model["R_timeout"] + (1.0 - h) * x
        new = np.where(live, np.bincount(ss, weights=w * x, minlength=S), 0.0)
        back = float((sign * (V - new))[live].max())    # > 0: a step against the monotone direction
        worst = max(worst, back)
        assert back <= mono_tol, f"robust {sense} iterate is not monotone at iteration {it} ({back:.3g})"
        new = np.maximum(new, V) if sense == "min" else np.minimum(new, V)
        change = float(np.abs(new - V).max())
        V = new
        if change <= tol:
            converged = True
            break
    return V, {"iterations": int(it), "converged": bool(converged), "last_change": change,
               "monotone_slack": worst, "rows": int(len(rows)), "rows_unrestricted": int((~np.isfinite(rr)).sum())}


# ---------- claims, coverage and the per-draw driver ----------
def x_cs_pair_stats(C: np.ndarray, u: np.ndarray, jt: np.ndarray) -> Dict[str, Any]:
    """Pairs, claims and errors (truth reversed or tied) of one claim matrix C[i, j] = 'i beats j', over all pairs and
    by type: AA (neither policy can leave the support), AU, UU, and sens = AU + UU (completion-sensitive)."""
    u = np.asarray(u, dtype=bool)
    jt = np.asarray(jt, dtype=np.float64)
    off = ~np.eye(len(u), dtype=bool)
    nu = u[:, None].astype(np.int64) + u[None, :].astype(np.int64)
    wrong = ~_x_v4_gt(jt, jt)
    C = np.asarray(C, dtype=bool) & off
    out: Dict[str, Any] = {}
    for sfx, tm in (("", off), ("_AA", off & (nu == 0)), ("_AU", off & (nu == 1)), ("_UU", off & (nu == 2)),
                    ("_sens", off & (nu >= 1))):
        out[f"pairs{sfx}"] = int(tm.sum() // 2)
        out[f"claims{sfx}"] = int((C & tm).sum())
        out[f"errors{sfx}"] = int((C & tm & wrong).sum())
    out["any_wrong"] = bool(out["errors"] > 0)
    return out


def x_cs_interval_stats(lo: np.ndarray, up: np.ndarray, jt: np.ndarray) -> Dict[str, Any]:
    """Coverage of J_true by the intervals [lo, up] of one policy set (tolerance XV4_TOL) and their mean width."""
    lo, up, jt = (np.asarray(x, dtype=np.float64) for x in (lo, up, jt))
    lo_ok, up_ok = lo <= jt + XV4_TOL, jt <= up + XV4_TOL
    cov = lo_ok & up_ok
    return {"n_policies": int(len(jt)), "n_covered": int(cov.sum()), "coverage": float(cov.mean()),
            "all_covered": bool(cov.all()), "lower_misses": int((~lo_ok).sum()), "upper_misses": int((~up_ok).sum()),
            "mean_width": float((up - lo).mean())}


def x_cs_identity(spec: dict, inst: CellInstance) -> dict:
    return {"kind": "xenv_v4_confidence", "draw": x_v4_draw_identity(spec, inst), "code": XV4_CS_CODE_HASH,
            "delta": float(XV4CFG.cs_delta), "scales": [float(x) for x in XV4CFG.cs_scales],
            "tol": float(XV4CFG.cs_tol), "max_iter": int(XV4CFG.cs_max_iter), "columns": XV4_CS_COLUMNS,
            "policies": "default (is_default) records of x_collect_records"}


def x_cs_check_rows(df: pd.DataFrame, v_min: float, v_max: float) -> Dict[str, Any]:
    """Invariants of one draw's confidence-set rows (asserted; re-run on cached tables) and a small report."""
    assert not df.duplicated(["policy", "scale"]).any(), "duplicated (policy, scale) confidence-set rows"
    L, Uf, Up, Un = (df[c].to_numpy(dtype=float) for c in ("L", "U_free", "U_proj", "U_nhb"))
    assert np.isfinite(np.c_[L, Uf, Up, Un]).all(), "non-finite confidence-set bound"
    gap = float((L - Uf).max(initial=-np.inf))
    assert gap <= XV4_TOL, f"L > U_free by {gap:.3g}"
    # exact in memory; XV4_TOL only guards a cached table against a float formatting change of the csv round trip
    assert float(np.abs(Un - np.minimum(Uf, Up)).max(initial=0.0)) <= XV4_TOL, "U_nhb must be min(U_free, U_proj)"
    below = float((v_min - L).max(initial=-np.inf))
    above = float((np.r_[Uf, Up] - v_max).max(initial=-np.inf))
    assert below <= XV4_TOL and above <= XV4_TOL, f"bound outside [v_min, v_max] ({below:.3g}, {above:.3g})"
    conv = np.stack([xs_as_bool(df[c]).to_numpy(dtype=bool)
                     for c in ("converged_lower", "converged_upper", "converged_upper_proj")], axis=1)
    it = df[["iter_lower", "iter_upper", "iter_upper_proj"]].to_numpy(dtype=np.int64)
    return {"rows": int(len(df)), "max_lower_minus_upper": gap,
            "max_nhb_minus_free": float((Un - Uf).max(initial=-np.inf)), "max_below_v_min": below,
            "max_above_v_max": above, "nonconverged_rows": int((~conv.all(axis=1)).sum()),
            "max_iterations": int(it.max(initial=0))}


def run_v4_cs_draw(spec: dict, draw: int) -> Tuple[pd.DataFrame, dict]:
    """Confidence-set bounds and certificates of the default policies of one draw (cached under v4cs/). The meta
    holds the per-scale x test rows under 'tests'."""
    cell, family, draw = spec["cell"], spec["family"], int(draw)
    assert family in ("gumbel", "garnet"), family
    t0 = time.perf_counter()
    inst = x_v4_instance(spec, draw)
    identity = x_cs_identity(spec, inst)
    key, ident_hash = "v4cs/" + x_cache_key(cell, draw), obj_hash(identity)[:20]
    build_s = time.perf_counter() - t0
    hit, meta_hit = XCACHE.get(key, identity), XCACHE.get(key + "/meta", identity)
    if hit is not None and meta_hit is not None:
        del inst
        assert list(hit.columns) == XV4_CS_COLUMNS, f"{key}: cached columns differ from XV4_CS_COLUMNS"
        meta = json.loads(str(meta_hit["meta_json"].iloc[0]))
        meta.update(cache="loaded", session_build_s=build_s,
                    checks=x_cs_check_rows(hit, meta["v_min"], meta["v_max"]))
        gc.collect()
        return hit, meta
    if (hit is None) != (meta_hit is None):
        print(f"  {key}: incomplete cache entry (results or meta missing) - recomputing this draw")
    XCACHE.require_allowed(key)
    learn = inst.learn_env
    t1 = time.perf_counter()
    records, keys = x_v4_default_records(inst)
    t2 = time.perf_counter()
    true_env, _, hosp, B = x_v4_rebuild_benchmark(spec, draw, inst)
    tgt = {lab: (env, sem, lift) for lab, env, sem, lift in inst.targets}
    T_true, sem_true, lift_true = tgt["true"]
    assert sem_true == "given" and lift_true is not None and T_true.S == true_env.S, cell
    model = x_cs_counts(true_env, hosp)
    del hosp
    v_min, v_max, live = model["v_min"], model["v_max"], model["live"]
    ev: Dict[bytes, Dict[str, Any]] = {}
    for rec, k in zip(records, keys):
        if k in ev:
            continue
        pi_true, pi_proj = B["lift"](rec["pi"]), B["lift"](x_project(rec["pi"], learn.mask, learn.live))
        assert np.array_equal(pi_true, lift_true(rec["pi"])), f"{rec['label']}: rebuilt lift != instance lift"
        e_true = x_v4_eval(T_true, pi_true, "given")
        e_mean, e_term = x_v4_eval(learn, rec["pi"], "given"), x_v4_eval(learn, rec["pi"], "terminate")
        bad = [(lab, e["status"]) for lab, e in (("true", e_true), ("bench_mean", e_mean), ("bench_term", e_term))
               if e["status"] != "ok"]
        assert not bad, f"{cell} draw {draw}: {rec['label']} is not proper on {bad}"
        ev[k] = {"J_true": e_true["return"], "J_bench_mean": e_mean["return"], "J_bench_term": e_term["return"],
                 "p_bench": e_mean["prob_any_violation"], "pi_true": pi_true, "pi_proj": pi_proj,
                 "proj_is_same": bool(np.array_equal(pi_true, pi_proj))}
    pols = [ev[k] for k in keys]
    n = len(pols)
    jt, jm, jterm, pb = (np.array([p[c] for p in pols]) for c in ("J_true", "J_bench_mean", "J_bench_term", "p_bench"))
    u = pb > XV4_PTOL
    used = np.zeros(model["S"] * model["A"], dtype=bool)
    for p in ev.values():
        for pi in (p["pi_true"], p["pi_proj"]):
            ss, aa = np.nonzero((pi > 0.0) & live[:, None])
            used[ss * model["A"] + aa] = True
    t3 = time.perf_counter()
    base = {"cell": cell, "family": family, "draw": draw}
    rows, tests, scales = [], [], {}
    for scale in XV4CFG.cs_scales:
        t_s = time.perf_counter()
        scale = float(scale)
        kind = "valid" if scale == 1.0 else "heuristic"
        r, r0 = x_cs_radii(model, scale, XV4CFG.cs_delta)
        memo: Dict[Tuple[bytes, str], Tuple[np.ndarray, Dict[str, Any]]] = {}
        got: Dict[str, List[Tuple[np.ndarray, Dict[str, Any]]]] = {"lower": [], "upper": [], "proj": []}
        for p in pols:
            for name, pi, sense in (("lower", p["pi_true"], "min"), ("upper", p["pi_true"], "max"),
                                    ("proj", p["pi_proj"], "max")):
                mk = (_x_pi_key(pi), sense)
                if mk not in memo:
                    memo[mk] = x_cs_robust_vector(model, r, pi, sense, XV4CFG.cs_tol, XV4CFG.cs_max_iter)
                got[name].append(memo[mk])
        Lv, Ufv, Upv = (np.stack([v for v, _ in got[nm]]) for nm in ("lower", "upper", "proj"))
        Unv = np.minimum(Ufv, Upv)
        L = x_cs_l1_extreme(model["d0hat"], r0, Lv, live, "min")
        Uf = x_cs_l1_extreme(model["d0hat"], r0, Ufv, live, "max")
        Up = x_cs_l1_extreme(model["d0hat"], r0, Upv, live, "max")
        Un = np.minimum(Uf, Up)
        PF = np.stack([x_cs_l1_extreme(model["d0hat"], r0, Lv[i][None, :] - Ufv, live, "min") for i in range(n)])
        PN = np.stack([x_cs_l1_extreme(model["d0hat"], r0, Lv[i][None, :] - Unv, live, "min") for i in range(n)])
        gap = float((L - Uf).max())
        assert gap <= XV4_TOL and (Un <= Uf).all(), f"{cell} draw {draw} scale {scale}: L > U_free by {gap:.3g}"
        assert (L >= v_min - XV4_TOL).all() and (Uf <= v_max + XV4_TOL).all(), "bound outside [v_min, v_max]"
        pair_slack = min(float((PF - (L[:, None] - Uf[None, :])).min()), float((PN - (L[:, None] - Un[None, :])).min()))
        assert pair_slack >= -XV4_CS_PAIR_TOL, f"a paired test is weaker than the unpaired one ({pair_slack:.3g})"
        claims = {"free": _x_v4_gt(L, Uf), "nhb": _x_v4_gt(L, Un), "paired_free": PF > XV4_TOL,
                  "paired_nhb": PN > XV4_TOL}
        ivs = {"free": (L, Uf), "nhb": (L, Un), "paired_free": (L, Uf), "paired_nhb": (L, Un)}
        ru = r[used]
        rad = {"r0": float(r0), "used_pairs": int(used.sum()),
               "radius_median": float(np.median(ru)) if len(ru) else float("nan"),
               "radius_median_finite": float(np.median(ru[np.isfinite(ru)])) if np.isfinite(ru).any() else float("nan"),
               "share_unrestricted": float((~np.isfinite(ru)).mean()) if len(ru) else float("nan")}
        for name in XV4_CS_TESTS:
            tests.append({**base, "scale": scale, "set_kind": kind, "test": name, "interval": XV4_CS_INTERVALS[name],
                          **x_cs_interval_stats(*ivs[name], jt), **x_cs_pair_stats(claims[name], u, jt), **rad})
        cov_f, cov_n = (L <= jt + XV4_TOL) & (jt <= Uf + XV4_TOL), (L <= jt + XV4_TOL) & (jt <= Un + XV4_TOL)
        for i, (rec, p) in enumerate(zip(records, pols)):
            (_, il), (_, iu), (_, ip) = got["lower"][i], got["upper"][i], got["proj"][i]
            d = rec["desc"]
            rows.append({**base, "policy": d["policy"], "method": d["method"], "variant": d["variant"], "hp": d["hp"],
                         "method_family": d["method_family"], "scale": scale, "set_kind": kind, "J_true": jt[i],
                         "J_bench_mean": jm[i], "J_bench_term": jterm[i], "p_bench": pb[i], "L": L[i], "U_free": Uf[i],
                         "U_proj": Up[i], "U_nhb": Un[i], "proj_is_same": p["proj_is_same"],
                         "covered_free": bool(cov_f[i]), "covered_nhb": bool(cov_n[i]),
                         "iter_lower": il["iterations"], "iter_upper": iu["iterations"],
                         "iter_upper_proj": ip["iterations"], "converged_lower": il["converged"],
                         "converged_upper": iu["converged"], "converged_upper_proj": ip["converged"],
                         "rows_used": il["rows"], "rows_unrestricted": il["rows_unrestricted"]})
        infos = [i_ for _, i_ in memo.values()]
        scales[str(scale)] = {**rad, "set_kind": kind, "robust_runs": len(memo),
                              "max_iterations": int(max(i_["iterations"] for i_ in infos)),
                              "mean_iterations": float(np.mean([i_["iterations"] for i_ in infos])),
                              "nonconverged_runs": int(sum(not i_["converged"] for i_ in infos)),
                              "max_monotone_slack": float(max(i_["monotone_slack"] for i_ in infos)),
                              "min_paired_minus_unpaired": pair_slack, "seconds": time.perf_counter() - t_s}
        del memo, got, Lv, Ufv, Upv, Unv, PF, PN
    v_range = v_max - v_min
    ref = {x["type"]: x for x in x_v4_pair_claims(jterm, jm, pb, v_range, jt)}
    for name, kind_ref, up in (("point_nhb", "cert_nhb", jm), ("point_free", "cert_free", jterm + pb * v_range)):
        st = x_cs_pair_stats(_x_v4_gt(jterm, up), u, jt)
        for ty, sfx in (("all", ""), ("AA", "_AA"), ("AU", "_AU"), ("UU", "_UU")):
            assert (st[f"claims{sfx}"], st[f"errors{sfx}"]) == (ref[ty][f"{kind_ref}_claims"],
                                                                 ref[ty][f"{kind_ref}_errors"]), (name, ty)
        tests.append({**base, "scale": float("nan"), "set_kind": "point", "test": name,
                      "interval": XV4_CS_INTERVALS[name], **x_cs_interval_stats(jterm, up, jt), **st,
                      "r0": float("nan"), "used_pairs": int(used.sum()), "radius_median": float("nan"),
                      "radius_median_finite": float("nan"), "share_unrestricted": float("nan")})
    t4 = time.perf_counter()
    df = pd.DataFrame(rows, columns=XV4_CS_COLUMNS)
    checks = x_cs_check_rows(df, v_min, v_max)
    meta = _x_compact({
        "cell": cell, "family": family, "draw": draw, "identity_hash": ident_hash, "code_hash": XV4_CS_CODE_HASH[:16],
        "cache": "computed", "v_min": v_min, "v_max": v_max, "hazard": model["hazard"], "n_policies": n,
        "unique_policies": len(ev), "true_states": model["S"], "true_live_states": model["m0"],
        "counts": {"steps": model["steps"], "informative_steps": model["informative_steps"],
                   "hospital_episodes": model["n_ep"], "logged_pairs": int((model["N"] > 0).sum()),
                   "max_row_support": model["max_row_support"], "outcomes_m": model["m"], "union_pairs_K": model["K"]},
        "scales": scales,
        "max_monotone_slack": float(max(v["max_monotone_slack"] for v in scales.values())),
        "min_paired_minus_unpaired": float(min(v["min_paired_minus_unpaired"] for v in scales.values())),
        "nonconverged_runs": int(sum(v["nonconverged_runs"] for v in scales.values())), "checks": checks,
        "timing": {"build_s": build_s, "methods_s": t2 - t1, "rebuild_eval_s": t3 - t2, "cs_s": t4 - t3,
                   "total_s": time.perf_counter() - t0}})
    meta["tests"] = tests                     # kept whole (not compacted): the per-draw x scale x test rows
    XCACHE.put(key + "/meta", identity, pd.DataFrame({"meta_json": [json.dumps(meta, default=_json_default)]}))
    XCACHE.put(key, identity, df)
    back = XCACHE.get(key, identity)
    assert back is not None and list(back.columns) == XV4_CS_COLUMNS and len(back) == len(df), f"{key}: cache write"
    for c in XV4_CS_FLOAT_COLUMNS:
        assert np.allclose(back[c].to_numpy(dtype=float), df[c].to_numpy(dtype=float), rtol=1e-14, atol=0.0,
                           equal_nan=True), f"{key}: cache round trip changed column {c}"
    del inst, records, ev, pols, model, df, rows
    gc.collect()
    return back, meta


# Objects whose source changes what a confidence-set draw computes (the row checks are re-run on cached tables).
XV4_CS_CODE = [x_cs_pack, _x_cs_extreme_packed, x_cs_l1_extreme, x_cs_counts, x_cs_radii, x_cs_robust_vector,
               x_cs_pair_stats, x_cs_interval_stats, x_cs_identity, run_v4_cs_draw]
XV4_CS_CODE_HASH = obj_hash({"common": XV4_COMMON_CODE_HASH, "code": code_hash(*XV4_CS_CODE),
                             "constants": {"tests": list(XV4_CS_TESTS), "point_tests": list(XV4_CS_POINT_TESTS),
                                           "intervals": XV4_CS_INTERVALS, "mono_tol": XV4_CS_MONO_TOL,
                                           "pair_tol": XV4_CS_PAIR_TOL, "columns": XV4_CS_COLUMNS,
                                           "float_columns": XV4_CS_FLOAT_COLUMNS}})


# ---------- self-test of the inner problem, summaries and the cell runner (analysis only, not hashed) ----------
def _x_cs_lp_reference(qhat: np.ndarray, r: float, U: np.ndarray, sup: np.ndarray, sense: str) -> float:
    """The inner problem as a linear program: variables (q, t), |q - qhat| <= t, sum t <= r, q in simplex(sup)."""
    S = len(qhat)
    c = np.r_[(1.0 if sense == "min" else -1.0) * U, np.zeros(S)]
    I = np.eye(S)
    A_ub = np.r_[np.c_[I, -I], np.c_[-I, -I]]
    b_ub = np.r_[qhat, -qhat]
    if np.isfinite(r):
        A_ub = np.r_[A_ub, np.r_[np.zeros(S), np.ones(S)][None, :]]
        b_ub = np.r_[b_ub, r]
    bounds = [(0.0, 1.0 if s_ else 0.0) for s_ in sup] + [(0.0, None)] * S
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=np.r_[np.ones(S), np.zeros(S)][None, :], b_eq=[1.0], bounds=bounds,
                  method="highs")
    assert res.status == 0, res.message
    return float(U @ res.x[:S])


def x_cs_selftest(n_instances: int = 200) -> Dict[str, Any]:
    """x_cs_l1_extreme against scipy's linprog on random rows: both senses, shared and per-row U and supports, tied
    values, radii tiny / moderate / >= 2 / inf, and all-zero rows with r = inf (N = 0)."""
    rng = np_rng("xenv", "v4", "cs-selftest")
    worst, rows = 0.0, 0
    for k in range(int(n_instances)):
        S, n, shared = int(rng.integers(2, 9)), int(rng.integers(1, 5)), k % 3 == 0
        sup = rng.random((n, S)) < 0.7
        j = rng.integers(0, S, n)
        if shared:                                         # one support and one U for every row
            sup[:], j[:] = sup[0], j[0]
        sup[np.arange(n), j] = True
        q = rng.dirichlet(np.ones(S), size=n) * (rng.random((n, S)) < 0.6) * sup
        q[np.arange(n), j] += q.sum(axis=1) == 0           # an empty draw puts its mass on the forced support state
        q /= q.sum(axis=1, keepdims=True)
        r = rng.choice(np.array([1e-4, 0.05, 0.3, 0.9, 1.7, 2.5, np.inf]), size=n)
        q[(r == np.inf) & (rng.random(n) < 0.3)] = 0.0    # unlogged rows (N = 0) are unrestricted
        U = rng.integers(-2, 3, size=(n, S)).astype(np.float64) if k % 2 else rng.normal(size=(n, S))
        for sense in ("min", "max"):
            got = (x_cs_l1_extreme(q, r, U[0], sup[0], sense) if shared
                   else x_cs_l1_extreme(q, r, U, sup, sense))
            for i in range(n):
                ref = _x_cs_lp_reference(q[i], float(r[i]), U[0] if shared else U[i], sup[i], sense)
                worst = max(worst, abs(float(got[i]) - ref))
                rows += 1
    return {"instances": int(n_instances), "rows_compared": rows, "max_abs_diff_vs_linprog": worst,
            "tolerance": 1e-8, "passes": bool(worst <= 1e-8)}


_XCS_SUM_COLUMNS = ("n_policies", "n_covered", "lower_misses", "upper_misses", "pairs", "claims", "errors",
                    "pairs_AA", "claims_AA", "errors_AA", "pairs_AU", "claims_AU", "errors_AU", "pairs_UU",
                    "claims_UU", "errors_UU", "pairs_sens", "claims_sens", "errors_sens")


def x_cs_summary_table(draws: pd.DataFrame, cells: Sequence[str]) -> pd.DataFrame:
    """Per cell, family pool and pooled, x scale x test: claims, share, errors and error rate (all pairs and the
    completion-sensitive pairs AU + UU), draws with a wrong claim or an uncovered policy, coverage and mean width."""
    if draws is None or not len(draws):
        return pd.DataFrame()
    fam_of = draws.drop_duplicates("cell").set_index("cell")["family"].to_dict()
    present = [c for c in cells if c in fam_of]
    units = [(c, [c]) for c in present]                    # family pools and 'pooled:all' always exist
    units += [(f"pooled:{f}", [c for c in present if fam_of[c] == f]) for f in sorted(set(fam_of.values()))]
    units.append(("pooled:all", present))
    out = []
    for name, cs in units:
        d = draws[draws["cell"].isin(cs)]
        for (scale, test), g in d.groupby(["scale", "test"], dropna=False, sort=False):
            s = g[list(_XCS_SUM_COLUMNS)].sum()
            out.append({"unit": name, "n_cells": len(cs), "scale": scale, "set_kind": g["set_kind"].iloc[0],
                        "test": test, "interval": g["interval"].iloc[0], "draws": int(len(g)),
                        **{c: int(s[c]) for c in _XCS_SUM_COLUMNS},
                        "share": _xv3_rate(s["claims"], s["pairs"]), "error_rate": _xv3_rate(s["errors"], s["claims"]),
                        "share_sens": _xv3_rate(s["claims_sens"], s["pairs_sens"]),
                        "error_rate_sens": _xv3_rate(s["errors_sens"], s["claims_sens"]),
                        "draws_any_wrong": int(xs_as_bool(g["any_wrong"]).sum()),
                        "draws_any_uncovered": int((~xs_as_bool(g["all_covered"])).sum()),
                        "coverage": _xv3_rate(s["n_covered"], s["n_policies"]),
                        "mean_width": float(np.average(g["mean_width"].to_numpy(float),
                                                       weights=g["n_policies"].to_numpy(float))),
                        "radius_median": float(g["radius_median"].median()),
                        "radius_median_finite": float(g["radius_median_finite"].median()),
                        "share_unrestricted": float(g["share_unrestricted"].mean()), "r0_mean": float(g["r0"].mean())})
    return pd.DataFrame(out)


def x_cs_run_checks(pol: pd.DataFrame, meta: pd.DataFrame) -> Dict[str, Any]:
    """Pooled invariant report of a run (every per-draw assertion has passed when this is called)."""
    def col(c: str, f: Callable, default):
        return float(f(pd.to_numeric(meta[c]).to_numpy(dtype=float))) if c in meta and len(meta) else default

    out = {"policy_rows": int(len(pol)),
           "max_lower_minus_upper": float((pol["L"] - pol["U_free"]).max()) if len(pol) else None,
           "max_nhb_minus_free": float((pol["U_nhb"] - pol["U_free"]).max()) if len(pol) else None,
           "max_monotone_slack": col("max_monotone_slack", np.max, None),
           "min_paired_minus_unpaired": col("min_paired_minus_unpaired", np.min, None),
           "nonconverged_runs": int(col("nonconverged_runs", np.sum, 0)),
           "nonconverged_policy_rows": int((~(xs_as_bool(pol["converged_lower"]) & xs_as_bool(pol["converged_upper"])
                                              & xs_as_bool(pol["converged_upper_proj"]))).sum()) if len(pol) else 0,
           "max_iterations": int(pol[["iter_lower", "iter_upper", "iter_upper_proj"]].to_numpy().max(initial=0))}
    out["passes"] = bool(len(pol) > 0 and out["max_lower_minus_upper"] <= XV4_TOL and out["max_nhb_minus_free"] <= 0.0
                         and out["max_monotone_slack"] is not None and out["min_paired_minus_unpaired"] is not None
                         and out["min_paired_minus_unpaired"] >= -XV4_CS_PAIR_TOL)
    return out


def x_cs_run(cells: Sequence[dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Every draw of every confidence-set cell (run_v4_cs_draw via x_v4_run_draws): policy rows, per-draw test rows,
    flattened draw meta, run summary."""
    tests: List[dict] = []

    def one(spec: dict, draw: int) -> Tuple[pd.DataFrame, dict]:
        df, meta = run_v4_cs_draw(spec, draw)
        meta = dict(meta)
        tests.extend(meta.pop("tests"))
        return df, meta

    frames, meta_df, summary = x_v4_run_draws(cells, one, "II.9c confidence sets")
    pol = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=XV4_CS_COLUMNS)
    return pol, pd.DataFrame(tests), meta_df, summary


XV4_CS_POLICIES = pd.DataFrame(columns=XV4_CS_COLUMNS)
XV4_CS_DRAWS = pd.DataFrame()
XV4_CS_TABLE = pd.DataFrame()
XV4_CS_META = pd.DataFrame()             # flattened per-draw meta (memory only; the release sums 'timing.total_s')
XV4_CS_SUMMARY: Dict[str, Any] = {"status": "not run (xenv_v4_confidence off)", "code_hash": XV4_CS_CODE_HASH,
                                  "checks": {}, "elapsed_s": 0.0, "draws_complete": None}
print("II.9c code hash:", XV4_CS_CODE_HASH[:16])

# %% [markdown]
# #### II.9c run
#
# Runs the self-test of the inner problem (it must pass), then every draw of every enabled confidence-set cell with
# the progress and ETA of II.6a, and writes the four outputs listed above (the flattened draw meta stays in
# memory as `XV4_CS_META`). A configured cell that is not enabled (family switch off, or not in the design) is
# skipped and listed in `v4_cs_summary.json`. The headline block of the json gives, per family pool and scale, the
# certified share and error rate of every test on all pairs and on the completion-sensitive pairs, beside the
# point-benchmark certificates. The pooled invariant checks are asserted after the json is written.

# %%
# ============================================================
# II.9c Confidence-set certificates: run
# ============================================================
EXECLOG.mark("II.9c Confidence-set certificates: run")
if EXPERIMENTS["xenv_v4_confidence"]:
    _t0 = time.perf_counter()
    _selftest = x_cs_selftest()
    print("II.9c inner-problem self-test:", _selftest)
    assert _selftest["passes"], f"x_cs_l1_extreme disagrees with linprog: {_selftest}"
    _cells, _skipped = x_v4_cells(XV4CFG.cs_cells)
    if _skipped:
        print("NOTE: confidence-set cells not enabled (family switch off or not in the design), skipped:", _skipped)
    if _cells:
        XV4_CS_POLICIES, XV4_CS_DRAWS, XV4_CS_META, _run = x_cs_run(_cells)
        XV4_CS_TABLE = x_cs_summary_table(XV4_CS_DRAWS, [str(s["cell"]) for s in _cells])
        XV4_CS_POLICIES.to_csv(XENV_TABLE_DIR / "v4_cs_policies.csv.gz", index=False, compression="gzip")
        XV4_CS_DRAWS.to_csv(XENV_TABLE_DIR / "v4_cs_draws.csv", index=False)
        XV4_CS_TABLE.to_csv(XENV_TABLE_DIR / "v4_cs_summary.csv", index=False)
        _checks = {**x_cs_run_checks(XV4_CS_POLICIES, XV4_CS_META), "test_rows": int(len(XV4_CS_DRAWS))}
        _hcols = ["unit", "scale", "set_kind", "test", "draws", "claims", "share", "errors", "error_rate", "share_sens",
                  "error_rate_sens", "draws_any_wrong", "draws_any_uncovered", "coverage", "mean_width"]
        _head = (XV4_CS_TABLE[XV4_CS_TABLE["unit"].astype(str).str.startswith("pooled:")][_hcols]
                 if len(XV4_CS_TABLE) else pd.DataFrame(columns=_hcols))
        _need = sum(x_v4_draws(s) for s in _cells)
        XV4_CS_SUMMARY = {
            "status": "run", "code_hash": XV4_CS_CODE_HASH,
            "config": {"delta": XV4CFG.cs_delta, "scales": list(XV4CFG.cs_scales), "tol": XV4CFG.cs_tol,
                       "max_iter": XV4CFG.cs_max_iter, "valid_scale": 1.0,
                       "note": "only scale 1.0 is covered by the union bound; smaller scales are heuristic"},
            "selftest": _selftest, "cells": [s["cell"] for s in _cells], "cells_skipped": _skipped,
            **{k: _run[k] for k in ("draws", "draws_computed", "draws_loaded", "draws_per_cell")},
            "draws_complete": bool(_run["draws"] == _need), "checks": _checks,
            "headline": _head.to_dict(orient="records"),
            "outputs": ["xenv/v4_cs_policies.csv.gz", "xenv/v4_cs_draws.csv", "xenv/v4_cs_summary.csv",
                        "xenv/v4_cs_summary.json"],
            "elapsed_s": time.perf_counter() - _t0}
        display(_head.drop(columns=["set_kind", "claims", "errors", "draws_any_uncovered"]).round(4))
        del _checks, _hcols, _head, _need, _run
    else:
        XV4_CS_SUMMARY = {"status": "not run: no configured confidence-set cell is enabled", "cells_skipped": _skipped,
                          "code_hash": XV4_CS_CODE_HASH, "selftest": _selftest, "checks": {}, "draws_complete": None,
                          "elapsed_s": time.perf_counter() - _t0}
    del _t0, _selftest, _cells, _skipped
save_json(XV4_CS_SUMMARY, XENV_TABLE_DIR / "v4_cs_summary.json")
print(json.dumps({k: v for k, v in XV4_CS_SUMMARY.items() if k not in ("headline", "draws_per_cell")}, indent=1,
                 default=_json_default))
# after the json is written, so that a failure can be inspected there (as the II.9 gate message says)
assert XV4_CS_SUMMARY.get("status") != "run" or XV4_CS_SUMMARY["checks"]["passes"], \
    f"II.9c invariant checks failed: {XV4_CS_SUMMARY['checks']} - see xenv/v4_cs_summary.json"
