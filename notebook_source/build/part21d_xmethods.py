# %% [markdown]
# ## II.4 Method suite: exact tabular offline-RL learners on the empirical model
#
# Every Part II learner is a deterministic function of its training data and a `LearnerView`. No learner reads a
# transition tensor, an initial distribution or a `FiniteEnv`. Data enter only through `XEmpirical`:
#
# * counts N(s,a) and maximum-likelihood rows P̂;
# * mean logged rewards r̂;
# * the empirical behaviour policy π̂_β, which uses the global logged-action frequency on states without data;
# * the empirical start distribution d̂₀.
#
# All methods use γ = 1 with V = 0 on absorbing states. The value-iteration methods (IQL, BCQ, VI-LCB, MBS-QI, SPIBB)
# return expected entry rewards in [v_min, v_max]; CQL returns regularised scores.
#
# **Conventions shared by all learners.**
# * *Iteration.* Fixed points are computed by synchronous iteration from V = 0. Iteration stops when
#   max|ΔQ| < `XCFG.tol`, capped at `XCFG.max_iterations`. Non-convergence is reported in `info["converged"]`,
#   never raised and never hidden.
# * *Unobserved pairs* (N(s,a) = 0) have zero rows in P̂. Each method states its convention:
#   * IQL keeps Part I's never-updated value 0;
#   * CQL pushes the unobserved actions of states with data toward −∞ (the conservative gradient is positive there);
#     rows of states without data stay 0;
#   * BCQ, MBS-QI and SPIBB never back up through these pairs;
#   * VI-LCB sets them to v_min;
#   * states without any data are valued 0 by BCQ, and v_min by MBS-QI and VI-LCB;
#   * the model-based constrained methods complete them with a jump to the sink paying `view.terminate_reward`
#     (pessimistic completion, `XEmpirical.completed_dense`).
# * *Exact evaluation on the empirical model* (SPIBB, Lagrangian, LP cross-check) splits the policy-induced chain
#   into transient states and closed classes without leakage. Rewards are paid only on entering absorbing states,
#   and a closed class never absorbs. Its value is therefore exactly 0, which is the limit of value iteration from
#   V = 0, and its occupancy is infinite once it is entered. No singular system is ever solved.
# * *Variants.*
#   * `native` is the method's own output.
#   * `masked` restricts extraction to the admissible mask M of the learner's environment. Zero mass outside M on
#     live states is asserted for every masked record, and for every method that is admissible by construction.
#   * The driver (Section II.6) adds the `projected` variant `x_project(native, M)` of every native record.
# * *Hyper-parameters.* Every tunable method has a pre-declared default `XCFG.*_default`, which is asserted to lie
#   in its grid. Ties in greedy extraction go to the lowest index, as in Part I, except where a row of the table
#   says otherwise.
#
# | Method (family) | Exact tabular definition | Reference |
# |---|---|---|
# | BC (stochastic), BC (behaviour) | π̂_β; argmax of counts; masked: argmax of counts over M, falling back to the global frequency over M | behaviour cloning (empirical MLE) |
# | Uniform-admissible (behaviour) | uniform over M(s) | trivial admissible baseline |
# | CQL (value) | full-batch Adam on (1/n)Σ N(Q−ȳ)² + α(1/n)[Σ_s N_s lse Q(s,·) − Σ N Q], with ȳ = r̂ + P̂ max Q_target (hard target sync); approximate fixed point after 400 target syncs (`XCFG.cql_steps` / `cql_target_every`); non-convergence (max \|ΔQ\| on observed pairs over the last sync interval > `XCFG.cql_converged_tol`) is flagged | Kumar et al. 2020 |
# | IQL, IQL (AW stochastic) (value) | in-sample fixed point: V(s) = N-weighted τ-expectile of Q(s,·) (exact sorting solution), Q = r̂ + P̂V on observed pairs; extraction by AW weights exp(clip(β(Q−V), −20, log w_max))·N, as in Part I | Kostrikov et al. 2022 |
# | BCQ (pessimistic) | VI restricted to G(s) = {a : π̂_β(a\|s)/max π̂_β ≥ τ}; masked: argmax over G∩M, falling back to M | Fujimoto et al. 2019 (discrete BCQ) |
# | VI-LCB (pessimistic) | Q = clip(r̂ − c·(v_max − v_min)/√max(N,1) + P̂V, v_min, v_max) on observed pairs, v_min elsewhere; ties go to the most frequently logged action | Rashidinejad et al. 2021; Jin et al. 2021 |
# | MBS-QI (pessimistic) | backups only through ζ = 1[N ≥ b]; V_ζ = v_min where no action is supported; states without supported actions use BC | Liu et al. 2020 |
# | SPIBB (pessimistic) | Π_b-SPIBB policy iteration on P̂: pairs with N < N∧ keep π̂_β, the remaining mass goes to argmax Q over the other pairs | Laroche et al. 2019 |
# | Lagrangian-MB (constrained) | cost c = 1[a ∉ M]; bisection on λ for greedy VI on r − λc in the completed model, until the expected violations from d̂₀ are ≤ κ | Altman 1999; Le, Voloshin & Yue 2019 (Batch Policy Learning under Constraints, ICML); Tessler et al. 2019 (RCPO) |
# | CMDP-LP (model) (constrained) | exact occupancy LP (`x_cmdp_lp`) on the completed model with E[violations] ≤ κ; zero-occupancy states use masked BC | Altman 1999; COptiDICE (Lee et al. 2022) |
# | SfBC (exact), SfBC (exact, argmax) (guided-prior) | π ∝ π̂_β·exp(α·A_IQL) at the default expectile; masked: renormalised on M | Chen et al. 2023 |
# | IDQL (exact) (guided-prior) | π ∝ π̂_β·\|τ − 1(Q < V)\| at the default expectile | Hansen-Estruch et al. 2023 |
# | FISOR-tab (exact) (constrained) | π ∝ π̂_β·exp(α·A_feas)·1[a ∈ M]; A_feas comes from the in-sample IQL fixed point whose expectile uses logged **admissible** actions only, so admissibility acts as feasibility in both the critic and the extraction | Zheng et al. 2024 |
# | Diffusion (exact, ·) (diffusion) | exact Bayes-optimal reverse kernels of the uniform-corruption chain started at π̂_β; Part I inference (uniform-admissible start, every-step mask, guidance by G = per-state standardised Q_IQL, deterministic argmax), plus final-step-only guidance and an unmasked control | Austin et al. 2021; Part I Section 12 |
#
# FISOR-tab uses the SfBC grid for α. With the unrestricted IQL critic it would equal `SfBC (exact) [masked]` exactly.
# The feasibility-restricted critic is what makes it a separate method, and it is the tabular analogue of FISOR's
# feasibility-dependent value learning.

# %%
# ============================================================
# II.4 Method suite
# ============================================================
EXECLOG.mark("II.4 Method suite")
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla

_X_TIE_RTOL = 1e-12          # relative tolerance for "tied" greedy scores
_X_LEAK_TOL = 1e-12          # row mass below 1 - tol leaves the live set (absorption, completion sink, no data)
# Module constants read by hashed functions: code_hash covers source text only, so the driver (II.6) and theory
# (II.5) identities carry their values.
X_NUMERIC_CONSTANTS = {"tie_rtol": _X_TIE_RTOL, "leak_tol": _X_LEAK_TOL, "strategy": dict(_X_STRATEGY)}
_X_SWITCHES = ("xenv_theory", "xenv_frontiers", "xenv_icu", "xenv_gumbel", "xenv_garnet", "xenv_tabular_crosscheck")


def _x_info(d: Dict[str, Any]) -> Dict[str, Any]:
    """Small JSON-able info dict: numpy scalars become Python scalars; arrays and objects are rejected."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = float(v)
        elif v is None or isinstance(v, str):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = _x_info(v)
        elif (isinstance(v, (list, tuple)) and len(v) <= 64
              and all(isinstance(x, (bool, int, float, str, np.number, np.bool_)) for x in v)):
            out[k] = [x.item() if isinstance(x, (np.number, np.bool_)) else x for x in v]
        else:
            raise TypeError(f"info field {k!r} is not a small JSON-able value ({type(v).__name__})")
    return out


def _x_iterate(step: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], live: np.ndarray,
               tol: Optional[float] = None, max_iter: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Synchronous fixed-point iteration from V = 0 (γ = 1, V = 0 on absorbing states).

    `step(V)` returns fresh arrays (Q [S, A], V_new [S]). Stops when max|ΔQ| < tol; hitting the cap is reported."""
    tol = XCFG.tol if tol is None else float(tol)
    max_iter = XCFG.max_iterations if max_iter is None else int(max_iter)
    V = np.zeros(len(live))
    Q_prev, delta, it = None, float("inf"), 0
    for it in range(1, max_iter + 1):
        Q, V_new = step(V)
        if not np.isfinite(Q).all():
            raise FloatingPointError("non-finite Q during fixed-point iteration")
        V_new[~live] = 0.0
        if Q_prev is not None:
            delta = float(np.max(np.abs(Q - Q_prev)))
        Q_prev, V = Q, V_new
        if delta < tol:
            break
    return Q_prev, V, {"iterations": it, "final_delta": delta, "converged": bool(delta < tol)}


def _x_argmax_tiebreak(scores: np.ndarray, allowed: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Argmax over `allowed`; scores within a relative 1e-12 of the best are tied and resolved by `secondary`
    (then the lowest index)."""
    s = np.where(allowed, np.asarray(scores, dtype=np.float64), -np.inf)
    best = s.max(axis=1, keepdims=True)
    tied = allowed & (s >= best - _X_TIE_RTOL * np.maximum(1.0, np.abs(best)))
    return masked_argmax(secondary, tied)


def _x_policy_rows(pi: np.ndarray) -> sp.csr_matrix:
    """Sparse [S, S*A] matrix whose row s holds pi(.|s) at the columns of the (s, a) pairs."""
    S, A = pi.shape
    return sp.csr_matrix((np.asarray(pi, dtype=np.float64).ravel(), (np.repeat(np.arange(S), A), np.arange(S * A))),
                         shape=(S, S * A))


def _x_chain(emp: XEmpirical, live: np.ndarray, pi: np.ndarray) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Live-to-live block of the policy-induced empirical chain. Missing row mass is absorption, the completion
    sink (unobserved pairs) or a state without data."""
    L = np.flatnonzero(live)
    P_pi = (_x_policy_rows(pi) @ emp.P).tocsr()
    return P_pi[L][:, L].tocsr(), L


def _x_markov_solve(P_LL: sp.spmatrix, r: np.ndarray, d0: Optional[np.ndarray] = None,
                    b: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Exact γ = 1 evaluation of a substochastic chain on live states (value, return, occupancy, violations).

    Closed classes without leakage never absorb. Their value is 0, because rewards are paid only on entry to
    absorbing states and value iteration from V = 0 converges to 0 there. Their expected visits, and hence the
    violations of a policy that violates inside them, are infinite once entered. Only the transient block,
    which is always non-singular, is factorised."""
    P = sp.csr_matrix(P_LL, dtype=np.float64)
    P.eliminate_zeros()
    n = P.shape[0]
    r = np.asarray(r, dtype=np.float64)
    if P.nnz and P.data.min() < 0:
        raise ValueError("negative transition probabilities")
    row = np.asarray(P.sum(axis=1)).ravel()
    if (row > 1.0 + 1e-9).any():
        raise ValueError(f"row sums exceed one (max {row.max():.12f})")
    leak = row < 1.0 - _X_LEAK_TOL
    n_comp, lab = csgraph.connected_components(P, directed=True, connection="strong")
    coo = P.tocoo()
    open_comp = np.zeros(n_comp, dtype=bool)
    cross = lab[coo.row] != lab[coo.col]
    open_comp[lab[coo.row[cross]]] = True
    open_comp[lab[leak]] = True
    rec = ~open_comp[lab]
    T = np.flatnonzero(~rec)
    v = np.zeros(n)
    out: Dict[str, Any] = {"recurrent_states": int(rec.sum())}
    lu = None
    if len(T):
        M = (sp.identity(len(T), format="csc") - P[T][:, T].tocsc()).tocsc()
        lu = spla.splu(M)
        vT = lu.solve(r[T])
        res = float(np.abs(M @ vT - r[T]).max())
        if not np.isfinite(vT).all() or res > 1e-8 * (1.0 + np.abs(r[T]).max() + np.abs(vT).max()):
            raise EvaluationInvariantError(f"empirical-chain value solve residual {res:.2e}")
        v[T] = vT
    out["v"] = v
    if d0 is not None:
        d0 = np.asarray(d0, dtype=np.float64)
        mu = np.zeros(n)
        if len(T):
            muT = lu.solve(d0[T], trans="T")
            res = float(np.abs(M.T @ muT - d0[T]).max())
            if not np.isfinite(muT).all() or res > 1e-8 * (1.0 + muT.max()) or muT.min() < -1e-9:
                raise EvaluationInvariantError(f"empirical-chain occupancy solve residual {res:.2e}")
            mu[T] = np.clip(muT, 0.0, None)
        inflow = d0 * rec
        if len(T) and rec.any():
            inflow = inflow + np.asarray(P[T].T @ mu[T]).ravel() * rec
        entered = np.zeros(n_comp, dtype=bool)
        entered[lab[inflow > 1e-15]] = True
        trapped_states = rec & entered[lab]
        out["occupancy"], out["trapped"], out["return"] = mu, float(inflow.sum()), float(d0 @ v)
        if b is not None:
            b = np.asarray(b, dtype=np.float64)
            viol = float(mu @ b)
            if (b[trapped_states] > 0).any():
                viol = float("inf")
            out["violations"] = viol
    return out


# ---------- behaviour cloning ----------
def x_bc(emp: XEmpirical, view: LearnerView) -> Dict[str, Any]:
    """Empirical behaviour cloning (Part I semantics with an explicit mask and live set)."""
    counts = emp.N_sa
    native = one_hot_policy(emp.pi_b.argmax(axis=1), view.A)
    adm_seen = (counts * view.mask).sum(axis=1) > 0
    masked = one_hot_policy(masked_argmax(np.where(adm_seen[:, None], counts, emp.global_pi[None, :]), view.mask),
                            view.A)
    info = {"masked_fallback_states": int((~adm_seen & view.live).sum()),
            "unseen_live_states": int((~emp.seen & view.live).sum())}
    return {"stochastic": emp.pi_b.copy(), "native": native, "masked": masked, "deterministic": native,
            "masked_greedy": masked, "info": info}


# ---------- CQL: exact full-batch objective, Adam, hard target sync ----------
def _x_cql_fit(emp: XEmpirical, view: LearnerView, alphas: Sequence[float], steps: int, lr: float,
               target_every: int) -> List[Dict[str, Any]]:
    """Exact empirical CQL for several α at once (independent problems stacked on a leading axis).

    Gradient of (1/n)Σ_sa N_sa (Q_sa − ȳ_sa)² + (α/n)[Σ_s N_s lse(Q_s) − Σ_sa N_sa Q_sa] on states with data;
    Adam as in torch (β = (0.9, 0.999), ε = 1e-8, bias-corrected). Rows of states without data stay 0.
    Convergence: max |ΔQ| on observed pairs over the last target-sync interval ≤ XCFG.cql_converged_tol (unobserved
    actions of rows with data drift toward −∞ under the conservative term, so they are excluded)."""
    S, A, H = view.S, view.A, len(alphas)
    rows = np.flatnonzero(emp.N_s > 0)
    if not view.live[rows].all():
        raise ValueError("CQL: data rows on absorbing states")
    N = emp.N_sa[rows]
    obs_r = emp.observed[rows]
    Ns = emp.N_s[rows][:, None]
    n = float(emp.n)
    c_td = 2.0 * N / n
    c_reg = np.asarray(alphas, dtype=np.float64)[:, None, None] / n
    q = np.zeros((H, len(rows), A))
    m1, m2, y = np.zeros_like(q), np.zeros_like(q), np.zeros_like(q)
    g, z, tmp = np.empty_like(q), np.empty_like(q), np.empty_like(q)
    b1, b2, eps = 0.9, 0.999, 1e-8
    full = np.zeros((S, A))

    def targets(qh: np.ndarray) -> np.ndarray:
        full[rows] = qh
        V = full.max(axis=1)
        V[~view.live] = 0.0
        return (emp.r_hat + emp.backup(V))[rows]

    q_sync = q.copy()
    last_dq, last_dq_obs = np.zeros(H), np.zeros(H)
    syncs = 0
    for step in range(1, steps + 1):
        if (step - 1) % target_every == 0:
            last_dq = np.abs(q - q_sync).max(axis=(1, 2))
            last_dq_obs = np.where(obs_r[None], np.abs(q - q_sync), 0.0).max(axis=(1, 2))
            q_sync = q.copy()
            for h in range(H):
                y[h] = targets(q[h])
            syncs += 1
        np.subtract(q, q.max(axis=2, keepdims=True), out=z)
        np.exp(z, out=z)
        z /= z.sum(axis=2, keepdims=True)
        z *= Ns
        z -= N
        z *= c_reg
        np.subtract(q, y, out=g)
        g *= c_td
        g += z
        m1 *= b1
        np.multiply(g, 1.0 - b1, out=tmp)
        m1 += tmp
        m2 *= b2
        np.multiply(g, g, out=tmp)
        tmp *= 1.0 - b2
        m2 += tmp
        np.sqrt(m2, out=tmp)
        tmp /= math.sqrt(1.0 - b2 ** step)
        tmp += eps
        np.divide(m1, tmp, out=tmp)
        tmp *= lr / (1.0 - b1 ** step)
        q -= tmp
    if not np.isfinite(q).all():
        raise FloatingPointError("CQL produced non-finite Q values")
    out = []
    obs = emp.observed
    for h, alpha in enumerate(alphas):
        Q = np.zeros((S, A))
        Q[rows] = q[h]
        V = Q.max(axis=1)
        V[~view.live] = 0.0
        bell = emp.r_hat + emp.backup(V)
        lse = np.log(np.exp(q[h] - q[h].max(axis=1, keepdims=True)).sum(axis=1)) + q[h].max(axis=1)
        info = {"alpha": float(alpha), "steps": int(steps), "lr": float(lr), "target_every": int(target_every),
                "target_syncs": syncs, "last_sync_max_abs_dq": float(last_dq[h]),
                "last_sync_max_abs_dq_observed": float(last_dq_obs[h]),
                "converged": bool(syncs >= 2 and last_dq_obs[h] <= XCFG.cql_converged_tol),
                "td_term": float((N * (q[h] - y[h]) ** 2).sum() / n),
                "conservative_term": float((emp.N_s[rows] * lse - (N * q[h]).sum(axis=1)).sum() / n),
                "td_residual_max_observed": float(np.abs(Q - bell)[obs].max()) if obs.any() else 0.0,
                "q_min": float(Q.min()), "q_max": float(Q.max())}
        native = x_greedy(Q)
        masked = x_greedy(Q, view.mask)
        out.append({"q": Q, "native": native, "masked": masked, "deterministic": native, "masked_greedy": masked,
                    "info": info})
    return out


def x_tab_cql(emp: XEmpirical, view: LearnerView, alpha: Optional[float] = None, steps: Optional[int] = None,
              lr: Optional[float] = None, target_every: Optional[int] = None) -> Dict[str, Any]:
    alpha = XCFG.cql_alpha_default if alpha is None else float(alpha)
    return _x_cql_fit(emp, view, (alpha,), XCFG.cql_steps if steps is None else int(steps),
                      XCFG.cql_lr if lr is None else float(lr),
                      XCFG.cql_target_every if target_every is None else int(target_every))[0]


# ---------- IQL: exact in-sample expectile fixed point ----------
def x_expectile(values: np.ndarray, weights: np.ndarray, tau: float) -> np.ndarray:
    """Exact weighted τ-expectile of every row: the unique v with τ Σ_{x>v} w(x − v) = (1 − τ) Σ_{x<v} w(v − x).

    The condition is piecewise linear and decreasing in v. After sorting each row, the root lies on the bracket
    between the last sorted value where the condition is ≥ 0 and the next one, and is solved there in closed form.
    Rows need non-negative weights with a positive sum."""
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    tau = float(tau)
    if x.ndim != 2 or x.shape != w.shape:
        raise ValueError(f"x_expectile: shapes {x.shape} and {w.shape}")
    if not 0.0 < tau < 1.0:
        raise ValueError(f"expectile {tau} outside (0, 1)")
    if x.shape[0] == 0:
        return np.zeros(0)
    if not (np.isfinite(x).all() and np.isfinite(w).all() and (w >= 0).all() and (w.sum(axis=1) > 0).all()):
        raise ValueError("x_expectile: finite values and non-negative weights with a positive row sum are required")
    K = x.shape[1]
    order = np.argsort(x, axis=1, kind="stable")
    xs = np.take_along_axis(x, order, axis=1)
    ws = np.take_along_axis(w, order, axis=1)
    W_lo = np.cumsum(ws, axis=1)
    S_lo = np.cumsum(ws * xs, axis=1)
    W_hi = W_lo[:, -1:] - W_lo
    S_hi = S_lo[:, -1:] - S_lo
    F = tau * (S_hi - xs * W_hi) - (1.0 - tau) * (xs * W_lo - S_lo)
    neg = F < 0.0
    k = np.clip(np.where(neg.any(axis=1), neg.argmax(axis=1) - 1, K - 1), 0, K - 1)
    r = np.arange(x.shape[0])
    v = ((tau * S_hi[r, k] + (1.0 - tau) * S_lo[r, k])
         / (tau * W_hi[r, k] + (1.0 - tau) * W_lo[r, k]))
    upper = np.where(k + 1 < K, xs[r, np.minimum(k + 1, K - 1)], np.inf)
    return np.clip(v, xs[r, k], upper)


def _x_aw_extraction(q: np.ndarray, v: np.ndarray, counts: np.ndarray, beta: float, max_weight: float,
                     mask: np.ndarray, live: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Part I `iql_extraction` with an explicit mask and live set (AW policy, weighted counts, masked greedy)."""
    S, A = q.shape
    adv = q - v[:, None]
    weights = np.exp(np.clip(beta * adv, -20.0, math.log(max_weight)))
    weighted = counts * weights
    global_pi = counts.sum(axis=0) / max(counts.sum(), 1.0)
    pi = np.tile(global_pi, (S, 1))
    seen = weighted.sum(axis=1) > 0
    pi[seen] = weighted[seen] / weighted[seen].sum(axis=1, keepdims=True)
    adm_w = np.where(mask, weighted, -np.inf)
    no_logged_adm = ~(np.where(mask, weighted, 0.0) > 0).any(axis=1)
    masked_actions = adm_w.argmax(axis=1)
    masked_actions[no_logged_adm] = masked_argmax(np.broadcast_to(global_pi, q.shape), mask)[no_logged_adm]
    return pi, weighted, one_hot_policy(masked_actions, A), int((no_logged_adm & live).sum())


def x_tab_iql(emp: XEmpirical, view: LearnerView, expectile: Optional[float] = None, beta: Optional[float] = None,
              max_weight: Optional[float] = None, support: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Exact tabular IQL fixed point: V(s) = expectile of Q(s,·) under weights N(s,·)·support(s,·), Q = r̂ + P̂V
    on observed pairs (0 elsewhere, as Part I's never-updated entries). `support=None` uses every logged action;
    FISOR-tab passes the admissible mask. States without supported logged actions keep V = 0."""
    tau = XCFG.iql_expectile_default if expectile is None else float(expectile)
    beta = XCFG.iql_beta if beta is None else float(beta)
    max_weight = XCFG.iql_max_weight if max_weight is None else float(max_weight)
    W = emp.N_sa if support is None else emp.N_sa * np.asarray(support, dtype=bool)
    rows = np.flatnonzero(view.live & (W.sum(axis=1) > 0))
    Wr = W[rows]
    obs = emp.observed

    def step(V):
        Q = np.where(obs, emp.r_hat + emp.backup(V), 0.0)
        V_new = np.zeros(view.S)
        V_new[rows] = x_expectile(Q[rows], Wr, tau)
        return Q, V_new

    Q, V, info = _x_iterate(step, view.live)
    resid = float(np.abs(np.where(obs, emp.r_hat + emp.backup(V), 0.0) - Q).max())
    pi, weighted, masked, n_fb = _x_aw_extraction(Q, V, emp.N_sa, beta, max_weight, view.mask, view.live)
    native = one_hot_policy(pi.argmax(axis=1), view.A)
    info.update(expectile=tau, beta=beta, max_weight=max_weight, fixed_point_residual=resid,
                masked_fallback_states=n_fb, support="logged actions" if support is None else "logged admissible actions",
                states_without_supported_action=int((view.live & emp.seen & (W.sum(axis=1) == 0)).sum()))
    return {"q": Q, "v": V, "aw_policy": pi, "aw_scores": weighted, "native": native, "masked": masked,
            "deterministic": native, "masked_greedy": masked, "info": info}


# ---------- support-constrained and pessimistic value iteration ----------
def x_behavior_threshold_set(pi_b: np.ndarray, tau: float, live: np.ndarray) -> np.ndarray:
    """Discrete-BCQ set G(s) = {a : π̂_β(a|s) / max_a' π̂_β(a'|s) ≥ τ} (Part I `behavior_threshold_set`)."""
    allowed = pi_b / np.maximum(pi_b.max(axis=1, keepdims=True), 1e-30) >= tau
    allowed[~live] = True
    return allowed


def x_tab_bcq(emp: XEmpirical, view: LearnerView, tau: Optional[float] = None) -> Dict[str, Any]:
    """Exact tabular discrete BCQ: Q = r̂ + P̂V, V(s) = max over G(s) (Part I extraction semantics)."""
    tau = XCFG.bcq_tau_default if tau is None else float(tau)
    G = x_behavior_threshold_set(emp.pi_b, tau, view.live)
    if not G.any(axis=1).all():
        raise AssertionError("BCQ: empty behaviour-threshold set")

    def step(V):
        Q = emp.r_hat + emp.backup(V)
        return Q, np.where(G, Q, -np.inf).max(axis=1)

    Q, V, info = _x_iterate(step, view.live)
    native = one_hot_policy(masked_argmax(Q, G), view.A)
    both = G & view.mask
    both_ok = both.any(axis=1)
    masked = one_hot_policy(masked_argmax(Q, np.where(both_ok[:, None], both, view.mask)), view.A)
    info.update(tau=tau, masked_fallback_states=int((~both_ok & view.live).sum()),
                mean_allowed_actions=float(G[view.live].sum(axis=1).mean()) if view.live.any() else 0.0)
    return {"q": Q, "v": V, "allowed": G, "native": native, "masked": masked, "deterministic": native,
            "masked_greedy": masked, "info": info}


def x_vi_lcb(emp: XEmpirical, view: LearnerView, c: Optional[float] = None) -> Dict[str, Any]:
    """VI-LCB (Hoeffding form, multiplier c): pessimistic, clipped backups on observed pairs, v_min elsewhere."""
    c = XCFG.lcb_c_default if c is None else float(c)
    lo, hi, live = view.v_min, view.v_max, view.live
    obs = emp.observed & live[:, None]
    base = emp.r_hat - c * (hi - lo) / np.sqrt(np.maximum(emp.N_sa, 1.0))

    def step(V):
        Q = np.where(obs, np.clip(base + emp.backup(V), lo, hi), lo)
        Q[~live] = 0.0
        return Q, Q.max(axis=1)

    Q, V, info = _x_iterate(step, live)
    allow = np.ones_like(view.mask)
    native = one_hot_policy(_x_argmax_tiebreak(Q, allow, emp.pi_b), view.A)
    masked = one_hot_policy(_x_argmax_tiebreak(Q, view.mask, emp.pi_b), view.A)
    info.update(c=c, states_at_floor=int((V[live] <= lo + 1e-12).sum()))
    return {"q": Q, "v": V, "native": native, "masked": masked, "info": info}


def x_mbs_qi(emp: XEmpirical, view: LearnerView, b: Optional[int] = None) -> Dict[str, Any]:
    """MBS-QI: backups through ζ = 1[N ≥ b] only; V_ζ = v_min where no action is supported."""
    b = XCFG.mbsqi_b_default if b is None else int(b)
    live = view.live
    zeta = (emp.N_sa >= b) & live[:, None]
    has = zeta.any(axis=1)

    def step(V):
        Q = emp.r_hat + emp.backup(V)
        return Q, np.where(has, np.where(zeta, Q, -np.inf).max(axis=1), view.v_min)

    Q, V, info = _x_iterate(step, live)
    bc = x_bc(emp, view)
    act = np.where(has, masked_argmax(Q, np.where(has[:, None], zeta, True)), bc["native"].argmax(axis=1))
    zm = zeta & view.mask
    hm = zm.any(axis=1)
    act_m = np.where(hm, masked_argmax(Q, np.where(hm[:, None], zm, True)), bc["masked"].argmax(axis=1))
    info.update(b=b, unsupported_live_states=int((~has & live).sum()),
                masked_fallback_states=int((~hm & live).sum()))
    return {"q": np.where(zeta, Q, view.v_min), "v": V, "zeta": zeta, "native": one_hot_policy(act, view.A),
            "masked": one_hot_policy(act_m, view.A), "info": info}


def x_spibb(emp: XEmpirical, view: LearnerView, n_wedge: Optional[int] = None, max_iter: int = 100) -> Dict[str, Any]:
    """Π_b-SPIBB (Laroche et al. 2019) with exact policy evaluation on P̂ (closed classes valued 0)."""
    n_wedge = XCFG.spibb_n_default if n_wedge is None else int(n_wedge)
    S, live, pb = view.S, view.live, emp.pi_b
    boot = emp.N_sa < n_wedge
    boot[~live] = True
    free_mask = ~boot
    has_free = free_mask.any(axis=1)
    kept = np.where(boot, pb, 0.0)
    free = 1.0 - kept.sum(axis=1)
    if np.abs(free[~has_free]).max(initial=0.0) > 1e-12:
        raise AssertionError("SPIBB: all-bootstrapped rows must keep the behaviour policy exactly")
    free[~has_free] = 0.0
    allowed = np.where(has_free[:, None], free_mask, True)
    idx = np.arange(S)
    r_hat = emp.r_hat
    pi, act, converged, it, recurrent = pb.copy(), None, False, 0, 0
    Q = V = None
    for it in range(1, max_iter + 1):
        P_LL, L = _x_chain(emp, live, pi)
        sol = _x_markov_solve(P_LL, (pi * r_hat).sum(axis=1)[L])
        recurrent = max(recurrent, sol["recurrent_states"])
        V = np.zeros(S)
        V[L] = sol["v"]
        Q = r_hat + emp.backup(V)
        new = masked_argmax(Q, allowed)
        if act is not None:
            q_new = Q[idx, new]
            keep = Q[idx, act] >= q_new - _X_TIE_RTOL * np.maximum(1.0, np.abs(q_new))
            new = np.where(keep, act, new)
            if np.array_equal(new[has_free], act[has_free]):
                converged = True
                break
        act = new
        pi = kept.copy()
        pi[has_free, act[has_free]] += free[has_free]
    lb = boot & live[:, None]
    if not np.array_equal(pi[lb], pb[lb]):
        raise AssertionError("SPIBB: a bootstrapped pair does not keep its behaviour probability")
    off = pi.copy()
    off[lb] = 0.0
    if has_free.any() and not np.allclose(off[has_free].sum(axis=1), free[has_free], atol=1e-12):
        raise AssertionError("SPIBB: the free mass is not placed on non-bootstrapped pairs")
    info = {"n_wedge": n_wedge, "iterations": it, "converged": converged,
            "states_with_free_mass": int((has_free & live).sum()),
            "bootstrapped_live_pairs": int(lb.sum()), "max_recurrent_states": recurrent,
            "mean_free_mass": float(free[has_free & live].mean()) if (has_free & live).any() else 0.0}
    return {"pi": pi, "q": Q, "v": V, "bootstrapped": boot, "info": info}


# ---------- model-based constrained methods on the completed empirical model ----------
def _x_model_reward_cost(emp: XEmpirical, view: LearnerView) -> Tuple[np.ndarray, np.ndarray]:
    """Expected rewards of `emp.completed_dense(view.sink)` (unobserved live pairs pay the termination reward) and
    the violation cost c = 1[a ∉ M] on live states."""
    live = view.live
    miss = ~emp.observed & live[:, None]
    r = np.where(miss, view.terminate_reward, emp.r_hat)
    r[~live] = 0.0
    cost = (~view.mask & live[:, None]).astype(np.float64)
    return r, cost


def _x_model_eval(emp: XEmpirical, view: LearnerView, pi: np.ndarray, r_model: np.ndarray,
                  cost: np.ndarray) -> Dict[str, Any]:
    """Exact evaluation from d̂₀ on the completed model. Its sparse form is P̂ plus a jump to the sink for
    unobserved pairs, which leaves the live set."""
    P_LL, L = _x_chain(emp, view.live, pi)
    sol = _x_markov_solve(P_LL, (pi * r_model).sum(axis=1)[L], emp.d0_hat[L], (pi * cost).sum(axis=1)[L])
    return {"return": sol["return"], "violations": sol["violations"], "trapped": sol["trapped"],
            "recurrent_states": sol["recurrent_states"]}


def _x_lagrangian_greedy(emp: XEmpirical, view: LearnerView, lam: float, r_model: np.ndarray, cost: np.ndarray,
                         evals: Dict[bytes, Dict[str, Any]]) -> Dict[str, Any]:
    """Greedy VI policy of r − λc on the completed model (admissible actions win exact ties). Value iteration
    always starts from V = 0: with γ = 1 a warm start could keep a stale value on a zero-reward closed cycle.
    Model evaluations are cached per deterministic policy (`evals`), because the greedy policy is piecewise
    constant in λ."""
    base = r_model - lam * cost
    live = view.live

    def step(V):
        Q = base + emp.backup(V)      # unobserved rows of P̂ are zero and V(sink) = 0: the completion's backup
        Q[~live] = 0.0
        return Q, Q.max(axis=1)

    Q, _, vi = _x_iterate(step, live)
    act = _x_argmax_tiebreak(Q, np.ones_like(view.mask), view.mask.astype(np.float64))
    key = act.astype(np.int64).tobytes()
    pi = one_hot_policy(act, view.A)
    if key not in evals:
        evals[key] = _x_model_eval(emp, view, pi, r_model, cost)
    ev = evals[key]
    return {"pi": pi, "lam": float(lam), "violations": ev["violations"], "model_return": ev["return"],
            "trapped": ev["trapped"], "vi": vi}


def x_lagrangian_mb(emp: XEmpirical, view: LearnerView, kappa: Optional[float] = None,
                    cache: Optional[Dict[str, Dict[Any, Dict[str, Any]]]] = None, rel_tol: float = 1e-6,
                    max_bisection: int = 60) -> Dict[str, Any]:
    """Lagrangian model-based policy: the smallest λ ∈ [0, λ_max] (bisection) whose greedy policy of r − λc meets
    E_model[violations] ≤ κ from d̂₀. λ_max = (v_max − v_min) + 1 makes every admissible action strictly preferred.
    `cache` ({"greedy": λ → result, "eval": policy → model evaluation}) may be shared across κ on the same data."""
    kappa = XCFG.lagrange_kappa_default if kappa is None else float(kappa)
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    r_model, cost = _x_model_reward_cost(emp, view)
    cache = {} if cache is None else cache
    greedy, evals = cache.setdefault("greedy", {}), cache.setdefault("eval", {})

    def at(lam: float) -> Dict[str, Any]:
        if lam not in greedy:
            greedy[lam] = _x_lagrangian_greedy(emp, view, lam, r_model, cost, evals)
        return greedy[lam]

    def feasible(res: Dict[str, Any]) -> bool:
        return res["violations"] <= kappa + 1e-12

    lam_max = (view.v_max - view.v_min) + 1.0
    lo, hi, steps = 0.0, lam_max, 0
    lo_res = None
    if feasible(at(0.0)):
        chosen = at(0.0)
    else:
        lo_res = at(0.0)
        if not feasible(at(lam_max)):
            raise AssertionError(f"Lagrangian: lambda_max={lam_max} still violates ({at(lam_max)['violations']})")
        while hi - lo > rel_tol * lam_max and steps < max_bisection:
            mid = 0.5 * (lo + hi)
            steps += 1
            if feasible(at(mid)):
                hi = mid
            else:
                lo, lo_res = mid, at(mid)
        chosen = at(hi)
    used = [g["vi"] for g in greedy.values()]
    info = {"kappa": kappa, "lambda": chosen["lam"], "lambda_max": lam_max, "bisection_steps": steps,
            "model_violations": chosen["violations"], "model_return": chosen["model_return"],
            "trapped_probability": chosen["trapped"],
            "infeasible_side_violations": None if lo_res is None else lo_res["violations"],
            "converged": all(u["converged"] for u in used),
            "vi_converged_all": all(u["converged"] for u in used), "vi_iterations_max": max(u["iterations"] for u in used),
            "greedy_solves": len(greedy), "distinct_policies_evaluated": len(evals)}
    return {"pi": chosen["pi"], "info": info}


def x_cmdp_lp_model(emp: XEmpirical, view: LearnerView, kappa: Optional[float] = None,
                    P_model: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Exact constrained LP on the completed empirical model (`x_cmdp_lp`, start distribution d̂₀).

    An infeasible LP is a legitimate empirical outcome: some closed class of admissible observed pairs may be
    unavoidable. It is reported (status) and counted, and the policy falls back to masked BC."""
    kappa = XCFG.mblp_kappa_default if kappa is None else float(kappa)
    P = emp.completed_dense(view.sink) if P_model is None else P_model
    r_model, cost = _x_model_reward_cost(emp, view)
    fallback = x_bc(emp, view)["masked"]
    try:
        pi, rep = x_cmdp_lp(P, r_model, emp.d0_hat, view.live, cost, kappa, fallback=fallback)
        status = "ok"
    except RuntimeError as exc:
        pi, rep, status = fallback.copy(), {"message": str(exc)[:160]}, "lp_failed_masked_bc_fallback"
    ev = _x_model_eval(emp, view, pi, r_model, cost)
    ok = status == "ok"
    info = {"kappa": kappa, "status": status, "lp_objective": rep.get("objective"), "lp_violations": rep.get("violations"),
            "model_return": ev["return"], "model_violations": ev["violations"],
            "lp_eval_gap": abs(ev["return"] - rep["objective"]) if ok else None,
            "zero_occupancy_live_states": int(view.live.sum()) - rep["reachable_live_states"] if ok else None,
            "randomised_states": rep.get("randomised_states"), "solve_s": rep.get("solve_s"),
            "message": rep.get("message")}
    return {"pi": pi, "info": info}


# ---------- behaviour prior x critic (exact, no sampling) ----------
def _x_tilted(pb: np.ndarray, log_w: np.ndarray, allowed: Optional[np.ndarray], live: np.ndarray) -> Tuple[np.ndarray, int]:
    """π ∝ π̂_β·exp(log_w) on `allowed` (log space; exact zeros where π̂_β = 0 or not allowed). Live rows without
    mass become uniform over `allowed` (counted); terminal rows keep the unrestricted tilt."""
    allow = np.ones(pb.shape, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    ok = (pb > 0) & allow
    empty = ~ok.any(axis=1)
    if allowed is None and empty.any():
        raise AssertionError("behaviour prior has an empty row")
    logits = np.where(ok, np.log(np.where(ok, pb, 1.0)) + log_w, -np.inf)
    logits[empty] = np.where(allow[empty], 0.0, -np.inf)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p, int((empty & live).sum())


def x_sfbc(emp: XEmpirical, view: LearnerView, alpha: Optional[float] = None,
           iql: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Exact SfBC: π ∝ π̂_β·exp(α·A) with A = Q − V of IQL at the default expectile (the limit of SfBC's
    candidate resampling with infinitely many behaviour candidates)."""
    alpha = XCFG.sfbc_alpha_default if alpha is None else float(alpha)
    iql = x_tab_iql(emp, view) if iql is None else iql
    tilt = alpha * (iql["q"] - iql["v"][:, None])
    native, _ = _x_tilted(emp.pi_b, tilt, None, view.live)
    masked, n_fb = _x_tilted(emp.pi_b, tilt, view.mask, view.live)
    return {"native": native, "masked": masked, "argmax": one_hot_policy(native.argmax(axis=1), view.A),
            "info": {"alpha": alpha, "expectile": iql["info"]["expectile"], "masked_fallback_states": n_fb,
                     "converged": bool(iql["info"]["converged"])}}


def x_idql(emp: XEmpirical, view: LearnerView, iql: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Exact IDQL implicit policy: π ∝ π̂_β·|τ − 1(Q < V)| at the IQL expectile τ."""
    iql = x_tab_iql(emp, view) if iql is None else iql
    tau = iql["info"]["expectile"]
    w = np.abs(tau - (iql["q"] < iql["v"][:, None]).astype(np.float64))
    native, _ = _x_tilted(emp.pi_b, np.log(w), None, view.live)
    masked, n_fb = _x_tilted(emp.pi_b, np.log(w), view.mask, view.live)
    return {"native": native, "masked": masked, "info": {"expectile": tau, "masked_fallback_states": n_fb,
                                                         "converged": bool(iql["info"]["converged"])}}


def x_fisor_tab(emp: XEmpirical, view: LearnerView, alpha: Optional[float] = None,
                critic: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """FISOR-style feasibility tilt: π ∝ π̂_β·exp(α·A_feas)·1[a ∈ M], with A_feas from the admissible-support
    IQL critic."""
    alpha = XCFG.sfbc_alpha_default if alpha is None else float(alpha)
    critic = x_tab_iql(emp, view, support=view.mask) if critic is None else critic
    if critic["info"]["support"] != "logged admissible actions":
        raise ValueError("FISOR-tab needs the admissible-support critic")
    pi, n_fb = _x_tilted(emp.pi_b, alpha * (critic["q"] - critic["v"][:, None]), view.mask, view.live)
    return {"pi": pi, "info": {"alpha": alpha, "expectile": critic["info"]["expectile"], "uniform_fallback_states": n_fb,
                               "states_without_admissible_logged_action":
                                   critic["info"]["states_without_supported_action"]}}


# ---------- exact categorical diffusion with the Bayes-optimal denoiser ----------
_X_DIFFUSION_CACHE: Dict[tuple, Any] = {}


def x_diffusion_for(A: int) -> "CategoricalDiffusion":
    """Cached uniform-corruption schedule of Part I (Section 12) for an action set of size A."""
    key = (int(A), int(CFG.diffusion_T), float(CFG.diffusion_beta_start), float(CFG.diffusion_beta_end))
    if key not in _X_DIFFUSION_CACHE:
        _X_DIFFUSION_CACHE[key] = CategoricalDiffusion(*key)
    return _X_DIFFUSION_CACHE[key]


def x_bayes_reverse_kernels(pi_b: np.ndarray, diffusion: "CategoricalDiffusion") -> Dict[int, np.ndarray]:
    """Exact Bayes-optimal reverse kernels of the uniform-corruption chain started at pi_b (per state):

    B_t(x_{t−1} | x_t, s) = Σ_{x0} p(x0 | x_t, s)·q(x_{t−1} | x_t, x0),
    with p(x0 | x_t, s) ∝ pi_b(x0 | s)·Q̄_t(x0, x_t).
    Returns {t: [S, K(x_t), K(x_{t−1})]} for t = 1..T; every row sums to one."""
    K = diffusion.K
    pb = validate_policy(pi_b, "diffusion behaviour prior", shape=(np.asarray(pi_b).shape[0], K))
    out: Dict[int, np.ndarray] = {}
    for t in range(1, diffusion.T + 1):
        joint = pb[:, None, :] * diffusion.Qbar[t].T[None, :, :]              # [S, x_t, x0]
        marg = joint.sum(axis=2, keepdims=True)                                # q_t(x_t | s) = (pi_b Q̄_t)(x_t)
        if not (marg > 0).all():
            raise AssertionError("forward marginal has a zero entry (Q̄_t must be strictly positive for t >= 1)")
        post = joint / marg
        B = np.ascontiguousarray(np.matmul(post.transpose(1, 0, 2), diffusion.posterior[t]).transpose(1, 0, 2))
        if not (np.allclose(B.sum(axis=2), 1.0, atol=1e-12) and (B >= -1e-15).all()):
            raise AssertionError(f"reverse kernel at t={t} is not a stochastic matrix")
        out[t] = np.clip(B, 0.0, None)
    return out


def x_guided_reverse(kernels: Dict[int, np.ndarray], critic_G: Optional[np.ndarray], guidance: float,
                     mask: np.ndarray, live: np.ndarray, init: Any = "uniform_admissible",
                     mask_mode: str = "every_step", guidance_mode: str = "every_step",
                     deterministic: bool = True) -> Tuple[np.ndarray, List[float]]:
    """Generic port of Part I `guided_reverse_policy`: exact marginal propagation of the reverse chain over all states.

    Same semantics: logits = log max(B_t, 1e-30) + g·G(s, x_{t−1}) when guided, −∞ outside the mask when masked,
    the marginal is re-masked and renormalised, and the trace records the maximum inadmissible mass on live states.
    `init` may also be an explicit [S, K] start distribution (used to verify the Bayes kernels)."""
    assert mask_mode in {"every_step", "final_only", "none"}
    assert guidance_mode in {"every_step", "final_only", "none"}
    if guidance_mode == "every_step" and guidance != 0 and mask_mode != "every_step":
        raise ValueError("Every-step guidance is only defined together with every-step masking.")
    T = max(kernels)
    if sorted(kernels) != list(range(1, T + 1)):
        raise ValueError("kernels must be indexed 1..T")
    S, K, _ = kernels[T].shape
    m = np.asarray(mask, dtype=bool)
    live = np.asarray(live, dtype=bool)
    if m.shape != (S, K) or not m.any(axis=1).all():
        raise ValueError("mask must be [S, K] with a non-empty row everywhere")
    if isinstance(init, str):
        assert init in {"uniform_admissible", "uniform_all"}
        p = m / m.sum(axis=1, keepdims=True) if init == "uniform_admissible" else np.full((S, K), 1.0 / K)
    else:
        p = validate_policy(init, "reverse-chain start distribution", shape=(S, K)).copy()
    if critic_G is not None and (np.shape(critic_G) != (S, K) or not np.isfinite(critic_G).all()):
        raise ValueError("critic_G must be a finite [S, K] array")
    trace: List[float] = []
    for t in range(T, 0, -1):
        logits = np.log(np.maximum(kernels[t], 1e-30))          # same floor as Part I's ReverseCache
        g = guidance if (guidance_mode == "every_step" or (guidance_mode == "final_only" and t == 1)) else 0.0
        if g != 0.0 and critic_G is not None:
            logits = logits + g * critic_G[:, None, :]
        masked_now = mask_mode == "every_step" or (mask_mode == "final_only" and t == 1)
        if masked_now:
            logits = np.where(m[:, None, :], logits, -np.inf)
        logits = logits - logits.max(axis=2, keepdims=True)
        kernel = np.exp(logits)
        kernel /= kernel.sum(axis=2, keepdims=True)
        p = np.matmul(p[:, None, :], kernel)[:, 0, :]
        if masked_now:
            p = np.where(m, p, 0.0)
        p = p / p.sum(axis=1, keepdims=True)
        trace.append(float(np.where(m, 0.0, p)[live].sum(axis=1).max()) if live.any() else 0.0)
    policy = one_hot_policy(p.argmax(axis=1), K) if deterministic else p
    if mask_mode != "none" and live.any():
        assert np.all(policy[live][~m[live]] <= 1e-14), "Masked reverse inference produced mass outside the mask."
        assert float(np.where(m, 0.0, p)[live].sum(axis=1).max()) <= 1e-12
    return policy, trace


print("Part II learners ready.")

# %% [markdown]
# ### II.4b Method catalogue and the per-draw method runner
#
# `run_xmethods(view, train, key)` fits the full suite on one training split. It returns records
# {method, variant, hp, is_default, family, pi, info}, where `pi` is validated with `validate_policy` and `info` is a
# small JSON-able dict (convergence, fall-back counts, model-side constraint diagnostics, fit time).
#
# Fits that several methods share are computed once per draw:
# * the default-expectile IQL critic is shared by IQL (AW stochastic), SfBC, IDQL and diffusion guidance;
# * the λ cache is shared across the Lagrangian κ grid;
# * the dense completed model is shared across the LP κ grid;
# * the Bayes kernels are shared by every diffusion variant.
#
# Two exactness checks of Proposition P7 run on every draw:
# * the reverse chain started at q_T = π̂_β Q̄_T returns π̂_β to 1e-10;
# * the unmasked chain started from uniform satisfies TV(output, π̂_β) ≤ ᾱ_T·TV(u, π̂_β).
#
# Every difference between a Part II diffusion policy and its behaviour prior is therefore caused by the inference
# modifications (admissible start, mask, guidance, argmax extraction), not by denoiser error. Part I's learned
# denoiser cannot separate the two.
#
# `X_METHOD_CATALOG` (saved as `xenv/xenv_method_catalog.csv`) documents each method: family, information used,
# use of the mask, hyper-parameter grid with its pre-declared default, and reference.

# %%
def x_method_table() -> List[Dict[str, Any]]:
    """Catalogue of the Part II method suite (family labels used by `run_xmethods`; grids read from XCFG)."""
    c = XCFG
    fmt = lambda t: ", ".join(str(v) for v in t)
    row = lambda method, family, variants, info, mask_use, hp, grid, default, ref: {
        "method": method, "family": family, "variants": variants, "information_used": info, "mask_use": mask_use,
        "hyperparameter": hp, "grid": grid, "default": default, "reference": ref}
    counts, model = "logged (s, a) counts", "empirical model (counts, P̂, r̂)"
    return [
        row("BC (stochastic)", "behaviour", "native", counts, "none", "", "", "", "behaviour cloning (empirical MLE)"),
        row("BC", "behaviour", "native, masked", counts, "extraction (masked)", "", "", "",
            "behaviour cloning (empirical MLE)"),
        row("Uniform-admissible", "behaviour", "native", "admissible mask only", "definition", "", "", "",
            "trivial admissible baseline"),
        row("CQL", "value", "native, masked", model, "extraction (masked)", "alpha", fmt(c.cql_alpha_grid),
            str(c.cql_alpha_default), "Kumar et al. 2020 (NeurIPS)"),
        row("IQL", "value", "native, masked", model, "extraction (masked)", "expectile", fmt(c.iql_expectile_grid),
            str(c.iql_expectile_default), "Kostrikov et al. 2022 (ICLR)"),
        row("IQL (AW stochastic)", "value", "native", model, "none", "expectile (fixed)", "",
            str(c.iql_expectile_default), "Kostrikov et al. 2022 (ICLR)"),
        row("BCQ", "pessimistic", "native, masked", model + " + behaviour ratio", "extraction (masked)", "tau",
            fmt(c.bcq_tau_grid), str(c.bcq_tau_default), "Fujimoto et al. 2019 (discrete BCQ)"),
        row("VI-LCB", "pessimistic", "native, masked", model + " + value range", "extraction (masked)", "c",
            fmt(c.lcb_c_grid), str(c.lcb_c_default), "Rashidinejad et al. 2021 (NeurIPS); Jin et al. 2021 (PEVI)"),
        row("MBS-QI", "pessimistic", "native, masked", model, "extraction (masked)", "b", fmt(c.mbsqi_b_grid),
            str(c.mbsqi_b_default), "Liu et al. 2020 (NeurIPS)"),
        row("SPIBB", "pessimistic", "native (stochastic)", model + " + behaviour policy", "none (projected by the driver)",
            "N_wedge", fmt(c.spibb_n_grid), str(c.spibb_n_default), "Laroche et al. 2019 (ICML)"),
        row("Lagrangian-MB", "constrained", "native", model + " + pessimistic completion + d̂0 + mask cost",
            "constraint (expected violations)", "kappa", fmt(c.lagrange_kappa_grid), str(c.lagrange_kappa_default),
            "Altman 1999; Le, Voloshin & Yue 2019 (Batch Policy Learning under Constraints, ICML); "
            "Tessler et al. 2019 (RCPO)"),
        row("CMDP-LP (model)", "constrained", "native", model + " + pessimistic completion + d̂0 + mask cost",
            "constraint (expected violations)", "kappa", fmt(c.mblp_kappa_grid), str(c.mblp_kappa_default),
            "Altman 1999; COptiDICE (Lee et al. 2022, ICLR)"),
        row("SfBC (exact)", "guided-prior", "native, masked", model + " + behaviour prior", "extraction (masked)",
            "alpha", fmt(c.sfbc_alpha_grid), str(c.sfbc_alpha_default), "Chen et al. 2023 (ICLR)"),
        row("SfBC (exact, argmax)", "guided-prior", "native", model + " + behaviour prior", "none", "alpha",
            fmt(c.sfbc_alpha_grid), str(c.sfbc_alpha_default), "Chen et al. 2023 (ICLR)"),
        row("IDQL (exact)", "guided-prior", "native, masked", model + " + behaviour prior", "extraction (masked)",
            "expectile (fixed)", "", str(c.iql_expectile_default), "Hansen-Estruch et al. 2023"),
        row("FISOR-tab (exact)", "constrained", "native", model + " + behaviour prior + mask",
            "critic support + extraction", "alpha", fmt(c.sfbc_alpha_grid), str(c.sfbc_alpha_default),
            "Zheng et al. 2024 (ICLR)"),
        row("Diffusion (exact, unguided)", "diffusion", "native", "behaviour prior (exact Bayes denoiser)",
            "inference (every reverse step)", "", "", "", "Austin et al. 2021 (D3PM); Part I Section 12"),
        row("Diffusion (exact, guided)", "diffusion", "native", "behaviour prior + IQL critic",
            "inference (every reverse step)", "guidance g", fmt(c.diffusion_guidance_grid),
            str(c.diffusion_guidance_default), "Dhariwal & Nichol 2021 (guidance); Part I Section 12"),
        row("Diffusion (exact, final-only)", "diffusion", "native", "behaviour prior + IQL critic",
            "inference (every reverse step); guidance at the final step only", "guidance g (fixed)", "",
            str(c.diffusion_guidance_default), "Part I Section 12"),
        row("Diffusion (exact, unmasked)", "diffusion", "native (stochastic)", "behaviour prior (exact Bayes denoiser)",
            "none", "", "", "", "Proposition P7 control"),
    ]


def run_xmethods(view: LearnerView, train: Dict[str, np.ndarray], key: str) -> List[dict]:
    """Fit the Part II method suite on `train` (Section II.4 contract) and return validated policy records."""
    S, A, live, M = view.S, view.A, view.live, view.mask
    emp = XEmpirical(train, S, A, view.terminals)
    if emp.n == 0:
        raise ValueError(f"{key}: empty training data")
    if not np.array_equal(emp.live, live):
        raise ValueError(f"{key}: the learner view and the data disagree on the absorbing states")
    if not live[train["state"]].all():
        raise ValueError(f"{key}: decision states must be live")
    if not np.isclose(emp.d0_hat.sum(), 1.0, atol=1e-12):
        raise ValueError(f"{key}: training data must contain complete episodes (timestep 0 rows)")
    for name in ("cql_alpha", "iql_expectile", "bcq_tau", "lcb_c", "mbsqi_b", "spibb_n", "lagrange_kappa",
                 "mblp_kappa", "sfbc_alpha", "diffusion_guidance"):
        grid, default = getattr(XCFG, f"{name}_grid"), getattr(XCFG, f"{name}_default")
        if default not in grid:
            raise ValueError(f"XCFG.{name}_default={default} is not in its grid {grid}")
    family = {r["method"]: r["family"] for r in x_method_table()}
    records: List[dict] = []
    clock = {"t": time.perf_counter()}

    def lap() -> float:
        now = time.perf_counter()
        dt, clock["t"] = now - clock["t"], now
        return dt

    def add(method: str, variant: str, pi: np.ndarray, hp: Optional[dict] = None, default: bool = True,
            info: Optional[dict] = None, admissible: bool = False) -> None:
        hp = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in (hp or {}).items()}
        label = x_label(method, variant, hp, default)
        pi = validate_policy(pi, f"{key} | {label}", shape=(S, A))
        if (variant == "masked" or admissible) and live.any():
            outside = float(x_inadmissible_mass(pi, M, live).max())
            if outside != 0.0:
                raise AssertionError(f"{key} | {label}: mass {outside:.3e} outside the admissible mask")
        records.append({"method": method, "variant": variant, "hp": hp, "is_default": bool(default),
                        "family": family[method], "pi": pi, "info": _x_info(info or {})})

    # behaviour references
    bc = x_bc(emp, view)
    t = lap()
    add("BC (stochastic)", "native", bc["stochastic"], info={**bc["info"], "fit_s": t})
    add("BC", "native", bc["native"], info={**bc["info"], "fit_s": t})
    add("BC", "masked", bc["masked"], info={**bc["info"], "fit_s": t})
    add("Uniform-admissible", "native", x_uniform_admissible(M, live), admissible=True)

    # value-based
    cql = _x_cql_fit(emp, view, XCFG.cql_alpha_grid, XCFG.cql_steps, XCFG.cql_lr, XCFG.cql_target_every)
    t = lap() / len(cql)
    for a, res in zip(XCFG.cql_alpha_grid, cql):
        d = a == XCFG.cql_alpha_default
        for var in ("native", "masked"):
            add("CQL", var, res[var], {"alpha": a}, d, {**res["info"], "fit_s": t})
    del cql
    iql_default = None
    for e in XCFG.iql_expectile_grid:
        res = x_tab_iql(emp, view, e)
        t = lap()
        d = e == XCFG.iql_expectile_default
        for var in ("native", "masked"):
            add("IQL", var, res[var], {"expectile": e}, d, {**res["info"], "fit_s": t})
        if d:
            iql_default = res
    add("IQL (AW stochastic)", "native", iql_default["aw_policy"], info=iql_default["info"])

    # support-constrained / pessimistic
    for name, method, fn, grid, dflt, hp_name in (
            ("bcq", "BCQ", x_tab_bcq, XCFG.bcq_tau_grid, XCFG.bcq_tau_default, "tau"),
            ("lcb", "VI-LCB", x_vi_lcb, XCFG.lcb_c_grid, XCFG.lcb_c_default, "c"),
            ("mbsqi", "MBS-QI", x_mbs_qi, XCFG.mbsqi_b_grid, XCFG.mbsqi_b_default, "b")):
        for v in grid:
            res = fn(emp, view, v)
            t = lap()
            for var in ("native", "masked"):
                add(method, var, res[var], {hp_name: v}, v == dflt, {**res["info"], "fit_s": t})
    for nw in XCFG.spibb_n_grid:
        res = x_spibb(emp, view, nw)
        add("SPIBB", "native", res["pi"], {"n_wedge": nw}, nw == XCFG.spibb_n_default, {**res["info"], "fit_s": lap()})

    # constrained, model-based (completed empirical model)
    lam_cache: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    for k in XCFG.lagrange_kappa_grid:
        res = x_lagrangian_mb(emp, view, k, cache=lam_cache)
        add("Lagrangian-MB", "native", res["pi"], {"kappa": k}, k == XCFG.lagrange_kappa_default,
            {**res["info"], "fit_s": lap()})
    del lam_cache
    P_model = emp.completed_dense(view.sink)
    for k in XCFG.mblp_kappa_grid:
        res = x_cmdp_lp_model(emp, view, k, P_model=P_model)
        add("CMDP-LP (model)", "native", res["pi"], {"kappa": k}, k == XCFG.mblp_kappa_default,
            {**res["info"], "fit_s": lap()})
    del P_model
    gc.collect()

    # behaviour prior x critic (exact)
    for a in XCFG.sfbc_alpha_grid:
        res = x_sfbc(emp, view, a, iql=iql_default)
        t = lap()
        d = a == XCFG.sfbc_alpha_default
        add("SfBC (exact)", "native", res["native"], {"alpha": a}, d, {**res["info"], "fit_s": t})
        add("SfBC (exact)", "masked", res["masked"], {"alpha": a}, d, {**res["info"], "fit_s": t})
        add("SfBC (exact, argmax)", "native", res["argmax"], {"alpha": a}, d, {**res["info"], "fit_s": t})
    res = x_idql(emp, view, iql=iql_default)
    t = lap()
    add("IDQL (exact)", "native", res["native"], info={**res["info"], "fit_s": t})
    add("IDQL (exact)", "masked", res["masked"], info={**res["info"], "fit_s": t})
    feas = x_tab_iql(emp, view, XCFG.iql_expectile_default, support=M)
    t_feas = lap()
    for a in XCFG.sfbc_alpha_grid:
        res = x_fisor_tab(emp, view, a, critic=feas)
        add("FISOR-tab (exact)", "native", res["pi"], {"alpha": a}, a == XCFG.sfbc_alpha_default,
            {**res["info"], "critic_converged": feas["info"]["converged"], "converged": feas["info"]["converged"],
             "fit_s": t_feas + lap()}, admissible=True)
    del feas

    # exact categorical diffusion with the Bayes-optimal denoiser of the empirical behaviour policy
    diff = x_diffusion_for(A)
    pb = emp.pi_b
    kernels = x_bayes_reverse_kernels(pb, diff)
    q_T = pb @ diff.Qbar[diff.T]
    back, _ = x_guided_reverse(kernels, None, 0.0, M, live, init=q_T, mask_mode="none", guidance_mode="none",
                               deterministic=False)
    roundtrip = float(np.abs(back - pb).max())
    if roundtrip > 1e-10:
        raise AssertionError(f"{key}: Bayes reverse chain from q_T does not return the behaviour prior ({roundtrip:.2e})")
    abar = float(diff.alpha_bar[-1])
    dinfo = {"T": diff.T, "alpha_bar_T": abar, "bayes_roundtrip_max_abs_err": roundtrip, "fit_s": lap()}
    G = normalize_critic_per_state(iql_default["q"], M)
    pi, tr = x_guided_reverse(kernels, None, 0.0, M, live)
    add("Diffusion (exact, unguided)", "native", pi, info={**dinfo, "g": 0.0, "max_trace_inadmissible": max(tr)},
        admissible=True)
    for g in XCFG.diffusion_guidance_grid:
        pi, tr = x_guided_reverse(kernels, G, g, M, live)
        add("Diffusion (exact, guided)", "native", pi, {"g": g}, g == XCFG.diffusion_guidance_default,
            {**dinfo, "max_trace_inadmissible": max(tr), "converged": iql_default["info"]["converged"]}, admissible=True)
    pi, tr = x_guided_reverse(kernels, G, XCFG.diffusion_guidance_default, M, live, guidance_mode="final_only")
    add("Diffusion (exact, final-only)", "native", pi,
        info={**dinfo, "g": XCFG.diffusion_guidance_default, "max_trace_inadmissible": max(tr),
              "converged": iql_default["info"]["converged"]}, admissible=True)
    pu, _ = x_guided_reverse(kernels, None, 0.0, M, live, init="uniform_all", mask_mode="none", guidance_mode="none",
                             deterministic=False)
    u = np.full(A, 1.0 / A)
    tv_out = 0.5 * np.abs(pu - pb).sum(axis=1)[live]
    tv_bound = abar * 0.5 * np.abs(u[None, :] - pb).sum(axis=1)[live]
    if live.any() and not (tv_out <= tv_bound + 1e-12).all():
        raise AssertionError(f"{key}: unmasked reverse chain violates TV(out, prior) <= alpha_bar_T TV(u, prior)")
    add("Diffusion (exact, unmasked)", "native", pu,
        info={**dinfo, "tv_to_prior_max": float(tv_out.max()) if live.any() else 0.0,
              "tv_bound_max": float(tv_bound.max()) if live.any() else 0.0, "tv_bound_global": abar * (1.0 - 1.0 / A)})
    del kernels
    return records


X_METHOD_CATALOG = pd.DataFrame(x_method_table())
X_METHOD_CATALOG.to_csv(XENV_TABLE_DIR / "xenv_method_catalog.csv", index=False)
display(X_METHOD_CATALOG[["method", "family", "variants", "mask_use", "hyperparameter", "grid", "default"]])

# %% [markdown]
# ### II.4c Self-test of the method suite (tiny random absorbing CMDP, seconds)
#
# This test checks the exact solvers against independent computations before any Part II result is produced. It
# draws a random 12-state, 4-action absorbing MDP with a random mask, rewards +1 and −1 on entry to two absorbing
# states, and data from a random behaviour policy. The learners see only the data and the `LearnerView`; the check
# code alone uses the generating MDP. The following must hold, otherwise the cell raises:
#
# * the expectile solver matches bisection;
# * IQL satisfies its fixed-point equations;
# * BCQ, VI-LCB and MBS-QI have Bellman residuals at the iteration tolerance;
# * SPIBB keeps every bootstrapped pair at π̂_β and is policy-stable;
# * the Lagrangian policy meets its model budget, and the infeasible side of the bisection violates it;
# * the sparse completed-model evaluation equals a dense solve on `XEmpirical.completed_dense`;
# * the LP objective equals the exact model value of the LP policy;
# * the Bayes reverse kernels are exact;
# * every masked record has zero inadmissible mass.
#
# The result is saved to `xenv/xenv_methods_selftest.json`.

# %%
def x_methods_selftest(n_live: int = 12, n_actions: int = 4, n_episodes: int = 600) -> Dict[str, Any]:
    rng = np_rng("xenv", "methods-selftest")
    A = n_actions
    S, goal, death = n_live + 2, n_live, n_live + 1
    P = np.zeros((S, A, S))
    for s in range(n_live):
        for a in range(A):
            succ = rng.choice(n_live, size=3, replace=False)
            h, q = rng.uniform(0.1, 0.3), rng.uniform()
            P[s, a, succ] = (1.0 - h) * rng.dirichlet(np.ones(3))
            P[s, a, goal] += h * q
            P[s, a, death] += h * (1.0 - q)
    P[goal, :, goal] = 1.0
    P[death, :, death] = 1.0
    entry = np.zeros(S)
    entry[goal], entry[death] = 1.0, -1.0
    d0 = np.zeros(S)
    d0[:4] = 0.25
    mask = rng.random((S, A)) < 0.6
    mask[np.arange(n_live), rng.integers(0, A, n_live)] = True
    env = FiniteEnv(name="methods-selftest", kind="variant", P=P, entry_reward=entry, d0=d0, terminals=(goal, death),
                    sink=death, goal=goal, mask=mask)
    data = x_sample_dataset(env, rng.dirichlet(np.full(A, 0.7), size=S), n_episodes, 60, rng)
    view = LearnerView.from_env(env)
    emp = XEmpirical(data, S, A, env.terminals)
    live, obs = view.live, emp.observed
    out: Dict[str, Any] = {}
    # expectile: closed form vs bisection on the first-order condition
    x = rng.normal(size=(200, 6))
    x[:20, 3] = x[:20, 1]                       # ties
    w = rng.integers(0, 4, size=(200, 6)).astype(float)
    w[w.sum(axis=1) == 0, 0] = 1.0
    errs = []
    for tau in (0.1, 0.5, 0.7, 0.9):
        v = x_expectile(x, w, tau)
        lo, hi = x.min(axis=1), x.max(axis=1)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f = (w * np.where(x > mid[:, None], tau, -(1.0 - tau)) * np.abs(x - mid[:, None])).sum(axis=1)
            lo, hi = np.where(f > 0, mid, lo), np.where(f > 0, hi, mid)
        errs.append(float(np.abs(v - 0.5 * (lo + hi)).max()))
    out["expectile_vs_bisection_max_err"] = max(errs)
    # IQL fixed point
    iql = x_tab_iql(emp, view)
    rows = np.flatnonzero(live & emp.seen)
    out["iql_v_equation_err"] = float(np.abs(iql["v"][rows] - x_expectile(iql["q"][rows], emp.N_sa[rows],
                                                                          XCFG.iql_expectile_default)).max())
    out["iql_q_equation_err"] = float(np.abs((emp.r_hat + emp.backup(iql["v"]) - iql["q"])[obs]).max())
    out["iql_converged"] = iql["info"]["converged"]
    # Bellman residuals of the value-iteration learners
    bcq = x_tab_bcq(emp, view)
    V_bcq = np.where(live, np.where(bcq["allowed"], bcq["q"], -np.inf).max(axis=1), 0.0)
    out["bcq_bellman_residual"] = float(np.abs(emp.r_hat + emp.backup(V_bcq) - bcq["q"]).max())
    lcb = x_vi_lcb(emp, view)
    V = np.where(live, lcb["q"].max(axis=1), 0.0)
    ref = np.where(obs & live[:, None], np.clip(emp.r_hat - XCFG.lcb_c_default * (view.v_max - view.v_min)
                                                / np.sqrt(np.maximum(emp.N_sa, 1.0)) + emp.backup(V),
                                                view.v_min, view.v_max), view.v_min)
    out["vi_lcb_bellman_residual"] = float(np.abs(np.where(live[:, None], ref, 0.0) - lcb["q"]).max())
    mbs = x_mbs_qi(emp, view)
    z = mbs["zeta"]
    Vz = np.where(live, np.where(z.any(axis=1), np.where(z, mbs["q"], -np.inf).max(axis=1), view.v_min), 0.0)
    out["mbs_qi_bellman_residual"] = float(np.abs((emp.r_hat + emp.backup(Vz) - mbs["q"])[z]).max()) if z.any() else 0.0
    # SPIBB: constraint and stability
    sp_res = x_spibb(emp, view)
    boot = sp_res["bootstrapped"] & live[:, None]
    out["spibb_bootstrapped_max_dev"] = float(np.abs(sp_res["pi"][boot] - emp.pi_b[boot]).max()) if boot.any() else 0.0
    out["spibb_converged"] = sp_res["info"]["converged"]
    # sparse completed-model evaluation vs a dense solve on completed_dense
    r_model, cost = _x_model_reward_cost(emp, view)
    Pc = emp.completed_dense(view.sink)
    pi_r = rng.dirichlet(np.ones(A), size=S)
    Lx = np.flatnonzero(live)
    Ppi = np.einsum("sa,san->sn", pi_r, Pc)[np.ix_(Lx, Lx)]
    mu = np.linalg.solve((np.eye(len(Lx)) - Ppi).T, emp.d0_hat[Lx])
    ev = _x_model_eval(emp, view, pi_r, r_model, cost)
    out["completed_model_sparse_vs_dense_violation_err"] = abs(ev["violations"] - float(mu @ (pi_r * cost).sum(axis=1)[Lx]))
    out["completed_model_sparse_vs_dense_return_err"] = abs(ev["return"] - float(mu @ (pi_r * r_model).sum(axis=1)[Lx]))
    # Lagrangian budget and tightness; LP objective vs exact model value
    cache: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    lag_ok, lp_gap = [], []
    for k in XCFG.lagrange_kappa_grid:
        lag = x_lagrangian_mb(emp, view, k, cache=cache)["info"]
        lag_ok.append(lag["model_violations"] <= k + 1e-12 and
                      (lag["infeasible_side_violations"] is None or lag["infeasible_side_violations"] > k))
        lp = x_cmdp_lp_model(emp, view, k, P_model=Pc)["info"]
        if lp["status"] == "ok":
            lp_gap.append(lp["lp_eval_gap"])
    out["lagrangian_budget_and_tightness_ok"] = bool(all(lag_ok))
    out["lp_eval_gap_max"] = float(max(lp_gap)) if lp_gap else None
    # full suite: masked invariants and Bayes-kernel checks run inside run_xmethods
    recs = run_xmethods(view, data, "methods-selftest")
    out["records"] = len(recs)
    out["masked_records_outside_mass_max"] = float(max(x_inadmissible_mass(r["pi"], view.mask, live).max()
                                                       for r in recs if r["variant"] == "masked"))
    out["bayes_roundtrip_max_abs_err"] = max(r["info"]["bayes_roundtrip_max_abs_err"] for r in recs
                                             if r["method"].startswith("Diffusion"))
    out["methods"] = sorted({r["method"] for r in recs})
    tol_vi = 10 * XCFG.tol
    out["passed"] = bool(out["expectile_vs_bisection_max_err"] < 1e-12 and out["iql_v_equation_err"] < 1e-12
                         and out["iql_q_equation_err"] < tol_vi and out["bcq_bellman_residual"] < tol_vi
                         and out["vi_lcb_bellman_residual"] < tol_vi and out["mbs_qi_bellman_residual"] < tol_vi
                         and out["spibb_bootstrapped_max_dev"] == 0.0 and out["spibb_converged"]
                         and out["completed_model_sparse_vs_dense_violation_err"] < 1e-9
                         and out["completed_model_sparse_vs_dense_return_err"] < 1e-9
                         and out["lagrangian_budget_and_tightness_ok"]
                         and (out["lp_eval_gap_max"] is None or out["lp_eval_gap_max"] < 1e-7)
                         and out["masked_records_outside_mass_max"] == 0.0
                         and out["bayes_roundtrip_max_abs_err"] < 1e-10 and out["iql_converged"])
    return out


X_METHODS_SELFTEST = None
if any(EXPERIMENTS[k] for k in _X_SWITCHES):
    X_METHODS_SELFTEST = x_methods_selftest()
    save_json(X_METHODS_SELFTEST, XENV_TABLE_DIR / "xenv_methods_selftest.json")
    print("Method-suite self-test:", {k: v for k, v in X_METHODS_SELFTEST.items() if k != "methods"})
    assert X_METHODS_SELFTEST["passed"], "Part II method-suite self-test failed; see xenv/xenv_methods_selftest.json"

# Objects whose source enters the Part II code hash.
XENV_METHOD_CODE = [_x_info, _x_iterate, _x_argmax_tiebreak, _x_policy_rows, _x_chain, _x_markov_solve, x_bc,
                    _x_cql_fit, x_tab_cql, x_expectile, _x_aw_extraction, x_tab_iql, x_behavior_threshold_set,
                    x_tab_bcq, x_vi_lcb, x_mbs_qi, x_spibb, _x_model_reward_cost, _x_model_eval, _x_lagrangian_greedy,
                    x_lagrangian_mb, x_cmdp_lp_model, _x_tilted, x_sfbc, x_idql, x_fisor_tab, x_diffusion_for,
                    x_bayes_reverse_kernels, x_guided_reverse, x_method_table, run_xmethods, CategoricalDiffusion,
                    normalize_critic_per_state]
print("Part II method suite ready:", len(X_METHOD_CATALOG), "methods.")
