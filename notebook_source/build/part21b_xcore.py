# %% [markdown]
# ## II.2 Core: finite-environment object, evaluator factory, sampler, empirical models, exact CMDP LP, result cache
#
# * `FiniteEnv` is an absorbing finite MDP whose rewards are paid **on entering absorbing states** (true for
#   ICU-Sepsis — reward 1 on entering survival, verified in Section 3 —, for the Gumbel simulator — +1 discharge,
#   −1 death — and for the Garnet family), together with an admissible mask. Its inadmissible rows follow the
#   environment's own semantics ("given"); the `terminate` rule replaces them by a jump to the sink (death) paying the
#   sink's entry reward. Exact evaluation re-uses `ExactMDPEvaluator` (Section 8) with the generic invariant
#   J = Σ_t P(absorb in t)·R_t.
# * Learners never see `P`: they receive a `LearnerView` (sizes, mask, absorbing states, reward range) and data.
# * `x_cmdp_lp` solves the exact constrained MDP over occupancy measures (γ = 1, absorbing; Altman 1999) with an
#   expected-violation budget κ. Flow conservation excludes non-absorbing (improper) behaviour on every state with
#   positive occupancy (a closed class cannot pass on its inflow); states with zero occupancy take the caller's
#   `fallback` row, so callers that need a proper policy everywhere pass a proper fallback or full-support weights.
# * `XCache` stores one identity-checked result file per (cell, draw): Part II resumes after a disconnect, rejects
#   stale results after any code / config / data change, and honours `FORCE_RETRAIN` and `ALLOW_TRAINING`.

# %%
# ============================================================
# II.2 Cross-environment core
# ============================================================
EXECLOG.mark("II.2 Cross-environment core")
import zlib

X_SEMANTICS = ("given", "terminate")
_X_STRATEGY = {"given": "mean", "terminate": "terminate"}   # ExactMDPEvaluator's "mean" = "use P as given"


@dataclass(eq=False)
class FiniteEnv:
    """Finite absorbing MDP, rewards on entering absorbing states, admissible mask (terminal rows all-True)."""
    name: str
    kind: str                                   # "icu" | "gumbel" | "garnet" | "benchmark" | "fqe" | "variant"
    P: np.ndarray = field(repr=False)           # [S, A, S]; inadmissible rows follow this env's own semantics
    entry_reward: np.ndarray = field(repr=False)  # [S]; non-zero only on absorbing states
    d0: np.ndarray = field(repr=False)
    terminals: Tuple[int, ...]
    sink: int                                   # target of forced termination (death)
    goal: int                                   # best absorbing state (the evaluator's "survival" state)
    mask: np.ndarray = field(repr=False)
    features: Optional[np.ndarray] = field(default=None, repr=False)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.P = np.asarray(self.P, dtype=np.float64)
        self.entry_reward = np.asarray(self.entry_reward, dtype=np.float64)
        self.d0 = np.asarray(self.d0, dtype=np.float64)
        self.terminals = tuple(sorted(int(t) for t in self.terminals))
        S, A, S2 = self.P.shape
        assert S == S2 and self.entry_reward.shape == (S,) and self.d0.shape == (S,), self.name
        live = np.ones(S, dtype=bool)
        live[list(self.terminals)] = False
        assert np.all(self.entry_reward[live] == 0.0), f"{self.name}: rewards must be paid only on entering absorbing states"
        assert int(self.sink) in self.terminals and int(self.goal) in self.terminals, self.name
        m = np.asarray(self.mask, dtype=bool).copy()
        assert m.shape == (S, A), self.name
        m[~live] = True
        assert m[live].any(axis=1).all(), f"{self.name}: a live state has an empty admissible set"
        self.mask = m
        self.sink, self.goal = int(self.sink), int(self.goal)
        self._live = live
        self._ev = None
        self._r_sa = None
        self._identity = None

    @property
    def S(self) -> int:
        return int(self.P.shape[0])

    @property
    def A(self) -> int:
        return int(self.P.shape[1])

    @property
    def live(self) -> np.ndarray:
        return self._live

    @property
    def r_sa(self) -> np.ndarray:
        if self._r_sa is None:
            r = np.einsum("san,n->sa", self.P, self.entry_reward, optimize=True)
            r[~self._live] = 0.0
            self._r_sa = r
        return self._r_sa

    @property
    def terminate_reward(self) -> float:
        return float(self.entry_reward[self.sink])

    @property
    def value_bounds(self) -> Tuple[float, float]:
        rt = self.entry_reward[list(self.terminals)]
        return float(min(0.0, rt.min())), float(max(0.0, rt.max()))

    def identity(self) -> dict:
        # `meta` enters the hash, so it may hold only construction parameters, never solver outputs (a value-iteration
        # float depends on the BLAS reduction order and would change the identity across machines).
        if self._identity is None:
            self._identity = {"name": self.name, "kind": self.kind, "terminals": list(self.terminals),
                              "sink": self.sink, "goal": self.goal, "meta": obj_hash(self.meta),
                              "arrays": array_dict_hash({"P": self.P, "entry": self.entry_reward, "d0": self.d0,
                                                         "mask": self.mask})}
        return self._identity

    def evaluator(self) -> ExactMDPEvaluator:
        if self._ev is None:
            self._ev = ExactMDPEvaluator(
                self.P, None, self.d0, self.mask, self.terminals, self.sink, self.goal, r_sa=self.r_sa,
                terminal_entry_rewards={t: float(self.entry_reward[t]) for t in self.terminals},
                terminate_reward=self.terminate_reward)
        return self._ev

    def variant(self, name: str, P: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None,
                d0: Optional[np.ndarray] = None, kind: str = "variant", **meta) -> "FiniteEnv":
        return FiniteEnv(name=name, kind=kind, P=self.P if P is None else P, entry_reward=self.entry_reward,
                         d0=self.d0 if d0 is None else d0, terminals=self.terminals, sink=self.sink, goal=self.goal,
                         mask=self.mask if mask is None else mask, features=self.features,
                         meta={**self.meta, **meta})


@dataclass(frozen=True, eq=False)
class LearnerView:
    """Everything a Part II learner may use besides its data: no transition model, no initial distribution."""
    S: int
    A: int
    mask: np.ndarray            # admissible mask with all-True terminal rows (operational mask)
    live: np.ndarray
    terminals: Tuple[int, ...]
    sink: int
    v_min: float
    v_max: float
    terminate_reward: float

    @staticmethod
    def from_env(env: FiniteEnv) -> "LearnerView":
        lo, hi = env.value_bounds
        return LearnerView(S=env.S, A=env.A, mask=env.mask.copy(), live=env.live.copy(), terminals=env.terminals,
                           sink=env.sink, v_min=lo, v_max=hi, terminate_reward=env.terminate_reward)


# ---------- semantics, imputation and policy helpers (mask and live set always explicit) ----------
def x_semantic_P(env: FiniteEnv, semantics: str) -> np.ndarray:
    """Dense transition tensor of `env` under a semantics: 'given' (env.P) or 'terminate' (jump to the sink)."""
    if semantics == "given":
        return env.P
    if semantics != "terminate":
        raise ValueError(semantics)
    P = env.P.copy()
    bad = ~env.mask & env.live[:, None]
    P[bad] = 0.0
    P[bad, env.sink] = 1.0
    return P


def x_impute_inadmissible(P: np.ndarray, mask: np.ndarray, live: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
    """Convex imputation: each inadmissible live row becomes Σ_a' W[s, a, a'] P[s, a'] over admissible a'.

    W=None is the uniform mean over admissible actions (the ICU-Sepsis 'mean' rule). W may be [S, A] (same weights
    for every inadmissible action of a state) or [S, A, A] (per inadmissible action)."""
    P = np.asarray(P, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    S, A, _ = P.shape
    out = P.copy()
    for s in np.flatnonzero(live):
        bad = np.flatnonzero(~m[s])
        if not len(bad):
            continue
        if W is None:
            w = np.where(m[s], 1.0, 0.0)
            w = np.broadcast_to(w / w.sum(), (len(bad), A))
        elif np.ndim(W) == 2:
            w = np.broadcast_to(np.asarray(W[s], dtype=np.float64), (len(bad), A))
        else:
            w = np.asarray(W[s, bad], dtype=np.float64)
        assert np.all(w[:, ~m[s]] == 0.0) and np.allclose(w.sum(axis=1), 1.0), "W must be a distribution over admissible actions"
        out[s, bad] = w @ P[s]
    return out


def x_project(pi: np.ndarray, mask: np.ndarray, live: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
    """Admissible projection pi_W(a'|s) = pi(a'|s)1[a' in M(s)] + Σ_{a not in M(s)} pi(a|s) W[s, a, a'] (live rows)."""
    pi = np.asarray(pi, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    S, A = pi.shape
    out = pi.copy()
    L = np.flatnonzero(live)
    if W is None:
        w = m[L] / m[L].sum(axis=1, keepdims=True)
        b = (pi[L] * ~m[L]).sum(axis=1, keepdims=True)
        out[L] = pi[L] * m[L] + b * w
    else:
        Wf = np.asarray(W, dtype=np.float64)
        if Wf.ndim == 2:
            Wf = np.broadcast_to(Wf[:, None, :], (S, A, A))
        for s in L:
            bad = ~m[s]
            out[s] = pi[s] * m[s] + pi[s, bad] @ Wf[s, bad]
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(out[L][~m[L]] == 0.0)
    return out


def x_inadmissible_mass(pi: np.ndarray, mask: np.ndarray, live: np.ndarray) -> np.ndarray:
    b = (np.asarray(pi, dtype=np.float64) * ~np.asarray(mask, dtype=bool)).sum(axis=1)
    b[~live] = 0.0
    return b


def x_uniform_admissible(mask: np.ndarray, live: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    out = np.full(m.shape, 1.0 / m.shape[1])
    out[live] = m[live] / m[live].sum(axis=1, keepdims=True)
    return out


def x_greedy(scores: np.ndarray, op_mask: Optional[np.ndarray] = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    act = scores.argmax(axis=1) if op_mask is None else masked_argmax(scores, op_mask)
    return one_hot_policy(act, scores.shape[1])


def x_evaluate(env: FiniteEnv, pi: np.ndarray, semantics: str, vectors: bool = False) -> Dict[str, Any]:
    """Exact evaluation; an improper (non-absorbing) policy is reported with status 'improper', never hidden.

    On a tabular FQE selection model (kind 'fqe', estimated from few dev counts) a nearly non-absorbing policy can
    fail the evaluator's residual invariants; there it is also reported as 'improper' (message kept). On benchmark
    and true-system targets every invariant failure raises."""
    try:
        out = env.evaluator().evaluate(pi, _X_STRATEGY[semantics], return_vectors=vectors)
        out["status"] = "ok"
    except ImproperPolicyError as exc:
        out = {"status": "improper", "error": str(exc)[:200]}
    except EvaluationInvariantError as exc:
        if env.kind != "fqe":
            raise
        out = {"status": "improper", "error": ("near-singular on the FQE model: " + str(exc))[:200]}
    return out


def x_optimal(env: FiniteEnv, semantics: str = "given", constrained: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
    return env.evaluator().optimal_policy(_X_STRATEGY[semantics], constrain_to_admissible=constrained,
                                          tie_break="admissible")


# ---------- vectorised dataset sampler (same fields as Part I datasets) ----------
def x_sample_dataset(env: FiniteEnv, behavior: np.ndarray, n_episodes: int, max_steps: int,
                     rng: np.random.Generator, semantics: str = "given") -> Dict[str, np.ndarray]:
    """Simulate episodes from `env` with a fixed behaviour policy. Rows are ordered by (episode, timestep)."""
    P = x_semantic_P(env, semantics)
    S, A = env.S, env.A
    beh = validate_policy(behavior, "behaviour policy", shape=(S, A))
    cumP = np.cumsum(P, axis=2)
    cumB = np.cumsum(beh, axis=1)
    cum0 = np.cumsum(env.d0)
    s = np.searchsorted(cum0, rng.random(n_episodes) * cum0[-1], side="right")
    s = np.minimum(s, S - 1)
    assert env.live[s].all() and (env.d0[s] > 0).all()
    ep = np.arange(n_episodes)
    cols = {k: [] for k in ("state", "action", "reward", "next_state", "terminated", "cutoff", "episode_id", "timestep")}
    term_mask = ~env.live
    for t in range(max_steps):
        if not len(ep):
            break
        rb = cumB[s]
        a = (rb <= (rng.random(len(s)) * rb[:, -1])[:, None]).sum(axis=1)
        a = np.minimum(a, A - 1)
        rows = cumP[s, a]
        ns = (rows <= (rng.random(len(s)) * rows[:, -1])[:, None]).sum(axis=1)
        ns = np.minimum(ns, S - 1)
        done = term_mask[ns]
        cut = (~done) & (t == max_steps - 1)
        cols["state"].append(s)
        cols["action"].append(a)
        cols["reward"].append(env.entry_reward[ns])
        cols["next_state"].append(ns)
        cols["terminated"].append(done)
        cols["cutoff"].append(cut)
        cols["episode_id"].append(ep)
        cols["timestep"].append(np.full(len(s), t))
        keep = ~done
        s, ep = ns[keep], ep[keep]
    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    order = np.lexsort((cat["timestep"], cat["episode_id"]))
    data = {
        "state": cat["state"][order].astype(np.int64),
        "action": cat["action"][order].astype(np.int64),
        "reward": cat["reward"][order].astype(np.float32),
        "next_state": cat["next_state"][order].astype(np.int64),
        "terminated": cat["terminated"][order].astype(bool),
        "truncated": np.zeros(len(order), dtype=bool),
        "cutoff": cat["cutoff"][order].astype(bool),
        "episode_id": cat["episode_id"][order].astype(np.int32),
        "timestep": cat["timestep"][order].astype(np.int32),
    }
    data["terminal"] = data["terminated"].astype(np.float32)
    data["logged_action_admissible"] = env.mask[data["state"], data["action"]]
    del cumP
    return data


def x_validate_dataset(data: Dict[str, np.ndarray], env: FiniteEnv) -> Dict[str, Any]:
    n = len(data["state"])
    for k in REQUIRED_FIELDS:
        assert k in data and len(data[k]) == n, k
    s, a, ns = data["state"], data["action"], data["next_state"]
    assert n > 0 and s.min() >= 0 and s.max() < env.S and ns.min() >= 0 and ns.max() < env.S
    assert a.min() >= 0 and a.max() < env.A
    assert env.live[s].all(), "decision states must be live"
    assert np.array_equal(data["terminated"], ~env.live[ns]), "terminated flag must equal entering an absorbing state"
    assert np.allclose(data["reward"], env.entry_reward[ns]), "logged reward must equal the entry reward"
    assert not (data["terminated"] & data["cutoff"]).any()
    ep, ts = data["episode_id"], data["timestep"]
    starts = np.r_[True, ep[1:] != ep[:-1]]
    assert np.all(ts[starts] == 0) and np.all(ts[~starts] == ts[np.flatnonzero(~starts) - 1] + 1)
    ends = np.r_[ep[1:] != ep[:-1], True]
    assert np.all(data["terminated"][ends] | data["cutoff"][ends]), "every episode ends by absorption or cutoff"
    assert np.all(~data["terminated"][~ends] & ~data["cutoff"][~ends])
    assert np.array_equal(ns[~ends], s[np.flatnonzero(~ends) + 1]), "consecutive rows must chain"
    _, ep_inv = np.unique(ep, return_inverse=True)
    ep_ret = np.bincount(ep_inv, weights=data["reward"].astype(np.float64))
    return {"transitions": int(n), "episodes": int(starts.sum()), "terminated_episodes": int(data["terminated"][ends].sum()),
            "cutoff_episodes": int(data["cutoff"][ends].sum()), "mean_return": float(ep_ret.mean()),
            "inadmissible_logged_fraction": float((~data["logged_action_admissible"]).mean()),
            "distinct_states": int(len(np.unique(s))), "distinct_pairs": int(len(np.unique(s * env.A + a)))}


def x_split_episodes(data: Dict[str, np.ndarray], dev_fraction: float, rng: np.random.Generator):
    eps = np.unique(data["episode_id"])
    dev_eps = rng.choice(eps, size=max(1, int(round(dev_fraction * len(eps)))), replace=False)
    is_dev = np.isin(data["episode_id"], dev_eps)
    return subset_rows(data, np.flatnonzero(~is_dev)), subset_rows(data, np.flatnonzero(is_dev))


# ---------- empirical (count) model used by learners ----------
class XEmpirical:
    """Maximum-likelihood counts from data. Rows of unobserved (s, a) pairs are zero (never silently imputed)."""

    def __init__(self, data: Dict[str, np.ndarray], S: int, A: int, terminals: Sequence[int]):
        self.S, self.A = int(S), int(A)
        s, a, ns = data["state"], data["action"], data["next_state"]
        flat = s * self.A + a
        self.N_sa = np.bincount(flat, minlength=self.S * self.A).reshape(self.S, self.A).astype(np.float64)
        self.N_s = self.N_sa.sum(axis=1)
        C = sp.csr_matrix((np.ones(len(s)), (flat, ns)), shape=(self.S * self.A, self.S))
        C.sum_duplicates()
        self.C = C
        self.P = sp.diags(1.0 / np.maximum(self.N_sa.ravel(), 1.0)) @ C
        self.P = sp.csr_matrix(self.P)
        rsum = np.bincount(flat, weights=np.asarray(data["reward"], dtype=np.float64), minlength=self.S * self.A)
        self.r_hat = (rsum / np.maximum(self.N_sa.ravel(), 1.0)).reshape(self.S, self.A)
        self.observed = self.N_sa > 0
        self.live = np.ones(self.S, dtype=bool)
        self.live[list(terminals)] = False
        self.terminals = tuple(int(t) for t in terminals)
        tot = self.N_sa.sum()
        self.global_pi = self.N_sa.sum(axis=0) / max(tot, 1.0)
        self.pi_b = np.tile(self.global_pi, (self.S, 1))
        seen = self.N_s > 0
        self.pi_b[seen] = self.N_sa[seen] / self.N_s[seen, None]
        self.seen = seen
        first = data["timestep"] == 0
        d0 = np.bincount(s[first], minlength=self.S).astype(np.float64)
        self.d0_hat = d0 / max(d0.sum(), 1.0)
        self.n = int(len(s))

    def backup(self, V: np.ndarray) -> np.ndarray:
        """[S, A] array of Σ_s' P̂(s'|s,a) V(s'); zero for unobserved pairs. Callers keep V = 0 on absorbing states."""
        return np.asarray(self.P @ np.asarray(V, dtype=np.float64)).reshape(self.S, self.A)

    def completed_dense(self, sink: int) -> np.ndarray:
        """P̂ with every unobserved live pair sent to the sink (pessimistic completion) and absorbing self-loops."""
        P = self.P.toarray().reshape(self.S, self.A, self.S)
        miss = ~self.observed & self.live[:, None]
        P[miss] = 0.0
        P[miss, sink] = 1.0
        for t in self.terminals:
            P[t] = 0.0
            P[t, :, t] = 1.0
        return P


def x_fqe_env(dev: Dict[str, np.ndarray], train: Dict[str, np.ndarray], base: FiniteEnv, name: str) -> FiniteEnv:
    """Tabular model-based FQE model of the dev episodes (as in Section 26, but environment-generic).

    Each (s, a) row uses dev counts if observed in dev, else training counts if observed there, else the state's
    action-pooled dev distribution (training-pooled if the state is unseen in dev); states unseen in both splits are
    pessimistic: every action jumps to the sink."""
    S, A = base.S, base.A
    e_d, e_t = XEmpirical(dev, S, A, base.terminals), XEmpirical(train, S, A, base.terminals)
    Pd = e_d.C.toarray().reshape(S, A, S)
    Pt = e_t.C.toarray().reshape(S, A, S)
    P = np.zeros((S, A, S))
    stats_ = {"rows_from_dev": 0, "rows_from_train": 0, "rows_state_pooled": 0, "states_unobserved": 0}
    for s in np.flatnonzero(base.live):
        nd, nt = e_d.N_sa[s], e_t.N_sa[s]
        pool_src = Pd[s] if nd.sum() > 0 else Pt[s]
        if pool_src.sum() == 0:
            P[s, :, base.sink] = 1.0
            stats_["states_unobserved"] += 1
            continue
        pooled = pool_src.sum(axis=0) / pool_src.sum()
        P[s] = np.where(nd[:, None] > 0, Pd[s] / np.maximum(nd[:, None], 1),
                        np.where(nt[:, None] > 0, Pt[s] / np.maximum(nt[:, None], 1), pooled[None, :]))
        stats_["rows_from_dev"] += int((nd > 0).sum())
        stats_["rows_from_train"] += int(((nd == 0) & (nt > 0)).sum())
        stats_["rows_state_pooled"] += int(((nd == 0) & (nt == 0)).sum())
    for t in base.terminals:
        P[t, :, t] = 1.0
    d0 = e_d.d0_hat if e_d.d0_hat.sum() > 0 else e_t.d0_hat
    del Pd, Pt
    return FiniteEnv(name=name, kind="fqe", P=P, entry_reward=base.entry_reward, d0=d0, terminals=base.terminals,
                     sink=base.sink, goal=base.goal, mask=base.mask, meta={"fqe_rows": stats_, "base": base.name})


# ---------- exact constrained MDP over occupancy measures ----------
def x_cmdp_lp(P: np.ndarray, r_sa: np.ndarray, d0: np.ndarray, live: np.ndarray, cost_sa: np.ndarray,
              kappa: float, fallback: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """max Σ x(s,a) r(s,a)  s.t.  Σ_a x(s',a) − Σ_{s,a} P(s'|s,a) x(s,a) = d0(s')  (live s'),  Σ x c ≤ κ,  x ≥ 0.

    x are expected visit counts of live (s, a) pairs (γ = 1, absorbing). Returns the policy x / Σ_a x (states with
    zero occupancy take the `fallback` row, default uniform) and the LP report."""
    T = np.flatnonzero(live)
    nT, A = len(T), P.shape[1]
    Ptt = sp.csr_matrix(np.asarray(P, dtype=np.float64)[T][:, :, T].reshape(nT * A, nT))
    E = sp.csr_matrix((np.ones(nT * A), (np.repeat(np.arange(nT), A), np.arange(nT * A))), shape=(nT, nT * A))
    A_eq = (E - Ptt.T).tocsr()
    c = -np.asarray(r_sa, dtype=np.float64)[T].ravel()
    kw = {}
    if np.isfinite(kappa):
        kw = {"A_ub": sp.csr_matrix(np.asarray(cost_sa, dtype=np.float64)[T].ravel()[None, :]), "b_ub": np.array([float(kappa)])}
    t0 = time.perf_counter()
    # Interior point with crossover (basic optimal solution) is several times faster than dual simplex at these
    # sizes; dual simplex is the fallback if the interior-point run does not report an optimum.
    solver = "highs-ipm"
    res = linprog(c, A_eq=A_eq, b_eq=np.asarray(d0, dtype=np.float64)[T], bounds=(0, None), method=solver, **kw)
    if res.status != 0:
        solver = "highs-ds"
        res = linprog(c, A_eq=A_eq, b_eq=np.asarray(d0, dtype=np.float64)[T], bounds=(0, None), method=solver, **kw)
    if res.status != 0:
        raise RuntimeError(f"CMDP LP failed (status {res.status}): {res.message}")
    x = np.clip(res.x, 0.0, None).reshape(nT, A)
    occ = x.sum(axis=1)
    S = P.shape[0]
    pi = np.full((S, A), 1.0 / A) if fallback is None else np.asarray(fallback, dtype=np.float64).copy()
    reach = occ > 1e-12
    pi[T[reach]] = x[reach] / occ[reach, None]
    info = {"objective": float(-res.fun), "kappa": float(kappa), "violations": float((x * np.asarray(cost_sa)[T]).sum()),
            "expected_decisions": float(occ.sum()), "reachable_live_states": int(reach.sum()),
            "randomised_states": int(((x[reach] > 1e-9).sum(axis=1) > 1).sum()), "solve_s": time.perf_counter() - t0,
            "message": str(res.message)[:120], "solver": solver}
    return pi, info


# ---------- per-(cell, draw) result cache ----------
# Errors of a cache file cut by a disconnect (truncated gzip / csv / json): such a file is a miss, never a crash.
XCACHE_READ_ERRORS = (json.JSONDecodeError, UnicodeDecodeError, OSError, EOFError, zlib.error,
                      pd.errors.EmptyDataError, pd.errors.ParserError)


def x_atomic_replace(tmp: Path, path: Path) -> None:
    """Move a fully written temporary file into place (os.replace; shutil.move fallback as in atomic_torch_save)."""
    try:
        os.replace(tmp, path)
    except OSError:
        shutil.move(str(tmp), str(path))


class XCache:
    """Identity-addressed result tables (csv.gz + identity sidecar) for Part II."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.written: set = set()
        self.loaded: set = set()

    def _paths(self, key: str, identity: dict):
        h = obj_hash(identity)
        d = self.root / slug(key)
        return d / f"{h[:20]}.csv.gz", d / f"{h[:20]}.identity.json", h

    def get(self, key: str, identity: dict) -> Optional[pd.DataFrame]:
        data_p, side_p, h = self._paths(key, identity)
        if not (data_p.exists() and side_p.exists()):
            return None
        try:
            meta = json.loads(side_p.read_text())
        except XCACHE_READ_ERRORS as exc:     # a write cut by a disconnect: treat as a miss and recompute
            print(f"WARNING: unreadable Part II cache sidecar {side_p} ({type(exc).__name__}); recomputing.")
            return None
        if meta.get("identity_hash") != h:
            raise CheckpointIdentityError(f"{data_p} does not match the requested Part II identity.")
        if FORCE_RETRAIN and meta.get("force_token") != FORCE_TOKEN:
            return None
        # to_csv writes shortest-repr floats; only the round-trip parser reads them back bit-exactly (the default
        # parser can be off by ~1e-13 relative), so cached and freshly computed draws are identical. Label columns are
        # read as strings: a column that is 'true' in every row (deploy_target of Gumbel / Garnet) would parse as bool.
        try:
            df = pd.read_csv(data_p, float_precision="round_trip",
                             dtype={"deploy_target": str, "target": str, "target_env": str})
        except XCACHE_READ_ERRORS as exc:
            print(f"WARNING: unreadable Part II cache file {data_p} ({type(exc).__name__}); recomputing.")
            return None
        if h not in self.written:
            self.loaded.add(h)
        return df

    def put(self, key: str, identity: dict, df: pd.DataFrame) -> Path:
        data_p, side_p, h = self._paths(key, identity)
        data_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = data_p.with_name(data_p.name + ".tmp")
        df.to_csv(tmp, index=False, compression="gzip")
        x_atomic_replace(tmp, data_p)
        side_tmp = side_p.with_name(side_p.name + ".tmp")
        save_json({"key": key, "identity_hash": h, "identity": identity, "saved_utc": utc_now(), "rows": int(len(df)),
                   "force_token": FORCE_TOKEN}, side_tmp)
        x_atomic_replace(side_tmp, side_p)
        self.written.add(h)
        return data_p

    @staticmethod
    def require_allowed(key: str) -> None:
        if not ALLOW_TRAINING:
            raise MissingArtifactError(f"ALLOW_TRAINING=False but no identity-matched Part II result for '{key}'.")


XCACHE = XCache(XENV_CACHE_DIR)


# ---------- the unit of Part II: one dataset draw of one environment cell ----------
@dataclass(eq=False)
class CellInstance:
    cell: str                         # e.g. "gumbel|best|n2000|tau5|eps0.05"
    family: str                       # "icu" | "gumbel" | "garnet"
    draw: int
    learn_env: FiniteEnv              # MDP on the learner's state space (constructed benchmark, or ICU-Sepsis)
    behavior: np.ndarray              # behaviour policy on learn_env that generated `data`
    data: Dict[str, np.ndarray]
    train: Dict[str, np.ndarray]
    dev: Dict[str, np.ndarray]
    # (label, env, semantics, lift): lift maps a learn_env policy to env's state space (None = same space).
    targets: List[Tuple[str, FiniteEnv, str, Optional[Callable[[np.ndarray], np.ndarray]]]]
    deploy_target: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def identity(self) -> dict:
        return {"cell": self.cell, "draw": int(self.draw), "learn_env": self.learn_env.identity(),
                "data": array_dict_hash(self.data), "train": array_dict_hash(self.train),
                "targets": [(lab, env.identity(), sem) for lab, env, sem, _ in self.targets],
                "deploy_target": self.deploy_target}


def x_label(method: str, variant: str, hp: Optional[dict] = None, is_default: bool = True) -> str:
    base = f"{method} [{variant}]"
    if is_default or not hp:
        return base
    return base + " {" + ", ".join(f"{k}={hp[k]}" for k in sorted(hp)) + "}"


# XCFG fields that choose which cells / draws exist or configure statistics and theory only (Section II.6 markdown).
# Defined here, before Section II.5, so that the theory-record cache (II.5) and the driver (II.6) share one identity.
_XCFG_NOT_IN_IDENTITY = ("icu_draws", "gumbel_draws", "garnet_draws", "gumbel_cells", "garnet_harm",
                         "garnet_behavior_eps", "gumbel_mc_pairs", "gumbel_mc_draws", "frontier_kappas", "bootstrap",
                         "equivalence_margin_normalized")
assert set(_XCFG_NOT_IN_IDENTITY) <= set(asdict(XCFG)), sorted(set(_XCFG_NOT_IN_IDENTITY) - set(asdict(XCFG)))


def x_result_xcfg() -> dict:
    """The XCFG fields that can change what a single (cell, draw) computes."""
    return {k: v for k, v in asdict(XCFG).items() if k not in _XCFG_NOT_IN_IDENTITY}


# Objects whose source enters the Part II code hash (extended by Sections II.3-II.6).
XENV_CORE_CODE = [FiniteEnv, LearnerView, x_semantic_P, x_impute_inadmissible, x_project, x_inadmissible_mass,
                  x_uniform_admissible, x_greedy, x_evaluate, x_optimal, x_sample_dataset, x_validate_dataset,
                  x_split_episodes, XEmpirical, x_fqe_env, x_cmdp_lp, CellInstance, x_label, ExactMDPEvaluator,
                  masked_argmax, one_hot_policy, validate_policy, subset_rows]
print("Part II core ready.")
