# %% [markdown]
# ## 8. Exact finite-MDP evaluator with strict validation and interpretable violation metrics (audit M09)
#
# The transition model is used **only** for evaluation, oracles, and stress tests — no `train_*` function
# receives `P`, `R`, or `d0`. For a policy π and handling rule h ∈ {mean, terminate}, with transient set 𝒯 and
# `F = I − P_π,𝒯𝒯`, the evaluator solves (one LU factorisation, three solves):
#
# * value `F v = r_π`, return `J = d0ᵀ v`;
# * expected decisions `F ℓ = 1`, `L = d0ᵀ ℓ`;
# * occupancy `Fᵀ μ = d0`, with the invariants `L = Σ μ`, absorption probabilities summing to one,
#   `J = P(absorb in survival)` (reward 1 only on entering survival), and `0 ≤ J ≤ 1`.
#
# Violation metrics (all relative to the released benchmark-support mask, never "safety"):
# pooled step-occupancy rate `U = μᵀb / Σμ`; expected violation count `μᵀb`; probability of at least one
# violation (identical under both rules because pre-violation dynamics coincide; under termination it equals
# the expected count); number of live states with positive inadmissible mass; maximum and unweighted mean
# statewise inadmissible mass. Singular (improper / non-absorbing) systems, large residuals, failed
# invariants, malformed policies, and non-converged value iteration all raise.

# %%
# ============================================================
# 8. Exact finite-MDP evaluator
# ============================================================
EXECLOG.mark("8. Exact evaluator")


class ImproperPolicyError(RuntimeError):
    pass


class EvaluationInvariantError(RuntimeError):
    pass


class ExactMDPEvaluator:
    def __init__(self, tx_mat, r_mat, d0, support_mask, terminal_states, death_state, survival_state,
                 r_sa: Optional[np.ndarray] = None, terminal_entry_rewards: Optional[Dict[int, float]] = None,
                 terminate_reward: float = 0.0):
        # terminal_entry_rewards=None keeps the ICU-Sepsis invariants (J = P(absorb in survival), 0 <= J <= 1).
        # Otherwise rewards are paid on entering absorbing states and J = sum_t P(absorb in t) * R_t is checked
        # (Part II environments). terminate_reward is the reward of a forced jump to death_state (0 for ICU-Sepsis).
        P = np.asarray(tx_mat, dtype=np.float64)
        self.d0 = np.asarray(d0, dtype=np.float64)
        self.S, self.A, s2 = P.shape
        if r_mat is None:
            # Expected immediate rewards supplied directly (used for perturbed / empirical MDPs).
            assert r_sa is not None and np.asarray(r_sa).shape == (self.S, self.A)
            R = None
        else:
            R = np.asarray(r_mat, dtype=np.float64)
            assert R.shape == P.shape and np.isfinite(R).all()
        assert self.S == s2 and self.d0.shape == (self.S,)
        assert np.isfinite(P).all() and np.isfinite(self.d0).all()
        assert (P >= -1e-12).all() and (self.d0 >= -1e-12).all()
        assert np.allclose(P.sum(axis=2), 1.0, atol=1e-8), "Transition rows do not sum to one."
        assert np.isclose(self.d0.sum(), 1.0, atol=1e-8)
        self.terminals = np.asarray(sorted(set(int(x) for x in terminal_states)), dtype=np.int64)
        self.live = np.ones(self.S, dtype=bool)
        self.live[self.terminals] = False
        assert np.allclose(self.d0[~self.live], 0.0), "Initial distribution puts mass on absorbing states."
        self.T = np.flatnonzero(self.live)
        self.death, self.survival = int(death_state), int(survival_state)
        self.mask = np.asarray(support_mask, dtype=bool).copy()
        assert self.mask.shape == (self.S, self.A)
        assert self.mask[self.live].any(axis=1).all()
        self.m_eff = self.mask.copy()
        self.m_eff[~self.live] = True
        self.P = P
        self.r_sa_mean = (np.einsum("san,san->sa", P, R, optimize=True) if R is not None
                          else np.asarray(r_sa, dtype=np.float64).copy())
        self.terminate_reward = float(terminate_reward)
        self.terminal_entry_rewards = (None if terminal_entry_rewards is None else
                                       {int(k): float(v) for k, v in terminal_entry_rewards.items()})
        if self.terminal_entry_rewards is not None:
            assert set(self.terminal_entry_rewards) == set(int(t) for t in self.terminals)
            assert np.isclose(self.terminal_entry_rewards[self.death], self.terminate_reward)
        # inadmissible -> death; the forced jump pays the death entry reward (0 for ICU-Sepsis)
        self.r_sa_term = np.where(self.m_eff, self.r_sa_mean, self.terminate_reward)
        del R

    # ---------- policy-induced chain ----------
    def check_policy(self, policy, validate: bool = True, repair: bool = False):
        if repair:
            pi, report = repair_policy_rows(policy, self.m_eff)
            return pi, report
        if validate:
            return validate_policy(policy, "evaluated policy", shape=(self.S, self.A)), None
        return np.asarray(policy, dtype=np.float64), None

    def induced(self, pi: np.ndarray, strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        if strategy == "mean":
            return np.einsum("sa,san->sn", pi, self.P, optimize=True), (pi * self.r_sa_mean).sum(axis=1)
        if strategy == "terminate":
            pi_adm = pi * self.m_eff
            P_pi = np.einsum("sa,san->sn", pi_adm, self.P, optimize=True)
            b = (pi * ~self.m_eff).sum(axis=1)
            P_pi[:, self.death] += b
            return P_pi, (pi_adm * self.r_sa_mean).sum(axis=1) + b * self.terminate_reward
        raise ValueError("strategy must be 'mean' or 'terminate'")

    def _factor(self, M: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter("error", sla.LinAlgWarning)
            try:
                lu = sla.lu_factor(M, check_finite=True)
            except (sla.LinAlgWarning, np.linalg.LinAlgError, ValueError) as exc:
                raise ImproperPolicyError("Singular Bellman system: improper / non-absorbing policy under gamma=1.") from exc
        if np.min(np.abs(np.diag(lu[0]))) < 1e-13:
            raise ImproperPolicyError("Numerically singular Bellman system (improper policy).")
        return lu

    def evaluate(self, policy: np.ndarray, strategy: str, validate: bool = True, repair: bool = False,
                 condition: bool = False, return_vectors: bool = False) -> Dict[str, Any]:
        pi, repair_report = self.check_policy(policy, validate=validate, repair=repair)
        P_pi, r_pi = self.induced(pi, strategy)
        T = self.T
        P_tt = P_pi[np.ix_(T, T)]
        F_mat = np.eye(len(T)) - P_tt
        lu = self._factor(F_mat)
        rT, d0T, ones = r_pi[T], self.d0[T], np.ones(len(T))
        v = sla.lu_solve(lu, rT)
        ell = sla.lu_solve(lu, ones)
        mu = sla.lu_solve(lu, d0T, trans=1)
        res_v = float(np.abs(F_mat @ v - rT).max())
        res_l = float(np.abs(F_mat @ ell - ones).max())
        res_mu = float(np.abs(F_mat.T @ mu - d0T).max())
        tol = 1e-8
        if not (np.isfinite(v).all() and np.isfinite(ell).all() and np.isfinite(mu).all()):
            raise ImproperPolicyError("Non-finite exact solution.")
        if res_v > tol * (1 + np.abs(rT).max()) or res_l > tol * (1 + ell.max()) or res_mu > tol * (1 + mu.max()):
            raise EvaluationInvariantError(f"Residuals too large: value {res_v:.2e}, length {res_l:.2e}, occupancy {res_mu:.2e}")
        if (ell < -1e-8).any() or (mu < -1e-10).any():
            raise EvaluationInvariantError("Negative expected lengths or occupancies.")
        J = float(d0T @ v)
        L = float(d0T @ ell)
        total_occ = float(mu.sum())
        absorb = mu @ P_pi[np.ix_(T, self.terminals)]
        absorb_by_state = {int(s): float(p) for s, p in zip(self.terminals, absorb)}
        if abs(L - total_occ) > 1e-8 * max(1.0, total_occ):
            raise EvaluationInvariantError(f"Expected length {L} != total occupancy {total_occ}.")
        if abs(absorb.sum() - 1.0) > 1e-8:
            raise EvaluationInvariantError(f"Absorption probabilities sum to {absorb.sum()} (improper policy?).")
        if self.terminal_entry_rewards is None:
            if abs(J - absorb_by_state[self.survival]) > 1e-8:
                raise EvaluationInvariantError("Return differs from the survival absorption probability.")
            if not (-1e-10 <= J <= 1 + 1e-10):
                raise EvaluationInvariantError(f"Return {J} outside [0, 1].")
        else:
            J_abs = sum(absorb_by_state[t] * R for t, R in self.terminal_entry_rewards.items())
            if abs(J - J_abs) > 1e-8 * max(1.0, abs(J)):
                raise EvaluationInvariantError(f"Return {J} differs from sum_t P(absorb t) R_t = {J_abs}.")
            lo, hi = min(self.terminal_entry_rewards.values()), max(self.terminal_entry_rewards.values())
            if not (lo - 1e-10 <= J <= hi + 1e-10):
                raise EvaluationInvariantError(f"Return {J} outside [{lo}, {hi}].")

        b = (pi * ~self.m_eff).sum(axis=1)
        exp_viol = float(mu @ b[T])
        if b[T].max() > 0:
            if strategy == "terminate":
                p_any = exp_viol
            else:
                P_term, _ = self.induced(pi, "terminate")
                lu_t = self._factor(np.eye(len(T)) - P_term[np.ix_(T, T)])
                mu_t = sla.lu_solve(lu_t, d0T, trans=1)
                p_any = float(mu_t @ b[T])
        else:
            p_any = 0.0
        out = {
            "return": J,
            "avg_episode_length": L,
            "inadmissible_rate": exp_viol / total_occ if total_occ > 0 else 0.0,
            "expected_violations": exp_viol,
            "prob_any_violation": p_any,
            "states_with_inadmissible_mass": int((b[T] > 1e-12).sum()),
            "max_state_inadmissible_mass": float(b[T].max()),
            "mean_state_inadmissible_mass": float(b[T].mean()),
            "absorb_death": absorb_by_state.get(self.death, 0.0),
            "absorb_survival": absorb_by_state.get(self.survival, 0.0),
            "absorb_other_terminal": float(sum(p for s, p in absorb_by_state.items() if s not in (self.death, self.survival))),
            "residual_value": res_v,
            "residual_length": res_l,
            "residual_occupancy": res_mu,
        }
        if condition:
            out["condition_number"] = float(np.linalg.cond(F_mat))
        if repair_report is not None:
            out["repair_report"] = repair_report
        if return_vectors:
            v_full = np.zeros(self.S)
            v_full[T] = v
            occ_full = np.zeros(self.S)
            occ_full[T] = mu
            out["value_vector"] = v_full
            out["occupancy_vector"] = occ_full
        return out

    # ---------- dynamic programming oracles (fail loudly) ----------
    def q_values(self, V: np.ndarray, strategy: str) -> np.ndarray:
        cont = np.einsum("san,n->sa", self.P, V, optimize=True)
        q = self.r_sa_mean + cont
        if strategy == "terminate":
            q = np.where(self.m_eff, q, self.terminate_reward + V[self.death])
        return q

    def optimal_policy(self, strategy: str = "mean", constrain_to_admissible: bool = False,
                       tol: float = 1e-12, max_iter: int = 200_000,
                       tie_break: str = "admissible") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        V = np.zeros(self.S)
        converged, it = False, 0
        for it in range(1, max_iter + 1):
            q = self.q_values(V, strategy)
            if constrain_to_admissible:
                q = np.where(self.m_eff, q, -np.inf)
            new_V = q.max(axis=1)
            new_V[self.terminals] = 0.0
            delta = float(np.max(np.abs(new_V - V)))
            V = new_V
            if delta < tol:
                converged = True
                break
        if not converged:
            raise RuntimeError(f"Value iteration did not converge in {max_iter} iterations (last delta {delta:.3e}).")
        q = self.q_values(V, strategy)
        if constrain_to_admissible:
            q = np.where(self.m_eff, q, -np.inf)
        # Tie-aware greedy choice that prefers admissible actions: under the mean rule an inadmissible action's
        # q is the mean of admissible q's, so exact ties occur (e.g. in single-admissible-action states); the
        # optimal policy is not unique and is reported with admissible tie-breaking.
        qb = q.max(axis=1, keepdims=True)
        tied = q >= qb - 1e-12 * np.maximum(1.0, np.abs(qb))
        pref = tied & self.m_eff
        act = np.where(pref.any(axis=1), pref.argmax(axis=1), tied.argmax(axis=1))
        if tie_break == "lowest_index":   # previous notebook's behaviour (np.argmax), kept for continuity
            act = q.argmax(axis=1)
        policy = one_hot_policy(act, self.A)
        exact = self.evaluate(policy, strategy)["return"]
        info = {"iterations": it, "dp_value": float(self.d0 @ V), "exact_return_of_greedy": exact,
                "abs_gap": abs(exact - float(self.d0 @ V)),
                "near_tie_live_states": int((tied.sum(axis=1) > 1)[self.live].sum()),
                "inadmissible_live_states_in_greedy": int(((policy * ~self.m_eff).sum(axis=1) > 0)[self.live].sum())}
        if info["abs_gap"] > 1e-8:
            raise EvaluationInvariantError(f"Greedy oracle evaluation differs from its DP value: {info}")
        return policy, V, info


# ---------- Analytic toy tests (independent closed forms) ----------
def _toy_mdp(improper: bool = False):
    S, A = 4, 2   # 0,1 transient; 2 death; 3 survival
    P = np.zeros((S, A, S))
    P[0, 0] = [0.1, 0.5, 0.2, 0.2] if not improper else [1.0, 0.0, 0.0, 0.0]
    P[0, 1] = P[0, 0]                     # inadmissible at state 0 -> mean of admissible rows
    P[1, 0] = [0.3, 0.0, 0.3, 0.4]
    P[1, 1] = [0.0, 0.2, 0.5, 0.3]
    for t in (2, 3):
        P[t, :, t] = 1.0
    R = np.zeros((S, A, S))
    R[:, :, 3] = 1.0
    d0 = np.array([1.0, 0.0, 0.0, 0.0])
    mask = np.array([[True, False], [True, True], [True, True], [True, True]])
    return ExactMDPEvaluator(P, R, d0, mask, [2, 3], 2, 3), P


def _inv2(M):
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    return np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) / det


def run_evaluator_toy_tests() -> Dict[str, Any]:
    ev, P = _toy_mdp()
    pi = np.array([[0.6, 0.4], [0.5, 0.5], [1.0, 0.0], [1.0, 0.0]])
    # closed forms
    Ptt_mean = np.array([0.6 * P[0, 0, :2] + 0.4 * P[0, 1, :2], 0.5 * P[1, 0, :2] + 0.5 * P[1, 1, :2]])
    r_mean = np.array([P[0, 0, 3], 0.5 * P[1, 0, 3] + 0.5 * P[1, 1, 3]])
    inv_m = _inv2(np.eye(2) - Ptt_mean)
    J_mean = (inv_m @ r_mean)[0]
    L_mean = (inv_m @ np.ones(2))[0]
    Ptt_term = np.array([0.6 * P[0, 0, :2], Ptt_mean[1]])
    r_term = np.array([0.6 * P[0, 0, 3], r_mean[1]])
    inv_t = _inv2(np.eye(2) - Ptt_term)
    J_term = (inv_t @ r_term)[0]
    p_any = (inv_t @ np.array([0.4, 0.0]))[0]
    em, et = ev.evaluate(pi, "mean"), ev.evaluate(pi, "terminate")
    # projection on the toy MDP
    proj = pi.copy()
    proj[0] = [1.0, 0.0]
    pm, pt = ev.evaluate(proj, "mean"), ev.evaluate(proj, "terminate")
    results = {
        "mean_return_error": abs(em["return"] - J_mean),
        "mean_length_error": abs(em["avg_episode_length"] - L_mean),
        "terminate_return_error": abs(et["return"] - J_term),
        "prob_any_violation_error_mean_rule": abs(em["prob_any_violation"] - p_any),
        "prob_any_violation_error_terminate_rule": abs(et["prob_any_violation"] - p_any),
        "projected_equals_original_mean_rule": abs(pm["return"] - em["return"]),
        "projected_equal_across_rules": abs(pm["return"] - pt["return"]),
        "projected_violations": pm["expected_violations"] + pt["expected_violations"],
        "closed_form_mean_return": J_mean,
        "closed_form_terminate_return": J_term,
    }
    for k in ["mean_return_error", "mean_length_error", "terminate_return_error", "prob_any_violation_error_mean_rule",
              "prob_any_violation_error_terminate_rule", "projected_equals_original_mean_rule",
              "projected_equal_across_rules", "projected_violations"]:
        assert results[k] < 1e-12, (k, results[k])
    # improper policy must raise
    ev_bad, _ = _toy_mdp(improper=True)
    try:
        ev_bad.evaluate(np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]), "mean")
        raise AssertionError("Improper policy was not rejected.")
    except ImproperPolicyError:
        results["improper_policy_rejected"] = True
    # malformed policies must raise; repair is opt-in and reported
    for label, bad in {"nan": np.array([[np.nan, 1.0], [0.5, 0.5], [1, 0], [1, 0]]),
                       "negative": np.array([[1.2, -0.2], [0.5, 0.5], [1, 0], [1, 0]]),
                       "empty_row": np.array([[0.0, 0.0], [0.5, 0.5], [1, 0], [1, 0]])}.items():
        try:
            ev.evaluate(bad, "mean")
            raise AssertionError(f"Malformed policy ({label}) was evaluated silently.")
        except PolicyValidationError:
            results[f"malformed_{label}_rejected"] = True
        rep = ev.evaluate(bad, "mean", repair=True)
        results[f"malformed_{label}_opt_in_repair_report"] = rep["repair_report"]
    return results


EVALUATOR_TOY_TESTS = run_evaluator_toy_tests()
print("Analytic evaluator tests passed:", json.dumps(EVALUATOR_TOY_TESTS, indent=1, default=_json_default))

# ---------- Benchmark evaluator (built only after every trainer above is defined) ----------
_EXACT = BENCH.exact_mdp()
EVALUATOR = ExactMDPEvaluator(_EXACT["tx_mat"], _EXACT["r_mat"], _EXACT["d_0"], SUPPORT_MASK,
                              BENCH.terminal_states, BENCH.death_state, BENCH.survival_state)
D0 = _EXACT["d_0"].copy()
del _EXACT
gc.collect()


def safe_evaluate(pi: np.ndarray, rule: str, evaluator: Optional["ExactMDPEvaluator"] = None, **kwargs) -> Optional[Dict[str, Any]]:
    """Exact evaluation that returns None (instead of raising) for improper policies; malformed policies still raise."""
    try:
        return (EVALUATOR if evaluator is None else evaluator).evaluate(pi, rule, **kwargs)
    except (ImproperPolicyError, EvaluationInvariantError) as exc:
        print(f"  evaluation skipped ({rule}): {exc}")
        return None


def evaluate_both(pi: np.ndarray, **kwargs) -> Dict[str, Dict[str, Any]]:
    return {rule: EVALUATOR.evaluate(pi, rule, **kwargs) for rule in ("mean", "terminate")}


def monte_carlo_estimate(pi: np.ndarray, strategy: str, n_episodes: int, key: str) -> Dict[str, float]:
    """Small Monte-Carlo cross-check through the real environment (evaluation RNG stream)."""
    env = gym.make(ICUSepsisBenchmark.ENV_ID, inadmissible_action_strategy=strategy)
    rng = np_rng("evaluation", "monte-carlo", key, strategy)
    cdf = np.cumsum(validate_policy(pi, key), axis=1)
    cdf[:, -1] = 1.0
    G = np.zeros(n_episodes)
    Ls = np.zeros(n_episodes)
    V = np.zeros(n_episodes)
    for ep in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        for _t in range(10_000):
            a = int(np.searchsorted(cdf[s], rng.random(), side="right"))
            a = min(a, A_N - 1)
            V[ep] += float(not SUPPORT_MASK[s, a])
            s, r, term, trunc, _ = env.step(a)
            G[ep] += float(r)
            Ls[ep] += 1
            if term or trunc:
                break
    env.close()
    return {"mc_return": float(G.mean()), "mc_return_se": float(G.std(ddof=1) / math.sqrt(n_episodes)),
            "mc_length": float(Ls.mean()), "mc_length_se": float(Ls.std(ddof=1) / math.sqrt(n_episodes)),
            "mc_violations": float(V.mean()), "mc_violations_se": float(V.std(ddof=1) / math.sqrt(n_episodes)),
            "mc_episodes": n_episodes}


def env_semantics_tests(n_pairs: int = 40, n_draws: int = 1500, n_mean_pairs: int = 5) -> Dict[str, Any]:
    """Termination / mean semantics of the pinned environment versus the evaluator (audit M09).

    The test sets the environment's internal current state directly (test-only hook) after a reset.
    """
    rng = np_rng("evaluation", "env-semantics")
    pairs = [(int(s), int(a)) for s in np.flatnonzero(LIVE) for a in np.flatnonzero(~SUPPORT_MASK[s])]
    pick = [pairs[i] for i in rng.choice(len(pairs), size=min(n_pairs, len(pairs)), replace=False)]
    env_t = gym.make(ICUSepsisBenchmark.ENV_ID, inadmissible_action_strategy="terminate")
    term_ok = []
    for s, a in pick:
        env_t.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        env_t.unwrapped._current_state = s
        ns, r, term, trunc, _ = env_t.step(a)
        one_hot = np.zeros((S_N, A_N))
        one_hot[:, 0] = 1.0
        one_hot[s] = 0.0
        one_hot[s, a] = 1.0
        P_row = EVALUATOR.induced(one_hot, "terminate")[0][s]
        term_ok.append(int(ns) == BENCH.death_state and float(r) == 0.0 and bool(term)
                       and abs(P_row[BENCH.death_state] - 1.0) < 1e-12)
    env_t.close()
    env_m = gym.make(ICUSepsisBenchmark.ENV_ID, inadmissible_action_strategy="mean")
    mean_rows = []
    for s, a in pick[:n_mean_pairs]:
        counts = np.zeros(S_N)
        for _ in range(n_draws):
            env_m.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
            env_m.unwrapped._current_state = s
            ns, *_ = env_m.step(a)
            counts[int(ns)] += 1
        emp = counts / n_draws
        p = EVALUATOR.P[s, a]
        bound = 5.0 * np.sqrt(p * (1 - p) / n_draws) + 2e-3
        mean_rows.append({"state": s, "action": a, "max_abs_dev": float(np.abs(emp - p).max()),
                          "within_bounds": bool((np.abs(emp - p) <= bound).all())})
    env_m.close()
    out = {"terminate_pairs_tested": len(term_ok), "terminate_all_consistent": bool(all(term_ok)),
           "mean_rule_rows": mean_rows, "mean_rule_all_within_bounds": bool(all(r["within_bounds"] for r in mean_rows))}
    return out


print("Exact evaluator ready (transition model reserved for evaluation, oracles and stress tests).")
