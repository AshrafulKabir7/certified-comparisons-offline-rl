# %% [markdown]
# ## II.5 Propositions on unsupported-action semantics, and exact constrained frontiers
#
# Part II's benchmark-validity results rest on seven elementary facts about absorbing MDPs whose unsupported actions
# are completed by a rule. Each is stated below with its assumptions and a proof sketch, and is **verified numerically
# before any experiment is interpreted**, with this scope: P1–P3 and P5 on ICU-Sepsis (P1/P2 where the released
# mean-rule identity holds), the draw-0 Gumbel and Garnet instances and 20 random absorbing MDPs; P4 on the Gumbel and
# Garnet instances and the random MDPs (no ground truth for ICU-Sepsis); P7 on ICU-Sepsis and the draw-0 benchmark
# instances; P6 by the Part I gradient tests (Section 13) and closed-form checks. The checks go to
# `xenv/proposition_checks.csv` (one row per proposition × environment × instance × check, aggregated over policies)
# and the per-proposition status to `xenv/proposition_summary.csv`. A check row has a tolerance or bound and `passes` ∈ {True, False}.
# Descriptive rows (counts, magnitudes, tightness) leave `passes` empty, and "not run" rows say why. The release
# gate (Section 34) requires `PROPOSITIONS_OK` not to be False.
#
# **Verification set.**
# * ICU-Sepsis (`X_ICU`).
# * The draw-0 instances of the ICU cell, the first Gumbel "best" cell, the Gumbel "oracle" cell and every Garnet
#   cell. Each instance contributes every policy that `run_xmethods` learns from its draw-0 training split (cached
#   under an identity in `XENV_CACHE_DIR/theory`, with the same learner key as the driver). It also contributes the
#   behaviour policy, the uniform-admissible and uniform policies, six Dirichlet(0.5) random policies and two random
#   policies concentrated (80 %) on unsupported actions. Policies are lifted to the true state space where a target
#   lives there.
# * 20 random absorbing MDPs (S = 12, A = 4). Each has 9 live states and 3 absorbing states with random entry rewards,
#   so r_term can lie strictly inside [v_min, v_max]. Masks are random, with one single-admissible state and one fully
#   admissible state per MDP, and W is a random convex imputation. The 20 MDPs are aggregated as the worst case per check.
#
# **Setting and notation.**
# * M = (𝒮, 𝒜, P, R, d₀), 𝒮 = 𝓛 ∪ 𝒯 (live / absorbing), γ = 1. The reward R(s′) is paid on *entering* s′ ∈ 𝒯, and
#   R = 0 on 𝓛, so r(s,a) = Σ_{s′} P(s′|s,a)R(s′). Every live state has an admissible set ∅ ≠ M(s) ⊆ 𝒜.
# * A policy π is *proper* if it is absorbed with probability 1. Then F_π = I − P_π[𝓛,𝓛] is invertible and:
#   v_π = F_π⁻¹r_π (v = 0 on 𝒯); J(π) = d₀ᵀv_π; occupancy μ_π = F_π⁻ᵀd₀ (expected visits); L = 1ᵀμ_π.
# * Inadmissible mass b_π(s) = Σ_{a∉M(s)} π(a|s), and E[viol] = μ_πᵀb_π. U_π = R + v_π is the value of *entering* a
#   state.
# * v_min = min(0, min_𝒯 R), v_max = max(0, max_𝒯 R) (`FiniteEnv.value_bounds`), and r_term = R(sink).
# * **Semantics.** *given* uses P. *terminate* replaces every inadmissible live row by δ_sink, so the forced jump pays
#   r_term.
# * **Convex imputation W.** For each live s and a ∉ M(s), W(·|s,a) is a distribution on M(s). P_W(·|s,a) =
#   Σ_{a′} W(a′|s,a)P(·|s,a′) on inadmissible pairs, and P_W = P elsewhere. The ICU-Sepsis mean rule is W uniform on
#   M(s).
# * **Projection** π_W(a′|s) = π(a′|s)1[a′∈M(s)] + Σ_{a∉M(s)} π(a|s)W(a′|s,a) (`x_project`).

# %% [markdown]
# ### Propositions
#
# **P1 (projection invariance under convex imputation).**
# * *Assumption:* the given rows are P_W for a convex W, i.e. every inadmissible row is a W-mixture of the admissible
#   rows of the same state.
# * *Statement:* for every policy π and s ∈ 𝓛, P_{π_W}(·|s) = P_π(·|s) and r_{π_W}(s) = r_π(s). Hence π is proper iff
#   π_W is, and v, J, ℓ, L, μ and the absorption distribution coincide, while π_W is admissible.
# * *Proof:* Σ_{a∉M}π(a|s)P_W(·|s,a) = Σ_{a′∈M}[Σ_{a∉M}π(a|s)W(a′|s,a)]P(·|s,a′). Adding the admissible part gives
#   P_{π_W}(·|s). The reward r is linear in the row. ∎
# * *Checks (tolerance 1e-9):* induced chains, J, L, μ, v and properness, on every environment whose given rows pass the
#   imputation test. These are ICU-Sepsis with the uniform W of the released mean rule, every constructed benchmark,
#   the oracle benchmark and the true-rows + mean completions. The random MDPs use random W.
# * *Controls:* a negative control projects with the wrong W. On the true environments the projection effect is
#   reported, not claimed.
#
# **P2 (optimum under convex imputation).**
# * *Assumptions:* those of P1, and an optimal proper policy exists. Value iteration converges here and its greedy
#   policy is evaluated exactly; every Gumbel and Garnet policy is proper because of the per-step absorption hazard.
# * *Statement:*
#   * V* = V*_M: the optimum over all policies equals the optimum over admissible policies.
#   * For s ∈ 𝓛 and a ∉ M(s), Q*(s,a) = Σ_{a′}W(a′|s,a)Q*(s,a′) ≤ max_{a′∈M(s)}Q*(s,a′), with equality iff W(·|s,a)
#     is supported on the admissible argmax. In particular every action ties when |M(s)| = 1.
# * *Consequences:*
#   * Unconstrained greedy policies with lowest-index tie-breaking can take inadmissible actions at zero loss under the
#     given rows (the "oracle artefact" of the earlier notebook), but they lose under termination.
#   * The exact violation-return frontier under such rows is flat.
# * *Proof:* sup over all ≥ sup over admissible, and by P1 every π is matched by the admissible π_W. Q*(s,a) =
#   P_W(·|s,a)ᵀU* is the W-average of the admissible Q*. ∎
#
# **P3 (termination gap).**
# * *Assumption:* π is proper under both semantics of the same environment (whatever its given rows are). Let τ be the
#   first decision with A_τ ∉ M(S_τ), and p = P(τ < ∞). p is the same under both semantics, because the processes
#   coincide up to and including (S_τ, A_τ).
# * *Statement:*
#   * J_given − J_term = Σ_{s∈𝓛} μ^term(s) Σ_{a∉M(s)} π(a|s)(Q^π_given(s,a) − r_term).
#   * Hence p(v_min − r_term) ≤ J_given − J_term ≤ p(v_max − r_term).
#   * Moreover p = E_term[viol] ≤ E_given[viol].
# * *Proof:* couple the two processes up to τ.
#   * No reward is collected before τ, because rewards are paid only on absorption and S_τ is live.
#   * After τ, termination pays r_term. The given rows pay, in expectation, Q^π_given(S_τ,A_τ), a convex combination of
#     terminal rewards, which lies in [v_min, v_max].
#   * By the strong Markov property, P(S_τ = s, A_τ = a) = μ^term(s)π(a|s).
#   * p ≤ E_given[viol] is Markov's inequality. ∎
# * *Corollary:* if r_term = v_min, then Q_term(s,a) = v_min ≤ max_{a′∈M(s)}Q_term(s,a′) for every a ∉ M(s). So the
#   unconstrained terminate optimum equals the admissible optimum, and **the terminate frontier is flat as well**.
#   This holds for ICU-Sepsis (r_term = 0), Gumbel (−1) and Garnet (0).
# * *Checks:* every learned policy of every draw-0 instance, on every target environment; tolerances 1e-10 / 1e-9.
#
# **P4 (exact benchmark error decomposition).**
# * *Assumptions:*
#   * true dynamics M0 = P0 and a completion M1 with P1(·|s,a) = P0(·|s,a) for all s ∈ 𝓛, a ∈ M(s);
#   * the same R, d₀, mask and absorbing rows;
#   * π proper in both.
# * *Statement (simulation lemma, γ = 1):*
#   * J_M0(π) − J_M1(π) = Σ_{s∈𝓛} μ1(s) Σ_{a∉M(s)} π(a|s)(P0(·|s,a) − P1(·|s,a))ᵀU0, where U0 = R + v0 (v0 = 0 on 𝒯) and
#     μ1 is the occupancy under M1.
#   * Hence |J_M0 − J_M1| ≤ Σ_sμ1(s)Σ_{a∉M}π(a|s)TV_sa·span(U0) ≤ E_M1[viol]·max_{s,a∉M}TV(P0(·|s,a), P1(·|s,a))·span(U0).
#   * For a benchmark B with lifted policies: J_true − J_B = [J_M0 − J_M1] + [J_M1 − J_B]. The first term is the
#     semantics of unsupported actions; the second is estimation, coverage and lifting.
#   * This holds for both completions: mean (M1 = true rows + mean rule) and terminate (M1 = true rows + termination,
#     with P1(·|s,a) = δ_sink, so the forced jump pays R(sink)). For the oracle construction the second term is
#     identically zero.
# * *Proof:*
#   * Δ = v0 − v1 satisfies Δ = δ + P1_π[𝓛,𝓛]Δ, with δ(s) = Σ_aπ(a|s)(P0 − P1)(·|s,a)ᵀU0. This uses U0 − U1 = Δ,
#     since R is shared.
#   * Multiply by d₀ᵀF1⁻¹ = μ1ᵀ.
#   * The admissible terms vanish because those rows coincide, and |(p − q)ᵀf| ≤ TV(p,q)·span(f). ∎
# * *Checks:* identity to 1e-9 and both bounds, for both completions, on lifted policies on the true state space. The
#   preconditions (shared space, coinciding admissible and absorbing rows) are asserted as checks.
#
# **P5 (admissible policies are semantics-invariant).**
# * *Assumption:* two targets on the same state space with the same d₀, R, mask, sink and admissible live rows; each may
#   use either semantics.
# * *Statement:* if b_π ≡ 0 on 𝓛, every evaluation quantity of π (J, L, μ, absorption distribution, properness) is
#   identical on both targets.
# * *Proof:* P_π uses admissible rows only, and termination rewrites inadmissible rows only. ∎
# * *Consequence:* completion rules can only change the values, and hence the ranking, of policies that use unsupported
#   actions, and P1's projection removes that dependence.
#
# **P6 (hard-masked CQL regulariser defect; Part I Sections 10 and 13).**
# * *Setting:* ℓ_legacy(Q; s, a) = T·log Σ_{a′∈M(s)} exp(Q(s,a′)/T) − Q(s,a).
# * *Statement:*
#   * For a logged a ∉ M(s), ∂ℓ/∂Q(s,a) = −1 (−1/B in a batch mean) and ℓ → −∞ as Q(s,a) → ∞: the regulariser *rewards*
#     inflating excluded logged actions.
#   * The admissible-row design ℓ_B = 1[a∈M(s)]·ℓ is ≥ 0, since T·logsumexp ≥ max ≥ Q(s,a), and is constant in every
#     excluded coordinate.
# * *Checks:* Section 13's torch gradient tests (`REG_GRAD_TESTS`), summarised, plus closed-form numpy checks.
#
# **P7 (exact categorical diffusion with the Bayes-optimal denoiser).**
# * *Assumptions:*
#   * uniform corruption Q_t = (1 − β_t)I + β_t11ᵀ/K with β_t ∈ (0,1), so Q̄_t = ᾱ_tI + (1 − ᾱ_t)11ᵀ/K;
#   * denoiser p(x0|x_t,s) ∝ π̂_β(x0|s)Q̄_t(x0,x_t);
#   * reverse kernels B_t(x_{t−1}|x_t,s) = Σ_{x0}p(x0|x_t,s)q(x_{t−1}|x_t,x0) (`x_bayes_reverse_kernels`).
# * *Statement:*
#   * (a) Started from q_T(·|s) = π̂_β(·|s)Q̄_T = ᾱ_Tπ̂_β + (1 − ᾱ_T)u, the reverse chain returns π̂_β(·|s) exactly.
#   * (b) From any start ρ, TV(ρB_T⋯B_1, π̂_β) ≤ TV(ρ, q_T).
#   * (c) For the uniform start u, TV(u, q_T) = ᾱ_T·TV(u, π̂_β) ≤ ᾱ_T(1 − 1/K).
# * *Proof:*
#   * B_t is the exact time reversal of the forward chain started at π̂_β: Σ_{x_t}q_t(x_t)B_t(·|x_t) = q_{t−1}.
#   * Markov kernels are TV non-expansive (data processing), with ρ ↦ ρB_T⋯B_1 and q_T ↦ π̂_β.
#   * u − q_T = ᾱ_T(u − π̂_β), and TV(u, π) ≤ 1 − 1/K. ∎
# * *Consequence:* the unmasked, unguided exact diffusion policy is within ᾱ_T(1 − 1/K) of stochastic behaviour
#   cloning in every state. This is ≈ 0.05 for the Part I schedule, ᾱ_T ≈ 0.0515. P7 bounds only this unmasked
#   stochastic chain from the uniform start: differences of the deployed diffusion policies come from the admissible
#   start, masking, guidance and argmax extraction, not from denoiser error.
# * *Checks:* per state, on ICU-Sepsis (π̂_β of MAIN) and on every draw-0 benchmark (π̂_β of the training split), plus a
#   cross-check of the "Diffusion (exact, unmasked)" policy of `run_xmethods`. The argmax agreement of final-step-only
#   guided diffusion with masked SfBC (both masked) and with unmasked SfBC (argmax) is reported descriptively.

# %% [markdown]
# ### Exact constrained frontiers
#
# **Problem.** For an environment E, semantics σ and budget κ:
# J*_σ(κ) = max_π J_σ(π) s.t. E_{σ,π}[#{t : A_t ∉ M(S_t)}] ≤ κ.
#
# **LP form** (`x_cmdp_lp`; Altman 1999). With expected visit counts x(s,a) ≥ 0 (γ = 1):
# max Σx·r s.t. Σ_a x(s′,a) − Σ_{s,a}P(s′|s,a)x(s,a) = d₀(s′) (s′ ∈ 𝓛), Σx·c ≤ κ, with c(s,a) = 1[a ∉ M(s)] on live
# states.
#
# **Properties.** J*(κ) is non-decreasing and concave in κ: the optimal value of an LP is concave in a right-hand side.
# J*(0) is the admissible optimum and J*(∞) the unconstrained optimum.
#
# **Cross-checks.**
# * J*(0) and J*(∞) against value iteration (`x_optimal`, constrained / unconstrained), to 1e-7.
# * The exact evaluation of the LP policy equals the LP objective to 1e-7, and its exact expected violations are
#   ≤ κ + 1e-7. On states the LP never visits, the policy plays the admissible VI optimum (or, if value iteration is
#   unavailable, a proper admissible policy from a full-support LP): this does not affect the return and keeps the
#   evaluated chain proper. A value-iteration failure is reported as "not run" for the VI cross-check only, and P2
#   and the P3 corollary are then verified through the occupancy LP.
# * Monotonicity and concavity on the κ grid, which always contains 0 and ∞.
#
# **Problems solved.**
# * ICU-Sepsis (given and terminate).
# * The first Gumbel "best" cell, draw 0: benchmark under given and terminate, and the true system with cost
#   1[a ∉ mask_true].
# * The Gumbel "oracle" cell, draw 0: the true system with its own mask_true, plus the oracle benchmark.
# * The Gumbel physician-control cells ("best" construction, physician other than 'optimal', e.g. withhold-vaso),
#   draw 0: benchmark under given and terminate, and the true system (frontiers and LP cross-checks only; their
#   propositions are not re-run).
# * Every Garnet cell, draw 0: the true system.
#
# **Expected flat frontiers.** Under given rows that are a convex imputation the frontier is flat by P2. Under
# termination with r_term = v_min it is flat by the P3 corollary. Both are reported as **findings**, not errors: the
# released benchmark under either completion rule cannot express any value of unsupported actions. True frontiers
# are flat wherever the mask contains the true optimum (optimal-physician Gumbel cells, Garnet h ≥ 0 at ε = 0.05);
# they show a trade-off in the controls (withhold-vaso about 0.12; Garnet h < 0 up to about 0.007) and a small one for
# Garnet h > 0 at ε = 0.30 (up to about 0.0015; draw-0 values of a pre-run check). The range of a true frontier
# is `value_of_unsupported_actions` in `xenv/xenv_environment_suite.csv`. Output:
# `xenv/exact_frontiers.csv` and the global `X_FRONTIERS`.

# %%
# ============================================================
# II.5 Propositions and exact frontiers: verification helpers
# ============================================================
EXECLOG.mark("II.5 Propositions and exact frontiers")

X_PROP_COLUMNS = ["proposition", "environment", "instance", "check", "value", "tolerance_or_bound", "passes", "notes"]
X_PROPOSITION_TITLES = {
    "P1": "projection invariance under convex imputation W",
    "P2": "optimum under convex imputation (admissible optimum, ties, flat frontier)",
    "P3": "termination gap: identity, bounds, flat terminate frontier when r_term = v_min",
    "P4": "exact benchmark error decomposition (simulation lemma, both completions)",
    "P5": "admissible policies are semantics-invariant",
    "P6": "hard-masked CQL regulariser defect",
    "P7": "exact categorical diffusion with the Bayes-optimal denoiser",
    "LP": "exact constrained frontier: LP versus value iteration and exact evaluation",
}
X_FRONTIER_COLUMNS = ["family", "cell", "draw", "target", "env", "env_kind", "semantics", "kappa", "objective",
                      "exact_return", "lp_violations", "exact_violations", "prob_any_violation", "avg_episode_length",
                      "expected_decisions", "reachable_live_states", "randomised_states", "J_opt_admissible",
                      "J_opt_unconstrained", "value_of_unsupported_actions", "flat_expected", "flat_reason",
                      "frontier_range", "gap_eval", "gap_vi", "feasibility_excess", "monotone_concave_violation",
                      "eval_status", "vi_status", "solve_s", "passes"]
X_IMPUTATION_TOL = 1e-12     # given rows are a convex imputation (same threshold as Section 3's mean-rule identity)
X_LP_TOL = 1e-7              # LP vs VI / exact-evaluation tolerance (HiGHS feasibility scale)
X_INVARIANT = "implementation invariant (not an independent check)"   # descriptive rows that hold by construction


class XPropositionLog:
    """Rows of proposition_checks.csv: checks (pass/fail against a bound), descriptive reports and 'not run' rows."""

    def __init__(self):
        self.rows: List[dict] = []

    def check(self, prop: str, environment: str, instance: str, check: str, value: float, bound: float,
              op: str = "le", notes: str = "") -> bool:
        v = float(value)
        if op == "le":
            ok = bool(np.isfinite(v) and v <= bound)
        elif op == "ge":
            ok = bool(np.isfinite(v) and v >= bound)
        else:
            raise ValueError(op)
        rule = "pass if value <= bound" if op == "le" else "pass if value >= bound"
        self.rows.append({"proposition": prop, "environment": environment, "instance": instance, "check": check,
                          "value": v, "tolerance_or_bound": float(bound), "passes": ok,
                          "notes": f"{rule}; {notes}" if notes else rule, "_op": op})
        return ok

    def report(self, prop: str, environment: str, instance: str, check: str, value: float, notes: str = "") -> None:
        self.rows.append({"proposition": prop, "environment": environment, "instance": instance, "check": check,
                          "value": float(value), "tolerance_or_bound": np.nan, "passes": pd.NA,
                          "notes": f"descriptive; {notes}" if notes else "descriptive", "_op": "report"})

    def not_run(self, prop: str, environment: str, instance: str, check: str, reason: str) -> None:
        self.rows.append({"proposition": prop, "environment": environment, "instance": instance, "check": check,
                          "value": np.nan, "tolerance_or_bound": np.nan, "passes": pd.NA,
                          "notes": f"not run: {reason}", "_op": "not_run"})

    def merge_aggregated(self, other: "XPropositionLog", environment: str, instance: str) -> None:
        """Worst case over `other`'s rows per (proposition, check): used for the random-MDP suite."""
        groups: Dict[Tuple[str, str, str], List[dict]] = {}
        for r in other.rows:
            groups.setdefault((r["proposition"], r["check"], r["_op"]), []).append(r)
        for (prop, check, op), rows in groups.items():
            if op in ("le", "ge"):
                margin = [np.nan_to_num((r["value"] - r["tolerance_or_bound"]) * (1 if op == "le" else -1), nan=np.inf)
                          for r in rows]
                worst = rows[int(np.argmax(margin))]
                n_fail = sum(not r["passes"] for r in rows)
                self.rows.append({**worst, "environment": environment, "instance": instance,
                                  "passes": bool(n_fail == 0),
                                  "notes": f"worst of {len(rows)} rows ({n_fail} failed); {worst['notes']}"})
            elif op == "report":
                vals = np.array([r["value"] for r in rows], dtype=np.float64)
                v = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
                self.rows.append({**rows[0], "environment": environment, "instance": instance, "value": v,
                                  "notes": f"descriptive; max over {len(rows)} rows"})
            else:
                self.rows.append({**rows[0], "environment": environment, "instance": instance,
                                  "notes": f"{rows[0]['notes']} ({len(rows)} rows)"})

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame([{k: r[k] for k in X_PROP_COLUMNS} for r in self.rows], columns=X_PROP_COLUMNS)
        df["passes"] = df["passes"].astype("boolean")
        return df


def x_tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Total-variation distance along the last axis."""
    return 0.5 * np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum(axis=-1)


def x_imputation_deviation(env: FiniteEnv, W: Optional[np.ndarray] = None) -> float:
    """max |P(.|s,a) - sum_a' W(a'|s,a) P(.|s,a')| over live inadmissible pairs (0 iff env's given rows are P_W)."""
    imp = x_impute_inadmissible(env.P, env.mask, env.live, W)
    dev = float(np.abs(imp - env.P).max())
    del imp
    return dev


def x_admissible_row_deviation(P_a: np.ndarray, P_b: np.ndarray, mask: np.ndarray, live: np.ndarray) -> float:
    """max |P_a(.|s,a) - P_b(.|s,a)| over live admissible pairs."""
    dev = 0.0
    for s in np.flatnonzero(live):
        m = mask[s]
        dev = max(dev, float(np.abs(P_a[s, m] - P_b[s, m]).max()))
    return dev


def x_inadmissible_tv(P_a: np.ndarray, P_b: np.ndarray, mask: np.ndarray, live: np.ndarray) -> np.ndarray:
    """[S, A] total variation between the rows of two models on live inadmissible pairs (0 elsewhere)."""
    tv = np.zeros(mask.shape)
    for s in np.flatnonzero(live):
        bad = ~mask[s]
        if bad.any():
            tv[s, bad] = x_tv(P_a[s, bad], P_b[s, bad])
    return tv


def x_shares_admissible_rows(e1: FiniteEnv, e2: FiniteEnv) -> Tuple[bool, float]:
    """P5 precondition: same space, d0, rewards, mask, sink, absorbing set and admissible live rows."""
    if e1 is e2:
        return True, 0.0
    if e1.P.shape != e2.P.shape or e1.terminals != e2.terminals or e1.sink != e2.sink:
        return False, np.inf
    if not (np.array_equal(e1.mask, e2.mask) and np.array_equal(e1.d0, e2.d0)
            and np.array_equal(e1.entry_reward, e2.entry_reward)):
        return False, np.inf
    dev = x_admissible_row_deviation(e1.P, e2.P, e1.mask, e1.live)
    return bool(dev <= X_IMPUTATION_TOL), dev


def x_theory_random_policies(mask: np.ndarray, live: np.ndarray, rng: np.random.Generator, n_dirichlet: int,
                             n_inadmissible: int, n_deterministic: int = 0) -> List[Tuple[str, np.ndarray]]:
    """Random test policies: Dirichlet(0.5) rows, rows with 80% mass on unsupported actions, random one-hot rows."""
    S, A = mask.shape
    bad = ~np.asarray(mask, dtype=bool) & live[:, None]
    has_bad = bad.any(axis=1)
    out = [(f"random Dirichlet(0.5) #{i}", rng.dirichlet(np.full(A, 0.5), size=S)) for i in range(n_dirichlet)]
    for i in range(n_inadmissible):
        base = rng.dirichlet(np.ones(A), size=S)
        on_bad = np.where(bad, base, 0.0)
        on_good = np.where(bad, 0.0, base)
        pi = base.copy()
        pi[has_bad] = (0.8 * on_bad[has_bad] / on_bad[has_bad].sum(axis=1, keepdims=True)
                       + 0.2 * on_good[has_bad] / on_good[has_bad].sum(axis=1, keepdims=True))
        out.append((f"random unsupported-heavy #{i}", pi))
    for i in range(n_deterministic):
        out.append((f"random deterministic #{i}", one_hot_policy(rng.integers(0, A, size=S), A)))
    return out


class XOptCache:
    """Memoised x_optimal(env, semantics, constrained) for one instance (the memo keeps the environments alive).

    A RuntimeError of x_optimal (VI non-convergence within its iteration cap, improper greedy policy, DP / exact gap)
    is recorded and returned as None; callers then verify through the exact occupancy LP (`x_lp_optimum`) and say so.
    Section II.3d censors trap states, so every benchmark policy is proper and this is a numerical safety net."""

    def __init__(self):
        self.memo: Dict[Tuple[int, str, bool], tuple] = {}

    def __call__(self, env: FiniteEnv, semantics: str, constrained: bool) -> Optional[tuple]:
        k = (id(env), semantics, bool(constrained))
        if k not in self.memo:
            try:
                self.memo[k] = (env, x_optimal(env, semantics, constrained=constrained), "")
            except RuntimeError as exc:
                self.memo[k] = (env, None, f"{type(exc).__name__}: {str(exc)[:160]}")
        return self.memo[k][1]

    def reason(self, env: FiniteEnv, semantics: str, constrained: bool) -> str:
        return self.memo[(id(env), semantics, bool(constrained))][2]


def x_lp_optimum(env: FiniteEnv, semantics: str, constrained: bool) -> float:
    """Optimal d0-weighted return over proper policies by the occupancy LP (kappa = 0 admissible, inf unconstrained)."""
    P = x_semantic_P(env, semantics)
    r = np.einsum("san,n->sa", P, env.entry_reward, optimize=True)
    r[~env.live] = 0.0
    cost = (~env.mask & env.live[:, None]).astype(np.float64)
    _, info = x_cmdp_lp(P, r, env.d0, env.live, cost, 0.0 if constrained else float("inf"))
    return float(info["objective"])


# ---------- per-policy quantities of P1-P4 ----------
def x_p1_policy(env: FiniteEnv, pi: np.ndarray, base: Dict[str, Any], W: Optional[np.ndarray]) -> Dict[str, float]:
    """P1 for one policy: pi versus its projection pi_W under env's given rows (base = x_evaluate(env, pi, 'given'))."""
    ev = env.evaluator()
    L = env.live
    pw = x_project(pi, env.mask, L, W)
    Pa, ra = ev.induced(pi, "mean")
    Pb, rb = ev.induced(pw, "mean")
    out = {"dP": float(np.abs(Pa[L] - Pb[L]).max()), "dr": float(np.abs(ra[L] - rb[L]).max()),
           "proj_mass": float(x_inadmissible_mass(pw, env.mask, L).max())}
    del Pa, Pb
    res = x_evaluate(env, pw, "given", vectors=True)
    out["status_mismatch"] = float(res["status"] != base["status"])
    nan = float("nan")
    out.update(dJ=nan, dL=nan, dmu=nan, dv=nan, dabs=nan)
    if res["status"] == "ok" and base["status"] == "ok":
        out.update(dJ=abs(res["return"] - base["return"]),
                   dL=abs(res["avg_episode_length"] - base["avg_episode_length"]),
                   dmu=float(np.abs(res["occupancy_vector"] - base["occupancy_vector"]).max()),
                   dv=float(np.abs(res["value_vector"] - base["value_vector"]).max()),
                   dabs=max(abs(res[k] - base[k]) for k in ("absorb_death", "absorb_survival", "absorb_other_terminal")))
    return out


def x_p2_env(env: FiniteEnv, opt: "XOptCache") -> Dict[str, Any]:
    """P2 on env's given rows: unconstrained vs admissible optimum, convexity of Q*, ties, lowest-index greedy."""
    V_unc, V_con = opt(env, "given", False)[1], opt(env, "given", True)[1]
    L, m = env.live, env.mask
    Q = env.evaluator().q_values(V_unc, "mean")[L]
    mL = m[L]
    bad = ~mL
    q_all = Q.max(axis=1)
    q_adm = np.where(mL, Q, -np.inf).max(axis=1)
    excess = float(np.where(bad, Q - q_adm[:, None], -np.inf).max()) if bad.any() else 0.0
    tied = bad & (Q >= (q_all - 1e-9 * np.maximum(1.0, np.abs(q_all)))[:, None])
    single = mL.sum(axis=1) == 1
    single_dev = (float(np.abs(Q[single] - q_adm[single, None]).max()) if (single.any() and env.A > 1)
                  else float("nan"))
    act = np.zeros(env.S, dtype=np.int64)
    act[L] = Q.argmax(axis=1)              # lowest index among exact maxima: the earlier notebook's oracle
    low = one_hot_policy(act, env.A)
    eg, et = x_evaluate(env, low, "given"), x_evaluate(env, low, "terminate")
    J_star = float(env.d0 @ V_unc)
    nan = float("nan")
    return {"dV": float(np.abs(V_unc[L] - V_con[L]).max()), "dJ": abs(float(env.d0 @ (V_unc - V_con))),
            "J_star": J_star, "convexity_excess": excess, "tied_states": int(tied.any(axis=1).sum()),
            "tied_single": int((tied.any(axis=1) & single).sum()), "single_states": int(single.sum()),
            "single_dev": single_dev, "low_inadmissible_states": int(((low[L] * bad).sum(axis=1) > 0).sum()),
            "low_status": f"{eg['status']}/{et['status']}",
            "low_gap_given": eg["return"] - J_star if eg["status"] == "ok" else nan,
            "low_gap_terminate": et["return"] - J_star if et["status"] == "ok" else nan,
            "low_violations": eg["expected_violations"] if eg["status"] == "ok" else nan}


def x_p3_policy(env: FiniteEnv, pi: np.ndarray, eg: Dict[str, Any], et: Dict[str, Any]) -> Dict[str, float]:
    """P3 for one proper policy: the given-terminate gap, its occupancy identity and the p-bounds."""
    L = env.live
    bad = ~env.mask & L[:, None]
    lo, hi = env.value_bounds
    rt = env.terminate_reward
    gap = eg["return"] - et["return"]
    p = et["prob_any_violation"]
    U = env.entry_reward + eg["value_vector"]
    Qg = np.einsum("san,n->sa", env.P, U, optimize=True)
    rhs = float(et["occupancy_vector"] @ np.where(bad, pi * (Qg - rt), 0.0).sum(axis=1))
    # independent value-form solve of P(at least one violation): (I - P_adm[L, L]) p = b
    T = np.flatnonzero(L)
    b = np.where(bad, pi, 0.0).sum(axis=1)
    P_adm = np.einsum("sa,san->sn", np.where(bad, 0.0, pi), env.P, optimize=True)[np.ix_(T, T)]
    p_ind = float(env.d0[T] @ np.linalg.solve(np.eye(len(T)) - P_adm, b[T]))
    return {"gap": gap, "p": p, "bound_excess": max(p * (lo - rt) - gap, gap - p * (hi - rt)),
            "identity": abs(gap - rhs), "p_rules": abs(eg["prob_any_violation"] - p), "p_indep": abs(p_ind - p),
            "p_term_viol": abs(et["expected_violations"] - p), "markov_excess": p - eg["expected_violations"]}


def x_p4_policy(M0: FiniteEnv, P1: np.ndarray, pi: np.ndarray, ev0: Dict[str, Any], ev1: Dict[str, Any],
                tv: np.ndarray, max_tv: float) -> Dict[str, float]:
    """P4 for one policy on the true space: simulation-lemma identity and the two TV bounds of J_M0 - J_M1."""
    bad = ~M0.mask & M0.live[:, None]
    U0 = M0.entry_reward + ev0["value_vector"]
    PU0 = np.einsum("san,n->sa", M0.P, U0, optimize=True)
    PU1 = np.einsum("san,n->sa", P1, U0, optimize=True)
    contrib = pi * (PU0 - PU1)
    d1 = ev1["occupancy_vector"]                      # zero on absorbing states
    rhs_bad = float(d1 @ np.where(bad, contrib, 0.0).sum(axis=1))
    rhs_all = float(d1 @ contrib.sum(axis=1))
    lhs = ev0["return"] - ev1["return"]
    span = float(U0.max() - U0.min())
    weighted = float(d1 @ np.where(bad, pi * tv, 0.0).sum(axis=1)) * span
    coarse = float(ev1["expected_violations"]) * max_tv * span
    return {"semantics_term": lhs, "identity": abs(lhs - rhs_bad), "admissible_part": abs(rhs_all - rhs_bad),
            "excess": abs(lhs) - coarse, "weighted_excess": abs(lhs) - weighted,
            "tightness": abs(lhs) / coarse if coarse > 0 else (0.0 if lhs == 0 else np.inf), "span": span}


# ---------- P6 and P7 closed forms ----------
def _x_masked_lse(q: np.ndarray, m: np.ndarray, T: float) -> np.ndarray:
    z = np.where(m, q / T, -np.inf)
    mx = z.max(axis=1)
    return T * (mx + np.log(np.exp(z - mx[:, None]).sum(axis=1)))


def x_p6_closed_form_checks(rng: np.random.Generator, n: int = 64, K: int = 8) -> Dict[str, float]:
    """Numpy closed-form checks of the legacy and admissible-row regularisers (no autograd)."""
    Q = 3.0 * rng.standard_normal((n, K))
    m = rng.random((n, K)) < 0.5
    m[np.arange(n), rng.integers(0, K, size=n)] = True
    a = rng.integers(0, K, size=n)
    for i in range(n // 4):                            # make sure some logged actions are excluded
        if (~m[i]).any():
            a[i] = int(np.flatnonzero(~m[i])[0])
    rows = np.arange(n)
    adm = m[rows, a]
    ex = np.flatnonzero(~adm)
    assert len(ex) and adm.any(), "P6 test batch needs excluded and admissible logged actions"
    out = {"legacy_fd_dev": 0.0, "legacy_untouched_rows_dev": 0.0, "legacy_mean_derivative_dev": 0.0,
           "design_b_min": np.inf, "design_b_excluded_invariance": 0.0, "batch": n, "excluded_rows": int(len(ex))}
    for T in (0.5, 1.0, 2.0):
        def legacy(q):
            return _x_masked_lse(q, m, T) - q[rows, a]
        base = legacy(Q)
        h = 2.5
        Q2 = Q.copy()
        Q2[ex, a[ex]] += h
        diff = legacy(Q2) - base
        out["legacy_fd_dev"] = max(out["legacy_fd_dev"], float(np.abs(diff[ex] + h).max()))
        out["legacy_untouched_rows_dev"] = max(out["legacy_untouched_rows_dev"], float(np.abs(diff[adm]).max()))
        i, e = int(ex[0]), 0.5                         # the legacy loss is linear in that coordinate
        Qp, Qm = Q.copy(), Q.copy()
        Qp[i, a[i]] += e
        Qm[i, a[i]] -= e
        deriv = (legacy(Qp).mean() - legacy(Qm).mean()) / (2 * e)
        out["legacy_mean_derivative_dev"] = max(out["legacy_mean_derivative_dev"], abs(deriv + 1.0 / n))
        db = np.where(adm, base, 0.0)
        out["design_b_min"] = min(out["design_b_min"], float(db.min()))
        Q3 = Q + np.where(~m, 5.0 * rng.standard_normal((n, K)), 0.0)   # perturb every excluded coordinate
        out["design_b_excluded_invariance"] = max(out["design_b_excluded_invariance"],
                                                  float(np.abs(np.where(adm, legacy(Q3), 0.0) - db).max()))
    return out


def x_reverse_marginal(kernels: Dict[int, np.ndarray], p_T: np.ndarray) -> np.ndarray:
    """Propagate per-state distributions [S, K] through B_T, ..., B_1 (kernels[t][s, x_t, x_{t-1}]); no mask."""
    p = np.asarray(p_T, dtype=np.float64)
    for t in sorted(kernels, reverse=True):
        p = np.einsum("sk,skj->sj", p, np.asarray(kernels[t], dtype=np.float64), optimize=True)
    return p


def x_p7_state_checks(pi_b: np.ndarray, diffusion: "CategoricalDiffusion", kernels: Dict[int, np.ndarray],
                      live: np.ndarray) -> Dict[str, Any]:
    """P7 (a)-(c) per live state for the Bayes-optimal reverse chain of the uniform-corruption process."""
    pi_b = np.asarray(pi_b, dtype=np.float64)
    S, K = pi_b.shape
    T = diffusion.T
    assert K == diffusion.K and sorted(kernels) == list(range(1, T + 1)), "kernels must map t = 1..T to [S, K, K]"
    rowsum_dev, kmin = 0.0, np.inf
    for t in range(1, T + 1):
        k = np.asarray(kernels[t], dtype=np.float64)
        assert k.shape == (S, K, K), f"kernel {t}: shape {k.shape} != {(S, K, K)}"
        rowsum_dev = max(rowsum_dev, float(np.abs(k.sum(axis=2) - 1.0).max()))
        kmin = min(kmin, float(k.min()))
    abar = float(diffusion.alpha_bar[-1])
    u = np.full((S, K), 1.0 / K)
    qT = pi_b @ diffusion.Qbar[T]
    out_q = x_reverse_marginal(kernels, qT)
    out_u = x_reverse_marginal(kernels, u)
    tv_q, tv_u = x_tv(out_q, pi_b)[live], x_tv(out_u, pi_b)[live]
    tv_uq, tv_ub = x_tv(u, qT)[live], x_tv(u, pi_b)[live]
    pos = tv_ub > 1e-12
    ratio = tv_u[pos] / tv_ub[pos]
    return {"abar": abar, "K": K, "kernel_rowsum_dev": rowsum_dev, "kernel_min": kmin,
            "qT_closed_dev": float(np.abs(qT - (abar * pi_b + (1.0 - abar) * u))[live].max()),
            "tv_from_qT": float(tv_q.max()), "tv_identity_dev": float(np.abs(tv_uq - abar * tv_ub).max()),
            "contraction_excess": float((tv_u - tv_uq).max()), "tv_from_u_max": float(tv_u.max()),
            "tv_from_u_mean": float(tv_u.mean()), "global_bound": abar * (1.0 - 1.0 / K),
            "ratio_max": float(ratio.max()) if len(ratio) else float("nan"),
            "ratio_median": float(np.median(ratio)) if len(ratio) else float("nan"), "out_u": out_u}


# ---------- random absorbing MDPs ----------
def x_tiny_absorbing_mdp(i: int, S: int = 12, A: int = 4) -> Tuple[FiniteEnv, np.ndarray]:
    """Random absorbing MDP (S-3 live states; death / goal / timeout absorbing) with a random mask and random W."""
    rng = np_rng("xenv", "theory", "tiny-mdp", int(i), int(S), int(A))
    nL = S - 3
    death, goal, timeout = nL, nL + 1, nL + 2
    R = np.zeros(S)
    R[death], R[goal], R[timeout] = rng.uniform(-1.0, 0.0), rng.uniform(0.5, 1.0), rng.uniform(-0.5, 0.5)
    P = np.zeros((S, A, S))
    for s in range(nL):
        for a in range(A):
            h = rng.uniform(0.05, 0.3)
            succ = rng.choice(nL, size=3, replace=False)
            P[s, a, succ] = (1.0 - h) * rng.dirichlet(np.ones(3))
            P[s, a, [death, goal, timeout]] = h * rng.dirichlet(np.ones(3))
    for t in (death, goal, timeout):
        P[t, :, t] = 1.0
    mask = np.ones((S, A), dtype=bool)
    for s in range(nL):
        k = 1 if s == 0 else (A if s == 1 else int(rng.integers(1, A + 1)))
        mask[s] = False
        mask[s, rng.choice(A, size=k, replace=False)] = True
    d0 = np.zeros(S)
    start = rng.choice(nL, size=4, replace=False)
    d0[start] = rng.dirichlet(np.ones(4))
    W = np.zeros((S, A, A))
    for s in range(nL):
        adm = np.flatnonzero(mask[s])
        for a in np.flatnonzero(~mask[s]):
            W[s, a, adm] = rng.dirichlet(np.full(len(adm), 0.7))
    env = FiniteEnv(name=f"tiny-{i:02d}", kind="variant", P=P, entry_reward=R, d0=d0,
                    terminals=(death, goal, timeout), sink=death, goal=goal, mask=mask, meta={"tiny_mdp": int(i)})
    return env, W


# ---------- exact constrained frontier ----------
def x_exact_frontier(env: FiniteEnv, semantics: str, kappas: Sequence[float], opt: "XOptCache", meta: Dict[str, Any],
                     W: Optional[np.ndarray] = None) -> pd.DataFrame:
    """CMDP-LP frontier of env under a semantics, cost 1[a not in M(s)] on live states, with VI / exact cross-checks."""
    P = x_semantic_P(env, semantics)
    r = np.einsum("san,n->sa", P, env.entry_reward, optimize=True)
    r[~env.live] = 0.0
    cost = (~env.mask & env.live[:, None]).astype(np.float64)
    nan = float("nan")
    con, unc = opt(env, semantics, True), opt(env, semantics, False)
    vi_status = ("ok" if (con is not None and unc is not None) else
                 "; ".join(x for x in (opt.reason(env, semantics, True), opt.reason(env, semantics, False)) if x))
    J_con = float(env.d0 @ con[1]) if con is not None else nan
    J_unc = float(env.d0 @ unc[1]) if unc is not None else nan
    if con is not None:
        fallback = con[0]
    else:   # proper admissible policy from every live state (full-support weights), as the driver's reference LP
        try:
            fallback, _ = x_cmdp_lp(P, r, 0.5 * env.d0 + 0.5 * env.live / env.live.sum(), env.live, cost, 0.0)
        except RuntimeError:
            fallback = None
    lo, _ = env.value_bounds
    if semantics == "given" and x_imputation_deviation(env, W) <= X_IMPUTATION_TOL:
        flat, reason = True, "P2: given rows are a convex imputation"
    elif semantics == "terminate" and env.terminate_reward == lo:
        flat, reason = True, "P3 corollary: r_term = v_min"
    else:
        flat, reason = False, ""
    ks = sorted(set(float(k) for k in kappas) | {0.0, float("inf")})
    rows = []
    for k in ks:
        try:
            pi, info = x_cmdp_lp(P, r, env.d0, env.live, cost, k, fallback=fallback)
        except RuntimeError as exc:
            info = {"objective": nan, "violations": nan, "expected_decisions": nan, "reachable_live_states": -1,
                    "randomised_states": -1, "solve_s": nan}
            ev = {"status": f"lp_failed: {str(exc)[:120]}"}
        else:
            ev = x_evaluate(env, pi, semantics)
        ok = ev["status"] == "ok"
        row = {**meta, "env": env.name, "env_kind": env.kind, "semantics": semantics, "kappa": k,
               "objective": info["objective"], "exact_return": ev["return"] if ok else nan,
               "lp_violations": info["violations"],
               "exact_violations": ev["expected_violations"] if ok else nan,
               "prob_any_violation": ev["prob_any_violation"] if ok else nan,
               "avg_episode_length": ev["avg_episode_length"] if ok else nan,
               "expected_decisions": info["expected_decisions"], "reachable_live_states": info["reachable_live_states"],
               "randomised_states": info["randomised_states"], "J_opt_admissible": J_con, "J_opt_unconstrained": J_unc,
               "value_of_unsupported_actions": J_unc - J_con, "flat_expected": flat, "flat_reason": reason,
               "eval_status": ev["status"], "vi_status": vi_status, "solve_s": info["solve_s"]}
        row["gap_eval"] = abs(row["exact_return"] - row["objective"]) if ok else nan
        row["feasibility_excess"] = (row["exact_violations"] - k) if (ok and np.isfinite(k)) else nan
        row["gap_vi"] = (abs(row["objective"] - J_con) if k == 0.0 else
                         abs(row["objective"] - J_unc) if not np.isfinite(k) else nan)   # NaN if VI is unavailable
        rows.append(row)
    obj = np.array([r_["objective"] for r_ in rows])
    mono = float(max(0.0, np.max(obj[:-1] - obj[1:])))
    conc = 0.0
    for j in range(1, len(ks) - 2):                  # consecutive finite triples (the last kappa is inf)
        lam = (ks[j + 1] - ks[j]) / (ks[j + 1] - ks[j - 1])
        conc = max(conc, float(lam * obj[j - 1] + (1 - lam) * obj[j + 1] - obj[j]))
    if not np.isfinite(obj).all():                   # a failed LP: the shape checks are undefined (and fail)
        mono = conc = nan
    rng_ = float(np.max(obj) - obj[0])
    for row in rows:
        row["frontier_range"] = rng_
        row["monotone_concave_violation"] = max(mono, conc)
        checks = [row["eval_status"] == "ok", bool(row["gap_eval"] <= X_LP_TOL),
                  bool(row["monotone_concave_violation"] <= X_LP_TOL)]
        if np.isfinite(row["feasibility_excess"]):
            checks.append(row["feasibility_excess"] <= X_LP_TOL)
        if np.isfinite(row["gap_vi"]):
            checks.append(row["gap_vi"] <= X_LP_TOL)
        row["passes"] = bool(all(checks))
    return pd.DataFrame(rows).reindex(columns=X_FRONTIER_COLUMNS)


def x_frontier_checks(log: XPropositionLog, fr: pd.DataFrame, instance: str) -> None:
    """LP cross-check rows (and the P2 / P3 flat-frontier findings) of one frontier for proposition_checks.csv."""
    env, sem = str(fr["env"].iloc[0]), str(fr["semantics"].iloc[0])
    name = f"{env} [{fr['target'].iloc[0]}, {sem}]"
    n_bad = int((fr["eval_status"] != "ok").sum())
    log.check("LP", name, instance, "LP policies proper (count improper)", n_bad, 0.0)
    log.check("LP", name, instance, "max |exact return of LP policy - LP objective|", fr["gap_eval"].max(), X_LP_TOL)
    vi = str(fr["vi_status"].iloc[0])
    for txt, sel in (("kappa = 0: |LP objective - admissible VI optimum|", fr["kappa"] == 0.0),
                     ("kappa = inf: |LP objective - unconstrained VI optimum|", ~np.isfinite(fr["kappa"]))):
        if vi == "ok":
            log.check("LP", name, instance, txt, fr.loc[sel, "gap_vi"].iloc[0], X_LP_TOL)
        else:   # the LP itself is still verified by the exact evaluation of its policy (rows above / below)
            log.not_run("LP", name, instance, txt, f"value iteration unavailable ({vi})")
    fe = fr["feasibility_excess"].dropna()
    log.check("LP", name, instance, "max (exact expected violations - kappa)", fe.max() if len(fe) else 0.0, X_LP_TOL)
    log.check("LP", name, instance, "frontier non-decreasing and concave in kappa (max violation)",
              fr["monotone_concave_violation"].iloc[0], X_LP_TOL)
    flat = bool(fr["flat_expected"].iloc[0])
    if flat:
        prop = "P2" if sem == "given" else "P3"
        log.check(prop, name, instance, "frontier is flat (max_kappa J*(kappa) - J*(0))", fr["frontier_range"].iloc[0],
                  X_LP_TOL, notes=f"finding, not an error: {fr['flat_reason'].iloc[0]}")
    else:
        log.report("LP", name, instance, "frontier range J*(inf) - J*(0) (value of unsupported actions)",
                   fr["frontier_range"].iloc[0])


def x_find_record(records: Sequence[dict], method: str, variant: str) -> Optional[dict]:
    """The default-hyper-parameter record of (method, variant), or None if run_xmethods did not return it."""
    hits = [r for r in records if r["method"] == method and r["variant"] == variant and bool(r["is_default"])]
    assert len(hits) <= 1, f"several default records for {method} [{variant}]"
    return hits[0] if hits else None

# %%
# ============================================================
# II.5 Propositions: per-instance verification
# ============================================================
def _x_p1_block(log: XPropositionLog, e: dict, inst: str, W: Optional[np.ndarray], note: str) -> None:
    env, en = e["env"], e["label"]
    if e["imputed"]:
        res = [x_p1_policy(env, p, g, W) for p, g in zip(e["pis"], e["g"])]
        n_imp = sum(g["status"] != "ok" for g in e["g"])
        nt = f"{note}; improper under given: {n_imp}"
        for key, txt in (("dJ", "max |J(pi) - J(pi_W)|"), ("dL", "max |L(pi) - L(pi_W)|"),
                         ("dmu", "max ||mu_pi - mu_pi_W||_inf (state occupancy)"),
                         ("dv", "max ||v_pi - v_pi_W||_inf"), ("dabs", "max |absorption probability difference|"),
                         ("dP", "max |P_pi - P_pi_W| on live rows"), ("dr", "max |r_pi - r_pi_W| on live rows")):
            vals = [r[key] for r in res if np.isfinite(r[key])]
            if vals:
                log.check("P1", en, inst, txt, max(vals), 1e-9, notes=nt)
            else:
                log.not_run("P1", en, inst, txt, "no policy proper under the given rows")
        log.report("P1", en, inst, "max inadmissible mass of pi_W", max(r["proj_mass"] for r in res),
                   notes=f"{X_INVARIANT}: x_project asserts zero mass outside the mask; {nt}")
        log.check("P1", en, inst, "properness of pi and pi_W disagrees (count)",
                  sum(r["status_mismatch"] for r in res), 0.0, notes=nt)
        if W is not None:   # negative control: the uniform projection is the wrong W for this environment
            ctrl = [x_p1_policy(env, p, g, None)["dJ"] for p, g in zip(e["pis"], e["g"])]
            ctrl = [v for v in ctrl if np.isfinite(v)]
            log.check("P1", en, inst, "negative control: projecting with the wrong (uniform) W changes J (max |dJ|)",
                      max(ctrl) if ctrl else np.nan, 1e-6, op="ge", notes="the check has power")
    else:
        res = [x_p1_policy(env, p, g, None) for p, g in zip(e["pis"], e["g"])]
        vals = [r["dJ"] for r in res if np.isfinite(r["dJ"])]
        log.report("P1", en, inst, "rows are not an imputation: max |J(pi) - J(pi_W)| (projection not value-preserving)",
                   max(vals) if vals else np.nan, notes=note)


def _x_p2_block(log: XPropositionLog, e: dict, inst: str, opt: "XOptCache") -> None:
    env, en = e["env"], e["label"]
    if opt(env, "given", False) is None or opt(env, "given", True) is None:
        why = opt.reason(env, "given", False) or opt.reason(env, "given", True)
        gap = x_lp_optimum(env, "given", False) - x_lp_optimum(env, "given", True)
        nt = f"value iteration unavailable ({why}); exact occupancy-LP route"
        if e["imputed"]:
            log.check("P2", en, inst, "LP route: |J*_unconstrained - J*_admissible|", abs(gap), X_LP_TOL, notes=nt)
        else:
            log.check("P2", en, inst, "LP route: J*_unconstrained - J*_admissible >= 0 (non-imputed rows)", gap,
                      -X_LP_TOL, op="ge", notes=nt)
            log.report("P2", en, inst, "value of unsupported actions J*_unconstrained - J*_admissible", gap, notes=nt)
        return
    if not e["imputed"]:
        V_unc, V_con = opt(env, "given", False)[1], opt(env, "given", True)[1]
        gap = float(env.d0 @ (V_unc - V_con))
        log.check("P2", en, inst, "J*_unconstrained - J*_admissible >= 0 (non-imputed rows)", gap, -1e-9, op="ge")
        log.report("P2", en, inst, "value of unsupported actions J*_unconstrained - J*_admissible", gap)
        return
    r = x_p2_env(env, opt)
    log.check("P2", en, inst, "max |V*_unconstrained - V*_admissible| (live states)", r["dV"], 1e-9)
    log.check("P2", en, inst, "|J*_unconstrained - J*_admissible|", r["dJ"], 1e-9)
    log.check("P2", en, inst, "convexity: max_{a not in M} Q*(s,a) - max_{a' in M} Q*(s,a')", r["convexity_excess"], 1e-9)
    if np.isfinite(r["single_dev"]):
        log.check("P2", en, inst, "single-admissible states: max |Q*(s,a) - Q*(s,a_M)| (exact ties)", r["single_dev"],
                  1e-9, notes=f"{r['single_states']} single-admissible live states")
    else:
        log.not_run("P2", en, inst, "single-admissible states: exact ties", "no single-admissible live state")
    log.report("P2", en, inst, "live states where an unsupported action ties the optimum", r["tied_states"],
               notes=f"{r['tied_single']} of them single-admissible; relative tolerance 1e-9")
    log.report("P2", en, inst, "lowest-index greedy (earlier oracle): live states choosing an unsupported action",
               r["low_inadmissible_states"], notes=f"evaluation status given/terminate {r['low_status']}")
    log.report("P2", en, inst, "lowest-index greedy: J_given - J*", r["low_gap_given"], notes="0 when it is proper")
    log.report("P2", en, inst, "lowest-index greedy: J_terminate - J*", r["low_gap_terminate"],
               notes=f"expected violations under given {r['low_violations']:.4g}: the oracle artefact")


def _x_p3_block(log: XPropositionLog, e: dict, inst: str, opt: "XOptCache", note: str) -> None:
    env, en = e["env"], e["label"]
    res = [x_p3_policy(env, p, g, t) for p, g, t, ok in zip(e["pis"], e["g"], e["t"], e["ok"]) if ok]
    nt = f"{note}; proper under both semantics: {len(res)}/{len(e['pis'])}"
    if not res:
        log.not_run("P3", en, inst, "termination gap", "no policy proper under both semantics")
    else:
        lo, hi = env.value_bounds
        rt = env.terminate_reward
        log.check("P3", en, inst, "p(v_min - r_term) <= J_given - J_term <= p(v_max - r_term): max excess",
                  max(r["bound_excess"] for r in res), 1e-10, notes=f"{nt}; v_min={lo:g}, v_max={hi:g}, r_term={rt:g}")
        log.check("P3", en, inst, "identity: |gap - sum_s mu_term(s) sum_{a not in M} pi (Q_given - r_term)|",
                  max(r["identity"] for r in res), 1e-9, notes=nt)
        log.report("P3", en, inst, "p identical under both semantics", max(r["p_rules"] for r in res),
                   notes=f"{X_INVARIANT}: both evaluations read p from the terminate chain; {nt}")
        log.check("P3", en, inst, "p equals an independent first-violation value solve", max(r["p_indep"] for r in res),
                  1e-9, notes=nt)
        log.report("P3", en, inst, "p = E_term[violations]", max(r["p_term_viol"] for r in res),
                   notes=f"{X_INVARIANT}: under termination the evaluator sets p_any = expected violations; {nt}")
        log.check("P3", en, inst, "Markov: p - E_given[violations] <= 0", max(r["markov_excess"] for r in res), 1e-12,
                  notes=nt)
        log.report("P3", en, inst, "max |J_given - J_term| over policies", max(abs(r["gap"]) for r in res), notes=nt)
        log.report("P3", en, inst, "max p over policies", max(r["p"] for r in res), notes=nt)
    lo, _ = env.value_bounds
    if env.terminate_reward == lo:
        ot, oc = opt(env, "terminate", False), opt(env, "given", True)
        txt = "corollary (r_term = v_min): max |V*_terminate,unconstrained - V*_admissible|"
        if ot is None or oc is None:
            why = opt.reason(env, "terminate", False) or opt.reason(env, "given", True)
            log.check("P3", en, inst, "corollary (r_term = v_min), LP route: |J*_terminate,unconstrained - J*_admissible|",
                      abs(x_lp_optimum(env, "terminate", False) - x_lp_optimum(env, "given", True)), X_LP_TOL,
                      notes=f"value iteration unavailable ({why})")
        else:
            log.check("P3", en, inst, txt, float(np.abs(ot[1] - oc[1])[env.live].max()), 1e-9)
    else:
        log.report("P3", en, inst, "r_term - v_min (> 0: flat-terminate corollary not claimed)",
                   env.terminate_reward - lo)


def _x_p5_block(log: XPropositionLog, entries: List[dict], learn_env: FiniteEnv,
                policies: Sequence[Tuple[str, np.ndarray, str]], inst: str) -> None:
    proj = [x_project(pi, learn_env.mask, learn_env.live) for _, pi, _ in policies]
    for e in entries:
        env = e["env"]
        e["adm"] = [validate_policy(e["lift"](p) if e["lift"] is not None else p, f"projected policy on {env.name}",
                                    shape=(env.S, env.A)) for p in proj]
        log.check("P5", e["label"], inst, "lifted projected policies: max inadmissible mass (precondition)",
                  max(float(x_inadmissible_mass(p, env.mask, env.live).max()) for p in e["adm"]), 0.0)
    groups: List[List[Tuple[dict, str]]] = []
    for e in entries:
        for sem in X_SEMANTICS:
            for g in groups:
                if x_shares_admissible_rows(g[0][0]["env"], e["env"])[0]:
                    g.append((e, sem))
                    break
            else:
                groups.append([(e, sem)])
    for g in groups:
        if len(g) < 2:
            continue
        e0, s0 = g[0]
        name = " = ".join(f"{e['env'].name}:{s}" for e, s in g)
        ref = [x_evaluate(e0["env"], p, s0, vectors=True) for p in e0["adm"]]
        d = {"pol": 0.0, "J": 0.0, "L": 0.0, "mu": 0.0, "abs": 0.0, "mismatch": 0, "improper": 0}
        for e, s in g[1:]:
            for p0, p, r0 in zip(e0["adm"], e["adm"], ref):
                d["pol"] = max(d["pol"], float(np.abs(p - p0).max()))
                r = x_evaluate(e["env"], p, s, vectors=True)
                if r["status"] != r0["status"]:
                    d["mismatch"] += 1
                    continue
                if r["status"] != "ok":
                    d["improper"] += 1
                    continue
                d["J"] = max(d["J"], abs(r["return"] - r0["return"]))
                d["L"] = max(d["L"], abs(r["avg_episode_length"] - r0["avg_episode_length"]))
                d["mu"] = max(d["mu"], float(np.abs(r["occupancy_vector"] - r0["occupancy_vector"]).max()))
                d["abs"] = max(d["abs"], max(abs(r[k] - r0[k]) for k in
                                             ("absorb_death", "absorb_survival", "absorb_other_terminal")))
        nt = f"{len(g)} targets; {len(proj)} admissible policies; jointly improper pairs {d['improper']}"
        log.check("P5", name, inst, "same admissible policy on every target (precondition)", d["pol"], 0.0, notes=nt)
        log.check("P5", name, inst, "max |dJ| across targets", d["J"], 1e-10, notes=nt)
        log.check("P5", name, inst, "max |dL| across targets", d["L"], 1e-10, notes=nt)
        log.check("P5", name, inst, "max ||d mu||_inf across targets", d["mu"], 1e-10, notes=nt)
        log.check("P5", name, inst, "max |d absorption probability| across targets", d["abs"], 1e-10, notes=nt)
        log.check("P5", name, inst, "properness disagrees across targets (count)", d["mismatch"], 0.0, notes=nt)


def _x_p4_block(log: XPropositionLog, entries: List[dict], tg: Dict[str, tuple], inst: str,
                completions: Sequence[Tuple[str, str, str]]) -> None:
    def entry(env):
        return next(x for x in entries if x["env"] is env)

    M0, s0, _ = tg["true"]
    assert s0 == "given", "the 'true' target must use the true dynamics ('given')"
    e0 = entry(M0)
    for comp, m1_lab, b_lab in completions:
        M1, s1, _ = tg[m1_lab]
        Bn, sb, _ = tg[b_lab]
        e1, eb = entry(M1), entry(Bn)
        en = f"{M0.name} vs {M1.name}:{s1} [{comp} completion]"
        same = (M1.P.shape == M0.P.shape and M1.terminals == M0.terminals and M1.sink == M0.sink
                and np.array_equal(M1.mask, M0.mask) and np.array_equal(M1.d0, M0.d0)
                and np.array_equal(M1.entry_reward, M0.entry_reward))
        if not log.check("P4", en, inst, "M1 shares state space, mask, d0, rewards and sink with M0 (precondition)",
                         0.0 if same else 1.0, 0.0):
            continue
        P1 = x_semantic_P(M1, s1)
        log.check("P4", en, inst, "admissible live rows of M0 and M1 coincide (precondition)",
                  x_admissible_row_deviation(M0.P, P1, M0.mask, M0.live), X_IMPUTATION_TOL)
        log.check("P4", en, inst, "absorbing rows of M0 and M1 coincide (precondition)",
                  max(float(np.abs(M0.P[t] - P1[t]).max()) for t in M0.terminals), 0.0)
        log.check("P4", en, inst, "same lifted policy under M0 and M1 (precondition)",
                  max(float(np.abs(a - b).max()) for a, b in zip(e0["pis"], e1["pis"])), 0.0)
        tv = x_inadmissible_tv(M0.P, P1, M0.mask, M0.live)
        max_tv = float(tv.max())
        k1, kb = ("g" if s1 == "given" else "t"), ("g" if sb == "given" else "t")
        res = []
        for i, pi in enumerate(e0["pis"]):
            ev0, ev1, evb = e0["g"][i], e1[k1][i], eb[kb][i]
            if not (ev0["status"] == ev1["status"] == evb["status"] == "ok"):
                continue
            r = x_p4_policy(M0, P1, pi, ev0, ev1, tv, max_tv)
            r["estimation_term"] = ev1["return"] - evb["return"]
            r["total"] = ev0["return"] - evb["return"]
            res.append(r)
        del P1
        nt = f"proper policies {len(res)}/{len(e0['pis'])}; benchmark target '{b_lab}' ({Bn.name}:{sb})"
        if not res:
            log.not_run("P4", en, inst, "simulation-lemma identity", "no policy proper on M0, M1 and the benchmark")
            continue
        log.check("P4", en, inst, "identity |dJ - sum_s mu1(s) sum_{a not in M} pi (P0 - P1)^T U0|",
                  max(r["identity"] for r in res), 1e-9, notes=nt)
        log.check("P4", en, inst, "admissible pairs contribute nothing |sum_all - sum_inadmissible|",
                  max(r["admissible_part"] for r in res), 1e-12, notes=nt)
        log.check("P4", en, inst, "bound: |dJ| - E_M1[viol] max TV span(U0) (max excess)",
                  max(r["excess"] for r in res), 1e-12, notes=nt)
        log.check("P4", en, inst, "weighted bound: |dJ| - sum_s mu1 sum_{a not in M} pi TV span(U0) (max excess)",
                  max(r["weighted_excess"] for r in res), 1e-12, notes=nt)
        log.report("P4", en, inst, "max TV(P0, P1) over live unsupported pairs", max_tv)
        log.report("P4", en, inst, "max tightness |dJ| / (E_M1[viol] max TV span(U0))",
                   max(r["tightness"] for r in res), notes=nt)
        log.report("P4", en, inst, "mean |semantics term J_M0 - J_M1|", np.mean([abs(r["semantics_term"]) for r in res]),
                   notes=nt)
        log.report("P4", en, inst, "mean |estimation + coverage term J_M1 - J_bench|",
                   np.mean([abs(r["estimation_term"]) for r in res]), notes=nt)
        log.report("P4", en, inst, "mean |J_true - J_bench|", np.mean([abs(r["total"]) for r in res]), notes=nt)
        if M1 is Bn and s1 == sb:
            log.check("P4", en, inst, "benchmark = true rows + completion: estimation term (max |J_M1 - J_bench|)",
                      max(abs(r["estimation_term"]) for r in res), 0.0, notes=nt)


def x_theory_checks(log: XPropositionLog, inst: str, learn_env: FiniteEnv,
                    targets: Sequence[Tuple[str, FiniteEnv, str, Optional[Callable]]],
                    policies: Sequence[Tuple[str, np.ndarray, str]], opt: "XOptCache", construction: Optional[str] = None,
                    learn_env_imputed: bool = True, env_W: Optional[Dict[int, np.ndarray]] = None,
                    completions: Optional[Sequence[Tuple[str, str, str]]] = None) -> Dict[str, Any]:
    """P1-P5 on one instance: every distinct target environment x every policy (lifted where the target needs it)."""
    env_W = {} if env_W is None else env_W
    assert any(env is learn_env for _, env, _, _ in targets), "the learner's environment must be one of the targets"
    tg = {lab: (env, sem, lift) for lab, env, sem, lift in targets}
    assert len(tg) == len(targets), "duplicate target labels"
    entries: List[dict] = []
    for lab, env, sem, lift in targets:
        e = next((x for x in entries if x["env"] is env), None)
        if e is None:
            entries.append({"env": env, "lift": lift, "labels": [lab]})
        else:
            assert (e["lift"] is None) == (lift is None), f"{lab}: inconsistent lift for {env.name}"
            e["labels"].append(lab)
    kinds = {k: sum(1 for _, _, kk in policies if kk == k) for k in ("learned", "reference", "random")}
    note = "policies " + ", ".join(f"{k} {v}" for k, v in kinds.items())
    for e in entries:
        env, lift = e["env"], e["lift"]
        e["label"] = f"{env.name} [{', '.join(e['labels'])}]"
        e["pis"] = [validate_policy(lift(pi) if lift is not None else pi, f"{name} on {env.name}", shape=(env.S, env.A))
                    for name, pi, _ in policies]
        e["g"] = [x_evaluate(env, p, "given", vectors=True) for p in e["pis"]]
        e["t"] = [x_evaluate(env, p, "terminate", vectors=True) for p in e["pis"]]
        e["ok"] = [g["status"] == "ok" and t["status"] == "ok" for g, t in zip(e["g"], e["t"])]
        W = env_W.get(id(env))
        wname = "random W" if W is not None else "uniform W (mean rule)"
        dev = x_imputation_deviation(env, W)
        required = ((env is learn_env and learn_env_imputed) or "true_rows_mean" in e["labels"]
                    or id(env) in env_W)
        if required:
            e["imputed"] = log.check("P1", e["label"], inst, f"given rows equal the convex imputation [{wname}] "
                                     "(precondition)", dev, X_IMPUTATION_TOL)
        else:
            e["imputed"] = bool(dev <= X_IMPUTATION_TOL)
            log.report("P1", e["label"], inst, f"deviation of the given rows from the convex imputation [{wname}]",
                       dev, notes="P1 / P2 are claimed only for imputed rows")
        _x_p1_block(log, e, inst, W, note)
        _x_p2_block(log, e, inst, opt)
        _x_p3_block(log, e, inst, opt, note)
    _x_p5_block(log, entries, learn_env, policies, inst)
    if "true" in tg:
        if completions is None:
            completions = ((("mean", "bench_mean", "bench_mean"), ("terminate", "bench_terminate", "bench_terminate"))
                           if construction == "oracle" else
                           (("mean", "true_rows_mean", "bench_mean"), ("terminate", "true_rows_terminate", "bench_terminate")))
        _x_p4_block(log, entries, tg, inst, completions)
    else:
        log.not_run("P4", learn_env.name, inst, "exact error decomposition", "no ground-truth target (ICU-Sepsis)")
    n_ok = sum(sum(e["ok"]) for e in entries)
    return {"targets": [e["label"] for e in entries], "policies": len(policies), **kinds,
            "proper_pairs": int(n_ok), "env_policy_pairs": int(len(entries) * len(policies))}


def x_theory_p7(log: XPropositionLog, env_name: str, inst: str, pi_b: np.ndarray, live: np.ndarray,
                records: Optional[Sequence[dict]] = None) -> Dict[str, float]:
    """P7 per state for one behaviour estimate, plus cross-checks against the run_xmethods diffusion records."""
    K = pi_b.shape[1]
    diffusion = x_diffusion_for(K)
    kernels = x_bayes_reverse_kernels(pi_b, diffusion)
    r = x_p7_state_checks(pi_b, diffusion, kernels, live)
    del kernels
    nt = f"K={K}, T={diffusion.T}, alpha_bar_T={r['abar']:.6g}, {int(live.sum())} live states"
    log.check("P7", env_name, inst, "Bayes kernels are Markov: max |row sum - 1|", r["kernel_rowsum_dev"], 1e-10, notes=nt)
    log.report("P7", env_name, inst, "Bayes kernels are non-negative: min entry", r["kernel_min"],
               notes=f"{X_INVARIANT}: x_bayes_reverse_kernels clips at 0; {nt}")
    log.check("P7", env_name, inst, "q_T closed form: max |pi_b Qbar_T - (abar pi_b + (1 - abar) u)|", r["qT_closed_dev"],
              1e-12, notes=nt)
    log.check("P7", env_name, inst, "(a) reverse chain from q_T returns pi_b: max_s TV", r["tv_from_qT"], 1e-10, notes=nt)
    log.check("P7", env_name, inst, "(c) TV(u, q_T) = abar TV(u, pi_b): max abs difference", r["tv_identity_dev"], 1e-12,
              notes=nt)
    log.check("P7", env_name, inst, "(b) from uniform: max_s [TV(out, pi_b) - TV(u, q_T)]", r["contraction_excess"],
              1e-12, notes=nt)
    log.check("P7", env_name, inst, "(c) from uniform: max_s TV(out, pi_b) <= abar (1 - 1/K)", r["tv_from_u_max"],
              r["global_bound"] + 1e-12, notes=nt)
    log.report("P7", env_name, inst, "mean_s TV(out from uniform, pi_b)", r["tv_from_u_mean"], notes=nt)
    log.report("P7", env_name, inst, "median_s TV(out, pi_b) / TV(u, pi_b) (bound: alpha_bar_T)", r["ratio_median"],
               notes=f"max ratio {r['ratio_max']:.4g}")
    if records is not None:
        rec = x_find_record(records, "Diffusion (exact, unmasked)", "native")
        if rec is None:
            log.not_run("P7", env_name, inst, "'Diffusion (exact, unmasked)' cross-check", "record not returned")
        else:
            dev = float(np.abs(np.asarray(rec["pi"])[live] - r["out_u"][live]).max())
            log.check("P7", env_name, inst, "'Diffusion (exact, unmasked)' policy = uniform-start reverse marginal "
                      "(max abs, live)", dev, 1e-9, notes="same pi_b (training split, global fallback for unseen states)")
        fin = x_find_record(records, "Diffusion (exact, final-only)", "native")
        sfm = x_find_record(records, "SfBC (exact)", "masked")
        sfa = x_find_record(records, "SfBC (exact, argmax)", "native")
        # The deployed diffusion records are masked, deterministic (argmax) policies, so they are compared by argmax
        # with masked SfBC (masked against masked); a TV against a stochastic policy would be large even when both
        # pick the same action.
        if fin is not None and sfm is not None:
            agree = np.asarray(fin["pi"])[live].argmax(axis=1) == np.asarray(sfm["pi"])[live].argmax(axis=1)
            log.report("P7", env_name, inst, "argmax agreement: final-only guided diffusion vs argmax of SfBC (exact) "
                       "[masked]", float(agree.mean()), notes="fraction of live states; both masked; default hyper-parameters")
        else:
            log.not_run("P7", env_name, inst, "argmax agreement: final-only guided vs SfBC (exact) [masked]",
                        "record(s) not returned")
        if fin is not None and sfa is not None:
            agree = np.asarray(fin["pi"])[live].argmax(axis=1) == np.asarray(sfa["pi"])[live].argmax(axis=1)
            log.report("P7", env_name, inst, "argmax agreement: final-only guided (masked) vs SfBC (exact, argmax) "
                       "(unmasked)", float(agree.mean()), notes="fraction of live states; SfBC unmasked")
    r.pop("out_u")
    return r


def x_theory_p6(log: XPropositionLog) -> None:
    """P6: summary of Section 13's torch gradient tests plus closed-form numpy checks."""
    rg = globals().get("REG_GRAD_TESTS")
    if rg is None:
        log.not_run("P6", "Part I Section 13 (torch)", "REG_GRAD_TESTS", "regulariser gradient tests",
                    "REG_GRAD_TESTS missing")
    else:
        B = int(rg["batch_rows"])
        nt = f"batch {B}, admissible logged rows {rg.get('admissible_logged_rows')}"
        if "legacy_excluded_logged_action_derivative" in rg:
            log.check("P6", "Part I Section 13 (torch)", "REG_GRAD_TESTS",
                      "legacy: |dL/dQ(excluded logged action) + 1/B|",
                      abs(float(rg["legacy_excluded_logged_action_derivative"]) + 1.0 / B), 1e-12, notes=nt)
        else:
            log.not_run("P6", "Part I Section 13 (torch)", "REG_GRAD_TESTS", "legacy excluded-action derivative",
                        "the test batch had no excluded logged action")
        log.check("P6", "Part I Section 13 (torch)", "REG_GRAD_TESTS",
                  "all Section 13 regulariser unit tests passed (design B: zero leakage, all-excluded batch, normalisation)",
                  float(bool(rg.get("passed"))), 1.0, op="ge", notes=nt)
    c = x_p6_closed_form_checks(np_rng("xenv", "theory", "p6"))
    nt = f"batch {c['batch']}, excluded logged rows {c['excluded_rows']}, temperatures 0.5 / 1 / 2"
    env, inst = "closed form (numpy)", "random Q, masks, logged actions"
    log.check("P6", env, inst, "legacy: l(Q + h e_sa) - l(Q) + h for excluded logged a (unbounded below)",
              c["legacy_fd_dev"], 1e-12, notes=nt)
    log.report("P6", env, inst, "legacy: rows with admissible logged actions unaffected", c["legacy_untouched_rows_dev"],
               notes=f"{X_INVARIANT}: only the excluded rows are perturbed; {nt}")
    log.check("P6", env, inst, "legacy batch mean: |d/dQ(excluded logged action) + 1/B|", c["legacy_mean_derivative_dev"],
              1e-12, notes=nt)
    log.check("P6", env, inst, "design B: min contribution >= 0", c["design_b_min"], -1e-12, op="ge", notes=nt)
    log.check("P6", env, inst, "design B: invariant to every excluded coordinate (max change)",
              c["design_b_excluded_invariance"], 0.0, notes=nt)


def x_theory_tiny_suite(log: XPropositionLog, n_mdps: int = 20, S: int = 12, A: int = 4,
                        kappas: Sequence[float] = (0.0, 0.01, 0.1, 0.5, float("inf"))) -> pd.DataFrame:
    """P1-P5 and the LP cross-checks on random absorbing MDPs (true rows P0, uniform and random-W completions)."""
    sub = XPropositionLog()
    frontiers = []
    for i in range(n_mdps):
        P0, W = x_tiny_absorbing_mdp(i, S, A)
        E_W = P0.variant(f"{P0.name}-W", P=x_impute_inadmissible(P0.P, P0.mask, P0.live, W))
        E_U = P0.variant(f"{P0.name}-mean", P=x_impute_inadmissible(P0.P, P0.mask, P0.live))
        targets = [("bench_mean", E_U, "given", None), ("bench_terminate", E_U, "terminate", None),
                   ("true", P0, "given", None), ("true_rows_mean", E_U, "given", None),
                   ("true_rows_terminate", P0, "terminate", None), ("true_rows_W", E_W, "given", None)]
        pols = [(n, p, "random") for n, p in
                x_theory_random_policies(P0.mask, P0.live, np_rng("xenv", "theory", "tiny-policies", i), 20, 5, 5)]
        opt = XOptCache()
        x_theory_checks(sub, f"tiny|{i:02d}", E_U, targets, pols, opt, construction="best", env_W={id(E_W): W},
                        completions=(("mean", "true_rows_mean", "bench_mean"),
                                     ("terminate", "true_rows_terminate", "bench_terminate"),
                                     ("random W", "true_rows_W", "true_rows_W")))
        for lab, env, sem, Wf in (("true", P0, "given", None), ("true_rows_terminate", P0, "terminate", None),
                                  ("bench_mean", E_U, "given", None), ("true_rows_W", E_W, "given", W)):
            fr = x_exact_frontier(env, sem, kappas, opt, {"family": "tiny", "cell": "tiny", "draw": i, "target": lab},
                                  W=Wf)
            x_frontier_checks(sub, fr, f"tiny|{i:02d}")
            frontiers.append(fr)
    log.merge_aggregated(sub, environment=f"random absorbing MDPs ({n_mdps} x S={S}, A={A})",
                         instance=f"tiny|00-{n_mdps - 1:02d}")
    return pd.concat(frontiers, ignore_index=True)


# ---------- draw-0 instances, learned policies and frontier problems ----------
def x_cell_construction(spec: dict) -> Optional[str]:
    """'best' / 'oracle' for Gumbel cells (from the cell key, cross-checked with spec['construction'] if present)."""
    parts = str(spec["cell"]).split("|")
    assert parts[0] == spec["family"], f"cell key {spec['cell']!r} does not start with its family"
    if spec["family"] != "gumbel":
        return None
    c = parts[1]
    assert c in ("best", "oracle"), spec["cell"]
    if "construction" in spec:
        assert spec["construction"] == c, (spec["construction"], spec["cell"])
    return c


def x_theory_jobs() -> List[Tuple[str, Optional[dict]]]:
    """(role, cell spec) for every instance of the verification set; role 'icu' with spec None = X_ICU alone."""
    icu = [s for s in X_CELLS if s["family"] == "icu"]
    best = [s for s in X_CELLS if s["family"] == "gumbel" and x_cell_construction(s) == "best"]
    oracle = [s for s in X_CELLS if s["family"] == "gumbel" and x_cell_construction(s) == "oracle"]
    jobs: List[Tuple[str, Optional[dict]]] = [("icu", icu[0] if icu else None)]
    if best:
        jobs.append(("gumbel_best", best[0]))
    if oracle:
        jobs.append(("gumbel_oracle", oracle[0]))
    # Physician controls (e.g. withhold-vaso: unsupported actions have value in truth): frontiers only, no theory block.
    jobs += [("gumbel_control", s) for s in best[1:] if s.get("physician", "optimal") != "optimal"]
    jobs += [("garnet", s) for s in X_CELLS if s["family"] == "garnet"]
    return jobs


def x_frontier_problems(role: str, inst: Optional["CellInstance"]) -> List[Tuple[str, FiniteEnv, str]]:
    """(target label, env, semantics) frontier problems of one verification instance."""
    if role == "icu":
        return [("bench_mean", X_ICU, "given"), ("bench_terminate", X_ICU, "terminate")]
    tg = {lab: (env, sem) for lab, env, sem, _ in inst.targets}
    want = {"gumbel_best": ["bench_mean", "bench_terminate", "true"], "gumbel_oracle": ["bench_mean", "true"],
            "gumbel_control": ["bench_mean", "bench_terminate", "true"], "garnet": ["true"]}[role]
    missing = [w for w in want if w not in tg]
    if missing:
        raise KeyError(f"{inst.cell}: frontier targets {missing} missing from the instance")
    return [(lab, tg[lab][0], tg[lab][1]) for lab in want]


def x_theory_records(inst: "CellInstance", key: str) -> List[dict]:
    """run_xmethods on the draw-0 training split, cached on disk under an identity (resumable, FORCE_RETRAIN-aware)."""
    # x_result_xcfg (Section II.2): draw counts, cells, frontier budgets and bootstrap settings do not change what
    # run_xmethods computes, so extending a lite run to the full design keeps these records.
    identity = {"purpose": "II.5 theory records", "instance": inst.identity(), "key": key,
                "code": code_hash(*XENV_CORE_CODE, *XENV_ENV_CODE, *XENV_METHOD_CODE), "xcfg": x_result_xcfg(),
                "numeric": X_NUMERIC_CONSTANTS,
                "diffusion": [CFG.diffusion_T, CFG.diffusion_beta_start, CFG.diffusion_beta_end], "run_mode": RUN_MODE}
    h = obj_hash(identity)
    d = XENV_CACHE_DIR / "theory" / slug(key)
    npz_p, side_p = d / f"{h[:20]}.npz", d / f"{h[:20]}.identity.json"
    S, A = inst.learn_env.S, inst.learn_env.A
    if npz_p.exists() and side_p.exists():
        try:
            meta = json.loads(side_p.read_text())
        except XCACHE_READ_ERRORS as exc:     # a write cut by a disconnect: a miss, recomputed below
            print(f"WARNING: unreadable theory-record sidecar {side_p} ({type(exc).__name__}); recomputing.")
            meta = None
        if meta is not None and meta.get("identity_hash") != h:
            raise CheckpointIdentityError(f"{side_p} does not match the requested Part II theory identity.")
        if meta is not None and not (FORCE_RETRAIN and meta.get("force_token") != FORCE_TOKEN):
            try:
                with np.load(npz_p) as z:
                    pis = [np.array(z[f"pi_{i}"]) for i in range(len(meta["records"]))]
            except (*XCACHE_READ_ERRORS, ValueError, KeyError, zipfile.BadZipFile) as exc:
                print(f"WARNING: unreadable theory-record file {npz_p} ({type(exc).__name__}); recomputing.")
                pis = None
            if pis is not None:
                recs = [{**m, "pi": validate_policy(p, m["method"], shape=(S, A))} for m, p in zip(meta["records"], pis)]
                print(f"  {key}: loaded {len(recs)} cached learner records")
                return recs
    XCache.require_allowed(key)
    recs = run_xmethods(LearnerView.from_env(inst.learn_env), inst.train, key)
    for r in recs:
        validate_policy(r["pi"], f"{r['method']} [{r['variant']}]", shape=(S, A))
    d.mkdir(parents=True, exist_ok=True)
    tmp = npz_p.with_name(npz_p.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **{f"pi_{i}": np.asarray(r["pi"], dtype=np.float64) for i, r in enumerate(recs)})
    x_atomic_replace(tmp, npz_p)
    side_tmp = side_p.with_name(side_p.name + ".tmp")
    save_json({"key": key, "identity_hash": h, "identity": identity, "saved_utc": utc_now(), "force_token": FORCE_TOKEN,
               "records": [{k: v for k, v in r.items() if k != "pi"} for r in recs]}, side_tmp)
    x_atomic_replace(side_tmp, side_p)
    return recs


def x_theory_policies(inst_label: str, learn_env: FiniteEnv, behavior: np.ndarray,
                      records: Sequence[dict]) -> List[Tuple[str, np.ndarray, str]]:
    """Learned records + references + random policies on the learner's state space: (name, pi, kind)."""
    S, A = learn_env.S, learn_env.A
    out = [(x_label(r["method"], r["variant"], r["hp"], r["is_default"]), r["pi"], "learned") for r in records]
    out += [("REF behaviour", behavior, "reference"),
            ("REF uniform-admissible", x_uniform_admissible(learn_env.mask, learn_env.live), "reference"),
            ("REF uniform (all actions)", np.full((S, A), 1.0 / A), "reference")]
    rng = np_rng("xenv", "theory", "policies", inst_label)
    out += [(n, p, "random") for n, p in x_theory_random_policies(learn_env.mask, learn_env.live, rng, 6, 2)]
    return [(n, validate_policy(p, n, shape=(S, A)), k) for n, p, k in out]


def x_proposition_summary(checks: pd.DataFrame, ran: bool) -> pd.DataFrame:
    rows = []
    for p, title in X_PROPOSITION_TITLES.items():
        sub = checks[checks["proposition"] == p]
        is_check = sub["passes"].notna()
        n_chk = int(is_check.sum())
        n_pass = int(sub.loc[is_check, "passes"].astype(bool).sum())
        n_nr = int(sub["notes"].astype(str).str.startswith("not run").sum())
        status = "not run" if (not ran or n_chk == 0) else ("verified" if n_pass == n_chk else "FAILED")
        rows.append({"proposition": p, "title": title, "checks": n_chk, "passed": n_pass, "failed": n_chk - n_pass,
                     "descriptive": int((~is_check).sum()) - n_nr, "not_run": n_nr,
                     "instances": "; ".join(sorted(set(sub["instance"].astype(str)))), "status": status,
                     "verified_in": "Section II.5 (xenv/proposition_checks.csv)"})
    return pd.DataFrame(rows)

# %%
# ============================================================
# II.5 Run: propositions on the verification set, exact frontiers
# ============================================================
_X_RUN_THEORY = bool(EXPERIMENTS["xenv_theory"])
_X_RUN_FRONTIERS = bool(EXPERIMENTS["xenv_frontiers"])
PROP_LOG = XPropositionLog()
X_THEORY_INFO: Dict[str, Any] = {"jobs": [], "theory": _X_RUN_THEORY, "frontiers": _X_RUN_FRONTIERS}
_x_frontiers: List[pd.DataFrame] = []
_t_theory = time.perf_counter()
if _X_RUN_THEORY or _X_RUN_FRONTIERS:
    assert globals().get("X_ICU") is not None, "II.5 needs X_ICU (Section II.3 builds it when any xenv switch is on)"

if _X_RUN_THEORY:
    _t0 = time.perf_counter()
    X_THEORY_INFO["tiny_frontiers"] = int(len(x_theory_tiny_suite(PROP_LOG)))
    x_theory_p6(PROP_LOG)
    _e_main = XEmpirical(MAIN.data, S_N, A_N, BENCH.terminal_states)
    x_theory_p7(PROP_LOG, X_ICU.name, "icu|MAIN (Part I dataset)", _e_main.pi_b, X_ICU.live)
    del _e_main
    print(f"II.5 random MDPs, P6 and ICU P7 done ({time.perf_counter() - _t0:.1f} s)")

if _X_RUN_THEORY or _X_RUN_FRONTIERS:
    for _role, _spec in x_theory_jobs():
        if _role == "gumbel_control" and not _X_RUN_FRONTIERS:   # a frontier-only job
            continue
        _t0 = time.perf_counter()
        _opt = XOptCache()
        _inst = build_cell_instance(_spec, 0) if _spec is not None else None
        _cell = _spec["cell"] if _spec is not None else "icu|released"
        _label = f"{_cell}|d0" if _spec is not None else "icu|X_ICU (no ICU cell: reference and random policies)"
        _info = {"role": _role, "cell": _cell}
        if _X_RUN_THEORY and _role != "gumbel_control":
            if _inst is not None:
                _key = f"{_cell}|draw0"           # the driver's run_xmethods key for draw 0
                _records = x_theory_records(_inst, _key)
                _learn, _targets, _behavior = _inst.learn_env, _inst.targets, _inst.behavior
            else:
                _records, _learn, _behavior = [], X_ICU, X_ICU_BEHAVIOR
                _targets = [("bench_mean", X_ICU, "given", None), ("bench_terminate", X_ICU, "terminate", None)]
            _pols = x_theory_policies(_label, _learn, _behavior, _records)
            _imputed = bool(MEAN_RULE_IDENTITY_HOLDS) if _role == "icu" else True
            if _role == "icu":
                PROP_LOG.check("P1", X_ICU.name, _label, "direct imputation test agrees with Section 3's mean-rule identity",
                               float((x_imputation_deviation(X_ICU) <= X_IMPUTATION_TOL) != _imputed), 0.0,
                               notes=f"MEAN_RULE_IDENTITY_HOLDS = {bool(MEAN_RULE_IDENTITY_HOLDS)}")
            _info["checks"] = x_theory_checks(PROP_LOG, _label, _learn, _targets, _pols, _opt,
                                              construction=x_cell_construction(_spec) if _spec is not None else None,
                                              learn_env_imputed=_imputed)
            if _inst is not None:
                assert _inst.learn_env.S == _learn.S
                if _role == "gumbel_oracle":   # the oracle benchmark is exactly the true rows + mean completion
                    _tg = {lab: env for lab, env, _, _ in _inst.targets}
                    _Bo, _Tm = _tg["bench_mean"], _tg["true"]
                    _imp = x_impute_inadmissible(_Tm.P, _Bo.mask, _Tm.live)
                    PROP_LOG.check("P4", _Bo.name, _label, "oracle benchmark = true rows + mean completion on its mask "
                                   "(max |dP| + 1[d0 differs from the true d0])",
                                   float(np.abs(_Bo.P - _imp).max()) + float(not np.array_equal(_Bo.d0, _Tm.d0)),
                                   X_IMPUTATION_TOL, notes="the estimation term of the oracle construction is zero")
                    del _imp, _Bo, _Tm, _tg
                _e = XEmpirical(_inst.train, _learn.S, _learn.A, _learn.terminals)
                x_theory_p7(PROP_LOG, _learn.name, _label, _e.pi_b, _learn.live, _records)
                del _e
            del _records, _pols
        if _X_RUN_FRONTIERS:
            for _tlab, _env, _sem in x_frontier_problems(_role, _inst):
                _fr = x_exact_frontier(_env, _sem, XCFG.frontier_kappas, _opt,
                                       {"family": "icu" if _role == "icu" else _spec["family"], "cell": _cell, "draw": 0,
                                        "target": _tlab})
                _x_frontiers.append(_fr)
                if _X_RUN_THEORY:
                    x_frontier_checks(PROP_LOG, _fr, _label)
        _info["seconds"] = time.perf_counter() - _t0
        X_THEORY_INFO["jobs"].append(_info)
        print(f"II.5 {_role:<14} {_cell:<36} {_info['seconds']:7.1f} s")
        del _inst, _opt
        gc.collect()

# %%
# ============================================================
# II.5 Summary, outputs and globals
# ============================================================
PROP_CHECKS = PROP_LOG.frame() if _X_RUN_THEORY else pd.DataFrame(columns=X_PROP_COLUMNS)
PROP_SUMMARY = x_proposition_summary(PROP_CHECKS, _X_RUN_THEORY)
_n_chk = int(PROP_CHECKS["passes"].notna().sum()) if len(PROP_CHECKS) else 0
PROPOSITIONS_OK = (bool(_n_chk > 0 and PROP_CHECKS.loc[PROP_CHECKS["passes"].notna(), "passes"].astype(bool).all())
                   if _X_RUN_THEORY else None)
X_FRONTIERS = (pd.concat(_x_frontiers, ignore_index=True) if _x_frontiers
               else pd.DataFrame(columns=X_FRONTIER_COLUMNS))
X_FRONTIERS_OK = bool(len(X_FRONTIERS) and X_FRONTIERS["passes"].astype(bool).all()) if _X_RUN_FRONTIERS else None
PROP_CHECKS.to_csv(XENV_TABLE_DIR / "proposition_checks.csv", index=False)
PROP_SUMMARY.to_csv(XENV_TABLE_DIR / "proposition_summary.csv", index=False)
X_FRONTIERS.to_csv(XENV_TABLE_DIR / "exact_frontiers.csv", index=False)
X_THEORY_INFO["seconds"] = time.perf_counter() - _t_theory
del _x_frontiers

if _X_RUN_THEORY:
    display(PROP_SUMMARY[["proposition", "title", "checks", "passed", "failed", "descriptive", "not_run", "status"]])
    _fail = PROP_CHECKS[PROP_CHECKS["passes"].eq(False).fillna(False)]
    if len(_fail):
        print(f"WARNING: {len(_fail)} proposition check(s) FAILED (the release gate will stop):")
        display(_fail)
    print(f"PROPOSITIONS_OK = {PROPOSITIONS_OK} ({_n_chk} checks)")
else:
    print("II.5 propositions not run (EXPERIMENTS['xenv_theory'] = False).")
if _X_RUN_FRONTIERS:
    _fs = (X_FRONTIERS.groupby(["family", "cell", "target", "semantics"], sort=False)
           .agg(J_kappa0=("objective", "first"), J_inf=("objective", "last"), flat_expected=("flat_expected", "first"),
                frontier_range=("frontier_range", "first"), passes=("passes", "all")).reset_index())
    display(_fs)
    print(f"X_FRONTIERS_OK = {X_FRONTIERS_OK}. Flat frontiers under convex-imputation rows (P2) or termination with "
          "r_term = v_min (P3) are findings, not errors.")
else:
    print("II.5 frontiers not run (EXPERIMENTS['xenv_frontiers'] = False).")
print(f"II.5 total {X_THEORY_INFO['seconds']:.1f} s")

XENV_THEORY_CODE = [XPropositionLog, x_tv, x_imputation_deviation, x_admissible_row_deviation, x_inadmissible_tv,
                    x_shares_admissible_rows, x_theory_random_policies, XOptCache, x_lp_optimum, x_p1_policy,
                    x_p2_env, x_p3_policy, x_p4_policy, _x_masked_lse, x_p6_closed_form_checks, x_reverse_marginal,
                    x_p7_state_checks, x_tiny_absorbing_mdp, x_exact_frontier, x_frontier_checks, x_find_record,
                    _x_p1_block, _x_p2_block, _x_p3_block, _x_p5_block, _x_p4_block, x_theory_checks, x_theory_p7,
                    x_theory_p6, x_theory_tiny_suite, x_cell_construction, x_theory_jobs, x_frontier_problems,
                    x_theory_records, x_theory_policies, x_proposition_summary]
