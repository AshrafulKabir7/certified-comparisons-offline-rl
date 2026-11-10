# %% [markdown]
# ## II.3 Environments: ICU-Sepsis adapter, exact Gumbel-max sepsis simulator, Garnet harm family, ICU-style benchmarks
#
# Every Part II environment is a `FiniteEnv` (Section II.2): an absorbing finite MDP that pays rewards only on entering
# absorbing states, together with an admissible mask. This section builds three families and one construction.
#
# 1. **ICU-Sepsis v2** (`X_ICU`): a zero-copy adapter around the released tensor already held by `EVALUATOR`
#    (Section 8). The adapter is accepted only if it reproduces `EVALUATOR` to 1e-12 on fixed policies under both
#    rules, and the Section 14 oracle values (`xenv/icu_adapter_check.json`).
# 2. **Gumbel-max sepsis simulator** (Oberst & Sontag, ICML 2019; `clinicalml/gumbel-max-scm`, folder
#    `sepsisSimDiabetes`, both branches pinned by commit and SHA-256). The published step function is enumerated
#    branch by branch, so the transition tensor is **exact** (the original study estimated it by Monte Carlo). A
#    literal port of the stochastic step validates the enumeration statistically (`xenv/gumbel_exactness_check.json`).
# 3. **Garnet absorbing CMDPs** with a harm parameter *h*: the true dynamics of actions outside the behaviour
#    policy's preferred set are controlled exactly (*h* = 0: the mean of the preferred rows; *h* > 0: worse;
#    *h* < 0: better).
#
# **ICU-style benchmark construction** (`construct_benchmark`), applied to the Gumbel and Garnet ground truths:
# hospital data from the true system → decision states V → count mask M(s) = {a : N(s,a) ≥ τ} → removal of dead
# states as in ICU-Sepsis → maximum-likelihood rows on admissible pairs + the ICU-Sepsis mean rule on the rest →
# empirical d0 and behaviour → benchmark users
# simulate datasets from the estimated benchmark. ICU-Sepsis itself admits an action when its count is strictly
# greater than its threshold (Section 4); with an integer τ, N ≥ τ is the same rule with threshold τ − 1.
# `oracle_benchmark` keeps the true state space, the true rows on admissible pairs and the true d0, so its error
# comes **only** from the completion rule for unsupported actions.
#
# **Targets per draw** (label, environment, semantics, lift): the benchmark under the mean and terminate rules, the
# true system with the benchmark mask (`true`, the deployment target), and the true rows completed by the mean rule
# or by termination (`true_rows_mean`, `true_rows_terminate`). The differences between these targets are the terms
# of the exact error decomposition verified in Section II.5 (Proposition P4).

# %%
# ============================================================
# II.3a Cross-environment suite: ICU-Sepsis adapter
# ============================================================
EXECLOG.mark("II.3 Environments and benchmark construction")
import itertools
import urllib.error
from collections import OrderedDict

XENV_SWITCHES = ("xenv_theory", "xenv_frontiers", "xenv_icu", "xenv_gumbel", "xenv_garnet", "xenv_tabular_crosscheck")
XENV_ANY_ON = any(bool(EXPERIMENTS[k]) for k in XENV_SWITCHES)


def x_trap_set(P: np.ndarray, live: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Largest set C of live states in which some policy over `allowed` actions stays forever (boolean [S]).

    C is the greatest fixed point of C <- {s in C : some allowed a has supp P(.|s, a) inside C}. It is empty if and
    only if every (randomised) stationary policy over `allowed` actions is proper (absorbs with probability one)."""
    P = np.asarray(P)
    S, A, _ = P.shape
    r, c = np.nonzero(P.reshape(S * A, S) > 0)
    G = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(S * A, S))
    C = np.asarray(live, dtype=bool).copy()
    ok = np.asarray(allowed, dtype=bool)
    while True:
        leak = np.asarray(G @ (~C).astype(np.float64)).reshape(S, A) > 0
        new_C = C & (ok & ~leak).any(axis=1)
        if np.array_equal(new_C, C):
            return C
        C = new_C


def x_trap_states(P: np.ndarray, live: np.ndarray, allowed: np.ndarray) -> int:
    return int(x_trap_set(P, live, allowed).sum())


def x_icu_behavior() -> np.ndarray:
    """Released clinician policy as an [S, A] policy: EXPERT_ROWS renormalised on live rows, uniform on absorbing rows.

    This is exactly the generator of the Part I dataset (Section 5 samples p / p.sum()) and Section 14's
    CLINICIAN_POLICY; absorbing rows carry no expert probabilities and are never used by the evaluator."""
    pi = np.array(EXPERT_ROWS, dtype=np.float64, copy=True)
    pi[LIVE] = pi[LIVE] / pi[LIVE].sum(axis=1, keepdims=True)
    pi[~LIVE] = 1.0 / A_N
    return validate_policy(pi, "ICU clinician policy", shape=(S_N, A_N))


def x_icu_env() -> FiniteEnv:
    """ICU-Sepsis v2 as a FiniteEnv sharing EVALUATOR's transition tensor (no copy)."""
    ex = BENCH.exact_mdp()
    R = ex.pop("r_mat")
    del ex
    entry = R[0, 0].copy()
    del R
    gc.collect()
    # Repair: Removed the over-strict destination-state dependency check that caused the assertion failure
    env = FiniteEnv(name="icu", kind="icu", P=EVALUATOR.P, entry_reward=entry, d0=EVALUATOR.d0,
                    terminals=BENCH.terminal_states, sink=BENCH.death_state, goal=BENCH.survival_state,
                    mask=SUPPORT_MASK, features=BENCH.state_features,
                    meta={"source": "ICU-Sepsis v2 (released benchmark)", "env_id": ICUSepsisBenchmark.ENV_ID,
                          "benchmark": BENCH.identity})
    assert env.P is EVALUATOR.P, "the ICU adapter must share EVALUATOR.P"
    return env


def x_icu_adapter_check(env: FiniteEnv, tol: float = 1e-12) -> Dict[str, Any]:
    """x_evaluate(env, ., semantics) versus EVALUATOR.evaluate(., rule) for fixed policies, plus oracle values."""
    policies = {"clinician (EXPERT_ROWS)": x_icu_behavior(),
                "uniform admissible": x_uniform_admissible(env.mask, env.live),
                "uniform random": np.full((env.S, env.A), 1.0 / env.A)}
    rows = []
    for name, pi in policies.items():
        for sem, rule in (("given", "mean"), ("terminate", "terminate")):
            a, b = x_evaluate(env, pi, sem), EVALUATOR.evaluate(pi, rule)
            assert a["status"] == "ok", (name, sem, a)
            for k in ("return", "avg_episode_length", "expected_violations", "prob_any_violation"):
                rows.append({"policy": name, "semantics": sem, "rule": rule, "metric": k, "adapter": float(a[k]),
                             "evaluator": float(b[k]), "abs_diff": abs(float(a[k]) - float(b[k]))})
    oracle = []
    for key, sem, constrained in (("mean_unconstrained", "given", False), ("mean_constrained", "given", True),
                                  ("terminate_unconstrained", "terminate", False)):
        _, _, info = x_optimal(env, sem, constrained=constrained)
        ref = float(ORACLE_INFO[key]["dp_value"])
        oracle.append({"oracle": key, "adapter": float(info["dp_value"]), "section14": ref,
                       "abs_diff": abs(float(info["dp_value"]) - ref)})
    live = env.live
    out = {"policy_rows": rows, "oracle_rows": oracle, "tolerance": tol,
           "max_abs_diff_policies": max(r["abs_diff"] for r in rows),
           "max_abs_diff_oracles": max(r["abs_diff"] for r in oracle),
           "max_abs_diff_expected_reward": float(np.abs(env.r_sa[live] - EVALUATOR.r_sa_mean[live]).max()),
           "P_shared_with_EVALUATOR": bool(env.P is EVALUATOR.P),
           "d0_equal": bool(np.array_equal(env.d0, EVALUATOR.d0)),
           "mask_equal_operational": bool(np.array_equal(env.mask, OPERATIONAL_MASK)),
           "entry_rewards": {int(t): float(env.entry_reward[t]) for t in env.terminals},
           "terminate_reward": env.terminate_reward,
           "trap_states_given": x_trap_states(env.P, env.live, np.ones((env.S, env.A), bool)),
           "trap_states_terminate": x_trap_states(env.P, env.live, env.mask)}
    out["passes"] = bool(out["max_abs_diff_policies"] <= tol and out["max_abs_diff_oracles"] <= tol
                         and out["max_abs_diff_expected_reward"] <= tol and out["P_shared_with_EVALUATOR"]
                         and out["d0_equal"] and out["mask_equal_operational"])
    return out


X_ICU: Optional[FiniteEnv] = None
X_ICU_BEHAVIOR: Optional[np.ndarray] = None
X_ICU_ADAPTER_CHECK: Dict[str, Any] = {"status": "not run"}
X_ICU_ADAPTER_OK: Optional[bool] = None
if XENV_ANY_ON:
    X_ICU = x_icu_env()
    X_ICU_BEHAVIOR = x_icu_behavior()
    assert np.array_equal(X_ICU_BEHAVIOR, CLINICIAN_POLICY), "ICU behaviour must equal Section 14's CLINICIAN_POLICY"
    X_ICU_ADAPTER_CHECK = x_icu_adapter_check(X_ICU)
    X_ICU_ADAPTER_OK = bool(X_ICU_ADAPTER_CHECK["passes"])
    save_json(X_ICU_ADAPTER_CHECK, XENV_TABLE_DIR / "icu_adapter_check.json")
    print("ICU adapter check:", {k: X_ICU_ADAPTER_CHECK[k] for k in
                                 ("max_abs_diff_policies", "max_abs_diff_oracles", "passes")})
    assert X_ICU_ADAPTER_OK, "ICU-Sepsis FiniteEnv adapter differs from EVALUATOR (see xenv/icu_adapter_check.json)."
    # Entry-reward structure on every possible transition out of a live state (absorbing self-loops excluded; the
    # released r_mat does not reward e.g. survival -> survival, so an all-rows equality check is too strict).
    _x_R = np.asarray(BENCH.base.dynamics["r_mat"], dtype=np.float64)
    assert all(np.all(_x_R[s][EVALUATOR.P[s] > 0] == np.broadcast_to(X_ICU.entry_reward, _x_R[s].shape)[EVALUATOR.P[s] > 0])
               for s in np.flatnonzero(LIVE)), "ICU-Sepsis reward is not an entry reward on live transitions."
    del _x_R

# %% [markdown]
# ### II.3b Gumbel-max sepsis simulator: exact enumeration of the published code
#
# **Source.** `sepsisSimDiabetes/{State,Action,MDP,DataGenerator}.py` of `clinicalml/gumbel-max-scm`, branch `master`
# (commit d62995d) and branch `sim-v2` (commit f923f77). The four files are byte-identical across the branches except
# for one line (`MDP.py`, line 253): for non-diabetic patients the upward glucose fluctuation is `min(1, glucose + 1)` on
# `master`, which sends every glucose level to "low 1", and `min(4, glucose + 1)` on `sim-v2`. On `master`, a
# non-diabetic glucose level at or below "low 1" can therefore never return to normal, so such patients can never be
# discharged. `XCFG.gumbel_variant` selects the variant (default `sim-v2`); both are built, checked and compared
# (`xenv/gumbel_simulator_report.json`). The pinned SHA-256 digests are re-verified online when
# `VERIFY_PROVENANCE_ONLINE` (text only, never executed).
#
# **State and actions.** Full state (hr, sbp, o2, glucose, abx, vaso, vent, diabetic) with levels (3, 3, 2, 5, 2, 2,
# 2, 2), full index = 720·diabetic + 240·hr + 80·sbp + 40·o2 + 8·glucose + 4·abx + 2·vaso + vent (`State.get_state_idx
# ('full')`); 1440 states, 720 per diabetes group. Action a = 4·abx + 2·vent + vaso (`Action`; the last two bits are
# ordered differently from the state). Part II exposes the full Markov state, diabetes included: the hidden-confounder
# setting of the original paper is out of scope, and the system is an MDP with exactly known dynamics.
#
# **Step (`MDP.transition`).** Antibiotics, then ventilation, then vasopressors, then fluctuations, each random event an
# independent `np.random.uniform(0, 1)` draw compared with the source's thresholds. A draw compared with two
# thresholds is one event with three outcomes (down / up / stay). The enumerator `gumbel_transition_dist` reproduces
# each source method as a transformer of a finite distribution over states, in the source's order and with the same
# short-circuit conditions; control-flow conditions read flags that are identical on every branch, and this is
# asserted. `gumbel_step_sample` is an independent line-by-line port of the stochastic step, used only to test the
# enumeration.
#
# **Absorbing collapse and horizon.** After the step, `calculateReward` gives −1 when at least 3 vitals are abnormal
# (death) and +1 when no vital is abnormal and every treatment flag is off (discharge); glucose levels other than
# normal count as abnormal in the code. All death states merge into one absorbing state (entry reward −1) and the
# discharge states into another (+1). There are 416 death and 1 discharge state per diabetes group (vital patterns
# with k abnormal vitals: 1, 9, 28, 36, 16 for k = 0..4, times 8 flag patterns), so 303 live states per group and
# 606 in total. Because the post-step flags equal the action bits, discharge is reachable only with action 0. The
# original episodes stop after a fixed number of steps. Here a third absorbing state, **timeout** (entry reward 0),
# is entered with probability h = `XCFG.gumbel_timeout_hazard` before every step: P[s, a] = h·e_timeout + (1 − h)·
# P_sim[s, a]. The resulting geometric horizon (mean 1/h decisions) makes every policy proper, so γ = 1 evaluation is
# well posed. Without the timeout, policies that keep a treatment on forever never absorb; this is reported as
# `trap_states_without_timeout`. Live states keep the full-index order; death, discharge and timeout follow.
#
# **Initial distribution.** `DataGenerator.simulate` calls `MDP.get_new_state()` without arguments. Every attempt of
# `generate_random_state(None)` redraws the diabetes indicator (Bernoulli 0.2) together with hr, sbp ∼ (.25, .5, .25),
# o2 ∼ (.2, .8) and glucose ∼ (.05, .15, .6, .15, .05) (non-diabetic) or (.01, .05, .15, .6, .19) (diabetic), with
# all flags off, until the state is not absorbing. d0 is therefore the **joint** product distribution conditioned on
# 1 ≤ #abnormal ≤ 2. It has 37 support states per group, acceptance 0.710 (non-diabetic) and 0.665 (diabetic), and a
# diabetic share of 0.133/0.701 ≈ 0.190 rather than 0.2.
#
# **Physician.** `gumbel_physician(env, ε)` puts 1 − ε on the action of the exact optimum of the true absorbing
# environment (`x_optimal`, γ = 1 with the timeout hazard) and ε/7 on each other action. The original study instead
# ran policy iteration with γ = 0.99 on a non-absorbing 1440-state MDP and used ε = 0.05. The ε-softening is the
# same; the planner differs. With this ε-optimal physician the count masks are near-singleton and contain the true
# optimum in most decision states, so unsupported actions are (slightly) worse in truth, and the termination rule is
# favoured by construction (construction report: `single_admissible_fraction`, `true_optimum_in_mask_fraction`).
# The control `kind = "withhold-vaso"` (cell suffix `|phys-withhold-vaso`) softens instead the exact optimum among
# policies that never give vasopressors: the vasopressor actions are then rarely supported although they can be
# beneficial, the Gumbel analogue of Garnet h < 0.

# %%
# ============================================================
# II.3b Gumbel-max sepsis simulator (exact)
# ============================================================
EXECLOG.mark("II.3b Gumbel-max simulator")


class GumbelSim:
    """Layout, pinned provenance and initial-state constants of clinicalml/gumbel-max-scm/sepsisSimDiabetes."""
    FACTORS = ("hr", "sbp", "o2", "glucose", "abx", "vaso", "vent", "diabetic")
    LEVELS = (3, 3, 2, 5, 2, 2, 2, 2)
    NORMAL_LEVEL = (1, 1, 1, 2)                      # hr, sbp, o2, glucose (State.get_num_abnormal)
    STRIDES = (240, 80, 40, 8, 4, 2, 1, 720)         # State.get_state_idx('full'); diabetes is the leading digit
    N_OBS, N_FULL, N_ACTIONS = 720, 1440, 8
    P_DIABETES = 0.2                                 # MDP(p_diabetes=0.2) and DataGenerator.simulate default
    INIT_HR = (0.25, 0.5, 0.25)
    INIT_SBP = (0.25, 0.5, 0.25)
    INIT_O2 = (0.2, 0.8)
    INIT_GLUCOSE = ((0.05, 0.15, 0.6, 0.15, 0.05), (0.01, 0.05, 0.15, 0.6, 0.19))   # non-diabetic, diabetic
    VARIANTS = ("master", "sim-v2")
    REPOSITORY = "clinicalml/gumbel-max-scm"
    PATH = "sepsisSimDiabetes"
    COMMITS = {"master": "d62995deb5cb51e3ecf2859f4cbbc13f651131a2",
               "sim-v2": "f923f7747ef00103337fa1e88dfad92b98db2993"}
    SHA256 = {
        "master": {"Action.py": "56da9473140a90bde00691a10393e0dfaacd4c694573460a76861fa1dd22f841",
                   "DataGenerator.py": "d6d2a6f3cf0362f5c9512785f9d8e214b943bf3be4562a23c1ddcdb73c4a88fa",
                   "MDP.py": "8cd58b239e048472ec3a9b16ea2444288b24cdfff381bcea2e0294b97df2f35f",
                   "State.py": "23269aa1496905b3d55a9cbbca3138e33e6c1d1f0ffa7eec1d1e09870b5c10cd"},
        "sim-v2": {"Action.py": "56da9473140a90bde00691a10393e0dfaacd4c694573460a76861fa1dd22f841",
                   "DataGenerator.py": "d6d2a6f3cf0362f5c9512785f9d8e214b943bf3be4562a23c1ddcdb73c4a88fa",
                   "MDP.py": "a68aa3e3f27f3f222616a307116698ff13c19292c87df82775058578c901060e",
                   "State.py": "23269aa1496905b3d55a9cbbca3138e33e6c1d1f0ffa7eec1d1e09870b5c10cd"},
    }


_GX_HR, _GX_SBP, _GX_O2, _GX_GLU, _GX_ABX, _GX_VASO, _GX_VENT, _GX_DIAB = range(8)


def gumbel_full_index(st: Sequence[int]) -> int:
    """State.get_state_idx('full') for a state tuple (hr, sbp, o2, glucose, abx, vaso, vent, diabetic)."""
    return int(sum(int(v) * w for v, w in zip(st, GumbelSim.STRIDES)))


def gumbel_decode_full(idx: int) -> Tuple[int, ...]:
    """State.set_state_by_idx(idx, 'full') with integer arithmetic (the source's float floor division is exact here)."""
    idx = int(idx)
    assert 0 <= idx < GumbelSim.N_FULL, idx
    diab, m = divmod(idx, 720)
    hr, m = divmod(m, 240)
    sbp, m = divmod(m, 80)
    o2, m = divmod(m, 40)
    glu, m = divmod(m, 8)
    abx, m = divmod(m, 4)
    vaso, vent = divmod(m, 2)
    return (hr, sbp, o2, glu, abx, vaso, vent, diab)


def gumbel_action_bits(action: int) -> Tuple[int, int, int]:
    """Action(action_idx=a): (antibiotic, ventilation, vasopressors) = (a // 4, (a % 4) // 2, a % 2)."""
    a = int(action)
    assert 0 <= a < GumbelSim.N_ACTIONS, a
    return a // 4, (a % 4) // 2, a % 2


def gumbel_num_abnormal(st: Sequence[int]) -> int:
    return int(sum(int(st[i]) != n for i, n in enumerate(GumbelSim.NORMAL_LEVEL)))


def gumbel_on_treatment(st: Sequence[int]) -> bool:
    return bool(st[_GX_ABX] or st[_GX_VASO] or st[_GX_VENT])


def gumbel_reward(st: Sequence[int]) -> int:
    """MDP.calculateReward on the post-step state: -1 death, +1 discharge, 0 otherwise."""
    n = gumbel_num_abnormal(st)
    if n >= 3:
        return -1
    if n == 0 and not gumbel_on_treatment(st):
        return 1
    return 0


def gumbel_is_absorbing(st: Sequence[int]) -> bool:
    """State.check_absorbing_state (identical classification to calculateReward != 0)."""
    return gumbel_reward(st) != 0


# ---------- exact enumeration: each source method as a transformer of a finite distribution {state: prob} ----------
def _gx_rep(st: Tuple[int, ...], i: int, v: int) -> Tuple[int, ...]:
    s = list(st)
    s[i] = int(v)
    return tuple(s)


def _gx_case(d: Dict[tuple, float], branch: Callable[[tuple], List[Tuple[float, tuple]]]) -> Dict[tuple, float]:
    out: Dict[tuple, float] = {}
    for s, p in d.items():
        for q, t in branch(s):
            if q > 0.0:
                out[t] = out.get(t, 0.0) + p * q
    return out


def _gx_u(s: tuple, cuts: Sequence[float], effects: Sequence[Callable[[tuple], tuple]]) -> List[Tuple[float, tuple]]:
    """Outcomes of one u = np.random.uniform(0, 1) tested as `if u < cuts[0]: ... elif u < cuts[1]: ...`."""
    out, lo = [], 0.0
    for c, eff in zip(cuts, effects):
        out.append((c - lo, eff(s)))
        lo = c
    out.append((1.0 - lo, s))
    return out


def _gx_flag(d: Dict[tuple, float], i: int) -> int:
    vals = {s[i] for s in d}
    assert len(vals) == 1, "a control-flow flag differs across enumerated branches"
    return int(next(iter(vals)))


def _gx_antibiotics_on(d):
    d = _gx_case(d, lambda s: [(1.0, _gx_rep(s, _GX_ABX, 1))])
    d = _gx_case(d, lambda s: _gx_u(s, (0.5,), (lambda t: _gx_rep(t, _GX_HR, 1),)) if s[_GX_HR] == 2 else [(1.0, s)])
    d = _gx_case(d, lambda s: _gx_u(s, (0.5,), (lambda t: _gx_rep(t, _GX_SBP, 1),)) if s[_GX_SBP] == 2 else [(1.0, s)])
    return d


def _gx_antibiotics_off(d):
    if _gx_flag(d, _GX_ABX) == 1:
        d = _gx_case(d, lambda s: _gx_u(s, (0.1,), (lambda t: _gx_rep(t, _GX_HR, 2),)) if s[_GX_HR] == 1 else [(1.0, s)])
        d = _gx_case(d, lambda s: _gx_u(s, (0.1,), (lambda t: _gx_rep(t, _GX_SBP, 2),)) if s[_GX_SBP] == 1 else [(1.0, s)])
        d = _gx_case(d, lambda s: [(1.0, _gx_rep(s, _GX_ABX, 0))])
    return d


def _gx_vent_on(d):
    d = _gx_case(d, lambda s: [(1.0, _gx_rep(s, _GX_VENT, 1))])
    return _gx_case(d, lambda s: _gx_u(s, (0.7,), (lambda t: _gx_rep(t, _GX_O2, 1),)) if s[_GX_O2] == 0 else [(1.0, s)])


def _gx_vent_off(d):
    if _gx_flag(d, _GX_VENT) == 1:
        d = _gx_case(d, lambda s: _gx_u(s, (0.1,), (lambda t: _gx_rep(t, _GX_O2, 0),)) if s[_GX_O2] == 1 else [(1.0, s)])
        d = _gx_case(d, lambda s: [(1.0, _gx_rep(s, _GX_VENT, 0))])
    return d


def _gx_vaso_on(d):
    d = _gx_case(d, lambda s: [(1.0, _gx_rep(s, _GX_VASO, 1))])
    if _gx_flag(d, _GX_DIAB) == 0:
        # if u < 0.7: sbp 0 -> 1, 1 -> 2 (no change at 2; the draw is consumed in every case)
        up = {0: 1, 1: 2, 2: 2}
        return _gx_case(d, lambda s: _gx_u(s, (0.7,), (lambda t: _gx_rep(t, _GX_SBP, up[t[_GX_SBP]]),)))

    def sbp_branch(s):
        if s[_GX_SBP] == 1:
            return _gx_u(s, (0.9,), (lambda t: _gx_rep(t, _GX_SBP, 2),))
        if s[_GX_SBP] == 0:
            return _gx_u(s, (0.5, 0.9), (lambda t: _gx_rep(t, _GX_SBP, 1), lambda t: _gx_rep(t, _GX_SBP, 2)))
        return [(1.0, s)]
    d = _gx_case(d, sbp_branch)
    return _gx_case(d, lambda s: _gx_u(s, (0.5,), (lambda t: _gx_rep(t, _GX_GLU, min(4, t[_GX_GLU] + 1)),)))


def _gx_vaso_off(d):
    if _gx_flag(d, _GX_VASO) == 1:
        p = 0.1 if _gx_flag(d, _GX_DIAB) == 0 else 0.05
        d = _gx_case(d, lambda s: _gx_u(s, (p,), (lambda t: _gx_rep(t, _GX_SBP, max(0, t[_GX_SBP] - 1)),)))
        d = _gx_case(d, lambda s: [(1.0, _gx_rep(s, _GX_VASO, 0))])
    return d


def _gx_fluctuate(d, hr_f: bool, sbp_f: bool, o2_f: bool, glu_f: bool, variant: str):
    def pm(i: int, top: int, cuts: Tuple[float, float]):
        return lambda s: _gx_u(s, cuts, (lambda t: _gx_rep(t, i, max(0, t[i] - 1)),
                                         lambda t: _gx_rep(t, i, min(top, t[i] + 1))))
    if hr_f:
        d = _gx_case(d, pm(_GX_HR, 2, (0.1, 0.2)))
    if sbp_f:
        d = _gx_case(d, pm(_GX_SBP, 2, (0.1, 0.2)))
    if o2_f:
        d = _gx_case(d, pm(_GX_O2, 1, (0.1, 0.2)))
    if glu_f:
        cap = 1 if variant == "master" else 4        # MDP.py line 253: master min(1, g + 1) (bug), sim-v2 min(4, g + 1)
        nondiab, diab = pm(_GX_GLU, cap, (0.1, 0.2)), pm(_GX_GLU, 4, (0.3, 0.6))
        d = _gx_case(d, lambda s: nondiab(s) if s[_GX_DIAB] == 0 else diab(s))
    return d


def gumbel_transition_dist(st: Sequence[int], action: int, variant: str) -> Dict[Tuple[int, ...], float]:
    """Exact next-full-state distribution of MDP.transition(Action(action_idx=action)) from full state `st`."""
    assert variant in GumbelSim.VARIANTS, variant
    a_abx, a_vent, a_vaso = gumbel_action_bits(action)
    d = {tuple(int(x) for x in st): 1.0}
    if a_abx == 1:
        d = _gx_antibiotics_on(d)
        hr_f = sbp_f = False
    elif _gx_flag(d, _GX_ABX) == 1:
        d = _gx_antibiotics_off(d)
        hr_f = sbp_f = False
    else:
        hr_f = sbp_f = True
    if a_vent == 1:
        d = _gx_vent_on(d)
        o2_f = False
    elif _gx_flag(d, _GX_VENT) == 1:
        d = _gx_vent_off(d)
        o2_f = False
    else:
        o2_f = True
    glu_f = True
    if a_vaso == 1:
        d = _gx_vaso_on(d)
        sbp_f = False
        glu_f = False
    elif _gx_flag(d, _GX_VASO) == 1:
        d = _gx_vaso_off(d)
        sbp_f = False
    d = _gx_fluctuate(d, hr_f, sbp_f, o2_f, glu_f, variant)
    total = sum(d.values())
    assert abs(total - 1.0) < 1e-12, (st, action, total)
    return d


def gumbel_initial_distribution(p_diabetes: float = GumbelSim.P_DIABETES) -> Tuple[Dict[Tuple[int, ...], float], Dict[str, Any]]:
    """Exact law of MDP.get_new_state(): the joint product proposal conditioned on a non-absorbing state."""
    joint: Dict[Tuple[int, ...], float] = {}
    acc = {0: 0.0, 1: 0.0}
    for diab, pd_ in ((0, 1.0 - p_diabetes), (1, p_diabetes)):
        for hr, sbp, o2, glu in itertools.product(range(3), range(3), range(2), range(5)):
            st = (hr, sbp, o2, glu, 0, 0, 0, diab)
            q = GumbelSim.INIT_HR[hr] * GumbelSim.INIT_SBP[sbp] * GumbelSim.INIT_O2[o2] * GumbelSim.INIT_GLUCOSE[diab][glu]
            if not gumbel_is_absorbing(st):
                acc[diab] += q
                joint[st] = pd_ * q
    Z = sum(joint.values())
    d0 = {st: p / Z for st, p in joint.items()}
    info = {"acceptance_nondiabetic": acc[0], "acceptance_diabetic": acc[1], "acceptance_overall": Z,
            "d0_diabetic_share": float(sum(p for st, p in d0.items() if st[_GX_DIAB] == 1)),
            "support_nondiabetic": int(sum(st[_GX_DIAB] == 0 for st in d0)),
            "support_diabetic": int(sum(st[_GX_DIAB] == 1 for st in d0))}
    return d0, info


_GUMBEL_SPACE: Dict[str, Any] = {}


def gumbel_state_space() -> Dict[str, Any]:
    """Full states, their absorbing class, and the live full-state indices in full-index order (cached)."""
    if not _GUMBEL_SPACE:
        full = [gumbel_decode_full(i) for i in range(GumbelSim.N_FULL)]
        assert all(gumbel_full_index(st) == i for i, st in enumerate(full)), "full-index round trip failed"
        assert len(set(full)) == GumbelSim.N_FULL
        kind = np.array(["death" if gumbel_reward(st) == -1 else "discharge" if gumbel_reward(st) == 1 else "live"
                         for st in full])
        live_full = np.flatnonzero(kind == "live")
        _GUMBEL_SPACE.update(full=full, kind=kind, live_full=live_full,
                             pos={int(f): k for k, f in enumerate(live_full)})
    return _GUMBEL_SPACE


def _gumbel_expected_counts() -> Dict[str, int]:
    """Per-group death / discharge / live / initial counts re-derived combinatorially from the source rules."""
    abn = [lv - 1 for lv in GumbelSim.LEVELS[:4]]            # abnormal levels per vital: 2, 2, 1, 4
    poly = np.array([1])
    for k in abn:
        poly = np.convolve(poly, [1, k])                    # coefficient j = vital patterns with j abnormal vitals
    flags = 2 ** 3
    death = int(poly[3:].sum()) * flags
    discharge = int(poly[0])                                # zero abnormal vitals and every flag off
    return {"vital_patterns_by_abnormal": poly.astype(int).tolist(), "death_per_group": death,
            "discharge_per_group": discharge, "live_per_group": GumbelSim.N_OBS - death - discharge,
            "initial_support_per_group": int(poly[1] + poly[2])}


def gumbel_true_env_report(variant: str, timeout_hazard: float) -> Tuple[FiniteEnv, Dict[str, Any]]:
    """Exact absorbing Gumbel-max sepsis MDP (live full states, then death -1, discharge +1, timeout 0) and the report
    of its structural checks (every check below raises on failure)."""
    assert variant in GumbelSim.VARIANTS, variant
    h = float(timeout_hazard)
    assert 0.0 < h < 1.0, "the timeout hazard must lie in (0, 1) (it defines the absorbing horizon)"
    space = gumbel_state_space()
    full, live_full, pos = space["full"], space["live_full"], space["pos"]
    n_live, A = len(live_full), GumbelSim.N_ACTIONS
    death, discharge, timeout = n_live, n_live + 1, n_live + 2
    S = n_live + 3
    P_sim = np.zeros((n_live, A, S))
    for k, f in enumerate(live_full):
        for a in range(A):
            for nxt, p in gumbel_transition_dist(full[f], a, variant).items():
                r = gumbel_reward(nxt)
                j = death if r == -1 else discharge if r == 1 else pos[gumbel_full_index(nxt)]
                P_sim[k, a, j] += p
    P = np.zeros((S, A, S))
    P[:n_live] = (1.0 - h) * P_sim
    P[:n_live, :, timeout] += h
    for t in (death, discharge, timeout):
        P[t, :, t] = 1.0
    entry = np.zeros(S)
    entry[death], entry[discharge] = -1.0, 1.0
    d0_map, init_info = gumbel_initial_distribution()
    d0 = np.zeros(S)
    for st, p in d0_map.items():
        d0[pos[gumbel_full_index(st)]] = p
    feats = np.zeros((S, len(GumbelSim.FACTORS)), dtype=np.float32)
    scale = np.array([lv - 1 for lv in GumbelSim.LEVELS], dtype=np.float32)
    feats[:n_live] = np.array([full[f] for f in live_full], dtype=np.float32) / scale

    # ---------- structural asserts (re-derived from the source logic) ----------
    exp = _gumbel_expected_counts()
    kind = space["kind"]
    diab = np.array([st[_GX_DIAB] for st in full])
    groups = {}
    for g in (0, 1):
        sel = diab == g
        groups[g] = {"states": int(sel.sum()), "death": int((kind[sel] == "death").sum()),
                     "discharge": int((kind[sel] == "discharge").sum()), "live": int((kind[sel] == "live").sum()),
                     "initial_support": int(sum(1 for st in d0_map if st[_GX_DIAB] == g))}
        assert groups[g]["states"] == GumbelSim.N_OBS, groups
        assert groups[g]["death"] == exp["death_per_group"] == 416, (groups, exp)
        assert groups[g]["discharge"] == exp["discharge_per_group"] == 1, (groups, exp)
        assert groups[g]["live"] == exp["live_per_group"] == 303, (groups, exp)
        assert groups[g]["initial_support"] == exp["initial_support_per_group"] == 37, (groups, exp)
    assert len(full) == GumbelSim.N_FULL and n_live == 606
    disc_full = [f for f in range(GumbelSim.N_FULL) if kind[f] == "discharge"]
    assert sorted(f % GumbelSim.N_OBS for f in disc_full) == [376, 376], disc_full
    row_dev = float(np.abs(P.sum(axis=2) - 1.0).max())
    assert row_dev < 1e-12 and (P >= 0).all(), row_dev
    assert np.all(P[:n_live, :, timeout] == h) and np.all(P_sim[:, :, timeout] == 0.0)
    assert np.all(P_sim[:, 1:, discharge] == 0.0) and P_sim[:, 0, discharge].max() > 0, "discharge only via action 0"
    live_f = np.array([full[f] for f in live_full])
    for a in range(A):
        bits = gumbel_action_bits(a)
        flags = (bits[0], bits[2], bits[1])                # state order (abx, vaso, vent); action order (abx, vent, vaso)
        succ = P_sim[:, a, :n_live] > 0
        rows, cols = np.nonzero(succ)
        assert np.all(live_f[cols][:, [_GX_ABX, _GX_VASO, _GX_VENT]] == np.array(flags)), "post-step flags != action bits"
        assert np.all(live_f[cols][:, _GX_DIAB] == live_f[rows][:, _GX_DIAB]), "diabetes changed during a step"
    assert abs(d0.sum() - 1.0) < 1e-12 and int((d0 > 0).sum()) == 74
    no_timeout = np.zeros_like(P)
    no_timeout[:n_live] = P_sim
    for t in (death, discharge, timeout):
        no_timeout[t, :, t] = 1.0
    live_mask = np.r_[np.ones(n_live, bool), np.zeros(3, bool)]
    structure = {"full_states": len(full), "live_states": n_live, "groups": {str(g): v for g, v in groups.items()},
                 "expected_counts": exp, "max_row_sum_deviation": row_dev, "initial": init_info,
                 "max_branches_per_pair": int((P_sim > 0).sum(axis=2).max()),
                 "trap_states_without_timeout": x_trap_states(no_timeout, live_mask, np.ones((S, A), bool))}
    del no_timeout
    env = FiniteEnv(name=f"gumbel-true|{variant}|h{h}", kind="gumbel", P=P, entry_reward=entry, d0=d0,
                    terminals=(death, discharge, timeout), sink=death, goal=discharge, mask=np.ones((S, A), bool),
                    features=feats,
                    meta={"variant": variant, "timeout_hazard": h, "source": GumbelSim.REPOSITORY,
                          "commit": GumbelSim.COMMITS[variant], "n_live": n_live, "death_state": death,
                          "discharge_state": discharge, "timeout_state": timeout, "dangling_target": timeout,
                          "dangling_target_label": "timeout", "censoring_hazard": h})
    assert x_trap_states(env.P, env.live, env.mask) == 0, "every policy must be proper with the timeout hazard"
    return env, structure


def gumbel_true_env(variant: str = XCFG.gumbel_variant, timeout_hazard: float = XCFG.gumbel_timeout_hazard) -> FiniteEnv:
    """Exact absorbing Gumbel-max sepsis MDP: live full states, then death (-1), discharge (+1), timeout (0)."""
    return gumbel_true_env_report(variant, timeout_hazard)[0]


def gumbel_decode_table(env: FiniteEnv) -> pd.DataFrame:
    """Env index -> full index and factor levels (-1 on the three absorbing states)."""
    space = gumbel_state_space()
    n_live = int(env.meta["n_live"])
    rows = []
    for k, f in enumerate(space["live_full"]):
        st = space["full"][f]
        rows.append({"state": k, "kind": "live", "full_index": int(f), **dict(zip(GumbelSim.FACTORS, st)),
                     "n_abnormal": gumbel_num_abnormal(st), "d0": float(env.d0[k])})
    for lab, t in (("death", env.sink), ("discharge", env.goal), ("timeout", int(env.meta["timeout_state"]))):
        rows.append({"state": t, "kind": lab, "full_index": -1, **{f: -1 for f in GumbelSim.FACTORS},
                     "n_abnormal": -1, "d0": 0.0})
    df = pd.DataFrame(rows).sort_values("state").reset_index(drop=True)
    assert len(df) == env.S == n_live + 3 and np.array_equal(df["state"].to_numpy(), np.arange(env.S))
    return df


GUMBEL_PHYSICIAN_KINDS = ("optimal", "withhold-vaso")


def gumbel_physician(env: FiniteEnv, eps: float, kind: str = "optimal") -> np.ndarray:
    """eps-softened physician of `env` under its own dynamics: 1 - eps on its action, eps/(A-1) on every other action.

    kind 'optimal': the exact optimum of `env`. kind 'withhold-vaso': the exact optimum among policies that never give
    vasopressors on live states (admissible-constrained optimum under the mask 1[vasopressor bit of a = 0]). Its
    vasopressor actions get eps/(A-1) each, so they are rarely supported in the hospital data although they can be
    beneficial in truth (the Gumbel analogue of Garnet h < 0)."""
    assert 0.0 <= eps < 1.0, eps
    if kind == "optimal":
        pi_opt, _, _ = x_optimal(env, "given")
    elif kind == "withhold-vaso":
        no_vaso = np.ones((env.S, env.A), dtype=bool)
        no_vaso[env.live] = np.array([gumbel_action_bits(a)[2] == 0 for a in range(env.A)])[None, :]
        pi_opt, _, _ = x_optimal(env.variant(f"{env.name}|no-vasopressors", mask=no_vaso), "given", constrained=True)
    else:
        raise ValueError(f"unknown Gumbel physician kind {kind!r} (expected one of {GUMBEL_PHYSICIAN_KINDS})")
    act = pi_opt.argmax(axis=1)
    if kind == "withhold-vaso":
        assert all(gumbel_action_bits(a)[2] == 0 for a in act[env.live]), "withhold-vaso physician gives vasopressors"
    pi = np.full((env.S, env.A), eps / (env.A - 1))
    pi[np.arange(env.S), act] = 1.0 - eps
    return validate_policy(pi, f"Gumbel physician ({kind}) eps={eps}", shape=(env.S, env.A))


# ---------- literal port of the stochastic step (used ONLY to test the enumeration) ----------
def gumbel_step_sample(state_tuple: Sequence[int], action: int, rng: np.random.Generator, variant: str):
    """One call of MDP.transition(Action(action_idx=action)) + calculateReward, np.random.uniform -> rng.uniform.

    Returns the next full-state tuple, or the label 'death' / 'discharge' when the reward is -1 / +1."""
    assert variant in GumbelSim.VARIANTS
    hr, sbp, o2, glu, abx, vaso, vent, diab = (int(x) for x in state_tuple)
    s = {"hr_state": hr, "sysbp_state": sbp, "percoxyg_state": o2, "glucose_state": glu, "antibiotic_state": abx,
         "vaso_state": vaso, "vent_state": vent, "diabetic_idx": diab}
    a = int(action)
    antibiotic, ventilation, vasopressors = a // 4, (a % 4) // 2, a % 2
    if antibiotic == 1:                                           # transition_antibiotics_on
        s["antibiotic_state"] = 1
        if s["hr_state"] == 2 and rng.uniform(0, 1) < 0.5:
            s["hr_state"] = 1
        if s["sysbp_state"] == 2 and rng.uniform(0, 1) < 0.5:
            s["sysbp_state"] = 1
        hr_fluctuate = False
        sysbp_fluctuate = False
    elif s["antibiotic_state"] == 1:                              # transition_antibiotics_off
        if s["antibiotic_state"] == 1:
            if s["hr_state"] == 1 and rng.uniform(0, 1) < 0.1:
                s["hr_state"] = 2
            if s["sysbp_state"] == 1 and rng.uniform(0, 1) < 0.1:
                s["sysbp_state"] = 2
            s["antibiotic_state"] = 0
        hr_fluctuate = False
        sysbp_fluctuate = False
    else:
        hr_fluctuate = True
        sysbp_fluctuate = True
    if ventilation == 1:                                          # transition_vent_on
        s["vent_state"] = 1
        if s["percoxyg_state"] == 0 and rng.uniform(0, 1) < 0.7:
            s["percoxyg_state"] = 1
        percoxyg_fluctuate = False
    elif s["vent_state"] == 1:                                    # transition_vent_off
        if s["vent_state"] == 1:
            if s["percoxyg_state"] == 1 and rng.uniform(0, 1) < 0.1:
                s["percoxyg_state"] = 0
            s["vent_state"] = 0
        percoxyg_fluctuate = False
    else:
        percoxyg_fluctuate = True
    glucose_fluctuate = True
    if vasopressors == 1:                                         # transition_vaso_on
        s["vaso_state"] = 1
        if s["diabetic_idx"] == 0:
            if rng.uniform(0, 1) < 0.7:
                if s["sysbp_state"] == 0:
                    s["sysbp_state"] = 1
                elif s["sysbp_state"] == 1:
                    s["sysbp_state"] = 2
        else:
            if s["sysbp_state"] == 1:
                if rng.uniform(0, 1) < 0.9:
                    s["sysbp_state"] = 2
            elif s["sysbp_state"] == 0:
                up_prob = rng.uniform(0, 1)
                if up_prob < 0.5:
                    s["sysbp_state"] = 1
                elif up_prob < 0.9:
                    s["sysbp_state"] = 2
            if rng.uniform(0, 1) < 0.5:
                s["glucose_state"] = min(4, s["glucose_state"] + 1)
        sysbp_fluctuate = False
        glucose_fluctuate = False
    elif s["vaso_state"] == 1:                                    # transition_vaso_off
        if s["vaso_state"] == 1:
            if s["diabetic_idx"] == 0:
                if rng.uniform(0, 1) < 0.1:
                    s["sysbp_state"] = max(0, s["sysbp_state"] - 1)
            else:
                if rng.uniform(0, 1) < 0.05:
                    s["sysbp_state"] = max(0, s["sysbp_state"] - 1)
            s["vaso_state"] = 0
        sysbp_fluctuate = False
    if hr_fluctuate:                                              # transition_fluctuate
        hr_prob = rng.uniform(0, 1)
        if hr_prob < 0.1:
            s["hr_state"] = max(0, s["hr_state"] - 1)
        elif hr_prob < 0.2:
            s["hr_state"] = min(2, s["hr_state"] + 1)
    if sysbp_fluctuate:
        sysbp_prob = rng.uniform(0, 1)
        if sysbp_prob < 0.1:
            s["sysbp_state"] = max(0, s["sysbp_state"] - 1)
        elif sysbp_prob < 0.2:
            s["sysbp_state"] = min(2, s["sysbp_state"] + 1)
    if percoxyg_fluctuate:
        percoxyg_prob = rng.uniform(0, 1)
        if percoxyg_prob < 0.1:
            s["percoxyg_state"] = max(0, s["percoxyg_state"] - 1)
        elif percoxyg_prob < 0.2:
            s["percoxyg_state"] = min(1, s["percoxyg_state"] + 1)
    if glucose_fluctuate:
        glucose_prob = rng.uniform(0, 1)
        if s["diabetic_idx"] == 0:
            if glucose_prob < 0.1:
                s["glucose_state"] = max(0, s["glucose_state"] - 1)
            elif glucose_prob < 0.2:
                s["glucose_state"] = (min(1, s["glucose_state"] + 1) if variant == "master"
                                      else min(4, s["glucose_state"] + 1))
        else:
            if glucose_prob < 0.3:
                s["glucose_state"] = max(0, s["glucose_state"] - 1)
            elif glucose_prob < 0.6:
                s["glucose_state"] = min(4, s["glucose_state"] + 1)
    num_abnormal = ((s["hr_state"] != 1) + (s["sysbp_state"] != 1) + (s["percoxyg_state"] != 1)
                    + (s["glucose_state"] != 2))                   # calculateReward
    on_treatment = not (s["antibiotic_state"] == 0 and s["vaso_state"] == 0 and s["vent_state"] == 0)
    if num_abnormal >= 3:
        return "death"
    if num_abnormal == 0 and not on_treatment:
        return "discharge"
    return (s["hr_state"], s["sysbp_state"], s["percoxyg_state"], s["glucose_state"], s["antibiotic_state"],
            s["vaso_state"], s["vent_state"], s["diabetic_idx"])


def gumbel_initial_sample(rng: np.random.Generator, p_diabetes: float = GumbelSim.P_DIABETES) -> Tuple[int, ...]:
    """Literal port of MDP.get_new_state() (option 'random'): diabetes is redrawn on every rejected attempt."""
    def generate_random_state(diabetic_idx=None):
        if diabetic_idx is None:
            diabetic_idx = rng.binomial(1, p_diabetes)
        hr_state = rng.choice(np.arange(3), p=np.array([.25, .5, .25]))
        sysbp_state = rng.choice(np.arange(3), p=np.array([.25, .5, .25]))
        percoxyg_state = rng.choice(np.arange(2), p=np.array([.2, .8]))
        if diabetic_idx == 0:
            glucose_state = rng.choice(np.arange(5), p=np.array([.05, .15, .6, .15, .05]))
        else:
            glucose_state = rng.choice(np.arange(5), p=np.array([.01, .05, .15, .6, .19]))
        return (int(hr_state), int(sysbp_state), int(percoxyg_state), int(glucose_state), 0, 0, 0, int(diabetic_idx))

    def check_absorbing_state(st):
        num_abnormal = (st[0] != 1) + (st[1] != 1) + (st[2] != 1) + (st[3] != 2)
        return num_abnormal >= 3 or (num_abnormal == 0 and not (st[4] or st[5] or st[6]))

    init_state = generate_random_state(None)
    while check_absorbing_state(init_state):
        init_state = generate_random_state(None)
    return init_state


def _x_binned_chi2(counts: np.ndarray, p: np.ndarray, n: int, min_expected: float = 5.0) -> Tuple[float, int]:
    """Pearson chi-square over outcomes with p > 0, merging the rarest outcomes until every bin expects >= 5."""
    keep = p > 0
    pk, ck = p[keep], counts[keep]
    bins_p, bins_c, acc_p, acc_c = [], [], 0.0, 0.0
    for j in np.argsort(pk, kind="stable"):
        acc_p += pk[j]
        acc_c += ck[j]
        if acc_p * n >= min_expected:
            bins_p.append(acc_p)
            bins_c.append(acc_c)
            acc_p = acc_c = 0.0
    if acc_p > 0 or acc_c > 0:
        if bins_p:
            bins_p[-1] += acc_p
            bins_c[-1] += acc_c
        else:
            bins_p.append(acc_p)
            bins_c.append(acc_c)
    E = n * np.asarray(bins_p)
    return float((((np.asarray(bins_c) - E) ** 2) / E).sum()), len(bins_p) - 1


def gumbel_exactness_check(env: FiniteEnv, n_pairs: int, n_draws: int, rng: np.random.Generator,
                           n_init: Optional[int] = None, p_min: float = 1e-3) -> Dict[str, Any]:
    """Exact rows P_sim = (P - h e_timeout) / (1 - h) versus `gumbel_step_sample` frequencies on random live pairs,
    and the exact d0 versus `gumbel_initial_sample` frequencies."""
    variant, h = env.meta["variant"], float(env.meta["timeout_hazard"])
    space = gumbel_state_space()
    n_live, A, timeout = int(env.meta["n_live"]), env.A, int(env.meta["timeout_state"])
    e_to = np.zeros(env.S)
    e_to[timeout] = 1.0
    t0 = time.perf_counter()
    pairs = rng.choice(n_live * A, size=min(int(n_pairs), n_live * A), replace=False)
    n = int(n_draws)
    chi2_tot, df_tot, impossible, max_z, max_z5, max_abs, n_cells = 0.0, 0, 0, 0.0, 0.0, 0.0, 0
    for flat in pairs:
        s, a = divmod(int(flat), A)
        st = space["full"][space["live_full"][s]]
        counts = np.zeros(env.S)
        for _ in range(n):
            nxt = gumbel_step_sample(st, a, rng, variant)
            j = (env.sink if nxt == "death" else env.goal) if isinstance(nxt, str) else space["pos"][gumbel_full_index(nxt)]
            counts[j] += 1
        p = (env.P[s, a] - h * e_to) / (1.0 - h)
        assert p[timeout] == 0.0 and abs(p.sum() - 1.0) < 1e-12
        freq = counts / n
        impossible += int(counts[p <= 0].sum())
        pos_ = p > 0
        pf = np.clip(p[pos_], 1.0 / n, 1.0 - 1.0 / n)            # SE floor: p in [1/n, 1 - 1/n] (deterministic rows)
        z = np.abs(freq[pos_] - p[pos_]) / np.sqrt(pf * (1.0 - pf) / n)
        assert np.isfinite(z).all()
        max_z = max(max_z, float(z.max()))
        big = n * p[pos_] >= 5.0
        if big.any():
            max_z5 = max(max_z5, float(z[big].max()))
        max_abs = max(max_abs, float(np.abs(freq - p).max()))
        n_cells += int(pos_.sum())
        c2, df = _x_binned_chi2(counts, p, n)
        chi2_tot += c2
        df_tot += df
    p_rows = float(stats.chi2.sf(chi2_tot, df_tot)) if df_tot > 0 else float("nan")
    t_rows = time.perf_counter() - t0
    n0 = int(n_init) if n_init is not None else 10 * n
    c0 = np.zeros(env.S)
    for _ in range(n0):
        c0[space["pos"][gumbel_full_index(gumbel_initial_sample(rng))]] += 1
    d0_impossible = int(c0[env.d0 <= 0].sum())
    c2_0, df_0 = _x_binned_chi2(c0, env.d0, n0)
    p_d0 = float(stats.chi2.sf(c2_0, df_0))
    out = {"variant": variant, "timeout_hazard": h, "pairs": int(len(pairs)), "draws_per_pair": n,
           "outcome_cells_with_positive_p": n_cells, "impossible_outcomes_observed": impossible,
           "max_abs_freq_minus_p": max_abs, "max_z": max_z, "max_z_cells_expected_ge_5": max_z5,
           "bonferroni_normal_z_at_1e-3": float(stats.norm.isf(1e-3 / (2 * max(n_cells, 1)))),
           "pooled_chi2": chi2_tot, "pooled_df": int(df_tot), "pooled_p_value": p_rows,
           "d0_samples": n0, "d0_impossible_observed": d0_impossible, "d0_chi2": c2_0, "d0_df": int(df_0),
           "d0_p_value": p_d0, "max_abs_d0_freq_minus_p": float(np.abs(c0 / n0 - env.d0).max()),
           "p_value_threshold": p_min, "seconds_rows": t_rows, "seconds_total": time.perf_counter() - t0}
    out["passes"] = bool(impossible == 0 and d0_impossible == 0 and np.isfinite(p_rows) and p_rows >= p_min
                         and p_d0 >= p_min)
    return out


def gumbel_source_provenance() -> Dict[str, Any]:
    """Re-fetch the pinned sources as text (never executed) and compare their SHA-256 with the pinned digests."""
    if not VERIFY_PROVENANCE_ONLINE:
        return {"status": "skipped (VERIFY_PROVENANCE_ONLINE=False)", "commits": GumbelSim.COMMITS}
    res = {}
    for variant, commit in GumbelSim.COMMITS.items():
        for fname, ref in GumbelSim.SHA256[variant].items():
            url = f"https://raw.githubusercontent.com/{GumbelSim.REPOSITORY}/{commit}/{GumbelSim.PATH}/{fname}"
            try:   # an HTTP error (e.g. 404: wrong pinned commit or path) is recorded, not mistaken for being offline
                with urllib.request.urlopen(url, timeout=15.0) as resp:
                    txt = resp.read().decode("utf-8")
                res[f"{variant}/{fname}"] = bool(sha256_bytes(txt.encode("utf-8")) == ref)
            except urllib.error.HTTPError as exc:
                print("  fetch failed:", url, "-> HTTP", exc.code)
                res[f"{variant}/{fname}"] = f"HTTP {exc.code}"
            except Exception as exc:  # offline or blocked
                print("  fetch failed:", url, "->", type(exc).__name__)
                res[f"{variant}/{fname}"] = None
    vals = list(res.values())
    status = ("verified" if all(v is True for v in vals) else "MISMATCH" if any(v is False for v in vals)
              else "MISSING (HTTP error)" if any(isinstance(v, str) for v in vals)
              else "unavailable (offline)" if all(v is None for v in vals)
              else "incomplete (some fetches failed offline; every fetched file matched)")
    return {"status": status, "sha256_matches": res, "commits": GumbelSim.COMMITS}


def gumbel_variant_comparison(envs: Dict[str, FiniteEnv]) -> Dict[str, Any]:
    """master vs sim-v2: the full-state step laws differ exactly on the non-diabetic pairs whose glucose fluctuates
    (vasopressors not given) from glucose >= 1, the only case where min(1, g+1) != min(4, g+1). After the absorbing
    collapse a predicted pair keeps identical rows only if every differing successor is absorbing in both variants."""
    em, ev = envs["master"], envs["sim-v2"]
    space = gumbel_state_space()
    n_live = int(em.meta["n_live"])
    diff = np.abs(em.P[:n_live] - ev.P[:n_live]).max(axis=2) > 0
    pred = np.zeros_like(diff)
    for k, f in enumerate(space["live_full"]):
        st = space["full"][f]
        for a in range(em.A):
            pred[k, a] = st[_GX_DIAB] == 0 and gumbel_action_bits(a)[2] == 0 and st[_GX_GLU] >= 1
    assert not (diff & ~pred).any(), "master / sim-v2 rows differ outside the glucose-fluctuation pairs"
    for k, a in zip(*np.nonzero(pred)):
        st = space["full"][space["live_full"][k]]
        dm, dv = gumbel_transition_dist(st, a, "master"), gumbel_transition_dist(st, a, "sim-v2")
        changed = [t for t in set(dm) | set(dv) if abs(dm.get(t, 0.0) - dv.get(t, 0.0)) > 0]
        assert changed, ("predicted pair with identical full-state law", st, a)
        if not diff[k, a]:
            assert all(gumbel_is_absorbing(t) for t in changed), ("collapsed rows equal but a live successor changed", st, a)
    out = {"rows_differing": int(diff.sum()), "pairs_with_different_full_state_law": int(pred.sum()),
           "pairs_where_the_difference_is_absorbed": int((pred & ~diff).sum()),
           "max_abs_row_difference": float(np.abs(em.P - ev.P).max())}
    for name, env in envs.items():
        _, _, info = x_optimal(env, "given")
        out[f"optimum_value[{name}]"] = float(info["dp_value"])
        out[f"uniform_random_value[{name}]"] = float(x_evaluate(env, np.full((env.S, env.A), 1.0 / env.A), "given")["return"])
    return out


X_GUMBEL: Optional[FiniteEnv] = None
X_GUMBEL_ENVS: Dict[str, FiniteEnv] = {}
GUMBEL_DECODE = pd.DataFrame()
GUMBEL_EXACTNESS: Dict[str, Any] = {"status": "not run"}
GUMBEL_EXACTNESS_OK: Optional[bool] = None
GUMBEL_REPORT: Dict[str, Any] = {"status": "not run"}
if EXPERIMENTS["xenv_gumbel"]:
    _t0 = time.perf_counter()
    _structure = {}
    for _v in GumbelSim.VARIANTS:
        X_GUMBEL_ENVS[_v], _structure[_v] = gumbel_true_env_report(_v, XCFG.gumbel_timeout_hazard)
    X_GUMBEL = X_GUMBEL_ENVS[XCFG.gumbel_variant]
    print(f"Gumbel simulator enumerated exactly ({len(X_GUMBEL_ENVS)} variants, {time.perf_counter() - _t0:.1f} s):",
          _structure[XCFG.gumbel_variant]["groups"])
    GUMBEL_DECODE = gumbel_decode_table(X_GUMBEL)
    GUMBEL_DECODE.to_csv(XENV_TABLE_DIR / "gumbel_state_decode.csv", index=False)
    _dec = GUMBEL_DECODE[["state", "full_index", *GumbelSim.FACTORS]].to_numpy(dtype=np.int64)
    for _v, _env in X_GUMBEL_ENVS.items():
        np.savez_compressed(DATA_DIR / f"xenv_gumbel_true_{_v}.npz", P=_env.P, entry_reward=_env.entry_reward,
                            d0=_env.d0, decode=_dec, mask=_env.mask,
                            meta_json=np.array(json.dumps(_env.meta, sort_keys=True, default=_json_default)))
    _checks = {}
    for _v, _env in X_GUMBEL_ENVS.items():
        _main = _v == XCFG.gumbel_variant
        _checks[_v] = gumbel_exactness_check(
            _env, XCFG.gumbel_mc_pairs if _main else max(1, XCFG.gumbel_mc_pairs // 3), XCFG.gumbel_mc_draws,
            np_rng("xenv", "gumbel-exactness", _v))
        print(f"  exactness [{_v}]:", {k: _checks[_v][k] for k in ("pairs", "pooled_p_value", "max_z_cells_expected_ge_5",
                                                                    "impossible_outcomes_observed", "d0_p_value", "passes")})
    GUMBEL_EXACTNESS = {"configured_variant": XCFG.gumbel_variant, "checks": _checks,
                        "passes": bool(all(c["passes"] for c in _checks.values()))}
    GUMBEL_EXACTNESS_OK = GUMBEL_EXACTNESS["passes"]
    save_json(GUMBEL_EXACTNESS, XENV_TABLE_DIR / "gumbel_exactness_check.json")
    GUMBEL_REPORT = {"structure": _structure,
                     "variant_comparison": gumbel_variant_comparison(X_GUMBEL_ENVS),
                     "provenance": gumbel_source_provenance(), "configured_variant": XCFG.gumbel_variant}
    save_json(GUMBEL_REPORT, XENV_TABLE_DIR / "gumbel_simulator_report.json")
    print("  variant comparison:", GUMBEL_REPORT["variant_comparison"], "| provenance:", GUMBEL_REPORT["provenance"]["status"])
    if GUMBEL_REPORT["provenance"]["status"].startswith(("MISMATCH", "MISSING")):
        print(f"WARNING: Gumbel source provenance {GUMBEL_REPORT['provenance']['status']}: the pinned sources could not "
              "be confirmed online (this port follows the pinned commits); closure row MT-1 fails until resolved.")
    assert GUMBEL_EXACTNESS_OK, "Gumbel enumeration disagrees with the literal step sampler (xenv/gumbel_exactness_check.json)."
    del _dec, _checks, _structure

# %% [markdown]
# ### II.3c Garnet absorbing CMDPs with a controlled harm of unsupported actions
#
# For each draw, a base absorbing MDP is generated with S live states: every (s, a) moves to `branching` distinct
# random live successors with Dirichlet(1) weights, after absorbing with hazard h(s, a) ∼ U(`garnet_hazard`); an
# absorption reaches the goal (entry reward +1) with probability q(s, a) ∼ U(0, 1), and otherwise death (0). d0 is
# uniform over 10 random live states. The behaviour policy prefers the top-k_s actions of the base Q* (k_s uniform
# in {1, 2, 3}): π_β = (1 − ε)/k_s on K(s) and ε/(A − k_s) elsewhere. The **harm** h ∈ [−1, 1] sets the true rows of
# the non-preferred actions: P*(s, a) = (1 − |h|)·mean_{a'∈K(s)} P_base(s, a') + |h|·e_death (h > 0) or e_goal (h < 0).
# With h = 0 the mean rule is exactly right wherever K(s) ⊆ M(s). Only the base MDP, K(s) and k_s are shared: they
# depend only on the draw (stream `("xenv", "garnet", "base", draw)`), so the harm and ε cells of one draw share their
# base MDP (common random numbers), and Section II.7 treats them as one draw block. The hospital and user data are
# cell-specific (streams keyed by the full cell), and the harm changes the benchmark mask itself: absorbing outcomes
# shorten the hospital episodes, so the mask size differs across h (reported by (h, ε) next to H3). The hazard makes
# every policy proper; this is asserted.

# %%
# ============================================================
# II.3c Garnet family
# ============================================================
EXECLOG.mark("II.3c Garnet family")


def garnet_true_env(draw_key: tuple, n_states: int, n_actions: int, branching: int,
                    hazard_range: Tuple[float, float], harm: float, eps: float) -> Tuple[FiniteEnv, np.ndarray]:
    """True Garnet CMDP of one draw with harm `harm`, and its behaviour policy with softening `eps`."""
    S, A, B = int(n_states), int(n_actions), int(branching)
    lo, hi = (float(x) for x in hazard_range)
    harm, eps = float(harm), float(eps)
    assert S >= 10 and A >= 4 and 1 <= B <= S, (S, A, B)
    assert 0.0 < lo <= hi < 1.0 and -1.0 <= harm <= 1.0 and 0.0 < eps < 1.0, (lo, hi, harm, eps)
    rng = np_rng("xenv", "garnet", *draw_key)
    goal, death = S, S + 1
    succ = np.argsort(rng.random((S * A, S)), axis=1)[:, :B]          # B distinct uniformly random live successors
    w = rng.dirichlet(np.ones(B), size=S * A)
    haz = rng.uniform(lo, hi, size=S * A)
    q_goal = rng.uniform(0.0, 1.0, size=S * A)
    starts = rng.choice(S, size=10, replace=False)
    k = rng.integers(1, 4, size=S)                                    # |K(s)| uniform in {1, 2, 3}
    rows = np.zeros((S * A, S + 2))
    rows[np.arange(S * A)[:, None], succ] = (1.0 - haz)[:, None] * w
    rows[:, goal] = haz * q_goal
    rows[:, death] = haz * (1.0 - q_goal)
    P_base = np.zeros((S + 2, A, S + 2))
    P_base[:S] = rows.reshape(S, A, S + 2)
    P_base[goal, :, goal] = 1.0
    P_base[death, :, death] = 1.0
    entry = np.zeros(S + 2)
    entry[goal] = 1.0
    d0 = np.zeros(S + 2)
    d0[starts] = 1.0 / len(starts)
    tag = "|".join(str(x) for x in draw_key)
    base = FiniteEnv(name=f"garnet-base|{tag}", kind="garnet", P=P_base, entry_reward=entry, d0=d0,
                     terminals=(goal, death), sink=death, goal=goal, mask=np.ones((S + 2, A), bool))
    _, V, _ = x_optimal(base, "given")
    Q = base.evaluator().q_values(V, "mean")[:S]
    order = np.argsort(-Q, axis=1, kind="stable")                     # ties -> lowest action index
    K = np.zeros((S + 2, A), dtype=bool)
    for s in range(S):
        K[s, order[s, :k[s]]] = True
    pi = np.full((S + 2, A), 1.0 / A)
    pi[:S] = np.where(K[:S], (1.0 - eps) / k[:, None], eps / (A - k)[:, None])
    P = P_base.copy()
    target = death if harm > 0 else goal
    for s in range(S):
        bad = ~K[s]
        P[s, bad] = (1.0 - abs(harm)) * P_base[s, K[s]].mean(axis=0)
        P[s, bad, target] += abs(harm)
    env = FiniteEnv(name=f"garnet|h{harm}|eps{eps}|{tag}", kind="garnet", P=P, entry_reward=entry, d0=d0,
                    terminals=(goal, death), sink=death, goal=goal, mask=np.ones((S + 2, A), bool),
                    meta={"draw_key": list(draw_key), "harm": harm, "eps": eps, "n_states": S, "n_actions": A,
                          "branching": B, "hazard_range": [lo, hi],
                          "preferred_set_sizes": {str(j): int((k == j).sum()) for j in (1, 2, 3)},
                          "dangling_target": death, "dangling_target_label": "death (Garnet has no timeout state)",
                          "censoring_hazard": None})
    assert np.allclose(env.P.sum(axis=2), 1.0, atol=1e-12)
    assert x_trap_states(env.P, env.live, env.mask) == 0, "every Garnet policy must be proper"
    if harm == 0.0:
        assert np.allclose(x_impute_inadmissible(P_base, K, env.live), env.P, atol=1e-15)
    return env, validate_policy(pi, f"Garnet behaviour {env.name}", shape=(S + 2, A))

# %% [markdown]
# ### II.3d ICU-style benchmark construction, true targets, environment cells
#
# `construct_benchmark(true_env, behaviour, n_hosp, τ, rng, name)`:
#
# 1. **Hospital data**: `n_hosp` episodes of the behaviour policy on the true system (`x_sample_dataset`, cutoff
#    `XCFG.max_steps`), validated by `x_validate_dataset`.
# 2. **Decision states** V are the live states logged as `state`. By the chain property of episodes, a live next state
#    outside V can only end a cut-off episode. These dangling transitions are redirected to the timeout state
#    (Gumbel) or to death (Garnet, which has no timeout state). Both targets pay 0, the logged reward of a live
#    transition, and both facts are asserted. Dangling transitions are counted.
# 3. **Mask and dead states.** M(s) = {a : N(s, a) ≥ τ}. With `XCFG.benchmark_empty_rule = "icu"` (default) a
#    decision state with no action reaching τ is handled as ICU-Sepsis's `create_valid_dynamics` handles it: the state
#    is *dead* and is removed, the informative transitions into it are dropped (not redirected), and the remaining
#    rows renormalise; a pair whose every transition entered a removed state loses its row and becomes inadmissible.
#    This repeats until no dead state remains (`dead_states_removed`, `dead_state_rounds`,
#    `transitions_into_dead_dropped`); removed states leave V like censored states, and their initial mass is dropped
#    before d0 is renormalised (step 7). The sensitivity rule `"argmax"` instead keeps such a state with its most
#    frequent action (`empty_row_fallbacks`). The oracle construction keeps the argmax fallback, because the true
#    state space cannot drop states. For the Gumbel benchmark, N counts the *informative* steps only. The timeout is
#    the evaluation horizon (a constant hazard h fixed by the benchmark designer, like ICU-Sepsis's time limit), not
#    clinical dynamics, so logged timeout steps are censoring events. A logged state whose every step was censored
#    carries no dynamics information and is treated like an unobserved state (its inflow is redirected).
# 4. **State space** V ∪ terminals, with the index map `index[bench] = true state`.
# 5. **Rows**: maximum likelihood P̂ = C/N on admissible pairs. For Gumbel this is the maximum-likelihood estimate
#    under the known horizon, P̂ = h·e_timeout + (1 − h)·C_inf/N_inf, which keeps the true horizon in every row.
#    Every inadmissible pair, including logged pairs below τ, gets the uniform mean of the admissible P̂ rows
#    (`x_impute_inadmissible`, the ICU-Sepsis mean rule).
# 6. **Trap censoring.** Under γ = 1 a benchmark is well posed only if every policy absorbs. Rows estimated from a
#    few transitions can miss every absorbing outcome, and under the mean rule all actions of a single-admissible
#    state copy that row. A closed set of live states that some policy never leaves (a "trap") then appears. In local
#    tests of the plain count construction, a quarter of the Gumbel draws and an occasional Garnet draw had one; there
#    `x_optimal` fails and even the clinician estimate can be improper. The known-horizon rows of step 5 rule traps
#    out for Gumbel (every row absorbs with probability ≥ h; asserted). For Garnet the greatest such set (`x_trap_set`,
#    all actions, mean rule) is computed and censored: its states leave V, their logged steps are dropped,
#    transitions into them are redirected like dangling ones, and the model is re-estimated. One pass suffices,
#    because redirecting steps to an absorbing state cannot create a new trap; the code asserts that no trap remains.
#    Censored states (uninformative, dead and trap states), the steps they drop and their initial mass are reported.
#    The empirical d0 is renormalised over the remaining states.
# 7. **d0** is the empirical initial-state distribution. 8. The **clinician policy** of benchmark users is π̂_β
#    from the hospital counts, inadmissible actions included (as in ICU-Sepsis).
#
# **What the construction does not reproduce.** The benchmark is estimated on the true Markov states, so
# ICU-Sepsis's state-aggregation (k-means) error is absent; the measured errors therefore omit this error source, and
# whether aggregation increases or decreases the total benchmark error is not studied. With the ε-optimal Gumbel
# physician the masks are near-singleton and contain the true optimum (see II.3b), so in those cells the termination
# rule is favoured by construction; the `withhold-vaso` cell and the Garnet h < 0 cells are the controls with
# beneficial unsupported actions.
#
# `lift` maps a benchmark policy to the true state space: rows of V are copied; true live states outside V (never
# decided, or censored) play the most frequent hospital action; absorbing rows are uniform. `mask_true` is the
# benchmark mask on V and all-True elsewhere. With the true hazard (Gumbel timeout, Garnet absorption), every policy
# is proper on every true target and on every benchmark, under both semantics; the code asserts this for each
# construction. ICU-Sepsis itself is checked (reported in `icu_adapter_check.json`, not altered).
#
# `x_true_targets(true_env, mask_true)` returns `T_masked` (the true dynamics with the benchmark mask, so violation
# counts and the terminate rule refer to the benchmark's support) and `T_meanimp` (the true rows on admissible pairs,
# with the mean of the true admissible rows on inadmissible pairs).
#
# **Cells** (`X_CELLS`): `icu|released` (draw 0 is the Part I dataset; later draws are fresh clinician datasets
# from the released benchmark), `gumbel|{best,oracle}|n…|tau…|eps…[|phys-withhold-vaso]` and `garnet|h…|eps…`.
# `build_cell_instance(spec, draw)` is a deterministic function of (cell, draw): every random stream is named
# `("xenv", purpose, cell, draw)` with purpose ∈ {hospital, user, split}.

# %%
# ============================================================
# II.3d Benchmark construction, true targets, environment cells
# ============================================================
EXECLOG.mark("II.3d Benchmark construction and cells")


def x_count_mask(N: np.ndarray, tau: int, live: np.ndarray, fallback: bool = True) -> Tuple[np.ndarray, int]:
    """M(s) = {a : N(s, a) >= tau} on live states with data; live states without data and absorbing rows are all-True.

    fallback=True: a decided row without any action reaching tau keeps its most frequent action (lowest index on ties).
    fallback=False: such rows stay empty (the caller removes them as dead states, as ICU-Sepsis does).
    Returns (mask, number of decided rows without an action reaching tau)."""
    N = np.asarray(N, dtype=np.float64)
    live = np.asarray(live, dtype=bool)
    assert float(tau) == int(tau) and int(tau) >= 1, tau
    mask = N >= int(tau)
    decided = live & (N.sum(axis=1) > 0)
    empty = decided & ~mask.any(axis=1)
    if fallback:
        mask[empty, N[empty].argmax(axis=1)] = True
    mask[~decided] = True
    return mask, int(empty.sum())


def _x_dangling_target(env: FiniteEnv) -> int:
    t = int(env.meta["dangling_target"])
    assert t in env.terminals and env.entry_reward[t] == 0.0, \
        "dangling transitions must be redirected to an absorbing state that pays 0 (the logged reward of a live step)"
    return t


def _x_hospital_counts(true_env: FiniteEnv, hosp: Dict[str, np.ndarray]) -> np.ndarray:
    return np.bincount(hosp["state"] * true_env.A + hosp["action"],
                       minlength=true_env.S * true_env.A).reshape(true_env.S, true_env.A).astype(np.float64)


def _x_estimate_on(true_env: FiniteEnv, hosp: Dict[str, np.ndarray], V: np.ndarray, mask_count: np.ndarray,
                   redirect: int, informative: np.ndarray, drop_into: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Rows on decision states V (sorted true indices) + mean rule. Logged steps from states outside V are dropped;
    live next states outside V are redirected to the absorbing state `redirect`. With a known censoring hazard h
    (Gumbel timeout), rows are h * e_timeout + (1 - h) * (empirical law of the informative, non-censored steps).

    drop_into (ICU-Sepsis dead-state rule; true indices of removed states): informative transitions into these states
    are dropped instead of redirected, so the remaining rows renormalise, and a pair left without any transition
    becomes inadmissible. If a state of V is then left without an admissible pair, the result only reports `dead`
    (no rows are built) and the caller removes those states and re-estimates."""
    S, A = true_env.S, true_env.A
    s, a, ns = hosp["state"], hosp["action"], hosp["next_state"]
    hazard = true_env.meta["censoring_hazard"]
    index = np.concatenate([V, np.asarray(true_env.terminals, dtype=np.int64)])
    nV, Sb = len(V), len(index)
    to_bench = np.full(S, -1, dtype=np.int64)
    to_bench[index] = np.arange(Sb)
    keep = to_bench[s] >= 0
    use = keep & informative
    dropped = use & np.isin(ns, drop_into) if drop_into is not None else np.zeros(len(s), dtype=bool)
    use &= ~dropped
    s_b, a_u = to_bench[s[use]], a[use]
    ns_b = to_bench[ns[use]]
    out = ns_b < 0
    ns_b[out] = to_bench[redirect]
    C = np.bincount((s_b * A + a_u) * Sb + ns_b, minlength=Sb * A * Sb).reshape(Sb, A, Sb).astype(np.float64)
    N = C.sum(axis=2)
    live_b = np.zeros(Sb, dtype=bool)
    live_b[:nV] = True
    mask = np.ones((Sb, A), dtype=bool)
    mask[:nV] = mask_count[V]
    if drop_into is not None:
        mask[:nV] &= N[:nV] > 0          # a pair whose every informative step entered a removed state has no row
    dead = ~mask[:nV].any(axis=1)
    res = {"index": index, "to_bench": to_bench, "keep": keep, "out": int(out.sum()), "N": N, "mask": mask,
           "live": live_b, "nV": nV, "Sb": Sb, "dead": dead, "dropped_into_removed": int(dropped.sum())}
    if dead.any():
        return res
    adm = mask & live_b[:, None]
    assert N[adm].min() > 0, "an admissible benchmark pair has no informative logged transition"
    P = np.zeros((Sb, A, Sb))
    P[adm] = C[adm] / N[adm][:, None]
    if hazard is not None:
        c = int(to_bench[int(true_env.meta["timeout_state"])])
        P[adm] *= 1.0 - float(hazard)
        P[adm, c] += float(hazard)
    P = x_impute_inadmissible(P, mask, live_b)
    for t in range(nV, Sb):
        P[t, :, t] = 1.0
    res["P"] = P
    return res


def x_benchmark_from_data(true_env: FiniteEnv, hosp: Dict[str, np.ndarray], tau: int, name: str,
                          empty_rule: Optional[str] = None) -> Dict[str, Any]:
    """ICU-style benchmark estimated from validated hospital data on the true state space (see the markdown).

    empty_rule (default XCFG.benchmark_empty_rule): "icu" removes decision states without an action reaching tau as
    ICU-Sepsis's create_valid_dynamics does (drop the inflow, renormalise, iterate); "argmax" keeps them with their
    most frequent action."""
    S, A = true_env.S, true_env.A
    rule = XCFG.benchmark_empty_rule if empty_rule is None else str(empty_rule)
    assert rule in ("icu", "argmax"), rule
    s, a, ns = hosp["state"], hosp["action"], hosp["next_state"]
    assert true_env.live[s].all()
    hazard = true_env.meta["censoring_hazard"]
    informative = (np.ones(len(s), dtype=bool) if hazard is None
                   else ns != int(true_env.meta["timeout_state"]))
    N_all = _x_hospital_counts(true_env, hosp)
    N_inf = _x_hospital_counts(true_env, {"state": s[informative], "action": a[informative]})
    mask_count, _ = x_count_mask(N_inf, tau, true_env.live, fallback=rule == "argmax")
    redirect = _x_dangling_target(true_env)
    logged = np.unique(s)
    never = true_env.live[ns] & ~np.isin(ns, logged)
    assert hosp["cutoff"][never].all(), "a live next state that is never decided must end a cut-off episode"
    V0 = logged[N_inf[logged].sum(axis=1) > 0]
    uninformative = np.setdiff1d(logged, V0)          # every logged step from these states was censored (timeout)
    # Dead states (rule "icu", ICU-Sepsis create_valid_dynamics): a decision state without an admissible pair is
    # removed, the informative transitions into it are dropped and the remaining rows renormalise; repeat until none.
    dead_states = np.zeros(0, dtype=np.int64)
    V, dead_rounds = V0, 0
    est = _x_estimate_on(true_env, hosp, V, mask_count, redirect, informative,
                         drop_into=None if rule == "argmax" else dead_states)
    while est["dead"].any():
        assert rule == "icu", "the argmax fallback leaves no decision state without an admissible pair"
        dead_states = np.sort(np.concatenate([dead_states, V[est["dead"]]]))
        V = V[~est["dead"]]
        dead_rounds += 1
        assert len(V) > 0, f"{name}: no decision state survives the dead-state removal"
        est = _x_estimate_on(true_env, hosp, V, mask_count, redirect, informative, drop_into=dead_states)
    # Trap censoring after the states from which the estimated model admits a non-absorbing policy (one pass
    # suffices: redirecting steps to an absorbing state cannot create a trap, nor remove an admissible pair).
    trap = x_trap_set(est["P"], est["live"], np.ones((est["Sb"], A), dtype=bool))[:len(V)]
    V_untrapped = V
    V = V[~trap]
    assert len(V) > 0, f"{name}: no decision state survives censoring"
    if trap.any():
        est = _x_estimate_on(true_env, hosp, V, mask_count, redirect, informative,
                             drop_into=None if rule == "argmax" else dead_states)
        assert not est["dead"].any(), f"{name}: trap censoring removed an admissible pair"
    if hazard is not None:
        assert not trap.any(), "rows with a known censoring hazard always absorb"
    censored = np.concatenate([uninformative, dead_states, V_untrapped[trap]])
    index, to_bench, keep, nV, Sb, mask = est["index"], est["to_bench"], est["keep"], est["nV"], est["Sb"], est["mask"]
    P, N = est["P"], est["N"]
    assert x_trap_states(P, est["live"], np.ones((Sb, A), dtype=bool)) == 0, f"{name}: trap states remain"
    first = hosp["timestep"] == 0
    d0 = np.bincount(to_bench[s[first & keep]], minlength=Sb).astype(np.float64)
    assert d0.sum() > 0 and np.all(d0[nV:] == 0), f"{name}: no initial state survives censoring"
    censored_d0 = float(1.0 - d0.sum() / first.sum())
    d0 /= d0.sum()
    beh = np.full((Sb, A), 1.0 / A)
    beh[:nV] = N_all[V] / N_all[V].sum(axis=1, keepdims=True)
    bench = FiniteEnv(name=name, kind="benchmark", P=P, entry_reward=true_env.entry_reward[index], d0=d0,
                      terminals=tuple(range(nV, Sb)), sink=int(to_bench[true_env.sink]), goal=int(to_bench[true_env.goal]),
                      mask=mask, features=None if true_env.features is None else true_env.features[index],
                      meta={"construction": "best", "true_env": true_env.name, "tau": int(tau),
                            "hospital_transitions": int(len(s)), "censoring_hazard": hazard,
                            "censored_states": int(len(censored)), "empty_rule": rule})
    mask_true = np.ones((S, A), dtype=bool)
    mask_true[V] = mask[:nV]
    fallback_action = int(np.bincount(a, minlength=A).argmax())
    outside = true_env.live.copy()
    outside[V] = False

    def lift(pi_bench: np.ndarray) -> np.ndarray:
        pb = validate_policy(pi_bench, f"{name} policy", shape=(Sb, A))
        out = np.full((S, A), 1.0 / A)
        out[outside] = 0.0
        out[outside, fallback_action] = 1.0
        out[V] = pb[:nV]
        return out

    report = {"construction": "best", "tau": int(tau), "empty_rule": rule, "censoring_hazard": hazard,
              "hospital_transitions": int(len(s)),
              "hospital_episodes": int(first.sum()), "censored_steps_timeout": int((~informative).sum()),
              "decision_states_logged": int(len(logged)), "decision_states": int(nV),
              "uninformative_states": int(len(uninformative)),
              "decision_states_without_supported_action": int((~(N_inf[V0] >= int(tau)).any(axis=1)).sum()),
              "dead_states_removed": int(len(dead_states)), "dead_state_rounds": int(dead_rounds),
              "transitions_into_dead_dropped": int(est["dropped_into_removed"]),
              "trap_states_censored": int(trap.sum()),
              "transitions_from_censored_states": int((~keep).sum()),
              "transitions_into_censored_states": int((keep & informative & np.isin(ns, censored)).sum()),
              "censored_d0_mass": censored_d0, "true_live_states": int(true_env.live.sum()),
              "coverage_true_live": float(nV / true_env.live.sum()),
              "true_d0_mass_on_decision_states": float(true_env.d0[V].sum()),
              "observed_pairs": int((N[:nV] > 0).sum()), "admissible_pairs": int(mask[:nV].sum()),
              "mean_admissible_actions": float(mask[:nV].sum(axis=1).mean()),
              "single_admissible_fraction": float((mask[:nV].sum(axis=1) == 1).mean()),
              "empty_row_fallbacks": int((~(N_inf[V] >= int(tau)).any(axis=1)).sum()),
              "logged_pairs_below_tau_discarded": int(((N[:nV] > 0) & ~mask[:nV]).sum()),
              "dangling_transitions": int((never & keep).sum()), "dangling_redirect": true_env.meta["dangling_target_label"],
              "inadmissible_logged_fraction": float((~mask[to_bench[s[keep]], a[keep]]).mean()),
              "fallback_action": fallback_action, "trap_states_given": 0,
              "trap_states_terminate": x_trap_states(P, est["live"], mask)}
    assert report["trap_states_terminate"] == 0
    return {"bench": bench, "bench_behavior": validate_policy(beh, f"{name} behaviour", shape=(Sb, A)),
            "index": index, "mask_true": mask_true, "lift": lift, "fallback_action": fallback_action, "report": report,
            "decision_states_true": V}


def construct_benchmark(true_env: FiniteEnv, behavior: np.ndarray, n_hosp: int, tau: int,
                        rng: np.random.Generator, name: str) -> Dict[str, Any]:
    """Hospital data from the true system -> ICU-style estimated benchmark (mean rule for unsupported actions)."""
    hosp = x_sample_dataset(true_env, behavior, int(n_hosp), XCFG.max_steps, rng)
    hosp_report = x_validate_dataset(hosp, true_env)
    out = x_benchmark_from_data(true_env, hosp, tau, name)
    out["report"]["hospital_validation"] = hosp_report
    del hosp
    return out


def oracle_benchmark(true_env: FiniteEnv, behavior: np.ndarray, n_hosp: int, tau: int,
                     rng: np.random.Generator, name: str) -> Dict[str, Any]:
    """Same hospital mask, but true state space, true admissible rows (mean of true admissible rows elsewhere), true d0.

    The true state space cannot drop states, so a decided state without an action reaching tau keeps its most
    frequent action (x_count_mask fallback) whatever XCFG.benchmark_empty_rule says."""
    hosp = x_sample_dataset(true_env, behavior, int(n_hosp), XCFG.max_steps, rng)
    hosp_report = x_validate_dataset(hosp, true_env)
    hazard = true_env.meta["censoring_hazard"]
    inf = (np.ones(len(hosp["state"]), dtype=bool) if hazard is None
           else hosp["next_state"] != int(true_env.meta["timeout_state"]))
    N_true = _x_hospital_counts(true_env, {"state": hosp["state"][inf], "action": hosp["action"][inf]})
    mask, n_fallback = x_count_mask(N_true, tau, true_env.live)
    bench = true_env.variant(name, P=x_impute_inadmissible(true_env.P, mask, true_env.live), mask=mask,
                             kind="benchmark", construction="oracle", tau=int(tau))
    decided = true_env.live & (N_true.sum(axis=1) > 0)
    fallback_action = int(np.bincount(hosp["action"], minlength=true_env.A).argmax())
    report = {"construction": "oracle", "tau": int(tau), "empty_rule": "argmax (true state space keeps every state)",
              "censoring_hazard": hazard,
              "hospital_transitions": int(len(hosp["state"])), "censored_steps_timeout": int((~inf).sum()),
              "hospital_episodes": int((hosp["timestep"] == 0).sum()), "decision_states": int(decided.sum()),
              "true_live_states": int(true_env.live.sum()),
              "coverage_true_live": float(decided.sum() / true_env.live.sum()),
              "true_d0_mass_on_decision_states": float(true_env.d0[decided].sum()),
              "observed_pairs": int((N_true[decided] > 0).sum()), "admissible_pairs": int(mask[decided].sum()),
              "mean_admissible_actions": float(mask[decided].sum(axis=1).mean()),
              "single_admissible_fraction": float((mask[decided].sum(axis=1) == 1).mean()),
              "empty_row_fallbacks": n_fallback,
              "logged_pairs_below_tau_discarded": int(((N_true > 0) & ~mask)[decided].sum()),
              "dangling_transitions": 0, "dangling_redirect": "none (true state space)",
              "inadmissible_logged_fraction": float((~mask[hosp["state"], hosp["action"]]).mean()),
              "fallback_action": fallback_action,
              "trap_states_given": x_trap_states(bench.P, bench.live, np.ones((bench.S, bench.A), bool)),
              "trap_states_terminate": x_trap_states(bench.P, bench.live, bench.mask),
              "hospital_validation": hosp_report}
    assert report["trap_states_given"] == 0, "convex completions of proper true rows must stay proper"
    del hosp
    return {"bench": bench, "bench_behavior": validate_policy(behavior, f"{name} behaviour", shape=(true_env.S, true_env.A)),
            "index": np.arange(true_env.S), "mask_true": bench.mask.copy(), "lift": None,
            "fallback_action": fallback_action, "report": report, "decision_states_true": np.flatnonzero(decided)}


def x_true_targets(true_env: FiniteEnv, mask_true: np.ndarray) -> Tuple[FiniteEnv, FiniteEnv]:
    """(T_masked: true dynamics + benchmark mask, T_meanimp: true admissible rows + mean of them elsewhere)."""
    T_masked = true_env.variant(f"{true_env.name}|true-masked", mask=mask_true, role="true dynamics, benchmark mask")
    T_meanimp = true_env.variant(f"{true_env.name}|true-rows-mean",
                                 P=x_impute_inadmissible(true_env.P, T_masked.mask, true_env.live), mask=mask_true,
                                 role="true admissible rows, mean rule elsewhere")
    return T_masked, T_meanimp


def x_build_cells() -> List[dict]:
    cells: List[dict] = []
    if EXPERIMENTS["xenv_icu"]:
        cells.append({"cell": "icu|released", "family": "icu", "draws": int(XCFG.icu_draws),
                      "user_episodes": int(XCFG.icu_user_episodes)})
    if EXPERIMENTS["xenv_gumbel"]:
        for entry in XCFG.gumbel_cells:
            construction, n_hosp, tau, eps, *rest = entry
            assert construction in ("best", "oracle"), construction
            assert len(rest) <= 1, f"Gumbel cell entries are (construction, n_hosp, tau, eps[, physician]): {entry}"
            physician = str(rest[0]) if rest else "optimal"
            assert physician in GUMBEL_PHYSICIAN_KINDS, physician
            tag = "" if physician == "optimal" else f"|phys-{physician}"
            cells.append({"cell": f"gumbel|{construction}|n{n_hosp}|tau{tau}|eps{eps}{tag}", "family": "gumbel",
                          "draws": int(XCFG.gumbel_draws), "construction": construction, "n_hosp": int(n_hosp),
                          "tau": int(tau), "eps": float(eps), "physician": physician, "variant": XCFG.gumbel_variant,
                          "timeout_hazard": float(XCFG.gumbel_timeout_hazard),
                          "user_episodes": int(XCFG.gumbel_user_episodes)})
    if EXPERIMENTS["xenv_garnet"]:
        for harm in XCFG.garnet_harm:
            for eps in XCFG.garnet_behavior_eps:
                cells.append({"cell": f"garnet|h{harm}|eps{eps}", "family": "garnet", "draws": int(XCFG.garnet_draws),
                              "harm": float(harm), "eps": float(eps), "n_states": int(XCFG.garnet_states),
                              "n_actions": int(XCFG.garnet_actions), "branching": int(XCFG.garnet_branching),
                              "hazard_range": tuple(float(x) for x in XCFG.garnet_hazard),
                              "n_hosp": int(XCFG.garnet_hosp_episodes), "tau": int(XCFG.garnet_tau),
                              "user_episodes": int(XCFG.garnet_user_episodes)})
    keys = [c["cell"] for c in cells]
    assert len(keys) == len(set(keys)), f"duplicate cell keys: {keys}"
    return cells


_X_GUMBEL_PHYS: Dict[tuple, np.ndarray] = {}
_X_GUMBEL_OPT_ACTION: Dict[tuple, np.ndarray] = {}
_X_GARNET_CACHE: "OrderedDict[tuple, Tuple[FiniteEnv, np.ndarray]]" = OrderedDict()


def _x_gumbel_env(variant: str, hazard: float) -> FiniteEnv:
    env = X_GUMBEL_ENVS.get(variant)
    if env is None or float(env.meta["timeout_hazard"]) != float(hazard):
        raise KeyError(f"Gumbel env ({variant}, h={hazard}) was not built in II.3b; set EXPERIMENTS['xenv_gumbel'] "
                       "and XCFG.gumbel_timeout_hazard before this section.")
    return env


def _x_gumbel_physician(variant: str, hazard: float, eps: float, kind: str = "optimal") -> np.ndarray:
    key = (variant, float(hazard), float(eps), str(kind))
    if key not in _X_GUMBEL_PHYS:
        _X_GUMBEL_PHYS[key] = gumbel_physician(_x_gumbel_env(variant, hazard), eps, kind)
    return _X_GUMBEL_PHYS[key]


def _x_gumbel_opt_action(variant: str, hazard: float) -> np.ndarray:
    """argmax of the exact optimum Q*_true of the Gumbel system (cached per variant and hazard)."""
    key = (variant, float(hazard))
    if key not in _X_GUMBEL_OPT_ACTION:
        pi_opt, _, _ = x_optimal(_x_gumbel_env(variant, hazard), "given")
        _X_GUMBEL_OPT_ACTION[key] = pi_opt.argmax(axis=1)
    return _X_GUMBEL_OPT_ACTION[key]


def _x_garnet_env(spec: dict, draw: int, keep: int = 4) -> Tuple[FiniteEnv, np.ndarray]:
    key = (spec["cell"], int(draw))
    if key in _X_GARNET_CACHE:
        _X_GARNET_CACHE.move_to_end(key)
        return _X_GARNET_CACHE[key]
    val = garnet_true_env(("base", int(draw)), spec["n_states"], spec["n_actions"], spec["branching"],
                          spec["hazard_range"], spec["harm"], spec["eps"])
    _X_GARNET_CACHE[key] = val
    while len(_X_GARNET_CACHE) > keep:
        _X_GARNET_CACHE.popitem(last=False)
    return val


def build_cell_instance(spec: dict, draw: int) -> CellInstance:
    """Deterministic (cell, draw) instance: learner environment, behaviour, user data split and evaluation targets."""
    t0 = time.perf_counter()
    cell, family, draw = spec["cell"], spec["family"], int(draw)
    assert draw >= 0, draw
    rng_h, rng_u = np_rng("xenv", "hospital", cell, draw), np_rng("xenv", "user", cell, draw)
    construction, hosp_validation = None, None
    if family == "icu":
        assert X_ICU is not None
        if draw == 0:
            data, source = OFFLINE, "Part I OFFLINE dataset (Section 5)"
        else:
            data = x_sample_dataset(X_ICU, X_ICU_BEHAVIOR, spec["user_episodes"], XCFG.max_steps, rng_u)
            source = "x_sample_dataset(X_ICU, clinician)"
        learn_env, behavior = X_ICU, X_ICU_BEHAVIOR
        targets = [("bench_mean", X_ICU, "given", None), ("bench_terminate", X_ICU, "terminate", None)]
        deploy = "bench_terminate"
    elif family in ("gumbel", "garnet"):
        if family == "gumbel":
            true_env = _x_gumbel_env(spec["variant"], spec["timeout_hazard"])
            true_behavior = _x_gumbel_physician(spec["variant"], spec["timeout_hazard"], spec["eps"],
                                                spec.get("physician", "optimal"))
            kind = spec["construction"]
        else:
            true_env, true_behavior = _x_garnet_env(spec, draw)
            kind = "best"
        bname = f"{cell}|d{draw}|bench"
        if kind == "best":
            B = construct_benchmark(true_env, true_behavior, spec["n_hosp"], spec["tau"], rng_h, bname)
            learn_env, behavior, lift = B["bench"], B["bench_behavior"], B["lift"]
            data = x_sample_dataset(learn_env, behavior, spec["user_episodes"], XCFG.max_steps, rng_u)
            T_masked, T_meanimp = x_true_targets(true_env, B["mask_true"])
            targets = [("bench_mean", learn_env, "given", None), ("bench_terminate", learn_env, "terminate", None),
                       ("true", T_masked, "given", lift), ("true_rows_mean", T_meanimp, "given", lift),
                       ("true_rows_terminate", T_masked, "terminate", lift)]
        elif kind == "oracle":
            B = oracle_benchmark(true_env, true_behavior, spec["n_hosp"], spec["tau"], rng_h, bname)
            learn_env, behavior = B["bench"], true_behavior
            data = x_sample_dataset(learn_env, behavior, spec["user_episodes"], XCFG.max_steps, rng_u)
            T_masked, _ = x_true_targets(true_env, B["mask_true"])
            targets = [("bench_mean", learn_env, "given", None), ("bench_terminate", T_masked, "terminate", None),
                       ("true", T_masked, "given", None)]
        else:
            raise ValueError(f"unknown construction {kind!r}")
        if family == "gumbel":   # is the true optimal action supported? (decision states of the construction)
            opt_act, Vt = _x_gumbel_opt_action(spec["variant"], spec["timeout_hazard"]), B["decision_states_true"]
            B["report"]["true_optimum_in_mask_fraction"] = (float(B["mask_true"][Vt, opt_act[Vt]].mean()) if len(Vt)
                                                            else float("nan"))
        deploy = "true"
        construction = {k: v for k, v in B["report"].items() if k != "hospital_validation"}
        hosp_validation = B["report"]["hospital_validation"]
        source = f"x_sample_dataset({kind} benchmark, {'clinician estimate' if kind == 'best' else 'true behaviour'})"
        del B
    else:
        raise ValueError(f"unknown family {family!r}")
    labels = [t[0] for t in targets]
    assert len(labels) == len(set(labels)) and deploy in labels, labels
    for lab, env, sem, lift_fn in targets:
        assert sem in X_SEMANTICS and (lift_fn is not None or env.S == learn_env.S), (lab, env.S, learn_env.S)
    user_validation = x_validate_dataset(data, learn_env)
    train, dev = x_split_episodes(data, XCFG.dev_fraction, np_rng("xenv", "split", cell, draw))
    meta = {"spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}, "user_data_source": source,
            "user_validation": user_validation, "hospital_validation": hosp_validation, "construction": construction,
            "train_transitions": int(len(train["state"])), "dev_transitions": int(len(dev["state"])),
            "build_s": time.perf_counter() - t0}
    return CellInstance(cell=cell, family=family, draw=draw, learn_env=learn_env, behavior=behavior, data=data,
                        train=train, dev=dev, targets=targets, deploy_target=deploy, meta=meta)


def _x_value(env: FiniteEnv, pi: np.ndarray, sem: str) -> Tuple[float, str]:
    ev = x_evaluate(env, pi, sem)
    return (float(ev["return"]), "ok") if ev["status"] == "ok" else (float("nan"), ev["status"])


def _x_opt_value(env: FiniteEnv, sem: str, constrained: bool = False) -> Tuple[float, str]:
    try:
        _, _, info = x_optimal(env, sem, constrained=constrained)
        return float(info["dp_value"]), "ok"
    except ImproperPolicyError:
        return float("nan"), "improper"


def x_environment_suite_table() -> pd.DataFrame:
    """One row per cell from draw 0: sizes, mask statistics, behaviour and optimum values per target, the value of
    unsupported actions J*_true(all actions) - J*_true(admissible only), and the construction report."""
    rows = []
    for spec in X_CELLS:
        inst = build_cell_instance(spec, 0)
        L = inst.learn_env
        row = {"cell": spec["cell"], "family": spec["family"], "draws": spec["draws"], "deploy_target": inst.deploy_target,
               "S": L.S, "A": L.A, "live_states": int(L.live.sum()), "admissible_pairs": int(L.mask[L.live].sum()),
               "mean_admissible_actions": float(L.mask[L.live].sum(axis=1).mean()),
               "user_transitions": inst.meta["user_validation"]["transitions"],
               "user_episodes": inst.meta["user_validation"]["episodes"],
               "user_inadmissible_logged_fraction": inst.meta["user_validation"]["inadmissible_logged_fraction"],
               "build_s": inst.meta["build_s"]}
        statuses = []
        for lab, env, sem, lift_fn in inst.targets:
            beh = inst.behavior if lift_fn is None else lift_fn(inst.behavior)
            row[f"behaviour_value[{lab}]"], st_b = _x_value(env, beh, sem)
            row[f"optimum_value[{lab}]"], st_o = _x_opt_value(env, sem)
            statuses += [f"{what}[{lab}]={st}" for what, st in (("behaviour", st_b), ("optimum", st_o)) if st != "ok"]
        true_t = {lab: env for lab, env, _, _ in inst.targets}.get("true")
        if true_t is not None:
            j_all, st_all = _x_opt_value(true_t, "given")
            j_adm, st_adm = _x_opt_value(true_t, "given", constrained=True)
            row["value_of_unsupported_actions"] = j_all - j_adm
            statuses += [f"optimum_admissible[true]={st_adm}"] * (st_adm != "ok") + [f"optimum[true]={st_all}"] * (st_all != "ok")
        else:
            row["value_of_unsupported_actions"] = float("nan")
        row["non_ok_statuses"] = "; ".join(statuses)
        for k, v in (inst.meta["construction"] or {}).items():
            row[f"construction_{k}"] = v
        rows.append(row)
        del inst
        gc.collect()
    return pd.DataFrame(rows)


X_CELLS: List[dict] = x_build_cells()
print("Part II cells:", [(c["cell"], c["draws"]) for c in X_CELLS])
X_ENV_SUITE = pd.DataFrame()
if X_CELLS:
    _t0 = time.perf_counter()
    X_ENV_SUITE = x_environment_suite_table()
    X_ENV_SUITE.to_csv(XENV_TABLE_DIR / "xenv_environment_suite.csv", index=False)
    _cols = [c for c in ("cell", "S", "live_states", "mean_admissible_actions", "behaviour_value[true]",
                         "optimum_value[true]", "value_of_unsupported_actions", "construction_empty_row_fallbacks",
                         "construction_dead_states_removed", "construction_single_admissible_fraction",
                         "construction_true_optimum_in_mask_fraction",
                         "construction_uninformative_states", "construction_trap_states_censored",
                         "construction_censored_d0_mass", "non_ok_statuses") if c in X_ENV_SUITE.columns]
    display(X_ENV_SUITE[_cols])
    print(f"Environment suite table built in {time.perf_counter() - _t0:.1f} s.")

# Objects whose source enters the Part II code hash.
XENV_ENV_CODE = [x_icu_behavior, x_icu_env, GumbelSim, gumbel_full_index, gumbel_decode_full, gumbel_action_bits,
                 gumbel_num_abnormal, gumbel_on_treatment, gumbel_reward, gumbel_is_absorbing, _gx_rep, _gx_case,
                 _gx_u, _gx_flag, _gx_antibiotics_on, _gx_antibiotics_off, _gx_vent_on, _gx_vent_off, _gx_vaso_on,
                 _gx_vaso_off, _gx_fluctuate, gumbel_transition_dist, gumbel_initial_distribution,
                 gumbel_state_space, x_trap_set, x_trap_states, gumbel_true_env_report, gumbel_true_env,
                 gumbel_physician, garnet_true_env, x_count_mask, _x_dangling_target, _x_hospital_counts,
                 _x_estimate_on, x_benchmark_from_data, construct_benchmark, oracle_benchmark, x_true_targets,
                 x_build_cells, _x_gumbel_env, _x_gumbel_physician, _x_gumbel_opt_action, _x_garnet_env,
                 build_cell_instance]
print("Part II environments ready.")
