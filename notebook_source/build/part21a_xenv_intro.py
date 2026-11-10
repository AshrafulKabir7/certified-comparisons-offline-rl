# %% [markdown]
# # Part II — Cross-environment exact study (cross-environment extension)
#
# Part I (Sections 0–31) is a deep case study of one benchmark. Part II asks the question that a
# reader needs answered before any ICU-Sepsis result can be interpreted:
#
# > **Unsupported actions are a modelling choice.** Offline-RL benchmarks built from clinical records must say
# > what happens when a policy takes an action that the data never (or rarely) supports. ICU-Sepsis replaces such
# > an action by the mean of the supported ones ("mean" rule) or ends the episode ("terminate" rule). How does that
# > choice change violation costs, optimal policies, and **method rankings** — and do benchmark rankings agree with
# > the rankings on the true system that generated the data?
#
# ICU-Sepsis cannot answer the last question, because its true dynamics for unsupported actions are unknown. Part II
# therefore rebuilds an **ICU-Sepsis-style benchmark** (hospital data → count threshold τ → admissible mask →
# estimated rows for supported pairs + a completion rule for the rest → benchmark users simulate datasets from it)
# on top of environments whose **true dynamics are known exactly**:
#
# | Environment family | True dynamics for unsupported actions | Role |
# |---|---|---|
# | ICU-Sepsis v2 (released benchmark) | unknown | the benchmark itself; mean / terminate rules; 50 user-dataset draws (draw 0 = the Part I dataset) |
# | Gumbel-max sepsis simulator (Oberst & Sontag, ICML 2019), exact enumeration | known | ground-truth benchmark-validity study across hospital-data size and threshold τ |
# | Garnet absorbing CMDPs with a harm parameter *h* for unsupported actions | known, controlled | phase diagram: when does the completion rule mislead? (*h* = 0: non-preferred unsupported actions are completed correctly; the semantics term vanishes where K(s) ⊆ M(s), see `table_error_decomposition.csv`) |
#
# **Research questions.** RQ1 (theory, Section II.5): seven propositions on unsupported-action semantics, verified
# numerically per proposition: P1–P3 and P5 on ICU-Sepsis (P1/P2 where the released mean-rule identity holds), the
# draw-0 Gumbel and Garnet instances and 20 random absorbing MDPs; P4 on the Gumbel and Garnet instances and the random
# MDPs (no ground truth for ICU-Sepsis); P7 on ICU-Sepsis and the draw-0 benchmark instances; P6 by the Part I
# gradient tests (Section 13) and closed-form checks. RQ2 (benchmark validity): Kendall-τ rank fidelity, top-1
# agreement and selection regret between benchmark and true values, with an exact error decomposition. RQ3 (methods): what masking,
# projection, pessimism, Lagrangian constraints, support filtering, guided behaviour priors and exact categorical
# diffusion buy over the trivial admissible projection, with rliable-style statistics over many dataset draws.
#
# **Unit of analysis.** All Part II learners are deterministic functions of their training data, so the experimental
# unit is the **dataset draw** (for Gumbel / Garnet: a full benchmark-construction replicate). Hyper-parameter grids
# have equal size per tunable method; results are reported with pre-declared defaults, with selection by tabular FQE
# on held-out episodes, and with oracle selection (upper bound, labelled).
#
# Sections: II.1 setup and preregistration · II.2 core (environment object, evaluator factory, sampler, empirical
# models, CMDP LP, cache) · II.3 environments and benchmark construction · II.4 method suite · II.5 propositions and
# exact constrained frontiers · II.6 driver · II.7 statistics, figures and tables. Part II is placed before the
# release sections (32–35) so that every output is packaged and mapped.

# %%
# ============================================================
# II.1 Part II setup and preregistration
# ============================================================
EXECLOG.mark("II.1 Cross-environment setup")
import scipy.sparse as sp
from scipy.optimize import linprog

XENV_TABLE_DIR = TABLE_DIR / "xenv"
XENV_FIG_DIR = FIG_DIR / "xenv"
XENV_CACHE_DIR = ROOT / "xenv_cache"
for _p in (XENV_TABLE_DIR, XENV_FIG_DIR, XENV_CACHE_DIR):
    _p.mkdir(parents=True, exist_ok=True)


@dataclass
class XConfig:
    # Dataset draws per environment cell (the experimental unit of Part II).
    icu_draws: int = 50
    gumbel_draws: int = 30
    garnet_draws: int = 30
    # Benchmark-user datasets and the train / dev split used for offline selection.
    icu_user_episodes: int = 10_000
    gumbel_user_episodes: int = 5_000
    garnet_user_episodes: int = 2_000
    max_steps: int = 100
    dev_fraction: float = 0.2

    # Gumbel-max sepsis simulator (exact enumeration of the published simulator code).
    gumbel_variant: str = "sim-v2"                 # "sim-v2" (glucose-cap fix) | "master" (published notebooks)
    gumbel_timeout_hazard: float = 0.05            # geometric horizon: mean 20 decisions, stationary optimum
    # (construction, hospital episodes, support threshold tau, physician epsilon[, physician kind]); kind 'optimal'
    # (default) softens the true optimum, 'withhold-vaso' the optimum without vasopressors (a control in which
    # unsupported actions are beneficial, parallel to Garnet h < 0; see gumbel_physician).
    gumbel_cells: Tuple[tuple, ...] = (
        ("best", 2_000, 5, 0.05), ("best", 2_000, 20, 0.05), ("best", 10_000, 5, 0.05),
        ("best", 10_000, 20, 0.05), ("best", 10_000, 20, 0.20), ("best", 10_000, 20, 0.05, "withhold-vaso"),
        ("oracle", 10_000, 20, 0.05))
    gumbel_mc_pairs: int = 300                     # (state, action) pairs checked against the step sampler
    gumbel_mc_draws: int = 4_000

    # Garnet absorbing CMDP family.
    garnet_states: int = 150
    garnet_actions: int = 8
    garnet_branching: int = 4
    garnet_hazard: Tuple[float, float] = (0.05, 0.20)   # per-step absorption probability range
    garnet_harm: Tuple[float, ...] = (-0.5, 0.0, 0.5, 1.0)
    garnet_behavior_eps: Tuple[float, ...] = (0.05, 0.30)
    garnet_hosp_episodes: int = 5_000
    garnet_tau: int = 5
    # Decision states without an action reaching tau: "icu" removes them like ICU-Sepsis's create_valid_dynamics
    # (drop inflow, renormalise, iterate); "argmax" keeps them with their most frequent action (sensitivity run).
    benchmark_empty_rule: str = "icu"

    # Method grids (equal size for comparable tunable methods; *_default is the pre-declared configuration).
    cql_alpha_grid: Tuple[float, ...] = (0.3, 1.0, 3.0)
    cql_alpha_default: float = 1.0
    cql_steps: int = 10_000                        # full-batch Adam steps on the exact empirical CQL objective
    cql_lr: float = 0.02
    cql_target_every: int = 25                     # 400 hard target syncs (Bellman backups)
    cql_converged_tol: float = 5e-3                # max |dQ| on observed pairs over the last sync interval
    iql_expectile_grid: Tuple[float, ...] = (0.6, 0.7, 0.9)
    iql_expectile_default: float = 0.7
    iql_beta: float = 3.0
    iql_max_weight: float = 100.0
    bcq_tau_grid: Tuple[float, ...] = (0.1, 0.3, 0.5)
    bcq_tau_default: float = 0.3
    lcb_c_grid: Tuple[float, ...] = (0.03, 0.1, 0.3)
    lcb_c_default: float = 0.1
    mbsqi_b_grid: Tuple[int, ...] = (1, 5, 20)
    mbsqi_b_default: int = 5
    spibb_n_grid: Tuple[int, ...] = (5, 20, 50)
    spibb_n_default: int = 20
    lagrange_kappa_grid: Tuple[float, ...] = (0.0, 0.01, 0.05)
    lagrange_kappa_default: float = 0.0
    mblp_kappa_grid: Tuple[float, ...] = (0.0, 0.01, 0.05)
    mblp_kappa_default: float = 0.0
    sfbc_alpha_grid: Tuple[float, ...] = (1.0, 3.0, 10.0)
    sfbc_alpha_default: float = 3.0
    diffusion_guidance_grid: Tuple[float, ...] = (0.5, 1.5, 3.0)
    diffusion_guidance_default: float = 1.5
    max_iterations: int = 20_000
    tol: float = 1e-10

    # Exact constrained frontiers (expected violation budget kappa; inf = unconstrained).
    frontier_kappas: Tuple[float, ...] = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, float("inf"))

    # Statistics.
    bootstrap: int = 5_000
    equivalence_margin_normalized: float = 0.05


def apply_xrun_mode(xc: XConfig, mode: str, profile: str) -> XConfig:
    if profile == "lite":
        xc.icu_draws, xc.gumbel_draws, xc.garnet_draws = 20, 15, 15
    if mode == "smoke":
        xc.icu_draws = xc.gumbel_draws = xc.garnet_draws = 2
        xc.icu_user_episodes = xc.gumbel_user_episodes = 300
        xc.garnet_user_episodes = 200
        xc.gumbel_cells = (("best", 300, 5, 0.05), ("oracle", 300, 5, 0.05))
        xc.gumbel_mc_pairs, xc.gumbel_mc_draws = 40, 500
        xc.garnet_states, xc.garnet_harm, xc.garnet_behavior_eps = 30, (0.0, 1.0), (0.05,)
        xc.garnet_hosp_episodes = 300
        xc.cql_steps, xc.cql_target_every = 200, 20
        xc.max_iterations = 5_000
        xc.frontier_kappas = (0.0, 0.1, float("inf"))
        xc.bootstrap = 300
    return xc


XCFG = apply_xrun_mode(XConfig(), RUN_MODE, COMPUTE_PROFILE)

# Canonical policy labels used by the preregistration, the driver and the statistics: f"{method} [{variant}]",
# variant in {native, masked, projected}; the label refers to the pre-declared default hyper-parameters.
XCOMPARATOR = "IQL [projected]"
XTREATMENTS = ["BC [masked]", "CQL [masked]", "IQL [masked]", "BCQ [masked]", "VI-LCB [masked]", "MBS-QI [masked]",
               "SPIBB [projected]", "Lagrangian-MB [native]", "CMDP-LP (model) [native]", "SfBC (exact) [masked]",
               "FISOR-tab (exact) [native]", "Diffusion (exact, guided) [native]"]

PREREG_PART2 = {
    "status": "DRAFT - lock (set status to LOCKED with a date) before the confirmatory Part II run",
    "unit": ("dataset draw: a full benchmark-construction replicate (hospital data, mask, estimated benchmark, user "
             "data) for Gumbel / Garnet; a user-dataset replicate for ICU-Sepsis. Learners are deterministic given data."),
    "deployment_target": "true dynamics where available (Gumbel, Garnet); terminate rule for ICU-Sepsis (no ground truth)",
    "normalized_score": "(J - J_behaviour) / (J*_target - J_behaviour) per (cell, draw, target); J* is the target's optimum",
    "default_policy_set": "every method at its *_default hyper-parameters, variants native / masked / projected",
    "H1_benchmark_fidelity": {
        "question": ("Does the completion rule for unsupported actions (mean imputation vs termination) change how faithfully "
                     "ICU-style benchmark rankings reproduce true-system rankings?"),
        "metric": ("Kendall tau-b between benchmark values and true values over the default policy set, per draw; a "
                   "benchmark that assigns every policy the same value scores tau = 0 (no ranking information); such "
                   "draws are counted and reported"),
        "test": "two-sided sign-flip on per-draw differences tau(mean) - tau(terminate) within each Gumbel estimated-benchmark cell (construction 'best'); Holm across cells",
        "alpha": 0.05,
    },
    "H2_projection_sufficiency": {
        "question": "Does any learned constraint-aware method improve the deployment value over the admissible projection of tabular IQL?",
        "comparator": XCOMPARATOR,
        "treatments": XTREATMENTS,
        "metric": "normalized deployment score",
        "test": ("two-sided sign-flip on paired differences within each environment family (ICU, Gumbel, Garnet; "
                 "ICU/Gumbel: (cell, draw) units pooled across cells (independent streams); Garnet: one unit per "
                 "base-MDP draw (mean over the harm x eps cells that share it); intervals resample Garnet draws jointly "
                 "across cells); Holm over treatments x families; TOST with the declared margin on the same units"),
        "equivalence_margin_normalized": XCFG.equivalence_margin_normalized,
        "alpha": 0.05,
    },
    "H3_harm_moderates_error": {
        "question": "Does the benchmark error of mean imputation grow with the harm |h| of unsupported actions (Garnet)?",
        "metric": "mean |J_true - J_bench_mean| over the default policy set, per draw",
        "test": ("Spearman correlation between |h| and the per-draw error, two-sided permutation p-value: permutation "
                 "of |h| within (eps, draw) blocks, 10 000 permutations (h = -0.5 kept as a beneficial-unsupported "
                 "control)"),
        "alpha": 0.05,
    },
    "exploratory": ["error decomposition (semantics vs estimation/coverage)", "exact constrained frontiers",
                    "performance profiles", "FQE vs oracle hyper-parameter selection", "exact tabular vs SGD (Part I) agreement"],
    # Confirmatory design and analysis settings (XConfig() defaults = the full paper design, independent of the
    # profile): every field except the simulator Monte-Carlo check sizes and the exploratory frontier budgets;
    # xs_prereg_report lists every realised deviation (`design_deviations`).
    "design": {k: v for k, v in asdict(XConfig()).items()
               if k not in ("gumbel_mc_pairs", "gumbel_mc_draws", "frontier_kappas")},
    "analysis": {"H1": "sign-flip per cell on all draws (constant-benchmark draws score tau = 0); Holm across cells",
                 "H2": ("ICU / Gumbel: (cell, draw) units pooled; Garnet: blocked by draw (mean over its cells), "
                        "bootstrap resamples Garnet draws jointly across cells"),
                 "H3": "permutation of |h| within (eps, draw) blocks, 10 000 permutations",
                 "sign_flip": "exact enumeration for n <= 16; Monte Carlo with 200 000 sign patterns (+1 correction) otherwise",
                 "bootstrap": "stratified percentile bootstrap, XCFG.bootstrap replicates"},
}
save_json(asdict(XCFG), XENV_TABLE_DIR / "xenv_config.json")
save_json(PREREG_PART2, XENV_TABLE_DIR / "xenv_preregistration.json")
print("Part II config:", {k: v for k, v in asdict(XCFG).items() if k.endswith("draws")}, "| profile:", COMPUTE_PROFILE)
