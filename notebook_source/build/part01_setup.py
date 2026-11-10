# %% [markdown]
# # Action Support and Policy Evaluation in the ICU-Sepsis Benchmark
# ## Google Colab research notebook (offline RL, exact finite-MDP evaluation)
#
# This notebook is the **revised implementation** of the constraint-aware offline-RL study
# (BC, tabular CQL / IQL / BCQ, twin-dueling Deep CQL, hard and soft admissibility-aware CQL,
# direct categorical policies, and a critic-guided categorical diffusion policy) on
# `Sepsis/ICU-Sepsis-v2` (`icu-sepsis==2.0.1`).
#
# It implements **every code-level repair and every new experiment** of the audit of the original study (the paper's Appendix F) — **Part I**, Sections 0–31 — and a
# **cross-environment extension, Part II** (Sections II.1–II.7), which re-runs the central question on environments whose
# true dynamics for unsupported actions are known exactly. Sections 32–35 then replay, package, gate and hand off
# both parts. 
#
# ### What changed relative to the previous notebook (audit ID → where it is fixed)
#
# | Audit ID | Repair implemented here | Section |
# |---|---|---|
# | **C01** | Mean-equivalent admissible projection of every policy; masked BC, deterministic masked IQL (advantage-weighted scores), uniform-admissible random; exact verification that projected and original policies induce identical mean-rule transition/reward vectors; replay of the *original* Drive artifacts when available | 7, 14, 19, 21 |
# | **C02** | Legacy hard AM-CQL kept only as a labelled failure-mode control; two documented repaired designs (A: full-action CQL + masked backup/extraction, B: masked CQL on admissible-logged rows with explicit normalization); TD-row ablation (retain vs exclude inadmissible logged transitions); Huber vs squared TD; conservative-weight sensitivity; separately labelled target-clipping and discounting ablations; Q-quantile / spread / gradient-norm / target / full-buffer Bellman-residual / float32-resolution diagnostics; all dependent diffusion policies recomputed | 10, 16, 23 |
# | **C03** | Seeding *before* model construction; named RNG streams (init / minibatch / corruption / evaluation / dataset); run identity = dataset hash + benchmark assets hash + normalizer hash + config hash + code hash + seed + run mode stored and validated in every checkpoint; incompatible checkpoints rejected; normalizer stored inside checkpoints; unconditional dataset-rebuild override removed; resumable checkpoints (optimizer, targets, step, dual variable, RNG states); order / unrelated-draw / resume / cache-invalidation tests | 2, 12, 13 |
# | **C04** | Every main and supplementary table is generated here; artifact replay (policy files + checkpoints); release directory with per-seed CSVs, policies, checkpoint metadata, environment lock, artifact inventory with sizes + SHA-256, table-to-command map, zip archive | 21, 32–34 |
# | **M01** | Admissible support coverage computed only on observed ∩ admissible pairs, separate observed / admissible / inadmissible counts, per-pair frequency quantiles, corrected histogram with asserted population | 6 |
# | **M02** | Released benchmark-support mask vs. support estimated from the simulated training data are separate objects; threshold inequality verified from the benchmark construction code; estimated-support experiment with false exclusions/inclusions, empty rows, fallback frequency | 3, 4, 25 |
# | **M03** | Preregistration block; two-sided exact sign-flip tests; Holm over the full family; attainability report; effect sizes + intervals; sign-flip test inversion; TOST equivalence; hierarchical (dataset → training-seed) analysis; replication planning | 1, 27, 30 |
# | **M04** | Episode-level train/dev split, normalizer fitted on training episodes only, tabular FQE selector validated against exact returns, locked evaluation on a fresh dataset, tuning-budget disclosure | 26 |
# | **M05** | Soft AM-CQL training/extraction alignment study, surrogate vs occupancy metrics, primal/dual residuals, per-state violation distributions, mean/min critic disagreement, margin/tie analysis, tolerance and temperature sensitivity | 10, 24 |
# | **M06** | Matched-capacity direct categorical policy, behaviour-prior × critic, advantage-weighted categorical imitation, critic-only, tabular-only / neural-only / mixed guidance, schedule length, initialization, masking and guidance placement, cost accounting (params, wall time, memory, policy construction, lookup latency) | 11–12, 18, 22 |
# | **M07** | Independent datasets, smaller data regimes, behaviour mixtures, MDP stress tests (transitions, initial-state distribution, admissible sets) with ranking persistence, severity/support subgroup diagnostics | 27–28 |
# | **M08** | Feature provenance/timing table built from the benchmark construction script (action derived from `input_4hourly` and `max_dose_vaso`, which are also clustering features), finite-tensor assertions, representation ablation | 4, 29 |
# | **M09** | Strict policy validation (repair is opt-in), conditioning + residual checks, length/occupancy agreement, absorption and return-bound checks, expected violation count and first-violation probability, statewise support failure, fail-loud value iteration, analytic toy tests, termination/mean semantics tested against the pinned environment, action codec verified against the environment mapping, Monte-Carlo cross-check | 8, 14, 20 |
# | **M10** | `terminated`, `truncated`, and local `cutoff` stored separately; stepping through the intended wrapper; bootstrapping across censoring boundaries; full trajectory validation; short-cutoff test | 5 |
# | **m04–m07** | Pearson r vs Spearman ρ labelled; Matplotlib-version-safe boxplot; pinned benchmark + recorded runtime lock; benchmark-support terminology; execution log with execution counts, timings, versions, GPU | throughout |
#
# **Scope statement.** Exact evaluation removes rollout sampling error *within the
# fixed ICU-Sepsis MDP*. It does not establish causal treatment effects, robustness to another hospital,
# or clinical safety. "Inadmissible" means *outside the released benchmark-support mask*, not medically unsafe.
#
# ### Cross-environment extension (Part II): what it asks and what is verified
#
# Thesis: **unsupported actions are a modelling choice.** An ICU-style benchmark (hospital data → count threshold
# τ → admissible mask → estimated rows for supported pairs + a completion rule for the rest) must decide what an
# unsupported action does; ICU-Sepsis offers the mean rule and termination. Part II measures what that choice does to
# violation costs, optima and method rankings, and — on simulators with known ground truth — whether benchmark
# rankings reproduce true-system rankings.
#
# | Research question | Environments | What is verified or measured | Section · switch |
# |---|---|---|---|
# | RQ1: formal semantics of unsupported actions (propositions P1–P7: projection invariance, optimum under convex imputation, termination gap, exact error decomposition, semantics invariance of admissible policies, hard-masked CQL defect, exact categorical diffusion) | ICU-Sepsis v2, exact Gumbel-max sepsis simulator, Garnet harm family, 20 random absorbing MDPs | per proposition: P1–P3 and P5 on ICU-Sepsis (P1/P2 where the released mean-rule identity holds), the draw-0 Gumbel and Garnet instances and 20 random absorbing MDPs; P4 on the Gumbel and Garnet instances and the random MDPs (no ground truth for ICU-Sepsis); P7 on ICU-Sepsis and the draw-0 benchmark instances; P6 by the Part I gradient tests (Section 13) and closed-form checks. Identities to 1e-9, bounds checked on every learned policy; by-construction facts reported, not counted | II.5 · `xenv_theory` |
# | Exact constrained optima | ICU-Sepsis (mean, terminate), Gumbel benchmark and true system, Garnet true systems | CMDP-LP frontier over an expected-violation budget κ; κ = 0 and κ = ∞ endpoints equal the exact constrained / unconstrained optima to 1e-7 | II.5 · `xenv_frontiers` |
# | RQ2: benchmark validity | Gumbel-max simulator (Oberst & Sontag, 2019; exact enumeration of the published code) across hospital-data size, τ and physician ε, plus an oracle construction; Garnet absorbing CMDPs with harm *h* of unsupported actions | Kendall τ-b rank fidelity, top-1 agreement and selection regret between benchmark and true values; exact error decomposition (completion semantics vs estimation / coverage); preregistered H1 and H3 | II.3, II.6–II.7 · `xenv_gumbel`, `xenv_garnet` |
# | RQ3: what constraint-aware methods buy over admissible projection | all three families (ICU-Sepsis deploy target: terminate rule) | 20 tabular methods with native / masked / projected variants (BC, CQL, IQL, BCQ, VI-LCB, MBS-QI, SPIBB, Lagrangian, CMDP-LP, SfBC, IDQL, FISOR-style, exact categorical diffusion); IQM of normalised scores with stratified bootstrap CIs, performance profiles, probability of improvement, preregistered H2 (sign-flip, Holm, TOST), default vs FQE vs oracle hyper-parameter selection | II.4, II.6–II.7 · `xenv_icu` (+ the two above) |
# | Implementation checks | ICU-Sepsis; Gumbel-max simulator | the generic evaluator equals the Part I evaluator to 1e-12; enumerated Gumbel rows match a literal port of the stochastic simulator step (Monte Carlo, chi-square); exact tabular learners agree with the Part I SGD learners | II.3, II.6 · `xenv_tabular_crosscheck` |
#
# The Part II experimental unit is the **dataset draw** (for Gumbel / Garnet a full benchmark-construction
# replicate); its learners are deterministic given data. Part II keeps its own result frames and never adds rows to
# the Part I registry.
#
# ### Version 3 (Section II.8): what this version adds
#
# v3 adds a bracketing theorem with certified
# comparisons, and tests outside the designs of II.3–II.6. Every function of Part I and of Sections II.1–II.7 is
# unchanged, so all identity-checked caches stay valid. II.8 has its own code hashes and cache keys (`v3/`,
# `v3boot/`, `v3harm/`).
#
# | Switch (default on) | Section | What it adds | Main outputs |
# |---|---|---|---|
# | `xenv_v3_physicians` | II.8b, II.8f | Gumbel cells with physicians that are not near-optimal: a diabetes-blind (confounded) one and an antibiotic-withholding one | `xenv/v3_results_all.csv.gz` |
# | `xenv_v3_aggregated` | II.8c, II.8f | Gumbel benchmarks on aggregated observation states with diabetes hidden (and glucose dropped in a second cell): state aggregation plus hidden confounding | `xenv/v3_results_all.csv.gz` |
# | `xenv_v3_garnet_random` | II.8d, II.8f | Garnet cells whose unsupported true rows are independent random rows, so no hidden benefit holds or fails at random | `xenv/v3_results_all.csv.gz` |
# | `xenv_v3_bootstrap` | II.8g | estimation-aware intervals from B = 10 bootstrap resamples of the hospital data, on 7 II.6 cells | `xenv/v3_bootstrap_*` |
# | `xenv_v3_icu_harm` | II.8h | ICU-Sepsis harm curves J(λ), λ ∈ [0, 1], for the Part I main / reference policies (seed 0) and the Part II draw-0 defaults | `xenv/v3_icu_harm_*` |
# | (any of the above) | II.8i | bracketing coverage (true rows and benchmark), certified comparisons, rank fidelity, bootstrap coverage | `xenv/v3_bracketing*.csv`, `xenv/v3_certified*.csv`, `xenv/v3_rank_fidelity*.csv`, `xenv/v3_bootstrap_coverage.csv`, `figures/xenv/fig_v3_*` |
#
# The release map (Section 33), the gate (Section 34: switch-aware, closure rows REV-1 … REV-3) and the hand-off
# (Section 35) cover every II.8 output. II.8 is exploratory. It is not part of `PREREG_PART2`, and its rows never
# enter `XRESULTS` or the preregistered tests H1–H3.

# %% [markdown]
# ## How to run this notebook on Google Colab
#
# 1. **Runtime → Change runtime type → GPU** (T4 or better). CPU works but is slow for the neural parts.
# 2. Run all cells top to bottom (**Runtime → Run all**). Google Drive is mounted so that every checkpoint
#    survives a disconnect. Re-running after a disconnect **resumes**: finished runs are re-used only when their
#    stored run identity matches exactly; interrupted neural runs resume from their last resumable checkpoint.
# 3. `RUN_MODE = "smoke"` (Section 1, or `SEPSIS_RUN_MODE`) runs a short debugging pass of every code path in
#    a separate output directory; `RUN_MODE = "paper"` runs the full protocol. `COMPUTE_PROFILE` sets the scale
#    of a paper run: `"full"` (default, used for the paper: 10 main seeds, 8 replication datasets × 2 training
#    seeds, 50 / 30 / 30 Part II draws per ICU / Gumbel / Garnet cell) or `"lite"` (5 seeds, 5 × 2 replication,
#    20 / 15 / 15 draws; its two-sided seed- and dataset-level tests cannot reject at 0.05).
# 4. Experiment blocks can be switched off in `EXPERIMENTS` (Section 1) or through `SEPSIS_EXPERIMENTS` (JSON;
#    unknown keys raise) and run in a later session; the identity-checked cache makes this safe. In the supplied
#    v4 notebook the six `xenv_v4_*` switches of Section II.9 are on by default; the final run switched II.9d,
#    II.9f and II.9g off.
# 5. **Clean run:** use a fresh `ROOT` (`SEPSIS_ROOT`) or set `FORCE_RETRAIN = True` (a forced run writes
#    `checkpoints/force_retrain_token.json`, so it survives disconnects; delete that file to start another forced
#    run). **Artifact replay:** keep `ROOT` and set `ALLOW_TRAINING = False`; a missing artifact then raises, and
#    Section 32 compares the replayed results with the previous session's exports. Checkpoint identities do not
#    depend on the accelerator, so GPU-trained artifacts can be replayed on a CPU (near-tie greedy actions may
#    differ).
# 6. If the original study's run directory exists at `SEPSIS_LEGACY_ROOT` (default
#    `/content/drive/MyDrive/sepsis_fresh_colab`), Section 21 replays its exported policies and checkpoints
#    read-only; otherwise Section 21 is skipped.
#
# **Compute.** The final run that produced the paper's numbers (`paper` + `full`, Parts I and II including
# II.9b, II.9c and II.9e) ran once, top to bottom, in one session from an empty run folder on a workstation
# with one NVIDIA RTX 4070 Ti SUPER, and logged 49.1 h: 22.6 h for Part I and 26.5 h for Part II, of which
# about 14 h were the II.9e sensitivity cells. The elapsed time of every cell is logged in
# `logs/execution_log.jsonl` and summarised in `tables/compute_summary.csv`.

# %% [markdown]
# ## 0. Environment setup — pinned benchmark, recorded scientific stack

# %%
# ============================================================
# 0. Environment setup
# ============================================================
# The benchmark is pinned exactly. The Colab-provided scientific stack (NumPy, pandas, SciPy,
# Matplotlib, PyTorch) is NOT reinstalled, because replacing it forces a runtime restart and can
# break CUDA wheels; instead its exact versions are checked against tested minimums and written to
# an environment lock file in Section 33 (audit m05).
import importlib
import importlib.util
import os
import subprocess
import sys

# Deterministic cuBLAS workspace: must be set before CUDA is initialised (audit C03).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PINNED_BENCHMARK = {"icu-sepsis": "2.0.1"}
MINIMUM_VERSIONS = {
    "numpy": "1.26",
    "pandas": "2.0",
    "scipy": "1.10",
    "matplotlib": "3.7",
    "torch": "2.1",
    "gymnasium": "0.28.1",
    "tqdm": "4.66",
}


def installed_version(dist_name: str):
    try:
        from importlib.metadata import version
        return version(dist_name)
    except Exception:
        return None


to_install = []
if installed_version("icu-sepsis") != PINNED_BENCHMARK["icu-sepsis"]:
    to_install.append(f"icu-sepsis=={PINNED_BENCHMARK['icu-sepsis']}")
for dist, minimum in [("gymnasium", MINIMUM_VERSIONS["gymnasium"]), ("tqdm", MINIMUM_VERSIONS["tqdm"])]:
    if installed_version(dist) is None:
        to_install.append(f"{dist}>={minimum}")

if to_install:
    print("Installing:", to_install)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
    importlib.invalidate_caches()
else:
    print("Benchmark and helper packages already present.")


def _version_tuple(v: str):
    out = []
    for part in str(v).split("+")[0].split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


version_report = {}
for dist, minimum in MINIMUM_VERSIONS.items():
    v = installed_version(dist)
    version_report[dist] = v
    if v is None:
        raise ImportError(f"Required package {dist} is not installed.")
    if _version_tuple(v) < _version_tuple(minimum):
        raise ImportError(f"{dist}=={v} is older than the tested minimum {minimum}.")
version_report["icu-sepsis"] = installed_version("icu-sepsis")
assert version_report["icu-sepsis"] == PINNED_BENCHMARK["icu-sepsis"], version_report
print("Runtime versions:", version_report)
