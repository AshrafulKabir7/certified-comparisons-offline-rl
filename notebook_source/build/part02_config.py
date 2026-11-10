# %% [markdown]
# ## 1. Imports, configuration, run modes, experiment switches, preregistration, execution log
#
# * `RUN_MODE = "paper"` is the full protocol; `"smoke"` is a fast debugging pass written to a separate
#   output directory so smoke checkpoints can never be mistaken for paper checkpoints (audit C03).
# * `COMPUTE_PROFILE` (environment variable `SEPSIS_COMPUTE_PROFILE`) sets the replication scale of a paper run.
#   `"full"` (default): 10 main training seeds (the smallest two-sided exact sign-flip p is
#   2/2¹⁰ ≈ 0.002, so the primary test and the Holm family can reject at 0.05), 8 independent replication datasets × 2
#   training seeds (dataset-level two-sided minimum 2/2⁸ ≈ 0.008), and 50 / 30 / 30 Part II draws per ICU / Gumbel /
#   Garnet cell. `"lite"`: 5 seeds, 5 × 2 replication, 20 / 15 / 15 draws — a cheaper run whose two-sided seed-level
#   and dataset-level tests **cannot** reject (minimum p 0.0625); use it for development, not for the paper. Both
#   profiles share `ROOT`: Part I run identities do not contain the seed list, so a later full run re-uses every lite
#   Part I checkpoint. Smoke mode keeps its own small overrides.
# * `EXPERIMENTS` switches every block on or off (overrides: `SEPSIS_EXPERIMENTS` as a JSON object; unknown keys
#   raise). The `xenv_*` switches control Part II and allow splitting the work across Colab sessions.
# * `PREREG` declares the primary comparison, endpoint, direction, comparison family, and practical
#   effect size **before** confirmatory runs (audit M03/M04). Edit it before the confirmatory study and
#   do not edit it after looking at results; its hash is stored in every results manifest.

# %%
# ============================================================
# 1. Imports, configuration and run bookkeeping
# ============================================================
import copy
import dataclasses
import gc
import hashlib
import inspect
import json
import math
import platform
import random
import shutil
import time
import urllib.request
import warnings
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import icu_sepsis  # noqa: F401  (registers the Sepsis/* environments)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange

try:
    from IPython.display import display
except Exception:  # pragma: no cover - plain Python fallback
    def display(obj):
        print(obj)

warnings.filterwarnings("once", message=".*does not have a deterministic implementation.*")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

# ---------------------------
# Run mode and global switches
# ---------------------------
RUN_MODE = os.environ.get("SEPSIS_RUN_MODE", "paper").strip().lower()   # "paper" | "smoke"
assert RUN_MODE in {"paper", "smoke"}, RUN_MODE
COMPUTE_PROFILE = os.environ.get("SEPSIS_COMPUTE_PROFILE", "full").strip().lower()   # "full" | "lite"
assert COMPUTE_PROFILE in {"full", "lite"}, COMPUTE_PROFILE

USE_GOOGLE_DRIVE = True        # persist all artifacts on Drive (recommended on Colab)
FORCE_REBUILD_DATA = False     # rebuild datasets even if an identity-matched file exists
FORCE_RETRAIN = False          # ignore identity-matched checkpoints and retrain
ALLOW_TRAINING = True          # False = artifact-replay mode: missing checkpoints raise instead of training
VERIFY_PROVENANCE_ONLINE = True  # fetch the benchmark construction scripts (parsed as text, never executed)
RELEASE_INCLUDE_WEIGHTS = True   # copy checkpoint weights into the release directory
RELEASE_INCLUDE_DATASET = True   # copy frozen datasets into the release (check the benchmark terms first)
SHOW_PROGRESS = True

# Every experiment block can be switched off and run in a later session (identity-checked caching).
EXPERIMENTS = {
    "legacy_artifact_replay": True,      # Phase 2: replay the original run's Drive artifacts read-only
    "reproducibility_tests": True,       # C03 closure tests
    "env_semantics_tests": True,         # M09: termination / mean semantics against the pinned env
    "monte_carlo_crosscheck": True,      # M09: small Monte-Carlo cross-check of exact evaluation
    "main_tabular": True,
    "main_neural_critics": True,
    "main_policy_nets": True,            # M06: matched-capacity direct categorical baselines
    "main_diffusion": True,
    "inference_ablations": True,         # Tables 6-7 + M06 masking/guidance/initialisation variants
    "critic_stability_ablations": True,  # C02 design/TD-row/loss/alpha/clipping ablations
    "include_discounted_ablation": False,  # gamma=0.99 changes the estimand; separately labelled if enabled
    "soft_sensitivity": True,            # M05 tolerance / temperature sensitivity
    "diffusion_schedule_ablation": True, # M06 schedule length (retrains the denoiser)
    "support_estimation": True,          # M02 estimated-support experiment
    "selection_protocol": True,          # M04 episode split + FQE selector validation
    "replication": True,                 # M03/M07 independent datasets x training seeds
    "replication_include_neural": True,  # include repaired critic + diffusion in the replication study
    "data_regimes": True,                # M07 smaller datasets + behaviour mixtures
    "mdp_stress_tests": True,            # M07 transition / initial-state / admissible-set stress tests
    "representation_ablation": True,     # M08 treatment-feature removal
    "release_package": True,             # C04 release directory + zip
    # Part II (cross-environment exact study, Sections II.1-II.7).
    "xenv_theory": True,                 # II.5 propositions P1-P7, numerically verified
    "xenv_frontiers": True,              # II.5 exact constrained (CMDP-LP) frontiers
    "xenv_icu": True,                    # II.6 ICU-Sepsis user-dataset draws
    "xenv_gumbel": True,                 # II.3/II.6 exact Gumbel-max sepsis simulator cells
    "xenv_garnet": True,                 # II.3/II.6 Garnet harm family
    "xenv_tabular_crosscheck": True,     # II.6 exact tabular learners vs Part I SGD learners (needs main_tabular)
    # Part II revision experiments (Section II.8): new cells with their own cache identities (II.6 draws unaffected).
    "xenv_v3_physicians": True,          # II.8 Gumbel physicians that are not near-optimal (confounded, withhold-abx)
    "xenv_v3_aggregated": True,          # II.8 Gumbel benchmarks on aggregated observation states (diabetes hidden)
    "xenv_v3_garnet_random": True,       # II.8 Garnet with independent random true rows for unsupported actions
    "xenv_v3_bootstrap": True,           # II.8g bootstrap-ensemble intervals on II.6 cells (needs xenv_gumbel / xenv_garnet)
    "xenv_v3_icu_harm": True,            # II.8h ICU-Sepsis harm curves J(lambda), Part I seed 0 + Part II draw 0
    # Part II second-revision experiments (Section II.9): new functions and cache keys (II.6 / II.8 draws unaffected).
    "xenv_v4_revision_robust": True,     # II.9b NHB-free certificates, pair types, kappa, II.8g all-policy coverage
    "xenv_v4_confidence": True,          # II.9c L1 confidence-set certificates on the true state space
    "xenv_v4_bootstrap": True,           # II.9d bootstrap B = 200, fixed mask and re-estimated construction
    "xenv_v4_sensitivity": True,         # II.9e timeout hazard, dead-state rule, softer physicians, abstractions
    "xenv_v4_lift": True,                # II.9f physician / uniform lifts and the range of J_true over all lifts
    "xenv_v4_clean_check": True,         # II.9g clean rerun of the Part II draws into an empty cache (ALLOW_TRAINING)
}
if os.environ.get("SEPSIS_EXPERIMENTS"):
    _exp_override = json.loads(os.environ["SEPSIS_EXPERIMENTS"])
    _exp_unknown = sorted(set(_exp_override) - set(EXPERIMENTS))
    assert not _exp_unknown, f"Unknown experiment switches in SEPSIS_EXPERIMENTS: {_exp_unknown}"
    assert all(isinstance(v, bool) for v in _exp_override.values()), "SEPSIS_EXPERIMENTS values must be JSON true/false."
    EXPERIMENTS.update(_exp_override)

# ---------------------------
# Output paths
# ---------------------------
IN_COLAB = importlib.util.find_spec("google.colab") is not None
if os.environ.get("SEPSIS_ROOT"):
    ROOT_BASE = Path(os.environ["SEPSIS_ROOT"])
elif IN_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    ROOT_BASE = Path("/content/drive/MyDrive/sepsis_revised_colab")
elif IN_COLAB:
    ROOT_BASE = Path("/content/sepsis_revised_colab")
else:
    ROOT_BASE = Path.cwd() / "sepsis_revised_outputs"

# Separate directory per run mode: a smoke checkpoint can never be loaded by a paper run (C03).
ROOT = ROOT_BASE / RUN_MODE
# The ORIGINAL run directory is only ever read (legacy replay, Section 21); nothing is written there.
LEGACY_ROOT = Path(os.environ.get("SEPSIS_LEGACY_ROOT", "/content/drive/MyDrive/sepsis_fresh_colab"))

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
POLICY_DIR = ROOT / "policies"
LOG_DIR = ROOT / "logs"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
RELEASE_DIR = ROOT / "release"
for p in [DATA_DIR, CKPT_DIR, POLICY_DIR, LOG_DIR, TABLE_DIR, FIG_DIR, RELEASE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Configuration
# ---------------------------
@dataclass
class Config:
    # Matched training seeds (main study) and the smaller seed set used for ablations. Ten seeds make the
    # two-sided exact sign-flip test able to reject (minimum p = 2/2^10); the "lite" profile uses 0-4.
    seeds: Tuple[int, ...] = tuple(range(10))
    ablation_seeds: Tuple[int, ...] = (0, 1, 2)

    # Benchmark / frozen offline dataset.
    dataset_seed: int = 2026
    dataset_episodes: int = 10_000
    max_episode_steps: int = 100       # local cutoff (censoring); the benchmark TimeLimit is 500
    behavior: str = "clinician"
    gamma: float = 1.0                 # training discount (evaluation is always the undiscounted return)

    # Generic optimisation.
    batch_size: int = 256
    hidden_dim: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 10.0
    target_tau: float = 0.005
    log_every: int = 500
    diag_every: int = 5_000            # full-buffer Bellman residual + full Q-table diagnostics
    resume_every: int = 5_000          # resumable-checkpoint interval for neural trainers

    # Tabular baselines.
    tab_cql_steps: int = 25_000
    tab_cql_lr: float = 3e-3
    tab_cql_alpha: float = 1.0
    iql_steps: int = 25_000
    iql_lr: float = 3e-3
    iql_expectile: float = 0.7
    iql_beta: float = 3.0
    iql_max_weight: float = 100.0
    bcq_steps: int = 25_000
    bcq_lr: float = 3e-3
    bcq_threshold: float = 0.3         # discrete-BCQ relative behaviour-probability threshold

    # Deep CQL / AM-CQL.
    deep_steps: int = 30_000
    cql_alpha: float = 1.0
    cql_temperature: float = 1.0

    # Soft AM-CQL.
    soft_initial_lambda: float = 1.0
    soft_dual_lr: float = 1e-2
    soft_target_inadmissible_mass: float = 0.0
    soft_lambda_max: float = 100.0
    soft_policy_temperature: float = 1.0

    # Categorical diffusion behaviour prior.
    diffusion_steps: int = 20_000
    diffusion_lr: float = 3e-4
    diffusion_T: int = 20
    diffusion_beta_start: float = 0.02
    diffusion_beta_end: float = 0.25
    diffusion_action_embed_dim: int = 32
    diffusion_time_embed_dim: int = 32
    diffusion_guidance_scale: float = 1.5
    diffusion_deterministic_policy: bool = True
    guidance_grid: Tuple[float, ...] = (0.0, 0.5, 1.5, 3.0)
    diffusion_T_grid: Tuple[int, ...] = (5, 50)

    # Direct categorical policies (matched budget with the diffusion prior).
    policy_net_steps: int = 20_000
    policy_net_lr: float = 3e-4
    aw_beta: float = 3.0
    aw_max_weight: float = 100.0

    # Statistics.
    bootstrap_samples: int = 10_000
    significance_alpha: float = 0.05

    # Robustness / selection. The dataset is the replication unit: 8 independent datasets x 2 nested training
    # seeds make a dataset-level two-sided test attainable (minimum p = 2/2^8); 2026 is the main dataset.
    replication_dataset_seeds: Tuple[int, ...] = tuple(range(2026, 2034))
    replication_train_seeds: Tuple[int, ...] = (0, 1)
    data_size_regimes: Tuple[int, ...] = (1_000, 3_000)
    behavior_mixtures: Tuple[str, ...] = ("clinician+uniform:0.25", "clinician+uniform_admissible:0.25")
    regime_train_seeds: Tuple[int, ...] = (0, 1, 2)
    regime_dataset_draws: int = 2      # independent dataset draws per reduced size (not prefixes of the main data)
    stress_seeds: Tuple[int, ...] = (0, 1, 2)
    mc_episodes: int = 10_000
    dev_fraction: float = 0.2
    fresh_eval_dataset_seed: int = 3030


def apply_profile(cfg: Config, profile: str) -> Config:
    """Replication scale of a paper run: 'full' keeps the defaults; 'lite' halves the seed and dataset counts."""
    if profile == "lite":
        cfg.seeds = (0, 1, 2, 3, 4)
        cfg.replication_dataset_seeds = (2026, 2027, 2028, 2029, 2030)
        cfg.replication_train_seeds = (0, 1)
    elif profile != "full":
        raise ValueError(f"unknown compute profile {profile!r}")
    return cfg


def apply_run_mode(cfg: Config, mode: str) -> Config:
    if mode == "smoke":
        cfg.seeds = (0, 1)
        cfg.ablation_seeds = (0, 1)
        cfg.dataset_episodes = 400
        cfg.tab_cql_steps = cfg.iql_steps = cfg.bcq_steps = 300
        cfg.deep_steps = 300
        cfg.diffusion_steps = 300
        cfg.policy_net_steps = 300
        cfg.log_every = 50
        cfg.diag_every = 150
        cfg.resume_every = 100
        cfg.diffusion_T_grid = (5,)
        cfg.bootstrap_samples = 500
        cfg.replication_dataset_seeds = (2026, 2027)
        cfg.replication_train_seeds = (0, 1)
        cfg.data_size_regimes = (150,)
        cfg.behavior_mixtures = ("clinician+uniform:0.25",)
        cfg.regime_train_seeds = (0, 1)
        cfg.regime_dataset_draws = 1
        cfg.stress_seeds = (0,)
        cfg.mc_episodes = 400
    return cfg


# The profile is applied first; smoke mode then imposes its own small overrides.
CFG = apply_run_mode(apply_profile(Config(), COMPUTE_PROFILE), RUN_MODE)
# On the main dataset the replication study registers its runs in the "main" group (Section 27), so its training
# seeds must be main seeds; otherwise main-study pairs would become unbalanced across methods.
assert set(CFG.replication_train_seeds) <= set(CFG.seeds), (CFG.replication_train_seeds, CFG.seeds)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_determinism() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


configure_determinism()

# ---------------------------
# Preregistration (audit M03 / M04) — edit BEFORE confirmatory runs, never after seeing results.
# ---------------------------
PRIMARY_DIFFUSION = "Diffusion-AM-CQL [primary critic B]"
PREREG = {
    "status": "DRAFT - lock (set status to LOCKED with a date) before the confirmatory Phase-4 run",
    "primary_question": (
        "Does critic-guided categorical diffusion improve exact return over the mean-equivalent "
        "admissible projection of tabular IQL when both use the same released benchmark-support mask?"
    ),
    "primary_treatment": PRIMARY_DIFFUSION,
    "primary_comparator": "Tabular IQL [projected]",
    "primary_endpoint": {"metric": "return", "rule": "terminate"},
    "alternative": "two-sided",
    "alpha": 0.05,
    "practical_effect": 0.005,       # absolute exact-return difference (0.5 percentage points)
    "equivalence_margin": 0.005,     # TOST margin (same scale)
    "primary_hard_critic": "hard_admrows",  # repaired design B guides the primary diffusion policy
    "secondary_comparators": [
        "Tabular IQL [masked greedy]",
        "Behavior Cloning [masked greedy]",
        "Tabular CQL [masked greedy]",
        "Masked critic ensemble [critic B + tabular CQL]",
        "Neural BC prior x critic [critic B + tabular CQL]",
        "AM-CQL (Hard, design B: admissible-row CQL)",
        "AM-CQL (Hard, design A: full-action CQL)",
        "Deep CQL [masked greedy]",
        "Diffusion-AM-CQL [legacy critic]",
    ],
    "secondary_rules": ["mean", "terminate"],
    "multiplicity": ("Holm over the distinct hypotheses of the secondary family (comparator x rule). When the treatment "
                     "and a comparator both have zero inadmissible mass, their mean- and terminate-rule returns coincide "
                     "by construction, so that comparison is counted once."),
    "experimental_unit_main": (
        f"training seed, conditional on one frozen dataset; {len(CFG.seeds)} seeds {list(CFG.seeds)} "
        f"(compute profile '{COMPUTE_PROFILE}', run mode '{RUN_MODE}'; the full paper profile uses seeds 0-9, "
        "so the smallest two-sided exact sign-flip p is 2/2^10)"),
    "experimental_unit_hierarchical": (
        f"dataset-generation seed, training seeds nested within dataset; {len(CFG.replication_dataset_seeds)} datasets "
        f"{list(CFG.replication_dataset_seeds)} x {len(CFG.replication_train_seeds)} training seeds "
        f"{list(CFG.replication_train_seeds)} (full profile: 8 x 2, smallest dataset-level two-sided p 2/2^8)"),
    "selection": (
        "Main-study configurations are fixed a priori (Table 3 settings). Inference sweeps are exploratory "
        "and never replace the declared configuration."
    ),
}

# ---------------------------
# Execution log (audit m07): execution counts, timings, success flags, section names
# ---------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ExecutionLog:
    def __init__(self, path: Path):
        self.path = path
        self.section = "startup"
        self._t0 = None
        self._start = None
        self.session_start = utc_now()

    def mark(self, section: str) -> None:
        self.section = section
        print(f"\n=== {section} ===")

    def pre(self, *args, **kwargs) -> None:
        self._t0 = time.perf_counter()
        self._start = utc_now()

    def post(self, result=None, *args, **kwargs) -> None:
        rec = {
            "session_start": self.session_start,
            "section": self.section,
            "execution_count": getattr(result, "execution_count", None),
            "success": bool(getattr(result, "success", True)),
            "error": (repr(getattr(result, "error_in_exec", None))
                      if getattr(result, "error_in_exec", None) is not None else None),
            "start_utc": self._start,
            "elapsed_s": (time.perf_counter() - self._t0) if self._t0 is not None else None,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass


EXECLOG = ExecutionLog(LOG_DIR / "execution_log.jsonl")
try:
    _ip = get_ipython()  # noqa: F821
    for _ev in ("pre_run_cell", "post_run_cell"):
        for _cb in list(_ip.events.callbacks.get(_ev, [])):
            if getattr(_cb, "__self__", None).__class__.__name__ == "ExecutionLog":
                _ip.events.unregister(_ev, _cb)
    _ip.events.register("pre_run_cell", EXECLOG.pre)
    _ip.events.register("post_run_cell", EXECLOG.post)
except Exception:
    _ip = None

GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
RUNTIME_INFO = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": GPU_NAME,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": installed_version("scipy"),
    "matplotlib": matplotlib.__version__,
    "gymnasium": gym.__version__,
    "icu_sepsis": installed_version("icu-sepsis"),
    "in_colab": IN_COLAB,
    "session_start_utc": EXECLOG.session_start,
    "run_mode": RUN_MODE,
    "compute_profile": COMPUTE_PROFILE,
}
# Replication design actually used by this session (read by Sections 30, 34 and the paper hand-off).
RUN_PROFILE = {
    "run_mode": RUN_MODE,
    "compute_profile": COMPUTE_PROFILE,
    "main_seeds": list(CFG.seeds),
    "ablation_seeds": list(CFG.ablation_seeds),
    "replication_dataset_seeds": list(CFG.replication_dataset_seeds),
    "replication_train_seeds": list(CFG.replication_train_seeds),
    "min_two_sided_signflip_p_main_seeds": 2.0 / 2 ** len(CFG.seeds),
    "min_two_sided_signflip_p_dataset_level": 2.0 / 2 ** len(CFG.replication_dataset_seeds),
    "paper_scale": bool(RUN_MODE == "paper" and COMPUTE_PROFILE == "full"),
}

print("Run mode:", RUN_MODE, "| compute profile:", COMPUTE_PROFILE,
      f"| main seeds: {len(CFG.seeds)} | replication: {len(CFG.replication_dataset_seeds)} datasets x "
      f"{len(CFG.replication_train_seeds)} training seeds")
print("Output root:", ROOT)
print("Device:", DEVICE, "| GPU:", GPU_NAME)
print("Enabled experiments:", [k for k, v in EXPERIMENTS.items() if v])


def _finite_json(o):
    if isinstance(o, dict):
        return {k: _finite_json(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite_json(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None   # strict JSON: NaN / +-Infinity -> null
    return o


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _finite_json(json.loads(json.dumps(obj, default=_json_default)))
    path.write_text(json.dumps(clean, indent=2, allow_nan=False), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if dataclasses.is_dataclass(o):
        return asdict(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


save_json(asdict(CFG), TABLE_DIR / "fresh_config.json")
save_json(PREREG, TABLE_DIR / "preregistration.json")
save_json(EXPERIMENTS, TABLE_DIR / "experiment_switches.json")
save_json(RUN_PROFILE, TABLE_DIR / "run_profile.json")
save_json(RUNTIME_INFO, LOG_DIR / f"runtime_{EXECLOG.session_start.replace(':', '')}.json")
