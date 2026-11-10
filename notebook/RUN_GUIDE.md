# Running the notebook

**Notebook:** `Sepsis_Offline_RL_v4.ipynb`

**Platforms:** Google Colab with one GPU, or a local machine with one CUDA GPU. The paper's final run used a local
workstation (Windows, RTX 4070 Ti SUPER, Python 3.13) with `local_setup_cell.py` as its first cell. Outside Colab,
without `SEPSIS_ROOT`, the notebook writes to `./sepsis_revised_outputs/<run mode>/`.

The notebook installs `icu-sepsis==2.0.1` itself. It checks the scientific stack against minimum
versions and records the exact versions in `environment_lock.txt`. The final run's versions are in
`results/release_metadata/environment_lock.txt`.

## Controls

Set these environment variables in a cell **before Section 1**, then run all cells:

```python
import os, json
os.environ["SEPSIS_RUN_MODE"] = "paper"          # "paper" (results of the paper) or "smoke" (minutes; separate folder)
os.environ["SEPSIS_COMPUTE_PROFILE"] = "full"    # "full" (paper: 10 main seeds, 8 x 2 replication) or "lite"
os.environ["SEPSIS_ROOT"] = "<new, empty output folder>"   # optional; default shown in Section 1
os.environ["SEPSIS_EXPERIMENTS"] = json.dumps({   # the switches of the final run
    "xenv_v4_bootstrap": False, "xenv_v4_lift": False, "xenv_v4_clean_check": False})
```

In Colab, mount Google Drive first (`from google.colab import drive; drive.mount("/content/drive")`) and point
`SEPSIS_ROOT` at a Drive folder, so that a disconnect does not lose finished work.

On a local machine, use `local_setup_cell.py` as the first cell instead: edit its four settings (run folder, folder
of the earlier study's artefacts, CPU threads, optional sections). Its step 3 asserts that three files of the
earlier study exist under `LEGACY_ROOT`; they are not part of this supplement, so remove that check (the
`for _rel in (...)` loop) and point `LEGACY_ROOT` at any folder outside the run folder. Section 21 is then skipped,
as described below. It sets the variables above, sizes the thread
pools before NumPy and PyTorch load, checks the GPU and the package versions, and inserts an empty placeholder
`google` package so that Section 1's Colab detection returns False. On Windows, start Jupyter with `PYTHONUTF8=1`.

Section 1 also defines:

- `FORCE_RETRAIN`: ignore identity-matched checkpoints and retrain.
- `ALLOW_TRAINING`: when `False`, the notebook runs in artifact-replay mode, and a missing checkpoint raises an
  error instead of starting training.

Every result is cached under an identity: the content hashes of its data, configuration and code. A rerun into the
same folder therefore loads every finished model and Part II draw, and recomputes only what is missing or what
changed. A run that stops can be resumed by running all again.

## Reproducing the paper

| Goal | How | Time |
|---|---|---|
| Regenerate the paper's tables and numbers from the stored results | `bash paper_scripts/reproduce_paper_assets.sh`. No notebook is needed. | about 2 min on a CPU |
| **Run from an empty folder (as the final run)** | Point `SEPSIS_ROOT` at a new, empty folder and run all, with `SEPSIS_RUN_MODE="paper"`, `SEPSIS_COMPUTE_PROFILE="full"` and the switches above. | 49.1 h logged on an RTX 4070 Ti SUPER with 20 CPU threads: 22.6 h Part I (GPU training), 26.5 h Part II (CPU-bound; about 14 h of it the II.9e sensitivity cells). Resumable across sessions through checkpoints and the draw cache. |
| Quick functional check | `SEPSIS_RUN_MODE="smoke"`, run all. It writes to a separate `smoke` folder. | well under 1 h (estimate) |

The trained checkpoints and the draw cache are not supplied (size limit), so artifact-replay mode
(`ALLOW_TRAINING=False`) cannot be used with the supplied files; a rerun retrains every model.

## Sections

| Sections | Content | Status in the paper |
|---|---|---|
| 0–31 | Part I: ICU-Sepsis case study (training, exact evaluation, statistics) | reported (App. F) |
| II.1–II.7 | Part II: cross-environment exact study (Gumbel-max simulator, Garnet MDPs, ICU-Sepsis; 16 cells, 500 draws) | reported (Sections 4–5, App. C–E, G) |
| II.8 | Additional experiments: physicians, aggregated states, random-harm Garnet, bootstrap (B = 10), ICU-Sepsis harm curves | reported (App. H) |
| II.9b | NHB-free certificates, hidden-benefit (κ) sensitivity, pair types | run; Section 5 and App. I report the same quantities, recomputed from the stored draws by `paper_scripts/make_robustness.py` |
| II.9c, II.9e | Confidence-set certificates; sensitivity cells | reported (App. J) |
| II.9d, II.9f, II.9g | B = 200 bootstrap with mask re-estimation; lift sensitivity; cache-free rerun check | switched off in the final run; not reported |
| 32–35 | Replay checks, release archive, final integrity gate, hand-off | run |

### Section II.9 switches

Each switch can be turned off through `SEPSIS_EXPERIMENTS`.

| Switch | Section | Content | Cost |
|---|---|---|---|
| `xenv_v4_revision_robust` | II.9b | NHB-free certificates, κ sensitivity, pair types and distinct policies for every cell; bootstrap coverage over all policies | < 1 min |
| `xenv_v4_confidence` | II.9c | Confidence-set certificates: Weissman L1 sets on the true rows, valid for all policies at once at scale 1.0; scales 0.3 and 0.1 are heuristic | 0.7 h in the final run (210 draws) |
| `xenv_v4_bootstrap` | II.9d | B = 200 bootstrap, with the mask, decision states, dead states and d0 re-estimated in every replicate | 4–8 h (estimate), off in the final run |
| `xenv_v4_sensitivity` | II.9e | Timeout η ∈ {0.02, 0.10}; dead-state rule `argmax`; physicians with ε ∈ {0.3, 0.5}; abstractions (glucose, k-means 50 and 150) | 13.9 h in the final run (510 draws) |
| `xenv_v4_lift` | II.9f | Physician and uniform policy lifts beside the default | about 2 h (estimate), off in the final run |
| `xenv_v4_clean_check` | II.9g | Reruns the 500 II.6 and 180 II.8 draws into an empty cache and compares them row by row | about 10.5 h at the final run's speed (the draws it repeats took 8.3 h and 2.2 h there), not resumable; redundant for a run from an empty folder |

Section II.9h writes `xenv/v4_overview.json` and `xenv/v4_stats_checks.json`. The final integrity gate (Section 34)
adds rows for II.9. It gates only on invariants and completion, never on coverage or error rates.

**Section 21** replays the earlier study's exported artefacts read-only, and only if they are at
`SEPSIS_LEGACY_ROOT`. They are not part of this supplement, so a rerun from it skips Section 21: it prints "Legacy
artifact directory not found", and the final gate marks the Legacy row "not run". The App. F.5 numbers come from the
stored `results/tables/legacy_replay.json`, `legacy_projection_controls_summary.csv` and the `LEGACY` rows of
`exact_results_all.csv`.

**Known limit of II.9c:** confidence sets need raw counts on unaggregated states, so they cover the simulator cells
only.
