# Supplementary material

**Which Comparisons Can a Clinical Offline RL Benchmark Support? Interval Reporting for Unsupported Actions**

About this release:

- Notebook-environment account metadata has been removed from every notebook.
- In markdown cells and code comments, local folder names and the authors' planning notes have been replaced.
  No line of code and no output has been changed.

Paper references in this README and in `TABLE_MAP.md` (such as "App. J" or "Table 1") refer to an earlier, longer version of the manuscript; the arXiv version is shorter and numbers its appendices and tables differently.
Inside the notebooks, "manuscript", "the original study" and "Table N" / "Figures N" refer to the report of the
earlier study audited in App. F. So do the release map `table_to_command_map.csv` and output names such as
`table1_reconciliation.csv`, `table6_*`, `table7_guidance_sweep.csv`, `statistics_historical_table5_one_sided.csv`
and `paper_fig*`.

## Quick start: regenerate the paper's tables and numbers (CPU, about 2 minutes)

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r paper_scripts/requirements.txt
PYTHON=python bash paper_scripts/reproduce_paper_assets.sh
```

The scripts run with `python -I`, which ignores user-site packages, so install into a virtual environment as above.

This runs no experiment. The scripts read three sources:

- the result files exported by the final run, in `results/`;
- the printed outputs of the final run, in `run_record/Sepsis_Offline_RL_v4_final_run.ipynb`;
- for Table 24 only, the facts files of the earlier run, in `paper_assets_expected/original_run/`.

From these they rebuild, in `regenerated/`:

- every generated table of the paper (all except Tables 5 and 25, which are written by hand in LaTeX; see
  `TABLE_MAP.md`);
- every data figure;
- the facts files that hold every number quoted in the text.

The driver then compares the output with `paper_assets_expected/`, which holds the files used in the paper.
Tables and facts must match. Figures are written for visual comparison only.

## Contents

| Path | Content |
|---|---|
| `notebook/Sepsis_Offline_RL_v4.ipynb` | The experiment notebook, outputs cleared. It runs in Google Colab or on a local machine with one CUDA GPU. |
| `notebook/RUN_GUIDE.md` | Run modes, switches, runtimes, and how to run it from an empty folder in Colab or locally. |
| `notebook/local_setup_cell.py` | The one cell that the final run added in front of the notebook (run folder, thread count, switches). |
| `notebook_source/` | Percent-format source files the notebook is assembled from. `python build_nb.py --out X.ipynb --py X.py` rebuilds it without executing anything. |
| `run_record/Sepsis_Offline_RL_v4_final_run.ipynb` | The executed final run, outputs kept. Every result in the paper comes from it. See `run_record/RUN_RECORDS.md`. |
| `run_record/earlier_run/` | Two executed notebooks of an earlier, multi-session run. Kept as history and for the replication in App. K; no number in the paper comes from them except that comparison. |
| `results/tables/` | Every CSV/JSON table that the final run exported: Part I at the top level, Part II (II.1–II.9) under `xenv/`. |
| `results/v3_tables/` | Tables of the additional experiments (Section II.8), the corrected cross-check and the compute totals, extracted from the final-run record by `paper_scripts/extract_v3.py`; four of them are then replaced by the run's full-precision CSV files by `use_exact_v3.py`. Used as the reference for the comparison. |
| `results/policies/` | The saved Part I policies, one file per method, variant and seed. |
| `results/data/` | Dataset manifests with content SHA-256, the frozen offline datasets sampled from the ICU-Sepsis MDP, and the enumerated Gumbel-max transition tensors. |
| `results/logs/` | Execution log of the final run (wall time of every executed cell), its runtime record, and per-model training and critic-diagnostic logs. |
| `results/release_metadata/` | Environment lock, input hashes, table-to-command map, artifact inventory, policy registry and checkpoint metadata, written by the release cell of the final run. `results/tables/audit_closure_checklist.csv` is the run's own export, shipped unchanged. Its IDs C01–C04, M01–M12 and m01–m08 index the corrections of the audit of the earlier study (App. F); its other rows (Legacy, MT-1–MT-5, REV-1–REV-9) are the notebook's own checks, and REV-6, REV-8 and REV-9 read "not run" because their switches were off in the final run. Its rows marked manuscript-only concern that study's write-up and have no notebook output; their "see Section 35" refers to a planning note that is not part of the shipped notebooks. |
| `paper_scripts/` | Scripts that build the paper's tables, figures and facts, plus the driver `reproduce_paper_assets.sh`. |
| `paper_assets_expected/` | The tables (`.tex`), figures and facts files of the paper. `original_run/` holds the facts files of the earlier run, which `make_repro_summary.py` compares with the final run (App. K, Table 24). |
| `TABLE_MAP.md` | Each table and figure of the paper mapped to its script, input files and notebook section. |
| `MANIFEST.sha256` | SHA-256 of every file in this folder. |

Not included, because of the 100 MB limit:

- the trained neural checkpoints (about 0.4 GB). Their identity metadata is in
  `results/release_metadata/checkpoint_metadata.csv`;
- the identity-addressed draw cache `xenv_cache/` (about 80 MB), which lets a rerun into the same folder reload the
  Part II draws instead of recomputing them.

Neither is part of this repository.

## The final run

Every result in the paper comes from one execution of the supplied notebook (App. K):

- run once, top to bottom, in one kernel session, starting from an empty run folder, so every model was trained and
  every Part II draw computed within the run;
- on a local workstation (Windows, one NVIDIA GeForce RTX 4070 Ti SUPER, Python 3.13, torch 2.11.0+cu130,
  numpy 2.1.3, pandas 2.2.3, scipy 1.16.3, gymnasium 1.3.0, icu-sepsis 2.0.1);
- 76 code cells executed in order, no error output; the final integrity gate passed;
- 49.1 h of logged cell time: 22.6 h for Part I and 26.5 h for Part II (Section 33 prints the total; it is stored as
  `compute` in `results/v3_tables/text_facts.json`).

The record differs from the supplied notebook only by one added first cell (`notebook/local_setup_cell.py`) and the
short markdown note above it; the anonymisation rewrites were applied identically to both files. The added cell sets the run folder, the folder of the earlier study's
artefacts that Section 21 replays, the number of CPU threads and the experiment switches, and it inserts an empty
placeholder `google` package so that the notebook's Colab detection returns False. Printed paths in the outputs name
the run folder on that machine.

**What ran:** Parts I and II, Sections 0–35 and II.1–II.9, with three exceptions. The switches turned off II.9d (the
B = 200 bootstrap with mask re-estimation), II.9f (policy-lift sensitivity) and II.9g (a cache-free rerun of the
Part II draws, which the empty run folder makes redundant). None of the three is reported in the paper. Section II.9c
(confidence-set certificates) and II.9e (sensitivity cells: softer physicians, timeout hazard, dead-state rule, state
abstractions) ran and are reported in App. J. The App. B.5 counts are computed by `paper_scripts/verify_bracket.py`
on random MDPs, not by the notebook.

**Section 21** replays the earlier study's exported artefacts read-only (App. F.5). The final run had them; they are
not part of this supplement, so a rerun from this folder skips Section 21 (it prints "Legacy artifact directory not
found" and the final gate marks the Legacy row "not run"). The App. F.5 numbers come from the stored
`results/tables/legacy_replay.json`, `legacy_projection_controls_summary.csv` and the `LEGACY` rows of
`exact_results_all.csv`.

## The earlier run (history)

Before the final run, the notebook ran in Google Colab on one T4 GPU over six sessions that resumed from
identity-checked caches. App. K compares it with the final run (Table 24): the pooled Gumbel results are identical,
and the 19 main-table Part I values (GPU-trained) differ by at most 0.0017.

During that run, an automated code assistant of the notebook environment edited three cells after errors (App. K,
notebook edit history): the FQE model-selection cell (Section 26), the ICU-Sepsis adapter (II.3) and the
exact-tabular versus SGD cross-check (II.6). The session-5 record in `run_record/earlier_run/` shows the edited
cells. The supplied notebook contains the corrected cells, and the final run executed them; the selection result and
the corrected cross-check (App. G.1, Table 16) come from the final run.

## External assets and licences

- **ICU-Sepsis benchmark:** Python package `icu-sepsis` 2.0.1 (MIT licence), installed by the notebook from PyPI and
  pinned with its asset hashes. Its authors built the MDP from the de-identified MIMIC-III database. Only the released
  MDP is used; the datasets here are sampled from it and contain no patient records.
- **Gumbel-max sepsis simulator:** `clinicalml/gumbel-max-scm` (MIT licence). The notebook enumerates the published
  step function of the four simulator files exactly. It pins both branches by commit and SHA-256 (`master` d62995d,
  `sim-v2` f923f77) and can re-verify the digests online; the files are fetched as text and never executed.
- **Python stack:** NumPy, pandas, SciPy, Matplotlib and PyTorch. Exact versions are in
  `results/release_metadata/environment_lock.txt` and `runtime.json`.

A licence for this code has not been chosen yet; please contact the authors before reusing it.
