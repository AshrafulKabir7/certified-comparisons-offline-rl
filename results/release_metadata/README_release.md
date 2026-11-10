# Release contents

Run mode: paper; created 2026-11-07T21:44:03+00:00.

* release/code/ — executed notebook (Colab export) and executed cell sources
* tables/ — every CSV/JSON behind the manuscript tables (see table_to_command_map.csv)
* figures/ — PNG + PDF figures
* policies/ — every evaluated policy (npz) per group and seed
* checkpoints/ — identity-addressed checkpoints (+ .identity.json metadata)
* data/ — dataset manifests (content SHA-256) and, if included, the frozen datasets; Part II exact Gumbel simulator matrices (xenv_gumbel_true_*.npz)
* logs/ — training and execution logs
* tables/xenv/, figures/xenv/ — Part II (cross-environment exact study) tables and figures; v3_* and fig_v3_* are the revision experiments of Section II.8 (REV-1 ... REV-3), v4_* and fig_v4_* the second-revision experiments of Section II.9 (REV-4 ... REV-9; status and headline numbers in tables/xenv/v4_overview.json)
* xenv_cache/ — identity-addressed Part II results per (cell, draw); replay with ALLOW_TRAINING=False
* tables/compute_summary.csv — logged wall time per section / session / part, with the GPU
* environment_lock.txt, runtime.json, artifact_inventory.csv, input_sha256.json

Compute profile: full (10 main seeds; 8 x 2 replication).

Clean training: a fresh ROOT (or FORCE_RETRAIN=True; delete checkpoints/force_retrain_token.json to start a new forced run).
Artifact replay: same ROOT, ALLOW_TRAINING=False. Checkpoints are device-independent in identity; CPU replay of GPU checkpoints is supported and labelled (near-tie greedy actions may differ).
