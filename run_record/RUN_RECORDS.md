# Run records

Executed notebooks with their outputs kept. What was removed or replaced:

- notebook-environment account metadata: per-cell execution info, output ids and widget state;
- in markdown cells and code comments, local folder names and the authors' planning notes.

No line of code and no output has been changed. The Section 34 outputs print the run's audit-closure checklist,
whose manuscript-only rows (C/M/m IDs of the audit of the earlier study, App. F) point to a planning note that is not
included; see `README.md`.

## `Sepsis_Offline_RL_v4_final_run.ipynb`: the final run

Every result in the paper comes from this run (App. K).

- One kernel session, started from an empty run folder; every model was trained and every Part II draw computed
  within the run.
- 76 code cells, execution counts 1 to 76 in order, no error output. The final integrity gate passed.
- Code cell 1 is `notebook/local_setup_cell.py` (run folder, thread count, experiment switches, Colab-detection
  placeholder). Code cells 2–76 equal the 75 code cells of `notebook/Sepsis_Offline_RL_v4.ipynb` line for line
  (checked when this folder was assembled).
- Switched off: II.9d, II.9f, II.9g (see `README.md`).

**Use by the paper scripts:** `paper_scripts/extract_v3.py` reads its printed tables and summary lines (Sections
II.6b, II.8, 26, 33, 34 and 35); its file name is recorded in `text_facts.json`.

## `earlier_run/`: two sessions of the earlier Colab run

Kept as history. App. K compares the earlier run with the final run; nothing else in the paper comes from these
records.

- `Sepsis_Offline_RL_v3_earlier_run_final_session.ipynb`: the sixth and last session of the earlier run. It reloaded
  Parts I and II.1–II.7 from identity-checked caches, recomputed the two outputs invalidated by the automated edits,
  and ran Section II.8. Its code predates Section II.9. The facts files built from it are in
  `paper_assets_expected/original_run/`.
- `Sepsis_Offline_RL_earlier_run_session5_with_automated_edits.ipynb`: session 5, with the three cells as an
  automated code assistant of the notebook environment left them (selection and adapter cells edited in session 3;
  cross-check cell changed in session 3 and rewritten in session 5). Its selection and cross-check outputs are
  invalid. The supplied notebook and the final run use the corrected cells.
