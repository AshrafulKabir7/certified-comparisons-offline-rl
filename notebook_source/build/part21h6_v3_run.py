# %% [markdown]
# ### II.8f Run the II.8 cells (and the prerequisites of II.8g–II.8h)
#
# The cell below runs every draw of every enabled II.8 cell (`X_V3_CELLS`) through `run_v3_cell_draw`, using
# `x_v3_run_cells` (II.8e). Each draw is loaded from `XCACHE` (key `v3/<cell>/draw<k>`) when its identity matches.
# Otherwise, if `ALLOW_TRAINING`, it is computed and cached. The table returned is always the one read back from the
# cache, so an interrupted, resumed run and an uninterrupted run give identical tables.
#
# | Output | Content |
# |---|---|
# | `XRESULTS_V3` · `xenv/v3_results_all.csv.gz` | one row per draw × policy × target, with exactly the columns of `XRESULTS` (`XRESULT_COLUMNS`) |
# | `XDRAW_META_V3` · `xenv/v3_draw_meta.csv` | construction and validation reports, record counts, reference solvers, invariant-check slacks and timings per draw |
# | `XV3_DRIVER_SUMMARY` · `xenv/v3_driver_summary.json` | code hash, cache use, the cross-draw checks of `x_check_results`, draws per cell and whether every configured draw is present |
#
# `XRESULTS_V3` is kept separate from `XRESULTS`. The II.7 statistics, the preregistered tests H1–H3 and the Part II
# gate use only the II.6 cells, which keeps the preregistered analysis unchanged. The II.8 rows enter the additional
# analyses (II.8i) and the II.8 gate of Section 34.
#
# **Prerequisites of II.8g–II.8h.** The bootstrap intervals (II.8g) and the ICU-Sepsis harm curves (II.8h) call
# `run_xmethods`, and II.8h evaluates on the ICU-Sepsis adapter `X_ICU`. When those switches are on but every switch
# of Sections II.3–II.7 and of II.8a–II.8e is off, the cell below runs the ICU-Sepsis adapter check and the method
# self-test itself, with the same code and output files as II.3a and II.4. The Section 34 gate requires the adapter
# check whenever an `xenv_*` switch is on.
#
# **Expected cost on Colab (CPU-bound).** These are estimates, not measurements of Colab. Local per-draw timings of
# the II.8 cells (harness, M1 CPU) were scaled by the Colab / local ratio measured on matched II.6 draws: about 1.8
# for Gumbel and 3.5 for Garnet, from the timings in `xenv_draw_meta.csv`. That gives about 20–25 s per physician
# draw, 10–12 s per aggregated draw and 4–5 s per random-row Garnet draw. With 30 draws per cell, the six cells take
# about 35–40 min, plus under a minute for the environment suite of II.8e. A cached draw reloads in about 3 s
# (instance rebuild and identity), so a re-run costs about 9 min.

# %%
# ============================================================
# II.8f II.8 driver: all II.8 cells and draws
# ============================================================
EXECLOG.mark("II.8 Revision experiments: driver")
XV3_EXTRA_SWITCHES = ("xenv_v3_bootstrap", "xenv_v3_icu_harm")
XV3_ALL_SWITCHES = (*XV3_SWITCHES, *XV3_EXTRA_SWITCHES)
assert all(k in EXPERIMENTS for k in XV3_ALL_SWITCHES), [k for k in XV3_ALL_SWITCHES if k not in EXPERIMENTS]
if any(bool(EXPERIMENTS[k]) for k in XV3_EXTRA_SWITCHES) and X_ICU_ADAPTER_OK is None:
    # Same code and output file as II.3a; runs only when every switch of II.3-II.7 and of II.8a-II.8e is off.
    X_ICU = x_icu_env()
    X_ICU_BEHAVIOR = x_icu_behavior()
    assert np.array_equal(X_ICU_BEHAVIOR, CLINICIAN_POLICY), "ICU behaviour must equal Section 14's CLINICIAN_POLICY"
    X_ICU_ADAPTER_CHECK = x_icu_adapter_check(X_ICU)
    X_ICU_ADAPTER_OK = bool(X_ICU_ADAPTER_CHECK["passes"])
    save_json(X_ICU_ADAPTER_CHECK, XENV_TABLE_DIR / "icu_adapter_check.json")
    assert X_ICU_ADAPTER_OK, "ICU-Sepsis FiniteEnv adapter differs from EVALUATOR (see xenv/icu_adapter_check.json)."
if any(bool(EXPERIMENTS[k]) for k in XV3_EXTRA_SWITCHES) and X_METHODS_SELFTEST is None:
    # Same self-test and output file as II.4; runs only when every switch of II.3-II.7 and of II.8a-II.8e is off.
    X_METHODS_SELFTEST = x_methods_selftest()
    save_json(X_METHODS_SELFTEST, XENV_TABLE_DIR / "xenv_methods_selftest.json")
    assert X_METHODS_SELFTEST["passed"], "Part II method-suite self-test failed; see xenv/xenv_methods_selftest.json"


def x_v3_draw_counts(results: pd.DataFrame, cells: Sequence[dict]) -> Dict[str, Any]:
    """Draws present per enabled cell against the configured number (spec['draws'])."""
    have = (results.groupby("cell")["draw"].nunique().to_dict() if len(results) else {})
    per = {str(s["cell"]): {"draws": int(have.get(s["cell"], 0)), "configured_draws": int(s["draws"])} for s in cells}
    return {"draws_per_cell": per,
            "draws_complete": bool(cells) and all(v["draws"] == v["configured_draws"] for v in per.values())}


XRESULTS_V3 = pd.DataFrame(columns=XRESULT_COLUMNS)
XDRAW_META_V3 = pd.DataFrame()
XV3_DRIVER_SUMMARY: Dict[str, Any] = {"status": "not run (xenv_v3_physicians, xenv_v3_aggregated and "
                                                "xenv_v3_garnet_random are off)", "code_hash": XENV_V3_CODE_HASH,
                                      "draws_complete": None}
if X_V3_CELLS:
    XRESULTS_V3, XDRAW_META_V3, _xv3s = x_v3_run_cells(X_V3_CELLS)
    assert list(XRESULTS_V3.columns) == XRESULT_COLUMNS
    XV3_DRIVER_SUMMARY = {"status": "run", **_xv3s, **x_v3_draw_counts(XRESULTS_V3, X_V3_CELLS)}
    for _c in ("checks.max_slack_vs_optimum", "checks.max_slack_vs_admissible_optimum", "checks.projection_identity_worst"):
        XV3_DRIVER_SUMMARY[f"max_{_c.split('.', 1)[1]}"] = (float(XDRAW_META_V3[_c].max())
                                                           if _c in XDRAW_META_V3 else None)
    XV3_DRIVER_SUMMARY["nonconverged_records"] = (int(XDRAW_META_V3["records.nonconverged"].sum())
                                                  if "records.nonconverged" in XDRAW_META_V3 else 0)
    XRESULTS_V3.to_csv(XENV_TABLE_DIR / "v3_results_all.csv.gz", index=False, compression="gzip")
    XDRAW_META_V3.to_csv(XENV_TABLE_DIR / "v3_draw_meta.csv", index=False)
    _dep = XRESULTS_V3[XRESULTS_V3["is_deploy_target"].astype(bool)
                       & XRESULTS_V3["policy"].isin([*XREF_POLICIES, XCOMPARATOR])]
    display(_dep.groupby(["cell", "policy"])["return"].agg(["mean", "std", "count"]).round(4))
    del _xv3s, _dep
    gc.collect()
save_json(XV3_DRIVER_SUMMARY, XENV_TABLE_DIR / "v3_driver_summary.json")
print(json.dumps({k: v for k, v in XV3_DRIVER_SUMMARY.items() if k not in ("improper_by_family_target", "draws_per_cell")},
                 indent=1, default=_json_default))
print("II.8 driver complete:", len(XRESULTS_V3), "result rows,", len(XDRAW_META_V3), "draws.")
