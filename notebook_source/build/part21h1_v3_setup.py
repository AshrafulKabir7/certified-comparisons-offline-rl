# %% [markdown]
# ## II.8 Additional experiments: environments and cells
#
# This section adds three groups of cells to Part II. They test the bracketing predictions outside the designs of
# Sections II.3–II.6. There, the Gumbel physician is ε-optimal, so its unsupported actions are mostly harmful, and
# the Garnet unsupported rows are exact mixtures by construction. The testbed also omitted two features of
# ICU-Sepsis: state aggregation and hidden confounding.
#
# | Switch | Cells | What is new |
# |---|---|---|
# | `xenv_v3_physicians` | `gumbel|best|n10000|tau20|eps0.05|phys-confounded`, `gumbel|best|n10000|tau20|eps0.05|phys-withhold-abx` | physicians that are not near-optimal (II.8b) |
# | `xenv_v3_aggregated` | `gumbel|agg-obs|n10000|tau20|eps0.05`, `gumbel|agg-proj|n10000|tau20|eps0.05` | benchmark built on aggregated observation states: diabetes hidden, glucose optionally dropped (II.8c) |
# | `xenv_v3_garnet_random` | `garnet|hrand|eps0.05`, `garnet|hrand|eps0.3` | true rows of the non-preferred actions are independent random rows, outside the harm-mixture family (II.8d) |
#
# **Cached draws stay valid.** No function of Sections II.1–II.7 is changed, and neither is `XConfig`, whose fields
# enter `x_result_xcfg`. New behaviour lives in new functions. Their source enters a separate code hash,
# `XENV_V3_CODE_HASH` (II.8e), which also folds in `XENV_CODE_HASH`. The new cells have new names, new RNG streams
# (named by the cell, as in II.3d) and cache keys under `v3/`. The cached draws of Section II.6 therefore keep their
# identities, and the new draws are cached separately.
#
# **Design.** `XV3Config` lists the cells. A cell has as many draws as its family has in `XCFG` (`gumbel_draws`,
# `garnet_draws`: 30 in the full profile, 15 in lite, 2 in smoke). Smoke mode shrinks the hospital data as in
# `apply_xrun_mode`. Learners, targets, reference rows, invariant checks and result columns are the ones of
# Section II.6. `run_v3_cell_draw` runs one draw, and `x_v3_run_cells` runs a list of cells and returns their rows.
# The run itself, the statistics and the figures come after this section.
#
# **Prerequisites.** These cells use `run_xmethods` (II.4) and, for Gumbel, the exact simulator of II.3b. If every
# switch of Sections II.3–II.7 is off, this cell runs the two checks that the Part II gate (Section 34) requires
# whenever an `xenv_*` switch is on: the ICU-Sepsis adapter check and the method-suite self-test. The Gumbel
# simulator is then enumerated in II.8b with the same function as II.3b; its Monte-Carlo exactness check runs only
# in II.3b.

# %%
# ============================================================
# II.8a Revision experiments: configuration and prerequisites
# ============================================================
EXECLOG.mark("II.8 Revision experiments: environments and cells")

XV3_SWITCHES = ("xenv_v3_physicians", "xenv_v3_aggregated", "xenv_v3_garnet_random")
XV3_ANY_ON = any(bool(EXPERIMENTS[k]) for k in XV3_SWITCHES)
XV3_KINDS = ("physician", "aggregated", "garnet-random")


@dataclass
class XV3Config:
    # Gumbel cells with other physicians: (construction, hospital episodes, tau, physician eps, physician kind).
    physician_cells: Tuple[tuple, ...] = (("best", 10_000, 20, 0.05, "confounded"),
                                          ("best", 10_000, 20, 0.05, "withhold-abx"))
    # Gumbel cells on aggregated states: (aggregation, hospital episodes, tau, physician eps, physician kind).
    aggregated_cells: Tuple[tuple, ...] = (("agg-obs", 10_000, 20, 0.05, "optimal"),
                                           ("agg-proj", 10_000, 20, 0.05, "optimal"))
    # Garnet cells with independent random rows for the non-preferred actions: behaviour eps per cell.
    garnet_random_eps: Tuple[float, ...] = (0.05, 0.30)


def apply_xv3_run_mode(c: XV3Config, mode: str) -> XV3Config:
    if mode == "smoke":
        c.physician_cells = tuple(("best", 300, 5, 0.05, k) for k in ("confounded", "withhold-abx"))
        c.aggregated_cells = tuple((g, 300, 5, 0.05, "optimal") for g in ("agg-obs", "agg-proj"))
        c.garnet_random_eps = (0.05,)
    return c


XV3CFG = apply_xv3_run_mode(XV3Config(), RUN_MODE)
save_json(asdict(XV3CFG), XENV_TABLE_DIR / "v3_config.json")

if XV3_ANY_ON and X_ICU_ADAPTER_OK is None:
    # Same code and output file as II.3a; runs only when every switch of II.3-II.7 is off.
    X_ICU = x_icu_env()
    X_ICU_BEHAVIOR = x_icu_behavior()
    assert np.array_equal(X_ICU_BEHAVIOR, CLINICIAN_POLICY), "ICU behaviour must equal Section 14's CLINICIAN_POLICY"
    X_ICU_ADAPTER_CHECK = x_icu_adapter_check(X_ICU)
    X_ICU_ADAPTER_OK = bool(X_ICU_ADAPTER_CHECK["passes"])
    save_json(X_ICU_ADAPTER_CHECK, XENV_TABLE_DIR / "icu_adapter_check.json")
    assert X_ICU_ADAPTER_OK, "ICU-Sepsis FiniteEnv adapter differs from EVALUATOR (see xenv/icu_adapter_check.json)."
if XV3_ANY_ON and X_METHODS_SELFTEST is None:
    # Same self-test and output file as II.4; runs only when every switch of II.3-II.7 is off.
    X_METHODS_SELFTEST = x_methods_selftest()
    save_json(X_METHODS_SELFTEST, XENV_TABLE_DIR / "xenv_methods_selftest.json")
    assert X_METHODS_SELFTEST["passed"], "Part II method-suite self-test failed; see xenv/xenv_methods_selftest.json"
print("II.8 switches:", {k: bool(EXPERIMENTS[k]) for k in XV3_SWITCHES}, "| config:", asdict(XV3CFG))
