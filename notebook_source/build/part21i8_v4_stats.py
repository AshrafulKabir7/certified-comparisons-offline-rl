# %% [markdown]
# ### II.9h Overview of the further experiments: status, headline numbers, checks and figures
#
# This cell collects the outputs of II.9b–II.9g. It computes nothing new about policies. For every subsection whose
# switch is on it reads the summary (`XV4_ROBUST_SUMMARY`, `XV4_CS_SUMMARY`, `XV4_BOOT_SUMMARY`, `XV4_SENS_SUMMARY`,
# `XV4_LIFT_SUMMARY`, `XV4_CLEAN`) from memory, or from its json file in `xenv/` when the global does not exist (the
# cell was not executed in this session). A subsection whose switch is off is not read at all, so a summary left on
# disk by an earlier session is never mixed in. The tables it needs (`XV4_CS_TABLE`, `XV4_CS_POLICIES`,
# `XV4_BOOT_COVERAGE`, `XV4_BOOT_IV`, `XV4_SENS_TABLE`) are read only for a subsection that ran (switch on, status
# `run`), and from the same place as its summary: from memory when the summary is in memory, else from `xenv/`. A
# subsection that is switched off, did not run, or has no (readable) summary is named in the status
# (`run | missing: … | switched off: …`), and its headline block and figure say why they are empty. Nothing here
# fails because a subsection is missing; a file cut by a disconnect is reported as unreadable. `draws_complete` is
# true when every required subsection ran with all its draws; a subsection is required when its switch is on, except
# II.9g in artifact-replay mode (`ALLOW_TRAINING = False`), as in the Section 34 gate.
#
# **Headline numbers** (`xenv/v4_overview.json`; the pooled units of each subsection):
# * II.9b: NHB and NHB-free certified share and error rate per pool and target set, with the draw-clustered intervals
#   of the error rates, also on the completion-sensitive pairs (AU + UU); the median breakdown fraction κ* of the
#   NHB-only claims; the II.8g intervals over all default policies;
# * II.9c: per test and scale, and for the point-benchmark certificates of the same draws: claims, certified share,
#   error rate (all and completion-sensitive pairs), draws with a wrong claim or an uncovered policy, coverage and mean
#   width;
# * II.9d: per variant (point, II.8g B = 10, fixed and re-estimated B = 200) and interval type: coverage over all
#   default policies, over p > 0 and over p = 0, certified share and error rate, family-wise draw shares, and the error
#   rate of the midpoint margin rule with as many claims (matched-coverage baseline);
# * II.9e: the differences new − base of every sensitivity cell and pool (coverage, certificates, Kendall τ, support
#   of the physician), with the kind, the pairing and the draws of the new cell, and the new cells that have no
#   reference cell;
# * II.9f: the range of J_true over all lifts (width, share of benchmark intervals that contain it, benchmark
#   certificates that hold under every lift) and the error rates under each lift;
# * II.9g: the pass flag of the clean rerun, its four criteria and bitwise identity.
#
# **Checks** (`xenv/v4_stats_checks.json`). For every subsection: switch, status, whether its draws are complete, and
# its invariant flag. The flags are the subsections' own: II.9b theorem and identity counts; II.9c self-test of the
# inner problem and bound order; II.9d per-replicate invariants (Theorem 1(iii)), unit weights and the true-rows
# bracket; II.9e self-test and theorem counts; II.9f its pass flag and the slack maxima of its asserted invariants
# within their tolerances (checked again here). II.9g has no invariant; its pass flag is reported, and the Section 34
# gate uses it. Two orderings are recounted here from the stored tables, since both hold by construction:
#
# * every confidence-set row: finite bounds, L ≤ U_free and U_nhb = min(U_free, U_proj) ≤ U_free;
# * every bootstrap interval (Theorem 1(iii) on every replicate, and monotone quantiles): lo ≤ up for the NHB
#   interval; up_free ≥ up_nhb; lo_free = lo_nhb; the min / max interval contains the quantile interval; the ends are
#   defined together (the point interval has no min / max ends). The tolerance is 4·1e-9: the per-replicate slacks
#   (each ≤ 1e-9) add up in up_free ≥ up_nhb.
#
# A non-zero count (a table without its columns counts too), or a subsection flag that is false, sets `passes` to
# false. The cell then raises after writing its outputs, as II.8i does. Coverage and error rates are outcomes:
# reported, never checked.
#
# **Figures** (style of `fig_v3_bootstrap`: horizontal bars per cell; a cross at 0 marks a value that is undefined or
# not available, e.g. an error rate without any certified pair, or a cell without II.8g draws):
#
# * `fig_v4_confidence` (rows: NHB-free and NHB certificates): for every II.9c cell, the certified share and the
#   share of certified pairs that are wrong. Bars show the paired confidence-set test (`paired_free` / `paired_nhb`)
#   at every scale (scale 1 valid, smaller scales heuristic) beside the point-benchmark certificate (`point_free` /
#   `point_nhb`) of the same policies;
# * `fig_v4_bootstrap` (rows: NHB interval, NHB-free interval): for every II.9d cell, the coverage of J_true over all
#   default policies, the certified share and the certification error of the point interval, II.8g (B = 10, fixed
#   mask), and the fixed and re-estimated bootstraps (B = 200).
#
# | Output | Content |
# |---|---|
# | `XV4_STATS` | overview, checks, per-subsection status, the small tables read, figure status, the current outputs |
# | `xenv/v4_overview.json` | status (what is missing), per-subsection status, headline numbers, the paths of every II.9 output with `exists` and `current` (written by a subsection that ran, or by this cell) |
# | `xenv/v4_stats_checks.json` | per-subsection status, completion and invariant flags, recounted orderings, inputs, figure status, `passes` |
# | `figures/xenv/fig_v4_confidence.*`, `figures/xenv/fig_v4_bootstrap.*` | the two figures above |
#
# **Expected cost on Colab** (estimate, not a measurement): under a minute. The summaries and tables are small.
# Reading `v4_cs_policies.csv.gz` and `v4_boot_intervals.csv.gz` (a few 10 000 rows each) from disk, when they are
# not in memory, adds a few seconds.

# %%
# ============================================================
# II.9h Second-revision overview, checks and figures
# ============================================================
EXECLOG.mark("II.9h Second-revision experiments: overview, checks and figures")

# subsection -> (switch, summary global, summary file in xenv/)
XV4H_SUBSECTIONS = {"II.9b": ("xenv_v4_revision_robust", "XV4_ROBUST_SUMMARY", "v4_robust_summary.json"),
                    "II.9c": ("xenv_v4_confidence", "XV4_CS_SUMMARY", "v4_cs_summary.json"),
                    "II.9d": ("xenv_v4_bootstrap", "XV4_BOOT_SUMMARY", "v4_boot_summary.json"),
                    "II.9e": ("xenv_v4_sensitivity", "XV4_SENS_SUMMARY", "v4_sens_summary.json"),
                    "II.9f": ("xenv_v4_lift", "XV4_LIFT_SUMMARY", "v4_lift_summary.json"),
                    "II.9g": ("xenv_v4_clean_check", "XV4_CLEAN", "v4_clean_check.json")}
assert {v[0] for v in XV4H_SUBSECTIONS.values()} == set(XV4_SWITCHES), "II.9h must cover every II.9 switch"
# tables read for the checks and figures: key -> (subsection, global, file in xenv/)
XV4H_TABLES = {"cs_table": ("II.9c", "XV4_CS_TABLE", "v4_cs_summary.csv"),
               "cs_policies": ("II.9c", "XV4_CS_POLICIES", "v4_cs_policies.csv.gz"),
               "boot_coverage": ("II.9d", "XV4_BOOT_COVERAGE", "v4_boot_coverage.csv"),
               "boot_intervals": ("II.9d", "XV4_BOOT_IV", "v4_boot_intervals.csv.gz"),
               "sens_table": ("II.9e", "XV4_SENS_TABLE", "v4_sens_summary.csv")}
XV4H_KEEP_TABLES = ("cs_table", "boot_coverage", "sens_table")     # kept in XV4_STATS (the others are large)
XV4H_RECOUNT_TABLES = {"II.9c": "cs_policies", "II.9d": "boot_intervals"}   # input of the recounted orderings
# Every II.9 output, relative to tables/ ('xenv/...') or to the figures ('figures/xenv/...'), by subsection.
XV4H_OUTPUTS = {
    "II.9a": ("xenv/v4_config.json",),
    "II.9b": ("xenv/v4_robust_pairs.csv.gz", "xenv/v4_robust_summary.csv", "xenv/v4_robust_gapbins.csv",
              "xenv/v4_robust_kappa.csv", "xenv/v4_v3boot_allpolicy.csv", "xenv/v4_robust_summary.json"),
    "II.9c": ("xenv/v4_cs_policies.csv.gz", "xenv/v4_cs_draws.csv", "xenv/v4_cs_summary.csv",
              "xenv/v4_cs_summary.json"),
    "II.9d": ("xenv/v4_boot_intervals.csv.gz", "xenv/v4_boot_draws.csv", "xenv/v4_boot_coverage.csv",
              "xenv/v4_boot_meta.csv", "xenv/v4_boot_summary.json"),
    "II.9e": ("xenv/v4_sens_results_all.csv.gz", "xenv/v4_sens_draw_meta.csv", "xenv/v4_sens_summary.csv",
              "xenv/v4_sens_certified_draws.csv", "xenv/v4_sens_rank_fidelity.csv",
              "xenv/v4_sens_rank_fidelity_draws.csv", "xenv/v4_sens_summary.json"),
    "II.9f": ("xenv/v4_lift_results.csv.gz", "xenv/v4_lift_summary.csv", "xenv/v4_lift_summary.json"),
    "II.9g": ("xenv/v4_clean_check.json", "xenv/v4_clean_diffs.csv"),
    "II.9h": ("xenv/v4_overview.json", "xenv/v4_stats_checks.json", "figures/xenv/fig_v4_confidence.png",
              "figures/xenv/fig_v4_confidence.pdf", "figures/xenv/fig_v4_bootstrap.png",
              "figures/xenv/fig_v4_bootstrap.pdf")}
# Figure rows: (label, confidence-set test, point-benchmark test) and (interval type, label); bootstrap variants.
XV4H_CS_FIG = (("NHB-free certificate", "paired_free", "point_free"), ("NHB certificate", "paired_nhb", "point_nhb"))
XV4H_BOOT_FIG_INTERVALS = (("nhb", "NHB interval"), ("free", "NHB-free interval"))
XV4H_BOOT_FIG_VARIANTS = ("point", "v3boot", "fixed", "reestimated")
# Headline columns per subsection.
XV4H_ROBUST_KEYS = ("draws", "pairs", "cert_nhb_share", "cert_nhb_error_rate", "cert_nhb_error_rate_ci_low",
                    "cert_nhb_error_rate_ci_high", "cert_free_share", "cert_free_error_rate",
                    "cert_free_error_rate_ci_low", "cert_free_error_rate_ci_high", "cert_nhb_cs_share",
                    "cert_nhb_cs_error_rate", "cert_free_cs_share", "cert_free_cs_error_rate")
XV4H_CS_KEYS = ("draws", "claims", "share", "errors", "error_rate", "share_sens", "error_rate_sens",
                "draws_any_wrong", "draws_any_uncovered", "coverage", "mean_width")
XV4H_BOOT_KEYS = ("draws", "coverage_all", "coverage_ppos", "coverage_pzero", "mean_width", "certified_share",
                  "error_rate", "certified_share_cs", "error_rate_cs", "draws_any_wrong_share",
                  "draws_all_covered_share", "margin_mid_error_rate")
XV4H_SENS_KEYS = ("bench_coverage_violating", "sem_upper_coverage_violating", "cc_bench_cert_nhb_share",
                  "cc_bench_cert_nhb_error_rate", "cc_bench_cert_free_share", "cc_bench_cert_free_error_rate",
                  "cc_true_rows_cert_nhb_share", "cc_true_rows_cert_nhb_error_rate", "cc_true_rows_cert_free_share",
                  "cc_true_rows_cert_free_error_rate", "tau_mean_mean", "tau_terminate_mean", "tau_diff_mean",
                  "tau_mean_delta", "tau_mean_delta_ci_low", "tau_mean_delta_ci_high", "tau_terminate_delta",
                  "tau_terminate_delta_ci_low", "tau_terminate_delta_ci_high", "tau_diff_delta",
                  "tau_diff_delta_ci_low", "tau_diff_delta_ci_high", "constr_mean_admissible_actions",
                  "constr_single_admissible_fraction")
XV4H_CLEAN_CHECKS = ("draws", "draws_planned", "draws_passing", "draws_exact", "draws_without_reference",
                     "rows_matched", "missing_in_rerun", "missing_in_reference", "status_mismatches", "rows_above_tol",
                     "certified_count_differences", "max_abs_diff")
# II.9f invariant slacks (x_v4_lift_checks 'max_<key>'): value tolerance, and the range tolerance of its 'tolerances'.
XV4H_LIFT_SLACKS = ("bench_rule_p_difference", "bench_negative_width", "bench_width_minus_bound",
                    "target_p_difference", "lower_bound_shortfall", "rows_negative_width", "rows_width_minus_bound")
XV4H_LIFT_RANGE_SLACKS = ("lift_below_worst", "lift_above_best", "range_negative_width")
# Tolerance of the recounted II.9d orderings: each replicate satisfies Theorem 1(iii) and the shared p to XV4_TOL
# (asserted in II.8g / II.9d), and up_free >= up_nhb composes these slacks with v_range <= 2.
XV4H_ORDER_TOL = 4 * XV4_TOL


def _xv4h_path(rel: str, table_dir: Path, fig_dir: Path) -> Path:
    """Absolute path of an output listed relative to tables/ ('xenv/...') or to the figures ('figures/xenv/...')."""
    if rel.startswith("figures/xenv/"):
        return Path(fig_dir) / rel[len("figures/xenv/"):]
    assert rel.startswith("xenv/"), rel
    return Path(table_dir) / rel[len("xenv/"):]


def xv4h_summary(name: str, table_dir: Path) -> Tuple[Dict[str, Any], str]:
    """Summary of one subsection: the global when it exists, else its json file in table_dir; ({}, 'missing') when
    neither, and a status that names the file when it cannot be read (e.g. a write cut by a disconnect)."""
    _, glob, fname = XV4H_SUBSECTIONS[name]
    s = globals().get(glob)
    if isinstance(s, dict) and s:
        return s, "memory"
    p = Path(table_dir) / fname
    if not p.exists():
        return {}, "missing"
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except XCACHE_READ_ERRORS as exc:
        return {"status": f"missing: xenv/{fname} unreadable ({type(exc).__name__})"}, f"xenv/{fname} (unreadable)"
    if not isinstance(s, dict):
        return {"status": f"missing: xenv/{fname} is not a summary"}, f"xenv/{fname} (not a summary)"
    return s, f"xenv/{fname}"


def xv4h_table(global_name: str, file_name: str, ran: bool, in_memory: bool, table_dir: Path
               ) -> Tuple[pd.DataFrame, str]:
    """A table of a subsection that ran, from the same place as its summary: the global when the summary is in memory
    (never a file of another session beside a summary of this one), else its file in table_dir (exact float round
    trip). An empty frame when the subsection did not run or the table is missing or unreadable."""
    if not ran:
        return pd.DataFrame(), "not used (subsection not run)"
    if in_memory:
        df = globals().get(global_name)
        if isinstance(df, pd.DataFrame):
            return df, (global_name if len(df) else f"{global_name} (empty)")
        return pd.DataFrame(), f"missing: {global_name} is not in memory (its cell did not run in this session)"
    p = Path(table_dir) / file_name
    if not p.exists():
        return pd.DataFrame(), "missing"
    try:
        return pd.read_csv(p, float_precision="round_trip", low_memory=False), f"xenv/{file_name}"
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), f"xenv/{file_name} (empty)"
    except XCACHE_READ_ERRORS as exc:                   # e.g. a gzip cut by a disconnect
        return pd.DataFrame(), f"xenv/{file_name} unreadable ({type(exc).__name__})"


def _xv4h_all(flags: Sequence[Optional[bool]]) -> Optional[bool]:
    """Combined flag: None if no flag is set, else whether every set flag is true."""
    f = [bool(x) for x in flags if x is not None]
    return all(f) if f else None


def _xv4h_num(x: Any) -> Optional[float]:
    """A finite float, or None (missing, null in json, NaN)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------- invariants (recounted orderings and the subsections' own flags) ----------
def xv4h_cs_order_counts(pol: pd.DataFrame) -> Dict[str, int]:
    """Orderings of the confidence-set rows of II.9c (all must be 0): finite bounds, L <= U_free, U_nhb <= U_free,
    U_nhb = min(U_free, U_proj); tolerance XV4_TOL. A table without these columns counts as one failure."""
    if pol is None or not len(pol):
        return {}
    cols = ("L", "U_free", "U_proj", "U_nhb")
    if not set(cols) <= set(pol.columns):
        return {"cs_table_missing_columns": len(set(cols) - set(pol.columns))}
    L, Uf, Up, Un = (pd.to_numeric(pol[c], errors="coerce").to_numpy(dtype=float) for c in cols)
    with np.errstate(invalid="ignore"):
        return {"cs_nonfinite_bounds": int((~np.isfinite(np.c_[L, Uf, Up, Un])).any(axis=1).sum()),
                "cs_lower_above_free_upper": int((L > Uf + XV4_TOL).sum()),
                "cs_nhb_above_free_upper": int((Un > Uf + XV4_TOL).sum()),
                "cs_nhb_not_min_of_free_and_proj": int((np.abs(Un - np.minimum(Uf, Up)) > XV4_TOL).sum())}


def xv4h_boot_order_counts(iv: pd.DataFrame) -> Dict[str, int]:
    """Orderings of the II.9d interval table (all must be 0), tolerance XV4H_ORDER_TOL: lo <= up (NHB),
    up_free >= up_nhb, lo_free = lo_nhb, the min / max interval containing the quantile interval, and the ends
    defined together (min / max ends too, except on the point rows, which have none). A table without these columns
    counts as one failure."""
    if iv is None or not len(iv):
        return {}
    cols = ("boot_variant", "lo_nhb", "up_nhb", "lo_free", "up_free", "lo_minmax", "up_minmax")
    if not set(cols) <= set(iv.columns):
        return {"boot_table_missing_columns": len(set(cols) - set(iv.columns))}
    g = {c: pd.to_numeric(iv[c], errors="coerce").to_numpy(dtype=float) for c in cols[1:]}
    nan = {c: np.isnan(v) for c, v in g.items()}
    boot = iv["boot_variant"].astype(str).to_numpy() != "point"
    tol = XV4H_ORDER_TOL
    with np.errstate(invalid="ignore"):
        return {"boot_nhb_lower_above_upper": int((g["lo_nhb"] > g["up_nhb"] + tol).sum()),
                "boot_free_upper_below_nhb_upper": int((g["up_free"] < g["up_nhb"] - tol).sum()),
                "boot_free_lower_differs": int((np.abs(g["lo_free"] - g["lo_nhb"]) > tol).sum()),
                "boot_minmax_inside_quantile": int(((g["lo_minmax"] > g["lo_nhb"] + tol)
                                                    | (g["up_minmax"] < g["up_nhb"] - tol)).sum()),
                "boot_ends_defined_apart": int(((nan["lo_nhb"] != nan["up_nhb"]) | (nan["lo_free"] != nan["lo_nhb"])
                                                | (nan["up_free"] != nan["up_nhb"])
                                                | (boot & ((nan["lo_minmax"] != nan["lo_nhb"])
                                                           | (nan["up_minmax"] != nan["up_nhb"])))).sum())}


def xv4h_invariants(name: str, s: Dict[str, Any], order_counts: Dict[str, int]
                    ) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Invariant flag of one subsection that ran (None when it reports none) and its evidence: the subsection's own
    flags and the orderings recounted here ('cs_*' for II.9c, 'boot_*' for II.9d). Coverage and error rates never
    enter."""
    chk = s.get("checks") or {}
    if name == "II.9b":
        return chk.get("passes"), {"theorem_counts": chk.get("theorem_counts"),
                                   "identity_counts": chk.get("identity_counts")}
    if name in ("II.9c", "II.9d"):
        own = {k: v for k, v in order_counts.items() if k.startswith("cs_" if name == "II.9c" else "boot_")}
        st = s.get("selftest") if name == "II.9c" else None
        st_ok = st.get("passes") if isinstance(st, dict) else None
        flag = _xv4h_all([chk.get("passes"), st_ok, (all(v == 0 for v in own.values()) if own else None)])
        return flag, {"checks": {k: v for k, v in chk.items() if not isinstance(v, (dict, list))},
                      "selftest_passes": st_ok, "recounted": own}
    if name == "II.9e":
        return chk.get("passes"), {"theorem_counts": chk.get("theorem_counts"),
                                   "selftest_passes": (chk.get("selftest") or {}).get("passes"),
                                   "draw_checks": chk.get("draw_checks")}
    if name == "II.9f":
        tol_b = _xv4h_num((chk.get("tolerances") or {}).get("bounds")) or XV4_TOL
        slack = {k: _xv4h_num(chk.get(f"max_{k}")) for k in (*XV4H_LIFT_SLACKS, *XV4H_LIFT_RANGE_SLACKS)}
        bad = [k for k, v in slack.items() if v is not None and v > (tol_b if k in XV4H_LIFT_RANGE_SLACKS else XV4_TOL)]
        have = any(v is not None for v in slack.values())
        return _xv4h_all([chk.get("passes"), (not bad) if have else None]), {
            "passes": chk.get("passes"), "max_slacks": slack, "above_tolerance": bad, "range_tolerance": tol_b}
    assert name == "II.9g", name
    return None, {"clean_pass": s.get("pass"), "note": "no invariant: the pass flag is reported and gated separately"}


# ---------- headline numbers ----------
def xv4h_headline_robust(s: Dict[str, Any]) -> Dict[str, Any]:
    """II.9b: certificates per pool and target set, the median kappa* of the NHB-only claims, and the II.8g intervals
    over all default policies (pooled units)."""
    hd, kp = s.get("headline") or {}, s.get("kappa") or {}
    return {"pools": {u: {tf: {k: r.get(k) for k in XV4H_ROBUST_KEYS if k in r} for tf, r in g.items()}
                      for u, g in hd.items()},
            "kappa_star_median_nhb_only": {
                u: {tf: {"nhb_only_claims": r.get("nhb_only_claims"),
                         "kappa_star_q50": (r.get("kappa_star_quantiles") or {}).get("kappa_star_q50")}
                    for tf, r in g.items()} for u, g in kp.items()},
            "v3boot_allpolicy": s.get("v3boot") or {}}


def xv4h_headline_cs(table: pd.DataFrame, s: Dict[str, Any]) -> Dict[str, Any]:
    """II.9c: per pooled unit (the cell itself without a pool) x test x scale ('point' for the point-benchmark
    certificates): claims, certified share, error rate (all and completion-sensitive pairs), wrong-claim and
    uncovered-policy draws, coverage and mean width."""
    recs = table.to_dict("records") if len(table) else list(s.get("headline") or [])
    out: Dict[str, Any] = {"config": s.get("config") or {}}
    units = list(dict.fromkeys(str(r["unit"]) for r in recs))
    pools = [u for u in units if u.startswith("pooled:")] or units
    for r in recs:
        if str(r["unit"]) not in pools:
            continue
        sc = _xv4h_num(r.get("scale"))
        lab = "point" if sc is None else f"scale {sc:g}"
        out.setdefault(str(r["unit"]), {})[f"{r['test']}|{lab}"] = {k: r.get(k) for k in XV4H_CS_KEYS if k in r}
    return out


def xv4h_headline_boot(cov: pd.DataFrame, s: Dict[str, Any]) -> Dict[str, Any]:
    """II.9d: per pooled unit x variant x interval type: coverage over all / p > 0 / p = 0 default policies, width,
    certificates, family-wise draw shares and the matched midpoint margin; B of both bootstraps and the failed
    re-estimated replicates."""
    v3 = globals().get("XV3BOOT")
    out: Dict[str, Any] = {"replicates": s.get("replicates"), "replicates_v3boot": getattr(v3, "replicates", None),
                           "quantiles": s.get("quantiles"),
                           "failed_replicates": {k: (s.get("failed_replicates") or {}).get(k)
                                                 for k in ("total", "replicates", "share", "draws_with_failure")}}
    if not len(cov):
        out["summary_headline"] = s.get("headline") or {}
        return out
    units = [str(u) for u in pd.unique(cov["cell"])]
    pools = [u for u in units if u.startswith("pooled:")] or units
    for r in cov[cov["cell"].astype(str).isin(pools)].to_dict("records"):
        out.setdefault(str(r["cell"]), {})[f"{r['boot_variant']}|{r['interval']}"] = {
            k: r[k] for k in XV4H_BOOT_KEYS if k in r}
    return out


def xv4h_headline_sens(table: pd.DataFrame, s: Dict[str, Any]) -> Dict[str, Any]:
    """II.9e: the differences new - base per sensitivity cell and per pool (rows 'new - base' of the II.9e table, with
    the kind, the pairing and the draws of the new side; the summary's per-cell deltas when the table is not
    available), and the new cells without a reference cell (no difference)."""
    if not len(table) or not {"row_type", "side", "cell"} <= set(table.columns):
        return {"cells": dict(s.get("deltas") or {}), "pools": {}, "without_reference": None}
    d = table[table["side"] == "new - base"]
    new = table[table["side"] == "new"]
    n_new = ({(str(r["row_type"]), str(r["cell"])): (None if _xv4h_num(r["n_draws"]) is None else int(r["n_draws"]))
              for r in new.to_dict("records")} if "n_draws" in new.columns else {})
    pick = {rt: {str(r["cell"]): {"v4_kind": r.get("v4_kind"), "reference": r.get("reference"),
                                  "paired": r.get("paired"), "n_draws": n_new.get((rt, str(r["cell"]))),
                                  **{k: r[k] for k in XV4H_SENS_KEYS if k in r and pd.notna(r[k])}}
                 for r in d[d["row_type"] == rt].to_dict("records")} for rt in ("cell", "pool")}
    alone = [str(c) for c in new.loc[new["row_type"] == "cell", "cell"] if str(c) not in pick["cell"]]
    return {"cells": pick["cell"], "pools": pick["pool"], "without_reference": alone}


def xv4h_headline_lift(s: Dict[str, Any]) -> Dict[str, Any]:
    """II.9f: the headline of its widest pool per lift, including the 'range' row (bounds over all lifts)."""
    return {"lift_kinds": s.get("lift_kinds"), "lift_bounds": s.get("lift_bounds"), **(s.get("headline") or {})}


def xv4h_headline_clean(s: Dict[str, Any]) -> Dict[str, Any]:
    """II.9g: pass flag, criteria, bitwise identity and the pooled comparison counts."""
    chk = s.get("checks") or {}
    return {**{k: s.get(k) for k in ("pass", "bitwise_identical", "criteria", "tolerance", "failing_draws_total")},
            "scope": (s.get("config") or {}).get("clean_scope"), **{k: chk.get(k) for k in XV4H_CLEAN_CHECKS}}


# ---------- figures (style of xs_fig_v3_bootstrap) ----------
def _xv4h_bars(ax, n_rows: int, series: Sequence[Tuple[str, np.ndarray, str]]) -> bool:
    """Grouped horizontal bars in percent: one group per row, one bar per series (label, values in [0, 1], colour).
    An undefined value (NaN) gets a muted cross just right of the axis instead of a bar. Returns whether a value was
    undefined."""
    y = np.arange(n_rows)
    k = max(len(series), 1)
    h = 0.8 / k
    undefined = False
    for i, (lab, v, col) in enumerate(series):
        v = 100.0 * np.asarray(v, dtype=float)
        yi = y - 0.4 + h * (i + 0.5)
        ax.barh(yi, np.nan_to_num(v), height=0.9 * h, color=col, label=lab)
        bad = ~np.isfinite(v)
        if bad.any():
            undefined = True
            ax.scatter(np.full(int(bad.sum()), 0.015), yi[bad], marker="x", s=16, color=XS_MUTED, linewidth=1.0,
                       zorder=3, transform=ax.get_yaxis_transform())   # x in axes fraction, y in data
    return undefined


def _xv4h_pct_limit(series: Sequence[Tuple[str, np.ndarray, str]]) -> float:
    """Right x limit (percent) of an error-rate panel: 5 % at least, else 10 % above the largest value."""
    v = np.concatenate([100.0 * np.asarray(s[1], dtype=float) for s in series]) if series else np.zeros(0)
    v = v[np.isfinite(v)]
    return max(5.0, 1.1 * float(v.max())) if len(v) else 5.0


def _xv4h_finish(fig, axes: np.ndarray, order: Sequence[str], undefined: bool, ncol: int) -> None:
    """Cell labels on the shared y axis (run order, top to bottom) and one legend under the figure, so that it never
    covers data or an axis label."""
    y = np.arange(len(order))
    for ax in axes[:, 0]:
        ax.set_yticks(y)
        ax.set_yticklabels([xs_cell_label(c) for c in order], fontsize=7)
    axes[0, 0].invert_yaxis()
    handles, labels = axes[-1, 0].get_legend_handles_labels()
    if undefined:
        handles.append(plt.Line2D([], [], marker="x", linestyle="none", markersize=4, color=XS_MUTED))
        labels.append("undefined or not available (e.g. no certified pair)")
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.08, 0.0), ncol=ncol)


def xs_fig_v4_confidence(table: pd.DataFrame, cells: Sequence[str], fig_dir: Path) -> str:
    """Per II.9c cell: certified share and certification error of the paired confidence-set tests at every scale,
    beside the point-benchmark certificates of the same policies (rows: NHB-free / NHB certificates)."""
    t = table[table["unit"].astype(str).isin([str(c) for c in cells])] if len(table) else table
    if t.empty:
        return "not produced: no II.9c confidence-set results"
    order = [str(c) for c in cells if str(c) in set(t["unit"].astype(str))]
    unit, test = t["unit"].astype(str), t["test"].astype(str)
    scale = pd.to_numeric(t["scale"], errors="coerce")
    scales = sorted({float(x) for x in scale.dropna()}, reverse=True)
    kinds = [k for k in XV4H_CS_FIG if test.isin(k[1:]).any()]
    if not kinds:
        return "not produced: the II.9c table has no paired confidence-set or point-benchmark test"

    def values(name: str, sc: Optional[float], col: str) -> np.ndarray:
        m = (test == name) & (scale.isna() if sc is None else np.isclose(scale, sc))
        if col not in t.columns or not m.any():
            return np.full(len(order), np.nan)
        return pd.to_numeric(t.loc[m, col], errors="coerce").set_axis(unit[m]).reindex(order).to_numpy(dtype=float)

    fig, axes = plt.subplots(len(kinds), 2, figsize=(12, len(kinds) * (0.9 + 0.5 * len(order)) + 0.4), squeeze=False,
                             sharey=True)
    undefined = False
    for r, (lab, cs_test, pt_test) in enumerate(kinds):
        for j, (col, xlab, what) in enumerate((("share", "certified pairs (%)", "certified share"),
                                               ("error_rate", "certified pairs that are wrong (%)",
                                                "certification error"))):
            series = [("point benchmark (same policies)", values(pt_test, None, col), XS_COLORS[0])]
            series += [(f"confidence set, scale {s:g} ({'valid' if s == 1.0 else 'heuristic'})",
                        values(cs_test, s, col), XS_COLORS[(2 + i) % len(XS_COLORS)]) for i, s in enumerate(scales)]
            ax = axes[r, j]
            undefined |= _xv4h_bars(ax, len(order), series)
            ax.set_xlim(0, 100 if col == "share" else _xv4h_pct_limit(series))
            ax.set_xlabel(xlab)
            ax.set_title(f"({'abcdefgh'[2 * r + j]}) {lab}: {what}", loc="left", color=XS_INK, fontsize=9)
    _xv4h_finish(fig, axes, order, undefined, ncol=3)
    return _xs_save(fig, fig_dir, "fig_v4_confidence")


def xs_fig_v4_bootstrap(cov: pd.DataFrame, cells: Sequence[str], fig_dir: Path, b_v3: Optional[int],
                        b_v4: Optional[int]) -> str:
    """Per II.9d cell: coverage of J_true over all default policies, certified share and certification error of the
    point interval, II.8g (B = 10, fixed mask) and the fixed and re-estimated bootstraps (rows: NHB / NHB-free)."""
    c = cov[cov["cell"].astype(str).isin([str(x) for x in cells])] if len(cov) else cov
    if c.empty:
        return "not produced: no II.9d bootstrap coverage rows"
    order = [str(x) for x in cells if str(x) in set(c["cell"].astype(str))]
    rows = [(it, lab) for it, lab in XV4H_BOOT_FIG_INTERVALS if (c["interval"] == it).any()]
    variants = [v for v in XV4H_BOOT_FIG_VARIANTS if (c["boot_variant"] == v).any()]
    if not rows or not variants:
        return "not produced: the II.9d coverage table has no NHB / NHB-free rows"
    b3, b4 = (f", B = {n}" if n else "" for n in (b_v3, b_v4))
    labels = {"point": "point benchmark", "v3boot": f"II.8g, fixed mask{b3}", "fixed": f"fixed mask{b4}",
              "reestimated": f"re-estimated construction{b4}"}
    panels = (("coverage_all", "coverage of J_true, all default policies (%)", "coverage"),
              ("certified_share", "certified pairs (%)", "certified share"),
              ("error_rate", "certified pairs that are wrong (%)", "certification error"))

    def values(variant: str, interval: str, col: str) -> np.ndarray:
        m = (c["boot_variant"] == variant) & (c["interval"] == interval)
        if col not in c.columns or not m.any():
            return np.full(len(order), np.nan)
        return (pd.to_numeric(c.loc[m, col], errors="coerce").set_axis(c.loc[m, "cell"].astype(str)).reindex(order)
                .to_numpy(dtype=float))

    fig, axes = plt.subplots(len(rows), 3, figsize=(12, len(rows) * (0.9 + 0.5 * len(order)) + 0.4), squeeze=False,
                             sharey=True)
    undefined = False
    for r, (it, lab) in enumerate(rows):
        for j, (col, xlab, what) in enumerate(panels):
            series = [(labels[v], values(v, it, col), XS_COLORS[XV4H_BOOT_FIG_VARIANTS.index(v)]) for v in variants]
            ax = axes[r, j]
            undefined |= _xv4h_bars(ax, len(order), series)
            ax.set_xlim(0, _xv4h_pct_limit(series) if col == "error_rate" else 100)
            ax.set_xlabel(xlab)
            ax.set_title(f"({'abcdefghi'[3 * r + j]}) {lab}: {what}", loc="left", color=XS_INK, fontsize=9)
    _xv4h_finish(fig, axes, order, undefined, ncol=3)
    return _xs_save(fig, fig_dir, "fig_v4_bootstrap")


# ---------- run ----------
def _xv4h_cells(s: Dict[str, Any], table: pd.DataFrame, col: str) -> List[str]:
    """Cells of a subsection in its run order (summary 'cells'), else in table order (pooled units left out)."""
    cells = [str(c) for c in (s.get("cells") or [])]
    if not cells and len(table) and col in table.columns:
        cells = [str(u) for u in pd.unique(table[col].astype(str)) if not str(u).startswith("pooled:")]
    return cells


def xv4h_run(table_dir: Path, fig_dir: Path) -> Dict[str, Any]:
    """Section II.9h end to end: status, headline numbers, checks, figures and the two json files (inputs read from and
    outputs written to table_dir / fig_dir). Returns XV4_STATS."""
    t0 = time.perf_counter()
    table_dir, fig_dir = Path(table_dir), Path(fig_dir)
    summ: Dict[str, Dict[str, Any]] = {}
    subs: Dict[str, Dict[str, Any]] = {}
    for name, (switch, _, _) in XV4H_SUBSECTIONS.items():
        on = bool(EXPERIMENTS[switch])
        # a switched-off subsection is not read, so a summary left by an earlier session is never mixed in
        s, src = xv4h_summary(name, table_dir) if on else ({}, "not read (switched off)")
        summ[name] = s
        st = (str(s.get("status", "missing: no summary in memory or on disk")) if on
              else f"switched off ({switch} = False)")
        # II.9g is not required in artifact-replay mode (ALLOW_TRAINING False), as in the Section 34 gate
        subs[name] = {"switch": switch, "enabled": on, "required": on and (name != "II.9g" or bool(ALLOW_TRAINING)),
                      "summary_source": src, "status": st, "ran": on and st == "run",
                      "draws_complete": s.get("draws_complete"), "code_hash": s.get("code_hash"),
                      "elapsed_s": s.get("elapsed_s")}
    tables: Dict[str, pd.DataFrame] = {}
    inputs: Dict[str, Any] = {"summaries": {k: v["summary_source"] for k, v in subs.items()}, "tables": {}}
    for key, (sub, glob, fname) in XV4H_TABLES.items():
        tables[key], src = xv4h_table(glob, fname, subs[sub]["ran"], subs[sub]["summary_source"] == "memory",
                                      table_dir)
        inputs["tables"][key] = {"source": src, "rows": int(len(tables[key]))}
    order_counts = {**xv4h_cs_order_counts(tables["cs_policies"]), **xv4h_boot_order_counts(tables["boot_intervals"])}
    for name, s in subs.items():
        if s["ran"]:
            flag, detail = xv4h_invariants(name, summ[name], order_counts)
            if name in XV4H_RECOUNT_TABLES:
                detail["recount_input"] = inputs["tables"][XV4H_RECOUNT_TABLES[name]]
        else:
            flag, detail = None, {"note": "not evaluated: the subsection did not run"}
        s.update(invariants_pass=None if flag is None else bool(flag), invariants=detail)
    missing = {k: v["status"] for k, v in subs.items() if v["enabled"] and not v["ran"]}
    off = [k for k, v in subs.items() if not v["enabled"]]
    parts = ["run"]
    if missing:
        parts.append("missing: " + "; ".join(f"{k} {st}" for k, st in missing.items()))
    if off:
        parts.append("switched off: " + ", ".join(off))
    status = " | ".join(parts)
    required = [v for v in subs.values() if v["required"]]
    draws_complete = None if not required else all(v["ran"] and v["draws_complete"] is True for v in required)
    makers: Dict[str, Callable[[], Dict[str, Any]]] = {
        "II.9b": lambda: xv4h_headline_robust(summ["II.9b"]),
        "II.9c": lambda: xv4h_headline_cs(tables["cs_table"], summ["II.9c"]),
        "II.9d": lambda: xv4h_headline_boot(tables["boot_coverage"], summ["II.9d"]),
        "II.9e": lambda: xv4h_headline_sens(tables["sens_table"], summ["II.9e"]),
        "II.9f": lambda: xv4h_headline_lift(summ["II.9f"]), "II.9g": lambda: xv4h_headline_clean(summ["II.9g"])}
    headline = {k: (fn() if subs[k]["ran"] else {"status": f"not available: {subs[k]['status']}"})
                for k, fn in makers.items()}
    figures: Dict[str, str] = {}
    if subs["II.9c"]["ran"]:
        figures["fig_v4_confidence"] = xs_fig_v4_confidence(
            tables["cs_table"], _xv4h_cells(summ["II.9c"], tables["cs_table"], "unit"), fig_dir)
    else:
        figures["fig_v4_confidence"] = f"not produced: II.9c {subs['II.9c']['status']}"
    if subs["II.9d"]["ran"]:
        v3 = globals().get("XV3BOOT")
        figures["fig_v4_bootstrap"] = xs_fig_v4_bootstrap(
            tables["boot_coverage"], _xv4h_cells(summ["II.9d"], tables["boot_coverage"], "cell"), fig_dir,
            getattr(v3, "replicates", None), summ["II.9d"].get("replicates"))
    else:
        figures["fig_v4_bootstrap"] = f"not produced: II.9d {subs['II.9d']['status']}"
    inv = {k: v["invariants_pass"] for k, v in subs.items()}
    passes = bool(all(f is not False for f in inv.values()) and all(n == 0 for n in order_counts.values()))
    checks = {"status": status, "code_hash": XV4_STATS_CODE_HASH, "passes": passes, "invariants_pass": inv,
              "order_counts": order_counts, "draws_complete": draws_complete, "missing": missing,
              "switched_off": off, "subsections": subs, "inputs": inputs, "figures": figures,
              "tolerances": {"value": XV4_TOL, "p": XV4_PTOL, "boot_order": XV4H_ORDER_TOL},
              "elapsed_s": time.perf_counter() - t0}
    save_json(checks, table_dir / "v4_stats_checks.json")
    written = {"xenv/v4_overview.json", "xenv/v4_stats_checks.json"}
    # 'current': written by a subsection that ran (II.9a always), or by this cell; a file of another session is not
    current = {"II.9a": True, **{k: v["ran"] for k, v in subs.items()}}
    outputs = {sec: [{"path": rel, "exists": bool(rel in written or _xv4h_path(rel, table_dir, fig_dir).exists()),
                      "current": bool((rel in written or figures.get(Path(rel).stem) == "written")
                                      if sec == "II.9h" else current[sec])} for rel in rels]
               for sec, rels in XV4H_OUTPUTS.items()}
    keys = ("switch", "enabled", "required", "status", "summary_source", "ran", "draws_complete", "invariants_pass",
            "code_hash", "elapsed_s")
    overview = {"status": status, "code_hash": XV4_STATS_CODE_HASH,
                "subsections": {k: {c: v[c] for c in keys} for k, v in subs.items()}, "missing": missing,
                "switched_off": off, "draws_complete": draws_complete, "checks_passes": passes, "headline": headline,
                "figures": figures, "outputs": outputs, "elapsed_s": time.perf_counter() - t0}
    save_json(overview, table_dir / "v4_overview.json")
    return {"overview": overview, "checks": checks, "subsections": subs, "figures": figures,
            "tables": {k: tables[k] for k in XV4H_KEEP_TABLES},
            "outputs": [o["path"] for v in outputs.values() for o in v if o["exists"] and o["current"]]}


# Objects whose source changes what II.9h reports, and the module constants they read (by value).
XV4_STATS_CODE = [_xv4h_path, xv4h_summary, xv4h_table, _xv4h_all, _xv4h_num, xv4h_cs_order_counts,
                  xv4h_boot_order_counts, xv4h_invariants, xv4h_headline_robust, xv4h_headline_cs, xv4h_headline_boot,
                  xv4h_headline_sens, xv4h_headline_lift, xv4h_headline_clean, _xv4h_bars, _xv4h_pct_limit,
                  _xv4h_finish, xs_fig_v4_confidence, xs_fig_v4_bootstrap, _xv4h_cells, xv4h_run]
XV4_STATS_CODE_HASH = obj_hash({
    "common": XV4_COMMON_CODE_HASH, "code": code_hash(*XV4_STATS_CODE),
    "constants": {"subsections": {k: list(v) for k, v in XV4H_SUBSECTIONS.items()},
                  "tables": {k: list(v) for k, v in XV4H_TABLES.items()}, "keep_tables": list(XV4H_KEEP_TABLES),
                  "recount_tables": XV4H_RECOUNT_TABLES, "outputs": {k: list(v) for k, v in XV4H_OUTPUTS.items()},
                  "cs_fig": [list(x) for x in XV4H_CS_FIG],
                  "boot_fig_intervals": [list(x) for x in XV4H_BOOT_FIG_INTERVALS],
                  "boot_fig_variants": list(XV4H_BOOT_FIG_VARIANTS), "robust_keys": list(XV4H_ROBUST_KEYS),
                  "cs_keys": list(XV4H_CS_KEYS), "boot_keys": list(XV4H_BOOT_KEYS), "sens_keys": list(XV4H_SENS_KEYS),
                  "clean_checks": list(XV4H_CLEAN_CHECKS), "lift_slacks": list(XV4H_LIFT_SLACKS),
                  "lift_range_slacks": list(XV4H_LIFT_RANGE_SLACKS), "order_tol": XV4H_ORDER_TOL,
                  "read_errors": [e.__name__ for e in XCACHE_READ_ERRORS]}})

XV4_STATS = xv4h_run(XENV_TABLE_DIR, XENV_FIG_DIR)
print("II.9h code hash:", XV4_STATS_CODE_HASH[:16], "| status:", XV4_STATS["overview"]["status"])
display(pd.DataFrame([{"subsection": k, **{c: v[c] for c in ("switch", "enabled", "required", "status",
                                                              "summary_source", "draws_complete", "invariants_pass")}}
                      for k, v in XV4_STATS["subsections"].items()]))
_t = XV4_STATS["tables"]["cs_table"]
if len(_t):
    display(_t[_t["unit"].astype(str).str.startswith("pooled:") | (_t["unit"].nunique() == 1)][
        [c for c in ("unit", "scale", "test", "draws", "share", "error_rate", "share_sens", "error_rate_sens",
                     "draws_any_wrong", "coverage") if c in _t.columns]].round(4))
_t = XV4_STATS["tables"]["boot_coverage"]
if len(_t):
    display(_t[_t["cell"].astype(str).str.startswith("pooled:") & _t["interval"].isin(["nhb", "free"])][
        [c for c in ("cell", "boot_variant", "interval", "draws", "coverage_all", "coverage_ppos", "coverage_pzero",
                     "certified_share", "error_rate") if c in _t.columns]].round(4))
del _t
print("II.9g clean check:", json.dumps(XV4_STATS["overview"]["headline"]["II.9g"], default=_json_default))
print("Figures:", XV4_STATS["figures"], "| draws complete:", XV4_STATS["overview"]["draws_complete"],
      "| checks pass:", XV4_STATS["checks"]["passes"])
if not XV4_STATS["checks"]["passes"]:
    raise AssertionError(f"II.9 invariant checks failed: {XV4_STATS['checks']['invariants_pass']}, recounted "
                         f"{XV4_STATS['checks']['order_counts']} (xenv/v4_stats_checks.json)")
