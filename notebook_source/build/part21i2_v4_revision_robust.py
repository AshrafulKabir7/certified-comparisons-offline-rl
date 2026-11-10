# %% [markdown]
# ### II.9b Robustness of certified comparisons: certificates without NHB, pair types, abstention, hidden benefit
#
# A certificate (Corollary 2) orders two policies through their brackets. The **NHB certificate**
# J_term(i) > J_W(j) assumes no hidden benefit (NHB) for the losing policy j. The **NHB-free certificate**
# J_term(i) > J_term(j) + p_j (max_T R − v_min) assumes nothing about the unsupported rows (Theorem 1(iv)). This section
# re-reads the stored exact results of every Part II simulator cell (II.6 and II.8, aggregated cells included) and of
# ICU-Sepsis, and scores both certificates. It splits them by pair type, merges identical policies, compares them with a
# single-rule abstention baseline at the same coverage, and measures how much hidden benefit would reverse each claim.
# It also scores the II.8g bootstrap intervals over **all** default policies. It simulates nothing: it is the notebook
# form of `paper_scripts/make_robustness.py` of the supplementary material, with the same definitions. The value tolerance is 1e-9. A policy
# "can leave the support" if p > 1e-12. A claim is an error if the truth reverses it or ties.
#
# **Inputs.** `XRESULTS` (II.6) and `XRESULTS_V3` (II.8), or `xenv/xenv_results_all.csv.gz` and
# `xenv/v3_results_all.csv.gz` when the globals are empty. The bootstrap rows come from `XV3_BOOT_RESULTS`, or
# `xenv/v3_bootstrap_results.csv.gz`. Only default, non-reference policies are used, as in II.8i. Every stored draw is
# analysed: `XV4CFG.max_draws` limits only the subsections that simulate.
#
# **Per (cell, draw).**
# 1. *Target sets.* `true_rows` = (J_term, J_W, p) = (`true_rows_terminate`, `true_rows_mean`, p on `true_rows_mean`).
#    The oracle cell has no `true_rows_*` rows: its benchmark rows are the true rows, so it uses its `bench_*` targets,
#    as `make_assets.py` does. `bench` = (`bench_terminate`, `bench_mean`, p on `bench_mean`). The truth is `true`.
#    ICU-Sepsis has `bench` only and no truth. If some policy is not proper on a needed target, that target set of the
#    draw is skipped and counted (II.8i skips such draws too).
# 2. *Claims.* `x_v4_pair_claims` with v_range = v_max − v_min (Gumbel 2, Garnet 1, ICU-Sepsis 1). It counts the NHB
#    and NHB-free certificates, the NHB-only claims and the testable claims (losing policy with p > 0). It also counts
#    the point claims of each rule and the margin rules: claim the k largest gaps of one rule, with k = the number of
#    certificates (abstention at matched coverage). Pair types: AA (neither policy can leave the support), AU (one),
#    UU (both). The counts are repeated after merging policies with identical values on every available target
#    (`x_v4_distinct`). Within a draw, policies are ordered by label as in `make_robustness.py`; this fixes how ties of
#    the margin rules are broken.
# 3. *Hidden-benefit sensitivity (κ).* E_j = p_j v_range − (J_W(j) − J_term(j)) ≥ 0 is the largest possible excess of
#    π_j's value over J_W(j) (Theorem 1(iv), attained). The κ-certificate is J_term(i) > J_W(j) + κ E_j: κ = 0 is the
#    NHB certificate and κ = 1 the NHB-free one. It is computed as J_term(i) > (1 − κ) J_W(j) + κ (J_term(j) +
#    p_j v_range), so both ends reproduce the two certificates exactly. Each claim is counted for every κ in
#    `XV4CFG.kappa_grid`. Every NHB-only claim also gets its breakdown fraction κ* = (J_term(i) − J_W(j)) / E_j and
#    its margin J_term(i) − J_W(j). E_j > 0 for every NHB-only claim, so κ* is finite. A claim is NHB-free iff κ* > 1,
#    and κ* ∈ (0, 1] for NHB-only claims, both up to the 1e-9 tolerance: a claim within 1e-9 of the NHB-free end stays
#    NHB-only, so κ* can exceed 1 by at most 1e-9 / E_j (as in `make_robustness.py`).
# 4. *Admissible-pair gaps* (`bench`, with truth). No completion can change an NHB-certified claim between two policies
#    with p = 0, so its errors are estimation and construction error. These claims are binned by the normalised gap
#    (J_term(i) − J_W(j)) / v_range.
#
# **Aggregates.** Units are every cell and the pools `gumbel_estimated` (the II.6 `best` cells), `gumbel_oracle`,
# `garnet_all`, `garnet_h-0.5`, `garnet_h0.0`, `garnet_h_ge_0.5`, `icu`, and one pool per II.8 kind (`v3_physician`,
# `v3_aggregated`, `v3_garnet-random`). Each unit is reported per target set, policy set (all / distinct) and pair type.
# Each row gives claims, errors, shares and error rates of every kind, and the draws with at least one certificate or
# one wrong certificate. Each row also gives a draw-clustered percentile interval of the `cert_nhb`, `cert_free` and
# `margin_*` error rates: 5,000 resamples of the draws within each cell, stream
# `("xenv", "v4", "robust-ci", unit, target set, kind)`. The same resamples are shared by every policy set and pair
# type of a unit. A resample without claims counts as rate 0, as in `make_robustness.py`. The **completion-sensitive
# headline** is in every row: the certified share and error among AU + UU pairs, for NHB and NHB-free.
#
# **II.8g bootstrap rows** (B = `XV3BOOT.replicates`, proper replicates only, quantiles `XV3BOOT.quantiles`). Every
# default policy gets four intervals:
#
# * NHB: [q_lo(J_term^b), q_hi(J_mean^b)];
# * NHB-free: [q_lo(J_term^b), q_hi(J_term^b + p^b v_range)];
# * the min / max versions of both.
#
# The point intervals [J_term, J_W] and [J_term, J_term + p v_range] of the original benchmark are scored beside them.
# Per cell and pooled, the table reports:
#
# * coverage of J_true over all default policies, and over p > 0 and p = 0 (p from replicate −1);
# * certified share and errors, also among AU + UU pairs;
# * draws with any wrong certificate, and draws in which every policy is covered;
# * margin rules at the same number of claims, on the point values.
#
# The two ends of a bootstrap interval are quantiles over different sets of proper replicates, so an interval can be
# inverted (lower end above upper end). Such intervals are counted in the checks and reported, not asserted. The
# number of replicates is read from the rows.
#
# **Invariants asserted.** They are counted, the outputs are written, and then the cell raises, as II.8i does.
#
# * Values lie in [v_min, v_max].
# * 0 ≤ J_W − J_term ≤ p v_range on both target sets.
# * J_term ≤ J_true ≤ J_term + p v_range on the true rows (Theorem 1(i), (iv)).
# * J_W = J_term on the true rows when p = 0.
# * The two rules (and the truth) share p.
# * κ = 0 and κ = 1 reproduce the NHB and NHB-free counts.
# * The per-draw NHB and point counts equal those of the II.8i code (`xv3_certified_draws`) on the same rows, and the
#   `XV3_STATS` table when it is in memory.
# * On the bootstrap rows: the per-replicate invariants of II.8g, and the NHB interval of II.8i.
#
# The pooled counts are also compared with the published values (`facts_robust.json`). For `gumbel_estimated` these
# are: true rows 152,237 NHB claims, 0 errors, 110,330 NHB-free, 41,907 NHB-only, 43,346 testable; bench 151,338
# claims, 15,981 errors. This comparison is asserted only for the full paper design: `RUN_MODE == "paper"`,
# `COMPUTE_PROFILE == "full"` and all 180 draws present. A mismatch there is counted like the invariants above. The
# `garnet_all` values are compared and reported only. Error rates and coverage are reported, never asserted.
#
# | Output | Content |
# |---|---|
# | `xenv/v4_robust_pairs.csv.gz` | per draw × target set × policy set × pair type: claims and errors of every kind |
# | `xenv/v4_robust_summary.csv` | per cell and pool × target set × policy set × pair type: shares, error rates, draw-clustered intervals, draws with a (wrong) certificate, completion-sensitive headline |
# | `xenv/v4_robust_gapbins.csv` | per cell and pool × gap bin: NHB-certified benchmark claims between two policies with p = 0, and their error rate |
# | `xenv/v4_robust_kappa.csv` | per cell and pool × target set × κ: claims, share of pairs and of AU + UU pairs, errors; quantiles of κ* and of the margin of the NHB-only claims |
# | `xenv/v4_v3boot_allpolicy.csv` | II.8g bootstrap, per cell and pooled × interval (point / bootstrap / minmax) × bound (NHB / NHB-free): coverage over all, p > 0 and p = 0 policies; certificates; draw-level counts; margin baselines |
# | `xenv/v4_robust_summary.json` | status, code hash, checks, self-check, headline numbers, κ block, gap bins, bootstrap headline, timing |
#
# `XV4_ROBUST` holds these tables, plus the per-draw κ counts and the per-policy and per-draw bootstrap tables.
# `XV4_ROBUST_SUMMARY` holds the summary json.
#
# **Expected cost on Colab.** This is an estimate, not a measurement. The analysis is pure numpy / pandas. Reading the
# result rows from the CSV files takes 10–20 s; nothing is read when the globals are in memory. On synthetic inputs of
# the same shape (local harness, M1 CPU), 300 draws of 50 policies were scored in 1.3 s, and 290 summary rows with
# their 5,000-resample intervals took 1.4 s (the intervals are matrix products of resample counts). Scaled to the
# roughly 700 draws of Part II and a Colab CPU two to three times slower, the claims take 5–15 s. The summary,
# κ and gap tables take 5–15 s, the II.8i recount 5–10 s and the bootstrap rows under 5 s. The whole cell takes about
# 0.5–1 min, with well under 1 GB of extra memory.

# %%
# ============================================================
# II.9b Robustness of certified comparisons (stored results, no simulation)
# ============================================================
EXECLOG.mark("II.9 Second-revision experiments: robustness of certified comparisons")
XV4R_SWITCH = "xenv_v4_revision_robust"
XV4R_VRANGE = {"gumbel": (-1.0, 1.0), "garnet": (0.0, 1.0), "icu": (0.0, 1.0)}   # [v_min, v_max]; r_term = v_min
assert XV4R_VRANGE == XV3_VRANGE, (XV4R_VRANGE, XV3_VRANGE)
XV4R_TARGETS = ("bench_mean", "bench_terminate", "true", "true_rows_mean", "true_rows_terminate")
XV4R_TARGET_SETS = {"true_rows": ("true_rows_terminate", "true_rows_mean", "p_true_rows_mean"),
                    "bench": ("bench_terminate", "bench_mean", "p_bench_mean")}   # (J_term, J_W, p) columns
XV4R_TYPES = ("all", "AA", "AU", "UU")
XV4R_CI_KINDS = ("cert_nhb", "cert_free", "margin_mean", "margin_term", "margin_mid", "margin_mean_free")
XV4R_CI_RESAMPLES = 5_000
XV4R_CI_LEVELS = (0.025, 0.975)
XV4R_KSTAR_Q = (0.10, 0.25, 0.50, 0.75, 0.90)
XV4R_GAP_BINS = (0.0, 0.0025, 0.005, 0.01, 0.025, 0.05, np.inf)
XV4R_RESULT_COLUMNS = ["cell", "family", "draw", "policy", "is_default", "is_reference", "target", "status", "return",
                       "prob_any_violation"]
XV4R_BOOT_COLUMNS = ["cell", "family", "draw", "policy", "replicate", "target", "status", "return",
                     "prob_any_violation"]
XV4R_BOOT_INTERVALS = {("point", "nhb"): ("point_lo", "point_up"),             # (interval, bound): (lower, upper)
                       ("point", "free"): ("point_lo", "point_up_free"),
                       ("bootstrap", "nhb"): ("boot_lo", "boot_up"),
                       ("bootstrap", "free"): ("boot_lo", "boot_up_free"),
                       ("minmax", "nhb"): ("minmax_lo", "minmax_up"),
                       ("minmax", "free"): ("minmax_lo", "minmax_up_free")}
# Published pooled counts (paper_assets_expected/facts_robust.json, 30 draws per cell). XV4R_KNOWN is asserted in the full paper
# design (XV4R_KNOWN_DRAWS draws present); XV4R_KNOWN_REPORTED is compared and reported only.
XV4R_KNOWN = {"true_rows:gumbel_estimated": {"cert_nhb_claims": 152_237, "cert_nhb_errors": 0,
                                             "cert_free_claims": 110_330, "cert_nhb_only_claims": 41_907,
                                             "cert_testable_claims": 43_346},
              "bench:gumbel_estimated": {"cert_nhb_claims": 151_338, "cert_nhb_errors": 15_981}}
XV4R_KNOWN_DRAWS = {"gumbel_estimated": 180, "garnet_all": 240}
XV4R_KNOWN_REPORTED = {"true_rows:gumbel_estimated": {"pairs": 220_500, "cert_free_errors": 0,
                                                      "point_mean_errors": 12_493, "point_term_errors": 4_851},
                       "bench:gumbel_estimated": {"pairs": 220_500, "cert_free_claims": 110_039,
                                                  "cert_free_errors": 15_084, "point_mean_errors": 30_436,
                                                  "point_term_errors": 20_329},
                       "true_rows:garnet_all": {"pairs": 294_000, "cert_nhb_claims": 249_930, "cert_nhb_errors": 2_396,
                                                "cert_free_claims": 225_494, "cert_free_errors": 0,
                                                "cert_nhb_only_claims": 24_436, "cert_testable_claims": 64_325},
                       "bench:garnet_all": {"pairs": 294_000, "cert_nhb_claims": 253_620, "cert_nhb_errors": 23_310,
                                            "cert_free_claims": 234_857, "cert_free_errors": 19_339}}


def _xv4r_rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def _xv4r_vrange(family: str) -> float:
    lo, hi = XV4R_VRANGE[str(family)]
    return float(hi - lo)


def xv4r_cell_group(cell: str, family: str, source: str, kind_of: Dict[str, str]) -> Tuple[str, float]:
    """(group, Garnet harm) of a Part II cell: 'gumbel_estimated', 'gumbel_oracle', 'garnet' or 'icu' for an II.6 cell,
    'v3_<kind>' for an II.8 cell (kind from X_V3_CELLS, else from the name)."""
    tok = str(cell).split("|")
    if source == "II.8":
        kind = kind_of.get(str(cell)) or ("aggregated" if tok[1].startswith("agg-") else
                                          "garnet-random" if family == "garnet" else "physician")
        return f"v3_{kind}", float("nan")
    if family == "icu":
        return "icu", float("nan")
    if family == "gumbel":
        assert tok[1] in ("best", "oracle"), cell
        return ("gumbel_estimated" if tok[1] == "best" else "gumbel_oracle"), float("nan")
    assert family == "garnet" and tok[1].startswith("h"), cell
    return "garnet", float(tok[1][1:])


def xv4r_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Long rows of the II.6 and II.8 cells on the five targets (XRESULTS / XRESULTS_V3, else their CSV files; column
    `source` = 'II.6' / 'II.8') and the II.8g bootstrap rows (XV3_BOOT_RESULTS, else its CSV file)."""
    info: Dict[str, Any] = {}
    frames = []
    for src, name, fname in (("II.6", "XRESULTS", "xenv_results_all.csv.gz"),
                             ("II.8", "XRESULTS_V3", "v3_results_all.csv.gz")):
        df, path = globals().get(name), XENV_TABLE_DIR / fname
        if df is not None and len(df):
            where, df = f"{name} (in memory)", df[XV4R_RESULT_COLUMNS]
        elif path.exists():
            where = f"xenv/{fname}"
            df = pd.read_csv(path, usecols=XV4R_RESULT_COLUMNS, float_precision="round_trip", low_memory=False)
        else:
            info[src] = {"source": "missing", "rows": 0, "cells": 0}
            continue
        df = df[df["target"].isin(XV4R_TARGETS)].assign(source=src)
        info[src] = {"source": where, "rows": int(len(df)), "cells": int(df["cell"].nunique())}
        frames.append(df)
    res = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=[*XV4R_RESULT_COLUMNS, "source"]))
    if len(frames) == 2:
        clash = set(frames[0]["cell"]) & set(frames[1]["cell"])
        assert not clash, f"II.6 and II.8 cells share names: {sorted(clash)}"
    boot, path = globals().get("XV3_BOOT_RESULTS"), XENV_TABLE_DIR / "v3_bootstrap_results.csv.gz"
    if boot is not None and len(boot):
        info["II.8g"], boot = "XV3_BOOT_RESULTS (in memory)", boot[XV4R_BOOT_COLUMNS].copy()
    elif path.exists():
        info["II.8g"] = "xenv/v3_bootstrap_results.csv.gz"
        boot = pd.read_csv(path, usecols=XV4R_BOOT_COLUMNS, float_precision="round_trip", low_memory=False)
    else:
        info["II.8g"], boot = "missing", pd.DataFrame(columns=XV4R_BOOT_COLUMNS)
    info["II.8g_rows"] = int(len(boot))
    return res, boot, info


def xv4r_wide(res: pd.DataFrame, kind_of: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Per (cell, draw, policy), default non-reference policies only: J and P(any unsupported decision) on the five
    targets (NaN unless the status is ok; p columns 'p_<target>'), family, source, group and Garnet harm. The oracle
    cell's true-rows columns are its benchmark columns. Sorted by (cell, draw, policy)."""
    r = res[xs_as_bool(res["is_default"]) & ~xs_as_bool(res["is_reference"])].copy()
    r["draw"] = pd.to_numeric(r["draw"]).astype(np.int64)
    k = ["cell", "draw", "policy"]
    dup = r.duplicated(k + ["target"])
    assert not dup.any(), f"{int(dup.sum())} duplicated (cell, draw, policy, target) rows"
    ok = r["status"].astype(str) == "ok"
    r["J"] = pd.to_numeric(r["return"], errors="coerce").where(ok)
    r["pv"] = pd.to_numeric(r["prob_any_violation"], errors="coerce").where(ok)
    idx = r.set_index(k + ["target"])
    J = idx["J"].unstack("target").reindex(columns=list(XV4R_TARGETS))
    Pv = idx["pv"].unstack("target").reindex(columns=list(XV4R_TARGETS)).add_prefix("p_")
    W = r.drop_duplicates(k).set_index(k)[["family", "source"]].join(J).join(Pv)
    cells = r.drop_duplicates("cell")[["cell", "family", "source"]]
    grp = {c: xv4r_cell_group(c, f, s, kind_of) for c, f, s in cells.itertuples(index=False)}
    cl = W.index.get_level_values("cell")
    W["group"] = cl.map(lambda c: grp[c][0]).to_numpy()
    W["harm"] = cl.map(lambda c: grp[c][1]).to_numpy(dtype=float)
    orc = (W["group"] == "gumbel_oracle").to_numpy()
    W["raw_true_rows"] = W[["true_rows_mean", "true_rows_terminate"]].notna().all(axis=1)
    for a, b in (("true_rows_mean", "bench_mean"), ("true_rows_terminate", "bench_terminate"),
                 ("p_true_rows_mean", "p_bench_mean"), ("p_true_rows_terminate", "p_bench_terminate")):
        assert W.loc[orc, a].isna().all(), f"the oracle cell has no {a} target (its benchmark rows are the true rows)"
        W.loc[orc, a] = W.loc[orc, b]
    checks = {"rows": int(len(r)), "nonok_rows": int((~ok).sum()), "policy_rows": int(len(W)),
              "cells": int(len(grp)), "draws": int(W.reset_index()[["cell", "draw"]].drop_duplicates().shape[0]),
              "oracle_policy_rows": int(orc.sum())}
    return W.sort_index(), checks


def xv4r_theorem_counts(W: pd.DataFrame) -> Dict[str, int]:
    """Theorem facts on the stored exact values (counts of violations, tolerance XV4_TOL; all must be 0)."""
    vr = W["family"].map(_xv4r_vrange).to_numpy(dtype=float)
    out = {"values_outside_value_range": 0}
    for fam, (lo, hi) in XV4R_VRANGE.items():
        v = W.loc[W["family"] == fam, list(XV4R_TARGETS)].to_numpy(dtype=float)
        out["values_outside_value_range"] += int(((v < lo - XV4_TOL) | (v > hi + XV4_TOL)).sum())
    tr = W[["true", "true_rows_mean", "true_rows_terminate"]].notna().all(axis=1).to_numpy()
    bn = W[["bench_mean", "bench_terminate"]].notna().all(axis=1).to_numpy()
    raw = W["raw_true_rows"].to_numpy(dtype=bool) & tr
    jt, tt, tm = (W[c].to_numpy(dtype=float) for c in ("true", "true_rows_terminate", "true_rows_mean"))
    bt, bm = W["bench_terminate"].to_numpy(dtype=float), W["bench_mean"].to_numpy(dtype=float)
    pt, pb = W["p_true_rows_mean"].to_numpy(dtype=float), W["p_bench_mean"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        out["negative_width_true_rows"] = int((tr & (tm - tt < -XV4_TOL)).sum())
        out["width_above_bound_true_rows"] = int((tr & (tm - tt > pt * vr + XV4_TOL)).sum())
        out["negative_width_bench"] = int((bn & (bm - bt < -XV4_TOL)).sum())
        out["width_above_bound_bench"] = int((bn & (bm - bt > pb * vr + XV4_TOL)).sum())
        out["lower_bound_violations_true_rows"] = int((tr & (jt < tt - XV4_TOL)).sum())
        out["nhb_free_upper_violations_true_rows"] = int((tr & (jt > tt + pt * vr + XV4_TOL)).sum())
        out["nonzero_width_admissible_true_rows"] = int((tr & (pt <= XV4_PTOL) & (np.abs(tm - tt) > XV4_TOL)).sum())
        out["rule_p_difference_bench"] = int(
            (bn & (np.abs(pb - W["p_bench_terminate"].to_numpy(dtype=float)) > XV4_TOL)).sum())
        out["rule_p_difference_true_rows"] = int(
            (raw & (np.abs(pt - W["p_true_rows_terminate"].to_numpy(dtype=float)) > XV4_TOL)).sum())
        out["truth_p_difference_true_rows"] = int(
            (raw & (np.abs(pt - W["p_true"].to_numpy(dtype=float)) > XV4_TOL)).sum())
    return out


def xv4r_kappa_claims(lo: np.ndarray, up: np.ndarray, p: np.ndarray, v_range: float, jt: Optional[np.ndarray],
                      grid: Sequence[float]) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """kappa-certificates lo_i > (1 - kappa) up_j + kappa (lo_j + p_j v_range) (+ XV4_TOL) for every kappa of `grid`,
    i.e. J_term(i) > J_W(j) + kappa E_j with E_j = p_j v_range - (J_W(j) - J_term(j)): kappa = 0 is cert_nhb and
    kappa = 1 cert_free of x_v4_pair_claims (the convex combination makes both ends exact). Returns the counts per kappa
    (all pairs and completion-sensitive AU + UU pairs; errors when `jt` is given) and, for the NHB-only claims, the
    breakdown fraction kappa* = (J_term(i) - J_W(j)) / E_j and the margin J_term(i) - J_W(j)."""
    lo, up, p = (np.asarray(x, dtype=float) for x in (lo, up, p))
    n = len(lo)
    off = ~np.eye(n, dtype=bool)
    u = p > XV4_PTOL
    cs = off & (u[:, None] | u[None, :])
    free = lo + p * float(v_range)
    nhb = _x_v4_gt(lo, up) & off
    wrong = None if jt is None else ~_x_v4_gt(np.asarray(jt, dtype=float), np.asarray(jt, dtype=float))
    base = {"n_policies": n, "pairs": int(off.sum() // 2), "pairs_cs": int(cs.sum() // 2),
            "nhb_claims": int(nhb.sum())}
    rows = []
    for kap in grid:
        C = _x_v4_gt(lo, (1.0 - float(kap)) * up + float(kap) * free) & off
        rows.append({**base, "kappa": float(kap), "claims": int(C.sum()), "claims_cs": int((C & cs).sum()),
                     "errors": np.nan if wrong is None else int((C & wrong).sum()),
                     "errors_cs": np.nan if wrong is None else int((C & cs & wrong).sum())})
    i, j = np.nonzero(nhb & ~_x_v4_gt(lo, free))
    excess = free - up
    assert (excess[j] > 0).all(), "an NHB-only claim needs a positive largest excess E_j of the losing policy"
    margin = lo[i] - up[j]
    return rows, margin / excess[j], margin


def xv4r_gap_claims(lo: np.ndarray, up: np.ndarray, p: np.ndarray, v_range: float, jt: np.ndarray) -> pd.DataFrame:
    """NHB-certified claims between two policies that cannot leave the support (p <= XV4_PTOL): normalised gap
    (J_term(i) - J_W(j)) / v_range, and whether the truth reverses or ties (`wrong`)."""
    lo, up, p, jt = (np.asarray(x, dtype=float) for x in (lo, up, p, jt))
    a = p <= XV4_PTOL
    C = _x_v4_gt(lo, up) & a[:, None] & a[None, :]
    np.fill_diagonal(C, False)
    i, j = np.nonzero(C)
    return pd.DataFrame({"gap": (lo[i] - up[j]) / float(v_range), "wrong": ~(jt[i] > jt[j] + XV4_TOL)})


def xv4r_draw(d: pd.DataFrame, grid: Sequence[float]) -> Dict[str, Any]:
    """Claims of one draw (rows of xv4r_wide, ordered by policy): x_v4_pair_claims on each target set for all and for
    distinct policies, the kappa counts, kappa* and margins of the NHB-only claims, and the admissible-pair benchmark
    claims. A target set absent from the cell is left out; one with a non-proper policy is skipped and listed."""
    fam = str(d["family"].iloc[0])
    vr = _xv4r_vrange(fam)
    out: Dict[str, Any] = {"pairs": [], "kappa": [], "kstar": {}, "gaps": None, "skipped": []}
    truth = fam != "icu"
    if truth and d["true"].isna().any():
        out["skipped"] = [tf for tf, cols in XV4R_TARGET_SETS.items() if d[list(cols)].notna().any().any()]
        return out
    jt = d["true"].to_numpy(dtype=float) if truth else None
    avail = [t for t in XV4R_TARGETS if d[t].notna().all()]
    keep = x_v4_distinct(d[avail].to_numpy(dtype=float)) if avail else np.ones(len(d), dtype=bool)
    for tf, (lc, uc, pc) in XV4R_TARGET_SETS.items():
        sub = d[[lc, uc, pc]]
        if sub.isna().all().all():
            continue
        if sub.isna().any().any():
            out["skipped"].append(tf)
            continue
        lo, up, p = (sub[c].to_numpy(dtype=float) for c in (lc, uc, pc))
        for dist in (False, True):
            m = keep if dist else np.ones(len(d), dtype=bool)
            for r in x_v4_pair_claims(lo[m], up[m], p[m], vr, None if jt is None else jt[m]):
                out["pairs"].append({"targets": tf, "distinct": dist, **r})
        krows, kstar, margin = xv4r_kappa_claims(lo, up, p, vr, jt, grid)
        out["kappa"] += [{"targets": tf, **r} for r in krows]
        out["kstar"][tf] = (kstar, margin)
        if tf == "bench" and truth:
            out["gaps"] = xv4r_gap_claims(lo, up, p, vr, jt)
    return out


def xv4r_claims(W: pd.DataFrame, grid: Sequence[float]
                ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
                           pd.DataFrame, Dict[str, int]]:
    """xv4r_draw over every (cell, draw) of W: per-draw claim rows, kappa rows, the NHB-only kappa* and margins per
    (cell, target set), the admissible-pair gap claims and the skipped (cell | target set) counts."""
    pair_rows: List[dict] = []
    k_rows: List[dict] = []
    gap_frames: List[pd.DataFrame] = []
    parts: Dict[Tuple[str, str], List[Tuple[np.ndarray, np.ndarray]]] = {}
    skipped: Counter = Counter()
    cells = list(pd.unique(W.index.get_level_values("cell")))
    for ci, cell in enumerate(cells, 1):
        t_cell = time.perf_counter()
        g = W.xs(cell, level="cell", drop_level=False)
        draws = g.index.get_level_values("draw")
        for draw, d in g.groupby(level="draw", sort=True):
            meta = {"cell": cell, "draw": int(draw), "family": str(d["family"].iloc[0]),
                    "source": str(d["source"].iloc[0]), "group": str(d["group"].iloc[0])}
            o = xv4r_draw(d, grid)
            pair_rows += [{**meta, **r} for r in o["pairs"]]
            k_rows += [{**meta, **r} for r in o["kappa"]]
            for tf, v in o["kstar"].items():
                parts.setdefault((cell, tf), []).append(v)
            if o["gaps"] is not None:
                gap_frames.append(o["gaps"].assign(cell=cell, draw=int(draw)))
            for tf in o["skipped"]:
                skipped[f"{cell}|{tf}"] += 1
        print(f"  [{ci}/{len(cells)}] {cell}: {draws.nunique()} draws, {len(g) // max(draws.nunique(), 1)} policies "
              f"per draw, {time.perf_counter() - t_cell:.1f} s")
    R = pd.DataFrame(pair_rows)
    K = pd.DataFrame(k_rows)
    KS = {key: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for key, v in parts.items()}
    G = (pd.concat(gap_frames, ignore_index=True) if gap_frames
         else pd.DataFrame({"gap": pd.Series(dtype=float), "wrong": pd.Series(dtype=bool),
                            "cell": pd.Series(dtype=str), "draw": pd.Series(dtype=np.int64)}))
    return R, K, KS, G, dict(skipped)


def xv4r_units(W: pd.DataFrame, order: Sequence[str]) -> Tuple[List[Tuple[str, str, List[str]]], Dict[str, List[str]]]:
    """Analysis units: every cell (driver order), then the pools of the paper and one pool per II.8 kind."""
    info = W.reset_index().drop_duplicates("cell").set_index("cell")[["group", "harm"]]
    cells = [c for c in order if c in info.index]
    grp, harm = info["group"], info["harm"]
    gar = grp == "garnet"
    masks = {"gumbel_estimated": grp == "gumbel_estimated", "gumbel_oracle": grp == "gumbel_oracle",
             "garnet_all": gar, "garnet_h-0.5": gar & np.isclose(harm, -0.5),
             "garnet_h0.0": gar & np.isclose(harm, 0.0), "garnet_h_ge_0.5": gar & (harm >= 0.5), "icu": grp == "icu"}
    masks.update({g: grp == g for g in sorted(set(grp)) if g.startswith("v3_")})
    pools = {name: [c for c in cells if bool(m.get(c, False))] for name, m in masks.items()}
    pools = {name: cs for name, cs in pools.items() if cs}
    return [(c, "cell", [c]) for c in cells] + [(name, "pool", cs) for name, cs in pools.items()], pools


def xv4r_summary_row(d: pd.DataFrame, kinds: Sequence[str]) -> Dict[str, Any]:
    """Pooled claims, shares and error rates of per-draw rows of one (target set, policy set, pair type), and the
    draws with at least one (wrong) certificate."""
    assert not d.duplicated(["cell", "draw"]).any(), "one row per (cell, draw) expected"
    s = d.sum(numeric_only=True)
    pairs = int(s["pairs"])
    err = len(d) > 0 and all(f"{k}_errors" in d.columns and d[f"{k}_errors"].notna().all() for k in kinds)
    o: Dict[str, Any] = {"draws": int(len(d)), "policies_per_draw": float(d["n_policies"].median()), "pairs": pairs}
    for nm in kinds:
        cl = int(s[f"{nm}_claims"])
        o[f"{nm}_claims"], o[f"{nm}_share"] = cl, _xv4r_rate(cl, pairs)
        o[f"{nm}_errors"] = int(s[f"{nm}_errors"]) if err else np.nan
        o[f"{nm}_error_rate"] = _xv4r_rate(s[f"{nm}_errors"], cl) if err else np.nan
    for nm in ("cert_nhb", "cert_free"):
        o[f"draws_with_{nm}"] = int((d[f"{nm}_claims"] > 0).sum())
        o[f"draws_with_{nm}_error"] = int((d[f"{nm}_errors"] > 0).sum()) if err else np.nan
    return o


def xv4r_cs_headline(d: pd.DataFrame) -> Dict[str, Any]:
    """Completion-sensitive headline from the AU and UU rows of one (unit, target set, policy set): certified share
    and error rate among pairs in which at least one policy can leave the support."""
    s = d.sum(numeric_only=True)
    pairs = int(s["pairs"]) if len(d) else 0
    o: Dict[str, Any] = {"cs_pairs": pairs}
    for nm in ("cert_nhb", "cert_free"):
        cl = int(s[f"{nm}_claims"]) if len(d) else 0
        err = len(d) > 0 and f"{nm}_errors" in d.columns and d[f"{nm}_errors"].notna().all()
        o[f"{nm}_cs_claims"], o[f"{nm}_cs_share"] = cl, _xv4r_rate(cl, pairs)
        o[f"{nm}_cs_errors"] = int(s[f"{nm}_errors"]) if err else np.nan
        o[f"{nm}_cs_error_rate"] = _xv4r_rate(s[f"{nm}_errors"], cl) if err else np.nan
    return o


def xv4r_resample_matrix(cells: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_draws) multiplicities of a draw-clustered bootstrap: the draws of every cell (in order of first
    appearance) are resampled with replacement within the cell."""
    cells = np.asarray(cells)
    n_boot = int(n_boot)
    M = np.zeros((n_boot, len(cells)))
    for c in pd.unique(cells):
        ix = np.flatnonzero(cells == c)
        pick = rng.integers(0, len(ix), size=(n_boot, len(ix)))
        flat = (np.arange(n_boot)[:, None] * len(ix) + pick).ravel()
        M[:, ix] = np.bincount(flat, minlength=n_boot * len(ix)).reshape(n_boot, len(ix))
    return M


def xv4r_ratio_ci(M: np.ndarray, errors: np.ndarray, claims: np.ndarray) -> Tuple[float, float]:
    """Percentile interval (XV4R_CI_LEVELS) of the pooled error rate sum(errors) / sum(claims) over the resamples M; a
    resample without claims counts as 0 (make_robustness.py); NaN when there is no claim at all."""
    errors, claims = np.asarray(errors, dtype=float), np.asarray(claims, dtype=float)
    if not claims.sum():
        return float("nan"), float("nan")
    est = (M @ errors) / np.maximum(M @ claims, 1.0)
    return float(np.quantile(est, XV4R_CI_LEVELS[0])), float(np.quantile(est, XV4R_CI_LEVELS[1]))


def xv4r_summary_table(R: pd.DataFrame, units: Sequence[Tuple[str, str, List[str]]]) -> pd.DataFrame:
    """Summary rows per unit x target set x policy set x pair type (xv4r_summary_row, xv4r_cs_headline and the
    draw-clustered intervals of the XV4R_CI_KINDS error rates)."""
    kinds = [c[:-len("_claims")] for c in R.columns if c.endswith("_claims")]
    rows = []
    for unit, utype, cells in units:
        U = R[R["cell"].isin(cells)]
        for tf in XV4R_TARGET_SETS:
            T = U[U["targets"] == tf]
            if T.empty:
                continue
            base = T[~T["distinct"].astype(bool) & (T["type"] == "all")].set_index(["cell", "draw"]).sort_index().index
            mats = {}
            if "cert_nhb_errors" in T.columns and T["cert_nhb_errors"].notna().all():
                cl = base.get_level_values("cell").to_numpy()
                mats = {nm: xv4r_resample_matrix(cl, XV4R_CI_RESAMPLES, np_rng("xenv", "v4", "robust-ci", unit, tf, nm))
                        for nm in XV4R_CI_KINDS}
            for dist in (False, True):
                D = T[T["distinct"].astype(bool) == dist]
                cs = xv4r_cs_headline(D[D["type"].isin(["AU", "UU"])])
                for ty in XV4R_TYPES:
                    d = D[D["type"] == ty]
                    r = {"unit": unit, "unit_type": utype, "n_cells": len(cells),
                         "source": "+".join(sorted(set(d["source"]))), "targets": tf, "distinct": dist, "type": ty,
                         **xv4r_summary_row(d, kinds), **cs}
                    a = d.set_index(["cell", "draw"]).reindex(base)
                    assert a["pairs"].notna().all(), f"{unit} {tf}: draws differ between policy sets / pair types"
                    for nm in XV4R_CI_KINDS:
                        lo, hi = (xv4r_ratio_ci(mats[nm], a[f"{nm}_errors"].to_numpy(dtype=float),
                                                a[f"{nm}_claims"].to_numpy(dtype=float)) if mats
                                  else (float("nan"), float("nan")))
                        r[f"{nm}_error_rate_ci_low"], r[f"{nm}_error_rate_ci_high"] = lo, hi
                    rows.append(r)
            del mats
    return pd.DataFrame(rows)


def xv4r_kappa_table(K: pd.DataFrame, KS: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
                     units: Sequence[Tuple[str, str, List[str]]]) -> pd.DataFrame:
    """Per unit x target set x kappa: claims, share of pairs and of completion-sensitive pairs, share of the NHB
    claims, errors; plus the quantiles of kappa* and of the margin over the unit's NHB-only claims."""
    rows = []
    for unit, utype, cells in units:
        for tf in XV4R_TARGET_SETS:
            d = K[K["cell"].isin(cells) & (K["targets"] == tf)]
            if d.empty:
                continue
            ks = np.concatenate([KS[(c, tf)][0] for c in cells if (c, tf) in KS] or [np.zeros(0)])
            mg = np.concatenate([KS[(c, tf)][1] for c in cells if (c, tf) in KS] or [np.zeros(0)])
            base = {"unit": unit, "unit_type": utype, "n_cells": len(cells), "targets": tf,
                    "nhb_only_claims": int(len(ks))}
            for q in XV4R_KSTAR_Q:
                base[f"kappa_star_q{int(round(100 * q)):02d}"] = float(np.quantile(ks, q)) if len(ks) else np.nan
            for q in XV4R_KSTAR_Q:
                base[f"nhb_only_margin_q{int(round(100 * q)):02d}"] = float(np.quantile(mg, q)) if len(mg) else np.nan
            base["nhb_only_margin_mean"] = float(mg.mean()) if len(mg) else np.nan
            for kap, g in d.groupby("kappa", sort=True):
                s = g.sum(numeric_only=True)
                err = g["errors"].notna().all()
                rows.append({**base, "kappa": float(kap), "draws": int(len(g)), "pairs": int(s["pairs"]),
                             "pairs_cs": int(s["pairs_cs"]), "nhb_claims": int(s["nhb_claims"]),
                             "claims": int(s["claims"]), "share": _xv4r_rate(s["claims"], s["pairs"]),
                             "claims_cs": int(s["claims_cs"]), "cs_share": _xv4r_rate(s["claims_cs"], s["pairs_cs"]),
                             "share_of_nhb_claims": _xv4r_rate(s["claims"], s["nhb_claims"]),
                             "errors": int(s["errors"]) if err else np.nan,
                             "error_rate": _xv4r_rate(s["errors"], s["claims"]) if err else np.nan,
                             "errors_cs": int(s["errors_cs"]) if err else np.nan,
                             "error_rate_cs": _xv4r_rate(s["errors_cs"], s["claims_cs"]) if err else np.nan,
                             "draws_with_error": int((g["errors"] > 0).sum()) if err else np.nan})
    return pd.DataFrame(rows)


def xv4r_gap_table(G: pd.DataFrame, units: Sequence[Tuple[str, str, List[str]]], cells_with_truth: Sequence[str]
                   ) -> pd.DataFrame:
    """Error rate of the admissible-pair benchmark claims by normalised gap bin (XV4R_GAP_BINS), per unit; rows 'all'
    and '(0, 0.01]' are totals."""
    rows = []
    have = set(cells_with_truth)
    for unit, utype, cells in units:
        cs = [c for c in cells if c in have]
        if not cs:
            continue
        g = G[G["cell"].isin(cs)]
        tab = g.groupby(pd.cut(g["gap"], list(XV4R_GAP_BINS), right=True), observed=False)["wrong"].agg(["size", "sum"])
        lab = [(str(b), int(r["size"]), int(r["sum"])) for b, r in tab.iterrows()]
        low = g["gap"] <= 0.01
        lab += [("all", int(len(g)), int(g["wrong"].sum())),
                ("(0, 0.01]", int(low.sum()), int(g.loc[low, "wrong"].sum()))]
        rows += [{"unit": unit, "unit_type": utype, "n_cells": len(cs), "bin": b, "claims": n, "errors": e,
                  "error_rate": _xv4r_rate(e, n)} for b, n, e in lab]
    return pd.DataFrame(rows)


def xv4r_boot_intervals(boot: pd.DataFrame, q: Tuple[float, float]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Per (cell, draw, default policy) of the II.8g bootstrap: J_true, the point values and p (replicate -1), and the
    intervals over the proper replicates of each rule: NHB [q_lo(J_term^b), q_hi(J_mean^b)], NHB-free
    [q_lo(J_term^b), q_hi(J_term^b + p^b v_range)] (p^b of the terminate row), and their min / max versions.
    Also the per-replicate invariant counts (replicates where both rules are proper)."""
    b = boot[XV4R_BOOT_COLUMNS].copy()
    b["draw"] = pd.to_numeric(b["draw"]).astype(np.int64)
    b["replicate"] = pd.to_numeric(b["replicate"]).astype(np.int64)
    for c in ("return", "prob_any_violation"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["vr"] = b["family"].map(_xv4r_vrange)
    k = ["cell", "draw", "policy"]
    dup = b.duplicated(k + ["replicate", "target"])
    assert not dup.any(), f"{int(dup.sum())} duplicated bootstrap rows"
    pt = b[b["replicate"] == XV3_BOOT_POINT].set_index(k + ["target"])
    assert (pt["status"].astype(str) == "ok").all(), "replicate -1 (original benchmark and truth) must be proper"
    J0, P0 = pt["return"].unstack("target"), pt["prob_any_violation"].unstack("target")
    iv = pd.DataFrame({"true": J0["true"], "point_lo": J0["bench_terminate"], "point_up": J0["bench_mean"],
                       "p_bench": P0["bench_mean"]})
    iv = iv.join(b.drop_duplicates(k).set_index(k)[["family", "vr"]])
    iv["point_up_free"] = iv["point_lo"] + iv["p_bench"] * iv["vr"]
    rep = b[(b["replicate"] >= 0) & (b["status"].astype(str) == "ok")]
    tm = rep[rep["target"] == "bench_terminate"].assign(free=lambda x: x["return"] + x["prob_any_violation"] * x["vr"])
    gt_, gf = tm.groupby(k)["return"], tm.groupby(k)["free"]
    gm = rep[rep["target"] == "bench_mean"].groupby(k)["return"]
    cols = {"boot_lo": gt_.quantile(float(q[0])), "boot_up": gm.quantile(float(q[1])),
            "boot_up_free": gf.quantile(float(q[1])), "minmax_lo": gt_.min(), "minmax_up": gm.max(),
            "minmax_up_free": gf.max(), "n_ok_terminate": gt_.size(), "n_ok_mean": gm.size()}
    for c, s in cols.items():
        iv[c] = s.reindex(iv.index)
    iv["n_replicates"] = b[b["replicate"] >= 0].groupby(k)["replicate"].nunique().reindex(iv.index)
    for c in ("n_ok_terminate", "n_ok_mean", "n_replicates"):
        iv[c] = iv[c].fillna(0).astype(int)
    wide = rep.set_index(k + ["replicate", "target"])[["return", "prob_any_violation"]].unstack("target").reindex(
        columns=pd.MultiIndex.from_product([["return", "prob_any_violation"], ["bench_mean", "bench_terminate"]]))
    jm, jtm = wide[("return", "bench_mean")], wide[("return", "bench_terminate")]
    pm, ptm = wide[("prob_any_violation", "bench_mean")], wide[("prob_any_violation", "bench_terminate")]
    vr = wide.index.get_level_values("cell").map(dict(zip(b["cell"], b["vr"]))).to_numpy(dtype=float)
    both = (jm.notna() & jtm.notna()).to_numpy()
    w = (jm - jtm).to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        dp = np.abs((pm - ptm).to_numpy(dtype=float))
        checks = {"policy_rows": int(len(iv)), "replicate_pairs_checked": int(both.sum()),
                  "replicate_rule_p_difference": int((both & (dp > XV4_TOL)).sum()),
                  "replicate_negative_width": int((both & (w < -XV4_TOL)).sum()),
                  "replicate_width_above_bound": int((both & (w > ptm.to_numpy(dtype=float) * vr + XV4_TOL)).sum()),
                  "point_negative_width": int((iv["point_up"] < iv["point_lo"] - XV4_TOL).sum()),
                  "point_free_below_nhb_upper": int((iv["point_up_free"] < iv["point_up"] - XV4_TOL).sum()),
                  # reported, not asserted: the two ends are quantiles over different sets of proper replicates
                  "inverted_bootstrap_intervals": int((iv["boot_lo"] > iv["boot_up"] + XV4_TOL).sum()),
                  "replicates_in_rows": int(iv["n_replicates"].max()) if len(iv) else 0}
    return iv.reset_index(), checks


def xv4r_interval_claims(lo: np.ndarray, up: np.ndarray, u: np.ndarray, wrong: np.ndarray) -> Dict[str, int]:
    """Certified claims lo_i > up_j (+ XV4_TOL) of one interval type and their errors (`wrong`: truth reverses or ties),
    over all pairs and over completion-sensitive pairs (u: p > XV4_PTOL for at least one of the two policies)."""
    n = len(lo)
    off = ~np.eye(n, dtype=bool)
    cs = off & (u[:, None] | u[None, :])
    C = _x_v4_gt(np.asarray(lo, dtype=float), np.asarray(up, dtype=float)) & off
    return {"n_policies": n, "pairs": int(off.sum() // 2), "claims": int(C.sum()), "errors": int((C & wrong).sum()),
            "cs_pairs": int(cs.sum() // 2), "cs_claims": int((C & cs).sum()),
            "cs_errors": int((C & cs & wrong).sum())}


def xv4r_boot_draws(iv: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, draw, interval, bound): certificates of the policies whose interval is defined, coverage counts,
    draw-level flags, and the margin rules on the point values at the same number of claims."""
    rows = []
    for (c, dr), d in iv.groupby(["cell", "draw"], sort=False):
        d = d.sort_values("policy")
        for (it, bd), (lc, uc) in XV4R_BOOT_INTERVALS.items():
            ok = d[[lc, uc]].notna().all(axis=1).to_numpy()
            dd = d[ok]
            lo, up, jt = (dd[x].to_numpy(dtype=float) for x in (lc, uc, "true"))
            cov = (jt >= lo - XV4_TOL) & (jt <= up + XV4_TOL)
            wrong = ~_x_v4_gt(jt, jt)
            r = xv4r_interval_claims(lo, up, dd["p_bench"].to_numpy(dtype=float) > XV4_PTOL, wrong)
            row = {"cell": c, "family": d["family"].iloc[0], "draw": int(dr), "interval": it, "bound": bd,
                   "policies": int(len(d)), "undefined": int((~ok).sum()), **r, "uncovered": int(len(d) - cov.sum()),
                   "any_wrong": bool(r["errors"] > 0), "all_covered": bool(ok.all() and cov.all())}
            pl, pu = dd["point_lo"].to_numpy(dtype=float), dd["point_up"].to_numpy(dtype=float)
            for nm, v in (("mean", pu), ("term", pl), ("mid", (pl + pu) / 2.0)):
                Cm = x_v4_margin_claims(v, r["claims"])
                row[f"margin_{nm}_claims"], row[f"margin_{nm}_errors"] = int(Cm.sum()), int((Cm & wrong).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def xv4r_boot_table(iv: pd.DataFrame, D: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Per bootstrap cell, family pool and overall pool x interval x bound: coverage of J_true over all default
    policies and over p > 0 / p = 0, mean width, certificates (also among AU + UU pairs), draw-level counts and the
    error rates of the margin rules at the same number of claims."""
    fams = sorted(set(iv["family"]))
    units = [(c, "cell", [c]) for c in order if c in set(iv["cell"])]
    units += [(f"pooled:{f}", "pool", sorted(set(iv.loc[iv["family"] == f, "cell"]))) for f in fams]
    if len(fams) > 1:
        units.append(("pooled:all", "pool", sorted(set(iv["cell"]))))
    rows = []
    for unit, utype, cs in units:
        d, g0 = iv[iv["cell"].isin(cs)], D[D["cell"].isin(cs)]
        pos = (d["p_bench"] > XV4_PTOL).to_numpy()
        jt = d["true"].to_numpy(dtype=float)
        for (it, bd), (lc, uc) in XV4R_BOOT_INTERVALS.items():
            lo, up = d[lc].to_numpy(dtype=float), d[uc].to_numpy(dtype=float)
            ok = ~(np.isnan(lo) | np.isnan(up))
            cov = ok & (jt >= lo - XV4_TOL) & (jt <= up + XV4_TOL)
            width = up - lo
            g = g0[(g0["interval"] == it) & (g0["bound"] == bd)]
            s = g.sum(numeric_only=True)
            r = {"unit": unit, "unit_type": utype, "n_cells": len(cs), "interval": it, "bound": bd,
                 "replicates": int(d["n_replicates"].max()) if len(d) else 0,
                 "quantiles": "|".join(str(x) for x in XV3BOOT.quantiles),
                 "n_policies": int(len(d)), "n_undefined": int((~ok).sum()), "n_p_pos": int(pos.sum()),
                 "n_p_zero": int((~pos).sum()), "coverage_all": _xv4r_rate(cov.sum(), ok.sum()),
                 "coverage_p_pos": _xv4r_rate((cov & pos).sum(), (ok & pos).sum()),
                 "coverage_p_zero": _xv4r_rate((cov & ~pos).sum(), (ok & ~pos).sum()),
                 "mean_width": float(width[ok].mean()) if ok.any() else np.nan,
                 "mean_width_p_pos": float(width[ok & pos].mean()) if (ok & pos).any() else np.nan,
                 "draws": int(len(g)), "pairs": int(s["pairs"]), "claims": int(s["claims"]),
                 "certified_share": _xv4r_rate(s["claims"], s["pairs"]), "errors": int(s["errors"]),
                 "error_rate": _xv4r_rate(s["errors"], s["claims"]), "cs_pairs": int(s["cs_pairs"]),
                 "cs_claims": int(s["cs_claims"]), "cs_share": _xv4r_rate(s["cs_claims"], s["cs_pairs"]),
                 "cs_errors": int(s["cs_errors"]), "cs_error_rate": _xv4r_rate(s["cs_errors"], s["cs_claims"]),
                 "draws_with_wrong_cert": int(g["any_wrong"].sum()), "draws_all_covered": int(g["all_covered"].sum())}
            for nm in ("mean", "term", "mid"):
                r[f"margin_{nm}_claims"] = int(s[f"margin_{nm}_claims"])
                r[f"margin_{nm}_error_rate"] = _xv4r_rate(s[f"margin_{nm}_errors"], s[f"margin_{nm}_claims"])
            rows.append(r)
    return pd.DataFrame(rows)


def xv4r_boot_vs_ii8i(iv: pd.DataFrame) -> Dict[str, Any]:
    """The NHB bootstrap interval, the point interval and J_true against II.8i's xv3_boot_intervals (XV3_STATS) on the
    same rows: an identity of two implementations (rows beyond XV4_TOL or with a different NaN pattern)."""
    st = globals().get("XV3_STATS") or {}
    ref = st.get("bootstrap_intervals") if isinstance(st, dict) else None
    if ref is None or not len(ref):
        return {"status": "not compared (XV3_STATS has no bootstrap intervals)", "mismatched_rows": 0}
    k = ["cell", "draw", "policy"]
    r = ref.assign(draw=pd.to_numeric(ref["draw"]).astype(np.int64))
    m = iv.merge(r, on=k, how="outer", suffixes=("", "_ii8i"), indicator=True)
    out = {"status": "compared", "rows": int(len(m)), "missing": int((m["_merge"] != "both").sum()),
           "max_abs_diff": {}, "mismatched_rows": int((m["_merge"] != "both").sum())}
    both = m[m["_merge"] == "both"]
    bad = np.zeros(len(both), dtype=bool)
    for c in ("true", "point_lo", "point_up", "boot_lo", "boot_up"):
        a, b_ = both[c].to_numpy(dtype=float), both[f"{c}_ii8i"].to_numpy(dtype=float)
        nan_diff = np.isnan(a) != np.isnan(b_)
        d = np.abs(np.where(np.isnan(a) | np.isnan(b_), 0.0, a - b_))
        out["max_abs_diff"][c] = float(d.max(initial=0.0))
        bad |= nan_diff | (d > XV4_TOL)
    out["mismatched_rows"] += int(bad.sum())
    return out


def xv4r_self_check(R: pd.DataFrame, res: pd.DataFrame, pools: Dict[str, List[str]],
                    kind_of: Dict[str, str]) -> Dict[str, Any]:
    """Per-draw NHB and point counts against II.8i's code (xv3_certified_draws on the same rows; XV3_STATS when in
    memory), and the pooled counts of gumbel_estimated and garnet_all against the published values (XV4R_KNOWN
    mismatches counted in the full paper design, never raised here)."""
    mine = R[~R["distinct"].astype(bool) & (R["type"] == "all")].set_index(["cell", "draw", "targets"])
    cols = {"pairs": "pairs", "certified": "cert_nhb_claims", "certified_errors": "cert_nhb_errors",
            "claims_mean": "point_mean_claims", "claims_terminate": "point_term_claims",
            "errors_mean": "point_mean_errors", "errors_terminate": "point_term_errors"}
    refs = []
    if len(res):   # default rows only: xv3_certified_draws keeps only these, so the counts are unchanged
        Wv3, _ = xv3_bracket_frame(res[xs_as_bool(res["is_default"])], kind_of)
        refs.append(("xv3_certified_draws (II.8i code) on the analysed rows", xv3_certified_draws(Wv3)))
        del Wv3
    st = globals().get("XV3_STATS") or {}
    cc3 = st.get("certified_draws") if isinstance(st, dict) else None
    if cc3 is not None and len(cc3):
        refs.append(("XV3_STATS['certified_draws'] (II.8i, II.8 cells)", cc3))
    out: Dict[str, Any] = {"comparisons": {}, "mismatched_draws": 0}
    for name, cc in refs:
        if cc is None or not len(cc):
            out["comparisons"][name] = {"draws": 0}
            continue
        ref = cc.assign(draw=pd.to_numeric(cc["draw"]).astype(np.int64)).set_index(["cell", "draw", "targets"])
        both, missing = ref.index.intersection(mine.index), ref.index.difference(mine.index)
        bad = np.zeros(len(both), dtype=bool)
        diff = {}
        for c, v in cols.items():
            x = ref.loc[both, c].to_numpy(dtype=float) != mine.loc[both, v].to_numpy(dtype=float)
            diff[c], bad = int(x.sum()), bad | x
        out["comparisons"][name] = {"draws": int(len(both)), "missing_here": int(len(missing)),
                                    "mismatched_draws": int(bad.sum()), "mismatches_by_column": diff}
        out["mismatched_draws"] += int(bad.sum()) + int(len(missing))
    full = RUN_MODE == "paper" and COMPUTE_PROFILE == "full"
    known: Dict[str, Any] = {}
    for key in sorted(set(XV4R_KNOWN) | set(XV4R_KNOWN_REPORTED)):
        tf, pool = key.split(":")
        d = R[(R["targets"] == tf) & ~R["distinct"].astype(bool) & (R["type"] == "all")
              & R["cell"].isin(pools.get(pool, []))]
        s = d.sum(numeric_only=True)
        exp = {**XV4R_KNOWN.get(key, {}), **XV4R_KNOWN_REPORTED.get(key, {})}
        got = {c: (int(s[c]) if c in s.index and len(d) else None) for c in exp}
        design = bool(full and len(d) == XV4R_KNOWN_DRAWS.get(pool, -1))
        asserted = sorted(XV4R_KNOWN.get(key, {})) if design else []
        wrong = [c for c in asserted if got[c] != exp[c]]
        known[key] = {"draws": int(len(d)), "expected": exp, "got": got, "match": got == exp,
                      "asserted_fields": asserted, "asserted_mismatches": wrong}
        if wrong:
            print(f"  II.9b self-check {key}: {', '.join(f'{c} {got[c]} != published {exp[c]}' for c in wrong)}")
    out["known_values"] = known
    out["known_values_asserted"] = any(v["asserted_fields"] for v in known.values())
    # counted into identity_counts by xv4r_run, so the cell raises after the outputs are written (as II.8i)
    out["published_value_mismatches"] = sum(len(v["asserted_mismatches"]) for v in known.values())
    return out


def xv4r_draw_counts(W: pd.DataFrame, iv: pd.DataFrame) -> Dict[str, Any]:
    """Draws present per enabled cell (x_v4_all_specs) and per II.8g bootstrap cell, against the configured draws."""
    have = W.reset_index().groupby("cell")["draw"].nunique().to_dict() if len(W) else {}
    per = {c: {"draws": int(have.get(c, 0)), "configured_draws": int(s["draws"])} for c, s in x_v4_all_specs().items()}
    bh = iv.groupby("cell")["draw"].nunique().to_dict() if len(iv) else {}
    bper = {str(s["cell"]): {"draws": int(bh.get(s["cell"], 0)), "configured_draws": int(s["draws"])}
            for s in globals().get("XV3_BOOT_CELLS", []) or []}
    cells_ok = bool(per) and all(v["draws"] == v["configured_draws"] for v in per.values())
    boot_ok = all(v["draws"] == v["configured_draws"] for v in bper.values())
    return {"draws_per_cell": per, "bootstrap_draws_per_cell": bper, "cells_complete": cells_ok if per else None,
            "bootstrap_complete": boot_ok if bper else None,
            "draws_complete": (cells_ok and boot_ok) if per else None}


def xv4r_run(table_dir: Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Section II.9b end to end: tables, checks, summary json content. Returns (XV4_ROBUST, XV4_ROBUST_SUMMARY)."""
    t0 = time.perf_counter()
    table_dir = Path(table_dir)
    outputs: List[str] = []

    def write_csv(df: pd.DataFrame, name: str) -> None:
        if name.endswith(".gz"):
            df.to_csv(table_dir / name, index=False, compression="gzip")
        else:
            df.to_csv(table_dir / name, index=False)
        outputs.append(f"xenv/{name}")

    res, boot, inputs = xv4r_inputs()
    kind_of = {str(s["cell"]): str(s["v3_kind"]) for s in (globals().get("X_V3_CELLS") or [])}
    out: Dict[str, pd.DataFrame] = {k: pd.DataFrame() for k in XV4R_FRAMES}
    checks: Dict[str, Any] = {"theorem_counts": {}, "identity_counts": {}}
    summary: Dict[str, Any] = {"inputs": inputs, "value_ranges": XV4R_VRANGE, "tolerance": XV4_TOL,
                               "p_tol": XV4_PTOL, "kappa_grid": list(XV4CFG.kappa_grid),
                               "ci": {"resamples": XV4R_CI_RESAMPLES, "levels": list(XV4R_CI_LEVELS)}}
    W = pd.DataFrame()
    if len(res):
        print(f"II.9b: {inputs['II.6']['rows']} II.6 rows ({inputs['II.6']['source']}), {inputs['II.8']['rows']} "
              f"II.8 rows ({inputs['II.8']['source']})")
        order = [str(c) for c in pd.unique(res["cell"])]
        W, checks["wide"] = xv4r_wide(res, kind_of)
        checks["theorem_counts"] = xv4r_theorem_counts(W)
        R, K, KS, G, checks["skipped_target_sets"] = xv4r_claims(W, XV4CFG.kappa_grid)
        assert len(R) and len(K), "II.9b: every draw was skipped (no proper policy set on any target set)"
        units, pools = xv4r_units(W, order)
        summary["pools"] = pools
        base = R[~R["distinct"].astype(bool) & (R["type"] == "all")].set_index(["cell", "draw", "targets"])
        n_bad = 0
        for kap, col in ((0.0, "cert_nhb_claims"), (1.0, "cert_free_claims")):
            kk = K[K["kappa"] == kap].set_index(["cell", "draw", "targets"])
            if len(kk):
                n_bad += int((kk["claims"].to_numpy() != base[col].reindex(kk.index).to_numpy()).sum())
        checks["identity_counts"]["kappa_endpoint_mismatches"] = n_bad
        checks["self_check"] = xv4r_self_check(R, res, pools, kind_of)
        checks["identity_counts"]["ii8i_certified_count_mismatches"] = checks["self_check"]["mismatched_draws"]
        checks["identity_counts"]["published_value_mismatches"] = checks["self_check"]["published_value_mismatches"]
        del res
        gc.collect()
        t1 = time.perf_counter()
        out["pairs"], out["kappa_draws"] = R, K
        out["summary"] = xv4r_summary_table(R, units)
        out["kappa"] = xv4r_kappa_table(K, KS, units)
        truth_cells = sorted(set(R.loc[(R["targets"] == "bench") & R["cert_nhb_errors"].notna(), "cell"])
                             if "cert_nhb_errors" in R.columns else [])
        out["gapbins"] = xv4r_gap_table(G, units, truth_cells)
        checks["aggregation_s"] = time.perf_counter() - t1
        write_csv(R, "v4_robust_pairs.csv.gz")
        write_csv(out["summary"], "v4_robust_summary.csv")
        write_csv(out["gapbins"], "v4_robust_gapbins.csv")
        write_csv(out["kappa"], "v4_robust_kappa.csv")
        S = out["summary"]
        hd = S[(S["unit_type"] == "pool") & ~S["distinct"].astype(bool) & (S["type"] == "all")]
        keys = ["draws", "pairs", "cert_nhb_share", "cert_nhb_error_rate", "cert_nhb_error_rate_ci_low",
                "cert_nhb_error_rate_ci_high", "cert_free_share", "cert_free_error_rate", "cert_free_error_rate_ci_low",
                "cert_free_error_rate_ci_high", "cert_nhb_only_share", "cert_nhb_only_error_rate", "cert_nhb_cs_share",
                "cert_nhb_cs_error_rate", "cert_free_cs_share", "cert_free_cs_error_rate", "margin_mean_error_rate",
                "margin_term_error_rate", "margin_mid_error_rate", "margin_mean_free_error_rate"]
        summary["headline"] = {u: {tf: {k: r[k] for k in keys} for tf, r in g.set_index("targets").iterrows()}
                               for u, g in hd.groupby("unit", sort=False)}
        kp = out["kappa"][out["kappa"]["unit_type"] == "pool"]
        summary["kappa"] = {
            u: {tf: {"nhb_only_claims": int(g["nhb_only_claims"].iloc[0]),
                     "kappa_star_quantiles": {c: g[c].iloc[0] for c in g.columns if c.startswith("kappa_star_q")},
                     "nhb_only_margin_quantiles": {c: g[c].iloc[0] for c in g.columns
                                                   if c.startswith("nhb_only_margin_")},
                     "by_kappa": {f"{r['kappa']:g}": {k: r[k] for k in ("claims", "share", "cs_share",
                                                                       "share_of_nhb_claims", "errors", "error_rate")}
                                  for _, r in g.iterrows()}}
                for tf, g in gu.groupby("targets", sort=False)}
            for u, gu in kp.groupby("unit", sort=False)}
        gb = out["gapbins"]
        gb = gb[gb["unit_type"] == "pool"] if len(gb) else pd.DataFrame(columns=["unit", "bin"])   # no truth cell
        summary["gap_bins"] = {u: {r["bin"]: {"claims": r["claims"], "error_rate": r["error_rate"]}
                                   for _, r in g.iterrows()} for u, g in gb.groupby("unit", sort=False)}
        del R, K, KS, G
        gc.collect()
    else:
        print("II.9b: no Part II results (XRESULTS, XRESULTS_V3 and their files are missing) - claims not analysed.")
    iv = pd.DataFrame()
    if len(boot):
        iv, checks["bootstrap"] = xv4r_boot_intervals(boot, tuple(XV3BOOT.quantiles))
        del boot
        checks["bootstrap_vs_ii8i"] = xv4r_boot_vs_ii8i(iv)
        border = [str(s["cell"]) for s in globals().get("XV3_BOOT_CELLS", []) or []]
        border += [c for c in pd.unique(iv["cell"]) if c not in border]
        out["v3boot_intervals"] = iv
        out["v3boot_draws"] = xv4r_boot_draws(iv)
        out["v3boot_allpolicy"] = xv4r_boot_table(iv, out["v3boot_draws"], border)
        write_csv(out["v3boot_allpolicy"], "v4_v3boot_allpolicy.csv")
        bp = out["v3boot_allpolicy"]
        summary["v3boot"] = {f"{r['unit']}|{r['interval']}|{r['bound']}": {
            k: r[k] for k in ("coverage_all", "coverage_p_pos", "coverage_p_zero", "certified_share", "error_rate",
                              "cs_share", "draws", "draws_with_wrong_cert", "draws_all_covered",
                              "margin_mid_error_rate")} for _, r in bp[bp["unit_type"] == "pool"].iterrows()}
        bc = checks["bootstrap"]
        checks["theorem_counts"].update({f"bootstrap_{k}": bc[k] for k in (
            "replicate_rule_p_difference", "replicate_negative_width", "replicate_width_above_bound",
            "point_negative_width", "point_free_below_nhb_upper")})
        checks["identity_counts"]["bootstrap_nhb_interval_vs_ii8i"] = checks["bootstrap_vs_ii8i"]["mismatched_rows"]
    else:
        checks["bootstrap"] = {"status": "not run: no II.8g bootstrap rows (xenv_v3_bootstrap off or no cell)"}
    counts = xv4r_draw_counts(W, iv)
    del W
    gc.collect()
    tc, ic = checks["theorem_counts"], checks["identity_counts"]
    checks["passes"] = bool(all(v == 0 for v in tc.values()) and all(v == 0 for v in ic.values()))
    status = ("run" if len(out["pairs"]) or len(iv) else
              "run: no inputs (no Part II results and no II.8g bootstrap rows)")
    summary.update({"status": status, "code_hash": XV4_ROBUST_CODE_HASH, "checks": checks,
                    "draws_complete": counts["draws_complete"], "draw_counts": counts,
                    "outputs": outputs + ["xenv/v4_robust_summary.json"], "elapsed_s": time.perf_counter() - t0})
    return out, summary


XV4R_FRAMES = ("pairs", "summary", "gapbins", "kappa", "kappa_draws", "v3boot_intervals", "v3boot_draws",
               "v3boot_allpolicy")
# Objects whose source changes what II.9b computes, and the module constants they read (by value).
XV4_ROBUST_CODE = [_xv4r_rate, _xv4r_vrange, xv4r_cell_group, xv4r_inputs, xv4r_wide, xv4r_theorem_counts,
                   xv4r_kappa_claims, xv4r_gap_claims, xv4r_draw, xv4r_claims, xv4r_units, xv4r_summary_row,
                   xv4r_cs_headline, xv4r_resample_matrix, xv4r_ratio_ci, xv4r_summary_table, xv4r_kappa_table,
                   xv4r_gap_table, xv4r_boot_intervals, xv4r_interval_claims, xv4r_boot_draws, xv4r_boot_table,
                   xv4r_boot_vs_ii8i, xv4r_self_check, xv4r_draw_counts, xv4r_run]
XV4_ROBUST_CODE_HASH = obj_hash({
    "common": XV4_COMMON_CODE_HASH, "code": code_hash(*XV4_ROBUST_CODE),
    "constants": {"vrange": XV4R_VRANGE, "targets": list(XV4R_TARGETS),
                  "target_sets": {k: list(v) for k, v in XV4R_TARGET_SETS.items()}, "types": list(XV4R_TYPES),
                  "ci_kinds": list(XV4R_CI_KINDS), "ci_resamples": XV4R_CI_RESAMPLES,
                  "ci_levels": list(XV4R_CI_LEVELS), "kstar_q": list(XV4R_KSTAR_Q),
                  "gap_bins": [str(x) for x in XV4R_GAP_BINS], "result_columns": XV4R_RESULT_COLUMNS,
                  "boot_columns": XV4R_BOOT_COLUMNS,
                  "boot_intervals": {f"{a}|{b}": list(v) for (a, b), v in XV4R_BOOT_INTERVALS.items()},
                  "boot_point": XV3_BOOT_POINT, "boot_quantiles": list(XV3BOOT.quantiles),
                  "kappa_grid": list(XV4CFG.kappa_grid), "known": XV4R_KNOWN, "known_draws": XV4R_KNOWN_DRAWS,
                  "known_reported": XV4R_KNOWN_REPORTED, "frames": list(XV4R_FRAMES)}})

XV4_ROBUST: Dict[str, pd.DataFrame] = {k: pd.DataFrame() for k in XV4R_FRAMES}
XV4_ROBUST_SUMMARY: Dict[str, Any] = {"status": f"not run ({XV4R_SWITCH} off)", "code_hash": XV4_ROBUST_CODE_HASH,
                                      "checks": {}, "elapsed_s": 0.0, "draws_complete": None}
if EXPERIMENTS[XV4R_SWITCH]:
    XV4_ROBUST, XV4_ROBUST_SUMMARY = xv4r_run(XENV_TABLE_DIR)
save_json(XV4_ROBUST_SUMMARY, XENV_TABLE_DIR / "v4_robust_summary.json")
print("II.9b code hash:", XV4_ROBUST_CODE_HASH[:16], "| status:", XV4_ROBUST_SUMMARY["status"], "| elapsed:",
      f"{XV4_ROBUST_SUMMARY['elapsed_s']:.1f} s", "| draws complete:", XV4_ROBUST_SUMMARY["draws_complete"])
if len(XV4_ROBUST["summary"]):
    _s = XV4_ROBUST["summary"]
    display(_s[(_s["unit_type"] == "pool") & ~_s["distinct"].astype(bool) & (_s["type"] == "all")][
        ["unit", "targets", "draws", "pairs", "cert_nhb_share", "cert_nhb_error_rate", "cert_nhb_error_rate_ci_low",
         "cert_nhb_error_rate_ci_high", "cert_free_share", "cert_free_error_rate", "cert_nhb_cs_share",
         "cert_free_cs_share", "margin_mid_error_rate"]].round(4))
    _k = XV4_ROBUST["kappa"]
    if len(_k):
        display(_k[(_k["unit_type"] == "pool") & (_k["targets"] == "bench")][
            ["unit", "kappa", "claims", "share", "cs_share", "error_rate", "nhb_only_claims",
             "kappa_star_q50"]].round(4))
    print("II.9b self-check (pooled counts against the published values):",
          json.dumps({k: {"match": v["match"], "draws": v["draws"]} for k, v in
                      XV4_ROBUST_SUMMARY["checks"]["self_check"]["known_values"].items()}))
    del _s, _k
if len(XV4_ROBUST["v3boot_allpolicy"]):
    _b = XV4_ROBUST["v3boot_allpolicy"]
    display(_b[_b["unit_type"] == "pool"][["unit", "interval", "bound", "coverage_all", "coverage_p_pos",
                                           "coverage_p_zero", "certified_share", "error_rate",
                                           "draws_with_wrong_cert", "draws_all_covered"]].round(4))
    del _b
if XV4_ROBUST_SUMMARY.get("checks", {}).get("passes") is False:
    _c = XV4_ROBUST_SUMMARY["checks"]
    raise AssertionError(f"II.9b checks failed: theorem {_c['theorem_counts']}, identity {_c['identity_counts']} "
                         "(xenv/v4_robust_summary.json)")
