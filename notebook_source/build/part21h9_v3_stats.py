# %% [markdown]
# ### II.8i Additional analyses: bracketing coverage, certified comparisons, rank fidelity, bootstrap intervals
#
# These analyses use the definitions of the paper. They match the paper-side analysis of the II.6 cells, so
# the II.8 numbers can be read next to those.
#
# * **Policies.** Bracketing uses every non-reference policy of every draw. Certified comparisons and rank fidelity
#   use the default policy set of each draw. A policy is **violating** if P(any unsupported decision) > 1e-12 on
#   `bench_mean`. The value tolerance is 1e-9.
# * **Bracketing on the true rows** (semantics only, where the theorem applies exactly; not for the aggregated cells,
#   whose benchmark state space differs from the true one). The lower bound is J(true rows + terminate), the upper
#   bound J(true rows + mean rule), and the truth J(true system). Reported: lower-bound violations (0 by Theorem
#   part i), upper-bound coverage overall, among violating policies, and among policies violating on the true rows;
#   the size of the upper-bound violation when it fails, also divided by p·(v_max − v_min); and the interval width.
# * **Bracketing at benchmark level** (completion plus estimation error): J_true ∈ [J_bench_terminate, J_bench_mean].
# * **Certified comparisons** per draw, over the default policies, on the true-rows targets and on the benchmark
#   targets. A pair (i, j) is certified if lower(i) > upper(j) + 1e-9. It is a certification error if
#   J_true(i) ≤ J_true(j) + 1e-9, and a reversal if J_true(i) < J_true(j) − 1e-9. Point claims under one rule,
#   J_rule(i) > J_rule(j) + 1e-9, are scored the same way. Shares are over unordered pairs, with counts pooled over
#   draws.
# * **Rank fidelity** (exploratory, not preregistered): Kendall τ-b, top-1 agreement and selection regret of the
#   benchmark ranking against the true ranking per draw (`xs_rank_fidelity_draws` of II.7), and the paired
#   difference τ(mean) − τ(terminate) with stratified-bootstrap intervals over draws.
# * **Bootstrap intervals** (II.8g). The interval is [q_2.5%(J_term^b), q_97.5%(J_mean^b)] over the proper
#   replicates; the point interval is [J_bench_terminate, J_bench_mean]. Coverage of J_true, and certified comparisons
#   with each kind of interval, are computed per cell and pooled.
# * **Hidden benefit** (random-row Garnet cells): per draw, the upper-bound coverage among violating policies against
#   the draw's share of non-preferred pairs whose action value beats the preferred mean (II.8d), with the Spearman
#   correlation over draws.
#
# The theorem facts (no lower-bound violation on the true rows, non-negative widths, zero width for policies that never
# take an unsupported action, values inside [v_min, v_max]) are counted in `xenv/v3_stats_checks.json` (`passes`).
# Coverage and certification rates are scientific outcomes; they are reported and never asserted.
#
# | Output | Content |
# |---|---|
# | `xenv/v3_bracketing.csv` | per II.8 cell and pooled by kind: semantics-only (`sem_*`) and benchmark-level (`bench_*`) bracketing statistics |
# | `xenv/v3_bracketing_draws.csv` | per cell and draw: violating policies, lower-bound violations, upper / benchmark coverage, hidden-benefit share |
# | `xenv/v3_certified.csv`, `xenv/v3_certified_draws.csv` | certified comparisons per cell (and pooled) and per draw, for `true_rows` and `bench` targets |
# | `xenv/v3_rank_fidelity.csv`, `xenv/v3_rank_fidelity_draws.csv` | rank fidelity per cell and completion, and τ(mean) − τ(terminate) |
# | `xenv/v3_bootstrap_intervals.csv` | per bootstrap cell, draw and default policy: J_true, point and bootstrap interval, coverage flags |
# | `xenv/v3_bootstrap_coverage.csv` | per bootstrap cell and pooled, for point and bootstrap intervals: coverage and certified comparisons |
# | `xenv/v3_stats_checks.json` | theorem checks, input sizes, figure status, outputs |
# | `figures/xenv/fig_v3_bracketing.*` | (a) upper-bound coverage (true rows) and benchmark-level coverage among violating policies per II.8 cell; (b) random-row Garnet: coverage per draw against the hidden-benefit share |
# | `figures/xenv/fig_v3_bootstrap.*` | per bootstrap cell, point vs bootstrap intervals: (a) coverage among violating policies, (b) certified share, (c) certified error rate |

# %%
# ============================================================
# II.8i Revision analyses
# ============================================================
EXECLOG.mark("II.8 Revision analyses")
XV3_BR_PTOL = 1e-12                       # violating: P(any unsupported decision) > XV3_BR_PTOL on bench_mean
XV3_VRANGE = {"gumbel": (-1.0, 1.0), "garnet": (0.0, 1.0), "icu": (0.0, 1.0)}   # [v_min, v_max]; r_term = v_min
XV3_BR_TARGETS = ["bench_mean", "bench_terminate", "true", "true_rows_mean", "true_rows_terminate"]


def _xv3_rate(num, den) -> float:
    return float(num / den) if den else float("nan")


def xv3_bracket_frame(results: pd.DataFrame, kind_of: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Per (cell, draw, policy) values on the bracketing targets and the bound / coverage flags (non-reference rows)."""
    cols = ["cell", "draw", "policy", "family", "v3_kind", "is_default", *XV3_BR_TARGETS, "p_bench", "p_true",
            "has_rows", "has_bench", "viol", "viol_true", "lo_ok", "up_ok", "b_lo_ok", "b_up_ok", "b_in", "exceed",
            "width", "bench_width", "exceed_ratio", "width_ratio"]
    if results is None or not len(results):
        return pd.DataFrame(columns=cols), {"rows": 0}
    r = results[results["target"].isin(XV3_BR_TARGETS) & ~xs_as_bool(results["is_reference"])].copy()
    r["draw"] = pd.to_numeric(r["draw"]).astype(np.int64)
    k = ["cell", "draw", "policy"]
    assert not r.duplicated(k + ["target"]).any(), "duplicated (cell, draw, policy, target) rows"
    ok = r["status"] == "ok"
    J = r[ok].set_index(k + ["target"])["return"].unstack("target")
    Pv = r[ok].set_index(k + ["target"])["prob_any_violation"].unstack("target")
    W = r.drop_duplicates(k).set_index(k)[["family", "is_default"]].copy()
    W["is_default"] = xs_as_bool(W["is_default"]).to_numpy()
    W = W.join(J.reindex(columns=XV3_BR_TARGETS))
    W["v3_kind"] = W.index.get_level_values("cell").map(lambda c: kind_of.get(c, ""))
    Pv = Pv.reindex(index=W.index, columns=XV3_BR_TARGETS)
    W["p_bench"] = Pv["bench_mean"]
    W["p_true"] = Pv["true_rows_mean"].where(Pv["true_rows_mean"].notna(), Pv["true"])
    W["has_rows"] = W[["true", "true_rows_mean", "true_rows_terminate"]].notna().all(axis=1)
    W["has_bench"] = W[["true", "bench_mean", "bench_terminate"]].notna().all(axis=1)
    W["viol"] = W["p_bench"] > XV3_BR_PTOL
    W["viol_true"] = W["p_true"] > XV3_BR_PTOL
    W["lo_ok"] = W["true"] >= W["true_rows_terminate"] - XV3_TOL
    W["up_ok"] = W["true"] <= W["true_rows_mean"] + XV3_TOL
    W["b_lo_ok"] = W["true"] >= W["bench_terminate"] - XV3_TOL
    W["b_up_ok"] = W["true"] <= W["bench_mean"] + XV3_TOL
    W["b_in"] = W["b_lo_ok"] & W["b_up_ok"]
    W["exceed"] = W["true"] - W["true_rows_mean"]
    W["width"] = W["true_rows_mean"] - W["true_rows_terminate"]
    W["bench_width"] = W["bench_mean"] - W["bench_terminate"]
    vr = W["family"].map(lambda f: XV3_VRANGE[f][1] - XV3_VRANGE[f][0])
    bnd = W["p_true"] * vr
    W["exceed_ratio"] = np.where(bnd > 0, W["exceed"] / bnd.where(bnd > 0, 1.0), np.nan)
    W["width_ratio"] = np.where(bnd > 0, W["width"] / bnd.where(bnd > 0, 1.0), np.nan)
    rng_bad = 0
    for fam, (lo, hi) in XV3_VRANGE.items():
        v = W.loc[W["family"] == fam, XV3_BR_TARGETS].to_numpy(dtype=float)
        rng_bad += int(((v < lo - XV3_TOL) | (v > hi + XV3_TOL)).sum())
    hr, hb = W["has_rows"], W["has_bench"]
    checks = {"rows": int(len(W)), "nonok_rows": int((~ok).sum()), "policy_rows_with_true_rows": int(hr.sum()),
              "policy_rows_with_bench": int(hb.sum()), "values_outside_value_range": rng_bad,
              "lower_bound_violations_true_rows": int((hr & ~W["lo_ok"]).sum()),
              "negative_width_true_rows": int((hr & (W["width"] < -XV3_TOL)).sum()),
              "negative_width_bench": int((hb & (W["bench_width"] < -XV3_TOL)).sum()),
              "nonzero_width_admissible": int((hr & ~W["viol_true"] & (W["width"].abs() > XV3_TOL)).sum()),
              "width_above_bound": int((hr & (W["width"] > bnd + XV3_TOL)).sum())}
    return W.reset_index()[cols], checks


def xv3_bracket_semantics(d: pd.DataFrame) -> Dict[str, Any]:
    d = d[d["has_rows"]]
    if d.empty:
        return {"n_rows": 0}
    v, f, vt = d[d["viol"]], d[~d["up_ok"]], d[d["viol_true"]]
    return {"n_rows": int(len(d)), "n_draws": int(d["draw"].nunique()), "n_violating": int(len(v)),
            "lower_violations": int((~d["lo_ok"]).sum()), "lower_holds_violating": _xv3_rate(v["lo_ok"].sum(), len(v)),
            "max_lower_shortfall": float((d["true_rows_terminate"] - d["true"]).max()),
            "upper_coverage_all": _xv3_rate(d["up_ok"].sum(), len(d)),
            "upper_coverage_violating": _xv3_rate(v["up_ok"].sum(), len(v)),
            "upper_coverage_violating_truep": _xv3_rate(vt["up_ok"].sum(), len(vt)), "n_violating_truep": int(len(vt)),
            "upper_failures": int(len(f)), "upper_exceedance_mean": float(f["exceed"].mean()) if len(f) else 0.0,
            "upper_exceedance_median": float(f["exceed"].median()) if len(f) else 0.0,
            "upper_exceedance_max": float(f["exceed"].max()) if len(f) else 0.0,
            "upper_exceedance_max_over_p_range": float(np.nanmax(f["exceed_ratio"])) if len(f) else 0.0,
            "width_mean_violating": float(v["width"].mean()) if len(v) else 0.0,
            "width_max_over_p_range": (float(np.nanmax(d["width_ratio"])) if d["width_ratio"].notna().any() else 0.0)}


def xv3_bracket_benchmark(d: pd.DataFrame) -> Dict[str, Any]:
    d = d[d["has_bench"]]
    if d.empty:
        return {"n_rows": 0}
    v = d[d["viol"]]
    return {"n_rows": int(len(d)), "n_violating": int(len(v)), "coverage_all": _xv3_rate(d["b_in"].sum(), len(d)),
            "coverage_violating": _xv3_rate(v["b_in"].sum(), len(v)),
            "lower_holds_violating": _xv3_rate(v["b_lo_ok"].sum(), len(v)),
            "upper_holds_violating": _xv3_rate(v["b_up_ok"].sum(), len(v)),
            "lower_violations_all": int((~d["b_lo_ok"]).sum()), "upper_violations_all": int((~d["b_up_ok"]).sum()),
            "coverage_nonviolating": _xv3_rate(d.loc[~d["viol"], "b_in"].sum(), int((~d["viol"]).sum()))}


def xv3_pairs(lo: np.ndarray, up: np.ndarray, jt: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Certified pairs and single-rule point claims over one policy set, scored against the truth when given."""
    lo, up = np.asarray(lo, dtype=float), np.asarray(up, dtype=float)
    n = len(lo)
    out = {"n_policies": n, "pairs": n * (n - 1) // 2}
    cert = lo[:, None] > up[None, :] + XV3_TOL
    np.fill_diagonal(cert, False)
    out["certified"] = int(cert.sum())
    pm = up[:, None] > up[None, :] + XV3_TOL
    pt = lo[:, None] > lo[None, :] + XV3_TOL
    out["claims_mean"], out["claims_terminate"] = int(pm.sum()), int(pt.sum())
    out["rule_dependent"] = int((pm & (lo[:, None] < lo[None, :] - XV3_TOL)).sum())
    if jt is not None:
        jt = np.asarray(jt, dtype=float)
        wrong = jt[:, None] <= jt[None, :] + XV3_TOL
        rev = jt[:, None] < jt[None, :] - XV3_TOL
        out["truth_ordered"] = int((jt[:, None] > jt[None, :] + XV3_TOL).sum())
        out["certified_errors"], out["certified_reversals"] = int((cert & wrong).sum()), int((cert & rev).sum())
        for nm, cl in (("mean", pm), ("terminate", pt)):
            out[f"errors_{nm}"], out[f"reversals_{nm}"] = int((cl & wrong).sum()), int((cl & rev).sum())
            unc = cl & ~cert
            out[f"claims_{nm}_uncertified"], out[f"errors_{nm}_uncertified"] = int(unc.sum()), int((unc & wrong).sum())
    return out


def xv3_cc_summary(d: pd.DataFrame) -> Dict[str, Any]:
    if d.empty:
        return {"draws": 0}
    s = d.drop(columns=[c for c in ("cell", "draw", "family", "targets", "kind") if c in d]).sum(numeric_only=True)
    share = d["certified"] / d["pairs"].where(d["pairs"] > 0)
    o = {"draws": int(len(d)), "policies_per_draw": int(d["n_policies"].median()), "pairs": int(s["pairs"]),
         "certified": int(s["certified"]), "certified_share": _xv3_rate(s["certified"], s["pairs"]),
         "certified_share_draw_min": float(share.min()), "certified_share_draw_max": float(share.max()),
         "rule_dependent_share": _xv3_rate(s["rule_dependent"], s["pairs"]),
         "claims_mean_share": _xv3_rate(s["claims_mean"], s["pairs"]),
         "claims_terminate_share": _xv3_rate(s["claims_terminate"], s["pairs"])}
    if "certified_errors" in d and d["certified_errors"].notna().all():
        o.update({"truth_ordered_share": _xv3_rate(s["truth_ordered"], s["pairs"]),
                  "certified_errors": int(s["certified_errors"]),
                  "certified_error_rate": _xv3_rate(s["certified_errors"], s["certified"]),
                  "certified_reversal_rate": _xv3_rate(s["certified_reversals"], s["certified"])})
        for nm in ("mean", "terminate"):
            o.update({f"point_{nm}_errors": int(s[f"errors_{nm}"]),
                      f"point_{nm}_error_rate": _xv3_rate(s[f"errors_{nm}"], s[f"claims_{nm}"]),
                      f"point_{nm}_reversal_rate": _xv3_rate(s[f"reversals_{nm}"], s[f"claims_{nm}"]),
                      f"point_{nm}_error_rate_uncertified": _xv3_rate(s[f"errors_{nm}_uncertified"],
                                                                      s[f"claims_{nm}_uncertified"])})
    return o


def xv3_certified_draws(W: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, draw, targets) certified-comparison counts over the default policies (true_rows and bench targets)."""
    rows = []
    for (c, dr), d in W[W["is_default"]].groupby(["cell", "draw"], sort=False):
        jt = d["true"].to_numpy(dtype=float)
        for tf, (lc, uc) in (("true_rows", ("true_rows_terminate", "true_rows_mean")),
                             ("bench", ("bench_terminate", "bench_mean"))):
            if d[[lc, uc, "true"]].isna().any().any():
                continue
            rows.append({"cell": c, "draw": int(dr), "family": d["family"].iloc[0], "kind": d["v3_kind"].iloc[0],
                         "targets": tf, **xv3_pairs(d[lc].to_numpy(dtype=float), d[uc].to_numpy(dtype=float), jt)})
    return pd.DataFrame(rows)


def xv3_pools(cells: Sequence[str], kind_of: Dict[str, str]) -> Dict[str, List[str]]:
    pools: Dict[str, List[str]] = {}
    for c in cells:
        pools.setdefault(f"pooled:{kind_of.get(c, c.split('|', 1)[0])}", []).append(c)
    return {k: v for k, v in pools.items() if len(v) > 1}


def xv3_bracketing_table(W: pd.DataFrame, cells: Sequence[str], kind_of: Dict[str, str]) -> pd.DataFrame:
    rows = []
    units = [(c, [c]) for c in cells] + list(xv3_pools(cells, kind_of).items())
    for name, cs in units:
        d = W[W["cell"].isin(cs)]
        if d.empty:
            continue
        rows.append({"cell": name, "kind": kind_of.get(name, name.split(":", 1)[-1]), "n_cells": len(cs),
                     **{f"sem_{k}": v for k, v in xv3_bracket_semantics(d).items()},
                     **{f"bench_{k}": v for k, v in xv3_bracket_benchmark(d).items()}})
    return pd.DataFrame(rows)


def xv3_bracketing_draws(W: pd.DataFrame, draw_meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Per (cell, draw) bracketing summary, with the hidden-benefit share of random-row Garnet draws."""
    rows = []
    for (c, dr), d in W.groupby(["cell", "draw"], sort=False):
        s, b = xv3_bracket_semantics(d), xv3_bracket_benchmark(d)
        rows.append({"cell": c, "draw": int(dr), "kind": d["v3_kind"].iloc[0],
                     "n_violating": s.get("n_violating", b.get("n_violating")),
                     "lower_violations": s.get("lower_violations"),
                     "upper_coverage_violating": s.get("upper_coverage_violating"),
                     "upper_exceedance_max": s.get("upper_exceedance_max"),
                     "bench_coverage_violating": b.get("coverage_violating")})
    out = pd.DataFrame(rows)
    hb = None
    if draw_meta is not None and len(draw_meta):
        hb = next((col for col in draw_meta.columns if str(col).endswith("rows_nonpreferred_beneficial_share")), None)
    if len(out):
        out["hidden_benefit_share"] = np.nan
        if hb is not None:
            m = draw_meta[["cell", "draw", hb]].copy()
            m["draw"] = pd.to_numeric(m["draw"]).astype(np.int64)
            out = out.drop(columns="hidden_benefit_share").merge(
                m.rename(columns={hb: "hidden_benefit_share"}), on=["cell", "draw"], how="left", validate="one_to_one")
    return out


def xv3_rank_fidelity(results: pd.DataFrame, cells: Sequence[str], n_boot: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rank fidelity per draw (xs_rank_fidelity_draws) and per cell x completion, plus tau(mean) - tau(terminate)."""
    cols = ["cell", "family", "completion", "ranked_by", "evaluated_on", "n_draws", "mean_policies"] + \
           [f"{m}_{s}" for m in XS_RANK_METRICS for s in ("mean", "ci_low", "ci_high", "n")]
    if results is None or not len(results):
        return pd.DataFrame(columns=XS_RANK_DRAW_COLUMNS), pd.DataFrame(columns=cols)
    norm, _ = xs_normalized_scores(xs_prepare_results(results))
    draws = xs_sort_cells(xs_rank_fidelity_draws(norm), list(cells), ["draw"])
    rows = []
    for (cell, completion), g in draws.groupby(["cell", "completion"], sort=False):
        r = {"cell": cell, "family": g["family"].iloc[0], "completion": completion, "ranked_by": g["ranked_by"].iloc[0],
             "evaluated_on": g["evaluated_on"].iloc[0], "n_draws": int(len(g)), "mean_policies": float(g["n_policies"].mean())}
        for m in XS_RANK_METRICS:
            v = g[m].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            point, reps = xs_stratified_bootstrap({cell: v[:, None]} if len(v) else {}, xs_stat_mean(),
                                                  ("v3rank", completion, m), n_boot)
            r[f"{m}_mean"] = point
            r[f"{m}_ci_low"], r[f"{m}_ci_high"] = xs_ci(reps)
            r[f"{m}_n"] = int(len(v))
        rows.append(r)
    for cell, g in draws.groupby("cell", sort=False):
        t = g.pivot_table(index="draw", columns="completion", values="kendall_tau_b", aggfunc="first")
        if not {"mean", "terminate"} <= set(t.columns):
            continue
        dlt = (t["mean"] - t["terminate"]).dropna().to_numpy(dtype=np.float64)
        point, reps = xs_stratified_bootstrap({cell: dlt[:, None]} if len(dlt) else {}, xs_stat_mean(),
                                              ("v3rank", "tau difference"), n_boot)
        lo, hi = xs_ci(reps)
        rows.append({"cell": cell, "family": g["family"].iloc[0], "completion": "mean - terminate",
                     "ranked_by": "bench_mean | bench_terminate", "evaluated_on": "true", "n_draws": int(len(dlt)),
                     "mean_policies": float(g["n_policies"].mean()), "kendall_tau_b_mean": point,
                     "kendall_tau_b_ci_low": lo, "kendall_tau_b_ci_high": hi, "kendall_tau_b_n": int(len(dlt))})
    return draws, pd.DataFrame(rows, columns=cols)


def xv3_boot_intervals(boot: pd.DataFrame, q: Tuple[float, float]) -> pd.DataFrame:
    """Per (cell, draw, policy): truth, point interval and bootstrap interval [q_lo(J_term^b), q_hi(J_mean^b)]."""
    cols = ["cell", "family", "draw", "policy", "true", "point_lo", "point_up", "p_bench", "viol", "boot_lo",
            "boot_up", "boot_lo_min", "boot_up_max", "n_ok_terminate", "n_ok_mean", "point_cover", "boot_cover",
            "boot_lo_ok", "boot_up_ok", "point_width", "boot_width"]
    if boot is None or not len(boot):
        return pd.DataFrame(columns=cols)
    k = ["cell", "draw", "policy"]
    b = boot.copy()
    b["draw"] = pd.to_numeric(b["draw"]).astype(np.int64)
    b["replicate"] = pd.to_numeric(b["replicate"]).astype(np.int64)
    pt = b[b["replicate"] == XV3_BOOT_POINT].set_index(k + ["target"])
    J0 = pt["return"].unstack("target")
    out = pd.DataFrame({"true": J0["true"], "point_lo": J0["bench_terminate"], "point_up": J0["bench_mean"],
                        "p_bench": pt["prob_any_violation"].unstack("target")["bench_mean"]})
    out = out.join(b.drop_duplicates(k).set_index(k)[["family"]])
    bs = b[(b["replicate"] >= 0) & (b["status"] == "ok")]
    g = bs.groupby(k + ["target"])["return"]
    tg = ["bench_mean", "bench_terminate"]
    qlo = g.quantile(float(q[0])).unstack("target").reindex(columns=tg)
    qhi = g.quantile(float(q[1])).unstack("target").reindex(columns=tg)
    mn, mx = g.min().unstack("target").reindex(columns=tg), g.max().unstack("target").reindex(columns=tg)
    n = g.size().unstack("target").reindex(columns=tg)
    out["boot_lo"] = qlo.reindex(out.index)["bench_terminate"]
    out["boot_up"] = qhi.reindex(out.index)["bench_mean"]
    out["boot_lo_min"] = mn.reindex(out.index)["bench_terminate"]
    out["boot_up_max"] = mx.reindex(out.index)["bench_mean"]
    out["n_ok_terminate"] = n.reindex(out.index)["bench_terminate"].fillna(0).astype(int)
    out["n_ok_mean"] = n.reindex(out.index)["bench_mean"].fillna(0).astype(int)
    out["viol"] = out["p_bench"] > XV3_BR_PTOL
    out["point_cover"] = (out["true"] >= out["point_lo"] - XV3_TOL) & (out["true"] <= out["point_up"] + XV3_TOL)
    out["boot_lo_ok"] = out["true"] >= out["boot_lo"] - XV3_TOL
    out["boot_up_ok"] = out["true"] <= out["boot_up"] + XV3_TOL
    out["boot_cover"] = out["boot_lo_ok"] & out["boot_up_ok"]
    out["point_width"] = out["point_up"] - out["point_lo"]
    out["boot_width"] = out["boot_up"] - out["boot_lo"]
    return out.reset_index()[cols]


def xv3_boot_coverage(iv: pd.DataFrame, cells: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Coverage and certified comparisons of point and bootstrap intervals per cell and pooled; per-draw counts."""
    if iv.empty:
        return pd.DataFrame(), pd.DataFrame()
    cc = []
    for (c, dr), d in iv.groupby(["cell", "draw"], sort=False):
        for kind, (lc, uc) in (("point", ("point_lo", "point_up")), ("bootstrap", ("boot_lo", "boot_up"))):
            dd = d[d[[lc, uc, "true"]].notna().all(axis=1)]
            cc.append({"cell": c, "draw": int(dr), "family": d["family"].iloc[0], "targets": kind,
                       "policies_without_interval": int(len(d) - len(dd)),
                       **xv3_pairs(dd[lc].to_numpy(float), dd[uc].to_numpy(float), dd["true"].to_numpy(float))})
    CC = pd.DataFrame(cc)
    fams = sorted(set(iv["family"]))
    units = [(c, [c]) for c in cells if c in set(iv["cell"])]
    units += [(f"pooled:{f}", sorted(set(iv.loc[iv["family"] == f, "cell"]))) for f in fams]
    if len(fams) > 1:
        units.append(("pooled:all", sorted(set(iv["cell"]))))
    rows = []
    for name, cs in units:
        d = iv[iv["cell"].isin(cs)]
        v = d[d["viol"]]
        for kind, cov, w, lo_ok, up_ok in (("point", "point_cover", "point_width", None, None),
                                           ("bootstrap", "boot_cover", "boot_width", "boot_lo_ok", "boot_up_ok")):
            defined = d[[w]].notna().all(axis=1)
            dv = v[v[w].notna()]
            r = {"cell": name, "n_cells": len(cs), "interval": kind, "n_rows": int(len(d)),
                 "n_undefined": int((~defined).sum()), "n_violating": int(len(dv)),
                 "coverage_all": _xv3_rate(d.loc[defined, cov].sum(), int(defined.sum())),
                 "coverage_violating": _xv3_rate(dv[cov].sum(), len(dv)),
                 "mean_width_violating": float(dv[w].mean()) if len(dv) else np.nan}
            if lo_ok:
                r["lower_holds_violating"] = _xv3_rate(dv[lo_ok].sum(), len(dv))
                r["upper_holds_violating"] = _xv3_rate(dv[up_ok].sum(), len(dv))
            else:
                r["lower_holds_violating"] = _xv3_rate((dv["true"] >= dv["point_lo"] - XV3_TOL).sum(), len(dv))
                r["upper_holds_violating"] = _xv3_rate((dv["true"] <= dv["point_up"] + XV3_TOL).sum(), len(dv))
            r.update({f"cc_{k_}": v_ for k_, v_ in
                      xv3_cc_summary(CC[CC["cell"].isin(cs) & (CC["targets"] == kind)]).items()})
            rows.append(r)
    return pd.DataFrame(rows), CC


def _xv3_legend_below(ax, ncol: int = 1) -> None:
    """Legend under the axes (below the x label), so it never covers data."""
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=ncol)


def _xv3_col(t: pd.DataFrame, name: str) -> np.ndarray:
    """Numeric column of `t` as floats (NaN when the column is absent, e.g. no cell with true-rows targets)."""
    return (pd.to_numeric(t[name], errors="coerce").to_numpy(dtype=float) if name in t.columns
            else np.full(len(t), np.nan))


def xs_fig_v3_bracketing(table: pd.DataFrame, draws: pd.DataFrame, cells: Sequence[str], fig_dir: Path) -> str:
    t = table[table["cell"].isin(cells)] if len(table) else table
    if t.empty:
        return "not produced: no II.8 cell results"
    t = t.set_index("cell").loc[[c for c in cells if c in set(t["cell"])]]
    rand = draws[draws["kind"] == "garnet-random"] if len(draws) else draws
    has_b = len(rand) and rand["hidden_benefit_share"].notna().any() and rand["upper_coverage_violating"].notna().any()
    fig, axes = plt.subplots(1, 2 if has_b else 1, figsize=(11, 3.6) if has_b else (6.5, 3.6), squeeze=False)
    ax = axes[0, 0]
    y = np.arange(len(t))
    h = 0.38
    sem = 100 * _xv3_col(t, "sem_upper_coverage_violating")
    ben = 100 * _xv3_col(t, "bench_coverage_violating")
    ax.barh(y - 0.2, np.nan_to_num(sem), height=h, color=XS_COLORS[0], label="upper bound on the true rows")
    ax.barh(y + 0.2, np.nan_to_num(ben), height=h, color=XS_COLORS[1], label="benchmark interval [J_term, J_mean]")
    for yi, s_ in zip(y, sem):
        if not np.isfinite(s_):
            ax.text(1, yi - 0.2, "n/a (aggregated states)", va="center", fontsize=7, color=XS_MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([xs_cell_label(c) for c in t.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("coverage of J_true among violating policies (%)")
    lv = _xv3_col(t, "sem_lower_violations")
    ax.set_title(f"(a) II.8 cells; lower-bound violations on the true rows: {int(np.nansum(lv))}", loc="left",
                 color=XS_INK, fontsize=9)
    _xv3_legend_below(ax, ncol=2)
    if has_b:
        ax = axes[0, 1]
        for i, (c, g) in enumerate(rand.groupby("cell", sort=False)):
            ax.scatter(g["hidden_benefit_share"], 100 * g["upper_coverage_violating"], s=30, color=XS_COLORS[i],
                       edgecolor="white", linewidth=0.8, label=xs_cell_label(c), zorder=3)
        ax.set_xlabel("hidden-benefit share of the draw")
        ax.set_ylabel("upper-bound coverage (%)")
        ax.set_title("(b) random-row Garnet draws (violating policies)", loc="left", color=XS_INK, fontsize=9)
        _xv3_legend_below(ax, ncol=2)
    return _xs_save(fig, fig_dir, "fig_v3_bracketing")


def xs_fig_v3_bootstrap(cov: pd.DataFrame, cells: Sequence[str], fig_dir: Path) -> str:
    c = cov[cov["cell"].isin(cells)] if len(cov) else cov
    if c.empty:
        return "not produced: no bootstrap results"
    order = [x for x in cells if x in set(c["cell"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    y = np.arange(len(order))
    h = 0.38
    for ax, col, xlab in zip(axes, ("coverage_violating", "cc_certified_share", "cc_certified_error_rate"),
                             ("coverage of J_true, violating policies (%)", "certified pairs (%)",
                              "certified pairs that are wrong (%)")):
        for i, kind in enumerate(("point", "bootstrap")):
            v = c[c["interval"] == kind].set_index("cell").reindex(order)[col]
            ax.barh(y + (i - 0.5) * 0.4, 100 * np.nan_to_num(pd.to_numeric(v, errors="coerce").to_numpy(float)),
                    height=h, color=XS_COLORS[i], label=f"{kind} interval")
        ax.set_xlabel(xlab)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([xs_cell_label(x) for x in order], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[1].set_xlim(0, 100)
    _xv3_legend_below(axes[0], ncol=2)
    axes[0].set_title("(a) coverage", loc="left", color=XS_INK, fontsize=9)
    axes[1].set_title("(b) certified share", loc="left", color=XS_INK, fontsize=9)
    axes[2].set_title("(c) certification error", loc="left", color=XS_INK, fontsize=9)
    return _xs_save(fig, fig_dir, "fig_v3_bootstrap")


def xv3_stats_run(results: pd.DataFrame, draw_meta: Optional[pd.DataFrame], cells_spec: Sequence[dict],
                  boot: pd.DataFrame, boot_cells: Sequence[dict], table_dir: Path, fig_dir: Path,
                  n_boot: int) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    """Section II.8i end to end: tables, figures and checks. Returns (XV3_STATS, done flags)."""
    t0 = time.perf_counter()
    table_dir, fig_dir = Path(table_dir), Path(fig_dir)
    kind_of = {str(s["cell"]): str(s["v3_kind"]) for s in cells_spec}
    cells = [str(s["cell"]) for s in cells_spec]
    outputs: List[str] = []

    def write_csv(df: pd.DataFrame, name: str) -> None:
        df.to_csv(table_dir / name, index=False)
        outputs.append(f"xenv/{name}")

    out: Dict[str, Any] = {}
    checks: Dict[str, Any] = {}
    done = {"cells": False, "bootstrap": False}
    figures: Dict[str, str] = {}
    if results is not None and len(results):
        W, checks["bracket_frame"] = xv3_bracket_frame(results, kind_of)
        out["bracketing"] = xv3_bracketing_table(W, cells, kind_of)
        out["bracketing_draws"] = xv3_bracketing_draws(W, draw_meta)
        CC = xv3_certified_draws(W)
        cc_rows = []
        for tf in ("true_rows", "bench"):
            units = [(c, [c]) for c in cells] + list(xv3_pools(cells, kind_of).items())
            for name, cs in units:
                d = CC[CC["cell"].isin(cs) & (CC["targets"] == tf)] if len(CC) else CC
                if len(d):
                    cc_rows.append({"cell": name, "targets": tf, **xv3_cc_summary(d)})
        out["certified"], out["certified_draws"] = pd.DataFrame(cc_rows), CC
        out["rank_fidelity_draws"], out["rank_fidelity"] = xv3_rank_fidelity(results, cells, n_boot)
        rb = out["bracketing_draws"]
        rand = rb[(rb["kind"] == "garnet-random") & rb["hidden_benefit_share"].notna()
                  & rb["upper_coverage_violating"].notna()] if len(rb) else rb
        checks["hidden_benefit_spearman"] = {
            c: {"draws": int(len(g)), "spearman_rho": xs_spearman(g["hidden_benefit_share"], g["upper_coverage_violating"])
                if len(g) >= 3 else None} for c, g in rand.groupby("cell")} if len(rand) else {}
        for name in ("bracketing", "bracketing_draws", "certified", "certified_draws", "rank_fidelity",
                     "rank_fidelity_draws"):
            write_csv(out[name], f"v3_{name}.csv")
        figures["fig_v3_bracketing"] = xs_fig_v3_bracketing(out["bracketing"], rb, cells, fig_dir)
        done["cells"] = True
        del W
    else:
        figures["fig_v3_bracketing"] = "not produced: XRESULTS_V3 is empty (II.8 cell switches off)"
    if boot is not None and len(boot):
        bcells = [str(s["cell"]) for s in boot_cells]
        iv = xv3_boot_intervals(boot, tuple(XV3BOOT.quantiles))
        cov, bcc = xv3_boot_coverage(iv, bcells)
        out["bootstrap_intervals"], out["bootstrap_coverage"], out["bootstrap_certified_draws"] = iv, cov, bcc
        write_csv(iv, "v3_bootstrap_intervals.csv")
        write_csv(cov, "v3_bootstrap_coverage.csv")
        checks["bootstrap"] = {"policy_rows": int(len(iv)), "undefined_bootstrap_intervals": int(iv["boot_width"].isna().sum()),
                               "negative_point_width": int((iv["point_width"] < -XV3_TOL).sum()),
                               "quantiles": list(XV3BOOT.quantiles), "replicates": int(XV3BOOT.replicates)}
        figures["fig_v3_bootstrap"] = xs_fig_v3_bootstrap(cov, bcells, fig_dir)
        done["bootstrap"] = True
    else:
        figures["fig_v3_bootstrap"] = "not produced: no bootstrap rows (xenv_v3_bootstrap off or no cell enabled)"
    outputs += [f"figures/xenv/{k}.{ext}" for k, v in figures.items() if v == "written" for ext in ("png", "pdf")]
    bf = checks.get("bracket_frame", {})
    theorem_counts = {k: bf.get(k, 0) for k in ("values_outside_value_range", "lower_bound_violations_true_rows",
                                                 "negative_width_true_rows", "negative_width_bench",
                                                 "nonzero_width_admissible", "width_above_bound")}
    theorem_counts["negative_point_width_bootstrap"] = checks.get("bootstrap", {}).get("negative_point_width", 0)
    checks.update({"theorem_counts": theorem_counts, "passes": all(v == 0 for v in theorem_counts.values()),
                   "done": done, "figures": figures, "outputs": outputs, "n_boot": int(n_boot),
                   "tolerance": XV3_TOL, "violating_threshold": XV3_BR_PTOL, "value_ranges": XV3_VRANGE,
                   "elapsed_s": time.perf_counter() - t0})
    save_json(checks, table_dir / "v3_stats_checks.json")
    outputs.append("xenv/v3_stats_checks.json")
    out.update({"checks": checks, "figures": figures, "outputs": outputs})
    return out, done


XV3_STATS, XV3_STATS_DONE = xv3_stats_run(XRESULTS_V3, XDRAW_META_V3, X_V3_CELLS, XV3_BOOT_RESULTS, XV3_BOOT_CELLS,
                                          XENV_TABLE_DIR, XENV_FIG_DIR, int(XCFG.bootstrap))
print("II.8i theorem checks:", XV3_STATS["checks"]["theorem_counts"], "| passes:", XV3_STATS["checks"]["passes"])
if "bracketing" in XV3_STATS:
    display(XV3_STATS["bracketing"][["cell", "sem_n_violating", "sem_lower_violations", "sem_upper_coverage_violating",
                                     "bench_n_violating", "bench_coverage_violating"]].round(4))
if "certified" in XV3_STATS and len(XV3_STATS["certified"]):
    _cols = [c for c in ("cell", "targets", "draws", "certified_share", "certified_error_rate", "point_mean_error_rate",
                         "point_terminate_error_rate") if c in XV3_STATS["certified"].columns]
    display(XV3_STATS["certified"][_cols].round(4))
if "bootstrap_coverage" in XV3_STATS and len(XV3_STATS["bootstrap_coverage"]):
    display(XV3_STATS["bootstrap_coverage"][["cell", "interval", "n_violating", "coverage_violating",
                                             "mean_width_violating", "cc_certified_share",
                                             "cc_certified_error_rate"]].round(4))
print("Figures:", XV3_STATS["figures"], "| done:", XV3_STATS_DONE)
if not XV3_STATS["checks"]["passes"]:
    raise AssertionError(f"II.8i theorem checks failed: {XV3_STATS['checks']['theorem_counts']} (xenv/v3_stats_checks.json)")
