"""Robustness analyses of certified comparisons, from the stored exact results (no rerun).

Usage: python -I make_robustness.py <colab paper/tables dir> <paper dir>
Reads <tables>/xenv/xenv_results_all.csv.gz (15 Part II simulator cells + 50 ICU-Sepsis draws) and
<tables>/../exact_results_all.csv (Part I). Writes <paper>/facts_robust.json and <paper>/tables/tab_robust*.tex.

Analyses:
  1. NHB-free certificates. Theorem 1(iv): over all completions of the supported rows, the value of pi lies in
     [J_term(pi), J_term(pi) + p_pi (max_T R - v_min)]. Certifying pi_1 > pi_2 against that upper end needs no
     assumption on the unsupported rows of either policy. Compared side by side with the NHB-conditional certificate
     (upper end J_W = J_mean) of the paper.
  2. Pair types. Pairs of default policies split by whether each policy makes an unsupported decision with positive
     probability (p > 0 on the same targets): AA both admissible, AU one, UU both.
  3. Distinct policies. Within a draw, default policies with identical values on all five targets (to 1e-12) are
     merged before counting pairs.
  4. Margin abstention at matched coverage. Per draw, a single rule claims i > j for the k pairs with the largest
     value gap, k = number of certified pairs in that draw; error rates are compared with certification.
  5. Draw-level counts: draws with at least one wrong certified claim; a draw-clustered bootstrap interval of the
     pooled error rate.
Definitions match make_assets.py (tolerance 1e-9, p > 1e-12, errors = truth reverses or ties).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
X = SRC / "xenv"
TOL, PTOL = 1e-9, 1e-12
# Absorbing rewards (notebook): Gumbel death -1, discharge +1, timeout 0; Garnet death 0, goal 1; ICU death 0, survival 1.
# The terminate rule jumps to death, so r_term = v_min; max_T R = v_max in every family.
VR = {"gumbel": (-1.0, 1.0), "garnet": (0.0, 1.0), "icu": (0.0, 1.0)}
TG = ["bench_mean", "bench_terminate", "true", "true_rows_mean", "true_rows_terminate"]
FACTS = json.loads((OUT / "facts.json").read_text())

xr = pd.read_csv(X / "xenv_results_all.csv.gz", low_memory=False)
xr = xr[xr["target"].isin(TG)]
assert (xr["status"] == "ok").all()
k = ["cell", "draw", "policy"]
J = xr.set_index(k + ["target"])["return"].unstack("target")
P = xr.set_index(k + ["target"])["prob_any_violation"].unstack("target")
J = J.join(xr.drop_duplicates(k).set_index(k)[["family", "is_default", "is_reference"]])
J["p_bench"], J["p_true"] = P["bench_mean"], P["true_rows_mean"]
orc = J.index.get_level_values("cell").str.contains("|oracle|", regex=False)
for a, b in (("true_rows_mean", "bench_mean"), ("true_rows_terminate", "bench_terminate"), ("p_true", "p_bench")):
    assert J.loc[orc, a].isna().all()
    J.loc[orc, a] = J.loc[orc, b]
D = J[J["is_default"].astype(bool) & ~J["is_reference"].astype(bool)].sort_index()

CELLS = sorted(D.index.get_level_values("cell").unique())
GUM_EST = [c for c in CELLS if c.startswith("gumbel|best|")]
GAR = [c for c in CELLS if c.startswith("garnet|")]
POOLS = {"gumbel_estimated": GUM_EST, "gumbel_oracle": [c for c in CELLS if "|oracle|" in c], "garnet_all": GAR,
         "garnet_h-0.5": [c for c in GAR if "|h-0.5|" in c], "garnet_h_ge_0.5": [c for c in GAR if c.split("|")[1] in ("h0.5", "h1.0")],
         "garnet_h0.0": [c for c in GAR if "|h0.0|" in c], "icu": ["icu|released"]}
TARGETS = {"true_rows": ("true_rows_terminate", "true_rows_mean", "p_true"), "bench": ("bench_terminate", "bench_mean", "p_bench")}


def gt(a, b):
    return a[:, None] > b[None, :] + TOL


def pair_stats(lo, up, p, rng, jt):
    """Ordered-pair claim matrices for one draw. Returns dict of count arrays keyed by claim kind and pair type."""
    n = len(lo)
    off = ~np.eye(n, dtype=bool)
    upf = lo + p * rng                       # NHB-free upper end, Theorem 1(iv)
    claims = {"cert_nhb": gt(lo, up), "cert_free": gt(lo, upf), "point_mean": gt(up, up), "point_term": gt(lo, lo)}
    claims["cert_nhb_only"] = claims["cert_nhb"] & ~claims["cert_free"]   # certified only because of NHB for the loser
    u = p > PTOL
    # With true rows a certified claim can fail only if the LOSING policy (column j) has p > 0: a point interval is the
    # truth (Thm 1(iii)) and a winner with p > 0 is protected by the lower bound (Thm 1(i)). These claims are the test.
    claims["cert_nhb_testable"] = claims["cert_nhb"] & u[None, :]
    typ = np.where(u[:, None] & u[None, :], "UU", np.where(u[:, None] | u[None, :], "AU", "AA"))
    wrong = None if jt is None else ~gt(jt, jt)
    return claims, typ, off, wrong


def margin_claims(v, kpairs):
    """Claims i > j for the kpairs unordered pairs with the largest gap |v_i - v_j| under one rule (ties at the
    threshold broken by order; gaps <= TOL never claimed)."""
    n = len(v)
    iu = np.triu_indices(n, 1)
    gap = np.abs(v[iu[0]] - v[iu[1]])
    order = np.argsort(-gap, kind="stable")[:kpairs]
    order = order[gap[order] > TOL]
    C = np.zeros((n, n), dtype=bool)
    i, j = iu[0][order], iu[1][order]
    hi = v[i] > v[j]
    C[np.where(hi, i, j), np.where(hi, j, i)] = True
    return C


rows = []
for (c, dr), d in D.groupby(level=["cell", "draw"]):
    fam = d["family"].iloc[0]
    rng = VR[fam][1] - VR[fam][0]
    jt = d["true"].to_numpy(float) if fam != "icu" else None
    for tf, (lc, uc, pc) in TARGETS.items():
        if d[lc].isna().any():
            continue
        lo, up, p = d[lc].to_numpy(float), d[uc].to_numpy(float), d[pc].to_numpy(float)
        assert (up - lo >= -1e-12).all() and (up - lo <= p * rng + 1e-12).all()
        assert not (gt(lo, lo + p * rng) & ~gt(lo, up)).any(), "NHB-free certificates are NHB certificates"
        # distinct policies: identical values on every available target
        vals = d[[t for t in TG if d[t].notna().all()]].to_numpy(float)
        _, first = np.unique(np.round(vals / 1e-12).astype(np.int64), axis=0, return_index=True)
        keep = np.zeros(len(d), dtype=bool)
        keep[first] = True
        for dist in (False, True):
            m = keep if dist else np.ones(len(d), dtype=bool)
            claims, typ, off, wrong = pair_stats(lo[m], up[m], p[m], rng, None if jt is None else jt[m])
            ncert = int(claims["cert_nhb"].sum())
            claims["margin_mean"] = margin_claims(up[m], ncert)
            claims["margin_term"] = margin_claims(lo[m], ncert)
            claims["margin_mid"] = margin_claims((lo[m] + up[m]) / 2, ncert)
            nf = int(claims["cert_free"].sum())
            claims["margin_mean_free"] = margin_claims(up[m], nf)
            for ty in ("all", "AA", "AU", "UU"):
                tm = off if ty == "all" else (off & (typ == ty))
                r = {"cell": c, "draw": int(dr), "family": fam, "targets": tf, "distinct": dist, "type": ty,
                     "n_policies": int(m.sum()), "pairs": int(tm.sum() // 2)}
                for nm, C in claims.items():
                    r[f"{nm}_claims"] = int((C & tm).sum())
                    if wrong is not None:
                        r[f"{nm}_errors"] = int((C & tm & wrong).sum())
                rows.append(r)
R = pd.DataFrame(rows)

# Reproduce the paper's existing numbers before reporting anything new.
cc = FACTS["certified_comparisons"]
for tf in ("true_rows", "bench"):
    for pool in ("gumbel_estimated", "garnet_all"):
        s = R[(R["targets"] == tf) & ~R["distinct"] & (R["type"] == "all") & R["cell"].isin(POOLS[pool])].sum(numeric_only=True)
        ref = cc[tf]["pooled:" + pool]
        assert int(s["cert_nhb_claims"]) == ref["certified"] and int(s["cert_nhb_errors"]) == ref["certified_errors"], (tf, pool)
        assert int(s["pairs"]) == ref["pairs"]
        assert int(s["point_mean_errors"]) == ref["point_mean_errors"] and int(s["point_term_errors"]) == ref["point_terminate_errors"]
s = R[(R["targets"] == "bench") & ~R["distinct"] & (R["type"] == "all") & (R["cell"] == "icu|released")].sum(numeric_only=True)
assert int(s["cert_nhb_claims"]) == cc["bench"]["icu|released"]["certified"]
print("reproduces facts.json certified counts and errors (true rows, benchmark, ICU)")

KINDS = ["cert_nhb", "cert_free", "cert_nhb_only", "cert_nhb_testable", "point_mean", "point_term", "margin_mean", "margin_term", "margin_mid",
         "margin_mean_free"]


def summ(d):
    s = d.sum(numeric_only=True)
    o = {"draws": int(d.groupby(["cell", "draw"]).ngroups), "pairs": int(s["pairs"])}
    for nm in KINDS:
        cl = int(s[f"{nm}_claims"])
        o[f"{nm}_claims"] = cl
        o[f"{nm}_share"] = cl / s["pairs"] if s["pairs"] else float("nan")
        if f"{nm}_errors" in d and d[f"{nm}_errors"].notna().all():
            e = int(s[f"{nm}_errors"])
            o[f"{nm}_errors"] = e
            o[f"{nm}_error_rate"] = e / cl if cl else float("nan")
    if "cert_nhb_errors" in d and d["cert_nhb_errors"].notna().all():
        g = d.groupby(["cell", "draw"])[["cert_nhb_errors", "cert_free_errors"]].sum()
        o["draws_with_cert_nhb_error"] = int((g["cert_nhb_errors"] > 0).sum())
        o["draws_with_cert_free_error"] = int((g["cert_free_errors"] > 0).sum())
    if "cert_nhb_only_claims" in d:
        g = d.groupby(["cell", "draw"])[["cert_nhb_claims", "cert_nhb_only_claims"]].sum()
        o["draws_with_cert_nhb"] = int((g["cert_nhb_claims"] > 0).sum())
        o["draws_with_cert_nhb_only"] = int((g["cert_nhb_only_claims"] > 0).sum())
    return o


rs = np.random.default_rng(20261025)


def cluster_ci(d, nm, B=5000):
    """Draw-clustered percentile bootstrap of the pooled error rate errors/claims (draws resampled within cells)."""
    g = d.groupby(["cell", "draw"])[[f"{nm}_errors", f"{nm}_claims"]].sum()
    cells = g.index.get_level_values("cell")
    e, c = g[f"{nm}_errors"].to_numpy(float), g[f"{nm}_claims"].to_numpy(float)
    idx = [np.flatnonzero(cells == cl) for cl in pd.unique(cells)]
    est = []
    for _ in range(B):
        pick = np.concatenate([ix[rs.integers(0, len(ix), len(ix))] for ix in idx])
        est.append(e[pick].sum() / max(c[pick].sum(), 1))
    return [float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))]


OUTF = {"definition": __doc__.split("Definitions")[0].strip(), "tolerance": TOL, "p_tol": PTOL, "value_ranges": VR,
        "source": "xenv_results_all.csv.gz (default, non-reference policies; 15 simulator cells x 30 draws, ICU 50 draws)"}
for tf in TARGETS:
    for dist in (False, True):
        key = f"{tf}{'_distinct' if dist else ''}"
        OUTF[key] = {}
        for pool, cells in POOLS.items():
            for ty in ("all", "AA", "AU", "UU"):
                d = R[(R["targets"] == tf) & (R["distinct"] == dist) & (R["type"] == ty) & R["cell"].isin(cells)]
                if len(d):
                    OUTF[key].setdefault(pool, {})[ty] = summ(d)
        for c in CELLS:
            d = R[(R["targets"] == tf) & (R["distinct"] == dist) & (R["type"] == "all") & (R["cell"] == c)]
            if len(d):
                OUTF[key].setdefault("cells", {})[c] = summ(d)
# draw-clustered intervals for the benchmark-level error rates (the paper's 10.6% and 9.2%)
OUTF["bench_cluster_ci"] = {}
for pool in ("gumbel_estimated", "garnet_all"):
    d = R[(R["targets"] == "bench") & ~R["distinct"] & (R["type"] == "all") & R["cell"].isin(POOLS[pool])]
    OUTF["bench_cluster_ci"][pool] = {nm: cluster_ci(d, nm) for nm in ("cert_nhb", "cert_free", "point_mean", "point_term", "margin_mean", "margin_term")}

# ---- ICU-Sepsis Part I (released rows, seed means), side by side
res = pd.read_csv(SRC / "exact_results_all.csv")
p1 = res[res["group"].isin(["main", "reference"]) & (res["status"] == "ok")]
v = p1.pivot_table(index=["group", "name", "seed"], columns="rule", values="return", aggfunc="first")
v["p"] = p1[p1["rule"] == "mean"].set_index(["group", "name", "seed"])["prob_any_violation"]
g = v.groupby(level=["group", "name"]).agg(lo=("terminate", "mean"), up=("mean", "mean"), p=("p", "mean"), pmax=("p", "max"))
# Seed means: the NHB-free end of the mean policy is bounded by the mean of the per-seed ends (linear), so use them.
v["upf"] = v["terminate"] + v["p"] * 1.0
g["upf"] = v.groupby(level=["group", "name"])["upf"].mean()
ICU_ALL = g.copy()          # Part I, all policies (g is reused below)
icu_rows = FACTS["icu_intervals"]["main_table"]
mt = g.reset_index().set_index("name").loc[[r["key"] for r in icu_rows]]
lab = [r["policy"] for r in icu_rows]
assert np.allclose(mt["lo"], [r["J_term"] for r in icu_rows]) and np.allclose(mt["up"], [r["J_mean"] for r in icu_rows])


def icu_block(t, labels=None):
    lo, up, upf = t["lo"].to_numpy(), t["up"].to_numpy(), t["upf"].to_numpy()
    Cn, Cf = gt(lo, up), gt(lo, upf)
    np.fill_diagonal(Cn, False)
    np.fill_diagonal(Cf, False)
    n = len(t)
    u = t["pmax"].to_numpy() > PTOL
    typ = np.where(u[:, None] & u[None, :], "UU", np.where(u[:, None] | u[None, :], "AU", "AA"))
    off = ~np.eye(n, dtype=bool)
    o = {"n_policies": n, "pairs": n * (n - 1) // 2, "cert_nhb": int(Cn.sum()), "cert_free": int(Cf.sum())}
    for ty in ("AA", "AU", "UU"):
        tm = off & (typ == ty)
        o[ty] = {"pairs": int(tm.sum() // 2), "cert_nhb": int((Cn & tm).sum()), "cert_free": int((Cf & tm).sum())}
    if labels is not None:
        o["nhb_only_pairs"] = [(labels[i], labels[j]) for i, j in zip(*np.nonzero(Cn & ~Cf))]
        o["rows"] = [{"policy": labels[i], "J_term": float(lo[i]), "J_mean": float(up[i]), "upper_free": float(upf[i]),
                      "p": float(t["p"].iloc[i]), "cert_above_nhb": int(Cn[i].sum()), "cert_above_free": int(Cf[i].sum()),
                      "cert_below_nhb": int(Cn[:, i].sum()), "cert_below_free": int(Cf[:, i].sum())} for i in range(n)]
    return o


OUTF["icu_part1_main_table"] = icu_block(mt, lab)
OUTF["icu_part1_all"] = icu_block(g)
assert OUTF["icu_part1_main_table"]["cert_nhb"] == FACTS["icu_certified"]["part1_main_table"]["certified"]
assert OUTF["icu_part1_all"]["cert_nhb"] == FACTS["icu_certified"]["part1_all"]["certified"]
(OUT / "facts_robust.json").write_text(json.dumps(OUTF, indent=1, default=float))
print("wrote facts_robust.json")


# ---- quick console summary
def pct(x):
    return "--" if x != x else f"{100 * x:.1f}"


for tf in TARGETS:
    for pool in ("gumbel_estimated", "garnet_all", "garnet_h-0.5", "garnet_h_ge_0.5", "gumbel_oracle", "icu"):
        if pool not in OUTF[tf]:
            continue
        for ty in ("all", "AA", "AU", "UU"):
            o = OUTF[tf][pool].get(ty)
            if not o:
                continue
            line = f"{tf:9s} {pool:16s} {ty:3s} pairs {o['pairs']:7d} | "
            for nm in KINDS:
                line += f"{nm} {pct(o[nm + '_share'])}%/{pct(o.get(nm + '_error_rate', float('nan')))}  "
            print(line)
print(json.dumps({kk: OUTF[kk] for kk in ("bench_cluster_ci",)}, indent=0))
for kk in ("icu_part1_main_table", "icu_part1_all"):
    o = OUTF[kk]
    print(kk, {x: o[x] for x in ("n_policies", "pairs", "cert_nhb", "cert_free", "AA", "AU", "UU")})
print(OUTF["icu_part1_main_table"]["nhb_only_pairs"])


# =====================================================================================================================
# Tables. tab_robust (main text, Section 5.1) and tab_pairtypes (appendix app:robust).
# =====================================================================================================================
TAB = OUT / "tables"


def p1(x):
    """Percentage with one decimal; an exact 0 prints as 0, a positive rate below 0.05% as <0.1."""
    if x is None or x != x:
        return "--"
    if x == 0:
        return "0"
    v = 100.0 * x
    return r"$<$0.1" if v < 0.05 else f"{v:.1f}"


def comb(o, types, nm):
    """Error rate of claim kind nm over several pair types (errors / claims)."""
    cl = sum(o[t][f"{nm}_claims"] for t in types)
    er = sum(o[t].get(f"{nm}_errors", 0) for t in types)
    return er / cl if cl else float("nan"), er, cl


ROBUST_ROWS = [("Gumbel, true rows (6 cells)", "true_rows", "gumbel_estimated"),
               ("Garnet, true rows (8 cells)", "true_rows", "garnet_all"),
               (r"\quad of which $h{=}{-}0.5$ (2 cells)", "true_rows", "garnet_h-0.5"),
               ("Gumbel, estimated rows", "bench", "gumbel_estimated"),
               ("Garnet, estimated rows", "bench", "garnet_all")]
lines = [r"\setlength{\tabcolsep}{3.5pt}%", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{NHB certificate} & \multicolumn{2}{c}{NHB-free certificate} & needs NHB"
         r" & \multicolumn{3}{c}{margin rule, same coverage} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}\cmidrule(lr){7-9}",
         r"Setting & share & err. & share & err. & err. & mean & term & mid \\", r"\midrule"]
OUTF["tab_robust"] = {}
for lab_, tf, pool in ROBUST_ROWS:
    o = OUTF[tf][pool]
    a = o["all"]
    up, _, _ = comb(o, ("AU", "UU"), "cert_nhb") if "cert_nhb_errors" in a else (float("nan"), 0, 0)
    only = a.get("cert_nhb_only_error_rate")
    lines.append(f"{lab_} & {p1(a['cert_nhb_share'])} & {p1(a.get('cert_nhb_error_rate'))} & {p1(a['cert_free_share'])} & "
                 f"{p1(a.get('cert_free_error_rate'))} & {p1(only)} & {p1(a.get('margin_mean_error_rate'))} & "
                 f"{p1(a.get('margin_term_error_rate'))} & {p1(a.get('margin_mid_error_rate'))} \\\\")
    OUTF["tab_robust"][lab_] = {"targets": tf, "pool": pool, "cert_nhb_error_rate_p_pos_pairs": up}
    if lab_.startswith(r"\quad"):
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_robust.tex").write_text("\n".join(lines) + "\n")

PT_POOLS = [("Gumbel (6 estimated cells)", "gumbel_estimated"), (r"Garnet $h{=}{-}0.5$", "garnet_h-0.5"),
            (r"Garnet $h{=}0$", "garnet_h0.0"), (r"Garnet $h{\ge}0.5$", "garnet_h_ge_0.5")]
TYLAB = {"AA": "neither with $p_\\pi{>}0$", "AU": "one with $p_\\pi{>}0$", "UU": "both with $p_\\pi{>}0$"}
lines = [r"\setlength{\tabcolsep}{3pt}%", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r" & & & \multicolumn{2}{c}{NHB certificate} & \multicolumn{2}{c}{NHB-free certificate} & \multicolumn{2}{c}{point err.} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Cells & pair type & pairs & share & err. & share & err. & mean & term \\", r"\midrule"]
for tf, tlab in (("true_rows", "true supported rows"), ("bench", "estimated rows (benchmark)")):
    lines.append(r"\multicolumn{9}{l}{\textit{%s}} \\" % tlab)
    for plab, pool in PT_POOLS:
        for i, (key, ty) in enumerate([(tf, "AA"), (tf, "AU"), (tf, "UU"), (tf + "_distinct", "all")]):
            o = OUTF[key][pool][ty]
            first = plab if i == 0 else ""
            tl = TYLAB.get(ty, "distinct policies")
            lines.append(f"{first} & {tl} & {o['pairs']:,} & {p1(o['cert_nhb_share'])} & {p1(o['cert_nhb_error_rate'])} & "
                         f"{p1(o['cert_free_share'])} & {p1(o['cert_free_error_rate'])} & {p1(o['point_mean_error_rate'])} & "
                         f"{p1(o['point_term_error_rate'])} \\\\".replace(",", "{,}"))
        lines.append(r"\addlinespace[2pt]")
    if tf == "true_rows":
        lines[-1] = r"\midrule"
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
(TAB / "tab_pairtypes.tex").write_text("\n".join(lines) + "\n")
(OUT / "facts_robust.json").write_text(json.dumps(OUTF, indent=1, default=float))
print("wrote tables/tab_robust.tex, tables/tab_pairtypes.tex")


# =====================================================================================================================
# Bootstrap intervals against abstention (approximate, local). The per-draw bootstrap certificates are on Drive only;
# v3_tables/bootstrap_v3.csv has each cell's bootstrap certified share (4 decimals). A single rule on the POINT benchmark
# values claims, per cell, the K = round(share * pairs) pairs with the largest gaps (pooled over the cell's draws; per-draw
# allocation as a check), and its error is compared with the bootstrap certified error of that cell.
# =====================================================================================================================
BOOT = pd.read_csv(OUT / "v3_tables" / "bootstrap_v3.csv")
BOOT = BOOT[(BOOT["interval"] == "bootstrap") & ~BOOT["cell"].str.startswith("pooled")]
full = {c: c for c in CELLS}
BOOT["cell"] = BOOT["cell"].map(lambda c: next(k for k in CELLS if k.startswith(c[:-3])) if c.endswith("...") else c)
assert BOOT["cell"].isin(CELLS).all() and len(BOOT) == 7


def gaps(d, score):
    """All unordered pairs of one draw: (gap, correct-if-claimed) for a claim of the higher-scored policy."""
    v, jt = score(d), d["true"].to_numpy(float)
    iu = np.triu_indices(len(v), 1)
    g = v[iu[0]] - v[iu[1]]
    hi = np.where(g > 0, iu[0], iu[1])
    lo = np.where(g > 0, iu[1], iu[0])
    return np.abs(g), ~(jt[hi] > jt[lo] + TOL)


SCORES = {"mean": lambda d: d["bench_mean"].to_numpy(float), "term": lambda d: d["bench_terminate"].to_numpy(float),
          "mid": lambda d: (d["bench_mean"].to_numpy(float) + d["bench_terminate"].to_numpy(float)) / 2}
bm = {}
for _, b in BOOT.iterrows():
    c = b["cell"]
    dd = [d for _, d in D.loc[c].groupby(level="draw")]
    npairs = sum(len(d) * (len(d) - 1) // 2 for d in dd)
    K = int(round(b["cc_certified_share"] * npairs))
    o = {"pairs": npairs, "claims": K, "boot_error_rate": float(b["cc_certified_error_rate"]),
         "boot_errors_approx": float(b["cc_certified_error_rate"]) * K}
    for nm, sc in SCORES.items():
        G = [gaps(d, sc) for d in dd]
        g = np.concatenate([x[0] for x in G]); w = np.concatenate([x[1] for x in G])
        order = np.argsort(-g, kind="stable")[:K]
        order = order[g[order] > TOL]
        o[f"{nm}_claims"], o[f"{nm}_errors"] = int(len(order)), int(w[order].sum())
        kd = int(round(b["cc_certified_share"] * len(dd[0]) * (len(dd[0]) - 1) / 2))     # per-draw allocation
        e = n = 0
        for gg, ww in G:
            od = np.argsort(-gg, kind="stable")[:kd]
            od = od[gg[od] > TOL]
            e, n = e + int(ww[od].sum()), n + len(od)
        o[f"{nm}_errors_perdraw"], o[f"{nm}_claims_perdraw"] = e, n
    bm[c] = o
POOLB = {"gumbel": [c for c in bm if c.startswith("gumbel")], "garnet": [c for c in bm if c.startswith("garnet")], "all": list(bm)}
OUTF["bootstrap_vs_margin"] = {"definition": "per cell: K = round(bootstrap certified share x pairs) claims of one rule on the "
                               "point benchmark values, largest gaps first (pooled over the cell's draws); *_perdraw: same share "
                               "per draw; boot errors from the rounded rates (approximate)", "cells": bm, "pools": {}}
for pool, cs in POOLB.items():
    o = {"claims": sum(bm[c]["claims"] for c in cs), "boot_error_rate": sum(bm[c]["boot_errors_approx"] for c in cs) /
         sum(bm[c]["claims"] for c in cs)}
    for nm in SCORES:
        for suf in ("", "_perdraw"):
            o[f"{nm}{suf}_error_rate"] = sum(bm[c][f"{nm}_errors{suf}"] for c in cs) / sum(bm[c][f"{nm}_claims{suf}"] for c in cs)
    OUTF["bootstrap_vs_margin"]["pools"][pool] = o
    print("bootstrap vs margin", pool, {k: round(v, 4) if isinstance(v, float) else v for k, v in o.items()})
(OUT / "facts_robust.json").write_text(json.dumps(OUTF, indent=1, default=float))


# =====================================================================================================================
# Certified claims between two admissible policies on the estimated benchmark: no completion changes their order, so
# their errors are estimation and construction error. Error rate by the certified gap, as a share of v_max - v_min.
# =====================================================================================================================
BINS = [0, 0.0025, 0.005, 0.01, 0.025, 0.05, np.inf]
gap_rows = []
for (c, dr), d in D.groupby(level=["cell", "draw"]):
    fam = d["family"].iloc[0]
    if fam == "icu":
        continue
    rng = VR[fam][1] - VR[fam][0]
    lo, up, p = d["bench_terminate"].to_numpy(float), d["bench_mean"].to_numpy(float), d["p_bench"].to_numpy(float)
    jt = d["true"].to_numpy(float)
    C = gt(lo, up)
    a = p <= PTOL
    C &= a[:, None] & a[None, :]
    i, j = np.nonzero(C)
    gap_rows.append(pd.DataFrame({"cell": c, "gap": (lo[i] - up[j]) / rng, "wrong": ~(jt[i] > jt[j] + TOL)}))
G = pd.concat(gap_rows)
G["bin"] = pd.cut(G["gap"], BINS, right=True)
OUTF["admissible_pair_gap"] = {"definition": "bench targets; NHB-certified claims with both policies admissible on the benchmark; "
                               "gap = (J_term(i) - J_mean(j)) / (v_max - v_min); wrong = truth reverses or ties", "pools": {}}
for pool, cells in (("gumbel_estimated", POOLS["gumbel_estimated"]), ("garnet_all", POOLS["garnet_all"]),
                    ("gumbel_oracle", POOLS["gumbel_oracle"])):
    g = G[G["cell"].isin(cells)]
    tab = g.groupby("bin", observed=False)["wrong"].agg(["size", "mean"])
    OUTF["admissible_pair_gap"]["pools"][pool] = {"claims": int(len(g)), "wrong": int(g["wrong"].sum()),
        "below_1pct_claims": int((g["gap"] <= 0.01).sum()), "below_1pct_error_rate": float(g.loc[g["gap"] <= 0.01, "wrong"].mean()),
        "bins": {str(b): {"claims": int(r["size"]), "error_rate": float(r["mean"]) if r["size"] else None} for b, r in tab.iterrows()}}
    print("AA gap", pool, len(g), int(g["wrong"].sum()), {str(b): (int(r["size"]), round(float(r["mean"]), 3) if r["size"] else None) for b, r in tab.iterrows()},
          "<=1%:", round(float(g.loc[g["gap"] <= 0.01, "wrong"].mean()), 3))
(OUT / "facts_robust.json").write_text(json.dumps(OUTF, indent=1, default=float))


# =====================================================================================================================
# Hidden-benefit sensitivity. For an ordered pair (i, j) on the same rows, the losing policy's
# largest possible excess over J_W is E_j = p_j v_range - (J_W(j) - J_term(j)) >= 0 (Theorem 1(iv), attained). The
# kappa-certificate J_term(i) > J_W(j) + kappa E_j is the NHB certificate at kappa = 0 and the NHB-free one at kappa = 1.
# kappa* = (J_term(i) - J_W(j)) / E_j is the breakdown fraction of an NHB certificate (> 1 iff NHB-free).
# =====================================================================================================================
KGRID = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)


def kappa_draw(lo, up, p, rng, jt=None):
    """Per kappa: certified ordered claims on completion-sensitive pairs (a policy with p > 0) and their errors;
    plus kappa* of every NHB-only claim."""
    n = len(lo)
    off = ~np.eye(n, dtype=bool)
    u = p > PTOL
    sens = off & (u[:, None] | u[None, :])
    E = p * rng - (up - lo)
    assert (E >= -1e-12).all()
    out = []
    wrong = None if jt is None else ~gt(jt, jt)
    for kap in KGRID:
        C = (lo[:, None] > up[None, :] + kap * E[None, :] + TOL) & off
        r = {"kappa": kap, "claims": int(C.sum()), "claims_sensitive": int((C & sens).sum()),
             "pairs_sensitive": int(sens.sum() // 2)}
        if wrong is not None:
            r["errors"], r["errors_sensitive"] = int((C & wrong).sum()), int((C & sens & wrong).sum())
        out.append(r)
    nhb = gt(lo, up) & off
    free = gt(lo, lo + p * rng) & off
    only = nhb & ~free
    i, j = np.nonzero(only)
    ks = (lo[i] - up[j]) / np.where(E[j] > 0, E[j], np.nan)
    marg = lo[i] - up[j]
    wr = None if jt is None else ~(jt[i] > jt[j] + TOL)
    return out, ks, marg, wr


KROWS, KSTAR = [], []
for (c, dr), d in D.groupby(level=["cell", "draw"]):
    fam = d["family"].iloc[0]
    rng = VR[fam][1] - VR[fam][0]
    jt = d["true"].to_numpy(float) if fam != "icu" else None
    for tf, (lc, uc, pc) in TARGETS.items():
        if d[lc].isna().any():
            continue
        lo, up, p = d[lc].to_numpy(float), d[uc].to_numpy(float), d[pc].to_numpy(float)
        rows_, ks, marg, wr = kappa_draw(lo, up, p, rng, jt)
        for r in rows_:
            KROWS.append({"cell": c, "draw": int(dr), "family": fam, "targets": tf, **r})
        for t_, (k_, m_) in enumerate(zip(ks, marg)):
            KSTAR.append({"cell": c, "draw": int(dr), "targets": tf, "kappa_star": float(k_), "margin": float(m_),
                          "wrong": (None if wr is None else bool(wr[t_]))})
KR, KS = pd.DataFrame(KROWS), pd.DataFrame(KSTAR)

# Part I ICU-Sepsis main table and all 73 policies (released rows, seed means)
def icu_kappa(t):
    lo, up, pp = t["lo"].to_numpy(), t["up"].to_numpy(), t["p"].to_numpy()
    rows_, ks, marg, _ = kappa_draw(lo, up, pp, 1.0)
    return rows_, ks, marg


OUTF["kappa"] = {"definition": "kappa-certificate J_term(i) > J_W(j) + kappa E_j, E_j = p_j v_range - (J_W(j) - J_term(j)); "
                 "kappa=0 NHB, kappa=1 NHB-free; kappa* = (J_term(i) - J_W(j)) / E_j of NHB-only claims; sensitive pairs: "
                 "at least one policy with p > 0", "grid": KGRID, "pools": {}}
for tf in TARGETS:
    for pool, cells in POOLS.items():
        d = KR[(KR["targets"] == tf) & KR["cell"].isin(cells)]
        if not len(d):
            continue
        gk = d.groupby("kappa").sum(numeric_only=True)
        k = KS[(KS["targets"] == tf) & KS["cell"].isin(cells)]
        o = {"by_kappa": {}}
        for kap, r in gk.iterrows():
            e = {"claims_sensitive": int(r["claims_sensitive"]), "pairs_sensitive": int(r["pairs_sensitive"]),
                 "share_sensitive": float(r["claims_sensitive"] / r["pairs_sensitive"]) if r["pairs_sensitive"] else None}
            if "errors" in r and not np.isnan(r["errors"]) and (d["family"] != "icu").all():
                e["errors_sensitive"] = int(r["errors_sensitive"])
                e["error_rate_sensitive"] = float(r["errors_sensitive"] / r["claims_sensitive"]) if r["claims_sensitive"] else None
            o["by_kappa"][str(kap)] = e
        if len(k):
            o["kappa_star_quantiles"] = {str(q): float(k["kappa_star"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
            o["margin_quantiles"] = {str(q): float(k["margin"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
            o["nhb_only_claims"] = int(len(k))
            if k["wrong"].notna().all():
                kk = k.copy()
                kk["bin"] = pd.cut(kk["kappa_star"], [0, 0.05, 0.1, 0.25, 0.5, 1.0])
                o["error_by_kappa_star_bin"] = {str(b): {"claims": int(len(x)), "error_rate": float(x["wrong"].mean()) if len(x) else None}
                                                for b, x in kk.groupby("bin", observed=False)}
        OUTF["kappa"]["pools"].setdefault(tf, {})[pool] = o
for nm, t in (("icu_part1_main_table", mt), ("icu_part1_all", ICU_ALL)):
    rows_, ks, marg = icu_kappa(t)
    OUTF["kappa"][nm] = {"by_kappa": {str(r["kappa"]): {"claims_sensitive": r["claims_sensitive"],
                                                         "pairs_sensitive": r["pairs_sensitive"]} for r in rows_},
                         "kappa_star": sorted(float(x) for x in ks), "margin": sorted(float(x) for x in marg)}
(OUT / "facts_robust.json").write_text(json.dumps(OUTF, indent=1, default=float))
for tf in TARGETS:
    for pool in ("gumbel_estimated", "garnet_all", "garnet_h-0.5", "icu"):
        o = OUTF["kappa"]["pools"].get(tf, {}).get(pool)
        if o:
            print("KAPPA", tf, pool, {k: (v["claims_sensitive"], v["pairs_sensitive"], round(v["share_sensitive"], 3) if v["share_sensitive"] is not None else None,
                                          round(v.get("error_rate_sensitive") or 0, 4)) for k, v in o["by_kappa"].items()},
                  "k* q:", {k: round(v, 3) for k, v in o.get("kappa_star_quantiles", {}).items()},
                  "bins:", {k: (v["claims"], round(v["error_rate"], 3) if v["error_rate"] is not None else None) for k, v in o.get("error_by_kappa_star_bin", {}).items()})
for nm in ("icu_part1_main_table", "icu_part1_all"):
    o = OUTF["kappa"][nm]
    print(nm, {k: (v["claims_sensitive"], v["pairs_sensitive"]) for k, v in o["by_kappa"].items()}, "k*:", [round(x, 3) for x in o["kappa_star"]][:40])

# ---- tab_kappa (appendix app:robust): share of completion-sensitive pairs certified (error) along kappa
KCOLS = (0.0, 0.1, 0.25, 0.5, 1.0)
KROWS_T = [("Gumbel, true rows", "true_rows", "gumbel_estimated"), ("Garnet, true rows", "true_rows", "garnet_all"),
           (r"\quad $h{=}{-}0.5$ (NHB fails)", "true_rows", "garnet_h-0.5"), ("Gumbel, estimated rows", "bench", "gumbel_estimated"),
           ("Garnet, estimated rows", "bench", "garnet_all"), (r"\icu{}, 50 draws", "bench", "icu")]
lines = [r"\setlength{\tabcolsep}{4pt}%", r"\begin{tabular}{l" + "r" * (len(KCOLS) + 1) + "}", r"\toprule",
         r" & \multicolumn{%d}{c}{$\kappa$} & median \\" % len(KCOLS), r"\cmidrule(lr){2-%d}" % (len(KCOLS) + 1),
         "Setting & " + " & ".join(("0 (NHB)" if k == 0 else "1 (free)" if k == 1 else f"{k:g}") for k in KCOLS)
         + r" & $\kappa^*$ \\", r"\midrule"]
for lab_, tf, pool in KROWS_T:
    o = OUTF["kappa"]["pools"][tf][pool]
    cells_ = []
    for k in KCOLS:
        e = o["by_kappa"][str(k)]
        s = p1(e["share_sensitive"])
        if "error_rate_sensitive" in e and e["error_rate_sensitive"] is not None:
            s += f" ({p1(e['error_rate_sensitive'])})"
        cells_.append(s)
    med = o.get("kappa_star_quantiles", {}).get("0.5")
    lines.append(f"{lab_} & " + " & ".join(cells_) + f" & {med:.2f} \\\\")
o = OUTF["kappa"]["icu_part1_main_table"]
ks_ = np.array(o["kappa_star"])
lines.append(r"\icu{}, audited main table & " + " & ".join(
    f"{o['by_kappa'][str(k)]['claims_sensitive']}/{o['by_kappa'][str(k)]['pairs_sensitive']}" for k in KCOLS)
    + f" & {np.median(ks_):.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_kappa.tex").write_text("\n".join(lines) + "\n")
print("wrote tables/tab_kappa.tex")
