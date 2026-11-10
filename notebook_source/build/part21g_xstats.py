# %% [markdown]
# ## II.7 Cross-environment statistics: normalised scores, rliable aggregates, benchmark fidelity, preregistered tests
#
# **Inputs.** `XRESULTS` (Section II.6: one row per cell × draw × policy × target, plus three reference rows per
# target), `XDRAW_META`, and `X_FRONTIERS` / `PROP_CHECKS` (Section II.5). Every table and JSON below is written even
# when Part II switches are off (header-only tables, "not run" entries), so the release map and the final gate never
# depend on which sessions ran.
#
# **Normalised score.** z = (J − J_β) / (J* − J_β) per (cell, draw, target), where J* is the target's unconstrained
# optimum and J_β the behaviour policy evaluated on the same target (the "REF" rows). z = 1 is optimal, z = 0 matches
# the behaviour policy, z < 0 is worse than it. z is undefined (NaN, counted) when J* − J_β < 1e-9. Improper
# (non-absorbing) evaluations have no return; they are excluded from every aggregate and counted.
#
# **Aggregates (rliable; Agarwal et al., NeurIPS 2021).** Tasks are environment cells; runs are dataset draws.
# * The interquartile mean (IQM) is the mean of the empirical quantile function over [0.25, 0.75]. Sorted scores
#   inside the interquartile range get weight 1 and the two boundary scores get fractional weight. The IQM therefore
#   equals rliable's `trim_mean(·, 0.25)` exactly when n is a multiple of 4, and interpolates otherwise.
# * Intervals are percentile intervals of a **stratified bootstrap**: draws are resampled with replacement within
#   each cell (`XCFG.bootstrap` replicates, streams `np_rng("xenv", "boot", …)`). Statistics that share a stream key
#   share resamples (common random numbers across policies), and paired statistics resample draws jointly.
#   Garnet cells of one draw share the base MDP, K(s) and k_s (Section II.3c), so Garnet draws are resampled
#   **jointly across cells** (one index vector per replicate, on the draws that are finite in every cell).
# * The probability of improvement P(X > Y) is rliable's: within each cell, the Mann–Whitney probability over all
#   draw pairs (ties count ½), averaged over cells.
# * Performance profiles give the fraction of (cell, draw) units with z ≥ x, with pointwise stratified-bootstrap
#   bands.
# * Exact values within 1e-9 are treated as ties in every rank, argmax and comparison. Proposition P1 makes some
#   policies analytically equal (for example a native policy and its projection under the mean rule), and
#   floating-point noise must not decide their order.
#
# **Benchmark fidelity (Gumbel, Garnet).** Each completion rule is compared with the truth, per draw, over the default
# policy set (mean → `bench_mean`, terminate → `bench_terminate`):
# * Kendall τ-b between benchmark values and true values;
# * top-1 agreement and selection regret (in z units of the true target) of the benchmark's choice, both as
#   expectations under uniform tie-breaking among the benchmark's top choices.
#
# The same statistics computed between the two benchmark rules ("rule sensitivity", every family including
# ICU-Sepsis) measure how much the rule alone reorders methods.
#
# **Error decomposition.** J_true − J_bench = (J_true − J_trueRows+completion) + (J_trueRows+completion − J_bench):
# * the semantics term: the completion rule is wrong about unsupported actions;
# * the estimation-and-coverage term: finite hospital data and unvisited states.
#
# In the oracle construction the benchmark *is* the true-rows completion, so all of its error is semantics.
#
# **Preregistered tests (`PREREG_PART2`).**
# * **H1:** per Gumbel B-est cell, a two-sided sign-flip test on per-draw τ(mean) − τ(terminate); Holm across cells.
#   A benchmark that gives every policy the same value carries no ranking information and scores τ = 0 (Kendall
#   τ-a); such draws stay in the test, are counted, and a sensitivity p-value without them is reported.
# * **H2:** per family, a two-sided sign-flip test on paired differences of deployment z (treatment −
#   `XCOMPARATOR`). ICU-Sepsis and Gumbel cells use independent streams on a fixed system, so their (cell, draw)
#   units are pooled across cells. The Garnet cells of one draw share a base MDP, so the Garnet test unit is the
#   draw block (the mean difference over the harm × ε cells of that draw). Holm over treatments × families; TOST at
#   `XCFG.equivalence_margin_normalized` (raw and Holm-adjusted) on the same units; IQM and mean differences with
#   stratified-bootstrap intervals.
# * **H3:** Spearman ρ between |h| and the per-draw benchmark error of mean imputation (Garnet). The two-sided
#   permutation p-value uses 10 000 permutations of |h| within (ε, draw) blocks: the harm levels of one draw share
#   its base MDP, and the null leaves the error free to depend on ε and on the draw. Permutations within ε only and
#   unrestricted permutations are reported as sensitivity checks, next to descriptive mask sizes and violation
#   counts by (h, ε), because the harm also changes how many actions the benchmark admits.
# * Sign-flip p-values are exact for n ≤ 16 and Monte Carlo otherwise (200 000 sign patterns, +1 correction); the
#   reported attainable minimum follows the method used.
#
# **Hyper-parameter selection.** For each method and variant, z is reported for three configurations: the declared
# default; the one that maximises tabular FQE on the held-out dev episodes; and the one that maximises the deployment
# return. The last is an oracle choice, an **upper bound** that no offline procedure attains. A unit in which the FQE
# model rates no candidate proper falls back to the declared default (counted in `fqe_fallback_units`), so the three
# columns describe the same units.
#
# **Invariants asserted on every row.** An assertion failure is a pipeline bug, not a statistical finding.
# * Complete design: every cell and draw of `X_CELLS` is present, with identical policy and target sets across draws.
# * Unique keys; status in {ok, improper}; finite metrics on every ok row; three reference rows per target
#   (optional on `fqe_dev`, which is used only for hyper-parameter selection).
# * Value ordering: J ≤ J* for every policy, J*_admissible ≤ J*, and J ≤ J*_admissible for policies with zero
#   expected violations.
# * Targets that share the admissible rows (benchmark targets; true-system targets) give every policy the same
#   P(any violation), and give admissible policies identical values (Proposition P5).

# %%
# ============================================================
# II.7 Cross-environment statistics
# ============================================================
EXECLOG.mark("II.7 Cross-environment statistics")
import re
from matplotlib.colors import LinearSegmentedColormap

XS_FAMILIES = ("icu", "gumbel", "garnet")
XS_DEPLOY = {"icu": "bench_terminate", "gumbel": "true", "garnet": "true"}   # deployment target per family
XS_REF = ("REF optimum", "REF optimum (admissible)", "REF behaviour")
XS_REF_COLS = ("ref_optimum", "ref_optimum_admissible", "ref_behaviour")
XS_FQE = "fqe_dev"
XS_DRAW_BLOCKED_FAMILIES = ("garnet",)   # cells of one draw share the base MDP (part21c ('base', draw))
XS_SIGNFLIP_MC_DRAWS = 200_000           # sign_flip_test's Monte-Carlo size for n > 16 (Section 30 default)
XS_COMPLETIONS = {"mean": ("bench_mean", "true_rows_mean"), "terminate": ("bench_terminate", "true_rows_terminate")}
XS_RANK_COMPARISONS = (("mean", "bench_mean", "true"), ("terminate", "bench_terminate", "true"),
                       ("rule sensitivity", "bench_mean", "bench_terminate"))
# Targets that share state space, initial distribution and admissible rows (so P5 applies within each group).
XS_TARGET_GROUP = {"bench_mean": "benchmark", "bench_terminate": "benchmark", "true": "true system",
                   "true_rows_mean": "true system", "true_rows_terminate": "true system"}
XS_RESULT_COLUMNS = ["cell", "family", "draw", "policy", "method", "variant", "hp", "is_default", "method_family",
                     "target", "semantics", "status", "return", "avg_episode_length", "inadmissible_rate",
                     "expected_violations", "prob_any_violation", "max_state_inadmissible_mass", "absorb_death",
                     "absorb_goal"]
XS_METRICS = ["return", "avg_episode_length", "inadmissible_rate", "expected_violations", "prob_any_violation",
              "max_state_inadmissible_mass", "absorb_death", "absorb_goal"]
XS_NORM_COLUMNS = ["cell", "family", "draw", "target", "semantics", "policy", "method", "variant", "hp", "is_default",
                   "method_family", "status", "return", "ref_optimum", "ref_optimum_admissible", "ref_behaviour",
                   "denominator", "z", "expected_violations", "prob_any_violation", "max_state_inadmissible_mass"]
XS_DENOM_MIN = 1e-9        # normalisation undefined below this optimum-behaviour gap
# Exact values closer than XS_TIE_ATOL are ties: P1 makes e.g. a native policy and its projection equal under the
# mean rule, and floating-point noise (~1e-15) must not decide their order in ranks, argmaxes or P(X > Y).
XS_TIE_ATOL = 1e-9
XS_VALUE_TOL = 1e-6        # tolerance of value-ordering invariants (exact solves are accurate to ~1e-8)
XS_PROB_TOL = 1e-8
XS_ZERO_MASS = 1e-12
XS_PROFILE_GRID = np.round(np.linspace(-1.0, 1.0, 81), 6)
XS_PROFILE_TOL = 1e-9      # z >= x is evaluated as z >= x - 1e-9, so exactly optimal policies count at x = 1
XS_H3_PERMUTATIONS = 10_000
XS_BEHAVIOUR_POLICY = "Behaviour (true) [native]"
# Categorical slots in fixed order (CVD-validated reference palette); ink and muted greys for text and context.
XS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
XS_INK, XS_MUTED, XS_NEUTRAL = "#0b0b0b", "#898781", "#f0efec"


# ---------- small utilities ----------
def xs_as_bool(s: pd.Series) -> pd.Series:
    """Boolean column after a CSV round trip ('True' / 'False' strings); missing -> False, anything else raises."""
    if s.dtype == bool:
        return s

    def conv(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return False
        if isinstance(v, (int, np.integer)) and int(v) in (0, 1):
            return bool(v)
        if isinstance(v, str) and v.strip().lower() in ("true", "false", "1", "0"):
            return v.strip().lower() in ("true", "1")
        raise ValueError(f"not a boolean flag: {v!r}")
    return s.map(conv).astype(bool)


def xs_as_frame(obj: Any) -> pd.DataFrame:
    """Section II.5 outputs as a DataFrame (None = not run)."""
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, (list, tuple)):
        return pd.DataFrame(list(obj))
    raise TypeError(f"expected a DataFrame or a list of records, got {type(obj)}")


def _xs_num(tok: str):
    return int(tok) if re.fullmatch(r"[+-]?\d+", tok) else float(tok)


_XS_CELL_RE = {
    "icu": re.compile(r"icu\|(?P<construction>.+)"),
    "gumbel": re.compile(r"gumbel\|(?P<construction>[A-Za-z_-]+)\|n(?P<n_hosp>\d+)\|tau(?P<tau>[0-9.eE+-]+)"
                         r"\|eps(?P<eps>[0-9.eE+-]+)(?:\|phys-(?P<physician>[a-z-]+))?"),
    "garnet": re.compile(r"garnet\|h(?P<harm>[0-9.eE+-]+)\|eps(?P<eps>[0-9.eE+-]+)"),
}


def xs_parse_cell(cell: str) -> Dict[str, Any]:
    """Factors encoded in an X_CELLS key (formats of Section II.3); unknown formats raise."""
    cell = str(cell)
    fam = cell.split("|", 1)[0]
    m = _XS_CELL_RE[fam].fullmatch(cell) if fam in _XS_CELL_RE else None
    if m is None:
        raise ValueError(f"unrecognised Part II cell key {cell!r}")
    # Garnet benchmarks are built from hospital data like the Gumbel B-est construction.
    out = {"family": fam, "construction": "best", "n_hosp": np.nan, "tau": np.nan, "eps": np.nan, "harm": np.nan,
           "physician": "optimal" if fam == "gumbel" else ""}
    for k, v in m.groupdict().items():
        if v is not None:      # the optional Gumbel physician suffix is absent for the default (optimal) physician
            out[k] = v if k in ("construction", "physician") else _xs_num(v)
    return out


def xs_cell_label(cell: str) -> str:
    return str(cell).split("|", 1)[-1]


def xs_sort_cells(df: pd.DataFrame, order: Sequence[str], extra: Sequence[str] = ()) -> pd.DataFrame:
    """Rows in the design order of X_CELLS (stable, then by `extra` columns)."""
    if df.empty or "cell" not in df.columns:
        return df
    pos = df["cell"].map({c: i for i, c in enumerate(order)})
    assert pos.notna().all(), f"cells outside the design: {sorted(set(df.loc[pos.isna(), 'cell']))}"
    return (df.assign(_xs_order=pos).sort_values(["_xs_order", *extra], kind="mergesort")
            .drop(columns="_xs_order").reset_index(drop=True))


def xs_policy_role(policy: str) -> str:
    return "comparator" if policy == XCOMPARATOR else ("treatment" if policy in XTREATMENTS else "other")


def xs_fmt_ci(v: float, lo: float, hi: float, digits: int = 3) -> str:
    if not np.isfinite(v):
        return ""
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ---------- rliable-style estimators ----------
def xs_iqm_weights(n: int) -> np.ndarray:
    """Weights of the sorted sample in the IQM: the mean of the empirical quantile function over [0.25, 0.75]."""
    i = np.arange(n, dtype=np.float64)
    w = np.clip(np.minimum(i + 1.0, 0.75 * n) - np.maximum(i, 0.25 * n), 0.0, None)
    return w / (0.5 * n)


def xs_iqm_rows(x: np.ndarray) -> np.ndarray:
    """IQM of every row of a [B, n] array of finite values (NaN rows for n = 0)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.shape[1] == 0:
        return np.full(x.shape[0], np.nan)
    return np.sort(x, axis=1) @ xs_iqm_weights(x.shape[1])


def xs_iqm(x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if not len(x):
        return np.nan
    if not np.isfinite(x).all():
        raise ValueError("IQM input must be finite (missing scores are dropped and counted by the caller)")
    return float(xs_iqm_rows(x)[0])


def xs_snap_ties(v: np.ndarray, atol: float = XS_TIE_ATOL) -> np.ndarray:
    """Values within `atol` of their sorted neighbour form one tie class; each value is replaced by its class minimum."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 2:
        return v.copy()
    order = np.argsort(v, kind="mergesort")
    s = v[order]
    new = np.r_[True, np.diff(s) > atol]
    out = np.empty_like(v)
    out[order] = s[new][np.cumsum(new) - 1]
    return out


def xs_top(v: np.ndarray) -> np.ndarray:
    """Mask of the values tied (within XS_TIE_ATOL) with the maximum."""
    s = xs_snap_ties(v)
    return s >= s.max()


def xs_poi_rows(x: np.ndarray, y: np.ndarray, atol: float = XS_TIE_ATOL) -> np.ndarray:
    """Row-wise P(X > Y) + 0.5 P(X = Y) over all (i, j) pairs of [B, n] and [B, m] arrays (Mann-Whitney; |x - y| <= atol
    is a tie)."""
    d = x[:, :, None] - y[:, None, :]
    return (d > atol).mean(axis=(1, 2)) + 0.5 * (np.abs(d) <= atol).mean(axis=(1, 2))


def xs_stratified_bootstrap(groups: Dict[str, np.ndarray], stat: Callable[[List[np.ndarray]], np.ndarray],
                            key: Tuple, n_boot: int, chunk: int = 500,
                            shared_draws: bool = False) -> Tuple[Any, np.ndarray]:
    """Stratified bootstrap (rliable): draws are resampled with replacement within each cell (stratum).

    `groups` maps cell -> [n_c, k] array of per-draw values (k paired columns, rows aligned by draw). `stat` maps a
    list of [B, n_c, k] arrays (one per cell) to a [B] or [B, G] array. Resample indices come from
    np_rng("xenv", "boot", *key, cell, n_c), so statistics with the same key and cell sizes share resamples.
    shared_draws=True (cells of one draw share a base MDP; groups from xs_cell_groups(..., aligned=True), rows sorted
    by draw) resamples the draws jointly: one index matrix np_rng("xenv", "boot", *key, "shared-draws", n) for every
    cell. Returns (point estimate, replicates [n_boot] or [n_boot, G])."""
    cells = [c for c, g in groups.items() if len(g)]
    if not cells:
        return np.nan, np.zeros(0)
    arrs = [np.asarray(groups[c], dtype=np.float64).reshape(len(groups[c]), -1) for c in cells]
    assert all(np.isfinite(a).all() for a in arrs), "bootstrap inputs must be finite"
    assert len({a.shape[1] for a in arrs}) == 1, "every stratum needs the same paired columns"
    point = np.asarray(stat([a[None] for a in arrs]))[0]
    if shared_draws:
        n = int(len(arrs[0]))
        assert all(len(a) == n for a in arrs), "shared-draw resampling needs draw-aligned groups of equal length"
        shared = np_rng("xenv", "boot", *key, "shared-draws", n).integers(0, n, size=(int(n_boot), n))
        idx = [shared] * len(arrs)
    else:
        idx = [np_rng("xenv", "boot", *key, c, int(len(a))).integers(0, len(a), size=(int(n_boot), len(a)))
               for c, a in zip(cells, arrs)]
    reps = [np.asarray(stat([a[i[s0:s0 + chunk]] for a, i in zip(arrs, idx)])) for s0 in range(0, int(n_boot), chunk)]
    reps = np.concatenate(reps, axis=0) if reps else np.zeros(0)
    return (float(point) if np.ndim(point) == 0 else point), reps


def xs_ci(reps: np.ndarray) -> Tuple[float, float]:
    """Percentile interval at level 1 - ALPHA of scalar bootstrap replicates."""
    reps = np.asarray(reps, dtype=np.float64)
    if reps.size == 0:
        return (np.nan, np.nan)
    lo, hi = np.quantile(reps, [ALPHA / 2, 1 - ALPHA / 2])
    return float(lo), float(hi)


def xs_stat_iqm(col: int = 0) -> Callable:
    return lambda parts: xs_iqm_rows(np.concatenate([p[..., col] for p in parts], axis=1))


def xs_stat_mean(col: int = 0) -> Callable:
    return lambda parts: np.concatenate([p[..., col] for p in parts], axis=1).mean(axis=1)


def xs_stat_iqm_diff(parts: List[np.ndarray]) -> np.ndarray:
    return xs_stat_iqm(0)(parts) - xs_stat_iqm(1)(parts)


def xs_stat_mean_diff(parts: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([p[..., 0] - p[..., 1] for p in parts], axis=1).mean(axis=1)


def xs_stat_poi(parts: List[np.ndarray]) -> np.ndarray:
    """rliable probability of improvement: within-cell Mann-Whitney P(col 0 > col 1), averaged over cells."""
    return np.mean([xs_poi_rows(p[..., 0], p[..., 1]) for p in parts], axis=0)


def xs_stat_profile(grid: np.ndarray) -> Callable:
    g = np.asarray(grid, dtype=np.float64)[None, None, :] - XS_PROFILE_TOL
    return lambda parts: (np.concatenate([p[..., 0] for p in parts], axis=1)[:, :, None] >= g).mean(axis=1)


def xs_cell_groups(W: pd.DataFrame, cols: Sequence[str], aligned: bool = False) -> Dict[str, np.ndarray]:
    """cell -> [n_c, len(cols)] rows of a (cell, draw)-indexed frame on which every column is finite.

    aligned=True (cells of one draw share a base MDP) keeps only the draws whose rows are finite in every cell
    present in W and returns each cell's rows sorted by draw, so every group has the same length and row i of every
    group is the same draw (for xs_stratified_bootstrap(..., shared_draws=True))."""
    out = {}
    if W.empty:
        return out
    V = W[list(cols)]
    if aligned:
        fin = pd.Series(np.isfinite(V.to_numpy(dtype=np.float64)).all(axis=1), index=V.index)
        ok = fin.unstack("cell", fill_value=False).astype(bool)          # draw x cell
        keep = np.sort(ok.index[ok.all(axis=1)].to_numpy())
        if not len(keep):
            return out
        for cell in dict.fromkeys(V.index.get_level_values("cell")):
            out[str(cell)] = V.xs(cell, level="cell").loc[keep].to_numpy(dtype=np.float64)
        return out
    for cell, g in V.groupby(level="cell", sort=False):
        v = g.to_numpy(dtype=np.float64)
        v = v[np.isfinite(v).all(axis=1)]
        if len(v):
            out[str(cell)] = v
    return out


def xs_min_attainable_p(n: int) -> float:
    """Smallest two-sided sign-flip p-value: 2 / 2^n by exact enumeration (n <= 16), and at least the Monte-Carlo
    floor 1 / (draws + 1) otherwise (sign_flip_test uses 200 000 sign patterns with the +1 correction)."""
    exact = min_attainable_p(n, "two-sided")
    return exact if n <= 16 else max(exact, 1.0 / (XS_SIGNFLIP_MC_DRAWS + 1))


def xs_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if len(rx) < 3 or np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def xs_permutation_spearman(x: np.ndarray, y: np.ndarray, strata: Optional[np.ndarray], n_perm: int,
                            key: Tuple) -> Dict[str, Any]:
    """Two-sided permutation p-value of Spearman's rho (y permuted within strata if given; +1 correction)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n = len(x)
    rho = xs_spearman(x, y)
    if not np.isfinite(rho):
        return {"rho": rho, "p_value": np.nan, "n": int(n), "n_permutations": 0}
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    norm_ = math.sqrt(float(rx @ rx) * float(ry @ ry))
    strata = np.zeros(n) if strata is None else np.asarray(strata)
    rng = np_rng("xenv", "H3", *key)
    idx = np.tile(np.arange(n), (int(n_perm), 1))
    for s in np.unique(strata):
        gi = np.flatnonzero(strata == s)
        idx[:, gi] = gi[np.argsort(rng.random((int(n_perm), len(gi))), axis=1)]
    null = (ry[idx] @ rx) / norm_
    p = (1 + int(np.sum(np.abs(null) >= abs(rho) - 1e-12))) / (1 + int(n_perm))
    return {"rho": rho, "p_value": float(p), "n": int(n), "n_permutations": int(n_perm),
            "n_strata": int(len(np.unique(strata)))}


def xs_selftests() -> Dict[str, Any]:
    """Millisecond checks of the estimators against independent definitions."""
    rng = np_rng("xenv", "stats-selftest")
    out = {}
    out["iqm_equals_trim_mean_when_n_mod_4_is_0"] = max(
        abs(xs_iqm(x) - float(stats.trim_mean(x, 0.25))) for x in (rng.normal(size=n) for n in (4, 8, 12, 40)))
    s = np.sort(rng.normal(size=9))
    out["iqm_interpolated_n9"] = abs(xs_iqm(s[::-1]) - (0.75 * s[2] + s[3:6].sum() + 0.75 * s[6]) / 4.5)
    out["iqm_n1_n2"] = abs(xs_iqm([3.0]) - 3.0) + abs(xs_iqm([1.0, 2.0]) - 1.5)
    x, y = rng.integers(0, 4, size=7).astype(float), rng.integers(0, 4, size=5).astype(float)
    out["poi_matches_probability_of_improvement"] = abs(float(xs_poi_rows(x[None], y[None])[0])
                                                        - probability_of_improvement(x, y))
    _, reps = xs_stratified_bootstrap({"a": np.zeros(5), "b": np.ones(3)}, xs_stat_mean(), ("selftest",), 200)
    out["stratified_bootstrap_keeps_strata"] = float(np.abs(reps - 3 / 8).max())
    xa = rng.normal(size=7)
    _, reps = xs_stratified_bootstrap({"a": xa, "b": xa + 1.0}, lambda p: p[1][..., 0].mean(1) - p[0][..., 0].mean(1),
                                      ("selftest",), 200, shared_draws=True)
    out["shared_draw_bootstrap_resamples_draws_jointly"] = float(np.abs(reps - 1.0).max())
    u, v = rng.normal(size=30), rng.normal(size=30)
    out["spearman_matches_scipy"] = abs(xs_spearman(u, v) - float(stats.spearmanr(u, v)[0]))
    t = xs_snap_ties(np.array([0.5 + 1e-15, 0.2, 0.5, 0.7]))
    out["tie_snapping"] = float(abs(t[0] - t[2]) + abs(t[1] - 0.2) + abs(t[3] - 0.7))
    bad = {k: v for k, v in out.items() if not v < 1e-12}
    assert not bad, f"II.7 estimator self-tests failed: {bad}"
    return out


# ---------- input preparation and structural invariants ----------
def xs_prepare_results(results: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Type-normalised copy of XRESULTS (schema, keys, status and reference labels asserted)."""
    if results is None or not len(results):
        return pd.DataFrame(columns=XS_RESULT_COLUMNS + ["is_ref"])
    missing = [c for c in XS_RESULT_COLUMNS if c not in results.columns]
    assert not missing, f"XRESULTS lacks the columns {missing}"
    xr = results[XS_RESULT_COLUMNS].copy()
    for c in ("cell", "family", "policy", "method", "variant", "hp", "method_family", "target", "semantics", "status"):
        xr[c] = xr[c].astype(str)
    xr["draw"] = pd.to_numeric(xr["draw"], errors="raise").astype(np.int64)
    for c in XS_METRICS:
        xr[c] = pd.to_numeric(xr[c], errors="raise").astype(np.float64)
    # Reference rows (Section II.6): policy in XS_REF, method = the policy label, variant "reference",
    # is_reference True. The label set defines them; the driver's flag must agree when present.
    xr["is_ref"] = xr["policy"].isin(XS_REF)
    if "is_reference" in results.columns:
        flag = xs_as_bool(results["is_reference"]).to_numpy()
        assert np.array_equal(flag, xr["is_ref"].to_numpy()), "is_reference disagrees with the REF policy labels"
    assert xr.loc[xr["is_ref"], "method"].isin(["REF", *XS_REF]).all(), "reference rows with a learner method label"
    # is_default is meaningful (and strictly parsed) for learned policies only; reference rows are never "default".
    flags = np.zeros(len(xr), dtype=bool)
    flags[~xr["is_ref"].to_numpy()] = xs_as_bool(xr.loc[~xr["is_ref"], "is_default"]).to_numpy()
    xr["is_default"] = flags
    bad = sorted(set(xr["status"]) - {"ok", "improper"})
    assert not bad, f"unexpected Part II evaluation status values {bad}"
    ok = xr["status"].eq("ok")
    fin = np.isfinite(xr.loc[ok, XS_METRICS].to_numpy(dtype=np.float64)).all(axis=0)
    assert fin.all(), f"'ok' Part II rows with non-finite {[c for c, f in zip(XS_METRICS, fin) if not f]}"
    xr.loc[~ok, XS_METRICS] = np.nan          # improper evaluations carry no metrics
    assert xr["cell"].str.split("|", n=1).str[0].eq(xr["family"]).all(), "family column disagrees with the cell key"
    assert set(xr["family"]) <= set(XS_FAMILIES), sorted(set(xr["family"]))
    dup = xr.duplicated(["cell", "draw", "target", "policy"])
    assert not dup.any(), f"{int(dup.sum())} duplicated (cell, draw, target, policy) rows in XRESULTS"
    assert (xr.groupby(["cell", "target"])["semantics"].nunique() == 1).all(), "a target has two semantics in a cell"
    assert not xr.loc[~xr["is_ref"], "method"].isin(["REF", *XS_REF]).any(), "a learned policy carries a reference label"
    assert not xr.loc[~xr["is_ref"], "variant"].eq("reference").any(), "a learned policy has variant 'reference'"
    assert (xr.loc[ok, "expected_violations"] >= -1e-12).all() and (xr.loc[ok, "prob_any_violation"] <= 1 + 1e-9).all()
    return xr


def xs_design_checks(xr: pd.DataFrame, cells_spec: Optional[Sequence[dict]], experiments: Dict[str, bool]) -> Dict[str, Any]:
    """Complete design: every X_CELLS cell and draw present, identical target / policy sets, reference rows."""
    out = {"families_enabled": [f for f in XS_FAMILIES if experiments[f"xenv_{f}"]],
           "families_present": [f for f in XS_FAMILIES if f in set(xr["family"])], "rows": int(len(xr)), "cells": {}}
    spec = {str(s["cell"]): s for s in (cells_spec or [])}
    if xr.empty:
        assert not spec, f"X_CELLS lists {len(spec)} cells but XRESULTS is empty (run Section II.6)"
        return out
    assert cells_spec is not None, "XRESULTS has rows but X_CELLS is undefined"
    unknown, absent = sorted(set(xr["cell"]) - set(spec)), sorted(set(spec) - set(xr["cell"]))
    assert not unknown, f"XRESULTS cells missing from X_CELLS: {unknown}"
    assert not absent, f"X_CELLS cells without results: {absent}"
    for fam in out["families_present"]:
        assert fam in out["families_enabled"], f"results for family '{fam}' although EXPERIMENTS['xenv_{fam}'] is False"
    for cell, g in xr.groupby("cell", sort=False):
        fam = str(g["family"].iloc[0])
        draws = set(g["draw"].tolist())
        assert draws == set(range(int(spec[cell]["draws"]))), \
            f"{cell}: draws {sorted(draws)} differ from the design 0..{int(spec[cell]['draws']) - 1}"
        tsets = g.groupby("draw")["target"].agg(lambda s: "\n".join(sorted(set(s))))
        assert tsets.nunique() == 1, f"{cell}: target sets differ across draws"
        targets = tsets.iloc[0].split("\n")
        assert XS_DEPLOY[fam] in targets, f"{cell}: deployment target '{XS_DEPLOY[fam]}' missing"
        learned = g[~g["is_ref"]]
        psets = learned.groupby(["draw", "target"])["policy"].agg(lambda s: "\n".join(sorted(s)))
        assert len(psets) == len(draws) * len(targets) and psets.nunique() == 1, \
            f"{cell}: the learned-policy set differs across draws or targets"
        nref = g[g["is_ref"]].groupby(["draw", "target"])["policy"].nunique()
        for t in targets:
            k = nref.xs(t, level="target") if t in nref.index.get_level_values("target") else pd.Series(dtype=int)
            if t == XS_FQE and k.empty:
                continue
            assert len(k) == len(draws) and (k == len(XS_REF)).all(), f"{cell}: reference rows incomplete on '{t}'"
        out["cells"][cell] = {"family": fam, "draws": len(draws), "targets": targets,
                              "learned_policies": len(psets.iloc[0].split("\n")),
                              "default_policies": int(learned.loc[learned["is_default"], "policy"].nunique())}
    return out


def xs_draw_meta_checks(meta: Optional[pd.DataFrame], xr: pd.DataFrame) -> Dict[str, Any]:
    meta = xs_as_frame(meta)
    out = {"rows": int(len(meta)), "columns": [str(c) for c in meta.columns]}
    if meta.empty:
        return out
    if {"cell", "draw"} <= set(meta.columns):
        have = set(zip(meta["cell"].astype(str), pd.to_numeric(meta["draw"]).astype(int)))
        out["result_draws_without_meta"] = int(len(set(zip(xr["cell"], xr["draw"].astype(int))) - have))
        if "deploy_target" in meta.columns:
            fam = meta["cell"].astype(str).str.split("|", n=1).str[0]
            wrong = meta.loc[meta["deploy_target"].astype(str) != fam.map(XS_DEPLOY), "cell"]
            assert wrong.empty, f"deployment targets disagree with XS_DEPLOY for {sorted(set(wrong))}"
    return out


def xs_normalized_scores(xr: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Learned-policy rows with their target's reference values and z = (J - J_b) / (J* - J_b)."""
    if xr.empty:
        return pd.DataFrame(columns=XS_NORM_COLUMNS), {"rows": 0}
    ref = xr[xr["is_ref"]]
    idx = ["cell", "draw", "target", "policy"]
    R = ref.set_index(idx)["return"].unstack("policy").reindex(columns=list(XS_REF))
    st = ref.set_index(idx)["status"].unstack("policy").reindex(columns=list(XS_REF))
    opt_bad = int((st[list(XS_REF[:2])] != "ok").to_numpy().sum())
    assert opt_bad == 0, f"{opt_bad} reference optima are not 'ok' evaluations"
    R.columns = list(XS_REF_COLS)
    pol = xr[~xr["is_ref"]].merge(R.reset_index(), on=["cell", "draw", "target"], how="left", validate="many_to_one")
    denom = pol["ref_optimum"] - pol["ref_behaviour"]
    defined = denom >= XS_DENOM_MIN
    pol["denominator"] = denom
    pol["z"] = np.where(defined, (pol["return"] - pol["ref_behaviour"]) / denom.where(defined), np.nan)
    grp = pol.drop_duplicates(["cell", "draw", "target"])
    small = grp["denominator"].notna() & (grp["denominator"] < XS_DENOM_MIN)
    report = {
        "rows": int(len(pol)), "z_finite_rows": int(np.isfinite(pol["z"]).sum()),
        "improper_rows": int(pol["status"].ne("ok").sum()),
        "improper_rows_by_family_target": {f"{f}|{t}": int(n) for (f, t), n in
                                           pol[pol["status"].ne("ok")].groupby(["family", "target"]).size().items()},
        "groups": int(len(grp)),
        "groups_denominator_below_1e-9": int(small.sum()),
        "rows_denominator_below_1e-9": int((pol["denominator"].notna() & ~defined).sum()),
        "groups_denominator_below_1e-9_by_target": {str(t): int(n) for t, n in grp[small].groupby("target").size().items()},
        "groups_without_reference_by_target": {str(t): int(n) for t, n in
                                               grp[grp["ref_optimum"].isna()].groupby("target").size().items()},
        "groups_reference_behaviour_not_ok": int((st["REF behaviour"] != "ok").sum()),
    }
    return pol[XS_NORM_COLUMNS].reset_index(drop=True), report


def xs_reference_checks(norm: pd.DataFrame) -> Dict[str, Any]:
    """Value ordering: J <= J*, J*_adm <= J*, J_b <= J*, and J <= J*_adm for policies without expected violations."""
    if norm.empty:
        return {"rows_checked": 0}
    ok = norm[norm["status"].eq("ok") & norm["ref_optimum"].notna()]
    scale = np.maximum(1.0, ok["ref_optimum"].abs())
    zero = ok["expected_violations"] <= XS_ZERO_MASS
    grp = norm[norm["ref_optimum"].notna()].drop_duplicates(["cell", "draw", "target"])
    gscale = np.maximum(1.0, grp["ref_optimum"].abs())
    out = {"rows_checked": int(len(ok)), "rows_without_expected_violations": int(zero.sum()),
           "max_excess_over_optimum": float(((ok["return"] - ok["ref_optimum"]) / scale).max()),
           "max_excess_over_admissible_optimum_zero_violation_policies":
               float(((ok["return"] - ok["ref_optimum_admissible"]) / scale)[zero].max()) if zero.any() else -np.inf,
           "max_excess_admissible_optimum_over_optimum":
               float(((grp["ref_optimum_admissible"] - grp["ref_optimum"]) / gscale).max()),
           "max_excess_behaviour_over_optimum": float(((grp["ref_behaviour"] - grp["ref_optimum"]) / gscale).max())}
    bad = {k: v for k, v in out.items() if k.startswith("max_") and np.isfinite(v) and v > XS_VALUE_TOL}
    assert not bad, f"value-ordering invariants violated (tolerance {XS_VALUE_TOL}): {bad}"
    beh = norm[norm["policy"].eq(XS_BEHAVIOUR_POLICY) & norm["status"].eq("ok") & norm["ref_behaviour"].notna()]
    if len(beh):   # reported only: the driver's behaviour record should coincide with the REF behaviour row
        out["behaviour_record_max_abs_diff_to_ref_behaviour"] = float((beh["return"] - beh["ref_behaviour"]).abs().max())
        out["behaviour_record_max_abs_z"] = float(np.nanmax(np.abs(beh["z"].to_numpy()))) if beh["z"].notna().any() else np.nan
    return out


def xs_semantic_group_checks(norm: pd.DataFrame) -> Dict[str, Any]:
    """Targets sharing the admissible rows: equal P(any violation) for every policy; equal value if admissible (P5)."""
    out = {}
    if norm.empty:
        return out
    sub = norm[norm["target"].isin(list(XS_TARGET_GROUP)) & norm["status"].eq("ok")]
    for grp_name in sorted(set(XS_TARGET_GROUP.values())):
        g = sub[sub["target"].map(XS_TARGET_GROUP) == grp_name]
        if g.empty:
            continue
        w = g.set_index(["cell", "draw", "policy", "target"])
        pa, ms, rt = (w[c].unstack("target") for c in ("prob_any_violation", "max_state_inadmissible_mass", "return"))
        multi = pa.notna().sum(axis=1) >= 2
        adm = ((ms <= XS_ZERO_MASS) | ms.isna()).all(axis=1) & (ms.notna().sum(axis=1) >= 2)
        spread_pa = (pa.max(axis=1) - pa.min(axis=1))[multi]
        spread_rt = (rt.max(axis=1) - rt.min(axis=1))[adm]
        out[grp_name] = {"policy_rows_compared": int(multi.sum()), "admissible_policy_rows": int(adm.sum()),
                         "max_prob_any_violation_spread": float(spread_pa.max()) if len(spread_pa) else 0.0,
                         "max_admissible_value_spread": float(spread_rt.max()) if len(spread_rt) else 0.0}
        assert out[grp_name]["max_prob_any_violation_spread"] <= XS_PROB_TOL, (grp_name, out[grp_name])
        assert out[grp_name]["max_admissible_value_spread"] <= XS_VALUE_TOL, (grp_name, out[grp_name])
    return out


# ---------- deployment scores, main table, profiles, probability of improvement ----------
def xs_deploy_scores(norm: pd.DataFrame, defaults_only: bool = True) -> pd.DataFrame:
    """Rows on each family's deployment target (default-hp policies unless defaults_only=False)."""
    if norm.empty:
        return norm
    d = norm[norm["target"].eq(norm["family"].map(XS_DEPLOY))]
    return d[d["is_default"]] if defaults_only else d


def xs_wide(d: pd.DataFrame, value: str) -> pd.DataFrame:
    """(cell, draw) x policy frame of one column."""
    return d.set_index(["cell", "draw", "policy"])[value].unstack("policy")


XS_MAIN_LONG_COLUMNS = ["family", "deploy_target", "policy", "method", "variant", "method_family", "role", "n_cells",
                        "n_units", "n_scores", "n_missing", "iqm", "iqm_ci_low", "iqm_ci_high", "median", "mean",
                        "mean_expected_violations", "mean_prob_any_violation", "comparator", "n_paired",
                        "poi_vs_comparator", "poi_ci_low", "poi_ci_high"]


def xs_main_long(norm: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """One row per (family, default policy): IQM [CI], median, violations, P(improvement vs XCOMPARATOR) [CI]."""
    dep = xs_deploy_scores(norm)
    if dep.empty:
        return pd.DataFrame(columns=XS_MAIN_LONG_COLUMNS)
    meta = dep.drop_duplicates("policy").set_index("policy")
    rows = []
    for fam in XS_FAMILIES:
        d = dep[dep["family"] == fam]
        if d.empty:
            continue
        Z, V, PA = (xs_wide(d, c) for c in ("z", "expected_violations", "prob_any_violation"))
        key = ("main", fam, XS_DEPLOY[fam])
        blk = fam in XS_DRAW_BLOCKED_FAMILIES
        for pol in Z.columns:
            g = xs_cell_groups(Z, [pol], aligned=blk)
            pooled = np.concatenate([v[:, 0] for v in g.values()]) if g else np.zeros(0)
            iqm, reps = xs_stratified_bootstrap(g, xs_stat_iqm(), key, n_boot, shared_draws=blk)
            lo, hi = xs_ci(reps)
            v_ok, pa_ok = V[pol].dropna(), PA[pol].dropna()
            r = {"family": fam, "deploy_target": XS_DEPLOY[fam], "policy": pol, "method": meta.at[pol, "method"],
                 "variant": meta.at[pol, "variant"], "method_family": meta.at[pol, "method_family"],
                 "role": xs_policy_role(pol), "n_cells": int(Z.index.get_level_values("cell").nunique()),
                 "n_units": int(len(Z)), "n_scores": int(len(pooled)), "n_missing": int(len(Z) - len(pooled)),
                 "iqm": iqm, "iqm_ci_low": lo, "iqm_ci_high": hi,
                 "median": float(np.median(pooled)) if len(pooled) else np.nan,
                 "mean": float(pooled.mean()) if len(pooled) else np.nan,
                 "mean_expected_violations": float(v_ok.mean()) if len(v_ok) else np.nan,
                 "mean_prob_any_violation": float(pa_ok.mean()) if len(pa_ok) else np.nan,
                 "comparator": XCOMPARATOR, "n_paired": 0, "poi_vs_comparator": np.nan,
                 "poi_ci_low": np.nan, "poi_ci_high": np.nan}
            if XCOMPARATOR in Z.columns:
                gp = xs_cell_groups(Z, [pol, XCOMPARATOR], aligned=blk)
                poi, preps = xs_stratified_bootstrap(gp, xs_stat_poi, key, n_boot, shared_draws=blk)
                r["n_paired"] = int(sum(len(v) for v in gp.values()))
                r["poi_vs_comparator"] = poi
                r["poi_ci_low"], r["poi_ci_high"] = xs_ci(preps)
            rows.append(r)
    return pd.DataFrame(rows, columns=XS_MAIN_LONG_COLUMNS)


def xs_policy_order(main_long: pd.DataFrame) -> List[str]:
    """Default policies ordered by the mean of their per-family IQMs (descending; undefined last)."""
    if main_long.empty:
        return []
    m = main_long.groupby("policy")["iqm"].mean()
    return list(m.sort_values(ascending=False, na_position="last", kind="mergesort").index)


def xs_main_wide(main_long: pd.DataFrame) -> pd.DataFrame:
    """table_xenv_main.csv: one row per default policy, one column block per family (its deployment target)."""
    base_cols = ["policy", "method", "variant", "method_family", "role"]
    if main_long.empty:
        return pd.DataFrame(columns=base_cols)
    out = main_long.drop_duplicates("policy").set_index("policy").loc[xs_policy_order(main_long),
                                                                       base_cols[1:]].reset_index()
    for fam in XS_FAMILIES:
        f = main_long[main_long["family"] == fam].set_index("policy").reindex(out["policy"])
        if f["iqm"].isna().all() and f["n_units"].isna().all():
            continue
        out[f"{fam}_target"] = XS_DEPLOY[fam]
        out[f"{fam}_iqm_ci"] = [xs_fmt_ci(a, b, c) for a, b, c in zip(f["iqm"], f["iqm_ci_low"], f["iqm_ci_high"])]
        for c in ("iqm", "iqm_ci_low", "iqm_ci_high", "median", "mean_expected_violations", "mean_prob_any_violation",
                  "poi_vs_comparator", "poi_ci_low", "poi_ci_high", "n_scores", "n_missing"):
            out[f"{fam}_{c}"] = f[c].to_numpy()
        out[f"{fam}_poi_ci"] = [xs_fmt_ci(a, b, c) for a, b, c in
                                zip(f["poi_vs_comparator"], f["poi_ci_low"], f["poi_ci_high"])]
    return out


XS_PROFILE_COLUMNS = ["family", "deploy_target", "policy", "role", "threshold", "fraction_at_least", "ci_low", "ci_high",
                      "n_scores"]


def xs_performance_profiles(norm: pd.DataFrame, n_boot: int, grid: np.ndarray = XS_PROFILE_GRID) -> pd.DataFrame:
    """Fraction of (cell, draw) units with z >= x per default policy and family, with stratified-bootstrap bands."""
    dep = xs_deploy_scores(norm)
    frames = []
    for fam in XS_FAMILIES:
        d = dep[dep["family"] == fam] if len(dep) else dep
        if d.empty:
            continue
        Z = xs_wide(d, "z")
        blk = fam in XS_DRAW_BLOCKED_FAMILIES
        for pol in Z.columns:
            g = xs_cell_groups(Z, [pol], aligned=blk)
            if not g:
                continue
            point, reps = xs_stratified_bootstrap(g, xs_stat_profile(grid), ("main", fam, XS_DEPLOY[fam]), n_boot,
                                                  shared_draws=blk)
            lo, hi = (np.quantile(reps, [ALPHA / 2, 1 - ALPHA / 2], axis=0) if len(reps)
                      else (np.full(len(grid), np.nan),) * 2)
            frames.append(pd.DataFrame({"family": fam, "deploy_target": XS_DEPLOY[fam], "policy": pol,
                                        "role": xs_policy_role(pol), "threshold": grid, "fraction_at_least": point,
                                        "ci_low": lo, "ci_high": hi, "n_scores": int(sum(len(v) for v in g.values()))}))
    return pd.concat(frames, ignore_index=True)[XS_PROFILE_COLUMNS] if frames else pd.DataFrame(columns=XS_PROFILE_COLUMNS)


def xs_poi_matrix(norm: pd.DataFrame, order: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Pooled rliable P(row > column) between default policies on the deployment targets, averaged over all cells."""
    cols = ["policy_x", "policy_y", "poi", "n_cells"]
    dep = xs_deploy_scores(norm)
    if dep.empty:
        return pd.DataFrame(columns=cols), {}
    Z = xs_wide(dep, "z")
    pols = [p for p in order if p in Z.columns] + [p for p in Z.columns if p not in set(order)]
    P = len(pols)
    acc, cnt = np.zeros((P, P)), np.zeros((P, P))
    for _, g in Z[pols].groupby(level="cell", sort=False):
        V = g.to_numpy(dtype=np.float64)
        F = np.isfinite(V)
        Vf = np.where(F, V, 0.0)
        valid = F[:, None, :, None] & F[None, :, None, :]                     # [draw a, draw b, policy i, policy j]
        diff = Vf[:, None, :, None] - Vf[None, :, None, :]
        score = (diff > XS_TIE_ATOL) + 0.5 * (np.abs(diff) <= XS_TIE_ATOL)
        num, den = (score * valid).sum(axis=(0, 1)), valid.sum(axis=(0, 1))
        has = den > 0
        acc[has] += num[has] / den[has]
        cnt[has] += 1
    M = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    both = np.isfinite(M) & np.isfinite(M.T)
    check = {"policies": P, "max_complementarity_error": float(np.abs(M + M.T - 1.0)[both].max()) if both.any() else 0.0,
             "max_diagonal_error": float(np.nanmax(np.abs(np.diag(M) - 0.5))) if np.isfinite(np.diag(M)).any() else 0.0}
    assert check["max_complementarity_error"] <= 1e-9 and check["max_diagonal_error"] <= 1e-12, check
    ii, jj = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    out = pd.DataFrame({"policy_x": np.asarray(pols, dtype=object)[ii.ravel()],
                        "policy_y": np.asarray(pols, dtype=object)[jj.ravel()],
                        "poi": M.ravel(), "n_cells": cnt.ravel().astype(int)})
    return out, check


# ---------- benchmark fidelity ----------
def xs_rank_agreement(src: np.ndarray, ref: np.ndarray, z_ref: np.ndarray) -> Dict[str, Any]:
    """Agreement of the ranking by `src` (benchmark values) with the ranking by `ref` over one policy set.

    Kendall tau-b; expected top-1 agreement and selection regret (z units of the reference target) under uniform
    tie-breaking among the policies tied at the top of `src`. Values within XS_TIE_ATOL are ties. A benchmark that
    gives every policy the same value (constant `src`, non-constant `ref`) has no ranking information and scores
    tau = 0 (Kendall tau-a; flagged `tau_benchmark_constant`); tau stays undefined only when `ref` is constant."""
    src, ref, z_ref = (np.asarray(v, dtype=np.float64) for v in (src, ref, z_ref))
    out = {"n_policies": int(len(src)), "kendall_tau_b": np.nan, "top1_agreement": np.nan, "selection_regret_z": np.nan,
           "selected_suboptimality_z": np.nan, "n_top_ties": 0, "tau_undefined_constant_values": False,
           "tau_benchmark_constant": False}
    if len(src) < 2:
        return out
    s_src, s_ref = xs_snap_ties(src), xs_snap_ties(ref)
    src_var, ref_var = np.ptp(s_src) > 0, np.ptp(s_ref) > 0
    if src_var and ref_var:
        out["kendall_tau_b"] = float(stats.kendalltau(s_src, s_ref)[0])
    elif ref_var:
        out["kendall_tau_b"] = 0.0
        out["tau_benchmark_constant"] = True
    else:
        out["tau_undefined_constant_values"] = True
    top_src, top_ref = xs_top(src), xs_top(ref)
    out["n_top_ties"] = int(top_src.sum())
    out["top1_agreement"] = float(top_ref[top_src].mean())
    if np.isfinite(z_ref).all():
        out["selection_regret_z"] = float(z_ref.max() - z_ref[top_src].mean())
        out["selected_suboptimality_z"] = float(1.0 - z_ref[top_src].mean())
    return out


XS_RANK_METRICS = ("kendall_tau_b", "top1_agreement", "selection_regret_z", "selected_suboptimality_z")
XS_RANK_DRAW_COLUMNS = (["cell", "family", "draw", "completion", "ranked_by", "evaluated_on", "n_policies",
                         "n_top_ties", "tau_undefined_constant_values", "tau_benchmark_constant"] + list(XS_RANK_METRICS)
                        + ["benchmark_choice", "reference_best"])


def xs_rank_fidelity_draws(norm: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, draw, comparison) agreement statistics over the default policy set."""
    keep = ["bench_mean", "bench_terminate", "true"]
    d = norm[norm["is_default"] & norm["target"].isin(keep)] if len(norm) else norm
    if d.empty:
        return pd.DataFrame(columns=XS_RANK_DRAW_COLUMNS)
    cell_targets = {c: set(t) for c, t in d.groupby("cell")["target"]}
    w = d.set_index(["cell", "draw", "policy", "target"])
    J, Zt = w["return"].unstack("target"), w["z"].unstack("target")
    rows = []
    for (cell, draw), g in J.groupby(level=["cell", "draw"], sort=False):
        zg = Zt.loc[g.index]
        pols = g.index.get_level_values("policy").to_numpy()
        for completion, src_t, ref_t in XS_RANK_COMPARISONS:
            if not {src_t, ref_t} <= cell_targets[cell]:
                continue
            ok = (g[src_t].notna() & g[ref_t].notna()).to_numpy()
            src, ref, zr = g[src_t].to_numpy()[ok], g[ref_t].to_numpy()[ok], zg[ref_t].to_numpy()[ok]
            r = {"cell": cell, "family": cell.split("|", 1)[0], "draw": int(draw), "completion": completion,
                 "ranked_by": src_t, "evaluated_on": ref_t, **xs_rank_agreement(src, ref, zr),
                 "benchmark_choice": "", "reference_best": ""}
            if ok.any():
                p = pols[ok]
                r["benchmark_choice"] = "; ".join(p[xs_top(src)])
                r["reference_best"] = "; ".join(p[xs_top(ref)])
            rows.append(r)
    return pd.DataFrame(rows, columns=XS_RANK_DRAW_COLUMNS)


def _xs_cell_factor_cols(cell: str) -> Dict[str, Any]:
    f = xs_parse_cell(cell)
    return {k: f[k] for k in ("construction", "n_hosp", "tau", "eps", "harm")}


def xs_rank_fidelity_table(draws: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """table_rank_fidelity.csv: per cell x comparison, mean over draws with bootstrap intervals over draws."""
    cols = (["cell", "family", "construction", "n_hosp", "tau", "eps", "harm", "completion", "ranked_by", "evaluated_on",
             "n_draws", "mean_policies"]
            + [f"{m}_{s}" for m in XS_RANK_METRICS for s in ("mean", "ci_low", "ci_high", "n")])
    if draws.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (cell, completion), g in draws.groupby(["cell", "completion"], sort=False):
        r = {"cell": cell, "family": g["family"].iloc[0], **_xs_cell_factor_cols(cell), "completion": completion,
             "ranked_by": g["ranked_by"].iloc[0], "evaluated_on": g["evaluated_on"].iloc[0], "n_draws": int(len(g)),
             "mean_policies": float(g["n_policies"].mean())}
        for m in XS_RANK_METRICS:
            v = g[m].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            point, reps = xs_stratified_bootstrap({cell: v[:, None]} if len(v) else {}, xs_stat_mean(),
                                                  ("rank", completion, m), n_boot)
            r[f"{m}_mean"] = point
            r[f"{m}_ci_low"], r[f"{m}_ci_high"] = xs_ci(reps)
            r[f"{m}_n"] = int(len(v))
        rows.append(r)
    return pd.DataFrame(rows, columns=cols)


XS_ERR_DRAW_COLUMNS = ["cell", "family", "construction", "n_hosp", "tau", "eps", "harm", "draw", "completion",
                       "n_policies", "mean_abs_total", "mean_abs_semantics", "mean_abs_estimation", "mean_total",
                       "mean_semantics", "mean_estimation", "decomposition_source"]


def xs_error_decomposition_draws(norm: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, draw, completion) means over default policies of the exact benchmark-error decomposition."""
    keep = ["true", "bench_mean", "bench_terminate", "true_rows_mean", "true_rows_terminate"]
    d = (norm[norm["is_default"] & norm["family"].isin(["gumbel", "garnet"]) & norm["target"].isin(keep)]
         if len(norm) else norm)
    if d.empty:
        return pd.DataFrame(columns=XS_ERR_DRAW_COLUMNS)
    cell_targets = {c: set(t) for c, t in d.groupby("cell")["target"]}
    J = d.set_index(["cell", "draw", "policy", "target"])["return"].unstack("target")
    rows = []
    for (cell, draw), g in J.groupby(level=["cell", "draw"], sort=False):
        info = xs_parse_cell(cell)
        for completion, (bench_t, rows_t) in XS_COMPLETIONS.items():
            if not {bench_t, "true"} <= cell_targets[cell]:
                continue
            if rows_t in cell_targets[cell]:
                rows_v, source = g[rows_t], "targets"
            elif info["construction"] == "oracle":
                # Oracle construction (Section II.3): the benchmark is the true rows + completion on the true state
                # space with the true d0, so it equals the true-rows target and the estimation term is exactly 0.
                rows_v, source = g[bench_t], "oracle construction (benchmark = true rows + completion)"
            else:
                raise AssertionError(f"{cell}: target '{rows_t}' is missing, so the error cannot be decomposed")
            ok = (g["true"].notna() & g[bench_t].notna() & rows_v.notna()).to_numpy()
            tot = (g["true"] - g[bench_t]).to_numpy()[ok]
            sem = (g["true"] - rows_v).to_numpy()[ok]
            est = (rows_v - g[bench_t]).to_numpy()[ok]
            mean = (lambda a: float(a.mean()) if len(a) else np.nan)
            rows.append({"cell": cell, "family": info["family"], **_xs_cell_factor_cols(cell), "draw": int(draw),
                         "completion": completion, "n_policies": int(ok.sum()), "mean_abs_total": mean(np.abs(tot)),
                         "mean_abs_semantics": mean(np.abs(sem)), "mean_abs_estimation": mean(np.abs(est)),
                         "mean_total": mean(tot), "mean_semantics": mean(sem), "mean_estimation": mean(est),
                         "decomposition_source": source})
    return pd.DataFrame(rows, columns=XS_ERR_DRAW_COLUMNS)


XS_ERR_METRICS = ("mean_abs_total", "mean_abs_semantics", "mean_abs_estimation")


def xs_error_decomposition_table(draws: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    cols = (["cell", "family", "construction", "n_hosp", "tau", "eps", "harm", "completion", "n_draws",
             "mean_policies"] + [f"{m}_{s}" for m in XS_ERR_METRICS for s in ("mean", "ci_low", "ci_high")]
            + ["mean_total", "mean_semantics", "mean_estimation", "decomposition_source"])
    if draws.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (cell, completion), g in draws.groupby(["cell", "completion"], sort=False):
        r = {"cell": cell, "family": g["family"].iloc[0], **_xs_cell_factor_cols(cell), "completion": completion,
             "n_draws": int(len(g)), "mean_policies": float(g["n_policies"].mean()),
             "decomposition_source": g["decomposition_source"].iloc[0]}
        for m in XS_ERR_METRICS:
            v = g[m].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            point, reps = xs_stratified_bootstrap({cell: v[:, None]} if len(v) else {}, xs_stat_mean(),
                                                  ("errdec", completion, m), n_boot)
            r[f"{m}_mean"] = point
            r[f"{m}_ci_low"], r[f"{m}_ci_high"] = xs_ci(reps)
        for m in ("mean_total", "mean_semantics", "mean_estimation"):
            v = g[m].dropna()
            r[m] = float(v.mean()) if len(v) else np.nan
        rows.append(r)
    return pd.DataFrame(rows, columns=cols)


# ---------- preregistered tests H1-H3 (PREREG_PART2) ----------
def xs_h1(rank_draws: pd.DataFrame, n_boot: int) -> Dict[str, Any]:
    """H1: per Gumbel B-est cell, two-sided sign-flip test on per-draw tau(mean) - tau(terminate); Holm across cells."""
    rd = rank_draws[rank_draws["family"].eq("gumbel") & rank_draws["completion"].isin(["mean", "terminate"])] \
        if len(rank_draws) else rank_draws
    cells = [c for c in dict.fromkeys(rd["cell"]) if xs_parse_cell(c)["construction"] == "best"] if len(rd) else []
    rows = []
    for cell in cells:
        sub = rd[rd["cell"] == cell]
        t = sub.pivot(index="draw", columns="completion", values="kendall_tau_b")
        v = t.reindex(columns=["mean", "terminate"]).dropna()
        d = (v["mean"] - v["terminate"]).to_numpy(dtype=np.float64)
        test = sign_flip_test(d, "two-sided", f"xenv|H1|{cell}")
        # Draws in which a benchmark gives every policy the same value score tau = 0 and stay in the primary test;
        # the sensitivity test drops them (flag on either completion).
        flag = (xs_as_bool(sub["tau_benchmark_constant"]) if "tau_benchmark_constant" in sub.columns
                else pd.Series(False, index=sub.index))
        const = flag.groupby(sub["draw"].to_numpy()).any().reindex(v.index, fill_value=False).astype(bool)
        d_ex = d[~const.to_numpy()]
        test_ex = sign_flip_test(d_ex, "two-sided", f"xenv|H1|{cell}|excluding-constant-benchmarks")
        md, reps = xs_stratified_bootstrap({cell: d[:, None]} if len(d) else {}, xs_stat_mean(), ("H1", cell), n_boot)
        lo, hi = xs_ci(reps)
        rows.append({"cell": cell, "n_draws": int(len(t)), "n_pairs": int(len(d)),
                     "n_tau_undefined": int(len(t) - len(d)), "n_benchmark_constant": int(flag.sum()),
                     "mean_tau_mean_rule": float(v["mean"].mean()) if len(v) else np.nan,
                     "mean_tau_terminate_rule": float(v["terminate"].mean()) if len(v) else np.nan,
                     "mean_difference": md, "ci_low": lo, "ci_high": hi, "p_value": test["p_value"],
                     "test_method": test["method"],
                     "min_attainable_p": xs_min_attainable_p(len(d)) if len(d) else np.nan,
                     "n_pairs_excluding_constant_benchmarks": int(len(d_ex)),
                     "p_value_excluding_constant_benchmarks": test_ex["p_value"]})
    adj = holm_adjust({r["cell"]: r["p_value"] for r in rows})
    m = int(sum(np.isfinite(r["p_value"]) for r in rows))
    for r in rows:
        r["holm_p"] = adj[r["cell"]]
        r["reject_holm"] = bool(np.isfinite(r["holm_p"]) and r["holm_p"] <= ALPHA)
        r["holm_first_step_attainable"] = bool(np.isfinite(r["min_attainable_p"])
                                               and r["min_attainable_p"] <= ALPHA / max(m, 1))
    return {"status": "run" if rows else "not run (no Gumbel B-est cell with both completions)",
            "family_size": m, "alpha": ALPHA, "metric": "Kendall tau-b, bench vs true, default policy set",
            "difference": "tau(mean) - tau(terminate)",
            "constant_benchmark": ("a benchmark that gives every policy the same value scores tau = 0 (Kendall tau-a, "
                                   "no ranking information); such draws stay in the primary test and are counted "
                                   "(n_benchmark_constant); p_value_excluding_constant_benchmarks drops them"),
            "rows": rows,
            "n_rejections_holm": int(sum(r["reject_holm"] for r in rows))}


XS_H2_COLUMNS = ["family", "deploy_target", "treatment", "comparator", "status", "n_units", "n_pairs", "n_missing_pairs",
                 "test_unit", "n_test_units",
                 "iqm_treatment", "iqm_comparator", "iqm_difference", "iqm_difference_ci_low", "iqm_difference_ci_high",
                 "mean_difference", "mean_difference_ci_low", "mean_difference_ci_high", "paired_win_rate",
                 "p_value", "test_method", "min_attainable_p", "tost_p", "equivalence_margin"]


def xs_h2(norm: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """H2: per family, paired deployment-z differences (treatment - comparator); sign-flip, Holm, TOST.

    Test units: (cell, draw) pairs pooled across cells for ICU-Sepsis and Gumbel (independent streams on a fixed
    system); for Garnet (XS_DRAW_BLOCKED_FAMILIES) one unit per base-MDP draw, the mean difference over the harm x eps
    cells that share it, and the bootstrap resamples Garnet draws jointly across cells."""
    dep = xs_deploy_scores(norm)
    margin = float(XCFG.equivalence_margin_normalized)
    rows = []
    for fam in XS_FAMILIES:
        d = dep[dep["family"] == fam] if len(dep) else dep
        if d.empty:
            continue
        Z = xs_wide(d, "z")
        blk = fam in XS_DRAW_BLOCKED_FAMILIES
        for treat in XTREATMENTS:
            r = {c: np.nan for c in XS_H2_COLUMNS}
            r.update({"family": fam, "deploy_target": XS_DEPLOY[fam], "treatment": treat, "comparator": XCOMPARATOR,
                      "status": "ok", "n_units": int(len(Z)), "n_pairs": 0, "n_missing_pairs": int(len(Z)),
                      "test_unit": "draw block" if blk else "cell x draw", "n_test_units": 0,
                      "equivalence_margin": margin, "test_method": ""})
            missing = [p for p in (treat, XCOMPARATOR) if p not in Z.columns]
            if missing:
                r["status"] = "missing policy: " + ", ".join(missing)
                rows.append(r)
                continue
            g = xs_cell_groups(Z, [treat, XCOMPARATOR], aligned=blk)
            D = Z[treat] - Z[XCOMPARATOR]
            D = D[np.isfinite(D.to_numpy(dtype=np.float64))]
            unit_d = D.to_numpy(dtype=np.float64)                       # finite (cell, draw) paired differences
            dv = D.groupby(level="draw").mean().to_numpy(dtype=np.float64) if blk else unit_d
            n, tkey = len(dv), f"xenv|H2|{fam}|{treat}"
            test = sign_flip_test(dv, "two-sided", tkey)
            iqm_d, ireps = xs_stratified_bootstrap(g, xs_stat_iqm_diff, ("main", fam, XS_DEPLOY[fam]), n_boot,
                                                   shared_draws=blk)
            mean_d, mreps = xs_stratified_bootstrap(g, xs_stat_mean_diff, ("main", fam, XS_DEPLOY[fam]), n_boot,
                                                    shared_draws=blk)
            pooled = np.concatenate(list(g.values())) if g else np.zeros((0, 2))
            nu = len(unit_d)
            r.update({"n_pairs": int(nu), "n_missing_pairs": int(len(Z) - nu), "n_test_units": int(n),
                      "iqm_treatment": xs_iqm(pooled[:, 0]), "iqm_comparator": xs_iqm(pooled[:, 1]),
                      "iqm_difference": iqm_d, "mean_difference": mean_d,
                      "paired_win_rate": (float(np.mean(unit_d > XS_TIE_ATOL) + 0.5 * np.mean(np.abs(unit_d) <= XS_TIE_ATOL))
                                          if nu else np.nan),
                      "p_value": test["p_value"], "test_method": test["method"],
                      "min_attainable_p": xs_min_attainable_p(n) if n else np.nan,
                      "tost_p": tost(dv, margin, tkey)})
            r["iqm_difference_ci_low"], r["iqm_difference_ci_high"] = xs_ci(ireps)
            r["mean_difference_ci_low"], r["mean_difference_ci_high"] = xs_ci(mreps)
            rows.append(r)
    h2 = pd.DataFrame(rows, columns=XS_H2_COLUMNS)
    extra = ["holm_p", "tost_holm_p", "holm_family_size", "reject_holm", "direction", "equivalent_tost",
             "equivalent_tost_holm", "holm_first_step_attainable"]
    if h2.empty:
        return pd.DataFrame(columns=XS_H2_COLUMNS + extra)
    hyp = h2["family"] + "|" + h2["treatment"]
    h2["holm_p"] = hyp.map(holm_adjust(dict(zip(hyp, h2["p_value"]))))
    h2["tost_holm_p"] = hyp.map(holm_adjust(dict(zip(hyp, h2["tost_p"]))))
    m = int(np.isfinite(h2["p_value"].to_numpy(dtype=np.float64)).sum())
    h2["holm_family_size"] = m
    h2["reject_holm"] = h2["holm_p"] <= ALPHA
    h2["direction"] = np.where(~h2["reject_holm"], "", np.where(h2["mean_difference"] > 0, "treatment better",
                                                               "comparator better"))
    h2["equivalent_tost"] = h2["tost_p"] <= ALPHA
    h2["equivalent_tost_holm"] = h2["tost_holm_p"] <= ALPHA
    h2["holm_first_step_attainable"] = h2["min_attainable_p"] <= ALPHA / max(m, 1)
    return h2


def xs_h3(err_draws: pd.DataFrame, n_perm: int = XS_H3_PERMUTATIONS) -> Dict[str, Any]:
    """H3 (Garnet): Spearman rho between |h| and the per-draw mean |J_true - J_bench_mean| over default policies."""
    e = err_draws[err_draws["family"].eq("garnet") & err_draws["completion"].eq("mean")] if len(err_draws) else err_draws
    e = e[np.isfinite(e["mean_abs_total"].to_numpy(dtype=np.float64))] if len(e) else e
    out = {"status": "not run (no Garnet draws)", "alpha": ALPHA, "metric": "mean |J_true - J_bench_mean| per draw",
           "predictor": "|h|", "n_draw_units": int(len(e))}
    if e.empty:
        return out
    x = np.abs(e["harm"].to_numpy(dtype=np.float64))
    y = e["mean_abs_total"].to_numpy(dtype=np.float64)
    eps = e["eps"].to_numpy(dtype=np.float64)
    # Exchangeability blocks of the design: the harm levels of one draw share its base MDP (part21c ('base', draw)).
    blocks = (e["eps"].astype(str) + "|" + e["draw"].astype(str)).to_numpy()
    primary = xs_permutation_spearman(x, y, blocks, n_perm, ("within-eps-draw",))
    strat = xs_permutation_spearman(x, y, eps, n_perm, ("stratified-eps",))
    unres = xs_permutation_spearman(x, y, None, n_perm, ("unrestricted",))
    out.update({"status": "run" if np.isfinite(primary["rho"]) else "not testable (fewer than 3 draws or constant |h|)",
                "spearman_rho": primary["rho"], "p_value": primary["p_value"],
                "test": "permutation of |h| within (eps, draw) blocks", "n_blocks": primary.get("n_strata", 0),
                "p_value_permutation_within_eps_draw": primary["p_value"],
                "p_value_permutation_within_eps": strat["p_value"],
                "p_value_permutation_unrestricted": unres["p_value"], "n_permutations": int(n_perm),
                "eps_strata": sorted(float(v) for v in np.unique(eps)),
                "reject": bool(np.isfinite(primary["p_value"]) and primary["p_value"] <= ALPHA),
                "per_eps_spearman_descriptive": [{"eps": float(v), "n": int((eps == v).sum()),
                                                  "rho": xs_spearman(x[eps == v], y[eps == v])} for v in np.unique(eps)],
                "mean_error_by_harm": [{"harm": float(h), "abs_harm": abs(float(h)), "n": int(len(g)),
                                        "mean_abs_error": float(g["mean_abs_total"].mean())}
                                       for h, g in e.groupby("harm")]})
    return out


def xs_h3_descriptive(err_draws: pd.DataFrame, norm: pd.DataFrame, draw_meta: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Descriptive companion of H3 (Garnet). The harm also changes the benchmark mask (absorbing outcomes shorten the
    hospital episodes), so H3 mixes the size of the harm with the number of unsupported actions. Reported by (h, eps):
    mean admissible actions of the benchmark (draw meta), mean expected violations of the default policies on the
    true target, the mean |error|; and, exploratory, the mean |error| per expected violation by h (ratio of means)."""
    e = (err_draws[err_draws["family"].eq("garnet") & err_draws["completion"].eq("mean")]
         if len(err_draws) else err_draws)
    if e.empty:
        return {"status": "not run (no Garnet draws)"}
    units = e[["cell", "draw", "harm", "eps", "mean_abs_total", "mean_abs_semantics"]].copy()
    units["draw"] = units["draw"].astype(np.int64)
    meta = xs_as_frame(draw_meta)
    col = "instance.construction.mean_admissible_actions"
    if col in meta.columns and {"cell", "draw"} <= set(meta.columns):
        m = pd.DataFrame({"cell": meta["cell"].astype(str), "draw": pd.to_numeric(meta["draw"]).astype(np.int64),
                          "mean_admissible_actions": pd.to_numeric(meta[col], errors="coerce")})
        units = units.merge(m.drop_duplicates(["cell", "draw"]), on=["cell", "draw"], how="left", validate="one_to_one")
    else:
        units["mean_admissible_actions"] = np.nan
    t = (norm[norm["family"].eq("garnet") & norm["is_default"] & norm["target"].eq("true") & norm["status"].eq("ok")]
         if len(norm) else norm)
    ev = (t.groupby(["cell", "draw"])["expected_violations"].mean().rename("mean_expected_violations_true")
          .reset_index() if len(t) else pd.DataFrame(columns=["cell", "draw", "mean_expected_violations_true"]))
    ev["draw"] = ev["draw"].astype(np.int64)
    units = units.merge(ev, on=["cell", "draw"], how="left", validate="one_to_one")
    mean = (lambda s: float(s.mean()) if s.notna().any() else np.nan)
    by_cell = [{"harm": float(h), "eps": float(ep), "n_draws": int(len(g)),
                "mean_admissible_actions": mean(g["mean_admissible_actions"]),
                "mean_expected_violations_true": mean(g["mean_expected_violations_true"]),
                "mean_abs_error": mean(g["mean_abs_total"]), "mean_abs_semantics_term": mean(g["mean_abs_semantics"])}
               for (h, ep), g in units.groupby(["harm", "eps"])]
    per_viol = []
    for h, g in units.groupby("harm"):
        v = mean(g["mean_expected_violations_true"])
        per_viol.append({"harm": float(h), "n_draw_units": int(len(g)), "mean_expected_violations_true": v,
                         "mean_abs_error_per_expected_violation": (mean(g["mean_abs_total"]) / v
                                                                   if np.isfinite(v) and v > 0 else np.nan),
                         "mean_abs_semantics_per_expected_violation": (mean(g["mean_abs_semantics"]) / v
                                                                       if np.isfinite(v) and v > 0 else np.nan)})
    return {"status": "descriptive", "by_harm_eps": by_cell, "exploratory_error_per_expected_violation": per_viol,
            "note": ("the harm changes the mask itself, so H3 mixes the size of the harm with the number of "
                     "unsupported actions; the per-violation ratios (ratio of means over draws) separate the two")}


def xs_design_deviations(prereg: Dict[str, Any], xcfg: Any) -> List[Dict[str, Any]]:
    """Every PREREG_PART2['design'] key whose realised XCFG value differs (tuples compared as JSON lists)."""
    out = []
    for k, v in (prereg.get("design") or {}).items():
        pre = json.loads(canonical_json(v))
        real = json.loads(canonical_json(getattr(xcfg, k, None)))
        if pre != real:
            out.append({"key": k, "preregistered": pre, "realised": real})
    return out


def xs_prereg_report(h1: Dict[str, Any], h2: pd.DataFrame, h3: Dict[str, Any]) -> Dict[str, Any]:
    status = str(PREREG_PART2.get("status", ""))
    locked = status.upper().startswith("LOCKED")
    h2_ok = h2[h2["status"].eq("ok")] if len(h2) else h2
    return {"prereg": PREREG_PART2, "prereg_hash": obj_hash(PREREG_PART2), "prereg_locked": locked,
            "warning": ("" if locked else "PREREG_PART2 is not locked: these tests are exploratory until the "
                        "preregistration is locked before the confirmatory run."),
            "design_deviations": xs_design_deviations(PREREG_PART2, XCFG),
            "alpha": ALPHA, "deploy_targets": XS_DEPLOY, "comparator": XCOMPARATOR, "treatments": XTREATMENTS,
            "H1": h1,
            "H2": {"status": "run" if len(h2_ok) else "not run", "table": "xenv/xenv_h2_family.csv",
                   "test_units": {f: ("draw block (mean over the cells sharing a base MDP)"
                                      if f in XS_DRAW_BLOCKED_FAMILIES else "(cell, draw) pooled across cells")
                                  for f in XS_FAMILIES},
                   "family_size": int(h2["holm_family_size"].iloc[0]) if len(h2) else 0,
                   "n_rejections_holm": int(h2_ok["reject_holm"].sum()) if len(h2_ok) else 0,
                   "n_equivalent_tost_holm": int(h2_ok["equivalent_tost_holm"].sum()) if len(h2_ok) else 0,
                   "rows": h2.to_dict(orient="records")},
            "H3": h3}


# ---------- hyper-parameter selection: default vs FQE vs oracle ----------
XS_HP_COLUMNS = (["family", "deploy_target", "method", "variant", "method_family", "n_configs", "configs", "n_units"]
                 + [f"{p}_{s}" for p in ("default", "fqe_selected", "oracle_selected_upper_bound")
                    for s in ("n", "iqm", "iqm_ci_low", "iqm_ci_high", "mean")]
                 + ["mean_fqe_minus_default", "mean_oracle_minus_fqe", "fqe_picks_default_rate", "fqe_modal_hp",
                    "fqe_modal_hp_rate", "fqe_target_available", "fqe_fallback_units"])


def _xs_select(m: pd.DataFrame, score: str, keys: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per unit, the configurations maximising `score` (ties within XS_TIE_ATOL); z averaged over the tie set."""
    s = m[score]
    mx = s.groupby([m[k] for k in keys]).transform("max")
    tie = s.notna() & (s >= mx - XS_TIE_ATOL)
    t = m[tie].copy()
    if t.empty:
        return pd.DataFrame(columns=["z", "default_share"], index=pd.MultiIndex.from_tuples([], names=keys)), t
    t["w"] = 1.0 / t.groupby(keys)["hp"].transform("size")
    agg = t.groupby(keys).agg(z=("z", lambda v: float(v.mean()) if v.notna().all() else np.nan),
                              default_share=("is_default", "mean"))
    return agg, t


def xs_hp_selection(norm: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """table_hp_selection.csv for every tunable (method, variant) and family, on the deployment target."""
    empty = pd.DataFrame(columns=XS_HP_COLUMNS)
    dep = xs_deploy_scores(norm, defaults_only=False)
    if dep.empty:
        return empty
    ncfg = dep.groupby(["family", "method", "variant"])["hp"].nunique()
    tun = ncfg[ncfg >= 2].reset_index()[["family", "method", "variant"]]
    if tun.empty:
        return empty
    dep = dep.merge(tun, on=["family", "method", "variant"], how="inner")
    fq = norm.loc[norm["target"].eq(XS_FQE), ["cell", "draw", "policy", "return"]].rename(columns={"return": "fqe_return"})
    m = dep.merge(fq, on=["cell", "draw", "policy"], how="left", validate="one_to_one")
    keys = ["family", "cell", "draw", "method", "variant"]
    units = m[keys].drop_duplicates().set_index(keys)
    dflt = m[m["is_default"]].groupby(keys)["z"]
    assert len(dflt.size()) == 0 or int(dflt.size().max()) == 1, "more than one default configuration in a unit"
    units["z_default"] = dflt.first()
    fsel, ft = _xs_select(m, "fqe_return", keys)
    osel, _ = _xs_select(m, "return", keys)
    units["z_fqe"] = fsel["z"]
    units["fqe_default_share"] = fsel["default_share"]
    units["z_oracle"] = osel["z"]
    # A unit whose candidates are all improper on the FQE model has no FQE choice: it falls back to the pre-declared
    # default, so default, FQE and oracle columns describe the same units (fqe_default_share stays NaN there).
    decided = units.index.isin(fsel.index)
    units["fqe_fallback"] = ~decided & (len(fq) > 0)
    if len(fq) > 0:
        units.loc[~decided, "z_fqe"] = units.loc[~decided, "z_default"]
    rows = []
    for (fam, meth, var), u in units.groupby(level=["family", "method", "variant"], sort=False):
        W = u.reset_index().set_index(["cell", "draw"])
        sub = dep[(dep["family"] == fam) & (dep["method"] == meth) & (dep["variant"] == var)]
        r = {"family": fam, "deploy_target": XS_DEPLOY[fam], "method": meth, "variant": var,
             "method_family": sub["method_family"].iloc[0], "n_configs": int(sub["hp"].nunique()),
             "configs": "; ".join(sorted(sub["hp"].unique())), "n_units": int(len(W)),
             "fqe_target_available": bool(len(fq) > 0), "fqe_fallback_units": int(W["fqe_fallback"].sum())}
        blk = fam in XS_DRAW_BLOCKED_FAMILIES
        for label, col in (("default", "z_default"), ("fqe_selected", "z_fqe"),
                           ("oracle_selected_upper_bound", "z_oracle")):
            g = xs_cell_groups(W, [col], aligned=blk)
            point, reps = xs_stratified_bootstrap(g, xs_stat_iqm(), ("hp", fam), n_boot, shared_draws=blk)
            v = W[col].dropna()
            r[f"{label}_n"] = int(len(v))
            r[f"{label}_iqm"] = point
            r[f"{label}_iqm_ci_low"], r[f"{label}_iqm_ci_high"] = xs_ci(reps)
            r[f"{label}_mean"] = float(v.mean()) if len(v) else np.nan
        a, b, c = W["z_fqe"], W["z_default"], W["z_oracle"]
        r["mean_fqe_minus_default"] = float((a - b).dropna().mean()) if (a - b).notna().any() else np.nan
        r["mean_oracle_minus_fqe"] = float((c - a).dropna().mean()) if (c - a).notna().any() else np.nan
        r["fqe_picks_default_rate"] = float(W["fqe_default_share"].mean()) if W["fqe_default_share"].notna().any() else np.nan
        mt = ft[(ft["family"] == fam) & (ft["method"] == meth) & (ft["variant"] == var)] if len(ft) else ft
        if len(mt):
            cnt = mt.groupby("hp")["w"].sum().sort_values(ascending=False, kind="mergesort")
            r["fqe_modal_hp"], r["fqe_modal_hp_rate"] = str(cnt.index[0]), float(cnt.iloc[0] / cnt.sum())
        else:
            r["fqe_modal_hp"], r["fqe_modal_hp_rate"] = "", np.nan
        rows.append(r)
    return pd.DataFrame(rows, columns=XS_HP_COLUMNS)


def xs_proposition_summary(props: pd.DataFrame, propositions_ok: Optional[bool]) -> Dict[str, Any]:
    if props.empty:
        return {"status": "not run", "PROPOSITIONS_OK": propositions_ok}
    assert {"proposition", "passes"} <= set(props.columns), f"PROP_CHECKS columns: {list(props.columns)}"
    flag = props["passes"].astype(str).str.strip().str.lower()
    per = {str(p): {"checks": int(len(g)), "passed": int((flag[g.index] == "true").sum()),
                    "failed": int((flag[g.index] == "false").sum())} for p, g in props.groupby("proposition")}
    return {"status": "run", "checks": int(len(props)), "passed": int((flag == "true").sum()),
            "failed": int((flag == "false").sum()), "other": int((~flag.isin(["true", "false"])).sum()),
            "per_proposition": per, "PROPOSITIONS_OK": propositions_ok}

# %% [markdown]
# ### II.7 figures
#
# | Figure | Content |
# |---|---|
# | `fig_xenv_frontiers` | exact CMDP frontiers: optimal value vs expected-violation budget κ per environment and target. The κ = ∞ value is the dashed line; on every benchmark the frontier is flat, under the mean rule by Proposition P2 and under termination by the P3 corollary (r_term = v_min); panels are (cell, environment) |
# | `fig_xenv_rank_fidelity` | Kendall τ-b, top-1 agreement and selection regret of the benchmark's ranking against the true system, per cell and completion rule |
# | `fig_xenv_harm_phase` | Garnet phase diagram: benchmark error and rank fidelity against the harm *h* of unsupported actions, per behaviour ε (*h* = 0: non-preferred unsupported actions are completed correctly; the semantics term vanishes where K(s) ⊆ M(s), see `table_error_decomposition.csv`) |
# | `fig_xenv_iqm` | IQM of the deployment score with stratified-bootstrap intervals for every default policy, per family |
# | `fig_xenv_profiles` | performance profiles of the H2 treatments against the comparator (small multiples of four treatments each) |
# | `fig_xenv_poi` | pooled probability of improvement between default policies (diverging scale centred at ½) |
# | `fig_xenv_error_decomposition` | mean absolute benchmark error and its semantics and estimation-and-coverage terms per cell |
#
# A figure whose inputs are empty (because a switch is off) is not drawn; the reason is recorded in
# `XSTATS["figures"]`.

# %%
def _xs_err(mean, lo, hi) -> np.ndarray:
    mean, lo, hi = (np.asarray(v, dtype=np.float64) for v in (mean, lo, hi))
    return np.vstack([np.clip(mean - lo, 0.0, None), np.clip(hi - mean, 0.0, None)])


def _xs_save(fig, fig_dir: Path, stem: str) -> str:
    save_figure(fig, Path(fig_dir) / stem, dpi=200)
    return "written"


def xs_fig_frontiers(fr: pd.DataFrame, fig_dir: Path) -> str:
    if fr.empty:
        return "not produced: X_FRONTIERS is empty (switch xenv_frontiers off or Section II.5 not run)"
    obj = next((c for c in ("objective", "lp_objective", "value", "return") if c in fr.columns), None)
    assert "kappa" in fr.columns and obj is not None, f"X_FRONTIERS needs 'kappa' and an objective column: {list(fr.columns)}"
    panel = next((c for c in ("environment", "env", "instance") if c in fr.columns), None)
    lines = [c for c in ("target", "semantics", "cost") if c in fr.columns and c != panel]
    fr = fr.assign(kappa=pd.to_numeric(fr["kappa"], errors="raise"), _obj=pd.to_numeric(fr[obj], errors="raise"))
    if panel is not None and "cell" in fr.columns:
        # Environment names are not unique across cells (e.g. the best and oracle Gumbel cells share the name of the
        # true system with a benchmark mask, but their masks differ), so panels are (cell, environment).
        fr = fr.assign(_panel=fr["cell"].astype(str) + "\n" + fr[panel].astype(str))
        panel = "_panel"
    if panel is not None:
        dup = fr.groupby([panel, *lines], sort=False)["kappa"].apply(lambda k: k.duplicated().any())
        assert not dup.any(), f"several frontiers share one figure line: {list(dup[dup].index)[:3]}"
    panels = list(dict.fromkeys(fr[panel])) if panel else ["all environments"]
    nc = min(3, len(panels))
    nr = int(math.ceil(len(panels) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.8 * nc, 3.6 * nr), squeeze=False)
    finite_k = fr.loc[np.isfinite(fr["kappa"]) & (fr["kappa"] > 0), "kappa"]
    lin = float(finite_k.min()) if len(finite_k) else 1e-3
    for ax, p in zip(axes.ravel(), panels):
        g = fr[fr[panel] == p] if panel else fr
        series = list(g.groupby(lines, sort=False)) if lines else [("", g)]
        for k, (lab, h) in enumerate(series):
            h = h.sort_values("kappa")
            fin = np.isfinite(h["kappa"].to_numpy())
            color = XS_COLORS[k % len(XS_COLORS)] if len(series) <= len(XS_COLORS) else None
            label = " / ".join(map(str, lab)) if isinstance(lab, tuple) else str(lab)
            ln, = ax.plot(h["kappa"][fin], h["_obj"][fin], marker="o", ms=4, lw=1.5, color=color, label=label or None)
            if (~fin).any():
                ax.axhline(float(h["_obj"][~fin].iloc[0]), color=ln.get_color(), lw=1, ls="--")
        ax.set_xscale("symlog", linthresh=lin)
        ax.set_title(str(p), fontsize=8)
        ax.set_xlabel("expected-violation budget κ (dashed: κ = ∞)")
        ax.set_ylabel("exact constrained optimum")
        ax.grid(alpha=0.3, lw=0.5)
        if lines:
            ax.legend(fontsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Exact CMDP frontiers (cost = 1[a not admissible])")
    return _xs_save(fig, fig_dir, "fig_xenv_frontiers")


def xs_fig_rank_fidelity(rank: pd.DataFrame, fig_dir: Path) -> str:
    r = rank[rank["completion"].isin(["mean", "terminate"])] if len(rank) else rank
    if r.empty:
        return "not produced: no benchmark-vs-true comparisons (Gumbel and Garnet off)"
    cells = list(dict.fromkeys(r["cell"]))
    x = np.arange(len(cells))
    panels = [("kendall_tau_b", "Kendall τ-b (benchmark vs true)"), ("top1_agreement", "top-1 agreement"),
              ("selection_regret_z", "selection regret (z, true target)")]
    fig, axes = plt.subplots(3, 1, figsize=(max(8.0, 0.55 * len(cells) + 3), 9), sharex=True)
    for ax, (m, title) in zip(axes, panels):
        for j, comp in enumerate(("mean", "terminate")):
            g = r[r["completion"] == comp].set_index("cell").reindex(cells)
            ax.bar(x + (j - 0.5) * 0.4, g[f"{m}_mean"], 0.38, color=XS_COLORS[j], edgecolor="white", lw=0.6,
                   yerr=_xs_err(g[f"{m}_mean"], g[f"{m}_ci_low"], g[f"{m}_ci_high"]), capsize=2,
                   error_kw={"elinewidth": 0.8, "ecolor": XS_INK}, label=f"{comp} rule")
        ax.axhline(0, color=XS_MUTED, lw=0.8)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
    axes[0].legend(fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(cells, rotation=40, ha="right", fontsize=7)
    fig.suptitle("Benchmark rank fidelity over the default policy set "
                 f"(mean over draws, {100 * (1 - ALPHA):.0f}% bootstrap CI)")
    return _xs_save(fig, fig_dir, "fig_xenv_rank_fidelity")


def xs_fig_harm_phase(err: pd.DataFrame, rank: pd.DataFrame, fig_dir: Path) -> str:
    e = err[err["family"] == "garnet"] if len(err) else err
    k = rank[(rank["family"] == "garnet") & rank["completion"].isin(["mean", "terminate"])] if len(rank) else rank
    if e.empty and k.empty:
        return "not produced: no Garnet results (switch xenv_garnet off)"
    eps_vals = sorted(set(e["eps"].dropna()) | set(k["eps"].dropna()))
    styles = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D")]
    harms = pd.concat([e["harm"], k["harm"]]).dropna()
    dodge = 0.012 * max(float(harms.max() - harms.min()), 1.0) if len(harms) else 0.0   # keeps error bars apart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (df, col, ylab) in zip(axes, ((e, "mean_abs_total", "mean |J_true − J_bench| (default policies)"),
                                          (k, "kendall_tau_b", "Kendall τ-b (benchmark vs true)"))):
        for j, comp in enumerate(("mean", "terminate")):
            for i, ev in enumerate(eps_vals):
                g = df[(df["completion"] == comp) & (df["eps"] == ev)].sort_values("harm") if len(df) else df
                if g.empty:
                    continue
                ls, mk = styles[i % len(styles)]
                ax.errorbar(g["harm"] + (2 * j + i - 1.5) * dodge, g[f"{col}_mean"],
                            yerr=_xs_err(g[f"{col}_mean"], g[f"{col}_ci_low"], g[f"{col}_ci_high"]),
                            color=XS_COLORS[j], ls=ls, marker=mk, ms=6, lw=1.5, capsize=3, label=f"{comp} rule, ε = {ev:g}")
        ax.axvline(0.0, color=XS_MUTED, lw=0.8, ls=":")
        ax.set_xlabel("harm h of unsupported actions")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=7)
    axes[0].set_title("(a) Benchmark error")
    axes[1].set_title("(b) Rank fidelity")
    fig.suptitle("Garnet phase diagram: when does the completion rule mislead? (h = 0: non-preferred unsupported "
                 "actions completed correctly; the semantics term vanishes where K(s) ⊆ M(s); points dodged)")
    return _xs_save(fig, fig_dir, "fig_xenv_harm_phase")


def xs_fig_iqm(main_long: pd.DataFrame, order: Sequence[str], fig_dir: Path) -> str:
    if main_long.empty:
        return "not produced: no deployment scores"
    fams = [f for f in XS_FAMILIES if f in set(main_long["family"])]
    pols = [p for p in order if p in set(main_long["policy"])]
    y = np.arange(len(pols))
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams) + 3.0, 0.22 * len(pols) + 1.8), sharey=True,
                             squeeze=False)
    groups = (("other", XS_MUTED, "other default policies"), ("treatment", XS_COLORS[0], "H2 treatments"),
              ("comparator", XS_COLORS[1], f"comparator: {XCOMPARATOR}"))
    for ax, fam in zip(axes[0], fams):
        f = main_long[main_long["family"] == fam].set_index("policy").reindex(pols)
        for role, color, label in groups:
            sel = (f["role"] == role).to_numpy()
            if sel.any():
                ax.errorbar(f["iqm"][sel], y[sel], xerr=_xs_err(f["iqm"][sel], f["iqm_ci_low"][sel], f["iqm_ci_high"][sel]),
                            fmt="o", ms=4, color=color, ecolor=color, elinewidth=1, capsize=2, label=label)
        if XCOMPARATOR in f.index and np.isfinite(f.at[XCOMPARATOR, "iqm"]):
            ax.axvline(f.at[XCOMPARATOR, "iqm"], color=XS_COLORS[1], lw=0.8, ls="--")
        for v in (0.0, 1.0):
            ax.axvline(v, color=XS_MUTED, lw=0.6)
        ax.set_title(f"{fam} (target: {XS_DEPLOY[fam]})", fontsize=9)
        ax.set_xlabel("IQM of normalised score z")
        ax.grid(axis="x", alpha=0.3, lw=0.5)
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels(pols, fontsize=7)
    axes[0][0].invert_yaxis()
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Deployment performance: IQM with {100 * (1 - ALPHA):.0f}% stratified-bootstrap CI "
                 "(z = 1 optimal, 0 behaviour)")
    return _xs_save(fig, fig_dir, "fig_xenv_iqm")


def xs_fig_profiles(prof: pd.DataFrame, fig_dir: Path) -> str:
    if prof.empty:
        return "not produced: no deployment scores"
    fams = [f for f in XS_FAMILIES if f in set(prof["family"])]
    treat = [t for t in XTREATMENTS if t in set(prof["policy"])]
    blocks = [treat[i:i + 4] for i in range(0, len(treat), 4)] or [[]]
    fig, axes = plt.subplots(len(fams), len(blocks), figsize=(4.4 * len(blocks), 3.4 * len(fams)), squeeze=False,
                             sharex=True, sharey=True)
    for i, fam in enumerate(fams):
        pf = prof[prof["family"] == fam]
        for j, block in enumerate(blocks):
            ax = axes[i][j]
            curves = [(XCOMPARATOR, XS_INK, 2.2)] + [(p, XS_COLORS[q], 1.4) for q, p in enumerate(block)]
            for pol, color, lw in curves:
                g = pf[pf["policy"] == pol]
                if g.empty:
                    continue
                ax.plot(g["threshold"], g["fraction_at_least"], color=color, lw=lw, label=pol)
                ax.fill_between(g["threshold"], g["ci_low"], g["ci_high"], color=color, alpha=0.12, lw=0)
            ax.set_title(f"{fam} (target: {XS_DEPLOY[fam]})", fontsize=9)
            ax.grid(alpha=0.3, lw=0.5)
            ax.legend(fontsize=6, loc="lower left")
            if i == len(fams) - 1:
                ax.set_xlabel("normalised score threshold x")
            if j == 0:
                ax.set_ylabel("fraction of (cell, draw) with z ≥ x", fontsize=8)
    fig.suptitle("Performance profiles of the H2 treatments vs the comparator (black), pointwise stratified-bootstrap bands")
    return _xs_save(fig, fig_dir, "fig_xenv_profiles")


def xs_fig_poi(poi: pd.DataFrame, order: Sequence[str], fig_dir: Path) -> str:
    if poi.empty:
        return "not produced: no deployment scores"
    pols = [p for p in order if p in set(poi["policy_x"])] + [p for p in dict.fromkeys(poi["policy_x"]) if p not in set(order)]
    M = (poi.pivot(index="policy_x", columns="policy_y", values="poi").reindex(index=pols, columns=pols)
         .to_numpy(dtype=np.float64))
    cmap = LinearSegmentedColormap.from_list("xs_diverging", [XS_COLORS[7], XS_NEUTRAL, XS_COLORS[0]]).with_extremes(
        bad="#ffffff")
    n = len(pols)
    fig, ax = plt.subplots(figsize=(0.2 * n + 4.0, 0.2 * n + 3.0))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0.0, vmax=1.0)
    fs = float(np.clip(300.0 / max(n, 1), 4.0, 8.0))
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(pols, rotation=90, fontsize=fs)
    ax.set_yticklabels(pols, fontsize=fs)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="P(row policy > column policy)")
    ax.set_title("Probability of improvement on the deployment targets (rliable; averaged over all cells)", fontsize=9)
    return _xs_save(fig, fig_dir, "fig_xenv_poi")


def xs_fig_error_decomposition(err: pd.DataFrame, fig_dir: Path) -> str:
    if err.empty:
        return "not produced: no benchmark-vs-true comparisons (Gumbel and Garnet off)"
    comps = [c for c in ("mean", "terminate") if c in set(err["completion"])]
    cells = list(dict.fromkeys(err["cell"]))
    x = np.arange(len(cells))
    terms = (("mean_abs_total", "|J_true − J_bench| (total)"), ("mean_abs_semantics", "|semantics term|"),
             ("mean_abs_estimation", "|estimation + coverage term|"))
    fig, axes = plt.subplots(len(comps), 1, figsize=(max(8.0, 0.6 * len(cells) + 3), 3.6 * len(comps)), squeeze=False,
                             sharex=True)
    for ax, comp in zip(axes[:, 0], comps):
        e = err[err["completion"] == comp].set_index("cell").reindex(cells)
        for k, (m, lab) in enumerate(terms):
            ax.bar(x + (k - 1) * 0.27, e[f"{m}_mean"], 0.25, color=XS_COLORS[k], edgecolor="white", lw=0.6,
                   yerr=_xs_err(e[f"{m}_mean"], e[f"{m}_ci_low"], e[f"{m}_ci_high"]), capsize=2,
                   error_kw={"elinewidth": 0.8, "ecolor": XS_INK}, label=lab)
        ax.set_title(f"Completion rule: {comp}", fontsize=9)
        ax.set_ylabel("mean absolute value error", fontsize=9)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(cells, rotation=40, ha="right", fontsize=7)
    fig.suptitle("Exact decomposition of the benchmark error over default policies (mean over draws, bootstrap CI)")
    return _xs_save(fig, fig_dir, "fig_xenv_error_decomposition")


# ---------- orchestration ----------
def xs_run(results: Optional[pd.DataFrame], draw_meta: Optional[pd.DataFrame], frontiers: Any, prop_checks: Any,
           propositions_ok: Optional[bool], cells_spec: Optional[Sequence[dict]], experiments: Dict[str, bool],
           table_dir: Path, fig_dir: Path, n_boot: int) -> Tuple[Dict[str, Any], bool]:
    """Section II.7 end to end: checks, every table, the preregistered tests and the figures. Returns (XSTATS, done)."""
    t0 = time.perf_counter()
    table_dir, fig_dir = Path(table_dir), Path(fig_dir)
    families_on = [f for f in XS_FAMILIES if experiments[f"xenv_{f}"]]
    assert results is not None or not families_on, \
        f"XRESULTS is undefined although {families_on} are switched on (run Section II.6 first)"
    xr = xs_prepare_results(results)
    checks = {"selftests": xs_selftests(), "design": xs_design_checks(xr, cells_spec, experiments),
              "draw_meta": xs_draw_meta_checks(draw_meta, xr)}
    norm, checks["normalization"] = xs_normalized_scores(xr)
    del xr
    gc.collect()
    checks["references"] = xs_reference_checks(norm)
    checks["semantic_groups"] = xs_semantic_group_checks(norm)
    outputs = []

    def write_csv(df: pd.DataFrame, name: str) -> None:
        df.to_csv(table_dir / name, index=False)
        outputs.append(f"xenv/{name}")

    write_csv(norm, "xenv_normalized_scores.csv")
    main_long = xs_main_long(norm, n_boot)
    order = xs_policy_order(main_long)
    main = xs_main_wide(main_long)
    write_csv(main, "table_xenv_main.csv")
    write_csv(main_long, "table_xenv_main_long.csv")
    cell_order = [str(s["cell"]) for s in (cells_spec or [])]
    rank_draws = xs_sort_cells(xs_rank_fidelity_draws(norm), cell_order, ["draw"])
    rank = xs_rank_fidelity_table(rank_draws, n_boot)
    write_csv(rank, "table_rank_fidelity.csv")
    write_csv(rank_draws, "table_rank_fidelity_draws.csv")
    err_draws = xs_sort_cells(xs_error_decomposition_draws(norm), cell_order, ["draw"])
    err = xs_error_decomposition_table(err_draws, n_boot)
    write_csv(err, "table_error_decomposition.csv")
    write_csv(err_draws, "table_error_decomposition_draws.csv")
    h1, h2, h3 = xs_h1(rank_draws, n_boot), xs_h2(norm, n_boot), xs_h3(err_draws)
    h3["descriptive_mask_and_violations"] = xs_h3_descriptive(err_draws, norm, draw_meta)
    prereg = xs_prereg_report(h1, h2, h3)
    save_json(prereg, table_dir / "xenv_prereg_tests.json")
    outputs.append("xenv/xenv_prereg_tests.json")
    write_csv(h2, "xenv_h2_family.csv")
    hp = xs_hp_selection(norm, n_boot)
    write_csv(hp, "table_hp_selection.csv")
    prof = xs_performance_profiles(norm, n_boot)
    write_csv(prof, "table_performance_profiles.csv")
    poi, checks["poi"] = xs_poi_matrix(norm, order)
    write_csv(poi, "xenv_poi_matrix.csv")
    fr, props = xs_as_frame(frontiers), xs_as_frame(prop_checks)
    checks["propositions"] = xs_proposition_summary(props, propositions_ok)
    checks["frontiers"] = {"rows": int(len(fr)), "columns": [str(c) for c in fr.columns]}
    figures = {"fig_xenv_frontiers": xs_fig_frontiers(fr, fig_dir),
               "fig_xenv_rank_fidelity": xs_fig_rank_fidelity(rank, fig_dir),
               "fig_xenv_harm_phase": xs_fig_harm_phase(err, rank, fig_dir),
               "fig_xenv_iqm": xs_fig_iqm(main_long, order, fig_dir),
               "fig_xenv_profiles": xs_fig_profiles(prof, fig_dir),
               "fig_xenv_poi": xs_fig_poi(poi, order, fig_dir),
               "fig_xenv_error_decomposition": xs_fig_error_decomposition(err, fig_dir)}
    outputs += [f"figures/xenv/{k}.{ext}" for k, v in figures.items() if v == "written" for ext in ("png", "pdf")]
    done = bool(checks["design"]["cells"]) and not norm.empty
    checks.update({"figures": figures, "outputs": outputs, "n_boot": int(n_boot), "ci_level": 1 - ALPHA,
                   "deploy_targets": XS_DEPLOY, "xenv_done": done, "elapsed_s": time.perf_counter() - t0})
    save_json(checks, table_dir / "xenv_stats_checks.json")
    outputs.append("xenv/xenv_stats_checks.json")
    xstats = {"normalized_scores": norm, "main": main, "main_long": main_long, "policy_order": order,
              "rank_fidelity": rank, "rank_fidelity_draws": rank_draws, "error_decomposition": err,
              "error_decomposition_draws": err_draws, "prereg_tests": prereg, "h2_family": h2, "hp_selection": hp,
              "performance_profiles": prof, "poi_matrix": poi, "checks": checks, "figures": figures,
              "outputs": outputs}
    return xstats, done


XSTATS, XENV_DONE = xs_run(
    results=globals().get("XRESULTS"), draw_meta=globals().get("XDRAW_META"), frontiers=globals().get("X_FRONTIERS"),
    prop_checks=globals().get("PROP_CHECKS"), propositions_ok=globals().get("PROPOSITIONS_OK"),
    cells_spec=globals().get("X_CELLS"), experiments=EXPERIMENTS, table_dir=XENV_TABLE_DIR, fig_dir=XENV_FIG_DIR,
    n_boot=int(XCFG.bootstrap))

_xn = XSTATS["checks"]["normalization"]
print(f"Normalised scores: {_xn.get('rows', 0)} rows, {_xn.get('z_finite_rows', 0)} finite; "
      f"{_xn.get('groups_denominator_below_1e-9', 0)} (cell, draw, target) groups with J* - J_b < 1e-9 "
      f"({_xn.get('rows_denominator_below_1e-9', 0)} rows set to NaN); {_xn.get('improper_rows', 0)} improper rows.")
if len(XSTATS["main"]):
    _fams = [f for f in XS_FAMILIES if f"{f}_iqm_ci" in XSTATS["main"].columns]
    display(XSTATS["main"][["policy", "role"] + [c for f in _fams for c in (f"{f}_iqm_ci", f"{f}_poi_ci",
                                                                            f"{f}_mean_expected_violations")]])
if len(XSTATS["rank_fidelity"]):
    display(XSTATS["rank_fidelity"][["cell", "completion", "n_draws", "kendall_tau_b_mean", "kendall_tau_b_ci_low",
                                     "kendall_tau_b_ci_high", "top1_agreement_mean", "selection_regret_z_mean"]].round(4))
_pt = XSTATS["prereg_tests"]
if _pt["warning"]:
    print("WARNING:", _pt["warning"])
print("H1:", _pt["H1"]["status"], "| Holm rejections:", _pt["H1"]["n_rejections_holm"], "of", _pt["H1"]["family_size"])
if len(XSTATS["h2_family"]):
    display(XSTATS["h2_family"][["family", "treatment", "test_unit", "n_test_units", "mean_difference",
                                 "mean_difference_ci_low", "mean_difference_ci_high", "iqm_difference", "p_value",
                                 "holm_p", "tost_holm_p", "reject_holm", "equivalent_tost_holm"]].round(5))
print("H3:", _pt["H3"]["status"], {k: _pt["H3"].get(k) for k in ("spearman_rho", "p_value",
                                                                 "p_value_permutation_within_eps",
                                                                 "p_value_permutation_unrestricted")})
if _pt["design_deviations"]:
    print("NOTE: realised Part II design differs from PREREG_PART2['design']:",
          [d["key"] for d in _pt["design_deviations"]])
print("Figures:", XSTATS["figures"])
print("XENV_DONE =", XENV_DONE, "(all Part II statistics written)" if XENV_DONE else
      "(Part II not run: every xenv family switch is off or X_CELLS is empty)")
