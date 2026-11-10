# %% [markdown]
# ### II.9g Clean rerun of Part II into an empty cache
#
# Every Part II number so far was produced through the identity-addressed cache `XCACHE`. Draws computed in different
# Colab sessions were joined by their identities; the v2 session, for example, reloaded all 500 II.6 draws from the
# cache (`xenv_driver_summary.json`). A run without caches is the strongest replay check. This section reruns the Part II
# draws from scratch into a new, **empty** cache directory and compares every rerun table row by row with the cached
# table of the same draw. A match shows that each cached result is what the current code computes from scratch: no
# stale entry under a matching identity, no state that the identity misses, and a deterministic pipeline.
#
# **Scope** (`XV4CFG.clean_scope`; `max_draws` caps both):
#
# | Scope | Draws rerun |
# |---|---|
# | `all` (paper + full) | every draw of every enabled II.6 cell (`X_CELLS`, family switches as in II.6a; `run_cell_draw`) and, with `clean_include_v3`, of every II.8 cell (`X_V3_CELLS`; `run_v3_cell_draw`). With `clean_include_v3_boot` also the II.8g bootstrap draws (`run_v3_boot_draw`). That is off by default: II.8g already regenerated the default policies, hospital data and benchmarks of its draws without reading the II.6 cache, and compared its replicate −1 values with the II.6 rows (`x_boot_point_check`). |
# | `sample` (lite profile and smoke mode) | the first `clean_sample_draws` draws of each of these cells |
#
# Part I (GPU training of the critics, policy networks and diffusion models) is out of scope. Its clean-training
# command is a fresh `ROOT` or `FORCE_RETRAIN` (Section 1), and its artifact replay is checked in Section 32. Only the
# CPU-bound exact computation of Part II is rerun here.
#
# **Steps.**
# 1. The rerun needs `ALLOW_TRAINING = True` (asserted in `x_v4_clean_run`). In artifact-replay mode
#    (`ALLOW_TRAINING = False`) every draw would have to be computed, so the rerun is not started and the section
#    records `not run`.
# 2. Create a new directory `XV4CFG.clean_cache_dir/run-<UTC stamp>-<pid>` on the Colab local disk. It must not be
#    inside `ROOT` and must not exist yet (asserted). Rebind the global `XCACHE` to `XCache(<that directory>)` inside
#    `try / finally` and restore it afterwards. Every II.6 / II.8 function reads `XCACHE` at call time, so the rerun
#    reads and writes only the clean cache and the main cache is never written. The in-memory memo of reference optima
#    (`_XREF_MEMO`) holds results, so it is emptied for the run and restored afterwards. The deterministic construction
#    caches are kept: the exactly enumerated Gumbel systems with their physician policies and true optimal actions,
#    the Garnet systems and the diffusion schedules. They are deterministic inputs of a draw, not results: the systems
#    and the physicians enter the draw identity through the instance hashes, and the true optimal actions enter only a
#    construction statistic of the draw meta.
# 3. Per draw, rerun the draw. It must report `cache = computed`, since the clean cache starts empty (asserted). A
#    rerun that raises (an invariant of the draw fails, say) is recorded as a failing draw with the message and has no
#    rerun rows; the check goes on with the next draw. Then take the cached table of the same draw: its rows in
#    `XRESULTS`, `XRESULTS_V3` or `XV3_BOOT_RESULTS` (the tables read back from the main cache in II.6a, II.8f and
#    II.8g). If those lack the draw, read the main-cache entry under the identity of the regenerated draw
#    (`x_v4_instance` and the draw identity); if that read raises, the draw has no cached table (recorded with the
#    message). Before the rerun, a self-check reads draw 0 of the first cell of each section both ways; the two tables
#    must be identical (reported as `reference_selfcheck`, not asserted).
# 4. **Rows.** Key on (cell, draw, policy, target), plus the replicate for II.8g. Count the rows missing on either side
#    and the status mismatches. For every numeric column, report the max |Δ| over the matched rows and the number of
#    rows in which exactly one side is NaN. Count the rows with |Δ| > `clean_tol`·max(1, |cached|) in any column.
#    Compare the label columns (method, hp, solver, converged, error, …) as text, and record whether each row is
#    exactly equal.
# 5. **Certified counts.** On both tables, compute the certified pairs of `x_v4_pair_claims` per draw, over the default
#    non-reference policies that are `ok` under both rules. Count NHB (`cert_nhb`) and NHB-free (`cert_free`) pairs on
#    the true rows (`true_rows_terminate` / `true_rows_mean`, p from `true_rows_mean`) and on the benchmark
#    (`bench_terminate` / `bench_mean`, p from `bench_mean`), with v_max − v_min from `XV3_VRANGE`. II.8g tables are
#    scored per replicate on the benchmark rules: replicate −1 as `bench` and the B replicates pooled as `boot_bench`.
# 6. Delete the clean cache directory (`shutil.rmtree`), also after a failure, unless the environment variable
#    `SEPSIS_KEEP_CLEAN_CACHE=1` is set.
#
# **Pass criterion.** It is printed and stored as `XV4_CLEAN["pass"]` with its four parts. It is never asserted here;
# the Section 34 gate uses it. The criterion: every |Δ| ≤ `clean_tol`·max(1, |cached|) with no NaN-pattern change;
# identical statuses; identical certified counts; and no row missing on either side, so every draw must have a cached
# table and a rerun that did not raise. Label columns (solver, converged, error text, …) are compared but do not enter
# the criterion. Bitwise equality, with every value and label identical, is reported separately as
# `bitwise_identical`. Another BLAS or CPU can change the last bits of a linear solve without changing any reported
# number.
#
# | Output | Content |
# |---|---|
# | `XV4_CLEAN` · `xenv/v4_clean_check.json` | status, code hash, scope and configuration, the pass flag and its four criteria, `bitwise_identical`, pooled checks (draws, draws whose rerun raised, rows, missing rows, status and label mismatches, max \|Δ\| per column, rows above tolerance, certified-count differences), the failing draws, the reference self-check, per-section driver summaries, `elapsed_s` and `draws_complete` |
# | `XV4_CLEAN_CELLS` · `xenv/v4_clean_diffs.csv` | per section × cell and pooled (`all`): draws (passing, exact, without a cached table, rerun raised), rows, missing rows, status mismatches, max \|Δ\| and NaN changes per numeric column, rows above tolerance, label mismatches, certified counts of both tables and their differences, rerun time |
# | `XV4_CLEAN_DRAWS` | the same per draw (in memory only) |
#
# **Expected cost on Colab (CPU-bound).** These are estimates, not measurements of this section. A rerun costs what
# computing its draws cost, plus well under a second per draw to write and read back the clean cache entry and to
# compare the two tables. The compute times recorded when the II.6 draws were computed (`timing.total_s` in
# `xenv_draw_meta.csv`) sum to about 107 min for the 500 draws: about 50 s per ICU-Sepsis draw (42 min), 14 s per
# Gumbel draw (48 min) and 4 s per Garnet draw (17 min). The 27.5 min of the II.6 driver in the v2 session
# (`xenv_driver_summary.json`) was the reload of cached draws, not their computation. The 180 II.8 draws take about
# 35–40 min (II.8f). Paper + full, scope `all`: about 2.5 h in all; the optional II.8g bootstrap adds 20–25 min. Lite
# (scope `sample`, 3 draws of each of the 16 II.6 and 6 II.8 cells): about 15 min. Smoke: 1–2 min. The self-check adds
# one instance rebuild per section (seconds). Before the rerun starts, the cell prints the expected time from the
# compute times recorded in the cached draw meta (`xenv_draw_meta.csv`, `v3_draw_meta.csv`, `v3_bootstrap_meta.csv`),
# and an ETA after every cell and section. The clean cache holds a few tens of MB on the local disk. The rerun is not
# resumable: after a disconnect it starts again in a new directory.

# %%
# ============================================================
# II.9g Clean rerun of Part II into an empty cache
# ============================================================
EXECLOG.mark("II.9 Second-revision experiments: clean rerun")
import tempfile

XV4_CLEAN_RESULT_KEYS = ("cell", "draw", "policy", "target")
XV4_CLEAN_BOOT_KEYS = ("cell", "draw", "policy", "replicate", "target")
XV4_CLEAN_BOOT_NUMERIC = ("return", "prob_any_violation", "expected_violations", "avg_episode_length")
# (name, lower rule, upper rule) of the certified-count comparison; p is read from the upper (mean-rule) target
XV4_CLEAN_RULES = (("true_rows", "true_rows_terminate", "true_rows_mean"), ("bench", "bench_terminate", "bench_mean"))
XV4_CLEAN_KEEP_ENV = "SEPSIS_KEEP_CLEAN_CACHE"     # "1" keeps the clean cache directory after the run
assert set(XV4_CLEAN_RESULT_KEYS) <= set(XRESULT_COLUMNS)
assert set(XV4_CLEAN_BOOT_KEYS) | set(XV4_CLEAN_BOOT_NUMERIC) <= set(XV3_BOOT_COLUMNS)


def x_v4_clean_root() -> Path:
    """A new directory for the clean cache on local disk: XV4CFG.clean_cache_dir / run-<UTC stamp>-<pid> (outside
    Colab, where /content does not exist, the same name under the local temporary directory); never inside ROOT."""
    base = Path(XV4CFG.clean_cache_dir)
    if not IN_COLAB and not base.parent.exists():
        base = Path(tempfile.gettempdir()) / base.name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (base / f"run-{stamp}-{os.getpid()}").resolve()
    run_root = Path(ROOT).resolve()
    assert root != run_root and run_root not in root.parents, f"{root}: the clean cache must not be inside ROOT"
    assert not root.exists(), f"{root} exists already: the clean cache must start empty"
    return root


def x_v4_clean_jobs() -> Tuple[List[dict], List[str]]:
    """The reruns of the clean check, one job per section (II.6, II.8, optionally II.8g): runner, cache-key prefix,
    identity function, the enabled cells with the draws in scope, the cached reference rows with their (cell, draw)
    index and the recorded draw meta. Returns the jobs and the cells left out (with the reason)."""
    scope = str(XV4CFG.clean_scope)
    assert scope in ("all", "sample"), f"XV4CFG.clean_scope must be 'all' or 'sample', not {scope!r}"
    g = globals()
    v3_cells = list(g.get("X_V3_CELLS") or [])
    plan = [("II.6", [s for s in X_CELLS if EXPERIMENTS[f"xenv_{s['family']}"]], run_cell_draw, "", x_v4_draw_identity,
             "XRESULTS", "XDRAW_META", XV4_CLEAN_RESULT_KEYS, tuple(_XMETRIC_SOURCE), XRESULT_COLUMNS)]
    skipped = [f"II.6 {s['cell']} (family switch off)" for s in X_CELLS if not EXPERIMENTS[f"xenv_{s['family']}"]]
    if XV4CFG.clean_include_v3:
        plan.append(("II.8", v3_cells, run_v3_cell_draw, "v3/", x_v4_draw_identity, "XRESULTS_V3", "XDRAW_META_V3",
                     XV4_CLEAN_RESULT_KEYS, tuple(_XMETRIC_SOURCE), XRESULT_COLUMNS))
    else:
        skipped += [f"II.8 {s['cell']} (clean_include_v3 off)" for s in v3_cells]
    if XV4CFG.clean_include_v3_boot:
        boot_cells = list(g.get("XV3_BOOT_CELLS") or [])
        if not boot_cells:
            skipped.append("II.8g bootstrap (xenv_v3_bootstrap off or no bootstrap cell enabled)")
        plan.append(("II.8g", boot_cells, run_v3_boot_draw, "v3boot/", x_boot_identity, "XV3_BOOT_RESULTS",
                     "XV3_BOOT_META", XV4_CLEAN_BOOT_KEYS, XV4_CLEAN_BOOT_NUMERIC, XV3_BOOT_COLUMNS))

    def n_draws(spec: dict) -> int:
        n = x_v4_draws(spec)
        return n if scope == "all" else min(n, int(XV4CFG.clean_sample_draws))

    jobs = []
    for section, specs, runner, prefix, ident, ref_name, meta_name, keys, numeric, columns in plan:
        ref = g.get(ref_name)
        ref = pd.DataFrame(columns=list(columns)) if ref is None else ref
        assert set(columns) <= set(ref.columns), f"{ref_name} lacks columns {sorted(set(columns) - set(ref.columns))}"
        index = ({} if not len(ref) else ref.groupby([ref["cell"].astype(str),
                                                      pd.to_numeric(ref["draw"]).astype(np.int64)], sort=False).indices)
        meta = g.get(meta_name)
        jobs.append({"section": section, "runner": runner, "prefix": prefix, "identity": ident, "keys": tuple(keys),
                     "numeric": tuple(numeric), "columns": list(columns), "reference_name": ref_name,
                     "reference": ref, "reference_index": index, "meta": pd.DataFrame() if meta is None else meta,
                     "specs": {str(s["cell"]): s for s in specs},
                     # "draws" enters no identity (x_draw_identity drops it); the runner gets the original spec
                     "cells": [{**s, "draws": n_draws(s)} for s in specs]})
    return jobs, skipped


def x_v4_clean_estimate(jobs: Sequence[dict]) -> Tuple[Optional[float], int]:
    """Expected rerun seconds of the draws in scope, from the per-cell mean of the compute times recorded in the cached
    draw meta ('timing.total_s', kept from the session that computed each draw), and the draws without a record."""
    total, unknown, known = 0.0, 0, False
    for job in jobs:
        m = job["meta"]
        per = (pd.to_numeric(m["timing.total_s"], errors="coerce").groupby(m["cell"].astype(str)).mean()
               if len(m) and {"cell", "timing.total_s"} <= set(m.columns) else pd.Series(dtype=np.float64))
        for s in job["cells"]:
            t = float(per.get(str(s["cell"]), np.nan))
            if np.isfinite(t):
                total, known = total + t * int(s["draws"]), True
            else:
                unknown += int(s["draws"])
    return (total if known else None), unknown


def x_v4_clean_reference(job: dict, spec: dict, draw: int, main: XCache,
                         memory: bool = True) -> Tuple[Optional[pd.DataFrame], str]:
    """The cached table of one draw: its rows in the in-memory results (read back from the main cache in II.6a / II.8f
    / II.8g), else the main-cache entry under the identity of the regenerated draw; (None, 'missing') if neither.
    memory=False reads the main cache directly (the self-check)."""
    pos = job["reference_index"].get((str(spec["cell"]), int(draw))) if memory else None
    if pos is not None:
        return job["reference"].iloc[pos][job["columns"]].reset_index(drop=True), "memory"
    inst = x_v4_instance(spec, draw)
    hit = main.get(job["prefix"] + x_cache_key(spec["cell"], draw), job["identity"](spec, inst))
    del inst
    return (None, "missing") if hit is None else (hit[job["columns"]], "main_cache")


def _x_v4_clean_keyed(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Copy with normalised key columns (integers for draw / replicate, text otherwise); keys must be unique."""
    out = df.copy()
    for k in keys:
        out[k] = pd.to_numeric(out[k]).astype(np.int64) if k in ("draw", "replicate") else out[k].astype(str)
    dup = out.duplicated(list(keys))
    assert not dup.any(), f"{int(dup.sum())} duplicated {tuple(keys)} rows"
    return out


def _x_v4_clean_text(s: pd.Series) -> np.ndarray:
    """Label values as text, missing values as '' (a CSV round trip turns empty cells into NaN)."""
    return s.astype(object).where(s.notna(), "").map(str).to_numpy()


def x_v4_clean_compare(new: Optional[pd.DataFrame], ref: Optional[pd.DataFrame], keys: Sequence[str],
                       numeric: Sequence[str], tol: float) -> Dict[str, Any]:
    """Row-by-row comparison of a rerun table with the cached table of the same draw (see markdown, step 4). A
    missing side (None: no cached table, or a rerun that raised) counts every row of the other side as missing."""
    keys, numeric = list(keys), list(numeric)
    out: Dict[str, Any] = {"rows_rerun": 0 if new is None else int(len(new)),
                           "rows_reference": 0 if ref is None else int(len(ref))}
    if new is None or ref is None:
        out.update({"rows_matched": 0, "missing_in_rerun": out["rows_reference"],
                    "missing_in_reference": out["rows_rerun"],
                    "status_mismatches": 0, **{f"max_abs_diff[{c}]": np.nan for c in numeric},
                    **{f"nan_mismatches[{c}]": 0 for c in numeric}, "rows_above_tol": 0, "rows_exact": 0,
                    "label_mismatches": 0, "label_mismatch_columns": ""})
        return out
    assert list(new.columns) == list(ref.columns), "the rerun and the cached table have different columns"
    labels = [c for c in new.columns if c not in keys and c not in numeric and c != "status"]
    m = _x_v4_clean_keyed(new, keys).merge(_x_v4_clean_keyed(ref, keys), on=keys, how="outer",
                                           suffixes=("_rerun", "_ref"), indicator=True, validate="one_to_one")
    b = m[(m["_merge"] == "both").to_numpy()]
    exact = _x_v4_clean_text(b["status_rerun"]) == _x_v4_clean_text(b["status_ref"])
    out.update({"rows_matched": int(len(b)), "missing_in_rerun": int((m["_merge"] == "right_only").sum()),
                "missing_in_reference": int((m["_merge"] == "left_only").sum()),
                "status_mismatches": int((~exact).sum())})
    above = np.zeros(len(b), dtype=bool)
    for c in numeric:
        x = pd.to_numeric(b[f"{c}_rerun"], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(b[f"{c}_ref"], errors="coerce").to_numpy(dtype=np.float64)
        nx, ny = np.isnan(x), np.isnan(y)
        d = np.where(nx | ny, 0.0, np.abs(x - y))
        out[f"max_abs_diff[{c}]"] = float(d.max(initial=0.0))
        out[f"nan_mismatches[{c}]"] = int((nx != ny).sum())
        scale = np.maximum(1.0, np.where(ny, 0.0, np.abs(y)))          # tolerance absolute below 1, relative above
        above |= (d > tol * scale) | (nx != ny)
        exact &= (x == y) | (nx & ny)
    bad, n_bad = [], 0
    for c in labels:
        neq = _x_v4_clean_text(b[f"{c}_rerun"]) != _x_v4_clean_text(b[f"{c}_ref"])
        if neq.any():
            bad.append(c)
            n_bad += int(neq.sum())
        exact &= ~neq
    out.update({"rows_above_tol": int(above.sum()), "rows_exact": int(exact.sum()), "label_mismatches": n_bad,
                "label_mismatch_columns": ";".join(bad)})
    return out


def x_v4_clean_certified(df: pd.DataFrame, family: str) -> Dict[str, int]:
    """Certified pairs of one draw's table (x_v4_pair_claims, type 'all'): NHB (cert_nhb) and NHB-free (cert_free)
    counts over the default non-reference policies that are ok under both rules, on the true rows and on the benchmark.
    Bootstrap tables are scored per replicate: replicate -1 under the rule name, the replicates pooled under
    'boot_<name>'. Keys '<name>.policies', '<name>.cert_nhb', '<name>.cert_free'."""
    v_lo, v_hi = XV3_VRANGE[family]
    d = df[xs_as_bool(df["is_default"]).to_numpy()]
    if "is_reference" in d.columns:
        d = d[~xs_as_bool(d["is_reference"]).to_numpy()]
    d = d.assign(target=d["target"].astype(str), status=d["status"].astype(str))
    rep = (pd.to_numeric(d["replicate"]).astype(np.int64).to_numpy() if "replicate" in d.columns
           else np.full(len(d), XV3_BOOT_POINT, dtype=np.int64))
    out: Dict[str, int] = {}
    for r in np.unique(rep):
        dr = d[rep == r]
        have = set(dr["target"])
        for name, lo_t, up_t in XV4_CLEAN_RULES:
            if not {lo_t, up_t} <= have:
                continue
            key = name if int(r) == XV3_BOOT_POINT else f"boot_{name}"
            sub = dr[dr["target"].isin([lo_t, up_t]) & (dr["status"] == "ok")].set_index(["policy", "target"])
            J = sub["return"].astype(np.float64).unstack("target").reindex(columns=[lo_t, up_t]).dropna()
            p = sub["prob_any_violation"].astype(np.float64).unstack("target").reindex(index=J.index, columns=[up_t])
            claims = {"cert_nhb_claims": 0, "cert_free_claims": 0}
            if len(J):
                claims = next(c for c in x_v4_pair_claims(J[lo_t].to_numpy(np.float64), J[up_t].to_numpy(np.float64),
                                                          p[up_t].to_numpy(np.float64), v_hi - v_lo)
                              if c["type"] == "all")
            for k, v in (("policies", len(J)), ("cert_nhb", claims["cert_nhb_claims"]),
                         ("cert_free", claims["cert_free_claims"])):
                out[f"{key}.{k}"] = out.get(f"{key}.{k}", 0) + int(v)
    return out


def x_v4_clean_selfcheck(job: dict, main: XCache) -> Dict[str, Any]:
    """Draw 0 of the first cell of a section read both ways: the in-memory table and the main-cache entry under the
    identity of the regenerated draw. They are the same table read back, so they must be identical (reported). This
    also exercises the main-cache path of x_v4_clean_reference in every run mode."""
    spec = job["specs"][str(job["cells"][0]["cell"])]
    out: Dict[str, Any] = {"cell": str(spec["cell"]), "draw": 0}
    if int(job["cells"][0]["draws"]) < 1 or (str(spec["cell"]), 0) not in job["reference_index"]:
        return {**out, "status": "not checked (draw 0 has no in-memory table)", "match": None}
    mem, _ = x_v4_clean_reference(job, spec, 0, main)
    try:
        hit, _ = x_v4_clean_reference(job, spec, 0, main, memory=False)
    except MemoryError:
        raise
    except Exception as exc:          # the rerun of the same draw meets the same error and records it
        return {**out, "status": f"failed: {type(exc).__name__}: {str(exc)[:200]}", "match": False}
    if hit is None:
        return {**out, "status": "no main-cache entry under the regenerated identity", "match": False}
    cmp = x_v4_clean_compare(hit, mem, job["keys"], job["numeric"], 0.0)
    match = cmp["rows_exact"] == cmp["rows_matched"] == cmp["rows_rerun"] == cmp["rows_reference"]
    return {**out, "status": "checked", "match": bool(match),
            **{k: cmp[k] for k in ("rows_reference", "rows_matched", "rows_exact", "status_mismatches",
                                   "rows_above_tol", "label_mismatch_columns")}}


def x_v4_clean_draw(job: dict, spec: dict, draw: int, main: XCache) -> Tuple[pd.DataFrame, dict]:
    """Rerun one draw into the clean cache (the rebound global XCACHE) and compare it with its cached table.
    A rerun that raises is recorded as a failing draw without rerun rows (see markdown, step 3).
    Returns one comparison row and a meta for x_v4_run_draws."""
    cell, draw = str(spec["cell"]), int(draw)
    orig = job["specs"][cell]
    assert XCACHE is not main, "the clean rerun must run with XCACHE rebound to the clean cache"
    t0 = time.perf_counter()
    new, error = None, ""
    try:
        new, meta = job["runner"](orig, draw)
    except MemoryError:
        raise
    except Exception as exc:          # a clean-check outcome, not a crash: recorded, and the next draw runs
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
        print(f"  WARNING: II.9g {job['section']} {cell} draw {draw}: the rerun raised {error}")
    if new is not None:
        assert meta["cache"] == "computed", \
            f"{cell} draw {draw}: {meta['cache']!r} although the clean cache started empty"
        assert list(new.columns) == job["columns"], f"{cell} draw {draw}: rerun columns differ"
    t1 = time.perf_counter()
    ref, source, ref_error = None, "missing", ""
    try:
        ref, source = x_v4_clean_reference(job, orig, draw, main)
    except MemoryError:
        raise
    except Exception as exc:          # the main-cache path rebuilds the instance; a failure means no cached table
        ref_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        print(f"  WARNING: II.9g {job['section']} {cell} draw {draw}: the cached table could not be read: {ref_error}")
    t2 = time.perf_counter()
    cmp = x_v4_clean_compare(new, ref, job["keys"], job["numeric"], float(XV4CFG.clean_tol))
    cn = {} if new is None else x_v4_clean_certified(new, orig["family"])
    cr = {} if ref is None else x_v4_clean_certified(ref, orig["family"])
    cert = {**{f"{k}.rerun": v for k, v in cn.items()}, **{f"{k}.reference": v for k, v in cr.items()}}
    cdiff = np.nan
    if new is not None and ref is not None:
        cdiff = float(sum(abs(cn.get(k, 0) - cr.get(k, 0)) for k in set(cn) | set(cr) if not k.endswith(".policies")))
    reasons = [f"the rerun raised {error}"] if error else []
    if ref is None:
        reasons.append("no cached table" + (f" ({ref_error})" if ref_error else ""))
    elif new is not None:
        reasons += [f"{cmp[k]} {txt}" for k, txt in (("missing_in_rerun", "cached rows missing in the rerun"),
                                                     ("missing_in_reference", "rerun rows missing in the cache"),
                                                     ("status_mismatches", "status mismatches"),
                                                     ("rows_above_tol", "rows with |diff| > clean_tol or a NaN change"))
                    if cmp[k]]
        if cdiff != 0:
            reasons.append(f"certified counts differ by {cdiff:g}")
    passes = not reasons
    exact = passes and cmp["rows_exact"] == cmp["rows_matched"]
    row = {"section": job["section"], "cell": cell, "family": orig["family"], "draw": draw, "reference": source,
           "passes": passes, "exact": exact, "failure": "; ".join(reasons), "rerun_error": error,
           "reference_error": ref_error, **cmp,
           "certified_count_differences": cdiff, **cert, "rerun_s": t1 - t0, "reference_s": t2 - t1,
           "compare_s": time.perf_counter() - t2}
    del new, ref
    return pd.DataFrame([row]), {"cache": "computed", "section": job["section"], "cell": cell, "draw": draw,
                                 "reference": source, "passes": passes, "rerun_error": error,
                                 "timing": {"rerun_s": t1 - t0, "reference_s": t2 - t1, "compare_s": row["compare_s"],
                                            "total_s": time.perf_counter() - t0}}


def x_v4_clean_run(jobs: List[dict], root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Rerun the draws of every job into the empty cache `root` and compare each with its cached table.

    The global XCACHE is rebound to XCache(root) inside try / finally and restored afterwards; the memo of reference
    optima is emptied for the run and restored. The clean cache is deleted afterwards (also after a failure) unless
    SEPSIS_KEEP_CLEAN_CACHE=1. Returns the per-draw comparison and a run report."""
    global XCACHE
    assert ALLOW_TRAINING, "the clean rerun computes every draw: it needs ALLOW_TRAINING = True"
    main, root = XCACHE, Path(root)
    assert not root.exists(), f"{root}: the clean cache must start empty"
    assert root.resolve() != Path(main.root).resolve() and Path(main.root).resolve() not in root.resolve().parents, \
        f"{root}: the clean cache must not be inside the main cache {main.root}"
    keep = os.environ.get(XV4_CLEAN_KEEP_ENV, "").strip() == "1"
    todo = [j for j in jobs if j["cells"]]
    expected, unknown = x_v4_clean_estimate(todo)
    print(f"II.9g clean rerun ({XV4CFG.clean_scope} scope): {sum(len(j['cells']) for j in todo)} cells, "
          f"{sum(int(s['draws']) for j in todo for s in j['cells'])} draws, into {root}")
    print(f"  expected time from the compute times recorded in the cached draw meta: {_x_fmt_s(expected)}"
          + (f" (+ {unknown} draws without a recorded time)" if unknown else ""))
    selfcheck = {j["section"]: x_v4_clean_selfcheck(j, main) for j in todo}     # reads only (main cache, memory)
    if any(v["match"] is False for v in selfcheck.values()):
        print("NOTE: II.9g in-memory and main-cache tables differ (reference self-check):",
              {k: v for k, v in selfcheck.items() if v["match"] is False})
    frames: List[pd.DataFrame] = []
    sections: Dict[str, Any] = {}
    ref_memo = dict(_XREF_MEMO)
    t_all = time.perf_counter()
    obs_s, exp_s = 0.0, 0.0
    clean: Optional[XCache] = None
    try:
        clean = XCache(root)
        XCACHE = clean
        _XREF_MEMO.clear()
        for ji, job in enumerate(todo, 1):
            fr, _, summ = x_v4_run_draws(job["cells"], lambda s, d, job=job: x_v4_clean_draw(job, s, d, main),
                                         f"II.9g {job['section']}")
            frames += fr
            sections[job["section"]] = summ
            done, _ = x_v4_clean_estimate([job])
            obs_s, exp_s = obs_s + float(summ["elapsed_s"]), exp_s + (done or 0.0)
            rest, _ = x_v4_clean_estimate(todo[ji:])
            eta = None if rest is None else rest * (obs_s / exp_s if exp_s > 0 else 1.0)
            print(f"II.9g {job['section']}: {summ['draws']} draws rerun and compared in "
                  f"{_x_fmt_s(summ['elapsed_s'])} | total elapsed {_x_fmt_s(time.perf_counter() - t_all)} | "
                  f"ETA remaining sections {_x_fmt_s(eta) if ji < len(todo) else '0 s'}")
    finally:
        XCACHE = main
        _XREF_MEMO.clear()
        _XREF_MEMO.update(ref_memo)
        written = 0 if clean is None else len(clean.written)
        if keep:
            print(f"II.9g: clean cache kept at {root} ({XV4_CLEAN_KEEP_ENV}=1)")
        else:
            shutil.rmtree(root, ignore_errors=True)
            try:
                root.parent.rmdir()              # the base directory too, if no other run is kept there
            except OSError:
                pass
    assert XCACHE is main
    draws = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return draws, {"cache_dir": str(root), "cache_kept": keep, "cache_deleted": not keep and not root.exists(),
                   "cache_entries_written": written, "expected_s": expected, "draws_without_recorded_time": unknown,
                   "reference_selfcheck": selfcheck, "sections": sections, "run_s": time.perf_counter() - t_all}


def x_v4_clean_cells(draws: pd.DataFrame) -> pd.DataFrame:
    """Per (section, cell) and pooled ('all'): draws, rows, max |diff| per numeric column, NaN changes, mismatches,
    rows above tolerance, certified counts of both tables and their differences, rerun time."""
    if draws.empty:
        return pd.DataFrame()
    sums = ["rows_rerun", "rows_reference", "rows_matched", "missing_in_rerun", "missing_in_reference",
            "status_mismatches", "rows_above_tol", "rows_exact", "label_mismatches",
            *[c for c in draws.columns if c.startswith("nan_mismatches[")], "rerun_s", "reference_s", "compare_s"]
    maxes = [c for c in draws.columns if c.startswith("max_abs_diff[")]
    certs = [c for c in draws.columns if c.endswith((".rerun", ".reference"))]
    rows = []
    for (sec, cell), g in [*draws.groupby(["section", "cell"], sort=False), (("all", "all"), draws)]:
        cd = pd.to_numeric(g["certified_count_differences"], errors="coerce")
        r = {"section": sec, "cell": cell, "draws": int(len(g)), "draws_passing": int(g["passes"].astype(bool).sum()),
             "draws_exact": int(g["exact"].astype(bool).sum()),
             "draws_without_reference": int((g["reference"] == "missing").sum()),
             "draws_from_main_cache": int((g["reference"] == "main_cache").sum()),
             "draws_rerun_failed": int((g["rerun_error"].fillna("").astype(str) != "").sum())}
        r.update({c: g[c].sum() for c in sums})
        r.update({c: (float(g[c].max()) if g[c].notna().any() else np.nan) for c in maxes})
        r["certified_count_differences"] = float(cd.sum(min_count=1))
        r["draws_with_certified_count_difference"] = int((cd > 0).sum())
        r.update({c: g[c].sum(min_count=1) for c in certs})
        r["label_mismatch_columns"] = ";".join(sorted({x for v in g["label_mismatch_columns"].fillna("")
                                                        for x in str(v).split(";") if x}))
        rows.append(r)
    return pd.DataFrame(rows)


def x_v4_clean_summary(draws: pd.DataFrame, cells: pd.DataFrame, jobs: Sequence[dict], skipped: Sequence[str],
                       run: Dict[str, Any], elapsed_s: float) -> Dict[str, Any]:
    """XV4_CLEAN: pass flag and criteria, bitwise flag, pooled checks, failing draws, driver summaries, timing."""
    tol = float(XV4CFG.clean_tol)
    planned = int(sum(int(s["draws"]) for j in jobs for s in j["cells"]))
    p = cells[cells["section"] == "all"].iloc[0]
    criteria = {"every_max_abs_diff_within_tol": bool(p["rows_above_tol"] == 0),
                "statuses_identical": bool(p["status_mismatches"] == 0),
                "certified_counts_identical": bool(p["certified_count_differences"] == 0),
                "no_missing_rows": bool(p["missing_in_rerun"] == 0 and p["missing_in_reference"] == 0
                                        and p["draws_without_reference"] == 0)}
    maxes = {c[len("max_abs_diff["):-1]: float(p[c]) for c in cells.columns
             if c.startswith("max_abs_diff[") and pd.notna(p[c])}
    checks = {k: (p[k].item() if hasattr(p[k], "item") else p[k]) for k in
              ("draws", "draws_passing", "draws_exact", "draws_without_reference", "draws_from_main_cache",
               "draws_rerun_failed", "rows_rerun", "rows_reference", "rows_matched", "missing_in_rerun",
               "missing_in_reference", "status_mismatches", "rows_above_tol", "rows_exact", "label_mismatches",
               "label_mismatch_columns", "certified_count_differences", "draws_with_certified_count_difference")}
    selfcheck = {k: v["match"] for k, v in run.get("reference_selfcheck", {}).items()}
    fail = draws.loc[~draws["passes"].astype(bool), ["section", "cell", "draw", "failure"]]
    return {"status": "run", "code_hash": XV4_CLEAN_CODE_HASH,
            "config": {k: v for k, v in asdict(XV4CFG).items() if k.startswith("clean_") or k == "max_draws"},
            "tolerance": tol, "pass": bool(all(criteria.values()) and len(draws) > 0),
            "bitwise_identical": bool(criteria["no_missing_rows"] and p["draws_exact"] == p["draws"]),
            "criteria": criteria, "checks": {**checks, "draws_planned": planned, "max_abs_diff": maxes,
                                             "reference_selfcheck_match": selfcheck},
            "cells": {f"{r['section']}|{r['cell']}": bool(r["draws_passing"] == r["draws"])
                      for _, r in cells[cells["section"] != "all"].iterrows()},
            "failing_draws": fail.head(50).to_dict("records"), "failing_draws_total": int(len(fail)),
            "cells_skipped": list(skipped), **run, "elapsed_s": float(elapsed_s),
            "draws_complete": bool(len(draws) == planned and planned > 0),
            "files": {"check": "xenv/v4_clean_check.json", "diffs": "xenv/v4_clean_diffs.csv"}}


XV4_CLEAN_CODE = [x_v4_clean_root, x_v4_clean_jobs, x_v4_clean_estimate, x_v4_clean_reference, _x_v4_clean_keyed,
                  _x_v4_clean_text, x_v4_clean_compare, x_v4_clean_certified, x_v4_clean_selfcheck, x_v4_clean_draw,
                  x_v4_clean_run, x_v4_clean_cells, x_v4_clean_summary]
XV4_CLEAN_CODE_HASH = obj_hash({"common": XV4_COMMON_CODE_HASH, "code": code_hash(*XV4_CLEAN_CODE),
                                "constants": {"result_keys": XV4_CLEAN_RESULT_KEYS, "boot_keys": XV4_CLEAN_BOOT_KEYS,
                                              "boot_numeric": XV4_CLEAN_BOOT_NUMERIC, "rules": XV4_CLEAN_RULES,
                                              "v_range": XV3_VRANGE, "boot_point": XV3_BOOT_POINT,
                                              "keep_env": XV4_CLEAN_KEEP_ENV, "result_columns": XRESULT_COLUMNS,
                                              "metrics": list(_XMETRIC_SOURCE), "boot_columns": XV3_BOOT_COLUMNS}})

XV4_CLEAN_DRAWS = pd.DataFrame()
XV4_CLEAN_CELLS = pd.DataFrame()
XV4_CLEAN: Dict[str, Any] = {"status": "not run (xenv_v4_clean_check off)", "code_hash": XV4_CLEAN_CODE_HASH,
                             "pass": None, "bitwise_identical": None, "criteria": {}, "checks": {}, "elapsed_s": 0.0,
                             "draws_complete": None}
if EXPERIMENTS["xenv_v4_clean_check"] and not ALLOW_TRAINING:
    XV4_CLEAN["status"] = "not run (ALLOW_TRAINING is False: the clean rerun computes every draw)"
    print("NOTE: II.9g clean rerun skipped: ALLOW_TRAINING is False (artifact-replay mode).")
elif EXPERIMENTS["xenv_v4_clean_check"]:
    _t0 = time.perf_counter()
    _jobs, _skipped = x_v4_clean_jobs()
    if _skipped:
        print("NOTE: II.9g leaves out:", _skipped)
    if sum(int(s["draws"]) for j in _jobs for s in j["cells"]) > 0:
        XV4_CLEAN_DRAWS, _run = x_v4_clean_run(_jobs, x_v4_clean_root())
        XV4_CLEAN_CELLS = x_v4_clean_cells(XV4_CLEAN_DRAWS)
        XV4_CLEAN = x_v4_clean_summary(XV4_CLEAN_DRAWS, XV4_CLEAN_CELLS, _jobs, _skipped, _run,
                                       time.perf_counter() - _t0)
        XV4_CLEAN_CELLS.to_csv(XENV_TABLE_DIR / "v4_clean_diffs.csv", index=False)
        _cols = [c for c in ("section", "cell", "draws", "draws_passing", "draws_exact", "draws_without_reference",
                             "draws_rerun_failed", "rows_rerun", "missing_in_rerun", "missing_in_reference",
                             "status_mismatches",
                             "rows_above_tol", "max_abs_diff[return]", "max_abs_diff[avg_episode_length]",
                             "certified_count_differences", "rerun_s") if c in XV4_CLEAN_CELLS.columns]
        display(XV4_CLEAN_CELLS[_cols])
        del _run, _cols
    else:
        XV4_CLEAN.update(status="not run (no Part II cell is enabled for the clean rerun)", cells_skipped=_skipped)
    del _jobs, _skipped, _t0
    gc.collect()
save_json(XV4_CLEAN, XENV_TABLE_DIR / "v4_clean_check.json")
print(json.dumps({k: v for k, v in XV4_CLEAN.items() if k not in ("failing_draws", "sections", "cells", "config")},
                 indent=1, default=_json_default))
if XV4_CLEAN.get("pass") is not None:
    print(f"II.9g pass criterion: every max |diff| <= clean_tol = {XV4CFG.clean_tol:g} with no NaN change, identical "
          f"statuses, identical certified counts (NHB and NHB-free; true rows and bench) and no row missing on either "
          f"side (every draw reran and has a cached table) -> {'PASS' if XV4_CLEAN['pass'] else 'FAIL'} "
          f"{XV4_CLEAN['criteria']}; bitwise identical (reported "
          f"separately): {XV4_CLEAN['bitwise_identical']}")
    if not XV4_CLEAN["pass"]:
        print(f"  {XV4_CLEAN['failing_draws_total']} failing draws, e.g.", XV4_CLEAN["failing_draws"][:5])
