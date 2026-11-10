# %% [markdown]
# ## II.9 Further experiments: estimation-aware certificates, sensitivity and a clean rerun
#
# This section adds further robustness experiments. Sections II.1–II.8 are not changed, so every cached draw keeps
# its identity. The new experiments use the exact simulators and the stored designs of Sections II.3–II.8.
#
# | Switch | Subsection | What it adds |
# |---|---|---|
# | `xenv_v4_revision_robust` | II.9b | certificates without NHB, pair types, distinct policies, abstention at matched coverage and draw-level counts for every Part II cell, the additional cells included; the coverage of the II.8g bootstrap intervals over **all** default policies |
# | `xenv_v4_confidence` | II.9c | confidence-set certificates: L1 confidence sets for the true rows and d0 built from the raw hospital counts, robust bounds on J_true for every policy, certificates valid on one event for all policies at once |
# | `xenv_v4_bootstrap` | II.9d | bootstrap with B = 200 that re-estimates the whole construction (mask, decision states, dead-state removal, d0) in every replicate, beside the fixed-mask bootstrap of II.8g; coverage over all default policies |
# | `xenv_v4_sensitivity` | II.9e | new cells: Gumbel timeout hazard η ∈ {0.02, 0.10}; the alternative dead-state rule (`argmax`) on the hospital data of the original cells; softer physicians (wider support); other state abstractions (glucose hidden, k-means clusters) |
# | `xenv_v4_lift` | II.9f | the lift of benchmark policies to the true state space: physician and uniform lifts beside the default, and the range of J_true over all lifts |
# | `xenv_v4_clean_check` | II.9g | a clean rerun of the Part II draws (and the II.8 cells) into an empty cache, compared row by row with the cached results |
#
# II.9h collects the outputs, writes `xenv/v4_overview.json` and the figures, and states what failed.
#
# **Cached draws stay valid.** New behaviour lives in new functions. Each subsection has its own code hash, which folds
# in `XENV_CODE_HASH` (and `XENV_V3_CODE_HASH` where II.8 functions are used), and its own cache keys (`v4cs/`,
# `v4boot/`, `v4sens/`, `v4lift/`). The clean check writes to a separate, empty cache directory.
#
# **Shared helpers (this cell).** `XV4Config` holds every setting; `x_v4_cells` resolves cell names against `X_CELLS`
# and `X_V3_CELLS` (a configured cell whose switch is off is skipped and listed); `x_v4_instance` builds the II.6 or
# II.8 instance of a draw; `x_v4_default_records` regenerates its default policies (Part II learners are deterministic
# functions of their data, as II.8g checked); `x_v4_rebuild_benchmark` regenerates the hospital episodes and the
# benchmark and asserts that the benchmark equals the instance's; `x_v4_pair_claims` scores the certificates of one
# policy set (with NHB, without NHB, NHB-only, testable claims, pair types and matched-coverage margins);
# `x_v4_run_draws` runs a per-draw function over cells with progress and ETA.

# %%
# ============================================================
# II.9a Second-revision experiments: configuration and shared helpers
# ============================================================
EXECLOG.mark("II.9 Second-revision experiments: setup")

XV4_SWITCHES = ("xenv_v4_revision_robust", "xenv_v4_confidence", "xenv_v4_bootstrap", "xenv_v4_sensitivity",
                "xenv_v4_lift", "xenv_v4_clean_check")
assert all(k in EXPERIMENTS for k in XV4_SWITCHES), [k for k in XV4_SWITCHES if k not in EXPERIMENTS]
XV4_ANY_ON = any(bool(EXPERIMENTS[k]) for k in XV4_SWITCHES)
XV4_TOL = 1e-9                 # value tolerance of comparisons (as XV3_TOL)
XV4_PTOL = 1e-12               # a policy "can leave the support" iff P(any unsupported decision) > XV4_PTOL

XV4_GUMBEL_BEST = ("gumbel|best|n2000|tau5|eps0.05", "gumbel|best|n2000|tau20|eps0.05", "gumbel|best|n10000|tau5|eps0.05",
                   "gumbel|best|n10000|tau20|eps0.05", "gumbel|best|n10000|tau20|eps0.2",
                   "gumbel|best|n10000|tau20|eps0.05|phys-withhold-vaso")
XV4_GARNET_ALL = tuple(f"garnet|h{h}|eps{e}" for h in (-0.5, 0.0, 0.5, 1.0) for e in (0.05, 0.3))


@dataclass
class XV4Config:
    # II.9c confidence-set certificates
    cs_cells: Tuple[str, ...] = ("gumbel|best|n2000|tau5|eps0.05", "gumbel|best|n10000|tau20|eps0.05",
                                 "gumbel|best|n10000|tau20|eps0.05|phys-withhold-vaso", "garnet|h-0.5|eps0.05",
                                 "garnet|h0.0|eps0.05", "garnet|h0.5|eps0.05", "garnet|h1.0|eps0.05")
    cs_delta: float = 0.05                         # P(true rows or d0 outside the set) <= delta (union bound)
    cs_scales: Tuple[float, ...] = (1.0, 0.3, 0.1)  # radius multipliers; 1.0 is the valid set, smaller ones heuristic
    cs_tol: float = 1e-8                           # stop the monotone robust iteration at this sup-norm change
    cs_max_iter: int = 5000                        # every iterate is a valid bound, so a cap only loosens it
    # II.9d bootstrap with the construction re-estimated in every replicate
    boot_cells: Tuple[str, ...] = ("gumbel|best|n10000|tau20|eps0.05", "gumbel|best|n2000|tau5|eps0.05",
                                   "garnet|h0.0|eps0.05", "garnet|h1.0|eps0.05")
    boot_replicates: int = 200
    boot_quantiles: Tuple[float, float] = (0.025, 0.975)
    boot_variants: Tuple[str, ...] = ("fixed", "reestimated")
    # II.9e sensitivity cells: (base cell, timeout hazard) and base cells rebuilt with the 'argmax' dead-state rule
    timeout_cells: Tuple[Tuple[str, float], ...] = (("gumbel|best|n10000|tau20|eps0.05", 0.02),
                                                    ("gumbel|best|n10000|tau20|eps0.05", 0.10),
                                                    ("gumbel|best|n2000|tau5|eps0.05", 0.02),
                                                    ("gumbel|best|n2000|tau5|eps0.05", 0.10))
    deadrule_cells: Tuple[str, ...] = XV4_GUMBEL_BEST + ("garnet|h1.0|eps0.05", "garnet|h1.0|eps0.3")
    # II.9e wider physician support: the base cell with a softer physician (eps), streams named by the new cell
    support_base: str = "gumbel|best|n10000|tau20|eps0.05"
    support_eps: Tuple[float, ...] = (0.3, 0.5)
    # II.9e other state abstractions of the Gumbel benchmark (the II.8 aggregated design with another phi):
    # 'glucose' hides glucose only (diabetes visible); 'kmeansK' clusters the observation features into K cells
    aggregation_base: Tuple[int, int, float] = (10_000, 20, 0.05)       # (hospital episodes, tau, physician eps)
    aggregations: Tuple[str, ...] = ("glucose", "kmeans50", "kmeans150")
    # II.9b hidden-benefit sensitivity: fractions kappa of the largest possible excess over J_W
    kappa_grid: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
    # II.9e lift sensitivity
    lift_cells: Tuple[str, ...] = XV4_GUMBEL_BEST + XV4_GARNET_ALL
    lift_kinds: Tuple[str, ...] = ("fallback", "physician", "uniform")
    lift_bounds: bool = False                      # range of J_true over all lifts by optimal control outside V
    #                                                (about 2 min per Gumbel draw: enable for a subset only)
    # II.9f clean rerun into an empty cache
    clean_scope: str = "all"                       # "all" draws, or "sample": the first clean_sample_draws of each cell
    clean_sample_draws: int = 3
    clean_include_v3: bool = True                  # also the II.8 cells (run_v3_cell_draw)
    clean_include_v3_boot: bool = False            # II.8g already rebuilt its 210 draws without the cache
    clean_cache_dir: str = "/content/xenv_clean_cache"   # local disk, never the Drive run root
    clean_tol: float = 1e-12                       # |diff| <= clean_tol * max(1, |cached value|)
    # all subsections: at most this many draws per cell (None = every configured draw of the cell)
    max_draws: Optional[int] = None


def apply_xv4_run_mode(c: XV4Config, mode: str, profile: str) -> XV4Config:
    if profile == "lite":
        c.max_draws = 10
        c.cs_scales = (1.0, 0.3)
        c.clean_scope = "sample"
    if mode == "smoke":
        c.max_draws = 2
        c.cs_scales = (1.0, 0.3)
        c.cs_max_iter = 500
        c.boot_replicates = 5
        c.timeout_cells = c.timeout_cells[:1]
        c.support_eps = c.support_eps[:1]
        c.aggregations = c.aggregations[:1]
        c.lift_kinds = ("fallback", "physician", "uniform")
        c.lift_bounds = True                       # exercised in smoke; off by default in paper mode (cost)
        c.clean_scope, c.clean_sample_draws = "sample", 1
        c.clean_include_v3_boot = True             # seconds in smoke mode; exercises the II.8g rerun path
    return c


XV4CFG = apply_xv4_run_mode(XV4Config(), RUN_MODE, COMPUTE_PROFILE)
save_json(asdict(XV4CFG), XENV_TABLE_DIR / "v4_config.json")


# ---------- cells ----------
def x_v4_all_specs() -> Dict[str, dict]:
    """Every enabled Part II cell spec by name: II.6 cells (X_CELLS) and II.8 cells (X_V3_CELLS, with v3_kind)."""
    out = {str(s["cell"]): s for s in X_CELLS}
    for s in globals().get("X_V3_CELLS", []) or []:
        assert str(s["cell"]) not in out, s["cell"]
        out[str(s["cell"])] = s
    return out


def x_v4_cells(names: Sequence[str], families: Sequence[str] = ("gumbel", "garnet"),
               allow_aggregated: bool = False) -> Tuple[List[dict], List[str]]:
    """(specs, skipped) for configured cell names. In smoke mode, names that do not exist (the smoke designs use
    other hospital sizes) are replaced by the enabled cells of the same family and construction."""
    specs = x_v4_all_specs()
    chosen, skipped = [], []
    for name in names:
        s = specs.get(str(name))
        if s is None and RUN_MODE == "smoke":
            fam, cons = str(name).split("|")[0], str(name).split("|")[1]
            alt = [v for k, v in specs.items() if v["family"] == fam and k.split("|")[1] == cons
                   and v not in chosen and "v3_kind" not in v]
            s = alt[0] if alt else None
        if s is None or s["family"] not in families or (s.get("v3_kind") == "aggregated" and not allow_aggregated):
            skipped.append(str(name))
            continue
        if s not in chosen:
            chosen.append(s)
    return chosen, skipped


def x_v4_draws(spec: dict) -> int:
    n = int(spec["draws"])
    return n if XV4CFG.max_draws is None else min(n, int(XV4CFG.max_draws))


# ---------- instances, policies, hospital data and benchmarks ----------
def x_v4_instance(spec: dict, draw: int) -> CellInstance:
    """The II.6 instance (build_cell_instance) or the II.8 instance (build_v3_instance) of a draw."""
    return build_v3_instance(spec, draw) if "v3_kind" in spec else build_cell_instance(spec, draw)


def x_v4_draw_identity(spec: dict, inst: CellInstance) -> dict:
    """The identity of the regenerated draw: x_v3_draw_identity for II.8 cells, x_draw_identity otherwise."""
    return x_v3_draw_identity(spec, inst) if "v3_kind" in spec else x_draw_identity(spec, inst)


def x_v4_default_records(inst: CellInstance) -> Tuple[List[dict], List[bytes]]:
    """Default records of a draw (II.6 learners on the same training split and key, + behaviour and projections)."""
    learn = inst.learn_env
    raw = run_xmethods(LearnerView.from_env(learn), inst.train, f"{inst.cell}|draw{int(inst.draw)}")
    records = [r for r in x_collect_records(raw, inst.behavior, learn) if r["is_default"]]
    del raw
    return records, [_x_pi_key(r["pi"]) for r in records]


def x_v4_true_system(spec: dict, hazard: Optional[float] = None) -> Tuple[FiniteEnv, np.ndarray]:
    """True system and true physician of a Gumbel spec at timeout hazard `hazard` (default: the spec's).
    Garnet systems depend on the draw; use x_v4_true_system_draw."""
    assert spec["family"] == "gumbel", spec["family"]
    h = float(spec["timeout_hazard"] if hazard is None else hazard)
    env = _x_v3_gumbel_env(spec["variant"], h)
    beh = _x_v3_gumbel_physician(spec["variant"], h, spec["eps"], spec.get("physician", "optimal"))
    return env, beh


def x_v4_true_system_draw(spec: dict, draw: int, hazard: Optional[float] = None) -> Tuple[FiniteEnv, np.ndarray]:
    """True system and true behaviour policy of one draw of a Gumbel or Garnet spec (aggregated cells excluded)."""
    if spec["family"] == "gumbel":
        assert spec.get("v3_kind") != "aggregated", "aggregated cells have no true-state-space construction"
        return x_v4_true_system(spec, hazard)
    assert spec["family"] == "garnet" and hazard is None, (spec["family"], hazard)
    if spec.get("v3_kind") == "garnet-random":
        env, beh, _ = _x_v3_garnet_env(spec, draw)
        return env, beh
    return _x_garnet_env(spec, draw)


def x_v4_hospital(spec: dict, draw: int, true_env: FiniteEnv, true_behavior: np.ndarray,
                  stream_cell: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Hospital episodes of a draw: the stream and call of construct_benchmark (stream named by `stream_cell`,
    default the spec's own cell)."""
    rng_h = np_rng("xenv", "hospital", str(stream_cell or spec["cell"]), int(draw))
    return x_sample_dataset(true_env, true_behavior, int(spec["n_hosp"]), XCFG.max_steps, rng_h)


def x_v4_rebuild_benchmark(spec: dict, draw: int, inst: CellInstance
                           ) -> Tuple[FiniteEnv, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """(true_env, true_behavior, hospital episodes, benchmark dict) of an estimated ('best') draw; the rebuilt
    benchmark must equal the instance's learner environment (asserted)."""
    assert spec.get("construction", "best") == "best" and spec.get("v3_kind") != "aggregated", spec["cell"]
    true_env, beh = x_v4_true_system_draw(spec, draw)
    hosp = x_v4_hospital(spec, draw, true_env, beh)
    B = x_benchmark_from_data(true_env, hosp, spec["tau"], inst.learn_env.name)
    assert obj_hash(B["bench"].identity()) == obj_hash(inst.learn_env.identity()), \
        f"{spec['cell']} draw {draw}: regenerated hospital data do not rebuild the benchmark"
    return true_env, beh, hosp, B


def x_v4_eval(env: FiniteEnv, pi: np.ndarray, semantics: str) -> Dict[str, Any]:
    """Exact evaluation that never raises on a model-specific failure (ok / improper / near-singular)."""
    return _x_v3_eval(env, pi, semantics)


# ---------- certificates of one policy set ----------
def _x_v4_gt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, None] > b[None, :] + XV4_TOL


def x_v4_margin_claims(v: np.ndarray, k: int) -> np.ndarray:
    """Ordered claims i > j for the k unordered pairs with the largest |v_i - v_j| (gaps <= XV4_TOL never claimed;
    ties at the threshold broken by index order)."""
    n = len(v)
    iu = np.triu_indices(n, 1)
    gap = np.abs(v[iu[0]] - v[iu[1]])
    order = np.argsort(-gap, kind="stable")[:max(int(k), 0)]
    order = order[gap[order] > XV4_TOL]
    C = np.zeros((n, n), dtype=bool)
    i, j = iu[0][order], iu[1][order]
    hi = v[i] > v[j]
    C[np.where(hi, i, j), np.where(hi, j, i)] = True
    return C


def x_v4_pair_claims(lo: np.ndarray, up: np.ndarray, p: np.ndarray, v_range: float,
                     jt: Optional[np.ndarray] = None) -> List[dict]:
    """Claims of one policy set, by pair type. lo / up: J_term / J_W of each policy on the same rows; p: P(any
    unsupported decision) on those rows; v_range = max_T R - v_min (= v_max - v_min in every Part II family).

    Kinds: cert_nhb (lo_i > up_j), cert_free (lo_i > lo_j + p_j v_range, Corollary 2 without NHB), cert_nhb_only,
    cert_testable (certified with a losing policy that can leave the support), point_mean / point_term (strict
    single-rule claims) and margin_mean / margin_term / margin_mid (the k largest gaps, k = number of NHB
    certificates; margin_mean_free with k = number of NHB-free certificates). Pair types: AA (neither policy can leave
    the support), AU (one), UU (both), all. With `jt` (true values) every kind also counts its errors (truth reverses
    or ties). Returns one dict per type with '<kind>_claims' and '<kind>_errors'."""
    lo, up, p = (np.asarray(x, dtype=float) for x in (lo, up, p))
    n = len(lo)
    off = ~np.eye(n, dtype=bool)
    u = p > XV4_PTOL
    claims = {"cert_nhb": _x_v4_gt(lo, up), "cert_free": _x_v4_gt(lo, lo + p * float(v_range)),
              "point_mean": _x_v4_gt(up, up), "point_term": _x_v4_gt(lo, lo)}
    claims["cert_testable"] = claims["cert_nhb"] & u[None, :]
    # NHB-free implies NHB because J_W <= J_term + p v_range (Theorem 1(iii)); on rows that meet that bound only up to
    # the solver tolerance (bootstrap replicates) the implication can fail inside a 1e-9 window, so the subset
    # property is enforced and only a violation beyond 10 XV4_TOL is an error.
    _viol = claims["cert_free"] & ~claims["cert_nhb"] & off
    assert not (_viol & (up > lo + p * float(v_range) + 10 * XV4_TOL)[None, :]).any(), \
        "an NHB-free certificate must be an NHB one (Theorem 1(iii))"
    claims["cert_free"] &= claims["cert_nhb"]
    claims["cert_nhb_only"] = claims["cert_nhb"] & ~claims["cert_free"]
    k_nhb, k_free = int((claims["cert_nhb"] & off).sum()), int((claims["cert_free"] & off).sum())
    claims["margin_mean"] = x_v4_margin_claims(up, k_nhb)
    claims["margin_term"] = x_v4_margin_claims(lo, k_nhb)
    claims["margin_mid"] = x_v4_margin_claims((lo + up) / 2.0, k_nhb)
    claims["margin_mean_free"] = x_v4_margin_claims(up, k_free)
    typ = np.where(u[:, None] & u[None, :], "UU", np.where(u[:, None] | u[None, :], "AU", "AA"))
    wrong = None if jt is None else ~_x_v4_gt(np.asarray(jt, dtype=float), np.asarray(jt, dtype=float))
    out = []
    for ty in ("all", "AA", "AU", "UU"):
        tm = off if ty == "all" else (off & (typ == ty))
        r = {"type": ty, "n_policies": n, "pairs": int(tm.sum() // 2)}
        for nm, C in claims.items():
            r[f"{nm}_claims"] = int((C & tm).sum())
            if wrong is not None:
                r[f"{nm}_errors"] = int((C & tm & wrong).sum())
        out.append(r)
    return out


def x_v4_distinct(values: np.ndarray) -> np.ndarray:
    """Boolean mask keeping the first of every group of rows (policies) with identical values to 1e-12."""
    v = np.asarray(values, dtype=float)
    _, first = np.unique(np.round(v / 1e-12).astype(np.int64), axis=0, return_index=True)
    keep = np.zeros(len(v), dtype=bool)
    keep[first] = True
    return keep


# ---------- per-draw driver ----------
def x_v4_run_draws(cells: Sequence[dict], fn: Callable[[dict, int], Tuple[pd.DataFrame, dict]], desc: str,
                   time_key: Callable[[dict], str] = lambda s: str(s["family"])
                   ) -> Tuple[List[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    """Run fn(spec, draw) -> (rows, meta with 'cache' in {'loaded', 'computed'}) over every draw of every cell, with
    the progress and ETA of II.6a. Returns the frames, the flattened meta and a small summary."""
    frames: List[pd.DataFrame] = []
    meta_rows: List[dict] = []
    times: Dict[str, List[float]] = {}
    status: Counter = Counter()
    t_all = time.perf_counter()
    print(f"{desc}: {len(cells)} cells, {sum(x_v4_draws(s) for s in cells)} draws")
    for ci, spec in enumerate(cells, 1):
        t_cell, n_loaded, nd = time.perf_counter(), 0, x_v4_draws(spec)
        bar = trange(nd, desc=f"{desc} [{ci}/{len(cells)}] {spec['cell']}", leave=False, disable=not SHOW_PROGRESS)
        for draw in bar:
            t0 = time.perf_counter()
            df, meta = fn(spec, draw)
            dt = time.perf_counter() - t0
            frames.append(df)
            meta_rows.append(_x_flatten({**meta, "session_wall_s": dt}))
            status[meta["cache"]] += 1
            if meta["cache"] == "loaded":
                n_loaded += 1
            else:
                times.setdefault(time_key(spec), []).append(dt)
            if SHOW_PROGRESS:
                bar.set_postfix_str(f"{dt:.1f}s {meta['cache']}")
            elif (draw + 1) % max(1, nd // 5) == 0 or draw + 1 == nd:
                print(f"    {spec['cell']} draw {draw + 1}/{nd} ({meta['cache']}, {dt:.1f} s); cell ETA "
                      f"{_x_fmt_s(x_eta_seconds([(time_key(spec), nd - draw - 1)], times))}")
        eta = x_eta_seconds([(time_key(s), x_v4_draws(s)) for s in cells[ci:]], times)
        print(f"  [{ci}/{len(cells)}] {spec['cell']}: {nd} draws ({n_loaded} cached) in "
              f"{_x_fmt_s(time.perf_counter() - t_cell)} | total elapsed {_x_fmt_s(time.perf_counter() - t_all)} | "
              f"ETA remaining cells {_x_fmt_s(eta) if ci < len(cells) else '0 s'}")
    summary = {"cells": [s["cell"] for s in cells], "draws": int(sum(status.values())),
               "draws_computed": int(status["computed"]), "draws_loaded": int(status["loaded"]),
               "elapsed_s": time.perf_counter() - t_all,
               "draws_per_cell": {s["cell"]: x_v4_draws(s) for s in cells}}
    return frames, pd.DataFrame(meta_rows), summary


XV4_COMMON_CODE = [x_v4_all_specs, x_v4_cells, x_v4_draws, x_v4_instance, x_v4_draw_identity, x_v4_default_records,
                   x_v4_true_system, x_v4_true_system_draw, x_v4_hospital, x_v4_rebuild_benchmark, x_v4_eval,
                   _x_v4_gt, x_v4_margin_claims, x_v4_pair_claims, x_v4_distinct]
XV4_COMMON_CODE_HASH = obj_hash({"xenv_code": XENV_CODE_HASH, "v3_code": globals().get("XENV_V3_CODE_HASH"),
                                 "v4_common": code_hash(*XV4_COMMON_CODE),
                                 "constants": {"tol": XV4_TOL, "ptol": XV4_PTOL}})
print("II.9 switches:", {k: bool(EXPERIMENTS[k]) for k in XV4_SWITCHES}, "| config:", asdict(XV4CFG))
print("II.9 common code hash:", XV4_COMMON_CODE_HASH[:16])
