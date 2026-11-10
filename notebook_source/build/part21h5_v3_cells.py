# %% [markdown]
# ### II.8e Cells, instances, identities and the II.8 driver
#
# `X_V3_CELLS` lists the enabled II.8 cells (`x_v3_build_cells`). Each spec carries the keys of an II.3d spec of its
# family, plus `v3_kind` ∈ {`physician`, `aggregated`, `garnet-random`}. `build_v3_instance(spec, draw)` mirrors
# `build_cell_instance`, the `best` construction of II.3d. Every random stream is named
# `("xenv", purpose, cell, draw)`, with purpose ∈ {hospital, user, split}. The hospital stream is used first, then the
# user stream for the benchmark-user data, then the split stream for the train / dev split. For a spec that describes
# an II.3d cell, the instance equals the one `build_cell_instance` builds (checked when this section was written).
#
# | `v3_kind` | true system and behaviour | construction | targets |
# |---|---|---|---|
# | `physician` | exact Gumbel system, `gumbel_physician_v3` | `construct_benchmark` | `bench_mean`, `bench_terminate`, `true`, `true_rows_mean`, `true_rows_terminate` |
# | `aggregated` | exact Gumbel system, `gumbel_physician_v3` (full-state physician) | `construct_aggregated_benchmark` | `bench_mean`, `bench_terminate`, `true` (lifted through φ) |
# | `garnet-random` | `garnet_random_rows_env` | `construct_benchmark` | the five targets of the `physician` row |
#
# The deployment target is `true` in every cell. `run_v3_cell_draw(spec, draw)` performs the steps of
# `run_cell_draw` (II.6) with the same learners (`run_xmethods` on the training episodes), the same records
# (behaviour + projection of every native record), the same exact evaluation, reference rows and invariant checks
# (`x_check_draw_rows`, projection identity strict), and the same result columns (`XRESULT_COLUMNS`). Its identity is
# `x_draw_identity` with the code hash replaced by `XENV_V3_CODE_HASH`. That hash covers the source of every II.8
# function on the path and the values of the module constants these functions read, and it folds in
# `XENV_CODE_HASH`. Results are cached in `XCACHE` under `v3/<cell>/draw<k>` (and `/meta`), and the table returned is
# always the one read back. `x_v3_run_cells(cells)` loops over cells and draws with the progress and ETA of II.6a. It
# returns the stacked rows, the flattened draw metadata and the cross-draw check `x_check_results`, and it writes
# nothing.
#
# `xenv/v3_environment_suite.csv` (draw 0 of every enabled II.8 cell) has the columns of
# `xenv_environment_suite.csv`, plus the II.8 keys and the flattened construction report: aggregation sizes and the
# behaviour-aggregation identity, physician kind, and the hidden-benefit statistics of the random Garnet rows.
# `xenv/v3_environment_checks.json` holds the structural pass flag of this section (`X_V3_ENV_CHECK["passes"]`).

# %%
# ============================================================
# II.8e II.8 cells, instances, identities, driver
# ============================================================
def x_v3_build_cells() -> List[dict]:
    """Enabled Section II.8 cells (spec keys as in x_build_cells, plus v3_kind)."""
    cells: List[dict] = []
    gumbel_common = {"family": "gumbel", "draws": int(XCFG.gumbel_draws), "variant": XCFG.gumbel_variant,
                     "timeout_hazard": float(XCFG.gumbel_timeout_hazard), "user_episodes": int(XCFG.gumbel_user_episodes)}
    if EXPERIMENTS["xenv_v3_physicians"]:
        for construction, n_hosp, tau, eps, kind in XV3CFG.physician_cells:
            assert construction == "best" and kind in GUMBEL_PHYSICIAN_KINDS_V3, (construction, kind)
            tag = "" if kind == "optimal" else f"|phys-{kind}"
            cells.append({"cell": f"gumbel|best|n{n_hosp}|tau{tau}|eps{eps}{tag}", "v3_kind": "physician",
                          "construction": "best", "n_hosp": int(n_hosp), "tau": int(tau), "eps": float(eps),
                          "physician": str(kind), **gumbel_common})
    if EXPERIMENTS["xenv_v3_aggregated"]:
        for construction, n_hosp, tau, eps, kind in XV3CFG.aggregated_cells:
            assert construction in ("agg-obs", "agg-proj") and kind in GUMBEL_PHYSICIAN_KINDS_V3, (construction, kind)
            tag = "" if kind == "optimal" else f"|phys-{kind}"
            cells.append({"cell": f"gumbel|{construction}|n{n_hosp}|tau{tau}|eps{eps}{tag}", "v3_kind": "aggregated",
                          "construction": construction, "aggregation": construction.split("-", 1)[1],
                          "n_hosp": int(n_hosp), "tau": int(tau), "eps": float(eps), "physician": str(kind),
                          **gumbel_common})
    if EXPERIMENTS["xenv_v3_garnet_random"]:
        for eps in XV3CFG.garnet_random_eps:
            cells.append({"cell": f"garnet|hrand|eps{eps}", "family": "garnet", "v3_kind": "garnet-random",
                          "draws": int(XCFG.garnet_draws), "row_model": "independent random rows", "eps": float(eps),
                          "n_states": int(XCFG.garnet_states), "n_actions": int(XCFG.garnet_actions),
                          "branching": int(XCFG.garnet_branching),
                          "hazard_range": tuple(float(x) for x in XCFG.garnet_hazard),
                          "n_hosp": int(XCFG.garnet_hosp_episodes), "tau": int(XCFG.garnet_tau),
                          "user_episodes": int(XCFG.garnet_user_episodes)})
    keys = [c["cell"] for c in cells]
    assert len(keys) == len(set(keys)), f"duplicate II.8 cell keys: {keys}"
    clash = sorted(set(keys) & {c["cell"] for c in X_CELLS})
    assert not clash, f"II.8 cells must not reuse II.3d cell names: {clash}"
    return cells


def build_v3_instance(spec: dict, draw: int) -> CellInstance:
    """Deterministic (cell, draw) instance of a Section II.8 cell (same streams and steps as build_cell_instance)."""
    t0 = time.perf_counter()
    cell, family, kind, draw = spec["cell"], spec["family"], spec["v3_kind"], int(draw)
    assert draw >= 0 and kind in XV3_KINDS, (draw, kind)
    rng_h, rng_u = np_rng("xenv", "hospital", cell, draw), np_rng("xenv", "user", cell, draw)
    bname = f"{cell}|d{draw}|bench"
    row_report = None
    if kind in ("physician", "aggregated"):
        assert family == "gumbel", family
        true_env = _x_v3_gumbel_env(spec["variant"], spec["timeout_hazard"])
        true_behavior = _x_v3_gumbel_physician(spec["variant"], spec["timeout_hazard"], spec["eps"], spec["physician"])
        if kind == "physician":
            assert spec["construction"] == "best", spec["construction"]
            B = construct_benchmark(true_env, true_behavior, spec["n_hosp"], spec["tau"], rng_h, bname)
        else:
            B = construct_aggregated_benchmark(true_env, true_behavior, spec["aggregation"], spec["n_hosp"],
                                               spec["tau"], rng_h, bname)
    else:
        assert family == "garnet", family
        true_env, true_behavior, row_report = _x_v3_garnet_env(spec, draw)
        B = construct_benchmark(true_env, true_behavior, spec["n_hosp"], spec["tau"], rng_h, bname)
    learn_env, behavior, lift = B["bench"], B["bench_behavior"], B["lift"]
    data = x_sample_dataset(learn_env, behavior, spec["user_episodes"], XCFG.max_steps, rng_u)
    T_masked, T_meanimp = x_true_targets(true_env, B["mask_true"])
    targets = [("bench_mean", learn_env, "given", None), ("bench_terminate", learn_env, "terminate", None),
               ("true", T_masked, "given", lift)]
    if kind != "aggregated":
        targets += [("true_rows_mean", T_meanimp, "given", lift), ("true_rows_terminate", T_masked, "terminate", lift)]
    del T_meanimp
    if family == "gumbel":   # is the true optimal action supported? (decision states of the construction)
        opt_act, Vt = _x_v3_gumbel_opt_action(spec["variant"], spec["timeout_hazard"]), B["decision_states_true"]
        B["report"]["true_optimum_in_mask_fraction"] = (float(B["mask_true"][Vt, opt_act[Vt]].mean()) if len(Vt)
                                                        else float("nan"))
    if row_report is not None:
        B["report"].update({f"rows_{k}": v for k, v in row_report.items()})
    deploy = "true"
    construction = {k: v for k, v in B["report"].items() if k != "hospital_validation"}
    hosp_validation = B["report"]["hospital_validation"]
    source = f"x_sample_dataset({spec.get('construction', 'best')} benchmark, clinician estimate)"
    del B
    labels = [t[0] for t in targets]
    assert len(labels) == len(set(labels)) and deploy in labels, labels
    for lab, env, sem, lift_fn in targets:
        assert sem in X_SEMANTICS and (lift_fn is not None or env.S == learn_env.S), (lab, env.S, learn_env.S)
    user_validation = x_validate_dataset(data, learn_env)
    train, dev = x_split_episodes(data, XCFG.dev_fraction, np_rng("xenv", "split", cell, draw))
    meta = {"spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}, "user_data_source": source,
            "user_validation": user_validation, "hospital_validation": hosp_validation, "construction": construction,
            "train_transitions": int(len(train["state"])), "dev_transitions": int(len(dev["state"])),
            "build_s": time.perf_counter() - t0}
    return CellInstance(cell=cell, family=family, draw=draw, learn_env=learn_env, behavior=behavior, data=data,
                        train=train, dev=dev, targets=targets, deploy_target=deploy, meta=meta)


def x_v3_draw_identity(spec: dict, inst: CellInstance) -> dict:
    """x_draw_identity (II.6) with the II.8 code hash."""
    ident = x_draw_identity(spec, inst)
    ident["code"] = XENV_V3_CODE_HASH
    ident["section"] = "II.8"
    return ident


def run_v3_cell_draw(spec: dict, draw: int) -> Tuple[pd.DataFrame, dict]:
    """All results of one draw of one Section II.8 cell (the steps and checks of run_cell_draw; cached under v3/)."""
    cell, family = spec["cell"], spec["family"]
    assert family in ("gumbel", "garnet") and spec["v3_kind"] in XV3_KINDS, (family, spec.get("v3_kind"))
    t0 = time.perf_counter()
    inst = build_v3_instance(spec, draw)
    assert (inst.cell, inst.family, int(inst.draw)) == (cell, family, int(draw)), (inst.cell, inst.family, inst.draw)
    identity = x_v3_draw_identity(spec, inst)
    key, ident_hash = "v3/" + x_cache_key(cell, draw), obj_hash(identity)[:20]
    strict = True
    build_s = time.perf_counter() - t0
    hit, meta_hit = XCACHE.get(key, identity), XCACHE.get(key + "/meta", identity)
    if hit is not None and meta_hit is not None:
        del inst
        assert list(hit.columns) == XRESULT_COLUMNS, f"{key}: cached columns differ from XRESULT_COLUMNS"
        meta = json.loads(str(meta_hit["meta_json"].iloc[0]))
        meta.update(cache="loaded", session_build_s=build_s, checks=x_check_draw_rows(hit, strict))
        gc.collect()
        return hit, meta
    if (hit is None) != (meta_hit is None):
        print(f"  {key}: incomplete cache entry (results or meta missing) - recomputing this draw")
    XCACHE.require_allowed(key)
    learn = inst.learn_env
    t1 = time.perf_counter()
    raw = run_xmethods(LearnerView.from_env(learn), inst.train, f"{cell}|draw{int(draw)}")
    records = x_collect_records(raw, inst.behavior, learn)
    del raw
    t2 = time.perf_counter()
    fqe = x_fqe_env(inst.dev, inst.train, learn, name=f"fqe|{cell}|draw{int(draw)}")
    t3 = time.perf_counter()
    rows, ev_stats = x_evaluate_targets(inst, records, fqe)
    df = pd.DataFrame(rows, columns=XRESULT_COLUMNS)
    t4 = time.perf_counter()
    checks = x_check_draw_rows(df, strict)
    t5 = time.perf_counter()
    nonconv = [r["label"] for r in records if r["desc"]["converged"] is False]
    lm = learn.mask[learn.live]
    meta = _x_compact({
        "cell": cell, "family": family, "v3_kind": spec["v3_kind"], "draw": int(draw),
        "deploy_target": inst.deploy_target, "identity_hash": ident_hash, "code_hash": XENV_V3_CODE_HASH[:16],
        "cache": "computed",
        "learn_env": {"name": learn.name, "S": learn.S, "A": learn.A, "live": int(learn.live.sum()),
                      "admissible_pairs": int(lm.sum()), "mean_admissible_actions": float(lm.sum(axis=1).mean())},
        "data": {"train_transitions": int(len(inst.train["state"])),
                 "train_episodes": int(np.unique(inst.train["episode_id"]).size),
                 "dev_transitions": int(len(inst.dev["state"])),
                 "dev_episodes": int(np.unique(inst.dev["episode_id"]).size)},
        "targets": [t[0] for t in inst.targets],
        "records": {"total": len(records), **dict(Counter(r["variant"] for r in records)),
                    "nonconverged": len(nonconv), "nonconverged_labels": nonconv},
        "evaluation": ev_stats, "fqe_rows": fqe.meta.get("fqe_rows", {}), "checks": checks,
        "timing": {"build_s": build_s, "methods_s": t2 - t1, "fqe_build_s": t3 - t2, "eval_s": t4 - t3,
                   "check_s": t5 - t4, "total_s": t5 - t0},
        "instance": inst.meta})
    XCACHE.put(key + "/meta", identity, pd.DataFrame({"meta_json": [json.dumps(meta, default=_json_default)]}))
    XCACHE.put(key, identity, df)
    back = XCACHE.get(key, identity)
    assert back is not None and list(back.columns) == XRESULT_COLUMNS and len(back) == len(df), f"{key}: cache write"
    for c in _XMETRIC_SOURCE:
        assert np.allclose(back[c].to_numpy(dtype=float), df[c].to_numpy(dtype=float), rtol=1e-14, atol=0.0,
                           equal_nan=True), f"{key}: cache round trip changed column {c}"
    assert (back["status"].astype(str).to_numpy() == df["status"].to_numpy()).all(), f"{key}: cache round trip"
    del inst, fqe, records, rows, df
    gc.collect()
    return back, meta


def x_v3_run_cells(cells: Sequence[dict], desc: str = "II.8") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Every draw of every cell in `cells` (run_v3_cell_draw): stacked rows, flattened draw meta, cross-draw checks.

    Progress and ETA as in II.6a; nothing is written to disk."""
    frames: List[pd.DataFrame] = []
    meta_rows: List[dict] = []
    times: Dict[str, List[float]] = {}
    status: Counter = Counter()
    t_all = time.perf_counter()
    print(f"{desc} driver: {len(cells)} cells, {sum(int(s['draws']) for s in cells)} draws")
    for ci, spec in enumerate(cells, 1):
        t_cell, n_loaded, nd = time.perf_counter(), 0, int(spec["draws"])
        bar = trange(nd, desc=f"{desc} [{ci}/{len(cells)}] {spec['cell']}", leave=False, disable=not SHOW_PROGRESS)
        for draw in bar:
            t0 = time.perf_counter()
            df, meta = run_v3_cell_draw(spec, draw)
            dt = time.perf_counter() - t0
            frames.append(df)
            meta_rows.append(_x_flatten({**meta, "session_wall_s": dt}))
            status[meta["cache"]] += 1
            if meta["cache"] == "loaded":
                n_loaded += 1
            else:
                times.setdefault(spec["v3_kind"], []).append(dt)
            if SHOW_PROGRESS:
                bar.set_postfix_str(f"{dt:.1f}s {meta['cache']}")
            elif (draw + 1) % max(1, nd // 5) == 0 or draw + 1 == nd:
                print(f"    {spec['cell']} draw {draw + 1}/{nd} ({meta['cache']}, {dt:.1f} s); "
                      f"cell ETA {_x_fmt_s(x_eta_seconds([(spec['v3_kind'], nd - draw - 1)], times))}")
        eta = x_eta_seconds([(s["v3_kind"], int(s["draws"])) for s in cells[ci:]], times)
        print(f"  [{ci}/{len(cells)}] {spec['cell']}: {nd} draws ({n_loaded} cached) in "
              f"{_x_fmt_s(time.perf_counter() - t_cell)} | total elapsed {_x_fmt_s(time.perf_counter() - t_all)} | "
              f"ETA remaining cells {_x_fmt_s(eta) if ci < len(cells) else '0 s'}")
    res = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=XRESULT_COLUMNS)
    assert list(res.columns) == XRESULT_COLUMNS
    summary = {"code_hash": XENV_V3_CODE_HASH, "cells": len(cells), "draws": int(sum(status.values())),
               "draws_computed": int(status["computed"]), "draws_loaded": int(status["loaded"]),
               "elapsed_s": time.perf_counter() - t_all, **x_check_results(res)}
    return res, pd.DataFrame(meta_rows), summary


# Objects whose source enters the II.8 code hash, and the module constants they read (by value).
XENV_V3_CODE = [gumbel_diabetes_counterpart, gumbel_physician_v3, _x_v3_gumbel_env, _x_v3_gumbel_physician,
                _x_v3_gumbel_opt_action, x_gumbel_aggregation, x_aggregated_env, x_aggregate_dataset,
                construct_aggregated_benchmark, garnet_random_rows_env, _x_v3_garnet_env, x_v3_build_cells,
                build_v3_instance, x_v3_draw_identity, run_v3_cell_draw]
XENV_V3_CONSTANTS = {"physician_kinds": list(GUMBEL_PHYSICIAN_KINDS), "physician_kinds_v3": list(GUMBEL_PHYSICIAN_KINDS_V3),
                     "v3_kinds": list(XV3_KINDS)}
XENV_V3_CODE_HASH = obj_hash({"xenv_code": XENV_CODE_HASH, "v3_code": code_hash(*XENV_V3_CODE),
                              "constants": XENV_V3_CONSTANTS})
print("II.8 code hash:", XENV_V3_CODE_HASH[:16], "| Part II code hash:", XENV_CODE_HASH[:16])


def _x_v3_flat(prefix: str, d: Optional[dict]) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_x_v3_flat(f"{prefix}{k}.", v))
        elif isinstance(v, (list, tuple, np.ndarray)):
            out[f"{prefix}{k}"] = canonical_json(v)
        else:
            out[f"{prefix}{k}"] = v
    return out


def x_v3_environment_suite_table(cells: Sequence[dict]) -> pd.DataFrame:
    """One row per II.8 cell from draw 0 (columns of x_environment_suite_table + II.8 keys + flattened report)."""
    rows = []
    for spec in cells:
        inst = build_v3_instance(spec, 0)
        L = inst.learn_env
        row = {"cell": spec["cell"], "family": spec["family"], "v3_kind": spec["v3_kind"],
               "physician": spec.get("physician"), "aggregation": spec.get("aggregation"),
               "draws": spec["draws"], "deploy_target": inst.deploy_target,
               "S": L.S, "A": L.A, "live_states": int(L.live.sum()), "admissible_pairs": int(L.mask[L.live].sum()),
               "mean_admissible_actions": float(L.mask[L.live].sum(axis=1).mean()),
               "user_transitions": inst.meta["user_validation"]["transitions"],
               "user_episodes": inst.meta["user_validation"]["episodes"],
               "user_inadmissible_logged_fraction": inst.meta["user_validation"]["inadmissible_logged_fraction"],
               "build_s": inst.meta["build_s"], "targets": ";".join(t[0] for t in inst.targets)}
        statuses = []
        for lab, env, sem, lift_fn in inst.targets:
            beh = inst.behavior if lift_fn is None else lift_fn(inst.behavior)
            row[f"behaviour_value[{lab}]"], st_b = _x_value(env, beh, sem)
            row[f"optimum_value[{lab}]"], st_o = _x_opt_value(env, sem)
            statuses += [f"{what}[{lab}]={st}" for what, st in (("behaviour", st_b), ("optimum", st_o)) if st != "ok"]
        true_t = {lab: env for lab, env, _, _ in inst.targets}["true"]
        j_all, st_all = _x_opt_value(true_t, "given")
        j_adm, st_adm = _x_opt_value(true_t, "given", constrained=True)
        row["value_of_unsupported_actions"] = j_all - j_adm
        statuses += [f"optimum_admissible[true]={st_adm}"] * (st_adm != "ok") + [f"optimum[true]={st_all}"] * (st_all != "ok")
        row["non_ok_statuses"] = "; ".join(statuses)
        row.update(_x_v3_flat("construction_", inst.meta["construction"]))
        rows.append(row)
        del inst
        gc.collect()
    return pd.DataFrame(rows)


X_V3_CELLS: List[dict] = x_v3_build_cells()
print("II.8 cells:", [(c["cell"], c["draws"]) for c in X_V3_CELLS])
X_V3_ENV_SUITE = pd.DataFrame()
X_V3_ENV_CHECK: Dict[str, Any] = {"status": "not run"}
if X_V3_CELLS:
    _t0 = time.perf_counter()
    X_V3_ENV_SUITE = x_v3_environment_suite_table(X_V3_CELLS)
    X_V3_ENV_SUITE.to_csv(XENV_TABLE_DIR / "v3_environment_suite.csv", index=False)
    _agg = X_V3_ENV_SUITE[X_V3_ENV_SUITE["v3_kind"] == "aggregated"]
    _id_col = "construction_aggregation_check_behaviour_value_abs_diff"
    X_V3_ENV_CHECK = {
        "cells": [c["cell"] for c in X_V3_CELLS], "code_hash": XENV_V3_CODE_HASH,
        "non_ok_statuses": {r["cell"]: r["non_ok_statuses"] for _, r in X_V3_ENV_SUITE.iterrows() if r["non_ok_statuses"]},
        "trap_states_terminate_max": int(X_V3_ENV_SUITE["construction_trap_states_terminate"].max()),
        "behaviour_aggregation_identity_max_abs_diff": (float(_agg[_id_col].max()) if len(_agg) else None),
        "seconds": time.perf_counter() - _t0}
    X_V3_ENV_CHECK["passes"] = bool(not X_V3_ENV_CHECK["non_ok_statuses"]
                                    and X_V3_ENV_CHECK["trap_states_terminate_max"] == 0
                                    and (not len(_agg) or X_V3_ENV_CHECK["behaviour_aggregation_identity_max_abs_diff"] <= 1e-9))
    save_json(X_V3_ENV_CHECK, XENV_TABLE_DIR / "v3_environment_checks.json")
    _cols = [c for c in ("cell", "S", "live_states", "mean_admissible_actions", "behaviour_value[true]",
                         "optimum_value[true]", "value_of_unsupported_actions",
                         "construction_single_admissible_fraction", "construction_true_optimum_in_mask_fraction",
                         "construction_aggregated_live_cells", "construction_full_d0_mass_in_decision_cells",
                         "construction_rows_nonpreferred_beneficial_share", "non_ok_statuses")
             if c in X_V3_ENV_SUITE.columns]
    display(X_V3_ENV_SUITE[_cols])
    print(f"II.8 environment suite built in {time.perf_counter() - _t0:.1f} s | checks pass: {X_V3_ENV_CHECK['passes']}")
    assert X_V3_ENV_CHECK["passes"], "II.8 environment checks failed (xenv/v3_environment_checks.json)"
    del _agg, _id_col, _cols
print("II.8 environments and cells ready.")
