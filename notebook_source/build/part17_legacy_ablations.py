# %% [markdown]
# ## 21. Legacy artifact replay — the ORIGINAL run's policies and checkpoints (audit Phase 2, C01, C03, C04)
#
# If `LEGACY_ROOT` (default `/content/drive/MyDrive/sepsis_fresh_colab`) exists, its files are read-only inputs:
# 1. every original exported policy is re-validated and re-evaluated and compared with the saved
#    `exact_results_all.csv` rows (reported maximum discrepancy);
# 2. the admissible projection, deterministic masked IQL (from the original advantage-weighted checkpoint
#    scores) and masked BC are evaluated on the **original** artifacts, which directly tests the audit's
#    analytical prediction (projected IQL ≈ 0.78997 under both rules versus original diffusion 0.78565);
# 3. the original dataset is compared with the regenerated one, the saved normalizer is compared with a
#    refit (overwrite check), and original critic checkpoints are replayed to recover their Q-scale and to
#    check that the exported actions follow from the checkpoints.

# %%
# ============================================================
# 21. Legacy artifact replay (read-only)
# ============================================================
EXECLOG.mark("21. Legacy artifact replay")
LEGACY = {"root": str(LEGACY_ROOT), "found": bool(LEGACY_ROOT.exists()),
          "replay_enabled": bool(EXPERIMENTS["legacy_artifact_replay"])}
LEGACY_NAMES = {"Tabular CQL": "Tabular_CQL", "Tabular IQL": "Tabular_IQL", "Deep CQL": "Deep_CQL",
                "AM-CQL (Hard)": "AM-CQL_(Hard)", "AM-CQL (Soft)": "AM-CQL_(Soft)", "Diffusion-AM-CQL": "Diffusion-AM-CQL"}
LEGACY_FIXED = {"Behavior Cloning": "BC_fixed", "Random": "Random_fixed", "Clinician": "Clinician_fixed",
                "Optimal Oracle": "Optimal_oracle", "Constrained Optimal Oracle": "Constrained_optimal_oracle"}

if EXPERIMENTS["legacy_artifact_replay"] and LEGACY_ROOT.exists():
    lp = LEGACY_ROOT / "policies"
    legacy_rows_path = LEGACY_ROOT / "tables" / "exact_results_all.csv"
    legacy_saved = pd.read_csv(legacy_rows_path) if legacy_rows_path.exists() else None
    replay_rows = []

    def _legacy_register(model: str, seed, pi: np.ndarray, extraction: str, mask_use: str, mask_source: str,
                         stochastic: bool, parent: str = "") -> None:
        register_policy(f"LEGACY {model}", FIXED if seed in ("fixed", "oracle") else int(seed), pi, group="legacy",
                        dataset="original run", family=f"Legacy {model.split(' [')[0]}", extraction=extraction,
                        stochastic=stochastic, mask_source=mask_source, mask_use=mask_use,
                        information="original exported artifact", parent=parent)

    for model, stem in {**LEGACY_FIXED}.items():
        f = lp / f"{stem}.npz"
        if f.exists():
            with np.load(f) as z:
                _legacy_register(model, "fixed", z["policy"], "original export", "see original", "see original",
                                 stochastic=model in ("Random", "Clinician"))
    for model, stem in LEGACY_NAMES.items():
        for seed in range(10):
            f = lp / f"{stem}_seed{seed}.npz"
            if f.exists():
                with np.load(f) as z:
                    _legacy_register(model, seed, z["policy"], "original export",
                                     "training+extraction" if model in ("AM-CQL (Hard)", "Diffusion-AM-CQL") else "none",
                                     "released benchmark mask" if model in ("AM-CQL (Hard)", "Diffusion-AM-CQL", "AM-CQL (Soft)") else "none",
                                     stochastic=False)
    legacy_names = [n for n in REGISTRY if n.startswith("LEGACY ")]
    # deterministic masked IQL from the original advantage-weighted checkpoint scores
    for seed in range(10):
        f = LEGACY_ROOT / "checkpoints" / f"tabular_iql_seed{seed}.pt"
        if f.exists():
            pl = load_torch_payload(f)
            aw = pl["policy"].numpy().astype(np.float64)
            qv = pl["q"].numpy().astype(np.float64)
            adm_aw = np.where(OPERATIONAL_MASK, aw, -np.inf)
            act = adm_aw.argmax(axis=1)
            none_logged = ~(np.where(OPERATIONAL_MASK, aw, 0.0) > 0).any(axis=1)
            act[none_logged] = masked_argmax(qv)[none_logged]
            _legacy_register("Tabular IQL [masked greedy]", seed, one_hot_policy(act),
                             "masked argmax of original AW scores", "extraction", "released benchmark mask", False,
                             parent="LEGACY Tabular IQL")
            _legacy_register("Tabular IQL [stochastic AW]", seed, aw, "original AW distribution", "none", "none", True,
                             parent="LEGACY Tabular IQL")
    if "LEGACY Behavior Cloning" in REGISTRY:
        legacy_data = LEGACY_ROOT / "data" / "offline_expert_10000eps_seed2026.npz"
        if legacy_data.exists():
            with np.load(legacy_data) as z:
                LD = {k: z[k] for k in z.files}
            same = {k: bool(np.array_equal(LD[k], OFFLINE[k])) for k in ("state", "action", "next_state", "episode_id")
                    if k in LD}
            same["reward"] = bool(np.array_equal(LD["reward"].astype(np.float32), OFFLINE["reward"]))
            same["timestep"] = bool(np.array_equal(LD["timestep"].astype(np.int64), OFFLINE["timestep"].astype(np.int64)))
            same["done_equals_terminated_or_truncated"] = bool(np.array_equal(
                LD["done"].astype(bool), OFFLINE["terminated"] | OFFLINE["truncated"])) if len(LD["done"]) == len(OFFLINE["terminated"]) else False
            LEGACY["dataset_identical_to_regenerated"] = same
            cnt = np.zeros((S_N, A_N))
            np.add.at(cnt, (LD["state"], LD["action"]), 1.0)
            _legacy_register("Behavior Cloning [masked greedy]", "fixed", one_hot_policy(masked_argmax(cnt)),
                             "masked argmax of original counts", "extraction", "released benchmark mask", False,
                             parent="LEGACY Behavior Cloning")
            norm_file = LEGACY_ROOT / "data" / "feature_normalizer.npz"
            if norm_file.exists():
                with np.load(norm_file) as z:
                    saved_mean, saved_std = z["mean"], z["std"]
                refit = FeatureNormalizer.fit(BENCH.state_features, LD["state"])
                LEGACY["saved_normalizer_vs_refit_max_abs_diff"] = float(max(np.abs(saved_mean - refit.mean).max(),
                                                                            np.abs(saved_std - refit.std).max()))
                legacy_x = ((BENCH.state_features - saved_mean) / saved_std).astype(np.float32)
                crit_rows = []
                for variant, stem in (("hard", "hard_cql"), ("deep", "deep_cql"), ("soft", "soft_cql")):
                    for seed in range(10):
                        f = LEGACY_ROOT / "checkpoints" / f"{stem}_seed{seed}.pt"
                        if not f.exists():
                            continue
                        pl = load_torch_payload(f, map_location=DEVICE)
                        net = TwinDuelingCQL(BENCH.feature_dim, A_N, 256).to(DEVICE)
                        net.load_state_dict(pl["model"])
                        q1, q2 = critic_tables(net, legacy_x)
                        qmin = np.minimum(q1, q2)
                        exported = REGISTRY.get({"hard": "LEGACY AM-CQL (Hard)", "deep": "LEGACY Deep CQL",
                                                 "soft": "LEGACY AM-CQL (Soft)"}[variant])
                        re_pi = greedy_policy(qmin, masked=(variant == "hard"))
                        agree = (float((re_pi[LIVE].argmax(1) == exported.by_seed[seed][LIVE].argmax(1)).mean())
                                 if exported is not None and seed in exported.by_seed else float("nan"))
                        crit_rows.append({"variant": variant, "seed": seed,
                                          "q_adm_max_abs": float(np.abs(qmin[LIVE][SUPPORT_MASK[LIVE]]).max()),
                                          "q_adm_max": float(qmin[LIVE][SUPPORT_MASK[LIVE]].max()),
                                          "action_agreement_with_exported_policy": agree,
                                          "final_logged_td_loss": float(pd.DataFrame(pl["log"])["td_loss"].iloc[-1]),
                                          "final_logged_cql_loss": float(pd.DataFrame(pl["log"])["cql_loss"].iloc[-1]),
                                          "final_lambda": float(pl["meta"].get("final_lambda", np.nan))})
                        del net
                LEGACY["critic_replay"] = crit_rows
                pd.DataFrame(crit_rows).to_csv(TABLE_DIR / "legacy_critic_diagnostics.csv", index=False)
    add_projection_controls(["legacy"])
    finalize_projection_checks()
    LEGACY_RESULTS = evaluate_registered([n for n in REGISTRY if n.startswith("LEGACY ")])
    LEGACY_RESULTS = LEGACY_RESULTS[LEGACY_RESULTS["group"] == "legacy"]
    if legacy_saved is not None:
        for _, row in legacy_saved.iterrows():
            nm = f"LEGACY {row['model']}"
            seed = FIXED if str(row["seed"]) in ("fixed", "oracle") else int(row["seed"])
            hit = LEGACY_RESULTS[(LEGACY_RESULTS["name"] == nm) & (LEGACY_RESULTS["seed"] == seed)
                                 & (LEGACY_RESULTS["rule"] == row["strategy"])]
            if len(hit):
                replay_rows.append({"model": row["model"], "seed": row["seed"], "rule": row["strategy"],
                                    "saved_return": row["return"], "replayed_return": float(hit["return"].iloc[0]),
                                    "abs_diff_return": abs(row["return"] - float(hit["return"].iloc[0])),
                                    "abs_diff_inadmissible_rate": abs(row["inadmissible_rate"] - float(hit["inadmissible_rate"].iloc[0])),
                                    "abs_diff_length": abs(row["avg_episode_length"] - float(hit["avg_episode_length"].iloc[0]))})
        LEGACY_REPLAY = pd.DataFrame(replay_rows)
        LEGACY_REPLAY.to_csv(TABLE_DIR / "legacy_policy_reevaluation.csv", index=False)
        if len(LEGACY_REPLAY):
            LEGACY["replayed_rows"] = int(len(LEGACY_REPLAY))
            LEGACY["max_abs_discrepancy"] = float(LEGACY_REPLAY[["abs_diff_return", "abs_diff_inadmissible_rate",
                                                                  "abs_diff_length"]].max().max())
    LEGACY_SUMMARY = summarize_results(LEGACY_RESULTS)
    LEGACY_SUMMARY.to_csv(TABLE_DIR / "legacy_projection_controls_summary.csv", index=False)
    show = LEGACY_SUMMARY[LEGACY_SUMMARY["name"].isin([
        "LEGACY Tabular IQL", "LEGACY Tabular IQL [projected]", "LEGACY Tabular IQL [masked greedy]",
        "LEGACY Behavior Cloning [masked greedy]", "LEGACY Tabular CQL [projected]", "LEGACY Deep CQL [projected]",
        "LEGACY AM-CQL (Hard)", "LEGACY Diffusion-AM-CQL"])]
    display(show[["name", "rule", "n_seeds", "return_mean", "return_sd", "inadmissible_rate_mean"]].round(5))
    if "LEGACY Diffusion-AM-CQL" in REGISTRY and "LEGACY Tabular IQL [projected]" in REGISTRY:
        common = sorted(set(REGISTRY["LEGACY Diffusion-AM-CQL"].by_seed) & set(REGISTRY["LEGACY Tabular IQL [projected]"].by_seed))
        dif = [EVALUATOR.evaluate(REGISTRY["LEGACY Diffusion-AM-CQL"].by_seed[s], "terminate")["return"]
               - EVALUATOR.evaluate(REGISTRY["LEGACY Tabular IQL [projected]"].by_seed[s], "terminate")["return"] for s in common]
        LEGACY["original_diffusion_minus_projected_iql_terminate_per_seed"] = dif
        LEGACY["original_diffusion_minus_projected_iql_terminate_mean"] = float(np.mean(dif)) if dif else None
    save_json(LEGACY, TABLE_DIR / "legacy_replay.json")
    print(json.dumps({k: v for k, v in LEGACY.items() if k != "critic_replay"}, indent=2, default=_json_default))
else:
    print("Legacy artifact directory not found (or replay disabled):", LEGACY_ROOT)
    save_json(LEGACY, TABLE_DIR / "legacy_replay.json")

# %% [markdown]
# ## 22. Inference ablations, guidance sweep, diffusion necessity, schedule length, and cost accounting (audit M06, C04)
#
# Reproduces the original study's Tables 6–7 from code (for both the legacy critic and the repaired primary critic) and
# adds the direct categorical alternatives the audit requires. All comparisons here are descriptive (mean ± SD
# over seeds and per-seed paired differences); inferential statistics are in Section 30.

# %%
# ============================================================
# 22. Inference ablations / diffusion necessity / cost
# ============================================================
EXECLOG.mark("22. Inference ablations and M06")


def refresh_results() -> None:
    global RESULTS, SUMMARY
    RESULTS = evaluate_registered()
    SUMMARY = summarize_results(RESULTS)
    RESULTS.to_csv(TABLE_DIR / "exact_results_all.csv", index=False)
    SUMMARY.to_csv(TABLE_DIR / "exact_results_summary.csv", index=False)


def paired_frame(a: str, b: str, rule: str, metric: str = "return", df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = RESULTS if df is None else df
    x = df[(df["name"] == a) & (df["rule"] == rule) & (df["status"] == "ok")].set_index("seed")[metric]
    y = df[(df["name"] == b) & (df["rule"] == rule) & (df["status"] == "ok")].set_index("seed")[metric]
    if FIXED in y.index and len(y) == 1:
        y = pd.Series(float(y.iloc[0]), index=x.index)
    common = sorted(set(x.index) & set(y.index))
    return pd.DataFrame({"seed": common, "a": x.loc[common].values, "b": y.loc[common].values,
                         "diff": (x.loc[common] - y.loc[common]).values})


def ablation_table(reference: str, rows: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    out = []
    for label, name in rows:
        sub = SUMMARY[(SUMMARY["name"] == name)]
        if sub.empty:
            continue
        rec = {"control": label, "policy": name, "mask_source": sub["mask_source"].iloc[0],
               "mask_use": sub["mask_use"].iloc[0], **mask_columns(sub["mask_use"].iloc[0]),
               "information": sub["information"].iloc[0]}
        for rule in RULES:
            r = sub[sub["rule"] == rule]
            if r.empty:
                continue
            pf = paired_frame(name, reference, rule)
            rec[f"return_{rule}"] = float(r["return_mean"].iloc[0])
            rec[f"return_{rule}_sd"] = float(r["return_sd"].iloc[0]) if pd.notna(r["return_sd"].iloc[0]) else np.nan
            rec[f"delta_J_vs_reference_{rule}"] = float(pf["diff"].mean()) if len(pf) else np.nan
            rec[f"delta_J_sd_{rule}"] = float(pf["diff"].std(ddof=1)) if len(pf) > 1 else np.nan
            rec[f"100U_{rule}"] = 100.0 * float(r["inadmissible_rate_mean"].iloc[0])
        out.append(rec)
    return pd.DataFrame(out)


refresh_results()

TABLE6_LEGACY = ablation_table("Diffusion-AM-CQL [legacy critic]", [
    ("Original diffusion, g = 1.5 (legacy critic)", "Diffusion-AM-CQL [legacy critic]"),
    ("No critic guidance, g = 0", "Diffusion [masked, no guidance (g=0)]"),
    ("No diffusion: masked critic ensemble", "Masked critic ensemble [legacy critic + tabular CQL]"),
    ("Hard critic only", "Diffusion [legacy hard critic only]"),
    ("Tabular critic only", "Diffusion [tabular CQL critic only]"),
    ("Stochastic final policy", "Diffusion-AM-CQL [legacy critic] [stochastic final]"),
    ("Deep CQL with inference mask", "Deep CQL [masked greedy]"),
    ("Tabular CQL with inference mask", "Tabular CQL [masked greedy]"),
    ("Projected tabular IQL (C01 control)", "Tabular IQL [projected]"),
    ("Masked greedy tabular IQL (C01 control)", "Tabular IQL [masked greedy]"),
])
TABLE6_PRIMARY = ablation_table(PRIMARY_DIFFUSION, [
    ("Primary diffusion, g = 1.5 (critic B)", PRIMARY_DIFFUSION),
    ("No critic guidance, g = 0", "Diffusion [masked, no guidance (g=0)]"),
    ("No diffusion: masked critic ensemble", "Masked critic ensemble [critic B + tabular CQL]"),
    ("Neural-only guidance (critic B)", "Diffusion [critic B only]"),
    ("Tabular-only guidance", "Diffusion [tabular CQL critic only]"),
    ("Stochastic final policy", f"{PRIMARY_DIFFUSION} [stochastic final]"),
    ("Initialisation uniform over 25", f"{PRIMARY_DIFFUSION} [init uniform over 25]"),
    ("Guidance at final step only", f"{PRIMARY_DIFFUSION} [guidance at final step only]"),
    ("Prior, final-step mask, no guidance", "Diffusion prior [final-step mask, no guidance]"),
    ("Prior x critic at final step only", "Diffusion prior x critic B [final-step mask + guidance]"),
    ("Unmasked unguided prior (stochastic)", "Diffusion prior [unmasked, unguided, stochastic]"),
    ("Unmasked unguided prior (greedy)", "Diffusion prior [unmasked, unguided, greedy]"),
    ("Direct categorical: neural BC masked greedy", "Neural BC (matched capacity) [masked greedy]"),
    ("Direct categorical: AW imitation (IQL advantages) masked greedy", "AW categorical imitation [masked greedy]"),
    ("Direct categorical: AW imitation (critic-B scores) masked greedy", "AW categorical imitation [critic B + tabular CQL] [masked greedy]"),
    ("Neural BC prior x critic", "Neural BC prior x critic [critic B + tabular CQL]"),
    ("Tabular BC prior x critic", "Tabular BC prior x critic [critic B + tabular CQL]"),
    ("Projected tabular IQL (C01 control)", "Tabular IQL [projected]"),
    ("Masked greedy tabular IQL (C01 control)", "Tabular IQL [masked greedy]"),
    ("Diffusion with design-A critic", "Diffusion-AM-CQL [critic A]"),
    ("Diffusion with legacy critic", "Diffusion-AM-CQL [legacy critic]"),
])
TABLE7 = []
for g in CFG.guidance_grid:
    for critic_label, nm in (("legacy critic", f"Diffusion-AM-CQL [legacy critic] [g={g}]"),
                             ("critic B", f"{PRIMARY_DIFFUSION} [g={g}]")):
        sub = SUMMARY[(SUMMARY["name"] == nm) & (SUMMARY["rule"] == "terminate")]
        if len(sub):
            TABLE7.append({"guidance_g": g, "critic": critic_label, "mask_source": sub["mask_source"].iloc[0],
                           "mask_use": sub["mask_use"].iloc[0], "return_mean": float(sub["return_mean"].iloc[0]),
                           "return_sd": float(sub["return_sd"].iloc[0]),
                           "inadmissible_occupancy_pct": 100 * float(sub["inadmissible_rate_mean"].iloc[0])})
TABLE7 = pd.DataFrame(TABLE7)
TABLE6_LEGACY.to_csv(TABLE_DIR / "table6_inference_ablations_legacy_critic.csv", index=False)
TABLE6_PRIMARY.to_csv(TABLE_DIR / "table6_inference_ablations_primary_critic.csv", index=False)
TABLE7.to_csv(TABLE_DIR / "table7_guidance_sweep.csv", index=False)
pd.concat([TABLE6_LEGACY.assign(reference="legacy critic"), TABLE6_PRIMARY.assign(reference="critic B")]).to_csv(
    TABLE_DIR / "post_training_controls_all.csv", index=False)
display(TABLE6_PRIMARY.round(5))
display(TABLE7.round(5))

# ---------- Schedule length ablation (retrains the denoiser; ablation seeds) ----------
if EXPERIMENTS["diffusion_schedule_ablation"] and EXPERIMENTS["main_diffusion"]:
    for T_alt in CFG.diffusion_T_grid:
        sched = CategoricalDiffusion(A_N, T_alt, CFG.diffusion_beta_start, CFG.diffusion_beta_end).schedule_audit()
        for seed in CFG.ablation_seeds:
            prior = train_diffusion_prior(MAIN, seed, T=T_alt)
            DIFF_PRIORS[("main", seed, T_alt)] = {k: prior[k] for k in ("log", "meta", "label")}
            cache = ReverseCache(prior["model"], prior["diffusion"], MAIN.all_x)
            rB = get_critic(PREREG["primary_hard_critic"], MAIN, seed)
            G_B = critic_ensemble(np.minimum(rB["q1"], rB["q2"]), get_tab("cql", MAIN, seed)["q"])
            pi, tr = guided_reverse_policy(cache, G_B, G_PRIMARY)
            register_diffusion(f"{PRIMARY_DIFFUSION} [T={T_alt}]", seed, pi, tr, "schedule_ablation",
                               f"T={T_alt} (alpha_bar_T={sched['alpha_bar_T_initial_token_coefficient']:.4f})",
                               "hard_admrows + tabular CQL")
            del cache
    refresh_results()
    SCHEDULE_TABLE = SUMMARY[SUMMARY["name"].str.contains(r"\[T=", regex=True) | (SUMMARY["name"] == PRIMARY_DIFFUSION)]
    SCHEDULE_TABLE.to_csv(TABLE_DIR / "table_diffusion_schedule_length.csv", index=False)
    display(SCHEDULE_TABLE[["name", "rule", "n_seeds", "return_mean", "return_sd"]].round(5))

# ---------- Cost accounting (training cost includes upstream critics for guided policies) ----------
cost_rows = []
for (kind, bname, seed), r in TAB.items():
    if bname == "main":
        cost_rows.append({"component": f"tabular_{kind}", "seed": seed, "train_s": r["meta"].get("elapsed_s"),
                          "params": r["meta"].get("params"), "peak_gpu_mb": None})
for (key, bname, seed), r in CRITICS.items():
    if bname == "main":
        cost_rows.append({"component": f"critic_{key}", "seed": seed, "train_s": r["meta"].get("elapsed_s"),
                          "params": r["meta"].get("params"), "peak_gpu_mb": r["meta"].get("peak_gpu_mb")})
for (kind, bname, seed), r in PNETS.items():
    if bname == "main":
        cost_rows.append({"component": f"policy_net_{kind}", "seed": seed, "train_s": r["meta"].get("elapsed_s"),
                          "params": r["meta"].get("params"), "peak_gpu_mb": r["meta"].get("peak_gpu_mb")})
for (bname, seed, T_), r in DIFF_PRIORS.items():
    if bname == "main":
        cost_rows.append({"component": f"diffusion_prior_T{T_}", "seed": seed, "train_s": r["meta"].get("elapsed_s"),
                          "params": r["meta"].get("params"), "peak_gpu_mb": r["meta"].get("peak_gpu_mb")})
COST = pd.DataFrame(cost_rows)
COST.to_csv(TABLE_DIR / "cost_accounting_components.csv", index=False)
if len(COST):
    comp = COST.groupby("component").agg(train_s_mean=("train_s", "mean"), params=("params", "max"),
                                         peak_gpu_mb=("peak_gpu_mb", "max")).reset_index()
    def _t(c):
        v = comp.loc[comp["component"] == c, "train_s_mean"]
        return float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else 0.0
    def _p(c):
        v = comp.loc[comp["component"] == c, "params"]
        return int(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else 0
    dcost = pd.DataFrame(DIFFUSION_COST) if DIFFUSION_COST else pd.DataFrame()
    pipeline = [
        {"policy": "Tabular IQL [projected]", "total_train_s": _t("tabular_iql"), "total_params": _p("tabular_iql"),
         "policy_construction_s": 0.0, "deployed": "lookup table"},
        {"policy": "Masked critic ensemble [critic B + tabular CQL]", "total_train_s": _t("critic_hard_admrows") + _t("tabular_cql"),
         "total_params": _p("critic_hard_admrows") + _p("tabular_cql"), "policy_construction_s": 0.0, "deployed": "lookup table"},
        {"policy": "Neural BC prior x critic [critic B + tabular CQL]",
         "total_train_s": _t("policy_net_bc") + _t("critic_hard_admrows") + _t("tabular_cql"),
         "total_params": _p("policy_net_bc") + _p("critic_hard_admrows") + _p("tabular_cql"),
         "policy_construction_s": 0.0, "deployed": "lookup table"},
        {"policy": "Neural BC (matched capacity) [masked greedy]", "total_train_s": _t("policy_net_bc"),
         "total_params": _p("policy_net_bc"), "policy_construction_s": 0.0, "deployed": "lookup table"},
        {"policy": "AW categorical imitation [masked greedy]", "total_train_s": _t("policy_net_aw") + _t("tabular_iql"),
         "total_params": _p("policy_net_aw") + _p("tabular_iql"), "policy_construction_s": 0.0, "deployed": "lookup table"},
        {"policy": "AW categorical imitation [critic B + tabular CQL] [masked greedy]",
         "total_train_s": _t("policy_net_awB") + _t("critic_hard_admrows") + _t("tabular_cql"),
         "total_params": _p("policy_net_awB") + _p("critic_hard_admrows") + _p("tabular_cql"),
         "policy_construction_s": 0.0, "deployed": "lookup table"},
        {"policy": PRIMARY_DIFFUSION,
         "total_train_s": _t(f"diffusion_prior_T{CFG.diffusion_T}") + _t("critic_hard_admrows") + _t("tabular_cql"),
         "total_params": _p(f"diffusion_prior_T{CFG.diffusion_T}") + _p("critic_hard_admrows") + _p("tabular_cql"),
         "policy_construction_s": float((dcost["cache_s"] + dcost["construct_s"]).mean()) if len(dcost) else np.nan,
         "deployed": "lookup table"},
    ]
    COST_SUMMARY = pd.DataFrame(pipeline)
    COST_SUMMARY["deployed_lookup_latency_us"] = float(dcost["lookup_us"].mean()) if len(dcost) else np.nan
    COST_SUMMARY["on_demand_full_reverse_chain_ms_per_state"] = np.where(
        COST_SUMMARY["policy"] == PRIMARY_DIFFUSION,
        float(dcost["ondemand_ms"].dropna().mean()) if len(dcost) and "ondemand_ms" in dcost else np.nan, np.nan)
    PIPE = {"Tabular IQL [projected]": ["tabular_iql"],
            "Masked critic ensemble [critic B + tabular CQL]": ["critic_hard_admrows", "tabular_cql"],
            "Neural BC prior x critic [critic B + tabular CQL]": ["policy_net_bc", "critic_hard_admrows", "tabular_cql"],
            "Neural BC (matched capacity) [masked greedy]": ["policy_net_bc"],
            "AW categorical imitation [masked greedy]": ["policy_net_aw", "tabular_iql"],
            "AW categorical imitation [critic B + tabular CQL] [masked greedy]": ["policy_net_awB", "critic_hard_admrows", "tabular_cql"],
            PRIMARY_DIFFUSION: [f"diffusion_prior_T{CFG.diffusion_T}", "critic_hard_admrows", "tabular_cql"]}
    gpu = pd.to_numeric(comp.set_index("component")["peak_gpu_mb"], errors="coerce")
    COST_SUMMARY["peak_gpu_mb_max_component"] = [gpu.reindex(PIPE.get(p, [])).max() for p in COST_SUMMARY["policy"]]
    COST_SUMMARY["policy_construction_memory_mb"] = np.where(
        COST_SUMMARY["policy"] == PRIMARY_DIFFUSION, CFG.diffusion_T * S_N * A_N * A_N * 8 / 2 ** 20, 0.0)
    COST_SUMMARY.to_csv(TABLE_DIR / "cost_accounting.csv", index=False)
    comp.to_csv(TABLE_DIR / "cost_accounting_component_summary.csv", index=False)
    display(comp)
    display(COST_SUMMARY)
    print("Note: the deployed diffusion policy is a precomputed 716-row lookup table; the reverse chain is a "
          "construction cost, not a per-decision latency.")
