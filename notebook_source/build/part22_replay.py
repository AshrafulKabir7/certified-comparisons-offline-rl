# %% [markdown]
# ## 32. Artifact replay: cross-session comparison, saved policy files, CPU checkpoint inference (audit C04)
#
# Three levels, reported separately so that *replay* is never confused with *fresh retraining*:
# 1. **Cross-session replay** — the previous session's `exact_results_all.csv` and policy-file hashes were
#    snapshotted (Section 7) *before* anything was overwritten; they are compared with this session's results.
#    With an unchanged ROOT this is a genuine replay check; after `FORCE_RETRAIN` it measures fresh-retraining
#    differences (expected up to the hardware tolerance).
# 2. **File round-trip** — every saved policy file of the paper-table groups is reloaded, hash-checked,
#    re-validated and re-evaluated.
# 3. **Checkpoint inference on CPU** — main-study critic and diffusion checkpoints are rebuilt on CPU using the
#    normaliser stored **inside** each checkpoint; re-extracted actions are compared with the exported policies.
#    Guidance tables for the diffusion replay come from the CPU-replayed critic B and tabular CQL.

# %%
# ============================================================
# 32. Artifact replay
# ============================================================
EXECLOG.mark("32. Artifact replay")
refresh_results()
finalize_projection_checks(verbose=False)
REPLAY_GROUPS = {"main", "reference", "inference_ablation", "legacy", "critic_ablation", "soft_sensitivity",
                 "schedule_ablation"}
LATEST_FILES = {}
for rec in POLICY_FILES:
    LATEST_FILES[(rec["policy"], rec["seed"])] = rec
res_index = RESULTS[RESULTS["status"] == "ok"].set_index(["name", "seed", "rule"])

# (1) cross-session replay
cross_rows, hash_mismatch = [], []
prev = PREV_SESSION["exact_results"]
if prev is not None and len(prev):
    prev_ok = prev[prev["status"] == "ok"].drop_duplicates(["name", "seed", "rule"], keep="last").set_index(["name", "seed", "rule"])
    common_idx = res_index.index.intersection(prev_ok.index)
    for key in common_idx:
        cur, old = res_index.loc[key], prev_ok.loc[key]
        cross_rows.append({"policy": key[0], "seed": key[1], "rule": key[2],
                           **{f"abs_diff_{m}": abs(float(cur[m]) - float(old[m]))
                              for m in ("return", "inadmissible_rate", "avg_episode_length")}})
    hash_mismatch = [r["file"] for r in LATEST_FILES.values()
                     if r["file"] in PREV_SESSION["policy_hashes"] and PREV_SESSION["policy_hashes"][r["file"]] != r["policy_sha256"]]
CROSS = pd.DataFrame(cross_rows)
CROSS.to_csv(TABLE_DIR / "cross_session_replay.csv", index=False)
CROSS_OK = (None if prev is None or CROSS.empty else
            bool(not hash_mismatch and CROSS.filter(like="abs_diff").max().max() <= 1e-10))

# (2) file round-trip
replay_rows = []
for (name, seed), rec in LATEST_FILES.items():
    if REGISTRY[name].group not in REPLAY_GROUPS:
        continue
    with np.load(ROOT / rec["file"]) as z:
        pi = validate_policy(z["policy"], f"replay {name}")
    hash_ok = array_hash(pi) == rec["policy_sha256"]
    for rule in RULES:
        if (name, seed, rule) not in res_index.index:
            continue
        m = EVALUATOR.evaluate(pi, rule)
        saved = res_index.loc[(name, seed, rule)]
        replay_rows.append({"policy": name, "seed": seed, "rule": rule, "file": rec["file"], "file_hash_matches": hash_ok,
                            "abs_diff_return": abs(m["return"] - float(saved["return"])),
                            "abs_diff_inadmissible_rate": abs(m["inadmissible_rate"] - float(saved["inadmissible_rate"])),
                            "abs_diff_length": abs(m["avg_episode_length"] - float(saved["avg_episode_length"]))})
POLICY_REEVAL = pd.DataFrame(replay_rows)
POLICY_REEVAL.to_csv(TABLE_DIR / "policy_reevaluation.csv", index=False)
REPLAY_SUMMARY = {
    "file_round_trip_rows": int(len(POLICY_REEVAL)),
    "all_file_hashes_match": bool(POLICY_REEVAL["file_hash_matches"].all()) if len(POLICY_REEVAL) else None,
    "max_abs_discrepancy": float(POLICY_REEVAL[["abs_diff_return", "abs_diff_inadmissible_rate",
                                                "abs_diff_length"]].max().max()) if len(POLICY_REEVAL) else None,
    "cross_session_rows": int(len(CROSS)),
    "cross_session_policy_hash_mismatches": (len(hash_mismatch) if prev is not None else None),
    "cross_session_max_abs_diff": (float(CROSS.filter(like="abs_diff").max().max()) if len(CROSS) else None),
    "cross_session_ok": CROSS_OK,
    "force_retrain": FORCE_RETRAIN,
}

# Projection return / length differences (C01 closure: per-seed equality under mean; projected equality across rules).
okx = RESULTS[RESULTS["status"] == "ok"].set_index(["name", "seed", "rule"])
pr_rows = []
for child in [n for n in REGISTRY if n.endswith(" [projected]")]:
    par = REGISTRY[child].parent
    for seed in REGISTRY[child].by_seed:
        ks = [(par, seed, "mean"), (par, seed, "terminate"), (child, seed, "mean"), (child, seed, "terminate")]
        if not all(k in okx.index for k in ks):
            continue
        om, ot, pm_, pt_ = (okx.loc[k] for k in ks)
        pr_rows.append({"policy": par, "seed": seed, "group": REGISTRY[child].group,
                        "dJ_orig_mean_minus_proj_mean": om["return"] - pm_["return"],
                        "dJ_proj_mean_minus_proj_terminate": pm_["return"] - pt_["return"],
                        "dL_orig_mean_minus_proj_mean": om["avg_episode_length"] - pm_["avg_episode_length"],
                        "dL_proj_mean_minus_proj_terminate": pm_["avg_episode_length"] - pt_["avg_episode_length"],
                        "dJ_native_mean_minus_native_terminate": om["return"] - ot["return"],
                        "dL_native_mean_minus_native_terminate": om["avg_episode_length"] - ot["avg_episode_length"]})
PROJ_DIFFS = pd.DataFrame(pr_rows)
PROJ_DIFFS.to_csv(TABLE_DIR / "projection_return_differences.csv", index=False)
if len(PROJ_DIFFS) and MEAN_RULE_IDENTITY_HOLDS:
    _cols = [c for c in PROJ_DIFFS.columns if c.startswith("d") and "native" not in c]
    assert PROJ_DIFFS[_cols].abs().max().max() <= 1e-9, "Projected returns/lengths are not mean-equivalent."

# (3) CPU checkpoint inference
cpu = torch.device("cpu")
ckpt_rows = []
CPU_Q = {}


def _x_from_normalizer(nm: dict) -> np.ndarray:
    return ((BENCH.state_features[:, nm["keep"]] - np.asarray(nm["mean"], dtype=np.float32))
            / np.asarray(nm["std"], dtype=np.float32)).astype(np.float32)


for (key, bname, seed), res in list(CRITICS.items()):
    if bname != "main" or key not in SPEC_BY_KEY:
        continue
    spec = res["spec"]
    payload = STORE.load_final(res["label"], critic_identity(spec, seed, MAIN), map_location="cpu", respect_force=False)
    if payload is None:
        ckpt_rows.append({"model": spec.display, "seed": seed, "status": "missing checkpoint"})
        continue
    nm = payload["meta"]["normalizer"]
    x = _x_from_normalizer(nm)
    net = TwinDuelingCQL(len(nm["keep"]), A_N, CFG.hidden_dim)
    net.load_state_dict(payload["model"])
    q1, q2 = critic_tables(net, x, device=cpu)
    CPU_Q[(key, seed)] = np.minimum(q1, q2)
    if spec.display not in REGISTRY or seed not in REGISTRY[spec.display].by_seed:
        continue
    pi = greedy_policy(np.minimum(q1, q2), masked=spec.masked_extraction)
    ref = REGISTRY[spec.display].by_seed[seed]
    ckpt_rows.append({"model": spec.display, "seed": seed, "status": "ok",
                      "trained_on": payload.get("device_type"), "replayed_on": "cpu",
                      "normalizer_matches_bundle": bool(np.allclose(x, MAIN.all_x, atol=1e-6)),
                      "max_abs_q_diff_cpu_vs_session": float(max(np.abs(q1 - res["q1"]).max(), np.abs(q2 - res["q2"]).max())),
                      "action_agreement_live_states": float((pi[LIVE].argmax(1) == ref[LIVE].argmax(1)).mean())})
for seed in CFG.seeds:
    key = ("main", seed, CFG.diffusion_T)
    if key not in DIFF_PRIORS or PRIMARY_DIFFUSION not in REGISTRY or seed not in REGISTRY[PRIMARY_DIFFUSION].by_seed:
        continue
    payload = STORE.load_final(DIFF_PRIORS[key]["label"], diffusion_identity(MAIN, seed, CFG.diffusion_T),
                               map_location="cpu", respect_force=False)
    if payload is None or (PREREG["primary_hard_critic"], seed) not in CPU_Q:
        continue
    nm = payload["meta"]["normalizer"]
    x_ck = _x_from_normalizer(nm)
    den = build_denoiser(x_ck.shape[1])
    den.load_state_dict(payload["model"])
    dif = CategoricalDiffusion(A_N, CFG.diffusion_T, CFG.diffusion_beta_start, CFG.diffusion_beta_end)
    cache = ReverseCache(den, dif, x_ck, device=cpu)
    G_B = critic_ensemble(CPU_Q[(PREREG["primary_hard_critic"], seed)], get_tab("cql", MAIN, seed)["q"])
    pi, _ = guided_reverse_policy(cache, G_B, G_PRIMARY)
    ref = REGISTRY[PRIMARY_DIFFUSION].by_seed[seed]
    ckpt_rows.append({"model": PRIMARY_DIFFUSION, "seed": seed, "status": "ok",
                      "trained_on": payload.get("device_type"), "replayed_on": "cpu",
                      "normalizer_matches_bundle": bool(np.allclose(x_ck, MAIN.all_x, atol=1e-6)),
                      "action_agreement_live_states": float((pi[LIVE].argmax(1) == ref[LIVE].argmax(1)).mean())})
    del cache
CHECKPOINT_REPLAY = pd.DataFrame(ckpt_rows)
CHECKPOINT_REPLAY.to_csv(TABLE_DIR / "checkpoint_replay_cpu.csv", index=False)
REPLAY_SUMMARY["checkpoint_replays"] = int(len(CHECKPOINT_REPLAY))
REPLAY_SUMMARY["min_action_agreement"] = (float(CHECKPOINT_REPLAY["action_agreement_live_states"].min())
                                          if len(CHECKPOINT_REPLAY) and "action_agreement_live_states" in CHECKPOINT_REPLAY else None)
print(json.dumps(REPLAY_SUMMARY, indent=2))
numerical = json.loads((TABLE_DIR / "numerical_audit.json").read_text())
numerical.update({"policy_replay": REPLAY_SUMMARY, "projection_identity_verified": PROJECTION_IDENTITY_VERIFIED,
                  "mean_rule_identity": MEAN_RULE_IDENTITY})
save_json(numerical, TABLE_DIR / "numerical_audit.json")
if REPLAY_SUMMARY["max_abs_discrepancy"] is not None:
    assert REPLAY_SUMMARY["max_abs_discrepancy"] <= 1e-12, "Replayed policy evaluation differs from the session results."
if CROSS_OK is False:
    print("NOTE: results differ from the previous session in this ROOT (see cross_session_replay.csv). This is expected "
          "after FORCE_RETRAIN or a code change; otherwise investigate.")
if REPLAY_SUMMARY["min_action_agreement"] is not None and REPLAY_SUMMARY["min_action_agreement"] < 1.0:
    print("NOTE: CPU checkpoint inference changed some greedy actions (near-ties between CPU and GPU arithmetic); "
          "see checkpoint_replay_cpu.csv.")
