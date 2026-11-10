# %% [markdown]
# ## 13. Reproducibility closure tests (audit C03) and regulariser unit tests (audit §6)
#
# **Regulariser unit tests (always run, no training):** finite gradients for the legacy, design-A and design-B
# regularisers; the legacy regulariser's derivative with respect to an *excluded* logged action's Q equals
# −1/B (the confirmed defect); design B passes zero gradient to excluded rows and to masked entries; an
# all-excluded batch returns 0 with zero gradient; the `batch` normalisation equals the `conditional` one times
# the fraction of admissible-logged rows.
#
# **Closure tests** (isolated temporary checkpoint stores, short budgets; they also run in artifact-replay mode
# because the temporary stores are allowed to train):
# 1. initial weights for a seed are identical in forward or reverse seed order and after unrelated RNG draws;
# 2. short critic and denoiser training reproduces after unrelated RNG draws and in a different seed order;
# 3. an interrupted run resumed from its resumable checkpoint reproduces the uninterrupted run — tested for a
#    hard critic, for the **soft critic (dual variable λ restored)**, and for the denoiser;
# 4. changing the dataset, objective, run mode, seed or *actual class code* changes the run identity, and a
#    checkpoint copied under a foreign identity is rejected rather than loaded.
#
# Declared hardware tolerance: bitwise on CPU; on CUDA a maximum absolute difference ≤ 1e-4 is accepted
# because some backward kernels (e.g. scatter-add for `gather`) are not deterministic on GPUs.

# %%
# ============================================================
# 13. Regulariser unit tests + reproducibility tests
# ============================================================
EXECLOG.mark("13. Reproducibility tests")
REPRO_TESTS: Dict[str, Any] = {}
HW_TOL = 0.0 if DEVICE.type == "cpu" else 1e-4

# ---------- Regulariser gradient tests (audit §6: test before long training) ----------
_rows = np.flatnonzero(LIVE & ~SUPPORT_MASK.all(axis=1))[:8]
_m = torch.as_tensor(OPERATIONAL_MASK[_rows])
_a = torch.as_tensor([int(np.flatnonzero(~OPERATIONAL_MASK[s])[0]) if i % 2 else int(np.flatnonzero(OPERATIONAL_MASK[s])[0])
                      for i, s in enumerate(_rows)])
_adm = _m.gather(1, _a[:, None]).squeeze(1)
_B = len(_rows)
REG_GRAD_TESTS = {"batch_rows": _B, "admissible_logged_rows": int(_adm.sum())}
for _spec in (SPEC_HARD_LEGACY, SPEC_HARD_A, SPEC_HARD_B, dataclasses.replace(SPEC_HARD_B, reg_normalization="batch")):
    _q = torch.randn(_B, A_N, dtype=torch.float64, generator=torch.Generator().manual_seed(0), requires_grad=True)
    cql_regularizer(_q, _a, _m, _adm, _spec).backward()
    assert torch.isfinite(_q.grad).all(), _spec.key
    if _spec.reg_domain == "masked_legacy":
        _g = _q.grad[~_adm, _a[~_adm]]
        assert torch.allclose(_g, torch.full_like(_g, -1.0 / _B)), "Legacy excluded-row derivative is not -1/B."
        REG_GRAD_TESTS["legacy_excluded_logged_action_derivative"] = float(_g.mean())
    if _spec.reg_domain == "masked_admissible_rows":
        assert float(_q.grad[~_adm].abs().max()) == 0.0, "Design B leaks gradient into excluded rows."
        assert float(_q.grad[_adm][~_m[_adm]].abs().max()) == 0.0, "Design B leaks gradient into masked actions."
_q = torch.randn(_B, A_N, dtype=torch.float64, requires_grad=True)
_r0 = cql_regularizer(_q, _a, _m, torch.zeros_like(_adm), SPEC_HARD_B)
_r0.backward()
assert float(_r0) == 0.0 and float(_q.grad.abs().max()) == 0.0, "All-excluded batch must give 0 with zero gradient."
_q = torch.randn(_B, A_N, dtype=torch.float64)
_rc = cql_regularizer(_q, _a, _m, _adm, SPEC_HARD_B)
_rb = cql_regularizer(_q, _a, _m, _adm, dataclasses.replace(SPEC_HARD_B, reg_normalization="batch"))
assert torch.allclose(_rb, _rc * _adm.double().mean()), "batch != conditional x admissible fraction."
REG_GRAD_TESTS["passed"] = True
REPRO_TESTS["regularizer_unit_tests"] = REG_GRAD_TESTS
save_json(REG_GRAD_TESTS, TABLE_DIR / "regularizer_gradient_tests.json")
print("Regulariser unit tests passed:", REG_GRAD_TESTS)

if EXPERIMENTS["reproducibility_tests"]:
    test_root = ROOT / "_repro_tests"
    shutil.rmtree(test_root, ignore_errors=True)
    saved_cfg = (CFG.resume_every, CFG.diag_every, CFG.log_every, CFG.diffusion_steps)
    try:
        # 1. initialisation independence
        builders = {
            "twin_critic": (lambda: TwinDuelingCQL(MAIN.input_dim, A_N, CFG.hidden_dim), f"twin_dueling_{MAIN.input_dim}x{CFG.hidden_dim}"),
            "denoiser": (lambda: build_denoiser(MAIN.input_dim), f"denoiser_{MAIN.input_dim}x{CFG.hidden_dim}"),
            "policy_net": (lambda: CategoricalPolicyNet(MAIN.input_dim, A_N, CFG.hidden_dim), f"policy_net_{MAIN.input_dim}x{CFG.hidden_dim}"),
        }
        init_rows = {}
        for name, (builder, key) in builders.items():
            ref = state_dict_hash(init_model(builder, key, 3))
            for s in (0, 1, 2):
                init_model(builder, key, s)
            torch.rand(1234)
            np.random.rand(77)
            random.random()
            fwd = state_dict_hash(init_model(builder, key, 3))
            for s in (4, 2, 1, 0):
                init_model(builder, key, s)
            rev = state_dict_hash(init_model(builder, key, 3))
            other = state_dict_hash(init_model(builder, key, 4))
            init_rows[name] = {"same_after_other_seeds_and_draws": ref == fwd, "same_in_reverse_order": ref == rev,
                               "different_seed_differs": ref != other}
            assert all(init_rows[name].values()), (name, init_rows[name])
        REPRO_TESTS["initialisation"] = init_rows

        # 2-3. short training determinism, seed-order independence, resume equivalence
        CFG.resume_every, CFG.diag_every, CFG.log_every = 50, 10_000, 50
        short = dataclasses.replace(SPEC_HARD_B, steps=200)
        short_soft = dataclasses.replace(SPEC_SOFT, steps=200)
        stA, stB, stC = (CheckpointStore(test_root / x, allow_training=True) for x in ("A", "B", "C"))
        rA = train_critic(short, MAIN, 1, tag="repro", store=stA)
        torch.rand(999)
        np.random.rand(5)
        train_critic(short, MAIN, 0, tag="repro", store=stB)
        rB = train_critic(short, MAIN, 1, tag="repro", store=stB)
        assert train_critic(short, MAIN, 1, tag="repro", store=stC, interrupt_at=100) is None, "Interrupt hook failed."
        rC = train_critic(short, MAIN, 1, tag="repro", store=stC)
        sA = train_critic(short_soft, MAIN, 1, tag="repro", store=stA)
        assert train_critic(short_soft, MAIN, 1, tag="repro", store=stC, interrupt_at=100) is None
        sC = train_critic(short_soft, MAIN, 1, tag="repro", store=stC)
        crit = {
            "max_abs_diff_rerun_after_unrelated_draws_and_other_seed": float(np.abs(rA["q1"] - rB["q1"]).max()),
            "max_abs_diff_resumed_vs_uninterrupted": float(np.abs(rA["q1"] - rC["q1"]).max()),
            "resumed_flag": bool(rC["meta"]["resumed"]),
            "soft_max_abs_q_diff_resumed_vs_uninterrupted": float(np.abs(sA["q1"] - sC["q1"]).max()),
            "soft_abs_lambda_diff_resumed_vs_uninterrupted": abs(sA["meta"]["final_lambda"] - sC["meta"]["final_lambda"]),
            "soft_resumed_flag": bool(sC["meta"]["resumed"]),
        }
        crit["passes_declared_tolerance"] = bool(max(
            crit["max_abs_diff_rerun_after_unrelated_draws_and_other_seed"], crit["max_abs_diff_resumed_vs_uninterrupted"],
            crit["soft_max_abs_q_diff_resumed_vs_uninterrupted"], crit["soft_abs_lambda_diff_resumed_vs_uninterrupted"]) <= HW_TOL)
        REPRO_TESTS["critic_training"] = crit

        CFG.diffusion_steps = 200
        dA = train_diffusion_prior(MAIN, 1, tag="repro", store=stA)
        torch.rand(31)
        train_diffusion_prior(MAIN, 0, tag="repro", store=stB)
        dB = train_diffusion_prior(MAIN, 1, tag="repro", store=stB)
        assert train_diffusion_prior(MAIN, 1, tag="repro", store=stC, interrupt_at=100) is None
        dC = train_diffusion_prior(MAIN, 1, tag="repro", store=stC)
        sdA, sdB, sdC = (m["model"].state_dict() for m in (dA, dB, dC))
        diffd = {
            "max_abs_param_diff_rerun": float(max((sdA[k].double() - sdB[k].double()).abs().max() for k in sdA)),
            "max_abs_param_diff_resumed": float(max((sdA[k].double() - sdC[k].double()).abs().max() for k in sdA)),
        }
        diffd["passes_declared_tolerance"] = bool(max(diffd.values()) <= HW_TOL)
        REPRO_TESTS["diffusion_training"] = diffd

        # 4. cache invalidation (including a real class-code edit) and foreign-identity rejection
        def _probe_class(body: str):
            ns = {}
            exec(compile(f"class _Probe:\n    def f(self):\n        return {body}\n", "<probe>", "exec"), ns)
            return ns["_Probe"]
        code_edit_detected = code_hash(_probe_class("1")) != code_hash(_probe_class("2"))
        base_ident = critic_identity(short, 1, MAIN)
        sub = make_bundle("main_subset", OFFLINE, OFFLINE_MANIFEST, rows=np.arange(len(OFFLINE["state"]) - 1))
        variants = {
            "dataset": critic_identity(short, 1, sub),
            "objective": critic_identity(dataclasses.replace(short, alpha=2.0), 1, MAIN),
            "run_mode": {**base_ident, "run_mode": "smoke" if RUN_MODE == "paper" else "paper"},
            "code_hash_value": {**base_ident, "code": "edited-code"},
            "seed": critic_identity(short, 2, MAIN),
        }
        h0 = obj_hash(base_ident)
        inval = {k: obj_hash(v) != h0 for k, v in variants.items()}
        inval["actual_class_code_edit"] = bool(code_edit_detected)
        assert all(inval.values()), inval
        label = "repro/hard_admrows/main/seed1"
        src, _, _ = stA._paths(label, base_ident)
        dst, _, _ = stA._paths(label, variants["objective"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        try:
            stA.load_final(label, variants["objective"], respect_force=False)
            rejected = False
        except CheckpointIdentityError:
            rejected = True
        assert rejected, "A checkpoint stored under a foreign identity was loaded."
        REPRO_TESTS["cache_invalidation"] = {**{f"identity_changes_with_{k}": v for k, v in inval.items()},
                                             "foreign_identity_checkpoint_rejected": rejected}
    finally:
        CFG.resume_every, CFG.diag_every, CFG.log_every, CFG.diffusion_steps = saved_cfg
        shutil.rmtree(test_root, ignore_errors=True)
    REPRO_TESTS["hardware_tolerance"] = HW_TOL
    REPRO_TESTS["device"] = str(DEVICE)
    save_json(REPRO_TESTS, TABLE_DIR / "reproducibility_tests.json")
    print(json.dumps(REPRO_TESTS, indent=2, default=_json_default))
    if not (REPRO_TESTS["critic_training"]["passes_declared_tolerance"]
            and REPRO_TESTS["diffusion_training"]["passes_declared_tolerance"]):
        print("WARNING: training reproducibility exceeded the declared hardware tolerance on this device.")
else:
    print("Reproducibility closure tests skipped (EXPERIMENTS['reproducibility_tests'] = False).")

# %% [markdown]
# ## 14. Reference policies, oracles, and environment-semantics tests (audit C01, M09)
#
# * Behaviour cloning (native argmax, masked greedy, stochastic empirical), uniform random over all 25
#   actions, **uniform over admissible actions**, the benchmark clinician policy, and value-iteration
#   oracles (unconstrained mean-optimal, admissibility-constrained, terminate-optimal). Oracles use the true
#   model and are privileged references, not offline competitors.
# * The rounded documentation baselines are kept only as a broad sanity check; the tight checks are the
#   analytic tests of Section 8, the evaluator invariants, the environment-semantics test below, and the
#   Monte-Carlo cross-check of Section 20.

# %%
# ============================================================
# 14. Reference policies and oracles
# ============================================================
EXECLOG.mark("14. Reference policies")

BC = train_behavior_cloning(MAIN)
register_policy("Behavior Cloning", FIXED, BC["deterministic"], scores=BC["counts"], group="reference",
                family="BC", extraction="argmax of empirical counts (unmasked)", stochastic=False,
                mask_source="none", mask_use="none", information="logged state-action counts")
register_policy("Behavior Cloning [masked greedy]", FIXED, BC["masked_greedy"], group="reference",
                family="BC", extraction="argmax of counts over admissible actions", stochastic=False,
                mask_source="released benchmark mask", mask_use="extraction",
                information="logged counts + released mask", parent="Behavior Cloning")
register_policy("Behavior Cloning [stochastic empirical]", FIXED, BC["stochastic"], group="reference",
                family="BC", extraction="empirical action distribution", stochastic=True,
                mask_source="none", mask_use="none", information="logged state-action counts")
register_policy("Random (uniform over 25)", FIXED, np.full((S_N, A_N), 1.0 / A_N), group="reference",
                family="Random", extraction="uniform", stochastic=True, mask_source="none", mask_use="none",
                information="none")
register_policy("Random (uniform admissible)", FIXED, uniform_admissible_policy(), group="reference",
                family="Random", extraction="uniform over admissible actions", stochastic=True,
                mask_source="released benchmark mask", mask_use="construction", information="released mask")

CLINICIAN_POLICY = EXPERT_ROWS.copy()
CLINICIAN_POLICY[LIVE] = CLINICIAN_POLICY[LIVE] / CLINICIAN_POLICY[LIVE].sum(axis=1, keepdims=True)
CLINICIAN_POLICY[~LIVE] = 1.0 / A_N   # absorbing rows carry no expert probabilities; never used by the evaluator
register_policy("Clinician", FIXED, CLINICIAN_POLICY, group="reference", family="Clinician",
                extraction="benchmark expert policy", stochastic=True, mask_source="none", mask_use="none",
                information="benchmark clinician policy")

ORACLE_INFO = {}
opt_mean, _, ORACLE_INFO["mean_unconstrained"] = EVALUATOR.optimal_policy("mean", constrain_to_admissible=False)
opt_cons, _, ORACLE_INFO["mean_constrained"] = EVALUATOR.optimal_policy("mean", constrain_to_admissible=True)
opt_term, _, ORACLE_INFO["terminate_unconstrained"] = EVALUATOR.optimal_policy("terminate", constrain_to_admissible=False)
opt_mean_low, _, ORACLE_INFO["mean_unconstrained_lowest_index_ties"] = EVALUATOR.optimal_policy(
    "mean", constrain_to_admissible=False, tie_break="lowest_index")
ORACLE_INFO["terminate_optimum_equals_constrained_optimum"] = bool(
    abs(ORACLE_INFO["terminate_unconstrained"]["dp_value"] - ORACLE_INFO["mean_constrained"]["dp_value"]) < 1e-8)
ORACLE_INFO["mean_optimum_equals_constrained_optimum"] = bool(
    abs(ORACLE_INFO["mean_unconstrained"]["dp_value"] - ORACLE_INFO["mean_constrained"]["dp_value"]) < 1e-8)
print("Note: the mean-optimal policy is not unique (exact q ties between admissible and mean-imputed inadmissible "
      "actions). The registered oracle uses admissible tie-breaking; the previous notebook's lowest-index "
      "tie-breaking is kept as a separately labelled policy, and its violations are a tie-breaking artefact.")
for name, pi, note in [("Optimal Oracle (mean-optimal)", opt_mean, "unconstrained value iteration under mean, admissible tie-breaking"),
                       ("Optimal Oracle (mean-optimal, lowest-index ties)", opt_mean_low,
                        "unconstrained value iteration under mean, np.argmax tie-breaking (previous notebook)"),
                       ("Constrained Optimal Oracle", opt_cons, "value iteration over admissible actions"),
                       ("Optimal Oracle (terminate-optimal)", opt_term, "value iteration under terminate")]:
    register_policy(name, FIXED, pi, group="reference", family="Oracle", extraction=note, stochastic=False,
                    mask_source="released benchmark mask" if "Constrained" in name else "none",
                    mask_use="construction" if "Constrained" in name else "none",
                    information="TRUE transition model (privileged)")
print(json.dumps(ORACLE_INFO, indent=2))

reference_baselines = {"Random (uniform over 25)": (0.78, 9.45), "Clinician": (0.78, 9.22),
                       "Optimal Oracle (mean-optimal)": (0.88, 10.99)}
sanity = []
for name, (ref_ret, ref_len) in reference_baselines.items():
    m = EVALUATOR.evaluate(REGISTRY[name].by_seed[FIXED], "mean", condition=True)
    sanity.append({"policy": name, "measured_return": m["return"], "reference_return_rounded": ref_ret,
                   "measured_length": m["avg_episode_length"], "reference_length_rounded": ref_len,
                   "condition_number": m["condition_number"]})
BENCHMARK_SANITY = pd.DataFrame(sanity)
display(BENCHMARK_SANITY)
BENCHMARK_SANITY.to_csv(TABLE_DIR / "benchmark_reference_sanity.csv", index=False)
assert (np.abs(BENCHMARK_SANITY["measured_return"] - BENCHMARK_SANITY["reference_return_rounded"]) <= 0.05).all()
assert (np.abs(BENCHMARK_SANITY["measured_length"] - BENCHMARK_SANITY["reference_length_rounded"]) <= 2.0).all()

ENV_SEMANTICS = {}
if EXPERIMENTS["env_semantics_tests"]:
    ENV_SEMANTICS = env_semantics_tests(n_pairs=40 if RUN_MODE == "paper" else 10,
                                        n_draws=1500 if RUN_MODE == "paper" else 300)
    print(json.dumps(ENV_SEMANTICS, indent=2))
    assert ENV_SEMANTICS["terminate_all_consistent"], "Terminate semantics differ between env and evaluator."
    if not ENV_SEMANTICS["mean_rule_all_within_bounds"]:
        print("WARNING: an empirical mean-rule transition row deviates beyond its 5-sigma bound.")
save_json({"oracles": ORACLE_INFO, "env_semantics": ENV_SEMANTICS, "codec": CODEC_CHECK,
           "evaluator_toy_tests": EVALUATOR_TOY_TESTS}, TABLE_DIR / "numerical_audit.json")
