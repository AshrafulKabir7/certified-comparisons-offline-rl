# %% [markdown]
# ## 15. Main study — tabular learners on the main seeds (0–9 in the full profile, 0–4 in lite)

# %%
# ============================================================
# 15. Main study: tabular learners
# ============================================================
EXECLOG.mark("15. Main study: tabular")
TAB: Dict[Tuple[str, str, int], Dict[str, Any]] = {}


def save_log(name: str, seed: int, df: pd.DataFrame) -> None:
    if df is not None and len(df):
        df.to_csv(LOG_DIR / f"{slug(name)}__seed{seed}.csv", index=False)


def run_tabular_suite(bundle: DataBundle, seeds: Sequence[int], group: str, prefix: str = "",
                      dataset_label: str = "main") -> None:
    for seed in seeds:
        cql = train_tabular_cql(bundle, seed)
        iql = train_tabular_iql(bundle, seed)
        bcq = train_tabular_bcq(bundle, seed)
        TAB[("cql", bundle.name, seed)], TAB[("iql", bundle.name, seed)], TAB[("bcq", bundle.name, seed)] = cql, iql, bcq
        common = dict(group=group, dataset=dataset_label)
        register_policy(f"{prefix}Tabular CQL", seed, greedy_policy(cql["q"], masked=False), scores=cql["q"],
                        family="Tabular CQL", extraction="unmasked greedy", stochastic=False, mask_source="none",
                        mask_use="none", information="logged transitions", **common)
        register_policy(f"{prefix}Tabular CQL [masked greedy]", seed, greedy_policy(cql["q"], masked=True),
                        family="Tabular CQL", extraction="masked greedy (inference mask)", stochastic=False,
                        mask_source="released benchmark mask", mask_use="extraction",
                        information="logged transitions + released mask", parent=f"{prefix}Tabular CQL", **common)
        register_policy(f"{prefix}Tabular IQL", seed, iql["deterministic"], scores=iql["aw_scores"],
                        family="Tabular IQL", extraction="argmax of advantage-weighted counts (unmasked)",
                        stochastic=False, mask_source="none", mask_use="none", information="logged transitions", **common)
        register_policy(f"{prefix}Tabular IQL [masked greedy]", seed, iql["masked_greedy"], family="Tabular IQL",
                        extraction="argmax of advantage-weighted scores over admissible actions", stochastic=False,
                        mask_source="released benchmark mask", mask_use="extraction",
                        information="logged transitions + released mask", parent=f"{prefix}Tabular IQL",
                        notes=f"global-frequency fallback in {iql['masked_fallback_states']} states without a logged admissible action",
                        **common)
        register_policy(f"{prefix}Tabular IQL [stochastic AW]", seed, iql["aw_policy"], family="Tabular IQL",
                        extraction="advantage-weighted count distribution", stochastic=True, mask_source="none",
                        mask_use="none", information="logged transitions", parent=f"{prefix}Tabular IQL", **common)
        register_policy(f"{prefix}Tabular BCQ (tau={CFG.bcq_threshold})", seed, bcq["deterministic"], scores=bcq["q"],
                        family="Tabular BCQ", extraction="greedy over behaviour-thresholded actions", stochastic=False,
                        mask_source="dataset-estimated (behaviour ratio >= tau)", mask_use="training+extraction",
                        information="logged transitions only", **common)
        register_policy(f"{prefix}Tabular BCQ (tau={CFG.bcq_threshold}) [masked greedy]", seed, bcq["masked_greedy"],
                        family="Tabular BCQ", extraction="greedy over (behaviour-thresholded ∩ released) actions",
                        stochastic=False, mask_source="dataset-estimated + released benchmark mask",
                        mask_use="training+extraction", information="logged transitions + released mask",
                        parent=f"{prefix}Tabular BCQ (tau={CFG.bcq_threshold})", **common)
        if group == "main":
            save_log("Tabular CQL", seed, cql["log"])
            save_log("Tabular IQL", seed, iql["log"])
            save_log("Tabular BCQ", seed, bcq["log"])


if EXPERIMENTS["main_tabular"]:
    run_tabular_suite(MAIN, CFG.seeds, group="main")
    print("Tabular main study complete:", [n for n in REGISTRY if n.startswith("Tabular")])

# %% [markdown]
# ## 16. Main study — Deep CQL, legacy hard AM-CQL, repaired hard designs A and B, soft AM-CQL (main seeds: 0–9 in the full profile, 0–4 in lite)

# %%
# ============================================================
# 16. Main study: neural critics
# ============================================================
EXECLOG.mark("16. Main study: neural critics")
CRITICS: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
MARGIN_ROWS: List[dict] = []


_TAB_TRAINERS = {"cql": train_tabular_cql, "iql": train_tabular_iql, "bcq": train_tabular_bcq}


def get_tab(kind: str, bundle: DataBundle, seed: int) -> Dict[str, Any]:
    """Identity-checked tabular result: in-memory, else loaded from the store (trained only if allowed)."""
    k = (kind, bundle.name, seed)
    if k not in TAB:
        TAB[k] = _TAB_TRAINERS[kind](bundle, seed)
    return TAB[k]


def get_critic(key: str, bundle: DataBundle, seed: int) -> Dict[str, Any]:
    """Identity-checked critic result: in-memory, else loaded from the store (trained only if allowed)."""
    k = (key, bundle.name, seed)
    if k not in CRITICS:
        CRITICS[k] = train_critic(SPEC_BY_KEY[key], bundle, seed)
    return CRITICS[k]


def masked_export_flag(spec: CriticSpec, suffix: str) -> bool:
    return suffix == "[masked greedy]" or (suffix == "" and spec.masked_extraction)


def critic_mask_meta(spec: CriticSpec, suffix: str) -> Tuple[str, str]:
    masked_export = masked_export_flag(spec, suffix)
    if spec.soft:
        return "released benchmark mask", "training penalty" + (" + extraction" if masked_export else "")
    train = [u for u, on in (("backup", spec.masked_backup), ("regulariser", spec.reg_domain != "full")) if on]
    if not train:
        return ("released benchmark mask", "extraction") if masked_export else ("none", "none")
    return "released benchmark mask", ("training+extraction" if masked_export else "training (" + "+".join(train) + ")")


def register_critic(res: Dict[str, Any], group: str, prefix: str = "", dataset_label: str = "main") -> None:
    spec = res["spec"]
    for suffix, (pi, extraction, stochastic) in critic_extractions(res).items():
        name = f"{prefix}{spec.display}{(' ' + suffix) if suffix else ''}"
        src, use = critic_mask_meta(spec, suffix)
        register_policy(name, res["seed"], pi, scores=np.minimum(res["q1"], res["q2"]) if suffix == "" else None,
                        family=f"Neural critic ({spec.key})", extraction=extraction, stochastic=stochastic,
                        mask_source=src, mask_use=use, information="logged transitions + 47-d features"
                        + (" + released mask" if src != "none" else ""), critic=spec.key, group=group,
                        dataset=dataset_label, parent="" if suffix == "" else f"{prefix}{spec.display}",
                        notes=spec.status)
        if not stochastic and not masked_export_flag(spec, suffix):
            scores = 0.5 * (res["q1"] + res["q2"]) if "mean critic" in suffix else np.minimum(res["q1"], res["q2"])
            MARGIN_ROWS.append({**tie_and_margin_report(scores, name), "seed": res["seed"]})


def run_critics(specs: Sequence[CriticSpec], bundle: DataBundle, seeds: Sequence[int], group: str,
                prefix: str = "", tag: str = "critic", dataset_label: str = "main") -> None:
    for seed in seeds:
        for spec in specs:
            res = train_critic(spec, bundle, seed, tag=tag)
            CRITICS[(spec.key, bundle.name, seed)] = res
            register_critic(res, group=group, prefix=prefix, dataset_label=dataset_label)
            if group == "main":
                save_log(spec.display, seed, res["log"])
                res["diag"].assign(variant=spec.key, seed=seed).to_csv(
                    LOG_DIR / f"diag_{slug(spec.key)}__seed{seed}.csv", index=False)


if EXPERIMENTS["main_neural_critics"]:
    run_critics(MAIN_CRITIC_SPECS, MAIN, CFG.seeds, group="main")
    print("Neural critic main study complete.")

# %% [markdown]
# ## 17. Main study — matched-capacity direct categorical policies (main seeds: 0–9 in the full profile, 0–4 in lite)

# %%
# ============================================================
# 17. Main study: direct categorical policies
# ============================================================
EXECLOG.mark("17. Main study: direct categorical policies")
PNETS: Dict[Tuple[str, str, int], Dict[str, Any]] = {}


def register_policy_net(res: Dict[str, Any], label: str, seed: int, group: str, dataset_label: str = "main",
                        information: str = "logged state-action pairs + 47-d features") -> None:
    probs = res["probs"]
    common = dict(group=group, dataset=dataset_label, family="Direct categorical")
    register_policy(f"{label} [stochastic]", seed, probs, extraction="network softmax", stochastic=True,
                    mask_source="none", mask_use="none", information=information, **common)
    register_policy(f"{label} [greedy]", seed, one_hot_policy(probs.argmax(axis=1)), extraction="unmasked argmax",
                    stochastic=False, mask_source="none", mask_use="none", information=information,
                    parent=f"{label} [stochastic]", **common)
    register_policy(f"{label} [masked greedy]", seed, greedy_policy(probs, masked=True),
                    extraction="argmax over admissible actions", stochastic=False,
                    mask_source="released benchmark mask", mask_use="extraction",
                    information=information + " + released mask", parent=f"{label} [stochastic]", **common)


if EXPERIMENTS["main_policy_nets"]:
    for seed in CFG.seeds:
        bc_net = train_policy_net(MAIN, seed, mode="bc")
        PNETS[("bc", "main", seed)] = bc_net
        register_policy_net(bc_net, "Neural BC (matched capacity)", seed, group="main")
        iql = get_tab("iql", MAIN, seed)
        adv = iql["q"] - iql["v"][:, None]
        aw_net = train_policy_net(MAIN, seed, mode="aw", advantages=adv)
        PNETS[("aw", "main", seed)] = aw_net
        register_policy_net(aw_net, "AW categorical imitation", seed, group="main",
                            information="logged pairs + tabular-IQL advantages + 47-d features")
        # Same critic scores as the primary diffusion policy (M06: preserve the same mask and critic scores).
        rB = get_critic(PREREG["primary_hard_critic"], MAIN, seed)
        G_B_aw = critic_ensemble(np.minimum(rB["q1"], rB["q2"]), get_tab("cql", MAIN, seed)["q"])
        adv_B = np.where(OPERATIONAL_MASK, G_B_aw, -1e3)   # standardised over admissible actions; inadmissible -> weight exp(-20)
        awB = train_policy_net(MAIN, seed, mode="aw", advantages=adv_B, tag="policy_net_awB")
        PNETS[("awB", "main", seed)] = awB
        register_policy_net(awB, "AW categorical imitation [critic B + tabular CQL]", seed, group="main",
                            information="logged pairs + critic-B/tabular-CQL ensemble scores + 47-d features")
        save_log("Neural BC", seed, bc_net["log"])
        save_log("AW categorical imitation", seed, aw_net["log"])
    print("Direct categorical policies complete.")

# %% [markdown]
# ## 18. Main study — diffusion priors, critic-guided policies, and inference-only ablations (main seeds: 0–9 in the full profile, 0–4 in lite)
#
# The denoiser is trained once per seed; every guided variant reuses its exact reverse kernels, so all
# inference ablations (Tables 6–7 of the original study, plus the new masking/initialisation/guidance-placement
# controls) are reproducible here rather than in an unreleased script. Variants guided by the legacy hard
# critic reproduce the original method and are labelled as such; the preregistered primary policy uses the
# repaired design-B critic.

# %%
# ============================================================
# 18. Main study: diffusion + inference ablations
# ============================================================
EXECLOG.mark("18. Main study: diffusion")
DIFFUSION_TRACES: Dict[str, Dict[int, List[float]]] = {}
DIFF_PRIORS: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
G_PRIMARY = CFG.diffusion_guidance_scale


def critic_qmin(key: str, bundle_name: str, seed: int) -> np.ndarray:
    r = CRITICS[(key, bundle_name, seed)]
    return np.minimum(r["q1"], r["q2"])


def register_diffusion(name: str, seed: int, pi: np.ndarray, trace: Optional[List[float]], group: str,
                       extraction: str, critic: str, stochastic: bool = False, mask_use: str = "inference (every reverse step)",
                       mask_source: str = "released benchmark mask", dataset_label: str = "main", notes: str = "") -> None:
    register_policy(name, seed, pi, family="Categorical diffusion", extraction=extraction, stochastic=stochastic,
                    mask_source=mask_source, mask_use=mask_use,
                    information="logged pairs + 47-d features + critics" + (" + released mask" if mask_source != "none" else ""),
                    critic=critic, group=group, dataset=dataset_label, notes=notes)
    if trace is not None:
        DIFFUSION_TRACES.setdefault(name, {})[seed] = trace


def run_diffusion_suite(bundle: DataBundle, seed: int, group: str, prefix: str = "", full_ablations: bool = True,
                        T: Optional[int] = None, dataset_label: str = "main", tab_bundle: Optional[DataBundle] = None,
                        tag: str = "diffusion") -> Dict[str, Any]:
    T = CFG.diffusion_T if T is None else T
    tab_bundle = bundle if tab_bundle is None else tab_bundle
    # Upstream objects are loaded from identity-checked checkpoints when their blocks were switched off.
    get_tab("cql", tab_bundle, seed)
    get_critic(PREREG["primary_hard_critic"], bundle, seed)
    if bundle.name == "main":
        get_critic("hard_legacy", bundle, seed)
        get_critic("hard_fullcql", bundle, seed)
    prior = train_diffusion_prior(bundle, seed, T=T, tag=tag)
    DIFF_PRIORS[(bundle.name, seed, T)] = {k: prior[k] for k in ("log", "meta", "label")}
    cache = ReverseCache(prior["model"], prior["diffusion"], bundle.all_x)
    tab_q = TAB[("cql", tab_bundle.name, seed)]["q"]
    qB = critic_qmin(PREREG["primary_hard_critic"], bundle.name, seed)
    G_B = critic_ensemble(qB, tab_q)
    t0 = time.perf_counter()
    pi_B, tr_B = guided_reverse_policy(cache, G_B, G_PRIMARY)
    construct_s = time.perf_counter() - t0
    name_primary = f"{prefix}{PRIMARY_DIFFUSION}"
    register_diffusion(name_primary, seed, pi_B, tr_B, group, "guided reverse chain, deterministic final",
                       "hard_admrows + tabular CQL", dataset_label=dataset_label)
    out = {"cache_s": cache.construction_s, "construct_s": construct_s}
    if ("hard_legacy", bundle.name, seed) in CRITICS:
        G_L = critic_ensemble(critic_qmin("hard_legacy", bundle.name, seed), tab_q)
        pi_L, tr_L = guided_reverse_policy(cache, G_L, G_PRIMARY)
        register_diffusion(f"{prefix}Diffusion-AM-CQL [legacy critic]", seed, pi_L, tr_L, group,
                           "guided reverse chain, deterministic final (original method)", "hard_legacy + tabular CQL",
                           dataset_label=dataset_label, notes="historical / failure-mode critic")
    if ("hard_fullcql", bundle.name, seed) in CRITICS:
        G_A = critic_ensemble(critic_qmin("hard_fullcql", bundle.name, seed), tab_q)
        pi_A, tr_A = guided_reverse_policy(cache, G_A, G_PRIMARY)
        register_diffusion(f"{prefix}Diffusion-AM-CQL [critic A]", seed, pi_A, tr_A, group,
                           "guided reverse chain, deterministic final", "hard_fullcql + tabular CQL", dataset_label=dataset_label)
    register_policy(f"{prefix}Masked critic ensemble [critic B + tabular CQL]", seed, one_hot_policy(masked_argmax(G_B)),
                    family="Critic only", extraction="masked argmax of the standardised critic ensemble (no diffusion)",
                    stochastic=False, mask_source="released benchmark mask", mask_use="extraction",
                    information="critics + released mask", critic="hard_admrows + tabular CQL", group=group,
                    dataset=dataset_label)
    if ("bc", bundle.name, seed) in PNETS:
        register_policy(f"{prefix}Neural BC prior x critic [critic B + tabular CQL]", seed,
                        prior_times_critic(PNETS[("bc", bundle.name, seed)]["probs"], G_B, G_PRIMARY),
                        family="Prior x critic", extraction="argmax of m * neural-BC prior * exp(g G)", stochastic=False,
                        mask_source="released benchmark mask", mask_use="extraction",
                        information="neural BC + critics + released mask", critic="hard_admrows + tabular CQL",
                        group=group, dataset=dataset_label)
    if full_ablations:
        bc_counts = train_behavior_cloning(bundle)["stochastic"]
        register_policy(f"{prefix}Tabular BC prior x critic [critic B + tabular CQL]", seed,
                        prior_times_critic(bc_counts, G_B, G_PRIMARY), family="Prior x critic",
                        extraction="argmax of m * empirical-BC prior * exp(g G)", stochastic=False,
                        mask_source="released benchmark mask", mask_use="extraction",
                        information="counts + critics + released mask", critic="hard_admrows + tabular CQL", group=group,
                        dataset=dataset_label)
        G_tab = critic_ensemble(tab_q)
        G_Bonly = critic_ensemble(qB)
        variants = {
            "Diffusion [masked, no guidance (g=0)]": dict(G=None, g=0.0, critic="none"),
            "Diffusion [critic B only]": dict(G=G_Bonly, g=G_PRIMARY, critic="hard_admrows"),
            "Diffusion [tabular CQL critic only]": dict(G=G_tab, g=G_PRIMARY, critic="tabular CQL"),
            f"{PRIMARY_DIFFUSION} [stochastic final]": dict(G=G_B, g=G_PRIMARY, critic="hard_admrows + tabular CQL", det=False),
            f"{PRIMARY_DIFFUSION} [init uniform over 25]": dict(G=G_B, g=G_PRIMARY, critic="hard_admrows + tabular CQL", init="uniform_all"),
            f"{PRIMARY_DIFFUSION} [guidance at final step only]": dict(G=G_B, g=G_PRIMARY, critic="hard_admrows + tabular CQL", gmode="final_only"),
            "Diffusion prior [final-step mask, no guidance]": dict(G=None, g=0.0, critic="none", init="uniform_all", mmode="final_only", gmode="none"),
            "Diffusion prior x critic B [final-step mask + guidance]": dict(G=G_B, g=G_PRIMARY, critic="hard_admrows + tabular CQL", init="uniform_all", mmode="final_only", gmode="final_only"),
            "Diffusion prior [unmasked, unguided, stochastic]": dict(G=None, g=0.0, critic="none", init="uniform_all", mmode="none", gmode="none", det=False),
            "Diffusion prior [unmasked, unguided, greedy]": dict(G=None, g=0.0, critic="none", init="uniform_all", mmode="none", gmode="none"),
        }
        if ("hard_legacy", bundle.name, seed) in CRITICS:
            G_L = critic_ensemble(critic_qmin("hard_legacy", bundle.name, seed), tab_q)
            G_Lonly = critic_ensemble(critic_qmin("hard_legacy", bundle.name, seed))
            variants.update({
                "Diffusion [legacy hard critic only]": dict(G=G_Lonly, g=G_PRIMARY, critic="hard_legacy"),
                "Diffusion-AM-CQL [legacy critic] [stochastic final]": dict(G=G_L, g=G_PRIMARY, critic="hard_legacy + tabular CQL", det=False),
            })
            register_policy(f"{prefix}Masked critic ensemble [legacy critic + tabular CQL]", seed,
                            one_hot_policy(masked_argmax(G_L)), family="Critic only",
                            extraction="masked argmax of the standardised critic ensemble (no diffusion)",
                            stochastic=False, mask_source="released benchmark mask", mask_use="extraction",
                            information="critics + released mask", critic="hard_legacy + tabular CQL", group=group,
                            dataset=dataset_label)
            for g in CFG.guidance_grid:
                variants[f"Diffusion-AM-CQL [legacy critic] [g={g}]"] = dict(G=G_L, g=g, critic="hard_legacy + tabular CQL")
        for g in CFG.guidance_grid:
            variants[f"{PRIMARY_DIFFUSION} [g={g}]"] = dict(G=G_B, g=g, critic="hard_admrows + tabular CQL")
        for vname, v in variants.items():
            mmode = v.get("mmode", "every_step")
            pi, tr = guided_reverse_policy(cache, v["G"], v["g"], init=v.get("init", "uniform_admissible"),
                                           mask_mode=mmode, guidance_mode=v.get("gmode", "every_step"),
                                           deterministic=v.get("det", True))
            register_diffusion(f"{prefix}{vname}", seed, pi, tr, group if group != "main" else "inference_ablation",
                               f"init={v.get('init', 'uniform_admissible')}, mask={mmode}, guidance={v.get('gmode', 'every_step')}, g={v['g']}",
                               v["critic"], stochastic=not v.get("det", True),
                               mask_use={"every_step": "inference (every reverse step)", "final_only": "inference (final step only)",
                                         "none": "none"}[mmode],
                               mask_source="none" if mmode == "none" else "released benchmark mask", dataset_label=dataset_label)
    out["lookup_us"] = lookup_latency_us(pi_B)
    if group == "main" and seed == CFG.seeds[0]:
        out["ondemand_ms"] = ondemand_reverse_latency_ms(prior["model"], prior["diffusion"], bundle.all_x)
    del cache
    gc.collect()
    return out


DIFFUSION_COST: List[dict] = []
if EXPERIMENTS["main_diffusion"]:
    for seed in CFG.seeds:
        c = run_diffusion_suite(MAIN, seed, group="main", full_ablations=EXPERIMENTS["inference_ablations"])
        DIFFUSION_COST.append({"seed": seed, **c})
        save_log("Diffusion prior", seed, DIFF_PRIORS[("main", seed, CFG.diffusion_T)]["log"])
        for nm in (PRIMARY_DIFFUSION, "Diffusion-AM-CQL [legacy critic]"):
            if nm in DIFFUSION_TRACES and seed in DIFFUSION_TRACES[nm]:
                save_json({"policy": nm, "seed": seed, "guidance_scale": G_PRIMARY, "T": CFG.diffusion_T,
                           "max_inadmissible_mass_each_reverse_step": DIFFUSION_TRACES[nm][seed]},
                          TABLE_DIR / f"diffusion_reverse_support_{slug(nm)}_seed{seed}.json")
    pd.DataFrame(DIFFUSION_COST).to_csv(TABLE_DIR / "diffusion_policy_construction_cost.csv", index=False)
    print("Diffusion main study complete.")
