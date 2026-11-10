# %% [markdown]
# ## 4. Benchmark audit and construction provenance (audit M02, M08, m04)
#
# * Admissible-set statistics and the SOFA association are reported with **Pearson r** and
#   **Spearman ρ** labelled separately (the previous notebook printed an unlabelled Pearson value while
#   the original study quoted Spearman; both are legitimate but different statistics).
# * The benchmark's construction scripts are fetched from the public repository at a pinned commit and
#   **parsed as text / AST only (never executed)** to verify: the support threshold inequality, the
#   inadmissible-row mapping method, the action definition, the 47-feature order, and whether
#   treatment variables are also clustering features. If the fetch fails (offline), an embedded
#   transcription made on 17 Oct 2026 from the same commit is used and flagged as such.

# %%
# ============================================================
# 4. Benchmark audit + construction provenance
# ============================================================
EXECLOG.mark("4. Benchmark audit and provenance")

admissible_sizes = SUPPORT_MASK.sum(axis=1)
live_sizes = admissible_sizes[LIVE]
pearson_r, pearson_p = stats.pearsonr(BENCH.sofa_scores[LIVE], live_sizes)
spearman_rho, spearman_p = stats.spearmanr(BENCH.sofa_scores[LIVE], live_sizes)

BENCHMARK_AUDIT = {
    "num_states": S_N,
    "num_actions": A_N,
    "feature_dim": BENCH.feature_dim,
    "terminal_states": {"death": BENCH.death_state, "survival": BENCH.survival_state, "s_inf": BENCH.sinf_state},
    "live_states": int(LIVE.sum()),
    "admissible_pairs_live": int(SUPPORT_MASK[LIVE].sum()),
    "admissible_per_state_min": int(live_sizes.min()),
    "admissible_per_state_max": int(live_sizes.max()),
    "admissible_per_state_mean": float(live_sizes.mean()),
    "states_with_single_admissible_action": int((live_sizes == 1).sum()),
    "sofa_vs_admissible_count_pearson_r": float(pearson_r),
    "sofa_vs_admissible_count_pearson_p": float(pearson_p),
    "sofa_vs_admissible_count_spearman_rho": float(spearman_rho),
    "sofa_vs_admissible_count_spearman_p": float(spearman_p),
    "benchmark_identity": BENCH.identity,
    "env_metadata": BENCH.env_metadata,
}
print(json.dumps(BENCHMARK_AUDIT, indent=2, default=_json_default))
save_json(BENCHMARK_AUDIT, TABLE_DIR / "benchmark_audit.json")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(live_sizes, bins=np.arange(live_sizes.min(), live_sizes.max() + 2) - 0.5)
ax.set_xlabel("Admissible actions in the released benchmark-support mask")
ax.set_ylabel("Live (non-terminal) states")
ax.set_title("Released benchmark-support set sizes (713 live states)")
fig.tight_layout()
fig.savefig(FIG_DIR / "benchmark_support_set_sizes.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------- Construction provenance (parsed, never executed) ----------
import ast
import re

PROVENANCE_REF = {
    "repository": "icu-sepsis/icu-sepsis",
    "commit": "c9a6d19b85d34883304afaf7cbe1ff02ede9e12b",
    "path": "packages/icu_sepsis_helpers/icu_sepsis_helpers/mdp_creation",
    "sha256": {
        "create_rl_table.py": "7f3c83b4845cec29dcc123d4cc8321223283a8827449b6e894357cc6aebea305",
        "create_matrices.py": "cb4db6738c70a245cbffe90cc792be0e03d89ca8954be0442ad2a5083564d4cd",
        "parse_matrices.py": "6bd716ab2bf502656d85c8292588f4e7999dde69fe3fbce3f2baf1ecb75f8a73",
    },
    "note": ("The helper package is not shipped in the icu-sepsis 2.0.1 wheel. The commit is the repository "
             "HEAD at audit time (2024-10-23); confirm it corresponds to the released 2.0.1 assets."),
}

# Embedded transcription (fallback when offline) of create_rl_table.py at the commit above.
EMBEDDED_FEATURE_ORDER = {
    "colbin": ["gender", "mechvent", "max_dose_vaso", "re_admission"],
    "colnorm": ["age", "Weight_kg", "GCS", "HR", "SysBP", "MeanBP", "DiaBP", "RR", "Temp_C", "FiO2_1",
                "Potassium", "Sodium", "Chloride", "Glucose", "Magnesium", "Calcium", "Hb", "WBC_count",
                "Platelets_count", "PTT", "PT", "Arterial_pH", "paO2", "paCO2", "Arterial_BE", "HCO3",
                "Arterial_lactate", "SOFA", "SIRS", "Shock_Index", "PaO2_FiO2", "cumulated_balance"],
    "collog": ["SpO2", "BUN", "Creatinine", "SGOT", "SGPT", "Total_bili", "INR",
               "input_total", "input_4hourly", "output_total", "output_4hourly"],
}
EMBEDDED_FACTS = {
    "threshold_rule_in_code": "s_a_mat <= transition_threshold  -> zeroed (admissible iff count > threshold)",
    "action_definition": "action = n_action_levels * level(input_4hourly) + level(max_dose_vaso)",
    "treatment_columns_are_clustering_features": True,
    "inadmissible_row_method": "uniform_unweighted (mean of admissible rows)",
    "feature_transforms": "binary: x-0.5; normal: z-score; log: z-score(log(x+0.1))",
    "construction_quirks": [
        "all_cols[3] ('re_admission') receives +0.6 after normalisation (code comment says 'max dose?')",
        "all_cols[44] ('input_4hourly') is multiplied by 2 after normalisation ('increase weight')",
    ],
}


def fetch_text(url: str, timeout: float = 15.0) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except Exception as exc:  # offline or blocked
        print("  fetch failed:", url, "->", type(exc).__name__)
        return None


def parse_list_assignments(source: str, names: Sequence[str]) -> Dict[str, List[str]]:
    out = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in names and tgt.id not in out:
                    out[tgt.id] = list(ast.literal_eval(node.value))
    return out


def nospace(s: str) -> str:
    return re.sub(r"\s+", "", s)


PROVENANCE = {"reference": PROVENANCE_REF, "source": "embedded transcription", "checks": {}}
fetched = {}
if VERIFY_PROVENANCE_ONLINE:
    base_url = f"https://raw.githubusercontent.com/{PROVENANCE_REF['repository']}/{PROVENANCE_REF['commit']}/{PROVENANCE_REF['path']}"
    for fname in PROVENANCE_REF["sha256"]:
        txt = fetch_text(f"{base_url}/{fname}")
        if txt is not None:
            fetched[fname] = txt

if len(fetched) == len(PROVENANCE_REF["sha256"]):
    PROVENANCE["source"] = "fetched at runtime from the pinned commit"
    PROVENANCE["fetched_sha256"] = {k: sha256_bytes(v.encode("utf-8")) for k, v in fetched.items()}
    PROVENANCE["sha256_matches_reference"] = {
        k: PROVENANCE["fetched_sha256"][k] == PROVENANCE_REF["sha256"][k] for k in fetched}
    lists = parse_list_assignments(fetched["create_rl_table.py"], ["colbin", "colnorm", "collog"])
    rl = nospace(fetched["create_rl_table.py"])
    mx = nospace(fetched["create_matrices.py"])
    pm = nospace(fetched["parse_matrices.py"])
    PROVENANCE["checks"] = {
        "feature_lists_match_embedded": lists == EMBEDDED_FEATURE_ORDER,
        "threshold_rule_count_le_threshold_removed": "below_thresh=np.where(s_a_mat<=transition_threshold)" in mx,
        "fluid_level_from_input_4hourly": "ac1=get_action_nums(Dataset_Raw,'input_4hourly',n_action_levels)" in rl,
        "vaso_level_from_max_dose_vaso": "ac2=get_action_nums(Dataset_Raw,'max_dose_vaso',n_action_levels)" in rl,
        "action_is_fluid_major": "Dataset['action']=n_action_levels*ac1+ac2" in rl,
        "states_clustered_on_all_47_features": ("cluster_states(Dataset,cols" in rl
                                                 and "n_action_levels,all_cols,n_clustering" in rl),
        "uniform_unweighted_mapping_implemented": "elifmethod=='uniform_unweighted':" in pm,
        "readmission_plus_0.6_quirk": "Dataset[all_cols[4-1]]=Dataset[all_cols[4-1]]+0.6" in rl,
        "input_4hourly_times_2_quirk": "Dataset[all_cols[45-1]]=Dataset[all_cols[45-1]]*2" in rl,
    }
    FEATURE_ORDER = lists if lists else EMBEDDED_FEATURE_ORDER
else:
    PROVENANCE["checks"] = {"note": "offline: embedded transcription used; verify manually", **EMBEDDED_FACTS}
    FEATURE_ORDER = EMBEDDED_FEATURE_ORDER
PROVENANCE["embedded_facts"] = EMBEDDED_FACTS
PROVENANCE["metadata_threshold"] = BENCH.env_metadata.get("threshold")
PROVENANCE["metadata_action_map_method"] = BENCH.env_metadata.get("action_map_method")
print(json.dumps(PROVENANCE, indent=2, default=_json_default))
save_json(PROVENANCE, TABLE_DIR / "benchmark_construction_provenance.json")

FEATURE_NAMES = FEATURE_ORDER["colbin"] + FEATURE_ORDER["colnorm"] + FEATURE_ORDER["collog"]
assert len(FEATURE_NAMES) == BENCH.feature_dim == 47, (len(FEATURE_NAMES), BENCH.feature_dim)

TREATMENT_RELATION = {
    "input_4hourly": ("defines the logged fluid level of the same 4-h bloc", "high"),
    "max_dose_vaso": ("defines the logged vasopressor level of the same 4-h bloc", "high"),
    "input_total": ("cumulative fluid input, may include the current bloc", "high"),
    "cumulated_balance": ("cumulative fluid balance, may include the current bloc", "medium"),
    "output_4hourly": ("urine output of the bloc, plausibly treatment-affected", "medium"),
    "output_total": ("cumulative urine output, plausibly treatment-affected", "medium"),
    "mechvent": ("concurrent intervention indicator", "medium"),
}
centroids_live = BENCH.state_features[LIVE].astype(np.float64)
feat_rows = []
for j, name in enumerate(FEATURE_NAMES):
    group = ("binary (x - 0.5)" if name in FEATURE_ORDER["colbin"]
             else "z-score" if name in FEATURE_ORDER["colnorm"] else "z-score(log(x + 0.1))")
    rel, risk = TREATMENT_RELATION.get(name, ("physiological / demographic", "low (verify measurement time within bloc)"))
    feat_rows.append({
        "index_0based": j,
        "index_1based": j + 1,
        "feature": name,
        "transform_group": group,
        "treatment_relation": rel,
        "timing_risk": risk,
        "is_clustering_feature": True,
        "centroid_mean_live": float(centroids_live[:, j].mean()),
        "centroid_sd_live": float(centroids_live[:, j].std()),
        "centroid_min_live": float(centroids_live[:, j].min()),
        "centroid_max_live": float(centroids_live[:, j].max()),
    })
FEATURE_PROVENANCE = pd.DataFrame(feat_rows)
FEATURE_PROVENANCE.to_csv(TABLE_DIR / "feature_provenance_timing.csv", index=False)
display(FEATURE_PROVENANCE[["index_1based", "feature", "transform_group", "treatment_relation", "timing_risk"]])

TREATMENT_FEATURES_DIRECT = tuple(FEATURE_NAMES.index(n) for n in ("input_4hourly", "max_dose_vaso", "input_total"))
TREATMENT_FEATURES_RELATED = tuple(sorted(set(TREATMENT_FEATURES_DIRECT) | {
    FEATURE_NAMES.index(n) for n in ("cumulated_balance", "output_4hourly", "output_total", "mechvent")}))
print("Centroids are in the benchmark's transformed units (see transform_group); they are not raw clinical units.")
print("Treatment-direct feature indices:", TREATMENT_FEATURES_DIRECT, "| treatment-related:", TREATMENT_FEATURES_RELATED)
