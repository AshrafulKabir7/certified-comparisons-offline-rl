# %% [markdown]
# ## 6. Exploratory data analysis of the frozen dataset, with corrected support coverage (audit M01, m04, m05)
#
# The ICU-Sepsis states are processed **cluster centroids** in transformed units, so every figure
# describes the *processed benchmark and the simulated clinician-policy dataset*, not individual
# MIMIC-III patients, and nothing here is a causal treatment-effect estimate.
#
# **Support-coverage repair (M01).** The previous notebook divided *all* observed state-action pairs
# (6,738, including inadmissible ones) by the number of admissible pairs (2,238), giving 301.07%, and its
# histogram silently dropped ratios above one. Here observed pairs are first intersected with the released
# mask on live states; observed admissible, observed inadmissible, and total observed pairs are separate
# fields; per-pair frequency quantiles are reported because "observed at least once" does not imply
# reliable estimation; and the histogram population is asserted to equal the 713 live states.

# %%
# ============================================================
# 6. EDA of the frozen offline dataset
# ============================================================
EXECLOG.mark("6. EDA")

EDA_DIR = FIG_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)
LEVELS = BENCH.codec.levels


def save_figure(fig, path_stem: Path, dpi: int = 300) -> None:
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)


def boxplot_compat(ax, data, labels, **kwargs):
    """Matplotlib >= 3.9 uses tick_labels; older versions use labels (audit m05)."""
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kwargs)


TRANSITIONS = pd.DataFrame({
    "state": OFFLINE["state"],
    "action": OFFLINE["action"],
    "reward": OFFLINE["reward"].astype(np.float64),
    "next_state": OFFLINE["next_state"],
    "terminated": OFFLINE["terminated"],
    "truncated": OFFLINE["truncated"],
    "cutoff": OFFLINE["cutoff"],
    "episode_id": OFFLINE["episode_id"].astype(np.int64),
    "timestep": OFFLINE["timestep"].astype(np.int64),
    "admissible": OFFLINE["logged_action_admissible"],
})
TRANSITIONS["fluid_level"] = TRANSITIONS["action"] // LEVELS
TRANSITIONS["vasopressor_level"] = TRANSITIONS["action"] % LEVELS
TRANSITIONS["sofa"] = BENCH.sofa_scores[TRANSITIONS["state"].to_numpy()]

EPISODES = (
    TRANSITIONS.groupby("episode_id", sort=True)
    .agg(length=("state", "size"), episode_return=("reward", "sum"), initial_state=("state", "first"),
         final_next_state=("next_state", "last"), terminated=("terminated", "last"),
         censored_trunc=("truncated", "last"), censored_cutoff=("cutoff", "last"),
         initial_sofa=("sofa", "first"), mean_sofa=("sofa", "mean"), max_sofa=("sofa", "max"))
    .reset_index()
)


def episode_outcome(final_state: int, terminated: bool) -> str:
    if not terminated:
        return "Censored (cutoff / time limit)"
    if final_state == BENCH.survival_state:
        return "Survival"
    if final_state == BENCH.death_state:
        return "Death"
    if final_state == BENCH.sinf_state:
        return "s_inf terminal"
    return "Other terminal"


EPISODES["outcome"] = [episode_outcome(int(s), bool(t)) for s, t in zip(EPISODES["final_next_state"], EPISODES["terminated"])]
TRANSITIONS = TRANSITIONS.merge(EPISODES[["episode_id", "outcome"]], on="episode_id", how="left", validate="many_to_one")

# ---------- Corrected support audit (M01) ----------
LOGGED_COUNTS = np.zeros((S_N, A_N), dtype=np.int64)
np.add.at(LOGGED_COUNTS, (OFFLINE["state"], OFFLINE["action"]), 1)
allowed = SUPPORT_MASK & LIVE[:, None]
observed = LOGGED_COUNTS > 0
covered = observed & allowed
admissible_pair_coverage = covered.sum() / allowed.sum()
per_state_coverage = covered[LIVE].sum(axis=1) / allowed[LIVE].sum(axis=1)
per_state_inadm_frac = np.divide(
    (LOGGED_COUNTS * ~SUPPORT_MASK)[LIVE].sum(axis=1), LOGGED_COUNTS[LIVE].sum(axis=1),
    out=np.zeros(int(LIVE.sum())), where=LOGGED_COUNTS[LIVE].sum(axis=1) > 0)
assert 0.0 <= admissible_pair_coverage <= 1.0
assert np.all((per_state_coverage >= 0.0) & (per_state_coverage <= 1.0))
assert np.all(covered[LIVE].sum(axis=1) <= allowed[LIVE].sum(axis=1))
hist_cov, hist_edges = np.histogram(per_state_coverage, bins=np.linspace(0, 1, 21))
assert hist_cov.sum() == LIVE.sum(), "Coverage histogram does not contain every live state."

pair_freq = LOGGED_COUNTS[allowed]
quantile_levels = [0.0, 0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.99, 1.0]
SUPPORT_AUDIT = {
    "live_states": int(LIVE.sum()),
    "observed_live_states": int((LOGGED_COUNTS[LIVE].sum(axis=1) > 0).sum()),
    "total_unique_observed_pairs": int(observed.sum()),
    "observed_admissible_pairs": int(covered.sum()),
    "observed_inadmissible_pairs": int((observed & ~SUPPORT_MASK & LIVE[:, None]).sum()),
    "admissible_pairs": int(allowed.sum()),
    "admissible_pair_coverage": float(admissible_pair_coverage),
    "states_with_full_admissible_coverage": int((per_state_coverage == 1.0).sum()),
    "admissible_pair_count_quantiles": {f"q{int(100 * q):02d}": float(np.quantile(pair_freq, q)) for q in quantile_levels},
    "admissible_pairs_with_fewer_than": {str(k): int((pair_freq < k).sum()) for k in (1, 5, 10, 21, 50, 100)},
    "logged_transitions": int(len(OFFLINE["state"])),
    "inadmissible_logged_transitions": int((~OFFLINE["logged_action_admissible"]).sum()),
    "inadmissible_logged_fraction": float((~OFFLINE["logged_action_admissible"]).mean()),
    "states_with_any_inadmissible_logged_action": int(((LOGGED_COUNTS * ~SUPPORT_MASK)[LIVE].sum(axis=1) > 0).sum()),
    "coverage_histogram_counts": hist_cov.tolist(),
    "coverage_histogram_edges": hist_edges.tolist(),
    "note": ("Coverage counts a pair once it is observed; per-pair counts show estimation support. "
             "Inadmissible transitions are retained in the data (not deleted to make a statistic valid)."),
}
save_json(SUPPORT_AUDIT, TABLE_DIR / "corrected_support_audit.json")
print(json.dumps(SUPPORT_AUDIT, indent=2))

STATE_SUPPORT = pd.DataFrame({
    "state": np.arange(S_N),
    "is_live": LIVE,
    "visits": LOGGED_COUNTS.sum(axis=1),
    "admissible_actions": SUPPORT_MASK.sum(axis=1) * LIVE,
    "observed_admissible_actions": covered.sum(axis=1),
    "observed_inadmissible_actions": (observed & ~SUPPORT_MASK & LIVE[:, None]).sum(axis=1),
    "min_count_over_admissible_actions": np.where(allowed, LOGGED_COUNTS, np.iinfo(np.int64).max).min(axis=1) * LIVE,
    "sofa": BENCH.sofa_scores,
})
STATE_SUPPORT["admissible_coverage"] = np.where(LIVE, STATE_SUPPORT["observed_admissible_actions"] / np.maximum(STATE_SUPPORT["admissible_actions"], 1), np.nan)
STATE_SUPPORT.to_csv(TABLE_DIR / "eda_state_support.csv", index=False)

ACTION_COUNTS = pd.DataFrame({
    "action": np.arange(A_N),
    "fluid_level": np.arange(A_N) // LEVELS,
    "vasopressor_level": np.arange(A_N) % LEVELS,
    "count": LOGGED_COUNTS.sum(axis=0),
})
ACTION_COUNTS["percentage"] = 100.0 * ACTION_COUNTS["count"] / ACTION_COUNTS["count"].sum()

EDA_SUMMARY = pd.DataFrame([
    ("Transitions", DATASET_REPORT["transitions"]),
    ("Episodes", DATASET_REPORT["episodes"]),
    ("Mean episode length", DATASET_REPORT["mean_length"]),
    ("Median episode length", DATASET_REPORT["median_length"]),
    ("Maximum episode length", DATASET_REPORT["max_length"]),
    ("Survival episodes", DATASET_REPORT["survival_episodes"]),
    ("Death episodes", DATASET_REPORT["death_episodes"]),
    ("s_inf terminal episodes", DATASET_REPORT["s_inf_episodes"]),
    ("Censored episodes (cutoff or time limit)", DATASET_REPORT["censored_episodes"]),
    ("Observed live states", SUPPORT_AUDIT["observed_live_states"]),
    ("Live-state coverage (%)", 100.0 * SUPPORT_AUDIT["observed_live_states"] / SUPPORT_AUDIT["live_states"]),
    ("Unique observed state-action pairs (all)", SUPPORT_AUDIT["total_unique_observed_pairs"]),
    ("Observed admissible pairs", SUPPORT_AUDIT["observed_admissible_pairs"]),
    ("Observed inadmissible pairs", SUPPORT_AUDIT["observed_inadmissible_pairs"]),
    ("Admissible pairs (released mask)", SUPPORT_AUDIT["admissible_pairs"]),
    ("Admissible-pair coverage (%)", 100.0 * SUPPORT_AUDIT["admissible_pair_coverage"]),
    ("Median logged count per admissible pair", SUPPORT_AUDIT["admissible_pair_count_quantiles"]["q50"]),
    ("Admissible pairs with < 21 logged transitions", SUPPORT_AUDIT["admissible_pairs_with_fewer_than"]["21"]),
    ("Logged inadmissible transitions", SUPPORT_AUDIT["inadmissible_logged_transitions"]),
    ("Logged inadmissible fraction (%)", 100.0 * SUPPORT_AUDIT["inadmissible_logged_fraction"]),
], columns=["metric", "value"])
display(EDA_SUMMARY)
EDA_SUMMARY.to_csv(TABLE_DIR / "eda_dataset_summary.csv", index=False)
EPISODES.to_csv(TABLE_DIR / "eda_episode_summary.csv", index=False)
ACTION_COUNTS.to_csv(TABLE_DIR / "eda_action_counts.csv", index=False)
TRANSITIONS.to_csv(TABLE_DIR / "eda_transitions_enriched.csv.gz", index=False, compression="gzip")
print(f"Table 1 reconciliation: admissible pairs = {SUPPORT_AUDIT['admissible_pairs']}, "
      f"admissible-pair coverage = {100 * SUPPORT_AUDIT['admissible_pair_coverage']:.3f}% "
      f"(observed admissible pairs = {SUPPORT_AUDIT['observed_admissible_pairs']}).")
# Values reported in the manuscript's Table 1 / the original notebook output (paper mode only is comparable).
MANUSCRIPT_TABLE1 = {"logged_transitions": 93563, "episodes": 10000, "total_unique_observed_pairs": 6738,
                     "admissible_pairs": 2238, "inadmissible_logged_transitions": 11106, "admissible_pair_coverage": 1.0}
_regen = {**SUPPORT_AUDIT, "episodes": DATASET_REPORT["episodes"]}
TABLE1_RECON = pd.DataFrame([{"quantity": k, "regenerated": float(_regen[k]), "manuscript_or_original": float(v),
                              "difference": float(_regen[k]) - float(v)} for k, v in MANUSCRIPT_TABLE1.items()])
TABLE1_RECON.to_csv(TABLE_DIR / "table1_reconciliation.csv", index=False)
display(TABLE1_RECON)
if RUN_MODE == "paper" and (TABLE1_RECON["difference"].abs() > 1e-12).any():
    print("WARNING: regenerated dataset statistics differ from manuscript Table 1 / the original run "
          "(see table1_reconciliation.csv).")

# ---------- Figures ----------
outcome_order = ["Survival", "Death", "s_inf terminal", "Censored (cutoff / time limit)", "Other terminal"]
oc = EPISODES["outcome"].value_counts()
labels_present = [x for x in outcome_order if x in oc.index]
fig, ax = plt.subplots(figsize=(7.5, 4.8))
bars = ax.bar(labels_present, [int(oc[x]) for x in labels_present])
for b, x in zip(bars, labels_present):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(oc[x]):,}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Episodes")
ax.set_title("Frozen offline dataset: episode outcomes")
ax.tick_params(axis="x", rotation=15)
save_figure(fig, EDA_DIR / "01_outcome_distribution")

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.hist(EPISODES["length"], bins=np.arange(1, int(EPISODES["length"].max()) + 2) - 0.5)
ax.axvline(EPISODES["length"].median(), linestyle="--", label=f"Median = {EPISODES['length'].median():.1f}")
ax.set_xlabel("Episode length (decisions)")
ax.set_ylabel("Episodes")
ax.set_title("Clinician-policy trajectory length")
ax.legend()
save_figure(fig, EDA_DIR / "02_episode_length_distribution")

grid_pct = 100.0 * LOGGED_COUNTS.sum(axis=0).reshape(LEVELS, LEVELS) / LOGGED_COUNTS.sum()
fig, ax = plt.subplots(figsize=(7.0, 5.8))
im = ax.imshow(grid_pct, origin="lower", aspect="auto", cmap="viridis")
ax.set_xlabel("Vasopressor level")
ax.set_ylabel("IV-fluid level")
ax.set_xticks(range(LEVELS))
ax.set_yticks(range(LEVELS))
ax.set_title("Logged clinician-policy treatment frequency (%)")
fig.colorbar(im, ax=ax).set_label("Transitions (%)")
for i in range(LEVELS):
    for j in range(LEVELS):
        ax.text(j, i, f"{grid_pct[i, j]:.1f}", ha="center", va="center", fontsize=8, color="w")
save_figure(fig, EDA_DIR / "03_clinician_action_grid")

box_labels, box_values = [], []
for label in ["Survival", "Death", "s_inf terminal"]:
    v = EPISODES.loc[EPISODES["outcome"] == label, "initial_sofa"].to_numpy(dtype=np.float64)
    if len(v):
        box_labels.append(label)
        box_values.append(v)
fig, ax = plt.subplots(figsize=(7.5, 4.8))
boxplot_compat(ax, box_values, box_labels, showfliers=False)
ax.set_ylabel("Initial-state mean SOFA")
ax.set_title("Initial severity by simulated outcome")
save_figure(fig, EDA_DIR / "04_initial_sofa_by_outcome")

fig, ax = plt.subplots(figsize=(8.0, 5.0))
traj_rows = []
for label in ["Survival", "Death"]:
    sub = TRANSITIONS[TRANSITIONS["outcome"] == label]
    if sub.empty:
        continue
    g = sub.groupby("timestep")["sofa"].agg(["mean", "std", "count"]).reset_index()
    g = g[g["count"] >= max(20, int(0.01 * max((EPISODES["outcome"] == label).sum(), 1)))].copy()
    if g.empty:
        continue
    g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"])
    traj_rows.append(g.assign(outcome=label))
    ax.plot(g["timestep"], g["mean"], label=label)
    ax.fill_between(g["timestep"].to_numpy(), (g["mean"] - g["ci95"]).to_numpy(), (g["mean"] + g["ci95"]).to_numpy(), alpha=0.18)
ax.set_xlabel("Timestep")
ax.set_ylabel("Mean state SOFA")
ax.set_title("SOFA along simulated clinician-policy episodes")
ax.legend()
save_figure(fig, EDA_DIR / "05_sofa_trajectory_by_outcome")
if traj_rows:
    pd.concat(traj_rows, ignore_index=True).to_csv(TABLE_DIR / "eda_sofa_trajectory_by_outcome.csv", index=False)

sev = TRANSITIONS.copy()
sev["sofa_quartile"] = pd.qcut(sev["sofa"], q=4, duplicates="drop")
INTENSITY_BY_SOFA = (sev.groupby("sofa_quartile", observed=True)
                     .agg(transitions=("action", "size"), mean_fluid_level=("fluid_level", "mean"),
                          mean_vasopressor_level=("vasopressor_level", "mean"), mean_sofa=("sofa", "mean"))
                     .reset_index())
INTENSITY_BY_SOFA["severity_group"] = [f"Q{i + 1}" for i in range(len(INTENSITY_BY_SOFA))]
INTENSITY_BY_SOFA.astype({"sofa_quartile": str}).to_csv(TABLE_DIR / "eda_treatment_intensity_by_sofa.csv", index=False)
fig, ax = plt.subplots(figsize=(7.5, 4.8))
x = np.arange(len(INTENSITY_BY_SOFA))
ax.bar(x - 0.18, INTENSITY_BY_SOFA["mean_fluid_level"], 0.36, label="IV fluid")
ax.bar(x + 0.18, INTENSITY_BY_SOFA["mean_vasopressor_level"], 0.36, label="Vasopressor")
ax.set_xticks(x)
ax.set_xticklabels(INTENSITY_BY_SOFA["severity_group"])
ax.set_xlabel("State SOFA quartile (Q1 lower to Q4 higher)")
ax.set_ylabel("Mean discretised treatment level")
ax.set_title("Logged treatment intensity across severity strata")
ax.legend()
save_figure(fig, EDA_DIR / "06_treatment_intensity_by_sofa")

top = STATE_SUPPORT[STATE_SUPPORT["visits"] > 0].sort_values("visits", ascending=False).head(25).sort_values("visits")
fig, ax = plt.subplots(figsize=(8.0, 7.0))
ax.barh(top["state"].astype(str), top["visits"])
ax.set_xlabel("Logged transitions")
ax.set_ylabel("State ID")
ax.set_title("Top 25 most visited states")
save_figure(fig, EDA_DIR / "07_top_state_visitation")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
axes[0].hist(per_state_coverage, bins=np.linspace(0, 1, 21))
axes[0].set_xlabel("Observed admissible actions / admissible actions")
axes[0].set_ylabel("Live states (n = 713)")
axes[0].set_title("(a) Admissible-pair coverage per state")
axes[1].hist(np.log10(np.maximum(pair_freq, 1)), bins=40)
axes[1].axvline(np.log10(21), linestyle="--", color="k", label="21 transitions")
axes[1].set_xlabel("log10 logged transitions per admissible pair")
axes[1].set_ylabel("Admissible pairs")
axes[1].set_title("(b) Estimation support per admissible pair")
axes[1].legend()
axes[2].hist(per_state_inadm_frac, bins=np.linspace(0, 1, 21))
axes[2].set_xlabel("Fraction of logged transitions that are inadmissible")
axes[2].set_ylabel("Live states")
axes[2].set_title("(c) Inadmissible logged actions per state")
save_figure(fig, EDA_DIR / "08_corrected_support_coverage")

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.scatter(BENCH.sofa_scores[LIVE], live_sizes, alpha=0.5, s=18)
ax.set_xlabel("State mean SOFA")
ax.set_ylabel("Admissible actions (released mask)")
ax.set_title(f"Severity vs support-set size (Spearman ρ = {spearman_rho:.3f}; Pearson r = {pearson_r:.3f})")
save_figure(fig, EDA_DIR / "09_sofa_vs_admissible_actions")
pd.DataFrame([{"spearman_rho": spearman_rho, "spearman_p": spearman_p, "pearson_r": pearson_r,
               "pearson_p": pearson_p, "n_states": int(LIVE.sum())}]).to_csv(
    TABLE_DIR / "eda_sofa_admissibility_association.csv", index=False)

Xc = BENCH.state_features[LIVE].astype(np.float64)
Xc = Xc - Xc.mean(axis=0, keepdims=True)
_, sv, vt = np.linalg.svd(Xc, full_matrices=False)
coords = Xc @ vt[:2].T
expl = sv ** 2 / (sv ** 2).sum()
visits_live = STATE_SUPPORT.loc[LIVE, "visits"].to_numpy(dtype=np.float64)
fig, ax = plt.subplots(figsize=(8.0, 6.2))
sc = ax.scatter(coords[:, 0], coords[:, 1], c=BENCH.sofa_scores[LIVE], cmap="viridis", alpha=0.72,
                s=16 + 70 * np.log1p(visits_live) / max(np.log1p(visits_live).max(), 1.0))
ax.set_xlabel(f"PC1 ({100 * expl[0]:.1f}% variance)")
ax.set_ylabel(f"PC2 ({100 * expl[1]:.1f}% variance)")
ax.set_title("State-centroid PCA (mean-centred, benchmark-transformed units)")
fig.colorbar(sc, ax=ax).set_label("Mean SOFA")
save_figure(fig, EDA_DIR / "10_state_centroid_pca")
pd.DataFrame({"state": np.flatnonzero(LIVE), "pc1": coords[:, 0], "pc2": coords[:, 1],
              "sofa": BENCH.sofa_scores[LIVE], "visits": visits_live.astype(int)}).to_csv(
    TABLE_DIR / "eda_state_pca_coordinates.csv", index=False)

fvar = BENCH.state_features[LIVE].var(axis=0)
top_idx = np.argsort(fvar)[-12:]
corr = np.corrcoef(BENCH.state_features[LIVE][:, top_idx].astype(np.float64), rowvar=False)
fig, ax = plt.subplots(figsize=(8.5, 7.0))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
lab = [FEATURE_NAMES[j] for j in top_idx]
ax.set_xticks(range(len(lab)))
ax.set_yticks(range(len(lab)))
ax.set_xticklabels(lab, rotation=45, ha="right")
ax.set_yticklabels(lab)
ax.set_title("Pearson correlation among the 12 highest-variance centroid features")
fig.colorbar(im, ax=ax).set_label("Pearson r")
save_figure(fig, EDA_DIR / "11_top_feature_correlation")
pd.DataFrame(corr, index=lab, columns=lab).to_csv(TABLE_DIR / "eda_top_feature_correlation.csv")

# ---------- Manuscript Figures 2 and 3 (previously produced by unreleased scripts) ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
axes[0, 0].hist(EPISODES["length"], bins=30)
axes[0, 0].set_title("(a) Trajectory length")
axes[0, 0].set_xlabel("Episode length (steps)")
axes[0, 0].set_ylabel("Simulated episodes")
im = axes[0, 1].imshow(grid_pct, origin="lower", cmap="Blues")
for i in range(LEVELS):
    for j in range(LEVELS):
        axes[0, 1].text(j, i, f"{grid_pct[i, j]:.1f}", ha="center", va="center", fontsize=7)
axes[0, 1].set_title("(b) Logged treatment frequency (%)")
axes[0, 1].set_xlabel("Vasopressor level")
axes[0, 1].set_ylabel("Fluid level")
ranked = np.sort(LOGGED_COUNTS[LIVE].sum(axis=1))[::-1]
axes[1, 0].plot(np.arange(len(ranked)), np.maximum(ranked, 1))
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("(c) Uneven state visitation")
axes[1, 0].set_xlabel("State rank by visitation")
axes[1, 0].set_ylabel("Logged transitions")
axes[1, 1].hist(live_sizes, bins=np.arange(1, live_sizes.max() + 2) - 0.5)
axes[1, 1].set_title("(d) Released support-set sizes")
axes[1, 1].set_xlabel("Admissible actions per state")
axes[1, 1].set_ylabel("Live states")
save_figure(fig, FIG_DIR / "paper_fig2_dataset_distributions")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
bins = np.linspace(0, 20, 26)
for label in ["Survival", "Death"]:
    v = EPISODES.loc[EPISODES["outcome"] == label, "initial_sofa"]
    axes[0].hist(v, bins=bins, density=True, histtype="step", linewidth=1.5, label=f"{label} ({len(v):,})")
axes[0].set_title("(a) Initial severity by simulated outcome")
axes[0].set_xlabel("Initial state mean SOFA")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[1].scatter(BENCH.sofa_scores[LIVE], live_sizes, s=8, alpha=0.5)
axes[1].set_title(f"(b) Severity vs support-set size (Spearman ρ = {spearman_rho:.3f})")
axes[1].set_xlabel("State mean SOFA")
axes[1].set_ylabel("Admissible action count")
save_figure(fig, FIG_DIR / "paper_fig3_severity")
print("EDA complete:", EDA_DIR)
