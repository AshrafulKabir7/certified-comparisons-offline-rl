# %% [markdown]
# ## 5. Freeze / load the offline dataset with explicit episode semantics (audit M10, C03)
#
# * The environment is stepped **through its registered wrapper chain** (`TimeLimit → OrderEnforcing →
#   PassiveEnvChecker → FlattenActionWrapper → ICUSepsisEnv`) with the flat action index; the previous
#   notebook reset the wrapper but stepped the unwrapped object.
# * Three end-of-episode events are stored separately: `terminated` (absorbing state reached),
#   `truncated` (benchmark `TimeLimit`, 500 steps) and `cutoff` (this notebook's local maximum length).
#   Learners bootstrap with `1 - terminal`, where `terminal = terminated`; a censored (truncated / cutoff)
#   transition is **not** converted into death.
# * The dataset file name and manifest contain a hash of the full generation specification (behaviour,
#   episodes, seed, cutoff, environment id, benchmark assets, generator code). A cached file is used only if
#   its recomputed content hash equals its manifest; the old unconditional `FORCE_REBUILD_DATA = True`
#   override is gone.
# * The clinician dataset uses the same generation stream as the previous notebook (`default_rng(2026)`),
#   so the main dataset is expected to be bit-identical to the original 93,563-transition dataset; Section 21
#   verifies this against the original file when it is available.

# %%
# ============================================================
# 5. Offline dataset specification, generation and validation
# ============================================================
EXECLOG.mark("5. Offline dataset")


@dataclass(frozen=True)
class DatasetSpec:
    n_episodes: int
    seed: int
    max_steps: int
    behavior: str = "clinician"
    strategy: str = "mean"

    def as_dict(self) -> dict:
        return {**asdict(self), "env_id": ICUSepsisBenchmark.ENV_ID, "benchmark": BENCH.identity}


class DatasetIntegrityError(RuntimeError):
    pass


def _validated_expert_rows() -> np.ndarray:
    ep = np.array(BENCH.expert_policy, dtype=np.float64, copy=True)
    live_rows = ep[LIVE]
    if not (np.isfinite(live_rows).all() and (live_rows >= 0).all() and (live_rows.sum(axis=1) > 0).all()):
        raise DatasetIntegrityError("Benchmark expert policy has invalid live rows.")
    return ep


EXPERT_ROWS = _validated_expert_rows()


def make_behavior(behavior: str) -> Callable[[int, np.random.Generator], int]:
    """'clinician' | 'uniform' | 'uniform_admissible' | 'clinician+uniform:EPS' | 'clinician+uniform_admissible:EPS'."""
    def clinician(s, rng):
        p = EXPERT_ROWS[int(s)].copy()
        p /= p.sum()
        return int(rng.choice(A_N, p=p))

    def uniform(s, rng):
        return int(rng.integers(0, A_N))

    def uniform_admissible(s, rng):
        return int(rng.choice(np.flatnonzero(SUPPORT_MASK[int(s)])))

    components = {"clinician": clinician, "uniform": uniform, "uniform_admissible": uniform_admissible}
    if "+" not in behavior:
        return components[behavior]
    base_name, rest = behavior.split("+", 1)
    mix_name, eps = rest.split(":")
    eps = float(eps)
    base_fn, mix_fn = components[base_name], components[mix_name]

    def mixture(s, rng):
        return mix_fn(s, rng) if rng.random() < eps else base_fn(s, rng)
    return mixture


def build_offline_dataset(spec: DatasetSpec) -> Dict[str, np.ndarray]:
    env = gym.make(ICUSepsisBenchmark.ENV_ID, inadmissible_action_strategy=spec.strategy)
    behavior = make_behavior(spec.behavior)
    # Dataset-generation stream: default_rng(seed), as in the original notebook (main dataset reproducible).
    rng = np.random.default_rng(spec.seed)
    cols = {k: [] for k in ["state", "action", "reward", "next_state", "terminated", "truncated", "cutoff",
                            "episode_id", "timestep"]}
    for ep in trange(spec.n_episodes, desc=f"Generating {spec.behavior} data (seed {spec.seed})",
                     leave=False, disable=not SHOW_PROGRESS):
        reset_seed = int(rng.integers(0, 2 ** 31 - 1))
        state, _ = env.reset(seed=reset_seed)
        state = int(state)
        for t in range(spec.max_steps):
            a = behavior(state, rng)
            next_state, reward, terminated, truncated, _ = env.step(BENCH.codec.to_env(a))
            next_state = int(next_state)
            terminated, truncated = bool(terminated), bool(truncated)
            cutoff = (t == spec.max_steps - 1) and not (terminated or truncated)
            cols["state"].append(state)
            cols["action"].append(a)
            cols["reward"].append(float(reward))
            cols["next_state"].append(next_state)
            cols["terminated"].append(terminated)
            cols["truncated"].append(truncated)
            cols["cutoff"].append(cutoff)
            cols["episode_id"].append(ep)
            cols["timestep"].append(t)
            state = next_state
            if terminated or truncated:
                break
    env.close()
    data = {
        "state": np.asarray(cols["state"], dtype=np.int64),
        "action": np.asarray(cols["action"], dtype=np.int64),
        "reward": np.asarray(cols["reward"], dtype=np.float32),
        "next_state": np.asarray(cols["next_state"], dtype=np.int64),
        "terminated": np.asarray(cols["terminated"], dtype=bool),
        "truncated": np.asarray(cols["truncated"], dtype=bool),
        "cutoff": np.asarray(cols["cutoff"], dtype=bool),
        "episode_id": np.asarray(cols["episode_id"], dtype=np.int32),
        "timestep": np.asarray(cols["timestep"], dtype=np.int32),
    }
    data["terminal"] = data["terminated"].astype(np.float32)          # bootstrapping mask: 1 - terminal
    data["logged_action_admissible"] = SUPPORT_MASK[data["state"], data["action"]]
    return data


REQUIRED_FIELDS = ("state", "action", "reward", "next_state", "terminated", "truncated", "cutoff",
                   "terminal", "episode_id", "timestep", "logged_action_admissible")


def validate_dataset(data: Dict[str, np.ndarray], spec: Optional[DatasetSpec] = None) -> Dict[str, Any]:
    """Strict integrity gate (audit M10). Raises DatasetIntegrityError on any violation."""
    def check(cond, msg):
        if not bool(cond):
            raise DatasetIntegrityError(msg)

    missing = set(REQUIRED_FIELDS).difference(data)
    check(not missing, f"Missing dataset fields: {sorted(missing)}")
    n = len(data["state"])
    check(n > 0 and all(len(data[k]) == n for k in REQUIRED_FIELDS), "Dataset arrays have unequal lengths.")
    s, a, ns = data["state"], data["action"], data["next_state"]
    r = data["reward"].astype(np.float64)
    term, trunc, cut = data["terminated"], data["truncated"], data["cutoff"]
    ep, ts = data["episode_id"].astype(np.int64), data["timestep"].astype(np.int64)

    check(np.isfinite(r).all(), "Non-finite rewards.")
    check(((s >= 0) & (s < S_N)).all() and ((ns >= 0) & (ns < S_N)).all(), "State index out of range.")
    check(((a >= 0) & (a < A_N)).all(), "Action index out of range.")
    check(LIVE[s].all(), "A transition starts from an absorbing (terminal) state.")
    check(np.array_equal(data["logged_action_admissible"], SUPPORT_MASK[s, a]), "Admissibility flags disagree with the mask.")
    check(np.array_equal(data["terminal"].astype(bool), term), "terminal != terminated.")

    starts = np.r_[True, ep[1:] != ep[:-1]]
    ends = np.r_[ep[1:] != ep[:-1], True]
    check((np.diff(ep) >= 0).all(), "Episode ids are not grouped / sorted.")
    n_ep = int(starts.sum())
    check(np.array_equal(ep[starts], np.arange(n_ep)), "Episode ids are not contiguous 0..E-1.")
    start_pos = np.flatnonzero(starts)
    first_index_of_row = start_pos[np.cumsum(starts) - 1]
    check(np.array_equal(ts, np.arange(n) - first_index_of_row), "Timesteps are not 0..L-1 within episodes.")
    inner = ~ends
    check(np.array_equal(ns[inner], s[np.flatnonzero(inner) + 1]), "next_state does not continue the episode.")
    check(not (term & ~ends).any(), "A terminal event occurs before the end of an episode.")
    check(not ((trunc | cut) & ~ends).any(), "A censoring event occurs before the end of an episode.")
    check((term | trunc | cut)[ends].all(), "An episode ends without a terminal or censoring event.")
    check(not (cut & (term | trunc)).any(), "cutoff overlaps with terminated/truncated.")
    is_terminal_next = ~LIVE[ns]
    check(np.array_equal(is_terminal_next, term), "terminated flag disagrees with the absorbing next state.")
    check(np.isin(r, [0.0, 1.0]).all(), "Rewards outside {0, 1}.")
    check(np.array_equal(r == 1.0, term & (ns == BENCH.survival_state)), "Reward/outcome inconsistency.")
    ep_return = np.bincount(ep, weights=r, minlength=n_ep)
    check(np.isin(ep_return, [0.0, 1.0]).all(), "Episode return outside {0, 1}.")
    lengths = np.bincount(ep, minlength=n_ep)
    if spec is not None:
        check(n_ep == spec.n_episodes, f"Expected {spec.n_episodes} episodes, found {n_ep}.")
        check(lengths.max() <= spec.max_steps, "Episode longer than the local cutoff.")

    final_next = ns[ends]
    return {
        "transitions": int(n),
        "episodes": n_ep,
        "terminated_episodes": int(term[ends].sum()),
        "survival_episodes": int((term[ends] & (final_next == BENCH.survival_state)).sum()),
        "death_episodes": int((term[ends] & (final_next == BENCH.death_state)).sum()),
        "s_inf_episodes": int((term[ends] & (final_next == BENCH.sinf_state)).sum()),
        "truncated_episodes": int(trunc[ends].sum()),
        "cutoff_episodes": int(cut[ends].sum()),
        "censored_episodes": int((trunc | cut)[ends].sum()),
        "mean_length": float(lengths.mean()),
        "median_length": float(np.median(lengths)),
        "max_length": int(lengths.max()),
        "inadmissible_logged_transitions": int((~data["logged_action_admissible"]).sum()),
        "inadmissible_logged_fraction": float((~data["logged_action_admissible"]).mean()),
    }


DATASET_CODE_HASH = code_hash(build_offline_dataset, make_behavior, validate_dataset, DatasetSpec)


def dataset_identity(spec: DatasetSpec) -> dict:
    return {**spec.as_dict(), "generator_code": DATASET_CODE_HASH}


def load_or_build_dataset(spec: DatasetSpec, save: bool = True) -> Tuple[Dict[str, np.ndarray], dict]:
    ident = dataset_identity(spec)
    h = obj_hash(ident)
    path = DATA_DIR / f"offline_{slug(spec.behavior)}_{spec.n_episodes}eps_seed{spec.seed}_max{spec.max_steps}_{h[:10]}.npz"
    manifest_path = path.with_suffix(".manifest.json")
    if save and path.exists() and manifest_path.exists() and not FORCE_REBUILD_DATA:
        manifest = json.loads(manifest_path.read_text())
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        content = array_dict_hash(data)
        if content != manifest["content_sha256"] or manifest["spec_hash"] != h:
            raise DatasetIntegrityError(f"Cached dataset {path} fails its manifest hash check.")
        validate_dataset(data, spec)
        print("Loaded identity-matched dataset:", path.name)
        return data, manifest
    if not ALLOW_TRAINING and save:
        raise MissingArtifactError(f"Artifact-replay mode: dataset {path.name} is missing.")
    data = build_offline_dataset(spec)
    report = validate_dataset(data, spec)
    manifest = {
        "spec": ident,
        "spec_hash": h,
        "content_sha256": array_dict_hash(data),
        "file": path.name,
        "created_utc": utc_now(),
        "report": report,
    }
    if save:
        np.savez_compressed(path, **data)
        manifest["file_sha256"] = sha256_file(path)
        save_json(manifest, manifest_path)
        print("Saved dataset:", path.name)
    return data, manifest


# ---------- Cutoff-path test with an intentionally short maximum length (audit M10 closure) ----------
_cut_spec = DatasetSpec(n_episodes=60, seed=11, max_steps=3)
_cut_data = build_offline_dataset(_cut_spec)
CUTOFF_TEST = validate_dataset(_cut_data, _cut_spec)
assert CUTOFF_TEST["cutoff_episodes"] > 0, "Short-cutoff test did not exercise the censoring path."
assert not _cut_data["terminal"][_cut_data["cutoff"]].any(), "Censored rows must not be marked terminal."
print("Short-cutoff test:", CUTOFF_TEST)
del _cut_data

# ---------- Main frozen dataset ----------
MAIN_SPEC = DatasetSpec(n_episodes=CFG.dataset_episodes, seed=CFG.dataset_seed,
                        max_steps=CFG.max_episode_steps, behavior=CFG.behavior)
OFFLINE, OFFLINE_MANIFEST = load_or_build_dataset(MAIN_SPEC)
DATASET_REPORT = validate_dataset(OFFLINE, MAIN_SPEC)
DATASET_PATH = DATA_DIR / OFFLINE_MANIFEST["file"]
print(json.dumps(DATASET_REPORT, indent=2))
save_json({"main": OFFLINE_MANIFEST, "report": DATASET_REPORT, "short_cutoff_test": CUTOFF_TEST},
          TABLE_DIR / "dataset_manifest_main.json")
