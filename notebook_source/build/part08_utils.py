# %% [markdown]
# ## 7. Shared utilities: normalizer bound to data, tensor buffer, strict policy validation, admissible projection, policy registry
#
# * The feature normalizer is visit-weighted over the *training* states of a data bundle (all
#   transitions in the main study; training episodes only when an episode split is used, Section 26).
#   Its hash is part of every run identity and it is stored inside every neural checkpoint.
# * `validate_policy` rejects malformed policies (shape, non-finite, negative, row sums). Repair is
#   an explicit opt-in diagnostic (`repair_policy_rows`) and is never applied silently (audit M09).
# * `project_to_support` is the **mean-equivalent admissible projection** (audit C01):
#   `π̃(a|s) = m(s,a)·[π(a|s) + b(s)/k(s)]`, where `b(s)` is the probability π puts on inadmissible
#   actions and `k(s)` the number of admissible actions. Under the benchmark's mean rule every inadmissible
#   action already uses the uniform mixture of admissible transition rows, so π and π̃ induce identical
#   mean-rule dynamics and rewards, while π̃ never selects an inadmissible action. This equivalence is
#   stated in Choudhary et al. (2024, §4.3); it is used here as a control, not claimed as new.
#   It differs from `renormalize_on_support`, which conditions π on admissibility.

# %%
# ============================================================
# 7. Shared utilities
# ============================================================
EXECLOG.mark("7. Shared utilities")


class FeatureNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, keep: Sequence[int]):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.keep = np.asarray(keep, dtype=np.int64)

    @classmethod
    def fit(cls, features: np.ndarray, state_indices: np.ndarray, drop: Sequence[int] = ()):
        keep = [j for j in range(features.shape[1]) if j not in set(drop)]
        x = np.asarray(features[np.asarray(state_indices)][:, keep], dtype=np.float32)  # visit-weighted
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean, std, keep)

    def transform(self, features: np.ndarray) -> np.ndarray:
        out = (np.asarray(features, dtype=np.float32)[:, self.keep] - self.mean) / self.std
        if not np.isfinite(out).all():
            raise FloatingPointError("Non-finite normalised features.")
        return out.astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "keep": self.keep.tolist()}

    @property
    def hash(self) -> str:
        return array_dict_hash({"mean": self.mean, "std": self.std, "keep": self.keep})


class OfflineTensorBuffer:
    def __init__(self, data: Dict[str, np.ndarray], all_x: np.ndarray, device: torch.device):
        t = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        self.device = device
        self.state = t(data["state"], torch.long)
        self.action = t(data["action"], torch.long)
        self.reward = t(data["reward"], torch.float32)
        self.next_state = t(data["next_state"], torch.long)
        self.terminal = t(data["terminal"], torch.float32)
        self.adm = t(data["logged_action_admissible"], torch.bool)
        self.all_x = t(all_x, torch.float32)
        self.mask = t(OPERATIONAL_MASK, torch.bool)
        self.n = int(len(data["state"]))
        assert torch.isfinite(self.all_x).all(), "Non-finite feature tensor."
        assert torch.isfinite(self.reward).all()

    def batch_from_index(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        s, ns = self.state[idx], self.next_state[idx]
        return {"s_idx": s, "a": self.action[idx], "r": self.reward[idx], "ns_idx": ns,
                "terminal": self.terminal[idx], "adm": self.adm[idx], "x": self.all_x[s], "nx": self.all_x[ns],
                "mask": self.mask[s], "next_mask": self.mask[ns]}

    def sample(self, batch_size: int, generator: torch.Generator) -> Dict[str, torch.Tensor]:
        idx = torch.randint(0, self.n, (batch_size,), generator=generator, device=self.device)
        return self.batch_from_index(idx)

    def chunks(self, size: int = 8192):
        for start in range(0, self.n, size):
            yield self.batch_from_index(torch.arange(start, min(start + size, self.n), device=self.device))


@dataclass
class DataBundle:
    name: str
    data: Dict[str, np.ndarray]
    manifest: dict
    normalizer: FeatureNormalizer
    feature_drop: Tuple[int, ...] = ()

    def __post_init__(self):
        self.data_hash = array_dict_hash(self.data)
        self.all_x = self.normalizer.transform(BENCH.state_features)
        self.counts = np.zeros((S_N, A_N), dtype=np.int64)
        np.add.at(self.counts, (self.data["state"], self.data["action"]), 1)
        self._buffer = None

    @property
    def input_dim(self) -> int:
        return int(self.all_x.shape[1])

    def identity(self) -> dict:
        return {"dataset_hash": self.data_hash, "normalizer_hash": self.normalizer.hash,
                "feature_drop": list(self.feature_drop), "transitions": int(len(self.data["state"]))}

    def buffer(self) -> OfflineTensorBuffer:
        if self._buffer is None:
            self._buffer = OfflineTensorBuffer(self.data, self.all_x, DEVICE)
        return self._buffer

    def release(self) -> None:
        self._buffer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def subset_rows(data: Dict[str, np.ndarray], rows: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[rows] for k, v in data.items()}


def make_bundle(name: str, data: Dict[str, np.ndarray], manifest: dict, rows: Optional[np.ndarray] = None,
                feature_drop: Sequence[int] = ()) -> DataBundle:
    d = data if rows is None else subset_rows(data, rows)
    norm = FeatureNormalizer.fit(BENCH.state_features, d["state"], drop=feature_drop)
    return DataBundle(name=name, data=d, manifest=manifest, normalizer=norm, feature_drop=tuple(feature_drop))


MAIN = make_bundle("main", OFFLINE, OFFLINE_MANIFEST)
np.savez(DATA_DIR / "feature_normalizer_main.npz", mean=MAIN.normalizer.mean, std=MAIN.normalizer.std,
         keep=MAIN.normalizer.keep)
print("Main bundle:", MAIN.identity())

# Feature provenance (M08): terminal-state encodings and the notebook's additional standardisation.
for _lab, _s in (("death", BENCH.death_state), ("survival", BENCH.survival_state), ("s_inf", BENCH.sinf_state)):
    FEATURE_PROVENANCE[f"{_lab}_state_{_s}_centroid_value"] = BENCH.state_features[_s].astype(np.float64)
_mu = np.full(BENCH.feature_dim, np.nan)
_sd = np.full(BENCH.feature_dim, np.nan)
_mu[MAIN.normalizer.keep] = MAIN.normalizer.mean
_sd[MAIN.normalizer.keep] = MAIN.normalizer.std
FEATURE_PROVENANCE["notebook_norm_mean"] = _mu
FEATURE_PROVENANCE["notebook_norm_std"] = _sd
FEATURE_PROVENANCE["notebook_norm_fit_on"] = "visit-weighted current (pre-action) states of the main dataset"
FEATURE_PROVENANCE.to_csv(TABLE_DIR / "feature_provenance_timing.csv", index=False)
print("Absorbing-state centroid vectors are all zeros:", bool(np.all(BENCH.state_features[~LIVE] == 0)))

# ---------- Policy validation, extraction and projection ----------
FIXED = -1  # seed label for fixed (non-learned) policies


class PolicyValidationError(ValueError):
    pass


def validate_policy(pi: np.ndarray, name: str = "policy", atol: float = 1e-8,
                    shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    pi = np.asarray(pi, dtype=np.float64)
    problems = []
    shape = (S_N, A_N) if shape is None else tuple(shape)
    if pi.shape != shape:
        raise PolicyValidationError(f"{name}: shape {pi.shape} != {shape}")
    if not np.isfinite(pi).all():
        problems.append(f"{int((~np.isfinite(pi)).sum())} non-finite entries")
    if (pi < -1e-12).any():
        problems.append(f"{int((pi < -1e-12).sum())} negative entries")
    rs = pi.sum(axis=1)
    if np.isfinite(rs).all() and not np.allclose(rs, 1.0, atol=atol):
        problems.append(f"{int((np.abs(rs - 1.0) > atol).sum())} rows do not sum to one")
    if problems:
        raise PolicyValidationError(f"{name}: " + "; ".join(problems))
    return np.clip(pi, 0.0, None)


def repair_policy_rows(pi: np.ndarray, fallback_mask: np.ndarray = OPERATIONAL_MASK) -> Tuple[np.ndarray, dict]:
    """Explicit opt-in diagnostic repair. Never used silently by the evaluator."""
    raw = np.asarray(pi, dtype=np.float64)
    out = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, 0.0, None)
    rs = out.sum(axis=1)
    empty = rs <= 0
    for s in np.flatnonzero(empty):
        out[s, fallback_mask[s]] = 1.0 / fallback_mask[s].sum()
    out = out / out.sum(axis=1, keepdims=True)
    report = {"nonfinite_entries": int((~np.isfinite(raw)).sum()), "negative_entries": int((raw < 0).sum()),
              "empty_rows_replaced": int(empty.sum())}
    return out, report


def one_hot_policy(actions: np.ndarray, num_actions: int = A_N) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int64)
    pi = np.zeros((len(actions), num_actions), dtype=np.float64)
    pi[np.arange(len(actions)), actions] = 1.0
    return pi


def masked_argmax(scores: np.ndarray, mask: np.ndarray = OPERATIONAL_MASK) -> np.ndarray:
    """Argmax restricted to `mask`; ties resolve to the lowest admissible index (deterministic)."""
    m = np.asarray(mask, dtype=bool)
    if not m.any(axis=1).all():
        raise ValueError("masked_argmax: a row has no allowed action.")
    return np.where(m, np.asarray(scores, dtype=np.float64), -np.inf).argmax(axis=1)


def greedy_policy(scores: np.ndarray, masked: bool) -> np.ndarray:
    if masked:
        return one_hot_policy(masked_argmax(scores, OPERATIONAL_MASK))
    return one_hot_policy(np.asarray(scores, dtype=np.float64).argmax(axis=1))


def inadmissible_mass(pi: np.ndarray, mask: np.ndarray = SUPPORT_MASK) -> np.ndarray:
    b = (np.asarray(pi) * (~mask)).sum(axis=1)
    b[~LIVE] = 0.0
    return b


def project_to_support(pi: np.ndarray, mask: np.ndarray = SUPPORT_MASK) -> np.ndarray:
    """Mean-equivalent admissible projection (C01). Terminal rows are left unchanged."""
    pi = validate_policy(pi, "projection input")
    m = np.asarray(mask, dtype=bool)[LIVE]
    if not m.any(axis=1).all():
        raise ValueError("Projection requires a non-empty admissible set in every live state.")
    out = pi.copy()
    live_pi = pi[LIVE]
    bad = (live_pi * ~m).sum(axis=1, keepdims=True)
    out[LIVE] = m * (live_pi + bad / m.sum(axis=1, keepdims=True))
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(out[LIVE][~m] == 0.0)
    return out


def renormalize_on_support(pi: np.ndarray, mask: np.ndarray = SUPPORT_MASK) -> Tuple[np.ndarray, int]:
    """Condition π on admissibility; rows with zero admissible mass fall back to uniform-admissible."""
    pi = validate_policy(pi, "renormalisation input")
    m = np.asarray(mask, dtype=bool)
    out = pi.copy()
    num = (pi * m)[LIVE]
    z = num.sum(axis=1, keepdims=True)
    fallback = (z[:, 0] <= 0)
    num[fallback] = m[LIVE][fallback]
    z = num.sum(axis=1, keepdims=True)
    out[LIVE] = num / z
    return out, int(fallback.sum())


def uniform_admissible_policy(mask: np.ndarray = SUPPORT_MASK) -> np.ndarray:
    out = np.full((S_N, A_N), 1.0 / A_N)
    m = mask[LIVE].astype(np.float64)
    out[LIVE] = m / m.sum(axis=1, keepdims=True)
    return out


def tie_and_margin_report(scores: np.ndarray, name: str, tol: float = 1e-9) -> Dict[str, Any]:
    """Greedy-admissibility diagnostics (audit M05): margins and ties between admissible and inadmissible."""
    q = np.asarray(scores, dtype=np.float64)[LIVE]
    m = SUPPORT_MASK[LIVE]
    has_bad = (~m).any(axis=1)
    best_adm = np.where(m, q, -np.inf).max(axis=1)
    best_bad = np.where(~m, q, -np.inf).max(axis=1)
    margin = (best_adm - best_bad)[has_bad]
    q_adm_sorted = np.sort(np.where(m, q, -np.inf), axis=1)
    multi = m.sum(axis=1) >= 2
    adm_ties = int((np.abs(q_adm_sorted[multi, -1] - q_adm_sorted[multi, -2]) <= tol * np.maximum(1.0, np.abs(q_adm_sorted[multi, -1]))).sum())
    return {
        "policy": name,
        "states_with_inadmissible_actions": int(has_bad.sum()),
        "min_margin_admissible_minus_inadmissible": float(margin.min()) if margin.size else float("nan"),
        "states_inadmissible_ranked_first": int((margin < 0).sum()),
        "states_tied_admissible_vs_inadmissible": int((np.abs(margin) <= tol).sum()),
        "states_tied_among_admissible": adm_ties,
    }


# ---------- Policy registry (every evaluated policy is a named, documented object) ----------
@dataclass
class PolicyEntry:
    name: str
    family: str
    extraction: str
    stochastic: bool
    mask_source: str          # "none" | "released benchmark mask" | "dataset-estimated (...)"
    mask_use: str             # "none" | "training" | "extraction" | "training+extraction" | "projection" | ...
    information: str          # data and side information used
    group: str = "main"
    dataset: str = "main"
    parent: str = ""
    critic: str = ""
    notes: str = ""
    by_seed: Dict[int, np.ndarray] = field(default_factory=dict)
    scores: Dict[int, np.ndarray] = field(default_factory=dict)

    def meta(self) -> dict:
        return {k: getattr(self, k) for k in ("name", "family", "extraction", "stochastic", "mask_source",
                                                 "mask_use", "information", "group", "dataset", "parent",
                                                 "critic", "notes")}


REGISTRY: Dict[str, PolicyEntry] = {}
POLICY_FILES: List[dict] = []

# Snapshot of the previous session's exported results and policy files BEFORE anything is overwritten, so
# Section 32 can perform a genuine cross-session replay comparison (audit C04), not only a file round-trip.
_SESSION_TAG = EXECLOG.session_start.replace(":", "")
PREV_SESSION = {"exact_results": None, "policy_hashes": {}}
_prev_results = TABLE_DIR / "exact_results_all.csv"
if _prev_results.exists():
    shutil.copy(_prev_results, LOG_DIR / f"prev_exact_results_{_SESSION_TAG}.csv")
    PREV_SESSION["exact_results"] = pd.read_csv(_prev_results)
for _f in POLICY_DIR.rglob("*.npz"):
    try:
        with np.load(_f) as _z:
            PREV_SESSION["policy_hashes"][str(_f.relative_to(ROOT))] = array_hash(np.asarray(_z["policy"], dtype=np.float64))
    except Exception:
        pass
if PREV_SESSION["policy_hashes"]:
    save_json(PREV_SESSION["policy_hashes"], LOG_DIR / f"prev_policy_hashes_{_SESSION_TAG}.json")
print("Previous-session artifacts found:", PREV_SESSION["exact_results"] is not None,
      "| policy files:", len(PREV_SESSION["policy_hashes"]))


def register_policy(name: str, seed: int, pi: np.ndarray, scores: Optional[np.ndarray] = None,
                    save: bool = True, **meta) -> np.ndarray:
    pi = validate_policy(pi, f"{name} seed={seed}")
    if name not in REGISTRY:
        required = {"family", "extraction", "stochastic", "mask_source", "mask_use", "information"}
        missing = required.difference(meta)
        if missing:
            raise KeyError(f"First registration of {name} needs metadata {sorted(missing)}")
        REGISTRY[name] = PolicyEntry(name=name, **meta)
    entry = REGISTRY[name]
    entry.by_seed[int(seed)] = pi
    if scores is not None:
        entry.scores[int(seed)] = np.asarray(scores, dtype=np.float64)
    if save:
        out = POLICY_DIR / slug(entry.group) / f"{slug(name)}__seed{int(seed)}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"policy": pi}
        if scores is not None:
            payload["scores"] = np.asarray(scores, dtype=np.float64)
        np.savez_compressed(out, **payload)
        POLICY_FILES.append({"policy": name, "seed": int(seed), "file": str(out.relative_to(ROOT)),
                             "policy_sha256": array_hash(pi)})
    return pi


def registry_frame() -> pd.DataFrame:
    return pd.DataFrame([{**e.meta(), "seeds": sorted(e.by_seed)} for e in REGISTRY.values()])


def entry_seeds(name: str) -> List[int]:
    return sorted(REGISTRY[name].by_seed)


print("Utilities ready.")
