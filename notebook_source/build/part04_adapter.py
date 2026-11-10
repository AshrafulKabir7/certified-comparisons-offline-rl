# %% [markdown]
# ## 3. Benchmark adapter, action codec verified against the environment, and the two support objects
#
# Terminology used everywhere below (audit m06 / M02):
#
# * **Released benchmark-support mask** `SUPPORT_MASK[s, a]` — read from the benchmark's
#   `admissible_actions`. In the benchmark construction code an action is admissible when its source
#   transition count is **strictly greater than** the threshold (`metadata.threshold = 20`; verified in
#   Section 4). It is *privileged side information* relative to a purely logged-data learner, and it is
#   not ground-truth clinical safety. "Inadmissible" = outside this mask.
# * **Training-dataset support estimate** — computed from counts in the simulated offline dataset
#   (Sections 9 and 25). It answers a different question from the released mask.
# * `OPERATIONAL_MASK` equals the released mask on live (non-terminal) states and is all-True on the
#   three absorbing states, only so that tensor operations on terminal rows are defined.

# %%
# ============================================================
# 3. ICU-Sepsis benchmark adapter
# ============================================================
EXECLOG.mark("3. Benchmark adapter")

from icu_sepsis.utils import constants as ICU_CONSTANTS


class ActionCodec:
    """Flat 25-action index <-> environment action, verified against the environment's own mapping."""

    def __init__(self, action_space, num_actions: int):
        self.num_actions = int(num_actions)
        self.levels = int(round(math.sqrt(self.num_actions)))
        assert self.levels * self.levels == self.num_actions
        if hasattr(action_space, "n"):
            self.kind = "discrete"
            assert int(action_space.n) == self.num_actions
        elif hasattr(action_space, "nvec"):
            self.kind = "multidiscrete"
            assert int(np.prod(action_space.nvec)) == self.num_actions
        else:
            raise TypeError(f"Unsupported action space: {action_space}")

    def components(self, action_idx: int) -> Tuple[int, int]:
        """(fluid level, vasopressor level) — fluid-major, as in the benchmark construction code."""
        a = int(action_idx)
        return a // self.levels, a % self.levels

    def to_env(self, action_idx: int):
        a = int(action_idx)
        if not 0 <= a < self.num_actions:
            raise ValueError(f"Invalid flat action index {a}")
        return a if self.kind == "discrete" else np.asarray(self.components(a), dtype=np.int64)


def find_wrapper(env, class_name: str):
    e = env
    while e is not None:
        if type(e).__name__ == class_name:
            return e
        e = getattr(e, "env", None)
    return None


class ICUSepsisBenchmark:
    ENV_ID = "Sepsis/ICU-Sepsis-v2"

    def __init__(self):
        self.env_mean = gym.make(self.ENV_ID, inadmissible_action_strategy="mean")
        self.env_terminate = gym.make(self.ENV_ID, inadmissible_action_strategy="terminate")
        self.base = self.env_mean.unwrapped
        self.num_states = int(self.base.num_states)
        self.num_actions = int(self.base.num_actions)
        self.codec = ActionCodec(self.env_mean.action_space, self.num_actions)

        self.state_features = np.asarray(self.base.state_cluster_centers, dtype=np.float32)
        self.sofa_scores = np.asarray(self.base.sofa_scores, dtype=np.float64)
        self.expert_policy = np.asarray(self.base.expert_policy, dtype=np.float64)
        self.env_metadata = dict(self.base.env_metadata)

        self.terminal_states = sorted(int(x) for x in ICU_CONSTANTS.STATES_TERMINAL)
        self.death_state = int(ICU_CONSTANTS.STATE_DEATH)
        self.survival_state = int(ICU_CONSTANTS.STATE_SURVIVAL)
        self.sinf_state = int(ICU_CONSTANTS.STATE_S_INF)
        self.live = np.ones(self.num_states, dtype=bool)
        self.live[self.terminal_states] = False

        admissible = self.base.dynamics["admissible_actions"]
        self.support_mask = np.zeros((self.num_states, self.num_actions), dtype=bool)
        for s, acts in enumerate(admissible):
            self.support_mask[s, np.asarray(list(acts), dtype=np.int64)] = True
        self.operational_mask = self.support_mask.copy()
        self.operational_mask[~self.live] = True
        self.feature_dim = int(self.state_features.shape[1])

        self.version = installed_version("icu-sepsis")
        asset_dir = Path(inspect.getfile(type(self.base))).parent / "assets"
        self.asset_hashes = {p.name: sha256_file(p) for p in sorted(asset_dir.glob("*")) if p.is_file()}
        self.assets_hash = obj_hash(self.asset_hashes)
        self.identity = {"env_id": self.ENV_ID, "icu_sepsis": self.version, "assets_hash": self.assets_hash}

    def exact_mdp(self) -> Dict[str, np.ndarray]:
        d = self.base.dynamics
        return {
            "tx_mat": np.array(d["tx_mat"], dtype=np.float64, copy=True),
            "r_mat": np.array(d["r_mat"], dtype=np.float64, copy=True),
            "d_0": np.array(d["d_0"], dtype=np.float64, copy=True),
        }

    def verify_action_codec(self) -> Dict[str, Any]:
        """Check the flat index against the environment's own wrapper + base-env mapping (audit M09)."""
        flat = find_wrapper(self.env_mean, "FlattenActionWrapper")
        rows = []
        for a in range(self.num_actions):
            env_pair = tuple(int(x) for x in flat.action(a)) if flat is not None else self.codec.components(a)
            base_idx = int(self.base._get_action_idx(np.asarray(env_pair, dtype=np.int64)))
            rows.append({"flat": a, "env_pair": env_pair, "base_index": base_idx,
                         "codec_pair": self.codec.components(a)})
        ok = all(r["base_index"] == r["flat"] and tuple(r["codec_pair"]) == tuple(r["env_pair"]) for r in rows)
        return {"wrapper_found": flat is not None, "action_space": str(self.env_mean.action_space),
                "all_25_actions_consistent": bool(ok),
                "convention": "flat = levels * fluid_level + vasopressor_level (fluid-major)"}


BENCH = ICUSepsisBenchmark()
S_N, A_N = BENCH.num_states, BENCH.num_actions
LIVE = BENCH.live
SUPPORT_MASK = BENCH.support_mask          # released benchmark-support mask
OPERATIONAL_MASK = BENCH.operational_mask  # released mask on live rows, all-True on absorbing rows

CODEC_CHECK = BENCH.verify_action_codec()
assert CODEC_CHECK["all_25_actions_consistent"], CODEC_CHECK
assert S_N == 716 and A_N == 25
assert SUPPORT_MASK[LIVE].any(axis=1).all(), "A live state has an empty released-support row."
assert np.isfinite(BENCH.state_features).all(), "Non-finite state-centroid features."
assert np.isfinite(BENCH.sofa_scores).all(), "Non-finite SOFA scores."

# ---------- Mean-rule identity audit (prerequisite for the admissible projection, audit C01) ----------
_EX = BENCH.exact_mdp()
_P, _R = _EX["tx_mat"], _EX["r_mat"]
_r_sa = np.einsum("san,san->sa", _P, _R, optimize=True)
max_row_dev, max_r_dev, n_bad_pairs = 0.0, 0.0, 0
for s in np.flatnonzero(LIVE):
    m = SUPPORT_MASK[s]
    if (~m).any():
        n_bad_pairs += int((~m).sum())
        max_row_dev = max(max_row_dev, float(np.abs(_P[s, ~m] - _P[s, m].mean(axis=0)).max()))
        max_r_dev = max(max_r_dev, float(np.abs(_r_sa[s, ~m] - _r_sa[s, m].mean()).max()))
reward_dest_only = bool(np.allclose(_R, _R[:1, :1, :], atol=0.0))
MEAN_RULE_IDENTITY = {
    "inadmissible_live_pairs": n_bad_pairs,
    "max_abs_dev_inadmissible_row_vs_mean_admissible_rows": max_row_dev,
    "max_abs_dev_expected_reward": max_r_dev,
    "reward_depends_only_on_destination_state": reward_dest_only,
    "reward_values": np.unique(_R).tolist(),
    "rewarding_destinations": np.unique(np.argwhere(_R != 0)[:, 2]).tolist(),
    "benchmark_metadata": BENCH.env_metadata,
}
MEAN_RULE_IDENTITY_HOLDS = bool(max_row_dev <= 1e-12 and max_r_dev <= 1e-12)
MEAN_RULE_IDENTITY["identity_holds_to_1e-12"] = MEAN_RULE_IDENTITY_HOLDS
del _EX, _P, _R, _r_sa
gc.collect()

print(json.dumps({"codec": CODEC_CHECK, "mean_rule_identity": MEAN_RULE_IDENTITY}, indent=2, default=_json_default))
if not MEAN_RULE_IDENTITY_HOLDS:
    print("WARNING: inadmissible transition rows are NOT the uniform mean of admissible rows in this build; "
          "the mean-equivalence of the admissible projection (Section 19) will be reported, not assumed.")
save_json({"codec": CODEC_CHECK, "mean_rule_identity": MEAN_RULE_IDENTITY, "benchmark_identity": BENCH.identity},
          TABLE_DIR / "benchmark_semantics_audit.json")
