# %% [markdown]
# ## 2. Provenance: hashing, named RNG streams, run identity, identity-checked and resumable checkpoints (audit C03 / C04)
#
# * **Named RNG streams.** Every stochastic component draws from its own stream derived from
#   `(stream name, keys…)`: model initialisation, minibatch indices, diffusion corruption, evaluation /
#   Monte-Carlo, dataset generation. Seeding happens *immediately before* model construction, so
#   initial weights never depend on earlier notebook execution or on run order.
# * **Common random numbers.** For a given training seed, all twin-critic variants share the same
#   initial weights and the same minibatch index sequence on the same dataset; differences between
#   critic variants are therefore attributable to their objectives rather than to different random inputs.
#   Pairing across *different* model families (e.g. tabular IQL seed 3 vs diffusion seed 3) shares only the
#   minibatch stream and the dataset; it is a documented pairing convention, not identical randomness.
# * **Run identity.** dataset content hash, benchmark assets hash + version, normalizer hash,
#   algorithm configuration hash, code hash (source of the trainer and its helpers), training seed and
#   run mode. The identity is stored in every checkpoint and validated on load; a mismatch raises
#   instead of silently loading. Checkpoint file names contain the identity hash, so changing the dataset,
#   objective, hyperparameters, code (functions *and* classes, hashed member by member), or run mode
#   automatically invalidates the cache. The accelerator type is recorded as metadata, not as identity, so
#   GPU-trained checkpoints can be replayed on a CPU runtime (resume states are device-specific).

# %%
# ============================================================
# 2. Provenance utilities
# ============================================================
EXECLOG.mark("2. Provenance utilities")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def obj_hash(obj) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def array_dict_hash(arrays: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[k]))
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def array_hash(a: np.ndarray) -> str:
    return array_dict_hash({"a": a})


def _const_repr(c) -> str:
    """Process-independent representation of a code constant (set/frozenset order depends on PYTHONHASHSEED)."""
    if inspect.iscode(c):
        return _code_signature(c)
    if isinstance(c, (set, frozenset)):
        return f"{type(c).__name__}({sorted(_const_repr(x) for x in c)!r})"
    if isinstance(c, tuple):
        return "(" + ", ".join(_const_repr(x) for x in c) + ",)"
    return repr(c)


def _code_signature(code) -> str:
    """Address-free, hash-seed-independent signature of a code object (fallback when no source is available)."""
    return repr((code.co_code.hex(), code.co_names, tuple(_const_repr(c) for c in code.co_consts)))


_GENERATED_FILES = {"<string>", getattr(dataclasses, "__file__", "")}


def _is_generated(fn) -> bool:
    """Methods synthesised by @dataclass (exec'd from '<string>' or defined in dataclasses.py)."""
    base = inspect.unwrap(fn)
    return (getattr(fn, "__module__", None) == "dataclasses"
            or getattr(base.__code__, "co_filename", "") in _GENERATED_FILES
            or getattr(fn.__code__, "co_filename", "") in _GENERATED_FILES)


def _source_text(o) -> str:
    """Source text used for hashing. Classes are hashed member by member, because inside IPython/Colab
    `inspect.getsource(<class defined in a cell>)` raises OSError while functions and methods work.
    Dataclass-generated methods are skipped (the field list covers them), which keeps the hash identical
    across Python processes and versions."""
    if inspect.isclass(o):
        parts = [o.__qualname__, repr([b.__qualname__ for b in o.__mro__])]
        is_dc = dataclasses.is_dataclass(o)
        if is_dc:
            parts.append(repr([(f.name, str(f.type),
                                None if f.default is dataclasses.MISSING else repr(f.default),
                                f.default_factory is not dataclasses.MISSING) for f in dataclasses.fields(o)]))
        for k, v in sorted(vars(o).items()):
            fn = v.fget if isinstance(v, property) else getattr(v, "__func__", v)
            if inspect.isfunction(fn):
                if is_dc and _is_generated(fn):
                    continue
                try:
                    parts.append(f"{k}:" + inspect.getsource(fn))
                except (OSError, TypeError):
                    parts.append(f"{k}:" + _code_signature(fn.__code__))
            elif not k.startswith("__") and " at 0x" not in repr(v):
                parts.append(f"{k}={v!r}")
        return "\n".join(parts)
    try:
        return inspect.getsource(o)
    except (OSError, TypeError):
        code = getattr(o, "__code__", None)
        if code is None:
            raise TypeError(f"code_hash cannot hash {o!r}")
        return repr((getattr(o, "__qualname__", ""), _code_signature(code)))


def code_hash(*objs) -> str:
    """Hash of the source code of trainers, classes and helpers (audit C03 'code revision')."""
    h = hashlib.sha256()
    for o in objs:
        h.update(_source_text(o).encode("utf-8", errors="replace"))
    return h.hexdigest()


class _CodeHashProbe:
    def f(self):
        return 1


assert code_hash(_CodeHashProbe) != sha256_bytes(repr(("_CodeHashProbe", b"", ())).encode()), \
    "code_hash fell back to a name-only hash for a class."


@dataclass(frozen=True)
class _FrozenProbe:
    a: int = 0
    b: int = 0
    c: int = 0


assert "frozenset(" not in _source_text(_FrozenProbe) and "__setattr__" not in _source_text(_FrozenProbe), \
    "class hash would depend on PYTHONHASHSEED (dataclass-generated methods leaked into the hash)."


def stream_seed(*keys) -> int:
    """Deterministic 62-bit seed for a named RNG stream."""
    digest = hashlib.sha256(canonical_json(["icu-sepsis-revised-v1", *keys]).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 62) - 1)


def np_rng(*keys) -> np.random.Generator:
    return np.random.default_rng(stream_seed(*keys))


def torch_generator(seed: int, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device="cuda" if device.type == "cuda" else "cpu")
    gen.manual_seed(int(seed) % (2 ** 63 - 1))
    return gen


def seed_for_init(seed: int) -> None:
    """Seed the global torch RNG immediately before module construction (initialisation stream)."""
    torch.manual_seed(int(seed) % (2 ** 63 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) % (2 ** 63 - 1))


def init_model(builder: Callable[[], nn.Module], arch_key: str, seed: int) -> nn.Module:
    """Construct a model from the named initialisation stream, independent of prior RNG state.

    The global torch RNG state is saved and restored, so construction also does not perturb any
    later code that (incorrectly) relies on the global stream.
    """
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        seed_for_init(stream_seed("init", arch_key, int(seed)))
        model = builder()
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return model.to(DEVICE)


def state_dict_hash(module: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return int(sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)))


def peak_gpu_memory_mb() -> Optional[float]:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / 2 ** 20)
    return None


def reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def slug(name: str) -> str:
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in "-_.":
            keep.append(ch)
        elif ch in " /:[]()+,=":
            keep.append("_")
    s = "".join(keep)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")[:120]


class CheckpointIdentityError(RuntimeError):
    pass


class MissingArtifactError(RuntimeError):
    pass


def load_torch_payload(path: Path, map_location="cpu"):
    # Our own checkpoints contain plain-Python metadata, so the full payload is requested explicitly.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older PyTorch without weights_only
        return torch.load(path, map_location=map_location)


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    try:
        os.replace(tmp, path)
    except OSError:
        shutil.move(str(tmp), str(path))


# FORCE_RETRAIN persists across Colab disconnects through a token: checkpoints written by the current
# forced run carry the token and are re-used after a reconnect; older checkpoints are ignored.
# Delete checkpoints/force_retrain_token.json to start a new forced run; set FORCE_RETRAIN=False afterwards.
FORCE_TOKEN_FILE = CKPT_DIR / "force_retrain_token.json"
if FORCE_RETRAIN and not FORCE_TOKEN_FILE.exists():
    save_json({"token": os.urandom(16).hex(), "created_utc": utc_now()}, FORCE_TOKEN_FILE)
FORCE_TOKEN = json.loads(FORCE_TOKEN_FILE.read_text())["token"] if FORCE_RETRAIN else None


class CheckpointStore:
    """Identity-addressed checkpoints with validation and resumable training state."""

    def __init__(self, root: Path, allow_training: Optional[bool] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.allow_training = allow_training      # None -> follow the global ALLOW_TRAINING switch
        self.session_hashes: set = set()          # identities written in this Python session
        self.loaded_hashes: set = set()           # identities loaded from earlier sessions

    @staticmethod
    def identity_hash(identity: dict) -> str:
        return obj_hash(identity)

    def _paths(self, label: str, identity: dict):
        h = self.identity_hash(identity)
        d = self.root / slug(label)
        return d / f"final_{h[:20]}.pt", d / f"resume_{h[:20]}.pt", h

    def stale_checkpoints(self, label: str, identity: dict) -> List[str]:
        final, _, _ = self._paths(label, identity)
        if not final.parent.exists():
            return []
        return sorted(p.name for p in final.parent.glob("final_*.pt") if p.name != final.name)

    @staticmethod
    def _force_rejects(payload_token, respect_force: bool) -> bool:
        return bool(FORCE_RETRAIN and respect_force and payload_token != FORCE_TOKEN)

    def load_final(self, label: str, identity: dict, map_location="cpu", respect_force: bool = True):
        final, _, h = self._paths(label, identity)
        if not final.exists():
            return None
        sidecar = final.with_suffix(".identity.json")
        if FORCE_RETRAIN and respect_force and sidecar.exists():
            if json.loads(sidecar.read_text()).get("force_token") != FORCE_TOKEN:
                return None
        payload = load_torch_payload(final, map_location=map_location)
        if self._force_rejects(payload.get("force_token"), respect_force):
            return None
        stored = payload.get("identity")
        if payload.get("identity_hash") != h or stored is None or obj_hash(stored) != h:
            raise CheckpointIdentityError(
                f"Checkpoint {final} does not match the requested run identity; refusing to load it."
            )
        if h not in self.session_hashes:
            self.loaded_hashes.add(h)
        return payload

    def save_final(self, label: str, identity: dict, payload: dict) -> Path:
        final, resume, h = self._paths(label, identity)
        payload = dict(payload)
        payload.update(identity=identity, identity_hash=h, saved_utc=utc_now(), runtime=RUNTIME_INFO,
                       force_token=FORCE_TOKEN, device_type=DEVICE.type)
        atomic_torch_save(payload, final)
        self.session_hashes.add(h)
        save_json({"label": label, "identity_hash": h, "identity": identity, "saved_utc": payload["saved_utc"],
                   "file": final.name, "force_token": FORCE_TOKEN, "device_type": DEVICE.type, "gpu": GPU_NAME},
                  final.with_suffix(".identity.json"))
        if resume.exists():
            resume.unlink()
        return final

    def load_resume(self, label: str, identity: dict, map_location="cpu"):
        _, resume, h = self._paths(label, identity)
        if not resume.exists():
            return None
        payload = load_torch_payload(resume, map_location=map_location)
        if self._force_rejects(payload.get("force_token"), True):
            return None
        if payload.get("identity_hash") != h:
            raise CheckpointIdentityError(f"Resume state {resume} has a different identity.")
        if payload.get("device_type") != DEVICE.type:
            print(f"  ignoring resume state {resume.name}: saved on {payload.get('device_type')}, now on {DEVICE.type}")
            return None
        return payload

    def save_resume(self, label: str, identity: dict, payload: dict) -> None:
        _, resume, h = self._paths(label, identity)
        payload = dict(payload)
        payload.update(identity=identity, identity_hash=h, saved_utc=utc_now(), force_token=FORCE_TOKEN,
                       device_type=DEVICE.type)
        atomic_torch_save(payload, resume)

    def require_training_allowed(self, label: str) -> None:
        allowed = ALLOW_TRAINING if self.allow_training is None else bool(self.allow_training)
        if not allowed:
            raise MissingArtifactError(
                f"ALLOW_TRAINING=False (artifact-replay mode) but no identity-matched checkpoint "
                f"exists for '{label}'."
            )


STORE = CheckpointStore(CKPT_DIR)


def generator_state(gen: torch.Generator) -> torch.Tensor:
    return gen.get_state().clone()


def restore_generator(gen: torch.Generator, state: torch.Tensor) -> None:
    gen.set_state(state.cpu() if state.device.type != "cpu" else state)


print("Provenance utilities ready. Checkpoint root:", CKPT_DIR)
