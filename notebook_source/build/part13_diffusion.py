# %% [markdown]
# ## 12. Categorical diffusion prior and exact critic-guided reverse inference (audit C03, M06)
#
# * **C03 repair:** the denoiser is constructed from its named initialisation stream *inside*
#   `init_model`, i.e. after seeding — the previous notebook built the network before seeding, so its
#   initial weights depended on earlier notebook execution. Minibatch indices and corruption
#   (timestep + noisy-token) draws now use separate streams.
# * **Schedule audit:** with β linearly spaced from 0.02 to 0.25 over T = 20 steps,
#   `Q̄_T = ᾱ_T I + (1 − ᾱ_T) U` with `ᾱ_T = Π(1 − β_t) ≈ 0.0515`: the terminal forward distribution is
#   **not exactly uniform**. Starting inference from a uniform distribution over admissible actions, masking at
#   every step and adding critic guidance make the procedure an explicitly *modified inference heuristic*,
#   not exact sampling from the learned diffusion model. Intermediate tokens are latent variables, not treatments.
# * Inference variants (all computed by exact marginal propagation over the finite state set):
#   initialisation (uniform-admissible vs uniform over all 25), masking placement (every step, final step only,
#   none), guidance placement (every step, final step only, none), guidance strength, critic pairing
#   (hard-only, tabular-only, mixed), deterministic vs stochastic final extraction.
# * The deployed policy is a **precomputed lookup table** over the 716 states; reverse-chain cost is a
#   policy-construction cost, reported separately from per-decision lookup latency.

# %%
# ============================================================
# 12. Categorical diffusion
# ============================================================
EXECLOG.mark("12. Categorical diffusion")


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    exponent = -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    angles = t.float()[:, None] * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return F.pad(emb, (0, 1)) if dim % 2 == 1 else emb


class CategoricalDenoiser(nn.Module):
    """Predict x0 from (state, corrupted treatment x_t, step t); treatment token factorised fluid x vasopressor."""

    def __init__(self, state_dim, num_actions, hidden_dim, action_embed_dim, time_embed_dim):
        super().__init__()
        levels = int(round(math.sqrt(num_actions)))
        assert levels * levels == num_actions
        self.levels, self.time_embed_dim = levels, time_embed_dim
        self.fluid_embedding = nn.Embedding(levels, action_embed_dim)
        self.vaso_embedding = nn.Embedding(levels, action_embed_dim)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
                                           nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        fused = hidden_dim + 2 * action_embed_dim + time_embed_dim
        self.head = nn.Sequential(nn.Linear(fused, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_actions))

    def forward(self, state_x, xt, t):
        fluid = torch.div(xt, self.levels, rounding_mode="floor")
        vaso = xt % self.levels
        h = torch.cat([self.state_encoder(state_x), self.fluid_embedding(fluid), self.vaso_embedding(vaso),
                       sinusoidal_time_embedding(t, self.time_embed_dim)], dim=1)
        return self.head(h)


class CategoricalDiffusion:
    """Uniform categorical corruption Q_t = (1 - beta_t) I + beta_t U with exact posteriors."""

    def __init__(self, num_actions: int, T: int, beta_start: float, beta_end: float):
        self.K, self.T = int(num_actions), int(T)
        self.betas = np.linspace(beta_start, beta_end, self.T, dtype=np.float64)
        eye = np.eye(self.K)
        U = np.ones((self.K, self.K)) / self.K
        self.Q, self.Qbar = [None], [eye]
        running = eye
        for beta in self.betas:
            qt = (1.0 - beta) * eye + beta * U
            running = running @ qt
            self.Q.append(qt)
            self.Qbar.append(running.copy())
        self.alpha_bar = np.cumprod(1.0 - self.betas)
        self.posterior = [None]
        for t in range(1, self.T + 1):
            post = np.zeros((self.K, self.K, self.K))
            for xt in range(self.K):
                num = self.Qbar[t - 1] * self.Q[t][:, xt][None, :]          # [x0, xprev]
                cond = num / np.maximum(self.Qbar[t][:, xt][:, None], 1e-30)
                post[xt] = cond / np.maximum(cond.sum(axis=1, keepdims=True), 1e-30)
            self.posterior.append(post)
            assert np.allclose(self.Q[t].sum(1), 1) and np.allclose(self.Qbar[t].sum(1), 1)
            assert np.allclose(post.sum(axis=2), 1.0, atol=1e-10)
        self._qbar_cache = {}

    def schedule_audit(self) -> dict:
        abar = float(self.alpha_bar[-1])
        closed = abar * np.eye(self.K) + (1 - abar) * np.ones((self.K, self.K)) / self.K
        return {"T": self.T, "beta_start": float(self.betas[0]), "beta_end": float(self.betas[-1]),
                "alpha_bar_T_initial_token_coefficient": abar,
                "max_abs_dev_Qbar_T_from_closed_form": float(np.abs(self.Qbar[-1] - closed).max()),
                "max_prob_of_initial_token_at_T": float(abar + (1 - abar) / self.K),
                "terminal_forward_exactly_uniform": bool(abar < 1e-12)}

    def sample_xt(self, x0: torch.Tensor, t: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        key = str(x0.device)
        if key not in self._qbar_cache:
            self._qbar_cache[key] = torch.as_tensor(np.stack(self.Qbar), dtype=torch.float32, device=x0.device)
        return torch.multinomial(self._qbar_cache[key][t, x0], 1, replacement=True, generator=generator).squeeze(1)


DIFFUSION_CODE_HASH = None


def diffusion_identity(bundle: DataBundle, seed: int, T: int) -> dict:
    hp = {"steps": CFG.diffusion_steps, "lr": CFG.diffusion_lr, "T": int(T), "beta_start": CFG.diffusion_beta_start,
          "beta_end": CFG.diffusion_beta_end, "action_embed": CFG.diffusion_action_embed_dim,
          "time_embed": CFG.diffusion_time_embed_dim, "hidden": CFG.hidden_dim, "batch": CFG.batch_size,
          "weight_decay": CFG.weight_decay, "grad_clip": CFG.grad_clip}
    return {"algo": "categorical_diffusion_prior", "hparams": hp, "seed": int(seed), "run_mode": RUN_MODE,
            "data": bundle.identity(), "benchmark": BENCH.identity, "code": DIFFUSION_CODE_HASH}


def build_denoiser(input_dim: int) -> CategoricalDenoiser:
    return CategoricalDenoiser(input_dim, A_N, CFG.hidden_dim, CFG.diffusion_action_embed_dim, CFG.diffusion_time_embed_dim)


def train_diffusion_prior(bundle: DataBundle, seed: int, T: Optional[int] = None, tag: str = "diffusion",
                          store: Optional[CheckpointStore] = None, interrupt_at: Optional[int] = None):
    store = STORE if store is None else store
    T = CFG.diffusion_T if T is None else int(T)
    diffusion = CategoricalDiffusion(A_N, T, CFG.diffusion_beta_start, CFG.diffusion_beta_end)
    ident = diffusion_identity(bundle, seed, T)
    label = f"{tag}/T{T}/{bundle.name}/seed{seed}"
    payload = store.load_final(label, ident, map_location=DEVICE)
    if payload is None:
        store.require_training_allowed(label)
        t0 = time.perf_counter()
        reset_peak_gpu_memory()
        buf = bundle.buffer()
        model = init_model(lambda: build_denoiser(bundle.input_dim), f"denoiser_{bundle.input_dim}x{CFG.hidden_dim}", seed)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG.diffusion_lr, weight_decay=CFG.weight_decay)
        gen_batch = torch_generator(stream_seed("minibatch", bundle.data_hash, seed), DEVICE)
        gen_noise = torch_generator(stream_seed("corruption", bundle.data_hash, seed, T), DEVICE)
        rows, start = [], 0
        prior_elapsed, prior_peak, segments = 0.0, 0.0, 1
        resume = store.load_resume(label, ident, map_location=DEVICE)
        if resume is not None:
            model.load_state_dict(resume["model"])
            opt.load_state_dict(resume["optimizer"])
            restore_generator(gen_batch, resume["gen_batch"])
            restore_generator(gen_noise, resume["gen_noise"])
            rows, start = resume["log"], int(resume["step"])
            prior_elapsed = float(resume.get("elapsed_s", 0.0))
            prior_peak = float(resume.get("peak_gpu_mb") or 0.0)
            segments = int(resume.get("segments", 1)) + 1
            print(f"  resumed {label} at step {start}")
        model.train()
        for step in trange(start, CFG.diffusion_steps, desc=f"diffusion T{T} {bundle.name} s{seed}",
                           leave=False, disable=not SHOW_PROGRESS):
            b = buf.sample(CFG.batch_size, gen_batch)
            x0 = b["a"]
            t = torch.randint(1, T + 1, (CFG.batch_size,), generator=gen_noise, device=DEVICE)
            xt = diffusion.sample_xt(x0, t, gen_noise)
            logits = model(b["x"], xt, t)
            loss = F.cross_entropy(logits, x0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            opt.step()
            if step % CFG.log_every == 0 or step == CFG.diffusion_steps - 1:
                rows.append({"step": step + 1, "loss": float(loss),
                             "x0_accuracy": float((logits.argmax(1) == x0).float().mean())})
            if (step + 1) % CFG.resume_every == 0 and step + 1 < CFG.diffusion_steps:
                store.save_resume(label, ident, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                                 "gen_batch": generator_state(gen_batch),
                                                 "gen_noise": generator_state(gen_noise), "step": step + 1, "log": rows,
                                                 "elapsed_s": prior_elapsed + (time.perf_counter() - t0),
                                                 "peak_gpu_mb": max(prior_peak, peak_gpu_memory_mb() or 0.0),
                                                 "segments": segments})
                if interrupt_at is not None and step + 1 >= interrupt_at:
                    return None
        model.eval()
        meta = {"elapsed_s": prior_elapsed + (time.perf_counter() - t0), "params": count_parameters(model),
                "peak_gpu_mb": (max(prior_peak, peak_gpu_memory_mb() or 0.0) if torch.cuda.is_available() else None),
                "steps": CFG.diffusion_steps, "T": T, "normalizer": bundle.normalizer.to_dict(),
                "resumed": resume is not None, "segments": segments}
        payload = {"model": model.state_dict(), "log": rows, "meta": meta}
        store.save_final(label, ident, payload)
        COST_ROWS.append({"model": f"Diffusion prior (T={T})", "label": label, "seed": seed,
                          "train_s": meta["elapsed_s"], "params": meta["params"],
                          "peak_gpu_mb": meta["peak_gpu_mb"], "steps": CFG.diffusion_steps})
    model = build_denoiser(bundle.input_dim).to(DEVICE)
    model.load_state_dict(payload["model"])
    model.eval()
    return {"model": model, "diffusion": diffusion, "log": pd.DataFrame(payload["log"]), "meta": payload["meta"],
            "label": label}


class ReverseCache:
    """Exact base reverse kernels B_t(x_{t-1} | x_t, s) for every state, cached once per trained denoiser."""

    @torch.no_grad()
    def __init__(self, model: CategoricalDenoiser, diffusion: CategoricalDiffusion, all_x: np.ndarray, batch: int = 8192,
                 device: Optional[torch.device] = None):
        dev = DEVICE if device is None else device
        t0 = time.perf_counter()
        model.eval()
        S, K = all_x.shape[0], diffusion.K
        self.T, self.K, self.S = diffusion.T, K, S
        x_all = torch.as_tensor(all_x, dtype=torch.float32, device=dev)
        s_idx = torch.arange(S, device=dev).repeat_interleave(K)
        xt_idx = torch.arange(K, device=dev).repeat(S)
        self.base = {}
        for t in range(self.T, 0, -1):
            probs = []
            for i in range(0, S * K, batch):
                sl = slice(i, i + batch)
                tt = torch.full((len(s_idx[sl]),), t, dtype=torch.long, device=dev)
                probs.append(torch.softmax(model(x_all[s_idx[sl]], xt_idx[sl], tt).double(), dim=1).cpu())
            p_x0 = torch.cat(probs).numpy().reshape(S, K, K)
            # Same floor (1e-30) as the original implementation before taking logarithms.
            self.base[t] = np.clip(np.einsum("ski,kij->skj", p_x0, diffusion.posterior[t], optimize=True), 1e-30, None)
        self.construction_s = time.perf_counter() - t0


def normalize_critic_per_state(q: np.ndarray, mask: np.ndarray = OPERATIONAL_MASK) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    cnt = m.sum(axis=1, keepdims=True)
    mu = np.where(m, q, 0.0).sum(axis=1, keepdims=True) / cnt
    sd = np.sqrt(np.where(m, (q - mu) ** 2, 0.0).sum(axis=1, keepdims=True) / cnt)
    return np.where(m, (q - mu) / np.maximum(sd, 1e-6), 0.0)


def critic_ensemble(*q_tables: np.ndarray, mask: np.ndarray = OPERATIONAL_MASK) -> np.ndarray:
    return np.mean([normalize_critic_per_state(q, mask) for q in q_tables], axis=0)


def guided_reverse_policy(cache: ReverseCache, critic_G: Optional[np.ndarray], guidance: float,
                          mask: np.ndarray = OPERATIONAL_MASK, init: str = "uniform_admissible",
                          mask_mode: str = "every_step", guidance_mode: str = "every_step",
                          deterministic: bool = True) -> Tuple[np.ndarray, List[float]]:
    assert init in {"uniform_admissible", "uniform_all"}
    assert mask_mode in {"every_step", "final_only", "none"}
    assert guidance_mode in {"every_step", "final_only", "none"}
    if guidance_mode == "every_step" and guidance != 0 and mask_mode != "every_step":
        raise ValueError("Every-step guidance is only defined together with every-step masking.")
    m = np.asarray(mask, dtype=bool)
    p = m / m.sum(axis=1, keepdims=True) if init == "uniform_admissible" else np.full((cache.S, cache.K), 1.0 / cache.K)
    trace = []
    for t in range(cache.T, 0, -1):
        logits = np.log(cache.base[t])
        g = guidance if (guidance_mode == "every_step" or (guidance_mode == "final_only" and t == 1)) else 0.0
        if g != 0.0 and critic_G is not None:
            logits = logits + g * critic_G[:, None, :]
        masked_now = mask_mode == "every_step" or (mask_mode == "final_only" and t == 1)
        if masked_now:
            logits = np.where(m[:, None, :], logits, -np.inf)
        logits = logits - logits.max(axis=2, keepdims=True)
        kernel = np.exp(logits)
        kernel /= kernel.sum(axis=2, keepdims=True)
        p = np.einsum("sk,skj->sj", p, kernel, optimize=True)
        if masked_now:
            p = np.where(m, p, 0.0)
        p = p / p.sum(axis=1, keepdims=True)
        trace.append(float(np.where(m, 0.0, p)[LIVE].sum(axis=1).max()))
    policy = one_hot_policy(p.argmax(axis=1)) if deterministic else p
    if mask_mode != "none":
        # Invariant with respect to the mask actually supplied (released or dataset-estimated).
        assert np.all(policy[LIVE][~m[LIVE]] <= 1e-14), "Masked reverse inference produced mass outside the mask."
        assert float(np.where(m, 0.0, p)[LIVE].sum(axis=1).max()) <= 1e-12
    return policy, trace


def prior_times_critic(prior: np.ndarray, critic_G: Optional[np.ndarray], guidance: float,
                       mask: np.ndarray = OPERATIONAL_MASK, deterministic: bool = True) -> np.ndarray:
    """pi(a|s) ∝ m(s,a) prior(a|s) exp(g G(s,a)) — the simplest behaviour-prior-times-critic policy."""
    logits = np.log(np.clip(prior, 1e-300, None))
    if critic_G is not None and guidance != 0.0:
        logits = logits + guidance * critic_G
    logits = np.where(mask, logits, -np.inf)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return one_hot_policy(p.argmax(axis=1)) if deterministic else p


@torch.no_grad()
def ondemand_reverse_latency_ms(model: CategoricalDenoiser, diffusion: CategoricalDiffusion, all_x: np.ndarray,
                                n_states: int = 20) -> float:
    """Wall time to run the full reverse chain for ONE state on demand (not the deployed lookup)."""
    rng = np_rng("evaluation", "latency")
    states = rng.choice(np.flatnonzero(LIVE), size=n_states, replace=False)
    K = diffusion.K
    times = []
    for s in states:
        t0 = time.perf_counter()
        x = torch.as_tensor(all_x[[s] * K], dtype=torch.float32, device=DEVICE)
        xt = torch.arange(K, device=DEVICE)
        p = SUPPORT_MASK[s] / SUPPORT_MASK[s].sum()
        for t in range(diffusion.T, 0, -1):
            p_x0 = torch.softmax(model(x, xt, torch.full((K,), t, device=DEVICE)).double(), 1).cpu().numpy()
            kern = np.einsum("ki,kij->kj", p_x0, diffusion.posterior[t])
            kern = np.where(SUPPORT_MASK[s][None, :], kern, 0.0)
            kern /= kern.sum(axis=1, keepdims=True)
            p = p @ kern
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(times))


def lookup_latency_us(policy: np.ndarray, n: int = 100_000) -> float:
    table = policy.argmax(axis=1)
    rng = np_rng("evaluation", "lookup")
    qs = rng.integers(0, S_N, size=n)
    t0 = time.perf_counter()
    acc = 0
    for s in qs:
        acc += int(table[s])
    return 1e6 * (time.perf_counter() - t0) / n


DIFFUSION_CODE_HASH = code_hash(sinusoidal_time_embedding, CategoricalDenoiser, CategoricalDiffusion,
                                build_denoiser, train_diffusion_prior, init_model, OfflineTensorBuffer, stream_seed,
                                torch_generator, generator_state, restore_generator)
SCHEDULE_AUDIT = CategoricalDiffusion(A_N, CFG.diffusion_T, CFG.diffusion_beta_start, CFG.diffusion_beta_end).schedule_audit()
print("Diffusion schedule audit:", SCHEDULE_AUDIT)
save_json(SCHEDULE_AUDIT, TABLE_DIR / "diffusion_schedule_audit.json")
