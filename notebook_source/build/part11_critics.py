# %% [markdown]
# ## 10. Twin-dueling critics: Deep CQL, legacy hard AM-CQL, two repaired hard designs, soft AM-CQL (audit C02, M05)
#
# All variants share one trainer parameterised by a `CriticSpec`, the same architecture, the same
# initial weights for a given seed, and the same minibatch index sequence (common random numbers).
#
# | Variant | CQL regulariser domain | Backup | Extraction | Status |
# |---|---|---|---|---|
# | Deep CQL | full action set | unmasked | unmasked min-critic greedy | baseline |
# | AM-CQL (Hard, legacy) | log-sum-exp over admissible actions **minus the unmasked logged-action Q** (incompatible domain: for an excluded logged action the regulariser's derivative w.r.t. its Q is −1) | masked | masked | **failure-mode control only** |
# | AM-CQL (Hard, design A: full-action CQL) | full action set (standard CQL) | masked | masked | repaired comparator |
# | AM-CQL (Hard, design B: admissible-row CQL) | admissible log-sum-exp, applied **only to rows whose logged action is admissible**; normalisation = mean over those rows (`conditional`) or sum / batch size (`batch`) | masked | masked | repaired, preregistered primary critic |
# | AM-CQL (Soft, legacy) | full | unmasked | unmasked min-critic greedy (training penalty uses the mean-critic softmax) | analysed in Section 24 |
#
# Additional switches used by the ablations: TD rows (`all` vs `admissible_only` — inadmissible logged
# transitions carry mean-imputed simulator outcomes, not observations of the forbidden intervention),
# TD loss (Huber vs squared), conservative weight α, a separately labelled TD-target clipping ablation
# (a stabilisation choice, **not** a repair of the regulariser), a separately labelled discounted
# (γ = 0.99, different estimand) ablation, and feature removal (Section 29).
#
# Diagnostics logged during training: TD / regulariser / total loss, gradient norm (pre-clipping), TD-target
# mean/min/max, batch admissible-Q quantiles and absolute maximum, within-state admissible action spread,
# and every `diag_every` steps the **full-buffer Bellman residual** and full Q-table statistics, including the
# ratio of the common per-state offset to the action spread and the number of states whose action
# differences are within 64 float32 ulps of the offset (where huge offsets could obscure action differences).

# %%
# ============================================================
# 10. Neural critics
# ============================================================
EXECLOG.mark("10. Neural critics")


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                       nn.Linear(hidden_dim // 2, num_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        a = self.advantage(h)
        return self.value(h) + a - a.mean(dim=1, keepdim=True)


class TwinDuelingCQL(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = DuelingQNetwork(input_dim, num_actions, hidden_dim)
        self.q2 = DuelingQNetwork(input_dim, num_actions, hidden_dim)
        self.tq1 = copy.deepcopy(self.q1)
        self.tq2 = copy.deepcopy(self.q2)
        for p in list(self.tq1.parameters()) + list(self.tq2.parameters()):
            p.requires_grad_(False)

    def online(self, x):
        return self.q1(x), self.q2(x)

    @torch.no_grad()
    def target(self, x):
        return self.tq1(x), self.tq2(x)


@torch.no_grad()
def polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def apply_hard_mask(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Finite large-negative masking (legacy behaviour); every row of the operational mask is non-empty."""
    return q.masked_fill(~mask, torch.finfo(q.dtype).min)


@dataclass(frozen=True)
class CriticSpec:
    key: str
    display: str
    reg_domain: str = "full"                 # full | masked_legacy | masked_admissible_rows
    reg_normalization: str = "conditional"   # conditional | batch   (masked_admissible_rows only)
    masked_backup: bool = False
    masked_extraction: bool = False
    td_loss: str = "huber"                   # huber | mse
    td_rows: str = "all"                     # all | admissible_only
    alpha: float = 1.0
    temperature: float = 1.0
    target_clip: Optional[Tuple[float, float]] = None
    gamma: float = 1.0
    soft: bool = False
    soft_tolerance: float = 0.0
    soft_temperature: float = 1.0
    soft_aggregation: str = "mean"           # mean | min
    steps: int = 0
    status: str = ""

    def hparams(self) -> dict:
        d = asdict(self)
        d.pop("display")
        d.pop("status")
        d.update(lr=CFG.lr, weight_decay=CFG.weight_decay, batch=CFG.batch_size, hidden=CFG.hidden_dim,
                 grad_clip=CFG.grad_clip, target_tau=CFG.target_tau,
                 soft_initial_lambda=CFG.soft_initial_lambda, soft_dual_lr=CFG.soft_dual_lr,
                 soft_lambda_max=CFG.soft_lambda_max)
        return d


def make_spec(key: str, display: str, **kw) -> CriticSpec:
    kw.setdefault("steps", CFG.deep_steps)
    kw.setdefault("alpha", CFG.cql_alpha)
    kw.setdefault("temperature", CFG.cql_temperature)
    kw.setdefault("gamma", CFG.gamma)
    return CriticSpec(key=key, display=display, **kw)


SPEC_DEEP = make_spec("deep", "Deep CQL", status="baseline")
SPEC_HARD_LEGACY = make_spec("hard_legacy", "AM-CQL (Hard, legacy)", reg_domain="masked_legacy",
                             masked_backup=True, masked_extraction=True, status="failure-mode control")
SPEC_HARD_A = make_spec("hard_fullcql", "AM-CQL (Hard, design A: full-action CQL)", reg_domain="full",
                        masked_backup=True, masked_extraction=True, status="repaired comparator")
SPEC_HARD_B = make_spec("hard_admrows", "AM-CQL (Hard, design B: admissible-row CQL)",
                        reg_domain="masked_admissible_rows", reg_normalization="conditional",
                        masked_backup=True, masked_extraction=True, status="repaired (preregistered primary)")
SPEC_SOFT = make_spec("soft_legacy", "AM-CQL (Soft, legacy)", soft=True,
                      soft_tolerance=CFG.soft_target_inadmissible_mass,
                      soft_temperature=CFG.soft_policy_temperature, status="soft surrogate")
MAIN_CRITIC_SPECS = [SPEC_DEEP, SPEC_HARD_LEGACY, SPEC_HARD_A, SPEC_HARD_B, SPEC_SOFT]
SPEC_BY_KEY = {s.key: s for s in MAIN_CRITIC_SPECS}
assert PREREG["primary_hard_critic"] in SPEC_BY_KEY


def cql_regularizer(q: torch.Tensor, a: torch.Tensor, mask: torch.Tensor, adm: torch.Tensor,
                    spec: CriticSpec) -> torch.Tensor:
    T = spec.temperature
    assert T > 0
    q_log = q.gather(1, a[:, None]).squeeze(1)
    if spec.reg_domain == "full":
        return (T * torch.logsumexp(q / T, dim=1) - q_log).mean()
    if spec.reg_domain == "masked_legacy":
        # Original (defective) objective, retained only as a labelled failure-mode control.
        return (T * torch.logsumexp(apply_hard_mask(q, mask) / T, dim=1) - q_log).mean()
    if spec.reg_domain == "masked_admissible_rows":
        if not bool(adm.any()):
            return q.sum() * 0.0
        q_ok, m_ok = q[adm], mask[adm]
        assert bool(m_ok.any(dim=1).all())
        lse = T * torch.logsumexp(q_ok.masked_fill(~m_ok, -torch.inf) / T, dim=1)
        contrib = lse - q_log[adm]
        if spec.reg_normalization == "conditional":
            return contrib.mean()
        if spec.reg_normalization == "batch":
            return contrib.sum() / q.shape[0]
        raise ValueError(spec.reg_normalization)
    raise ValueError(spec.reg_domain)


def td_elementwise(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "huber":
        return F.smooth_l1_loss(pred, target, reduction="none")
    if kind == "mse":
        return F.mse_loss(pred, target, reduction="none")
    raise ValueError(kind)


def soft_policy_logits(q1: torch.Tensor, q2: torch.Tensor, aggregation: str) -> torch.Tensor:
    return 0.5 * (q1 + q2) if aggregation == "mean" else torch.minimum(q1, q2)


def inadmissible_probability_mass(logits: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    return (torch.softmax(logits / temperature, dim=1) * (~mask).float()).sum(dim=1)


@torch.no_grad()
def critic_tables(model: TwinDuelingCQL, all_x: np.ndarray, batch: int = 2048,
                  device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    x = torch.as_tensor(all_x, dtype=torch.float32, device=DEVICE if device is None else device)
    q1s, q2s = [], []
    for i in range(0, len(x), batch):
        a, b = model.online(x[i:i + batch])
        q1s.append(a.double().cpu())
        q2s.append(b.double().cpu())
    return torch.cat(q1s).numpy(), torch.cat(q2s).numpy()


@torch.no_grad()
def full_diagnostics(model: TwinDuelingCQL, buf: OfflineTensorBuffer, bundle: DataBundle, spec: CriticSpec,
                     step: int) -> dict:
    q1, q2 = critic_tables(model, bundle.all_x)
    qmin = np.minimum(q1, q2)[LIVE]
    m = SUPPORT_MASK[LIVE]
    adm_vals = qmin[m]
    q_adm = np.where(m, qmin, np.nan)
    offset = np.nanmean(q_adm, axis=1)
    spread = np.nanmax(q_adm, axis=1) - np.nanmin(q_adm, axis=1)
    multi = m.sum(axis=1) >= 2
    eps32 = float(np.finfo(np.float32).eps)
    near_res = multi & (spread <= 64 * eps32 * np.maximum(np.abs(offset), 1.0))
    sq, n_rows = 0.0, 0
    abs_sum = 0.0
    sq_adm, n_adm, sq_in, n_in = 0.0, 0, 0.0, 0
    model.eval()
    for b in buf.chunks():
        o1, o2 = model.online(b["x"])
        q_sa = torch.minimum(o1, o2).gather(1, b["a"][:, None]).squeeze(1)
        n_on = 0.5 * sum(model.online(b["nx"]))
        if spec.masked_backup:
            n_on = apply_hard_mask(n_on, b["next_mask"])
        na = n_on.argmax(dim=1)
        t1, t2 = model.target(b["nx"])
        tq = torch.minimum(t1, t2)
        if spec.masked_backup:
            tq = apply_hard_mask(tq, b["next_mask"])
        y = b["r"] + spec.gamma * (1.0 - b["terminal"]) * tq.gather(1, na[:, None]).squeeze(1)
        diff = (q_sa - y).double()
        d2 = diff ** 2
        sq += float(d2.sum())
        abs_sum += float(diff.abs().sum())
        n_rows += int(diff.numel())
        adm = b["adm"]
        sq_adm += float(d2[adm].sum())
        n_adm += int(adm.sum())
        sq_in += float(d2[~adm].sum())
        n_in += int((~adm).sum())
    model.train()
    td_only_adm = spec.td_rows == "admissible_only"
    return {
        "step": step,
        "bellman_residual_rmse_full_buffer": math.sqrt(sq / max(n_rows, 1)),
        "bellman_residual_rmse_adm_logged_rows": math.sqrt(sq_adm / max(n_adm, 1)),
        "bellman_residual_rmse_inadm_logged_rows": math.sqrt(sq_in / n_in) if n_in else float("nan"),
        "bellman_residual_rmse_td_objective_rows": (math.sqrt(sq_adm / max(n_adm, 1)) if td_only_adm
                                                    else math.sqrt(sq / max(n_rows, 1))),
        "bellman_residual_mae_full_buffer": abs_sum / max(n_rows, 1),
        "q_adm_max_abs": float(np.abs(adm_vals).max()),
        "q_adm_min": float(adm_vals.min()),
        "q_adm_max": float(adm_vals.max()),
        "q_adm_q01": float(np.quantile(adm_vals, 0.01)),
        "q_adm_q50": float(np.quantile(adm_vals, 0.50)),
        "q_adm_q99": float(np.quantile(adm_vals, 0.99)),
        "q_inadm_max": float(qmin[~m].max()) if (~m).any() else float("nan"),
        "mean_state_spread": float(spread[multi].mean()) if multi.any() else 0.0,
        "median_offset_to_spread_ratio": float(np.median(np.abs(offset[multi]) / np.maximum(spread[multi], 1e-30))) if multi.any() else 0.0,
        "states_spread_within_64_ulp_of_offset": int(near_res.sum()),
        "critic_disagreement_frac": float((q1[LIVE].argmax(1) != q2[LIVE].argmax(1)).mean()),
    }


CRITIC_CODE_HASH = None  # set after all helper definitions below


def critic_identity(spec: CriticSpec, seed: int, bundle: DataBundle) -> dict:
    return {"algo": "twin_dueling_critic", "spec": spec.hparams(), "seed": int(seed), "run_mode": RUN_MODE,
            "data": bundle.identity(), "benchmark": BENCH.identity, "code": CRITIC_CODE_HASH}


def train_critic(spec: CriticSpec, bundle: DataBundle, seed: int, tag: str = "critic",
                 interrupt_at: Optional[int] = None, store: Optional[CheckpointStore] = None) -> Optional[Dict[str, Any]]:
    """Train (or load) one critic. `interrupt_at` simulates a disconnect for the resume test."""
    store = STORE if store is None else store
    ident = critic_identity(spec, seed, bundle)
    label = f"{tag}/{spec.key}/{bundle.name}/seed{seed}"
    payload = store.load_final(label, ident)
    if payload is None:
        store.require_training_allowed(label)
        payload = _train_critic_loop(spec, bundle, seed, ident, label, store, interrupt_at)
        if payload is None:
            return None
    return {"spec": spec, "seed": seed, "bundle": bundle.name, "q1": payload["q1"].numpy(),
            "q2": payload["q2"].numpy(), "log": pd.DataFrame(payload["log"]),
            "diag": pd.DataFrame(payload["diag"]), "meta": payload["meta"], "label": label}


def _train_critic_loop(spec, bundle, seed, ident, label, store, interrupt_at):
    t0 = time.perf_counter()
    reset_peak_gpu_memory()
    buf = bundle.buffer()
    model = init_model(lambda: TwinDuelingCQL(bundle.input_dim, A_N, CFG.hidden_dim),
                       f"twin_dueling_{bundle.input_dim}x{CFG.hidden_dim}", seed)
    params = list(model.q1.parameters()) + list(model.q2.parameters())
    opt = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=CFG.weight_decay)
    gen = torch_generator(stream_seed("minibatch", bundle.data_hash, seed), DEVICE)
    lam = torch.tensor(CFG.soft_initial_lambda if spec.soft else 0.0, device=DEVICE)
    rows, diag_rows, start = [], [], 0
    prior_elapsed, prior_peak, segments = 0.0, 0.0, 1
    resume = store.load_resume(label, ident, map_location=DEVICE)
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        restore_generator(gen, resume["generator"])
        lam = resume["lambda"].to(device=DEVICE, dtype=torch.float32)
        rows, diag_rows, start = resume["log"], resume["diag"], int(resume["step"])
        prior_elapsed = float(resume.get("elapsed_s", 0.0))
        prior_peak = float(resume.get("peak_gpu_mb") or 0.0)
        segments = int(resume.get("segments", 1)) + 1
        print(f"  resumed {label} at step {start}")
    model.train()
    for step in trange(start, spec.steps, desc=f"{spec.key} {bundle.name} s{seed}", leave=False, disable=not SHOW_PROGRESS):
        b = buf.sample(CFG.batch_size, gen)
        q1, q2 = model.online(b["x"])
        q1_sa = q1.gather(1, b["a"][:, None]).squeeze(1)
        q2_sa = q2.gather(1, b["a"][:, None]).squeeze(1)
        with torch.no_grad():
            n1, n2 = model.online(b["nx"])
            nscore = 0.5 * (n1 + n2)
            if spec.masked_backup:
                nscore = apply_hard_mask(nscore, b["next_mask"])
            na = nscore.argmax(dim=1)
            t1, t2 = model.target(b["nx"])
            tq = torch.minimum(t1, t2)
            if spec.masked_backup:
                tq = apply_hard_mask(tq, b["next_mask"])
            y = b["r"] + spec.gamma * (1.0 - b["terminal"]) * tq.gather(1, na[:, None]).squeeze(1)
            if spec.target_clip is not None:
                y = y.clamp(spec.target_clip[0], spec.target_clip[1])
        w = b["adm"].float() if spec.td_rows == "admissible_only" else torch.ones_like(y)
        wsum = w.sum().clamp_min(1.0)
        td1 = (td_elementwise(q1_sa, y, spec.td_loss) * w).sum() / wsum
        td2 = (td_elementwise(q2_sa, y, spec.td_loss) * w).sum() / wsum
        reg1 = cql_regularizer(q1, b["a"], b["mask"], b["adm"], spec)
        reg2 = cql_regularizer(q2, b["a"], b["mask"], b["adm"], spec)
        loss = td1 + td2 + spec.alpha * (reg1 + reg2)
        mass = torch.zeros((), device=DEVICE)
        if spec.soft:
            logits = soft_policy_logits(q1, q2, spec.soft_aggregation)
            mass = inadmissible_probability_mass(logits, b["mask"], spec.soft_temperature).mean()
            loss = loss + lam.detach() * (mass - spec.soft_tolerance)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, CFG.grad_clip)
        opt.step()
        if spec.soft:
            with torch.no_grad():
                lam.add_(CFG.soft_dual_lr * (mass.detach() - spec.soft_tolerance)).clamp_(0.0, CFG.soft_lambda_max)
        polyak_update(model.tq1, model.q1, CFG.target_tau)
        polyak_update(model.tq2, model.q2, CFG.target_tau)

        if step % CFG.log_every == 0 or step == spec.steps - 1:
            with torch.no_grad():
                qm = torch.minimum(q1, q2)
                adm_q = qm[b["mask"]]
                spread = (qm.masked_fill(~b["mask"], -torch.inf).max(1).values
                          - qm.masked_fill(~b["mask"], torch.inf).min(1).values)
                qs = torch.quantile(adm_q.float(), torch.tensor([0.01, 0.5, 0.99], device=DEVICE))
            rows.append({
                "step": step + 1, "loss": float(loss), "td_loss": float(td1 + td2), "cql_loss": float(reg1 + reg2),
                "grad_norm": float(grad_norm), "inadmissible_mass_surrogate": float(mass),
                "primal_residual": float(mass) - spec.soft_tolerance if spec.soft else 0.0,
                "lambda": float(lam), "target_mean": float(y.mean()), "target_min": float(y.min()),
                "target_max": float(y.max()), "q_adm_q01": float(qs[0]), "q_adm_q50": float(qs[1]),
                "q_adm_q99": float(qs[2]), "q_adm_max_abs": float(adm_q.abs().max()),
                "mean_admissible_spread": float(spread.mean()), "frac_rows_admissible": float(b["adm"].float().mean()),
            })
        if (step + 1) % CFG.diag_every == 0 or step == spec.steps - 1:
            diag_rows.append(full_diagnostics(model, buf, bundle, spec, step + 1))
        if (step + 1) % CFG.resume_every == 0 and step + 1 < spec.steps:
            store.save_resume(label, ident, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                             "generator": generator_state(gen), "lambda": lam.detach().cpu(),
                                             "step": step + 1, "log": rows, "diag": diag_rows,
                                             "elapsed_s": prior_elapsed + (time.perf_counter() - t0),
                                             "peak_gpu_mb": max(prior_peak, peak_gpu_memory_mb() or 0.0),
                                             "segments": segments})
            if interrupt_at is not None and step + 1 >= interrupt_at:
                return None
    q1t, q2t = critic_tables(model, bundle.all_x)
    meta = {"elapsed_s": prior_elapsed + (time.perf_counter() - t0), "params": count_parameters(model),
            "peak_gpu_mb": (max(prior_peak, peak_gpu_memory_mb() or 0.0) if torch.cuda.is_available() else None),
            "steps": spec.steps, "final_lambda": float(lam), "normalizer": bundle.normalizer.to_dict(),
            "resumed": resume is not None, "segments": segments}
    payload = {"model": model.state_dict(), "q1": torch.from_numpy(q1t), "q2": torch.from_numpy(q2t),
               "log": rows, "diag": diag_rows, "meta": meta}
    store.save_final(label, ident, payload)
    COST_ROWS.append({"model": spec.display, "label": label, "seed": seed, "train_s": meta["elapsed_s"],
                      "params": meta["params"], "peak_gpu_mb": meta["peak_gpu_mb"], "steps": spec.steps})
    return payload


def load_critic_model(result: Dict[str, Any], bundle: DataBundle) -> TwinDuelingCQL:
    """Rebuild a critic network from its identity-validated checkpoint (used by artifact replay)."""
    ident = critic_identity(result["spec"], result["seed"], bundle)
    payload = STORE.load_final(result["label"], ident, map_location=DEVICE, respect_force=False)
    if payload is None:
        raise MissingArtifactError(result["label"])
    model = TwinDuelingCQL(bundle.input_dim, A_N, CFG.hidden_dim).to(DEVICE)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


CRITIC_CODE_HASH = code_hash(DuelingQNetwork, TwinDuelingCQL, polyak_update, apply_hard_mask, CriticSpec, make_spec,
                             cql_regularizer, td_elementwise, soft_policy_logits, inadmissible_probability_mass,
                             critic_tables, full_diagnostics, train_critic, _train_critic_loop, init_model,
                             OfflineTensorBuffer, stream_seed, torch_generator, generator_state, restore_generator)


# ---------- Policy extraction from critic tables ----------
def critic_extractions(res: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, str, bool]]:
    """Return {suffix: (policy, extraction description, stochastic)} for one trained critic."""
    spec = res["spec"]
    q1, q2 = res["q1"], res["q2"]
    qmin, qmean = np.minimum(q1, q2), 0.5 * (q1 + q2)
    out = {"": (greedy_policy(qmin, masked=spec.masked_extraction),
                "masked min-critic greedy" if spec.masked_extraction else "unmasked min-critic greedy", False)}
    if not spec.masked_extraction:
        out["[masked greedy]"] = (greedy_policy(qmin, masked=True), "masked min-critic greedy (inference mask)", False)
    else:
        # Constraint-placement control: mask used in training only (backup / regulariser), unmasked export.
        out["[unmasked greedy]"] = (greedy_policy(qmin, masked=False),
                                    "unmasked min-critic greedy (mask used in training only)", False)
    if spec.soft:
        logits = soft_policy_logits(torch.from_numpy(q1), torch.from_numpy(q2), spec.soft_aggregation).numpy()
        z = logits / spec.soft_temperature
        z = z - z.max(axis=1, keepdims=True)
        sm = np.exp(z)
        sm /= sm.sum(axis=1, keepdims=True)
        out["[aligned greedy: mean critic]"] = (greedy_policy(qmean, masked=False),
                                                "unmasked greedy on the training aggregation (mean critic)", False)
        out["[aligned stochastic: training softmax]"] = (sm, "training softmax (mean critic, T_p), stochastic", True)
    return out


print("Neural critic code hash:", CRITIC_CODE_HASH[:16])
