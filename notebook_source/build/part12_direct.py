# %% [markdown]
# ## 11. Direct categorical policies: matched-capacity neural BC and advantage-weighted categorical imitation (audit M06)
#
# The action space has only 25 categories, so a direct categorical network can represent any per-state
# action distribution without a denoising chain. These baselines use the same state encoder as the diffusion
# denoiser, the same optimiser, learning rate, batch size, minibatch stream and number of updates as the
# diffusion prior. The advantage-weighted variant reweights the cross-entropy by
# `exp(clip(β·A(s,a), −20, log w_max))` (batch-normalised) with advantages from tabular IQL of the same seed.

# %%
# ============================================================
# 11. Direct categorical policy networks
# ============================================================
EXECLOG.mark("11. Direct categorical policies")


class CategoricalPolicyNet(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
                                           nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.state_encoder(x))


POLICYNET_CODE_HASH = None


def train_policy_net(bundle: DataBundle, seed: int, mode: str = "bc", advantages: Optional[np.ndarray] = None,
                     tag: str = "policy_net", store: Optional[CheckpointStore] = None) -> Dict[str, Any]:
    assert mode in {"bc", "aw"}
    store = STORE if store is None else store
    hp = {"mode": mode, "steps": CFG.policy_net_steps, "lr": CFG.policy_net_lr, "batch": CFG.batch_size,
          "hidden": CFG.hidden_dim, "weight_decay": CFG.weight_decay, "grad_clip": CFG.grad_clip}
    if mode == "aw":
        hp.update(aw_beta=CFG.aw_beta, aw_max_weight=CFG.aw_max_weight, advantages_hash=array_hash(advantages))
    ident = {"algo": "categorical_policy_net", "hparams": hp, "seed": int(seed), "run_mode": RUN_MODE,
             "data": bundle.identity(), "benchmark": BENCH.identity, "code": POLICYNET_CODE_HASH}
    label = f"{tag}/{mode}/{bundle.name}/seed{seed}"
    payload = store.load_final(label, ident)
    if payload is None:
        store.require_training_allowed(label)
        t0 = time.perf_counter()
        reset_peak_gpu_memory()
        buf = bundle.buffer()
        model = init_model(lambda: CategoricalPolicyNet(bundle.input_dim, A_N, CFG.hidden_dim),
                           f"policy_net_{bundle.input_dim}x{CFG.hidden_dim}", seed)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG.policy_net_lr, weight_decay=CFG.weight_decay)
        gen = torch_generator(stream_seed("minibatch", bundle.data_hash, seed), DEVICE)
        adv_t = None
        if mode == "aw":
            adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)
        rows, start = [], 0
        prior_elapsed, prior_peak, segments = 0.0, 0.0, 1
        resume = store.load_resume(label, ident, map_location=DEVICE)
        if resume is not None:
            model.load_state_dict(resume["model"])
            opt.load_state_dict(resume["optimizer"])
            restore_generator(gen, resume["generator"])
            rows, start = resume["log"], int(resume["step"])
            prior_elapsed = float(resume.get("elapsed_s", 0.0))
            prior_peak = float(resume.get("peak_gpu_mb") or 0.0)
            segments = int(resume.get("segments", 1)) + 1
        model.train()
        for step in trange(start, CFG.policy_net_steps, desc=f"policy-net {mode} {bundle.name} s{seed}",
                           leave=False, disable=not SHOW_PROGRESS):
            b = buf.sample(CFG.batch_size, gen)
            logits = model(b["x"])
            ce = F.cross_entropy(logits, b["a"], reduction="none")
            if mode == "aw":
                w = torch.exp(torch.clamp(CFG.aw_beta * adv_t[b["s_idx"], b["a"]], -20.0, math.log(CFG.aw_max_weight)))
                w = w / w.mean()
                loss = (w * ce).mean()
            else:
                loss = ce.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            opt.step()
            if step % CFG.log_every == 0 or step == CFG.policy_net_steps - 1:
                rows.append({"step": step + 1, "loss": float(loss),
                             "accuracy": float((logits.argmax(1) == b["a"]).float().mean())})
            if (step + 1) % CFG.resume_every == 0 and step + 1 < CFG.policy_net_steps:
                store.save_resume(label, ident, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                                 "generator": generator_state(gen), "step": step + 1, "log": rows,
                                                 "elapsed_s": prior_elapsed + (time.perf_counter() - t0),
                                                 "peak_gpu_mb": max(prior_peak, peak_gpu_memory_mb() or 0.0),
                                                 "segments": segments})
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(torch.as_tensor(bundle.all_x, device=DEVICE)).double(), dim=1).cpu()
        meta = {"elapsed_s": prior_elapsed + (time.perf_counter() - t0), "params": count_parameters(model),
                "peak_gpu_mb": (max(prior_peak, peak_gpu_memory_mb() or 0.0) if torch.cuda.is_available() else None),
                "steps": CFG.policy_net_steps, "normalizer": bundle.normalizer.to_dict(),
                "resumed": resume is not None, "segments": segments}
        payload = {"model": model.state_dict(), "probs": probs, "log": rows, "meta": meta}
        store.save_final(label, ident, payload)
        COST_ROWS.append({"model": f"Neural categorical ({mode})", "label": label, "seed": seed,
                          "train_s": meta["elapsed_s"], "params": meta["params"],
                          "peak_gpu_mb": meta["peak_gpu_mb"], "steps": CFG.policy_net_steps})
    probs = payload["probs"].numpy()
    probs = probs / probs.sum(axis=1, keepdims=True)
    return {"probs": probs, "log": pd.DataFrame(payload["log"]), "meta": payload["meta"], "label": label}


POLICYNET_CODE_HASH = code_hash(CategoricalPolicyNet, train_policy_net, init_model, OfflineTensorBuffer, stream_seed,
                                torch_generator, generator_state, restore_generator)
print("Direct categorical policy code hash:", POLICYNET_CODE_HASH[:16])
