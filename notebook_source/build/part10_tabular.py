# %% [markdown]
# ## 9. Tabular learners: BC, tabular CQL, tabular IQL (adaptation), tabular discrete BCQ (adaptation), and dataset-estimated support
#
# * **Tabular IQL** is an adaptation of IQL (expectile value regression + advantage-weighted extraction
#   from logged action counts), not a reproduction of every detail of neural IQL.
# * **Tabular discrete BCQ** follows the discrete-BCQ rule of Fujimoto et al. (2019): actions whose
#   behaviour probability relative to the most likely action is below τ (= 0.3) are excluded from both the
#   backup and the extraction. The behaviour model here is the count-based maximum-likelihood estimate (the
#   tabular analogue of BCQ's learned behaviour network). It is listed separately from simple
#   **count-threshold support masks**, which are *not* BCQ.
# * Every learner returns its native export plus explicitly labelled masked-greedy variants; the
#   admissible projection of every policy is added generically in Section 19.

# %%
# ============================================================
# 9. Tabular learners and dataset-estimated support
# ============================================================
EXECLOG.mark("9. Tabular learners")

COST_ROWS: List[dict] = []   # training / construction cost accounting (audit M06)


def tabular_identity(algo: str, hp: dict, seed: int, bundle: DataBundle, code: str) -> dict:
    return {"algo": algo, "hparams": hp, "seed": int(seed), "run_mode": RUN_MODE, "data": bundle.identity(),
            "benchmark": BENCH.identity, "code": code}


def _tabular_tensors(bundle: DataBundle):
    d = bundle.data
    t = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt, device=DEVICE)
    return (t(d["state"], torch.long), t(d["action"], torch.long), t(d["reward"], torch.float32),
            t(d["next_state"], torch.long), t(d["terminal"], torch.float32), len(d["state"]))


def train_behavior_cloning(bundle: DataBundle) -> Dict[str, np.ndarray]:
    """Empirical (maximum-likelihood) behaviour cloning; unseen states use the global action distribution."""
    counts = bundle.counts.astype(np.float64)
    global_pi = counts.sum(axis=0) / max(counts.sum(), 1.0)
    pi = np.tile(global_pi, (S_N, 1))
    seen = counts.sum(axis=1) > 0
    pi[seen] = counts[seen] / counts[seen].sum(axis=1, keepdims=True)
    native = one_hot_policy(pi.argmax(axis=1))
    # Masked greedy: most frequent logged admissible action; states without any logged admissible action
    # fall back to the global logged-action frequency over admissible actions (not to the lowest index).
    adm_seen = (counts * OPERATIONAL_MASK).sum(axis=1) > 0
    masked = one_hot_policy(masked_argmax(np.where(adm_seen[:, None], counts, global_pi[None, :]), OPERATIONAL_MASK))
    return {"stochastic": pi, "deterministic": native, "masked_greedy": masked, "counts": counts,
            "masked_fallback_states": int((~adm_seen & LIVE).sum())}


def train_tabular_cql(bundle: DataBundle, seed: int, alpha: float = None, steps: int = None,
                      lr: float = None, tag: str = "tabular_cql") -> Dict[str, Any]:
    alpha = CFG.tab_cql_alpha if alpha is None else alpha
    steps = CFG.tab_cql_steps if steps is None else steps
    lr = CFG.tab_cql_lr if lr is None else lr
    hp = {"alpha": alpha, "steps": steps, "lr": lr, "batch": CFG.batch_size, "gamma": CFG.gamma,
          "temperature": CFG.cql_temperature, "tau": CFG.target_tau, "grad_clip": CFG.grad_clip}
    ident = tabular_identity("tabular_cql", hp, seed, bundle, code_hash(train_tabular_cql, _tabular_tensors, stream_seed, torch_generator))
    label = f"{tag}/{bundle.name}/seed{seed}"
    payload = STORE.load_final(label, ident)
    if payload is not None:
        return {"q": payload["q"].numpy(), "log": pd.DataFrame(payload["log"]), "meta": payload["meta"]}
    STORE.require_training_allowed(label)
    t0 = time.perf_counter()
    s_all, a_all, r_all, ns_all, d_all, n = _tabular_tensors(bundle)
    q = nn.Parameter(torch.zeros(S_N, A_N, device=DEVICE))
    target_q = torch.zeros_like(q.data)
    opt = torch.optim.Adam([q], lr=lr)
    gen = torch_generator(stream_seed("minibatch", bundle.data_hash, seed), DEVICE)
    rows = []
    for step in trange(steps, desc=f"Tabular CQL {bundle.name} s{seed}", leave=False, disable=not SHOW_PROGRESS):
        idx = torch.randint(0, n, (CFG.batch_size,), generator=gen, device=DEVICE)
        s, a, r, ns, d = s_all[idx], a_all[idx], r_all[idx], ns_all[idx], d_all[idx]
        q_sa = q[s, a]
        with torch.no_grad():
            target = r + CFG.gamma * (1.0 - d) * target_q[ns].max(dim=1).values
        td = F.mse_loss(q_sa, target)
        cons = (torch.logsumexp(q[s] / CFG.cql_temperature, dim=1) * CFG.cql_temperature - q_sa).mean()
        loss = td + alpha * cons
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q], CFG.grad_clip)
        opt.step()
        with torch.no_grad():
            target_q.mul_(1.0 - CFG.target_tau).add_(q.data, alpha=CFG.target_tau)
        if step % CFG.log_every == 0 or step == steps - 1:
            rows.append({"step": step + 1, "loss": float(loss), "td_loss": float(td), "cql_loss": float(cons)})
    q_np = q.detach().cpu().numpy().astype(np.float64)
    meta = {"elapsed_s": time.perf_counter() - t0, "params": S_N * A_N, "steps": steps}
    STORE.save_final(label, ident, {"q": torch.from_numpy(q_np), "log": rows, "meta": meta})
    COST_ROWS.append({"model": "Tabular CQL", "label": label, "seed": seed, "train_s": meta["elapsed_s"],
                      "params": meta["params"], "steps": steps})
    return {"q": q_np, "log": pd.DataFrame(rows), "meta": meta}


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.square()).mean()


def iql_extraction(q: np.ndarray, v: np.ndarray, counts: np.ndarray, beta: float, max_weight: float):
    """Advantage-weighted extraction from logged counts. Returns AW policy, weighted counts and masked greedy."""
    adv = q - v[:, None]
    weights = np.exp(np.clip(beta * adv, -20.0, math.log(max_weight)))
    weighted = counts * weights
    global_pi = counts.sum(axis=0) / max(counts.sum(), 1.0)
    pi = np.tile(global_pi, (S_N, 1))
    seen = weighted.sum(axis=1) > 0
    pi[seen] = weighted[seen] / weighted[seen].sum(axis=1, keepdims=True)
    # Deterministic masked IQL: argmax of advantage-weighted scores over admissible actions. Q is never
    # updated for unlogged pairs (it stays exactly 0), so states without a logged admissible action fall back
    # to the global logged-action frequency over admissible actions instead of an arbitrary lowest index.
    adm_w = np.where(OPERATIONAL_MASK, weighted, -np.inf)
    no_logged_adm = ~(np.where(OPERATIONAL_MASK, weighted, 0.0) > 0).any(axis=1)
    masked_actions = adm_w.argmax(axis=1)
    masked_actions[no_logged_adm] = masked_argmax(np.broadcast_to(global_pi, q.shape), OPERATIONAL_MASK)[no_logged_adm]
    return pi, weighted, one_hot_policy(masked_actions), int((no_logged_adm & LIVE).sum())


def train_tabular_iql(bundle: DataBundle, seed: int, expectile: float = None, beta: float = None,
                      steps: int = None, lr: float = None, tag: str = "tabular_iql") -> Dict[str, Any]:
    expectile = CFG.iql_expectile if expectile is None else expectile
    beta = CFG.iql_beta if beta is None else beta
    steps = CFG.iql_steps if steps is None else steps
    lr = CFG.iql_lr if lr is None else lr
    hp = {"expectile": expectile, "beta": beta, "steps": steps, "lr": lr, "batch": CFG.batch_size,
          "gamma": CFG.gamma, "max_weight": CFG.iql_max_weight, "grad_clip": CFG.grad_clip}
    ident = tabular_identity("tabular_iql", hp, seed, bundle,
                             code_hash(train_tabular_iql, expectile_loss, iql_extraction, _tabular_tensors,
                                       stream_seed, torch_generator))
    label = f"{tag}/{bundle.name}/seed{seed}"
    payload = STORE.load_final(label, ident)
    if payload is None:
        STORE.require_training_allowed(label)
        t0 = time.perf_counter()
        s_all, a_all, r_all, ns_all, d_all, n = _tabular_tensors(bundle)
        q = nn.Parameter(torch.zeros(S_N, A_N, device=DEVICE))
        v = nn.Parameter(torch.zeros(S_N, device=DEVICE))
        opt_q = torch.optim.Adam([q], lr=lr)
        opt_v = torch.optim.Adam([v], lr=lr)
        gen = torch_generator(stream_seed("minibatch", bundle.data_hash, seed), DEVICE)
        rows = []
        for step in trange(steps, desc=f"Tabular IQL {bundle.name} s{seed}", leave=False, disable=not SHOW_PROGRESS):
            idx = torch.randint(0, n, (CFG.batch_size,), generator=gen, device=DEVICE)
            s, a, r, ns, d = s_all[idx], a_all[idx], r_all[idx], ns_all[idx], d_all[idx]
            v_loss = expectile_loss(q[s, a].detach() - v[s], expectile)
            opt_v.zero_grad(set_to_none=True)
            v_loss.backward()
            opt_v.step()
            with torch.no_grad():
                target = r + CFG.gamma * (1.0 - d) * v[ns]
            q_loss = F.mse_loss(q[s, a], target)
            opt_q.zero_grad(set_to_none=True)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_([q], CFG.grad_clip)
            opt_q.step()
            if step % CFG.log_every == 0 or step == steps - 1:
                rows.append({"step": step + 1, "q_loss": float(q_loss), "v_loss": float(v_loss)})
        q_np = q.detach().cpu().numpy().astype(np.float64)
        v_np = v.detach().cpu().numpy().astype(np.float64)
        meta = {"elapsed_s": time.perf_counter() - t0, "params": S_N * A_N + S_N, "steps": steps}
        payload = {"q": torch.from_numpy(q_np), "v": torch.from_numpy(v_np), "log": rows, "meta": meta}
        STORE.save_final(label, ident, payload)
        COST_ROWS.append({"model": "Tabular IQL", "label": label, "seed": seed, "train_s": meta["elapsed_s"],
                          "params": meta["params"], "steps": steps})
    q_np, v_np = payload["q"].numpy(), payload["v"].numpy()
    pi, weighted, masked, n_fallback = iql_extraction(q_np, v_np, bundle.counts.astype(np.float64), beta, CFG.iql_max_weight)
    return {"q": q_np, "v": v_np, "aw_policy": pi, "aw_scores": weighted, "deterministic": one_hot_policy(pi.argmax(axis=1)),
            "masked_greedy": masked, "masked_fallback_states": n_fallback, "log": pd.DataFrame(payload["log"]),
            "meta": payload["meta"]}


def behavior_threshold_set(counts: np.ndarray, tau: float) -> np.ndarray:
    counts = counts.astype(np.float64)
    global_pi = counts.sum(axis=0) / max(counts.sum(), 1.0)
    pb = np.tile(global_pi, (S_N, 1))
    seen = counts.sum(axis=1) > 0
    pb[seen] = counts[seen] / counts[seen].sum(axis=1, keepdims=True)
    allowed = pb / np.maximum(pb.max(axis=1, keepdims=True), 1e-30) >= tau
    allowed[~LIVE] = True
    return allowed


def train_tabular_bcq(bundle: DataBundle, seed: int, tau: float = None, steps: int = None, lr: float = None,
                      tag: str = "tabular_bcq") -> Dict[str, Any]:
    tau = CFG.bcq_threshold if tau is None else tau
    steps = CFG.bcq_steps if steps is None else steps
    lr = CFG.bcq_lr if lr is None else lr
    hp = {"tau": tau, "steps": steps, "lr": lr, "batch": CFG.batch_size, "gamma": CFG.gamma,
          "target_tau": CFG.target_tau, "grad_clip": CFG.grad_clip}
    ident = tabular_identity("tabular_bcq", hp, seed, bundle,
                             code_hash(train_tabular_bcq, behavior_threshold_set, _tabular_tensors,
                                       stream_seed, torch_generator))
    label = f"{tag}/{bundle.name}/seed{seed}"
    G = behavior_threshold_set(bundle.counts, tau)
    payload = STORE.load_final(label, ident)
    if payload is None:
        STORE.require_training_allowed(label)
        t0 = time.perf_counter()
        s_all, a_all, r_all, ns_all, d_all, n = _tabular_tensors(bundle)
        G_t = torch.as_tensor(G, device=DEVICE)
        q = nn.Parameter(torch.zeros(S_N, A_N, device=DEVICE))
        target_q = torch.zeros_like(q.data)
        opt = torch.optim.Adam([q], lr=lr)
        gen = torch_generator(stream_seed("minibatch", bundle.data_hash, seed), DEVICE)
        rows = []
        for step in trange(steps, desc=f"Tabular BCQ {bundle.name} s{seed}", leave=False, disable=not SHOW_PROGRESS):
            idx = torch.randint(0, n, (CFG.batch_size,), generator=gen, device=DEVICE)
            s, a, r, ns, d = s_all[idx], a_all[idx], r_all[idx], ns_all[idx], d_all[idx]
            with torch.no_grad():
                next_a = q[ns].masked_fill(~G_t[ns], -torch.inf).argmax(dim=1)
                target = r + CFG.gamma * (1.0 - d) * target_q[ns].gather(1, next_a[:, None]).squeeze(1)
            loss = F.mse_loss(q[s, a], target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([q], CFG.grad_clip)
            opt.step()
            with torch.no_grad():
                target_q.mul_(1.0 - CFG.target_tau).add_(q.data, alpha=CFG.target_tau)
            if step % CFG.log_every == 0 or step == steps - 1:
                rows.append({"step": step + 1, "td_loss": float(loss)})
        q_np = q.detach().cpu().numpy().astype(np.float64)
        meta = {"elapsed_s": time.perf_counter() - t0, "params": S_N * A_N, "steps": steps}
        payload = {"q": torch.from_numpy(q_np), "log": rows, "meta": meta}
        STORE.save_final(label, ident, payload)
        COST_ROWS.append({"model": "Tabular BCQ", "label": label, "seed": seed, "train_s": meta["elapsed_s"],
                          "params": meta["params"], "steps": steps})
    q_np = payload["q"].numpy()
    native = one_hot_policy(masked_argmax(q_np, G))
    both = G & OPERATIONAL_MASK
    both_ok = both.any(axis=1)
    masked_mask = np.where(both_ok[:, None], both, OPERATIONAL_MASK)
    return {"q": q_np, "allowed": G, "deterministic": native,
            "masked_greedy": one_hot_policy(masked_argmax(q_np, masked_mask)),
            "masked_fallback_states": int((~both_ok & LIVE).sum()),
            "log": pd.DataFrame(payload["log"]), "meta": payload["meta"]}


# ---------- Dataset-estimated support (audit M02) ----------
def estimated_support_mask(counts: np.ndarray, rule: str, param: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Support estimated from the simulated training data — a different object from the released mask."""
    counts = np.asarray(counts)
    if rule == "count_at_least":
        est = counts >= param
    elif rule == "behavior_ratio_at_least":
        est = behavior_threshold_set(counts, param)
    else:
        raise ValueError(rule)
    est = est.copy()
    est[~LIVE] = True
    empty = LIVE & ~est.any(axis=1)
    for s in np.flatnonzero(empty):   # fallback: most frequently logged action (or all actions if unseen)
        if counts[s].sum() > 0:
            est[s, int(counts[s].argmax())] = True
        else:
            est[s] = True
    rel = SUPPORT_MASK[LIVE]
    e = est[LIVE]
    inter = (rel & e).sum(axis=1)
    union = (rel | e).sum(axis=1)
    report = {
        "rule": rule, "param": param,
        "estimated_pairs_live": int(e.sum()),
        "released_pairs_live": int(rel.sum()),
        "false_exclusions": int((rel & ~e).sum()),
        "false_inclusions": int((~rel & e).sum()),
        "empty_rows_before_fallback": int(empty.sum()),
        "fallback_rows": int(empty.sum()),
        "mean_state_jaccard": float(np.mean(inter / np.maximum(union, 1))),
        "states_identical_to_released": int((e == rel).all(axis=1).sum()),
    }
    return est, report


print("Tabular learners ready.")
