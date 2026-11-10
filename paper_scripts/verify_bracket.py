"""Numerical verification of the bracketing theorem (thm:bracket) and the certified-comparison corollary
(cor:certified) on random finite absorbing MDPs, by exact linear solves (no rollouts).

Setting (paper Sec. 2): live states S, absorbing states T (rewards paid on entry, gamma = 1), mask M(s) != {},
true rows P* on supported pairs shared by every completion; completions differ only on unsupported live pairs:
  P_W    convex imputation (W random per pair, or uniform = mean rule),
  P_term jump to the sink s_dagger paying r_term = R(s_dagger),
  P*     the 'true' completion (several families, below).
Checks
  A  lower bound (i):    J_term <= J*   for every proper truth when r_term = v_min (also when r_term = min_T R > 0,
                         and a failure case when the sink is not the worst absorbing state);
  B  upper bound (ii):   J* <= J_W     under intermediate harm (P* = (1-lam) P_W + lam delta_sink) and under
                         generic, policy-specific NHB rows that are not mixtures;
  C  counterexamples:    NHB fails (beneficial unsupported actions) -> upper bound fails; size <= p (v_max - v_min),
                         attained by a one-state construction; Garnet-type h < 0 rows fail NHB strictly;
  D  width (iii):        0 <= J_W - J_term <= p (v_max - v_min); J* - J_term <= p (v_max - v_min) for every truth;
                         the jump-to-best completion attains J_term + p (R_max - r_term) exactly;
  E  zero unsupported mass: admissible policies, and policies unsupported only on unreachable states, have width 0;
  F  properness:         proper under P or P_W  =>  proper under P_term and under every intermediate-harm mixture;
                         the converse fails (counted);
  G  sharpness/monotonicity: J(lam) non-increasing in lam (uniform and componentwise), endpoints J_W and J_term;
  H  certified comparisons: never wrong under NHB truths; error rates of single-rule point rankings;
  I  Garnet harm family: on states with K(s) subset of M(s), h in [0,1] is intermediate harm with
                         lam = h k / (m - h (m - k)); h < 0 violates NHB at every unsupported pair;
  J  P4 identity and the value range U in [min_T R, max_T R] (so U^W >= v_min = r_term).
Usage: python verify_bracket.py [--quick]
Writes ../theory_verification.json (next to main.tex): every raw check under "checks" and the counts quoted in the
paper (App. "Numerical check") under "summary". The paper quotes these counts only from that file.
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

TOL = 1e-10
QUICK = "--quick" in sys.argv
N_MDP = 60 if QUICK else 300          # MDPs per reward structure
N_POL = 14                            # policies per MDP
N_TRUTH = 4                           # truths per family per MDP

# reward structures on the absorbing states (index 0 is always the sink s_dagger)
REWARDS = {
    "gumbel_like(-1,+1,0)": np.array([-1.0, 1.0, 0.0]),   # death -1 (= v_min), discharge +1, timeout 0
    "garnet_icu_like(0,1)": np.array([0.0, 1.0]),          # death 0 (= v_min), goal/survival 1
    "random_neg(sink=min)": None,                           # random rewards, sink = min_T R <= 0
}


# --------------------------------------------------------------------------------------------------------------
# MDP machinery
# --------------------------------------------------------------------------------------------------------------
class MDP:
    def __init__(self, P, R, d0, mask, nS, sink=None):
        self.P = P                      # (N, A, N), absorbing rows self-loops
        self.R = R                      # (N,), zero on live states
        self.d0 = d0                    # (N,), supported on live states
        self.mask = mask                # (N, A) bool; all True on absorbing rows
        self.nS = nS                    # live states are 0..nS-1
        self.N, self.A = P.shape[0], P.shape[1]
        self.live = np.arange(nS)
        self.term = np.arange(nS, self.N)
        self.sink = nS if sink is None else sink
        rt = R[self.term]
        self.vmin, self.vmax = min(0.0, rt.min()), max(0.0, rt.max())
        self.Rmin_T, self.Rmax_T = rt.min(), rt.max()
        self.rterm = R[self.sink]
        self.bad = (~mask) & (np.arange(self.N) < nS)[:, None]   # unsupported live pairs

    def with_rows(self, P):
        return MDP(P, self.R, self.d0, self.mask, self.nS, self.sink)


def random_mdp(rng, R_T, *, nS=None, A=None, loops=False, orphan=False, branching=None):
    nS = nS or int(rng.integers(4, 31))
    A = A or int(rng.integers(2, 7))
    nT = len(R_T)
    N = nS + nT
    P = np.zeros((N, A, N))
    B = branching or int(rng.integers(1, min(4, nS) + 1))
    for s in range(nS):
        for a in range(A):
            succ = rng.choice(nS, size=B, replace=False)
            if orphan:
                succ = np.where(succ == nS - 1, (nS - 2) if nS > 1 else succ, succ)
            w = rng.dirichlet(np.ones(B))
            u = rng.random()
            haz = 0.0 if (loops and u < 0.6) else rng.uniform(0.05, 0.4)
            row = np.zeros(N)
            if loops and u < 0.2:                       # deterministic self-loop (improper if always chosen)
                succ, w = np.array([s]), np.array([1.0])
            np.add.at(row, succ, (1 - haz) * w)
            row[nS:] += haz * rng.dirichlet(np.ones(nT))
            P[s, a] = row
    for t in range(nS, N):
        P[t, :, t] = 1.0
    R = np.zeros(N)
    R[nS:] = R_T
    d0 = np.zeros(N)
    k0 = int(rng.integers(1, nS + 1)) if not orphan else int(rng.integers(1, nS))
    starts = rng.choice(nS - (1 if orphan else 0), size=min(k0, nS - (1 if orphan else 0)), replace=False)
    d0[starts] = rng.dirichlet(np.ones(len(starts)))
    mask = np.ones((N, A), bool)
    for s in range(nS):
        m = rng.random(A) < rng.uniform(0.2, 0.8)
        if not m.any():
            m[rng.integers(A)] = True
        mask[s] = m
    if orphan:   # the orphan (last live state) has every non-first action unsupported
        mask[nS - 1] = False
        mask[nS - 1, 0] = True
    return MDP(P, R, d0, mask, nS)


def induced(m: MDP, pi):
    return np.einsum("sa,sat->st", pi, m.P)


def is_proper(m: MDP, pi):
    """Graph test: proper iff every live state reaches T in the support graph of P_pi (exact zeros)."""
    adj = induced(m, pi) > 0
    can = np.zeros(m.N, bool)
    can[m.term] = True
    while True:
        new = can | adj[:, can].any(axis=1)
        if (new == can).all():
            break
        can = new
    return bool(can[m.live].all())


def evaluate(m: MDP, pi):
    Ppi = induced(m, pi)
    L = m.live
    F = np.eye(m.nS) - Ppi[np.ix_(L, L)]
    r = Ppi[L] @ m.R
    v = np.linalg.solve(F, r)
    mu = np.linalg.solve(F.T, m.d0[L])
    vfull = np.zeros(m.N)
    vfull[L] = v
    U = m.R + vfull
    return {"J": float(m.d0[L] @ v), "v": vfull, "U": U, "mu": mu,
            "Q": np.einsum("sat,t->sa", m.P, U), "nviol": float(mu @ unsupported_mass(m, pi)[L])}


def unsupported_mass(m: MDP, pi):
    return (pi * m.bad).sum(axis=1)


# completions -------------------------------------------------------------------------------------------------
def random_W(rng, m: MDP, uniform=False):
    W = np.zeros((m.N, m.A, m.A))
    for s in range(m.nS):
        sup = np.flatnonzero(m.mask[s])
        for a in np.flatnonzero(~m.mask[s]):
            W[s, a, sup] = 1.0 / len(sup) if uniform else rng.dirichlet(np.full(len(sup), 0.7))
    return W


def complete_W(m: MDP, W):
    P = m.P.copy()
    for s in range(m.nS):
        for a in np.flatnonzero(~m.mask[s]):
            P[s, a] = W[s, a] @ m.P[s]
    return m.with_rows(P)


def complete_jump(m: MDP, target):
    P = m.P.copy()
    P[m.bad] = 0.0
    P[m.bad, target] = 1.0
    return m.with_rows(P)


def complete_term(m: MDP):
    return complete_jump(m, m.sink)


def mix_rows(mW: MDP, target, lam):
    """Unsupported rows (1 - lam) P_W + lam delta_target; lam scalar or (N, A) array."""
    P = mW.P.copy()
    lam = np.broadcast_to(np.asarray(lam, float), (mW.N, mW.A))
    for s in range(mW.nS):
        for a in np.flatnonzero(~mW.mask[s]):
            row = (1 - lam[s, a]) * mW.P[s, a]
            row[target] += lam[s, a]
            P[s, a] = row
    return mW.with_rows(P)


def random_unsupported_rows(rng, m: MDP):
    """Arbitrary truth: every unsupported row an independent random row over B live successors + absorbing states."""
    P = m.P.copy()
    nT = len(m.term)
    for s in range(m.nS):
        for a in np.flatnonzero(~m.mask[s]):
            B = int(rng.integers(1, min(4, m.nS) + 1))
            succ = rng.choice(m.nS, size=B, replace=False)
            haz = rng.uniform(0.05, 0.6)
            row = np.zeros(m.N)
            np.add.at(row, succ, (1 - haz) * rng.dirichlet(np.ones(B)))
            row[m.nS:] += haz * rng.dirichlet(np.full(nT, 0.5))
            P[s, a] = row
    return m.with_rows(P)


def nhb_rows_for_policy(rng, mW: MDP, UW):
    """Policy-specific NHB truth that is NOT an intermediate-harm mixture: a random row q, and where q.U^W exceeds
    P_W.U^W it is mixed with delta_sink just enough (times a random factor >= 1) to satisfy NHB."""
    base = random_unsupported_rows(rng, mW)
    P = base.P.copy()
    for s in range(mW.nS):
        for a in np.flatnonzero(~mW.mask[s]):
            q = P[s, a].copy()
            gap = q @ UW - mW.P[s, a] @ UW
            if gap > 0:
                beta = min(1.0, gap / (q @ UW - mW.rterm) * rng.uniform(1.0, 1.5))
                q = (1 - beta) * q
                q[mW.sink] += beta
            P[s, a] = q
    return mW.with_rows(P)


def nhb_holds(mW: MDP, mT: MDP, pi, UW):
    worst = -np.inf
    for s in range(mW.nS):
        for a in np.flatnonzero((~mW.mask[s]) & (pi[s] > 0)):
            worst = max(worst, mT.P[s, a] @ UW - mW.P[s, a] @ UW)
    return worst <= TOL, worst


def policy_iteration(m: MDP, allowed):
    """Optimal deterministic policy over `allowed` (N, A) bool by exact policy iteration (every policy proper)."""
    a0 = np.argmax(allowed, axis=1)
    pi = np.zeros((m.N, m.A))
    pi[np.arange(m.N), a0] = 1.0
    for _ in range(500):
        e = evaluate(m, pi)
        Q = np.where(allowed, e["Q"], -np.inf)
        cur = (pi * np.where(allowed, e["Q"], 0)).sum(axis=1)
        better = Q.max(axis=1) > cur + 1e-12
        better[m.nS:] = False
        if not better.any():
            return pi, e
        pi[better] = 0.0
        pi[better, Q[better].argmax(axis=1)] = 1.0
    raise RuntimeError("policy iteration did not converge")


# policies ----------------------------------------------------------------------------------------------------
def random_policies(rng, m: MDP, k, orphan=False):
    pols, kinds = [], []
    for i in range(k):
        kind = ["dirichlet", "deterministic", "admissible", "near_admissible", "sparse_unsup"][i % 5]
        pi = np.full((m.N, m.A), 1.0 / m.A)
        for s in range(m.nS):
            sup = np.flatnonzero(m.mask[s])
            if kind == "dirichlet":
                pi[s] = rng.dirichlet(np.full(m.A, rng.choice([0.3, 1.0])))
            elif kind == "deterministic":
                pi[s] = 0.0
                pi[s, rng.integers(m.A)] = 1.0
            elif kind in ("admissible", "near_admissible", "sparse_unsup"):
                pi[s] = 0.0
                pi[s, sup] = rng.dirichlet(np.ones(len(sup)))
                uns = np.flatnonzero(~m.mask[s])
                if kind == "near_admissible" and len(uns):
                    e = rng.uniform(0, 0.15)
                    pi[s] *= 1 - e
                    pi[s, uns] += e * rng.dirichlet(np.ones(len(uns)))
                if kind == "sparse_unsup" and len(uns) and rng.random() < 0.3:
                    pi[s] = 0.0
                    pi[s, rng.choice(uns)] = 1.0
        if orphan and kind == "admissible":
            # unsupported mass only on the unreachable orphan state
            s = m.nS - 1
            pi[s] = 0.0
            pi[s, 1:] = 1.0 / (m.A - 1)
            kind = "unsup_only_unreachable"
        pols.append(pi)
        kinds.append(kind)
    return pols, kinds


# --------------------------------------------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(20261023)
    out = {k: 0 for k in ("F.FAIL_properP_not_term", "F.FAIL_properW_not_term", "F.FAIL_properW_not_mixture",
                          "F.converse_fails_term_proper_W_improper", "D.FAIL_width_negative", "D.FAIL_width_exceeds",
                          "B.FAIL_IH_not_NHB", "B.FAIL_upper_under_NHB[ih]", "B.FAIL_upper_under_NHB[nhb_generic]",
                          "A.FAIL_lower[arb]", "A.FAIL_lower[ih]", "A.FAIL_lower[ben]", "A.FAIL_lower[nhb_generic]",
                          "H.certified_wrong[ih]", "H.FAIL_certified_disagrees_with_a_rule",
                          "H2.n_certified_pairs_tested", "H2.FAIL_certified_wrong_under_NHB_for_pi2_only",
                          "skip.improper_truth[arb]", "skip.improper_under_W",
                          "K.FAIL_positive_value_but_NHB_for_optimal", "K.FAIL_IH_positive_value_of_unsupported")}

    def bump(key, val=1):
        out[key] = out.get(key, 0) + val

    def track_max(key, val):
        out[key] = max(out.get(key, -np.inf), val)

    def track_min(key, val):
        out[key] = min(out.get(key, np.inf), val)

    for rname, R_T in REWARDS.items():
        for it in range(N_MDP):
            if R_T is None:
                nT = int(rng.integers(2, 5))
                R_T_ = np.sort(rng.uniform(-2, 2, size=nT))
                R_T_[0] = min(R_T_[0], 0.0)               # sink is the worst absorbing state and <= 0: r_term = v_min
            else:
                R_T_ = R_T
            loops = (it % 3 == 0)                          # 1/3 of MDPs have zero-hazard rows (improper policies exist)
            orphan = (it % 5 == 1)
            m = random_mdp(rng, R_T_, loops=loops, orphan=orphan)
            assert abs(m.rterm - m.vmin) < 1e-15 and m.rterm == m.Rmin_T
            W = random_W(rng, m, uniform=(it % 2 == 0))
            mW, mT = complete_W(m, W), complete_term(m)
            goal = m.nS + int(np.argmax(m.R[m.term]))
            mBest = complete_jump(m, goal)
            pols, kinds = random_policies(rng, m, N_POL, orphan=orphan)
            truths_arb = [random_unsupported_rows(rng, m) for _ in range(N_TRUTH)]
            truths_ih = [mix_rows(mW, m.sink, rng.uniform(0, 1, size=(m.N, m.A))) for _ in range(N_TRUTH)]
            truths_ben = [mix_rows(mW, goal, rng.uniform(0.05, 1, size=(m.N, m.A))) for _ in range(N_TRUTH)]
            ev_cache = []
            if not loops:
                piM, eM = policy_iteration(m, m.mask)            # admissible optimum (supported rows only)
                JM = eM["J"]
                for fam, truths in (("ih", truths_ih), ("ben", truths_ben), ("arb", truths_arb)):
                    for mS in truths:
                        pio, eo = policy_iteration(mS, np.ones_like(m.mask))
                        vu = eo["J"] - JM                        # true value of unsupported actions
                        track_min(f"K.min value_of_unsupported[{fam}]", vu)
                        if vu > 1e-9:
                            bump(f"K.n_positive_value_of_unsupported[{fam}]")
                            okW, _ = nhb_holds(mW, mS, pio, evaluate(mW, pio)["U"])
                            if okW:
                                bump("K.FAIL_positive_value_but_NHB_for_optimal")
                        if fam == "ih" and vu > 1e-9:
                            bump("K.FAIL_IH_positive_value_of_unsupported")
            for pi, kind in zip(pols, kinds):
                pW, pT = is_proper(mW, pi), is_proper(mT, pi)
                pP = is_proper(m, pi)
                # F: properness implications
                bump("F.n_policy_checks")
                if pP and not pT:
                    bump("F.FAIL_properP_not_term")
                if pW and not pT:
                    bump("F.FAIL_properW_not_term")
                if pT and not pW:
                    bump("F.converse_fails_term_proper_W_improper")
                if pW:
                    for lam in (0.0, 0.1, 0.5, 0.9, 1.0):
                        if not is_proper(mix_rows(mW, m.sink, lam), pi):
                            bump("F.FAIL_properW_not_mixture")
                    lam_rand = rng.uniform(0, 1, size=(m.N, m.A))
                    if not is_proper(mix_rows(mW, m.sink, lam_rand), pi):
                        bump("F.FAIL_properW_not_mixture")
                if not pW:
                    bump("skip.improper_under_W")
                    ev_cache.append(None)
                    continue
                eW, eT = evaluate(mW, pi), evaluate(mT, pi)
                b = unsupported_mass(m, pi)
                p = float(eT["mu"] @ b[m.live])                  # P(first unsupported decision) = E_term[N_viol]
                # p is rule-free: same supported-row sub-chain under W (cross-check by a direct solve)
                Psup = np.einsum("sa,sat->st", pi * m.mask, m.P)[np.ix_(m.live, m.live)]
                p_direct = float(np.linalg.solve(np.eye(m.nS) - Psup.T, m.d0[m.live]) @ b[m.live])
                track_max("D.max_abs(p_term - p_direct)", abs(p - p_direct))
                track_min("D.min(E_W[Nviol] - p)", eW["nviol"] - p)
                span = m.vmax - m.vmin
                width = eW["J"] - eT["J"]
                bump("D.n_policies")
                track_min("D.min_width", width)
                if width < -TOL:
                    bump("D.FAIL_width_negative")
                if p > 0:
                    track_max("D.max width/(p*span)", width / (p * span))
                if width > p * span + TOL:
                    bump("D.FAIL_width_exceeds")
                # J: value range and U^W >= v_min = r_term
                for e in (eW, eT):
                    track_min("J.min(U - min_T R)", float((e["U"] - m.Rmin_T).min()))
                    track_min("J.min(U - v_min)", float((e["U"] - m.vmin).min()))
                    track_max("J.max(U - max_T R)", float((e["U"] - m.Rmax_T).max()))
                # jump-to-best completion: J = J_term + p (R_max - r_term) exactly (no-assumption sharp upper end)
                eB = evaluate(mBest, pi)
                track_max("D.max_abs(J_best - J_term - p(Rmax-rterm))",
                          abs(eB["J"] - eT["J"] - p * (m.Rmax_T - m.rterm)))
                # E: zero unsupported mass
                if p == 0.0 or kind in ("admissible", "unsup_only_unreachable"):
                    bump(f"E.n_p0[{kind}]")
                    if kind == "unsup_only_unreachable":
                        assert b[m.nS - 1] > 0 and p == 0.0
                    track_max("E.max_abs_width_when_p0", abs(width))
                # G: monotone in uniform lambda, endpoints
                Js = [evaluate(mix_rows(mW, m.sink, lam), pi)["J"] for lam in np.linspace(0, 1, 11)]
                track_max("G.max_abs(J(0)-J_W)", abs(Js[0] - eW["J"]))
                track_max("G.max_abs(J(1)-J_term)", abs(Js[-1] - eT["J"]))
                track_max("G.max_increase_in_lambda", float(np.max(np.diff(Js))))
                lam1 = rng.uniform(0, 1, size=(m.N, m.A))
                lam2 = np.minimum(1, lam1 + rng.uniform(0, 0.5, size=(m.N, m.A)))
                J1 = evaluate(mix_rows(mW, m.sink, lam1), pi)["J"]
                J2 = evaluate(mix_rows(mW, m.sink, lam2), pi)["J"]
                track_max("G.max(J(lam') - J(lam)), lam'>=lam", J2 - J1)
                ev_cache.append((eW, eT, p))
                if not loops:
                    track_max("K.max(J_W(pi) - J*_M)", eW["J"] - JM)
                # A: lower bound under arbitrary truths; B: upper bound under IH; C: beneficial truths
                for fam, truths in (("arb", truths_arb), ("ih", truths_ih), ("ben", truths_ben)):
                    for mS in truths:
                        if not is_proper(mS, pi):
                            bump(f"skip.improper_truth[{fam}]")
                            continue
                        eS = evaluate(mS, pi)
                        bump(f"A.n[{fam}]")
                        if eS["J"] < eT["J"] - TOL:
                            bump(f"A.FAIL_lower[{fam}]")
                        track_min(f"A.min(J*-J_term)[{fam}]", eS["J"] - eT["J"])
                        if eS["J"] - eT["J"] > p * span + TOL:
                            bump(f"D.FAIL_truth_gap_exceeds[{fam}]")
                        if p > 0:
                            track_max(f"D.max (J*-J_term)/(p*span)[{fam}]", (eS["J"] - eT["J"]) / (p * span))
                        ok, worst = nhb_holds(mW, mS, pi, eW["U"])
                        upper_ok = eS["J"] <= eW["J"] + TOL
                        if p > 0:
                            bump(f"B.n_p>0[{fam}]")
                            bump(f"B.upper_holds_p>0[{fam}]", int(upper_ok))
                            bump(f"B.nhb_holds_p>0[{fam}]", int(ok))
                            track_max(f"C.max (J*-J_W)/(p*span)[{fam}]", (eS["J"] - eW["J"]) / (p * span))
                        if ok and not upper_ok:
                            bump(f"B.FAIL_upper_under_NHB[{fam}]")
                        if fam == "ih" and not ok:
                            bump("B.FAIL_IH_not_NHB")
                        # P4 identity, P0 = P_W, P1 = truth
                        delta = sum(eS["mu"][s] * pi[s, a] * ((mW.P[s, a] - mS.P[s, a]) @ eW["U"])
                                    for s in range(m.nS) for a in np.flatnonzero(~m.mask[s]))
                        track_max("J.max_abs_P4_identity_residual", abs((eW["J"] - eS["J"]) - delta))
                # B': generic policy-specific NHB truths (not mixtures)
                for _ in range(2):
                    mS = nhb_rows_for_policy(rng, mW, eW["U"])
                    if not is_proper(mS, pi):
                        bump("skip.improper_truth[nhb_generic]")
                        continue
                    ok, worst = nhb_holds(mW, mS, pi, eW["U"])
                    assert ok, worst
                    eS = evaluate(mS, pi)
                    bump("B.n[nhb_generic]")
                    if eS["J"] > eW["J"] + TOL:
                        bump("B.FAIL_upper_under_NHB[nhb_generic]")
                    if eS["J"] < eT["J"] - TOL:
                        bump("A.FAIL_lower[nhb_generic]")
            # H: certified comparisons (pairs of proper policies), truths: IH (NHB for all), arbitrary, beneficial
            idx = [i for i, e in enumerate(ev_cache) if e is not None]
            for fam, truths in (("ih", truths_ih), ("arb", truths_arb), ("ben", truths_ben)):
                for mS in truths:
                    Jt = {}
                    for i in idx:
                        if is_proper(mS, pols[i]):
                            Jt[i] = evaluate(mS, pols[i])["J"]
                    for i in Jt:
                        for j in Jt:
                            if i == j:
                                continue
                            eWi, eTi, _ = ev_cache[i]
                            eWj, eTj, _ = ev_cache[j]
                            bump(f"H.pairs[{fam}]")
                            truth_tie_or_rev = Jt[i] <= Jt[j] + TOL
                            if eTi["J"] > eWj["J"] + TOL:
                                bump(f"H.certified[{fam}]")
                                if truth_tie_or_rev:
                                    bump(f"H.certified_wrong[{fam}]")
                                # certified claims agree with both single rules
                                if not (eWi["J"] > eWj["J"] and eTi["J"] > eTj["J"]):
                                    bump("H.FAIL_certified_disagrees_with_a_rule")
                            for rule, Ji, Jj in (("mean/W", eWi["J"], eWj["J"]), ("term", eTi["J"], eTj["J"])):
                                if Ji > Jj + TOL:
                                    bump(f"H.claims[{fam}][{rule}]")
                                    if truth_tie_or_rev:
                                        bump(f"H.wrong[{fam}][{rule}]")
            # H2: certified i > j must hold for EVERY truth that satisfies NHB for pi_j only (policy-specific rows)
            cert = [(i, j) for i in idx for j in idx if i != j and ev_cache[i][1]["J"] > ev_cache[j][0]["J"] + TOL]
            for (i, j) in cert[:6]:
                mS = nhb_rows_for_policy(rng, mW, ev_cache[j][0]["U"])
                if not (is_proper(mS, pols[i]) and is_proper(mS, pols[j])):
                    continue
                out["H2.n_certified_pairs_tested"] += 1
                if not evaluate(mS, pols[i])["J"] > evaluate(mS, pols[j])["J"]:
                    out["H2.FAIL_certified_wrong_under_NHB_for_pi2_only"] += 1

    # C': one-state tightness constructions -------------------------------------------------------------------
    for vmin, vmax in ((-1.0, 1.0), (0.0, 1.0)):
        for q in (0.1, 0.5, 1.0):
            # live s0; T = {sink (v_min), good (v_max)}; M(s0) = {a0}; a0 -> sink; true a1 -> good
            N, A = 3, 2
            P = np.zeros((N, A, N))
            P[0, 0, 1] = 1.0
            P[0, 1, 2] = 1.0     # true completion of a1 (beneficial)
            P[1, :, 1] = P[2, :, 2] = 1.0
            R = np.array([0.0, vmin, vmax])
            d0 = np.array([1.0, 0, 0])
            mask = np.ones((N, A), bool)
            mask[0, 1] = False
            mS = MDP(P, R, d0, mask, 1)
            mW = complete_W(mS, random_W(rng, mS, uniform=True))
            mT = complete_term(mS)
            pi = np.array([[1 - q, q], [0.5, 0.5], [0.5, 0.5]])
            JS, JW, JT = evaluate(mS, pi)["J"], evaluate(mW, pi)["J"], evaluate(mT, pi)["J"]
            pexp = q
            assert abs(JT - JW) < 1e-15 and abs(JT - vmin) < 1e-15, (JT, JW)   # J_term = J_W = v_min
            out[f"C.tight(vmin={vmin},vmax={vmax},q={q}): (J*-J_W)/(p*span)"] = (JS - JW) / (pexp * (vmax - vmin))
            # width tightness: a0 -> good instead
            P2 = P.copy()
            P2[0, 0] = 0.0
            P2[0, 0, 2] = 1.0
            m2 = MDP(P2, R, d0, mask, 1)
            mW2, mT2 = complete_W(m2, random_W(rng, m2, uniform=True)), complete_term(m2)
            out[f"D.tight(vmin={vmin},vmax={vmax},q={q}): width/(p*span)"] = \
                (evaluate(mW2, pi)["J"] - evaluate(mT2, pi)["J"]) / (pexp * (vmax - vmin))

    # F': converse of 'proper under P => proper under P_term' fails: unsupported a1 self-loops in truth
    N, A = 3, 2
    P = np.zeros((N, A, N)); P[0, 0, 1] = 1.0; P[0, 1, 0] = 1.0; P[1, :, 1] = P[2, :, 2] = 1.0
    mask = np.ones((N, A), bool); mask[0, 1] = False
    m1 = MDP(P, np.array([0.0, -1.0, 1.0]), np.array([1.0, 0, 0]), mask, 1)
    pi = np.array([[0.0, 1.0], [.5, .5], [.5, .5]])
    out["F'.one-state: proper under (truth, term, mean)"] = str((is_proper(m1, pi), is_proper(complete_term(m1), pi),
                                                             is_proper(complete_W(m1, random_W(rng, m1, True)), pi)))

    # A': hypotheses needed: sink not the worst absorbing state -> lower bound can fail; positive rewards with
    # r_term = min_T R > 0 = v_min -> the lower bound still holds (only r_term <= min_T R is used)
    for label, R_T_ in (("sink_not_worst(R=[0,-1,1])", np.array([0.0, -1.0, 1.0])),
                        ("positive_rewards(R=[0.2,1]) r_term=min_T R>v_min", np.array([0.2, 1.0]))):
        nfail = ntot = 0
        for _ in range(N_MDP):
            m = random_mdp(rng, R_T_)
            mT = complete_term(m)
            pols, _ = random_policies(rng, m, 6)
            for pi in pols:
                if not is_proper(m, pi):
                    continue
                for _ in range(2):
                    mS = random_unsupported_rows(rng, m)
                    if not is_proper(mS, pi):
                        continue
                    ntot += 1
                    nfail += int(evaluate(mS, pi)["J"] < evaluate(mT, pi)["J"] - TOL)
        out[f"A'.lower_failures[{label}]"] = f"{nfail}/{ntot}"

    # I: Garnet harm family ------------------------------------------------------------------------------------
    garnet_checks(rng, out)

    # report ---------------------------------------------------------------------------------------------------
    fails = {k: v for k, v in out.items() if "FAIL" in k and v}
    for k in sorted(out):
        v = out[k]
        print(f"{k:70s} {v:.3e}" if isinstance(v, float) else f"{k:70s} {v}")
    print("\nFAILURES:", fails if fails else "none")
    write_json(out, fails)
    return 0 if not fails else 1


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def write_json(out, fails):
    """Write every check and the derived counts that the paper quotes to ../theory_verification.json."""
    o = {k: _plain(v) for k, v in out.items()}
    fam = ("arb", "ih", "ben")
    n_p0 = {k.split("[", 1)[1].rstrip("]"): v for k, v in o.items() if k.startswith("E.n_p0[")}
    garnet = [int(x) for x in o["I.garnet NHB pairs ok (h>=0) / strictly violated (h<0) / all unsupported pairs"]
              .split("/")]
    sink_fail, sink_n = (int(x) for x in o["A'.lower_failures[sink_not_worst(R=[0,-1,1])]"].split("/"))
    pos_fail, pos_n = (int(x) for x in o["A'.lower_failures[positive_rewards(R=[0.2,1]) r_term=min_T R>v_min]"]
                       .split("/"))
    cert = o["H.certified[ih]"]
    summary = {
        "seed": 20261023,
        "quick": QUICK,
        "n_reward_structures": len(REWARDS),
        "n_mdps_per_reward_structure": N_MDP,
        "n_mdps": len(REWARDS) * N_MDP,
        "n_mdps_with_zero_hazard_rows": len(REWARDS) * len(range(0, N_MDP, 3)),
        "n_mdps_without_zero_hazard_rows": len(REWARDS) * (N_MDP - len(range(0, N_MDP, 3))),
        "policies_per_mdp": N_POL,
        "truths_per_family_per_mdp": N_TRUTH,
        "tolerance": TOL,
        "lower_bound_evaluations_per_family": o["A.n[ih]"],
        "lower_bound_failures": {f: o[f"A.FAIL_lower[{f}]"] for f in fam + ("nhb_generic",)},
        "nhb_only_evaluations": o["B.n[nhb_generic]"],
        "upper_bound_cases_p_positive_per_family": o["B.n_p>0[ih]"],
        "upper_bound_holds_p_positive": {f: o[f"B.upper_holds_p>0[{f}]"] for f in fam},
        "upper_bound_failures_under_nhb": {"ih": o["B.FAIL_upper_under_NHB[ih]"],
                                           "nhb_generic": o["B.FAIL_upper_under_NHB[nhb_generic]"]},
        "nhb_termwise_holds_p_positive_arbitrary": o["B.nhb_holds_p>0[arb]"],
        "max_width_over_p_span_random": o["D.max width/(p*span)"],
        "max_upper_excess_over_p_span_random": max(o["C.max (J*-J_W)/(p*span)[arb]"],
                                                   o["C.max (J*-J_W)/(p*span)[ben]"]),
        "tight_constructions_ratio_min": min(v for k, v in o.items()
                                             if k.startswith(("C.tight", "D.tight"))),
        "jump_to_best_formula_max_residual": o["D.max_abs(J_best - J_term - p(Rmax-rterm))"],
        "p4_identity_max_residual": o["J.max_abs_P4_identity_residual"],
        "n_policies_p_zero": int(sum(n_p0.values())),
        "n_policies_p_zero_unreachable_only": n_p0.get("unsup_only_unreachable", 0),
        "max_width_p_zero": o["E.max_abs_width_when_p0"],
        "properness_policy_checks": o["F.n_policy_checks"],
        "properness_implication_failures": o["F.FAIL_properP_not_term"] + o["F.FAIL_properW_not_term"]
        + o["F.FAIL_properW_not_mixture"],
        "properness_converse_failures": o["F.converse_fails_term_proper_W_improper"],
        "max_increase_in_lambda": o["G.max_increase_in_lambda"],
        "certified_pairs": cert,
        "ordered_pairs": o["H.pairs[ih]"],
        "certified_share": cert / o["H.pairs[ih]"],
        "certified_wrong": {f: o[f"H.certified_wrong[{f}]"] for f in fam},
        "certified_wrong_share": {f: o[f"H.certified_wrong[{f}]"] / cert for f in fam},
        "nhb_pi2_only_certified_pairs_tested": o["H2.n_certified_pairs_tested"],
        "nhb_pi2_only_certified_wrong": o["H2.FAIL_certified_wrong_under_NHB_for_pi2_only"],
        "point_ranking_wrong_share_ih": {"W": o["H.wrong[ih][mean/W]"] / o["H.claims[ih][mean/W]"],
                                         "term": o["H.wrong[ih][term]"] / o["H.claims[ih][term]"]},
        "garnet_unsupported_pairs": garnet[2],
        "garnet_nhb_ok_h_nonneg": garnet[0],
        "garnet_nhb_strictly_violated_h_neg": garnet[1],
        "garnet_ih_form_max_residual": o["I.garnet max residual of intermediate-harm form (h>=0, K in M)"],
        "garnet_upper_failures_h_nonneg": o["I.FAIL garnet upper bound violated at h>=0 (K in M)"],
        "value_of_unsupported_positive_truths": sum(v for k, v in o.items()
                                                    if k.startswith("K.n_positive_value_of_unsupported")),
        "value_of_unsupported_positive_under_ih": o["K.FAIL_IH_positive_value_of_unsupported"],
        "positive_value_but_nhb_for_optimal": o["K.FAIL_positive_value_but_NHB_for_optimal"],
        "max_JW_minus_admissible_optimum": o["K.max(J_W(pi) - J*_M)"],
        "sink_not_worst_lower_failures": sink_fail,
        "sink_not_worst_cases": sink_n,
        "rterm_min_T_R_positive_lower_failures": pos_fail,
        "rterm_min_T_R_positive_cases": pos_n,
        "failures": sorted(fails),
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "theory_verification.json")
    with open(path, "w") as fh:
        json.dump({"script": "scripts/verify_bracket.py", "summary": summary, "checks": o}, fh, indent=1,
                  sort_keys=True)
    print("wrote", os.path.normpath(path))


def garnet_checks(rng, out):
    """Garnet-type truth: preferred set K(s); unsupported-in-truth rows of non-preferred actions are
    (1-|h|) mean_K + |h| delta_{death or goal}. On states with K(s) subset of M(s), and P_W the mean of the TRUE
    rows over M(s): h >= 0 gives P* = (1-lam) P_W + lam delta_death with lam = h k / (m - h (m - k)); h < 0 gives
    (P* - P_W).U = |h| (k/m) (R(goal) - P_K.U) >= 0 for every U <= R(goal)."""
    max_res, n_ok, n_strict, n_pairs, n_upper_ok_neg, n_upper_neg, n_viol_pos = 0.0, 0, 0, 0, 0, 0, 0
    for it in range(40 if QUICK else 150):
        nS, A = int(rng.integers(6, 25)), int(rng.integers(4, 8))
        base = random_mdp(rng, np.array([0.0, 1.0]), nS=nS, A=A, branching=3)   # death (sink) 0, goal 1
        death, goal = nS, nS + 1
        for h in (-1.0, -0.5, 0.0, 0.5, 1.0):
            P = base.P.copy()
            K = np.zeros((base.N, A), bool)
            M = np.ones((base.N, A), bool)
            for s in range(nS):
                k = int(rng.integers(1, 4))
                K[s, rng.choice(A, size=k, replace=False)] = True
                PK = base.P[s, K[s]].mean(axis=0)
                for a in np.flatnonzero(~K[s]):
                    row = (1 - abs(h)) * PK
                    row[death if h > 0 else goal] += abs(h)
                    P[s, a] = row
                # mask: K(s) subset of M(s) (the case of the claim), plus random extra supported actions
                extra = (~K[s]) & (rng.random(A) < 0.4)
                M[s] = K[s] | extra
            mS = MDP(P, base.R, base.d0, M, nS)                      # truth with the benchmark mask
            mW = complete_W(mS, random_W(rng, mS, uniform=True))      # true rows + mean
            for s in range(nS):
                mm, k = M[s].sum(), K[s].sum()
                for a in np.flatnonzero(~M[s]):
                    if h >= 0:
                        lam = h * k / (mm - h * (mm - k))
                        pred = (1 - lam) * mW.P[s, a]
                        pred[death] += lam
                        max_res = max(max_res, float(np.abs(pred - mS.P[s, a]).max()))
            pi = np.full((base.N, A), 1.0 / A)               # uniform policy, reaches unsupported pairs
            eW = evaluate(mW, pi)
            eS = evaluate(mS, pi)
            for s in range(nS):
                for a in np.flatnonzero(~M[s]):
                    n_pairs += 1
                    gap = (mS.P[s, a] - mW.P[s, a]) @ eW["U"]
                    if h >= 0:
                        n_ok += int(gap <= TOL)
                    elif h < 0:
                        n_strict += int(gap > 0)
            if h < 0:
                n_upper_neg += 1
                n_upper_ok_neg += int(eS["J"] <= eW["J"] + TOL)
            else:
                n_viol_pos += int(eS["J"] > eW["J"] + TOL)
    out["I.garnet max residual of intermediate-harm form (h>=0, K in M)"] = max_res
    out["I.garnet NHB pairs ok (h>=0) / strictly violated (h<0) / all unsupported pairs"] = \
        f"{n_ok} / {n_strict} / {n_pairs}"
    out["I.garnet upper bound holds for uniform policy at h<0"] = f"{n_upper_ok_neg}/{n_upper_neg}"
    out["I.FAIL garnet upper bound violated at h>=0 (K in M)"] = n_viol_pos


if __name__ == "__main__":
    sys.exit(main())
