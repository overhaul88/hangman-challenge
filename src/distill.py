"""Strategy v6 — Expert Iteration: distill the PIMC search operator into a fast policy.

research_strategy.md §5.1. The v3/v4/v5 failure was the *estimator* (sparse-reward policy
gradient on a frozen near-optimal greedy with a v2-anchored tiny residual), not the problem.
Here we keep the provable one-step-improvement operator (PIMC) as a *teacher* and learn its
decision by **supervised distillation** — no sparse reward, no stochastic->argmax transfer gap,
no KL-to-v2 anchor:

  improvement step (exact):  Q_hat(o,c) = P(win | guess c, then v2)   [pimc_fast.FastPIMC]
  projection step (learned): minimise KL( student(.|o) || softmax(Q_hat(o,.)/temp) )

The student is a feed-forward residual over v2's reference logits (zero-init -> starts at v2,
so the safe-selection gate can always fall back to exactly v2):

    logits(o) = l_ref(o)/tau + g_theta(h(o)) ,   g_theta zero-init,   masked to allowed.

Two sub-commands:
  gen   : play v2 on TRAIN words (with optional epsilon exploration for coverage), collect
          states, label them with batched PIMC, save a label bank.
  train : fit the student to the label bank, select on a held-out paired eval (seeded with v2),
          and run the authoritative paired final gate (ship student only if it beats v2).
"""
import argparse
import os
import random
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import target_to_letter
from data import load_words, split_words
from evaluate import play_game
from rl_policy import NEG_MASK
from pimc_fast import FastPIMC, VectorizedV2
from fast_belief import DictBelief

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


# ---------------------------------------------------------------- student
class DistillPolicy(nn.Module):
    def __init__(self, feat_dim: int, tau: float = 1.0, hidden: int = 256):
        super().__init__()
        self.tau = tau
        self.g = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 26),
        )
        nn.init.zeros_(self.g[-1].weight)
        nn.init.zeros_(self.g[-1].bias)

    def logits(self, l_ref, h, allowed):
        lo = l_ref / self.tau + self.g(h)
        return lo.masked_fill(~allowed, NEG_MASK)


def make_student_guess_fn(policy: DistillPolicy, v2: VectorizedV2):
    policy.eval()

    @torch.no_grad()
    def guess_fn(board, guessed):
        l_ref, h, allowed = v2.features([(list(board), set(guessed))])
        lo = policy.logits(l_ref, h, allowed)
        return target_to_letter(int(lo[0].argmax().item()))
    return guess_fn


# ---------------------------------------------------------------- label generation
def collect_states(v2: VectorizedV2, words: List[str], n_games: int, epsilon: float,
                   rng: random.Random):
    """Play v2 (eps-greedy for coverage) on sampled train words; return visited states."""
    sample = rng.sample(words, min(n_games, len(words)))
    states = []
    for w in sample:
        board = ["_"] * len(w); guessed = set(); wrong = 0
        steps = 0
        while wrong < 6 and "_" in board and steps < 30:
            states.append((list(board), set(guessed)))
            l_ref, _, allowed = v2.features([(board, guessed)])
            if rng.random() < epsilon:
                allow_idx = allowed[0].nonzero(as_tuple=True)[0].tolist()
                gi = rng.choice(allow_idx)
            else:
                gi = int(l_ref[0].masked_fill(~allowed[0], -float("inf")).argmax().item())
            g = target_to_letter(gi)
            guessed.add(g)
            if g in w:
                for i, c in enumerate(w):
                    if c == g:
                        board[i] = g
            else:
                wrong += 1
            steps += 1
    return states


def gen(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _ = split_words(load_words())
    v2 = VectorizedV2(device)
    db = DictBelief(train)
    pimc = FastPIMC(train, v2, n_samples=args.n_samples,
                    max_candidates=args.max_candidates, seed=args.seed, belief=db)
    rng = random.Random(args.seed)

    print(f"collecting states from {args.n_games} v2 games (eps={args.epsilon})...", flush=True)
    states = collect_states(v2, train, args.n_games, args.epsilon, rng)
    rng.shuffle(states)
    states = states[:args.max_states]
    print(f"  {len(states)} states; labeling with batched PIMC "
          f"(n_samples={args.n_samples}, K={args.max_candidates})...", flush=True)

    L, H, A, Q = [], [], [], []
    t0 = time.time()
    rb = args.root_batch
    for s in range(0, len(states), rb):
        recs = pimc.label_states_batched(states[s:s + rb], chunk=args.chunk)
        for r in recs:
            if not r["has_belief"]:
                continue
            L.append(r["l_ref"]); H.append(r["h"]); A.append(r["allowed"]); Q.append(r["q_target"])
        el = time.time() - t0
        done = min(s + rb, len(states))
        print(f"  [{done}/{len(states)}] kept={len(L)} {el/60:.1f} min "
              f"({el/max(1,done):.2f}s/state)", flush=True)

    bank = dict(l_ref=torch.stack(L), h=torch.stack(H),
                allowed=torch.stack(A), q_target=torch.stack(Q),
                feat_dim=v2.feat_dim)
    out = os.path.join(MODELS_DIR, args.bank)
    torch.save(bank, out)
    print(f"saved {len(L)} labeled states -> {out}  ({(time.time()-t0)/60:.1f} min)")


# ---------------------------------------------------------------- training
def paired_eval(guess_fn, v2_fn, sample):
    a_w = b_w = a_only = b_only = 0
    a_wrong = b_wrong = 0
    for w in sample:
        aw, awr = play_game(w, guess_fn)
        bw, bwr = play_game(w, v2_fn)
        a_w += int(aw); b_w += int(bw); a_wrong += awr; b_wrong += bwr
        if aw and not bw: a_only += 1
        elif bw and not aw: b_only += 1
    n = len(sample)
    return dict(wr=a_w / n, v2_wr=b_w / n, delta=(a_w - b_w) / n,
                avg_wrong=a_wrong / n, v2_avg_wrong=b_wrong / n,
                student_only=a_only, v2_only=b_only)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_words, ev = split_words(load_words())
    v2 = VectorizedV2(device)

    bank = torch.load(os.path.join(MODELS_DIR, args.bank), map_location=device)
    l_ref = bank["l_ref"].to(device); h = bank["h"].to(device)
    allowed = bank["allowed"].to(device); q_target = bank["q_target"].to(device)
    feat_dim = bank["feat_dim"]
    M = l_ref.shape[0]
    print(f"label bank: {M} states, feat_dim={feat_dim}", flush=True)

    # Teacher distribution: softmax(Q/temp) over allowed letters.
    teacher_logits = (q_target / args.temp).masked_fill(~allowed, -float("inf"))
    teacher = F.softmax(teacher_logits, dim=-1)

    policy = DistillPolicy(feat_dim, tau=args.tau, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.wd)

    def v2_fn(board, guessed):
        return target_to_letter(v2.guess_idx([(list(board), set(guessed))])[0])

    rng = random.Random(0)
    eval_sample = rng.sample(ev, min(args.eval_num, len(ev)))

    # safe-selection seeded with v2 (zero-residual student == v2)
    best_wr = paired_eval(make_student_guess_fn(policy, v2), v2_fn, eval_sample)["v2_wr"]
    print(f"v2 held-out (paired, n={len(eval_sample)}) = {best_wr:.4f}", flush=True)
    best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    idx = torch.arange(M, device=device)
    for ep in range(1, args.epochs + 1):
        policy.train()
        perm = idx[torch.randperm(M, device=device)]
        tot = 0.0
        for s in range(0, M, args.batch):
            b = perm[s:s + args.batch]
            lo = policy.logits(l_ref[b], h[b], allowed[b])
            logp = F.log_softmax(lo.masked_fill(~allowed[b], -1e9), dim=-1)
            loss = F.kl_div(logp, teacher[b], reduction="batchmean")
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step(); tot += loss.item()
        if ep % args.eval_every == 0 or ep == args.epochs:
            r = paired_eval(make_student_guess_fn(policy, v2), v2_fn, eval_sample)
            flag = ""
            if r["wr"] > best_wr:
                best_wr = r["wr"]
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
                flag = " *best"
            print(f"ep {ep:3d} loss={tot/max(1,M//args.batch):.4f} | held-out wr={r['wr']:.4f} "
                  f"(v2={r['v2_wr']:.4f} d={r['delta']:+.4f} s_only={r['student_only']} "
                  f"v2_only={r['v2_only']}){flag}", flush=True)

    policy.load_state_dict(best_state)
    torch.save(dict(model=policy.state_dict(), feat_dim=feat_dim, tau=args.tau,
                    hidden=args.hidden), os.path.join(MODELS_DIR, args.out))

    # authoritative paired final gate on a large held-out sample
    rng2 = random.Random(args.final_seed)
    final_sample = rng2.sample(ev, min(args.final_num, len(ev)))
    print(f"\n=== Final paired gate (held-out, n={len(final_sample)}) ===", flush=True)
    r = paired_eval(make_student_guess_fn(policy, v2), v2_fn, final_sample)
    ship = r["wr"] > r["v2_wr"]
    print(f"v6 distilled : wr={r['wr']:.4f} avg-wrong={r['avg_wrong']:.2f}")
    print(f"v2 baseline  : wr={r['v2_wr']:.4f} avg-wrong={r['v2_avg_wrong']:.2f}")
    print(f"delta = {r['delta']:+.4f}  | student_only={r['student_only']} v2_only={r['v2_only']}")
    print(f"SHIP v6: {ship}  (else fall back to v2-equivalent zero-residual policy)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--n_games", type=int, default=600)
    g.add_argument("--max_states", type=int, default=6000)
    g.add_argument("--epsilon", type=float, default=0.15)
    g.add_argument("--n_samples", type=int, default=48)
    g.add_argument("--max_candidates", type=int, default=7)
    g.add_argument("--root_batch", type=int, default=256)
    g.add_argument("--chunk", type=int, default=4096)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--bank", default="distill_labels.pt")
    g.set_defaults(func=gen)

    t = sub.add_parser("train")
    t.add_argument("--bank", default="distill_labels.pt")
    t.add_argument("--out", default="hangman_distill.pt")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--wd", type=float, default=1e-5)
    t.add_argument("--temp", type=float, default=0.1)
    t.add_argument("--tau", type=float, default=1.0)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--eval_every", type=int, default=5)
    t.add_argument("--eval_num", type=int, default=2000)
    t.add_argument("--final_num", type=int, default=5000)
    t.add_argument("--final_seed", type=int, default=123)
    t.set_defaults(func=train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
