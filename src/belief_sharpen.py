"""Belief-sharpening hybrid (Strategy-v6 component B).

research_strategy.md §2.6 shows the v3/v4/v5 decision-layer is capped because most held-out
losses are *belief* failures on OOV words, which a frozen-belief decision layer cannot fix.
This module attacks the belief itself, cheaply and without training, by fusing v2's *neural*
letter marginal with the *exact* consistent-set marginal from the train dictionary — the
reveal-all-occurrences + absent-letter rules made explicit combinatorially:

    s_dict(c) = (# consistent words with c at some blank) / (# consistent words)

where the consistent set is the train-corpus words matching the board (same filter PIMC uses,
``lookahead.consistent_words``). We blend, gated by the consistent-set size N:

    beta(N) -> 1  when N is small   (endgame: the dictionary pins the answer; the neural
                                     marginal smears it)
    beta(N) -> 0  when N is huge or 0 (openings / true OOV: defer to v2, which generalizes)

    s3 = beta(N) * s_dict + (1 - beta(N)) * s2 ;  guess argmax over unguessed letters.

This is a third, maximally-diverse belief source — and belief diversity is exactly what gave
v2 its original +3.7-pt ensemble jump (strategy2.md §3).

Run as a script to sweep the gate and report paired held-out win-rate vs v2.
"""
import argparse
import math
import random

import numpy as np
import torch

from vocab import target_to_letter
from data import load_words, split_words
from evaluate import play_game
from fast_belief import DictBelief
from pimc_fast import VectorizedV2


def soft_gate(n: int, n_lo: float, n_hi: float) -> float:
    """beta(N): 1 for N<=n_lo, ramps log-linearly to 0 at N>=n_hi, 0 for N==0."""
    if n <= 0:
        return 0.0
    if n <= n_lo:
        return 1.0
    if n >= n_hi:
        return 0.0
    return (math.log(n_hi) - math.log(n)) / (math.log(n_hi) - math.log(n_lo))


def make_guess_fn(v2: VectorizedV2, db: DictBelief, n_lo: float, n_hi: float,
                  beta_cap: float = 1.0):
    """Build a guess_fn(board, guessed)->letter for the sharpened belief."""
    @torch.no_grad()
    def guess_fn(board, guessed):
        guessed = set(guessed)
        l_ref, _, allowed = v2.features([(list(board), guessed)])
        s2 = torch.exp(l_ref[0]).cpu().numpy()         # v2 neural marginal
        allowed0 = allowed[0].cpu().numpy()
        s_dict, n = db.query(board, guessed)
        beta = beta_cap * soft_gate(n, n_lo, n_hi)
        s3 = beta * s_dict + (1.0 - beta) * s2
        s3 = np.where(allowed0, s3, -np.inf)
        return target_to_letter(int(s3.argmax()))
    return guess_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", default="50:2000,200:5000,500:20000,100:100000",
                    help="comma list of n_lo:n_hi gate pairs to sweep")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, ev = split_words(load_words())
    v2 = VectorizedV2(device)
    db = DictBelief(train)

    def v2_fn(board, guessed):
        return target_to_letter(v2.guess_idx([(list(board), set(guessed))])[0])

    # Paired: same sampled words for v2 and every variant.
    rng = random.Random(args.seed)
    sample = rng.sample(ev, min(args.num, len(ev)))

    def score(fn):
        wins = tw = 0
        for w in sample:
            won, wrong = play_game(w, fn)
            wins += int(won); tw += wrong
        return wins / len(sample), tw / len(sample)

    v2_wr, v2_aw = score(v2_fn)
    print(f"[v2 baseline]            wr={v2_wr:.4f} aw={v2_aw:.2f}  (n={len(sample)})", flush=True)

    for pair in args.grid.split(","):
        n_lo, n_hi = (float(x) for x in pair.split(":"))
        fn = make_guess_fn(v2, db, n_lo, n_hi)
        wr, aw = score(fn)
        print(f"[sharpen lo={n_lo:.0f} hi={n_hi:.0f}] wr={wr:.4f} aw={aw:.2f} "
              f"delta={wr - v2_wr:+.4f}", flush=True)


if __name__ == "__main__":
    main()
