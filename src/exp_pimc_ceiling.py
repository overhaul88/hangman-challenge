"""Experiment 1 (decisive): measure the one-step decision-layer ceiling on held-out.

Plays PIMC (provable one-step policy improvement over v2) vs v2 on the SAME held-out words
(paired) and reports win-rate + a McNemar-style discordance breakdown. This is the number
v3/v4/v5 never measured: it says how much *any* decision layer on top of the frozen v2 belief
can add on out-of-vocabulary held-out words.

    PIMC ~= v2   -> decision layer is tapped out; pursue belief upgrades instead.
    PIMC  > v2   -> a real, generalizing gap exists that sparse-reward RL failed to extract;
                    distill PIMC into a fast policy (distill.py).
"""
import argparse
import json
import random
import time

import torch

from data import load_words, split_words
from evaluate import play_game
from vocab import target_to_letter
from pimc_fast import FastPIMC, VectorizedV2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--max_candidates", type=int, default=8)
    ap.add_argument("--switch_margin", type=float, default=0.0)
    ap.add_argument("--split", default="held", choices=["held", "train"],
                    help="evaluate on held-out (OOV) or train (in-distribution) words")
    ap.add_argument("--out", default="../models/pimc_ceiling.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, ev = split_words(load_words())
    from fast_belief import DictBelief
    db = DictBelief(train)
    v2 = VectorizedV2(device)
    pimc = FastPIMC(train, v2, n_samples=args.n_samples,
                    max_candidates=args.max_candidates,
                    switch_margin=args.switch_margin, seed=args.seed, belief=db)
    eval_words = ev if args.split == "held" else train

    def v2_fn(board, guessed):
        return target_to_letter(v2.guess_idx([(list(board), set(guessed))])[0])

    rng = random.Random(args.seed)
    sample = rng.sample(eval_words, min(args.num, len(eval_words)))

    v2_wins = pimc_wins = 0
    v2_wrong = pimc_wrong = 0
    both = v2_only = pimc_only = neither = 0
    t0 = time.time()
    for k, w in enumerate(sample, 1):
        v2_won, v2_wr = play_game(w, v2_fn)
        p_won, p_wr = play_game(w, pimc.best_action)
        v2_wins += int(v2_won); pimc_wins += int(p_won)
        v2_wrong += v2_wr; pimc_wrong += p_wr
        if v2_won and p_won: both += 1
        elif v2_won and not p_won: v2_only += 1
        elif p_won and not v2_won: pimc_only += 1
        else: neither += 1
        if k % 10 == 0:
            el = time.time() - t0
            print(f"  [{k}/{len(sample)}] v2={v2_wins/k:.3f} pimc={pimc_wins/k:.3f} "
                  f"(pimc_only={pimc_only} v2_only={v2_only}) {el/k:.1f}s/game", flush=True)

    n = len(sample)
    res = dict(n=n, seed=args.seed, n_samples=args.n_samples,
               max_candidates=args.max_candidates, switch_margin=args.switch_margin,
               v2_wr=v2_wins / n, pimc_wr=pimc_wins / n,
               v2_avg_wrong=v2_wrong / n, pimc_avg_wrong=pimc_wrong / n,
               delta=(pimc_wins - v2_wins) / n,
               both=both, v2_only=v2_only, pimc_only=pimc_only, neither=neither,
               seconds=time.time() - t0)
    print("\n=== PIMC one-step ceiling (paired, held-out) ===")
    print(f"v2   : wr={res['v2_wr']:.4f}  avg-wrong={res['v2_avg_wrong']:.2f}")
    print(f"PIMC : wr={res['pimc_wr']:.4f}  avg-wrong={res['pimc_avg_wrong']:.2f}")
    print(f"delta= {res['delta']:+.4f}  | discordant: pimc_only={pimc_only} v2_only={v2_only}")
    print(f"(n={n}, {res['seconds']/60:.1f} min)")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
