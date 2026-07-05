"""Strategy v7 — pseudo-OOV (morphological suffix) split of the train corpus.

strategy7.md §7. The real held-out set is *fully* out-of-vocabulary (0 overlap with
train), so to reward the encoder for generalising rather than memorising we carve a
**pseudo-OOV validation set out of the train split itself**: we cluster train words by
their morphological suffix (the last 3-4 characters) and hold out whole, distinctive
suffix clusters. During RL training 80% of episodes draw from the remaining in-vocab
words and 20% from this pseudo-OOV set; checkpoints are selected on pseudo-OOV win-rate.

The split is *deterministic* and reconstructable from a small suffix list, so we only
persist the suffix list (+ counts) to JSON; `load_split` rebuilds the exact word lists
from the seed-42 train split.

Key identity: a pseudo-OOV word is any train word whose lowercase form ends with one of
the configured suffixes; everything else is in-vocab.
"""
import argparse
import json
import os
from collections import Counter
from typing import List, Sequence, Tuple

from data import load_words, split_words

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# Default held-out suffixes (the two large, morphologically distinctive clusters called
# out in strategy7.md §7). Override with --suffixes / the JSON's "suffixes" field.
DEFAULT_SUFFIXES = ("tion", "ness")


def partition(words: Sequence[str], suffixes: Sequence[str]
              ) -> Tuple[List[str], List[str]]:
    """Split `words` into (in_vocab, pseudo_oov) by suffix membership.

    A word is pseudo-OOV iff it ends with any suffix in `suffixes`.
    """
    suf = tuple(s.lower() for s in suffixes)
    in_vocab, pseudo_oov = [], []
    for w in words:
        (pseudo_oov if w.endswith(suf) else in_vocab).append(w)
    return in_vocab, pseudo_oov


def suggest_suffixes(words: Sequence[str], k: int = 2, min_len: int = 3,
                     max_len: int = 4, top: int = 25) -> List[Tuple[str, int]]:
    """Rank candidate held-out suffixes by cluster size (last `min_len..max_len` chars).

    Returns the `top` largest suffix clusters as (suffix, count) so a human can pick the
    `k` most morphologically distinctive ones. We count the longest suffix window per
    word to avoid double-counting nested windows.
    """
    counts: Counter = Counter()
    for w in words:
        if len(w) > max_len:
            counts[w[-max_len:]] += 1
        elif len(w) >= min_len:
            counts[w[-min_len:]] += 1
    return counts.most_common(top)


def load_split(suffixes: Sequence[str] = DEFAULT_SUFFIXES, seed: int = 42, frac: float = 0.8
               ) -> Tuple[List[str], List[str], List[str]]:
    """Rebuild (in_vocab, pseudo_oov, held_out) from the corpus.

    `held_out` is the real seed-42 eval split (fully OOV; never trained on). `in_vocab`
    and `pseudo_oov` partition the train split by `suffixes`.
    """
    train, held_out = split_words(load_words(), seed=seed, frac=frac)
    in_vocab, pseudo_oov = partition(train, suffixes)
    return in_vocab, pseudo_oov, held_out


def load_split_from_json(path: str) -> Tuple[List[str], List[str], List[str]]:
    with open(path) as f:
        cfg = json.load(f)
    return load_split(cfg["suffixes"], cfg.get("seed", 42), cfg.get("frac", 0.8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffixes", default=",".join(DEFAULT_SUFFIXES),
                    help="comma-separated held-out suffixes (pseudo-OOV)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--suggest", action="store_true",
                    help="print the largest suffix clusters and exit")
    ap.add_argument("--out", default=os.path.join(MODELS_DIR, "oov_split.json"))
    args = ap.parse_args()

    train, held_out = split_words(load_words(), seed=args.seed, frac=args.frac)
    if args.suggest:
        print("Largest train suffix clusters (suffix: count):")
        for s, c in suggest_suffixes(train):
            print(f"  -{s}: {c}")
        return

    suffixes = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    in_vocab, pseudo_oov = partition(train, suffixes)
    cfg = {
        "suffixes": suffixes,
        "seed": args.seed,
        "frac": args.frac,
        "n_train": len(train),
        "n_in_vocab": len(in_vocab),
        "n_pseudo_oov": len(pseudo_oov),
        "n_held_out": len(held_out),
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"suffixes={suffixes}")
    print(f"train={len(train)}  in_vocab={len(in_vocab)}  "
          f"pseudo_oov={len(pseudo_oov)}  held_out(OOV)={len(held_out)}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
