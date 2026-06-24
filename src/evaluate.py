"""Offline evaluation: simulate full Hangman games and report win-rate.

Provides the shared inference primitives (used by both training-time checkpointing
and the online client):
  - predict_letter:   new encoder + absent/present conditioning + mean|noisy_or aggregation
  - play_game / evaluate_winrate: generic game simulator driven by any guess_fn

Run as a script to compare the new encoder (mean vs noisy-or) against the old MoE
baseline on the shared seed-42 eval split.
"""
import argparse
import os
import random
from typing import Callable, List, Set, Tuple

import torch
import torch.nn as nn

from vocab import (CHARS, CHAR_TO_IDX, MASK_IDX, NUM_LETTERS, MAX_WRONG,
                   letter_to_target, target_to_letter)
from data import load_words, split_words
from model import HangmanEncoder

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


# ----------------------------------------------------------------------------
# New-model inference
# ----------------------------------------------------------------------------
def _aggregate(probs: torch.Tensor, hidden_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """probs: (L,26) per-position distributions; hidden_mask: (L,) bool.
    Returns a (26,) score per letter aggregated over hidden positions."""
    hp = probs[hidden_mask]  # (H, 26)
    if hp.numel() == 0:
        return torch.zeros(NUM_LETTERS, device=probs.device)
    if mode == "mean":
        return hp.mean(dim=0)
    elif mode == "noisy_or":
        # P(letter appears in >=1 hidden position) = 1 - prod(1 - p_i)
        return 1.0 - torch.prod(1.0 - hp, dim=0)
    raise ValueError(mode)


@torch.no_grad()
def predict_letter(model: nn.Module, board: List[str], guessed: Set[str],
                   device, mode: str = "noisy_or") -> str:
    """Choose the next letter given the current board and the set of guessed letters."""
    revealed = set(c for c in board if c != "_")
    absent = guessed - revealed  # wrong guesses

    input_ids = [MASK_IDX if c == "_" else CHAR_TO_IDX[c] for c in board]
    input_t = torch.tensor([input_ids], dtype=torch.long, device=device)

    absent_mh = torch.zeros(1, NUM_LETTERS, device=device)
    for ch in absent:
        absent_mh[0, letter_to_target(ch)] = 1.0
    present_mh = torch.zeros(1, NUM_LETTERS, device=device)
    for ch in revealed:
        present_mh[0, letter_to_target(ch)] = 1.0

    logits = model(input_t, absent_mh, present_mh, pad_mask=None)  # (1, L, 26)
    probs = torch.softmax(logits[0], dim=-1)                        # (L, 26)
    hidden_mask = (input_t[0] == MASK_IDX)
    scores = _aggregate(probs, hidden_mask, mode)                   # (26,)

    for ch in guessed:
        scores[letter_to_target(ch)] = -float("inf")
    return target_to_letter(int(torch.argmax(scores).item()))


# ----------------------------------------------------------------------------
# Generic game simulator
# ----------------------------------------------------------------------------
def play_game(word: str, guess_fn: Callable[[List[str], Set[str]], str]
              ) -> Tuple[bool, int]:
    """Play one game with a guess_fn(board, guessed)->letter. Returns (won, wrong_count)."""
    guessed: Set[str] = set()
    wrong = 0
    board = ["_"] * len(word)
    while wrong < MAX_WRONG and "_" in board:
        g = guess_fn(board, guessed)
        guessed.add(g)
        if g in word:
            for i, c in enumerate(word):
                if c == g:
                    board[i] = g
        else:
            wrong += 1
    return ("_" not in board), wrong


def evaluate_winrate(words: List[str], guess_fn, num_samples: int = 1000,
                     seed: int = 0) -> Tuple[float, float]:
    rng = random.Random(seed)
    sample = rng.sample(words, min(num_samples, len(words)))
    wins, total_wrong = 0, 0
    for w in sample:
        won, wrong = play_game(w, guess_fn)
        wins += int(won)
        total_wrong += wrong
    return wins / len(sample), total_wrong / len(sample)


# ----------------------------------------------------------------------------
# Old MoE baseline (re-implemented for apples-to-apples comparison)
# ----------------------------------------------------------------------------
class _OldBiLSTM(nn.Module):
    def __init__(self, vocab=28, emb=128, hid=512, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hid * 2, 26)

    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return self.fc(out)


class _OldGate(nn.Module):
    def __init__(self, input_dim=159, hidden=(256, 128), out=5):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_moe_baseline(device):
    names = ["short", "medium", "long", "common", "rare"]
    experts = []
    for n in names:
        m = _OldBiLSTM().to(device)
        m.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"expert_{n}_bilstm.pt"),
                                     map_location=device))
        m.eval()
        experts.append(m)
    gate = _OldGate().to(device)
    gate.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_gating_network.pt"),
                                    map_location=device))
    gate.eval()

    @torch.no_grad()
    def _expert_probs(input_t):
        mask = (input_t == MASK_IDX).float()
        out = []
        for m in experts:
            p = torch.softmax(m(input_t), dim=-1)
            out.append((p * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1))
        return [x.squeeze(0) for x in out]  # 5 x (26,)

    @torch.no_grad()
    def guess_fn(board, guessed):
        revealed = set(c for c in board if c != "_")
        incorrect = sum(1 for ch in guessed if ch not in revealed)
        input_ids = [MASK_IDX if c == "_" else CHAR_TO_IDX[c] for c in board]
        input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        probs_list = _expert_probs(input_t)

        L = len(input_ids)
        blanks = sum(1 for c in board if c == "_")
        guessed_vec = torch.zeros(26, device=device)
        for ch in guessed:
            if ch in CHARS:
                guessed_vec[letter_to_target(ch)] = 1.0
        base = torch.tensor([L / 20.0, blanks / L if L else 0.0, incorrect / 6.0],
                            device=device)
        feat = torch.cat([base, guessed_vec, torch.cat(probs_list)]).unsqueeze(0)
        weights = torch.softmax(gate(feat).squeeze(0), dim=0)
        weighted = torch.zeros(26, device=device)
        for w, p in zip(weights, probs_list):
            weighted += w * p
        for ch in guessed:
            if ch in CHARS:
                weighted[letter_to_target(ch)] = -float("inf")
        return target_to_letter(int(torch.argmax(weighted).item()))

    return guess_fn


def load_encoder(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = HangmanEncoder(**cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(MODELS_DIR, "hangman_encoder.pt"))
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--baseline", action="store_true", help="also evaluate the old MoE")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_words = split_words(load_words())
    print(f"Eval words: {len(eval_words)} | sampling {args.num}")

    if os.path.exists(args.ckpt):
        model = load_encoder(args.ckpt, device)
        for mode in ("mean", "noisy_or"):
            wr, aw = evaluate_winrate(
                eval_words, lambda b, g, m=mode: predict_letter(model, b, g, device, m),
                num_samples=args.num)
            print(f"[encoder/{mode:8s}] win-rate={wr:.4f}  avg-wrong={aw:.2f}")
    else:
        print(f"[skip] no encoder checkpoint at {args.ckpt}")

    if args.baseline:
        gfn = load_moe_baseline(device)
        wr, aw = evaluate_winrate(eval_words, gfn, num_samples=args.num)
        print(f"[old MoE        ] win-rate={wr:.4f}  avg-wrong={aw:.2f}")


if __name__ == "__main__":
    main()
