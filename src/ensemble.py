"""Ensemble inference: blend the enhanced encoder with the old MoE at the score level.

Both produce a 26-dim letter-score vector for the current board:
  - encoder: per-position softmax averaged over hidden cells (negative-evidence aware)
  - MoE:     gated weighted average of the 5 experts' blank-averaged probs
We blend `alpha*encoder + (1-alpha)*moe`, mask guessed letters, and pick the argmax.

Run as a script to sweep alpha on the shared eval split.
"""
import argparse
import os

import torch

from vocab import (CHARS, CHAR_TO_IDX, MASK_IDX, NUM_LETTERS,
                   letter_to_target, target_to_letter)
from data import load_words, split_words
from evaluate import (load_encoder, load_moe_baseline, _aggregate,
                      evaluate_winrate)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


@torch.no_grad()
def encoder_scores(model, board, guessed, device, mode="mean"):
    revealed = set(c for c in board if c != "_")
    absent = guessed - revealed
    input_ids = [MASK_IDX if c == "_" else CHAR_TO_IDX[c] for c in board]
    input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    absent_mh = torch.zeros(1, NUM_LETTERS, device=device)
    for ch in absent:
        absent_mh[0, letter_to_target(ch)] = 1.0
    present_mh = torch.zeros(1, NUM_LETTERS, device=device)
    for ch in revealed:
        present_mh[0, letter_to_target(ch)] = 1.0
    logits = model(input_t, absent_mh, present_mh, pad_mask=None)
    probs = torch.softmax(logits[0], dim=-1)
    hidden = (input_t[0] == MASK_IDX)
    return _aggregate(probs, hidden, mode)  # (26,) in [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(MODELS_DIR, "hangman_encoder.pt"))
    ap.add_argument("--num", type=int, default=3000)
    ap.add_argument("--alphas", default="0.4,0.5,0.6,0.7")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_words = split_words(load_words())

    model = load_encoder(args.ckpt, device)
    moe_score = _moe_scorer(device)

    for a in [float(x) for x in args.alphas.split(",")]:
        def guess_fn(board, guessed, a=a):
            enc = encoder_scores(model, board, guessed, device, "mean")
            moe = moe_score(board, guessed)
            blend = a * enc + (1 - a) * moe
            for ch in guessed:
                if ch in CHARS:
                    blend[letter_to_target(ch)] = -float("inf")
            return target_to_letter(int(torch.argmax(blend).item()))
        wr, aw = evaluate_winrate(eval_words, guess_fn, num_samples=args.num)
        print(f"[ensemble alpha={a:.2f}] win-rate={wr:.4f}  avg-wrong={aw:.2f}", flush=True)


def _moe_scorer(device):
    """Like load_moe_baseline but returns the 26-dim blended-prob vector (pre-argmax)."""
    from evaluate import _OldBiLSTM, _OldGate
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
    def score(board, guessed):
        revealed = set(c for c in board if c != "_")
        incorrect = sum(1 for ch in guessed if ch not in revealed)
        input_ids = [MASK_IDX if c == "_" else CHAR_TO_IDX[c] for c in board]
        input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        mask = (input_t == MASK_IDX).float()
        probs_list = []
        for m in experts:
            p = torch.softmax(m(input_t), dim=-1)
            probs_list.append(((p * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1)).squeeze(0))
        L = len(input_ids)
        blanks = sum(1 for c in board if c == "_")
        gvec = torch.zeros(26, device=device)
        for ch in guessed:
            if ch in CHARS:
                gvec[letter_to_target(ch)] = 1.0
        base = torch.tensor([L / 20.0, blanks / L if L else 0.0, incorrect / 6.0], device=device)
        feat = torch.cat([base, gvec, torch.cat(probs_list)]).unsqueeze(0)
        w = torch.softmax(gate(feat).squeeze(0), dim=0)
        out = torch.zeros(26, device=device)
        for wi, p in zip(w, probs_list):
            out += wi * p
        return out
    return score


if __name__ == "__main__":
    main()
