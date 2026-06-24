"""Train the enhanced Hangman encoder.

- Realistic letter-level masking + absent/present conditioning (see data.py).
- CrossEntropy loss only on hidden positions.
- Curriculum that widens from easy (mostly-revealed, few wrong) toward the full game
  range (including all-blank openings and up to 6 wrong guesses).
- Checkpoints the model with the best *simulated game win-rate* on the eval split.
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vocab import IGNORE_INDEX, NUM_LETTERS
from data import load_words, split_words, HangmanStateDataset, collate_fn
from model import HangmanEncoder
from evaluate import evaluate_winrate, predict_letter

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# (start_epoch -> (hidden_frac_range, max_wrong))
CURRICULUM = {
    0:  ((0.20, 0.50), 2),
    4:  ((0.30, 0.70), 4),
    8:  ((0.30, 1.00), 6),
    12: ((0.20, 1.00), 6),
}


def build_config(args):
    return dict(arch=args.arch, d_model=args.d_model, nhead=args.nhead,
                num_layers=args.layers, dim_feedforward=args.ff,
                dropout=args.dropout, max_len=args.max_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--arch", default="transformer", choices=["transformer", "bilstm"])
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=40)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--eval_num", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(MODELS_DIR, "hangman_encoder.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    words = load_words()
    train_words, eval_words = split_words(words)
    print(f"Train: {len(train_words)}  Eval: {len(eval_words)}")

    dataset = HangmanStateDataset(train_words)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True,
                        drop_last=True)

    cfg = build_config(args)
    model = HangmanEncoder(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.arch} | params={n_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    warmup = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))  # cosine decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_wr = -1.0
    for epoch in range(args.epochs):
        if epoch in CURRICULUM:
            rng, mw = CURRICULUM[epoch]
            dataset.set_curriculum(rng, mw)
            print(f"  curriculum -> hidden_frac={rng} max_wrong={mw}")

        model.train()
        t0, total_loss = time.time(), 0.0
        for inputs, absent, present, targets, pad_mask in loader:
            inputs = inputs.to(device, non_blocking=True)
            absent = absent.to(device, non_blocking=True)
            present = present.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs, absent, present, pad_mask)
            loss = criterion(logits.reshape(-1, NUM_LETTERS), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        msg = f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | {time.time()-t0:.0f}s"

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            wr, aw = evaluate_winrate(
                eval_words,
                lambda b, g: predict_letter(model, b, g, device, "noisy_or"),
                num_samples=args.eval_num)
            msg += f" | win-rate={wr:.4f} avg-wrong={aw:.2f}"
            if wr > best_wr:
                best_wr = wr
                os.makedirs(MODELS_DIR, exist_ok=True)
                torch.save({"model": model.state_dict(), "config": cfg,
                            "win_rate": wr, "epoch": epoch + 1}, args.out)
                msg += "  -> saved best"
        print(msg, flush=True)

    print(f"Done. Best win-rate={best_wr:.4f}. Saved to {args.out}")


if __name__ == "__main__":
    main()
