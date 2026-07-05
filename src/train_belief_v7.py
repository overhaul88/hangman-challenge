"""Strategy v7 — Stage 1: supervised belief fine-tune (the cheap OOV lever).

strategy7.md §0/§7 establish the one load-bearing fact every prior strategy confirmed:
the *decision layer* has negative OOV headroom; the only lever that can beat held-out
0.635 is the **encoder belief's OOV generalisation**. v7's plan moves that belief with
RL (BPTT through the encoder, per-step 5-BiLSTM MoE + frozen-trunk recompute) — measured
here at ~0.4 s/collect-step, i.e. ~3 days for the 2M-episode spec.

This module reaches the *same lever* far more cheaply. The encoder belief `p_enc` is, by
construction, "predict the hidden letters given the revealed board + negative evidence" —
which is exactly the masked-LM objective already used as v7's auxiliary loss. So instead of
pushing a high-variance TD signal through the encoder, we fine-tune the encoder's top-2
blocks + head **directly with MLM** (~42 ms/step, ~10x cheaper than one collect-step and
far lower variance), and we apply v7's *actual* novelty — the pseudo-OOV curriculum — as
the **checkpoint-selection criterion**: every `eval_every` steps we measure win-rate on the
held-out −tion/−ness suffix clusters under the **deployment rule** (0.30·p_enc + 0.70·p_moe,
argmax) and keep the best. The −tion/−ness words and the real 45k OOV set are never trained on.

Freeze map, trainable set, and fusion weight are identical to v7 (`belief_v7.BeliefEngine`,
TrunkFeatures alpha=0.30). The final paired gate is the same no-regression discipline as
§13: ship only if the fine-tuned belief strictly beats strategy2 on the real OOV held-out.

Endpoints
---------
  TRAIN (default): fine-tune + pseudo-OOV selection + final held-out gate
    ~/miniconda3/envs/vessel/bin/python src/train_belief_v7.py --steps 60000
  EVALUATE (paired held-out gate of a saved encoder vs strategy2):
    ~/miniconda3/envs/vessel/bin/python src/train_belief_v7.py --eval_only --ckpt models/hangman_belief_v7.pt
  SMOKE (tiny wiring check):
    ~/miniconda3/envs/vessel/bin/python src/train_belief_v7.py --smoke
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vocab import target_to_letter, MAX_WRONG, NUM_LETTERS, IGNORE_INDEX
from data import load_words, split_words, HangmanStateDataset, collate_fn
from evaluate import play_game
from rl_features import TrunkFeatures
from belief_v7 import BeliefEngine
from oov_split import load_split, load_split_from_json

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
ALPHA = 0.30   # encoder weight in the v2 fused belief: s2 = 0.30*p_enc + 0.70*p_moe


# ----------------------------------------------------------------------------
# Batched deployment-rule evaluation: play the fused belief greedily (no DRQN).
#   guess = argmax_{allowed} (0.30 * p_enc + 0.70 * p_moe)
# This is exactly strategy2's decision rule, but with the (fine-tuned) encoder.
# Returns (win_rate, avg_wrong, per_word_wins, sample).
# ----------------------------------------------------------------------------
@torch.no_grad()
def fused_winrate(engine: BeliefEngine, words, device, num_samples=1000, seed=0,
                  sample=None, batch=256, max_steps=30):
    if sample is None:
        sample = random.Random(seed).sample(words, min(num_samples, len(words)))
    wins = [False] * len(sample)
    total_wrong = 0
    for i in range(0, len(sample), batch):
        chunk = sample[i:i + batch]
        B = len(chunk)
        boards = [["_"] * len(w) for w in chunk]
        guessed = [[] for _ in chunk]
        wrong = [0] * B
        done = [False] * B
        t = 0
        while not all(done) and t < max_steps:
            states = [(boards[b], guessed[b]) for b in range(B)]
            penc = engine.p_enc_from_states(states)
            pmoe = engine.p_moe_from_states(states)
            _, _, allowed = engine.state_features(states)
            score = (ALPHA * penc + (1.0 - ALPHA) * pmoe).masked_fill(~allowed, -1.0)
            action = score.argmax(-1).tolist()
            for b in range(B):
                if done[b]:
                    continue
                g = target_to_letter(action[b])
                guessed[b].append(g)
                w = chunk[b]
                if g in w:
                    for j, c in enumerate(w):
                        if c == g:
                            boards[b][j] = g
                    if "_" not in boards[b]:
                        done[b] = True
                else:
                    wrong[b] += 1
                    if wrong[b] >= MAX_WRONG:
                        done[b] = True
            t += 1
        for b in range(B):
            wins[i + b] = "_" not in boards[b]
            total_wrong += wrong[b]
    n = len(sample)
    return sum(wins) / n, total_wrong / n, wins, sample


# ----------------------------------------------------------------------------
# Forgetting guard: MLM accuracy over hidden positions on a fixed eval batch.
# ----------------------------------------------------------------------------
@torch.no_grad()
def mlm_accuracy(engine: BeliefEngine, batch, device):
    ids, ab, pr, tgt, pad = [t.to(device) for t in batch]
    engine.encoder.train()
    logits = engine.encoder(ids, ab, pr, pad)
    engine.encoder.eval()
    pred = logits.argmax(-1)
    keep = tgt != IGNORE_INDEX
    return (pred[keep] == tgt[keep]).float().mean().item()


# ----------------------------------------------------------------------------
# Paired final gate on the real (fully OOV) held-out split: fine-tuned fused
# belief vs strategy2 (original encoder, same rule). Mirrors train_drqn_v7.paired_gate.
# ----------------------------------------------------------------------------
def paired_gate(engine, held_out, device, num, seed):
    trunk = TrunkFeatures(device)   # fully-frozen strategy2 (original encoder ⊕ MoE)

    def v2_gfn(board, guessed):
        return target_to_letter(trunk.v2_guess_idx(list(board), set(guessed)))

    v7_wr, v7_aw, v7_wins, sample = fused_winrate(engine, held_out, device, num, seed)
    v2_wins, v2_wrong = [], 0
    for w in sample:
        won, wr = play_game(w, v2_gfn)
        v2_wins.append(won); v2_wrong += wr

    both = v2_only = v7_only = neither = 0
    for a, b in zip(v2_wins, v7_wins):
        both += int(a and b); v2_only += int(a and not b)
        v7_only += int(b and not a); neither += int(not a and not b)
    v2_wr = sum(v2_wins) / len(sample)
    return dict(n=len(sample), v2_wr=v2_wr, v2_aw=v2_wrong / len(sample),
                v7_wr=v7_wr, v7_aw=v7_aw, delta=v7_wr - v2_wr,
                both=both, v2_only=v2_only, v7_only=v7_only, neither=neither)


def save_ckpt(path, engine, pseudo_oov_wr, step, suffixes):
    torch.save({"encoder": engine.encoder.state_dict(),
                "encoder_config": engine.encoder_config,
                "pseudo_oov_wr": pseudo_oov_wr, "step": step,
                "suffixes": suffixes, "alpha": ALPHA}, path)


def load_for_eval(path, device, encoder_ckpt):
    ckpt = torch.load(path, map_location=device)
    engine = BeliefEngine(device, encoder_ckpt=encoder_ckpt)
    engine.encoder.load_state_dict(ckpt["encoder"])
    engine.encoder.eval()
    return engine, ckpt


def make_mlm_iter(words, batch_size, hidden_frac_range, workers):
    ds = HangmanStateDataset(words, hidden_frac_range=hidden_frac_range, max_wrong=MAX_WRONG)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        collate_fn=collate_fn, num_workers=workers,
                        persistent_workers=workers > 0)
    while True:
        for b in loader:
            yield b


# ============================================================================
# TRAINING ENDPOINT
# ============================================================================
def train(args, device):
    if args.oov_split and os.path.exists(args.oov_split):
        in_vocab, pseudo_oov, held_out = load_split_from_json(args.oov_split)
        suffixes = "(from json)"
    else:
        suffixes = [s.strip() for s in args.suffixes.split(",") if s.strip()]
        in_vocab, pseudo_oov, held_out = load_split(suffixes)
    if not pseudo_oov:
        raise ValueError("pseudo-OOV set is empty; pick suffixes that match the corpus")
    print(f"in_vocab={len(in_vocab)}  pseudo_oov={len(pseudo_oov)}  held_out(OOV)={len(held_out)}")

    engine = BeliefEngine(device, encoder_ckpt=args.encoder, n_finetune_layers=args.finetune_layers)
    engine.set_encoder_trainable(True)
    n_train = sum(p.numel() for p in engine.trainable_parameters())
    print(f"encoder: {engine.n_layers} layers, fine-tuning top {args.finetune_layers} + head "
          f"({n_train:,} trainable params)  lr={args.lr}")

    opt = torch.optim.Adam(engine.trainable_parameters(), lr=args.lr)
    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP mixed precision: {'on' if use_amp else 'off'}")

    mlm_iter = make_mlm_iter(in_vocab, args.batch, (args.hidden_lo, args.hidden_hi), args.workers)
    # fixed forgetting-guard batch drawn from the held-out pseudo-OOV cluster
    guard_iter = make_mlm_iter(pseudo_oov, args.batch, (args.hidden_lo, args.hidden_hi), 0)
    guard_batch = next(guard_iter)

    os.makedirs(MODELS_DIR, exist_ok=True)
    suffix_list = ([s for s in args.suffixes.split(",")] if isinstance(suffixes, str) else suffixes)

    # ---- step-0 baseline: fused belief with the ORIGINAL encoder == strategy2 ----
    base_wr, base_aw, _, _ = fused_winrate(engine, pseudo_oov, device, args.eval_num, args.eval_seed)
    base_acc = mlm_accuracy(engine, guard_batch, device)
    print(f"[step 0 / strategy2 baseline] pseudoOOV wr={base_wr:.4f} aw={base_aw:.2f} "
          f"mlm_acc={base_acc:.3f}", flush=True)
    # pseudo-OOV win-rate saturates near ceiling on the easy -tion/-ness clusters, so it ties
    # often; break ties on avg-wrong (efficiency) which stays continuous and tracks robustness.
    best_wr, best_aw = base_wr, base_aw
    save_ckpt(args.out, engine, best_wr, 0, suffix_list)   # safe default == v2

    t0 = time.time()
    run_loss = 0.0
    for step in range(1, args.steps + 1):
        ids, ab, pr, tgt, pad = [t.to(device) for t in next(mlm_iter)]
        engine.encoder.train()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = engine.encoder(ids, ab, pr, pad)
            loss = F.cross_entropy(logits.reshape(-1, NUM_LETTERS), tgt.reshape(-1),
                                   ignore_index=IGNORE_INDEX)
        engine.encoder.eval()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(engine.trainable_parameters(), args.clip)
        scaler.step(opt)
        scaler.update()
        run_loss += loss.item()

        if step % args.eval_every == 0 or step == args.steps:
            wr, aw, _, _ = fused_winrate(engine, pseudo_oov, device, args.eval_num, args.eval_seed)
            acc = mlm_accuracy(engine, guard_batch, device)
            tag = ""
            if wr > best_wr or (wr == best_wr and aw < best_aw):
                best_wr, best_aw = wr, aw
                save_ckpt(args.out, engine, best_wr, step, suffix_list)
                tag = " -> saved best"
            drop = base_acc - acc
            warn = "  [!] MLM acc dropped >5pt" if drop > 0.05 else ""
            print(f"step={step} loss={run_loss/args.eval_every:.4f} | pseudoOOV wr={wr:.4f} "
                  f"aw={aw:.2f} (best={best_wr:.4f}, base={base_wr:.4f}){tag} | "
                  f"mlm_acc={acc:.3f} (base={base_acc:.3f}){warn} [{(time.time()-t0)/60:.1f}m]",
                  flush=True)
            run_loss = 0.0

    print("\n=== Final held-out (OOV) gate: v7-belief vs strategy2 ===")
    engine_b, _ = load_for_eval(args.out, device, args.encoder)
    res = paired_gate(engine_b, held_out, device, args.final_eval_num, args.eval_seed)
    print(f"strategy2  : win-rate={res['v2_wr']:.4f}  avg-wrong={res['v2_aw']:.2f}")
    print(f"v7-belief  : win-rate={res['v7_wr']:.4f}  avg-wrong={res['v7_aw']:.2f}")
    print(f"delta = {res['delta']:+.4f}  | discordant: v7_only={res['v7_only']} "
          f"v2_only={res['v2_only']} (both={res['both']} neither={res['neither']}, n={res['n']})")
    if res["v7_wr"] > res["v2_wr"]:
        print(f"SHIP v7-belief (> strategy2 {res['v2_wr']:.4f}).  Saved -> {args.out}")
    else:
        print(f"v7-belief did NOT beat strategy2 on held-out OOV; ship strategy2 (no-regression).")


# ============================================================================
# EVALUATION ENDPOINT
# ============================================================================
def run_eval_only(args, device):
    _, _, held_out = (load_split_from_json(args.oov_split)
                      if (args.oov_split and os.path.exists(args.oov_split))
                      else load_split([s.strip() for s in args.suffixes.split(",") if s.strip()]))
    if not os.path.exists(args.ckpt):
        print(f"[no checkpoint at {args.ckpt}] nothing to evaluate.")
        return
    engine, ckpt = load_for_eval(args.ckpt, device, args.encoder)
    print(f"Loaded belief checkpoint (pseudo-OOV wr={ckpt.get('pseudo_oov_wr')}, "
          f"step={ckpt.get('step')})")
    res = paired_gate(engine, held_out, device, args.final_eval_num, args.eval_seed)
    print("\n=== Held-out (OOV) gate: v7-belief vs strategy2 ===")
    print(f"strategy2  : win-rate={res['v2_wr']:.4f}  avg-wrong={res['v2_aw']:.2f}")
    print(f"v7-belief  : win-rate={res['v7_wr']:.4f}  avg-wrong={res['v7_aw']:.2f}")
    print(f"delta = {res['delta']:+.4f}  | discordant: v7_only={res['v7_only']} "
          f"v2_only={res['v2_only']} (both={res['both']} neither={res['neither']}, n={res['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", default=os.path.join(MODELS_DIR, "hangman_belief_v7.pt"))
    ap.add_argument("--out", default=os.path.join(MODELS_DIR, "hangman_belief_v7.pt"))
    ap.add_argument("--encoder", default=os.path.join(MODELS_DIR, "hangman_encoder.pt"))
    ap.add_argument("--oov_split", default=os.path.join(MODELS_DIR, "oov_split.json"))
    ap.add_argument("--suffixes", default="tion,ness")
    # schedule (defaults sized for ~1.5-2.5h on a 4 GB GTX 1650)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--finetune_layers", type=int, default=2)
    ap.add_argument("--hidden_lo", type=float, default=0.3, help="min fraction of unique letters hidden")
    ap.add_argument("--hidden_hi", type=float, default=1.0, help="max fraction hidden (1.0 = opening board)")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=3_000)
    ap.add_argument("--eval_num", type=int, default=1_000)
    ap.add_argument("--final_eval_num", type=int, default=3_000)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        (args.steps, args.batch, args.eval_every, args.eval_num, args.final_eval_num) = (
            60, 16, 30, 200, 200)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.eval_only:
        run_eval_only(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
