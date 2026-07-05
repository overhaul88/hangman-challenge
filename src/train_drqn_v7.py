"""Strategy v7 — training & evaluation endpoints for the belief-unfreezing DRQN.

strategy7.md §4-§13. Phase-2 RL fine-tuning of the strategy2 encoder's top blocks together
with a fresh GRU DRQN over the two beliefs, on an 80/20 in-vocab / pseudo-OOV word mix.

Two optimisers run together (Adam on the DRQN @3e-4, Adam on the encoder top-2 + head
@1e-5). Each learning step: sample a prioritised batch of full episodes, recompute `p_enc`
WITH grad, run BPTT through the GRU, take the Double-DQN loss (IS-weighted), add 0.1·MLM on
a fresh batch of simulated boards, clip the two groups separately, step both optimisers and
EMA-update the target net. The encoder top blocks are frozen for the first `warmup_episodes`
(DRQN-only) before they start moving. Checkpoints are selected on **pseudo-OOV** win-rate.

Endpoints
---------
  TRAIN (default):
    ~/miniconda3/envs/vessel/bin/python src/train_drqn_v7.py --episodes 2000000
  EVALUATE (paired held-out gate of a saved checkpoint vs strategy2; no training):
    ~/miniconda3/envs/vessel/bin/python src/train_drqn_v7.py --eval_only --ckpt models/hangman_drqn_v7.pt
  SMOKE (tiny wiring check; NOT a real run):
    ~/miniconda3/envs/vessel/bin/python src/train_drqn_v7.py --smoke
"""
import argparse
import os
import random
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from vocab import target_to_letter, MAX_WRONG, NUM_LETTERS, IGNORE_INDEX
from data import load_words, split_words, HangmanStateDataset, collate_fn
from evaluate import play_game
from rl_features import TrunkFeatures
from belief_v7 import BeliefEngine
from policy_drqn_v7 import GRUDRQN, NEG_MASK
from per_replay import PrioritizedEpisodeReplay
from env_v7 import VecDenseHangmanEnv
from oov_split import load_split, load_split_from_json

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


# ----------------------------------------------------------------------------
# Batch assembly: turn replayed episodes into time-major tensors, recomputing the
# encoder belief p_enc WITH GRAD (the v7-defining step).
# ----------------------------------------------------------------------------
def assemble_batch(engine: BeliefEngine, episodes, device):
    lengths = [int(e["action"].shape[0]) for e in episodes]
    B, Tmax = len(episodes), max(lengths)

    penc = torch.zeros(Tmax, B, NUM_LETTERS, device=device)
    pmoe = torch.zeros(Tmax, B, NUM_LETTERS, device=device)
    gvec = torch.zeros(Tmax, B, NUM_LETTERS, device=device)
    wrong = torch.zeros(Tmax, B, device=device)
    tidx = torch.zeros(Tmax, B, device=device)
    action = torch.zeros(Tmax, B, dtype=torch.long, device=device)
    reward = torch.zeros(Tmax, B, device=device)
    done = torch.ones(Tmax, B, device=device)
    mask = torch.zeros(Tmax, B, device=device)
    reset = torch.zeros(Tmax, B, device=device)
    allowed = torch.ones(Tmax, B, NUM_LETTERS, dtype=torch.bool, device=device)

    # --- grad-enabled p_enc from the CACHED frozen trunk, grouped by word length ---
    # The frozen lower-encoder hidden (H3) was cached at collection time; only the trainable
    # top blocks + head are recomputed here (with grad), so this is ~3x cheaper than the full
    # forward while being numerically/gradient-identical (frozen-trunk grad is zero).
    byL = defaultdict(list)
    for b, e in enumerate(episodes):
        byL[e["trunk"].shape[1]].append(b)            # trunk: (T,L,384) -> L = shape[1]
    for L, bs in byL.items():
        Hs, ms, ts, bb = [], [], [], []
        for b in bs:
            e, T = episodes[b], lengths[b]
            Hs.append(e["trunk"]); ms.append(e["mask"])
            ts += list(range(T)); bb += [b] * T
        p = engine.p_enc_from_trunk(torch.cat(Hs, 0), torch.cat(ms, 0))
        penc[torch.tensor(ts, device=device), torch.tensor(bb, device=device)] = p

    # --- everything else (cached / derived) ---
    for b, e in enumerate(episodes):
        T = lengths[b]
        pmoe[:T, b] = e["p_moe"]
        gv = (e["absent"] + e["present"]).clamp(max=1.0)
        gvec[:T, b] = gv
        wrong[:T, b] = e["absent"].sum(1)
        allowed[:T, b] = gv == 0.0
        action[:T, b] = e["action"]
        reward[:T, b] = e["reward"]
        done[:T, b] = e["done"]
        mask[:T, b] = 1.0
        reset[0, b] = 1.0
        tidx[:T, b] = torch.arange(T, device=device).float() / 11.0

    x = torch.cat([penc, pmoe, gvec, (wrong / 6.0).unsqueeze(-1), tidx.unsqueeze(-1)], dim=-1)
    return dict(x=x, allowed=allowed, action=action, reward=reward,
                done=done, mask=mask, reset=reset)


# ----------------------------------------------------------------------------
# Double-DQN loss with PER importance weights; returns per-episode priorities.
# ----------------------------------------------------------------------------
def drqn_loss_v7(online, target, batch, weights, gamma):
    x, AL = batch["x"], batch["allowed"]
    A, R, D, M, RS = batch["action"], batch["reward"], batch["done"], batch["mask"], batch["reset"]
    Tm, B = A.shape
    dev = x.device

    q_online, _ = online.q_sequence(x, AL, RS, online.initial_state(B, dev))   # grad
    with torch.no_grad():
        q_target, _ = target.q_sequence(x, AL, RS, target.initial_state(B, dev))
    q_taken = q_online.gather(-1, A.unsqueeze(-1)).squeeze(-1)                  # (Tm,B)

    with torch.no_grad():
        boot_all = torch.zeros(Tm, B, device=dev)
        if Tm > 1:
            a_star = q_online.detach()[1:].argmax(-1)                          # (Tm-1,B)
            boot = q_target[1:].gather(-1, a_star.unsqueeze(-1)).squeeze(-1)
            boot_all[:Tm - 1] = boot
        y = R + gamma * (1.0 - D) * boot_all

    td = q_taken - y
    denom_b = M.sum(0).clamp(min=1.0)                                          # (B,)
    loss_b = ((td ** 2) * M).sum(0) / denom_b                                  # (B,)
    loss = (weights * loss_b).sum() / weights.sum().clamp(min=1e-8)            # IS-weighted

    with torch.no_grad():
        prio_b = (td.abs() * M).sum(0) / denom_b                              # (B,)
        mtot = M.sum().clamp(min=1.0)
        stats = {"loss": loss.item(),
                 "q": (q_taken * M).sum().item() / mtot.item(),
                 "y": (y * M).sum().item() / mtot.item()}
    return loss, prio_b.cpu(), stats


# ----------------------------------------------------------------------------
# Auxiliary MLM (catastrophic-forgetting guard) on fresh simulated boards.
# ----------------------------------------------------------------------------
def make_mlm_iter(words, batch_size, hidden_frac_range=(0.3, 1.0), max_wrong=6, workers=0):
    ds = HangmanStateDataset(words, hidden_frac_range=hidden_frac_range, max_wrong=max_wrong)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        collate_fn=collate_fn, num_workers=workers)
    while True:
        for batch in loader:
            yield batch


def mlm_loss(engine, batch, device):
    ids, ab, pr, tgt, pad = [t.to(device) for t in batch]
    # Run the padded MLM forward in train() mode (as strategy2 was trained): eval mode would
    # try the eval-only attention fast path, which is incompatible with a padding mask under
    # autograd. Restore eval() afterwards so acting/eval beliefs stay dropout-free.
    engine.encoder.train()
    logits = engine.encoder(ids, ab, pr, pad)                # (B,L,26), grad on top blocks
    engine.encoder.eval()
    return F.cross_entropy(logits.reshape(-1, NUM_LETTERS), tgt.reshape(-1),
                           ignore_index=IGNORE_INDEX)


# ----------------------------------------------------------------------------
# Batched greedy evaluation (deployment rules): threads the GRU state per game.
# Returns (win_rate, avg_wrong, per_word_wins, sample_order).
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_v7(engine, drqn, words, device, num_samples=2000, seed=0,
                sample=None, batch=256, max_steps=30):
    if sample is None:
        sample = random.Random(seed).sample(words, min(num_samples, len(words)))
    drqn.eval()
    wins = [False] * len(sample)
    total_wrong = 0
    for i in range(0, len(sample), batch):
        chunk = sample[i:i + batch]
        B = len(chunk)
        boards = [["_"] * len(w) for w in chunk]
        guessed = [[] for _ in chunk]
        wrong = [0] * B
        done = [False] * B
        h = drqn.initial_state(B, device)
        reset = torch.ones(B, device=device)
        t = 0
        while not all(done) and t < max_steps:
            states = [(boards[b], guessed[b]) for b in range(B)]
            penc = engine.p_enc_from_states(states)
            pmoe = engine.p_moe_from_states(states)
            gvec, wr, allowed = engine.state_features(states)
            tcol = torch.full((B, 1), t / 11.0, device=device)
            x = torch.cat([penc, pmoe, gvec, (wr / 6.0).unsqueeze(-1), tcol], dim=-1)
            q, h = drqn.q_step(x, h, reset, allowed)
            reset = torch.zeros(B, device=device)
            action = q.argmax(-1).tolist()
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
# Paired final gate: v7 vs strategy2 on the real (fully OOV) held-out split.
# ----------------------------------------------------------------------------
def paired_gate(engine, drqn, held_out, device, num, seed):
    trunk = TrunkFeatures(device)   # fresh, fully-frozen strategy2 (encoder ⊕ MoE)

    def v2_gfn(board, guessed):
        return target_to_letter(trunk.v2_guess_idx(list(board), set(guessed)))

    v7_wr, v7_aw, v7_wins, sample = evaluate_v7(engine, drqn, held_out, device, num, seed)
    v2_wins, v2_wrong = [], 0
    for w in sample:
        won, wr = play_game(w, v2_gfn)
        v2_wins.append(won); v2_wrong += wr

    both = v2_only = v7_only = neither = 0
    for a, b in zip(v2_wins, v7_wins):
        both += int(a and b); v2_only += int(a and not b)
        v7_only += int(b and not a); neither += int(not a and not b)
    return dict(n=len(sample), v2_wr=sum(v2_wins) / len(sample), v2_aw=v2_wrong / len(sample),
                v7_wr=v7_wr, v7_aw=v7_aw, delta=v7_wr - sum(v2_wins) / len(sample),
                both=both, v2_only=v2_only, v7_only=v7_only, neither=neither)


def epsilon_at(ep_done, total, eps_start, eps_end, decay_frac):
    span = max(1, int(decay_frac * total))
    return eps_start + min(1.0, ep_done / span) * (eps_end - eps_start)


# ----------------------------------------------------------------------------
# Checkpoint I/O
# ----------------------------------------------------------------------------
def save_ckpt(path, online, drqn_cfg, engine, pseudo_oov_wr, episodes, suffixes):
    torch.save({"drqn": online.state_dict(), "drqn_config": drqn_cfg,
                "encoder": engine.encoder.state_dict(),
                "encoder_config": engine.encoder_config,
                "pseudo_oov_wr": pseudo_oov_wr, "episodes": episodes,
                "suffixes": suffixes}, path)


def load_for_eval(path, device, encoder_ckpt):
    ckpt = torch.load(path, map_location=device)
    engine = BeliefEngine(device, encoder_ckpt=encoder_ckpt)
    engine.encoder.load_state_dict(ckpt["encoder"])   # fine-tuned weights
    engine.encoder.eval()
    drqn = GRUDRQN(**ckpt["drqn_config"]).to(device)
    drqn.load_state_dict(ckpt["drqn"]); drqn.eval()
    return engine, drqn, ckpt


# ============================================================================
# TRAINING ENDPOINT
# ============================================================================
def train(args, device):
    # --- data split (in-vocab / pseudo-OOV / real held-out OOV) ---
    if args.oov_split and os.path.exists(args.oov_split):
        in_vocab, pseudo_oov, held_out = load_split_from_json(args.oov_split)
        suffixes = "(from json)"
    else:
        suffixes = [s.strip() for s in args.suffixes.split(",") if s.strip()]
        in_vocab, pseudo_oov, held_out = load_split(suffixes)
    if not pseudo_oov:
        raise ValueError("pseudo-OOV set is empty; pick suffixes that match the corpus")
    print(f"in_vocab={len(in_vocab)}  pseudo_oov={len(pseudo_oov)}  held_out(OOV)={len(held_out)}")

    # --- belief engine (encoder top-2 + head trainable; MoE frozen) ---
    engine = BeliefEngine(device, encoder_ckpt=args.encoder, n_finetune_layers=args.finetune_layers)
    engine.set_encoder_trainable(False)   # warmup: DRQN-only first
    engine.enable_frozen_cache(args.cache_cap)   # memoize frozen p_moe + trunk (collection speedup)
    print(f"encoder: {engine.n_layers} layers, fine-tuning top {args.finetune_layers} + head "
          f"({sum(p.numel() for p in engine.trainable_parameters()):,} trainable params)")
    print(f"frozen-oracle cache: cap={args.cache_cap:,} states")

    # --- DRQN online/target ---
    drqn_cfg = dict(input_dim=80, gru_hidden=args.gru, head_hidden=args.head_hidden,
                    n_letters=NUM_LETTERS)
    online = GRUDRQN(**drqn_cfg).to(device)
    target = GRUDRQN(**drqn_cfg).to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)

    opt_drqn = torch.optim.Adam(online.parameters(), lr=args.lr_drqn)
    opt_enc = torch.optim.Adam(engine.trainable_parameters(), lr=args.lr_enc)

    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP mixed precision: {'on' if use_amp else 'off'}")

    buffer = PrioritizedEpisodeReplay(args.capacity, alpha=args.per_alpha)
    env = VecDenseHangmanEnv(in_vocab, pseudo_oov, args.n_envs,
                             pseudo_oov_frac=args.pseudo_oov_frac, seed=args.seed)
    mlm_iter = make_mlm_iter(in_vocab, args.mlm_batch, workers=args.mlm_workers)

    N = args.n_envs
    obs = env.reset()
    h = online.initial_state(N, device)
    reset = torch.ones(N, device=device)
    t_env = [0] * N
    ongoing = [defaultdict(list) for _ in range(N)]
    win_hist = deque(maxlen=4000)
    episodes_done = 0
    grad_steps = 0
    last_eval = 0
    best_wr = -1.0
    unfrozen = False
    os.makedirs(MODELS_DIR, exist_ok=True)
    t0 = time.time()

    while episodes_done < args.episodes:
        # ---- warmup boundary: unfreeze encoder top blocks ----
        if not unfrozen and episodes_done >= args.warmup_episodes:
            engine.set_encoder_trainable(True)
            unfrozen = True
            print(f"[warmup done @ {episodes_done} eps] unfroze encoder top "
                  f"{args.finetune_layers} blocks + head", flush=True)

        eps = epsilon_at(episodes_done, args.episodes, args.eps_start, args.eps_end,
                         args.eps_decay_frac)

        # ---------------- collection (prior-guided epsilon-greedy) ----------------
        online.eval()
        for _ in range(args.collect_steps):
            states = [o.state for o in obs]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                # memoized frozen oracles (p_moe + trunk H3) + fresh top-block p_enc, one shot
                penc, pmoe, caches, absents, presents = engine.collection_beliefs(states)
            # derive guessed/wrong/allowed from the (cached) absent/present — no re-parse/forward
            absent_t = torch.stack(absents).to(device)      # (N,26)
            present_t = torch.stack(presents).to(device)    # (N,26)
            gvec = (absent_t + present_t).clamp(max=1.0)
            wrong = absent_t.sum(1)
            allowed = gvec == 0.0
            tcol = torch.tensor([[t_env[i] / 11.0] for i in range(N)], device=device)
            # GRU collection step runs in fp32 (outside autocast) -> force fp32 belief inputs.
            penc, pmoe = penc.float(), pmoe.float()
            x = torch.cat([penc, pmoe, gvec, (wrong / 6.0).unsqueeze(-1), tcol], dim=-1)
            with torch.no_grad():
                q, h = online.q_step(x, h, reset, allowed)
            greedy = q.argmax(-1)
            prior = 0.30 * penc + 0.70 * pmoe
            prior_logits = torch.log(prior + 1e-9).masked_fill(~allowed, NEG_MASK)
            prior_sample = Categorical(logits=prior_logits).sample()
            explore = torch.rand(N, device=device) < eps
            action = torch.where(explore, prior_sample, greedy)

            letters = [target_to_letter(a) for a in action.tolist()]
            obs, rewards, dones, infos = env.step(letters)

            pmoe_c = pmoe.detach().cpu()
            act_c = action.detach().cpu()
            for i in range(N):
                H3_i, mask_i = caches[i]                                 # cached frozen trunk
                ep = ongoing[i]
                ep["trunk"].append(H3_i); ep["mask"].append(mask_i)
                ep["absent"].append(absents[i]); ep["present"].append(presents[i])
                ep["p_moe"].append(pmoe_c[i])
                ep["action"].append(int(act_c[i]))
                ep["reward"].append(float(rewards[i]))
                ep["done"].append(1.0 if dones[i] else 0.0)
                if dones[i]:
                    buffer.add_episode({
                        "trunk": torch.stack(ep["trunk"]),     # (T,L,384) fp16
                        "mask": torch.stack(ep["mask"]),       # (T,L) bool
                        "absent": torch.stack(ep["absent"]),
                        "present": torch.stack(ep["present"]),
                        "p_moe": torch.stack(ep["p_moe"]),
                        "action": torch.tensor(ep["action"], dtype=torch.long),
                        "reward": torch.tensor(ep["reward"], dtype=torch.float32),
                        "done": torch.tensor(ep["done"], dtype=torch.float32),
                    })
                    episodes_done += 1
                    win_hist.append(1.0 if infos[i]["win"] else 0.0)
                    ongoing[i] = defaultdict(list)
                    t_env[i] = 0
                else:
                    t_env[i] += 1
            reset = torch.tensor([1.0 if d else 0.0 for d in dones], device=device)

        # ---------------- learning ----------------
        stats = {"loss": float("nan"), "q": float("nan"), "y": float("nan")}
        mlm_val = float("nan")
        if len(buffer) >= args.learning_starts and buffer.can_sample(args.batch_episodes):
            online.train()
            for _ in range(args.train_steps):
                episodes, idx, w = buffer.sample(args.batch_episodes, args.per_beta, device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    batch = assemble_batch(engine, episodes, device)
                    loss, prio_b, stats = drqn_loss_v7(online, target, batch, w, args.gamma)
                    total = loss
                    if unfrozen:
                        mlm = mlm_loss(engine, next(mlm_iter), device)
                        total = total + args.mlm_lambda * mlm
                        mlm_val = mlm.item()

                opt_drqn.zero_grad(set_to_none=True)
                opt_enc.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                # unscale before clipping each parameter group separately (norms unchanged)
                scaler.unscale_(opt_drqn)
                torch.nn.utils.clip_grad_norm_(online.parameters(), args.drqn_clip)
                if unfrozen:
                    scaler.unscale_(opt_enc)
                    torch.nn.utils.clip_grad_norm_(engine.trainable_parameters(), args.enc_clip)
                scaler.step(opt_drqn)
                if unfrozen:
                    scaler.step(opt_enc)
                scaler.update()
                with torch.no_grad():
                    for tp, op in zip(target.parameters(), online.parameters()):
                        tp.data.mul_(1.0 - args.tau).add_(op.data, alpha=args.tau)
                buffer.update_priorities(idx, prio_b.tolist())
                grad_steps += 1

        # ---------------- periodic pseudo-OOV eval + checkpoint selection ----------------
        if episodes_done - last_eval >= args.eval_every and len(buffer) >= args.learning_starts:
            last_eval = episodes_done
            wr, aw, _, _ = evaluate_v7(engine, online, pseudo_oov, device,
                                       args.eval_num, seed=args.eval_seed)
            tag = ""
            if wr > best_wr:
                best_wr = wr
                save_ckpt(args.out, online, drqn_cfg, engine, best_wr, episodes_done,
                          [s for s in args.suffixes.split(",")] if isinstance(suffixes, str)
                          else suffixes)
                tag = " -> saved best"
            recent = sum(win_hist) / len(win_hist) if win_hist else float("nan")
            tot_lookup = engine.cache_hits + engine.cache_miss
            hit_rate = engine.cache_hits / max(1, tot_lookup)
            print(f"eps={episodes_done} gstep={grad_steps} epsilon={eps:.2f} "
                  f"roll_wr={recent:.3f} buf={len(buffer)} loss={stats['loss']:.3f} "
                  f"q={stats['q']:.2f} mlm={mlm_val:.3f} | pseudoOOV wr={wr:.4f} "
                  f"(best={best_wr:.4f}){tag} | cache_hit={hit_rate:.2f}(n={len(engine._cache)}) "
                  f"[{(time.time()-t0)/60:.1f}m]", flush=True)

    # ---------------- final paired held-out gate ----------------
    print("\n=== Final held-out (OOV) gate: v7 vs strategy2 ===")
    if best_wr < 0:   # never selected (e.g. short run) — save current as the candidate
        save_ckpt(args.out, online, drqn_cfg, engine, best_wr, episodes_done,
                  suffixes if not isinstance(suffixes, str) else None)
    engine_b, drqn_b, _ = load_for_eval(args.out, device, args.encoder)
    res = paired_gate(engine_b, drqn_b, held_out, device, args.final_eval_num, args.eval_seed)
    print(f"strategy2 : win-rate={res['v2_wr']:.4f}  avg-wrong={res['v2_aw']:.2f}")
    print(f"v7 (best) : win-rate={res['v7_wr']:.4f}  avg-wrong={res['v7_aw']:.2f}")
    print(f"delta = {res['delta']:+.4f}  | discordant: v7_only={res['v7_only']} "
          f"v2_only={res['v2_only']} (both={res['both']} neither={res['neither']}, n={res['n']})")
    if res["v7_wr"] > res["v2_wr"]:
        print(f"SHIP v7 (> strategy2 {res['v2_wr']:.4f}).  Saved -> {args.out}")
    else:
        print(f"v7 did NOT beat strategy2 on held-out OOV; ship strategy2 (no-regression).")


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
    engine, drqn, ckpt = load_for_eval(args.ckpt, device, args.encoder)
    print(f"Loaded v7 checkpoint (pseudo-OOV wr={ckpt.get('pseudo_oov_wr')}, "
          f"episodes={ckpt.get('episodes')})")
    res = paired_gate(engine, drqn, held_out, device, args.final_eval_num, args.eval_seed)
    print("\n=== Held-out (OOV) gate: v7 vs strategy2 ===")
    print(f"strategy2 : win-rate={res['v2_wr']:.4f}  avg-wrong={res['v2_aw']:.2f}")
    print(f"v7        : win-rate={res['v7_wr']:.4f}  avg-wrong={res['v7_aw']:.2f}")
    print(f"delta = {res['delta']:+.4f}  | discordant: v7_only={res['v7_only']} "
          f"v2_only={res['v2_only']} (both={res['both']} neither={res['neither']}, n={res['n']})")


def main():
    ap = argparse.ArgumentParser()
    # endpoints
    ap.add_argument("--eval_only", action="store_true",
                    help="evaluation endpoint: paired held-out v7-vs-strategy2 on --ckpt")
    ap.add_argument("--smoke", action="store_true", help="tiny wiring check (not a real run)")
    ap.add_argument("--ckpt", default=os.path.join(MODELS_DIR, "hangman_drqn_v7.pt"))
    ap.add_argument("--out", default=os.path.join(MODELS_DIR, "hangman_drqn_v7.pt"))
    ap.add_argument("--encoder", default=os.path.join(MODELS_DIR, "hangman_encoder.pt"))
    ap.add_argument("--oov_split", default=os.path.join(MODELS_DIR, "oov_split.json"))
    ap.add_argument("--suffixes", default="tion,ness", help="used if --oov_split missing")
    # schedule (defaults sized for a ~8h run on a 4 GB GTX 1650 with AMP + trunk caching;
    # pass --episodes 2000000 --warmup_episodes 100000 for the full spec budget)
    # Belief collection saturates ~150 env-steps/s at N>=256 (latency-bound: 5 seq. BiLSTMs +
    # per-length buckets; peak GPU mem is only ~0.3/4 GB, so we are NOT memory-bound). At that
    # rate ~300k episodes fit in ~8h once training is interleaved. Bigger N amortises kernel
    # launches (~2.5x over N=32); collect/train are rebalanced to hold the N=32 data:grad ratio.
    ap.add_argument("--episodes", type=int, default=300_000)
    ap.add_argument("--warmup_episodes", type=int, default=20_000)
    ap.add_argument("--n_envs", type=int, default=256)
    ap.add_argument("--collect_steps", type=int, default=4)
    ap.add_argument("--train_steps", type=int, default=8)
    ap.add_argument("--batch_episodes", type=int, default=16)
    ap.add_argument("--capacity", type=int, default=100_000)
    ap.add_argument("--learning_starts", type=int, default=5_000)
    ap.add_argument("--cache_cap", type=int, default=0,
                    help="frozen-oracle memoization cap (states, LRU); 0=off (recurrence too low to help)")
    # optimisation
    ap.add_argument("--lr_drqn", type=float, default=3e-4)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005, help="target EMA coefficient")
    ap.add_argument("--mlm_lambda", type=float, default=0.1)
    ap.add_argument("--mlm_batch", type=int, default=32)
    ap.add_argument("--mlm_workers", type=int, default=2,
                    help="DataLoader workers for the aux-MLM batch prep")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True,
                    help="mixed-precision (fp16) training on CUDA (default on)")
    ap.add_argument("--no_amp", dest="amp", action="store_false",
                    help="disable AMP (fp32 everywhere)")
    ap.add_argument("--enc_clip", type=float, default=0.5)
    ap.add_argument("--drqn_clip", type=float, default=1.0)
    ap.add_argument("--per_alpha", type=float, default=0.6)
    ap.add_argument("--per_beta", type=float, default=0.4)
    ap.add_argument("--pseudo_oov_frac", type=float, default=0.2,
                    help="fraction of episodes drawn from the pseudo-OOV cluster")
    # exploration
    ap.add_argument("--eps_start", type=float, default=0.5)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_frac", type=float, default=0.6)
    # model
    ap.add_argument("--gru", type=int, default=256)
    ap.add_argument("--head_hidden", type=int, default=128)
    ap.add_argument("--finetune_layers", type=int, default=2)
    # eval
    ap.add_argument("--eval_every", type=int, default=25_000, help="episodes between pseudo-OOV evals")
    ap.add_argument("--eval_num", type=int, default=1000)
    ap.add_argument("--final_eval_num", type=int, default=3000)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        (args.episodes, args.warmup_episodes, args.n_envs, args.collect_steps,
         args.train_steps, args.batch_episodes, args.capacity, args.learning_starts,
         args.eval_every, args.eval_num, args.final_eval_num) = (
            500, 120, 8, 8, 2, 8, 5_000, 120, 200, 200, 200)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True   # autotune cuDNN kernels for the BiLSTM experts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.eval_only:
        run_eval_only(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
