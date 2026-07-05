"""Strategy v5: off-policy Dueling Residual DRQN + Curriculum on the frozen v2 ensemble.

A value-based pivot from the on-policy PPO of v3/v4. The frozen Strategy-v2 trunk supplies the
belief features (`l_ref`, `h`, `allowed`); a Dueling Residual DRQN learns the action-value
`Q(o,a) ~= P(win | guess a, then play on)` off-policy from an episode replay buffer. Using RL
strictly for decision theory / risk management (not to re-learn the language) is what value-based
+ replay buys us over policy gradients (whose advantage variance on this OOV POMDP is chaotic).

Pieces (all in `src/`):
  - rl_features.TrunkFeatures        : frozen v2 trunk -> (l_ref, h, allowed_mask).        [no grad]
  - policy_drqn.DuelingResidualDRQN  : LSTM + dueling head + zero-init residual over l_ref.
  - replay.EpisodeReplayBuffer       : whole-game replay storing trunk features (off-policy reuse).
  - env_v4.VecCurriculumEnv          : parallel curriculum belief-MDP rollouts (train split).
  - curriculum.Curriculum            : phase manager (length / budget / hidden-fraction).

Algorithm: Double-DQN target + dueling Q + Huber loss + Polyak target net + prior-guided
epsilon-greedy exploration. Pure terminal reward (+1 win / 0 loss), gamma=1.0, so Q is literally
P(win). Curriculum front-loads wins to give the value bootstrap a dense early signal.

Safety carried from v3/v4 (the ">= v2" guarantee):
  * Zero-init residual -> the deterministic greedy policy == v2 at init (skip invariant).
  * Value-based conservatism (frozen trunk + g_head weight-decay + slow Polyak target +
    Double-DQN) replaces the PPO KL trust region.
  * Held-out safe selection seeded with v2 + an authoritative paired final gate that ships v5
    only if it strictly beats v2, else falls back to the exact zero-residual policy (== v2).

Held-out eval ALWAYS uses the deployment rule (all-blank openings, 6 wrong allowed), so v5's
numbers are directly comparable to v2 (0.635) / v3 / v4.

Run (training):
  ~/miniconda3/envs/vessel/bin/python src/train_drqn.py --updates 300
Run (evaluation only — paired v2 vs v5 on the held-out split):
  ~/miniconda3/envs/vessel/bin/python src/train_drqn.py --eval_only
Smoke (tiny wiring check; NOT a real run):
  ~/miniconda3/envs/vessel/bin/python src/train_drqn.py --smoke
"""
import argparse
import os
import random
import time
from collections import deque

import torch
import torch.nn.functional as F

from vocab import target_to_letter, MAX_WRONG
from data import load_words, split_words
from rl_features import TrunkFeatures
from rl_policy import NEG_MASK
from policy_drqn import DuelingResidualDRQN
from replay import EpisodeReplayBuffer
from env_v4 import VecCurriculumEnv
from curriculum import Curriculum

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


# ----------------------------------------------------------------------------
# Batched greedy held-out evaluation (deployment rules) — threads LSTM state per game.
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_drqn(policy, trunk, words, device, num_samples, seed=0,
                  batch=256, max_wrong=MAX_WRONG, max_steps=30):
    rng = random.Random(seed)
    sample = rng.sample(words, min(num_samples, len(words)))
    wins = 0
    total_wrong = 0
    for i in range(0, len(sample), batch):
        chunk = sample[i:i + batch]
        B = len(chunk)
        boards = [["_"] * len(w) for w in chunk]
        guessed = [set() for _ in chunk]
        wrong = [0] * B
        done = [False] * B
        lstm_state = policy.initial_state(B, device)
        reset = torch.ones(B, device=device)  # reset state at game start
        steps = 0
        while not all(done) and steps < max_steps:
            states = [(boards[b], guessed[b]) for b in range(B)]
            l_ref, h, allowed = trunk.compute(states)
            q, lstm_state = policy.q_step(l_ref, h, allowed, lstm_state, reset)
            reset = torch.zeros(B, device=device)
            action = q.argmax(-1)
            letters = [target_to_letter(a) for a in action.tolist()]
            for b in range(B):
                if done[b]:
                    continue
                g = letters[b]
                guessed[b].add(g)
                w = chunk[b]
                if g in w:
                    for j, c in enumerate(w):
                        if c == g:
                            boards[b][j] = g
                    if "_" not in boards[b]:
                        done[b] = True
                else:
                    wrong[b] += 1
                    if wrong[b] >= max_wrong:
                        done[b] = True
            steps += 1
        for b in range(B):
            if "_" not in boards[b]:
                wins += 1
            total_wrong += wrong[b]
    n = len(sample)
    return wins / n, total_wrong / n


# ----------------------------------------------------------------------------
# Double-DQN dueling Huber loss over a replayed batch of padded episodes.
# ----------------------------------------------------------------------------
def drqn_loss(online, target, batch, gamma, huber_beta):
    L, H, AL = batch["l_ref"], batch["h"], batch["allowed"]
    A, R, D, M, RS = (batch["action"], batch["reward"], batch["done"],
                      batch["mask"], batch["reset"])
    Tm, B = A.shape
    dev = L.device

    q_online, _ = online.q_sequence(L, H, AL, RS, online.initial_state(B, dev))   # (Tm,B,26) grad
    with torch.no_grad():
        q_target, _ = target.q_sequence(L, H, AL, RS, target.initial_state(B, dev))

    q_taken = q_online.gather(-1, A.unsqueeze(-1)).squeeze(-1)                      # (Tm,B)

    with torch.no_grad():
        # next-state value at position t+1: Double-DQN (online selects, target evaluates).
        boot_all = torch.zeros(Tm, B, device=dev)
        if Tm > 1:
            a_star = q_online.detach()[1:].argmax(-1)                              # (Tm-1,B)
            boot = q_target[1:].gather(-1, a_star.unsqueeze(-1)).squeeze(-1)       # (Tm-1,B)
            boot_all[:Tm - 1] = boot
        # done (incl. all padding rows, which are done=1) zeroes the bootstrap.
        y = R + gamma * (1.0 - D) * boot_all

    td = F.smooth_l1_loss(q_taken, y, reduction="none", beta=huber_beta)
    denom = M.sum().clamp(min=1.0)
    loss = (td * M).sum() / denom
    stats = {"loss": loss.item(),
             "q": (q_taken * M).sum().item() / denom.item(),
             "y": (y * M).sum().item() / denom.item()}
    return loss, stats


def epsilon_at(update, updates, eps_start, eps_end, decay_frac):
    span = max(1, int(decay_frac * updates))
    frac = min(1.0, (update - 1) / span)
    return eps_start + frac * (eps_end - eps_start)


def _new_ongoing(n):
    return [{"l_ref": [], "h": [], "allowed": [], "action": [], "reward": [], "done": []}
            for _ in range(n)]


def _finalize(ep, buffer):
    """Stack one env's per-step lists into tensors and push to the replay buffer."""
    if not ep["action"]:
        return
    buffer.add_episode(
        l_ref=torch.stack(ep["l_ref"]),
        h=torch.stack(ep["h"]),
        allowed=torch.stack(ep["allowed"]),
        action=torch.tensor(ep["action"], dtype=torch.long),
        reward=torch.tensor(ep["reward"], dtype=torch.float32),
        done=torch.tensor(ep["done"], dtype=torch.float32),
    )


def build_config(args, feat_dim):
    return dict(feat_dim=feat_dim, proj=args.proj, lstm_hidden=args.lstm_hidden,
                n_lstm_layers=args.n_lstm_layers, head_hidden=args.head_hidden, tau=args.tau)


# ----------------------------------------------------------------------------
def run_eval_only(args, device):
    """Evaluation endpoint: paired held-out comparison of the shipped v5 vs v2 (zero residual)."""
    _, eval_words = split_words(load_words())
    trunk = TrunkFeatures(device)
    print(f"Eval words: {len(eval_words)} | paired {args.final_eval_num} games, seed {args.eval_seed}")

    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        cfg = ckpt["config"]
        best = DuelingResidualDRQN(**cfg).to(device)
        best.load_state_dict(ckpt["model"])
        best.eval()
        fresh = DuelingResidualDRQN(**cfg).to(device)   # zero-residual == v2
        fresh.eval()
        v2_wr, v2_aw = evaluate_drqn(fresh, trunk, eval_words, device, args.final_eval_num,
                                     seed=args.eval_seed)
        rl_wr, rl_aw = evaluate_drqn(best, trunk, eval_words, device, args.final_eval_num,
                                     seed=args.eval_seed)
        tag = "fallback(==v2)" if ckpt.get("fallback") else f"best@upd{ckpt.get('update')}"
        print(f"v2 (zero residual): win-rate={v2_wr:.4f} avg-wrong={v2_aw:.2f}")
        print(f"v5 DRQN [{tag}]:     win-rate={rl_wr:.4f} avg-wrong={rl_aw:.2f}")
        print(f"delta = {rl_wr - v2_wr:+.4f}")
    else:
        # No trained checkpoint yet: report the zero-residual (== v2) baseline as a sanity check.
        fresh = DuelingResidualDRQN(**build_config(args, trunk.feat_dim)).to(device)
        fresh.eval()
        wr, aw = evaluate_drqn(fresh, trunk, eval_words, device, args.final_eval_num,
                               seed=args.eval_seed)
        print(f"[no checkpoint at {args.ckpt}] zero-residual DRQN (== v2): "
              f"win-rate={wr:.4f} avg-wrong={aw:.2f}")


def main():
    ap = argparse.ArgumentParser()
    # endpoints
    ap.add_argument("--eval_only", action="store_true",
                    help="evaluation endpoint: paired held-out v2-vs-v5 on --ckpt, no training")
    ap.add_argument("--ckpt", default=os.path.join(MODELS_DIR, "hangman_drqn.pt"))
    ap.add_argument("--out", default=os.path.join(MODELS_DIR, "hangman_drqn.pt"))
    ap.add_argument("--smoke", action="store_true")
    # training schedule
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--n_envs", type=int, default=64)
    ap.add_argument("--collect_steps", type=int, default=32, help="env-steps collected per update")
    ap.add_argument("--train_steps", type=int, default=8, help="grad updates per update")
    ap.add_argument("--batch_episodes", type=int, default=64)
    ap.add_argument("--capacity", type=int, default=200_000, help="replay capacity (transitions)")
    ap.add_argument("--learning_starts", type=int, default=5_000)
    # optimisation
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="L2 on the g_head residual only")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--huber_beta", type=float, default=1.0)
    ap.add_argument("--polyak", type=float, default=0.995, help="target net EMA coefficient")
    ap.add_argument("--max_grad_norm", type=float, default=10.0)
    # exploration
    ap.add_argument("--eps_start", type=float, default=0.5)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_frac", type=float, default=0.6)
    # model
    ap.add_argument("--proj", type=int, default=256)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--n_lstm_layers", type=int, default=1)
    ap.add_argument("--head_hidden", type=int, default=256)
    ap.add_argument("--tau", type=float, default=1.0)
    # env / curriculum
    ap.add_argument("--shaping", default="none", choices=["none", "progress"])
    ap.add_argument("--curriculum_state", default=None)
    # eval
    ap.add_argument("--eval_every", type=int, default=30)
    ap.add_argument("--eval_num", type=int, default=1000)
    ap.add_argument("--final_eval_num", type=int, default=3000)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        args.updates, args.n_envs, args.collect_steps = 12, 16, 8
        args.batch_episodes, args.learning_starts, args.train_steps = 8, 200, 4
        args.eval_every, args.eval_num, args.final_eval_num = 6, 200, 200

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.eval_only:
        run_eval_only(args, device)
        return

    train_words, eval_words = split_words(load_words())
    print(f"Train: {len(train_words)}  Eval: {len(eval_words)}")

    trunk = TrunkFeatures(device)
    cfg = build_config(args, trunk.feat_dim)
    online = DuelingResidualDRQN(**cfg).to(device)
    target = DuelingResidualDRQN(**cfg).to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)

    # Weight-decay only on the residual head g_head (keeps the v2 correction small / general).
    g_ids = {id(p) for p in online.g_head.parameters()}
    opt = torch.optim.AdamW([
        {"params": list(online.g_head.parameters()), "weight_decay": args.weight_decay},
        {"params": [p for p in online.parameters() if id(p) not in g_ids], "weight_decay": 0.0},
    ], lr=args.lr)

    # ---- baseline: zero-init DRQN == v2 (skip invariant + the fallback) ----
    online.eval()
    t0 = time.time()
    base_wr, base_aw = evaluate_drqn(online, trunk, eval_words, device,
                                     args.eval_num, seed=args.eval_seed)
    print(f"[baseline v2 == zero-residual DRQN] win-rate={base_wr:.4f} "
          f"avg-wrong={base_aw:.2f} ({args.eval_num} games, {time.time()-t0:.0f}s)")
    best_wr = base_wr
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"model": online.state_dict(), "config": cfg, "win_rate": best_wr,
                "update": 0, "baseline_v2": base_wr}, args.out)

    # ---- curriculum + envs + replay ----
    curriculum = Curriculum(state_file=args.curriculum_state)
    env = VecCurriculumEnv(train_words, args.n_envs, curriculum, shaping=args.shaping,
                           gamma=args.gamma, seed=args.seed)
    buffer = EpisodeReplayBuffer(args.capacity, trunk.feat_dim, n_letters=trunk.n_letters)
    phase_every = max(1, args.updates // curriculum.num_phases)

    N = args.n_envs
    obs = env.reset()
    next_done = torch.ones(N, device=device)             # episodes start fresh
    lstm_state = online.initial_state(N, device)
    ongoing = _new_ongoing(N)
    win_hist = deque(maxlen=2000)
    grad_steps = 0

    for update in range(1, args.updates + 1):
        # curriculum advancement (update-based; ends on the all-blank deployment phase)
        if update > 1 and (update - 1) % phase_every == 0:
            if curriculum.advance_phase():
                print(f"  -> curriculum phase {curriculum.current_phase}: "
                      f"{curriculum.get_current_config()}")

        eps = epsilon_at(update, args.updates, args.eps_start, args.eps_end, args.eps_decay_frac)

        # ---------- collection (prior-guided epsilon-greedy, threaded LSTM state) ----------
        online.eval()
        for _ in range(args.collect_steps):
            states = [o.state for o in obs]
            l_ref, h, allowed = trunk.compute(states)
            with torch.no_grad():
                q, lstm_state = online.q_step(l_ref, h, allowed, lstm_state, next_done)
            greedy = q.argmax(-1)
            explore = torch.rand(N, device=device) < eps
            prior_logits = l_ref.masked_fill(~allowed, NEG_MASK)
            prior_sample = torch.distributions.Categorical(logits=prior_logits).sample()
            action = torch.where(explore, prior_sample, greedy)            # (N,)

            letters = [target_to_letter(a) for a in action.tolist()]
            obs, rewards, dones, infos = env.step(letters)

            l_ref_c = l_ref.detach().cpu()
            h_c = h.detach().cpu()
            al_c = allowed.detach().cpu()
            act_c = action.detach().cpu()
            for i in range(N):
                ep = ongoing[i]
                ep["l_ref"].append(l_ref_c[i])
                ep["h"].append(h_c[i])
                ep["allowed"].append(al_c[i])
                ep["action"].append(int(act_c[i]))
                ep["reward"].append(float(rewards[i]))
                ep["done"].append(1.0 if dones[i] else 0.0)
                if dones[i]:
                    _finalize(ep, buffer)
                    win_hist.append(1.0 if infos[i]["win"] else 0.0)
                    ongoing[i] = {"l_ref": [], "h": [], "allowed": [], "action": [],
                                  "reward": [], "done": []}
            next_done = torch.tensor([1.0 if d else 0.0 for d in dones], device=device)

        # ---------- learning (off-policy: Double-DQN dueling Huber, Polyak target) ----------
        stats = {"loss": float("nan"), "q": float("nan"), "y": float("nan")}
        if len(buffer) >= args.learning_starts and buffer.can_sample(args.batch_episodes):
            online.train()
            for _ in range(args.train_steps):
                batch = buffer.sample(args.batch_episodes, device)
                loss, stats = drqn_loss(online, target, batch, args.gamma, args.huber_beta)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), args.max_grad_norm)
                opt.step()
                grad_steps += 1
                with torch.no_grad():
                    for tp, op in zip(target.parameters(), online.parameters()):
                        tp.data.mul_(args.polyak).add_(op.data, alpha=1.0 - args.polyak)

        recent = sum(win_hist) / len(win_hist) if win_hist else float("nan")
        msg = (f"upd {update:3d}/{args.updates} ph{curriculum.current_phase} eps={eps:.2f} | "
               f"roll_wr={recent:.3f} buf={len(buffer)} gsteps={grad_steps} "
               f"loss={stats['loss']:.4f} q={stats['q']:.3f} y={stats['y']:.3f}")

        if update % args.eval_every == 0 or update == args.updates:
            online.eval()
            wr, aw = evaluate_drqn(online, trunk, eval_words, device,
                                   args.eval_num, seed=args.eval_seed)
            msg += f" || held-out wr={wr:.4f} aw={aw:.2f} (best={best_wr:.4f})"
            if wr > best_wr:
                best_wr = wr
                torch.save({"model": online.state_dict(), "config": cfg, "win_rate": wr,
                            "update": update, "baseline_v2": base_wr}, args.out)
                msg += "  -> saved best"
        print(msg, flush=True)

    # ---------- authoritative paired final gate (v2 vs v5) ----------
    print("\n=== Final held-out evaluation (paired, deployment rules) ===")
    ckpt = torch.load(args.out, map_location=device)
    best = DuelingResidualDRQN(**ckpt["config"]).to(device)
    best.load_state_dict(ckpt["model"]); best.eval()
    fresh = DuelingResidualDRQN(**cfg).to(device); fresh.eval()  # zero-residual == v2

    v2_wr, v2_aw = evaluate_drqn(fresh, trunk, eval_words, device,
                                 args.final_eval_num, seed=args.eval_seed)
    rl_wr, rl_aw = evaluate_drqn(best, trunk, eval_words, device,
                                 args.final_eval_num, seed=args.eval_seed)
    print(f"v2 (zero residual): win-rate={v2_wr:.4f} avg-wrong={v2_aw:.2f}")
    print(f"v5 (DRQN best @upd {ckpt['update']}): win-rate={rl_wr:.4f} avg-wrong={rl_aw:.2f}")
    print(f"delta = {rl_wr - v2_wr:+.4f}  ({args.final_eval_num} paired games, seed {args.eval_seed})")
    if rl_wr > v2_wr:
        print(f"Shipped DRQN policy (> {v2_wr:.4f}). Saved -> {args.out}")
    else:
        torch.save({"model": fresh.state_dict(), "config": cfg, "win_rate": v2_wr,
                    "update": 0, "baseline_v2": base_wr, "fallback": True}, args.out)
        print(f"Did not beat v2 on the full eval; fell back to zero-residual (==v2). "
              f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
