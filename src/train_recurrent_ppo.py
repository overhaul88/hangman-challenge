"""Strategy v4: Recurrent PPO + Curriculum Learning on the frozen v2 ensemble.

Dependency-free Recurrent PPO (CleanRL `ppo_lstm` recurrence pattern) — chosen over
SB3-contrib because the residual skip-connection, finite action masking, and KL-to-v2 anchor
are exact and fully controlled here, and gymnasium/SB3 are not installed in this env.

Pieces (all in `src/`):
  - rl_features.TrunkFeatures   : frozen v2 trunk -> (l_ref, h, allowed_mask).        [no grad]
  - policy_recurrent.RecurrentResidualPolicy : LSTM + zero-init residual over l_ref + value head.
  - env_v4.VecCurriculumEnv     : parallel curriculum belief-MDP rollouts (train split).
  - curriculum.Curriculum       : phase manager (length / budget / hidden-fraction).

Guarantees carried from v3:
  * Zero-init residual -> the deterministic recurrent policy == v2 at init (skip invariant).
  * KL anchor to the softened v2 reference + held-out safe selection seeded with v2 +
    authoritative paired final gate with fallback to the exact zero-residual policy.
  * Finite action masking (no -inf nan).

Objective: pure terminal reward (+1 win / 0 loss), gamma=1.0, so the sigmoid P(win) critic is
consistent with the returns; the curriculum supplies the dense early-win signal. Held-out eval
always uses the deployment rule (all-blank openings, 6 wrong allowed).

Run:
  ~/miniconda3/envs/vessel/bin/python src/train_recurrent_ppo.py --updates 240
  ~/miniconda3/envs/vessel/bin/python src/train_recurrent_ppo.py --smoke
"""
import argparse
import os
import time
from collections import deque

import torch
import torch.nn.functional as F

from vocab import target_to_letter, MAX_WRONG
from data import load_words, split_words
from rl_features import TrunkFeatures
from policy_recurrent import RecurrentResidualPolicy
from env_v4 import VecCurriculumEnv
from curriculum import Curriculum

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


# ----------------------------------------------------------------------------
# Batched recurrent held-out evaluation (deterministic) — deployment rules.
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_recurrent(policy, trunk, words, device, num_samples, seed=0,
                       batch=256, max_wrong=MAX_WRONG, max_steps=30):
    import random
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
            action, _, _, _, lstm_state = policy.act_step(
                l_ref, h, allowed, lstm_state, reset, deterministic=True)
            reset = torch.zeros(B, device=device)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=240)
    ap.add_argument("--n_envs", type=int, default=64)
    ap.add_argument("--rollout_len", type=int, default=32)
    ap.add_argument("--update_epochs", type=int, default=4)
    ap.add_argument("--num_minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--kl_coef", type=float, default=0.1)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--target_kl", type=float, default=0.03)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--shaping", default="none", choices=["none", "progress"])
    ap.add_argument("--eval_every", type=int, default=30)
    ap.add_argument("--eval_num", type=int, default=1000)
    ap.add_argument("--final_eval_num", type=int, default=3000)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curriculum_state", default=None)
    ap.add_argument("--out", default=os.path.join(MODELS_DIR, "hangman_recurrent.pt"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.updates, args.n_envs, args.rollout_len = 12, 16, 8
        args.eval_every, args.eval_num, args.final_eval_num = 6, 200, 200

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_words, eval_words = split_words(load_words())
    print(f"Train: {len(train_words)}  Eval: {len(eval_words)}")

    trunk = TrunkFeatures(device)
    policy = RecurrentResidualPolicy(trunk.feat_dim, lstm_hidden=args.lstm_hidden,
                                     tau=args.tau).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    cfg = dict(feat_dim=trunk.feat_dim, lstm_hidden=args.lstm_hidden, tau=args.tau)

    # ---- baseline: zero-init recurrent policy == v2 (skip invariant + the fallback) ----
    policy.eval()
    t0 = time.time()
    base_wr, base_aw = evaluate_recurrent(policy, trunk, eval_words, device,
                                          args.eval_num, seed=args.eval_seed)
    print(f"[baseline v2 == zero-residual recurrent] win-rate={base_wr:.4f} "
          f"avg-wrong={base_aw:.2f} ({args.eval_num} games, {time.time()-t0:.0f}s)")
    best_wr = base_wr
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"model": policy.state_dict(), "config": cfg, "win_rate": best_wr,
                "update": 0, "baseline_v2": base_wr}, args.out)

    # ---- curriculum + envs ----
    curriculum = Curriculum(state_file=args.curriculum_state)
    env = VecCurriculumEnv(train_words, args.n_envs, curriculum, shaping=args.shaping,
                           gamma=args.gamma, seed=args.seed)
    phase_every = max(1, args.updates // curriculum.num_phases)

    N, T = args.n_envs, args.rollout_len
    envsperbatch = max(1, N // args.num_minibatches)

    obs = env.reset()
    next_done = torch.ones(N, device=device)            # episodes start fresh
    lstm_state = policy.initial_state(N, device)
    win_hist = deque(maxlen=2000)

    for update in range(1, args.updates + 1):
        # curriculum advancement (timestep/update-based; ends on the deployment phase)
        if update > 1 and (update - 1) % phase_every == 0:
            if curriculum.advance_phase():
                print(f"  -> curriculum phase {curriculum.current_phase}: "
                      f"{curriculum.get_current_config()}")

        # ---------- rollout ----------
        L = torch.empty(T, N, trunk.n_letters, device=device)
        H = torch.empty(T, N, trunk.feat_dim, device=device)
        AL = torch.empty(T, N, trunk.n_letters, dtype=torch.bool, device=device)
        ACT = torch.empty(T, N, dtype=torch.long, device=device)
        LOGP = torch.empty(T, N, device=device)
        VAL = torch.empty(T, N, device=device)
        REW = torch.empty(T, N, device=device)
        DON = torch.empty(T, N, device=device)
        initial_lstm_state = (lstm_state[0].clone(), lstm_state[1].clone())

        policy.eval()
        for t in range(T):
            DON[t] = next_done
            states = [o.state for o in obs]
            l_ref, h, allowed = trunk.compute(states)
            with torch.no_grad():
                action, logp, value, _, lstm_state = policy.act_step(
                    l_ref, h, allowed, lstm_state, next_done, deterministic=False)
            L[t], H[t], AL[t] = l_ref, h, allowed
            ACT[t], LOGP[t], VAL[t] = action, logp, value
            letters = [target_to_letter(a) for a in action.tolist()]
            obs, rewards, dones, infos = env.step(letters)
            for info in infos:
                if info.get("terminal"):
                    win_hist.append(1.0 if info["win"] else 0.0)
            REW[t] = torch.tensor(rewards, device=device)
            next_done = torch.tensor(dones, dtype=torch.float32, device=device)

        # bootstrap value (does not mutate carried lstm_state)
        with torch.no_grad():
            states = [o.state for o in obs]
            l_ref, h, allowed = trunk.compute(states)
            _, _, next_value, _, _ = policy.act_step(
                l_ref, h, allowed, lstm_state, next_done, deterministic=False)
            adv = torch.zeros(T, N, device=device)
            last = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterm, nextval = 1.0 - next_done, next_value
                else:
                    nextnonterm, nextval = 1.0 - DON[t + 1], VAL[t + 1]
                delta = REW[t] + args.gamma * nextval * nextnonterm - VAL[t]
                last = delta + args.gamma * args.gae_lambda * nextnonterm * last
                adv[t] = last
            ret = adv + VAL

        # ---------- PPO update (per-env minibatch, BPTT over T) ----------
        policy.train()
        envinds = torch.randperm(N, device=device)
        stop = False
        stats = {}
        for _ in range(args.update_epochs):
            if stop:
                break
            envinds = envinds[torch.randperm(N, device=device)]
            for s in range(0, N, envsperbatch):
                mb = envinds[s:s + envsperbatch]
                logp, value, ent, kl = policy.evaluate_sequence(
                    L[:, mb], H[:, mb], AL[:, mb], ACT[:, mb], DON[:, mb],
                    (initial_lstm_state[0][:, mb], initial_lstm_state[1][:, mb]))
                newlogp = logp.reshape(-1)
                a = adv[:, mb].reshape(-1)
                a = (a - a.mean()) / (a.std() + 1e-8)
                logratio = newlogp - LOGP[:, mb].reshape(-1)
                ratio = logratio.exp()
                pg = -torch.min(ratio * a,
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                v_loss = F.mse_loss(value.reshape(-1), ret[:, mb].reshape(-1))
                ent_loss = ent.mean()
                kl_loss = kl.mean()
                loss = pg + args.vf_coef * v_loss - args.ent_coef * ent_loss + args.kl_coef * kl_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    approx_kl = (-logratio).mean().item()
                stats = dict(pg=pg.item(), v=v_loss.item(), ent=ent_loss.item(),
                             kl_ref=kl_loss.item(), akl=approx_kl)
            if stats.get("akl", 0.0) > args.target_kl:
                stop = True

        recent = sum(win_hist) / len(win_hist) if win_hist else float("nan")
        msg = (f"upd {update:3d}/{args.updates} ph{curriculum.current_phase} | "
               f"roll_wr={recent:.3f} pg={stats['pg']:+.3f} v={stats['v']:.3f} "
               f"ent={stats['ent']:.3f} kl_ref={stats['kl_ref']:.4f} akl={stats['akl']:.4f}")

        if update % args.eval_every == 0 or update == args.updates:
            policy.eval()
            wr, aw = evaluate_recurrent(policy, trunk, eval_words, device,
                                        args.eval_num, seed=args.eval_seed)
            msg += f" || held-out wr={wr:.4f} aw={aw:.2f} (best={best_wr:.4f})"
            if wr > best_wr:
                best_wr = wr
                torch.save({"model": policy.state_dict(), "config": cfg, "win_rate": wr,
                            "update": update, "baseline_v2": base_wr}, args.out)
                msg += "  -> saved best"
        print(msg, flush=True)

    # ---------- authoritative paired final eval (v2 vs v4) ----------
    print("\n=== Final held-out evaluation (paired, deployment rules) ===")
    ckpt = torch.load(args.out, map_location=device)
    best = RecurrentResidualPolicy(**ckpt["config"]).to(device)
    best.load_state_dict(ckpt["model"]); best.eval()
    fresh = RecurrentResidualPolicy(trunk.feat_dim, lstm_hidden=args.lstm_hidden,
                                    tau=args.tau).to(device)
    fresh.eval()  # zero-residual == v2

    v2_wr, v2_aw = evaluate_recurrent(fresh, trunk, eval_words, device,
                                      args.final_eval_num, seed=args.eval_seed)
    rl_wr, rl_aw = evaluate_recurrent(best, trunk, eval_words, device,
                                      args.final_eval_num, seed=args.eval_seed)
    print(f"v2 (zero residual): win-rate={v2_wr:.4f} avg-wrong={v2_aw:.2f}")
    print(f"v4 (recurrent best @upd {ckpt['update']}): win-rate={rl_wr:.4f} avg-wrong={rl_aw:.2f}")
    print(f"delta = {rl_wr - v2_wr:+.4f}  ({args.final_eval_num} paired games, seed {args.eval_seed})")
    if rl_wr > v2_wr:
        print(f"Shipped recurrent policy (> {v2_wr:.4f}). Saved -> {args.out}")
    else:
        torch.save({"model": fresh.state_dict(), "config": cfg, "win_rate": v2_wr,
                    "update": 0, "baseline_v2": base_wr, "fallback": True}, args.out)
        print(f"Did not beat v2 on the full eval; fell back to zero-residual (==v2). "
              f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
