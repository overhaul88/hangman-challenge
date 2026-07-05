"""Strategy v3: PPO fine-tuning of the frozen v2 ensemble (approximate policy iteration).

Pieces (all under `src/`):
  - rl_features.TrunkFeatures : frozen v2 trunk -> (l_ref, h, allowed_mask).  [no grad]
  - rl_policy.ResidualPolicy  : zero-init residual actor + value head over the trunk.
  - rl_env.VecHangmanEnv      : parallel belief-MDP rollouts on the TRAIN split.
  - evaluate.evaluate_winrate : held-out simulated-game scoring (the selection metric).

Why this is "never worse than v2":
  1. The residual head is zero-initialized -> the deterministic policy at step 0 is
     argmax(l_ref) == v2, letter-for-letter (verified by rl_features/rl_policy tests).
  2. PPO-clip + a KL penalty to the (softened) v2 reference distribution form a trust
     region that keeps each step a conservative improvement.
  3. Held-out model selection is **seeded with the v2 baseline**: we save a checkpoint
     only when it strictly beats the best-so-far, and best-so-far starts at v2's own
     win-rate (and the initial saved model is the zero-residual policy == v2). So the
     shipped model is >= v2 by construction.

Objective / critic consistency:
  Training uses the **pure terminal reward** (+1 win / 0 loss, shaping off) with gamma=1.0,
  so the discounted return of a state is exactly its eventual win indicator in [0,1].
  The value head is sigmoid-bounded to (0,1) and therefore predicts P(win) consistently.
  (rl_env supports potential-based shaping, but that would need an unbounded critic; it is
  left off here so the trained objective equals the evaluation metric, win-rate.)

Run with the vessel python:
  ~/miniconda3/envs/vessel/bin/python src/train_rl.py --updates 300
  ~/miniconda3/envs/vessel/bin/python src/train_rl.py --smoke      # tiny sanity run
"""
import argparse
import os
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence

from vocab import target_to_letter
from data import load_words, split_words
from evaluate import evaluate_winrate
from rl_features import TrunkFeatures
from rl_policy import ResidualPolicy, make_guess_fn
from rl_env import VecHangmanEnv

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


# ---------------------------------------------------------------- rollout storage
class Rollout:
    """Flat per-step buffers; features are stored (trunk is frozen) so PPO epochs reuse them."""

    def __init__(self):
        self.l_ref, self.h, self.allowed = [], [], []
        self.actions, self.logp, self.values = [], [], []
        self.rewards, self.dones = [], []

    def add(self, l_ref, h, allowed, actions, logp, values, rewards, dones):
        self.l_ref.append(l_ref); self.h.append(h); self.allowed.append(allowed)
        self.actions.append(actions); self.logp.append(logp); self.values.append(values)
        self.rewards.append(rewards); self.dones.append(dones)


@torch.no_grad()
def compute_gae(rewards, values, dones, bootstrap_value, gamma, lam):
    """rewards/values/dones: (T, N) tensors; bootstrap_value: (N,). Returns (adv, ret) (T, N)."""
    T, N = rewards.shape
    adv = torch.zeros(T, N, device=rewards.device)
    last = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        next_value = bootstrap_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--n_envs", type=int, default=64)
    ap.add_argument("--rollout_len", type=int, default=32)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--kl_coef", type=float, default=0.1, help="anchor to softened v2 reference")
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=1.0, help="temperature on l_ref")
    ap.add_argument("--target_kl", type=float, default=0.03, help="early-stop PPO epochs above this")
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--eval_num", type=int, default=1000)
    ap.add_argument("--final_eval_num", type=int, default=3000)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(MODELS_DIR, "hangman_rl.pt"))
    ap.add_argument("--smoke", action="store_true", help="tiny run for a fast sanity check")
    args = ap.parse_args()

    if args.smoke:
        args.updates, args.n_envs, args.rollout_len = 3, 16, 8
        args.eval_every, args.eval_num, args.final_eval_num = 3, 200, 200

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_words, eval_words = split_words(load_words())
    print(f"Train: {len(train_words)}  Eval: {len(eval_words)}")

    trunk = TrunkFeatures(device)
    policy = ResidualPolicy(trunk.feat_dim, hidden=args.hidden, tau=args.tau).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    cfg = dict(feat_dim=trunk.feat_dim, hidden=args.hidden, tau=args.tau)

    # ---- v2 baseline (the fallback) = deterministic zero-residual policy on held-out ----
    policy.eval()
    det_guess = make_guess_fn(policy, trunk, device, deterministic=True)
    t0 = time.time()
    base_wr, base_aw = evaluate_winrate(eval_words, det_guess, num_samples=args.eval_num,
                                        seed=args.eval_seed)
    print(f"[baseline v2] held-out win-rate={base_wr:.4f} avg-wrong={base_aw:.2f} "
          f"({args.eval_num} games, {time.time()-t0:.0f}s)")
    best_wr = base_wr
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Save the zero-residual policy first so the shipped file is >= v2 even if RL never wins.
    torch.save({"model": policy.state_dict(), "config": cfg, "win_rate": best_wr,
                "update": 0, "baseline_v2": base_wr}, args.out)
    print(f"  saved initial (==v2) checkpoint -> {args.out}")

    # ---- PPO ----
    env = VecHangmanEnv(train_words, args.n_envs, shaping="none", gamma=args.gamma,
                        seed=args.seed)
    obs = env.reset()
    win_hist = deque(maxlen=2000)  # recent completed-episode outcomes for logging

    for update in range(1, args.updates + 1):
        policy.eval()
        roll = Rollout()
        # ---------------- collect a rollout ----------------
        for _ in range(args.rollout_len):
            states = [o.state for o in obs]
            l_ref, h, allowed = trunk.compute(states)
            with torch.no_grad():
                action, logp, value, _ = policy.act(l_ref, h, allowed, deterministic=False)
            letters = [target_to_letter(a) for a in action.tolist()]
            obs, rewards, dones, infos = env.step(letters)
            for info in infos:
                if info.get("terminal"):
                    win_hist.append(1.0 if info["win"] else 0.0)
            roll.add(l_ref, h, allowed, action, logp, value,
                     torch.tensor(rewards, device=device, dtype=torch.float32),
                     torch.tensor(dones, device=device, dtype=torch.float32))

        # bootstrap value of the final observation
        with torch.no_grad():
            l_ref, h, allowed = trunk.compute([o.state for o in obs])
            _, boot_value = policy.forward(l_ref, h, allowed)

        rewards = torch.stack(roll.rewards)            # (T, N)
        values = torch.stack(roll.values)             # (T, N)
        dones = torch.stack(roll.dones)               # (T, N)
        adv, ret = compute_gae(rewards, values, dones, boot_value, args.gamma, args.gae_lambda)

        # flatten
        b_l_ref = torch.cat(roll.l_ref)               # (T*N, 26)
        b_h = torch.cat(roll.h)                        # (T*N, D)
        b_allowed = torch.cat(roll.allowed)           # (T*N, 26)
        b_actions = torch.cat(roll.actions)           # (T*N,)
        b_logp = torch.cat(roll.logp)                 # (T*N,)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # ---------------- PPO update ----------------
        policy.train()
        n = b_l_ref.shape[0]
        idx = torch.randperm(n, device=device)
        stop = False
        last_stats = {}
        for _ in range(args.ppo_epochs):
            if stop:
                break
            for s in range(0, n, args.minibatch):
                mb = idx[s:s + args.minibatch]
                logits, value = policy.forward(b_l_ref[mb], b_h[mb], b_allowed[mb])
                ref_logits = policy.reference_logits(b_l_ref[mb], b_allowed[mb])
                dist = Categorical(logits=logits)
                ref = Categorical(logits=ref_logits)
                new_logp = dist.log_prob(b_actions[mb])
                entropy = dist.entropy().mean()
                kl_ref = kl_divergence(dist, ref).mean()

                logratio = new_logp - b_logp[mb]
                ratio = logratio.exp()
                a = b_adv[mb]
                pg = -torch.min(ratio * a,
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                v_loss = F.mse_loss(value, b_ret[mb])
                loss = pg + args.vf_coef * v_loss - args.ent_coef * entropy + args.kl_coef * kl_ref

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    approx_kl = (-logratio).mean().item()  # KL(old||new) estimator
                last_stats = dict(pg=pg.item(), v=v_loss.item(), ent=entropy.item(),
                                  kl_ref=kl_ref.item(), approx_kl=approx_kl)
            if last_stats.get("approx_kl", 0.0) > args.target_kl:
                stop = True  # trust region: stop tightening this batch

        recent_wr = sum(win_hist) / len(win_hist) if win_hist else float("nan")
        msg = (f"upd {update:3d}/{args.updates} | rollout_wr={recent_wr:.3f} "
               f"pg={last_stats['pg']:+.3f} v={last_stats['v']:.3f} ent={last_stats['ent']:.3f} "
               f"kl_ref={last_stats['kl_ref']:.4f} akl={last_stats['approx_kl']:.4f}")

        # ---------------- periodic held-out selection ----------------
        if update % args.eval_every == 0 or update == args.updates:
            policy.eval()
            det_guess = make_guess_fn(policy, trunk, device, deterministic=True)
            wr, aw = evaluate_winrate(eval_words, det_guess, num_samples=args.eval_num,
                                      seed=args.eval_seed)
            msg += f" || held-out wr={wr:.4f} aw={aw:.2f} (best={best_wr:.4f})"
            if wr > best_wr:
                best_wr = wr
                torch.save({"model": policy.state_dict(), "config": cfg, "win_rate": wr,
                            "update": update, "baseline_v2": base_wr}, args.out)
                msg += "  -> saved best"
        print(msg, flush=True)

    # ---------------- final apples-to-apples eval (v2 vs best RL) ----------------
    print("\n=== Final held-out evaluation ===")
    ckpt = torch.load(args.out, map_location=device)
    best = ResidualPolicy(**ckpt["config"]).to(device)
    best.load_state_dict(ckpt["model"])
    best.eval()
    fresh = ResidualPolicy(trunk.feat_dim, hidden=args.hidden, tau=args.tau).to(device)
    fresh.eval()  # zero-residual == v2

    v2_wr, v2_aw = evaluate_winrate(eval_words, make_guess_fn(fresh, trunk, device),
                                    num_samples=args.final_eval_num, seed=args.eval_seed)
    rl_wr, rl_aw = evaluate_winrate(eval_words, make_guess_fn(best, trunk, device),
                                    num_samples=args.final_eval_num, seed=args.eval_seed)
    print(f"v2 (zero residual): win-rate={v2_wr:.4f} avg-wrong={v2_aw:.2f}")
    print(f"v3 (RL best @upd {ckpt['update']}): win-rate={rl_wr:.4f} avg-wrong={rl_aw:.2f}")
    print(f"delta = {rl_wr - v2_wr:+.4f}  ({args.final_eval_num} paired games, seed {args.eval_seed})")

    # Authoritative safe-improvement gate: ship the RL policy only if it strictly beats v2 on
    # the full paired eval; otherwise resave the zero-residual policy (== v2 exactly). This makes
    # the shipped checkpoint provably >= v2 on the reported metric, not just on the noisy
    # training-time selection eval.
    if rl_wr > v2_wr:
        print(f"Shipped RL policy (>{v2_wr:.4f}). Saved -> {args.out}")
    else:
        torch.save({"model": fresh.state_dict(), "config": cfg, "win_rate": v2_wr,
                    "update": 0, "baseline_v2": base_wr, "fallback": True}, args.out)
        print(f"RL did not beat v2 on the full eval; fell back to zero-residual (==v2). "
              f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
