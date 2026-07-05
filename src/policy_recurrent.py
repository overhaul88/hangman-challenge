"""Recurrent residual actor-critic for Strategy v4.

Transfers the Recurrent-PPO idea (an LSTM in the policy, so the agent carries memory across
the within-game guess sequence) while preserving the Strategy-v3 safety mechanisms:

  * **Residual skip-connection (≥ v2).** Action logits = `l_ref/τ + g_θ(lstm_out)` with the
    residual head `g_θ` zero-initialised. At init `g_θ ≡ 0`, so the deterministic policy is
    `argmax(l_ref)` == v2 *for any LSTM weights/state*. The LSTM only feeds `g_θ` (and the
    value head), so a fresh recurrent policy reproduces v2 exactly.
  * **Finite action masking.** Disallowed (already-guessed) letters are masked with a large
    finite negative (`rl_policy.NEG_MASK`), never `-inf` (which makes entropy()/KL backward
    nan).

LSTM handling follows the standard CleanRL `ppo_lstm` pattern: `get_states` runs the LSTM
over a time sequence and zeroes the recurrent state at episode boundaries via a per-step
reset mask (`reset[t] == 1` means "obs at t starts a fresh episode → reset before step t").

Inputs are the frozen-trunk features from `rl_features.TrunkFeatures`:
    l_ref (.,26), h (.,feat_dim), allowed_mask (.,26) bool.
"""
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical, kl_divergence

from rl_policy import NEG_MASK

State = Tuple[torch.Tensor, torch.Tensor]  # (h, c), each (n_layers, B, hidden)


class RecurrentResidualPolicy(nn.Module):
    def __init__(self, feat_dim: int, n_letters: int = 26, proj: int = 256,
                 lstm_hidden: int = 128, n_lstm_layers: int = 1, head_hidden: int = 256,
                 tau: float = 1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_letters = n_letters
        self.lstm_hidden = lstm_hidden
        self.n_lstm_layers = n_lstm_layers
        self.tau = tau

        self.proj = nn.Sequential(nn.Linear(feat_dim, proj), nn.GELU())
        self.lstm = nn.LSTM(proj, lstm_hidden, n_lstm_layers)
        for name, p in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(p, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(p, 1.0)

        # Residual policy head (zero-init final layer -> g_theta == 0 at init -> policy == v2).
        self.g_head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, n_letters),
        )
        nn.init.zeros_(self.g_head[-1].weight)
        nn.init.zeros_(self.g_head[-1].bias)

        # Value head (sigmoid -> P(win) in (0,1), consistent with terminal-reward returns).
        self.v_head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def initial_state(self, batch: int, device) -> State:
        z = torch.zeros(self.n_lstm_layers, batch, self.lstm_hidden, device=device)
        return (z, z.clone())

    # ---- core recurrence: run the LSTM over a (T,B,*) sequence with per-step resets ----
    def get_states(self, feat_seq: torch.Tensor, lstm_state: State,
                   reset_seq: torch.Tensor) -> Tuple[torch.Tensor, State]:
        """feat_seq (T,B,proj); reset_seq (T,B) float. Returns out (T,B,hidden), final state."""
        outs = []
        for t in range(feat_seq.shape[0]):
            keep = (1.0 - reset_seq[t]).view(1, -1, 1)  # zero the state where episode restarted
            lstm_state = (keep * lstm_state[0], keep * lstm_state[1])
            o, lstm_state = self.lstm(feat_seq[t:t + 1], lstm_state)  # o (1,B,hidden)
            outs.append(o)
        return torch.cat(outs, dim=0), lstm_state

    def _masked_logits(self, l_ref, lstm_out, allowed):
        logits = l_ref / self.tau + self.g_head(lstm_out)
        return logits.masked_fill(~allowed, NEG_MASK)

    # ---- single rollout step (B envs) ----
    def act_step(self, l_ref, h, allowed, lstm_state: State, reset,
                 deterministic: bool = False):
        feat = self.proj(h).unsqueeze(0)                       # (1,B,proj)
        out, lstm_state = self.get_states(feat, lstm_state, reset.unsqueeze(0))
        out = out.squeeze(0)                                   # (B,hidden)
        logits = self._masked_logits(l_ref, out, allowed)
        value = torch.sigmoid(self.v_head(out)).squeeze(-1)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value, dist.entropy(), lstm_state

    # ---- batched sequence evaluation for the PPO update (BPTT over T) ----
    def evaluate_sequence(self, l_ref, h, allowed, actions, reset_seq, init_state: State):
        """All (T,B,*); actions/reset (T,B). Returns logp, value, entropy, kl_to_ref (T,B)."""
        feat = self.proj(h)                                    # (T,B,proj)
        out, _ = self.get_states(feat, init_state, reset_seq)  # (T,B,hidden)
        logits = self._masked_logits(l_ref, out, allowed)
        ref_logits = (l_ref / self.tau).masked_fill(~allowed, NEG_MASK)
        dist = Categorical(logits=logits)
        ref = Categorical(logits=ref_logits)
        value = torch.sigmoid(self.v_head(out)).squeeze(-1)
        return dist.log_prob(actions), value, dist.entropy(), kl_divergence(dist, ref)
