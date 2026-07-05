"""Dueling residual Deep Recurrent Q-Network (DRQN) for Strategy v5.

A value-based (Q-learning) analogue of the Strategy-v4 recurrent policy
(`policy_recurrent.RecurrentResidualPolicy`). It keeps the same trunk inputs,
the same LSTM-reset recurrence, and the same v2 skip-connection safety net, but
replaces the actor-critic head with a **dueling Q-head**:

  * **Dueling aggregation.** `Q = V + (A - mean_a A)` where `V = v_head(lstm_out)`
    is a single state value and `A` is the per-action advantage. The mean is taken
    over the *allowed* actions only (already-guessed letters do not contribute to
    the centering), which keeps the identifiability of V/A consistent with masking.
  * **Residual skip-connection (>= v2).** The advantage is seeded by the frozen v2
    prior: `A = l_ref/tau + g_theta(lstm_out)` with the residual head `g_theta`
    zero-initialised. At init `g_theta == 0`, so `A == l_ref/tau`; since V and the
    masked mean are constant across actions they cancel in the argmax, hence
    `argmax_allowed Q == argmax_allowed l_ref` == v2 *for any LSTM weights/state/V*.
    A freshly constructed network's greedy guess therefore equals the v2 baseline.
  * **Finite action masking.** Disallowed letters are set to `rl_policy.NEG_MASK`
    (a large finite negative), never `-inf` (which makes some backward passes nan).
  * **Recurrence.** The observation `o = (board, guessed)` is already a sufficient
    statistic, so the LSTM adds no information; it is kept for value-stability and
    faithfulness to the DRQN spec, adding only representational capacity.

LSTM handling follows the CleanRL `ppo_lstm` pattern: `get_states` runs the LSTM
over a time sequence and zeroes the recurrent state at episode boundaries via a
per-step reset mask (`reset[t] == 1` means "obs at t starts a fresh episode ->
reset before step t"). The DRQN target network is just a second instance of this
same class with copied weights.

Inputs are the frozen-trunk features from `rl_features.TrunkFeatures`:
    l_ref (.,26), h (.,feat_dim), allowed_mask (.,26) bool.
"""
from typing import Tuple

import torch
import torch.nn as nn

from rl_policy import NEG_MASK

State = Tuple[torch.Tensor, torch.Tensor]  # (h, c), each (n_layers, B, hidden)


class DuelingResidualDRQN(nn.Module):
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

        # Advantage residual head (zero-init final layer -> g_theta == 0 at init ->
        # advantage == l_ref/tau -> greedy action == v2).
        self.g_head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, n_letters),
        )
        nn.init.zeros_(self.g_head[-1].weight)
        nn.init.zeros_(self.g_head[-1].bias)

        # State-value head (raw scalar, NOT sigmoid'd: this is a Q-value baseline).
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

    # ---- dueling Q with a v2 residual skip (broadcasts over arbitrary leading dims) ----
    def _dueling_q(self, l_ref: torch.Tensor, lstm_out: torch.Tensor,
                   allowed: torch.Tensor) -> torch.Tensor:
        """l_ref (.,26); lstm_out (.,hidden); allowed (.,26) bool. Returns Q (.,26)."""
        adv = l_ref / self.tau + self.g_head(lstm_out)          # (.,26)
        value = self.v_head(lstm_out)                           # (.,1)
        # Mean of the advantage over allowed actions only.
        adv_for_mean = adv.masked_fill(~allowed, 0.0)
        count = allowed.sum(-1, keepdim=True).clamp(min=1)      # (.,1)
        adv_mean = adv_for_mean.sum(-1, keepdim=True) / count   # (.,1)
        q = value + (adv - adv_mean)                            # (.,26)
        return q.masked_fill(~allowed, NEG_MASK)

    # ---- single rollout step (B envs); used for collection AND evaluation ----
    def q_step(self, l_ref: torch.Tensor, h: torch.Tensor, allowed: torch.Tensor,
               lstm_state: State, reset: torch.Tensor) -> Tuple[torch.Tensor, State]:
        """l_ref (B,26), h (B,feat_dim), allowed (B,26) bool, reset (B,) float."""
        feat = self.proj(h).unsqueeze(0)                       # (1,B,proj)
        out, new_state = self.get_states(feat, lstm_state, reset.unsqueeze(0))
        out = out.squeeze(0)                                   # (B,hidden)
        q = self._dueling_q(l_ref, out, allowed)               # (B,26)
        return q, new_state

    # ---- batched sequence evaluation for the DRQN loss (BPTT over T) ----
    def q_sequence(self, l_ref: torch.Tensor, h: torch.Tensor, allowed: torch.Tensor,
                   reset_seq: torch.Tensor, init_state: State) -> Tuple[torch.Tensor, State]:
        """All (T,B,*); reset_seq (T,B) float. Returns Q (T,B,26), final state."""
        feat = self.proj(h)                                    # (T,B,proj)
        out, final_state = self.get_states(feat, init_state, reset_seq)  # (T,B,hidden)
        q = self._dueling_q(l_ref, out, allowed)               # (T,B,26)
        return q, final_state


if __name__ == "__main__":
    torch.manual_seed(0)
    B, feat_dim, n = 8, 415, 26
    net = DuelingResidualDRQN(feat_dim=feat_dim)

    # Random inputs; force column 0 True so every row has >= 1 allowed action.
    l_ref = torch.randn(B, n)
    h = torch.randn(B, feat_dim)
    allowed = torch.rand(B, n) > 0.5
    allowed[:, 0] = True

    # ---- q_step with a random non-zero recurrent state and reset=0 (state NOT wiped) ----
    rand_state: State = (torch.randn(net.n_lstm_layers, B, net.lstm_hidden),
                         torch.randn(net.n_lstm_layers, B, net.lstm_hidden))
    reset = torch.zeros(B)
    Q, _ = net.q_step(l_ref, h, allowed, rand_state, reset)

    ref = l_ref.masked_fill(~allowed, NEG_MASK)
    assert torch.equal(Q.argmax(-1), ref.argmax(-1)), "q_step greedy != v2"
    assert not torch.isnan(Q).any(), "NaN in q_step Q"
    print("skip-invariant (q_step) PASS")

    # ---- q_sequence over (T=4,B=8); reset only at t=0 ----
    T = 4
    l_ref_s = torch.randn(T, B, n)
    h_s = torch.randn(T, B, feat_dim)
    allowed_s = torch.rand(T, B, n) > 0.5
    allowed_s[..., 0] = True
    reset_seq = torch.zeros(T, B)
    reset_seq[0] = 1.0
    init_state = net.initial_state(B, l_ref_s.device)
    Qs, _ = net.q_sequence(l_ref_s, h_s, allowed_s, reset_seq, init_state)

    ref_s = l_ref_s.masked_fill(~allowed_s, NEG_MASK)
    assert torch.equal(Qs.argmax(-1), ref_s.argmax(-1)), "q_sequence greedy != v2"
    assert not torch.isnan(Qs).any(), "NaN in q_sequence Q"
    print("skip-invariant (q_sequence) PASS")

    print(f"Q shape {tuple(Q.shape)}, Q_seq shape {tuple(Qs.shape)}")
