"""Strategy v7 — the GRU Deep Recurrent Q-Network over the two beliefs.

strategy7.md §3. Unlike v5's `policy_drqn.DuelingResidualDRQN` (LSTM-128, dueling head,
zero-init residual over the frozen v2 logits), v7's DRQN is a **plain** recurrent Q-net
whose input is the 80-dim belief vector

    x_t = [ p_enc(26) | p_moe(26) | guessed(26) | wrong/6 (1) | t/11 (1) ]

It is a GRU(80->256) followed by a Q-head FC(256->128, ReLU)->FC(128->26). There is no
`l_ref` skip-connection, so a fresh network does NOT reproduce v2 — the no-regression
guarantee is provided instead by the final paired safe-selection gate (train_drqn_v7.py).

Recurrence follows the CleanRL `ppo_lstm` pattern used elsewhere in this codebase: a per-
step `reset` mask zeroes the GRU state at episode boundaries (reset[t]==1 -> wipe before
step t). Already-guessed letters are masked to `NEG_MASK` (a large finite negative; never
-inf, which can NaN the backward pass) before the argmax / bootstrap.
"""
from typing import Tuple

import torch
import torch.nn as nn

NEG_MASK = -1e9


class GRUDRQN(nn.Module):
    def __init__(self, input_dim: int = 80, gru_hidden: int = 256,
                 head_hidden: int = 128, n_letters: int = 26):
        super().__init__()
        self.input_dim = input_dim
        self.gru_hidden = gru_hidden
        self.n_letters = n_letters

        self.gru = nn.GRU(input_dim, gru_hidden)        # time-major (T,B,*)
        for name, p in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(p, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(p, 1.0)

        self.head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, n_letters),
        )

    def initial_state(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(1, batch, self.gru_hidden, device=device)

    # ---- run the GRU over a (T,B,input) sequence, zeroing state at episode boundaries ----
    def _run(self, x_seq: torch.Tensor, h: torch.Tensor,
             reset_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = x_seq.shape[0]
        if T == 0:
            return x_seq.new_zeros(0, x_seq.shape[1], self.gru_hidden), h
        # Fast path: no mid-sequence resets (always the case for BPTT training, which only
        # resets at t=0, and for single-step collection). Apply the t=0 reset, then run the
        # whole sequence in ONE GRU call — one kernel launch instead of T. Identical output
        # for a reset-free interior; falls back to the per-step loop if an interior reset exists.
        if T == 1 or torch.count_nonzero(reset_seq[1:]) == 0:
            h = (1.0 - reset_seq[0]).view(1, -1, 1) * h
            return self.gru(x_seq, h)
        outs = []
        for t in range(T):
            keep = (1.0 - reset_seq[t]).view(1, -1, 1)   # (1,B,1)
            h = keep * h
            o, h = self.gru(x_seq[t:t + 1], h)           # o (1,B,hidden)
            outs.append(o)
        return torch.cat(outs, dim=0), h

    def _q(self, gru_out: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        # Cast to fp32: the head matmul may run in fp16 under AMP, but the finite NEG_MASK
        # sentinel (-1e9) overflows fp16, and Q-masking/argmax/TD-loss are stabler in fp32.
        q = self.head(gru_out).float()
        return q.masked_fill(~allowed, NEG_MASK)

    # ---- single step (B envs); used for collection AND evaluation ----
    def q_step(self, x: torch.Tensor, h: torch.Tensor, reset: torch.Tensor,
               allowed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x (B,input); h (1,B,hidden); reset (B,) float; allowed (B,26) bool."""
        out, new_h = self._run(x.unsqueeze(0), h, reset.unsqueeze(0))
        return self._q(out.squeeze(0), allowed), new_h

    # ---- batched sequence (BPTT over T) for the DRQN loss ----
    def q_sequence(self, x_seq: torch.Tensor, allowed_seq: torch.Tensor,
                   reset_seq: torch.Tensor, init_state: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_seq (T,B,input); allowed_seq (T,B,26) bool; reset_seq (T,B). Returns Q (T,B,26)."""
        out, final = self._run(x_seq, init_state, reset_seq)
        return self._q(out, allowed_seq), final


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, IN = 8, 5, 80
    net = GRUDRQN(input_dim=IN)

    # q_step
    x = torch.randn(B, IN)
    allowed = torch.rand(B, 26) > 0.4
    allowed[:, 0] = True
    h0 = net.initial_state(B, x.device)
    q, h1 = net.q_step(x, h0, torch.ones(B), allowed)
    assert q.shape == (B, 26) and h1.shape == (1, B, 256)
    assert not torch.isnan(q).any()
    # masked letters must never win the argmax
    assert torch.all(allowed.gather(-1, q.argmax(-1, keepdim=True)).squeeze(-1))

    # q_sequence
    xs = torch.randn(T, B, IN)
    al = torch.rand(T, B, 26) > 0.4
    al[..., 0] = True
    rs = torch.zeros(T, B); rs[0] = 1.0
    qs, _ = net.q_sequence(xs, al, rs, net.initial_state(B, xs.device))
    assert qs.shape == (T, B, 26) and not torch.isnan(qs).any()
    print("policy_drqn_v7 self-test PASS", tuple(q.shape), tuple(qs.shape))
