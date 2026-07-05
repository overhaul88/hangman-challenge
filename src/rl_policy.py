"""Residual actor-critic policy for Strategy v3 (PPO fine-tuning of frozen v2).

The policy logits are the frozen Strategy-v2 reference logits PLUS a learned residual
that is ZERO at initialization, so a freshly constructed policy reproduces v2 *exactly*
(the skip-connection invariant). PPO then nudges the residual to improve on v2 while a
KL anchor to the (masked) reference distribution keeps it from drifting.

The trunk (`src/rl_features.py`) is responsible for turning game states into the three
inputs this module consumes:
    l_ref        (N, 26) float : v2 reference logits = log(blended v2 score).
    h            (N, D)  float : pooled state feature, D == trunk.feat_dim.
    allowed_mask (N, 26) bool  : True for letters not yet guessed (valid actions).

This module only ever sees those tensors (and uses the trunk solely inside
`make_guess_fn` to build them), keeping the frozen front-end's responsibilities out.
"""
from typing import Callable, Iterable, Sequence, Tuple

import torch
import torch.nn as nn

from vocab import target_to_letter

# Mask value for disallowed actions. A large *finite* negative (not -inf): exp() of it is
# ~0 so disallowed letters are never sampled/argmaxed and contribute ~0 to entropy/KL, but
# unlike -inf its gradient is finite — -inf masking makes entropy()/kl_divergence() backward
# produce nan (0*-inf), which would corrupt the heads after the first optimizer step.
NEG_MASK = -1e9


class ResidualPolicy(nn.Module):
    """Actor-critic head over the frozen v2 trunk with a zero-init logit residual."""

    def __init__(self, feat_dim: int, n_letters: int = 26, hidden: int = 256,
                 tau: float = 1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_letters = n_letters
        self.tau = tau

        # Policy residual: g_head(h) == 0 at init so logits == l_ref / tau initially.
        self.g_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_letters),
        )
        # Zero-initialize the final linear layer (weight AND bias).
        nn.init.zeros_(self.g_head[-1].weight)
        nn.init.zeros_(self.g_head[-1].bias)

        # Value head: estimates P(win) in (0, 1) via sigmoid (applied in forward).
        self.v_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, l_ref: torch.Tensor, h: torch.Tensor,
                allowed_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (masked logits (N,26), value (N,) in (0,1))."""
        logits = l_ref / self.tau + self.g_head(h)
        logits = logits.masked_fill(~allowed_mask, NEG_MASK)
        value = torch.sigmoid(self.v_head(h)).squeeze(-1)
        return logits, value

    def reference_logits(self, l_ref: torch.Tensor,
                         allowed_mask: torch.Tensor) -> torch.Tensor:
        """Masked v2 reference logits (l_ref / tau, no residual) for the KL anchor."""
        return (l_ref / self.tau).masked_fill(~allowed_mask, NEG_MASK)

    def act(self, l_ref: torch.Tensor, h: torch.Tensor, allowed_mask: torch.Tensor,
            deterministic: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) an action; return (action, logprob, value, entropy)."""
        logits, value = self.forward(l_ref, h, allowed_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate_actions(self, l_ref: torch.Tensor, h: torch.Tensor,
                         allowed_mask: torch.Tensor, actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-score given actions under the current policy (PPO update)."""
        logits, value = self.forward(l_ref, h, allowed_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()


def make_guess_fn(policy: ResidualPolicy, trunk, device,
                  deterministic: bool = True
                  ) -> Callable[[Sequence[str], Iterable[str]], str]:
    """Build a guess_fn(board, guessed)->letter so evaluate.evaluate_winrate can score the policy."""

    @torch.no_grad()
    def guess_fn(board: Sequence[str], guessed: Iterable[str]) -> str:
        l_ref, h, allowed = trunk.compute([(board, guessed)])
        logits, _ = policy.forward(l_ref, h, allowed)
        idx = int(logits[0].argmax(-1).item())
        return target_to_letter(idx)

    return guess_fn
