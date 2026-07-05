"""Frozen v2 trunk → RL features (the shared contract for Strategy v3).

This module is the single source of the belief-MDP front-end. It wraps the *frozen*
Strategy-v2 components (the negative-evidence char Transformer encoder + the original
gated MoE) and, for any batch of game states, produces exactly three tensors:

    l_ref        (N, 26) float : log( s2 + eps ), the v2 reference logits.
                                 s2 = alpha*encoder_mean + (1-alpha)*moe  (alpha=0.30),
                                 i.e. the same blended score Strategy v2 argmaxes over.
    h            (N, D)  float : the pooled trunk feature consumed by the RL heads:
                                 [ masked-mean encoder hidden (feat_dim)
                                   | lives/6, wrong/6, blanks/L, L/20, |G|/26   (5 globals)
                                   | guessed multi-hot (26) ]
                                 D = encoder.feat_dim + 5 + 26.
    allowed_mask (N, 26) bool  : True for letters not yet guessed (valid actions).

Design guarantees:
  * **Exactness / skip-connection invariant.** States are grouped by board length and run
    *without padding* per length bucket, so the encoder and the MoE BiLSTMs see the same
    inputs they would single-sample. Hence `argmax_{allowed} l_ref` reproduces v2's guess
    bit-for-bit, batched or not. This is what makes "v3 with a zero residual == v2" hold.
  * **Frozen.** All trunk params are eval()+requires_grad_(False); `compute` runs under
    no_grad. The RL heads receive l_ref/h as detached inputs, so gradients only ever flow
    through the (small) policy/value heads.

The state primitive is the codebase convention `(board, guessed)`:
    board   : list[str] of length L, revealed letters in place, "_" for blanks.
    guessed : iterable[str] of letters already guessed (hits and misses).
"""
import os
from typing import Iterable, List, Sequence, Tuple

import torch

from vocab import (CHARS, CHAR_TO_IDX, MASK_IDX, NUM_LETTERS, MAX_WRONG,
                   letter_to_target)
from evaluate import load_encoder, _OldBiLSTM, _OldGate

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

State = Tuple[Sequence[str], Iterable[str]]  # (board, guessed)


class TrunkFeatures:
    """Frozen v2 trunk that maps game states to (l_ref, h, allowed_mask)."""

    def __init__(self, device, encoder_ckpt: str = None, alpha: float = 0.30,
                 eps: float = 1e-9):
        self.device = device
        self.alpha = alpha
        self.eps = eps

        # --- encoder (v2 negative-evidence Transformer) ---
        encoder_ckpt = encoder_ckpt or os.path.join(MODELS_DIR, "hangman_encoder.pt")
        self.encoder = load_encoder(encoder_ckpt, device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # --- MoE experts + gate (original Strategy v1) ---
        names = ["short", "medium", "long", "common", "rare"]
        self.experts = []
        for n in names:
            m = _OldBiLSTM().to(device)
            m.load_state_dict(torch.load(
                os.path.join(MODELS_DIR, f"expert_{n}_bilstm.pt"), map_location=device))
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self.experts.append(m)
        self.gate = _OldGate().to(device)
        self.gate.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, "best_gating_network.pt"), map_location=device))
        self.gate.eval()
        for p in self.gate.parameters():
            p.requires_grad_(False)

        self.enc_feat_dim = self.encoder.feat_dim
        self.n_globals = 5
        self.n_letters = NUM_LETTERS

    @property
    def feat_dim(self) -> int:
        """D = dimension of the pooled state feature `h`."""
        return self.enc_feat_dim + self.n_globals + self.n_letters

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _parse(board: Sequence[str], guessed: Iterable[str]):
        board = list(board)
        guessed = set(guessed)
        revealed = set(c for c in board if c != "_")
        absent = set(c for c in guessed if c in CHARS) - revealed
        return board, guessed, revealed, absent

    # ------------------------------------------------------------------ main API
    @torch.no_grad()
    def compute(self, states: List[State]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """States -> (l_ref (N,26), h (N,D), allowed_mask (N,26) bool), all on device.

        Length-bucketed so per-bucket forwards are unpadded and therefore numerically
        identical to single-sample inference (the v2-exactness guarantee).
        """
        n = len(states)
        dev = self.device
        l_ref = torch.empty(n, self.n_letters, device=dev)
        h = torch.empty(n, self.feat_dim, device=dev)
        allowed = torch.ones(n, self.n_letters, dtype=torch.bool, device=dev)

        # group indices by board length
        buckets: dict = {}
        parsed = []
        for i, (board, guessed) in enumerate(states):
            b, g, rev, abs_ = self._parse(board, guessed)
            parsed.append((b, g, rev, abs_))
            buckets.setdefault(len(b), []).append(i)

        for L, idxs in buckets.items():
            G = len(idxs)
            input_ids = torch.empty(G, L, dtype=torch.long, device=dev)
            absent_mh = torch.zeros(G, self.n_letters, device=dev)
            present_mh = torch.zeros(G, self.n_letters, device=dev)
            gvec = torch.zeros(G, self.n_letters, device=dev)
            blanks = torch.empty(G, device=dev)
            wrong = torch.empty(G, device=dev)
            n_guessed = torch.empty(G, device=dev)
            allow_block = torch.ones(G, self.n_letters, dtype=torch.bool, device=dev)

            for r, i in enumerate(idxs):
                board, guessed, revealed, absent = parsed[i]
                ids = [MASK_IDX if c == "_" else CHAR_TO_IDX[c] for c in board]
                input_ids[r] = torch.tensor(ids, dtype=torch.long, device=dev)
                for ch in absent:
                    absent_mh[r, letter_to_target(ch)] = 1.0
                for ch in revealed:
                    present_mh[r, letter_to_target(ch)] = 1.0
                for ch in guessed:
                    if ch in CHARS:
                        gvec[r, letter_to_target(ch)] = 1.0
                        allow_block[r, letter_to_target(ch)] = False
                blanks[r] = sum(1 for c in board if c == "_")
                wrong[r] = len(absent)
                n_guessed[r] = sum(1 for ch in guessed if ch in CHARS)

            mask = (input_ids == MASK_IDX)                     # (G, L) hidden positions
            mask_f = mask.float()
            denom = mask_f.sum(1).clamp(min=1.0)               # (G,)

            # ---- encoder: pooled hidden + mean-aggregated probs ----
            hid = self.encoder.encode(input_ids, absent_mh, present_mh, pad_mask=None)
            enc_logits = self.encoder.head(hid)                # (G, L, 26)
            enc_probs = torch.softmax(enc_logits, dim=-1)
            enc_mean = (enc_probs * mask_f.unsqueeze(-1)).sum(1) / denom.unsqueeze(1)  # (G,26)
            pooled = (hid * mask_f.unsqueeze(-1)).sum(1) / denom.unsqueeze(1)          # (G,feat)

            # ---- MoE: gated blend of the 5 experts (matches _moe_scorer exactly) ----
            expert_probs = []
            for m in self.experts:
                p = torch.softmax(m(input_ids), dim=-1)
                expert_probs.append((p * mask_f.unsqueeze(-1)).sum(1) / denom.unsqueeze(1))
            base = torch.stack([
                torch.full((G,), L / 20.0, device=dev),
                blanks / L if L else torch.zeros(G, device=dev),
                wrong / 6.0,
            ], dim=1)                                          # (G, 3)
            feat = torch.cat([base, gvec, torch.cat(expert_probs, dim=1)], dim=1)  # (G,159)
            w = torch.softmax(self.gate(feat), dim=1)          # (G, 5)
            moe = torch.zeros(G, self.n_letters, device=dev)
            for k, p in enumerate(expert_probs):
                moe = moe + w[:, k:k + 1] * p

            # ---- v2 blended score and reference logits ----
            s2 = self.alpha * enc_mean + (1.0 - self.alpha) * moe   # (G, 26)
            block_lref = torch.log(s2 + self.eps)

            # ---- pooled state feature h ----
            lives = (MAX_WRONG - wrong).clamp(min=0)
            globals_ = torch.stack([
                lives / 6.0, wrong / 6.0,
                blanks / L if L else torch.zeros(G, device=dev),
                torch.full((G,), L / 20.0, device=dev),
                n_guessed / 26.0,
            ], dim=1)                                          # (G, 5)
            block_h = torch.cat([pooled, globals_, gvec], dim=1)   # (G, feat+5+26)

            idx_t = torch.tensor(idxs, device=dev)
            l_ref.index_copy_(0, idx_t, block_lref)
            h.index_copy_(0, idx_t, block_h)
            allowed.index_copy_(0, idx_t, allow_block)

        return l_ref, h, allowed

    @torch.no_grad()
    def v2_guess_idx(self, board: Sequence[str], guessed: Iterable[str]) -> int:
        """The letter index (0..25) Strategy v2 would guess — for the exactness test."""
        l_ref, _, allowed = self.compute([(board, guessed)])
        scores = l_ref[0].clone()
        scores[~allowed[0]] = -float("inf")
        return int(torch.argmax(scores).item())
