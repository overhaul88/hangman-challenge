"""Model for the enhanced Hangman solver.

`HangmanEncoder` is a char-level encoder that predicts, at every position, a
distribution over the 26 letters. Crucially it is conditioned on a *guessed-state*
vector built from the absent (wrong-guess) and present (revealed) letter multi-hots,
so its predictions are a proper posterior over the hidden letters given everything the
player knows.

Default architecture is a Transformer encoder; a BiLSTM variant is available behind the
`arch` flag for ablation.
"""
from typing import Optional

import torch
import torch.nn as nn

from vocab import VOCAB_SIZE, PAD_IDX, NUM_LETTERS


class HangmanEncoder(nn.Module):
    def __init__(self,
                 arch: str = "transformer",
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_len: int = 40):
        super().__init__()
        self.arch = arch
        self.d_model = d_model

        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_IDX)
        # Guessed-state conditioning: [absent(26) | present(26)] -> d_model, added to every token.
        self.state_proj = nn.Linear(2 * NUM_LETTERS, d_model)

        if arch == "transformer":
            self.pos_embedding = nn.Embedding(max_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif arch == "bilstm":
            self.pos_embedding = None
            self.encoder = nn.LSTM(d_model, d_model, num_layers=num_layers,
                                   batch_first=True, bidirectional=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.dropout = nn.Dropout(dropout)
        head_in = d_model * 2 if arch == "bilstm" else d_model
        self.head = nn.Linear(head_in, NUM_LETTERS)

    @property
    def feat_dim(self) -> int:
        """Dimension of the pre-head hidden representation returned by `encode`."""
        return self.d_model * 2 if self.arch == "bilstm" else self.d_model

    def encode(self,
               input_ids: torch.Tensor,        # (B, L) long
               absent: torch.Tensor,           # (B, 26) float
               present: torch.Tensor,          # (B, 26) float
               pad_mask: Optional[torch.Tensor] = None,  # (B, L) bool, True = pad
               ) -> torch.Tensor:
        """Return the per-position pre-head hidden states (B, L, feat_dim).

        This is the representation fed to `self.head`; downstream consumers (e.g. the RL
        residual policy) pool it to form a state feature. `forward` is exactly
        `head(encode(...))`, so existing behavior is unchanged.
        """
        B, L = input_ids.shape
        x = self.embedding(input_ids)                      # (B, L, d)
        state = self.state_proj(torch.cat([absent, present], dim=-1))  # (B, d)
        x = x + state.unsqueeze(1)                         # broadcast over positions

        if self.arch == "transformer":
            pos_ids = torch.arange(L, device=input_ids.device).clamp_max(
                self.pos_embedding.num_embeddings - 1)
            x = x + self.pos_embedding(pos_ids).unsqueeze(0)
            x = self.dropout(x)
            x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, L, d)
        else:
            x = self.dropout(x)
            x, _ = self.encoder(x)                          # (B, L, 2d)
        return x

    def forward(self,
                input_ids: torch.Tensor,        # (B, L) long
                absent: torch.Tensor,           # (B, 26) float
                present: torch.Tensor,          # (B, 26) float
                pad_mask: Optional[torch.Tensor] = None,  # (B, L) bool, True = pad
                ) -> torch.Tensor:
        x = self.encode(input_ids, absent, present, pad_mask)  # (B, L, feat_dim)
        return self.head(x)                                 # (B, L, 26)
