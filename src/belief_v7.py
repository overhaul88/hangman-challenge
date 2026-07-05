"""Strategy v7 — the belief engine: a *trainable* encoder belief + a *frozen* MoE oracle.

strategy7.md §2-§3. This is the one structural departure from v5: the encoder belief
`p_enc` must be produced by a **grad-enabled** forward (so the RL TD signal can reshape
the encoder's top blocks), whereas `rl_features.TrunkFeatures` is `@torch.no_grad()` and
freezes everything. `BeliefEngine` therefore:

  * loads the strategy2 encoder (`hangman_encoder.pt`) and **fine-tunes only its top two
    transformer blocks + output head** (everything else — lower blocks, token/positional/
    state embeddings — stays frozen for the whole run);
  * loads the original gated MoE (5 BiLSTM experts + gate) as a **frozen oracle** for
    `p_moe` (computed under `no_grad`, never updated);
  * exposes `p_enc_from_tensors` (the grad path used on replay) and the no-grad
    `p_enc_from_states` / `p_moe_from_states` used for acting and evaluation.

Both beliefs are 26-dim letter marginals obtained by **mean-pooling per-position softmax
probabilities over the hidden (MASK) positions** — matching strategy2's A/B-tested mean
aggregation.

The two beliefs disjointly cover the guessed letters: a correctly-guessed letter is
revealed (`present`), a wrongly-guessed one is `absent`, so
    guessed = present ∪ absent,  allowed = ¬guessed,  wrong = |absent|.
We exploit this to derive the DRQN's `guessed`/`wrong`/`allowed` features without storing
them separately.
"""
import os
from collections import defaultdict, OrderedDict
from typing import Iterable, List, Sequence, Tuple

import torch

from vocab import (CHARS, CHAR_TO_IDX, MASK_IDX, NUM_LETTERS, MAX_WRONG,
                   letter_to_target)
from evaluate import load_encoder, _OldBiLSTM, _OldGate

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
State = Tuple[Sequence[str], Iterable[str]]  # (board, guessed)


class BeliefEngine:
    def __init__(self, device, encoder_ckpt: str = None, n_finetune_layers: int = 2):
        self.device = device
        self.n_finetune = n_finetune_layers

        # --- encoder (strategy2): top-N transformer blocks + head trainable, rest frozen ---
        encoder_ckpt = encoder_ckpt or os.path.join(MODELS_DIR, "hangman_encoder.pt")
        ckpt = torch.load(encoder_ckpt, map_location=device)
        self.encoder_config = ckpt.get("config", {})
        self.encoder = load_encoder(encoder_ckpt, device)
        # Keep the encoder in eval() throughout: we want deterministic, dropout-free
        # beliefs while STILL backpropagating into the top blocks (eval() disables dropout
        # but does not stop gradients).
        self.encoder.eval()
        if self.encoder.arch != "transformer":
            raise ValueError("BeliefEngine expects the transformer encoder")
        # The TransformerEncoder NestedTensor fast-path (eval + padding mask) is incompatible
        # with autograd; disable it so the grad-enabled MLM forward (which passes a padding
        # mask) takes the standard path.
        self.encoder.encoder.enable_nested_tensor = False
        self._layers = self.encoder.encoder.layers          # nn.ModuleList
        self.n_layers = len(self._layers)
        if self.n_finetune > self.n_layers:
            raise ValueError(f"n_finetune {self.n_finetune} > n_layers {self.n_layers}")

        # Freeze everything, then mark the top-N blocks + head as the trainable set.
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._trainable: List[torch.nn.Parameter] = []
        for blk in self._layers[self.n_layers - self.n_finetune:]:
            self._trainable += list(blk.parameters())
        self._trainable += list(self.encoder.head.parameters())
        # Default: frozen (Phase-2 warmup unfreezes via set_encoder_trainable(True)).
        self.set_encoder_trainable(False)

        # --- MoE oracle (original strategy1): 5 BiLSTM experts + gate, FULLY frozen ---
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

        self.feat_dim = self.encoder.feat_dim  # 384

        # --- frozen-oracle memoization (collection speedup) ---
        # p_moe (5-BiLSTM MoE, ~70% of a collect-step) and the frozen lower-trunk hidden H3
        # are EXACT, constant functions of the canonical state — both are fully frozen for the
        # whole run. So we memoize (H3, mask, p_moe, absent, present) per state and recompute
        # only the trainable top blocks (p_enc) per visit. Hangman openings + early boards
        # recur on every episode reset, so the collection hit-rate is high and the memoized
        # values are bit-identical to recomputing them (no approximation).
        self._cache: "OrderedDict" = OrderedDict()
        self._cache_cap = 0            # 0 = disabled (enable_frozen_cache turns it on)
        self.cache_hits = 0
        self.cache_miss = 0

    # ------------------------------------------------------------------ frozen-oracle cache
    def enable_frozen_cache(self, cap: int) -> None:
        """cap > 0 enables LRU memoization of the frozen oracles; cap <= 0 disables it."""
        self._cache_cap = int(cap)
        self._cache.clear()
        self.cache_hits = self.cache_miss = 0

    def _cache_get(self, key):
        v = self._cache.get(key)
        if v is not None:
            self._cache.move_to_end(key)          # LRU: keep hot openings resident
        return v

    def _cache_put(self, key, value) -> None:
        self._cache[key] = value
        if len(self._cache) > self._cache_cap:
            self._cache.popitem(last=False)       # evict least-recently-used

    @torch.no_grad()
    def collection_beliefs(self, states: List[State]):
        """One-shot acting-time beliefs for collection (single parse + single trunk pass).

        Returns (penc (N,26) dev, pmoe (N,26) dev, caches list[(H3 fp16 cpu, mask cpu)],
                 absents list[(26,) cpu], presents list[(26,) cpu]). penc is always computed
                 from the (memoized or fresh) frozen trunk H3 with the CURRENT top-block
                 weights (which move during training); p_moe and H3 are frozen/exact.

        Frozen-oracle memoization (`enable_frozen_cache`) is OPTIONAL: measured hangman state
        recurrence is low (~0.24 hit at N=32), so it's off by default — the one-pass structure
        below is the real win over the old three-pass (acting_beliefs + p_moe + state_features)
        collection, since the frozen trunk and MoE are computed once per step, batched by length.
        """
        dev = self.device
        N = len(states)
        use_cache = self._cache_cap > 0
        parsed = [self._parse(b, g) for (b, g) in states]      # (board,guessed,revealed,absent)
        keys = [("".join(p[0]), frozenset(p[3])) for p in parsed] if use_cache else [None] * N

        # local store: per-index (H3 fp16 cpu, mask cpu, p_moe cpu, absent cpu, present cpu)
        vals: list = [None] * N
        if use_cache:
            for i in range(N):
                v = self._cache_get(keys[i])
                if v is not None:
                    vals[i] = v
        miss = [i for i in range(N) if vals[i] is None]

        if miss:
            byL = defaultdict(list)
            for i in miss:
                byL[len(parsed[i][0])].append(i)
            for L, idxs in byL.items():
                ids, ab, pr = self._tokenise_group(parsed, idxs, L)
                H = self.frozen_trunk(ids, ab, pr)             # (G,L,384)
                m = (ids == MASK_IDX)                          # (G,L)
                pm = self._moe_group(ids, ab, pr, L)           # (G,26)
                Hc, mc = H.to(torch.float16).cpu(), m.cpu()
                pmc, abc, prc = pm.cpu(), ab.cpu(), pr.cpu()
                for r, i in enumerate(idxs):
                    v = (Hc[r], mc[r], pmc[r], abc[r], prc[r])
                    vals[i] = v
                    if use_cache:
                        self._cache_put(keys[i], v)
            self.cache_miss += len(miss)
        self.cache_hits += N - len(miss)

        # ---- assemble; recompute penc (top blocks, current weights) batched by length ----
        pmoe = torch.empty(N, NUM_LETTERS, device=dev)
        caches: list = [None] * N
        absents: list = [None] * N
        presents: list = [None] * N
        byL = defaultdict(list)
        for i in range(N):
            H, m, pm, ab, pr = vals[i]
            caches[i] = (H, m); absents[i] = ab; presents[i] = pr
            pmoe[i] = pm.to(dev)
            byL[H.shape[0]].append(i)                          # H: (L,384)

        penc = torch.empty(N, NUM_LETTERS, device=dev)
        for L, idxs in byL.items():
            Hs = torch.stack([caches[i][0] for i in idxs]).to(dev)   # (G,L,384) fp16
            ms = torch.stack([caches[i][1] for i in idxs]).to(dev)   # (G,L)
            penc[torch.tensor(idxs, device=dev)] = self.p_enc_from_trunk(Hs, ms)
        return penc, pmoe, caches, absents, presents

    # ------------------------------------------------------------------ trainable set
    def set_encoder_trainable(self, flag: bool) -> None:
        """Toggle gradient on the top-N encoder blocks + head (Phase-2 warmup control)."""
        for p in self._trainable:
            p.requires_grad_(flag)
        self._enc_trainable = flag

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        return self._trainable

    # ------------------------------------------------------------------ parsing/tokenisation
    @staticmethod
    def _parse(board: Sequence[str], guessed: Iterable[str]):
        board = list(board)
        guessed = set(guessed)
        revealed = set(c for c in board if c != "_")
        absent = set(c for c in guessed if c in CHARS) - revealed
        return board, guessed, revealed, absent

    def _buckets(self, states: List[State]):
        """Group state indices by board length; return parsed states + {L: [idx,...]}."""
        parsed, buckets = [], defaultdict(list)
        for i, (board, guessed) in enumerate(states):
            p = self._parse(board, guessed)
            parsed.append(p)
            buckets[len(p[0])].append(i)
        return parsed, buckets

    def _tokenise_group(self, parsed, idxs, L):
        """Build the unpadded encoder inputs for one length bucket on device."""
        dev = self.device
        G = len(idxs)
        input_ids = torch.empty(G, L, dtype=torch.long, device=dev)
        absent_mh = torch.zeros(G, NUM_LETTERS, device=dev)
        present_mh = torch.zeros(G, NUM_LETTERS, device=dev)
        for r, i in enumerate(idxs):
            board, guessed, revealed, absent = parsed[i]
            input_ids[r] = torch.tensor(
                [MASK_IDX if c == "_" else CHAR_TO_IDX[c] for c in board],
                dtype=torch.long, device=dev)
            for ch in absent:
                absent_mh[r, letter_to_target(ch)] = 1.0
            for ch in revealed:
                present_mh[r, letter_to_target(ch)] = 1.0
        return input_ids, absent_mh, present_mh

    def tok_one(self, board: Sequence[str], guessed: Iterable[str]):
        """Single-state encoder inputs as CPU tensors (for replay storage).

        Returns (input_ids (L,) long, absent (26,) float, present (26,) float).
        """
        _, _, revealed, absent = self._parse(board, guessed)
        input_ids = torch.tensor(
            [MASK_IDX if c == "_" else CHAR_TO_IDX[c] for c in board], dtype=torch.long)
        absent_mh = torch.zeros(NUM_LETTERS)
        present_mh = torch.zeros(NUM_LETTERS)
        for ch in absent:
            absent_mh[letter_to_target(ch)] = 1.0
        for ch in revealed:
            present_mh[letter_to_target(ch)] = 1.0
        return input_ids, absent_mh, present_mh

    # ------------------------------------------------------------------ encoder belief
    # The encoder splits at the freeze boundary: a FROZEN trunk (embedding/state/pos +
    # layers[0:n_frozen]) and a TRAINABLE top (layers[n_frozen:] + head). The trunk output
    # is constant for the whole run, so v7 caches it per replay step (`frozen_trunk`) and
    # only recomputes the cheap trainable top with grad (`p_enc_from_trunk`) at train time
    # (~3x cheaper than re-running the full 6-layer forward). Grad through the frozen trunk
    # is zero anyway, so stopping at the cached hidden is exactness-preserving.
    @property
    def _n_frozen(self) -> int:
        return self.n_layers - self.n_finetune

    def _embed(self, input_ids: torch.Tensor, absent: torch.Tensor,
               present: torch.Tensor) -> torch.Tensor:
        """Replicate HangmanEncoder.encode's pre-transformer embedding (frozen, eval->no dropout)."""
        enc = self.encoder
        x = enc.embedding(input_ids)                                   # (G,L,d)
        state = enc.state_proj(torch.cat([absent, present], dim=-1))   # (G,d)
        x = x + state.unsqueeze(1)
        L = input_ids.shape[1]
        pos_ids = torch.arange(L, device=input_ids.device).clamp_max(
            enc.pos_embedding.num_embeddings - 1)
        x = x + enc.pos_embedding(pos_ids).unsqueeze(0)
        return enc.dropout(x)                                          # eval() => identity

    @torch.no_grad()
    def frozen_trunk(self, input_ids: torch.Tensor, absent: torch.Tensor,
                     present: torch.Tensor) -> torch.Tensor:
        """No-grad output of the frozen lower trunk: (G,L,384). Cacheable for the whole run."""
        x = self._embed(input_ids, absent, present)
        for blk in self._layers[:self._n_frozen]:
            x = blk(x)
        return x

    def p_enc_from_trunk(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """GRAD-ENABLED p_enc from a cached frozen-trunk hidden. hidden (G,L,384); mask (G,L).

        Runs only the trainable top blocks + head, then mean-pools softmax over MASK positions.
        Grads flow to layers[n_frozen:] + head iff unfrozen. `hidden` is upcast to fp32 so an
        fp16-cached trunk feeds cleanly (autocast re-casts for the matmuls when enabled).
        """
        x = hidden.float()
        for blk in self._layers[self._n_frozen:]:
            x = blk(x)
        logits = self.encoder.head(x)                                  # (G,L,26)
        probs = torch.softmax(logits, dim=-1)
        m = mask.float()                                               # (G,L)
        denom = m.sum(1).clamp(min=1.0)
        return (probs * m.unsqueeze(-1)).sum(1) / denom.unsqueeze(1)   # (G,26)

    def p_enc_from_tensors(self, input_ids: torch.Tensor, absent: torch.Tensor,
                           present: torch.Tensor) -> torch.Tensor:
        """GRAD-ENABLED p_enc for one length bucket (full forward; trunk part is no-grad)."""
        hid = self.frozen_trunk(input_ids, absent, present)            # (G,L,384) no-grad
        return self.p_enc_from_trunk(hid, input_ids == MASK_IDX)       # top blocks w/ grad

    @torch.no_grad()
    def p_enc_from_states(self, states: List[State]) -> torch.Tensor:
        """No-grad p_enc for acting/eval. Returns (N,26) on device."""
        out = torch.empty(len(states), NUM_LETTERS, device=self.device)
        parsed, buckets = self._buckets(states)
        for L, idxs in buckets.items():
            ids, ab, pr = self._tokenise_group(parsed, idxs, L)
            out[torch.tensor(idxs, device=self.device)] = self.p_enc_from_tensors(ids, ab, pr)
        return out

    @torch.no_grad()
    def acting_beliefs(self, states: List[State]):
        """Collection helper: ONE frozen-trunk pass that yields both the acting belief and the
        per-state trunk cache for replay. Returns (p_enc (N,26) on device, caches) where
        caches[i] = (H3 (L,384) fp16 CPU, mask (L,) bool CPU)."""
        penc = torch.empty(len(states), NUM_LETTERS, device=self.device)
        caches = [None] * len(states)
        parsed, buckets = self._buckets(states)
        for L, idxs in buckets.items():
            ids, ab, pr = self._tokenise_group(parsed, idxs, L)
            H = self.frozen_trunk(ids, ab, pr)                         # (G,L,384)
            m = (ids == MASK_IDX)                                      # (G,L) bool
            penc[torch.tensor(idxs, device=self.device)] = self.p_enc_from_trunk(H, m)
            Hc, mc = H.to(torch.float16).cpu(), m.cpu()
            for r, i in enumerate(idxs):
                caches[i] = (Hc[r], mc[r])
        return penc, caches

    # ------------------------------------------------------------------ MoE oracle belief
    @torch.no_grad()
    def _moe_group(self, input_ids, absent_mh, present_mh, L) -> torch.Tensor:
        """Frozen gated-MoE belief for one length bucket -> (G,26). Matches rl_features."""
        dev = self.device
        G = input_ids.shape[0]
        mask_f = (input_ids == MASK_IDX).float()
        denom = mask_f.sum(1).clamp(min=1.0)
        # gvec / blanks / wrong derived from present+absent (disjoint over guessed letters)
        gvec = (absent_mh + present_mh).clamp(max=1.0)                 # (G,26)
        wrong = absent_mh.sum(1)                                       # (G,)
        blanks = mask_f.sum(1)                                         # (G,)

        expert_probs = []
        for m in self.experts:
            p = torch.softmax(m(input_ids), dim=-1)                    # (G,L,26)
            expert_probs.append((p * mask_f.unsqueeze(-1)).sum(1) / denom.unsqueeze(1))
        base = torch.stack([
            torch.full((G,), L / 20.0, device=dev),
            blanks / L if L else torch.zeros(G, device=dev),
            wrong / 6.0,
        ], dim=1)                                                      # (G,3)
        feat = torch.cat([base, gvec, torch.cat(expert_probs, dim=1)], dim=1)  # (G,159)
        w = torch.softmax(self.gate(feat), dim=1)                      # (G,5)
        moe = torch.zeros(G, NUM_LETTERS, device=dev)
        for k, p in enumerate(expert_probs):
            moe = moe + w[:, k:k + 1] * p
        return moe

    @torch.no_grad()
    def p_moe_from_states(self, states: List[State]) -> torch.Tensor:
        out = torch.empty(len(states), NUM_LETTERS, device=self.device)
        parsed, buckets = self._buckets(states)
        for L, idxs in buckets.items():
            ids, ab, pr = self._tokenise_group(parsed, idxs, L)
            out[torch.tensor(idxs, device=self.device)] = self._moe_group(ids, ab, pr, L)
        return out

    # ------------------------------------------------------------------ DRQN scalar features
    @torch.no_grad()
    def state_features(self, states: List[State]):
        """Return (guessed_vec (N,26) float, wrong (N,) float, allowed (N,26) bool)."""
        dev = self.device
        N = len(states)
        gvec = torch.zeros(N, NUM_LETTERS, device=dev)
        wrong = torch.zeros(N, device=dev)
        for i, (board, guessed) in enumerate(states):
            _, gset, revealed, absent = self._parse(board, guessed)
            for ch in gset:
                if ch in CHARS:
                    gvec[i, letter_to_target(ch)] = 1.0
            wrong[i] = float(len(absent))
        allowed = gvec == 0.0
        return gvec, wrong, allowed
