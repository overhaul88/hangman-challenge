"""Vectorized PIMC one-step lookahead — the fast, GPU-batched policy-improvement operator.

This is a drop-in, *much* faster re-implementation of the Monte-Carlo lookahead in
``lookahead.py``. The original is correct but deploys the GPU v2 policy once per
rollout-step per sampled belief word, which is >1 min/game on the 4 GB GTX 1650 and made
a held-out sweep compute-prohibitive (see strategy3.md A.5 / research_strategy.md §2.7).

The bottleneck is the *continuation policy* v2 inside the rollouts. Here we keep the exact
same math but **simulate every belief rollout in parallel** and call v2 **once per
rollout-depth on the whole batch** via ``rl_features.TrunkFeatures`` (length-bucketed,
numerically identical to single-sample v2). For a root state we therefore pay ~horizon
batched GPU forwards instead of n_samples*horizon single forwards.

Definition (identical to lookahead.PIMCLookahead):

    Q_hat(o, c) = P( win | guess c now, then follow v2 to the end ),  estimated over words
                  sampled from the belief support (train-corpus words consistent with o).

``best_action`` = argmax_c Q_hat with the v2 move kept as the incumbent (tie-break to v2 and
an optional std-error switch margin), so it is provably >= v2 on the shared sample
(Policy Improvement Theorem). It has the ``guess_fn(board, guessed) -> letter`` signature
so it plugs straight into ``evaluate.evaluate_winrate``.

It also exposes ``label_state`` which returns the cached frozen-trunk features
``(l_ref, h, allowed)`` of the root *together with* a 26-dim Q-target vector, so the same
operator doubles as the teacher for Strategy-v6 distillation (see distill.py).
"""
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from vocab import CHARS, MAX_WRONG, letter_to_target, target_to_letter
from rl_features import TrunkFeatures
from lookahead import consistent_words

State = Tuple[List[str], set]


class VectorizedV2:
    """The frozen v2 policy, batched. argmax_allowed(l_ref) == v2's guess (verified)."""

    def __init__(self, device, alpha: float = 0.30):
        self.trunk = TrunkFeatures(device, alpha=alpha)
        self.device = device

    @property
    def feat_dim(self) -> int:
        return self.trunk.feat_dim

    @torch.no_grad()
    def features(self, states: List[State]):
        """States -> (l_ref (N,26), h (N,D), allowed (N,26) bool)."""
        return self.trunk.compute(states)

    @torch.no_grad()
    def guess_idx(self, states: List[State]) -> List[int]:
        """v2's chosen letter index (0..25) for each state."""
        l_ref, _, allowed = self.trunk.compute(states)
        scores = l_ref.masked_fill(~allowed, -float("inf"))
        return scores.argmax(dim=-1).tolist()


class _SimGame:
    """A single in-flight rollout (one belief word, first guess forced)."""
    __slots__ = ("word", "board", "guessed", "wrong", "done", "won")

    def __init__(self, word: str, board: List[str], guessed: set, wrong: int):
        self.word = word
        self.board = list(board)
        self.guessed = set(guessed)
        self.wrong = wrong
        self.done = False
        self.won = False

    def apply(self, letter: str, max_wrong: int):
        # Guard: the batched v2 never repeats a guess (already-guessed letters are masked),
        # but be defensive.
        if letter in self.guessed:
            self.done = True
            self.won = "_" not in self.board
            return
        self.guessed.add(letter)
        if letter in self.word:
            for i, c in enumerate(self.word):
                if c == letter:
                    self.board[i] = letter
        else:
            self.wrong += 1
        if "_" not in self.board:
            self.done, self.won = True, True
        elif self.wrong >= max_wrong:
            self.done, self.won = True, False


class FastPIMC:
    def __init__(self, corpus_words: Iterable[str], v2: VectorizedV2,
                 n_samples: int = 64, max_candidates: int = 8,
                 switch_margin: float = 0.0, max_wrong: int = MAX_WRONG,
                 seed: int = 0, belief=None):
        self.v2 = v2
        self.n_samples = n_samples
        self.max_candidates = max_candidates
        self.switch_margin = switch_margin
        self.max_wrong = max_wrong
        self.rng = random.Random(seed)
        # Fast vectorized consistent-word lookup (preferred); fall back to the
        # pure-Python filter only if no DictBelief is supplied.
        self.belief = belief
        if belief is not None:
            self.corpus_by_len = None
        else:
            self.corpus_by_len = {}
            for w in corpus_words:
                self.corpus_by_len.setdefault(len(w), []).append(w)
        self._last_n = 0
        self._last_base: Optional[str] = None

    def _consistent(self, board, guessed) -> List[str]:
        if self.belief is not None:
            return self.belief.consistent(board, guessed)
        return consistent_words(self.corpus_by_len, list(board), set(guessed))

    # ---------------------------------------------------------------- belief
    def _sample_belief(self, pool: List[str]) -> List[str]:
        if len(pool) <= self.n_samples:
            return list(pool)
        return self.rng.sample(pool, self.n_samples)

    # ---------------------------------------------------------------- core rollout
    @torch.no_grad()
    def _rollout_batch(self, games: List[_SimGame], chunk: int = 4096):
        """Play all games to termination, choosing every (non-forced) guess with batched v2.

        v2 is evaluated on the whole active set per depth (chunked to fit GPU memory), so the
        number of GPU forwards is ~horizon regardless of how many games are rolled together.
        """
        for _ in range(40):  # generous horizon cap
            active = [g for g in games if not g.done]
            if not active:
                break
            states = [(g.board, g.guessed) for g in active]
            idxs: List[int] = []
            for s in range(0, len(states), chunk):
                idxs.extend(self.v2.guess_idx(states[s:s + chunk]))
            for g, gi in zip(active, idxs):
                g.apply(target_to_letter(gi), self.max_wrong)

    @torch.no_grad()
    def q_values(self, board: Sequence[str], guessed: Iterable[str]) -> Dict[str, float]:
        """Estimate Q_hat(o,c) for candidate letters. {} when no belief support exists."""
        board = list(board)
        guessed = set(guessed)
        pool = self._consistent(board, guessed)
        if not pool:
            self._last_n, self._last_base = 0, None
            return {}
        sample = self._sample_belief(pool)
        n = len(sample)

        # Candidate letters = unguessed letters at a blank in some sampled word.
        blanks = [i for i in range(len(board)) if board[i] == "_"]
        cand_freq: Dict[str, int] = {}
        for w in sample:
            seen = set()
            for i in blanks:
                ch = w[i]
                if ch not in guessed and ch not in seen:
                    seen.add(ch)
                    cand_freq[ch] = cand_freq.get(ch, 0) + 1
        candidates = list(cand_freq)
        if self.max_candidates and len(candidates) > self.max_candidates:
            candidates = sorted(candidates, key=cand_freq.get,
                                reverse=True)[:self.max_candidates]

        # Always include v2's own move so argmax_c Q is provably >= v2 on this sample.
        base_choice = target_to_letter(self.v2.guess_idx([(board, guessed)])[0])
        if base_choice not in guessed and base_choice not in candidates:
            candidates.append(base_choice)

        wrong0 = sum(1 for g in guessed if g not in set(board) - {"_"})

        # Build n*|candidates| games and roll them all out together.
        games: List[_SimGame] = []
        spans: Dict[str, Tuple[int, int]] = {}
        for c in candidates:
            start = len(games)
            for w in sample:
                g = _SimGame(w, board, guessed, wrong0)
                g.apply(c, self.max_wrong)  # forced first guess
                games.append(g)
            spans[c] = (start, len(games))
        self._rollout_batch(games)

        q: Dict[str, float] = {}
        for c, (s, e) in spans.items():
            q[c] = sum(1 for g in games[s:e] if g.won) / max(1, e - s)
        self._last_n = n
        self._last_base = base_choice if base_choice not in guessed else None
        return q

    # ---------------------------------------------------------------- policy
    def best_action(self, board: Sequence[str], guessed: Iterable[str]) -> str:
        q = self.q_values(board, guessed)
        if not q:
            # No belief support (OOV / contradiction) -> defer to v2.
            return target_to_letter(self.v2.guess_idx([(list(board), set(guessed))])[0])
        base = self._last_base
        if base is None or base not in q:
            return max(q, key=q.get)
        q_base = q[base]
        n = max(1, self._last_n)
        thresh = q_base + self.switch_margin * (0.5 / n) ** 0.5
        best_c, best_q = base, q_base
        for c, v in q.items():
            if v > thresh and v > best_q:
                best_c, best_q = c, v
        return best_c

    # ---------------------------------------------------------------- teacher API
    @torch.no_grad()
    def label_state(self, board: Sequence[str], guessed: Iterable[str]):
        """Return (l_ref(26), h(D), allowed(26) bool, q_target(26), has_belief).

        q_target[c] = Q_hat for candidate letters, 0 elsewhere; disallowed letters are 0.
        has_belief is False when the consistent set is empty (no PIMC signal -> the caller
        should skip this state, falling back to the v2 target which the skip head already is).
        """
        board = list(board)
        guessed = set(guessed)
        l_ref, h, allowed = self.v2.features([(board, guessed)])
        l_ref, h, allowed = l_ref[0], h[0], allowed[0]
        q = self.q_values(board, guessed)
        q_target = torch.zeros(26, device=l_ref.device)
        for c, v in q.items():
            if c in CHARS:
                q_target[letter_to_target(c)] = v
        return l_ref, h, allowed, q_target, bool(q)

    @torch.no_grad()
    def label_states_batched(self, states: List[State], chunk: int = 4096):
        """Label MANY root states at once by rolling out all their candidate-games together.

        Returns a list aligned with ``states`` of dicts with keys
        ``l_ref (26), h (D), allowed (26) bool, q_target (26), has_belief (bool)`` (tensors on
        CPU). Roots with no belief support get ``has_belief=False`` and a zero q_target (the
        caller skips them; the skip head already encodes v2 there).

        The whole point: GPU forwards scale with rollout *depth*, not with #roots, so labeling
        is ~ (#roots / batch-of-games) times faster than calling ``label_state`` per root.
        """
        n = len(states)
        l_ref_all, h_all, allowed_all = self.v2.features(states)
        base_scores = l_ref_all.masked_fill(~allowed_all, -float("inf"))
        base_idx = base_scores.argmax(dim=-1).tolist()

        out = [None] * n
        games: List[_SimGame] = []
        meta: List[Tuple[int, str]] = []   # (root_index, candidate_letter)
        counts: Dict[Tuple[int, str], int] = {}

        for i, (board, guessed) in enumerate(states):
            board = list(board); guessed = set(guessed)
            pool = self._consistent(board, guessed)
            base_choice = target_to_letter(base_idx[i])
            rec = dict(l_ref=l_ref_all[i].cpu(), h=h_all[i].cpu(),
                       allowed=allowed_all[i].cpu(),
                       q_target=torch.zeros(26), has_belief=False)
            out[i] = rec
            if not pool:
                continue
            sample = self._sample_belief(pool)
            blanks = [j for j in range(len(board)) if board[j] == "_"]
            cand_freq: Dict[str, int] = {}
            for w in sample:
                seen = set()
                for j in blanks:
                    ch = w[j]
                    if ch not in guessed and ch not in seen:
                        seen.add(ch); cand_freq[ch] = cand_freq.get(ch, 0) + 1
            candidates = list(cand_freq)
            if self.max_candidates and len(candidates) > self.max_candidates:
                candidates = sorted(candidates, key=cand_freq.get,
                                    reverse=True)[:self.max_candidates]
            if base_choice not in guessed and base_choice not in candidates:
                candidates.append(base_choice)
            wrong0 = sum(1 for g in guessed if g not in set(board) - {"_"})
            rec["has_belief"] = True
            for c in candidates:
                counts[(i, c)] = len(sample)
                for w in sample:
                    g = _SimGame(w, board, guessed, wrong0)
                    g.apply(c, self.max_wrong)
                    games.append(g); meta.append((i, c))

        self._rollout_batch(games, chunk=chunk)

        wins: Dict[Tuple[int, str], int] = {}
        for g, key in zip(games, meta):
            if g.won:
                wins[key] = wins.get(key, 0) + 1
        for (i, c), tot in counts.items():
            out[i]["q_target"][letter_to_target(c)] = wins.get((i, c), 0) / max(1, tot)
        return out


# ---------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import time
    from data import load_words, split_words

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _ = split_words(load_words())
    v2 = VectorizedV2(device)
    pimc = FastPIMC(train, v2, n_samples=48, max_candidates=8, seed=0)

    # A mid-game board derived from a real train word.
    w = "consistent"
    board = list("c_ns_st_nt")
    guessed = set("cnst")
    t0 = time.time()
    q = pimc.q_values(board, guessed)
    print("Q sample:", {k: round(v, 3) for k, v in sorted(q.items(), key=lambda x: -x[1])[:6]})
    print("best_action:", pimc.best_action(board, guessed),
          "| v2:", target_to_letter(v2.guess_idx([(board, guessed)])[0]))
    print(f"one root Q in {time.time()-t0:.2f}s  (n_belief={pimc._last_n})")
