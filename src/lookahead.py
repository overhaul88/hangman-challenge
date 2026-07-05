"""Determinized Monte-Carlo (PIMC) one-step lookahead for Strategy v3.

This is the model-based *policy improvement operator*. Given the current game
state ``(board, guessed)`` and a BASE policy ``guess_fn(board, guessed) -> letter``
(Strategy v2), it estimates, for every candidate next letter ``c``, the action-value

    Q_hat(o, c) = P(win | guess c now, then follow the base policy to the end),

by sampling words from the *belief support* -- the corpus words consistent with the
observed board and guesses -- and rolling each one out with the first guess forced to
``c`` and all subsequent guesses chosen by ``guess_fn``. By the Policy Improvement
Theorem the greedy operator ``argmax_c Q_hat`` is provably no worse than the base
policy, so ``best_action`` is itself a valid ``guess_fn`` and serves as the offline
"ceiling" benchmark, a source of value labels for a critic, and a distillation target.

It is deliberately too slow to deploy live.

Conventions (see ``src/data.py`` / ``src/vocab.py``):
    State primitive ``(board, guessed)``: ``board`` is ``list[str]`` length L
    ('_' for blanks), ``guessed`` is an iterable of guessed letters.
"""
import random
from typing import Callable, Dict, Iterable, List, Sequence

from vocab import CHARS, MAX_WRONG

GuessFn = Callable[[Sequence[str], Iterable[str]], str]


# ----------------------------------------------------------------------------
# 1. Belief support (consistency filter)
# ----------------------------------------------------------------------------
def consistent_words(corpus_by_len: Dict[int, List[str]],
                     board: Sequence[str], guessed: Iterable[str]) -> List[str]:
    """Words from the corpus consistent with the observation.

    A guessed letter reveals ALL of its occurrences, so a letter that is revealed
    anywhere cannot appear at any *blank* position, and an absent (wrong-guess)
    letter cannot appear at all. Concretely, with
        revealed = set(board) - {'_'}        (correctly-guessed letters)
        absent   = set(guessed) - revealed   (wrong guesses)
    a word ``w`` of length L is consistent iff for every position i:
        - if board[i] != '_':  w[i] == board[i]
        - else:                w[i] not in (revealed | absent)
    The blank rule subsumes "the word contains none of the absent letters".

    ``corpus_by_len`` is a dict ``L -> list[str]`` (build once for speed).
    """
    L = len(board)
    words = corpus_by_len.get(L)
    if not words:
        return []

    revealed = set(board) - {"_"}
    absent = set(guessed) - revealed

    # Revealed (fixed) positions; blank positions are handled via the count rule below.
    fixed = [(i, board[i]) for i in range(L) if board[i] != "_"]

    # A revealed letter is shown at ALL its occurrences, so a consistent word must
    # contain it EXACTLY as many times as it is fixed on the board (no extra copy may
    # hide in a blank). Pre-tally the required count per revealed letter.
    need_count = {r: sum(1 for _, ch in fixed if ch == r) for r in revealed}

    out = []
    for w in words:
        # Fixed positions must match. (Cheap, fails fast for most candidates.)
        ok = True
        for i, ch in fixed:
            if w[i] != ch:
                ok = False
                break
        if not ok:
            continue
        # No absent letter may appear anywhere.
        if absent and not absent.isdisjoint(w):
            continue
        # Each revealed letter must appear exactly its fixed-count times (no blank copy).
        for r, c in need_count.items():
            if w.count(r) != c:
                ok = False
                break
        if ok:
            out.append(w)
    return out


# ----------------------------------------------------------------------------
# 2. Rollout
# ----------------------------------------------------------------------------
def rollout_win(word: str, board: Sequence[str], guessed: Iterable[str],
                first_letter: str, guess_fn: GuessFn,
                max_wrong: int = MAX_WRONG) -> bool:
    """Play out a KNOWN word from ``(board, guessed)`` and report whether it wins.

    The first guess is forced to ``first_letter``; every subsequent guess is taken
    from ``guess_fn(board, guessed)``. Operates on copies and never mutates inputs.
    Returns True iff the board is fully revealed before exhausting ``max_wrong``.
    """
    board = list(board)
    guessed = set(guessed)
    # Count wrong guesses already on the record (guessed but never revealed).
    revealed = set(board) - {"_"}
    wrong = sum(1 for g in guessed if g not in revealed)

    g = first_letter
    while wrong < max_wrong and "_" in board:
        if g in guessed:
            break  # base policy repeated a guess (no progress possible) -> stop
        guessed.add(g)
        if g in word:
            for i, c in enumerate(word):
                if c == g:
                    board[i] = g
        else:
            wrong += 1
        if "_" not in board or wrong >= max_wrong:
            break
        g = guess_fn(board, guessed)
    return "_" not in board


# ----------------------------------------------------------------------------
# 3. The operator
# ----------------------------------------------------------------------------
class PIMCLookahead:
    """Perfect-Information-Monte-Carlo one-step lookahead policy-improvement operator.

    ``best_action`` has the ``guess_fn(board, guessed) -> letter`` signature, so it can
    be scored directly by ``evaluate.evaluate_winrate`` as the provable-improvement
    ceiling over the supplied base policy.
    """

    def __init__(self, corpus_words: Iterable[str], guess_fn: GuessFn,
                 max_wrong: int = MAX_WRONG, n_samples: int = 64,
                 max_candidates: int = 0, switch_margin: float = 0.0, seed: int = 0):
        """``max_candidates`` (>0) prunes the candidate set to the top-K letters by
        belief-marginal frequency (how many sampled words contain the letter at a blank)
        before rolling out -- the argmax is overwhelmingly among the most frequent
        candidates, so this trades a little accuracy for a large speedup. 0 = no pruning.

        ``switch_margin`` (>=0) makes ``best_action`` conservative against finite-sample
        maximization bias: it leaves the base policy's move only when some candidate beats
        it by more than ``switch_margin`` standard errors of the Q difference. With noisy
        Q estimates a plain argmax systematically over-picks letters whose Q was inflated
        by chance, which can push the lookahead *below* the base policy and break the
        improvement guarantee. The margin restores "no worse than base" at the cost of
        switching less aggressively. 0.0 == faithful argmax_c Q_hat (with base tie-break).
        """
        self.guess_fn = guess_fn
        self.max_wrong = max_wrong
        self.n_samples = n_samples
        self.max_candidates = max_candidates
        self.switch_margin = switch_margin
        self.rng = random.Random(seed)

        # Build the length-bucketed corpus once.
        self.corpus_by_len: Dict[int, List[str]] = {}
        for w in corpus_words:
            self.corpus_by_len.setdefault(len(w), []).append(w)

    def _sample_belief(self, pool: List[str]) -> List[str]:
        """Draw up to ``n_samples`` words from the consistent set.

        Without replacement when the pool is large enough, otherwise with
        replacement (which here is just the whole pool, since sampling k>=len
        with replacement adds no information beyond using every word once)."""
        if len(pool) <= self.n_samples:
            return list(pool)
        return self.rng.sample(pool, self.n_samples)

    def q_values(self, board: Sequence[str], guessed: Iterable[str]) -> Dict[str, float]:
        """Estimate Q_hat(o, c) for each unguessed candidate letter.

        Candidates are the unguessed letters that appear at some blank position in at
        least one consistent word. Letters absent from every consistent word are
        guaranteed misses (Q == 0) and are omitted.

        Each Q[c] is the mean of ``rollout_win`` over the sampled belief words with the
        first guess forced to ``c``.

        Fallback: if the consistent set is EMPTY (e.g. a held-out word not in the
        corpus, or contradictory observations), there is no belief to roll out, so we
        return ``{}``. ``best_action`` / ``value`` then defer to the base policy. This
        keeps the operator from fabricating Q estimates with no support.
        """
        guessed = set(guessed)
        pool = consistent_words(self.corpus_by_len, board, guessed)
        if not pool:
            return {}

        sample = self._sample_belief(pool)

        # Candidate letters: unguessed letters occupying a blank in some SAMPLED word.
        # We enumerate over the sample (not the full pool) because Q is estimated over the
        # sample: a letter never appearing at a blank in any sampled word would score Q==0
        # anyway, so restricting to the sample is both consistent and far cheaper than
        # scanning a potentially huge belief set.
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
            # Keep the K letters most frequently appearing at a blank in the belief.
            candidates = sorted(candidates, key=cand_freq.get,
                                reverse=True)[:self.max_candidates]

        # Pruning must never drop the base policy's own move, or argmax_c Q could score
        # below the base action and break the policy-improvement guarantee. Always include
        # the base choice so best_action is provably >= the base policy on the same sample.
        base_choice = self.guess_fn(board, guessed)
        if base_choice not in guessed and base_choice not in candidates:
            candidates.append(base_choice)

        q: Dict[str, float] = {}
        n = len(sample)
        for c in candidates:
            wins = 0
            for w in sample:
                if rollout_win(w, board, guessed, c, self.guess_fn, self.max_wrong):
                    wins += 1
            q[c] = wins / n
        # Stash the sample size and base move so best_action can apply switch_margin
        # without recomputing the belief.
        self._last_n = n
        self._last_base = base_choice if base_choice not in guessed else None
        return q

    def best_action(self, board: Sequence[str], guessed: Iterable[str]) -> str:
        """argmax_c Q_hat; falls back to the base policy when no belief exists.

        The base policy's own move is the incumbent and is kept unless another candidate's
        estimated Q exceeds it by more than ``switch_margin`` standard errors of the
        difference. With ``switch_margin == 0`` this is a faithful argmax that merely
        tie-breaks toward the base action (so best_action is >= base on the shared sample);
        with ``switch_margin > 0`` it additionally guards against finite-sample
        maximization bias, where a plain argmax over noisy estimates over-picks chance-
        inflated candidates and can fall below the base policy.

        Has the ``guess_fn`` signature, so it can be passed straight to
        ``evaluate.evaluate_winrate`` as the PIMC policy."""
        q = self.q_values(board, guessed)
        if not q:
            return self.guess_fn(board, guessed)

        base_choice = getattr(self, "_last_base", None)
        if base_choice is None or base_choice not in q:
            return max(q, key=q.get)

        q_base = q[base_choice]
        # SE of a difference of two Bernoulli sample-means over n draws is bounded by
        # sqrt(0.5 / n); require a candidate to clear q_base by switch_margin * SE.
        n = max(1, getattr(self, "_last_n", 1))
        thresh = q_base + self.switch_margin * (0.5 / n) ** 0.5
        best_c, best_q = base_choice, q_base
        for c, v in q.items():
            if v > thresh and v > best_q:
                best_c, best_q = c, v
        return best_c

    def value(self, board: Sequence[str], guessed: Iterable[str]) -> float:
        """One-step-improved value estimate: max_c Q_hat (0.0 when no belief exists)."""
        q = self.q_values(board, guessed)
        if not q:
            return 0.0
        return max(q.values())
