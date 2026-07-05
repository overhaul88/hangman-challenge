"""Curriculum-driven belief-MDP environment for Strategy v4.

`CurriculumHangmanEnv` reuses the verified transition / reward / potential-based-shaping
logic of `rl_env.HangmanEnv` and only changes `reset`, so that episodes are drawn according
to the current `Curriculum` phase:

  * word sampled within the phase's word-length range,
  * wrong-guess budget set from the phase (a training-wheels knob; the final phase uses the
    true deployment value of 6),
  * a *partial-reveal opening* whose difficulty is set by the phase's hidden-fraction range
    (1.0 == all-blank opening == deployment).

`VecCurriculumEnv` runs many such envs in parallel, all sharing ONE `Curriculum` instance so
that a phase advance by the trainer takes effect for every env on its next reset.

Evaluation never uses this env — held-out games are played all-blank with 6 wrong allowed
(see `train_recurrent_ppo.evaluate_recurrent`), so v4's metrics stay comparable to v2/v3.
"""
import random
from typing import List, Optional, Sequence, Tuple

from vocab import MAX_WRONG
from rl_env import HangmanEnv, Obs, VecHangmanEnv
from curriculum import Curriculum


class CurriculumHangmanEnv(HangmanEnv):
    """Single belief-MDP game whose reset() obeys the shared curriculum phase."""

    def __init__(self, words: Sequence[str], curriculum: Curriculum,
                 shaping: str = "progress", shaping_coef: float = 0.1,
                 gamma: float = 0.99, seed: Optional[int] = None):
        super().__init__(words, max_wrong=MAX_WRONG, shaping=shaping,
                         shaping_coef=shaping_coef, gamma=gamma, seed=seed)
        self.curriculum = curriculum
        # Length-bucket the corpus once; cache eligible lists per (min,max) range.
        self.by_len: dict = {}
        for w in self.words:
            self.by_len.setdefault(len(w), []).append(w)
        self._eligible_cache: dict = {}

    def _eligible(self, min_len: int, max_len: int) -> List[str]:
        key = (min_len, max_len)
        pool = self._eligible_cache.get(key)
        if pool is None:
            pool = [w for L, ws in self.by_len.items() if min_len <= L <= max_len for w in ws]
            if not pool:
                pool = list(self.words)  # safety fallback
            self._eligible_cache[key] = pool
        return pool

    def reset(self, word: Optional[str] = None, *, seed=None, options=None) -> Obs:
        cfg = self.curriculum.get_current_config()
        min_len, max_len = cfg["word_length_range"]
        self.max_wrong = cfg["max_attempts"]
        lo, hi = cfg.get("hidden_frac_range", (1.0, 1.0))

        if word is None:
            word = self._rng.choice(self._eligible(min_len, max_len))
        self.word = word
        self._unique = set(word)

        # Partial-reveal opening: hide a (curriculum-controlled) fraction of UNIQUE letters.
        frac = self._rng.uniform(lo, hi)
        n_hidden = max(1, round(frac * len(self._unique)))
        n_hidden = min(n_hidden, len(self._unique))
        hidden = set(self._rng.sample(sorted(self._unique), n_hidden))
        revealed = self._unique - hidden

        self.board = [c if c in revealed else "_" for c in word]
        self.guessed = list(revealed)         # already-correct letters are "guessed"
        self.wrong = 0
        self._n_guesses = len(self.guessed)
        return self._obs()


class VecCurriculumEnv(VecHangmanEnv):
    """Parallel CurriculumHangmanEnvs sharing one Curriculum (reuses VecHangmanEnv.step)."""

    def __init__(self, words: Sequence[str], n_envs: int, curriculum: Curriculum,
                 shaping: str = "progress", shaping_coef: float = 0.1,
                 gamma: float = 0.99, seed: int = 0):
        self.n_envs = n_envs
        self.curriculum = curriculum
        self.envs = [
            CurriculumHangmanEnv(words, curriculum, shaping=shaping,
                                 shaping_coef=shaping_coef, gamma=gamma, seed=seed + i)
            for i in range(n_envs)
        ]
