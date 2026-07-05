"""Belief-MDP simulator for Hangman (Strategy v3) — what PPO rolls out.

The game: a hidden word; each turn the agent guesses one letter. A correct guess
reveals ALL of that letter's positions at once; a wrong guess costs a life. The
episode ends in a loss at ``MAX_WRONG`` (==6) wrong guesses, or a win when the
board has no blanks left.

The observation exposes the codebase's state primitive ``(board, guessed)`` used by
the rest of the system (`rl_features.py`, `rl_policy.py`):
    board   : list[str] of length L, revealed letters in place, "_" for blanks.
    guessed : list[str] of letters already guessed, in guess order (hits and misses).

Reward = base + potential-based shaping:
    base    : +1.0 on a winning terminal transition, else 0.0.
    shaping : F = gamma*Phi(next) - Phi(curr), with
              Phi(o) = shaping_coef * (#unique word letters revealed) / (#unique word letters)
              and Phi(terminal) = 0 (both win and loss). Being potential-based with a
              zeroed terminal potential, this leaves the optimal policy unchanged.

Knowing the hidden word to compute Phi is privileged TRAIN-TIME info; the agent's
policy only ever sees the board, never the word.

Pure Python (stdlib only) — no torch.
"""
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from vocab import MAX_WRONG


@dataclass
class Obs:
    """A single belief-MDP observation; ``state`` is the downstream primitive."""
    board: List[str]      # list[str], '_' for blanks
    guessed: List[str]    # list[str], in order of guesses
    lives: int            # MAX_WRONG - wrong
    wrong: int

    @property
    def state(self) -> Tuple[List[str], List[str]]:
        """The (board, guessed) primitive consumed downstream."""
        return (self.board, self.guessed)


class HangmanEnv:
    """A single-game Hangman environment with potential-based reward shaping."""

    def __init__(self, words: Sequence[str], max_wrong: int = MAX_WRONG,
                 shaping: str = "progress", shaping_coef: float = 0.1,
                 gamma: float = 0.99, seed: Optional[int] = None):
        if not words:
            raise ValueError("HangmanEnv requires a non-empty list of words")
        self.words = list(words)
        self.max_wrong = max_wrong
        self.shaping = shaping
        self.shaping_coef = shaping_coef
        self.gamma = gamma
        self._rng = random.Random(seed)

        # Mutable per-episode state (set in reset()).
        self.word: str = ""
        self._unique: set = set()      # unique letters in the hidden word
        self.board: List[str] = []
        self.guessed: List[str] = []
        self.wrong: int = 0
        self._n_guesses: int = 0       # total guesses (= len of useful action stream)

    # ------------------------------------------------------------------ helpers
    @property
    def lives(self) -> int:
        return self.max_wrong - self.wrong

    def _potential(self) -> float:
        """Phi(curr) = coef * (#unique word letters revealed) / (#unique word letters)."""
        if self.shaping == "none" or not self._unique:
            return 0.0
        revealed = sum(1 for c in self._unique if c in self.board)
        return self.shaping_coef * revealed / len(self._unique)

    def _obs(self) -> Obs:
        return Obs(board=list(self.board), guessed=list(self.guessed),
                   lives=self.lives, wrong=self.wrong)

    def _is_solved(self) -> bool:
        return "_" not in self.board

    # ------------------------------------------------------------------ API
    def reset(self, word: Optional[str] = None) -> Obs:
        """Start a new game; sample a word if none given. Returns the initial Obs."""
        self.word = word if word is not None else self._rng.choice(self.words)
        self._unique = set(self.word)
        self.board = ["_"] * len(self.word)
        self.guessed = []
        self.wrong = 0
        self._n_guesses = 0
        return self._obs()

    def step(self, letter: str) -> Tuple[Obs, float, bool, dict]:
        """Apply a single-letter guess; return (obs, reward, done, info)."""
        phi_curr = self._potential()

        # Robust to a repeated guess: no new information, no double counting.
        new_guess = letter not in self.guessed
        if new_guess:
            self.guessed.append(letter)
            self._n_guesses += 1
            if letter in self._unique:
                for i, c in enumerate(self.word):
                    if c == letter:
                        self.board[i] = letter
            else:
                self.wrong += 1

        win = self._is_solved()
        done = win or (self.wrong >= self.max_wrong)

        # Potential of the resulting state; zeroed on any terminal transition so the
        # shaping is optimality-preserving for episodic/absorbing terminals.
        phi_next = 0.0 if done else self._potential()
        if self.shaping == "none":
            shaping = 0.0
        else:
            shaping = self.gamma * phi_next - phi_curr

        base = 1.0 if win else 0.0
        reward = base + shaping

        info = {"win": win, "wrong": self.wrong, "lives": self.lives}
        if done:
            info["word"] = self.word
            info["episode_len"] = self._n_guesses
        return self._obs(), reward, done, info


class VecHangmanEnv:
    """Vectorized parallel HangmanEnvs with standard autoreset on terminal."""

    def __init__(self, words: Sequence[str], n_envs: int, max_wrong: int = MAX_WRONG,
                 shaping: str = "progress", shaping_coef: float = 0.1,
                 gamma: float = 0.99, seed: int = 0):
        self.n_envs = n_envs
        # Distinct per-env seed so sub-envs don't sample identical word streams.
        self.envs = [
            HangmanEnv(words, max_wrong=max_wrong, shaping=shaping,
                       shaping_coef=shaping_coef, gamma=gamma, seed=seed + i)
            for i in range(n_envs)
        ]

    def reset(self) -> List[Obs]:
        return [env.reset() for env in self.envs]

    def step(self, letters: Sequence[str]
             ) -> Tuple[List[Obs], List[float], List[bool], List[dict]]:
        """Step every sub-env; autoreset any that terminated.

        For a terminated env, ``infos[i]`` holds its terminal stats with
        ``terminal=True`` and ``obs_list[i]`` is the fresh post-reset Obs (standard
        vec-env autoreset). Non-terminal envs get ``infos[i] = {'terminal': False}``.
        """
        obs_list: List[Obs] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[dict] = []
        for env, letter in zip(self.envs, letters):
            obs, reward, done, info = env.step(letter)
            rewards.append(reward)
            dones.append(done)
            if done:
                info["terminal"] = True
                infos.append(info)
                obs = env.reset()  # autoreset; expose the fresh initial Obs
            else:
                infos.append({"terminal": False})
            obs_list.append(obs)
        return obs_list, rewards, dones, infos
