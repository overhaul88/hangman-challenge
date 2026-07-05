"""Strategy v7 — dense-reward Hangman environment with a pseudo-OOV word sampler.

strategy7.md §6-§7. Two differences from `rl_env`:

  1. **Dense reward** (Part 5) instead of the sparse terminal reward of v3/v4/v5:
        correct guess revealing k positions      -> +2k
        + that guess resolves the word (final)    -> +10 · lives_remaining / max_lives
        + win (board fully revealed)              -> +20
        wrong guess                               -> -2
        + loss (6th wrong)                        -> -10
        repeated / already-guessed letter         -> -50   (should never fire; Q-masking)
     The win/loss asymmetry (+20 vs -10) keeps the agent from learning a timid loss-avoider.

  2. **Pseudo-OOV sampling**: each episode draws from the pseudo-OOV held-out cluster with
     probability `pseudo_oov_frac` (default 0.2), else from the in-vocab pool. Openings are
     all-blank (deployment rule) — the v7 "curriculum" is over the word *distribution*, not
     partial reveals.

Reuses `rl_env.Obs` so the (board, guessed) state primitive matches the rest of the system.
Pure Python (stdlib) — no torch.
"""
import random
from typing import List, Optional, Sequence, Tuple

from vocab import MAX_WRONG
from rl_env import Obs


class DenseHangmanEnv:
    """Single-game env with the v7 dense reward and an 80/20 in-vocab/pseudo-OOV sampler."""

    def __init__(self, in_vocab: Sequence[str], pseudo_oov: Sequence[str],
                 pseudo_oov_frac: float = 0.2, max_wrong: int = MAX_WRONG,
                 seed: Optional[int] = None):
        if not in_vocab:
            raise ValueError("DenseHangmanEnv requires a non-empty in_vocab list")
        self.in_vocab = list(in_vocab)
        self.pseudo_oov = list(pseudo_oov) if pseudo_oov else []
        self.frac = pseudo_oov_frac if self.pseudo_oov else 0.0
        self.max_wrong = max_wrong
        self._rng = random.Random(seed)

        self.word = ""
        self._unique: set = set()
        self.board: List[str] = []
        self.guessed: List[str] = []
        self.wrong = 0
        self._n_guesses = 0

    @property
    def lives(self) -> int:
        return self.max_wrong - self.wrong

    def _obs(self) -> Obs:
        return Obs(board=list(self.board), guessed=list(self.guessed),
                   lives=self.lives, wrong=self.wrong)

    def _sample_word(self) -> str:
        if self.frac and self._rng.random() < self.frac:
            return self._rng.choice(self.pseudo_oov)
        return self._rng.choice(self.in_vocab)

    def reset(self, word: Optional[str] = None) -> Obs:
        self.word = word if word is not None else self._sample_word()
        self._unique = set(self.word)
        self.board = ["_"] * len(self.word)   # all-blank opening (deployment rule)
        self.guessed = []
        self.wrong = 0
        self._n_guesses = 0
        return self._obs()

    def step(self, letter: str) -> Tuple[Obs, float, bool, dict]:
        reward = 0.0

        if letter in self.guessed:
            # Should never happen (the DRQN masks guessed actions); penalise defensively.
            reward += -50.0
        else:
            self.guessed.append(letter)
            self._n_guesses += 1
            if letter in self._unique:
                k = sum(1 for c in self.word if c == letter)   # newly revealed positions
                for i, c in enumerate(self.word):
                    if c == letter:
                        self.board[i] = letter
                reward += 2.0 * k
                if "_" not in self.board:  # this correct guess resolved the word
                    reward += 10.0 * (self.lives / self.max_wrong)
            else:
                self.wrong += 1
                reward += -2.0

        win = "_" not in self.board
        loss = self.wrong >= self.max_wrong
        done = win or loss
        if win:
            reward += 20.0
        elif loss:
            reward += -10.0

        info = {"win": win, "wrong": self.wrong, "lives": self.lives}
        if done:
            info["word"] = self.word
            info["episode_len"] = self._n_guesses
        return self._obs(), reward, done, info


class VecDenseHangmanEnv:
    """Parallel DenseHangmanEnvs with standard autoreset on terminal (mirrors VecHangmanEnv)."""

    def __init__(self, in_vocab: Sequence[str], pseudo_oov: Sequence[str], n_envs: int,
                 pseudo_oov_frac: float = 0.2, max_wrong: int = MAX_WRONG, seed: int = 0):
        self.n_envs = n_envs
        self.envs = [
            DenseHangmanEnv(in_vocab, pseudo_oov, pseudo_oov_frac=pseudo_oov_frac,
                            max_wrong=max_wrong, seed=seed + i)
            for i in range(n_envs)
        ]

    def reset(self) -> List[Obs]:
        return [e.reset() for e in self.envs]

    def step(self, letters: Sequence[str]
             ) -> Tuple[List[Obs], List[float], List[bool], List[dict]]:
        obs_list, rewards, dones, infos = [], [], [], []
        for env, letter in zip(self.envs, letters):
            obs, reward, done, info = env.step(letter)
            rewards.append(reward)
            dones.append(done)
            if done:
                info["terminal"] = True
                infos.append(info)
                obs = env.reset()       # autoreset; expose the fresh initial Obs
            else:
                infos.append({"terminal": False})
            obs_list.append(obs)
        return obs_list, rewards, dones, infos


if __name__ == "__main__":
    env = VecDenseHangmanEnv(["apple", "banana", "cherry"], ["station", "kindness"],
                             n_envs=3, pseudo_oov_frac=0.5, seed=0)
    obs = env.reset()
    assert len(obs) == 3
    # play 'a' everywhere once
    obs, r, d, info = env.step(["a", "a", "a"])
    assert len(r) == 3
    print("env_v7 self-test PASS  rewards=", [round(x, 2) for x in r])
