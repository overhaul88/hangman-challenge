"""Dataset + data utilities for the enhanced Hangman solver.

Key difference from the original `HangmanDataset`: we mask by *letter*, not by
position. A sampled set of letters is hidden at *all* of its occurrences, exactly
mirroring a real Hangman board (guessing a letter reveals every occurrence at once).
We additionally sample a set of *absent* letters (wrong guesses) and expose them as a
multi-hot "negative evidence" feature, plus a multi-hot of the *revealed* letters.

Each item yields:
    input_ids        (L,)  long  -- board: revealed letters keep their idx, hidden -> MASK
    absent_multihot  (26,) float -- letters known to be absent (wrong guesses)
    present_multihot (26,) float -- letters currently revealed on the board
    target           (L,)  long  -- true letter (0..25) at hidden positions, IGNORE else
"""
import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from vocab import (
    CHARS, CHAR_TO_IDX, MASK_IDX, PAD_IDX, NUM_LETTERS, IGNORE_INDEX,
    letter_to_target,
)

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "dataset", "words_250000_train.txt")


def load_words(path: str = DEFAULT_CORPUS) -> List[str]:
    """Load + clean the corpus: lowercase, alphabetic, length > 2."""
    with open(path, "r") as f:
        words = f.read().splitlines()
    return [w.lower() for w in words if w.isalpha() and len(w) > 2]


def split_words(words: List[str], seed: int = 42, frac: float = 0.8
                ) -> Tuple[List[str], List[str]]:
    """Reproduce the original seed-42 80/20 split so eval sets are comparable."""
    words = list(words)
    rng = random.Random(seed)
    rng.shuffle(words)
    split_idx = int(len(words) * frac)
    return words[:split_idx], words[split_idx:]


class HangmanStateDataset(Dataset):
    """Samples a realistic in-progress board for each word.

    hidden_frac_range: fraction of the word's *unique* letters to hide. A high
        fraction (near 1.0) corresponds to an opening board (mostly blank); a low
        fraction corresponds to a near-solved endgame board.
    max_wrong: upper bound on the number of absent (wrong-guess) letters to sample.
    """

    def __init__(self, words: List[str],
                 hidden_frac_range: Tuple[float, float] = (0.3, 1.0),
                 max_wrong: int = 6):
        self.words = words
        self.hidden_frac_range = hidden_frac_range
        self.max_wrong = max_wrong

    def set_curriculum(self, hidden_frac_range: Tuple[float, float], max_wrong: int):
        self.hidden_frac_range = hidden_frac_range
        self.max_wrong = max_wrong

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: int):
        word = self.words[idx]
        unique = list(set(word))

        # --- choose which letters are hidden (masked at ALL their positions) ---
        lo, hi = self.hidden_frac_range
        frac = random.uniform(lo, hi)
        n_hidden = max(1, round(frac * len(unique)))
        n_hidden = min(n_hidden, len(unique))
        hidden_letters = set(random.sample(unique, n_hidden))
        revealed_letters = set(unique) - hidden_letters

        # --- sample wrong guesses (letters not in the word) as negative evidence ---
        absent_candidates = [c for c in CHARS if c not in unique]
        n_wrong = random.randint(0, self.max_wrong)
        n_wrong = min(n_wrong, len(absent_candidates))
        absent_letters = set(random.sample(absent_candidates, n_wrong)) if n_wrong else set()

        # --- build tensors ---
        input_ids, target = [], []
        for ch in word:
            if ch in hidden_letters:
                input_ids.append(MASK_IDX)
                target.append(letter_to_target(ch))
            else:
                input_ids.append(CHAR_TO_IDX[ch])
                target.append(IGNORE_INDEX)

        absent_multihot = torch.zeros(NUM_LETTERS)
        for ch in absent_letters:
            absent_multihot[letter_to_target(ch)] = 1.0
        present_multihot = torch.zeros(NUM_LETTERS)
        for ch in revealed_letters:
            present_multihot[letter_to_target(ch)] = 1.0

        return (torch.tensor(input_ids, dtype=torch.long),
                absent_multihot,
                present_multihot,
                torch.tensor(target, dtype=torch.long))


def collate_fn(batch):
    """Pad a batch and build the transformer padding mask."""
    inputs, absent, present, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_IDX)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=IGNORE_INDEX)
    absent = torch.stack(absent)
    present = torch.stack(present)
    pad_mask = (inputs_padded == PAD_IDX)  # True where padded -> ignored by attention
    return inputs_padded, absent, present, targets_padded, pad_mask
