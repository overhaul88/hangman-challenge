"""Shared vocabulary constants for the Hangman solver.

Index convention (must match the original strategy so checkpoints stay comparable):
    <PAD> -> 0
    a..z  -> 1..26
    _     -> 27   (mask / unknown position)

Letter *targets* and per-position model *outputs* use the 0..25 range
(i.e. char_idx - 1).
"""
import string

CHARS = string.ascii_lowercase

CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARS)}  # a..z -> 1..26
CHAR_TO_IDX["_"] = 27
CHAR_TO_IDX["<PAD>"] = 0
IDX_TO_CHAR = {idx: ch for ch, idx in CHAR_TO_IDX.items()}

PAD_IDX = 0
MASK_IDX = 27
VOCAB_SIZE = len(CHAR_TO_IDX)  # 28
NUM_LETTERS = 26
IGNORE_INDEX = -100  # CrossEntropy ignore for non-hidden / padded positions
MAX_WRONG = 6        # game allows 6 incorrect guesses


def letter_to_target(ch: str) -> int:
    """Map a lowercase letter to its 0..25 target/output index."""
    return CHAR_TO_IDX[ch] - 1


def target_to_letter(idx: int) -> str:
    """Map a 0..25 output index back to its letter."""
    return IDX_TO_CHAR[idx + 1]
