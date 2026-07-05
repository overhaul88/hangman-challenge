"""Vectorized exact dictionary belief over the train corpus (numpy).

The pure-Python ``lookahead.consistent_words`` scans a whole length-bucket per call, which is
far too slow for a held-out sweep (the belief-sharpening eval calls it once per move). This
module pre-encodes each length bucket once as numpy arrays and answers a consistency query +
letter marginal with a few vectorized ops.

Key simplification for the marginal: an **unguessed** letter can only occur at *blank*
positions (a guessed-correct letter is revealed at all its cells, a guessed-wrong letter is
absent). Hence for unguessed ``c``:

    s_dict(c) = P(c present in the word | belief) = (# consistent words containing c) / N

which is just the column mean of a boolean "contains" matrix over the consistent rows.

Consistency of a word ``w`` (length L) with observation ``(board, guessed)``:
  * fixed positions match:                w[pos] == board[pos]   for revealed positions
  * no absent letter occurs anywhere:     count_w[absent] == 0
  * revealed letters occur exactly as shown (no extra copy hiding in a blank):
                                          count_w[r] == (#times r is fixed on the board)
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

from vocab import CHARS, letter_to_target


class DictBelief:
    def __init__(self, words: List[str]):
        self.by_len: Dict[int, dict] = {}
        groups: Dict[int, List[str]] = {}
        for w in words:
            groups.setdefault(len(w), []).append(w)
        for L, ws in groups.items():
            n = len(ws)
            arr = np.empty((n, L), dtype=np.int8)
            contains = np.zeros((n, 26), dtype=bool)
            count = np.zeros((n, 26), dtype=np.int16)
            for i, w in enumerate(ws):
                codes = [ord(c) - 97 for c in w]
                arr[i] = codes
                for cd in codes:
                    contains[i, cd] = True
                    count[i, cd] += 1
            self.by_len[L] = dict(arr=arr, contains=contains, count=count, words=ws)

    def _mask(self, board: Sequence[str], guessed) -> Tuple[np.ndarray, dict]:
        """Boolean consistency mask over the length bucket (or (None, None))."""
        L = len(board)
        g = self.by_len.get(L)
        if g is None:
            return None, None
        arr, contains, count = g["arr"], g["contains"], g["count"]
        guessed = set(guessed)
        revealed = set(board) - {"_"}
        absent = set(c for c in guessed if c in CHARS) - revealed

        mask = np.ones(arr.shape[0], dtype=bool)
        need: Dict[str, int] = {}
        for pos in range(L):
            ch = board[pos]
            if ch != "_":
                mask &= arr[:, pos] == (ord(ch) - 97)
                need[ch] = need.get(ch, 0) + 1
        if not mask.any():
            return mask, g
        for ch in absent:
            mask &= ~contains[:, ord(ch) - 97]
        for ch, c in need.items():
            mask &= count[:, ord(ch) - 97] == c
        return mask, g

    def query(self, board: Sequence[str], guessed) -> Tuple[np.ndarray, int]:
        """Return (s_dict (26,) float32, N) for the observation. N==0 -> zeros."""
        mask, g = self._mask(board, guessed)
        if g is None or not mask.any():
            return np.zeros(26, dtype=np.float32), 0
        n = int(mask.sum())
        s = g["contains"][mask].mean(axis=0).astype(np.float32)  # P(letter present)
        for ch in set(guessed):
            if ch in CHARS:
                s[letter_to_target(ch)] = 0.0
        return s, n

    def consistent(self, board: Sequence[str], guessed) -> List[str]:
        """The actual list of train words consistent with the observation (for PIMC)."""
        mask, g = self._mask(board, guessed)
        if g is None or not mask.any():
            return []
        words = g["words"]
        return [words[i] for i in np.nonzero(mask)[0]]


if __name__ == "__main__":
    import time
    from data import load_words, split_words
    train, _ = split_words(load_words())
    t0 = time.time()
    db = DictBelief(train)
    print(f"indexed {len(train)} words in {time.time()-t0:.1f}s")
    t0 = time.time()
    for _ in range(2000):
        s, n = db.query(list("c_ns_st_nt"), set("cnst"))
    print(f"2000 queries in {time.time()-t0:.3f}s  | example N={n} top="
          f"{[(CHARS[i], round(float(s[i]),2)) for i in s.argsort()[::-1][:5]]}")
    s, n = db.query(["_"] * 8, set())
    print(f"opening len8: N={n} top={[(CHARS[i], round(float(s[i]),2)) for i in s.argsort()[::-1][:5]]}")
