"""Online Hangman client using the enhanced ensemble policy.

The server plumbing (URL selection, request/retry, game loop) is reused verbatim from
the original notebook. Only `guess()` is replaced. The default policy blends the new
negative-evidence-aware char Transformer encoder with the original gated MoE at the
score level (`alpha*encoder + (1-alpha)*moe`, best offline at alpha=0.3). Policies:
  - ensemble : encoder + MoE blend (best; default)
  - encoder  : encoder only
  - moe      : original MoE only
If neither model loads it falls back to the original dictionary-frequency policy.

The access token is read from the project-root `.env` (key `TREXQUANT_API`) by default,
or can be overridden with `--token`.

Usage:
    python play_online.py --games 100 [--policy ensemble] [--alpha 0.3] [--practice 1]
    python play_online.py --token YOUR_TOKEN --games 100   # explicit override
"""
import argparse
import collections
import json
import os
import re
import string
import time

import requests
import torch

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

try:
    from urllib.parse import parse_qs
except ImportError:
    from urlparse import parse_qs

from vocab import CHAR_TO_IDX, MASK_IDX, NUM_LETTERS, letter_to_target, target_to_letter
from model import HangmanEncoder
from evaluate import _aggregate, load_encoder
from ensemble import encoder_scores, _moe_scorer
from config import get_token

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
CORPUS = os.path.join(BASE_DIR, "..", "dataset", "words_250000_train.txt")


class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None,
                 ckpt=None, policy="ensemble", alpha=0.3, agg="mean"):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []

        self.full_dictionary = self.build_dictionary(CORPUS)
        self.full_dictionary_common_letter_sorted = collections.Counter(
            "".join(self.full_dictionary)).most_common()
        self.current_dictionary = []

        # --- models ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy
        self.alpha = alpha
        self.agg = agg
        self.model = None
        self.moe_score = None
        ckpt = ckpt or os.path.join(MODELS_DIR, "hangman_encoder.pt")
        try:
            if policy in ("ensemble", "encoder"):
                self.model = load_encoder(ckpt, self.device)
                print(f"Loaded encoder from {ckpt}")
            if policy in ("ensemble", "moe"):
                self.moe_score = _moe_scorer(self.device)
                print("Loaded MoE experts + gate")
        except Exception as e:
            print(f"[WARN] models not loaded ({e}); using dictionary fallback.")
            self.model = None
            self.moe_score = None

    # -------------------- guess policy --------------------
    @torch.no_grad()
    def guess(self, word):  # word like "_ p p _ e "
        if self.model is None and self.moe_score is None:
            return self._baseline_guess(word)

        board = list(re.sub(r"\s+", "", word.strip()))
        board = [c.lower() if c != "_" else c for c in board]
        guessed = set(self.guessed_letters)

        # Build the blended 26-dim letter score for the current policy.
        scores = None
        if self.model is not None:  # encoder component
            enc = encoder_scores(self.model, board, guessed, self.device, self.agg)
            scores = enc if scores is None else scores
        if self.policy == "ensemble":
            moe = self.moe_score(board, guessed)
            scores = self.alpha * enc + (1 - self.alpha) * moe
        elif self.policy == "moe":
            scores = self.moe_score(board, guessed)

        for ch in guessed:
            if ch in string.ascii_lowercase:
                scores[letter_to_target(ch)] = -float("inf")
        if torch.isinf(scores).all():
            for ch in string.ascii_lowercase:
                if ch not in guessed:
                    return ch
            return "e"
        return target_to_letter(int(torch.argmax(scores).item()))

    # -------------------- (legacy single-encoder path kept for reference) --------------------
    @torch.no_grad()
    def _encoder_only_guess(self, word):
        board = list(re.sub(r"\s+", "", word.strip()))
        guessed = set(self.guessed_letters)
        revealed = set(c for c in board if c != "_")
        absent = guessed - revealed

        input_ids = [MASK_IDX if c == "_" else CHAR_TO_IDX.get(c.lower(), MASK_IDX)
                     for c in board]
        input_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        absent_mh = torch.zeros(1, NUM_LETTERS, device=self.device)
        for ch in absent:
            if ch in string.ascii_lowercase:
                absent_mh[0, letter_to_target(ch)] = 1.0
        present_mh = torch.zeros(1, NUM_LETTERS, device=self.device)
        for ch in revealed:
            if ch in string.ascii_lowercase:
                present_mh[0, letter_to_target(ch)] = 1.0

        logits = self.model(input_t, absent_mh, present_mh, pad_mask=None)
        probs = torch.softmax(logits[0], dim=-1)
        hidden_mask = (input_t[0] == MASK_IDX)
        scores = _aggregate(probs, hidden_mask, self.agg)
        for ch in guessed:
            if ch in string.ascii_lowercase:
                scores[letter_to_target(ch)] = -float("inf")
        if torch.isinf(scores).all():
            for ch in string.ascii_lowercase:
                if ch not in guessed:
                    return ch
            return "e"
        return target_to_letter(int(torch.argmax(scores).item()))

    # -------------------- dictionary fallback (original) --------------------
    def _baseline_guess(self, word):
        clean_word = word[::2].replace("_", ".")
        len_word = len(clean_word)
        new_dictionary = [w for w in self.current_dictionary
                          if len(w) == len_word and re.match(clean_word, w)]
        self.current_dictionary = new_dictionary
        c = collections.Counter("".join(new_dictionary))
        for letter, _ in c.most_common():
            if letter not in self.guessed_letters:
                return letter
        for letter, _ in self.full_dictionary_common_letter_sorted:
            if letter not in self.guessed_letters:
                return letter
        return "!"

    # -------------------- server plumbing (verbatim from notebook) --------------------
    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com']
        data = {link: 0 for link in links}
        try:
            for link in links:
                requests.get(link, timeout=10)
                for i in range(10):
                    s = time.time()
                    requests.get(link, timeout=10)
                    data[link] = time.time() - s
            link = sorted(data.items(), key=lambda x: x[1])[0][0]
        except Exception as e:
            # No network (e.g. offline dry-run): fall back to the default host.
            print(f"[WARN] URL ping failed ({e}); defaulting to {links[0]}")
            link = links[0]
        return link + '/trexsim/hangman'

    def build_dictionary(self, path):
        with open(path, "r") as f:
            return f.read().splitlines()

    def start_game(self, practice=True, verbose=True):
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
        response = self.request("/new_game", {"practice": practice})
        if response.get('status') == "approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print(f"New game {game_id}. tries={tries_remains}. word={word}")
            while tries_remains > 0:
                guess_letter = self.guess(word)
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print(f"Guessing: {guess_letter}")
                try:
                    res = self.request("/guess_letter",
                                       {"request": "guess_letter", "game_id": game_id,
                                        "letter": guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError on request.')
                    continue
                if verbose:
                    print(f"Server: {res}")
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                if status == "success":
                    if verbose:
                        print(f"Won game {game_id}")
                    return True
                elif status == "failed":
                    if verbose:
                        print(f"Lost game {game_id}: {res.get('reason', 'tries exceeded')}")
                    return False
                elif status == "ongoing":
                    word = res.get('word')
        else:
            print("Failed to start a new game")
        return False

    def my_status(self):
        return self.request("/my_status", {})

    def request(self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"
        if self.access_token:
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token
        time.sleep(0.2)
        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET", self.hangman_url + path, timeout=self.timeout,
                    params=args, data=post_args, verify=False)
                break
            except requests.HTTPError as e:
                response = json.loads(e.read())
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)
        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            result = {"access_token": query_str["access_token"][0]}
            if "expires" in query_str:
                result["expires"] = query_str["expires"][0]
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')
        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result


class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""
        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
            except (KeyError, TypeError):
                self.message = result
        Exception.__init__(self, self.message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None,
                    help="trexsim access token (defaults to TREXQUANT_API in .env)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--practice", type=int, default=1)
    ap.add_argument("--policy", default="ensemble", choices=["ensemble", "encoder", "moe"])
    ap.add_argument("--alpha", type=float, default=0.3, help="encoder weight in ensemble")
    ap.add_argument("--agg", default="mean", choices=["mean", "noisy_or"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    token = args.token or get_token()
    if not token:
        raise SystemExit("No access token: set TREXQUANT_API in .env or pass --token.")

    api = HangmanAPI(access_token=token, timeout=2000, ckpt=args.ckpt,
                     policy=args.policy, alpha=args.alpha, agg=args.agg)
    wins = 0
    for i in range(args.games):
        won = api.start_game(practice=args.practice, verbose=args.verbose)
        wins += int(won)
        print(f"Game {i+1}/{args.games}: {'WON' if won else 'lost'} | running {wins}/{i+1} = {wins/(i+1):.3f}")
        time.sleep(0.5)
    print(f"\nFinal: {wins}/{args.games} = {wins/args.games:.4f}")
    try:
        print("Server status:", api.my_status())
    except Exception as e:
        print("status error:", e)


if __name__ == "__main__":
    main()
