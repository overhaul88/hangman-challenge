"""Curriculum manager for Strategy v4 (Recurrent PPO).

Transfers the phase-based curriculum from the Recurrent-PPO blog and extends it with a
third axis — the *hidden-fraction* of an opening board — to mirror Strategy v2's supervised
curriculum (open from mostly-revealed boards toward all-blank openings).

Each phase specifies:
    word_length_range : (min_len, max_len) words sampled for episodes
    max_attempts      : training-time wrong-guess budget (a "training wheels" knob;
                        annealed down to the true deployment value of 6)
    hidden_frac_range : (lo, hi) fraction of a word's UNIQUE letters hidden at reset
                        (1.0 == all-blank opening; <1.0 == partially revealed opening)

The FINAL phase equals the deployment distribution: all lengths 3-20, budget 6, all-blank
(hidden_frac 1.0). This guarantees the last segment of training matches exactly what the
held-out evaluation (and the live game) presents, so v4's numbers stay comparable to v2/v3.
"""
import json
import os


# (length range, attempt budget, hidden-fraction range) per phase.
DEFAULT_PHASES = {
    1: {"word_length_range": (3, 6),  "max_attempts": 10, "hidden_frac_range": (0.3, 0.7)},
    2: {"word_length_range": (3, 8),  "max_attempts": 9,  "hidden_frac_range": (0.4, 0.8)},
    3: {"word_length_range": (3, 10), "max_attempts": 8,  "hidden_frac_range": (0.5, 0.9)},
    4: {"word_length_range": (3, 14), "max_attempts": 7,  "hidden_frac_range": (0.6, 1.0)},
    5: {"word_length_range": (3, 20), "max_attempts": 6,  "hidden_frac_range": (0.8, 1.0)},
    6: {"word_length_range": (3, 20), "max_attempts": 6,  "hidden_frac_range": (1.0, 1.0)},
}


class Curriculum:
    """Manages curriculum phases with optional JSON state persistence."""

    def __init__(self, phases: dict = None, state_file: str = None):
        self.phases = phases or dict(DEFAULT_PHASES)
        self.current_phase = 1
        self.state_file = state_file
        if state_file:
            self.load_state()

    @property
    def num_phases(self) -> int:
        return len(self.phases)

    def get_current_config(self) -> dict:
        return self.phases[self.current_phase]

    def advance_phase(self):
        if self.current_phase < self.num_phases:
            self.current_phase += 1
            self.save_state()
            return True
        return False

    def regress_phase(self):
        if self.current_phase > 1:
            self.current_phase -= 1
            self.save_state()
            return True
        return False

    def save_state(self):
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({"current_phase": self.current_phase}, f)

    def load_state(self):
        try:
            with open(self.state_file, "r") as f:
                self.current_phase = json.load(f).get("current_phase", 1)
        except (FileNotFoundError, json.JSONDecodeError):
            self.current_phase = 1
