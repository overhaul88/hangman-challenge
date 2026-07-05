"""Strategy v7 — prioritised episode replay that keeps the encoder gradient alive.

strategy7.md §5/§10 (the load-bearing caveat). v7's whole thesis is that the RL signal
reshapes the encoder's top blocks. That only works if `p_enc` is **recomputed with grad
at train time** — so this buffer stores the *raw encoder inputs* per step
(`input_ids`, `absent`, `present`) rather than a detached `p_enc`. The frozen MoE belief
`p_moe` never changes, so it is cached. Storing a detached `p_enc` here would silently sever
the gradient and collapse v7 back to a v5-style frozen-belief DRQN.

Per stored step (all CPU; L = the episode's fixed word length):
    input_ids (T,L) long   -- encoder board tokens (MASK at blanks)
    absent    (T,26) float -- wrong-guess multi-hot (encoder negative evidence)
    present   (T,26) float -- revealed-letter multi-hot
    p_moe     (T,26) float -- cached frozen MoE belief
    action    (T,)  long
    reward    (T,)  float
    done      (T,)  float   (1.0 only at terminal step)

Sampling is proportional PER at the **episode** level: P(ep) ∝ priority^alpha, where the
priority is the episode's mean |TD| (seeded to the running max for freshly added episodes),
with importance-sampling weights w ∝ (N·P)^(-beta) / max_w to debias the gradient.
Eviction is FIFO by transition count.
"""
from typing import Dict, List

import numpy as np
import torch

Episode = Dict[str, torch.Tensor]


class PrioritizedEpisodeReplay:
    def __init__(self, capacity_transitions: int, alpha: float = 0.6, eps: float = 1e-4):
        self.capacity = int(capacity_transitions)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._eps: List[Episode] = []
        self._prio: List[float] = []
        self._n = 0          # total transitions
        self._max_prio = 1.0

    def __len__(self) -> int:
        return self._n

    @property
    def num_episodes(self) -> int:
        return len(self._eps)

    def add_episode(self, ep: Episode) -> None:
        T = int(ep["action"].shape[0])
        if T == 0:
            return
        cpu_ep = {k: v.detach().to("cpu") for k, v in ep.items()}
        self._eps.append(cpu_ep)
        self._prio.append(self._max_prio)   # new episodes sampled eagerly
        self._n += T
        self._evict()

    def _evict(self) -> None:
        while self._n > self.capacity and len(self._eps) > 1:
            old = self._eps.pop(0)
            self._prio.pop(0)
            self._n -= int(old["action"].shape[0])

    def can_sample(self, batch_episodes: int) -> bool:
        return self.num_episodes >= batch_episodes

    def sample(self, batch_episodes: int, beta: float, device):
        """Return (episodes_on_device, indices np.ndarray, is_weights tensor(B,) on device)."""
        prio = np.asarray(self._prio, dtype=np.float64)
        probs = prio ** self.alpha
        probs /= probs.sum()
        idx = np.random.choice(len(self._eps), size=batch_episodes, replace=False, p=probs)

        weights = (len(self._eps) * probs[idx]) ** (-beta)
        weights /= weights.max()
        w = torch.tensor(weights, dtype=torch.float32, device=device)

        episodes = [{k: v.to(device) for k, v in self._eps[i].items()} for i in idx]
        return episodes, idx, w

    def update_priorities(self, idx: np.ndarray, priorities) -> None:
        for i, p in zip(idx.tolist(), priorities):
            val = float(p) + self.eps
            self._prio[i] = val
            if val > self._max_prio:
                self._max_prio = val


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    buf = PrioritizedEpisodeReplay(capacity_transitions=60, alpha=0.6)

    def fake(T, L):
        return {
            "input_ids": torch.randint(1, 28, (T, L)),
            "absent": (torch.rand(T, 26) > 0.8).float(),
            "present": (torch.rand(T, 26) > 0.8).float(),
            "p_moe": torch.softmax(torch.randn(T, 26), -1),
            "action": torch.randint(0, 26, (T,)),
            "reward": torch.randn(T),
            "done": torch.tensor([0.0] * (T - 1) + [1.0]),
        }

    total = 0
    while total <= 80:
        T, L = np.random.randint(3, 8), np.random.randint(4, 9)
        buf.add_episode(fake(T, L))
        total += T
    assert len(buf) <= 60, f"eviction failed: {len(buf)}"

    eps, idx, w = buf.sample(4, beta=0.4, device="cpu")
    assert len(eps) == 4 and w.shape == (4,)
    assert all("input_ids" in e for e in eps)
    buf.update_priorities(idx, [1.0, 2.0, 0.5, 3.0])
    print("per_replay self-test PASS  episodes=%d transitions=%d max_prio=%.2f"
          % (buf.num_episodes, len(buf), buf._max_prio))
