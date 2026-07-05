"""Episode-level experience replay for the off-policy Dueling Residual DRQN (Strategy v5).

Hangman games are short (<= ~16 guesses), so the buffer stores **whole episodes** rather
than individual transitions: the DRQN replays each game through its LSTM from a zero start
state (no burn-in -- the short-episode simplification of R2D2). Each per-step transition
stores the **frozen-trunk features** of the state where the agent acted
(`l_ref`, `h`, `allowed`) so the expensive trunk forward is never recomputed on replay,
plus the `action` taken, the `reward`, and the `done` flag.

Eviction is FIFO by *transition* count: whenever adding an episode would push the total
number of stored transitions over `capacity_transitions`, the oldest episodes are dropped
until the buffer is within capacity. Episodes are stored on CPU.

`sample` draws `batch_episodes` distinct episodes, right-pads them to the batch's max length
`Tmax`, and returns **time-major** `(Tmax, B, ...)` tensors on `device`, plus:
  * a validity `mask` (1.0 for real steps, 0.0 for padding) used to exclude padded steps
    from the loss, and
  * an episode-start `reset` mask (1.0 at t==0 for every episode) consumed by
    `policy_recurrent.get_states` / the DRQN's `q_sequence` to zero the LSTM state at each
    episode boundary (every sampled episode starts fresh from a zero state).
Time-major layout is required because the network iterates dim 0 as time and treats dim 1
as the batch/env dimension.

Padding (positions t >= episode length) uses `done=1.0` and `mask=0.0`; padded `allowed`
rows are set **all-True** so a downstream masked-mean over allowed actions never divides by
zero (padded steps are excluded from the loss via `mask` regardless).
"""
from collections import deque
from typing import Deque, Dict, List

import torch

# One stored episode: per-step feature/target tensors, all CPU, length T.
Episode = Dict[str, torch.Tensor]


class EpisodeReplayBuffer:
    def __init__(self, capacity_transitions: int, feat_dim: int, n_letters: int = 26):
        self.capacity_transitions = int(capacity_transitions)
        self.feat_dim = int(feat_dim)
        self.n_letters = int(n_letters)
        self._episodes: Deque[Episode] = deque()
        self._n_transitions = 0  # running sum of episode lengths

    def __len__(self) -> int:
        """Total number of stored transitions (sum of episode lengths)."""
        return self._n_transitions

    @property
    def num_episodes(self) -> int:
        """Number of stored episodes."""
        return len(self._episodes)

    def add_episode(self, l_ref, h, allowed, action, reward, done) -> None:
        """Store one episode of length T. All inputs are per-step tensors:
            l_ref   (T, n_letters) float
            h       (T, feat_dim)  float
            allowed (T, n_letters) bool
            action  (T,) long
            reward  (T,) float
            done    (T,) float   (1.0 only at the last step, else 0.0)
        Tensors are detached and moved to CPU. Empty (T==0) episodes are ignored.
        Evicts the oldest episodes (FIFO) until within `capacity_transitions`.
        """
        T = int(l_ref.shape[0])
        if T == 0:
            return
        ep: Episode = {
            "l_ref": l_ref.detach().to("cpu", torch.float32),
            "h": h.detach().to("cpu", torch.float32),
            "allowed": allowed.detach().to("cpu", torch.bool),
            "action": action.detach().to("cpu", torch.long),
            "reward": reward.detach().to("cpu", torch.float32),
            "done": done.detach().to("cpu", torch.float32),
        }
        self._episodes.append(ep)
        self._n_transitions += T
        self._evict()

    def _evict(self) -> None:
        """Drop oldest episodes until total transitions <= capacity (keep >= 1 episode)."""
        while self._n_transitions > self.capacity_transitions and len(self._episodes) > 1:
            old = self._episodes.popleft()
            self._n_transitions -= int(old["action"].shape[0])

    def can_sample(self, batch_episodes: int) -> bool:
        """True iff at least `batch_episodes` episodes are stored."""
        return self.num_episodes >= batch_episodes

    def sample(self, batch_episodes: int, device) -> Dict[str, torch.Tensor]:
        """Sample `batch_episodes` distinct episodes, right-pad to the batch's max length
        Tmax, and return time-major `(Tmax, B, ...)` tensors on `device` (see module docs)."""
        B = int(batch_episodes)
        idx = torch.randperm(self.num_episodes)[:B].tolist()
        eps: List[Episode] = [self._episodes[i] for i in idx]
        lengths = [int(ep["action"].shape[0]) for ep in eps]
        Tmax = max(lengths)
        nl, fd = self.n_letters, self.feat_dim

        # Padded defaults: zeros for features/action/reward, done=1, mask=0, allowed=all-True.
        l_ref = torch.zeros(Tmax, B, nl, dtype=torch.float32)
        h = torch.zeros(Tmax, B, fd, dtype=torch.float32)
        allowed = torch.ones(Tmax, B, nl, dtype=torch.bool)
        action = torch.zeros(Tmax, B, dtype=torch.long)
        reward = torch.zeros(Tmax, B, dtype=torch.float32)
        done = torch.ones(Tmax, B, dtype=torch.float32)
        mask = torch.zeros(Tmax, B, dtype=torch.float32)
        reset = torch.zeros(Tmax, B, dtype=torch.float32)

        for b, (ep, T) in enumerate(zip(eps, lengths)):
            l_ref[:T, b] = ep["l_ref"]
            h[:T, b] = ep["h"]
            allowed[:T, b] = ep["allowed"]
            action[:T, b] = ep["action"]
            reward[:T, b] = ep["reward"]
            done[:T, b] = ep["done"]
            mask[:T, b] = 1.0
            reset[0, b] = 1.0  # every sampled episode starts a fresh LSTM state

        out = {
            "l_ref": l_ref, "h": h, "allowed": allowed, "action": action,
            "reward": reward, "done": done, "mask": mask, "reset": reset,
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }
        return {k: v.to(device) for k, v in out.items()}


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    torch.manual_seed(0)
    random.seed(0)

    FEAT_DIM, NL, CAP = 415, 26, 50
    buffer = EpisodeReplayBuffer(capacity_transitions=CAP, feat_dim=FEAT_DIM)

    def fake_episode(T: int):
        l_ref = torch.randn(T, NL)
        h = torch.randn(T, FEAT_DIM)
        allowed = torch.rand(T, NL) > 0.4
        allowed[:, 0] = True  # ensure at least one allowed action per real step
        action = torch.randint(0, NL, (T,))
        reward = torch.randn(T)
        done = torch.zeros(T)
        done[-1] = 1.0  # one-hot at terminal step
        return l_ref, h, allowed, action, reward, done

    # 2. Add enough episodes that total transitions exceed capacity -> eviction triggers.
    total_added = 0
    n_added = 0
    while total_added <= CAP + 20:
        T = random.randint(3, 8)
        buffer.add_episode(*fake_episode(T))
        total_added += T
        n_added += 1
    assert len(buffer) <= CAP, f"len {len(buffer)} > cap {CAP}"
    assert buffer.num_episodes < n_added, "FIFO eviction did not shrink episode count"

    # 3. Sample and check structure.
    B = 4
    batch = buffer.sample(batch_episodes=B, device="cpu")
    lengths = batch["lengths"]
    Tmax = int(lengths.max())
    assert lengths.shape == (B,)

    seq_keys = ["l_ref", "h", "allowed", "action", "reward", "done", "mask", "reset"]
    for k in seq_keys:
        assert batch[k].shape[0] == Tmax, f"{k} time dim {batch[k].shape[0]} != Tmax {Tmax}"
        assert batch[k].shape[1] == B, f"{k} batch dim {batch[k].shape[1]} != B {B}"

    # dtypes
    assert batch["allowed"].dtype == torch.bool
    assert batch["action"].dtype == torch.long
    for k in ["l_ref", "h", "reward", "done", "mask", "reset"]:
        assert batch[k].dtype == torch.float32, f"{k} dtype {batch[k].dtype}"

    # mask row-sums equal sampled lengths
    assert torch.equal(batch["mask"].sum(0).long(), lengths), "mask sums != lengths"

    # reset: 1 at t==0, 0 elsewhere
    assert torch.all(batch["reset"][0] == 1.0)
    assert torch.all(batch["reset"][1:] == 0.0)

    # padded positions: allowed all-True, done==1, mask==0
    for b in range(B):
        T = int(lengths[b])
        if T < Tmax:
            assert torch.all(batch["allowed"][T:, b]), "padded allowed not all-True"
            assert torch.all(batch["done"][T:, b] == 1.0), "padded done != 1"
            assert torch.all(batch["mask"][T:, b] == 0.0), "padded mask != 0"

    # 4. can_sample boundary
    assert buffer.can_sample(buffer.num_episodes) is True
    assert buffer.can_sample(buffer.num_episodes + 1) is False

    shapes = {k: tuple(batch[k].shape) for k in seq_keys + ["lengths"]}
    print("replay self-test PASS")
    print(f"  num_episodes={buffer.num_episodes}  transitions={len(buffer)}  Tmax={Tmax}  B={B}")
    print(f"  shapes={shapes}")
