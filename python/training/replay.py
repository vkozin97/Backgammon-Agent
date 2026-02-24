from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random
import time

import numpy as np


@dataclass
class ReplayItem:
    state_vector: np.ndarray
    agent_id: str
    opponent_id: str
    game_id: str
    step_index: int
    epoch: int
    terminal_outcome: float
    recency_index: int
    timestamp: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: deque[ReplayItem] = deque(maxlen=capacity)
        self._counter = 0

    def add(self, **kwargs) -> None:
        self.items.append(ReplayItem(recency_index=self._counter, timestamp=time.time(), **kwargs))
        self._counter += 1

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, batch_size: int, alpha_recency: float, alpha_uniform: float, recency_window: int,
               max_samples_per_game_in_batch: int | None = None) -> list[ReplayItem]:
        all_items = list(self.items)
        if not all_items:
            return []
        recency_n = int(batch_size * alpha_recency)
        uniform_n = batch_size - recency_n
        max_epoch = max(x.epoch for x in all_items)
        recency_pool = [x for x in all_items if x.epoch >= max_epoch - recency_window + 1]
        recency = random.choices(recency_pool or all_items, k=min(recency_n, len(recency_pool or all_items)))
        uniform = random.choices(all_items, k=min(uniform_n, len(all_items)))
        mix = recency + uniform
        if len(mix) < batch_size:
            mix += random.choices(all_items, k=batch_size - len(mix))
        if max_samples_per_game_in_batch is None:
            return mix[:batch_size]
        counts: Counter[str] = Counter()
        filtered: list[ReplayItem] = []
        for item in mix:
            if counts[item.game_id] >= max_samples_per_game_in_batch:
                continue
            filtered.append(item)
            counts[item.game_id] += 1
        while len(filtered) < batch_size and all_items:
            c = random.choice(all_items)
            if counts[c.game_id] < max_samples_per_game_in_batch:
                filtered.append(c)
                counts[c.game_id] += 1
            else:
                break
        return filtered[:batch_size]
