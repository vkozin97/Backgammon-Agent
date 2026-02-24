from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
import sqlite3
import tempfile
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
    def __init__(self, storage_dir: str | None = None):
        base_dir = Path(storage_dir) if storage_dir else Path(tempfile.gettempdir()) / "backgammon_replay"
        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = base_dir / "replay.sqlite3"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS replay (
                recency_index INTEGER PRIMARY KEY,
                state_vector BLOB NOT NULL,
                state_dim INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                opponent_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                terminal_outcome REAL NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )
        self._conn.commit()
        self._weight_cache: dict[int, list[float]] = {}

    def add(self, **kwargs) -> None:
        state_vector = np.asarray(kwargs["state_vector"], dtype=np.float32)
        self._conn.execute(
            """
            INSERT INTO replay (
                state_vector, state_dim, agent_id, opponent_id, game_id, step_index, epoch, terminal_outcome, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state_vector.tobytes(),
                int(state_vector.size),
                kwargs["agent_id"],
                kwargs["opponent_id"],
                kwargs["game_id"],
                int(kwargs["step_index"]),
                int(kwargs["epoch"]),
                float(kwargs["terminal_outcome"]),
                time.time(),
            ),
        )
        self._conn.commit()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM replay").fetchone()
        return int(row[0] if row else 0)

    def _load_all_items(self) -> list[ReplayItem]:
        rows = self._conn.execute(
            """
            SELECT recency_index, state_vector, state_dim, agent_id, opponent_id, game_id, step_index, epoch, terminal_outcome, timestamp
            FROM replay
            ORDER BY recency_index ASC
            """
        ).fetchall()
        items: list[ReplayItem] = []
        for row in rows:
            vec = np.frombuffer(row[1], dtype=np.float32, count=int(row[2])).copy()
            items.append(
                ReplayItem(
                    recency_index=int(row[0]),
                    state_vector=vec,
                    agent_id=row[3],
                    opponent_id=row[4],
                    game_id=row[5],
                    step_index=int(row[6]),
                    epoch=int(row[7]),
                    terminal_outcome=float(row[8]),
                    timestamp=float(row[9]),
                )
            )
        return items


    def _build_recency_weights(self, count: int, target_center_mass: float = 0.8) -> list[float]:
        if count <= 1:
            return [1.0]
        cached = self._weight_cache.get(count)
        if cached is not None:
            return cached

        x = np.linspace(0.0, 1.0, count, dtype=np.float64)

        def weights_for(beta: float) -> np.ndarray:
            exp_beta = np.exp(beta)
            return (np.exp(beta * x) - 1.0) / (exp_beta - 1.0)

        lo, hi = 1e-6, 32.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            w = weights_for(mid)
            w_sum = float(np.sum(w))
            if w_sum <= 0.0:
                lo = mid
                continue
            com = float(np.sum(x * w) / w_sum)
            if com < target_center_mass:
                lo = mid
            else:
                hi = mid

        weights = weights_for((lo + hi) / 2.0).tolist()
        self._weight_cache[count] = weights
        return weights

    def sample(
        self,
        batch_size: int,
        alpha_recency: float,
        alpha_uniform: float,
        recency_window: int,
        max_samples_per_game_in_batch: int | None = None,
    ) -> list[ReplayItem]:
        all_items = self._load_all_items()
        if not all_items:
            return []

        recency_weights = self._build_recency_weights(len(all_items), target_center_mass=0.8)

        recency_n = int(batch_size * alpha_recency)
        uniform_n = batch_size - recency_n

        weighted_items = random.choices(all_items, weights=recency_weights, k=max(recency_n, 0)) if recency_n > 0 else []
        uniform_items = random.choices(all_items, k=max(uniform_n, 0)) if uniform_n > 0 else []

        mix = weighted_items + uniform_items
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

    def get_meta(self) -> dict[str, int | str]:
        row = self._conn.execute("SELECT MAX(recency_index) FROM replay").fetchone()
        counter = int(row[0]) + 1 if row and row[0] is not None else 0
        return {"size": len(self), "counter": counter, "db_path": str(self.db_path)}

    def close(self) -> None:
        self._conn.close()
