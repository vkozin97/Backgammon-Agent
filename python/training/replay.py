from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import tempfile
import time

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        storage_dir: str | None = None,
        recency_decay: float = 0.9,
        replay_window_epochs: int = 20,
        clear_existing: bool = False,
    ):
        base_dir = Path(storage_dir) if storage_dir else Path(tempfile.gettempdir()) / "backgammon_replay"
        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = base_dir / "replay.sqlite3"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
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
                terminal_outcome BLOB NOT NULL,
                terminal_outcome_dim INTEGER NOT NULL,
                action_meta TEXT,
                match_length INTEGER,
                match_agent_1_id TEXT,
                match_agent_2_id TEXT,
                match_number INTEGER,
                game_number_in_match INTEGER,
                final_dave_value INTEGER,
                final_reward_value INTEGER,
                timestamp REAL NOT NULL
            )
            """
        )
        self._ensure_optional_columns()
        self._conn.commit()
        self._conn.execute("PRAGMA busy_timeout=5000")
        if clear_existing:
            self._conn.execute("DELETE FROM replay")
            self._conn.commit()

        self._rng = np.random.default_rng()
        self._recency_decay = float(recency_decay)
        self._replay_window_epochs = max(int(replay_window_epochs), 1)
        self._commit_every = 256
        self._pending_rows: list[tuple] = []

        rows = self._conn.execute(
            "SELECT recency_index, state_vector, terminal_outcome, terminal_outcome_dim, agent_id, epoch FROM replay ORDER BY recency_index ASC"
        ).fetchall()
        self._load_rows_into_memory(rows)

    def _load_rows_into_memory(self, rows: list[tuple]) -> None:
        self._recency_indices: list[int] = []
        self._states: list[np.ndarray] = []
        self._outcomes: list[np.ndarray] = []
        self._agent_ids: list[str] = []
        self._epochs: list[int] = []
        self._agent_to_positions: dict[str, list[int]] = {}
        self._id_to_pos: dict[int, int] = {}
        for recency_index, state_blob, terminal_outcome_blob, terminal_outcome_dim, agent_id, epoch in rows:
            rid = int(recency_index)
            pos = len(self._recency_indices)
            self._recency_indices.append(rid)
            self._states.append(np.frombuffer(state_blob, dtype=np.float32).copy())
            self._outcomes.append(np.frombuffer(terminal_outcome_blob, dtype=np.float32, count=int(terminal_outcome_dim)).copy())
            aid = str(agent_id)
            self._agent_ids.append(aid)
            self._epochs.append(int(epoch))
            self._agent_to_positions.setdefault(aid, []).append(pos)
            self._id_to_pos[rid] = pos

        self._size = len(self._recency_indices)
        self._next_recency_index = int(self._recency_indices[-1]) + 1 if self._size > 0 else 1

    def _ensure_optional_columns(self) -> None:
        existing = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(replay)").fetchall()}
        optional_columns = {
            "action_meta": "TEXT",
            "match_length": "INTEGER",
            "match_agent_1_id": "TEXT",
            "match_agent_2_id": "TEXT",
            "match_number": "INTEGER",
            "game_number_in_match": "INTEGER",
            "final_dave_value": "INTEGER",
            "final_reward_value": "INTEGER",
        }
        altered = False
        for col_name, col_type in optional_columns.items():
            if col_name in existing:
                continue
            self._conn.execute(f"ALTER TABLE replay ADD COLUMN {col_name} {col_type}")
            altered = True
        if altered:
            self._conn.commit()

    def _window_positions(self) -> np.ndarray:
        if self._size == 0:
            return np.empty((0,), dtype=np.int64)
        max_epoch = max(self._epochs)
        min_epoch = int(max_epoch - self._replay_window_epochs + 1)
        positions = [i for i, epoch in enumerate(self._epochs) if int(epoch) >= min_epoch]
        if not positions:
            return np.arange(self._size, dtype=np.int64)
        return np.asarray(positions, dtype=np.int64)

    def _insert_pending_rows(self) -> None:
        if not self._pending_rows:
            return
        self._conn.executemany(
            """
            INSERT INTO replay (
                state_vector, state_dim, agent_id, opponent_id, game_id, step_index, epoch, terminal_outcome, terminal_outcome_dim,
                action_meta, match_length, match_agent_1_id, match_agent_2_id, match_number, game_number_in_match, final_dave_value, final_reward_value, timestamp
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._pending_rows,
        )
        self._conn.commit()
        self._pending_rows = []

    def _add_single(self, **kwargs) -> None:
        state_vector = np.asarray(kwargs["state_vector"], dtype=np.float32)
        terminal_outcome = np.asarray(kwargs["terminal_outcome"], dtype=np.float32).reshape(-1)

        rid = self._next_recency_index
        self._next_recency_index += 1
        self._recency_indices.append(rid)
        self._id_to_pos[rid] = self._size
        self._states.append(state_vector)
        self._outcomes.append(terminal_outcome)
        agent_id = str(kwargs["agent_id"])
        self._agent_ids.append(agent_id)
        self._epochs.append(int(kwargs["epoch"]))
        self._agent_to_positions.setdefault(agent_id, []).append(self._size)

        self._pending_rows.append(
            (
                state_vector.tobytes(),
                int(state_vector.size),
                kwargs["agent_id"],
                kwargs["opponent_id"],
                kwargs["game_id"],
                int(kwargs["step_index"]),
                int(kwargs["epoch"]),
                terminal_outcome.tobytes(),
                int(terminal_outcome.size),
                json.dumps(kwargs.get("action_meta", {}), ensure_ascii=False),
                kwargs.get("match_length"),
                kwargs.get("match_agent_1_id"),
                kwargs.get("match_agent_2_id"),
                kwargs.get("match_number"),
                kwargs.get("game_number_in_match"),
                kwargs.get("final_dave_value"),
                kwargs.get("final_reward_value"),
                time.time(),
            )
        )
        self._size += 1

    def add(self, **kwargs) -> None:
        self._add_single(**kwargs)
        if len(self._pending_rows) >= self._commit_every:
            self._insert_pending_rows()

    def add_many(self, records: list[dict]) -> None:
        if not records:
            return

        for rec in records:
            self._add_single(**rec)
        if len(self._pending_rows) >= self._commit_every:
            self._insert_pending_rows()

    def __len__(self) -> int:
        return self._size

    def _flush_if_needed(self) -> None:
        self._insert_pending_rows()

    def _sample_positions(
        self,
        batch_size: int,
        alpha_recency: float,
        alpha_uniform: float,
    ) -> np.ndarray:
        if self._size == 0:
            return np.empty((0,), dtype=np.int64)

        candidate_positions = self._window_positions()
        total_states = int(candidate_positions.size)
        if total_states <= 0:
            return np.empty((0,), dtype=np.int64)

        local_idx = np.arange(total_states, dtype=np.float64)
        age_ratio = (float(total_states - 1) - local_idx) / float(max(total_states, 1))
        exponent = age_ratio * float(self._replay_window_epochs)
        weights = float(alpha_uniform) + float(alpha_recency) * (float(self._recency_decay) ** exponent)
        weights = np.clip(weights, 0.0, None)
        if float(np.sum(weights)) <= 0.0:
            probs = np.full((total_states,), 1.0 / float(total_states), dtype=np.float64)
        else:
            probs = weights / np.sum(weights)
        sampled_local = self._rng.choice(total_states, size=int(batch_size), replace=True, p=probs)
        return candidate_positions[sampled_local].astype(np.int64, copy=False)

    def sample_with_agent_ids(
        self,
        batch_size: int,
        alpha_recency: float,
        alpha_uniform: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sampled_pos = self._sample_positions(batch_size, alpha_recency, alpha_uniform)
        if sampled_pos.size == 0:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype="<U1"),
            )

        states = np.stack([self._states[ix] for ix in sampled_pos]).astype(np.float32)
        outcomes = np.stack([self._outcomes[ix] for ix in sampled_pos]).astype(np.float32)
        agent_ids = np.array([self._agent_ids[ix] for ix in sampled_pos], dtype=np.str_)
        return states, outcomes, agent_ids

    def sample_stratified_with_agent_ids(
        self,
        batch_sizes_by_agent: dict[str, int],
        alpha_recency: float,
        alpha_uniform: float,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self._size == 0:
            return {
                aid: (
                    np.empty((0, 0), dtype=np.float32),
                    np.empty((0, 0), dtype=np.float32),
                )
                for aid in batch_sizes_by_agent
            }

        total_required = sum(max(int(v), 0) for v in batch_sizes_by_agent.values())
        pooled_pos = self._sample_positions(total_required, alpha_recency, alpha_uniform)
        pooled_agent_ids = np.array([self._agent_ids[ix] for ix in pooled_pos], dtype=np.str_)

        for agent_id, batch_size in batch_sizes_by_agent.items():
            target_size = int(batch_size)
            if target_size <= 0:
                result[agent_id] = (np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32))
                continue

            agent_positions = self._agent_to_positions.get(agent_id, [])
            if not agent_positions:
                result[agent_id] = (np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32))
                continue

            selected = pooled_pos[pooled_agent_ids == agent_id]
            if selected.size < target_size:
                refill = self._rng.choice(np.asarray(agent_positions, dtype=np.int64), size=target_size - selected.size, replace=True)
                selected = np.concatenate([selected, refill])
            else:
                take_idx = self._rng.permutation(selected.size)[:target_size]
                selected = selected[take_idx]

            states = np.stack([self._states[ix] for ix in selected]).astype(np.float32)
            outcomes = np.stack([self._outcomes[ix] for ix in selected]).astype(np.float32)
            result[agent_id] = (states, outcomes)

        return result

    def sample(
        self,
        batch_size: int,
        alpha_recency: float,
        alpha_uniform: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        states, outcomes, _ = self.sample_with_agent_ids(
            batch_size,
            alpha_recency,
            alpha_uniform,
        )
        return states, outcomes

    def get_meta(self) -> dict[str, int | str]:
        self._flush_if_needed()
        counter = int(self._next_recency_index)
        return {"size": self._size, "counter": counter, "db_path": str(self.db_path)}

    def delete_from_epoch(self, min_epoch_inclusive: int) -> None:
        self._flush_if_needed()
        self._conn.execute("DELETE FROM replay WHERE epoch >= ?", (int(min_epoch_inclusive),))
        self._conn.commit()
        rows = self._conn.execute(
            "SELECT recency_index, state_vector, terminal_outcome, terminal_outcome_dim, agent_id, epoch FROM replay ORDER BY recency_index ASC"
        ).fetchall()
        self._load_rows_into_memory(rows)

    def close(self) -> None:
        self._flush_if_needed()
        self._conn.close()
