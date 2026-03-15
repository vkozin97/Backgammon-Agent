from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


def filter_replay_matches(
    replay_storage_dir: str,
    *,
    epoch: Optional[int] = None,
    match_length: Optional[int] = None,
    agent_1_id: Optional[str] = None,
    agent_2_id: Optional[str] = None,
    final_dave_value: Optional[int] = None,
    final_reward_value: Optional[int] = None,
) -> list[str]:
    db_path = Path(replay_storage_dir) / "replay.sqlite3"
    if not db_path.exists():
        return []

    clauses = ["1=1"]
    params: list[object] = []

    if epoch is not None:
        clauses.append("epoch = ?")
        params.append(int(epoch))
    if match_length is not None:
        clauses.append("match_length = ?")
        params.append(int(match_length))
    if agent_1_id is not None:
        clauses.append("match_agent_1_id = ?")
        params.append(str(agent_1_id))
    if agent_2_id is not None:
        clauses.append("match_agent_2_id = ?")
        params.append(str(agent_2_id))
    if final_dave_value is not None:
        clauses.append("final_dave_value = ?")
        params.append(int(final_dave_value))
    if final_reward_value is not None:
        clauses.append("final_reward_value = ?")
        params.append(int(final_reward_value))

    where = " AND ".join(clauses)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT game_id
            FROM replay
            WHERE {where}
            ORDER BY game_id
            """,
            params,
        ).fetchall()
    return [str(r[0]) for r in rows]
