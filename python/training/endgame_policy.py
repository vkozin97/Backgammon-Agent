from __future__ import annotations

import pickle
import importlib.util
from pathlib import Path
import sys
from threading import Lock

import numpy as np


_MODULE_PATH = Path(__file__).resolve()
_ENDGAME_PATH = _MODULE_PATH.parent / "endgame_helper" / "endgame_positions_n15.pkl"
_ENDGAME_POSITIONS = None
_LOAD_LOCK = Lock()


def _endgame_positions_class():
    module_name = "_backgammon_endgame_table"
    module = sys.modules.get(module_name)
    if module is None:
        source = _MODULE_PATH.parents[1] / "endgame.py"
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load endgame module from {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.EndgamePositions


class _EndgameUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        # Files produced by `python endgame.py` store this class under __main__.
        if module == "__main__" and name == "EndgamePositions":
            return _endgame_positions_class()
        # NumPy 2 writes this module path; NumPy 1.x exposes the same classes
        # under numpy.core. This keeps the generated database portable.
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def get_endgame_positions(path: Path | None = None):
    """Load the bear-off database once per Python process."""
    global _ENDGAME_POSITIONS
    if _ENDGAME_POSITIONS is None:
        with _LOAD_LOCK:
            if _ENDGAME_POSITIONS is None:
                source = _ENDGAME_PATH if path is None else Path(path)
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Endgame database is required but was not found: {source}. "
                        "Generate it with `python endgame.py`."
                    )
                with source.open("rb") as stream:
                    _ENDGAME_POSITIONS = _EndgameUnpickler(stream).load()
    return _ENDGAME_POSITIONS


def is_race_stage_no_hit(raw_state: np.ndarray) -> bool:
    raw = np.asarray(raw_state).reshape(-1)
    if raw.size < 52:
        return False
    points = raw[:24]
    opp_points = raw[24:48]
    bar, opp_bar = int(raw[48]), int(raw[50])
    mine_last = int(np.max(np.flatnonzero(points))) if np.any(points) else -1
    opp_last = int(np.min(np.flatnonzero(opp_points))) if np.any(opp_points) else 24
    return bar == 0 and opp_bar == 0 and mine_last < opp_last


def can_use_endgame_database(raw_state: np.ndarray) -> bool:
    """The n15 database represents only positions with all checkers in home."""
    raw = np.asarray(raw_state).reshape(-1)
    return is_race_stage_no_hit(raw) and not np.any(raw[6:24]) and int(np.sum(raw[:6])) <= 15


def endgame_position(raw_state: np.ndarray) -> tuple[int, ...]:
    # Engine indices run from the exit outwards; the table uses the opposite
    # order: distance 6 through distance 1.
    return tuple(int(x) for x in np.asarray(raw_state).reshape(-1)[:6][::-1])


def move_bears_off_all(raw_state: np.ndarray, move: np.ndarray) -> bool:
    """Detect a winning bear-off without relying on endless-mode done codes."""
    raw = np.asarray(raw_state).reshape(-1)
    mv = np.asarray(move).reshape(-1)
    checkers_on_board = int(np.sum(raw[:24])) + int(raw[48])
    borne_off = sum(
        1
        for step in range(min(mv.size // 2, 4))
        if int(mv[2 * step]) != 255 and int(mv[2 * step + 1]) == 25
    )
    return checkers_on_board > 0 and borne_off >= checkers_on_board


def to_mover_perspective(before_state: np.ndarray, after_state: np.ndarray) -> np.ndarray:
    before = np.asarray(before_state).reshape(-1)
    after = np.asarray(after_state).reshape(-1)
    if after.size > 52 and before.size > 52 and int(after[52]) == int(before[52]) + 1:
        converted = after.copy()
        converted[:24] = after[24:48][::-1]
        converted[24:48] = after[:24][::-1]
        converted[48:52] = after[[50, 51, 48, 49]]
        converted[52] = before[52]
        return converted
    return after


def endgame_expectation(raw_state: np.ndarray) -> float:
    """Expected rolls for either a table position or a full engine raw_state."""
    state = np.asarray(raw_state).reshape(-1)
    if state.size == 6:
        return position_expectation(state)
    return position_expectation(endgame_position(state))


def position_expectation(position: tuple[int, ...] | list[int] | np.ndarray) -> float:
    """Expected rolls for a table position ordered distance 6 ... distance 1."""
    normalized = tuple(int(x) for x in np.asarray(position).reshape(-1))
    if len(normalized) != 6:
        raise ValueError(f"Endgame position must contain 6 point counts, got {len(normalized)}")
    database = get_endgame_positions()
    position_id = database.pos2id[normalized]
    return float(database.expectations[position_id])


def reset_endgame_positions_for_tests() -> None:
    global _ENDGAME_POSITIONS
    _ENDGAME_POSITIONS = None
