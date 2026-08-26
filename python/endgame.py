from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class EndgamePositions:
    def __init__(self, n: int = 15):
        self.positions = [(0, 0, 0, 0, 0, 0)]
        self.pos2id = {(0, 0, 0, 0, 0, 0): 0}
        self.size_group_starts = [0]

        for _size in range(1, n + 1):
            self._build_next_size_positions()

        self.expectations = np.zeros(len(self.positions), dtype=float)
        self.comb2cubes, self.cubes2comb = self._get_cubes_combinations()
        self.best_next_positions = (-1) * np.ones((len(self), len(self.comb2cubes)), dtype="int32")

    def _get_cubes_combinations(self):
        comb2cubes = []
        for cube1 in range(1, 7):
            for cube2 in range(cube1, 7):
                comb2cubes.append((cube1, cube2))
        cubes2comb = dict()
        for i, cubes in enumerate(comb2cubes):
            cubes2comb[cubes] = i

        return comb2cubes, cubes2comb

    def _build_next_size_positions(self):
        size = len(self.size_group_starts)
        size_group_start = len(self.positions)
        self.size_group_starts.append(size_group_start)

        prev_size_group_start = self._get_size_group_start(size - 1)
        prev_size_group_finish = size_group_start

        for position in self.positions[prev_size_group_start:prev_size_group_finish]:
            for i in range(5, -1, -1):
                new_position = list(position)
                new_position[i] += 1
                new_position = tuple(new_position)

                if new_position not in self.pos2id:
                    self.pos2id[new_position] = len(self.positions)
                    self.positions.append(new_position)

        return 1

    def _get_size_group_start(self, size):
        if size >= len(self.size_group_starts):
            return None
        return self.size_group_starts[size]

    def _get_len_of_size_group(self, size):
        size_group_start = self._get_size_group_start(size)
        if size_group_start is None:
            return 0

        next_size_group_start = self._get_size_group_start(size + 1)
        if next_size_group_start is None:
            return len(self.positions) - size_group_start

        return next_size_group_start - size_group_start

    def _apply_cube_inplace(self, position_listed, i, cube_value):
        if position_listed[i] == 0:
            return 0

        # Нельзя снимать шашку перебором, если есть шашки дальше от выхода
        if cube_value > 6 - i and sum(position_listed[0:i]) > 0:
            return 0

        # Снятие шашки ровно по кубику
        if cube_value == 6 - i:
            position_listed[i] -= 1
            return 1

        # Обычный ход внутри доски
        if cube_value < 6 - i:
            position_listed[i] -= 1
            position_listed[i + cube_value] += 1
            return 1

        # Снятие шашки перебором
        if cube_value > 6 - i:
            position_listed[i] -= 1
            return 1

        return 0

    def _revert_cube_inplace(self, position_listed, i, cube_value):
        # Если ход был снятием шашки с доски
        if cube_value >= 6 - i:
            position_listed[i] += 1
        else:
            position_listed[i + cube_value] -= 1
            position_listed[i] += 1

    def _get_next_positions_ordered_cubes(
        self, current_position_listed, cubes, final_position_ids, used_cubes_number, maked_moves=0
    ):
        # recursively appends two lists:
        #     "final_position_ids" with possible final positions after use all of cubes in its initial order
        #     "used_cubes" with the number of used cubes to reach this final position

        if maked_moves == len(cubes):
            final_position_ids.append(self.pos2id[tuple(current_position_listed)])
            used_cubes_number.append(maked_moves)
            return

        is_final_position = True
        for i in range(6):
            res = self._apply_cube_inplace(current_position_listed, i, cubes[maked_moves])
            if res:
                is_final_position = False
                self._get_next_positions_ordered_cubes(
                    current_position_listed,
                    cubes,
                    final_position_ids,
                    used_cubes_number,
                    maked_moves + 1,
                )
                self._revert_cube_inplace(current_position_listed, i, cubes[maked_moves])

        if is_final_position:
            final_position_ids.append(self.pos2id[tuple(current_position_listed)])
            used_cubes_number.append(maked_moves)

    def get_legal_next_positions_ids(self, current_position, cube1, cube2):
        current_position_listed = list(current_position)
        final_position_ids = []
        used_cubes_number = []
        if cube1 == cube2:
            cubes = [cube1] * 4
            self._get_next_positions_ordered_cubes(
                current_position_listed,
                cubes,
                final_position_ids,
                used_cubes_number,
            )
        else:
            cubes = [cube1, cube2]
            self._get_next_positions_ordered_cubes(
                current_position_listed,
                cubes,
                final_position_ids,
                used_cubes_number,
            )
            cubes = [cube2, cube1]
            self._get_next_positions_ordered_cubes(
                current_position_listed,
                cubes,
                final_position_ids,
                used_cubes_number,
            )

        max_length = max(used_cubes_number)
        final_position_ids = set(
            [
                final_position_ids[i]
                for i in range(len(final_position_ids))
                if used_cubes_number[i] == max_length
            ]
        )
        return list(final_position_ids)

    def get_legal_next_positions(self, current_position, cube1, cube2):
        final_position_ids = self.get_legal_next_positions_ids(current_position, cube1, cube2)
        final_positions = [self.positions[i] for i in final_position_ids]
        return final_positions

    def _get_best_next_position_id(self, current_position, cube1, cube2):
        if cube1 > cube2:
            cube1, cube2 = cube2, cube1

        current_position_id = self.pos2id[current_position]
        comb_id = self.cubes2comb[(cube1, cube2)]

        cached = self.best_next_positions[current_position_id, comb_id]
        if cached != -1:
            return cached

        legal_next_position_ids = self.get_legal_next_positions_ids(current_position, cube1, cube2)

        best_pos_id = legal_next_position_ids[0]
        best_exp = self.expectations[best_pos_id]

        for pos_id in legal_next_position_ids[1:]:
            if self.expectations[pos_id] < best_exp:
                best_exp = self.expectations[pos_id]
                best_pos_id = pos_id

        self.best_next_positions[current_position_id, comb_id] = best_pos_id
        return best_pos_id

    def get_best_next_position(self, current_position, cube1, cube2):
        best_position_id = self._get_best_next_position_id(current_position, cube1, cube2)
        return self.positions[best_position_id]

    def compute_expectations(self):
        self.expectations[0] = 0.0

        for pos_id in range(1, len(self.positions)):
            current_position = self.positions[pos_id]
            exp_value = 0.0

            for cube1, cube2 in self.comb2cubes:
                best_next_pos_id = self._get_best_next_position_id(current_position, cube1, cube2)

                if cube1 == cube2:
                    prob = 1.0 / 36.0
                else:
                    prob = 1.0 / 18.0

                assert self.expectations[best_next_pos_id] != -1
                exp_value += prob * (1.0 + self.expectations[best_next_pos_id])

            self.expectations[pos_id] = exp_value

    def __len__(self):
        return len(self.positions)


def main():
    endgame_positions = EndgamePositions(n=15)
    endgame_positions.compute_expectations()

    output_dir = Path(__file__).resolve().parent / "training" / "endgame_helper"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "endgame_positions_n15.pkl"

    with output_file.open("wb") as f:
        pickle.dump(endgame_positions, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved endgame helper: {output_file}")


if __name__ == "__main__":
    main()
