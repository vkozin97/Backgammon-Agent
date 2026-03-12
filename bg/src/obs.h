#pragma once
#include "types.h"
#include <cstddef>

namespace bg {

	// Compact observation.
	// [0..23]   points[i]
	// [24..47]  opp_points[i]
	// [48] bar, [49] off, [50] opp_bar, [51] opp_off
	// [52] dice_a, [53] dice_b
	// [54] ply_norm
	static constexpr int OBS_COMPACT_DIM = 58;

	// Extended observation layout is aligned with python/training/observation.py:
	// 10 vector channels of length 24 + 23 scalar features.
	static constexpr int OBS_EXTENDED_VECTOR_CHANNELS = 10;
	static constexpr int OBS_EXTENDED_POINTS_DIM = 24;
	static constexpr int OBS_EXTENDED_SCALARS_DIM = 23;
	static constexpr int OBS_EXTENDED_DIM = OBS_EXTENDED_VECTOR_CHANNELS * OBS_EXTENDED_POINTS_DIM + OBS_EXTENDED_SCALARS_DIM;

	void get_obs_compact(const State& s, const Dice& d, int mine_score, int opp_score, int dave_value, float* out /*len=58*/);
	void get_obs_extended(const State& s, const Dice& d, int mine_score, int opp_score, int dave_value, int n_games, uint8_t cube_available_mine, uint8_t cube_available_opp, uint8_t is_crawford_game, uint8_t double_offered, float* out /*len=263*/);

} // namespace bg
