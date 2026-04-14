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
	// [55] mine_all_in_home, [56] opp_all_in_home, [57] race_stage_no_hit
	// [58] mine_score, [59] opp_score, [60] dave_value
	static constexpr int OBS_COMPACT_DIM = 61;

	// Extended observation layout is aligned with python/training/observation.py:
	// 10 vector channels of length 24 + 26 scalar features.
	static constexpr int OBS_EXTENDED_VECTOR_CHANNELS = 10;
	static constexpr int OBS_EXTENDED_POINTS_DIM = 24;
	static constexpr int OBS_EXTENDED_SCALARS_DIM = 26;
	static constexpr int OBS_EXTENDED_DIM = OBS_EXTENDED_VECTOR_CHANNELS * OBS_EXTENDED_POINTS_DIM + OBS_EXTENDED_SCALARS_DIM;

	void get_obs_compact(const State& s, const Dice& d, int mine_score, int opp_score, int dave_value, float* out /*len=61*/);
	void get_obs_extended(const State& s, const Dice& d, int mine_score, int opp_score, int dave_value, int n_games, uint8_t cube_available_mine, uint8_t cube_available_opp, uint8_t is_crawford_game, uint8_t double_offered, float* out /*len=266*/);

} // namespace bg
