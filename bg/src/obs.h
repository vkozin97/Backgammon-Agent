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
	static constexpr int OBS_COMPACT_DIM = 55;

	// Extended observation layout is aligned with python/training/observation.py:
	// 10 vector channels of length 24 + 14 scalar features.
	static constexpr int OBS_EXTENDED_VECTOR_CHANNELS = 10;
	static constexpr int OBS_EXTENDED_POINTS_DIM = 24;
	static constexpr int OBS_EXTENDED_SCALARS_DIM = 14;
	static constexpr int OBS_EXTENDED_DIM = OBS_EXTENDED_VECTOR_CHANNELS * OBS_EXTENDED_POINTS_DIM + OBS_EXTENDED_SCALARS_DIM;

	void get_obs_compact(const State& s, const Dice& d, float* out /*len=55*/);
	void get_obs_extended(const State& s, const Dice& d, float* out /*len=254*/);

} // namespace bg
