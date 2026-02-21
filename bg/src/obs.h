#pragma once
#include "types.h"
#include <cstddef>

namespace bg {

	// Компактный obs (фиксированный размер)
	// Предлагаю так (float):
	// [0..23]   points[i]
	// [24..47]  opp_points[i]
	// [48] bar, [49] off, [50] opp_bar, [51] opp_off
	// [52] dice_a, [53] dice_b
	// [54] ply_norm (например ply/1000 для отладки)
	// Итого 55
	static constexpr int OBS_COMPACT_DIM = 55;

	// Extended obs: compact + метрики.
	// Пока заведём минимум (можно расширять):
	// pip_mine, pip_opp, blots_mine, blots_opp, anchors_mine, anchors_opp
	static constexpr int OBS_EXTENDED_EXTRA = 6;
	static constexpr int OBS_EXTENDED_DIM = OBS_COMPACT_DIM + OBS_EXTENDED_EXTRA;

	void get_obs_compact(const State& s, const Dice& d, float* out /*len=55*/);
	void get_obs_extended(const State& s, const Dice& d, float* out /*len=61*/);

} // namespace bg