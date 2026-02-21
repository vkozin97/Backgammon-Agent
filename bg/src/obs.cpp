#include "obs.h"
#include <algorithm>

namespace bg {

	static int pip_count_mine(const State& s) {
		// Канонический вид: считаем, что дом "снизу", и движение "вниз" по индексам.
		// Пипсы = сумма (кол-во на пункте i) * (i+1) + bar*25 (условно)
		int pip = 0;
		for (int i = 0; i < 24; ++i) pip += int(s.points[i]) * (i + 1);
		pip += int(s.bar) * 25;
		return pip;
	}

	static int pip_count_opp(const State& s) {
		// Для соперника в канонической системе можно считать зеркально.
		// Простейший вариант: (24 - i) или тоже (i+1) — пока важно лишь “согласованность”.
		int pip = 0;
		for (int i = 0; i < 24; ++i) pip += int(s.opp_points[i]) * (24 - i);
		pip += int(s.opp_bar) * 25;
		return pip;
	}

	static int count_blots(const std::array<uint8_t, 24>& a) {
		int blots = 0;
		for (auto v : a) if (v == 1) blots++;
		return blots;
	}

	static int count_anchors(const std::array<uint8_t, 24>& a) {
		// "анкоры" как количество пунктов с >=2 шашками
		int anchors = 0;
		for (auto v : a) if (v >= 2) anchors++;
		return anchors;
	}

	void get_obs_compact(const State& s, const Dice& d, float* out) {
		// points
		for (int i = 0; i < 24; ++i) out[i] = float(s.points[i]);
		for (int i = 0; i < 24; ++i) out[24 + i] = float(s.opp_points[i]);

		out[48] = float(s.bar);
		out[49] = float(s.off);
		out[50] = float(s.opp_bar);
		out[51] = float(s.opp_off);

		out[52] = float(d.a);
		out[53] = float(d.b);

		out[54] = float(s.ply) / 1000.0f; // чисто для дебага/визуализации
	}

	void get_obs_extended(const State& s, const Dice& d, float* out) {
		get_obs_compact(s, d, out);

		// Метрики (пока базовые; потом расширим под твой список)
		const int pip_m = pip_count_mine(s);
		const int pip_o = pip_count_opp(s);

		const int blots_m = count_blots(s.points);
		const int blots_o = count_blots(s.opp_points);

		const int anch_m = count_anchors(s.points);
		const int anch_o = count_anchors(s.opp_points);

		int base = OBS_COMPACT_DIM;
		out[base + 0] = float(pip_m);
		out[base + 1] = float(pip_o);
		out[base + 2] = float(blots_m);
		out[base + 3] = float(blots_o);
		out[base + 4] = float(anch_m);
		out[base + 5] = float(anch_o);
	}

} // namespace bg