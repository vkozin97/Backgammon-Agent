#include "obs.h"

#include <array>

namespace bg {
namespace {

struct MiniState {
	std::array<uint8_t, 24> points{};
	std::array<uint8_t, 24> opp_points{};
	uint8_t bar{0};
	uint8_t opp_bar{0};
};

static int pip_count_mine(const State& s) {
	int pip = 0;
	for (int i = 0; i < 24; ++i) pip += int(s.points[i]) * (i + 1);
	pip += int(s.bar) * 25;
	return pip;
}

static int pip_count_opp(const State& s) {
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
	int anchors = 0;
	for (auto v : a) if (v >= 2) anchors++;
	return anchors;
}

static void collect_legal_steps(const MiniState& st, int die, std::array<std::pair<int, int>, 24>& out_steps, int& out_count) {
	out_count = 0;
	if (st.bar > 0) {
		int to = 24 - die;
		if (to >= 0 && to < 24 && st.opp_points[size_t(to)] < 2) {
			out_steps[size_t(out_count++)] = {-1, to};
		}
		return;
	}

	for (int from = 0; from < 24; ++from) {
		if (st.points[size_t(from)] == 0) continue;
		int to = from - die;
		if (to >= 0 && to < 24 && st.opp_points[size_t(to)] < 2) {
			out_steps[size_t(out_count++)] = {from, to};
		}
	}
}

static bool dfs_can_hit(MiniState st, const int* dice, int dice_n, int k, int target_idx, bool already_hit) {
	if (already_hit) return true;
	if (k >= dice_n) return false;

	std::array<std::pair<int, int>, 24> steps{};
	int n_steps = 0;
	collect_legal_steps(st, dice[k], steps, n_steps);
	if (n_steps == 0) {
		return dfs_can_hit(st, dice, dice_n, k + 1, target_idx, already_hit);
	}

	for (int i = 0; i < n_steps; ++i) {
		MiniState nx = st;
		auto [from, to] = steps[size_t(i)];
		if (from == -1) nx.bar--;
		else nx.points[size_t(from)]--;

		bool landed_on_target = (to == target_idx && st.opp_points[size_t(target_idx)] == 1);
		if (nx.opp_points[size_t(to)] == 1) {
			nx.opp_points[size_t(to)] = 0;
			nx.opp_bar++;
		}
		nx.points[size_t(to)]++;

		if (dfs_can_hit(nx, dice, dice_n, k + 1, target_idx, already_hit || landed_on_target)) return true;
	}
	return false;
}

static bool dfs_can_cover(MiniState st, const int* dice, int dice_n, int k, int target_idx) {
	if (st.points[size_t(target_idx)] >= 2) return true;
	if (k >= dice_n) return false;

	std::array<std::pair<int, int>, 24> steps{};
	int n_steps = 0;
	collect_legal_steps(st, dice[k], steps, n_steps);
	if (n_steps == 0) {
		return dfs_can_cover(st, dice, dice_n, k + 1, target_idx);
	}

	for (int i = 0; i < n_steps; ++i) {
		MiniState nx = st;
		auto [from, to] = steps[size_t(i)];
		if (from == -1) nx.bar--;
		else nx.points[size_t(from)]--;

		if (nx.opp_points[size_t(to)] == 1) {
			nx.opp_points[size_t(to)] = 0;
			nx.opp_bar++;
		}
		nx.points[size_t(to)]++;
		if (dfs_can_cover(nx, dice, dice_n, k + 1, target_idx)) return true;
	}
	return false;
}

static void compute_prob_vectors(const std::array<uint8_t, 24>& points,
                                 const std::array<uint8_t, 24>& opp_points,
                                 uint8_t bar,
                                 uint8_t opp_bar,
                                 float* threatened,
                                 float* cover) {
	for (int i = 0; i < 24; ++i) {
		threatened[i] = 0.0f;
		cover[i] = 0.0f;
		if (points[size_t(i)] != 1) continue;

		float p_hit = 0.0f;
		float p_cover = 0.0f;
		for (int a = 1; a <= 6; ++a) {
			for (int b = 1; b <= 6; ++b) {
				MiniState st_hit{};
				st_hit.points = opp_points;
				st_hit.opp_points = points;
				st_hit.bar = opp_bar;
				st_hit.opp_bar = bar;

				MiniState st_cover{};
				st_cover.points = points;
				st_cover.opp_points = opp_points;
				st_cover.bar = bar;
				st_cover.opp_bar = opp_bar;

				int dice[4]{};
				int dn = 0;
				if (a == b) {
					for (int k = 0; k < 4; ++k) dice[k] = a;
					dn = 4;
				} else {
					dice[0] = a;
					dice[1] = b;
					dn = 2;
				}

				if (dfs_can_hit(st_hit, dice, dn, 0, i, false)) p_hit += 1.0f / 36.0f;
				if (dfs_can_cover(st_cover, dice, dn, 0, i)) p_cover += 1.0f / 36.0f;
			}
		}
		threatened[i] = p_hit;
		cover[i] = p_cover;
	}
}

} // namespace

void get_obs_compact(const State& s, const Dice& d, float* out) {
	for (int i = 0; i < 24; ++i) out[i] = float(s.points[i]);
	for (int i = 0; i < 24; ++i) out[24 + i] = float(s.opp_points[i]);

	out[48] = float(s.bar);
	out[49] = float(s.off);
	out[50] = float(s.opp_bar);
	out[51] = float(s.opp_off);

	out[52] = float(d.a);
	out[53] = float(d.b);
	out[54] = float(s.ply) / 1000.0f;
}

void get_obs_extended(const State& s, const Dice& d, float* out) {
	(void)d;
	const int base_points = 0;
	const int base_opp_points = 24;
	const int base_blots = 48;
	const int base_opp_blots = 72;
	const int base_anchors = 96;
	const int base_opp_anchors = 120;
	const int base_hit_prob_mine = 144;
	const int base_cover_prob_mine = 168;
	const int base_hit_prob_opp = 192;
	const int base_cover_prob_opp = 216;
	const int base_scalars = 240;

	for (int i = 0; i < 24; ++i) {
		out[base_points + i] = float(s.points[i]);
		out[base_opp_points + i] = float(s.opp_points[i]);
		out[base_blots + i] = s.points[i] == 1 ? 1.0f : 0.0f;
		out[base_opp_blots + i] = s.opp_points[i] == 1 ? 1.0f : 0.0f;
		out[base_anchors + i] = s.points[i] >= 2 ? 1.0f : 0.0f;
		out[base_opp_anchors + i] = s.opp_points[i] >= 2 ? 1.0f : 0.0f;
	}

	compute_prob_vectors(s.points, s.opp_points, s.bar, s.opp_bar, out + base_hit_prob_mine, out + base_cover_prob_mine);

	std::array<uint8_t, 24> rev_points{};
	std::array<uint8_t, 24> rev_opp_points{};
	for (int i = 0; i < 24; ++i) {
		rev_points[size_t(i)] = s.opp_points[size_t(23 - i)];
		rev_opp_points[size_t(i)] = s.points[size_t(23 - i)];
	}
	float hit_opp_rev[24]{};
	float cover_opp_rev[24]{};
	compute_prob_vectors(rev_points, rev_opp_points, s.opp_bar, s.bar, hit_opp_rev, cover_opp_rev);
	for (int i = 0; i < 24; ++i) {
		out[base_hit_prob_opp + i] = hit_opp_rev[23 - i];
		out[base_cover_prob_opp + i] = cover_opp_rev[23 - i];
	}

	out[base_scalars + 0] = float(s.bar);
	out[base_scalars + 1] = float(s.off);
	out[base_scalars + 2] = float(s.opp_bar);
	out[base_scalars + 3] = float(s.opp_off);
	out[base_scalars + 4] = float(pip_count_mine(s));
	out[base_scalars + 5] = float(pip_count_opp(s));
	out[base_scalars + 6] = float(count_blots(s.points));
	out[base_scalars + 7] = float(count_blots(s.opp_points));
	out[base_scalars + 8] = float(count_anchors(s.points));
	out[base_scalars + 9] = float(count_anchors(s.opp_points));

	float blot_pips_mine = 0.0f;
	float blot_pips_opp = 0.0f;
	float anchor_pips_mine = 0.0f;
	float anchor_pips_opp = 0.0f;
	for (int i = 0; i < 24; ++i) {
		if (s.points[i] == 1) blot_pips_mine += float(i + 1);
		if (s.opp_points[i] == 1) blot_pips_opp += float(24 - i);
		if (s.points[i] >= 2) anchor_pips_mine += float(i + 1);
		if (s.opp_points[i] >= 2) anchor_pips_opp += float(24 - i);
	}
	out[base_scalars + 10] = blot_pips_mine;
	out[base_scalars + 11] = blot_pips_opp;
	out[base_scalars + 12] = anchor_pips_mine;
	out[base_scalars + 13] = anchor_pips_opp;
}

} // namespace bg
