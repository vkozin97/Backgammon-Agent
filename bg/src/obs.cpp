#include "obs.h"

#include <array>

namespace bg {
namespace {

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

static bool roll_hits_blot(const std::array<uint8_t, 24>& points,
					   const std::array<uint8_t, 24>& opp_points,
					   uint8_t opp_bar,
					   int blot_idx,
					   int die_a,
					   int die_b) {
	int pos = blot_idx;
	int bar_left = int(opp_bar);
	int dice[4]{};
	int dice_n = 0;
	if (die_a == die_b) {
		for (int k = 0; k < 4; ++k) dice[k] = die_a;
		dice_n = 4;
	} else {
		dice[0] = die_a;
		dice[1] = die_b;
		dice_n = 2;
	}

	for (int k = 0; k < dice_n; ++k) {
		const int die = dice[k];
		if (bar_left > 0) {
			const int to = 24 - die;
			if (to < 0 || to >= 24) continue;
			if (points[size_t(to)] >= 2) return false;
			bar_left--;
			continue;
		}

		const int to = pos - die;
		if (to < 0 || to >= 24) return false;
		if (opp_points[size_t(to)] > 0) return true;
		if (points[size_t(to)] > 0) return false;
		pos = to;
	}
	return false;
}

static bool roll_covers_blot(const std::array<uint8_t, 24>& points,
					 const std::array<uint8_t, 24>& opp_points,
					 int blot_idx,
					 int die_a,
					 int die_b) {
	int pos = blot_idx;
	int dice[4]{};
	int dice_n = 0;
	if (die_a == die_b) {
		for (int k = 0; k < 4; ++k) dice[k] = die_a;
		dice_n = 4;
	} else {
		dice[0] = die_a;
		dice[1] = die_b;
		dice_n = 2;
	}

	for (int k = 0; k < dice_n; ++k) {
		const int die = dice[k];
		const int to = pos - die;
		if (to < 0 || to >= 24) return false;
		if (opp_points[size_t(to)] > 0) return false;
		if (points[size_t(to)] > 0) return true;
		pos = to;
	}
	return false;
}

static void compute_prob_vectors(const std::array<uint8_t, 24>& points,
                                 const std::array<uint8_t, 24>& opp_points,
                                 uint8_t opp_bar,
                                 float* threatened,
                                 float* cover) {
	for (int i = 0; i < 24; ++i) {
		threatened[i] = 0.0f;
		cover[i] = 0.0f;
		if (points[size_t(i)] != 1) continue;

		int hit_count = 0;
		int cover_count = 0;
		for (int a = 1; a <= 6; ++a) {
			for (int b = 1; b <= 6; ++b) {
				if (roll_hits_blot(points, opp_points, opp_bar, i, a, b)) hit_count++;
				if (roll_covers_blot(points, opp_points, i, a, b)) cover_count++;
			}
		}
		threatened[i] = float(hit_count) / 36.0f;
		cover[i] = float(cover_count) / 36.0f;
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

	compute_prob_vectors(s.points, s.opp_points, s.opp_bar, out + base_hit_prob_mine, out + base_cover_prob_mine);

	std::array<uint8_t, 24> rev_points{};
	std::array<uint8_t, 24> rev_opp_points{};
	for (int i = 0; i < 24; ++i) {
		rev_points[size_t(i)] = s.opp_points[size_t(23 - i)];
		rev_opp_points[size_t(i)] = s.points[size_t(23 - i)];
	}
	float hit_opp_rev[24]{};
	float cover_opp_rev[24]{};
	compute_prob_vectors(rev_points, rev_opp_points, s.bar, hit_opp_rev, cover_opp_rev);
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
