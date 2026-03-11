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

static int run_hit_order(const std::array<uint8_t, 24>& points,
    const std::array<uint8_t, 24>& opp_points,
    uint8_t opp_bar,
    int blot_idx,
    const int* dice,
    int dice_n) {
    int pos = blot_idx;
    int bar_left = int(opp_bar);
    std::array<uint8_t, 24> opp_from_bar{};

    int k = 0;
    while (k < dice_n) {
        const int die = dice[k];
        if (bar_left > 0) {
            const int to = die - 1;
            if (to >= 0 && to < 24) {
                if (points[size_t(to)] >= 2) return 0;
                if (to == pos) return 1;
                opp_from_bar[size_t(to)]++;
                bar_left--;
            }
            ++k;
            continue;
        }

        const int to = pos - die;
        if (to < 0 || to >= 24) return 0;
        if (opp_points[size_t(to)] + opp_from_bar[size_t(to)] > 0) return 1;
        if (points[size_t(to)] >= 2) return 0;
        pos = to;
        ++k;
    }
    return 0;
}

static int roll_hits_blot(const std::array<uint8_t, 24>& points,
    const std::array<uint8_t, 24>& opp_points,
    uint8_t opp_bar,
    int blot_idx,
    int die_a,
    int die_b) {
    if (die_a == die_b) {
        int dice[4]{ die_a, die_a, die_a, die_a };
        return run_hit_order(points, opp_points, opp_bar, blot_idx, dice, 4);
    }
    int dice_ab[2]{ die_a, die_b };
    int dice_ba[2]{ die_b, die_a };
    int res = run_hit_order(points, opp_points, opp_bar, blot_idx, dice_ab, 2);
    if (!res)
        return 2 * run_hit_order(points, opp_points, opp_bar, blot_idx, dice_ba, 2);
    return 2 * res;
}

static int run_cover_order(const std::array<uint8_t, 24>& points,
    const std::array<uint8_t, 24>& opp_points,
    uint8_t bar,
    int blot_idx,
    const int* dice,
    int dice_n) {
    int pos = blot_idx;
    int bar_left = int(bar);
    std::array<uint8_t, 24> from_bar{};

    int k = 0;
    while (k < dice_n) {
        const int die = dice[k];
        if (bar_left > 0) {
            const int to = 24 - die;
            if (to >= 0 && to < 24) {
                if (opp_points[size_t(to)] >= 2) return 0;
                if (to == pos) return 1;
                from_bar[size_t(to)]++;
                bar_left--;
            }
            ++k;
            continue;
        }

        const int to = pos + die;
        if (to < 0 || to >= 24) return 0;
        if (opp_points[size_t(to)] >= 2) return 0;
        if (points[size_t(to)] + from_bar[size_t(to)] > 0) return 1;
        pos = to;
        ++k;
    }
    return 0;
}

static int roll_covers_blot(const std::array<uint8_t, 24>& points,
    const std::array<uint8_t, 24>& opp_points,
    uint8_t bar,
    int blot_idx,
    int die_a,
    int die_b) {
    if (die_a == die_b) {
        int dice[4]{ die_a, die_a, die_a, die_a };
        return run_cover_order(points, opp_points, bar, blot_idx, dice, 4);
    }
    int dice_ab[2]{ die_a, die_b };
    int dice_ba[2]{ die_b, die_a };
    int res = run_cover_order(points, opp_points, bar, blot_idx, dice_ab, 2);
    if (!res)
        return 2 * run_cover_order(points, opp_points, bar, blot_idx, dice_ba, 2);
    return 2 * res;
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

        int hit_count = 0;
        int cover_count = 0;
        for (int a = 1; a <= 6; ++a) {
            for (int b = a; b <= 6; ++b) {
                hit_count += roll_hits_blot(points, opp_points, opp_bar, i, a, b);
                cover_count += roll_covers_blot(points, opp_points, bar, i, a, b);
            }
        }
        threatened[i] = float(hit_count) / 36.0f;
        cover[i] = float(cover_count) / 36.0f;
    }
}

} // namespace

void get_obs_compact(const State& s, const Dice& d, int mine_score, int opp_score, int dave_value, float* out) {
	for (int i = 0; i < 24; ++i) out[i] = float(s.points[i]);
	for (int i = 0; i < 24; ++i) out[24 + i] = float(s.opp_points[i]);

	out[48] = float(s.bar);
	out[49] = float(s.off);
	out[50] = float(s.opp_bar);
	out[51] = float(s.opp_off);

	out[52] = float(d.a);
	out[53] = float(d.b);
	out[54] = float(s.ply) / 1000.0f;
	out[55] = float(mine_score);
	out[56] = float(opp_score);
	out[57] = float(dave_value);
}

void get_obs_extended(const State& s, const Dice& d, int mine_score, int opp_score, int dave_value, int n_games, uint8_t cube_available_mine, uint8_t cube_available_opp, uint8_t is_crawford_game, uint8_t double_offered, float* out) {
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
	out[base_scalars + 14] = float(mine_score);
	out[base_scalars + 15] = float(opp_score);
	out[base_scalars + 16] = float(dave_value);
	out[base_scalars + 17] = (n_games < 0) ? 11.0f : float(std::max(0, n_games - mine_score));
	out[base_scalars + 18] = (n_games < 0) ? 11.0f : float(std::max(0, n_games - opp_score));
	out[base_scalars + 19] = float(cube_available_mine);
	out[base_scalars + 20] = float(cube_available_opp);
	out[base_scalars + 21] = float(is_crawford_game);
	out[base_scalars + 22] = float(double_offered);
}

} // namespace bg
