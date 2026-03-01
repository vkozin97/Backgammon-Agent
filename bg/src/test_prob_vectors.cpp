#include <array>
#include <cstdint>
#include <iostream>

namespace {

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
            const int to = 24 - die;
            if (to >= 0 && to < 24) {
                if (points[size_t(to)] >= 2) return 0;
                opp_from_bar[size_t(to)]++;
                bar_left--;
            }
            ++k;
            continue;
        }

        const int to = pos + die;
        if (to < 0 || to >= 24) return 0;
        if (opp_points[size_t(to)] + opp_from_bar[size_t(to)] > 0) return 1;
        if (points[size_t(to)] > 0) return 0;
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
        int dice[4]{die_a, die_a, die_a, die_a};
        return run_hit_order(points, opp_points, opp_bar, blot_idx, dice, 4);
    }
    int dice_ab[2]{die_a, die_b};
    int dice_ba[2]{die_b, die_a};
    return run_hit_order(points, opp_points, opp_bar, blot_idx, dice_ab, 2)
        + run_hit_order(points, opp_points, opp_bar, blot_idx, dice_ba, 2);
}

static int run_cover_order(const std::array<uint8_t, 24>& points,
                           const std::array<uint8_t, 24>& opp_points,
                           int blot_idx,
                           const int* dice,
                           int dice_n) {
    int pos = blot_idx;
    for (int k = 0; k < dice_n; ++k) {
        const int die = dice[k];
        const int to = pos - die;
        if (to < 0 || to >= 24) return 0;
        if (opp_points[size_t(to)] > 0) return 0;
        if (points[size_t(to)] > 0) return 1;
        pos = to;
    }
    return 0;
}

static int roll_covers_blot(const std::array<uint8_t, 24>& points,
                            const std::array<uint8_t, 24>& opp_points,
                            int blot_idx,
                            int die_a,
                            int die_b) {
    if (die_a == die_b) {
        int dice[4]{die_a, die_a, die_a, die_a};
        return run_cover_order(points, opp_points, blot_idx, dice, 4);
    }
    int dice_ab[2]{die_a, die_b};
    int dice_ba[2]{die_b, die_a};
    return run_cover_order(points, opp_points, blot_idx, dice_ab, 2)
        + run_cover_order(points, opp_points, blot_idx, dice_ba, 2);
}

}  // namespace

int main() {
    std::array<uint8_t, 24> points{};
    std::array<uint8_t, 24> opp_points{};
    uint8_t opp_bar = 1;
    int blot_idx = 10;
    int die_a = 3;
    int die_b = 5;

    points[10] = 1;
    points[13] = 2;
    points[18] = 1;
    opp_points[15] = 1;
    opp_points[20] = 1;

    const int hit_result = roll_hits_blot(points, opp_points, opp_bar, blot_idx, die_a, die_b);
    const int cover_result = roll_covers_blot(points, opp_points, blot_idx, die_a, die_b);

    std::cout << "roll_hits_blot=" << hit_result << '\n';
    std::cout << "roll_covers_blot=" << cover_result << '\n';
    return 0;
}
