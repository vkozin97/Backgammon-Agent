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
    if !run_hit_order(points, opp_points, opp_bar, blot_idx, dice_ab, 2)
        return 2 * run_hit_order(points, opp_points, opp_bar, blot_idx, dice_ba, 2)
    else return 0;
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
            for (int b = a; b <= 6; ++b) {
                hit_count += roll_hits_blot(points, opp_points, opp_bar, i, a, b);
                cover_count += roll_covers_blot(points, opp_points, i, a, b);
            }
        }
        threatened[i] = float(hit_count) / 36.0f;
        cover[i] = float(cover_count) / 36.0f;
    }
}

}  // namespace

int main() {
    std::array<uint8_t, 24> points{0,0,0,0,1,4,0,2,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,2};
    std::array<uint8_t, 24> opp_points{1,0,1,0,0,0,0,0,0,0,0,4,0,0,0,1,3,0,4,1,0,0,0,0};
    uint8_t opp_bar = 0;
    int blot_idx = 4;

    std::array<float> threatened;
    std::array<float> cover;
    compute_prob_vectors(points, opp_points, opp_bar, threatened, cover);
    // const int hit_result = roll_hits_blot(points, opp_points, opp_bar, blot_idx, die_a, die_b);
    // const int cover_result = roll_covers_blot(points, opp_points, blot_idx, die_a, die_b);

    std::cout << "threatened=" << "\n";
    for (int i = 0; i < 24; ++i)
        cout << threatened[i] << " ";
    std::cout << "\n" << "cover" << "\n";
    for (int i = 0; i < 24; ++i)
        cout << cover[i] << " ";
    
    return 0;
}
