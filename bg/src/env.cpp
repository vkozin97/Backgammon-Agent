#include "env.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <set>
#include <vector>

namespace bg {

BackgammonEnv::BackgammonEnv(uint64_t seed)
    : rng_(seed ? seed : std::random_device{}()) {}

void BackgammonEnv::reset_standard() {
    s_ = State{};

    // Standard start position in canonical orientation.
    s_.points[23] = 2;
    s_.points[12] = 5;
    s_.points[7] = 3;
    s_.points[5] = 5;

    s_.opp_points[0] = 2;
    s_.opp_points[11] = 5;
    s_.opp_points[16] = 3;
    s_.opp_points[18] = 5;

    s_.bar = s_.opp_bar = 0;
    s_.off = s_.opp_off = 0;
    s_.ply = 0;

    dice_ = Dice{1, 1};
}

Dice BackgammonEnv::roll_dice() {
    std::uniform_int_distribution<int> d(1, 6);
    dice_.a = static_cast<uint8_t>(d(rng_));
    dice_.b = static_cast<uint8_t>(d(rng_));
    return dice_;
}

void BackgammonEnv::apply_single(uint8_t from, uint8_t to) {
    if (from == 255 || to == 255) return;

    auto& P = s_.points;
    if (from < 24) {
        if (P[from] == 0) return;
        P[from]--;
    } else if (from == BAR) {
        if (s_.bar == 0) return;
        s_.bar--;
    } else {
        return;
    }

    if (to < 24) {
        P[to]++;
    } else if (to == OFF) {
        s_.off++;
    } else if (to == BAR) {
        s_.bar++;
    }
}

size_t BackgammonEnv::legal_moves(std::vector<Move>& out) const {
    out.clear();

    struct LocalState {
        std::array<uint8_t, 24> points{};
        uint8_t bar{0};
    };

    auto legal_single_steps = [](const LocalState& st, int die,
                                 std::vector<std::pair<uint8_t, uint8_t>>& steps) {
        steps.clear();

        // If there are checkers on bar, only bar entry is legal.
        if (st.bar > 0) {
            int to = 24 - die;
            if (to >= 0 && to < 24) {
                steps.emplace_back(BAR, static_cast<uint8_t>(to));
            }
            return;
        }

        for (int from = 23; from >= 0; --from) {
            if (st.points[from] == 0) continue;
            int to = from - die;
            if (to >= 0) {
                steps.emplace_back(static_cast<uint8_t>(from), static_cast<uint8_t>(to));
            } else {
                steps.emplace_back(static_cast<uint8_t>(from), OFF);
            }
        }
    };

    auto apply_step_local = [](LocalState& st, uint8_t from, uint8_t to) {
        if (from == BAR) {
            if (st.bar == 0) return;
            st.bar--;
        } else if (from < 24) {
            if (st.points[from] == 0) return;
            st.points[from]--;
        } else {
            return;
        }

        if (to < 24) {
            st.points[to]++;
        }
    };

    std::vector<std::vector<int>> dice_orders;
    if (dice_.is_double()) {
        dice_orders.push_back({dice_.a, dice_.a, dice_.a, dice_.a});
    } else {
        dice_orders.push_back({dice_.a, dice_.b});
        dice_orders.push_back({dice_.b, dice_.a});
    }

    std::set<std::array<uint8_t, 8>> unique_moves;
    size_t max_used_dice = 0;

    std::function<void(const std::vector<int>&, size_t, LocalState&, Move&, size_t)> dfs;
    dfs = [&](const std::vector<int>& dice_seq, size_t idx, LocalState& cur_state,
              Move& cur_move, size_t used_dice) {
        auto record_move = [&]() {
            if (used_dice == 0) return;

            if (used_dice > max_used_dice) {
                max_used_dice = used_dice;
                unique_moves.clear();
            }
            if (used_dice == max_used_dice) {
                std::array<uint8_t, 8> key{};
                for (int k = 0; k < 4; ++k) {
                    key[2 * k] = cur_move.from[k];
                    key[2 * k + 1] = cur_move.to[k];
                }
                unique_moves.insert(key);
            }
        };

        if (idx >= dice_seq.size()) {
            record_move();
            return;
        }

        std::vector<std::pair<uint8_t, uint8_t>> steps;
        legal_single_steps(cur_state, dice_seq[idx], steps);

        if (steps.empty()) {
            record_move();
            return;
        }

        for (const auto& [from, to] : steps) {
            LocalState next_state = cur_state;
            Move next_move = cur_move;

            next_move.from[used_dice] = from;
            next_move.to[used_dice] = to;
            apply_step_local(next_state, from, to);

            dfs(dice_seq, idx + 1, next_state, next_move, used_dice + 1);
        }
    };

    for (const auto& order : dice_orders) {
        LocalState root_state{s_.points, s_.bar};
        Move root_move;
        dfs(order, 0, root_state, root_move, 0);
    }

    out.reserve(unique_moves.size());
    for (const auto& key : unique_moves) {
        Move m;
        for (int k = 0; k < 4; ++k) {
            m.from[k] = key[2 * k];
            m.to[k] = key[2 * k + 1];
        }
        out.push_back(m);
    }

    return out.size();
}

float BackgammonEnv::step_apply(const Move& m, bool& done) {
    done = false;

    for (int k = 0; k < 4; ++k) {
        apply_single(m.from[k], m.to[k]);
    }

    s_.ply++;

    if (s_.off >= 15) {
        done = true;
        return +1.0f;
    }
    return 0.0f;
}

bool BackgammonEnv::validate_invariants() const {
    auto sum_arr = [](const std::array<uint8_t, 24>& a) -> int {
        int s = 0;
        for (auto v : a) s += v;
        return s;
    };

    int mine = sum_arr(s_.points) + s_.bar + s_.off;
    int opp = sum_arr(s_.opp_points) + s_.opp_bar + s_.opp_off;

    if (mine > 15 || opp > 15) return false;
    return true;
}

void BackgammonEnv::get_state_raw(int16_t* out) const {
    for (int i = 0; i < 24; ++i) out[i] = int16_t(s_.points[i]);
    for (int i = 0; i < 24; ++i) out[24 + i] = int16_t(s_.opp_points[i]);
    out[48] = int16_t(s_.bar);
    out[49] = int16_t(s_.off);
    out[50] = int16_t(s_.opp_bar);
    out[51] = int16_t(s_.opp_off);
    out[52] = int16_t(s_.ply);
}

void BackgammonEnv::get_dice_raw(uint8_t* out) const {
    out[0] = dice_.a;
    out[1] = dice_.b;
}

}  // namespace bg
