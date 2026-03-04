#include "env.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace bg {

namespace {

static constexpr uint8_t INTERNAL_BAR = 24;
static constexpr uint8_t INTERNAL_OFF = 25;

bool is_board_coord(uint8_t p) { return p >= 1 && p <= 24; }
uint8_t board_to_idx(uint8_t p) { return static_cast<uint8_t>(24 - p); }
uint8_t idx_to_board(uint8_t i) { return static_cast<uint8_t>(24 - i); }

bool decode_from(uint8_t from_pub, uint8_t& from_idx_or_bar) {
    if (from_pub == BAR) { from_idx_or_bar = INTERNAL_BAR; return true; }
    if (!is_board_coord(from_pub)) return false;
    from_idx_or_bar = board_to_idx(from_pub);
    return true;
}

bool decode_to(uint8_t to_pub, uint8_t& to_idx_or_off) {
    if (to_pub == OFF) { to_idx_or_off = INTERNAL_OFF; return true; }
    if (!is_board_coord(to_pub)) return false;
    to_idx_or_off = board_to_idx(to_pub);
    return true;
}

uint8_t encode_from(uint8_t from_idx_or_bar) { return from_idx_or_bar == INTERNAL_BAR ? BAR : idx_to_board(from_idx_or_bar); }
uint8_t encode_to(uint8_t to_idx_or_off) { return to_idx_or_off == INTERNAL_OFF ? OFF : idx_to_board(to_idx_or_off); }

bool all_checkers_in_home_or_off(const State& s) {
    for (int p = 6; p < 24; ++p) if (s.points[p] > 0) return false;
    return true;
}

bool has_checker_farther_from_off(const State& s, uint8_t from) {
    for (int idx = 5; idx > static_cast<int>(from); --idx) if (s.points[idx] > 0) return true;
    return false;
}

bool can_bear_off_from(const State& s, uint8_t from, uint8_t die) {
    if (from >= 24 || die > 6 || die == 0 || !all_checkers_in_home_or_off(s)) return false;
    const uint8_t distance_to_off = static_cast<uint8_t>(from + 1);
    if (die == distance_to_off) return true;
    if (die > distance_to_off) return !has_checker_farther_from_off(s, from);
    return false;
}

bool is_legal_single_step(const State& s, uint8_t from, uint8_t to, uint8_t die) {
    if (from == 255 || to == 255) return false;
    if (s.bar > 0 && from != INTERNAL_BAR) return false;

    if (from == INTERNAL_BAR) {
        if (s.bar == 0 || to >= 24) return false;
        return s.opp_points[to] < 2;
    }

    if (from >= 24 || s.points[from] == 0) return false;

    if (to < 24) {
        if (to >= from || s.opp_points[to] >= 2) return false;
        return true;
    }
    if (to == INTERNAL_OFF) return can_bear_off_from(s, from, die);
    return false;
}

bool apply_single_checked(State& s, uint8_t from, uint8_t to, uint8_t die) {
    if (!is_legal_single_step(s, from, to, die)) return false;

    if (from == INTERNAL_BAR) s.bar--; else s.points[from]--;

    if (to < 24) {
        if (s.opp_points[to] == 1) {
            s.opp_points[to] = 0;
            s.opp_bar++;
        }
        s.points[to]++;
    } else {
        s.off++;
    }
    return true;
}

}  // namespace

BackgammonEnv::BackgammonEnv(uint64_t seed, int n_games)
    : rng_(seed ? seed : std::random_device{}()), n_games_(std::max(1, n_games)) {}

void BackgammonEnv::start_new_game(bool first_game) {
    s_ = State{};
    s_.points[23] = 2; s_.points[12] = 5; s_.points[7] = 3; s_.points[5] = 5;
    s_.opp_points[0] = 2; s_.opp_points[11] = 5; s_.opp_points[16] = 3; s_.opp_points[18] = 5;
    s_.bar = s_.opp_bar = 0;
    s_.off = s_.opp_off = 0;

    if (first_game) {
        std::uniform_int_distribution<int> starter(0, 1);
        current_player_white_ = starter(rng_) == 0;
    } else {
        current_player_white_ = previous_game_loser_ == 0;
    }
    second_player_white_ = !current_player_white_;

    first_turn_in_game_ = true;
    dave_value_ = 1;
    cube_owner_ = -1;
    pending_double_by_ = -1;
    crawford_active_ = !crawford_used_ && (white_score_ == n_games_ - 1 || black_score_ == n_games_ - 1);
    if (crawford_active_) crawford_used_ = true;

    dice_ = Dice{1, 1};
}

void BackgammonEnv::reset_standard() {
    white_score_ = 0;
    black_score_ = 0;
    dave_value_ = 1;
    crawford_used_ = false;
    previous_game_loser_ = -1;
    s_.ply = 0;
    start_new_game(true);
}

uint8_t BackgammonEnv::double_possible_for_current() const {
    if (crawford_active_) return 0;
    if (first_turn_in_game_) return current_player_white_ == second_player_white_ ? 1 : 0;
    if (cube_owner_ < 0) return 1;
    return (cube_owner_ == (current_player_white_ ? 0 : 1)) ? 1 : 0;
}

Dice BackgammonEnv::roll_dice() {
    std::uniform_int_distribution<int> d(1, 6);
    if (first_turn_in_game_) {
        const int a = d(rng_);
        std::uniform_int_distribution<int> e(1, 5);
        int b = e(rng_);
        if (b >= a) ++b;
        dice_.a = static_cast<uint8_t>(a);
        dice_.b = static_cast<uint8_t>(b);
    } else {
        dice_.a = static_cast<uint8_t>(d(rng_));
        dice_.b = static_cast<uint8_t>(d(rng_));
    }
    return dice_;
}

void BackgammonEnv::apply_single(uint8_t from, uint8_t to) {
    if (from == 255 || to == 255) return;
    auto& P = s_.points;
    if (from < 24) { if (P[from] == 0) return; P[from]--; }
    else if (from == INTERNAL_BAR) { if (s_.bar == 0) return; s_.bar--; }
    else return;

    if (to < 24) P[to]++;
    else if (to == INTERNAL_OFF) s_.off++;
    else if (to == INTERNAL_BAR) s_.bar++;
}

void BackgammonEnv::swap_perspective() {
    std::array<uint8_t, 24> next_points{};
    std::array<uint8_t, 24> next_opp_points{};
    for (int i = 0; i < 24; ++i) {
        next_points[i] = s_.opp_points[23 - i];
        next_opp_points[i] = s_.points[23 - i];
    }
    s_.points = next_points;
    s_.opp_points = next_opp_points;
    std::swap(s_.bar, s_.opp_bar);
    std::swap(s_.off, s_.opp_off);
}

std::pair<uint8_t, size_t> BackgammonEnv::legal_moves(std::vector<Move>& out, bool unique_states) const {
    out.clear();

    struct LocalState {
        std::array<uint8_t, 24> points{};
        std::array<uint8_t, 24> opp_points{};
        uint8_t bar{0}; uint8_t off{0}; uint8_t opp_bar{0}; uint8_t opp_off{0};
    };
    struct FinalStateKey {
        std::array<uint8_t, 24> points{};
        std::array<uint8_t, 24> opp_points{};
        uint8_t bar{0}; uint8_t off{0}; uint8_t opp_bar{0}; uint8_t opp_off{0};
        bool operator<(const FinalStateKey& o) const {
            if (points != o.points) return points < o.points;
            if (opp_points != o.opp_points) return opp_points < o.opp_points;
            if (bar != o.bar) return bar < o.bar;
            if (off != o.off) return off < o.off;
            if (opp_bar != o.opp_bar) return opp_bar < o.opp_bar;
            return opp_off < o.opp_off;
        }
        bool operator==(const FinalStateKey& o) const {
            return points == o.points && opp_points == o.opp_points && bar == o.bar && off == o.off && opp_bar == o.opp_bar && opp_off == o.opp_off;
        }
    };
    struct CandidateMove { Move move{}; FinalStateKey state{}; uint8_t used_dice{0}; };

    auto legal_single_steps = [](const LocalState& st, int die, std::vector<std::pair<uint8_t, uint8_t>>& steps) {
        steps.clear();
        if (st.bar > 0) {
            int to = 24 - die;
            if (to >= 0 && to < 24 && st.opp_points[to] < 2) steps.emplace_back(INTERNAL_BAR, static_cast<uint8_t>(to));
            return;
        }
        bool all_in_home = true;
        for (int p = 6; p < 24; ++p) if (st.points[p] > 0) { all_in_home = false; break; }

        for (int from = 23; from >= 0; --from) {
            if (st.points[from] == 0) continue;
            int to = from - die;
            if (to >= 0) {
                if (st.opp_points[to] < 2) steps.emplace_back(static_cast<uint8_t>(from), static_cast<uint8_t>(to));
                continue;
            }
            if (!all_in_home) continue;
            if (die == from + 1) { steps.emplace_back(static_cast<uint8_t>(from), INTERNAL_OFF); continue; }
            bool has_higher = false;
            for (int h = from + 1; h <= 5; ++h) if (st.points[h] > 0) { has_higher = true; break; }
            if (die > from + 1 && !has_higher) steps.emplace_back(static_cast<uint8_t>(from), INTERNAL_OFF);
        }
    };

    auto apply_step_local = [](LocalState& st, uint8_t from, uint8_t to) {
        if (from == INTERNAL_BAR) { if (st.bar == 0) return; st.bar--; }
        else if (from < 24) { if (st.points[from] == 0) return; st.points[from]--; }
        else return;

        if (to < 24) {
            if (st.opp_points[to] == 1) { st.opp_points[to] = 0; st.opp_bar++; }
            st.points[to]++;
        } else if (to == INTERNAL_OFF) st.off++;
    };

    std::vector<std::vector<int>> dice_orders;
    if (dice_.is_double()) dice_orders.push_back({dice_.a, dice_.a, dice_.a, dice_.a});
    else { dice_orders.push_back({dice_.a, dice_.b}); dice_orders.push_back({dice_.b, dice_.a}); }

    std::vector<CandidateMove> candidates;
    uint8_t max_used_dice = 0;

    std::function<void(const std::vector<int>&, size_t, LocalState&, Move&, uint8_t)> dfs;
    dfs = [&](const std::vector<int>& seq, size_t idx, LocalState& st, Move& mv, uint8_t used) {
        auto record = [&]() {
            if (used == 0) return;
            max_used_dice = std::max(max_used_dice, used);
            CandidateMove c; c.move = mv;
            c.state = {st.points, st.opp_points, st.bar, st.off, st.opp_bar, st.opp_off};
            c.used_dice = used;
            candidates.push_back(c);
        };
        if (idx >= seq.size()) { record(); return; }
        std::vector<std::pair<uint8_t, uint8_t>> steps;
        legal_single_steps(st, seq[idx], steps);
        if (steps.empty()) { record(); return; }
        for (const auto& [from, to] : steps) {
            LocalState next = st;
            Move next_move = mv;
            next_move.from[used] = from;
            next_move.to[used] = to;
            apply_step_local(next, from, to);
            dfs(seq, idx + 1, next, next_move, used + 1);
        }
    };

    for (const auto& order : dice_orders) {
        LocalState root{s_.points, s_.opp_points, s_.bar, s_.off, s_.opp_bar, s_.opp_off};
        Move mv;
        dfs(order, 0, root, mv, 0);
    }

    if (candidates.empty() || max_used_dice == 0) return {double_possible_for_current(), 0};

    std::vector<size_t> selected;
    for (size_t i = 0; i < candidates.size(); ++i) if (candidates[i].used_dice == max_used_dice) selected.push_back(i);

    std::sort(selected.begin(), selected.end(), [&](size_t l, size_t r) {
        const Move& a = candidates[l].move;
        const Move& b = candidates[r].move;
        return std::lexicographical_compare(a.from.begin(), a.from.end(), b.from.begin(), b.from.end()) ||
               (a.from == b.from && std::lexicographical_compare(a.to.begin(), a.to.end(), b.to.begin(), b.to.end()));
    });
    selected.erase(std::unique(selected.begin(), selected.end(), [&](size_t l, size_t r) {
        return candidates[l].move.from == candidates[r].move.from && candidates[l].move.to == candidates[r].move.to;
    }), selected.end());

    if (unique_states) {
        std::vector<size_t> by_state = selected;
        std::sort(by_state.begin(), by_state.end(), [&](size_t l, size_t r) {
            const auto& a = candidates[l].state;
            const auto& b = candidates[r].state;
            if (a == b) return l < r;
            return a < b;
        });
        std::vector<size_t> minimal;
        for (size_t i = 0; i < by_state.size(); ++i) if (i == 0 || !(candidates[by_state[i]].state == candidates[by_state[i - 1]].state)) minimal.push_back(by_state[i]);
        selected.swap(minimal);
    }

    out.reserve(selected.size());
    for (size_t idx : selected) {
        Move m = candidates[idx].move;
        for (int k = 0; k < 4; ++k) {
            if (m.from[k] != 255 && m.to[k] != 255) {
                m.from[k] = encode_from(m.from[k]);
                m.to[k] = encode_to(m.to[k]);
            }
        }
        out.push_back(m);
    }

    return {double_possible_for_current(), out.size()};
}

int BackgammonEnv::classify_win_reward() const {
    if (s_.opp_off > 0) return 1;
    bool kox = s_.opp_bar > 0;
    if (!kox) {
        for (int i = 6; i < 24; ++i) if (s_.opp_points[i] > 0) { kox = true; break; }
    }
    if (dave_value_ == 1) return 1;
    if (kox) return 3;
    return 2;
}

uint8_t BackgammonEnv::finish_game_and_maybe_match(int winner_color, int reward_points) {
    int points = reward_points * dave_value_;
    if (winner_color == 0) { white_score_ += points; previous_game_loser_ = 1; }
    else { black_score_ += points; previous_game_loser_ = 0; }
    if (white_score_ >= n_games_ || black_score_ >= n_games_) return 2;

    s_.ply++;
    start_new_game(false);
    return 1;
}

std::tuple<float, int, uint8_t, uint8_t> BackgammonEnv::step_apply(uint8_t apply_double, const Move& actions, uint8_t accept_double) {
    uint8_t double_accepted = 0;

    if (pending_double_by_ >= 0) {
        if (accept_double) {
            dave_value_ *= 2;
            cube_owner_ = current_player_white_ ? 0 : 1;
            double_accepted = 1;
            pending_double_by_ = -1;
        } else {
            const int winner_color = pending_double_by_;
            pending_double_by_ = -1;
            const uint8_t done = finish_game_and_maybe_match(winner_color, 1);
            return {1.0f, dave_value_, 0, done};
        }
    }

    if (apply_double && double_possible_for_current()) {
        pending_double_by_ = current_player_white_ ? 0 : 1;
    }

    for (int k = 0; k < 4; ++k) {
        if (actions.from[k] == 255 || actions.to[k] == 255) continue;
        if (!apply_micro_step(actions.from[k], actions.to[k], 0)) break;
    }

    if (s_.off >= 15) {
        const int reward = classify_win_reward();
        const uint8_t done = finish_game_and_maybe_match(current_player_white_ ? 0 : 1, reward);
        return {static_cast<float>(reward), dave_value_, double_accepted, done};
    }

    commit_turn();
    return {0.0f, dave_value_, double_accepted, 0};
}

bool BackgammonEnv::apply_micro_step(uint8_t from, uint8_t to, uint8_t die) {
    uint8_t from_internal = 255;
    uint8_t to_internal = 255;
    if (!decode_from(from, from_internal) || !decode_to(to, to_internal)) return false;
    return apply_single_checked(s_, from_internal, to_internal, die);
}

void BackgammonEnv::commit_turn() {
    swap_perspective();
    s_.ply++;
    current_player_white_ = !current_player_white_;
    if (first_turn_in_game_) first_turn_in_game_ = false;
}

void BackgammonEnv::set_state_raw(const int16_t* in) {
    pending_double_by_ = -1;
    for (int i = 0; i < 24; ++i) s_.points[i] = static_cast<uint8_t>(std::clamp<int>(in[i], 0, 15));
    for (int i = 0; i < 24; ++i) s_.opp_points[i] = static_cast<uint8_t>(std::clamp<int>(in[24 + i], 0, 15));
    s_.bar = static_cast<uint8_t>(std::clamp<int>(in[48], 0, 15));
    s_.off = static_cast<uint8_t>(std::clamp<int>(in[49], 0, 15));
    s_.opp_bar = static_cast<uint8_t>(std::clamp<int>(in[50], 0, 15));
    s_.opp_off = static_cast<uint8_t>(std::clamp<int>(in[51], 0, 15));
    s_.ply = static_cast<uint8_t>(std::clamp<int>(in[52], 0, 255));
    white_score_ = std::max(0, int(in[53]));
    black_score_ = std::max(0, int(in[54]));
    dave_value_ = std::max(1, int(in[55]));
    n_games_ = std::max(1, int(in[56]));
    current_player_white_ = in[57] != 0;
}

bool BackgammonEnv::validate_invariants() const {
    auto sum_arr = [](const std::array<uint8_t, 24>& a) -> int { int s = 0; for (auto v : a) s += v; return s; };
    int mine = sum_arr(s_.points) + s_.bar + s_.off;
    int opp = sum_arr(s_.opp_points) + s_.opp_bar + s_.opp_off;
    return mine <= 15 && opp <= 15;
}

void BackgammonEnv::get_state_raw(int16_t* out) const {
    for (int i = 0; i < 24; ++i) out[i] = int16_t(s_.points[i]);
    for (int i = 0; i < 24; ++i) out[24 + i] = int16_t(s_.opp_points[i]);
    out[48] = int16_t(s_.bar);
    out[49] = int16_t(s_.off);
    out[50] = int16_t(s_.opp_bar);
    out[51] = int16_t(s_.opp_off);
    out[52] = int16_t(s_.ply);
    out[53] = int16_t(white_score_);
    out[54] = int16_t(black_score_);
    out[55] = int16_t(dave_value_);
    out[56] = int16_t(n_games_);
    out[57] = int16_t(current_player_white_ ? 1 : 0);
}

void BackgammonEnv::get_dice_raw(uint8_t* out) const {
    out[0] = dice_.a;
    out[1] = dice_.b;
}

void BackgammonEnv::set_dice_raw(const uint8_t* in) {
    dice_.a = std::clamp<uint8_t>(in[0], 1, 6);
    dice_.b = std::clamp<uint8_t>(in[1], 1, 6);
}

}  // namespace bg
