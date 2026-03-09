#include "env.h"
#include "obs.h"
#include "ascii.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static bool is_finite_array(const float* a, int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(a[i])) return false;
    }
    return true;
}

static int run_selftest() {
    bg::BackgammonEnv env(123);
    env.reset_standard();

    constexpr int total_steps = 10000;
    for (int t = 0; t < total_steps; ++t) {
        env.roll_dice();

        if (!env.validate_invariants()) {
            std::cerr << "[FAIL] invariants before step at t=" << t << "\n";
            std::cerr << bg::to_ascii(env.state(), env.current_dice()) << "\n";
            return 1;
        }

        float obs_compact[bg::OBS_COMPACT_DIM];
        float obs_extended[bg::OBS_EXTENDED_DIM];
        bg::get_obs_compact(env.state(), env.current_dice(), env.mine_score(), env.opp_score(), env.dave_value(), obs_compact);
        bg::get_obs_extended(env.state(), env.current_dice(), env.mine_score(), env.opp_score(), env.dave_value(), env.n_games(), env.cube_available_mine(), env.cube_available_opp(), obs_extended);

        if (!is_finite_array(obs_compact, bg::OBS_COMPACT_DIM) || !is_finite_array(obs_extended, bg::OBS_EXTENDED_DIM)) {
            std::cerr << "[FAIL] non-finite observation at t=" << t << "\n";
            return 1;
        }

        std::vector<bg::Move> moves;
        auto [double_possible, move_count] = env.legal_moves(moves);
        (void)move_count;

        bg::Move move{};
        if (!moves.empty()) {
            move = moves.front();
        }

        auto [reward, _dave_after, accepted, done] = env.step_apply(double_possible, move, 1);
        (void)accepted;
        (void)reward;

        if (!env.validate_invariants()) {
            std::cerr << "[FAIL] invariants after step at t=" << t << "\n";
            std::cerr << bg::to_ascii(env.state(), env.current_dice()) << "\n";
            return 1;
        }

        if (done) {
            env.reset_standard();
        }
    }

    std::cout << "[OK] selftest passed: " << total_steps << " steps\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--selftest") {
        return run_selftest();
    }

    bg::BackgammonEnv env(123);
    env.reset_standard();

    auto d = env.roll_dice();
    std::cout << bg::to_ascii(env.state(), d) << "\n";

    // Compact obs
    float obs_c[bg::OBS_COMPACT_DIM];
    bg::get_obs_compact(env.state(), env.current_dice(), env.mine_score(), env.opp_score(), env.dave_value(), obs_c);
    std::cout << "OBS_COMPACT_DIM=" << bg::OBS_COMPACT_DIM << "\n";
    std::cout << "obs_compact[0..10]: ";
    for (int i = 0; i < 11; ++i) std::cout << obs_c[i] << " ";
    std::cout << "\n";

    // Extended obs
    float obs_e[bg::OBS_EXTENDED_DIM];
    bg::get_obs_extended(env.state(), env.current_dice(), env.mine_score(), env.opp_score(), env.dave_value(), env.n_games(), env.cube_available_mine(), env.cube_available_opp(), obs_e);
    std::cout << "OBS_EXTENDED_DIM=" << bg::OBS_EXTENDED_DIM << "\n";
    std::cout << "extended metrics (last " << bg::OBS_EXTENDED_SCALARS_DIM << "): ";
    for (int i = bg::OBS_EXTENDED_DIM - bg::OBS_EXTENDED_SCALARS_DIM; i < bg::OBS_EXTENDED_DIM; ++i) {
        std::cout << obs_e[i] << " ";
    }
    std::cout << "\n";

    // Legal moves
    std::vector<bg::Move> moves;
    auto [double_possible, move_count] = env.legal_moves(moves);
    std::cout << "double_possible=" << static_cast<int>(double_possible) << " legal_moves count=" << move_count << "\n";

    if (!moves.empty()) {
        auto [reward, _dave_after, accepted, done] = env.step_apply(double_possible, moves.front(), 1);
        std::cout << "Applied first move. reward=" << reward
                  << " done=" << static_cast<int>(done)
                  << " accepted=" << static_cast<int>(accepted) << "\n";
        std::cout << bg::to_ascii(env.state(), env.current_dice()) << "\n";
    }

    return 0;
}
