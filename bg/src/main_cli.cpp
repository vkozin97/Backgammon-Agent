#include "env.h"
#include "obs.h"
#include "ascii.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

static bool is_finite_array(const float* a, int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(a[i])) return false;
    }
    return true;
}

static int run_selftest() {
    bg::BackgammonEnv env(123);
    env.reset_standard();

    int total_steps = 10000;
    for (int t = 0; t < total_steps; ++t) {
        env.roll_dice();

        if (!env.validate_invariants()) {
            std::cerr << "[FAIL] invariants before step at t=" << t << "\n";
            std::cerr << bg::to_ascii(env.state(), env.current_dice()) << "\n";
            return 1;
        }

        float oc[bg::OBS_COMPACT_DIM];
        float oe[bg::OBS_EXTENDED_DIM];
        bg::get_obs_compact(env.state(), env.current_dice(), env.mine_score(), env.opp_score(), env.dave_value(), oc);
        bg::get_obs_extended(env.state(), env.current_dice(), env.mine_score(), env.opp_score(), env.dave_value(), oe);

        if (!is_finite_array(oc, bg::OBS_COMPACT_DIM) || !is_finite_array(oe, bg::OBS_EXTENDED_DIM)) {
            std::cerr << "[FAIL] non-finite obs at t=" << t << "\n";
            return 1;
        }

        std::vector<bg::Move> moves;
        auto [double_possible, _mcount] = env.legal_moves(moves);
        (void)double_possible;
        if (moves.empty()) {
        auto [_reward, _dave_after, _accepted, done] = env.step_apply(0, m, 1);
        }

    bg::get_obs_compact(env.state(), env.current_dice(), env.mine_score(), env.opp_score(), env.dave_value(), obs_c);
    bg::get_obs_extended(env.state(), env.current_dice(), env.mine_score(), env.opp_score(), env.dave_value(), obs_e);

    auto [double_possible, _mcount2] = env.legal_moves(moves);
        auto [r, _dave_after, accepted, done] = env.step_apply(double_possible, moves[0], 1);
        std::cout << "Applied first move. reward=" << r << " done=" << int(done) << " accepted=" << int(accepted) << "\n";
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
    bg::get_obs_compact(env.state(), env.current_dice(), obs_c);
    std::cout << "OBS_COMPACT_DIM=" << bg::OBS_COMPACT_DIM << "\n";
    std::cout << "obs_compact[0..10]: ";
    for (int i = 0; i < 11; ++i) std::cout << obs_c[i] << " ";
    std::cout << "\n";

    // Extended obs
    float obs_e[bg::OBS_EXTENDED_DIM];
    bg::get_obs_extended(env.state(), env.current_dice(), obs_e);
    std::cout << "OBS_EXTENDED_DIM=" << bg::OBS_EXTENDED_DIM << "\n";
    std::cout << "extended metrics (last 6): ";
    for (int i = bg::OBS_COMPACT_DIM; i < bg::OBS_EXTENDED_DIM; ++i) std::cout << obs_e[i] << " ";
    std::cout << "\n";

    // Legal moves (stub)
    std::vector<bg::Move> moves;
    env.legal_moves(moves);
    std::cout << "legal_moves (stub) count=" << moves.size() << "\n";
    if (!moves.empty()) {
        bool done = false;
        float r = env.step_apply(moves[0], done);
        std::cout << "Applied first move. reward=" << r << " done=" << done << "\n";
        std::cout << bg::to_ascii(env.state(), env.current_dice()) << "\n";
    }
    return 0;
}