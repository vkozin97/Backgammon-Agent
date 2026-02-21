#include "env.h"
#include <algorithm>
#include <cstdint>

namespace bg {

    BackgammonEnv::BackgammonEnv(uint64_t seed)
        : rng_(seed ? seed : std::random_device{}()) {}

    void BackgammonEnv::reset_standard() {
        s_ = State{};
        // Стандартная расстановка backgammon (для тестов/визуализации).
        // Индексация 0..23. Мы позже зафиксируем ориентацию под "короткие нарды".
        // Наши:
        // 2 на 23, 5 на 12, 3 на 7, 5 на 5 (пример для демонстрации)
        s_.points[23] = 2;
        s_.points[12] = 5;
        s_.points[7] = 3;
        s_.points[5] = 5;

        // Соперник:
        s_.opp_points[0] = 2;
        s_.opp_points[11] = 5;
        s_.opp_points[16] = 3;
        s_.opp_points[18] = 5;

        s_.bar = s_.opp_bar = 0;
        s_.off = s_.opp_off = 0;
        s_.ply = 0;

        dice_ = Dice{ 1,1 };
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
        // минимальная "безопасность"
        if (from < 24) {
            if (P[from] == 0) return;
            P[from]--;
        }
        else if (from == BAR) {
            if (s_.bar == 0) return;
            s_.bar--;
        }
        else {
            return;
        }

        if (to < 24) {
            P[to]++;
        }
        else if (to == OFF) {
            s_.off++;
        }
        else if (to == BAR) {
            s_.bar++;
        }
    }

    size_t BackgammonEnv::legal_moves(std::vector<Move>& out) const {
        out.clear();

        // ---- ЗАГЛУШКА ДЛЯ ТЕСТОВ ----
        // Идея: для каждого die пробуем сдвинуть одну шашку "вниз" (i - die),
        // если есть шашка на i. Это НЕ правила нард, а только для отладки интерфейса.
        const int dice_vals[2] = { dice_.a, dice_.b };

        for (int di = 0; di < 2; ++di) {
            int die = dice_vals[di];
            for (int i = 23; i >= 0; --i) {
                if (s_.points[i] == 0) continue;
                int j = i - die;
                Move m;
                m.from[0] = static_cast<uint8_t>(i);
                if (j >= 0) m.to[0] = static_cast<uint8_t>(j);
                else        m.to[0] = OFF;
                out.push_back(m);
            }
        }
        return out.size();
    }

    float BackgammonEnv::step_apply(const Move& m, bool& done) {
        done = false;

        for (int k = 0; k < 4; ++k) {
            apply_single(m.from[k], m.to[k]);
        }

        s_.ply++;

        // Заглушка терминальности: если вывели 15 шашек — победа.
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

        // В backgammon обычно 15 шашек на игрока
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

} // namespace bg