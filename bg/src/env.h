#pragma once
#include "types.h"
#include <random>
#include <string>
#include <vector>

namespace bg {

	class BackgammonEnv {
	public:
		explicit BackgammonEnv(uint64_t seed = 0);

		void reset_standard();            // стандартная расстановка (пока как в backgammon)
		const State& state() const { return s_; }

		Dice roll_dice();
		Dice current_dice() const { return dice_; }

		// TODO: полноценная генерация ходов.
		// Сейчас оставлен "заглушечный" генератор, чтобы можно было тестировать пайплайн UI.
		size_t legal_moves(std::vector<Move>& out) const;

		// TODO: полноценный step с проверкой валидности.
		// Сейчас применяет Move "как есть" минимально безопасно.
		float step_apply(const Move& m, bool& done);

		// Инварианты/санити-чеки (полезно для тестов)
		bool validate_invariants() const;

		// Для UI: вернуть "сырой" State как компактный массив int16
		// layout:
		// [0..23]   mine points
		// [24..47]  opp points
		// [48] mine_bar [49] mine_off [50] opp_bar [51] opp_off [52] ply
		void get_state_raw(int16_t* out /*len=53*/) const;

		// Для UI: текущие кости
		void get_dice_raw(uint8_t* out /*len=2*/) const;

	private:
		State s_{};
		Dice dice_{};
		std::mt19937_64 rng_;

		// Заглушечная "логика движения": считаем, что наши шашки ходят "вниз" по индексам.
		// Реальные правила подменим позже.
		void apply_single(uint8_t from, uint8_t to);
	};

} // namespace bg