#pragma once
#include "types.h"
#include <random>
#include <string>
#include <vector>

namespace bg {

	class BackgammonEnv {
	public:
		explicit BackgammonEnv(uint64_t seed = 0);

		void reset_standard();            // ñòàíäàðòíàÿ ðàññòàíîâêà (ïîêà êàê â backgammon)
		const State& state() const { return s_; }

		Dice roll_dice();
		Dice current_dice() const { return dice_; }

		// TODO: ïîëíîöåííàÿ ãåíåðàöèÿ õîäîâ.
		// Ñåé÷àñ îñòàâëåí "çàãëóøå÷íûé" ãåíåðàòîð, ÷òîáû ìîæíî áûëî òåñòèðîâàòü ïàéïëàéí UI.
		size_t legal_moves(std::vector<Move>& out) const;

		void swap_perspective();
		bool apply_micro_step(uint8_t from, uint8_t to);
		void commit_turn();
		void set_state_raw(const int16_t* in /*len=53*/);
		// TODO: ïîëíîöåííûé step ñ ïðîâåðêîé âàëèäíîñòè.
		// Ñåé÷àñ ïðèìåíÿåò Move "êàê åñòü" ìèíèìàëüíî áåçîïàñíî.
		float step_apply(const Move& m, bool& done);

		// Èíâàðèàíòû/ñàíèòè-÷åêè (ïîëåçíî äëÿ òåñòîâ)
		bool validate_invariants() const;

		// Äëÿ UI: âåðíóòü "ñûðîé" State êàê êîìïàêòíûé ìàññèâ int16
		// layout:
		// [0..23]   mine points
		// [24..47]  opp points
		// [48] mine_bar [49] mine_off [50] opp_bar [51] opp_off [52] ply
		void get_state_raw(int16_t* out /*len=53*/) const;

		// Äëÿ UI: òåêóùèå êîñòè
		void get_dice_raw(uint8_t* out /*len=2*/) const;
		void set_dice_raw(const uint8_t* in /*len=2*/);

	private:
		State s_{};
		Dice dice_{};
		std::mt19937_64 rng_;

		// Çàãëóøå÷íàÿ "ëîãèêà äâèæåíèÿ": ñ÷èòàåì, ÷òî íàøè øàøêè õîäÿò "âíèç" ïî èíäåêñàì.
		// Ðåàëüíûå ïðàâèëà ïîäìåíèì ïîçæå.
		void apply_single(uint8_t from, uint8_t to);
	};

} // namespace bg
