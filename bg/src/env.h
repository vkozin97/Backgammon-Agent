#pragma once
#include "types.h"
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace bg {

	class BackgammonEnv {
	public:
		explicit BackgammonEnv(uint64_t seed = 0, int n_games = 5);

		void reset_standard();            // ñòàíäàðòíàÿ ðàññòàíîâêà (ïîêà êàê â backgammon)
		const State& state() const { return s_; }

		Dice roll_dice();
		Dice current_dice() const { return dice_; }

		// TODO: ïîëíîöåííàÿ ãåíåðàöèÿ õîäîâ.
		// Ñåé÷àñ îñòàâëåí "çàãëóøå÷íûé" ãåíåðàòîð, ÷òîáû ìîæíî áûëî òåñòèðîâàòü ïàéïëàéí UI.
		std::pair<uint8_t, size_t> legal_moves(std::vector<Move>& out, bool unique_states = false) const;

		void swap_perspective();
		bool apply_micro_step(uint8_t from, uint8_t to, uint8_t die = 0);
		void commit_turn();
		void set_state_raw(const int16_t* in /*len=69*/);
		void set_state_full(const int16_t* in /*len=69*/);
		// TODO: ïîëíîöåííûé step ñ ïðîâåðêîé âàëèäíîñòè.
		// Ñåé÷àñ ïðèìåíÿåò Move "êàê åñòü" ìèíèìàëüíî áåçîïàñíî.
		std::tuple<float, int, uint8_t, uint8_t> step_apply(uint8_t apply_double, const Move& actions, uint8_t accept_double);
		bool request_double();
		std::tuple<int, uint8_t, uint8_t> resolve_pending_double(uint8_t accept_double);

		// Èíâàðèàíòû/ñàíèòè-÷åêè (ïîëåçíî äëÿ òåñòîâ)
		bool validate_invariants() const;

		// Äëÿ UI: âåðíóòü "ñûðîé" State êàê êîìïàêòíûé ìàññèâ int16
		// layout:
		// [0..23]   mine points
		// [24..47]  opp points
		// [48] mine_bar [49] mine_off [50] opp_bar [51] opp_off [52] ply
		// [53] white_score [54] black_score [55] dave_value [56] n_games [57] white_to_move
		// [58] crawford_used [59] is_crawford_game [60] first_turn [61] white_to_move_dup [62] second_player_white
		// [63] cube_owner [64] previous_game_loser [65] pending_accept [66] cube_available_mine [67] cube_available_opp [68] double_offered
		void get_state_raw(int16_t* out /*len=69*/) const;
		void get_state_full(int16_t* out /*len=69*/) const;
		int mine_score() const { return current_player_white_ ? white_score_ : black_score_; }
		int opp_score() const { return current_player_white_ ? black_score_ : white_score_; }
		int dave_value() const { return dave_value_; }
		int n_games() const { return endless_mode_ ? -1 : n_games_; }
		uint8_t cube_available_mine() const;
		uint8_t cube_available_opp() const;
		uint8_t is_crawford_game() const { return crawford_active_ ? 1 : 0; }
		uint8_t double_offered() const { return double_offered_in_match_ ? 1 : 0; }

		// Äëÿ UI: òåêóùèå êîñòè
		void get_dice_raw(uint8_t* out /*len=2*/) const;
		void set_dice_raw(const uint8_t* in /*len=2*/);

	private:
		State s_{};
		Dice dice_{};
		std::mt19937_64 rng_;
		int n_games_{5};
		int white_score_{0};
		int black_score_{0};
		int dave_value_{1};
		bool crawford_used_{false};
		bool crawford_active_{false};
		bool first_turn_in_game_{true};
		bool current_player_white_{true};
		bool second_player_white_{false};
		int cube_owner_{-1};
		int previous_game_loser_{-1};
		int pending_double_by_{-1};
		uint8_t accept_double_next_offer_{1};
		bool double_offered_in_match_{false};
		bool endless_mode_{false};

		void start_new_game(bool first_game);
		uint8_t double_possible_for_current() const;
		uint8_t cube_available_for_white(bool white_player) const;
		int classify_win_reward() const;
		uint8_t finish_game_and_maybe_match(int winner_color, int reward_points);

		// Çàãëóøå÷íàÿ "ëîãèêà äâèæåíèÿ": ñ÷èòàåì, ÷òî íàøè øàøêè õîäÿò "âíèç" ïî èíäåêñàì.
		// Ðåàëüíûå ïðàâèëà ïîäìåíèì ïîçæå.
		void apply_single(uint8_t from, uint8_t to);
	};

} // namespace bg
