#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "env.h"
#include "obs.h"

namespace py = pybind11;

PYBIND11_MODULE(bg_env, m) {
    m.doc() = "Backgammon C++ env (pybind11)";

    py::class_<bg::Move>(m, "Move")
        .def(py::init<>())
        .def_readwrite("from_", &bg::Move::from)
        .def_readwrite("to", &bg::Move::to);

    py::class_<bg::BackgammonEnv>(m, "Env")
        .def(py::init<uint64_t, int>(), py::arg("seed") = 0, py::arg("n_games") = 5)
        .def("reset", [](bg::BackgammonEnv& self) { self.reset_standard(); })
        .def("roll_dice", [](bg::BackgammonEnv& self) {
            auto d = self.roll_dice();
            return py::make_tuple(d.a, d.b);
        })
        .def("current_dice", [](bg::BackgammonEnv& self) {
            auto d = self.current_dice();
            return py::make_tuple(d.a, d.b);
        })
        .def("set_dice", [](bg::BackgammonEnv& self,
                            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> dice) {
            if (dice.ndim() != 1 || dice.shape(0) != 2) throw std::runtime_error("set_dice: expected (2,)");
            auto d = dice.unchecked<1>();
            uint8_t tmp[2]{d(0), d(1)};
            self.set_dice_raw(tmp);
        })
        .def("get_state_raw", [](bg::BackgammonEnv& self) {
            py::array_t<int16_t> arr({69});
            auto buf = arr.mutable_unchecked<1>();
            int16_t tmp[69];
            self.get_state_raw(tmp);
            for (int i = 0; i < 69; ++i) buf(i) = tmp[i];
            return arr;
        })
        .def("set_state_raw", [](bg::BackgammonEnv& self,
                                 py::array_t<int16_t, py::array::c_style | py::array::forcecast> st) {
            if (st.ndim() != 1 || st.shape(0) != 69) throw std::runtime_error("set_state_raw: expected (69,)");
            auto a = st.unchecked<1>();
            int16_t tmp[69];
            for (int i = 0; i < 69; ++i) tmp[i] = a(i);
            self.set_state_full(tmp);
        })
        .def("get_state_full", [](bg::BackgammonEnv& self) {
            py::array_t<int16_t> arr({69});
            auto buf = arr.mutable_unchecked<1>();
            int16_t tmp[69];
            self.get_state_full(tmp);
            for (int i = 0; i < 69; ++i) buf(i) = tmp[i];
            return arr;
        })
        .def("set_state_full", [](bg::BackgammonEnv& self,
                                  py::array_t<int16_t, py::array::c_style | py::array::forcecast> st) {
            if (st.ndim() != 1 || st.shape(0) != 69) throw std::runtime_error("set_state_full: expected (69,)");
            auto a = st.unchecked<1>();
            int16_t tmp[69];
            for (int i = 0; i < 69; ++i) tmp[i] = a(i);
            self.set_state_full(tmp);
        })
        .def("apply_micro_step", [](bg::BackgammonEnv& self, uint8_t from, uint8_t to, uint8_t die) {
            bool valid = self.apply_micro_step(from, to, die);
            py::array_t<int16_t> arr({69});
            auto buf = arr.mutable_unchecked<1>();
            int16_t tmp[69];
            self.get_state_raw(tmp);
            for (int i = 0; i < 69; ++i) buf(i) = tmp[i];
            return py::make_tuple(valid, arr);
        }, py::arg("from"), py::arg("to"), py::arg("die") = 0)
        .def("commit_turn", [](bg::BackgammonEnv& self) { self.commit_turn(); })
        .def("get_obs_compact", [](bg::BackgammonEnv& self) {
            py::array_t<float> arr({bg::OBS_COMPACT_DIM});
            auto buf = arr.mutable_unchecked<1>();
            float tmp[bg::OBS_COMPACT_DIM];
            bg::get_obs_compact(self.state(), self.current_dice(), self.mine_score(), self.opp_score(), self.dave_value(), tmp);
            for (int i = 0; i < bg::OBS_COMPACT_DIM; ++i) buf(i) = tmp[i];
            return arr;
        })
        .def("get_obs_extended", [](bg::BackgammonEnv& self) {
            py::array_t<float> arr({bg::OBS_EXTENDED_DIM});
            auto buf = arr.mutable_unchecked<1>();
            float tmp[bg::OBS_EXTENDED_DIM];
            bg::get_obs_extended(self.state(), self.current_dice(), self.mine_score(), self.opp_score(), self.dave_value(), self.n_games(), self.cube_available_mine(), self.cube_available_opp(), self.is_crawford_game(), self.double_offered(), tmp);
            for (int i = 0; i < bg::OBS_EXTENDED_DIM; ++i) buf(i) = tmp[i];
            return arr;
        })
        .def("legal_moves", [](bg::BackgammonEnv& self, bool unique_states) {
            std::vector<bg::Move> moves;
            auto [double_possible, _] = self.legal_moves(moves, unique_states);
            py::array_t<uint8_t> arr({(py::ssize_t)moves.size(), (py::ssize_t)8});
            auto a = arr.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < (py::ssize_t)moves.size(); ++i) {
                for (int k = 0; k < 4; ++k) {
                    a(i, 2 * k) = moves[i].from[k];
                    a(i, 2 * k + 1) = moves[i].to[k];
                }
            }
            return py::make_tuple(double_possible, arr);
        }, py::arg("unique_states") = false)
        .def("step_index", [](bg::BackgammonEnv& self, int idx, uint8_t apply_double, uint8_t accept_double) {
            std::vector<bg::Move> moves;
            self.legal_moves(moves);
            if (idx < 0 || idx >= (int)moves.size()) throw std::runtime_error("step_index: idx out of range");
            return self.step_apply(apply_double, moves[idx], accept_double);
        }, py::arg("idx"), py::arg("apply_double") = 0, py::arg("accept_double") = 1)
        .def("step_move", [](bg::BackgammonEnv& self,
                             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mv,
                             uint8_t apply_double,
                             uint8_t accept_double) {
            if (mv.ndim() != 1 || mv.shape(0) != 8) throw std::runtime_error("step_move: expected (8,)");
            auto a = mv.unchecked<1>();
            bg::Move m;
            for (int k = 0; k < 4; ++k) { m.from[k] = a(2 * k); m.to[k] = a(2 * k + 1); }
            return self.step_apply(apply_double, m, accept_double);
        }, py::arg("mv"), py::arg("apply_double") = 0, py::arg("accept_double") = 1)
        .def("request_double", &bg::BackgammonEnv::request_double)
        .def("resolve_pending_double", &bg::BackgammonEnv::resolve_pending_double, py::arg("accept_double"));
}
