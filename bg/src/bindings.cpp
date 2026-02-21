#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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
        .def(py::init<uint64_t>(), py::arg("seed") = 0)

        .def("reset", [](bg::BackgammonEnv& self) {
        self.reset_standard();
            })

        .def("roll_dice", [](bg::BackgammonEnv& self) {
                auto d = self.roll_dice();
                return py::make_tuple(d.a, d.b);
            })

                .def("current_dice", [](bg::BackgammonEnv& self) {
                auto d = self.current_dice();
                return py::make_tuple(d.a, d.b);
                    })

                // raw-state для UI
                        .def("get_state_raw", [](bg::BackgammonEnv& self) {
                        py::array_t<int16_t> arr({ 53 });
                        auto buf = arr.mutable_unchecked<1>();
                        int16_t tmp[53];
                        self.get_state_raw(tmp);
                        for (int i = 0; i < 53; ++i) buf(i) = tmp[i];
                        return arr;
                            })

                        // compact obs
                                .def("get_obs_compact", [](bg::BackgammonEnv& self) {
                                py::array_t<float> arr({ bg::OBS_COMPACT_DIM });
                                auto buf = arr.mutable_unchecked<1>();
                                float tmp[bg::OBS_COMPACT_DIM];
                                bg::get_obs_compact(self.state(), self.current_dice(), tmp);
                                for (int i = 0; i < bg::OBS_COMPACT_DIM; ++i) buf(i) = tmp[i];
                                return arr;
                                    })

                                // extended obs
                                        .def("get_obs_extended", [](bg::BackgammonEnv& self) {
                                        py::array_t<float> arr({ bg::OBS_EXTENDED_DIM });
                                        auto buf = arr.mutable_unchecked<1>();
                                        float tmp[bg::OBS_EXTENDED_DIM];
                                        bg::get_obs_extended(self.state(), self.current_dice(), tmp);
                                        for (int i = 0; i < bg::OBS_EXTENDED_DIM; ++i) buf(i) = tmp[i];
                                        return arr;
                                            })

                                        // legal moves (пока stub)
                                                .def("legal_moves", [](bg::BackgammonEnv& self) {
                                                std::vector<bg::Move> moves;
                                                self.legal_moves(moves);

                                                // uint8 numpy: shape = [num_moves, 8]
                                                // layout: from1,to1,from2,to2,from3,to3,from4,to4
                                                py::array_t<uint8_t> arr({ (py::ssize_t)moves.size(), (py::ssize_t)8 });
                                                auto a = arr.mutable_unchecked<2>();

                                                for (py::ssize_t i = 0; i < (py::ssize_t)moves.size(); ++i) {
                                                    for (int k = 0; k < 4; ++k) {
                                                        a(i, 2 * k) = moves[i].from[k];
                                                        a(i, 2 * k + 1) = moves[i].to[k];
                                                    }
                                                }
                                                return arr;
                                                    })

                                                // apply move by index from freshly generated legal list (удобно, но медленно)
                                                        .def("step_index", [](bg::BackgammonEnv& self, int idx) {
                                                        std::vector<bg::Move> moves;
                                                        self.legal_moves(moves);
                                                        if (idx < 0 || idx >= (int)moves.size()) {
                                                            throw std::runtime_error("step_index: idx out of range");
                                                        }
                                                        bool done = false;
                                                        float reward = self.step_apply(moves[idx], done);
                                                        return py::make_tuple(reward, done);
                                                            })

                                                        // apply move passed from Python: mv shape (8,)
                                                                .def("step_move", [](bg::BackgammonEnv& self,
                                                                    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mv) {
                                                                        if (mv.ndim() != 1 || mv.shape(0) != 8) {
                                                                            throw std::runtime_error("step_move: expected uint8 array with shape (8,)");
                                                                        }
                                                                        auto a = mv.unchecked<1>();

                                                                        bg::Move m;
                                                                        for (int k = 0; k < 4; ++k) {
                                                                            m.from[k] = a(2 * k);
                                                                            m.to[k] = a(2 * k + 1);
                                                                        }

                                                                        bool done = false;
                                                                        float reward = self.step_apply(m, done);
                                                                        return py::make_tuple(reward, done);
                                                                    })
                                                                ;
}