#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "env.h"
#include "obs.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

class BatchedBackgammonEnv {
public:
    BatchedBackgammonEnv(size_t n_matches, py::object n_games, py::object endless_mode, uint64_t seed = 0) {
        envs_.reserve(n_matches);

        auto cast_ints = [&](py::object value, const char* name) -> std::vector<int> {
            std::vector<int> out;
            if (py::isinstance<py::int_>(value)) {
                out.assign(n_matches, value.cast<int>());
                return out;
            }
            if (!py::isinstance<py::sequence>(value)) {
                throw std::runtime_error(std::string(name) + " must be int or sequence[int]");
            }
            py::sequence seq = value.cast<py::sequence>();
            if (static_cast<size_t>(seq.size()) != n_matches) {
                throw std::runtime_error(std::string(name) + " sequence length must be equal to n_matches");
            }
            out.reserve(n_matches);
            for (size_t i = 0; i < n_matches; ++i) out.push_back(seq[i].cast<int>());
            return out;
        };

        auto cast_bools = [&](py::object value, const char* name) -> std::vector<bool> {
            std::vector<bool> out;
            if (py::isinstance<py::bool_>(value)) {
                out.assign(n_matches, value.cast<bool>());
                return out;
            }
            if (!py::isinstance<py::sequence>(value)) {
                throw std::runtime_error(std::string(name) + " must be bool or sequence[bool]");
            }
            py::sequence seq = value.cast<py::sequence>();
            if (static_cast<size_t>(seq.size()) != n_matches) {
                throw std::runtime_error(std::string(name) + " sequence length must be equal to n_matches");
            }
            out.reserve(n_matches);
            for (size_t i = 0; i < n_matches; ++i) out.push_back(seq[i].cast<bool>());
            return out;
        };

        const std::vector<int> n_games_vec = cast_ints(n_games, "n_games");
        const std::vector<bool> endless_mode_vec = cast_bools(endless_mode, "endless_mode");
        for (size_t i = 0; i < n_matches; ++i) {
            envs_.emplace_back(seed + static_cast<uint64_t>(i), n_games_vec[i], endless_mode_vec[i]);
        }
    }

    size_t size() const { return envs_.size(); }

    void reset() {
        for (auto& env : envs_) env.reset_standard();
    }

    py::array_t<uint8_t> roll_dice() {
        py::array_t<uint8_t> arr({(py::ssize_t)envs_.size(), (py::ssize_t)2});
        auto out = arr.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < (py::ssize_t)envs_.size(); ++i) {
            auto d = envs_[i].roll_dice();
            out(i, 0) = d.a;
            out(i, 1) = d.b;
        }
        return arr;
    }

    py::list legal_moves(bool unique_states = false) const {
        py::list out;
        for (const auto& env : envs_) {
            std::vector<bg::Move> moves;
            auto [double_possible, _] = env.legal_moves(moves, unique_states);
            py::array_t<uint8_t> arr({(py::ssize_t)moves.size(), (py::ssize_t)8});
            auto a = arr.mutable_unchecked<2>();
            for (py::ssize_t r = 0; r < (py::ssize_t)moves.size(); ++r) {
                for (int k = 0; k < 4; ++k) {
                    a(r, 2 * k) = moves[r].from[k];
                    a(r, 2 * k + 1) = moves[r].to[k];
                }
            }
            out.append(py::make_tuple(double_possible, std::move(arr)));
        }
        return out;
    }


    py::array_t<float> get_obs_extended(size_t n_threads = 0) const {
        py::array_t<float> arr({(py::ssize_t)envs_.size(), (py::ssize_t)bg::OBS_EXTENDED_DIM});
        auto out = arr.mutable_unchecked<2>();

        auto worker = [&](size_t begin, size_t end) {
            float tmp[bg::OBS_EXTENDED_DIM];
            for (size_t i = begin; i < end; ++i) {
                bg::get_obs_extended(envs_[i].state(), envs_[i].current_dice(), envs_[i].mine_score(), envs_[i].opp_score(), envs_[i].dave_value(), envs_[i].n_games(), envs_[i].cube_available_mine(), envs_[i].cube_available_opp(), envs_[i].is_crawford_game(), envs_[i].double_offered(), tmp);
                for (int j = 0; j < bg::OBS_EXTENDED_DIM; ++j) {
                    out((py::ssize_t)i, j) = tmp[j];
                }
            }
        };

        const size_t n = envs_.size();
        size_t workers = n_threads;
        if (workers == 0) {
            workers = std::max<size_t>(1, std::thread::hardware_concurrency());
        }
        if (workers <= 1 || n < 2) {
            worker(0, n);
            return arr;
        }
        workers = std::min(workers, n);

        std::vector<std::thread> threads;
        threads.reserve(workers);
        const size_t chunk = (n + workers - 1) / workers;
        for (size_t t = 0; t < workers; ++t) {
            const size_t begin = t * chunk;
            if (begin >= n) break;
            const size_t end = std::min(n, begin + chunk);
            threads.emplace_back(worker, begin, end);
        }
        for (auto& th : threads) th.join();

        return arr;
    }
    py::array_t<int16_t> get_states_raw() const {
        py::array_t<int16_t> arr({(py::ssize_t)envs_.size(), (py::ssize_t)69});
        auto out = arr.mutable_unchecked<2>();
        int16_t tmp[69];
        for (py::ssize_t i = 0; i < (py::ssize_t)envs_.size(); ++i) {
            envs_[i].get_state_raw(tmp);
            for (int j = 0; j < 69; ++j) out(i, j) = tmp[j];
        }
        return arr;
    }

    void set_states_raw(py::array_t<int16_t, py::array::c_style | py::array::forcecast> states) {
        if (states.ndim() != 2 || states.shape(0) != (py::ssize_t)envs_.size() || states.shape(1) != 69) {
            throw std::runtime_error("set_states_raw: expected int16 array with shape (N, 69)");
        }
        auto a = states.unchecked<2>();
        for (py::ssize_t i = 0; i < (py::ssize_t)envs_.size(); ++i) {
            int16_t tmp69[69];
            for (int j = 0; j < 69; ++j) tmp69[j] = a(i, j);
            envs_[i].set_state_full(tmp69);
        }
    }

    py::tuple step_apply(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> moves_arr,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> apply_doubles_arr,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> accept_doubles_arr
    ) {
        if (moves_arr.ndim() != 2 || moves_arr.shape(0) != (py::ssize_t)envs_.size() || moves_arr.shape(1) != 8) {
            throw std::runtime_error("step_apply: expected uint8 array with shape (N, 8)");
        }
        if (apply_doubles_arr.ndim() != 1 || apply_doubles_arr.shape(0) != (py::ssize_t)envs_.size()) {
            throw std::runtime_error("step_apply: expected apply_doubles shape (N,)");
        }
        if (accept_doubles_arr.ndim() != 1 || accept_doubles_arr.shape(0) != (py::ssize_t)envs_.size()) {
            throw std::runtime_error("step_apply: expected accept_doubles shape (N,)");
        }
        auto mv = moves_arr.unchecked<2>();
        auto ad = apply_doubles_arr.unchecked<1>();
        auto ac = accept_doubles_arr.unchecked<1>();
        py::array_t<float> rewards({(py::ssize_t)envs_.size()});
        py::array_t<uint8_t> accepted({(py::ssize_t)envs_.size()});
        py::array_t<uint8_t> done({(py::ssize_t)envs_.size()});
        auto r = rewards.mutable_unchecked<1>();
        auto a = accepted.mutable_unchecked<1>();
        auto d = done.mutable_unchecked<1>();

        for (py::ssize_t i = 0; i < (py::ssize_t)envs_.size(); ++i) {
            bg::Move m;
            for (int k = 0; k < 4; ++k) {
                m.from[k] = mv(i, 2 * k);
                m.to[k] = mv(i, 2 * k + 1);
            }
            auto [reward, _dave_after, accepted_code, done_code] = envs_[i].step_apply(ad(i), m, ac(i));
            r(i) = reward;
            a(i) = accepted_code;
            d(i) = done_code;
        }
        return py::make_tuple(rewards, accepted, done);
    }

private:
    std::vector<bg::BackgammonEnv> envs_;
};

}  // namespace

PYBIND11_MODULE(batched_bg_env, m) {
    m.doc() = "Batched Backgammon C++ env (pybind11)";

    py::class_<BatchedBackgammonEnv>(m, "Env")
        .def(py::init<size_t, py::object, py::object, uint64_t>(), py::arg("n_matches"), py::arg("n_games"), py::arg("endless_mode") = false, py::arg("seed") = 0)
        .def("size", &BatchedBackgammonEnv::size)
        .def("reset", &BatchedBackgammonEnv::reset)
        .def("roll_dice", &BatchedBackgammonEnv::roll_dice)
        .def("legal_moves", &BatchedBackgammonEnv::legal_moves, py::arg("unique_states") = false)
        .def("get_states_raw", &BatchedBackgammonEnv::get_states_raw)
        .def("set_states_raw", &BatchedBackgammonEnv::set_states_raw)
                .def("step_apply", &BatchedBackgammonEnv::step_apply, py::arg("moves_arr"), py::arg("apply_doubles_arr"), py::arg("accept_doubles_arr"))
        .def("get_obs_extended", &BatchedBackgammonEnv::get_obs_extended, py::arg("n_threads") = 0);
}
