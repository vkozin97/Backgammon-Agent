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
    BatchedBackgammonEnv(size_t n_envs, uint64_t seed = 0) {
        envs_.reserve(n_envs);
        for (size_t i = 0; i < n_envs; ++i) {
            envs_.emplace_back(seed + static_cast<uint64_t>(i));
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
            env.legal_moves(moves, unique_states);
            py::array_t<uint8_t> arr({(py::ssize_t)moves.size(), (py::ssize_t)8});
            auto a = arr.mutable_unchecked<2>();
            for (py::ssize_t r = 0; r < (py::ssize_t)moves.size(); ++r) {
                for (int k = 0; k < 4; ++k) {
                    a(r, 2 * k) = moves[r].from[k];
                    a(r, 2 * k + 1) = moves[r].to[k];
                }
            }
            out.append(std::move(arr));
        }
        return out;
    }


    py::array_t<float> get_obs_extended(size_t n_threads = 0) const {
        py::array_t<float> arr({(py::ssize_t)envs_.size(), (py::ssize_t)bg::OBS_EXTENDED_DIM});
        auto out = arr.mutable_unchecked<2>();

        auto worker = [&](size_t begin, size_t end) {
            float tmp[bg::OBS_EXTENDED_DIM];
            for (size_t i = begin; i < end; ++i) {
                bg::get_obs_extended(envs_[i].state(), envs_[i].current_dice(), tmp);
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
        py::array_t<int16_t> arr({(py::ssize_t)envs_.size(), (py::ssize_t)53});
        auto out = arr.mutable_unchecked<2>();
        int16_t tmp[53];
        for (py::ssize_t i = 0; i < (py::ssize_t)envs_.size(); ++i) {
            envs_[i].get_state_raw(tmp);
            for (int j = 0; j < 53; ++j) out(i, j) = tmp[j];
        }
        return arr;
    }

    void set_states_raw(py::array_t<int16_t, py::array::c_style | py::array::forcecast> states) {
        if (states.ndim() != 2 || states.shape(0) != (py::ssize_t)envs_.size() || states.shape(1) != 53) {
            throw std::runtime_error("set_states_raw: expected int16 array with shape (N, 53)");
        }
        auto a = states.unchecked<2>();
        int16_t tmp[53];
        for (py::ssize_t i = 0; i < (py::ssize_t)envs_.size(); ++i) {
            for (int j = 0; j < 53; ++j) tmp[j] = a(i, j);
            envs_[i].set_state_raw(tmp);
        }
    }

    py::tuple step_apply(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> moves_arr) {
        if (moves_arr.ndim() != 2 || moves_arr.shape(0) != (py::ssize_t)envs_.size() || moves_arr.shape(1) != 8) {
            throw std::runtime_error("step_apply: expected uint8 array with shape (N, 8)");
        }
        auto mv = moves_arr.unchecked<2>();
        py::array_t<float> rewards({(py::ssize_t)envs_.size()});
        py::array_t<uint8_t> done({(py::ssize_t)envs_.size()});
        auto r = rewards.mutable_unchecked<1>();
        auto d = done.mutable_unchecked<1>();

        for (py::ssize_t i = 0; i < (py::ssize_t)envs_.size(); ++i) {
            bg::Move m;
            for (int k = 0; k < 4; ++k) {
                m.from[k] = mv(i, 2 * k);
                m.to[k] = mv(i, 2 * k + 1);
            }
            bool is_done = false;
            r(i) = envs_[i].step_apply(m, is_done);
            d(i) = is_done ? 1 : 0;
        }
        return py::make_tuple(rewards, done);
    }

private:
    std::vector<bg::BackgammonEnv> envs_;
};

}  // namespace

PYBIND11_MODULE(batched_bg_env, m) {
    m.doc() = "Batched Backgammon C++ env (pybind11)";

    py::class_<BatchedBackgammonEnv>(m, "Env")
        .def(py::init<size_t, uint64_t>(), py::arg("n_envs"), py::arg("seed") = 0)
        .def("size", &BatchedBackgammonEnv::size)
        .def("reset", &BatchedBackgammonEnv::reset)
        .def("roll_dice", &BatchedBackgammonEnv::roll_dice, py::call_guard<py::gil_scoped_release>())
        .def("legal_moves", &BatchedBackgammonEnv::legal_moves, py::arg("unique_states") = false, py::call_guard<py::gil_scoped_release>())
        .def("get_states_raw", &BatchedBackgammonEnv::get_states_raw, py::call_guard<py::gil_scoped_release>())
        .def("set_states_raw", &BatchedBackgammonEnv::set_states_raw, py::call_guard<py::gil_scoped_release>())
        .def("step_apply", &BatchedBackgammonEnv::step_apply, py::call_guard<py::gil_scoped_release>())
        .def("get_obs_extended", &BatchedBackgammonEnv::get_obs_extended, py::arg("n_threads") = 0, py::call_guard<py::gil_scoped_release>());
}
