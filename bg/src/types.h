#pragma once
#include <array>
#include <cstdint>

namespace bg {

static constexpr uint8_t BAR = 24;
static constexpr uint8_t OFF = 25;

struct Dice {
  uint8_t a{1}; // 1..6
  uint8_t b{1}; // 1..6
  bool is_double() const { return a == b; }
};

struct Move {
  // до 4 шагов (дубль). 255 = пусто.
  std::array<uint8_t, 4> from{255,255,255,255};
  std::array<uint8_t, 4> to  {255,255,255,255};
};

struct State {
  // Канонический вид: "текущий игрок" всегда в points/bar/off,
  // соперник — в opp_points/opp_bar/opp_off.
  std::array<uint8_t, 24> points{};
  std::array<uint8_t, 24> opp_points{};
  uint8_t bar{0}, opp_bar{0};
  uint8_t off{0}, opp_off{0};
  uint8_t ply{0};  // номер полухода (для отладки/истории)
};

} // namespace bg