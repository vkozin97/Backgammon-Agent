# Backgammon-Agent

Высокопроизводительная среда для коротких нард (Backgammon) с C++-ядром и Python-обвязкой через `pybind11`.

## Цель проекта

Проект готовит базу для RL/self-play обучения:
- быстрый и детерминируемый игровой core на C++;
- Python API для экспериментов, обучения и визуального дебага;
- минимальные и расширенные наблюдения (observation) для будущих policy/value моделей.

## Текущий статус (актуально)

### Реализовано
- Игровая логика среды в `bg/src/env.cpp`.
- Генерация наблюдений в `bg/src/obs.cpp`.
- Python-модуль `bg_env` через `pybind11` (`bg/src/bindings.cpp`).
- CLI-утилита для проверки логики (`bg/src/main_cli.cpp`).
- Pygame viewer для ручной проверки ходов (`python/viewer_pygame.py`).
- Минимальный smoke-тест Python API (`python/test_env.py`).

### В работе / не реализовано в репозитории
- Пайплайн RL-обучения (self-play loop, replay buffer, тренировка сети).
- Метрики качества агента, baseline-оценка и автоматические эксперименты.
- CI-процессы (сборка/тесты) и пакетная дистрибуция Python-модуля.

## Структура репозитория

```text
Backgammon-Agent/
├─ bg/
│  ├─ CMakeLists.txt
│  └─ src/
│     ├─ env.cpp / env.h         # core-логика игры
│     ├─ obs.cpp / obs.h         # compact/extended observation
│     ├─ bindings.cpp            # pybind11 bindings
│     ├─ ascii.cpp / ascii.h     # ASCII визуализация
│     └─ main_cli.cpp            # CLI проверка
├─ python/
│  ├─ requirements.txt           # зависимости Python (без локальных путей)
│  ├─ test_env.py                # smoke test Python API
│  └─ viewer_pygame.py           # визуализация и ручная отладка
└─ README.md
```

## Требования

Проверенный стек пользователя:
- Python `3.9.25`
- pip `26.0.1`

Также требуется:
- CMake `>= 3.16`
- C++20 компилятор (MSVC / GCC / Clang)

## Установка зависимостей (Python)

```bash
pip install -r python/requirements.txt
```

## Сборка

### Windows (MSVC, пример)

```bash
cd bg
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Согласно `bg/CMakeLists.txt`, собранный модуль `bg_env.pyd` складывается в директорию `python/`.

## Быстрая проверка

После сборки из корня репозитория:

```bash
python python/test_env.py
```

Ожидается успешный импорт `bg_env`, генерация legal moves и один корректный вызов `step_move`.

## Технические детали API

### C++ класс среды
Ключевые методы (`BackgammonEnv`):
- `reset_standard()`
- `roll_dice()`
- `current_dice() const`
- `legal_moves(std::vector<Move>& out) const`
- `step_apply(const Move& m, bool& done)`
- `get_state_raw(int16_t out[53]) const`

### Raw state
Формат: `int16[53]`
- `0..23` — шашки текущего игрока
- `24..47` — шашки соперника
- `48` — mine_bar
- `49` — mine_off
- `50` — opp_bar
- `51` — opp_off
- `52` — ply

### Observation
- `Compact`: 55 float
- `Extended`: 61 float (доп. фичи позиции вычисляются в C++)

## Примечания

- Проект сейчас ориентирован на correctness и скорость env-шага.
- Для обучения нейросети потребуется отдельный training-контур поверх текущего API.
