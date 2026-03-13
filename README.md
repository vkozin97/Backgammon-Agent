# Backgammon-Agent

Высокопроизводительная среда для коротких нард (Backgammon) с C++-ядром и Python-обвязкой через `pybind11`.

## Цель проекта

Проект включает:
- быстрый и детерминируемый игровой core на C++;
- Python API для экспериментов, обучения и визуального дебага;
- value-only training pipeline для 12 агентов (4 архитектуры по 3 агента).

## Что реализовано в value-only pipeline

В `python/training/` реализованы модули:
- `config.py` — полный набор гиперпараметров и загрузка/сохранение конфига;
- `agents.py` — 12 обучаемых агентов (группы A/B/C/D) с обязательным dropout;
- `league.py` — лига матчей каждый-с-каждым и против `RandomAgent`;
- `replay.py` — единый replay-buffer с recency+uniform sampling (80/20);
- `pipeline.py` — цикл эпох: генерация партий, обучение, метрики, checkpoint.
- `plotting.py` — отрисовка графиков по сохранённым метрикам.
- `render_plots.py` — CLI-скрипт для перерисовки графиков из checkpoint-ов без запуска новых эпох.

## Запуск обучения

```bash
python python/train_value_only.py
```

По умолчанию запускается 3 эпохи и сохраняются артефакты в:
- `python/training_stats/epoch_XXXX/`
- `python/training_stats/plots/winrates/`
- `python/training_stats/plots/loss/`

Для каждого агента сохраняются графики:
- по одному графику `winrate vs <opponent>` для каждого соперника;
- 1 график `train loss`.

Итого: число графиков winrate на агента равно числу соперников, плюс 1 график `train loss`.

## Перерисовка графиков без обучения

```bash
PYTHONPATH=python python python/render_plots.py --experiment-dir training_stats
```

## Тесты

```bash
PYTHONPATH=python pytest -q python/tests/test_training_pipeline.py
```

## Сборка C++ модуля окружения

```bash
cd bg
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Если `bg_env` недоступен, training pipeline использует встроенный fallback-env для smoke/CI прогона.

## Формат `extended_obs` (263 признака)

`extended_obs` возвращается из `env.get_obs_extended()` и имеет размер `263 = 10*24 + 23`.

### Векторные каналы (по 24 значения)
- `[0:24]` — `points`: шашки текущего игрока по пунктам.
- `[24:48]` — `opp_points`: шашки соперника по пунктам.
- `[48:72]` — `blots`: индикатор одиночной шашки текущего игрока (`1.0`, если ровно 1 шашка на пункте).
- `[72:96]` — `opp_blots`: то же для соперника.
- `[96:120]` — `anchors`: индикатор блока из 2+ шашек текущего игрока.
- `[120:144]` — `opp_anchors`: то же для соперника.
- `[144:168]` — `hit_prob_mine`: вероятность быть побитым для каждого blot текущего игрока (в долях от 36 исходов броска).
- `[168:192]` — `cover_prob_mine`: вероятность закрыть свой blot для каждого пункта.
- `[192:216]` — `hit_prob_opp`: аналог `hit_prob` для blot соперника.
- `[216:240]` — `cover_prob_opp`: аналог `cover_prob` для blot соперника.

### Скаляры (23 значения)
- `[240] bar`, `[241] off`, `[242] opp_bar`, `[243] opp_off`.
- `[244] pip_count_mine`, `[245] pip_count_opp`.
- `[246] blots_mine`, `[247] blots_opp`.
- `[248] anchors_mine`, `[249] anchors_opp`.
- `[250] blot_pips_mine`, `[251] blot_pips_opp`.
- `[252] anchor_pips_mine`, `[253] anchor_pips_opp`.
- `[254] mine_score`, `[255] opp_score`.
- `[256] dave_value` (значение куба удвоения).
- `[257] my_left_to_win`, `[258] opp_left_to_win` (в endless-режиме выставляются в `1.0`).
- `[259] cube_available_mine`, `[260] cube_available_opp`.
- `[261] is_crawford_game`, `[262] double_offered`.

## Выход агентов (`ValueAgent`)

По умолчанию модель имеет `output_dim = 31` и после `predict_proba(...)` выдаёт вектор вероятностей из 31 компоненты:

- `[:12]` — `my_match_head`: распределение по исходам «сколько очков осталось мне до победы в матче» (индексы `0..11`).
- `[12:24]` — `opp_match_head`: аналогично для соперника.
- `[24]` — `accept_prob`: вероятность принять удвоение (`sigmoid`).
- `[25:31]` — `reward_head`: распределение по награде из фиксированного набора `[-3, -2, -1, +1, +2, +3]`.

Практически в лиге используется:
- `p_win = probs[0]` как оценка вероятности победы в матче для текущего игрока;
- `accept_prob = probs[24]` для решения по принятию удвоения;
- матожидание награды: `dot(probs[25:31], [-3,-2,-1,1,2,3])`.
