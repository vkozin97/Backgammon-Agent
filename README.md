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
