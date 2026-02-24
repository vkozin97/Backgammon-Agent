# Backgammon-Agent

Высокопроизводительная среда для коротких нард (Backgammon) с C++-ядром и Python-обвязкой через `pybind11`.

## Цель проекта

Проект включает:
- быстрый и детерминируемый игровой core на C++;
- Python API для экспериментов, обучения и визуального дебага;
- value-only training pipeline для 12 агентов (3 архитектуры по 4 агента).

## Что реализовано в value-only pipeline

В `python/training/` реализованы модули:
- `config.py` — полный набор гиперпараметров и загрузка/сохранение конфига;
- `agents.py` — 12 обучаемых агентов (группы A/B/C) с обязательным dropout;
- `league.py` — лига матчей каждый-с-каждым и против `RandomAgent`/`BaselineAgent`;
- `replay.py` — единый replay-buffer с recency+uniform sampling (80/20);
- `pipeline.py` — цикл эпох: генерация партий, обучение, метрики, checkpoint, plotting.

## Запуск обучения

```bash
python python/train_value_only.py
```

По умолчанию запускается 3 эпохи и сохраняются артефакты в:
- `python/artifacts/checkpoints/epoch_XXXX/`
- `python/artifacts/checkpoints/plots/`

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
