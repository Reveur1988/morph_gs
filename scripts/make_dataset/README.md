# make_dataset — пайплайн сборки датасета

Четыре скрипта для сборки `gs_dataset_v2.h5` из данных FAIR-MAST.
Запускаются последовательно из корня репозитория.

---

## Шаг 1 — Обнаружение разрядов (`01_discover_shots.py`)

Сканирует FAIR-MAST S3, фильтрует пригодные разряды, сохраняет манифест с разбивкой train/val/test.

```bash
uv run python scripts/make_dataset/01_discover_shots.py \
    --out data/shot_manifest.json \
    --n-shots 600 \
    --workers 32
```

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--out` | `shot_manifest.json` | Путь к выходному манифесту |
| `--start` | `10000` | Начало диапазона shot ID |
| `--end` | `30473` | Конец диапазона shot ID |
| `--n-candidates` | `8000` | Сколько ID сканировать |
| `--n-shots` | `500` | Целевое число разрядов |
| `--workers` | `32` | Потоки для параллельного сканирования |
| `--seed` | `42` | Seed для воспроизводимости |
| `--cache` | — | Путь к кешу (повтор без S3-сканирования) |
| `--min-valid-slices` | `30` | Минимум плазменных срезов на разряд |

Быстрый тест:
```bash
uv run python scripts/make_dataset/01_discover_shots.py \
    --start 28000 --end 30500 --n-candidates 200 --n-shots 50 \
    --out data/shot_manifest_test.json
```

Выходные файлы: `shot_manifest.json`, `shot_manifest_rejected.json`.

---

## Шаг 2 — Обработка разрядов на Dask-кластере (`02_dask_process_shots.py`)

Для каждого разряда из манифеста: скачивает данные с FAIR-MAST S3, запускает cold-solve
FreeGSNKE, сохраняет сошедшиеся срезы в per-shot NPZ.

Проверить доступность кластера:
```bash
uv run python -c "
from dask.distributed import Client
c = Client('tcp://192.168.0.103:8786')
w = c.scheduler_info()['workers']
print(f'workers={len(w)}, total_threads={sum(v[\"nthreads\"] for v in w.values())}')
c.close()
"
```

Основной запуск:
```bash
uv run python scripts/make_dataset/02_dask_process_shots.py \
    --scheduler tcp://192.168.0.103:8786 \
    --shot-list data/shot_manifest.json \
    --shots-dir data/shots \
    2>&1 | tee data/log_dask.txt
```

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--scheduler` | — | Адрес Dask scheduler (`tcp://HOST:8786`) |
| `--shot-list` | — | Манифест от шага 1 |
| `--shots-dir` | — | Директория для per-shot NPZ |
| `--n-times` | `30` | Число временных срезов на разряд |
| `--batch-size` | `0` | Лимит разрядов (0 = все) |
| `--log-out` | `<shots-dir>/run_log.json` | JSON-лог результатов |
| `--dry-run` | — | Показать задачи без отправки |

Скрипт идемпотентен: уже обработанные разряды (NPZ существует) пропускаются.

---

## Шаг 3 — Сборка HDF5 (`03_build_dataset.py`)

Собирает per-shot NPZ в единый self-contained HDF5-датасет.

```bash
uv run python scripts/make_dataset/03_build_dataset.py \
    --shots-dir data/shots \
    --out data/gs_dataset_v2.h5 \
    --stats-out data/gs_dataset_v2.stats.json
```

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--shots-dir` | — | Директория с per-shot NPZ |
| `--out` | — | Путь к выходному HDF5 |
| `--stats-out` | `<out>.stats.json` | JSON с нормировочной статистикой |
| `--gzip` | `4` | Уровень gzip-сжатия |
| `--dry-run` | — | Сводка без записи файлов |

После завершения `data/gs_dataset_v2.h5` содержит всё необходимое для обучения.

Текущий датасет (v2): **1 979 разрядов, 30 528 образцов**
(train=21 173 / val=4 696 / test=4 659), ~1.9 GB.

Нормировочная статистика вычисляется только по train split.

---

## Шаг 4 — Верификация (`04_verify_dataset.py`)

Структурная и физическая проверка HDF5-файла.

```bash
# Полная проверка (включает K5 cold-solve, ~30 сек)
uv run python scripts/make_dataset/04_verify_dataset.py --h5 data/gs_dataset_v2.h5

# Быстрая структурная проверка без запуска солвера
uv run python scripts/make_dataset/04_verify_dataset.py --h5 data/gs_dataset_v2.h5 --no-k5
```
