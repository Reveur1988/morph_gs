# mt-experiments

Адаптация предобученной PDE foundation-модели [MORPH](https://github.com/camlab-ethz/MORPH) к задаче генерации начального приближения ψ(R,Z) для итерационного решателя [FreeGSNKE](https://github.com/FusionComputingLab/freegsnke) уравнения Грэда–Шафранова. Данные — открытый набор разрядов токамака MAST [FAIR-MAST](https://github.com/ukaea/fair-mast).

**Идея.** Вместо полной замены решателя нейросеть генерирует начальное приближение (warm-start), которое подаётся в FreeGSNKE как стартовая точка нулевой итерации Ньютона–Крылова. Решатель остаётся финальным арбитром физической корректности; ускорение измеряется сокращением числа NK-итераций до сходимости.

**Результат.** Все четыре исследованных конфигурации достигают порогового отношения итераций ≤ 0,5 относительно холодного старта. Лучшая конфигурация (`bilinear_warmup`) даёт среднее отношение **0,3229** (95% ДИ [0,3129; 0,3334]) на тестовом подмножестве FAIR-MAST из 1000 точек — сокращение числа итераций ~68%.

## Структура репозитория

```
src/morph_gs/              — библиотека: препроцессинг, датасет, модель, конфиг солвера
scripts/
├── make_dataset/          — пайплайн сборки датасета из FAIR-MAST
├── bilinear_frozen/       — конфигурация A1: BilinearDecoder, заморожен backbone
├── bilinear_warmup/       — конфигурация A2: BilinearDecoder, полное дообучение (лучшая)
├── upsamp_frozen/         — конфигурация B1: UpsamplingDecoder, заморожен backbone
├── upsamp_warmup/         — конфигурация B2: UpsamplingDecoder, полное дообучение
└── analyze_results.py     — агрегация результатов и bootstrap CI
data/                      — датасет (gs_dataset_v2.h5, gs_dataset_v2.stats.json)
weights/                   — предобученные веса MORPH (morph-Ti-FM-max_ar1_ep225.pth)
results/                   — результаты обучения и валидации (генерируется скриптами)
MORPH/                     — исходный код MORPH (submodule)
freegsnke/                 — локальная сборка FreeGSNKE
```

## Архитектура

Модель **MorphGSE** состоит из двух компонентов:

- **Backbone** — `ViT3DRegression` с весами MORPH-Ti (`morph-Ti-FM-max_ar1_ep225.pth`): 4 трансформерных блока, dim=256, 4 головы внимания, patch_size=8, ~9,86 млн параметров. Принимает тензор `(B, 1, 3, 1, 1, 72, 72)` (три входных поля на сетке 72×72 с нулевым паддингом).

- **Выходной декодер** — восстанавливает пространственное разрешение от карты токенов 9×9 до поля ψ на сетке 65×65:
  - `BilinearDecoder` (~388 тыс. параметров): фиксированная билинейная интерполяция + обучаемые Conv2d-блоки
  - `UpsamplingDecoder` (~883 тыс. параметров): транспонированно-свёрточное повышение разрешения (3 стадии ConvTranspose2d)

**Входные поля** (на сетке 65×65, RMIN=0.06, RMAX=2.0, ZMIN=−2.0, ZMAX=2.0 м):
- `psi_init` — вакуумное поле от катушек (функции Грина)
- `pprime_map` — профиль p′(ψ), проецированный в 2D через нормировку потоковой координаты
- `ffprime_map` — профиль FF′(ψ) аналогично

**Целевое поле**: ψ(R,Z) на сетке 65×65, совместимой с FreeGSNKE.

**Функция потерь**: MSE в нормированном пространстве (по статистикам обучающей выборки).

## Четыре конфигурации

| Конфигурация | Декодер | Режим дообучения | Метрика C1 | 95% ДИ |
|---|---|---|---|---|
| A1: `bilinear_frozen` | BilinearDecoder | Заморожен backbone | 0,3997 | [0,3886; 0,4111] |
| A2: `bilinear_warmup` | BilinearDecoder | Полное + warmup/LLRD | **0,3229** | [0,3129; 0,3334] |
| B1: `upsamp_frozen` | UpsamplingDecoder | Заморожен backbone | 0,4460 | [0,4318; 0,4602] |
| B2: `upsamp_warmup` | UpsamplingDecoder | Полное + warmup/LLRD | 0,4081 | [0,3950; 0,4216] |

Все конфигурации достигают порогового значения C1 ≤ 0,5. Метрика C1 = среднее N_warm/N_cold по точкам, где оба запуска сошлись и финальные поля ψ согласованы (макс. абс. разность < 1% от размаха ψ холодного старта).

## Быстрый старт

### Предусловия

```bash
uv sync
```

> **Примечание.** Папки `data/`, `weights/` и файлы `*.pth` добавлены в `.gitignore` и не хранятся в репозитории — их нужно получить отдельно.

Датасет должен лежать в `data/gs_dataset_v2.h5` и `data/gs_dataset_v2.stats.json`.  
Инструкция по сборке датасета из FAIR-MAST — в [scripts/make_dataset/README.md](scripts/make_dataset/README.md).

Предобученные веса MORPH-Ti (`morph-Ti-FM-max_ar1_ep225.pth`) нужно положить в `weights/` — скачиваются с HuggingFace Hub через MORPH-репозиторий.

### Обучение

Каждая конфигурация запускается из соответствующей директории:

```bash
# Лучшая конфигурация: BilinearDecoder с полным дообучением
uv run python scripts/bilinear_warmup/train.py --seed 0
uv run python scripts/bilinear_warmup/train.py --seed 1
uv run python scripts/bilinear_warmup/train.py --seed 42

# Остальные конфигурации аналогично
uv run python scripts/bilinear_frozen/train.py --seed 0
uv run python scripts/upsamp_warmup/train.py --seed 0
uv run python scripts/upsamp_frozen/train.py --seed 0
```

Требования к железу: GPU с ≥24 ГБ памяти, Python 3.11.  
Результаты сохраняются в `results/<config>/seed<N>/`.

### Валидация (benchmark)

```bash
uv run python scripts/bilinear_warmup/validate.py \
    --ckpt results/bilinear_warmup/seed0/morphgse_best.pth \
    --seed 0
```

Выходной CSV содержит для каждой тестовой точки: `shot_id`, `iter_cold`, `iter_warm`, `ratio`, `converged_cold`, `converged_warm`, `converged_both`, `psi_consistent`.

### Агрегация результатов

```bash
uv run python scripts/analyze_results.py
```

Вычисляет метрику C1 и bootstrap 95% ДИ (≥2000 ресэмплов) по агрегированным CSV трёх сидов.

## Датасет

Источник: FAIR-MAST — архив разрядов установки MAST.  
Разбиение фиксировано на уровне `shot_id` (один разряд не попадает в два сплита):

| Сплит | Временных срезов | Разрядов |
|---|---|---|
| train | 21 173 | 1 369 |
| val | 4 696 | 305 |
| test | 4 659 | 305 |

Пайплайн сборки: `scripts/make_dataset/` — четыре скрипта от обнаружения разрядов до верификации HDF5-артефакта.  
Нормировочные статистики: `data/gs_dataset_v2.stats.json`.

## Конфигурация решателя

Параметры FreeGSNKE зафиксированы в `src/morph_gs/config.py`:
`SOLVER_TOL=1e-3`, `SOLVER_MAXITS=100`, `SOLVER_PICARD_HANDOVER=0.11`.  
Warm-start адаптер — `warm_solve` в `src/morph_gs/solver.py`.

## Цитирование

Работа выполнена в рамках ВКР на программе ИТМО. Набор данных: FAIR-MAST (2024). Базовая модель: MORPH.
