# `scripts/upsamp_frozen` — L1: frozen backbone, upsampling decoder

Backbone заморожен полностью. Обучается только upsampling decoder (~883K параметров).

Результаты: `results/upsamp_frozen/`

## Обучение

```bash
# pretrained backbone (frozen):
uv run python scripts/upsamp_frozen/train.py \
    --ckpt models/morph-Ti-FM-max_ar1_ep225.pth \
    --device cuda:0 --seed 42 --max-samples 10000

# random backbone (frozen, контроль):
uv run python scripts/upsamp_frozen/train.py \
    --device cuda:0 --seed 42 --max-samples 10000
```

## Валидация

```bash
uv run python scripts/upsamp_frozen/validate.py \
    --weights results/upsamp_frozen/pretrained/N10000/seed42/morphgse_best.pth \
    --n-pairs 1000 --seed 42 \
    --scheduler tcp://192.168.0.103:8786
```

## Параметры train.py

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--ckpt` | `None` | Предобученные веса MORPH FM (`.pth`). Без флага — random init |
| `--device` | `cpu` | `cuda:0`, `cuda:1`, `cpu` |
| `--seed` | `42` | Seed для воспроизводимости |
| `--max-samples` | `0` | Лимит обучающих пар (0 = весь train split) |
| `--epochs` | `500` | Максимальное число эпох |
| `--batch-size` | `16` | Размер батча |
| `--lr-decoder` | `1e-3` | Learning rate декодера |
| `--weight-decay` | `1e-2` | Weight decay AdamW |
| `--early-stopping-patience` | `20` | Эпох без улучшения до останова |
| `--early-stopping-min-epochs` | `10` | Минимум эпох перед ранним остановом |
| `--out-dir` | `results/upsamp_frozen/{pretrained\|random}/N{N}/seed{seed}/` | Директория |
| `--h5` | `data/gs_dataset_v2.h5` | Датасет |
| `--stats` | `data/gs_dataset_v2.stats.json` | Нормировочная статистика |

## Архитектура

```
Вход: (B, 1, 3, 1, 1, 72, 72)
    ↓  MORPH backbone (ViT3DRegression, frozen)
    ↓  z: (B, 1, 81, 256)
    ↓  reshape → (B, 256, 9, 9)
    ↓  UpsamplingDecoder: ConvTranspose2d 9→18→36→72
    ↓  crop :65, :65
Выход: (B, 65, 65)
```

Декодер: 3 стадии ConvTranspose2d(k=4, s=2) + Conv3×3 + GroupNorm + GELU + Conv1×1.
Параметры: ~883K в декодере, ~9.86M в backbone (заморожен).

## Исторические данные

Предыдущие запуски этого скрипта (под именем `morph_adapted --frozen`) лежат в
`results/morph_baseline/frozen/`.
