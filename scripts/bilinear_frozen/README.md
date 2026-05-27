# `scripts/bilinear_frozen` — L2a: frozen backbone, bilinear decoder

Backbone заморожен полностью. Обучается только bilinear decoder (~388K параметров).

Отличие от `upsamp_frozen`: апсемплинг фиксированный (билинейная интерполяция),
без обучаемых ConvTranspose2d. Позволяет проверить, важен ли обучаемый апсемплинг.

Результаты: `results/bilinear_frozen/`

## Обучение

```bash
# pretrained backbone (frozen) + bilinear decoder:
uv run python scripts/bilinear_frozen/train.py \
    --ckpt models/morph-Ti-FM-max_ar1_ep225.pth \
    --device cuda:0 --seed 42 --max-samples 10000

# random backbone (frozen, контроль):
uv run python scripts/bilinear_frozen/train.py \
    --device cuda:0 --seed 42 --max-samples 10000
```

## Валидация

```bash
uv run python scripts/bilinear_frozen/validate.py \
    --weights results/bilinear_frozen/pretrained/N10000/seed42/morphgse_best.pth \
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
| `--out-dir` | `results/bilinear_frozen/{pretrained\|random}/N{N}/seed{seed}/` | Директория |

## Архитектура

```
Вход: (B, 1, 3, 1, 1, 72, 72)
    ↓  MORPH backbone (ViT3DRegression, frozen)
    ↓  z: (B, 1, 81, 256)
    ↓  reshape → (B, 256, 9, 9)
    ↓  BilinearDecoder: bilinear 9→18→36→72 + Conv3×3 после каждого шага
    ↓  crop :65, :65
Выход: (B, 65, 65)
```

Декодер: 3 стадии F.interpolate(×2) + Conv3×3 + GroupNorm + GELU + Conv1×1.
Параметры: ~388K в декодере, ~9.86M в backbone (заморожен).
