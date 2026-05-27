# `scripts/upsamp_warmup` — L3: full fine-tune, upsampling decoder (warmup + LLRD)

Двухфазная процедура обучения: сначала прогрев decoder при замороженном backbone,
затем полное дообучение всех весов с layer-wise LR decay (LLRD).

Результаты: `results/upsamp_warmup/`

## Обучение

```bash
# pretrained + warmup + LLRD:
uv run python scripts/upsamp_warmup/train.py \
    --ckpt models/morph-Ti-FM-max_ar1_ep225.pth \
    --device cuda:0 --seed 42 --max-samples 10000

# random init + warmup + LLRD (контроль):
uv run python scripts/upsamp_warmup/train.py \
    --device cuda:0 --seed 42 --max-samples 10000

# ablation: без LLRD (flat LR):
uv run python scripts/upsamp_warmup/train.py \
    --ckpt models/morph-Ti-FM-max_ar1_ep225.pth \
    --device cuda:0 --seed 42 --max-samples 10000 --no-llrd \
    --out-dir results/upsamp_warmup/ablation/pretrained_noLLRD_N10000_seed42
```

## Валидация

```bash
uv run python scripts/upsamp_warmup/validate.py \
    --weights results/upsamp_warmup/pretrained/N10000/seed42/morphgse_best.pth \
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
| `--total-epochs` | `200` | Суммарное число эпох (warmup + main) |
| `--warmup-epochs` | `5` | Эпох в фазе 1 (decoder only) |
| `--batch-size` | `16` | Размер батча |
| `--lr-decoder` | `1e-3` | LR декодера во всех фазах |
| `--lr-backbone-top` | `1e-5` | LR верхнего блока backbone в фазе 2 |
| `--llrd-factor` | `0.75` | Коэффициент LLRD на блок |
| `--no-llrd` | flag | Ablation: единый lr для всего backbone |
| `--weight-decay` | `1e-2` | Weight decay AdamW |
| `--early-stopping-patience` | `20` | Эпох без улучшения до останова (фаза 2) |
| `--out-dir` | авто | Директория для чекпоинта и лога |
| `--h5` | `data/gs_dataset_v2.h5` | Датасет |
| `--stats` | `data/gs_dataset_v2.stats.json` | Нормировочная статистика |

## Архитектура

```
Вход: (B, 1, 3, 1, 1, 72, 72)
    ↓  MORPH backbone (ViT3DRegression, дообучается с LLRD)
    ↓  z: (B, 1, 81, 256)
    ↓  reshape → (B, 256, 9, 9)
    ↓  UpsamplingDecoder: ConvTranspose2d 9→18→36→72
    ↓  crop :65, :65
Выход: (B, 65, 65)
```

Параметры: ~883K в декодере, ~9.86M в backbone. Всего ~10.75M.

## Исторические данные

Предыдущие запуски этого скрипта (под именем `morph_adapted_warmup`) лежат в
`results/morph_warmup/`.
