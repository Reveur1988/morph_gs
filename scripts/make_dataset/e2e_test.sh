#!/usr/bin/env bash
# E2E pipeline test using only morph_gs package CLIs.
#
# Usage:
#   ./scripts/e2e_test.sh                         # 5 epochs (default)
#   ./scripts/e2e_test.sh --epochs 20
#   ./scripts/e2e_test.sh --epochs 5 --skip-download
#   ./scripts/e2e_test.sh --clean

set -euo pipefail
trap 'echo ""; echo "❌  E2E test FAILED at line $LINENO. Workspace: $WORKSPACE"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

WORKSPACE="$REPO_ROOT/experiments/14_refactor_src/e2e_workspace"
FM_CKPT="$REPO_ROOT/models/morph-Ti-FM-max_ar1_ep225.pth"
EPOCHS=5
SKIP_DOWNLOAD=0

# ── parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)        EPOCHS="$2";    shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1; shift   ;;
        --clean)
            echo "Cleaning workspace: $WORKSPACE"
            rm -rf "$WORKSPACE"
            echo "Done."
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$WORKSPACE/shots" "$WORKSPACE/checkpoints" "$WORKSPACE/results"

SHOTS=(30425 27385 24920)
declare -A SHOT_SPLIT=([30425]="train" [27385]="val" [24920]="test")

# ── STEP 1/6: FM checkpoint ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo "STEP 1/6: FM checkpoint pre-check"
echo "========================================"

if [ ! -f "$FM_CKPT" ]; then
    echo "FM checkpoint not found. Downloading from HuggingFace..."
    mkdir -p "$REPO_ROOT/models"
    uv run python - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os
from pathlib import Path

target = Path(os.environ.get("FM_CKPT", "models/morph-Ti-FM-max_ar1_ep225.pth"))
target.parent.mkdir(parents=True, exist_ok=True)
tmp = hf_hub_download(repo_id="mahindrautela/MORPH",
                      filename="models/FM/morph-Ti-FM-max_ar1_ep225.pth")
if Path(tmp) != target:
    shutil.copy(tmp, target)
print(f"Downloaded → {target}")
PYEOF
    if [ ! -f "$FM_CKPT" ]; then
        echo "ERROR: FM checkpoint download failed."
        echo "Manually place morph-Ti-FM-max_ar1_ep225.pth in models/"
        echo "Source: https://huggingface.co/mahindrautela/MORPH"
        exit 1
    fi
fi
echo "FM checkpoint OK: $FM_CKPT ($(du -h "$FM_CKPT" | cut -f1))"

# ── STEP 2/6: process shots → NPZ ────────────────────────────────────────────
echo ""
echo "========================================"
echo "STEP 2/6: Process shots (download + cold-solve)"
echo "========================================"

if [ "$SKIP_DOWNLOAD" -eq 1 ]; then
    echo "--skip-download set; skipping shot processing."
else
    for shot in "${SHOTS[@]}"; do
        npz="$WORKSPACE/shots/shot_${shot}.npz"
        if [ -f "$npz" ]; then
            echo "Shot $shot already processed: $npz"
        else
            echo "Processing shot $shot (split: ${SHOT_SPLIT[$shot]})..."
            uv run morph-gs-process-shot \
                --shot    "$shot" \
                --out-dir "$WORKSPACE/shots" \
                --split   "${SHOT_SPLIT[$shot]}"
        fi
    done
fi

# ── STEP 3/6: build HDF5 dataset ─────────────────────────────────────────────
echo ""
echo "========================================"
echo "STEP 3/6: Build HDF5 dataset"
echo "========================================"

H5="$WORKSPACE/gs_dataset_e2e.h5"
STATS_JSON="$WORKSPACE/gs_dataset_e2e.stats.json"

if [ -f "$H5" ] && [ -f "$STATS_JSON" ]; then
    echo "Dataset already exists: $H5"
else
    uv run morph-gs-build-dataset \
        --shots-dir "$WORKSPACE/shots" \
        --out        "$H5" \
        --stats-out  "$STATS_JSON"
    echo "Dataset written: $H5"
fi

# ── STEP 4/6: train ──────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "STEP 4/6: Train ($EPOCHS epochs)"
echo "========================================"

BEST_CKPT="$WORKSPACE/checkpoints/morphgs_ft1_lora_best.pth"

if [ -f "$BEST_CKPT" ]; then
    echo "Checkpoint already exists: $BEST_CKPT"
else
    uv run morph-gs-train \
        --h5            "$H5" \
        --stats         "$STATS_JSON" \
        --ckpt          "$FM_CKPT" \
        --out-dir       "$WORKSPACE/checkpoints" \
        --ft-level      1 \
        --lora-r-attn   16 --lora-r-mlp 16 --lora-alpha 32 --lora-p 0.05 \
        --lr-head       1e-3 --lr-lora 5e-4 --weight-decay 1e-2 \
        --batch-size    2 \
        --epochs        "$EPOCHS" \
        --seed          42 \
        --run-id        e2e \
        --early-stopping-patience 999 --early-stopping-min-epochs 999
    echo "Training done: $BEST_CKPT"
fi

# ── STEP 5/6: pair runs ───────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "STEP 5/6: Pair runs (cold + warm)"
echo "========================================"

PAIRS_CSV="$WORKSPACE/results/pair_runs.csv"

uv run morph-gs-eval-pairs \
    --h5      "$H5" \
    --weights "$BEST_CKPT" \
    --n-pairs 3 \
    --seed    42 \
    --device  cpu \
    --out     "$PAIRS_CSV"

echo "Pair runs written: $PAIRS_CSV"

# ── STEP 6/6: report ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "STEP 6/6: Report"
echo "========================================"

REPORT="$WORKSPACE/results/e2e_report.md"

BEST_VAL=$(python3 -c "
import json
try:
    log = json.load(open('$WORKSPACE/checkpoints/train_log.json'))
    print(log.get('meta',{}).get('early_stop',{}).get('best_val_loss','n/a'))
except Exception:
    print('n/a')
" 2>/dev/null)

cat > "$REPORT" <<EOF
# E2E Test Report
Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Epochs: $EPOCHS
Shots: ${SHOTS[*]}

## Train
Best val_loss: $BEST_VAL

## Pair Runs
$(column -t -s, < "$PAIRS_CSV" 2>/dev/null || cat "$PAIRS_CSV")

## Verdict
$([ -s "$PAIRS_CSV" ] && echo "✅ Pipeline alive" || echo "❌ Pipeline broken")
EOF

cat "$REPORT"
echo ""
echo "========================================"
echo "✅  E2E test completed."
echo "========================================"
