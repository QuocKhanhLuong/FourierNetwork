#!/bin/bash
# Test Advanced Architectures
# 1. Asymmetric DCN Depth (2-3-4 blocks per stage)
# 2. Hybrid Inverted Residual + DCN
# 3. With/Without PointRend

echo "========================================"
echo "Advanced Architecture Test"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Working dir: $(pwd)"

# Config
EPOCHS=100
BATCH_SIZE=8
LR=0.0001
DATA_DIR="preprocessed_data/ACDC"

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Run test
python src/training/test_advanced_arch.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR

echo ""
echo "✓ Advanced architecture test completed!"
