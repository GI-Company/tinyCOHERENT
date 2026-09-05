#!/bin/bash
set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <steps> <in_model> <out_model> [extra_flags...]"
    exit 1
fi

STEPS=$1
IN_MODEL=$2
OUT_MODEL=$3
shift 3
EXTRA_FLAGS="$@"
LEDGER="training_ledger_v2.csv"

# Ensure ledger exists with header
if [ ! -f "$LEDGER" ]; then
    echo "block_end_model,val_loss,del_k1,del_k2,del_k3,del_k5,avg_del,rand_corr,sample_hash" > "$LEDGER"
fi

echo "=== Running $STEPS steps: $IN_MODEL -> $OUT_MODEL ==="

# If IN_MODEL is "scratch", we start from scratch. Otherwise we resume.
if [ "$IN_MODEL" = "scratch" ]; then
    ./build/train_scale --out "$OUT_MODEL" --steps "$STEPS" $EXTRA_FLAGS
else
    ./build/train_scale --resume "$IN_MODEL" --out "$OUT_MODEL" --steps "$STEPS" $EXTRA_FLAGS
fi

echo "=== Extracting Validation Loss ==="
VAL_OUT=$(./build/eval_frozen "$OUT_MODEL")
VAL_LOSS=$(echo "$VAL_OUT" | awk '/Exact Validation Loss:/ {print $4}')

echo "=== Running Faithcheck ==="
FAITH_OUT=$(./build/faithcheck "$OUT_MODEL" - --csv)

echo "=== Generating Samples and Hashing ==="
SAMPLE_OUT=$(./build/official_samples "$OUT_MODEL")
SAMPLE_HASH=$(echo "$SAMPLE_OUT" | shasum -a 256 | awk '{print $1}')

echo "=== Appending to Ledger ==="
echo "$OUT_MODEL,$VAL_LOSS,$FAITH_OUT,$SAMPLE_HASH" >> "$LEDGER"

echo "Ledger updated."
