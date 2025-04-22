#!/bin/bash

NUM_TRAIN_DOCS=100_000 # for a small experiment
# NUM_TRAIN_DOCS=5_000_000 # actual training experiment

python scripts/train_model.py \
    --model-type lz \
    --model-save-dir data/object/lz \
    --embedding-dim 256 \
    --lz-pca \
    --lz-num-train-docs $NUM_TRAIN_DOCS \
    --lz-max-spa-size 15_000_000_000 \
    --lz-save-interval 50_000 \
    --weight-type log_loss \
    --device cpu