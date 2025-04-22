#!/bin/bash

NUM_TRAIN_DOCS=10_000 # for a small experiment
# NUM_TRAIN_DOCS=400_000 # actual training experiment

python scripts/train_model.py \
    --model-type ngram \
    --model-save-dir data/object/ngram \
    --embedding-dim 256 \
    --ngram-pca \
    --ngram-num-train-docs $NUM_TRAIN_DOCS \
    --device cuda:7 \
    --overwrite