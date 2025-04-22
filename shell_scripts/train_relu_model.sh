#!/bin/bash

python scripts/train_model.py \
    --model-type relu \
    --model-save-dir data/object/relu \
    --embedding-dim 256 \
    --classification-problem-json mteb_info/classification_problems.json \
    --classifier-epochs 10 \
    --device cuda:7