#!/bin/bash
python scripts/train_model.py \
    --model-type ngram \
    --model-save-dir data/object/ngram_no_pca \
    --no-ngram-pca \