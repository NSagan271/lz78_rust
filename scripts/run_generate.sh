#!/bin/bash

# cargo run -r -p lz78-experiments --bin generate -- \
#     --save-path spa_outputs/c4-realnews \
#     --dataset c4 \
#     --topk 5\
#     -t 0.1 \
#     -n 1000 \
#     --seed-data "This"

cargo run -r -p lz78-experiments --bin generate -- \
    --save-path spa_outputs/shakespeare-ctw-1 \
    --dataset shakespeare \
    --seed-data "This" \
    --min-context 10 \
    --min-count 4 \
    --topk 5\
    -t 0.1 \
    -n 800