#!/bin/bash

# cargo run -r -p lz78-experiments --bin train -- \
#     -s ./spa_outputs/c4-realnews \
#     -e c4 \
#     --data-dir ./data \
#     --start-at-root

cargo run -r -p lz78-experiments --bin train -- \
    -s ./spa_outputs/c4-ctw-1a \
    --dataset c4 \
    --data-dir ./data \
    --spa-type lz78ctw \
    --gamma 0.1 \
    --start-at-root