#!/bin/bash

# cargo run -r -p lz78-experiments --bin train -- \
#     -s ./spa_outputs/c4-realnews \
#     -e c4 \
#     --data-dir ./data \
#     --start-at-root

cargo run -r -p lz78-experiments --bin train -- \
    -s ./spa_outputs/shakespeare \
    --dataset shakespeare \
    --data-dir ./data \
    --start-at-root