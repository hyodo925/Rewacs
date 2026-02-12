#!/bin/bash

MODES=(
  learning_based_only
  # rule_based_only
  # switching_all
  # switching_data_only
)

SEEDS=(0 1 2 3 4)
# HUMANS=(10)
HUMANS=(5)

for SEED in "${SEEDS[@]}"; do
  for MODE in "${MODES[@]}"; do
    for HUMAN in "${HUMANS[@]}"; do
      uv run python finetune_awac_with_flow_odpr.py \
        --train-finetune_mode "$MODE" \
        --train-random_seed "$SEED" \
        --sim-human_num "$HUMAN" 
    done
  done
done
