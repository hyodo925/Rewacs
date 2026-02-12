#!/bin/bash

MODES=(
  # learning_based_only
  # rule_based_only
  switching_all
  # switching_data_only
)

SEEDS=(0 1 2 3 4)
# SEEDS=(0)
# HUMANS=(5)
HUMANS=(5 6 7 8 9 10)

for HUMAN in "${HUMANS[@]}"; do
  for MODE in "${MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      uv run python3 finetune_awac_meta_critic_sep_sampling.py \
        --train-pre_explor False \
        --train-sep_sampling True \
        --train-finetune_mode "$MODE" \
        --train-random_seed "$SEED" \
        --sim-human_num "$HUMAN" 
    done
  done
done

