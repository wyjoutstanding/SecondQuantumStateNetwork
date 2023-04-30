#!/bin/bash

MOLECULE_LIST=(
    'H2' # 783
    # "H2O" # 962
    # "O2" # 977
    # "CO2" # 280
    # "CH4" # 297
    # "CNa2O3" # 10340
    # 'Aspirin' # 2244
    # 'VitaminC' # 54670067
)

BASIS_LIST=(
    'STO-3G'
    '3-21G'
    '6-31G'
    '6-311G*'
    '6-311+G*'
    '6-311++G**'
    '6-311++G(2df,2pd)'
)
for MOLECULE in "${MOLECULE_LIST[@]}"
do
    for BASIS in "${BASIS_LIST[@]}"
    do
        args=(
            --config_file './exp_configs/vmc.yaml' SYSTEM.RANDOM_SEED 666 MISC.NUM_TRIALS 1 MISC.SAME_LOCAL_SEED 'True'
            DDP.WORLD_SIZE 1 DDP.NODE_IDX 0 DDP.LOCAL_WORLD_SIZE 1 SYSTEM.NUM_GPUS 1 DDP.MASTER_ADDR 'vqmc-chemistry' DDP.MASTER_PORT 12653
            MODEL.MODEL_NAME 'made' MODEL.HIDDEN_DEPTH 1 MODEL.HIDDEN_WIDTH 64
            DATA.MOLECULE "${MOLECULE}" DATA.BASIS "${BASIS}" DATA.NUM_SAMPLES 1e12
            TRAIN.NUM_EPOCHS 50000 TRAIN.LEARNING_RATE 0.0005 TRAIN.OPTIMIZER_NAME 'adam' TRAIN.APPLY_SR 'False' TRAIN.BATCH_SIZE 500 TRAIN.INNER_ITER 1 TRAIN.ENABLE_UNIQS 'True'
        )
        CUDA_VISIBLE_DEVICES=0 python -m main "${args[@]}"
    done
done
