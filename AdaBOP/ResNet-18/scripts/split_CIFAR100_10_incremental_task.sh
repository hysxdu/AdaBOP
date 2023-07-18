#!/usr/bin/env bash
GPUID=0
OUTDIR=outputs_5runs/owm_ewc_based_combgrad_cifar100_10tasks_resnet_0901
mkdir -p $OUTDIR
REPEAT=5
# python -u main.py --schedule 20 40 60 80 --reg_coef 500 --model_lr 1e-4 --head_lr 1e-3 --owm_lr 5e-5 --bn_lr 5e-4  --model_weight_decay 5e-5 --agent_type owm --agent_name owm_based --dataset CIFAR100 --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --force_out_dim 0  --first_split_size 10 --other_split_size 10  --batch_size 32 --model_name resnet18 --model_type resnet | tee ${OUTDIR}/owm_epoch_80_bn32_lr1e-4_headlr1e-3_bnlr5e-4_owmlr5e-5_wdecay5e-5_regcoef_500.log

python -u main.py --schedule 1 --reg_coef 500 --model_lr 1e-4 --head_lr 1e-3 --owm_lr 1e-4 --bn_lr 5e-4  --model_weight_decay 5e-5 --agent_type owm --agent_name owm_based --dataset CIFAR100 --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --force_out_dim 0  --first_split_size 10 --other_split_size 10  --batch_size 32 --model_name resnet18 --model_type resnet  | tee ${OUTDIR}/owm_epoch_80_bn32_lr1e-4_headlr1e-3_bnlr5e-4_owmlr1e-4_wdecay5e-5_regcoef_500_alpha1.log