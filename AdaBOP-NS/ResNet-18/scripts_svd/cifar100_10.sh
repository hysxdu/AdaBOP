GPUID=3
REPEAT=1
############################################
OUTDIR=paper_results/cifar100-10
mkdir -p $OUTDIR
python -u main.py --schedule 30 60 80 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --force_out_dim 0  --first_split_size 10 --other_split_size 10  --batch_size 32 --model_name resnet18 --model_type resnet | tee ${OUTDIR}/svd_epoch_80_bn32_lr1e-4_headlr1e-3_bnlr5e-4_svdlr5e-5_wdecay5e-5_regcoef_100_eigvec_gt_ada10_combine0.log