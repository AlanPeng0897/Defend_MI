#exp_config=chestxray.yml
STYLEGAN_PKL=pretrained/stylegan/chestxray-stylegan/network-snapshot-004838.pkl
exp_id=Chestxray.1.data
lr=1e-4 
lambda_kl=1e-3
l=0-11
method=layeredflow
permute=shuffle
K=10
flow_coupling=additive
L=3
flow_use_actnorm=1
glow=1
lambda_miner_entropy=0
lambda_attack=1
run_target_feat_eval=1

for fixed_id in {0..7}; do
output_dir=attack_results/chestxray/$1/id${fixed_id}

cmd="attack_stylegan.py \
--flow_permutation ${permute} \
--flow_K ${K} \
--flow_glow ${glow} \
--flow_coupling ${flow_coupling} \
--flow_L ${L} \
--flow_use_actnorm ${flow_use_actnorm} \
--network ${STYLEGAN_PKL} \
--l_identity ${l} \
--db 0 \
--run_target_feat_eval ${run_target_feat_eval} \
--method ${method} \
--exp_config $2 \
--prior_model 0 \
--fixed_id ${fixed_id} \
--save_model_epochs 100,200,300 \
--resume 0 \
--patience 50 \
--output_dir ${output_dir} \
--log_iter_every 100  \
--viz_every 1 \
--epochs $3 \
--save_samples_every 1000 \
--lambda_weight_reg 0 \
--lambda_attack 1 \
--lambda_prior 0 \
--lambda_miner_entropy ${lambda_miner_entropy} \
--lambda_kl ${lambda_kl} \
--ewc_type None \
--lr ${lr} \
--batchSize 24 \
"

python $cmd
done
