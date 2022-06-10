#source activate ebm

# Almost unused
smoothness_lambda=0
smoothness_nsize=0
smoothness_extract_feat=none
rs=0

# Constants
use_labels=0
n_conditions=1
method=dcgan
lrD2lrG=1
lrD=2e-4
sn=1
norm='bn'

db=0
# gen=secret
# dc=disc_secret.yaml
gen=basic
dc=disc_base.yaml


# # for d in celeba-target celeba-aux; do
# for d in celeba_crop-target celeba_crop-aux; do
# for target_dataset in "" ; do
# for nz in 50 20; do
# for dn in 0.1 0.2; do
# for g_scale in 1; do
# for lrD in 2e-4; do
# for aug in "" color; do
# for nf in 128; do

# for d in celeba-target celeba-aux; do
for d in -1; do
for target_dataset in "" ; do
for nz in 50; do
for dn in 0.1; do
for g_scale in 1; do
for lrD in 2e-4; do
for aug in ""; do
for nf in 10; do

cls_path="''"
case "$target_dataset" in
cifar10)
    cls_path=./mm/classifiers/cifar10/50/ckpt.pt
;;
celeba-target)
	cls_path=./mm/classifiers/celeba/100-1e-2-64/ckpt.pt
;;
celeba-db)
	cls_path=./mm/classifiers/celeba/100-1e-2-64/ckpt.pt
;;
esac


case "$dc" in
disc_base.yaml)
    dcn=base
;;
disc_secret.yaml)
    dcn=secret
;;
disc_resnet.yaml)
    dcn=resnet
;;
esac


case $method in
l2_aux)
	model=l2_aux
	use_sigmoid=0
	output_type=standard
;;
wgan_aux)
	model=wgan_aux
	use_sigmoid=0
	output_type=adv_l2c
;;
dcgan)
	model=dcgan
	use_sigmoid=1
	output_type=standard
;;
dcgan_aux)
	model=dcgan_aux
	use_sigmoid=1
	output_type=standard
;;
dcgan_aux_c)
	model=dcgan_aux
	use_sigmoid=1
	output_type=standardc
esac

#output_dir=./mm/run-mm-augment/${d}-${target_dataset}/Dec10/rs${rs}-${method}-G_${gen}-nf${nf}-${dcn}-${nz}-${lrD}-${lrD2lrG}-${sn}-${dn}-${g_scale}-${aug}-${smoothness_lambda}-${smoothness_nsize}-${norm}
#output_dir=./DCGAN/celeba/h10&500
#exp_config=mnist/hsic_10&500.yml

cmd="main.py \
--exp_config $1 \
--db ${db} \
--ngf ${nf} \
--g_norm ${norm} \
--eval_only 0 \
--ckpt_path=$2/generator.pt \
--resume $3 \
--g_sn $sn \
--d_noise ${dn} \
--model ${model} \
--dataroot ./data/ \
--dataset $d \

--output_dir $2 \
--use_labels ${use_labels} \
--gen ${gen} \
--n_conditions $n_conditions \
--log_iter_every 100  \
--disc_config ${dc} \
--lrD ${lrD} \
--lrD2lrG ${lrD2lrG} \
--nz ${nz} \
--g_z_scale ${g_scale} \
--augment ${aug} \
--g_conditioning_method add \
--disc_kwargs is_conditional:i.${use_labels},n_conditions:i.${n_conditions},sn:i.${sn},embed_condition:i.1,output_type:s.${output_type},use_sigmoid:i.${use_sigmoid},ndf:i.${nf} \
"

if [ $3 == 1 ]
then
echo $cmd
python $cmd
else
sbatch <<< \
"#!/bin/bash
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p t4v2,t4v1,p100,rtx6000
#SBATCH --time=200:00:00
#SBATCH --output=/h/wangkuan/slurm/%j-out.txt
#SBATCH --error=/h/wangkuan/slurm/%j-err.txt
#SBATCH --qos=normal

#necessary env
#source activate ebm

echo $cmd
python $cmd
"
fi

# echo python -m pytorch_fid ${output_dir}/samples ./mm/data/cifar10
done
done
done
done
done
done
done
done