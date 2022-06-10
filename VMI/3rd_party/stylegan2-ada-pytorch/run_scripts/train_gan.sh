
#source activate gan

for cfg in stylegan2; do

output_dir=/home/ubuntu/peng/code/MID/attack_dataset/chestxray/train_gan/

cmd="train.py \
	--outdir ${output_dir} \
	--cfg ${cfg} \
	--data /home/ubuntu/peng/code/MID/attack_dataset/chestxray/gan_data \
	--gpus 1 \
	--snap 10 \
	--metrics fid50k_full \
	--resume_from_prev 0
"

#if [ $1 == 0 ]
#then
python $cmd
#break 100
#else
#sbatch <<< \
#"#!/bin/bash
##SBATCH --mem=64G
##SBATCH -c 4
##SBATCH --gres=gpu:2
##SBATCH -p rtx6000
##SBATCH --time=200:00:00
##SBATCH --output=logs/train_gan-%j-out.txt
##SBATCH --error=logs/train_gan-%j-err.txt
##SBATCH --qos=normal
#
##necessary env
#source activate gan

#echo $cmd
#python $cmd
#"
#fi

done
#done
#done
#done
#done
#done
#done
#done


