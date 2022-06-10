
export PATH=/pkgs/anaconda3/bin:$PATH
export ROOT1=/h/yanf
source activate /h/wangkuan/anaconda3/envs/gan 

for fixed_id in {0..9}; do

output_dir=/scratch/hdd001/home/yanf/stylegan/celeba-aux/attack-noise/April22-id${fixed_id}

cmd="attack.py \
	--outdir ${output_dir} \
	--network /scratch/hdd001/home/wangkuan/stylegan/celeba-aux/Mar22-auto/00001-celeba-aux-auto2-resumefromprev/network-snapshot-002298.pkl \
	--fixed_id ${fixed_id} \
"

if [ $1 == 0 ] 
then
python $cmd
break 100
else
sbatch <<< \
"#!/bin/bash
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --gres=gpu:2
#SBATCH -p rtx6000
#SBATCH --time=200:00:00
#SBATCH --output=logs/train_gan-%j-out.txt
#SBATCH --error=logs/train_gan-%j-err.txt
#SBATCH --qos=normal

#necessary env
source activate /h/wangkuan/anaconda3/envs/gan 

echo $cmd
python $cmd
"
fi

done
done
done
done
done
done




