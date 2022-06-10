for styles in 0,1,2 3,4,5,6 7,8,9; do

python style_mixing.py \
	--styles=${styles} \
	--outdir=out-celeba/try5/${styles} \
	--rows=55,59 \
	--cols=1789,21  \
	--network=/scratch/hdd001/home/wangkuan/stylegan/celeba-aux/Mar22-auto/00001-celeba-aux-auto2-resumefromprev/network-snapshot-002298.pkl

done

	# --rows=85,75,1500 \
	# --cols=55,1789,293  \