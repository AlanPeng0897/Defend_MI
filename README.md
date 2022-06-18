# Bilateral Dependency Optimization: Defending Against Model-inversion Attacks
Hi, this is the code for our work "Bilateral Dependency Optimization: Defending Against Model-inversion Attacks" 
presented in KDD2022.

If you find this code useful in your research then please cite

```

@inproceedings{peng2022bilateral,
title={Bilateral Dependency Optimization: Defending Against Model-inversion Attacks},
author={Peng, Xiong and Liu, Feng and Zhang, Jingfeng and Lan, Long and Ye, Junjie and Liu, Tongliang and Han, Bo},
booktitle={KDD},
year={2022}
}

```

# Requiements
This code has been tested on Ubuntu 16.04/18.04, with Python 3.7, Pytorch 1.7 and CUDA 10.2/11.0

# Getting started
Download relevent datasets: CelebA, MNIST.
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- MNIST
    ```
    python prepare_dataset.py
    ```

The directory of dataset are orgnized as follows (take CelebA as an example):
    
    attack_dataset  
    -- CelebA  
    ---- *.txt  
    ---- Img  
    ------ 000001.png
    

# Train with BiDO 
You can also skip to the next section for defending against MI attack with well-trained defense models.
- For GMI and DMI
    ```
    #dataset:celeba, mnist, cifar; 
    #measure:COCO, HSIC; 
    #balancing hyper-parameters: tune them in train_HSIC.py
    python train_HSIC.py --measure=HSIC --dataset=celeba
    ```
    for DMI, if you trained a defense model yourself, you have to train the attacker specific to it additionally.
    ```
    # put you hyper-parameters in k+1_gan_HSIC.py first
    python k+1_gan_HSIC --dataset=celeba --defense=HSIC
    ```
- For VMI  
    please refer to this section (Defending against MI attacks - VMI - train with BiDO).
    


# Defending against MI attacks 
Here we only provide the weights file of the well-trained defense models achieve the best trade-off between model robustness and utility, which are highlighted in the experimental results.
- GMI
    - Weights file (defense model / eval model / GAN) :
        - place [pretrained VGG16](https://1drv.ms/u/s!An_XOOYcXU0GggMxd_xImjJ1m1fk?e=VD8Dsp) in `BiDO/target_model/`
        - place [defense model](https://1drv.ms/u/s!An_XOOYcXU0Gggb4NdzXqxrsa7vL?e=gOhPou) in `BiDO/target_model/celeba/HSIC/`
        - place [evaluation classifer](https://1drv.ms/u/s!An_XOOYcXU0GgXwM2Nc_QrJqFLeM?e=0C88Ih) in `GMI/eval_model/`
        - place [GAN](https://1drv.ms/u/s!An_XOOYcXU0GgWnu2qmbl3BZGHyT?e=6rz14z) in `GMI/result/models_celeba_gan/`

    - Launch attack
        ```
        #balancing hyper-parameters: (0.05, 0.5)
        python attack.py --dataset=celeba --defense=HSIC
        ```
    - Calculate FID
        ```
        # sample real images from training set
        cd attack_res/celeba/pytorch-fid && python private_domain.py 
        # calculate FID between fake and real images
        python fid_score.py ../celeba/trainset/ ../celeba/HSIC/all/
        ```
        
- DMI
    - Weights file (defense model / eval model / GAN) :
        - place [defense model](https://1drv.ms/u/s!An_XOOYcXU0GggTyiELgboDjOa0y?e=OufV3X) in `BiDO/target_model/mnist/COCO/`
        - place [evaluation classifer](https://1drv.ms/u/s!An_XOOYcXU0GgXqBElsXK0DQCKAD?e=07oQq4) in `DMI/eval_model/`
        - place [improved GAN for celeba](https://1drv.ms/u/s!An_XOOYcXU0GgW4HgzYQCTBu7Coq?e=di6QmO) in `DMI/improvedGAN/celeba/HSIC/`
        - place [improved GAN for mnist](https://1drv.ms/u/s!An_XOOYcXU0GgghNCBXxSHRX--Rq?e=CJeK1X) in `DMI/improvedGAN/mnist/COCO/`
    - Launch attack
        ```
        #balancing hyper-parameters: (0.05, 0.5)
        python recovery.py --dataset=celeba --defense=HSIC
        #balancing hyper-parameters: (1, 50)
        python recovery.py --dataset=mnist --defense=COCO
        ```
    - Calculate FID
        ```
        # celeba
        cd attack_res/celeba/pytorch-fid && python private_domain.py 
        python fid_score.py ../celeba/trainset/ ../celeba/HSIC/all/ --dataset=celeba
        # mnist
        cd attack_res/mnist/pytorch-fid && python private_domain.py 
        python fid_score.py ../mnist/trainset/ ../mnist/COCO/all/ --dataset=mnist
        ```

- VMI  
To run this code, you need ~38G of memory for data loading, the attacking of 20 identities takes ~20 hours on a TiTAN-V GPU (12G).
    - data (CelebA)
        ```
        #create a link to CelebA
        cd VMI/data && ln -s ../../attack_data/CelebA/Img img_align_celeba
        python celeba.py
        ```
    - Weights file (defense model / eval model / GAN) :
        - place [defense model](https://1drv.ms/u/s!An_XOOYcXU0GgX_ffiscTaShdhQT?e=3FbD2r) in `VMI/clf_results/celeba/hsic_0.1&2/`
        - place [ir_se50.pth](https://1drv.ms/u/s!An_XOOYcXU0GggcLEgg4_yq0_y5l?e=lZyneT) in `VMI/3rd_party/InsightFace_Pytorch/work_space/save/`; place [evaluation classifer](https://1drv.ms/u/s!An_XOOYcXU0GggA9oEsLocMnR-M5?e=8fsxFD) in `VMI/pretrained/eval_clf/celeba/`
        - place [StyleGAN](https://1drv.ms/u/s!An_XOOYcXU0GggWTCCJV7CAhThpR?e=osrWUK) in `VMI/pretrained/stylegan/neurips2021-celeba-stylegan/`
    - Launch attack
        ```
        #balancing hyper-parameters: (0.1, 2)
        cd VMI
        # x.sh  (1st) path/to/attack_results (2nd) config_file  (3rd) batch_size
        ./run_scripts/neurips2021-celeba-stylegan-flow.sh  'hsic_0.1&2'  'hsic_0.1&2.yml' 32
        ```
    - Train with BiDO
        ```
        python classify_mnist.py --epochs=100 --dataset=celeba --output_dir=./clf_results/celeba/hsic_0.1&2 --model=ResNetClsH --measure=hsic --a1=0.1 --a2=2
        ```
































