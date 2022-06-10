import os, sys

sys.path.append('../../../BiDO/')
from utils import load_json, save_tensor_images, init_dataloader

if __name__ == '__main__':
    file = "../../../BiDO/config/mnist.json"
    args = load_json(json_file=file)
    file_path = '../../../attack_dataset/MNIST/trainset.txt'
    save_img_dir = "./trainset"
    os.makedirs(save_img_dir, exist_ok=True)

    args["dataset"]["img_path"] = '../../../attack_dataset/MNIST/Img'
    model_name = args['dataset']['model_name']
    args[model_name]['batch_size'] = 1
    private_loader = init_dataloader(args, file_path, mode="attack")

    num_count = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }
    for i, (imgs, iden) in enumerate(private_loader):
        iden = int(iden)

        if num_count[iden] < 100:
            num_count[iden] += 1
            print("save {} image of iden {}".format(num_count[iden], iden + 1))
            save_tensor_images(imgs,
                               os.path.join(save_img_dir, "iden_{:03d}_{}th.png".format(iden + 1, num_count[iden])))
