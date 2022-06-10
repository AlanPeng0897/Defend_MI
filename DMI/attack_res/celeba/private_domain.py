import os, sys

sys.path.append('../../../BiDO/')
from utils import load_json, save_tensor_images, init_dataloader


if __name__ == '__main__':
    file = "../../../BiDO/config/celeba.json"
    args = load_json(json_file=file)
    file_path = '../../../attack_dataset/CelebA/trainset.txt'
    save_img_dir = "./trainset"
    os.makedirs(save_img_dir, exist_ok=True)

    args["dataset"]["img_path"] = '../../../attack_dataset/CelebA/Img'
    model_name = args['dataset']['model_name']
    args[model_name]['batch_size'] = 1
    private_loader = init_dataloader(args, file_path, mode="attack")

    last_iden = -1
    cnt = 0
    for i, (imgs, iden) in enumerate(private_loader):
        if iden >= 300:
            break

        iden = int(iden)
        if iden != last_iden:
            last_iden = iden
            cnt = 0
        else:
            cnt += 1

        if cnt >= 5:
            continue

        print("save {} image of iden {}".format(cnt, iden + 1))
        save_tensor_images(imgs, os.path.join(save_img_dir, "iden_{:03d}_{}th.png".format(iden + 1, cnt)))
