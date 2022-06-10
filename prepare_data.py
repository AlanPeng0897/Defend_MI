import os,sys, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append('BiDO')
import utils

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(mnist_path, train=True, transform=transform, download=True)
    testset = torchvision.datasets.MNIST(mnist_path, train=False, transform=transform, download=True)

    train_loader = DataLoader(trainset, batch_size=1)
    test_loader = DataLoader(testset, batch_size=1)
    cnt = 0

    for imgs, labels in train_loader:
        cnt += 1
        img_name = str(cnt) + '_' + str(labels.item()) + '.png'
        utils.save_tensor_images(imgs, os.path.join(mnist_img_path, img_name))
    print("number of train files:", cnt)

    for imgs, labels in test_loader:
        cnt += 1
        img_name = str(cnt) + '_' + str(labels.item()) + '.png'
        utils.save_tensor_images(imgs, os.path.join(mnist_img_path, img_name))


def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(cifar_path, train=True, transform=transform, download=True)
    testset = torchvision.datasets.CIFAR10(cifar_path, train=False, transform=transform, download=True)

    train_loader = DataLoader(trainset, batch_size=1)
    test_loader = DataLoader(testset, batch_size=1)
    cnt = 0

    for imgs, labels in train_loader:
        cnt += 1
        img_name = str(cnt) + '_' + str(labels.item()) + '.png'
        utils.save_tensor_images(imgs, os.path.join(cifar_img_path, img_name))
    cnt_train = cnt
    print("number of train files:", cnt_train)

    for imgs, labels in test_loader:
        cnt += 1
        img_name = str(cnt) + '_' + str(labels.item()) + '.png'
        utils.save_tensor_images(imgs, os.path.join(cifar_img_path, img_name))

    print("number of test files:", cnt - cnt_train)


if __name__ == "__main__":
    mnist_path = "./attack_dataset/mnist_tmp"
    mnist_img_path = "./attack_dataset/MNIST/Img"
    os.makedirs(mnist_path, exist_ok=True)
    os.makedirs(mnist_img_path, exist_ok=True)
    cifar_path = "./attack_dataset/cifar_tmp"
    cifar_img_path = "./attack_dataset/CIFAR/Img"
    os.makedirs(cifar_path, exist_ok=True)
    os.makedirs(cifar_img_path, exist_ok=True)

    load_cifar10()
    load_mnist()
    print("ok")
