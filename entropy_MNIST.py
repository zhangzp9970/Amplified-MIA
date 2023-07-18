import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from tqdm import tqdm
from torchplus.utils import Init

if __name__ == "__main__":
    batch_size = 128
    train_epoches = 100
    class_num = 10
    root_dir = "./logZZPMAIN.entropy"
    target_pkl = "/path/to/target_classifier_mnist.pkl"
    h = 32
    w = 32

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        backup_filename=__file__,
        tensorboard=True,
        comment=f"MNIST entropy",
    )
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir = init.get_log_dir()
    data_workers = 2

    transform = Compose(
        [
            Grayscale(num_output_channels=1),
            Resize((h, w)),
            ToTensor(),
        ]
    )

    mnist_train_ds = MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    mnist_test_ds = MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    mnist_train_ds_len = len(mnist_train_ds)
    mnist_test_ds_len = len(mnist_test_ds)

    train_ds = mnist_train_ds
    test_ds = mnist_test_ds

    train_ds_len = len(train_ds)
    test_ds_len = len(test_ds)

    print(train_ds_len)
    print(test_ds_len)

    # for train
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )
    # for evaluate
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )

    train_dl_len = len(train_dl)
    test_dl_len = len(test_dl)

    class Classifier(nn.Module):
        def __init__(self, out_features):
            super(Classifier, self).__init__()
            self.out_features = out_features
            self.conv1 = nn.Conv2d(1, 128, 3, 1, 1)
            self.conv2 = nn.Conv2d(128, 256, 3, 1, 1)
            self.conv3 = nn.Conv2d(256, 512, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(512)
            self.mp1 = nn.MaxPool2d(2, 2)
            self.mp2 = nn.MaxPool2d(2, 2)
            self.mp3 = nn.MaxPool2d(2, 2)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.fc1 = nn.Linear(8192, 50)
            self.dropout = nn.Dropout()
            self.fc2 = nn.Linear(50, self.out_features)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.mp1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.mp2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.mp3(x)
            x = self.relu3(x)
            x = x.view(-1, 8192)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    target_classifier = Classifier(class_num).train(False).to(output_device)
    target_classifier.requires_grad_(False)

    assert os.path.exists(target_pkl)
    target_classifier.load_state_dict(
        torch.load(open(target_pkl, "rb"), map_location=output_device)
    )

    with torch.no_grad():
        for alpha in range(1, 21):
            after_softmax_list_by_label = [[] for i in range(class_num)]
            for i, (im, label) in enumerate(
                tqdm(train_dl, desc=f"testing train {alpha}")
            ):
                im = im.to(output_device)
                label = label.to(output_device)
                bs, c, h, w = im.shape
                out = target_classifier.forward(im)
                after_softmax = F.softmax(out, dim=-1)
                after_softmax = after_softmax.pow(1 / alpha)
                predict = torch.argmax(after_softmax, dim=-1)
                for b in range(bs):
                    after_softmax_list_by_label[label[b]].append(after_softmax[b])

            after_softmax_list_by_label = [
                torch.stack(after_softmax_list_by_label[i]).cpu()
                for i in range(class_num)
            ]
            after_softmax_list_number = [
                after_softmax_list_by_label[i].shape[0] for i in range(class_num)
            ]
            after_softmax_number = sum(after_softmax_list_number)
            after_softmax_list_number = [
                after_softmax_list_number[i] / after_softmax_number
                for i in range(class_num)
            ]
            H = 0
            for i in range(class_num):
                after_softmax_list = after_softmax_list_by_label[i]
                H_one_class = 0
                for j in range(class_num):
                    after_softmax_j = after_softmax_list[:, j]
                    hist, bin_edges = torch.histogram(
                        after_softmax_j, bins=100, density=True
                    )
                    H_col = 0
                    for bin in range(bin_edges.shape[0] - 1):
                        if hist[bin] != 0:
                            H_col += (
                                -1
                                * hist[bin]
                                * torch.log2(hist[bin])
                                * (bin_edges[bin + 1] - bin_edges[bin])
                            )
                    H_one_class += H_col
                H += H_one_class * after_softmax_list_number[i]

            writer.add_scalar("entropy", H, alpha)

        for alpha in range(2, 11):
            after_softmax_list_by_label = [[] for i in range(class_num)]
            for i, (im, label) in enumerate(
                tqdm(train_dl, desc=f"testing train {alpha*10}")
            ):
                im = im.to(output_device)
                label = label.to(output_device)
                bs, c, h, w = im.shape
                out = target_classifier.forward(im)
                after_softmax = F.softmax(out, dim=-1)
                after_softmax = after_softmax.pow(1 / (alpha * 10))
                predict = torch.argmax(after_softmax, dim=-1)
                for b in range(bs):
                    after_softmax_list_by_label[label[b]].append(after_softmax[b])

            after_softmax_list_by_label = [
                torch.stack(after_softmax_list_by_label[i]).cpu()
                for i in range(class_num)
            ]
            after_softmax_list_number = [
                after_softmax_list_by_label[i].shape[0] for i in range(class_num)
            ]
            after_softmax_number = sum(after_softmax_list_number)
            after_softmax_list_number = [
                after_softmax_list_number[i] / after_softmax_number
                for i in range(class_num)
            ]
            H = 0
            for i in range(class_num):
                after_softmax_list = after_softmax_list_by_label[i]
                H_one_class = 0
                for j in range(class_num):
                    after_softmax_j = after_softmax_list[:, j]
                    hist, bin_edges = torch.histogram(
                        after_softmax_j, bins=100, density=True
                    )
                    H_col = 0
                    for bin in range(bin_edges.shape[0] - 1):
                        if hist[bin] != 0:
                            H_col += (
                                -1
                                * hist[bin]
                                * torch.log2(hist[bin])
                                * (bin_edges[bin + 1] - bin_edges[bin])
                            )
                    H_one_class += H_col
                H += H_one_class * after_softmax_list_number[i]

            writer.add_scalar("entropy", H, alpha * 10)

        for alpha in range(1, 21):
            after_softmax_list_by_label = [[] for i in range(class_num)]
            for i, (im, label) in enumerate(
                tqdm(train_dl, desc=f"testing train {alpha*100}")
            ):
                im = im.to(output_device)
                label = label.to(output_device)
                bs, c, h, w = im.shape
                out = target_classifier.forward(im)
                after_softmax = F.softmax(out, dim=-1)
                after_softmax = after_softmax.pow(1 / (alpha * 100))
                predict = torch.argmax(after_softmax, dim=-1)
                for b in range(bs):
                    after_softmax_list_by_label[label[b]].append(after_softmax[b])

            after_softmax_list_by_label = [
                torch.stack(after_softmax_list_by_label[i]).cpu()
                for i in range(class_num)
            ]
            after_softmax_list_number = [
                after_softmax_list_by_label[i].shape[0] for i in range(class_num)
            ]
            after_softmax_number = sum(after_softmax_list_number)
            after_softmax_list_number = [
                after_softmax_list_number[i] / after_softmax_number
                for i in range(class_num)
            ]
            H = 0
            for i in range(class_num):
                after_softmax_list = after_softmax_list_by_label[i]
                H_one_class = 0
                for j in range(class_num):
                    after_softmax_j = after_softmax_list[:, j]
                    hist, bin_edges = torch.histogram(
                        after_softmax_j, bins=100, density=True
                    )
                    H_col = 0
                    for bin in range(bin_edges.shape[0] - 1):
                        if hist[bin] != 0:
                            H_col += (
                                -1
                                * hist[bin]
                                * torch.log2(hist[bin])
                                * (bin_edges[bin + 1] - bin_edges[bin])
                            )
                    H_one_class += H_col
                H += H_one_class * after_softmax_list_number[i]

            writer.add_scalar("entropy", H, alpha * 100)
    writer.close()
