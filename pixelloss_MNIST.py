import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from tqdm import tqdm
from torchplus.utils import Init
from torchplus.nn import PixelLoss
from piq import SSIMLoss

if __name__ == "__main__":
    batch_size = 128
    train_epoches = 10
    log_interval = 100
    test_interval = log_interval
    class_num = 10
    root_dir = "./logZZPMAIN.measure"
    test_pkl = "/path/to/target_classifier.pkl"
    inv_pkl = "/path/to/myinversion_48.pkl"
    h = 32
    w = 32
    pownum = 1 / 11

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        backup_filename=__file__,
        tensorboard=True,
        comment="MNIST ATTACK measure 48",
    )
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir = init.get_log_dir()
    data_workers = 2

    transform = Compose([Grayscale(num_output_channels=1), Resize((h, w)), ToTensor()])

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

    # for evaluate
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )
    # for attack
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )

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

    class Inversion(nn.Module):
        def __init__(self, in_channels):
            super(Inversion, self).__init__()
            self.in_channels = in_channels
            self.deconv1 = nn.ConvTranspose2d(self.in_channels, 512, 4, 1)
            self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
            self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
            self.deconv4 = nn.ConvTranspose2d(128, 1, 4, 2, 1)
            self.bn1 = nn.BatchNorm2d(512)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(128)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.sigmod = nn.Sigmoid()

        def forward(self, x):
            x = x.view(-1, self.in_channels, 1, 1)
            x = self.deconv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.deconv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.deconv4(x)
            x = self.sigmod(x)
            return x

    class PowerAmplification(nn.Module):
        def __init__(
            self, in_features: int, alpha: float = None, device=None, dtype=None
        ) -> None:
            super(PowerAmplification, self).__init__()
            factory_kwargs = {"device": device, "dtype": dtype}
            self.in_features = in_features
            if alpha is not None:
                self.alpha = Parameter(torch.tensor([alpha], **factory_kwargs))
            else:
                self.alpha = Parameter(torch.rand(1, **factory_kwargs))

        def forward(self, input: Tensor) -> Tensor:
            alpha = self.alpha.expand(self.in_features)
            return torch.pow(input, alpha)

    myclassifier = Classifier(class_num).train(False).to(output_device)
    myinversion = Inversion(class_num).train(False).to(output_device)
    myamplification = (
        PowerAmplification(class_num, 1 / 16).train(False).to(output_device)
    )

    assert os.path.exists(test_pkl)
    myclassifier.load_state_dict(
        torch.load(open(test_pkl, "rb"), map_location=output_device)
    )

    assert os.path.exists(inv_pkl)
    myinversion.load_state_dict(
        torch.load(open(inv_pkl, "rb"), map_location=output_device)
    )

    with torch.no_grad():
        r = 0
        plall = 0
        mseall = 0
        ssimloss = 0
        for i, (im, label) in enumerate(tqdm(train_dl, desc=f"priv")):
            r += 1
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            out = myclassifier.forward(im)
            after_softmax = F.softmax(out, dim=-1)
            after_softmax = myamplification.forward(after_softmax)
            rim = myinversion.forward(after_softmax)
            mse = F.mse_loss(rim, im)
            pl = PixelLoss(13)(rim, im)
            ssim = SSIMLoss()(rim, im)
            mseall += mse
            plall += pl
            ssimloss += ssim

        plavg = plall / r
        mseavg = mseall / r
        ssimavg = ssimloss / r
        writer.add_text("priv Pixel Loss", f"{plavg}")
        writer.add_text("priv MSE", f"{mseavg}")
        writer.add_text("priv SSIM", f"{ssimavg}")

    with torch.no_grad():
        r = 0
        plall = 0
        mseall = 0
        ssimloss = 0
        for i, (im, label) in enumerate(tqdm(test_dl, desc=f"aux")):
            r += 1
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            out = myclassifier.forward(im)
            after_softmax = F.softmax(out, dim=-1)
            after_softmax = myamplification.forward(after_softmax)
            rim = myinversion.forward(after_softmax)
            mse = F.mse_loss(rim, im)
            pl = PixelLoss(13)(rim, im)
            ssim = SSIMLoss()(rim, im)
            mseall += mse
            plall += pl
            ssimloss += ssim

        plavg = plall / r
        mseavg = mseall / r
        ssimavg = ssimloss / r
        writer.add_text("aux Pixel Loss", f"{plavg}")
        writer.add_text("aux MSE", f"{mseavg}")
        writer.add_text("aux SSIM", f"{ssimavg}")
    writer.close()
