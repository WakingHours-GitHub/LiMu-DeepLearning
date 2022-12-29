import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np


"""
AE也是基于encoder-decoder架构的.



"""

writer = SummaryWriter("AutoEncoder/logs")

BATCH_SIZE = 128
LR = 1e-2
NUM_EPOCH = 200

device = torch.device("cuda:0")


class encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 20), nn.ReLU(),
        )

    def forward(self, X):
        X = nn.Flatten()(X)
        X = self.backbone(X)
        return X


class decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),  # 如果是VAE这里要变成10. 也就是sample
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),  # map到(0, 1)
        )

    def forward(self, X):
        X: torch.Tensor = self.backbone(X)
        X = X.reshape(-1, 1, 28, 28)
        return X


class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, X):
        return self.decoder(self.encoder(X))


class VAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, X):
        h: torch.Tensor = self.encoder(X)
        # sample:
        mu, sigma = h.chunk(2, dim=1)  # 也就是在第一维, 分出两个块.
        h = mu + sigma * torch.randn_like(sigma)  # eps 这样才有反向传播.
        kld = 0.5 * torch.sum(torch.pow(mu, 2) + torch.pow(sigma, 2) -
                              torch.log(1e-8 + torch.pow(sigma, 2)) - 1) / (X.shape[0]*28*28)

        return self.decoder(h), kld


def main() -> None:
    mnist_train = DataLoader(torchvision.datasets.MNIST("./data", train=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]), download=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    mnist_test = DataLoader(torchvision.datasets.MNIST("./data", train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]), download=True),
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    # net = AutoEncoder().to(device)
    net = VAutoEncoder().to(device)

    loss_fn = nn.MSELoss().to(device)
    trainer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    for epoch in range(NUM_EPOCH):
        net.eval()
        for features, _ in mnist_train:
            features = features.to(device)
            y_hat, kld = net(features)

            loss = loss_fn(features, y_hat)
            if kld is not None:
                elbo = - loss - 1.0 * kld
                loss = - elbo

            trainer.zero_grad()
            loss.backward()
            trainer.step()
        print(loss.item(), kld.item())

        with torch.no_grad():
            net.eval()
            test, _ = net(iter(mnist_test).next()[0].to(device))
            writer.add_images(f"{epoch} AE generate", test)


if __name__ == "__main__":
    main()

writer.close()
