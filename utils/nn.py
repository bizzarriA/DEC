import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act

    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act

    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, out_dim=12):
        super().__init__()
        self.encoder = nn.Sequential(Encoder(input_dim, 500, True),
                                     Encoder(500, 500, True),
                                     Encoder(500, 2000, True),
                                     Encoder(2000, out_dim, False))
        self.decoder = nn.Sequential(Decoder(out_dim, 2000, True),
                                     Decoder(2000, 500, True),
                                     Decoder(500, 500, True),
                                     Decoder(500, input_dim, False))

    def forward(self, x):
        x = self.encoder(x)
        gen = self.decoder(x)
        return x, gen


# Creating a PyTorch class
class AEConv(torch.nn.Module):
    def __init__(self, channel=3, input_dim=32):
        super(AEConv, self).__init__()
        n_layer = 3
        stride = 2
        padding = 1
        k_size = 4
        dim = input_dim
        for _ in range(n_layer):
            dim = int(((dim - k_size + 2*padding)/stride) + 1)
        print(dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 12, k_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(12, 24, k_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(24, 48, k_size, stride=stride, padding=padding),
            nn.ReLU(),
            # nn.Conv2d(48, 96, 4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(96, 96, 4, stride=2, padding=1),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48*dim*dim, 48*dim),
            nn.Linear(48*dim, 48),
            nn.Linear(48, 16)
            # nn.ReLU(),
            # 10
            # nn.Softmax(),
        )
        self.decoder = nn.Sequential(
            # nn.Linear(10, 96*14*14),
            # nn.ReLU(),
            nn.Linear(16, 48*dim*dim),
            nn.Unflatten(1, (48, dim, dim)),
            # nn.ConvTranspose2d(96, 96, 4, stride=2, padding=1),
            # nn.ReLU()
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(48, 24, k_size, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, k_size, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, k_size, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Cluster(nn.Module):
    def __init__(self, center, alpha):
        super().__init__()
        self.center = center
        self.alpha = alpha

    def forward(self, x):
        square_dist = torch.pow(x[:, None, :] - self.center, 2).sum(dim=2)
        nom = torch.pow(1 + square_dist / self.alpha, -(self.alpha + 1) / 2)
        denom = nom.sum(dim=1, keepdim=True)
        return nom / denom


def get_p(q):
    with torch.no_grad():
        f = q.sum(dim=0, keepdim=True)
        nom = q ** 2 / f
        denom = nom.sum(dim=1, keepdim=True)
    return nom / denom


class DEC(nn.Module):
    def __init__(self, encoder, center, alpha=1):
        super().__init__()
        self.encoder = encoder
        self.cluster = Cluster(center, alpha)
        self.softmax = nn.Softmax()

    def forward(self, x):
        z = self.encoder(x)
        x = self.cluster(z)
        return x
