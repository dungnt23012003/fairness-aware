import pandas as pd
import torch
from torch.utils.data import Dataset


class OlympicDataset(Dataset):

    def __init__(self, data: pd.DataFrame, transform=None):
        self.data = torch.from_numpy(data.astype(float)).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_content = self.data[idx]
        return data_content


class Generator(torch.nn.Module):
    def __init__(self, z_dim, img_dim, ns_G):
        super().__init__()
        self.gen = torch.nn.Sequential(
            torch.nn.Linear(z_dim, 256),
            torch.nn.LeakyReLU(ns_G),
            torch.nn.Linear(256, img_dim),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(torch.nn.Module):
    def __init__(self, in_features, ns_D):
        super().__init__()
        self.disc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.LeakyReLU(ns_D),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class FairLossFunc(torch.nn.Module):
    def __init__(self, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index):
        super(FairLossFunc, self).__init__()
        self._S_start_index = S_start_index
        self._Y_start_index = Y_start_index
        self._underpriv_index = underpriv_index
        self._priv_index = priv_index
        self._undesire_index = undesire_index
        self._desire_index = desire_index

    def forward(self, x, loss_scale):
        G = x[:, self._S_start_index:self._S_start_index + 2]
        I = x[:, self._Y_start_index:self._Y_start_index + 2]
        d_sp = loss_scale * abs(torch.mean(G[:, self._underpriv_index] * I[:, self._desire_index]) / (x[:, self._S_start_index + self._underpriv_index].sum()) - torch.mean(G[:, self._priv_index] * I[:, self._desire_index]) / (x[:, self._S_start_index + self._priv_index].sum()))

        return d_sp
