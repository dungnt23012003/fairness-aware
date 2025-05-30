import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.functional import gumbel_softmax, leaky_relu


class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64)):
        super(ClassifierMLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[1], 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)

    def train_model(self, train_loader, num_epochs=50, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            self.train()  # switch to training mode
            total_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def evaluate_model(self, test_loader):
        self.eval()  # switch to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy


class OlympicDataset(Dataset):

    def __init__(self, data: pd.DataFrame, transform=None):
        self.data = torch.from_numpy(data.astype(float)).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_content = self.data[idx]
        return data_content


class Generator(torch.nn.Module):
    def __init__(self, z_dim, img_dim, ns_G, continuous_columns_dict, categorical_columns_dict):
        super().__init__()
        self._input_dim = z_dim
        self._output_dim = img_dim
        self._ns_G = ns_G
        self._discrete_columns = categorical_columns_dict
        self._num_continuous_columns = len(continuous_columns_dict)

        self.lin = nn.Linear(self._input_dim, self._output_dim)

        self.lin_numerical = nn.Linear(self._output_dim, self._num_continuous_columns)

        self.lin_cat = nn.ModuleDict()
        for key, value in self._discrete_columns.items():
            self.lin_cat[key] = nn.Linear(self._output_dim, value)

    def forward(self, x):
        x = leaky_relu(self.lin(x), negative_slope=self._ns_G)
        x_numerical = leaky_relu(self.lin_numerical(x), negative_slope=self._ns_G)

        x_cat = []
        for key in self.lin_cat:
            x_cat.append(gumbel_softmax(self.lin_cat[key](x), tau=0.2))
        x_final = torch.cat((x_numerical, *x_cat), 1)
        return x_final


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
        G = gumbel_softmax(G, tau=0.1, hard=False, dim=1)
        I = x[:, self._Y_start_index:self._Y_start_index + 2]
        I = gumbel_softmax(I, tau=0.1, hard=False, dim=1)
        d_sp = loss_scale * abs((G[:, self._underpriv_index] * I[:, self._desire_index]).sum() / G[:, self._underpriv_index].sum() - (G[:, self._priv_index] * I[:, self._desire_index]).sum() / G[:, self._priv_index].sum())
        return d_sp
