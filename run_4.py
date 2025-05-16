import argparse
import math
import os

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from normalization import *
from onehotencoding import *
from pathlib import Path
from modules import *
import torchvision.transforms as trans
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import warnings
from preprocessing import preprocessing_before_train
from fairness.my_useful_functions import statistical_parity_loss
warnings.filterwarnings('ignore')
from fairness.my_useful_functions import calculate_performance_equalized_odds

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
file_debugging = open('running/test.txt', 'w')

def run(
        source='olympics.csv',
        output='olympics_generation.csv',
        continuous_columns=['Age', 'Height', 'Weight'],
        categorical_columns=['Sex', 'Year', 'Season', 'City', 'Sport', 'Medal', 'AOS', 'AOE'],
        device="cpu",
        batch_size=32,
        epochs=50,
        lr=0.0001,
        ns_G=0.8,
        ns_D=0.1,
        amount=1,
        protected_attribute='sex',
        unpriv_value=0,
        class_label='class-label',
        fair_loss_scale=0.1

):
    # Load data
    source_path = ROOT / 'data' / 'Origins' / source
    out_path = ROOT / 'data' / 'Generations' / output

    df = pd.read_csv(source_path)
    # thay do
    df_train = df.copy()

    preprocessing_before_train(df_train, source)

    le = preprocessing.LabelEncoder()
    for i in df_train.columns:
        if df_train[i].dtypes == 'object':
            df_train[i] = le.fit_transform(df_train[i])

    length = len(df_train.columns)
    X_train = df_train.iloc[:, :length - 1]
    y_train = df_train[class_label]

    mlp = MLPClassifier()
    knn = KNeighborsClassifier(n_neighbors=5)
    dt = DecisionTreeClassifier()
    svm = SVC(probability=True)
    logisticregression = LogisticRegression()

    mlp.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    logisticregression.fit(X_train, y_train)

    # thay do

    source_columns = df.columns
    df = df[continuous_columns+categorical_columns]

    # Find continuous data and normalization
    dict_min_max_value = {}
    df_conti = df[continuous_columns].astype('int64')

    for column in continuous_columns:
        min_val = df_conti[column].min()
        max_val = df_conti[column].max()
        dict_min_max_value.update({column: [min_val, max_val]})
    print(dict_min_max_value)
    norm_list = continuous_columns
    norm_types = ['standard' for i in range(len(norm_list))]
    df_conti_norm, dict_conti = norm(df_conti, norm_list, norm_types)
    print(dict_conti)
    # Find categorical data and one hot encoding
    df_category = df[categorical_columns].astype('category')
    cate_name, cate_class_number, cate_class, df_category_ohe = one_hot_encoding(df_category)

    # Combine data
    df_combine = pd.concat([df_conti_norm, df_category_ohe], axis=1)

    # Reshape data
    df_length = len(df_combine.columns)
    input_data = df_combine.to_numpy().flatten().reshape(-1, 1, 1, df_length)

    # ToTensor transform
    transforms = trans.Compose(
        [trans.ToTensor()]
    )

    # Parameters
    z_dim = 64
    image_dim = 1 * df_length * 1
    num_epochs = epochs
    if num_epochs == 200:
        a = 1e-2 / 100
        r = 1.02872
    elif num_epochs == 100:
        a = 1e-2 / 100
        r = 1.0673
    else:
        a = 1e-2 / 100
        r = 1.15884

    # Model initialization
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)

    # input_data.tofile("running/test.txt", sep=" ", format='%s')
    dataset = OlympicDataset(input_data, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    disc = Discriminator(image_dim, ns_D).to(device)
    gen = Generator(z_dim, image_dim, ns_G).to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    df_real_continuous = df[df_conti.columns.to_numpy()].astype('int64')
    df_real_categorical = df[df_category.columns.to_numpy()].astype('category')
    df_real = pd.concat([df_real_continuous, df_real_categorical], axis=1)

    # Train and generate
    step = 0
    for epoch in range(num_epochs):
        for batch_idx, real in enumerate(loader):
            real = real.view(-1, 1 * df_length).to(device)
            batch_size = real.shape[0]

            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))

            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # where the second option of maximizing doesn't suffer from
            # saturating gradients
            output = disc(fake).view(-1)
            # change object function
            fake_df = pd.DataFrame(fake.cpu().flatten().reshape(-1, df_length).detach().numpy())
            fake_df = fake_df.rename(columns={i: df_combine.columns[i] for i in range(df_combine.columns.shape[0])})
            df_fake_categorical = one_hot_decoding(fake_df.iloc[:, len(continuous_columns):])[categorical_columns].astype(
                'category')
            df_fake_continuous = denorm(fake_df[continuous_columns], norm_list, norm_types, dict_conti).apply(
                np.ceil).astype(
                'int64')

            df_fake = pd.concat([df_fake_continuous, df_fake_categorical], axis=1)
            df_fake = df_fake[source_columns]

            preprocessing_before_train(df_fake, source)

            le = preprocessing.LabelEncoder()
            for i in df_fake.columns:
                if df_fake[i].dtypes == 'category':
                    df_fake[i] = le.fit_transform(df_fake[i])

            length = len(df_fake.columns)
            X_test = df_fake.iloc[:, :length - 1]
            y_test = df_fake[class_label]

            loss = 0.0

            feature = X_test.keys().tolist()
            sa_index = feature.index(protected_attribute)
            saValue = unpriv_value

            y_predict = mlp.predict(X_test)
            tmp = calculate_performance_equalized_odds(X_test.values, y_test.values, y_predict, sa_index, saValue)[
                'fairness'].__round__(4)
            if not math.isnan(tmp) and not math.isinf(tmp):
                loss += tmp
            else:
                loss += 1.0

            y_predict = knn.predict(X_test)
            tmp = calculate_performance_equalized_odds(X_test.values, y_test.values, y_predict, sa_index, saValue)[
                'fairness'].__round__(4)
            if not math.isnan(tmp) and not math.isinf(tmp):
                loss += tmp
            else:
                loss += 1.0

            y_predict = dt.predict(X_test)
            tmp = calculate_performance_equalized_odds(X_test.values, y_test.values, y_predict, sa_index, saValue)[
                'fairness'].__round__(4)
            if not math.isnan(tmp) and not math.isinf(tmp):
                loss += tmp
            else:
                loss += 1.0

            y_predict = svm.predict(X_test)
            tmp = calculate_performance_equalized_odds(X_test.values, y_test.values, y_predict, sa_index, saValue)[
                'fairness'].__round__(4)
            if not math.isnan(tmp) and not math.isinf(tmp):
                loss += tmp
            else:
                loss += 1.0

            y_predict = logisticregression.predict(X_test)
            tmp = calculate_performance_equalized_odds(X_test.values, y_test.values, y_predict, sa_index, saValue)[
                'fairness'].__round__(4)
            if not math.isnan(tmp) and not math.isinf(tmp):
                loss += tmp
            else:
                loss += 1.0

            # change object function
            lossG = criterion(output, torch.ones_like(output)) + abs(fair_loss_scale*loss/5)
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 1, df_length)
                    step += 1

        if epoch == 0:
            fake_df = pd.DataFrame(fake.cpu().flatten().reshape(-1, df_length).detach().numpy())
            fake_df = fake_df.rename(columns={i: df_combine.columns[i] for i in range(df_combine.columns.shape[0])})
            final = fake_df.astype('category')
            loop_num = int(len(df_real) * a // len(fake_df))
            for i in range(1 if loop_num == 0 else loop_num):
                noise = torch.randn(batch_size, z_dim).to(device)
                fake = gen(noise)
                fake_df = pd.DataFrame(fake.cpu().flatten().reshape(-1, df_length).detach().numpy())
                fake_df = fake_df.rename(columns={i: df_combine.columns[i] for i in range(df_combine.columns.shape[0])})
                demo = fake_df

                final = pd.concat([final, demo]).reset_index(drop=True)

        else:
            loop_num = int(len(df_real) * a // len(fake_df) * amount)
            for i in range(loop_num + 1):
                noise = torch.randn(batch_size, z_dim).to(device)
                fake = gen(noise)
                fake_df = pd.DataFrame(fake.cpu().flatten().reshape(-1, df_length).detach().numpy())
                fake_df = fake_df.rename(columns={i: df_combine.columns[i] for i in range(df_combine.columns.shape[0])})
                demo = fake_df

                final = pd.concat([final, demo]).reset_index(drop=True)

        a = a * r

    # Save generated results
    df_fake_categorical = one_hot_decoding(final.iloc[:, len(continuous_columns):])[categorical_columns].astype(
        'category')
    df_fake_continuous = denorm(final[continuous_columns], norm_list, norm_types, dict_conti).apply(np.ceil).astype(
        'int64')

    df_fake = pd.concat([df_fake_continuous, df_fake_categorical], axis=1)
    df_fake = df_fake[source_columns]
    row_to_drop = []
    for row in df_fake.index:
        for column in continuous_columns:
            if df_fake.iloc[row][column] < dict_min_max_value[column][0] or df_fake.iloc[row][column] > dict_min_max_value[column][1]:
                row_to_drop.append(row)
    df_fake = df_fake.drop(index=row_to_drop)
    df_fake.to_csv(out_path, index=False)
    print('Synthetic data has been saved to ', out_path)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='olympics.csv', help='source of data file')
    parser.add_argument('--output', type=str, default='olympics_generation.csv',
                        help='source of data file')
    parser.add_argument('--continuous_columns', nargs="*", type=str, default=['Age', 'Height', 'Weight'],
                        help='list of continuous columns')
    parser.add_argument('--categorical_columns', nargs="*", type=str,
                        default=['Sex', 'Year', 'Season', 'City', 'Sport', 'Medal', 'AOS', 'AOE'],
                        help='list of categorical columns')
    parser.add_argument('--device', type=str, default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--ns_G', type=float, default=0.8, help='leakyRelu negative slope of generator')
    parser.add_argument('--ns_D', type=float, default=0.1, help='leakyRelu negative slope of discriminator')
    parser.add_argument('--amount', type=float, default=1, help='percentage of generated data size over real data size')
    parser.add_argument('--protected_attribute', type=str, default='sex', help='protected attibute')
    parser.add_argument('--unpriv_value', type=int, default=0, help='unprivileged value')
    parser.add_argument('--class_label', type=str, default='class-label', help='class label')
    parser.add_argument('--fair_loss_scale', type=float, default=0.1, help='fair_loss_scale')

    opt = parser.parse_args()

    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
