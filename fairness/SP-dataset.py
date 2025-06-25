import math

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from compute_abroca import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from my_useful_functions import calculate_performance_statistical_parity_dataset, \
    calculate_performance_statistical_parity, calculate_performance_equalized_odds, \
    calculate_performance_equal_opportunity, calculate_performance_predictive_parity, \
    calculate_performance_predictive_equality, calculate_performance_treatment_equality
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
import sklearn.metrics as metrics
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
from preprocessing import preprocessing_before_train
import matplotlib

matplotlib.use('TkAgg')
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


if __name__ == '__main__':

    file_listss = [[['adult.csv', 'adult_generation.csv', 'adult_generation_5.csv', 'adult_generation_6_gender.csv', 'adult_generation_10_gender.csv', 'adult_generation_9_gender.csv', 'adult_generation_12_gender.csv'],
                    ['adult.csv', 'adult_generation.csv', 'adult_generation_5.csv', 'adult_generation_6_race.csv', 'adult_generation_10_race.csv', 'adult_generation_9_race.csv', 'adult_generation_12_race.csv']],
                   [['credit-card-clients.csv', 'credit-card-clients_generation.csv',
                     'credit-card-clients_generation_5.csv',
                     'credit-card-clients_generation_6_SEX.csv', 'credit-card-clients_generation_10_SEX.csv',
                     'credit-card-clients_generation_9_SEX.csv',
                     'credit-card-clients_generation_12_SEX.csv'],
                    ['credit-card-clients.csv', 'credit-card-clients_generation.csv',
                     'credit-card-clients_generation_5.csv',
                     'credit-card-clients_generation_6_AGE.csv', 'credit-card-clients_generation_10_AGE.csv',
                     'credit-card-clients_generation_9_AGE.csv',
                     'credit-card-clients_generation_12_AGE.csv']],
                   [['german-credit-data.csv', 'german-credit-data_generation.csv',
                     'german-credit-data_generation_5.csv',
                     'german-credit-data_generation_6_sex.csv', 'german-credit-data_generation_10_sex.csv',
                     'german-credit-data_generation_9_sex.csv',
                     'german-credit-data_generation_12_sex.csv']
                    ],
                   [['credit-scoring.csv', 'credit-scoring_generation.csv',
                     'credit-scoring_generation_5.csv', 'credit-scoring_generation_6_Sex.csv',
                     'credit-scoring_generation_10_Sex.csv', 'credit-scoring_generation_9_Sex.csv',
                     'credit-scoring_generation_12_Sex.csv'],
                    ['credit-scoring.csv', 'credit-scoring_generation.csv',
                     'credit-scoring_generation_5.csv', 'credit-scoring_generation_6_Age.csv',
                     'credit-scoring_generation_10_Age.csv', 'credit-scoring_generation_9_Age.csv',
                     'credit-scoring_generation_12_Age.csv']],
                   [['dutch-census.csv', 'dutch-census_generation.csv',
                     'dutch-census_generation_5.csv', 'dutch-census_generation_6_sex.csv',
                     'dutch-census_generation_10_sex.csv', 'dutch-census_generation_9_sex.csv',
                     'dutch-census_generation_12_sex.csv']],
                   [['PAKDD.csv', 'PAKDD_generation.csv', 'PAKDD_generation_5.csv',
                     'PAKDD_generation_6_SEX.csv', 'PAKDD_generation_10_SEX.csv', 'PAKDD_generation_9_SEX.csv', 'PAKDD_generation_12_SEX.csv']
                    ]
                 ]

    protected_attribute_lists = [['gender', 'race'],
                                 ['SEX', 'AGE'],
                                 ['sex'],
                                 ['Sex', 'Age'],
                                 ['sex'],
                                 ['SEX']]

    class_labels = ['class-label', 'class-label', 'class-label', 'class-label', 'class-label', 'class-label', 'class-label']
    p_Group_lists = [[0, 0],
                     [0, 0],
                     [0],
                     [0, 0],
                     [0],
                     [0]]
    file = open(ROOT / '..' / 'result' / 'SP.csv', 'w')
    result = []

    for protected_attribute_list, p_Group_list, file_lists, class_label in zip(protected_attribute_lists, p_Group_lists, file_listss, class_labels):
        print(protected_attribute_list)
        for file_list, protected_attribute, p_Group in zip(file_lists, protected_attribute_list, p_Group_list):
            result_attribute = []
            for f in file_list:
                if f.__contains__('generation'):
                    df = pd.read_csv(ROOT / '..' / 'data' / 'Generations' / f, sep=",")
                else:
                    df = pd.read_csv(ROOT / '..' / 'data' / 'Origins' / f, sep=",")

                le = preprocessing.LabelEncoder()
                for i in df.columns:
                    if df[i].dtypes == 'object':
                        df[i] = le.fit_transform(df[i])
                # Splitting data into train and test
                length = len(df.columns)
                X = df.iloc[:, :length - 1]
                y = df[class_label]
                feature = X.keys().tolist()
                sa_index = feature.index(protected_attribute)
                result_tmp = calculate_performance_statistical_parity_dataset(X.values, y.values, [], sa_index, p_Group)['fairness'].__round__(4)
                result_attribute.append(result_tmp)
            result.append(result_attribute)

    print(result)
    arr = np.array(result)
    arr_tmp = np.zeros(np.shape(arr))
    print(np.shape(arr))

    for att in range(np.shape(arr)[0]):
        min_position = -1
        for gen in range(np.shape(arr)[1]):
            if not math.isnan(arr[att][gen]):
                min_position = gen
                break
        if min_position != -1:
            for gen in range(np.shape(arr)[1]):
                if abs(arr[att][gen]) <= abs(arr[att][min_position]):
                    min_position = gen
            arr_tmp[att][min_position] = 1

    datasets = ['Adult\\_gender', 'Adult\\_race', 'Credit card clients\\_SEX', 'Credit card client\\_AGE', 'German credit data\\_sex', 'Credit scoring\\_Sex', 'Credit scoring\\_Age', 'Dutch census\\_sex', 'PAKDD\\_SEX']
    file.write("\\begin{table}[H]\n")
    file.write("\\begin{center}\n")
    file.write("\\caption{Statistical Parity Dataset}\n")
    file.write("\\begin{tabular}{c c c c c c c c c c c c}\n")

    file.write("\\hline\n")
    file.write("\\textbf{Dataset} &\\textbf{Origin} &\\textbf{DGGAN} &\\textbf{CTGAN} &\\textbf{TabFairGan} &\\textbf{FixedTabFairGanNoSM} &\\textbf{TabFairGanEOd} &\\textbf{DGGanEOd} \\\\\n")
    file.write("\\hline\n")
    for att in range(np.shape(arr)[0]):
        file.write(f'{datasets[att]} ')
        for gen in range(np.shape(arr)[1]):
            if arr_tmp[att][gen] == 0:
                file.write("&" + str(arr[att][gen]) + " ")
            else:
                file.write("&\\textbf{\\textcolor{red}{" + str(arr[att][gen]) + "}} ")
        file.write("\\\\\n")
    file.write("\\hline\n")
    file.write("\\end{tabular}\n")
    file.write("\\end{center}\n")
    file.write("\\end{table}\n")

    file.close()