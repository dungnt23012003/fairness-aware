import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from compute_abroca import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from my_useful_functions import calculate_performance_statistical_parity_dataset, calculate_performance_statistical_parity,calculate_performance_equalized_odds,calculate_performance_equal_opportunity,calculate_performance_predictive_parity,calculate_performance_predictive_equality,calculate_performance_treatment_equality
from sklearn import preprocessing
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import sklearn.metrics as metrics
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

def load_adult(file, model):
    print(file)

    if file.__contains__('generation'):
        df = pd.read_csv(ROOT / '..' / 'data' / 'Generations' / file, sep=",")
    else:
        df = pd.read_csv(ROOT / '..' / 'data' / 'Origins' / file, sep=",")
    protected_attribute_list = ['sex', 'race', 'age']
    majority_group_name_list = ["Male", "White", "From 25 to 65"]
    minority_group_name_list = ["Female", "Non-White", "Other"]
    class_label = 'income'
    filename = f'D://PycharmProjects//DGGAN//AbrocaPlot//{model}//{model}.' + file + '.abroca.'

    df['sex'] = [1 if v == ' Male' else 0 for v in df['sex']]
    df['age'] = [1 if 25 <= v <= 65 else 0 for v in df['age']]
    df['race'] = [1 if v == " White" else 0 for v in df['race']]
    df['income'] = [1 if v == " >50K" else 0 for v in df['income']]

    le = preprocessing.LabelEncoder()
    for i in df.columns:
        if df[i].dtypes == 'object':
            df[i] = le.fit_transform(df[i])
    # Splitting data into train and test
    length = len(df.columns)
    X = df.iloc[:, :length - 1]
    y = df[class_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Get index
    feature = X.keys().tolist()
    sa_index_list = []
    for v in protected_attribute_list:
        sa_index_list.append(feature.index(v))

    p_Group = 0

    return X_train, X_test, y_train, y_test, sa_index_list, p_Group, protected_attribute_list, filename, majority_group_name_list, minority_group_name_list


def load_australian(file, model):
    print(file)
    if file.__contains__('generation'):
        df = pd.read_csv(ROOT / '..' / 'data' / 'Generations' / file, sep=",")
    else:
        df = pd.read_csv(ROOT / '..' / 'data' / 'Origins' / file, sep=",")
    protected_attribute_list = ['One', 'Two']
    majority_group_name_list = ["Male", "From 25 to 65"]
    minority_group_name_list = ["Female", "Other"]
    class_label = 'Fifteen'
    filename = f'D://PycharmProjects//DGGAN//AbrocaPlot//{model}//{model}.' + file + '.abroca.'

    df['Two'] = [1 if 25 <= v <= 65 else 0 for v in df['Two']]

    le = preprocessing.LabelEncoder()
    for i in df.columns:
        if df[i].dtypes == 'object':
            df[i] = le.fit_transform(df[i])
    # Splitting data into train and test
    length = len(df.columns)
    X = df.iloc[:, :length - 1]
    y = df[class_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Get index
    feature = X.keys().tolist()
    sa_index_list = []
    for v in protected_attribute_list:
        sa_index_list.append(feature.index(v))

    p_Group = 0

    return X_train, X_test, y_train, y_test, sa_index_list, p_Group, protected_attribute_list, filename, majority_group_name_list, minority_group_name_list


def load_insurance(file, model):
    print(file)
    if file.__contains__('generation'):
        df = pd.read_csv(ROOT / '..' / 'data' / 'Generations' / file, sep=",")
    else:
        df = pd.read_csv(ROOT / '..' / 'data' / 'Origins' / file, sep=",")
    protected_attribute_list = ['Gender', 'HumanAge']
    majority_group_name_list = ["Male", "From 25 to 65"]
    minority_group_name_list = ["Female", "Other"]
    class_label = 'Response'
    filename = f'D:/PycharmProjects/DGGAN/AbrocaPlot/{model}//{model}.' + file + '.abroca.'

    df['Gender'] = [1 if v == 'Male' else 0 for v in df['Gender']]
    df['HumanAge'] = [1 if 25 <= v <= 65 else 0 for v in df['HumanAge']]

    le = preprocessing.LabelEncoder()
    for i in df.columns:
        if df[i].dtypes == 'object':
            df[i] = le.fit_transform(df[i])
    # Splitting data into train and test
    length = len(df.columns)
    X = df.iloc[:, :length - 1]
    y = df[class_label]
    X_train = []
    X_test =[]
    y_train = []
    y_test = []
    skf = StratifiedKFold(n_splits=10)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train.append(X.iloc[train_index])
        X_test.append(X.iloc[test_index])
        y_train.append(y.iloc[train_index])
        y_test.append(y.iloc[test_index])
    # Get index
    feature = X.keys().tolist()
    sa_index_list = []
    for v in protected_attribute_list:
        sa_index_list.append(feature.index(v))

    p_Group = 0

    return X_train, X_test, y_train, y_test, sa_index_list, p_Group, protected_attribute_list, filename, majority_group_name_list, minority_group_name_list
def run_experiment(X_train, X_test, y_train, y_test, sa_index_list, p_Group, protected_attribute_list, filename, majority_group_name_list, minority_group_name_list, model):

    model.fit(X_train, y_train)
    y_predicts = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)

    result = []
    # Print measures
    for sa_index, majority_group_name, minority_group_name, protected_attribute in zip(sa_index_list, majority_group_name_list, minority_group_name_list, protected_attribute_list):
        result_attr = []
        print(protected_attribute)

        print("Statistical parity dataset:")
        Statistical_parity_dataset = calculate_performance_statistical_parity_dataset(X_test.values, y_test.values, y_predicts, sa_index, p_Group)['fairness'].__round__(4)
        print(Statistical_parity_dataset)
        result_attr.append(Statistical_parity_dataset)

        print("Statistical parity:")
        Statistical_parity = calculate_performance_statistical_parity(X_test.values, y_test.values, y_predicts, sa_index, p_Group)['fairness'].__round__(4)
        print(Statistical_parity)
        result_attr.append(Statistical_parity)


        print("Equal opportunity")
        Equal_opportunity = calculate_performance_equal_opportunity(X_test.values, y_test.values, y_predicts, sa_index, p_Group)['fairness'].__round__(4)
        print(Equal_opportunity)
        result_attr.append(Equal_opportunity)

        print("Equalized odds")
        Equalized_odds = calculate_performance_equalized_odds(X_test.values, y_test.values, y_predicts, sa_index, p_Group)['fairness'].__round__(4)
        print(Equalized_odds)
        result_attr.append(Equalized_odds)

        print("Predictive parity")
        Predictive_parity = calculate_performance_predictive_parity(X_test.values, y_test.values, y_predicts, sa_index, p_Group)['fairness'].__round__(4)
        print(Predictive_parity)
        result_attr.append(Predictive_parity)

        print("Predictive equality")
        Predictive_equality = calculate_performance_predictive_equality(X_test.values, y_test.values, y_predicts, sa_index, p_Group)['fairness'].__round__(4)
        print(Predictive_equality)
        result_attr.append(Predictive_equality)

        print("Treatment equality")
        Treatment_equality = calculate_performance_treatment_equality(X_test.values, y_test.values, y_predicts, sa_index, p_Group)['fairness'].__round__(4)
        print(Treatment_equality)
        result_attr.append(Treatment_equality)

        # make predictions
        X_test['pred_proba'] = y_pred_probs[:, 1:2]
        X_test['true_label'] = y_test
        df_test = X_test

        # Compute Abroca
        Abroca = compute_abroca(df_test, pred_col='pred_proba', label_col='true_label',
                               protected_attr_col=protected_attribute,
                               majority_protected_attr_val=1, n_grid=10000,
                               plot_slices=False, majority_group_name=majority_group_name,
                               minority_group_name=minority_group_name, file_name=filename + protected_attribute + ".png").__round__(4)

        print("ABROCA:", Abroca)
        result_attr.append(Abroca)

        result.append(result_attr)
    return result


if __name__ == '__main__':
    list_model = ['MLP']
    for m in list_model:
        file = open(ROOT / '..' / 'result' / f'{m}', 'w')
        print(m)
        result = []
        file_list = ['adult.csv', 'adult_generation.csv', 'adult_generation_2.csv']
        for f in file_list:
            if m == 'MLP':
                model = MLPClassifier()
            elif m == 'kNN':
                model = KNeighborsClassifier(n_neighbors=5)
            elif m == 'NB':
                model = GaussianNB()
            else:
                model = LogisticRegression()

            X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, filename, majority_group_name, minority_group_name = load_adult(f, m)
            result.append(run_experiment(X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, filename, majority_group_name, minority_group_name, model))
        arr = np.array(result)
        for att in range(np.shape(arr)[1]):
            for score in range(np.shape(arr)[2]):
                for data in range(np.shape(arr)[0]):
                    file.write(str(arr[data][att][score]) + ' ')
                file.write('|')
            file.write('\n')

        result = []
        file_list = ['australian.csv', 'australian_generation.csv', 'australian_generation_2.csv']
        for f in file_list:
            if m == 'MLP':
                model = MLPClassifier()
            elif m == 'kNN':
                model = KNeighborsClassifier(n_neighbors=5)
            elif m == 'NB':
                model = GaussianNB()
            else:
                model = LogisticRegression()
            X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, filename, majority_group_name, minority_group_name = load_australian(f, m)
            result.append(run_experiment(X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, filename, majority_group_name, minority_group_name, model))
        arr = np.array(result)
        for att in range(np.shape(arr)[1]):
            for score in range(np.shape(arr)[2]):
                for data in range(np.shape(arr)[0]):
                    file.write(str(arr[data][att][score]) + ' ')
                file.write('|')

            file.write('\n')

        result = []
        file_list = ['insurance.csv', 'insurance_generation.csv', 'insurance_generation_2.csv']
        for f in file_list:
            if m == 'MLP':
                model = MLPClassifier()
            elif m == 'kNN':
                model = KNeighborsClassifier(n_neighbors=5)
            elif m == 'NB':
                model = GaussianNB()
            else:
                model = LogisticRegression()
            X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, filename, majority_group_name, minority_group_name = load_insurance(f, m)
            result.append(run_experiment(X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, filename, majority_group_name, minority_group_name, model))
        arr = np.array(result)
        for att in range(np.shape(arr)[1]):
            for score in range(np.shape(arr)[2]):
                for data in range(np.shape(arr)[0]):
                    file.write(str(arr[data][att][score]) + ' ')
                file.write('|')
            file.write('\n')

        file.close()

