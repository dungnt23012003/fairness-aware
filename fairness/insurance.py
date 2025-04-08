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
import matplotlib

matplotlib.use('TkAgg')
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

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


def load_insurance(file, protected_attribute, class_label):

    print(file)
    if file.__contains__('generation'):
        df = pd.read_csv(ROOT / '..' / 'data' / 'Generations' / file, sep=",")
    else:
        df = pd.read_csv(ROOT / '..' / 'data' / 'Origins' / file, sep=",")

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
    X_test = []
    y_train = []
    y_test = []

    skf = StratifiedKFold(n_splits=10)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train.append(X.iloc[train_index])
        X_test.append(X.iloc[test_index])
        y_train.append(y.iloc[train_index])
        y_test.append(y.iloc[test_index])

    feature = X.keys().tolist()
    sa_index = feature.index(protected_attribute)

    return X_train, X_test, y_train, y_test, sa_index


def run_experiment(X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name):
    result = []

    model_list = ['MLP', 'KNN', 'DT', 'SVM', 'LR']
    for m in model_list:
        result_tmp = []
        print(m)
        for X_train_fold, X_test_fold, y_train_fold, y_test_fold in zip(X_train, X_test, y_train, y_test):
            if m == 'MLP':
                model = MLPClassifier()
            elif m == 'KNN':
                model = KNeighborsClassifier(n_neighbors=5)
            elif m == 'DT':
                model = DecisionTreeClassifier()
            elif m == 'SVM':
                model = SVC(probability=True)
            else:
                model = LogisticRegression()


            model.fit(X_train_fold, y_train_fold)
            y_predicts_fold = model.predict(X_test_fold)
            y_pred_probs_fold = model.predict_proba(X_test_fold)

            result_fold = []
            # print(protected_attribute)

            # print("Statistical parity dataset:")
            Statistical_parity_dataset = calculate_performance_statistical_parity_dataset(X_test_fold.values, y_test_fold.values, y_predicts_fold, sa_index, p_Group)['fairness'].__round__(4)
            # print(Statistical_parity_dataset)
            result_fold.append(Statistical_parity_dataset)

            # print("Statistical parity:")
            Statistical_parity = calculate_performance_statistical_parity(X_test_fold.values, y_test_fold.values, y_predicts_fold, sa_index, p_Group)['fairness'].__round__(4)
            # print(Statistical_parity)
            result_fold.append(Statistical_parity)

            # print("Equal opportunity")
            Equal_opportunity = calculate_performance_equal_opportunity(X_test_fold.values, y_test_fold.values, y_predicts_fold, sa_index, p_Group)['fairness'].__round__(4)
            # print(Equal_opportunity)
            result_fold.append(Equal_opportunity)

            # print("Equalized odds")
            Equalized_odds = calculate_performance_equalized_odds(X_test_fold.values, y_test_fold.values, y_predicts_fold, sa_index, p_Group)['fairness'].__round__(4)
            # print(Equalized_odds)
            result_fold.append(Equalized_odds)

            # print("Predictive parity")
            Predictive_parity = calculate_performance_predictive_parity(X_test_fold.values, y_test_fold.values, y_predicts_fold, sa_index, p_Group)['fairness'].__round__(4)
            # print(Predictive_parity)
            result_fold.append(Predictive_parity)

            # print("Predictive equality")
            Predictive_equality = calculate_performance_predictive_equality(X_test_fold.values, y_test_fold.values, y_predicts_fold, sa_index, p_Group)['fairness'].__round__(4)
            # print(Predictive_equality)
            result_fold.append(Predictive_equality)

            # print("Treatment equality")
            Treatment_equality = calculate_performance_treatment_equality(X_test_fold.values, y_test_fold.values, y_predicts_fold, sa_index, p_Group)['fairness'].__round__(4)
            # print(Treatment_equality)
            result_fold.append(Treatment_equality)

            # make predictions
            df_test = X_test_fold.copy()
            df_test['pred_proba'] = y_pred_probs_fold[:, 1:2]
            df_test['true_label'] = y_test_fold

            filename = ""
            # Compute Abroca
            Abroca = compute_abroca(df_test, pred_col='pred_proba', label_col='true_label',
                                    protected_attr_col=protected_attribute,
                                    majority_protected_attr_val=1, n_grid=10000,
                                    plot_slices=False, majority_group_name=majority_group_name,
                                    minority_group_name=minority_group_name,
                                    file_name= filename + protected_attribute + ".png").__round__(4)

            # print("ABROCA:", Abroca)
            result_fold.append(Abroca)
            result_tmp.append(result_fold)
        arr = np.array(result_tmp)
        result_each_model = []
        for j in range(np.shape(arr)[1]):
            sum = 0
            num_nan = 0
            num = 0
            num_inf = 0
            for i in range(np.shape(arr)[0]):
                if arr[i][j] == math.nan:
                    num_nan = num_nan + 1
                elif arr[i][j] == math.inf:
                    num_inf = num_inf + 1
                else:
                    sum = sum + arr[i][j]
                    num = num + 1
            if num > num_inf + num_nan:
                result_each_model.append(sum/num)
            elif num_inf > num_nan:
                result_each_model.append(math.inf)
            else:
                result_each_model.append(math.nan)
        result.append(result_each_model)
    return result


if __name__ == '__main__':

    file_list = ['insurance.csv', 'insurance_generation.csv', 'insurance_generation_2.csv'] # need to change

    protected_attribute_list = ['Gender', 'HumanAge']
    majority_group_name_list = ["Male", "From 25 to 65"]
    minority_group_name_list = ["Female", "Other"]
    class_label = 'Response'
    p_Group_list = [0, 0] # need to change

    for protected_attribute, majority_group_name, minority_group_name, p_Group in zip(protected_attribute_list, majority_group_name_list, minority_group_name_list, p_Group_list):
        file = open(ROOT / '..' / 'result' / 'insurance' / f'{protected_attribute}.csv', 'w') # need to change
        result = []
        print(protected_attribute)
        for f in file_list:
            X_train, X_test, y_train, y_test, sa_index = load_insurance(f, protected_attribute, class_label)
            test = run_experiment(X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name)
            result.append(test)

        arr = np.array(result)
        arr_tmp = np.zeros(np.shape(arr))
        for model in range(np.shape(arr)[1]):
            for score in range(np.shape(arr)[2]):
                min_position = 0
                for gen in range(np.shape(arr)[0]):
                    if abs(arr[gen][model][score]) < abs(arr[min_position][model][score]):
                        min_position = gen
                arr_tmp[min_position][model][score] = 1

        file.write("\\begin table[H]\n")
        file.write("\\begin{center}\n")
        file.write("\\caption{adult\\_" + protected_attribute + "}\n")
        file.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n")
        file.write("\\hline\n")
        file.write("\\textbf{Dataset}&\\multicolumn{7}{|c|}{\\textbf{SP dataset}}\\\\\n")
        file.write("\\hline\n")

        for gen in range(np.shape(arr)[0]):
            if arr_tmp[gen][0][0] == 0:
                file.write(file_list[gen].replace("_", "\\_") + "&\\multicolumn{7}{|c|}{" + str(arr[gen][0][0].__round__(4)) + "}\\\\\n")
            else:
                file.write(file_list[gen].replace("_", "\\_") + "&\\multicolumn{7}{|c|}{\\textbf{\\textcolor{red}{" + str(arr[gen][0][0].__round__(4)) + "}}}\\\\\n")
            file.write("\\hline\n")

        model_list = ['MLP', 'KNN', 'DT', 'SVM', 'LR']
        for model in range(np.shape(arr)[1]):
            file.write("\\textbf{}&\\multicolumn{7}{|c|}{\\textbf{" + model_list[model] + "}} \\\\\n")
            file.write("\\cline{2-8}\n")
            file.write("\\textbf{} &SP &EO &EOdd &PP &PE &TE &Abroca \\\\\n")
            file.write("\\hline\n")
            for gen in range(np.shape(arr)[0]):
                file.write(file_list[gen].replace("_", "\\_") + " ")
                for score in range(1, np.shape(arr)[2]):
                    if arr_tmp[gen][model][score] == 0:
                        file.write("&" + str(arr[gen][model][score].__round__(4)) + " ")
                    else:
                        file.write("&\\textbf{\\textcolor{red}{" + str(arr[gen][model][score].__round__(4)) + "}} ")
                file.write("\\\\\n")
                file.write("\\hline\n")

        file.write("\\end{tabular}\n")
        file.write("\\end{center}\n")
        file.write("\\end{table}\n")

        file.close()

