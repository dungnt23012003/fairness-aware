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


def load_adult(file, protected_attribute, class_label):

    print(file)
    if file.__contains__('generation'):
        df = pd.read_csv(ROOT / '..' / 'data' / 'Generations' / file, sep=",")
    else:
        df = pd.read_csv(ROOT / '..' / 'data' / 'Origins' / file, sep=",")

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

    skf = StratifiedKFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train.append(X.iloc[train_index])
        X_test.append(X.iloc[test_index])
        y_train.append(y.iloc[train_index])
        y_test.append(y.iloc[test_index])

    feature = X.keys().tolist()
    sa_index = feature.index(protected_attribute)

    return X_train, X_test, y_train, y_test, sa_index


def run_experiment(X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name, f):
    result = []
    model_list = ['MLP', 'KNN', 'DT', 'LR']
    # model_list = ['MLP', 'KNN', 'DT', 'SVM', 'LR']
    for m in model_list:
        result_tmp = []
        print(m)
        num_fold = 0
        for X_train_fold, X_test_fold, y_train_fold, y_test_fold in zip(X_train, X_test, y_train, y_test):
            num_fold += 1
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
            Statistical_parity = calculate_performance_statistical_parity(X_test_fold.values, y_test_fold.values,
                                                                          y_predicts_fold, sa_index, p_Group)
            # print(Statistical_parity)
            result_fold.append(Statistical_parity['accuracy'].__round__(4))
            result_fold.append(Statistical_parity['balanced_accuracy'].__round__(4))
            result_fold.append(Statistical_parity['fairness'].__round__(4))

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

            filename = ROOT / '..' / 'result' / 'credit-card-clients' / 'ABROCA' / f'{protected_attribute}.{m}.{f}.{num_fold}.png'
            # Compute Abroca
            Abroca = compute_abroca(df_test, pred_col='pred_proba', label_col='true_label',
                                    protected_attr_col=protected_attribute,
                                    majority_protected_attr_val=1, n_grid=10000,
                                    plot_slices=False, majority_group_name=majority_group_name,
                                    minority_group_name=minority_group_name,
                                    file_name=filename).__round__(4)

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
                if math.isnan(arr[i][j]):
                    num_nan = num_nan + 1
                elif math.isinf(arr[i][j]):
                    num_inf = num_inf + 1
                else:
                    sum = sum + arr[i][j]
                    num = num + 1
            if num != 0:
                result_each_model.append(sum/num)
            elif num_inf > num_nan:
                result_each_model.append(math.inf)
            else:
                result_each_model.append(math.nan)
        result.append(result_each_model)
    return result


if __name__ == '__main__':

    file_lists = [['credit-card-clients.csv', 'credit-card-clients_generation.csv', 'credit-card-clients_generation_2.csv', 'credit-card-clients_generation_5.csv', 'credit-card-clients_generation_6_SEX.csv', 'credit-card-clients_generation_7_SEX.csv', 'credit-card-clients_generation_8_SEX.csv'],
                  ['credit-card-clients.csv', 'credit-card-clients_generation.csv', 'credit-card-clients_generation_2.csv', 'credit-card-clients_generation_5.csv', 'credit-card-clients_generation_6_AGE.csv', 'credit-card-clients_generation_7_AGE.csv', 'credit-card-clients_generation_8_AGE.csv']]

    protected_attribute_list = ['SEX', 'AGE']
    majority_group_name_list = ['Male', 'From 25 to 65']
    minority_group_name_list = ['Female', 'Other']
    class_label = 'class-label'
    p_Group_list = [0, 0]

    for protected_attribute, majority_group_name, minority_group_name, p_Group, file_list in zip(protected_attribute_list, majority_group_name_list, minority_group_name_list, p_Group_list, file_lists):
        file = open(ROOT / '..' / 'result' / 'credit-card-clients' / f'{protected_attribute}.csv', 'w')
        result = []
        print(protected_attribute)
        for f in file_list:
            X_train, X_test, y_train, y_test, sa_index = load_adult(f, protected_attribute, class_label)
            test = run_experiment(X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name, f)
            result.append(test)

        arr = np.array(result)
        arr_tmp = np.zeros(np.shape(arr))
        for model in range(np.shape(arr)[1]):
            for score in range(np.shape(arr)[2]):
                if score == 1 or score == 2:
                    max_position = -1
                    for gen in range(np.shape(arr)[0]):
                        if not math.isnan(arr[gen][model][score]):
                            max_position = gen
                            break
                    if max_position != -1:
                        for gen in range(np.shape(arr)[0]):
                            if abs(arr[gen][model][score]) >= abs(arr[max_position][model][score]):
                                max_position = gen
                        arr_tmp[max_position][model][score] = 1
                else:
                    min_position = -1
                    for gen in range(np.shape(arr)[0]):
                        if not math.isnan(arr[gen][model][score]):
                            min_position = gen
                            break
                    if min_position != -1:
                        for gen in range(np.shape(arr)[0]):
                            if abs(arr[gen][model][score]) <= abs(arr[min_position][model][score]):
                                min_position = gen
                        arr_tmp[min_position][model][score] = 1

        file.write("\\begin{table}[H]\n")
        file.write("\\begin{center}\n")
        file.write("\\caption{Credit card clients dataset: performance of predictive models. Protected attribute: " + protected_attribute + "}\n")
        file.write("\\begin{tabular}{c c c c c c c c c c}\n")
        file.write("\\hline\n")
        file.write("\\textbf{Method}&\\multicolumn{9}{c}{\\textbf{Predictive model}} \\\\\n")
        model_list = ['MLP', 'KNN', 'DT', 'LR']
        # model_list = ['MLP', 'KNN', 'DT', 'SVM', 'LR']
        for model in range(np.shape(arr)[1]):
            file.write("\\hline\n")
            file.write("\\textbf{}&\\multicolumn{9}{c}{\\textbf{" + model_list[model] + "}} \\\\\n")
            file.write("\\textbf{} &Acc &BA &SP &EO &EOd &PP &PE &TE &ABROCA \\\\\n")
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

