


import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def preprocessing_before_train(df_train, source):
    if source.__contains__("adult"):
        df_train['gender'] = [1 if v == 'Male' else 0 for v in df_train['gender']]
        df_train['age'] = [1 if 25 <= v <= 65 else 0 for v in df_train['age']]
        df_train['race'] = [1 if v == 'White' else 0 for v in df_train['race']]
    elif source.__contains__("bank-marketing"):
        df_train['age'] = [1 if 25 <= v <= 65 else 0 for v in df_train['age']]  # need to change
        df_train['class-label'] = [1 if v == "yes" else 0 for v in df_train['class-label']]
    elif source.__contains__("credit-card-clients"):
        df_train['AGE'] = [1 if 25 <= v <= 65 else 0 for v in df_train['AGE']]
    elif source.__contains__("german-credit-data"):
        df_train['age'] = [1 if 25 <= v <= 65 else 0 for v in df_train['age']]
        df_train['sex'] = [1 if v == 'male' else 0 for v in df_train['sex']]
    elif source.__contains__("kdd-census-income"):
        df_train['sex'] = [1 if v == 'Male' else 0 for v in df_train['sex']]
        df_train['age'] = [1 if 25 <= v <= 65 else 0 for v in df_train['age']]
        df_train['race'] = [1 if v == 'White' else 0 for v in df_train['race']]
    elif source.__contains__("credit-scoring"):
        df_train['Age'] = [1 if 25 <= v <= 65 else 0 for v in df_train['Age']]
    elif source.__contains__("PAKDD"):
        df_train['SEX'] = [1 if v == 'M' else 0 for v in df_train['SEX']]
        df_train['AGE'] = [1 if 25 <= v <= 65 else 0 for v in df_train['AGE']]
    return df_train


if __name__ == '__main__':
    df = pd.read_csv(ROOT/'data'/'Origins'/'PAKDD.csv')
    count = df['AGE'].value_counts()
    count.to_csv(ROOT/'data'/'Origins'/'PAKDD_TEST.csv')