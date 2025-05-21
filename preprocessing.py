


import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def preprocessing_before_train(df, source):
    if source.__contains__("adult"):
        df['gender'] = [1 if v == 'Male' else 0 for v in df['gender']]
        df['age'] = [1 if 25 <= v <= 65 else 0 for v in df['age']]
        df['race'] = [1 if v == 'White' else 0 for v in df['race']]
    elif source.__contains__("bank-marketing"):
        df['age'] = [1 if 25 <= v <= 65 else 0 for v in df['age']]  # need to change
        df['class-label'] = [1 if v == "yes" else 0 for v in df['class-label']]
    elif source.__contains__("credit-card-clients"):
        df['AGE'] = [1 if 25 <= v <= 65 else 0 for v in df['AGE']]
    elif source.__contains__("german-credit-data"):
        df['age'] = [1 if 25 <= v <= 65 else 0 for v in df['age']]
        df['sex'] = [1 if v == 'male' else 0 for v in df['sex']]
    elif source.__contains__("kdd-census-income"):
        df['sex'] = [1 if v == 'Male' else 0 for v in df['sex']]
        df['age'] = [1 if 25 <= v <= 65 else 0 for v in df['age']]
        df['race'] = [1 if v == 'White' else 0 for v in df['race']]
    elif source.__contains__("credit-scoring"):
        df['Age'] = [1 if 25 <= v <= 65 else 0 for v in df['Age']]
    elif source.__contains__("PAKDD"):
        df['SEX'] = [1 if v == 'M' else 0 for v in df['SEX']]
        df['AGE'] = [1 if 25 <= v <= 65 else 0 for v in df['AGE']]
    return df


