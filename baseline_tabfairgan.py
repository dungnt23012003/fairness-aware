import pandas as pd
from tabfairgan import TFG
from pathlib import Path
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def run_credit_card_clients():
    # Load your dataset
    df = pd.read_csv(ROOT / 'data' / 'Origins' / 'credit-card-clients.csv')
    df['class-label'] = ['1' if v == 1 else '0' for v in df['class-label']]
    df['AGE'] = ['25<=x<=65' if 25 <= v <= 65 else 'x>65 or x<25' for v in df['AGE']]
    # Define fairness configuration
    fairness_config = {
        'fair_epochs': 50,
        'lamda': 0.5,
        'S': 'AGE',
        'Y': 'class-label',
        'S_under': 'x>65 or x<25',
        'Y_desire': '1'
    }

    # Initialize TabFairGAN with fairness constraints
    tfg = TFG(df, epochs=50, batch_size=32, device='cuda:0', fairness_config=fairness_config)

    # Train the model
    tfg.train()

    # Generate synthetic data
    fake_df = tfg.generate_fake_df(num_rows=len(df))
    fake_df['AGE'] = [40 if v == '25<=x<=65' else 15 for v in df['AGE']]
    fake_df.to_csv(ROOT / 'data' / 'Generations' / 'credit-card-clients_generation_6_AGE.csv')


def run_adult():
    # Load your dataset
    df = pd.read_csv(ROOT / 'data' / 'Origins' / 'adult.csv')
    df['class-label'] = ['1' if v == 1 else '0' for v in df['class-label']]
    df['age'] = ['25<=x<=65' if 25 <= v <= 65 else 'x>65 or x<25' for v in df['age']]
    # Define fairness configuration
    fairness_config = {
        'fair_epochs': 50,
        'lamda': 0.5,
        'S': 'age',
        'Y': 'class-label',
        'S_under': 'x>65 or x<25',
        'Y_desire': '1'
    }

    # Initialize TabFairGAN with fairness constraints
    tfg = TFG(df, epochs=50, batch_size=32, device='cuda:0', fairness_config=fairness_config)

    # Train the model
    tfg.train()

    # Generate synthetic data
    fake_df = tfg.generate_fake_df(num_rows=len(df))
    fake_df['age'] = [40 if v == '25<=x<=65' else 15 for v in df['age']]
    # fake_df.to_csv(ROOT / 'data' / 'Generations' / 'adult_generation_6_age.csv')


def run_bank_marketing():
    # Load your dataset
    df = pd.read_csv(ROOT / 'data' / 'Origins' / 'bank-marketing.csv')
    df['age'] = ['25<=x<=65' if 25 <= v <= 65 else 'x>65 or x<25' for v in df['age']]
    # Define fairness configuration
    fairness_config = {
        'fair_epochs': 50,
        'lamda': 0.5,
        'S': 'age',
        'Y': 'class-label',
        'S_under': 'x>65 or x<25',
        'Y_desire': 'yes'
    }

    # Initialize TabFairGAN with fairness constraints
    tfg = TFG(df, epochs=50, batch_size=32, device='cuda:0', fairness_config=fairness_config)

    # Train the model
    tfg.train()

    # Generate synthetic data
    fake_df = tfg.generate_fake_df(num_rows=len(df))
    fake_df['age'] = [40 if v == '25<=x<=65' else 15 for v in df['age']]
    fake_df.to_csv(ROOT / 'data' / 'Generations' / 'bank-marketing_generation_6_age.csv')


def run_german_credit():
    # Load your dataset
    df = pd.read_csv(ROOT / 'data' / 'Origins' / 'german-credit-data.csv')
    df['class-label'] = ['1' if v == 1 else '0' for v in df['class-label']]
    # Define fairness configuration
    fairness_config = {
        'fair_epochs': 50,
        'lamda': 0.5,
        'S': 'sex',
        'Y': 'class-label',
        'S_under': 'female',
        'Y_desire': '1'
    }

    # Initialize TabFairGAN with fairness constraints
    tfg = TFG(df, epochs=50, batch_size=32, device='cuda:0', fairness_config=fairness_config)

    # Train the model
    tfg.train()

    # Generate synthetic data
    fake_df = tfg.generate_fake_df(num_rows=len(df))
    fake_df.to_csv(ROOT / 'data' / 'Generations' / 'german-credit-data_generation_6_sex.csv', index=False)


def run_credit_scoring():
    # Load your dataset
    df = pd.read_csv(ROOT / 'data' / 'Origins' / 'credit-scoring.csv')
    df['class-label'] = ['1' if v == 1 else '0' for v in df['class-label']]
    df['Age'] = ['25<=x<=65' if 25 <= v <= 65 else 'x>65 or x<25' for v in df['Age']]
    # Define fairness configuration
    fairness_config = {
        'fair_epochs': 50,
        'lamda': 0.5,
        'S': 'Age',
        'Y': 'class-label',
        'S_under': 'x>65 or x<25',
        'Y_desire': '1'
    }

    # Initialize TabFairGAN with fairness constraints
    tfg = TFG(df, epochs=50, batch_size=32, device='cuda:0', fairness_config=fairness_config)

    # Train the model
    tfg.train()

    # Generate synthetic data
    fake_df = tfg.generate_fake_df(num_rows=len(df))
    fake_df.to_csv(ROOT / 'data' / 'Generations' / 'credit-scoring_generation_6_Age.csv', index=False)


if __name__ == '__main__':

    # file_name = ROOT / 'data' / 'Generations' / 'adult_generation_7_race.csv'
    # df = pd.read_csv(file_name)
    # df['gender'] = [1 if v == 'Male' else 0 for v in df['gender']]
    # df['age'] = [1 if 25 <= v <= 65 else 0 for v in df['age']]
    # df['race'] = [1 if v == 'White' else 0 for v in df['race']]
    # df.to_csv(file_name, index=False)

    # file_name = ROOT / 'data' / 'Generations' / 'bank-marketing_generation_7_age.csv'
    # df = pd.read_csv(file_name)
    # df['age'] = [1 if 25 <= v <= 65 else 0 for v in df['age']]  # need to change
    # df['class-label'] = [1 if v == "yes" else 0 for v in df['class-label']]
    # df.to_csv(file_name, index=False)
    #
    # file_name = ROOT / 'data' / 'Generations' / 'credit-card-clients_generation_7_AGE.csv'
    # df = pd.read_csv(file_name)
    # df['AGE'] = [1 if 25 <= v <= 65 else 0 for v in df['AGE']]
    # df['SEX'] = [1 if v == 1 else 0 for v in df['SEX']]
    # df.to_csv(file_name, index=False)
    #
    # file_name = ROOT / 'data' / 'Generations' / 'credit-scoring_generation_6_Age.csv'
    # df = pd.read_csv(file_name)
    # df['Age'] = [1 if 25 <= v <= 65 else 0 for v in df['Age']]
    # df['Sex'] = [1 if v == 1 else 0 for v in df['Sex']]
    # df.to_csv(file_name, index=False)
    #
    # file_name = ROOT / 'data' / 'Generations' / 'german-credit-data_generation_6_age.csv'
    # df = pd.read_csv(file_name)
    # df['age'] = [1 if 25 <= v <= 65 else 0 for v in df['age']]
    # df['sex'] = [1 if v == 'male' else 0 for v in df['sex']]
    # df.to_csv(file_name, index=False)

    logits = torch.randn(10, 2)
    print(logits)
    a = torch.nn.functional.gumbel_softmax(logits, tau=0.1, hard=False)
    print(a)
    b = torch.nn.functional.gumbel_softmax(logits, tau=0.1, hard=True)
    print(b)
