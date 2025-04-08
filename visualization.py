import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sdmetrics.visualization import get_column_plot
from sdmetrics.reports.single_table import QualityReport

def visualization(real, fake, categorical_columns, continuous_columns):
    real_table = pd.read_csv(real)
    synthetic_table = pd.read_csv(fake)
    for column in categorical_columns:
        fig = get_column_plot(
            real_data=real_table,
            synthetic_data=synthetic_table,
            column_name=column,
            plot_type='bar'
        )
        fig.show()
        time.sleep(3)


    for column in continuous_columns:
        fig = get_column_plot(
            real_data=real_table,
            synthetic_data=synthetic_table,
            column_name=column,
            plot_type='distplot'
        )
        fig.show()
        time.sleep(3)


def report(real, fake, categorical_columns, continuous_columns, metadata):

    df_real = pd.read_csv(real)
    df_fake = pd.read_csv(fake)
    df_real[continuous_columns] = df_real[continuous_columns].astype('int64')
    df_real[categorical_columns] = df_real[categorical_columns].astype('category')
    df_fake[continuous_columns] = df_fake[continuous_columns].astype('int64')
    df_fake[categorical_columns] = df_fake[categorical_columns].astype('category')

    my_report = QualityReport()
    my_report.generate(df_real, df_fake, metadata)
    print(my_report.get_score())


if __name__ == '__main__':
    csv_real = 'data/insurance.csv'
    csv_fake = 'generations/insurance_generation_2.csv'

    categorical_columns = ['Gender', 'DrivingLicense', 'RegionCode', 'PreviouslyInsured', 'VehicleAge', 'VehicleDamage', 'Response']
    continuous_columns = ['HumanAge', 'AnnualPremium', 'PolicySalesChannel', 'Vintage']

    metadata = {
        "columns": {
            'HumanAge': {
                "sdtype": "numerical",
            },
            'AnnualPremium': {
                "sdtype": "numerical"
            },
            'PolicySalesChannel': {
                "sdtype": "numerical",
            },
            'Vintage': {
                "sdtype": "numerical"
            },
            'Gender': {
                "sdtype": "categorical"
            },
            'DrivingLicense': {
                "sdtype": "categorical"
            },
            'RegionCode': {
                "sdtype": "categorical"
            },
            'PreviouslyInsured': {
                "sdtype": "categorical"
            },
            'VehicleAge': {
                "sdtype": "categorical"
            },
            'VehicleDamage': {
                "sdtype": "categorical"
            },
            'Response': {
                "sdtype": "categorical"
            },
        }
    }

    # csv_real = 'data/australian.csv'
    # csv_fake = 'generations/australian_generation_2.csv'
    #
    # categorical_columns = ['One', 'Four', 'Five', 'Six', 'Eight', 'Nine', 'Eleven', 'Twelve', 'Fifteen']
    # continuous_columns = ['Two', 'Three', 'Seven', 'Ten', 'Thirteen', 'FOurteen']
    #
    # metadata = {
    #     "columns": {
    #         'Two': {
    #             "sdtype": "numerical",
    #         },
    #         'Three': {
    #             "sdtype": "numerical"
    #         },
    #         'Seven': {
    #             "sdtype": "numerical",
    #         },
    #         'Ten': {
    #             "sdtype": "numerical"
    #         },
    #         'Thirteen': {
    #             "sdtype": "numerical"
    #         },
    #         'FOurteen': {
    #             "sdtype": "numerical"
    #         },
    #         'One': {
    #             "sdtype": "categorical"
    #         },
    #         'Four': {
    #             "sdtype": "categorical"
    #         },
    #         'Five': {
    #             "sdtype": "categorical"
    #         },
    #         'Six': {
    #             "sdtype": "categorical"
    #         },
    #         'Eight': {
    #             "sdtype": "categorical"
    #         },
    #         'Nine': {
    #             "sdtype": "categorical"
    #         },
    #         'Eleven': {
    #             "sdtype": "categorical"
    #         },
    #         'Twelve': {
    #             "sdtype": "categorical"
    #         },
    #         'Fifteen': {
    #             "sdtype": "categorical"
    #         },
    #     }
    # }

    # csv_real = 'data/olympics.csv'
    # csv_fake = 'generations/olympics_generation.csv'
    #
    # categorical_columns = ['Sex', 'Year', 'Season', 'City', 'Sport', 'Medal', 'AOS', 'AOE']
    # continuous_columns = ['Age', 'Height', 'Weight']
    #
    # metadata = {
    #     "columns": {
    #         'Age': {
    #             "sdtype": "numerical",
    #         },
    #         'Height': {
    #             "sdtype": "numerical"
    #         },
    #         'Weight': {
    #             "sdtype": "numerical",
    #         },
    #         'Sex': {
    #             "sdtype": "categorical"
    #         },
    #         'Year': {
    #             "sdtype": "categorical"
    #         },
    #         'Season': {
    #             "sdtype": "categorical"
    #         },
    #         'City': {
    #             "sdtype": "categorical"
    #         },
    #         'Sport': {
    #             "sdtype": "categorical"
    #         },
    #         'Medal': {
    #             "sdtype": "categorical"
    #         },
    #         'AOS': {
    #             "sdtype": "categorical"
    #         },
    #         'AOE': {
    #             "sdtype": "categorical"
    #         },
    #     }
    # }

    report(csv_real, csv_fake, categorical_columns, continuous_columns, metadata)
    visualization(csv_real, csv_fake, categorical_columns, continuous_columns)

