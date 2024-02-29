from scipy.stats import pearsonr
import torch
import pandas as pd


def calculate_pc(x1,x2):
    correlation_coefficient, p_value = pearsonr(x1, x2)
    return correlation_coefficient


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_all_features():
    df = pd.read_csv(get_data_file())
    columns = list(df.columns)
    return columns[0:-1]


def get_data_file():
    return "data/lucas_s2.csv"


if __name__ == "__main__":
    print(get_all_features())