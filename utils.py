from scipy.stats import pearsonr
import torch
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np


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
    # x = np.array([1,2,3,4,5,6,7,8,9])
    # y = x*10
    # print(calculate_pc(y,x))
    # print(r2_score(y,x))
    x = torch.tensor([[1,2,3],[10,20,30]])
    y = x[:,[0,2]]
    print(y)