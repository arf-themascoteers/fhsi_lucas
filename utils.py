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


def is_test():
    return False


if __name__ == "__main__":
    savi = np.array([1,2,3,4,5,6,7,8,9])
    soc = np.array([10,20,30,40,50,60,70,80,90])
    print("Pearson",calculate_pc(soc,savi))
    print("R2",r2_score(soc,savi))
