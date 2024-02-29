from scipy.stats import pearsonr
import torch


def calculate_pc(x1,x2):
    correlation_coefficient, p_value = pearsonr(x1, x2)
    return correlation_coefficient


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
