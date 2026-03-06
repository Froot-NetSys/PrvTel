from datetime import datetime
from lzma import CHECK_CRC32
import numpy as np
import torch
from rdt.transformers import numerical, categorical
import pandas as pd
import socket
import os
import glob

# Graph Visualisation
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 54
plt.rcParams["figure.figsize"] = (5,9)

from sklearn.preprocessing import StandardScaler, RobustScaler


def set_seed(seed):
    """
    Sets random seeds for numpy and PyTorch to ensure reproducibility.
    
    Args:
        seed (int): The random seed value to use
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_file_size(path):
    """
    Returns the size (in bytes) of the given file(s).
    Accepts either a list of file paths or a glob string.
    """
    file_paths = path
    if isinstance(path, str):
        file_paths = glob.glob(path)
    size = 0
    for file_path in file_paths:
        size += os.path.getsize(file_path)
    return size


def socket_is_used(port=8097, hostname='localhost'):
    """
    Checks if a network socket port is already in use.
    
    Args:
        port (int): Port number to check (default: 8097)
        hostname (str): Hostname to check (default: 'localhost')
    
    Returns:
        bool: True if port is in use, False otherwise
    """
    is_used = False
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((hostname, port))
    except socket.error:
        is_used = True
    finally:
        s.close()
    return is_used
