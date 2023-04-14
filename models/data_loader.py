import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"


def get_tabular_train_test_data(split_percent=0.8):
    X = datasets.load_boston()["data"]
    y = datasets.load_boston()["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - split_percent, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def get_tabular_train_test_dataloader(batch_size=32, split_percent=0.8):
    X_train, X_test, y_train, y_test = get_tabular_train_test_data(split_percent)
    X_train = torch.tensor(X_train.astype("float32"), dtype=torch.float32)
    X_test = torch.tensor(X_test.astype("float32"), dtype=torch.float32)
    y_train = torch.tensor(y_train.astype("float32"), dtype=torch.float32)
    y_test = torch.tensor(y_test.astype("float32"), dtype=torch.float32)
    train_dataloader = DataLoader(
        list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        list(zip(X_test, y_test)), batch_size=len(X_test), shuffle=False
    )
    return train_dataloader, test_dataloader, nn.MSELoss()


def _get_sequential_data(split_percent=0.8):
    df = pd.read_csv(URL, usecols=[1], engine="python")
    data = np.array(df.values.astype("float32"))
    scaler = StandardScaler()
    data = scaler.fit_transform(data).flatten()
    n = len(data)
    # Point for splitting data into train and test
    split = int(n * split_percent)
    train_data = data[range(split)]
    test_data = data[split:]
    return train_data, test_data

def get_sequential_train_test_data(split_percent=0.8):
    train_data, test_data = _get_sequential_data(split_percent)
    X_train, y_train = train_data[:-1], train_data[1:]
    X_test, y_test = test_data[:-1], test_data[1:]
    return X_train, X_test, y_train, y_test

def get_sequential_train_test_dataloader(batch_size=1024, split_percent=0.8):
    X_train, X_test, y_train, y_test = get_sequential_train_test_data(split_percent)
    X_train = torch.tensor(X_train.astype("float32"), dtype=torch.float32)
    X_test = torch.tensor(X_test.astype("float32"), dtype=torch.float32)
    y_train = torch.tensor(y_train.astype("float32"), dtype=torch.float32)
    y_test = torch.tensor(y_test.astype("float32"), dtype=torch.float32)

    train_dataloader = DataLoader(
        list(zip(X_train, y_train)), batch_size=batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False
    )

    return train_dataloader, test_dataloader, nn.MSELoss()

def get_sequence_to_sequence_train_test_data(split_percent=0.8, sequence_length = 100, output_length = 15):
    train_data, test_data = _get_sequential_data(split_percent)
    train_window_batch = torch.tensor(train_data).unfold(0, sequence_length+output_length, 1)
    test_window_batch = torch.tensor(test_data).unfold(0, sequence_length+output_length, 1)
    X_train = [i[:sequence_length] for i in train_window_batch]
    y_train = [i[sequence_length:] for i in train_window_batch]
    X_test = [i[:sequence_length] for i in test_window_batch]
    y_test = [i[sequence_length:] for i in test_window_batch]
    return X_train, X_test, y_train, y_test

def get_sequence_to_sequence_train_test_dataloader(batch_size=1024, split_percent=0.8):
    X_train, X_test, y_train, y_test = get_sequence_to_sequence_train_test_data(split_percent)
    train_dataloader = DataLoader(
        list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False
    )
    return train_dataloader, test_dataloader, nn.MSELoss()