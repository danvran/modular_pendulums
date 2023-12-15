import pandas
import numpy
import torch
import os

print(os.getcwd())

def get_data_names() -> list[str]:
    df = pandas.read_csv("./simdata/NormalOperation.csv", sep=",")
    df = df.loc[:, df.any()]  # drop all columns containing only 0 values
    names = list(df.columns.values)
    return names

def get_normal_data(n_penduli: int, reduced = False) -> numpy.ndarray:
    filepath = f"./simdata/{n_penduli}/NormalOperation.csv"
    df = pandas.read_csv(filepath, sep=",")
    df = df.loc[:, df.any()]  # drop all columns containing only 0 values
    arr = df.to_numpy()
    if reduced:
        return arr[0:1000, :]
    else:
        return arr

def get_anomaly_data(filename: str, n_penduli: str, suffix="", reduced = False) -> numpy.ndarray:
    filepath = os.path.join(f"./simdata/{n_penduli}{suffix}/", filename)
    df = pandas.read_csv(filepath, sep=",")
    df = df.loc[:, df.any()]  # drop all columns containing only 0 values
    arr = df.to_numpy()
    if reduced:
        return arr[0:500, :]
    else:
        return arr

def time_series_sliding_window_generator(arr: numpy.ndarray, window: int, step: int) -> numpy.ndarray:
    r"""
    Generate time window slices of a numpy ndarray based on windowsize and step values and return a new numpy ndarray.
    The output array will have one more dimension than the input array. This dimension represents the samples.
    """
    if not type(arr) is numpy.ndarray:
        dtype = str(type(arr))
        dtype = dtype.replace('<' , "")
        dtype = dtype.replace('>' , "")
        message = F"expected arr to be of type 'numpy.ndarray' but got {dtype} instead"
        raise TypeError(message)
    if not type(window) is int:
        raise TypeError("windowsize must be of type int")
    if window < 1:
        raise ValueError("windowsize must be 1 or greater")
    if not type(step) is int:
        raise TypeError("step must be of type int")
    if step < 1:
        raise ValueError("step must be 1 or greater")
    
    time_steps = arr.shape[0]
    idx = 0
    buffer = []
    while idx+window < time_steps:
        buffer.append(arr[idx:idx+window])
        idx += step
    return numpy.asarray(buffer)


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, np_data: numpy.ndarray, device: str):
        self.torch_data = torch.tensor(np_data, dtype=torch.float32).to(device)

    def __len__(self):
        return self.torch_data.shape[0]

    def __getitem__(self, idx):
        return self.torch_data[idx]


def create_dataloaders(train_ds: TorchDataset, val_ds: TorchDataset, test_ds: TorchDataset, batch_size: int):
    """
    Build train, validation and tests dataloader using the toy data
    :return: Train, validation and test data loaders
    """
    print(f'Datasets shapes: Train={train_ds.shape}; Validation={val_ds.shape}; Test={test_ds.shape}')
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
