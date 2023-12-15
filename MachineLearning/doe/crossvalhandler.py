"""
Copyright © 2023 Daniel Vranješ
You may use, distribute and modify this code under the MIT license.
You should have received a copy of the MIT license with this file.
If not, please visit https://github.com/danvran/modular_pendulums
"""

import numpy

def get_cross_val_sets(in_data: numpy.ndarray, data_percentage: int=100, one_cv: bool=False) -> list[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]:
    r"""
    description: make train, val, test splits of a numpy array
    param: data: multidimensional numpy array where the highest dimension represents a sample
    paramm: data_percentage: choose how much percent of the train data to use for training
    return: cross val sets: list where each entry contains the data for one cross validation
    """
    if data_percentage > 100:
        raise ValueError("data_percentage must be between 1 and 100")
    data = in_data.copy()
    rng = numpy.random.default_rng(seed=42)
    rng.shuffle(data)  # in place shuffle along 0th axis
    cross_val_sets_list = []
    n_cross_validations = 5
    if one_cv:
        n_cross_validations = 1
    n_splits = n_cross_validations + 2
    modulo_cross_rest = data.shape[0] % n_splits
    if modulo_cross_rest != 0:
        data = data[:-modulo_cross_rest]  # remove some data to precisely fit number of cross-vals
    cross_val_sets = numpy.split(data, n_splits)  # returns a list of numpy arrays with equal sizes
    for cross_val_index in range(n_cross_validations):
        test_data = cross_val_sets[cross_val_index]
        val_data = cross_val_sets[cross_val_index + 1]
        train_data = cross_val_sets[:cross_val_index] + cross_val_sets[cross_val_index + 1 + 1:]
        train_data = numpy.concatenate(train_data) # convert list to numpy array
        #rng.shuffle(train_data)  # in place shuffle along 0th axis
        if data_percentage < 100:
            train_data = train_data[:int(round(train_data.shape[0]*data_percentage/100))]  # take a portion of the data
        cross_val_sets_list.append(tuple((train_data, val_data, test_data)))
    return cross_val_sets_list


def get_3_cross_val_sets(in_data: numpy.ndarray) -> list[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]:
    r"""
    description: make train, val, test splits of a numpy array
    param: data: multidimensional numpy array where the highest dimension represents a sample
    paramm: data_percentage: choose how much percent of the train data to use for training
    return: cross val sets: list where each entry contains the data for one cross validation
    """
    data = in_data.copy()
    rng = numpy.random.default_rng(seed=42)
    rng.shuffle(data)  # in place shuffle along 0th axis
    cross_val_sets_list = []
    n_splits = 3
    modulo_cross_rest = data.shape[0] % n_splits
    if modulo_cross_rest != 0:
        data = data[:-modulo_cross_rest]  # remove some data to precisely fit number of cross-vals
    cross_val_sets = numpy.split(data, n_splits)  # returns a list of numpy arrays with equal sizes
    
    test_data = cross_val_sets[0]
    val_data = cross_val_sets[1]
    train_data = cross_val_sets[2]
    cross_val_sets_list.append(tuple((train_data, val_data, test_data)))

    test_data = cross_val_sets[1]
    val_data = cross_val_sets[2]
    train_data = cross_val_sets[0]
    cross_val_sets_list.append(tuple((train_data, val_data, test_data)))

    test_data = cross_val_sets[2]
    val_data = cross_val_sets[0]
    train_data = cross_val_sets[1]
    cross_val_sets_list.append(tuple((train_data, val_data, test_data)))

    return cross_val_sets_list