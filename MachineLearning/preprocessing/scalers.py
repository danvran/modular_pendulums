"""
Copyright © 2023 Daniel Vranješ
You may use, distribute and modify this code under the MIT license.
You should have received a copy of the MIT license with this file.
If not, please visit https://github.com/danvran/modular_pendulums
"""

import numpy

def get_abs_max(val1, val2):
    if val1 < 0:
        abs_val1 = val1 * -1
    else:
        abs_val1 = val1
    if val2 < 0:
        abs_val2 = val2 * -1
    else:
        abs_val2 = val2
    return max(abs_val1, abs_val2)


def scale_np_arr(arr: numpy.ndarray, dim='columns') -> numpy.ndarray:
    r"""
    Scales values per row or per column by dividing by the highest absolute value within each row or column
    :param: arr: input data
    :param: dim: rows or columns
    :
    """
    scaled_arr = arr
    max_values = []
    print(arr.shape)  # (rows, columns)
    print(arr.shape[0])
    print(arr.shape[1])
    columns = arr.shape[1]
    rows = arr.shape[0]
    #print(f"{columns} columns and {rows} rows")
    if dim == 'columns':
        for idx in range(columns):
            maxi = numpy.max(arr[:, idx])
            mini = numpy.min(arr[:, idx])
            abs_max = get_abs_max(maxi, mini)
            scaled_arr[:, idx] = arr[:, idx]/abs_max
        
    else:
        for idx in range(rows):
            maxi = numpy.max(arr[idx, :])
            mini = numpy.min(arr[idx, :])
            abs_max = get_abs_max(maxi, mini)
            scaled_arr[idx, :] = arr[idx, :]/abs_max

    return scaled_arr


def scale_np_array_rows(arr: numpy.ndarray, scaling_factors: numpy.ndarray):
    r"""
    Scales every row of a 2D numpy array based on the scaling factors provided
    
    Parameters:
    arr (numpy.ndarray): The 2D numpy array to be scaled
    scaling_factors (numpy.ndarray): A 1D numpy array of scaling factors, one for each row in arr
    
    Returns:
    numpy.ndarray: A 2D numpy array where every row is scaled based on the corresponding scaling factor
    """
    # Ensure the input arrays are of the correct shape
    assert len(arr.shape) == 2, "Input array must be 2D"
    assert len(scaling_factors.shape) == 1, "Scaling factors must be a 1D array"
    assert arr.shape[0] == scaling_factors.shape[0], "Number of scaling factors must match number of rows in input array"
    
    # Scale each row of the array based on the corresponding scaling factor
    scaled_arr = numpy.zeros_like(arr)
    for i in range(arr.shape[0]):
        scaled_arr[i,:] = arr[i,:] / scaling_factors[i]
        
    return scaled_arr

def scale_arr_per_col(arr: numpy.ndarray, scaling_factors: numpy.ndarray) -> numpy.ndarray:
    """
    Scales every column of a 2D numpy array based on the scaling factors and returns the scaled array
    
    Parameters:
    arr (numpy.ndarray): The 2D numpy array to be scaled
    scaling_factors (numpy.ndarray): The scaling factors for each column
    
    Returns:
    numpy.ndarray: The scaled array
    """
    # Ensure the input arrays are of the correct shape
    assert len(arr.shape) == 2, "Input array must be 2D"
    assert len(scaling_factors.shape) == 1, "Scaling factors must be 2D numpy array"
    assert scaling_factors.shape[0] == arr.shape[1], "Scaling factors must have the same length as the number of columns in the input array"
    # Scale each column of the input array based on the corresponding scaling factor
    scaled_arr = arr / scaling_factors
    return scaled_arr


def get_abs_max_per_row(arr: numpy.ndarray) -> numpy.ndarray:
    """
    Returns the absolute maximum value of each row of a 2D numpy array in the form of a 1D numpy array
    
    Parameters:
    arr (numpy.ndarray): The 2D numpy array
    
    Returns:
    numpy.ndarray: A 1D numpy array where each element is the absolute maximum value of the corresponding row of the input array
    """
    # Ensure the input array is of the correct shape
    assert len(arr.shape) == 2, "Input array must be 2D"
    # Compute the absolute maximum value of each row
    abs_max = numpy.max(numpy.abs(arr), axis=1)
    return abs_max


def get_abs_max_per_col(arr: numpy.ndarray) ->  numpy.ndarray:
    """
    Returns the absolute maximum value of each column of a 2D numpy array in the form of a 1D numpy array
    
    Parameters:
    arr (numpy.ndarray): The 2D numpy array
    
    Returns:
    numpy.ndarray: A 1D numpy array where each element is the absolute maximum value of the corresponding column of the input array
    """
    # Ensure the input array is of the correct shape
    assert len(arr.shape) == 2, "Input array must be 2D"
    # Compute the absolute maximum value of each column
    abs_max = numpy.max(numpy.abs(arr), axis=0)
    return abs_max