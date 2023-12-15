"""
Copyright © 2023 Daniel Vranješ
You may use, distribute and modify this code under the MIT license.
You should have received a copy of the MIT license with this file.
If not, please visit https://github.com/danvran/modular_pendulums
"""

import os
from simdata import dataloader
from preprocessing import scalers
import pandas
import torch
from torchmodels import LSTM_AE_POS
from torcheval.tools import module_summary

single = True  # True: Single Anomaly data | False: Multi Anomaly data

if single:
    print("Single Error")
    suffix = "single"
else:
    print("Multi Error")
    suffix = ""

def compute_f1_score(true_positive: int, false_positive: int, false_negative: int):
    """
    Compute the F1 score given the number of true positives, false positives, and false negatives.

    Parameters:
    - true_positive: Number of true positive instances
    - false_positive: Number of false positive instances
    - false_negative: Number of false negative instances

    Returns:
    - F1 score (float)
    """
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    # Calculate F1 score using the harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score

def compute_mcc(true_positive: int, true_negative: int, false_positive: int, false_negative: int):
    """
    Compute the Matthews correlation coefficient given the number of true positives, true negatives,
    false positives, and false negatives.

    Parameters:
    - true_positive: Number of true positive instances
    - true_negative: Number of true negative instances
    - false_positive: Number of false positive instances
    - false_negative: Number of false negative instances

    Returns:
    - Matthews correlation coefficient (float)
    """
    numerator = (true_positive * true_negative) - (false_positive * false_negative)
    denominator = ((true_positive + false_positive) * (true_positive + false_negative) *
                   (true_negative + false_positive) * (true_negative + false_negative))

    mcc = numerator / (denominator**0.5) if denominator != 0 else 0

    return mcc

files = os.listdir("./experiments")
filtered_files = [item for item in files if "txt" not in item and "pkl" not in item]
filtered_files = [item for item in filtered_files if os.path.isfile(f"./experiments/{item}")]
if 'full_eval_table.csv' in filtered_files:
    filtered_files.remove('full_eval_table.csv')
if '.DS_Store' in filtered_files:
    filtered_files.remove('.DS_Store')

configurations = [item.split("_") for item in filtered_files]

def check_if_already_evaluated(configuration, df):
    row_exists = (df['Model'] == configuration[0]) & (df['n_penduli'] == int(configuration[1])) & (df['latent_size'] == int(configuration[2]))
    row_exists = row_exists.any()
    return row_exists

if os.path.exists("./experiments/full_eval_table.csv"):
    df_existing = pandas.read_csv("./experiments/full_eval_table.csv")
    configurations_reduced = [configuration for configuration in configurations if not check_if_already_evaluated(configuration, df_existing)]
    configurations = configurations_reduced

windowsize = 100
stepsize = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# make empty df to store experiments
columns = ['Model', 'n_penduli', 'latent_size', 'trainable_parameters','sensitivity', 'specificity', 'mean', 'f1', 'tp', 'fp', 'tn', 'fn', 'f1', 'mcc', 'CV']
df = pandas.DataFrame(columns=columns)

def get_model(modeltype: str, n_penduli: int, encoding_size: int):
    if n_penduli == 2:
        if modeltype == "monolith":
            model = LSTM_AE_POS.GlobalLstmAe(enc_input_size=10, enc_hidden_size=encoding_size, dec_hidden_size=encoding_size, dec_output_size=4)
        elif modeltype == "instance modular":
            model = LSTM_AE_POS.TwoPendulumLstmAe(enc_hidden_size=encoding_size, dec_hidden_size=encoding_size)
        elif modeltype == "type modular":
            model = LSTM_AE_POS.TwoPendulumTypeLstmAe(enc_hidden_size=encoding_size, dec_hidden_size=encoding_size)
        elif modeltype == "index semantic":
            model = LSTM_AE_POS.GlobalInformedLstmAe(enc_input_size=10, enc_hidden_size=encoding_size, dec_hidden_size=encoding_size, dec_output_size=4)
        else:
            raise Exception(f"{modeltype} is not a valid option")
    if n_penduli == 4:
        if modeltype == "monolith":
            model = LSTM_AE_POS.GlobalLstmAe(enc_input_size=20, enc_hidden_size=encoding_size, dec_hidden_size=encoding_size, dec_output_size=8)
        elif modeltype == "instance modular":
            model = LSTM_AE_POS.FourPendulumLstmAe(enc_hidden_size=encoding_size, dec_hidden_size=encoding_size)
        elif modeltype == "type modular":
            model = LSTM_AE_POS.FourPendulumTypeLstmAe(enc_hidden_size=encoding_size, dec_hidden_size=encoding_size)
        elif modeltype == "index semantic":
            model = LSTM_AE_POS.GlobalInformedLstmAe(enc_input_size=20, enc_hidden_size=encoding_size, dec_hidden_size=encoding_size, dec_output_size=8)
        else:
            raise Exception(f"{modeltype} is not a valid option")
    if n_penduli == 6:
        if modeltype == "monolith":
            model = LSTM_AE_POS.GlobalLstmAe(enc_input_size=30, enc_hidden_size=encoding_size, dec_hidden_size=encoding_size, dec_output_size=12)
        elif modeltype == "instance modular":
            model = LSTM_AE_POS.SixPendulumLstmAe(enc_hidden_size=encoding_size, dec_hidden_size=encoding_size)
        elif modeltype == "type modular":
            model = LSTM_AE_POS.SixPendulumTypeLstmAe(enc_hidden_size=encoding_size, dec_hidden_size=encoding_size)
        elif modeltype == "index semantic":
            model = LSTM_AE_POS.GlobalInformedLstmAe(enc_input_size=30, enc_hidden_size=encoding_size, dec_hidden_size=encoding_size, dec_output_size=12)
        else:
            raise Exception(f"{modeltype} is not a valid option")
    if n_penduli == 8:
        if modeltype == "monolith":
            model = LSTM_AE_POS.GlobalLstmAe(enc_input_size=40, enc_hidden_size=encoding_size, dec_hidden_size=encoding_size, dec_output_size=16)
        elif modeltype == "instance modular":
            model = LSTM_AE_POS.EightPendulumLstmAe(enc_hidden_size=encoding_size, dec_hidden_size=encoding_size)
        elif modeltype == "type modular":
            model = LSTM_AE_POS.EightPendulumTypeLstmAe(enc_hidden_size=encoding_size, dec_hidden_size=encoding_size)
        elif modeltype == "index semantic":
            model = LSTM_AE_POS.GlobalInformedLstmAe(enc_input_size=40, enc_hidden_size=encoding_size, dec_hidden_size=encoding_size, dec_output_size=16)
        else:
            raise Exception(f"{modeltype} is not a valid option")
    if n_penduli == 10:
        if modeltype == "monolith":
            model = LSTM_AE_POS.GlobalLstmAe(enc_input_size=50, enc_hidden_size=encoding_size, dec_hidden_size=encoding_size, dec_output_size=20)
        elif modeltype == "instance modular":
            model = LSTM_AE_POS.TenPendulumLstmAe(enc_hidden_size=encoding_size, dec_hidden_size=encoding_size)
        elif modeltype == "type modular":
            model = LSTM_AE_POS.TenPendulumTypeLstmAe(enc_hidden_size=encoding_size, dec_hidden_size=encoding_size)
        elif modeltype == "index semantic":
            model = LSTM_AE_POS.GlobalInformedLstmAe(enc_input_size=50, enc_hidden_size=encoding_size, dec_hidden_size=encoding_size, dec_output_size=20)
        else:
            raise Exception(f"{modeltype} is not a valid option")
    return model

def count_boolean_values(lst: list[bool]) -> tuple[int, int]:
    true_count = lst.count(True)
    false_count = lst.count(False)
    return (true_count, false_count)


for idx, configuration in enumerate(configurations):
    # configuration: (modeltype, n_penduli, encoding_size)
    print(f"Working on {configuration} {idx+1}/{len(configurations)}")

    arr_normal_data = dataloader.get_normal_data(int(configuration[1]))
    anom_data = []
    anom_files = os.listdir(f"./simdata/{configuration[1]}{suffix}")
    anom_files.remove('NormalOperation.csv')
    anom_files.sort()

    for file in anom_files:
        anom_data.append(dataloader.get_anomaly_data(file, configuration[1], suffix=suffix))
        print(f"File: {file}")

    test_sample_length = 100000  # to match the length of anomalous test data
    arr_normal_data_train = arr_normal_data[test_sample_length:, :]  # only used to get the scaling factors
    arr_normal_data_test = arr_normal_data[:test_sample_length, :]  # first split is for testing to be the same as the anomalous data
    scalers_np = scalers.get_abs_max_per_col(arr_normal_data_train)
    test_ds_list = [arr_normal_data_test]
    for file in anom_data:
        test_ds_list.append(file)

    # scale and slice data
    for i in range(len(test_ds_list)):
        test_ds_list[i] = scalers.scale_arr_per_col(test_ds_list[i], scalers_np)
        test_ds_list[i] = dataloader.time_series_sliding_window_generator(test_ds_list[i], windowsize, stepsize)
        # print(f"Max value in test_ds_[{i}]: {numpy.max(test_ds_list[i])}")
    ds_OK = dataloader.TorchDataset(test_ds_list[0], device)

    model = get_model(configuration[0], int(configuration[1]), int(configuration[2]))
    thresholds = f"./experiments/{configuration[0]}_{configuration[1]}_{configuration[2]}_{configuration[3]}.pkl"
    model.load_thresholds(thresholds)
    state_dict = f"./experiments/{configuration[0]}_{configuration[1]}_{configuration[2]}_{configuration[3]}"
    model.load_state_dict(torch.load(state_dict))
    model.eval()
    
    model_summary = module_summary.get_module_summary(model)
    trainable_params = int(model_summary.num_trainable_parameters)

    with torch.no_grad():
        normal_tests = []
        for entry in ds_OK:
            prediction, indices = model.detect_anomaly(entry)
            normal_tests.append(prediction)
        false_positive, true_negative = count_boolean_values(normal_tests)
        if (false_positive+true_negative) != 0:
            specificity = true_negative/(false_positive+true_negative) # Specificity = TrueNegative / (FalsePositive + TrueNegative)
        else:
            specificity = -1
        anomalous_tests = []
        for idx, data_set in enumerate(test_ds_list[1:]):
            ds_KO = dataloader.TorchDataset(data_set, device)
            for entry in ds_KO:
                prediction, indices = model.detect_anomaly(entry)
                anomalous_tests.append(prediction)
        true_positive, false_negative = count_boolean_values(anomalous_tests)
        if (false_negative+true_positive) != 0:
            sensitivity = true_positive/(false_negative+true_positive)  # Sensitivity = TruePositive / (TruePositive + FalseNegative)
        else:
            sensitivity = -1
        # Some analysis
        adjusted_correct_ratio_mean = (specificity + sensitivity)/2
        columns = ['Model', 'n_penduli', 'latent_size', 'trainable_parameters','sensitivity', 'specificity', 'mean', 'f1', 'tp', 'fp', 'tn', 'fn', 'f1', 'mcc', 'CV']
        new_row = pandas.Series({'Model': configuration[0],
                                 'n_penduli': int(configuration[1]),
                                 'latent_size': int(configuration[2]),
                                 'trainable_parameters': trainable_params,
                                 'sensitivity': sensitivity,
                                 'specificity': specificity,
                                 'mean': (sensitivity+specificity)/2,
                                 'tp': true_positive,
                                 'fp': false_positive,
                                 'tn': true_negative,
                                 'fn': false_negative,
                                 'f1': compute_f1_score(true_positive, false_positive, false_negative),
                                 'mcc': compute_mcc(true_positive, true_negative, false_positive, false_negative),
                                 'CV': int(configuration[3])}, index=columns)
        df = pandas.concat([df, new_row.to_frame().T], ignore_index=True)

if not df.empty:
    table_path = f"./experiments/full_eval_table.csv"
    df = df.sort_values(by=['Model', 'n_penduli', 'latent_size', 'CV'], ascending=[True, True, True, True])
    df.to_csv(table_path, index=False)
else:
    print("Empty data frame. Nothing saved")