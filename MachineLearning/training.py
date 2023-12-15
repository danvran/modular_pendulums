"""
Copyright © 2023 Daniel Vranješ
You may use, distribute and modify this code under the MIT license.
You should have received a copy of the MIT license with this file.
If not, please visit https://github.com/danvran/modular_pendulums
"""

import os
import torch
from simdata import dataloader
from doe import crossvalhandler
from torchmodels import LSTM_AE_POS
from preprocessing import scalers
from torcheval.tools import module_summary

modeltype = "monolith" # monolith, index semantic, type modular, instance modular
pendulums = [2, 4, 6, 8, 10]

experimental = False
reduction = False
windowsize = 100
stepsize = 50
epochs = 20

def main(modeltype: str, n_penduli: int, encoding_size: int):  
    # Run grid search / cross validations
    for cv_idx, cross_val_data in enumerate(lst_cross_val_data):
        print(f"Starting Cross Validation #{cv_idx}")
        exp_path = os.path.join(home, f"./experiments")
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)

        # set train val test data
        train_ds, val_ds, test_ds = cross_val_data
        train_ds = dataloader.TorchDataset(train_ds, device)
        val_ds = dataloader.TorchDataset(val_ds, device)
        test_ds = dataloader.TorchDataset(test_ds, device)
        # train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)

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

        
        history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs = model.train_model(train_ds, val_ds, test_ds, epochs)
        modelpath = os.path.join(exp_path, f"{modeltype}_{n_penduli}_{encoding_size}_{cv_idx}")
        torch.save(model.state_dict(), modelpath)

        model_summary = module_summary.get_module_summary(model)
        summary_path = os.path.join(exp_path, f"{modeltype}_{n_penduli}_{encoding_size}_{cv_idx}_summary.txt")
        with open(summary_path, 'w') as w:
            w.write(str(model_summary))
            w.write(f"Best Epoch: {best_epoch}\n")
        
        threshold_path = os.path.join(exp_path, f"{modeltype}_{n_penduli}_{encoding_size}_{cv_idx}.pkl")
        model.export_thresholds(threshold_path)

        if experimental:
            break  # run only one cross val during development


for n_penduli in pendulums:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    home = os.getcwd()
    
    # Get Data
    arr_normal_data = dataloader.get_normal_data(n_penduli)

    test_sample_length = 100000  # fixed for all files

    # Split Normal into Train and Test Data
    arr_normal_data_train = arr_normal_data[test_sample_length:, :]  # second split is for training
    scalers_np = scalers.get_abs_max_per_col(arr_normal_data_train)  # Save max values from normal sensor data as scaling factors for all data
    
    if reduction:
        arr_normal_data_train = arr_normal_data_train[:10000, :]
    
    print("Loaded Data")

    # Scale to range [-1, 1] and make time series slices from continuous data
    arr_normal_data_train = scalers.scale_arr_per_col(arr_normal_data_train, scalers_np)
    arr_normal_data_windowed_train = dataloader.time_series_sliding_window_generator(arr_normal_data_train, windowsize, stepsize)

    print("Preprocessed Data")

    # Make cross validation sets and store them in a list of multiple (train, val, test)-tuples
    lst_cross_val_data = crossvalhandler.get_3_cross_val_sets(arr_normal_data_windowed_train)
    print("Made CV Sets")

    if modeltype == "modular":
        sizes = [50, 100, 150]
    else:
        sizes = [200, 300, 500, 800]
    
    for size in sizes:
        print(f"Working on size {size}")
        main(modeltype = modeltype, n_penduli = n_penduli , encoding_size = size)

print("Finished Program")
