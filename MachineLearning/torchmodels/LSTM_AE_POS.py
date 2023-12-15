"""
Copyright © 2023 Daniel Vranješ
You may use, distribute and modify this code under the MIT license.
You should have received a copy of the MIT license with this file.
If not, please visit https://github.com/danvran/modular_pendulums
"""

import numpy
import torch
from torch.utils.tensorboard import SummaryWriter
import copy
import pickle

# Encoder Class
class Encoder(torch.nn.Module):
    r"""
    Docstring
    """
    def __init__(self, input_size, hidden_size, dropout_ratio=0):
        super(Encoder, self).__init__()
        self.lstm_enc = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout_ratio, batch_first=True)

    def forward(self, x):
        out, (last_h_state, last_c_state) = self.lstm_enc(x)
        return out

# Decoder Class
class Decoder(torch.nn.Module):
    r"""
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_ratio=0):
        super(Decoder, self).__init__()
        self.lstm_dec = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout_ratio, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, z):
        out, (last_h_state, last_c_state) = self.lstm_dec(z)
        out = self.fc(out)
        return out

# LSTM Auto-Encoder Class
class LstmAe(torch.nn.Module):
    r"""Encoder-Decoder LSTM Network
    In -> LSTM Encoder -> Hidden States -> LSTM Decoder -> Hidden States -> Fully Connected Layer -> Out
    """
    def __init__(self, enc_input_size: int, enc_hidden_size: int, dec_hidden_size: int, dec_output_size: int):
        """ init function
        Args:
            enc_input_size (int): Number of input features
            enc_hidden_size (int): Number of hidden encoder features
            dec_hidden_size (int): Number of hidden decoder features
        """
        super(LstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(input_size=enc_input_size, hidden_size=enc_hidden_size)
        self.decoder = Decoder(input_size=enc_hidden_size, hidden_size=dec_hidden_size, output_size=dec_output_size)

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_dec

class GlobalLstmAe(torch.nn.Module):
    r"""Encoder-Decoder LSTM Network
    In -> LSTM Encoder -> Hidden States -> LSTM Decoder -> Hidden States -> Fully Connected Layer -> Out
    """
    def __init__(self, enc_input_size: int, enc_hidden_size: int, dec_hidden_size: int, dec_output_size: int):
        """ init function
        Args:
            enc_input_size (int): Number of input features
            enc_hidden_size (int): Number of hidden encoder features
            dec_hidden_size (int): Number of hidden decoder features

        """
        super(GlobalLstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(input_size=enc_input_size, hidden_size=enc_hidden_size)
        self.decoder = Decoder(input_size=enc_hidden_size, hidden_size=dec_hidden_size, output_size=dec_output_size)
        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        self.threshold = 0.0
        self.summary_name = f"experiments/monolith_{int(enc_input_size/5)}_{enc_hidden_size}_tensorboard"
        self.POSidxs = []
        if enc_input_size == 10:
            self.POSidxs = [0, 1, 5, 6]
        if enc_input_size == 20:
            self.POSidxs = [0, 1, 5, 6, 10, 11, 15, 16]
        if enc_input_size == 30:
            self.POSidxs = [0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25, 26]
        if enc_input_size == 40:
            self.POSidxs = [0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25, 26,
                            30, 31, 35, 36]
        if enc_input_size == 50:
            self.POSidxs = [0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25, 26,
                            30, 31, 35, 36, 40, 41, 45, 46]
        if enc_input_size == 100:
            self.POSidxs = [0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25, 26, 30, 31,
                            35, 36, 40, 41, 45, 46, 50, 51, 55, 56, 60, 61,
                            65, 66, 70, 71, 75, 76, 80, 81, 85, 86, 90, 91,
                            95, 96]

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_dec
        
    def train_model(self, train_ds, val_ds, test_ds, n_epochs: int):
        """
        train_ds: 
        val_ds: 
        test_ds:
        n_epochs: int value
        """
        writer = SummaryWriter(self.summary_name)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        history = dict(train=[], val=[])  # save train and validation losses
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 10000.0  # arbitrary high loss
        best_epoch = -1
        for epoch in range(1, n_epochs + 1):
            #print(f"Epoch {epoch}")
            self = self.train()
            train_losses = []
            for batch_idx, data in enumerate(train_ds):
                optimizer.zero_grad()
                prediction = self(data)
                loss = criterion(prediction, data[:, self.POSidxs])
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            val_losses = []
            self = self.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_ds):
                    prediction = self(data)
                    loss = criterion(prediction, data[:, self.POSidxs])
                    val_losses.append(loss.item())
            mean_train_loss = numpy.mean(train_losses)
            mean_val_loss = numpy.mean(val_losses)
            history['train'].append(mean_train_loss)
            writer.add_scalar('train_loss', mean_train_loss, epoch)
            history['val'].append(mean_val_loss)
            writer.add_scalar('val_loss', mean_val_loss, epoch)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.load_state_dict(best_model_wts)
        self.eval()
        test_losses = []
        test_pairs = []
        with torch.no_grad():
            for batch_idx, data in enumerate(test_ds):
                prediction = self(data)
                # command data is removed since it is not to be learned
                loss = criterion(prediction, data[:, self.POSidxs])
                # loss = criterion(prediction, data[self.input_size:])
                test_losses.append(loss.item())
                writer.add_scalar('test_loss', loss.item(), batch_idx)
                test_pairs.append((data.numpy(force=True), prediction.numpy(force=True)))
        writer.close()
        mean_test_loss = numpy.mean(test_losses)
        std_test_loss = numpy.std(test_losses)
        self.threshold = 2.33 * std_test_loss + mean_test_loss
        return history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs
    
    def inference_mse(self, x) -> tuple[torch.FloatTensor | torch.cuda.FloatTensor, float]:
        x_hat = self(x)
        reconstruction_error = self.criterion(x_hat, x[:, self.POSidxs])
        return x_hat, reconstruction_error.item()
    
    def detect_anomaly(self, x) -> tuple[bool, list[int]]:
        x_hat, mse = self.inference_mse(x)
        if mse > self.threshold:
            return (True, [1])  # only one possible idx for anomaly
        else:
            return (False, [])  # no idx of no anomaly
        
    def get_thresholds(self):
        return [("Gloabal Threshold", self.threshold)]
    
    def export_thresholds(self, filepath):
        """
        
        """
        thresholds = self.get_thresholds()
        with open(filepath, 'wb') as file:
            # A new file will be created
            pickle.dump(thresholds, file)
    
    def load_thresholds(self, filepath):
        """
        
        """
        with open(filepath, 'rb') as file: 
            # Call load method to deserialze
            thresholds = pickle.load(file)
            self.threshold = thresholds[0][1]


class GlobalInformedLstmAe(torch.nn.Module):
    r"""Encoder-Decoder LSTM Network
    In -> LSTM Encoder -> Hidden States -> LSTM Decoder -> Hidden States -> Fully Connected Layer -> Out
    """
    def __init__(self, enc_input_size: int, enc_hidden_size: int, dec_hidden_size: int, dec_output_size: int):
        """ init function
        Args:
            enc_input_size (int): Number of input features
            enc_hidden_size (int): Number of hidden encoder features
            dec_hidden_size (int): Number of hidden decoder features

        """
        super(GlobalInformedLstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(input_size=enc_input_size, hidden_size=enc_hidden_size)
        self.decoder = Decoder(input_size=enc_hidden_size, hidden_size=dec_hidden_size, output_size=dec_output_size)
        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        self.threshold = 0.0
        self.summary_name = f"experiments/index_semantic_{int(enc_input_size/5)}_{enc_hidden_size}_tensorboard"
        self.POSidxs = []
        self.LOSSidxs = []
        self.thresholds = []
        self.n_pendulums = 0

        self.threshold_idxs = []  # here we store the threshold values for anom detect after training

        if enc_input_size == 10:
            self.POSidxs = [0, 1, 5, 6]
            self.LOSSidxs = [[0, 1], [2, 3]]

            self.thresholds = [0.0, 0.0]
            self.n_pendulums = 2
            self.threshP1 = 0.0
            self.threshP2 = 0.0
        if enc_input_size == 20:
            self.POSidxs = [0, 1, 5, 6, 10, 11, 15, 16]
            self.LOSSidxs = [[0, 1], [2, 3], [4, 5], [6, 7]]
            self.thresholds = [0.0, 0.0, 0.0, 0.0]
            self.n_pendulums = 4
            self.threshP1 = 0.0
            self.threshP2 = 0.0
            self.threshP3 = 0.0
            self.threshP4 = 0.0
        if enc_input_size == 30:
            self.POSidxs = [0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25, 26]
            self.LOSSidxs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]
            self.thresholds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.n_pendulums = 6
            self.threshP1 = 0.0
            self.threshP2 = 0.0
            self.threshP3 = 0.0
            self.threshP4 = 0.0
            self.threshP5 = 0.0
            self.threshP6 = 0.0
        if enc_input_size == 40:
            self.POSidxs = [0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25, 26,
                            30, 31, 35, 36]
            self.LOSSidxs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13],
                            [14, 15]]
            self.thresholds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.n_pendulums = 8
            self.threshP1 = 0.0
            self.threshP2 = 0.0
            self.threshP3 = 0.0
            self.threshP4 = 0.0
            self.threshP5 = 0.0
            self.threshP6 = 0.0
            self.threshP7 = 0.0
            self.threshP8 = 0.0
        if enc_input_size == 50:
            self.POSidxs = [0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25, 26,
                            30, 31, 35, 36, 40, 41, 45, 46]
            self.LOSSidxs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13],
                            [14, 15], [16, 17], [18, 19]]
            self.thresholds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.n_pendulums = 10
            self.threshP1 = 0.0
            self.threshP2 = 0.0
            self.threshP3 = 0.0
            self.threshP4 = 0.0
            self.threshP5 = 0.0
            self.threshP6 = 0.0
            self.threshP7 = 0.0
            self.threshP8 = 0.0
            self.threshP9 = 0.0
            self.threshP10 = 0.0

            

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        if self.n_pendulums == 2:
            p1 = x_dec[:, self.LOSSidxs[0]]
            p2 = x_dec[:, self.LOSSidxs[1]]
            return (p1, p2)
        if self.n_pendulums == 4:
            p1 = x_dec[:, self.LOSSidxs[0]]
            p2 = x_dec[:, self.LOSSidxs[1]]
            p3 = x_dec[:, self.LOSSidxs[2]]
            p4 = x_dec[:, self.LOSSidxs[3]]
            return (p1, p2, p3, p4)
        if self.n_pendulums == 6:
            p1 = x_dec[:, self.LOSSidxs[0]]
            p2 = x_dec[:, self.LOSSidxs[1]]
            p3 = x_dec[:, self.LOSSidxs[2]]
            p4 = x_dec[:, self.LOSSidxs[3]]
            p5 = x_dec[:, self.LOSSidxs[4]]
            p6 = x_dec[:, self.LOSSidxs[5]]
            return (p1, p2, p3, p4, p5, p6)
        if self.n_pendulums == 8:
            p1 = x_dec[:, self.LOSSidxs[0]]
            p2 = x_dec[:, self.LOSSidxs[1]]
            p3 = x_dec[:, self.LOSSidxs[2]]
            p4 = x_dec[:, self.LOSSidxs[3]]
            p5 = x_dec[:, self.LOSSidxs[4]]
            p6 = x_dec[:, self.LOSSidxs[5]]
            p7 = x_dec[:, self.LOSSidxs[6]]
            p8 = x_dec[:, self.LOSSidxs[7]]
            return (p1, p2, p3, p4, p5, p6, p7, p8)
        if self.n_pendulums == 10:
            p1 = x_dec[:, self.LOSSidxs[0]]
            p2 = x_dec[:, self.LOSSidxs[1]]
            p3 = x_dec[:, self.LOSSidxs[2]]
            p4 = x_dec[:, self.LOSSidxs[3]]
            p5 = x_dec[:, self.LOSSidxs[4]]
            p6 = x_dec[:, self.LOSSidxs[5]]
            p7 = x_dec[:, self.LOSSidxs[6]]
            p8 = x_dec[:, self.LOSSidxs[7]]
            p9 = x_dec[:, self.LOSSidxs[8]]
            p10 = x_dec[:, self.LOSSidxs[9]]
            return (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)
    
        
    def train_model(self, train_ds, val_ds, test_ds, n_epochs: int):
        """
        train_ds: 
        val_ds: 
        test_ds:
        n_epochs: int value
        """
        writer = SummaryWriter(self.summary_name)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        history = dict(train=[], val=[])  # save train and validation losses
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 10000.0  # arbitrary high loss
        best_epoch = -1
        for epoch in range(1, n_epochs + 1):
            #print(f"Epoch {epoch}")
            self = self.train()
            train_losses = []
            for batch_idx, data in enumerate(train_ds):
                optimizer.zero_grad()
                predictions = self(data)
                temp_train_losses = []
                for idx, _ in enumerate(predictions):
                    temp_train_losses.append(criterion(predictions[idx], data[:, self.LOSSidxs[idx]]))
                loss = sum(temp_train_losses)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            val_losses = []
            self = self.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_ds):
                    predictions = self(data)
                    temp_val_losses = []
                    for idx, _ in enumerate(predictions):
                        temp_val_losses.append(criterion(predictions[idx], data[:, self.LOSSidxs[idx]]))
                    loss = sum(temp_val_losses)
                    val_losses.append(loss.item())
            mean_train_loss = numpy.mean(train_losses)
            mean_val_loss = numpy.mean(val_losses)
            history['train'].append(mean_train_loss)
            writer.add_scalar('train_loss', mean_train_loss, epoch)
            history['val'].append(mean_val_loss)
            writer.add_scalar('val_loss', mean_val_loss, epoch)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.load_state_dict(best_model_wts)
        self.eval()
        # Testing
        test_losses = []
        test_pairs = []
        test_losses_P1 = []
        test_losses_P2 = []
        test_losses_P3 = []
        test_losses_P4 = []
        test_losses_P5 = []
        test_losses_P6 = []
        test_losses_P7 = []
        test_losses_P8 = []
        test_losses_P9 = []
        test_losses_P10 = []

        with torch.no_grad():
            for batch_idx, data in enumerate(test_ds):
                predictions = self(data)
                # command data is removed since it is not to be learned
                temp_test_losses = []
                for idx, _ in enumerate(predictions):
                    temp_test_losses.append(criterion(predictions[idx], data[:, self.LOSSidxs[idx]]))
                
                if self.n_pendulums == 2:
                    test_losses_P1.append(temp_test_losses[0])
                    test_losses_P2.append(temp_test_losses[1])
                if self.n_pendulums == 4:
                    test_losses_P1.append(temp_test_losses[0])
                    test_losses_P2.append(temp_test_losses[1])
                    test_losses_P3.append(temp_test_losses[2])
                    test_losses_P4.append(temp_test_losses[3])
                if self.n_pendulums == 6:
                    test_losses_P1.append(temp_test_losses[0])
                    test_losses_P2.append(temp_test_losses[1])
                    test_losses_P3.append(temp_test_losses[2])
                    test_losses_P4.append(temp_test_losses[3])
                    test_losses_P5.append(temp_test_losses[4])
                    test_losses_P6.append(temp_test_losses[5])
                if self.n_pendulums == 8:
                    test_losses_P1.append(temp_test_losses[0])
                    test_losses_P2.append(temp_test_losses[1])
                    test_losses_P3.append(temp_test_losses[2])
                    test_losses_P4.append(temp_test_losses[3])
                    test_losses_P5.append(temp_test_losses[4])
                    test_losses_P6.append(temp_test_losses[5])
                    test_losses_P7.append(temp_test_losses[6])
                    test_losses_P8.append(temp_test_losses[7])
                if self.n_pendulums == 10:
                    test_losses_P1.append(temp_test_losses[0])
                    test_losses_P2.append(temp_test_losses[1])
                    test_losses_P3.append(temp_test_losses[2])
                    test_losses_P4.append(temp_test_losses[3])
                    test_losses_P5.append(temp_test_losses[4])
                    test_losses_P6.append(temp_test_losses[5])
                    test_losses_P7.append(temp_test_losses[6])
                    test_losses_P8.append(temp_test_losses[7])
                    test_losses_P9.append(temp_test_losses[8])
                    test_losses_P10.append(temp_test_losses[9])

                loss = sum(temp_test_losses)
                # loss = criterion(prediction, data[self.input_size:])
                test_losses.append(loss.item())
                writer.add_scalar('test_loss', loss.item(), batch_idx)
    
        writer.close()

        mean_test_loss = numpy.mean(test_losses)
        std_test_loss = numpy.std(test_losses)

        if self.n_pendulums == 2:
            mean_test_loss_P1 = numpy.mean(test_losses_P1)
            std_test_loss_P1 = numpy.std(test_losses_P1)
            self.threshP1 = 2.33 * std_test_loss_P1 + mean_test_loss_P1
            self.threshold_idxs.append(self.threshP1)
            mean_test_loss_P2 = numpy.mean(test_losses_P2)
            std_test_loss_P2 = numpy.std(test_losses_P2)
            self.threshP2 = 2.33 * std_test_loss_P2 + mean_test_loss_P2
            self.threshold_idxs.append(self.threshP2)
        if self.n_pendulums == 4:
            mean_test_loss_P1 = numpy.mean(test_losses_P1)
            std_test_loss_P1 = numpy.std(test_losses_P1)
            self.threshP1 = 2.33 * std_test_loss_P1 + mean_test_loss_P1
            self.threshold_idxs.append(self.threshP1)
            mean_test_loss_P2 = numpy.mean(test_losses_P2)
            std_test_loss_P2 = numpy.std(test_losses_P2)
            self.threshP2 = 2.33 * std_test_loss_P2 + mean_test_loss_P2
            self.threshold_idxs.append(self.threshP2)

            mean_test_loss_P3 = numpy.mean(test_losses_P3)
            std_test_loss_P3 = numpy.std(test_losses_P3)
            self.threshP3 = 2.33 * std_test_loss_P3 + mean_test_loss_P3
            self.threshold_idxs.append(self.threshP3)
            mean_test_loss_P4 = numpy.mean(test_losses_P4)
            std_test_loss_P4 = numpy.std(test_losses_P4)
            self.threshP4 = 2.33 * std_test_loss_P4 + mean_test_loss_P4
            self.threshold_idxs.append(self.threshP4)
        if self.n_pendulums == 6:
            mean_test_loss_P1 = numpy.mean(test_losses_P1)
            std_test_loss_P1 = numpy.std(test_losses_P1)
            self.threshP1 = 2.33 * std_test_loss_P1 + mean_test_loss_P1
            self.threshold_idxs.append(self.threshP1)
            mean_test_loss_P2 = numpy.mean(test_losses_P2)
            std_test_loss_P2 = numpy.std(test_losses_P2)
            self.threshP2 = 2.33 * std_test_loss_P2 + mean_test_loss_P2
            self.threshold_idxs.append(self.threshP2)

            mean_test_loss_P3 = numpy.mean(test_losses_P3)
            std_test_loss_P3 = numpy.std(test_losses_P3)
            self.threshP3 = 2.33 * std_test_loss_P3 + mean_test_loss_P3
            self.threshold_idxs.append(self.threshP3)
            mean_test_loss_P4 = numpy.mean(test_losses_P4)
            std_test_loss_P4 = numpy.std(test_losses_P4)
            self.threshP4 = 2.33 * std_test_loss_P4 + mean_test_loss_P4
            self.threshold_idxs.append(self.threshP4)

            mean_test_loss_P5 = numpy.mean(test_losses_P5)
            std_test_loss_P5 = numpy.std(test_losses_P5)
            self.threshP5 = 2.33 * std_test_loss_P5 + mean_test_loss_P5
            self.threshold_idxs.append(self.threshP5)
            mean_test_loss_P6 = numpy.mean(test_losses_P6)
            std_test_loss_P6 = numpy.std(test_losses_P6)
            self.threshP6 = 2.33 * std_test_loss_P6 + mean_test_loss_P6
            self.threshold_idxs.append(self.threshP6)

        if self.n_pendulums == 8:
            mean_test_loss_P1 = numpy.mean(test_losses_P1)
            std_test_loss_P1 = numpy.std(test_losses_P1)
            self.threshP1 = 2.33 * std_test_loss_P1 + mean_test_loss_P1
            self.threshold_idxs.append(self.threshP1)
            mean_test_loss_P2 = numpy.mean(test_losses_P2)
            std_test_loss_P2 = numpy.std(test_losses_P2)
            self.threshP2 = 2.33 * std_test_loss_P2 + mean_test_loss_P2
            self.threshold_idxs.append(self.threshP2)

            mean_test_loss_P3 = numpy.mean(test_losses_P3)
            std_test_loss_P3 = numpy.std(test_losses_P3)
            self.threshP3 = 2.33 * std_test_loss_P3 + mean_test_loss_P3
            self.threshold_idxs.append(self.threshP3)
            mean_test_loss_P4 = numpy.mean(test_losses_P4)
            std_test_loss_P4 = numpy.std(test_losses_P4)
            self.threshP4 = 2.33 * std_test_loss_P4 + mean_test_loss_P4
            self.threshold_idxs.append(self.threshP4)

            mean_test_loss_P5 = numpy.mean(test_losses_P5)
            std_test_loss_P5 = numpy.std(test_losses_P5)
            self.threshP5 = 2.33 * std_test_loss_P5 + mean_test_loss_P5
            self.threshold_idxs.append(self.threshP5)
            mean_test_loss_P6 = numpy.mean(test_losses_P6)
            std_test_loss_P6 = numpy.std(test_losses_P6)
            self.threshP6 = 2.33 * std_test_loss_P6 + mean_test_loss_P6
            self.threshold_idxs.append(self.threshP6)

            mean_test_loss_P7 = numpy.mean(test_losses_P7)
            std_test_loss_P7 = numpy.std(test_losses_P7)
            self.threshP7 = 2.33 * std_test_loss_P7 + mean_test_loss_P7
            self.threshold_idxs.append(self.threshP7)
            mean_test_loss_P8 = numpy.mean(test_losses_P8)
            std_test_loss_P8 = numpy.std(test_losses_P8)
            self.threshP8 = 2.33 * std_test_loss_P8 + mean_test_loss_P8
            self.threshold_idxs.append(self.threshP8)

        if self.n_pendulums == 10:
            mean_test_loss_P1 = numpy.mean(test_losses_P1)
            std_test_loss_P1 = numpy.std(test_losses_P1)
            self.threshP1 = 2.33 * std_test_loss_P1 + mean_test_loss_P1
            self.threshold_idxs.append(self.threshP1)
            mean_test_loss_P2 = numpy.mean(test_losses_P2)
            std_test_loss_P2 = numpy.std(test_losses_P2)
            self.threshP2 = 2.33 * std_test_loss_P2 + mean_test_loss_P2
            self.threshold_idxs.append(self.threshP2)

            mean_test_loss_P3 = numpy.mean(test_losses_P3)
            std_test_loss_P3 = numpy.std(test_losses_P3)
            self.threshP3 = 2.33 * std_test_loss_P3 + mean_test_loss_P3
            self.threshold_idxs.append(self.threshP3)
            mean_test_loss_P4 = numpy.mean(test_losses_P4)
            std_test_loss_P4 = numpy.std(test_losses_P4)
            self.threshP4 = 2.33 * std_test_loss_P4 + mean_test_loss_P4
            self.threshold_idxs.append(self.threshP4)

            mean_test_loss_P5 = numpy.mean(test_losses_P5)
            std_test_loss_P5 = numpy.std(test_losses_P5)
            self.threshP5 = 2.33 * std_test_loss_P5 + mean_test_loss_P5
            self.threshold_idxs.append(self.threshP5)
            mean_test_loss_P6 = numpy.mean(test_losses_P6)
            std_test_loss_P6 = numpy.std(test_losses_P6)
            self.threshP6 = 2.33 * std_test_loss_P6 + mean_test_loss_P6
            self.threshold_idxs.append(self.threshP6)

            mean_test_loss_P7 = numpy.mean(test_losses_P7)
            std_test_loss_P7 = numpy.std(test_losses_P7)
            self.threshP7 = 2.33 * std_test_loss_P7 + mean_test_loss_P7
            self.threshold_idxs.append(self.threshP7)
            mean_test_loss_P8 = numpy.mean(test_losses_P8)
            std_test_loss_P8 = numpy.std(test_losses_P8)
            self.threshP8 = 2.33 * std_test_loss_P8 + mean_test_loss_P8
            self.threshold_idxs.append(self.threshP8)

            mean_test_loss_P9 = numpy.mean(test_losses_P9)
            std_test_loss_P9 = numpy.std(test_losses_P9)
            self.threshP9 = 2.33 * std_test_loss_P9 + mean_test_loss_P9
            self.threshold_idxs.append(self.threshP9)
            mean_test_loss_P10 = numpy.mean(test_losses_P10)
            std_test_loss_P10 = numpy.std(test_losses_P10)
            self.threshP10 = 2.33 * std_test_loss_P10 + mean_test_loss_P10
            self.threshold_idxs.append(self.threshP10)


        return history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs
    
    def inference_mse(self, x) -> tuple[torch.FloatTensor | torch.cuda.FloatTensor, float]:
        x_hat = self(x)
        reconstruction_error = self.criterion(x_hat, x[:, self.POSidxs])
        return x_hat, reconstruction_error.item()
    
    # ToDo: Funktion für anom detect
    def detect_anomaly(self, x) -> tuple[bool, list[int]]:
        r"""
        Based on the given input x this function computes residuals for its modules and compares
        this to trained thresholds for mse values. In case of at least one anomaly it returns TRUE
        else it returns FALSE. It also retuns a list with indices, where an anomaly was found
        """
        anomaly = False
        anom_idx = []  # to store the indices of modules which have detected anomalies
        predictions = self(x)
        losses = []
        for idx, _ in enumerate(predictions):
            losses.append(self.criterion(predictions[idx], x[:, self.LOSSidxs[idx]]).item())
            if losses[idx] > self.threshold_idxs[idx]:
                anom_idx.append(idx)
        if anom_idx:
            anomaly = True
        return anomaly, anom_idx
    
        
    def get_thresholds(self):
        if self.n_pendulums == 10:
            thresholds =[("Threshold P1", self.threshP1),
                        ("Threshold P2", self.threshP2),
                        ("Threshold P3", self.threshP3),
                        ("Threshold P4", self.threshP4),
                        ("Threshold P5", self.threshP5),
                        ("Threshold P6", self.threshP6),
                        ("Threshold P7", self.threshP7),
                        ("Threshold P8", self.threshP8),
                        ("Threshold P9", self.threshP9),
                        ("Threshold P10", self.threshP10)]
        if self.n_pendulums == 8:
            thresholds =[("Threshold P1", self.threshP1),
                        ("Threshold P2", self.threshP2),
                        ("Threshold P3", self.threshP3),
                        ("Threshold P4", self.threshP4),
                        ("Threshold P5", self.threshP5),
                        ("Threshold P6", self.threshP6),
                        ("Threshold P7", self.threshP7),
                        ("Threshold P8", self.threshP8)]
        if self.n_pendulums == 6:
            thresholds =[("Threshold P1", self.threshP1),
                        ("Threshold P2", self.threshP2),
                        ("Threshold P3", self.threshP3),
                        ("Threshold P4", self.threshP4),
                        ("Threshold P5", self.threshP5),
                        ("Threshold P6", self.threshP6)]
        if self.n_pendulums == 4:
            thresholds =[("Threshold P1", self.threshP1),
                        ("Threshold P2", self.threshP2),
                        ("Threshold P3", self.threshP3),
                        ("Threshold P4", self.threshP4)]
        if self.n_pendulums == 2:
            thresholds =[("Threshold P1", self.threshP1),
                        ("Threshold P2", self.threshP2)]
        return thresholds
        
    
    def export_thresholds(self, filepath):
        """
        
        """
        thresholds = self.get_thresholds()
        with open(filepath, 'wb') as file:
            # A new file will be created
            pickle.dump(thresholds, file)
    
    def load_thresholds(self, filepath):
        """
        
        """
        with open(filepath, 'rb') as file: 
            # Call load method to deserialze
            thresholds = pickle.load(file)
            if self.n_pendulums == 10:
                self.threshP1 = thresholds[0][1]
                self.threshP2 = thresholds[1][1]
                self.threshP3 = thresholds[2][1]
                self.threshP4 = thresholds[3][1]
                self.threshP5 = thresholds[4][1]
                self.threshP6 = thresholds[5][1]
                self.threshP7 = thresholds[6][1]
                self.threshP8 = thresholds[7][1]
                self.threshP9 = thresholds[8][1]
                self.threshP10 = thresholds[9][1]

                self.threshold_idxs.append(self.threshP1)
                self.threshold_idxs.append(self.threshP2)
                self.threshold_idxs.append(self.threshP3)
                self.threshold_idxs.append(self.threshP4)
                self.threshold_idxs.append(self.threshP5)
                self.threshold_idxs.append(self.threshP6)
                self.threshold_idxs.append(self.threshP7)
                self.threshold_idxs.append(self.threshP8)
                self.threshold_idxs.append(self.threshP9)
                self.threshold_idxs.append(self.threshP10)


            if self.n_pendulums == 8:
                self.threshP1 = thresholds[0][1]
                self.threshP2 = thresholds[1][1]
                self.threshP3 = thresholds[2][1]
                self.threshP4 = thresholds[3][1]
                self.threshP5 = thresholds[4][1]
                self.threshP6 = thresholds[5][1]
                self.threshP7 = thresholds[6][1]
                self.threshP8 = thresholds[7][1]

                self.threshold_idxs.append(self.threshP1)
                self.threshold_idxs.append(self.threshP2)
                self.threshold_idxs.append(self.threshP3)
                self.threshold_idxs.append(self.threshP4)
                self.threshold_idxs.append(self.threshP5)
                self.threshold_idxs.append(self.threshP6)
                self.threshold_idxs.append(self.threshP7)
                self.threshold_idxs.append(self.threshP8)
            if self.n_pendulums == 6:
                self.threshP1 = thresholds[0][1]
                self.threshP2 = thresholds[1][1]
                self.threshP3 = thresholds[2][1]
                self.threshP4 = thresholds[3][1]
                self.threshP5 = thresholds[4][1]
                self.threshP6 = thresholds[5][1]

                self.threshold_idxs.append(self.threshP1)
                self.threshold_idxs.append(self.threshP2)
                self.threshold_idxs.append(self.threshP3)
                self.threshold_idxs.append(self.threshP4)
                self.threshold_idxs.append(self.threshP5)
                self.threshold_idxs.append(self.threshP6)
            if self.n_pendulums == 4:
                self.threshP1 = thresholds[0][1]
                self.threshP2 = thresholds[1][1]
                self.threshP3 = thresholds[2][1]
                self.threshP4 = thresholds[3][1]

                self.threshold_idxs.append(self.threshP1)
                self.threshold_idxs.append(self.threshP2)
                self.threshold_idxs.append(self.threshP3)
                self.threshold_idxs.append(self.threshP4)
            if self.n_pendulums == 2:
                self.threshP1 = thresholds[0][1]
                self.threshP2 = thresholds[1][1]

                self.threshold_idxs.append(self.threshP1)
                self.threshold_idxs.append(self.threshP2)


class TwoPendulumLstmAe(torch.nn.Module):
    """
    This class contains one neural network for every component type of the technical system.
    Anomalies are detected per component (instance).
    """
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super(TwoPendulumLstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary_name = f"experiments/instance_modular_2_{enc_hidden_size}_tensorboard"

        self.Pendulum1 = [0,1,2,3,4]
        self.Pendulum2 = [5,6,7,8,9]
        self.Pendulum1POS = [0,1]
        self.Pendulum2POS = [5,6]
        
        self.module_idxs = [  # this list tells us, which module number belongs to which module
            [0, 1, 2, 3, 4],  # Pendulum1
            [5, 6, 7, 8, 9]  # Pendulum2
        ]

        self.threshold_idxs = []  # here we store the threshold values for anom detect after training

        self.lossdatalist = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1,
            self.Pendulum2
        ]

        self.lossdatalistPOS = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1POS,
            self.Pendulum2POS
        ]

        self.threshPendulum1 = 0.0
        self.threshPendulum2 = 0.0
        
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.LstmPendulum1 = LstmAe(enc_input_size=len(self.Pendulum1), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum2 = LstmAe(enc_input_size=len(self.Pendulum2), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        
        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)


    def forward(self, x):
        """ Forward pass of MultiModularLstmAe
        Split the input according to the indices provided by the lists and process them separately.
        """

        inPendulum1 = x[:, self.Pendulum1]  # The valve actuator is the only component directly influenced by the command
        inPendulum2 = x[:, self.Pendulum2]  
        
        Pendulum1 = self.LstmPendulum1(inPendulum1)
        Pendulum2 = self.LstmPendulum2(inPendulum2)

        return (Pendulum1, Pendulum2)
    
    def train_model(self, train_ds, val_ds, test_ds, n_epochs: int):
        """
        train_ds: 
        val_ds: 
        test_ds:
        n_epochs: int value
        """
        writer = SummaryWriter(self.summary_name)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        history = dict(train=[], val=[])  # save train and validation losses
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 10000.0  # arbitrary high loss
        best_epoch = -1
        for epoch in range(1, n_epochs + 1):
            #print(f"Epoch {epoch}")
            self = self.train()
            train_losses = []
            for batch_idx, data in enumerate(train_ds):
                optimizer.zero_grad()
                predictions = self(data)
                temp_train_losses = []
                for idx, _ in enumerate(predictions):
                    temp_train_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]))
                loss = sum(temp_train_losses)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            self = self.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_ds):
                    predictions = self(data)
                    temp_val_losses = []
                    for idx, _ in enumerate(predictions):
                        temp_val_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                    loss = sum(temp_val_losses)
                    val_losses.append(loss)
            mean_train_loss = numpy.mean(train_losses)
            mean_val_loss = numpy.mean(val_losses)
            history['train'].append(mean_train_loss)
            writer.add_scalar('train_loss', mean_train_loss, epoch)
            history['val'].append(mean_val_loss)
            writer.add_scalar('val_loss', mean_val_loss, epoch)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.load_state_dict(best_model_wts)
        self.eval()
        # Losses for each module
        test_losses = []
        test_losses_Pendulum1 = []
        test_losses_Pendulum2 = []

        test_pairs = []  # remains for compatibility but is unused here
        with torch.no_grad():
            for batch_idx, data in enumerate(test_ds):
                # ToDo: Gen sub-data sets & assign losses. Put in function?
                predictions = self(data)
                temp_test_losses = []
                for idx, _ in enumerate(predictions):
                    temp_test_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                
                test_losses_Pendulum1.append(temp_test_losses[0])
                test_losses_Pendulum2.append(temp_test_losses[1])

                loss = sum(temp_test_losses)
                test_losses.append(loss)
                writer.add_scalar('test_loss', loss, batch_idx)

        writer.close()
        mean_test_loss = numpy.mean(test_losses)
        std_test_loss = numpy.std(test_losses)
        mean_test_loss_Pendulum1 = numpy.mean(test_losses_Pendulum1)
        std_test_loss_Pendulum1 = numpy.std(test_losses_Pendulum1)
        mean_test_loss_Pendulum2 = numpy.mean(test_losses_Pendulum2)
        std_test_loss_Pendulum2 = numpy.std(test_losses_Pendulum2)
       
        self.threshPendulum1 = 2.33 * std_test_loss_Pendulum1 + mean_test_loss_Pendulum1
        self.threshold_idxs.append(self.threshPendulum1)
        self.threshPendulum2 = 2.33 * std_test_loss_Pendulum2 + mean_test_loss_Pendulum2
        self.threshold_idxs.append(self.threshPendulum2)
   
        
        return history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs
    
    # ToDo: Funktion für anom detect
    def detect_anomaly(self, x) -> tuple[bool, list[int]]:
        r"""
        Based on the given input x this function computes residuals for its modules and compares
        this to trained thresholds for mse values. In case of at least one anomaly it returns TRUE
        else it returns FALSE. It also retuns a list with indices, where an anomaly was found
        """
        anomaly = False
        anom_idx = []  # to store the indices of modules which have detected anomalies
        predictions = self(x)
        losses = []
        for idx, _ in enumerate(predictions):
            losses.append(self.criterion(predictions[idx], x[:, self.lossdatalistPOS[idx]]).item())
            if losses[idx] > self.threshold_idxs[idx]:
                anom_idx.append(idx)
        if anom_idx:
            anomaly = True
        return (anomaly, anom_idx)
        
    def get_thresholds(self):
        thresholds = [("Threshold Pendulum1", self.threshPendulum1),
                        ("Threshold Pendulum2", self.threshPendulum2)]
        return thresholds
    
    def export_thresholds(self, filepath):
        """
        
        """
        thresholds = self.get_thresholds()
        with open(filepath, 'wb') as file:
            # A new file will be created
            pickle.dump(thresholds, file)
    
    def load_thresholds(self, filepath):
        """
        
        """
        with open(filepath, 'rb') as file: 
            # Call load method to deserialze
            thresholds = pickle.load(file)  # loads a list of tuples (name, value)
            self.threshPendulum1 = thresholds[0][1]
            self.threshPendulum2 = thresholds[1][1]

            self.threshold_idxs.append(self.threshPendulum1)
            self.threshold_idxs.append(self.threshPendulum2)


class FourPendulumLstmAe(torch.nn.Module):
    """
    This class contains one neural network for every component type of the technical system.
    Anomalies are detected per component (instance).
    """
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super(FourPendulumLstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary_name = f"experiments/instance_modular_4_{enc_hidden_size}_tensorboard"

        self.Pendulum1 = [0,1,2,3,4]
        self.Pendulum2 = [5,6,7,8,9]
        self.Pendulum3 = [10,11,12,13,14]
        self.Pendulum4 = [15,16,17,18,19]
        self.Pendulum1POS = [0,1]
        self.Pendulum2POS = [5,6]
        self.Pendulum3POS = [10,11]
        self.Pendulum4POS = [15,16]

        
        self.module_idxs = [  # this list tells us, which module number belongs to which module
            [0, 1, 2, 3, 4],  # Pendulum1
            [5, 6, 7, 8, 9],  # Pendulum2
            [10, 11, 12, 13, 14],  # Pendulum3
            [15, 16, 17, 18, 19]  # Pendulum4
        ]

        self.threshold_idxs = []  # here we store the threshold values for anom detect after training

        self.lossdatalist = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1,
            self.Pendulum2,
            self.Pendulum3,
            self.Pendulum4,
        ]

        self.lossdatalistPOS = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1POS,
            self.Pendulum2POS,
            self.Pendulum3POS,
            self.Pendulum4POS,
        ]

        self.threshPendulum1 = 0.0
        self.threshPendulum2 = 0.0
        self.threshPendulum3 = 0.0
        self.threshPendulum4 = 0.0
        
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.LstmPendulum1 = LstmAe(enc_input_size=len(self.Pendulum1), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum2 = LstmAe(enc_input_size=len(self.Pendulum2), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum3 = LstmAe(enc_input_size=len(self.Pendulum3), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum4 = LstmAe(enc_input_size=len(self.Pendulum4), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        
        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)


    def forward(self, x):
        """ Forward pass of MultiModularLstmAe
        Split the input according to the indices provided by the lists and process them separately.
        """

        inPendulum1 = x[:, self.Pendulum1]  # The valve actuator is the only component directly influenced by the command
        inPendulum2 = x[:, self.Pendulum2]
        inPendulum3 = x[:, self.Pendulum3]
        inPendulum4 = x[:, self.Pendulum4]
        
        Pendulum1 = self.LstmPendulum1(inPendulum1)
        Pendulum2 = self.LstmPendulum2(inPendulum2)
        Pendulum3 = self.LstmPendulum3(inPendulum3)
        Pendulum4 = self.LstmPendulum4(inPendulum4)

        return (Pendulum1, Pendulum2, Pendulum3, Pendulum4)
    
    def train_model(self, train_ds, val_ds, test_ds, n_epochs: int):
        """
        train_ds: 
        val_ds: 
        test_ds:
        n_epochs: int value
        """
        writer = SummaryWriter(self.summary_name)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        history = dict(train=[], val=[])  # save train and validation losses
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 10000.0  # arbitrary high loss
        best_epoch = -1
        for epoch in range(1, n_epochs + 1):
            #print(f"Epoch {epoch}")
            self = self.train()
            train_losses = []

            for batch_idx, data in enumerate(train_ds):
                optimizer.zero_grad()
                predictions = self(data)
                temp_train_losses = []
                for idx, _ in enumerate(predictions):
                    temp_train_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]))
                loss = sum(temp_train_losses)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            self = self.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_ds):
                    predictions = self(data)
                    temp_val_losses = []
                    for idx, _ in enumerate(predictions):
                        temp_val_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                    loss = sum(temp_val_losses)
                    val_losses.append(loss)
            mean_train_loss = numpy.mean(train_losses)
            mean_val_loss = numpy.mean(val_losses)
            history['train'].append(mean_train_loss)
            writer.add_scalar('train_loss', mean_train_loss, epoch)
            history['val'].append(mean_val_loss)
            writer.add_scalar('val_loss', mean_val_loss, epoch)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.load_state_dict(best_model_wts)
        self.eval()
        # Losses for each module
        test_losses = []
        test_losses_Pendulum1 = []
        test_losses_Pendulum2 = []
        test_losses_Pendulum3 = []
        test_losses_Pendulum4 = []

        test_pairs = []  # remains for compatibility but is unused here
        with torch.no_grad():
            for batch_idx, data in enumerate(test_ds):
                # ToDo: Gen sub-data sets & assign losses. Put in function?
                predictions = self(data)
                temp_test_losses = []
                for idx, _ in enumerate(predictions):
                    temp_test_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                
                test_losses_Pendulum1.append(temp_test_losses[0])
                test_losses_Pendulum2.append(temp_test_losses[1])
                test_losses_Pendulum3.append(temp_test_losses[2])
                test_losses_Pendulum4.append(temp_test_losses[3])

                loss = sum(temp_test_losses)
                test_losses.append(loss)
                writer.add_scalar('test_loss', loss, batch_idx)

        writer.close()
        mean_test_loss = numpy.mean(test_losses)
        std_test_loss = numpy.std(test_losses)
        mean_test_loss_Pendulum1 = numpy.mean(test_losses_Pendulum1)
        std_test_loss_Pendulum1 = numpy.std(test_losses_Pendulum1)
        mean_test_loss_Pendulum2 = numpy.mean(test_losses_Pendulum2)
        std_test_loss_Pendulum2 = numpy.std(test_losses_Pendulum2)
        mean_test_loss_Pendulum3 = numpy.mean(test_losses_Pendulum3)
        std_test_loss_Pendulum3 = numpy.std(test_losses_Pendulum3)
        mean_test_loss_Pendulum4 = numpy.mean(test_losses_Pendulum4)
        std_test_loss_Pendulum4 = numpy.std(test_losses_Pendulum4)
       
        self.threshPendulum1 = 2.33 * std_test_loss_Pendulum1 + mean_test_loss_Pendulum1
        self.threshold_idxs.append(self.threshPendulum1)
        self.threshPendulum2 = 2.33 * std_test_loss_Pendulum2 + mean_test_loss_Pendulum2
        self.threshold_idxs.append(self.threshPendulum2)
        self.threshPendulum3 = 2.33 * std_test_loss_Pendulum3 + mean_test_loss_Pendulum3
        self.threshold_idxs.append(self.threshPendulum3)
        self.threshPendulum4 = 2.33 * std_test_loss_Pendulum4 + mean_test_loss_Pendulum4
        self.threshold_idxs.append(self.threshPendulum4)
        
        return history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs
    
    # ToDo: Funktion für anom detect
    def detect_anomaly(self, x) -> tuple[bool, list[int]]:
        r"""
        Based on the given input x this function computes residuals for its modules and compares
        this to trained thresholds for mse values. In case of at least one anomaly it returns TRUE
        else it returns FALSE. It also retuns a list with indices, where an anomaly was found
        """
        anomaly = False
        anom_idx = []  # to store the indices of modules which have detected anomalies
        predictions = self(x)
        losses = []
        for idx, _ in enumerate(predictions):
            losses.append(self.criterion(predictions[idx], x[:, self.lossdatalistPOS[idx]]).item())
            if losses[idx] > self.threshold_idxs[idx]:
                anom_idx.append(idx)
        if anom_idx:
            anomaly = True
        return (anomaly, anom_idx)
        
    def get_thresholds(self):
        thresholds = [("Threshold Pendulum1", self.threshPendulum1),
                        ("Threshold Pendulum2", self.threshPendulum2),
                        ("Threshold Pendulum3", self.threshPendulum3),
                        ("Threshold Pendulum4", self.threshPendulum4)
                        ]
        return thresholds
    
    def export_thresholds(self, filepath):
        """
        
        """
        thresholds = self.get_thresholds()
        with open(filepath, 'wb') as file:
            # A new file will be created
            pickle.dump(thresholds, file)
    
    def load_thresholds(self, filepath):
        """
        
        """
        with open(filepath, 'rb') as file: 
            # Call load method to deserialze
            thresholds = pickle.load(file)  # loads a list of tuples (name, value)
            self.threshPendulum1 = thresholds[0][1]
            self.threshPendulum2 = thresholds[1][1]
            self.threshPendulum3 = thresholds[2][1]
            self.threshPendulum4 = thresholds[3][1]

            self.threshold_idxs.append(self.threshPendulum1)
            self.threshold_idxs.append(self.threshPendulum2)
            self.threshold_idxs.append(self.threshPendulum3)
            self.threshold_idxs.append(self.threshPendulum4)


class SixPendulumLstmAe(torch.nn.Module):
    """
    This class contains one neural network for every component type of the technical system.
    Anomalies are detected per component (instance).
    """
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super(SixPendulumLstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary_name = f"experiments/instance_modular_6_{enc_hidden_size}_tensorboard"

        self.Pendulum1 = [0,1,2,3,4]
        self.Pendulum2 = [5,6,7,8,9]
        self.Pendulum3 = [10,11,12,13,14]
        self.Pendulum4 = [15,16,17,18,19]
        self.Pendulum5 = [20,21,22,23,24]
        self.Pendulum6 = [25,26,27,28,29]
        self.Pendulum1POS = [0,1]
        self.Pendulum2POS = [5,6]
        self.Pendulum3POS = [10,11]
        self.Pendulum4POS = [15,16]
        self.Pendulum5POS = [20,21]
        self.Pendulum6POS = [25,26]

        
        self.module_idxs = [  # this list tells us, which module number belongs to which module
            [0, 1, 2, 3, 4],  # Pendulum1
            [5, 6, 7, 8, 9],  # Pendulum2
            [10, 11, 12, 13, 14],  # Pendulum3
            [15, 16, 17, 18, 19],  # Pendulum4
            [20,21,22,23,24],  # Pendulum5
            [25,26,27,28,29]  # Pendulum6

        ]

        self.threshold_idxs = []  # here we store the threshold values for anom detect after training

        self.lossdatalist = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1,
            self.Pendulum2,
            self.Pendulum3,
            self.Pendulum4,
            self.Pendulum5,
            self.Pendulum6
        ]

        self.lossdatalistPOS = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1POS,
            self.Pendulum2POS,
            self.Pendulum3POS,
            self.Pendulum4POS,
            self.Pendulum5POS,
            self.Pendulum6POS
        ]

        self.threshPendulum1 = 0.0
        self.threshPendulum2 = 0.0
        self.threshPendulum3 = 0.0
        self.threshPendulum4 = 0.0
        self.threshPendulum5 = 0.0
        self.threshPendulum6 = 0.0
        
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.LstmPendulum1 = LstmAe(enc_input_size=len(self.Pendulum1), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum2 = LstmAe(enc_input_size=len(self.Pendulum2), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum3 = LstmAe(enc_input_size=len(self.Pendulum3), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum4 = LstmAe(enc_input_size=len(self.Pendulum4), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum5 = LstmAe(enc_input_size=len(self.Pendulum5), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum6 = LstmAe(enc_input_size=len(self.Pendulum6), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        
        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)


    def forward(self, x):
        """ Forward pass of MultiModularLstmAe
        Split the input according to the indices provided by the lists and process them separately.
        """

        inPendulum1 = x[:, self.Pendulum1]  # The valve actuator is the only component directly influenced by the command
        inPendulum2 = x[:, self.Pendulum2]
        inPendulum3 = x[:, self.Pendulum3]
        inPendulum4 = x[:, self.Pendulum4]
        inPendulum5 = x[:, self.Pendulum5]
        inPendulum6 = x[:, self.Pendulum6]
        
        Pendulum1 = self.LstmPendulum1(inPendulum1)
        Pendulum2 = self.LstmPendulum2(inPendulum2)
        Pendulum3 = self.LstmPendulum3(inPendulum3)
        Pendulum4 = self.LstmPendulum4(inPendulum4)
        Pendulum5 = self.LstmPendulum5(inPendulum5)
        Pendulum6 = self.LstmPendulum6(inPendulum6)

        return (Pendulum1, Pendulum2, Pendulum3, Pendulum4, Pendulum5, Pendulum6)
    
    def train_model(self, train_ds, val_ds, test_ds, n_epochs: int):
        """
        train_ds: 
        val_ds: 
        test_ds:
        n_epochs: int value
        """
        writer = SummaryWriter(self.summary_name)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        history = dict(train=[], val=[])  # save train and validation losses
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 10000.0  # arbitrary high loss
        best_epoch = -1
        for epoch in range(1, n_epochs + 1):
            #print(f"Epoch {epoch}")
            self = self.train()
            train_losses = []

            for batch_idx, data in enumerate(train_ds):
                optimizer.zero_grad()
                predictions = self(data)
                temp_train_losses = []
                for idx, _ in enumerate(predictions):
                    temp_train_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]))
                loss = sum(temp_train_losses)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            self = self.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_ds):
                    predictions = self(data)
                    temp_val_losses = []
                    for idx, _ in enumerate(predictions):
                        temp_val_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                    loss = sum(temp_val_losses)
                    val_losses.append(loss)
            mean_train_loss = numpy.mean(train_losses)
            mean_val_loss = numpy.mean(val_losses)
            history['train'].append(mean_train_loss)
            writer.add_scalar('train_loss', mean_train_loss, epoch)
            history['val'].append(mean_val_loss)
            writer.add_scalar('val_loss', mean_val_loss, epoch)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.load_state_dict(best_model_wts)
        self.eval()
        # Losses for each module
        test_losses = []
        test_losses_Pendulum1 = []
        test_losses_Pendulum2 = []
        test_losses_Pendulum3 = []
        test_losses_Pendulum4 = []
        test_losses_Pendulum5 = []
        test_losses_Pendulum6 = []

        test_pairs = []  # remains for compatibility but is unused here
        with torch.no_grad():
            for batch_idx, data in enumerate(test_ds):
                # ToDo: Gen sub-data sets & assign losses. Put in function?
                predictions = self(data)
                temp_test_losses = []
                for idx, _ in enumerate(predictions):
                    temp_test_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                
                test_losses_Pendulum1.append(temp_test_losses[0])
                test_losses_Pendulum2.append(temp_test_losses[1])
                test_losses_Pendulum3.append(temp_test_losses[2])
                test_losses_Pendulum4.append(temp_test_losses[3])
                test_losses_Pendulum5.append(temp_test_losses[4])
                test_losses_Pendulum6.append(temp_test_losses[5])

                loss = sum(temp_test_losses)
                test_losses.append(loss)
                writer.add_scalar('test_loss', loss, batch_idx)

        writer.close()
        mean_test_loss = numpy.mean(test_losses)
        std_test_loss = numpy.std(test_losses)
        mean_test_loss_Pendulum1 = numpy.mean(test_losses_Pendulum1)
        std_test_loss_Pendulum1 = numpy.std(test_losses_Pendulum1)
        mean_test_loss_Pendulum2 = numpy.mean(test_losses_Pendulum2)
        std_test_loss_Pendulum2 = numpy.std(test_losses_Pendulum2)
        mean_test_loss_Pendulum3 = numpy.mean(test_losses_Pendulum3)
        std_test_loss_Pendulum3 = numpy.std(test_losses_Pendulum3)
        mean_test_loss_Pendulum4 = numpy.mean(test_losses_Pendulum4)
        std_test_loss_Pendulum4 = numpy.std(test_losses_Pendulum4)
        mean_test_loss_Pendulum5 = numpy.mean(test_losses_Pendulum5)
        std_test_loss_Pendulum5 = numpy.std(test_losses_Pendulum5)
        mean_test_loss_Pendulum6 = numpy.mean(test_losses_Pendulum6)
        std_test_loss_Pendulum6 = numpy.std(test_losses_Pendulum6)
       
        self.threshPendulum1 = 2.33 * std_test_loss_Pendulum1 + mean_test_loss_Pendulum1
        self.threshold_idxs.append(self.threshPendulum1)
        self.threshPendulum2 = 2.33 * std_test_loss_Pendulum2 + mean_test_loss_Pendulum2
        self.threshold_idxs.append(self.threshPendulum2)
        self.threshPendulum3 = 2.33 * std_test_loss_Pendulum3 + mean_test_loss_Pendulum3
        self.threshold_idxs.append(self.threshPendulum3)
        self.threshPendulum4 = 2.33 * std_test_loss_Pendulum4 + mean_test_loss_Pendulum4
        self.threshold_idxs.append(self.threshPendulum4)
        self.threshPendulum5 = 2.33 * std_test_loss_Pendulum5 + mean_test_loss_Pendulum5
        self.threshold_idxs.append(self.threshPendulum5)
        self.threshPendulum6 = 2.33 * std_test_loss_Pendulum6 + mean_test_loss_Pendulum6
        self.threshold_idxs.append(self.threshPendulum6)
        
        return history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs
    
    # ToDo: Funktion für anom detect
    def detect_anomaly(self, x) -> tuple[bool, list[int]]:
        r"""
        Based on the given input x this function computes residuals for its modules and compares
        this to trained thresholds for mse values. In case of at least one anomaly it returns TRUE
        else it returns FALSE. It also retuns a list with indices, where an anomaly was found
        """
        anomaly = False
        anom_idx = []  # to store the indices of modules which have detected anomalies
        predictions = self(x)
        losses = []
        for idx, _ in enumerate(predictions):
            losses.append(self.criterion(predictions[idx], x[:, self.lossdatalistPOS[idx]]).item())
            if losses[idx] > self.threshold_idxs[idx]:
                anom_idx.append(idx)
        if anom_idx:
            anomaly = True
        return (anomaly, anom_idx)
        
    def get_thresholds(self):
        thresholds = [("Threshold Pendulum1", self.threshPendulum1),
                        ("Threshold Pendulum2", self.threshPendulum2),
                        ("Threshold Pendulum3", self.threshPendulum3),
                        ("Threshold Pendulum4", self.threshPendulum4),
                        ("Threshold Pendulum5", self.threshPendulum5),
                        ("Threshold Pendulum6", self.threshPendulum6)
                        ]
        return thresholds
    
    def export_thresholds(self, filepath):
        """
        
        """
        thresholds = self.get_thresholds()
        with open(filepath, 'wb') as file:
            # A new file will be created
            pickle.dump(thresholds, file)
    
    def load_thresholds(self, filepath):
        """
        
        """
        with open(filepath, 'rb') as file: 
            # Call load method to deserialze
            thresholds = pickle.load(file)  # loads a list of tuples (name, value)
            self.threshPendulum1 = thresholds[0][1]
            self.threshPendulum2 = thresholds[1][1]
            self.threshPendulum3 = thresholds[2][1]
            self.threshPendulum4 = thresholds[3][1]
            self.threshPendulum5 = thresholds[4][1]
            self.threshPendulum6 = thresholds[5][1]


            self.threshold_idxs.append(self.threshPendulum1)
            self.threshold_idxs.append(self.threshPendulum2)
            self.threshold_idxs.append(self.threshPendulum3)
            self.threshold_idxs.append(self.threshPendulum4)
            self.threshold_idxs.append(self.threshPendulum5)
            self.threshold_idxs.append(self.threshPendulum6)


class EightPendulumLstmAe(torch.nn.Module):
    """
    This class contains one neural network for every component type of the technical system.
    Anomalies are detected per component (instance).
    """
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super(EightPendulumLstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary_name = f"experiments/instance_modular_8_{enc_hidden_size}_tensorboard"

        self.Pendulum1 = [0,1,2,3,4]
        self.Pendulum2 = [5,6,7,8,9]
        self.Pendulum3 = [10,11,12,13,14]
        self.Pendulum4 = [15,16,17,18,19]
        self.Pendulum5 = [20,21,22,23,24]
        self.Pendulum6 = [25,26,27,28,29]
        self.Pendulum7 = [30,31,32,33,34]
        self.Pendulum8 = [35,36,37,38,39]
        self.Pendulum1POS = [0,1]
        self.Pendulum2POS = [5,6]
        self.Pendulum3POS = [10,11]
        self.Pendulum4POS = [15,16]
        self.Pendulum5POS = [20,21]
        self.Pendulum6POS = [25,26]
        self.Pendulum7POS = [30,31]
        self.Pendulum8POS = [35,36]
        
        self.module_idxs = [  # this list tells us, which module number belongs to which module
            [0, 1, 2, 3, 4],  # Pendulum1
            [5, 6, 7, 8, 9],  # Pendulum2
            [10, 11, 12, 13, 14],  # Pendulum3
            [15, 16, 17, 18, 19],  # Pendulum4
            [20, 21, 22, 23, 24],  # Pendulum5
            [25, 26, 27, 28, 29],  # Pendulum6
            [30, 31, 32, 33, 34],  # Pendulum7
            [35, 36, 37, 38, 39],  # Pendulum8

        ]

        self.threshold_idxs = []  # here we store the threshold values for anom detect after training

        self.lossdatalist = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1,
            self.Pendulum2,
            self.Pendulum3,
            self.Pendulum4,
            self.Pendulum5,
            self.Pendulum6,
            self.Pendulum7,
            self.Pendulum8
        ]

        self.lossdatalistPOS = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1POS,
            self.Pendulum2POS,
            self.Pendulum3POS,
            self.Pendulum4POS,
            self.Pendulum5POS,
            self.Pendulum6POS,
            self.Pendulum7POS,
            self.Pendulum8POS
        ]

        self.threshPendulum1 = 0.0
        self.threshPendulum2 = 0.0
        self.threshPendulum3 = 0.0
        self.threshPendulum4 = 0.0
        self.threshPendulum5 = 0.0
        self.threshPendulum6 = 0.0
        self.threshPendulum7 = 0.0
        self.threshPendulum8 = 0.0
        
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.LstmPendulum1 = LstmAe(enc_input_size=len(self.Pendulum1), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum2 = LstmAe(enc_input_size=len(self.Pendulum2), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum3 = LstmAe(enc_input_size=len(self.Pendulum3), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum4 = LstmAe(enc_input_size=len(self.Pendulum4), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum5 = LstmAe(enc_input_size=len(self.Pendulum5), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum6 = LstmAe(enc_input_size=len(self.Pendulum6), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum7 = LstmAe(enc_input_size=len(self.Pendulum7), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum8 = LstmAe(enc_input_size=len(self.Pendulum8), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)

        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)


    def forward(self, x):
        """ Forward pass of MultiModularLstmAe
        Split the input according to the indices provided by the lists and process them separately.
        """

        inPendulum1 = x[:, self.Pendulum1]  # The valve actuator is the only component directly influenced by the command
        inPendulum2 = x[:, self.Pendulum2]
        inPendulum3 = x[:, self.Pendulum3]
        inPendulum4 = x[:, self.Pendulum4]
        inPendulum5 = x[:, self.Pendulum5]
        inPendulum6 = x[:, self.Pendulum6]
        inPendulum7 = x[:, self.Pendulum7]
        inPendulum8 = x[:, self.Pendulum8]

        
        Pendulum1 = self.LstmPendulum1(inPendulum1)
        Pendulum2 = self.LstmPendulum2(inPendulum2)
        Pendulum3 = self.LstmPendulum3(inPendulum3)
        Pendulum4 = self.LstmPendulum4(inPendulum4)
        Pendulum5 = self.LstmPendulum5(inPendulum5)
        Pendulum6 = self.LstmPendulum6(inPendulum6)
        Pendulum7 = self.LstmPendulum7(inPendulum7)
        Pendulum8 = self.LstmPendulum8(inPendulum8)

        return (Pendulum1, Pendulum2, Pendulum3, Pendulum4, Pendulum5, Pendulum6, Pendulum7, Pendulum8)
    
    def train_model(self, train_ds, val_ds, test_ds, n_epochs: int):
        """
        train_ds: 
        val_ds: 
        test_ds:
        n_epochs: int value
        """
        writer = SummaryWriter(self.summary_name)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        history = dict(train=[], val=[])  # save train and validation losses
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 10000.0  # arbitrary high loss
        best_epoch = -1
        for epoch in range(1, n_epochs + 1):
            #print(f"Epoch {epoch}")
            self = self.train()
            train_losses = []

            for batch_idx, data in enumerate(train_ds):
                optimizer.zero_grad()
                predictions = self(data)
                temp_train_losses = []
                for idx, _ in enumerate(predictions):
                    temp_train_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]))
                loss = sum(temp_train_losses)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            self = self.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_ds):
                    predictions = self(data)
                    temp_val_losses = []
                    for idx, _ in enumerate(predictions):
                        temp_val_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                    loss = sum(temp_val_losses)
                    val_losses.append(loss)
            mean_train_loss = numpy.mean(train_losses)
            mean_val_loss = numpy.mean(val_losses)
            history['train'].append(mean_train_loss)
            writer.add_scalar('train_loss', mean_train_loss, epoch)
            history['val'].append(mean_val_loss)
            writer.add_scalar('val_loss', mean_val_loss, epoch)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.load_state_dict(best_model_wts)
        self.eval()
        # Losses for each module
        test_losses = []
        test_losses_Pendulum1 = []
        test_losses_Pendulum2 = []
        test_losses_Pendulum3 = []
        test_losses_Pendulum4 = []
        test_losses_Pendulum5 = []
        test_losses_Pendulum6 = []
        test_losses_Pendulum7 = []
        test_losses_Pendulum8 = []

        test_pairs = []  # remains for compatibility but is unused here
        with torch.no_grad():
            for batch_idx, data in enumerate(test_ds):
                # ToDo: Gen sub-data sets & assign losses. Put in function?
                predictions = self(data)
                temp_test_losses = []
                for idx, _ in enumerate(predictions):
                    temp_test_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                
                test_losses_Pendulum1.append(temp_test_losses[0])
                test_losses_Pendulum2.append(temp_test_losses[1])
                test_losses_Pendulum3.append(temp_test_losses[2])
                test_losses_Pendulum4.append(temp_test_losses[3])
                test_losses_Pendulum5.append(temp_test_losses[4])
                test_losses_Pendulum6.append(temp_test_losses[5])
                test_losses_Pendulum7.append(temp_test_losses[6])
                test_losses_Pendulum8.append(temp_test_losses[7])

                loss = sum(temp_test_losses)
                test_losses.append(loss)
                writer.add_scalar('test_loss', loss, batch_idx)

        writer.close()
        mean_test_loss = numpy.mean(test_losses)
        std_test_loss = numpy.std(test_losses)
        mean_test_loss_Pendulum1 = numpy.mean(test_losses_Pendulum1)
        std_test_loss_Pendulum1 = numpy.std(test_losses_Pendulum1)
        mean_test_loss_Pendulum2 = numpy.mean(test_losses_Pendulum2)
        std_test_loss_Pendulum2 = numpy.std(test_losses_Pendulum2)
        mean_test_loss_Pendulum3 = numpy.mean(test_losses_Pendulum3)
        std_test_loss_Pendulum3 = numpy.std(test_losses_Pendulum3)
        mean_test_loss_Pendulum4 = numpy.mean(test_losses_Pendulum4)
        std_test_loss_Pendulum4 = numpy.std(test_losses_Pendulum4)
        mean_test_loss_Pendulum5 = numpy.mean(test_losses_Pendulum5)
        std_test_loss_Pendulum5 = numpy.std(test_losses_Pendulum5)
        mean_test_loss_Pendulum6 = numpy.mean(test_losses_Pendulum6)
        std_test_loss_Pendulum6 = numpy.std(test_losses_Pendulum6)
        mean_test_loss_Pendulum7 = numpy.mean(test_losses_Pendulum7)
        std_test_loss_Pendulum7 = numpy.std(test_losses_Pendulum7)
        mean_test_loss_Pendulum8 = numpy.mean(test_losses_Pendulum8)
        std_test_loss_Pendulum8 = numpy.std(test_losses_Pendulum8)
       
        self.threshPendulum1 = 2.33 * std_test_loss_Pendulum1 + mean_test_loss_Pendulum1
        self.threshold_idxs.append(self.threshPendulum1)
        self.threshPendulum2 = 2.33 * std_test_loss_Pendulum2 + mean_test_loss_Pendulum2
        self.threshold_idxs.append(self.threshPendulum2)
        self.threshPendulum3 = 2.33 * std_test_loss_Pendulum3 + mean_test_loss_Pendulum3
        self.threshold_idxs.append(self.threshPendulum3)
        self.threshPendulum4 = 2.33 * std_test_loss_Pendulum4 + mean_test_loss_Pendulum4
        self.threshold_idxs.append(self.threshPendulum4)
        self.threshPendulum5 = 2.33 * std_test_loss_Pendulum5 + mean_test_loss_Pendulum5
        self.threshold_idxs.append(self.threshPendulum5)
        self.threshPendulum6 = 2.33 * std_test_loss_Pendulum6 + mean_test_loss_Pendulum6
        self.threshold_idxs.append(self.threshPendulum6)
        self.threshPendulum7 = 2.33 * std_test_loss_Pendulum7 + mean_test_loss_Pendulum7
        self.threshold_idxs.append(self.threshPendulum7)
        self.threshPendulum8 = 2.33 * std_test_loss_Pendulum8 + mean_test_loss_Pendulum8
        self.threshold_idxs.append(self.threshPendulum8)
        
        return history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs
    
    # ToDo: Funktion für anom detect
    def detect_anomaly(self, x) -> tuple[bool, list[int]]:
        r"""
        Based on the given input x this function computes residuals for its modules and compares
        this to trained thresholds for mse values. In case of at least one anomaly it returns TRUE
        else it returns FALSE. It also retuns a list with indices, where an anomaly was found
        """
        anomaly = False
        anom_idx = []  # to store the indices of modules which have detected anomalies
        predictions = self(x)
        losses = []
        for idx, _ in enumerate(predictions):
            losses.append(self.criterion(predictions[idx], x[:, self.lossdatalistPOS[idx]]).item())
            if losses[idx] > self.threshold_idxs[idx]:
                anom_idx.append(idx)
        if anom_idx:
            anomaly = True
        return (anomaly, anom_idx)
        
    def get_thresholds(self):
        thresholds = [("Threshold Pendulum1", self.threshPendulum1),
                        ("Threshold Pendulum2", self.threshPendulum2),
                        ("Threshold Pendulum3", self.threshPendulum3),
                        ("Threshold Pendulum4", self.threshPendulum4),
                        ("Threshold Pendulum5", self.threshPendulum5),
                        ("Threshold Pendulum6", self.threshPendulum6),
                        ("Threshold Pendulum7", self.threshPendulum7),
                        ("Threshold Pendulum8", self.threshPendulum8)
                        ]
        return thresholds
    
    def export_thresholds(self, filepath):
        """
        
        """
        thresholds = self.get_thresholds()
        with open(filepath, 'wb') as file:
            # A new file will be created
            pickle.dump(thresholds, file)
    
    def load_thresholds(self, filepath):
        """
        
        """
        with open(filepath, 'rb') as file: 
            # Call load method to deserialze
            thresholds = pickle.load(file)  # loads a list of tuples (name, value)
            self.threshPendulum1 = thresholds[0][1]
            self.threshPendulum2 = thresholds[1][1]
            self.threshPendulum3 = thresholds[2][1]
            self.threshPendulum4 = thresholds[3][1]
            self.threshPendulum5 = thresholds[4][1]
            self.threshPendulum6 = thresholds[5][1]
            self.threshPendulum7 = thresholds[6][1]
            self.threshPendulum8 = thresholds[7][1]


            self.threshold_idxs.append(self.threshPendulum1)
            self.threshold_idxs.append(self.threshPendulum2)
            self.threshold_idxs.append(self.threshPendulum3)
            self.threshold_idxs.append(self.threshPendulum4)
            self.threshold_idxs.append(self.threshPendulum5)
            self.threshold_idxs.append(self.threshPendulum6)
            self.threshold_idxs.append(self.threshPendulum7)
            self.threshold_idxs.append(self.threshPendulum8)


class TenPendulumLstmAe(torch.nn.Module):
    """
    This class contains one neural network for every component type of the technical system.
    Anomalies are detected per component (instance).
    """
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super(TenPendulumLstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary_name = f"experiments/instance_modular_10_{enc_hidden_size}_tensorboard"

        self.Pendulum1 = [0,1,2,3,4]
        self.Pendulum2 = [5,6,7,8,9]
        self.Pendulum3 = [10,11,12,13,14]
        self.Pendulum4 = [15,16,17,18,19]
        self.Pendulum5 = [20,21,22,23,24]
        self.Pendulum6 = [25,26,27,28,29]
        self.Pendulum7 = [30,31,32,33,34]
        self.Pendulum8 = [35,36,37,38,39]
        self.Pendulum9 = [40,41,42,43,44]
        self.Pendulum10 = [45,46,47,48,49]
        
        self.Pendulum1POS = [0,1]
        self.Pendulum2POS = [5,6]
        self.Pendulum3POS = [10,11]
        self.Pendulum4POS = [15,16]
        self.Pendulum5POS = [20,21]
        self.Pendulum6POS = [25,26]
        self.Pendulum7POS = [30,31]
        self.Pendulum8POS = [35,36]
        self.Pendulum9POS = [40,41]
        self.Pendulum10POS = [45,46]

        self.module_idxs = [  # this list tells us, which module number belongs to which module
            [0, 1, 2, 3, 4],  # Pendulum1
            [5, 6, 7, 8, 9],  # Pendulum2
            [10, 11, 12, 13, 14],  # Pendulum3
            [15, 16, 17, 18, 19],  # Pendulum4
            [20, 21, 22, 23, 24],  # Pendulum5
            [25, 26, 27, 28, 29],  # Pendulum6
            [30, 31, 32, 33, 34],  # Pendulum7
            [35, 36, 37, 38, 39],  # Pendulum8
            [40, 41, 42, 43, 44],
            [45, 46, 47, 48, 49]

        ]

        self.threshold_idxs = []  # here we store the threshold values for anom detect after training

        self.lossdatalist = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1,
            self.Pendulum2,
            self.Pendulum3,
            self.Pendulum4,
            self.Pendulum5,
            self.Pendulum6,
            self.Pendulum7,
            self.Pendulum8,
            self.Pendulum9,
            self.Pendulum10
        ]

        self.lossdatalistPOS = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1POS,
            self.Pendulum2POS,
            self.Pendulum3POS,
            self.Pendulum4POS,
            self.Pendulum5POS,
            self.Pendulum6POS,
            self.Pendulum7POS,
            self.Pendulum8POS,
            self.Pendulum9POS,
            self.Pendulum10POS
        ]

        self.threshPendulum1 = 0.0
        self.threshPendulum2 = 0.0
        self.threshPendulum3 = 0.0
        self.threshPendulum4 = 0.0
        self.threshPendulum5 = 0.0
        self.threshPendulum6 = 0.0
        self.threshPendulum7 = 0.0
        self.threshPendulum8 = 0.0
        self.threshPendulum9 = 0.0
        self.threshPendulum10 = 0.0
        
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.LstmPendulum1 = LstmAe(enc_input_size=len(self.Pendulum1), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum2 = LstmAe(enc_input_size=len(self.Pendulum2), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum3 = LstmAe(enc_input_size=len(self.Pendulum3), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum4 = LstmAe(enc_input_size=len(self.Pendulum4), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum5 = LstmAe(enc_input_size=len(self.Pendulum5), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum6 = LstmAe(enc_input_size=len(self.Pendulum6), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum7 = LstmAe(enc_input_size=len(self.Pendulum7), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum8 = LstmAe(enc_input_size=len(self.Pendulum8), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum9 = LstmAe(enc_input_size=len(self.Pendulum9), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum10 = LstmAe(enc_input_size=len(self.Pendulum10), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)

        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)


    def forward(self, x):
        """ Forward pass of MultiModularLstmAe
        Split the input according to the indices provided by the lists and process them separately.
        """

        inPendulum1 = x[:, self.Pendulum1]  # The valve actuator is the only component directly influenced by the command
        inPendulum2 = x[:, self.Pendulum2]
        inPendulum3 = x[:, self.Pendulum3]
        inPendulum4 = x[:, self.Pendulum4]
        inPendulum5 = x[:, self.Pendulum5]
        inPendulum6 = x[:, self.Pendulum6]
        inPendulum7 = x[:, self.Pendulum7]
        inPendulum8 = x[:, self.Pendulum8]
        inPendulum9 = x[:, self.Pendulum9]
        inPendulum10 = x[:, self.Pendulum10]

        Pendulum1 = self.LstmPendulum1(inPendulum1)
        Pendulum2 = self.LstmPendulum2(inPendulum2)
        Pendulum3 = self.LstmPendulum3(inPendulum3)
        Pendulum4 = self.LstmPendulum4(inPendulum4)
        Pendulum5 = self.LstmPendulum5(inPendulum5)
        Pendulum6 = self.LstmPendulum6(inPendulum6)
        Pendulum7 = self.LstmPendulum7(inPendulum7)
        Pendulum8 = self.LstmPendulum8(inPendulum8)
        Pendulum9 = self.LstmPendulum9(inPendulum9)
        Pendulum10 = self.LstmPendulum10(inPendulum10)

        return (Pendulum1, Pendulum2, Pendulum3, Pendulum4, Pendulum5, Pendulum6, Pendulum7, Pendulum8, Pendulum9, Pendulum10)
    
    def train_model(self, train_ds, val_ds, test_ds, n_epochs: int):
        """
        train_ds: 
        val_ds: 
        test_ds:
        n_epochs: int value
        """
        writer = SummaryWriter(self.summary_name)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        history = dict(train=[], val=[])  # save train and validation losses
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 10000.0  # arbitrary high loss
        best_epoch = -1
        for epoch in range(1, n_epochs + 1):
            #print(f"Epoch {epoch}")
            self = self.train()
            train_losses = []

            for batch_idx, data in enumerate(train_ds):
                optimizer.zero_grad()
                predictions = self(data)
                temp_train_losses = []
                for idx, _ in enumerate(predictions):
                    temp_train_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]))
                loss = sum(temp_train_losses)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            self = self.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_ds):
                    predictions = self(data)
                    temp_val_losses = []
                    for idx, _ in enumerate(predictions):
                        temp_val_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                    loss = sum(temp_val_losses)
                    val_losses.append(loss)
            mean_train_loss = numpy.mean(train_losses)
            mean_val_loss = numpy.mean(val_losses)
            history['train'].append(mean_train_loss)
            writer.add_scalar('train_loss', mean_train_loss, epoch)
            history['val'].append(mean_val_loss)
            writer.add_scalar('val_loss', mean_val_loss, epoch)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.load_state_dict(best_model_wts)
        self.eval()
        # Losses for each module
        test_losses = []
        test_losses_Pendulum1 = []
        test_losses_Pendulum2 = []
        test_losses_Pendulum3 = []
        test_losses_Pendulum4 = []
        test_losses_Pendulum5 = []
        test_losses_Pendulum6 = []
        test_losses_Pendulum7 = []
        test_losses_Pendulum8 = []
        test_losses_Pendulum9 = []
        test_losses_Pendulum10 = []

        test_pairs = []  # remains for compatibility but is unused here
        with torch.no_grad():
            for batch_idx, data in enumerate(test_ds):
                # ToDo: Gen sub-data sets & assign losses. Put in function?
                predictions = self(data)
                temp_test_losses = []
                for idx, _ in enumerate(predictions):
                    temp_test_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                
                test_losses_Pendulum1.append(temp_test_losses[0])
                test_losses_Pendulum2.append(temp_test_losses[1])
                test_losses_Pendulum3.append(temp_test_losses[2])
                test_losses_Pendulum4.append(temp_test_losses[3])
                test_losses_Pendulum5.append(temp_test_losses[4])
                test_losses_Pendulum6.append(temp_test_losses[5])
                test_losses_Pendulum7.append(temp_test_losses[6])
                test_losses_Pendulum8.append(temp_test_losses[7])
                test_losses_Pendulum9.append(temp_test_losses[8])
                test_losses_Pendulum10.append(temp_test_losses[9])

                loss = sum(temp_test_losses)
                test_losses.append(loss)
                writer.add_scalar('test_loss', loss, batch_idx)

        writer.close()
        mean_test_loss = numpy.mean(test_losses)
        std_test_loss = numpy.std(test_losses)
        mean_test_loss_Pendulum1 = numpy.mean(test_losses_Pendulum1)
        std_test_loss_Pendulum1 = numpy.std(test_losses_Pendulum1)
        mean_test_loss_Pendulum2 = numpy.mean(test_losses_Pendulum2)
        std_test_loss_Pendulum2 = numpy.std(test_losses_Pendulum2)
        mean_test_loss_Pendulum3 = numpy.mean(test_losses_Pendulum3)
        std_test_loss_Pendulum3 = numpy.std(test_losses_Pendulum3)
        mean_test_loss_Pendulum4 = numpy.mean(test_losses_Pendulum4)
        std_test_loss_Pendulum4 = numpy.std(test_losses_Pendulum4)
        mean_test_loss_Pendulum5 = numpy.mean(test_losses_Pendulum5)
        std_test_loss_Pendulum5 = numpy.std(test_losses_Pendulum5)
        mean_test_loss_Pendulum6 = numpy.mean(test_losses_Pendulum6)
        std_test_loss_Pendulum6 = numpy.std(test_losses_Pendulum6)
        mean_test_loss_Pendulum7 = numpy.mean(test_losses_Pendulum7)
        std_test_loss_Pendulum7 = numpy.std(test_losses_Pendulum7)
        mean_test_loss_Pendulum8 = numpy.mean(test_losses_Pendulum8)
        std_test_loss_Pendulum8 = numpy.std(test_losses_Pendulum8)
        mean_test_loss_Pendulum9 = numpy.mean(test_losses_Pendulum9)
        std_test_loss_Pendulum9 = numpy.std(test_losses_Pendulum9)
        mean_test_loss_Pendulum10 = numpy.mean(test_losses_Pendulum10)
        std_test_loss_Pendulum10 = numpy.std(test_losses_Pendulum10)
       
        self.threshPendulum1 = 2.33 * std_test_loss_Pendulum1 + mean_test_loss_Pendulum1
        self.threshold_idxs.append(self.threshPendulum1)
        self.threshPendulum2 = 2.33 * std_test_loss_Pendulum2 + mean_test_loss_Pendulum2
        self.threshold_idxs.append(self.threshPendulum2)
        self.threshPendulum3 = 2.33 * std_test_loss_Pendulum3 + mean_test_loss_Pendulum3
        self.threshold_idxs.append(self.threshPendulum3)
        self.threshPendulum4 = 2.33 * std_test_loss_Pendulum4 + mean_test_loss_Pendulum4
        self.threshold_idxs.append(self.threshPendulum4)
        self.threshPendulum5 = 2.33 * std_test_loss_Pendulum5 + mean_test_loss_Pendulum5
        self.threshold_idxs.append(self.threshPendulum5)
        self.threshPendulum6 = 2.33 * std_test_loss_Pendulum6 + mean_test_loss_Pendulum6
        self.threshold_idxs.append(self.threshPendulum6)
        self.threshPendulum7 = 2.33 * std_test_loss_Pendulum7 + mean_test_loss_Pendulum7
        self.threshold_idxs.append(self.threshPendulum7)
        self.threshPendulum8 = 2.33 * std_test_loss_Pendulum8 + mean_test_loss_Pendulum8
        self.threshold_idxs.append(self.threshPendulum8)
        self.threshPendulum9 = 2.33 * std_test_loss_Pendulum9 + mean_test_loss_Pendulum9
        self.threshold_idxs.append(self.threshPendulum9)
        self.threshPendulum10 = 2.33 * std_test_loss_Pendulum10 + mean_test_loss_Pendulum10
        self.threshold_idxs.append(self.threshPendulum10)
        
        return history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs
    
    # ToDo: Funktion für anom detect
    def detect_anomaly(self, x) -> tuple[bool, list[int]]:
        r"""
        Based on the given input x this function computes residuals for its modules and compares
        this to trained thresholds for mse values. In case of at least one anomaly it returns TRUE
        else it returns FALSE. It also retuns a list with indices, where an anomaly was found
        """
        anomaly = False
        anom_idx = []  # to store the indices of modules which have detected anomalies
        predictions = self(x)
        losses = []
        for idx, _ in enumerate(predictions):
            losses.append(self.criterion(predictions[idx], x[:, self.lossdatalistPOS[idx]]).item())
            if losses[idx] > self.threshold_idxs[idx]:
                anom_idx.append(idx)
        if anom_idx:
            anomaly = True
        return (anomaly, anom_idx)
        
    def get_thresholds(self):
        thresholds = [("Threshold Pendulum1", self.threshPendulum1),
                        ("Threshold Pendulum2", self.threshPendulum2),
                        ("Threshold Pendulum3", self.threshPendulum3),
                        ("Threshold Pendulum4", self.threshPendulum4),
                        ("Threshold Pendulum5", self.threshPendulum5),
                        ("Threshold Pendulum6", self.threshPendulum6),
                        ("Threshold Pendulum7", self.threshPendulum7),
                        ("Threshold Pendulum8", self.threshPendulum8),
                        ("Threshold Pendulum9", self.threshPendulum9),
                        ("Threshold Pendulum10", self.threshPendulum10)
                        ]
        return thresholds
    
    def export_thresholds(self, filepath):
        """
        
        """
        thresholds = self.get_thresholds()
        with open(filepath, 'wb') as file:
            # A new file will be created
            pickle.dump(thresholds, file)
    
    def load_thresholds(self, filepath):
        """
        
        """
        with open(filepath, 'rb') as file: 
            # Call load method to deserialze
            thresholds = pickle.load(file)  # loads a list of tuples (name, value)
            self.threshPendulum1 = thresholds[0][1]
            self.threshPendulum2 = thresholds[1][1]
            self.threshPendulum3 = thresholds[2][1]
            self.threshPendulum4 = thresholds[3][1]
            self.threshPendulum5 = thresholds[4][1]
            self.threshPendulum6 = thresholds[5][1]
            self.threshPendulum7 = thresholds[6][1]
            self.threshPendulum8 = thresholds[7][1]
            self.threshPendulum9 = thresholds[8][1]
            self.threshPendulum10 = thresholds[9][1]


            self.threshold_idxs.append(self.threshPendulum1)
            self.threshold_idxs.append(self.threshPendulum2)
            self.threshold_idxs.append(self.threshPendulum3)
            self.threshold_idxs.append(self.threshPendulum4)
            self.threshold_idxs.append(self.threshPendulum5)
            self.threshold_idxs.append(self.threshPendulum6)
            self.threshold_idxs.append(self.threshPendulum7)
            self.threshold_idxs.append(self.threshPendulum8)
            self.threshold_idxs.append(self.threshPendulum9)
            self.threshold_idxs.append(self.threshPendulum10)


class TwoPendulumTypeLstmAe(torch.nn.Module):
    """
    This class contains one neural network for every component type of the technical system.
    Anomalies are detected per component (instance).
    """
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super(TwoPendulumTypeLstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary_name = f"experiments/type_modular_2_{enc_hidden_size}_tensorboard"

        self.Pendulum1 = [0,1,2,3,4]
        self.Pendulum2 = [5,6,7,8,9]
        self.Pendulum1POS = [0,1]
        self.Pendulum2POS = [5,6]
        
        self.module_idxs = [  # this list tells us, which module number belongs to which module
            [0, 1, 2, 3, 4],  # Pendulum1
            [5, 6, 7, 8, 9]  # Pendulum2
        ]

        self.threshold_idxs = []  # here we store the threshold values for anom detect after training

        self.lossdatalist = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1,
            self.Pendulum2
        ]

        self.lossdatalistPOS = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1POS,
            self.Pendulum2POS
        ]

        self.threshPendulum1 = 0.0
        self.threshPendulum2 = 0.0
        
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.LstmPendulum1 = LstmAe(enc_input_size=len(self.Pendulum1), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        self.LstmPendulum2 = LstmAe(enc_input_size=len(self.Pendulum2), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        
        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)


    def forward(self, x):
        """ Forward pass of MultiModularLstmAe
        Split the input according to the indices provided by the lists and process them separately.
        """

        inPendulum1 = x[:, self.Pendulum1]  # The valve actuator is the only component directly influenced by the command
        inPendulum2 = x[:, self.Pendulum2]  
        
        Pendulum1 = self.LstmPendulum1(inPendulum1)
        Pendulum2 = self.LstmPendulum1(inPendulum2)

        return (Pendulum1, Pendulum2)
    
    def train_model(self, train_ds, val_ds, test_ds, n_epochs: int):
        """
        train_ds: 
        val_ds: 
        test_ds:
        n_epochs: int value
        """
        writer = SummaryWriter(self.summary_name)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        history = dict(train=[], val=[])  # save train and validation losses
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 10000.0  # arbitrary high loss
        best_epoch = -1
        for epoch in range(1, n_epochs + 1):
            #print(f"Epoch {epoch}")
            self = self.train()
            train_losses = []
            for batch_idx, data in enumerate(train_ds):
                optimizer.zero_grad()
                predictions = self(data)
                temp_train_losses = []
                for idx, _ in enumerate(predictions):
                    temp_train_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]))
                loss = sum(temp_train_losses)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            self = self.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_ds):
                    predictions = self(data)
                    temp_val_losses = []
                    for idx, _ in enumerate(predictions):
                        temp_val_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                    loss = sum(temp_val_losses)
                    val_losses.append(loss)
            mean_train_loss = numpy.mean(train_losses)
            mean_val_loss = numpy.mean(val_losses)
            history['train'].append(mean_train_loss)
            writer.add_scalar('train_loss', mean_train_loss, epoch)
            history['val'].append(mean_val_loss)
            writer.add_scalar('val_loss', mean_val_loss, epoch)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.load_state_dict(best_model_wts)
        self.eval()
        # Losses for each module
        test_losses = []
        test_losses_Pendulum1 = []
        test_losses_Pendulum2 = []

        test_pairs = []  # remains for compatibility but is unused here
        with torch.no_grad():
            for batch_idx, data in enumerate(test_ds):
                # ToDo: Gen sub-data sets & assign losses. Put in function?
                predictions = self(data)
                temp_test_losses = []
                for idx, _ in enumerate(predictions):
                    temp_test_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                
                test_losses_Pendulum1.append(temp_test_losses[0])
                test_losses_Pendulum2.append(temp_test_losses[1])

                loss = sum(temp_test_losses)
                test_losses.append(loss)
                writer.add_scalar('test_loss', loss, batch_idx)

        writer.close()
        mean_test_loss = numpy.mean(test_losses)
        std_test_loss = numpy.std(test_losses)
        mean_test_loss_Pendulum1 = numpy.mean(test_losses_Pendulum1)
        std_test_loss_Pendulum1 = numpy.std(test_losses_Pendulum1)
        mean_test_loss_Pendulum2 = numpy.mean(test_losses_Pendulum2)
        std_test_loss_Pendulum2 = numpy.std(test_losses_Pendulum2)
       
        self.threshPendulum1 = 2.33 * std_test_loss_Pendulum1 + mean_test_loss_Pendulum1
        self.threshold_idxs.append(self.threshPendulum1)
        self.threshPendulum2 = 2.33 * std_test_loss_Pendulum2 + mean_test_loss_Pendulum2
        self.threshold_idxs.append(self.threshPendulum2)
   
        
        return history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs
    
    # ToDo: Funktion für anom detect
    def detect_anomaly(self, x) -> tuple[bool, list[int]]:
        r"""
        Based on the given input x this function computes residuals for its modules and compares
        this to trained thresholds for mse values. In case of at least one anomaly it returns TRUE
        else it returns FALSE. It also retuns a list with indices, where an anomaly was found
        """
        anomaly = False
        anom_idx = []  # to store the indices of modules which have detected anomalies
        predictions = self(x)
        losses = []
        for idx, _ in enumerate(predictions):
            losses.append(self.criterion(predictions[idx], x[:, self.lossdatalistPOS[idx]]).item())
            if losses[idx] > self.threshold_idxs[idx]:
                anom_idx.append(idx)
        if anom_idx:
            anomaly = True
        return (anomaly, anom_idx)
        
    def get_thresholds(self):
        thresholds = [("Threshold Pendulum1", self.threshPendulum1),
                        ("Threshold Pendulum2", self.threshPendulum2)]
        return thresholds
    
    def export_thresholds(self, filepath):
        """
        
        """
        thresholds = self.get_thresholds()
        with open(filepath, 'wb') as file:
            # A new file will be created
            pickle.dump(thresholds, file)
    
    def load_thresholds(self, filepath):
        """
        
        """
        with open(filepath, 'rb') as file: 
            # Call load method to deserialze
            thresholds = pickle.load(file)  # loads a list of tuples (name, value)
            self.threshPendulum1 = thresholds[0][1]
            self.threshPendulum2 = thresholds[1][1]

            self.threshold_idxs.append(self.threshPendulum1)
            self.threshold_idxs.append(self.threshPendulum2)


class FourPendulumTypeLstmAe(torch.nn.Module):
    """
    This class contains one neural network for every component type of the technical system.
    Anomalies are detected per component (instance).
    """
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super(FourPendulumTypeLstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary_name = f"experiments/type_modular_4_{enc_hidden_size}_tensorboard"

        self.Pendulum1 = [0,1,2,3,4]
        self.Pendulum2 = [5,6,7,8,9]
        self.Pendulum3 = [10,11,12,13,14]
        self.Pendulum4 = [15,16,17,18,19]
        self.Pendulum1POS = [0,1]
        self.Pendulum2POS = [5,6]
        self.Pendulum3POS = [10,11]
        self.Pendulum4POS = [15,16]

        
        self.module_idxs = [  # this list tells us, which module number belongs to which module
            [0, 1, 2, 3, 4],  # Pendulum1
            [5, 6, 7, 8, 9],  # Pendulum2
            [10, 11, 12, 13, 14],  # Pendulum3
            [15, 16, 17, 18, 19]  # Pendulum4
        ]

        self.threshold_idxs = []  # here we store the threshold values for anom detect after training

        self.lossdatalist = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1,
            self.Pendulum2,
            self.Pendulum3,
            self.Pendulum4,
        ]

        self.lossdatalistPOS = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1POS,
            self.Pendulum2POS,
            self.Pendulum3POS,
            self.Pendulum4POS,
        ]

        self.threshPendulum1 = 0.0
        self.threshPendulum2 = 0.0
        self.threshPendulum3 = 0.0
        self.threshPendulum4 = 0.0
        
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.LstmPendulum1 = LstmAe(enc_input_size=len(self.Pendulum1), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        
        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)


    def forward(self, x):
        """ Forward pass of MultiModularLstmAe
        Split the input according to the indices provided by the lists and process them separately.
        """

        inPendulum1 = x[:, self.Pendulum1]  # The valve actuator is the only component directly influenced by the command
        inPendulum2 = x[:, self.Pendulum2]
        inPendulum3 = x[:, self.Pendulum3]
        inPendulum4 = x[:, self.Pendulum4]
        
        Pendulum1 = self.LstmPendulum1(inPendulum1)
        Pendulum2 = self.LstmPendulum1(inPendulum2)
        Pendulum3 = self.LstmPendulum1(inPendulum3)
        Pendulum4 = self.LstmPendulum1(inPendulum4)

        return (Pendulum1, Pendulum2, Pendulum3, Pendulum4)
    
    def train_model(self, train_ds, val_ds, test_ds, n_epochs: int):
        """
        train_ds: 
        val_ds: 
        test_ds:
        n_epochs: int value
        """
        writer = SummaryWriter(self.summary_name)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        history = dict(train=[], val=[])  # save train and validation losses
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 10000.0  # arbitrary high loss
        best_epoch = -1
        for epoch in range(1, n_epochs + 1):
            #print(f"Epoch {epoch}")
            self = self.train()
            train_losses = []

            for batch_idx, data in enumerate(train_ds):
                optimizer.zero_grad()
                predictions = self(data)
                temp_train_losses = []
                for idx, _ in enumerate(predictions):
                    temp_train_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]))
                loss = sum(temp_train_losses)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            self = self.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_ds):
                    predictions = self(data)
                    temp_val_losses = []
                    for idx, _ in enumerate(predictions):
                        temp_val_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                    loss = sum(temp_val_losses)
                    val_losses.append(loss)
            mean_train_loss = numpy.mean(train_losses)
            mean_val_loss = numpy.mean(val_losses)
            history['train'].append(mean_train_loss)
            writer.add_scalar('train_loss', mean_train_loss, epoch)
            history['val'].append(mean_val_loss)
            writer.add_scalar('val_loss', mean_val_loss, epoch)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.load_state_dict(best_model_wts)
        self.eval()
        # Losses for each module
        test_losses = []
        test_losses_Pendulum1 = []
        test_losses_Pendulum2 = []
        test_losses_Pendulum3 = []
        test_losses_Pendulum4 = []

        test_pairs = []  # remains for compatibility but is unused here
        with torch.no_grad():
            for batch_idx, data in enumerate(test_ds):
                # ToDo: Gen sub-data sets & assign losses. Put in function?
                predictions = self(data)
                temp_test_losses = []
                for idx, _ in enumerate(predictions):
                    temp_test_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                
                test_losses_Pendulum1.append(temp_test_losses[0])
                test_losses_Pendulum2.append(temp_test_losses[1])
                test_losses_Pendulum3.append(temp_test_losses[2])
                test_losses_Pendulum4.append(temp_test_losses[3])

                loss = sum(temp_test_losses)
                test_losses.append(loss)
                writer.add_scalar('test_loss', loss, batch_idx)

        writer.close()
        mean_test_loss = numpy.mean(test_losses)
        std_test_loss = numpy.std(test_losses)
        mean_test_loss_Pendulum1 = numpy.mean(test_losses_Pendulum1)
        std_test_loss_Pendulum1 = numpy.std(test_losses_Pendulum1)
        mean_test_loss_Pendulum2 = numpy.mean(test_losses_Pendulum2)
        std_test_loss_Pendulum2 = numpy.std(test_losses_Pendulum2)
        mean_test_loss_Pendulum3 = numpy.mean(test_losses_Pendulum3)
        std_test_loss_Pendulum3 = numpy.std(test_losses_Pendulum3)
        mean_test_loss_Pendulum4 = numpy.mean(test_losses_Pendulum4)
        std_test_loss_Pendulum4 = numpy.std(test_losses_Pendulum4)
       
        self.threshPendulum1 = 2.33 * std_test_loss_Pendulum1 + mean_test_loss_Pendulum1
        self.threshold_idxs.append(self.threshPendulum1)
        self.threshPendulum2 = 2.33 * std_test_loss_Pendulum2 + mean_test_loss_Pendulum2
        self.threshold_idxs.append(self.threshPendulum2)
        self.threshPendulum3 = 2.33 * std_test_loss_Pendulum3 + mean_test_loss_Pendulum3
        self.threshold_idxs.append(self.threshPendulum3)
        self.threshPendulum4 = 2.33 * std_test_loss_Pendulum4 + mean_test_loss_Pendulum4
        self.threshold_idxs.append(self.threshPendulum4)
        
        return history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs
    
    # ToDo: Funktion für anom detect
    def detect_anomaly(self, x) -> tuple[bool, list[int]]:
        r"""
        Based on the given input x this function computes residuals for its modules and compares
        this to trained thresholds for mse values. In case of at least one anomaly it returns TRUE
        else it returns FALSE. It also retuns a list with indices, where an anomaly was found
        """
        anomaly = False
        anom_idx = []  # to store the indices of modules which have detected anomalies
        predictions = self(x)
        losses = []
        for idx, _ in enumerate(predictions):
            losses.append(self.criterion(predictions[idx], x[:, self.lossdatalistPOS[idx]]).item())
            if losses[idx] > self.threshold_idxs[idx]:
                anom_idx.append(idx)
        if anom_idx:
            anomaly = True
        return (anomaly, anom_idx)
        
    def get_thresholds(self):
        thresholds = [("Threshold Pendulum1", self.threshPendulum1),
                        ("Threshold Pendulum2", self.threshPendulum2),
                        ("Threshold Pendulum3", self.threshPendulum3),
                        ("Threshold Pendulum4", self.threshPendulum4)
                        ]
        return thresholds
    
    def export_thresholds(self, filepath):
        """
        
        """
        thresholds = self.get_thresholds()
        with open(filepath, 'wb') as file:
            # A new file will be created
            pickle.dump(thresholds, file)
    
    def load_thresholds(self, filepath):
        """
        
        """
        with open(filepath, 'rb') as file: 
            # Call load method to deserialze
            thresholds = pickle.load(file)  # loads a list of tuples (name, value)
            self.threshPendulum1 = thresholds[0][1]
            self.threshPendulum2 = thresholds[1][1]
            self.threshPendulum3 = thresholds[2][1]
            self.threshPendulum4 = thresholds[3][1]

            self.threshold_idxs.append(self.threshPendulum1)
            self.threshold_idxs.append(self.threshPendulum2)
            self.threshold_idxs.append(self.threshPendulum3)
            self.threshold_idxs.append(self.threshPendulum4)


class SixPendulumTypeLstmAe(torch.nn.Module):
    """
    This class contains one neural network for every component type of the technical system.
    Anomalies are detected per component (instance).
    """
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super(SixPendulumTypeLstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary_name = f"experiments/type_modular_6_{enc_hidden_size}_tensorboard"

        self.Pendulum1 = [0,1,2,3,4]
        self.Pendulum2 = [5,6,7,8,9]
        self.Pendulum3 = [10,11,12,13,14]
        self.Pendulum4 = [15,16,17,18,19]
        self.Pendulum5 = [20,21,22,23,24]
        self.Pendulum6 = [25,26,27,28,29]
        self.Pendulum1POS = [0,1]
        self.Pendulum2POS = [5,6]
        self.Pendulum3POS = [10,11]
        self.Pendulum4POS = [15,16]
        self.Pendulum5POS = [20,21]
        self.Pendulum6POS = [25,26]

        
        self.module_idxs = [  # this list tells us, which module number belongs to which module
            [0, 1, 2, 3, 4],  # Pendulum1
            [5, 6, 7, 8, 9],  # Pendulum2
            [10, 11, 12, 13, 14],  # Pendulum3
            [15, 16, 17, 18, 19],  # Pendulum4
            [20,21,22,23,24],  # Pendulum5
            [25,26,27,28,29]  # Pendulum6

        ]

        self.threshold_idxs = []  # here we store the threshold values for anom detect after training

        self.lossdatalist = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1,
            self.Pendulum2,
            self.Pendulum3,
            self.Pendulum4,
            self.Pendulum5,
            self.Pendulum6
        ]

        self.lossdatalistPOS = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1POS,
            self.Pendulum2POS,
            self.Pendulum3POS,
            self.Pendulum4POS,
            self.Pendulum5POS,
            self.Pendulum6POS
        ]

        self.threshPendulum1 = 0.0
        self.threshPendulum2 = 0.0
        self.threshPendulum3 = 0.0
        self.threshPendulum4 = 0.0
        self.threshPendulum5 = 0.0
        self.threshPendulum6 = 0.0
        
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.LstmPendulum1 = LstmAe(enc_input_size=len(self.Pendulum1), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)
        
        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)


    def forward(self, x):
        """ Forward pass of MultiModularLstmAe
        Split the input according to the indices provided by the lists and process them separately.
        """

        inPendulum1 = x[:, self.Pendulum1]  # The valve actuator is the only component directly influenced by the command
        inPendulum2 = x[:, self.Pendulum2]
        inPendulum3 = x[:, self.Pendulum3]
        inPendulum4 = x[:, self.Pendulum4]
        inPendulum5 = x[:, self.Pendulum5]
        inPendulum6 = x[:, self.Pendulum6]
        
        Pendulum1 = self.LstmPendulum1(inPendulum1)
        Pendulum2 = self.LstmPendulum1(inPendulum2)
        Pendulum3 = self.LstmPendulum1(inPendulum3)
        Pendulum4 = self.LstmPendulum1(inPendulum4)
        Pendulum5 = self.LstmPendulum1(inPendulum5)
        Pendulum6 = self.LstmPendulum1(inPendulum6)

        return (Pendulum1, Pendulum2, Pendulum3, Pendulum4, Pendulum5, Pendulum6)
    
    def train_model(self, train_ds, val_ds, test_ds, n_epochs: int):
        """
        train_ds: 
        val_ds: 
        test_ds:
        n_epochs: int value
        """
        writer = SummaryWriter(self.summary_name)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        history = dict(train=[], val=[])  # save train and validation losses
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 10000.0  # arbitrary high loss
        best_epoch = -1
        for epoch in range(1, n_epochs + 1):
            #print(f"Epoch {epoch}")
            self = self.train()
            train_losses = []

            for batch_idx, data in enumerate(train_ds):
                optimizer.zero_grad()
                predictions = self(data)
                temp_train_losses = []
                for idx, _ in enumerate(predictions):
                    temp_train_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]))
                loss = sum(temp_train_losses)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            self = self.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_ds):
                    predictions = self(data)
                    temp_val_losses = []
                    for idx, _ in enumerate(predictions):
                        temp_val_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                    loss = sum(temp_val_losses)
                    val_losses.append(loss)
            mean_train_loss = numpy.mean(train_losses)
            mean_val_loss = numpy.mean(val_losses)
            history['train'].append(mean_train_loss)
            writer.add_scalar('train_loss', mean_train_loss, epoch)
            history['val'].append(mean_val_loss)
            writer.add_scalar('val_loss', mean_val_loss, epoch)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.load_state_dict(best_model_wts)
        self.eval()
        # Losses for each module
        test_losses = []
        test_losses_Pendulum1 = []
        test_losses_Pendulum2 = []
        test_losses_Pendulum3 = []
        test_losses_Pendulum4 = []
        test_losses_Pendulum5 = []
        test_losses_Pendulum6 = []

        test_pairs = []  # remains for compatibility but is unused here
        with torch.no_grad():
            for batch_idx, data in enumerate(test_ds):
                # ToDo: Gen sub-data sets & assign losses. Put in function?
                predictions = self(data)
                temp_test_losses = []
                for idx, _ in enumerate(predictions):
                    temp_test_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                
                test_losses_Pendulum1.append(temp_test_losses[0])
                test_losses_Pendulum2.append(temp_test_losses[1])
                test_losses_Pendulum3.append(temp_test_losses[2])
                test_losses_Pendulum4.append(temp_test_losses[3])
                test_losses_Pendulum5.append(temp_test_losses[4])
                test_losses_Pendulum6.append(temp_test_losses[5])

                loss = sum(temp_test_losses)
                test_losses.append(loss)
                writer.add_scalar('test_loss', loss, batch_idx)

        writer.close()
        mean_test_loss = numpy.mean(test_losses)
        std_test_loss = numpy.std(test_losses)
        mean_test_loss_Pendulum1 = numpy.mean(test_losses_Pendulum1)
        std_test_loss_Pendulum1 = numpy.std(test_losses_Pendulum1)
        mean_test_loss_Pendulum2 = numpy.mean(test_losses_Pendulum2)
        std_test_loss_Pendulum2 = numpy.std(test_losses_Pendulum2)
        mean_test_loss_Pendulum3 = numpy.mean(test_losses_Pendulum3)
        std_test_loss_Pendulum3 = numpy.std(test_losses_Pendulum3)
        mean_test_loss_Pendulum4 = numpy.mean(test_losses_Pendulum4)
        std_test_loss_Pendulum4 = numpy.std(test_losses_Pendulum4)
        mean_test_loss_Pendulum5 = numpy.mean(test_losses_Pendulum5)
        std_test_loss_Pendulum5 = numpy.std(test_losses_Pendulum5)
        mean_test_loss_Pendulum6 = numpy.mean(test_losses_Pendulum6)
        std_test_loss_Pendulum6 = numpy.std(test_losses_Pendulum6)
       
        self.threshPendulum1 = 2.33 * std_test_loss_Pendulum1 + mean_test_loss_Pendulum1
        self.threshold_idxs.append(self.threshPendulum1)
        self.threshPendulum2 = 2.33 * std_test_loss_Pendulum2 + mean_test_loss_Pendulum2
        self.threshold_idxs.append(self.threshPendulum2)
        self.threshPendulum3 = 2.33 * std_test_loss_Pendulum3 + mean_test_loss_Pendulum3
        self.threshold_idxs.append(self.threshPendulum3)
        self.threshPendulum4 = 2.33 * std_test_loss_Pendulum4 + mean_test_loss_Pendulum4
        self.threshold_idxs.append(self.threshPendulum4)
        self.threshPendulum5 = 2.33 * std_test_loss_Pendulum5 + mean_test_loss_Pendulum5
        self.threshold_idxs.append(self.threshPendulum5)
        self.threshPendulum6 = 2.33 * std_test_loss_Pendulum6 + mean_test_loss_Pendulum6
        self.threshold_idxs.append(self.threshPendulum6)
        
        return history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs
    
    # ToDo: Funktion für anom detect
    def detect_anomaly(self, x) -> tuple[bool, list[int]]:
        r"""
        Based on the given input x this function computes residuals for its modules and compares
        this to trained thresholds for mse values. In case of at least one anomaly it returns TRUE
        else it returns FALSE. It also retuns a list with indices, where an anomaly was found
        """
        anomaly = False
        anom_idx = []  # to store the indices of modules which have detected anomalies
        predictions = self(x)
        losses = []
        for idx, _ in enumerate(predictions):
            losses.append(self.criterion(predictions[idx], x[:, self.lossdatalistPOS[idx]]).item())
            if losses[idx] > self.threshold_idxs[idx]:
                anom_idx.append(idx)
        if anom_idx:
            anomaly = True
        return (anomaly, anom_idx)
        
    def get_thresholds(self):
        thresholds = [("Threshold Pendulum1", self.threshPendulum1),
                        ("Threshold Pendulum2", self.threshPendulum2),
                        ("Threshold Pendulum3", self.threshPendulum3),
                        ("Threshold Pendulum4", self.threshPendulum4),
                        ("Threshold Pendulum5", self.threshPendulum5),
                        ("Threshold Pendulum6", self.threshPendulum6)
                        ]
        return thresholds
    
    def export_thresholds(self, filepath):
        """
        
        """
        thresholds = self.get_thresholds()
        with open(filepath, 'wb') as file:
            # A new file will be created
            pickle.dump(thresholds, file)
    
    def load_thresholds(self, filepath):
        """
        
        """
        with open(filepath, 'rb') as file: 
            # Call load method to deserialze
            thresholds = pickle.load(file)  # loads a list of tuples (name, value)
            self.threshPendulum1 = thresholds[0][1]
            self.threshPendulum2 = thresholds[1][1]
            self.threshPendulum3 = thresholds[2][1]
            self.threshPendulum4 = thresholds[3][1]
            self.threshPendulum5 = thresholds[4][1]
            self.threshPendulum6 = thresholds[5][1]


            self.threshold_idxs.append(self.threshPendulum1)
            self.threshold_idxs.append(self.threshPendulum2)
            self.threshold_idxs.append(self.threshPendulum3)
            self.threshold_idxs.append(self.threshPendulum4)
            self.threshold_idxs.append(self.threshPendulum5)
            self.threshold_idxs.append(self.threshPendulum6)


class EightPendulumTypeLstmAe(torch.nn.Module):
    """
    This class contains one neural network for every component type of the technical system.
    Anomalies are detected per component (instance).
    """
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super(EightPendulumTypeLstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary_name = f"experiments/type_modular_8_{enc_hidden_size}_tensorboard"

        self.Pendulum1 = [0,1,2,3,4]
        self.Pendulum2 = [5,6,7,8,9]
        self.Pendulum3 = [10,11,12,13,14]
        self.Pendulum4 = [15,16,17,18,19]
        self.Pendulum5 = [20,21,22,23,24]
        self.Pendulum6 = [25,26,27,28,29]
        self.Pendulum7 = [30,31,32,33,34]
        self.Pendulum8 = [35,36,37,38,39]
        self.Pendulum1POS = [0,1]
        self.Pendulum2POS = [5,6]
        self.Pendulum3POS = [10,11]
        self.Pendulum4POS = [15,16]
        self.Pendulum5POS = [20,21]
        self.Pendulum6POS = [25,26]
        self.Pendulum7POS = [30,31]
        self.Pendulum8POS = [35,36]
        
        self.module_idxs = [  # this list tells us, which module number belongs to which module
            [0, 1, 2, 3, 4],  # Pendulum1
            [5, 6, 7, 8, 9],  # Pendulum2
            [10, 11, 12, 13, 14],  # Pendulum3
            [15, 16, 17, 18, 19],  # Pendulum4
            [20, 21, 22, 23, 24],  # Pendulum5
            [25, 26, 27, 28, 29],  # Pendulum6
            [30, 31, 32, 33, 34],  # Pendulum7
            [35, 36, 37, 38, 39],  # Pendulum8

        ]

        self.threshold_idxs = []  # here we store the threshold values for anom detect after training

        self.lossdatalist = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1,
            self.Pendulum2,
            self.Pendulum3,
            self.Pendulum4,
            self.Pendulum5,
            self.Pendulum6,
            self.Pendulum7,
            self.Pendulum8
        ]

        self.lossdatalistPOS = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1POS,
            self.Pendulum2POS,
            self.Pendulum3POS,
            self.Pendulum4POS,
            self.Pendulum5POS,
            self.Pendulum6POS,
            self.Pendulum7POS,
            self.Pendulum8POS
        ]

        self.threshPendulum1 = 0.0
        self.threshPendulum2 = 0.0
        self.threshPendulum3 = 0.0
        self.threshPendulum4 = 0.0
        self.threshPendulum5 = 0.0
        self.threshPendulum6 = 0.0
        self.threshPendulum7 = 0.0
        self.threshPendulum8 = 0.0
        
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.LstmPendulum1 = LstmAe(enc_input_size=len(self.Pendulum1), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)

        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)


    def forward(self, x):
        """ Forward pass of MultiModularLstmAe
        Split the input according to the indices provided by the lists and process them separately.
        """

        inPendulum1 = x[:, self.Pendulum1]  # The valve actuator is the only component directly influenced by the command
        inPendulum2 = x[:, self.Pendulum2]
        inPendulum3 = x[:, self.Pendulum3]
        inPendulum4 = x[:, self.Pendulum4]
        inPendulum5 = x[:, self.Pendulum5]
        inPendulum6 = x[:, self.Pendulum6]
        inPendulum7 = x[:, self.Pendulum7]
        inPendulum8 = x[:, self.Pendulum8]

        
        Pendulum1 = self.LstmPendulum1(inPendulum1)
        Pendulum2 = self.LstmPendulum1(inPendulum2)
        Pendulum3 = self.LstmPendulum1(inPendulum3)
        Pendulum4 = self.LstmPendulum1(inPendulum4)
        Pendulum5 = self.LstmPendulum1(inPendulum5)
        Pendulum6 = self.LstmPendulum1(inPendulum6)
        Pendulum7 = self.LstmPendulum1(inPendulum7)
        Pendulum8 = self.LstmPendulum1(inPendulum8)

        return (Pendulum1, Pendulum2, Pendulum3, Pendulum4, Pendulum5, Pendulum6, Pendulum7, Pendulum8)
    
    def train_model(self, train_ds, val_ds, test_ds, n_epochs: int):
        """
        train_ds: 
        val_ds: 
        test_ds:
        n_epochs: int value
        """
        writer = SummaryWriter(self.summary_name)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        history = dict(train=[], val=[])  # save train and validation losses
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 10000.0  # arbitrary high loss
        best_epoch = -1
        for epoch in range(1, n_epochs + 1):
            #print(f"Epoch {epoch}")
            self = self.train()
            train_losses = []

            for batch_idx, data in enumerate(train_ds):
                optimizer.zero_grad()
                predictions = self(data)
                temp_train_losses = []
                for idx, _ in enumerate(predictions):
                    temp_train_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]))
                loss = sum(temp_train_losses)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            self = self.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_ds):
                    predictions = self(data)
                    temp_val_losses = []
                    for idx, _ in enumerate(predictions):
                        temp_val_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                    loss = sum(temp_val_losses)
                    val_losses.append(loss)
            mean_train_loss = numpy.mean(train_losses)
            mean_val_loss = numpy.mean(val_losses)
            history['train'].append(mean_train_loss)
            writer.add_scalar('train_loss', mean_train_loss, epoch)
            history['val'].append(mean_val_loss)
            writer.add_scalar('val_loss', mean_val_loss, epoch)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.load_state_dict(best_model_wts)
        self.eval()
        # Losses for each module
        test_losses = []
        test_losses_Pendulum1 = []
        test_losses_Pendulum2 = []
        test_losses_Pendulum3 = []
        test_losses_Pendulum4 = []
        test_losses_Pendulum5 = []
        test_losses_Pendulum6 = []
        test_losses_Pendulum7 = []
        test_losses_Pendulum8 = []

        test_pairs = []  # remains for compatibility but is unused here
        with torch.no_grad():
            for batch_idx, data in enumerate(test_ds):
                # ToDo: Gen sub-data sets & assign losses. Put in function?
                predictions = self(data)
                temp_test_losses = []
                for idx, _ in enumerate(predictions):
                    temp_test_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                
                test_losses_Pendulum1.append(temp_test_losses[0])
                test_losses_Pendulum2.append(temp_test_losses[1])
                test_losses_Pendulum3.append(temp_test_losses[2])
                test_losses_Pendulum4.append(temp_test_losses[3])
                test_losses_Pendulum5.append(temp_test_losses[4])
                test_losses_Pendulum6.append(temp_test_losses[5])
                test_losses_Pendulum7.append(temp_test_losses[6])
                test_losses_Pendulum8.append(temp_test_losses[7])

                loss = sum(temp_test_losses)
                test_losses.append(loss)
                writer.add_scalar('test_loss', loss, batch_idx)

        writer.close()
        mean_test_loss = numpy.mean(test_losses)
        std_test_loss = numpy.std(test_losses)
        mean_test_loss_Pendulum1 = numpy.mean(test_losses_Pendulum1)
        std_test_loss_Pendulum1 = numpy.std(test_losses_Pendulum1)
        mean_test_loss_Pendulum2 = numpy.mean(test_losses_Pendulum2)
        std_test_loss_Pendulum2 = numpy.std(test_losses_Pendulum2)
        mean_test_loss_Pendulum3 = numpy.mean(test_losses_Pendulum3)
        std_test_loss_Pendulum3 = numpy.std(test_losses_Pendulum3)
        mean_test_loss_Pendulum4 = numpy.mean(test_losses_Pendulum4)
        std_test_loss_Pendulum4 = numpy.std(test_losses_Pendulum4)
        mean_test_loss_Pendulum5 = numpy.mean(test_losses_Pendulum5)
        std_test_loss_Pendulum5 = numpy.std(test_losses_Pendulum5)
        mean_test_loss_Pendulum6 = numpy.mean(test_losses_Pendulum6)
        std_test_loss_Pendulum6 = numpy.std(test_losses_Pendulum6)
        mean_test_loss_Pendulum7 = numpy.mean(test_losses_Pendulum7)
        std_test_loss_Pendulum7 = numpy.std(test_losses_Pendulum7)
        mean_test_loss_Pendulum8 = numpy.mean(test_losses_Pendulum8)
        std_test_loss_Pendulum8 = numpy.std(test_losses_Pendulum8)
       
        self.threshPendulum1 = 2.33 * std_test_loss_Pendulum1 + mean_test_loss_Pendulum1
        self.threshold_idxs.append(self.threshPendulum1)
        self.threshPendulum2 = 2.33 * std_test_loss_Pendulum2 + mean_test_loss_Pendulum2
        self.threshold_idxs.append(self.threshPendulum2)
        self.threshPendulum3 = 2.33 * std_test_loss_Pendulum3 + mean_test_loss_Pendulum3
        self.threshold_idxs.append(self.threshPendulum3)
        self.threshPendulum4 = 2.33 * std_test_loss_Pendulum4 + mean_test_loss_Pendulum4
        self.threshold_idxs.append(self.threshPendulum4)
        self.threshPendulum5 = 2.33 * std_test_loss_Pendulum5 + mean_test_loss_Pendulum5
        self.threshold_idxs.append(self.threshPendulum5)
        self.threshPendulum6 = 2.33 * std_test_loss_Pendulum6 + mean_test_loss_Pendulum6
        self.threshold_idxs.append(self.threshPendulum6)
        self.threshPendulum7 = 2.33 * std_test_loss_Pendulum7 + mean_test_loss_Pendulum7
        self.threshold_idxs.append(self.threshPendulum7)
        self.threshPendulum8 = 2.33 * std_test_loss_Pendulum8 + mean_test_loss_Pendulum8
        self.threshold_idxs.append(self.threshPendulum8)
        
        return history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs
    
    # ToDo: Funktion für anom detect
    def detect_anomaly(self, x) -> tuple[bool, list[int]]:
        r"""
        Based on the given input x this function computes residuals for its modules and compares
        this to trained thresholds for mse values. In case of at least one anomaly it returns TRUE
        else it returns FALSE. It also retuns a list with indices, where an anomaly was found
        """
        anomaly = False
        anom_idx = []  # to store the indices of modules which have detected anomalies
        predictions = self(x)
        losses = []
        for idx, _ in enumerate(predictions):
            losses.append(self.criterion(predictions[idx], x[:, self.lossdatalistPOS[idx]]).item())
            if losses[idx] > self.threshold_idxs[idx]:
                anom_idx.append(idx)
        if anom_idx:
            anomaly = True
        return (anomaly, anom_idx)
        
    def get_thresholds(self):
        thresholds = [("Threshold Pendulum1", self.threshPendulum1),
                        ("Threshold Pendulum2", self.threshPendulum2),
                        ("Threshold Pendulum3", self.threshPendulum3),
                        ("Threshold Pendulum4", self.threshPendulum4),
                        ("Threshold Pendulum5", self.threshPendulum5),
                        ("Threshold Pendulum6", self.threshPendulum6),
                        ("Threshold Pendulum7", self.threshPendulum7),
                        ("Threshold Pendulum8", self.threshPendulum8)
                        ]
        return thresholds
    
    def export_thresholds(self, filepath):
        """
        
        """
        thresholds = self.get_thresholds()
        with open(filepath, 'wb') as file:
            # A new file will be created
            pickle.dump(thresholds, file)
    
    def load_thresholds(self, filepath):
        """
        
        """
        with open(filepath, 'rb') as file: 
            # Call load method to deserialze
            thresholds = pickle.load(file)  # loads a list of tuples (name, value)
            self.threshPendulum1 = thresholds[0][1]
            self.threshPendulum2 = thresholds[1][1]
            self.threshPendulum3 = thresholds[2][1]
            self.threshPendulum4 = thresholds[3][1]
            self.threshPendulum5 = thresholds[4][1]
            self.threshPendulum6 = thresholds[5][1]
            self.threshPendulum7 = thresholds[6][1]
            self.threshPendulum8 = thresholds[7][1]


            self.threshold_idxs.append(self.threshPendulum1)
            self.threshold_idxs.append(self.threshPendulum2)
            self.threshold_idxs.append(self.threshPendulum3)
            self.threshold_idxs.append(self.threshPendulum4)
            self.threshold_idxs.append(self.threshPendulum5)
            self.threshold_idxs.append(self.threshPendulum6)
            self.threshold_idxs.append(self.threshPendulum7)
            self.threshold_idxs.append(self.threshPendulum8)


class TenPendulumTypeLstmAe(torch.nn.Module):
    """
    This class contains one neural network for every component type of the technical system.
    Anomalies are detected per component (instance).
    """
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super(TenPendulumTypeLstmAe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary_name = f"experiments/type_modular_10_{enc_hidden_size}_tensorboard"

        self.Pendulum1 = [0,1,2,3,4]
        self.Pendulum2 = [5,6,7,8,9]
        self.Pendulum3 = [10,11,12,13,14]
        self.Pendulum4 = [15,16,17,18,19]
        self.Pendulum5 = [20,21,22,23,24]
        self.Pendulum6 = [25,26,27,28,29]
        self.Pendulum7 = [30,31,32,33,34]
        self.Pendulum8 = [35,36,37,38,39]
        self.Pendulum9 = [40,41,42,43,44]
        self.Pendulum10 = [45,46,47,48,49]
        
        self.Pendulum1POS = [0,1]
        self.Pendulum2POS = [5,6]
        self.Pendulum3POS = [10,11]
        self.Pendulum4POS = [15,16]
        self.Pendulum5POS = [20,21]
        self.Pendulum6POS = [25,26]
        self.Pendulum7POS = [30,31]
        self.Pendulum8POS = [35,36]
        self.Pendulum9POS = [40,41]
        self.Pendulum10POS = [45,46]

        self.module_idxs = [  # this list tells us, which module number belongs to which module
            [0, 1, 2, 3, 4],  # Pendulum1
            [5, 6, 7, 8, 9],  # Pendulum2
            [10, 11, 12, 13, 14],  # Pendulum3
            [15, 16, 17, 18, 19],  # Pendulum4
            [20, 21, 22, 23, 24],  # Pendulum5
            [25, 26, 27, 28, 29],  # Pendulum6
            [30, 31, 32, 33, 34],  # Pendulum7
            [35, 36, 37, 38, 39],  # Pendulum8
            [40, 41, 42, 43, 44],
            [45, 46, 47, 48, 49]

        ]

        self.threshold_idxs = []  # here we store the threshold values for anom detect after training

        self.lossdatalist = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1,
            self.Pendulum2,
            self.Pendulum3,
            self.Pendulum4,
            self.Pendulum5,
            self.Pendulum6,
            self.Pendulum7,
            self.Pendulum8,
            self.Pendulum9,
            self.Pendulum10
        ]

        self.lossdatalistPOS = [  # this list represents the order of module outputs during inference, do not change
            self.Pendulum1POS,
            self.Pendulum2POS,
            self.Pendulum3POS,
            self.Pendulum4POS,
            self.Pendulum5POS,
            self.Pendulum6POS,
            self.Pendulum7POS,
            self.Pendulum8POS,
            self.Pendulum9POS,
            self.Pendulum10POS
        ]

        self.threshPendulum1 = 0.0
        self.threshPendulum2 = 0.0
        self.threshPendulum3 = 0.0
        self.threshPendulum4 = 0.0
        self.threshPendulum5 = 0.0
        self.threshPendulum6 = 0.0
        self.threshPendulum7 = 0.0
        self.threshPendulum8 = 0.0
        self.threshPendulum9 = 0.0
        self.threshPendulum10 = 0.0
        
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.LstmPendulum1 = LstmAe(enc_input_size=len(self.Pendulum1), enc_hidden_size=self.enc_hidden_size, dec_hidden_size=self.dec_hidden_size, dec_output_size=2)

        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)


    def forward(self, x):
        """ Forward pass of MultiModularLstmAe
        Split the input according to the indices provided by the lists and process them separately.
        """

        inPendulum1 = x[:, self.Pendulum1]  # The valve actuator is the only component directly influenced by the command
        inPendulum2 = x[:, self.Pendulum2]
        inPendulum3 = x[:, self.Pendulum3]
        inPendulum4 = x[:, self.Pendulum4]
        inPendulum5 = x[:, self.Pendulum5]
        inPendulum6 = x[:, self.Pendulum6]
        inPendulum7 = x[:, self.Pendulum7]
        inPendulum8 = x[:, self.Pendulum8]
        inPendulum9 = x[:, self.Pendulum9]
        inPendulum10 = x[:, self.Pendulum10]

        Pendulum1 = self.LstmPendulum1(inPendulum1)
        Pendulum2 = self.LstmPendulum1(inPendulum2)
        Pendulum3 = self.LstmPendulum1(inPendulum3)
        Pendulum4 = self.LstmPendulum1(inPendulum4)
        Pendulum5 = self.LstmPendulum1(inPendulum5)
        Pendulum6 = self.LstmPendulum1(inPendulum6)
        Pendulum7 = self.LstmPendulum1(inPendulum7)
        Pendulum8 = self.LstmPendulum1(inPendulum8)
        Pendulum9 = self.LstmPendulum1(inPendulum9)
        Pendulum10 = self.LstmPendulum1(inPendulum10)

        return (Pendulum1, Pendulum2, Pendulum3, Pendulum4, Pendulum5, Pendulum6, Pendulum7, Pendulum8, Pendulum9, Pendulum10)
    
    def train_model(self, train_ds, val_ds, test_ds, n_epochs: int):
        """
        train_ds: 
        val_ds: 
        test_ds:
        n_epochs: int value
        """
        writer = SummaryWriter(self.summary_name)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        history = dict(train=[], val=[])  # save train and validation losses
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 10000.0  # arbitrary high loss
        best_epoch = -1
        for epoch in range(1, n_epochs + 1):
            #print(f"Epoch {epoch}")
            self = self.train()
            train_losses = []

            for batch_idx, data in enumerate(train_ds):
                optimizer.zero_grad()
                predictions = self(data)
                temp_train_losses = []
                for idx, _ in enumerate(predictions):
                    temp_train_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]))
                loss = sum(temp_train_losses)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            self = self.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_ds):
                    predictions = self(data)
                    temp_val_losses = []
                    for idx, _ in enumerate(predictions):
                        temp_val_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                    loss = sum(temp_val_losses)
                    val_losses.append(loss)
            mean_train_loss = numpy.mean(train_losses)
            mean_val_loss = numpy.mean(val_losses)
            history['train'].append(mean_train_loss)
            writer.add_scalar('train_loss', mean_train_loss, epoch)
            history['val'].append(mean_val_loss)
            writer.add_scalar('val_loss', mean_val_loss, epoch)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.load_state_dict(best_model_wts)
        self.eval()
        # Losses for each module
        test_losses = []
        test_losses_Pendulum1 = []
        test_losses_Pendulum2 = []
        test_losses_Pendulum3 = []
        test_losses_Pendulum4 = []
        test_losses_Pendulum5 = []
        test_losses_Pendulum6 = []
        test_losses_Pendulum7 = []
        test_losses_Pendulum8 = []
        test_losses_Pendulum9 = []
        test_losses_Pendulum10 = []

        test_pairs = []  # remains for compatibility but is unused here
        with torch.no_grad():
            for batch_idx, data in enumerate(test_ds):
                # ToDo: Gen sub-data sets & assign losses. Put in function?
                predictions = self(data)
                temp_test_losses = []
                for idx, _ in enumerate(predictions):
                    temp_test_losses.append(criterion(predictions[idx], data[:, self.lossdatalistPOS[idx]]).item())
                
                test_losses_Pendulum1.append(temp_test_losses[0])
                test_losses_Pendulum2.append(temp_test_losses[1])
                test_losses_Pendulum3.append(temp_test_losses[2])
                test_losses_Pendulum4.append(temp_test_losses[3])
                test_losses_Pendulum5.append(temp_test_losses[4])
                test_losses_Pendulum6.append(temp_test_losses[5])
                test_losses_Pendulum7.append(temp_test_losses[6])
                test_losses_Pendulum8.append(temp_test_losses[7])
                test_losses_Pendulum9.append(temp_test_losses[8])
                test_losses_Pendulum10.append(temp_test_losses[9])

                loss = sum(temp_test_losses)
                test_losses.append(loss)
                writer.add_scalar('test_loss', loss, batch_idx)

        writer.close()
        mean_test_loss = numpy.mean(test_losses)
        std_test_loss = numpy.std(test_losses)
        mean_test_loss_Pendulum1 = numpy.mean(test_losses_Pendulum1)
        std_test_loss_Pendulum1 = numpy.std(test_losses_Pendulum1)
        mean_test_loss_Pendulum2 = numpy.mean(test_losses_Pendulum2)
        std_test_loss_Pendulum2 = numpy.std(test_losses_Pendulum2)
        mean_test_loss_Pendulum3 = numpy.mean(test_losses_Pendulum3)
        std_test_loss_Pendulum3 = numpy.std(test_losses_Pendulum3)
        mean_test_loss_Pendulum4 = numpy.mean(test_losses_Pendulum4)
        std_test_loss_Pendulum4 = numpy.std(test_losses_Pendulum4)
        mean_test_loss_Pendulum5 = numpy.mean(test_losses_Pendulum5)
        std_test_loss_Pendulum5 = numpy.std(test_losses_Pendulum5)
        mean_test_loss_Pendulum6 = numpy.mean(test_losses_Pendulum6)
        std_test_loss_Pendulum6 = numpy.std(test_losses_Pendulum6)
        mean_test_loss_Pendulum7 = numpy.mean(test_losses_Pendulum7)
        std_test_loss_Pendulum7 = numpy.std(test_losses_Pendulum7)
        mean_test_loss_Pendulum8 = numpy.mean(test_losses_Pendulum8)
        std_test_loss_Pendulum8 = numpy.std(test_losses_Pendulum8)
        mean_test_loss_Pendulum9 = numpy.mean(test_losses_Pendulum9)
        std_test_loss_Pendulum9 = numpy.std(test_losses_Pendulum9)
        mean_test_loss_Pendulum10 = numpy.mean(test_losses_Pendulum10)
        std_test_loss_Pendulum10 = numpy.std(test_losses_Pendulum10)
       
        self.threshPendulum1 = 2.33 * std_test_loss_Pendulum1 + mean_test_loss_Pendulum1
        self.threshold_idxs.append(self.threshPendulum1)
        self.threshPendulum2 = 2.33 * std_test_loss_Pendulum2 + mean_test_loss_Pendulum2
        self.threshold_idxs.append(self.threshPendulum2)
        self.threshPendulum3 = 2.33 * std_test_loss_Pendulum3 + mean_test_loss_Pendulum3
        self.threshold_idxs.append(self.threshPendulum3)
        self.threshPendulum4 = 2.33 * std_test_loss_Pendulum4 + mean_test_loss_Pendulum4
        self.threshold_idxs.append(self.threshPendulum4)
        self.threshPendulum5 = 2.33 * std_test_loss_Pendulum5 + mean_test_loss_Pendulum5
        self.threshold_idxs.append(self.threshPendulum5)
        self.threshPendulum6 = 2.33 * std_test_loss_Pendulum6 + mean_test_loss_Pendulum6
        self.threshold_idxs.append(self.threshPendulum6)
        self.threshPendulum7 = 2.33 * std_test_loss_Pendulum7 + mean_test_loss_Pendulum7
        self.threshold_idxs.append(self.threshPendulum7)
        self.threshPendulum8 = 2.33 * std_test_loss_Pendulum8 + mean_test_loss_Pendulum8
        self.threshold_idxs.append(self.threshPendulum8)
        self.threshPendulum9 = 2.33 * std_test_loss_Pendulum9 + mean_test_loss_Pendulum9
        self.threshold_idxs.append(self.threshPendulum9)
        self.threshPendulum10 = 2.33 * std_test_loss_Pendulum10 + mean_test_loss_Pendulum10
        self.threshold_idxs.append(self.threshPendulum10)
        
        return history, best_loss, best_epoch, best_model_wts, mean_test_loss, std_test_loss, test_pairs
    
    # ToDo: Funktion für anom detect
    def detect_anomaly(self, x) -> tuple[bool, list[int]]:
        r"""
        Based on the given input x this function computes residuals for its modules and compares
        this to trained thresholds for mse values. In case of at least one anomaly it returns TRUE
        else it returns FALSE. It also retuns a list with indices, where an anomaly was found
        """
        anomaly = False
        anom_idx = []  # to store the indices of modules which have detected anomalies
        predictions = self(x)
        losses = []
        for idx, _ in enumerate(predictions):
            losses.append(self.criterion(predictions[idx], x[:, self.lossdatalistPOS[idx]]).item())
            if losses[idx] > self.threshold_idxs[idx]:
                anom_idx.append(idx)
        if anom_idx:
            anomaly = True
        return (anomaly, anom_idx)
        
    def get_thresholds(self):
        thresholds = [("Threshold Pendulum1", self.threshPendulum1),
                        ("Threshold Pendulum2", self.threshPendulum2),
                        ("Threshold Pendulum3", self.threshPendulum3),
                        ("Threshold Pendulum4", self.threshPendulum4),
                        ("Threshold Pendulum5", self.threshPendulum5),
                        ("Threshold Pendulum6", self.threshPendulum6),
                        ("Threshold Pendulum7", self.threshPendulum7),
                        ("Threshold Pendulum8", self.threshPendulum8),
                        ("Threshold Pendulum9", self.threshPendulum9),
                        ("Threshold Pendulum10", self.threshPendulum10)
                        ]
        return thresholds
    
    def export_thresholds(self, filepath):
        """
        
        """
        thresholds = self.get_thresholds()
        with open(filepath, 'wb') as file:
            # A new file will be created
            pickle.dump(thresholds, file)
    
    def load_thresholds(self, filepath):
        """
        
        """
        with open(filepath, 'rb') as file: 
            # Call load method to deserialze
            thresholds = pickle.load(file)  # loads a list of tuples (name, value)
            self.threshPendulum1 = thresholds[0][1]
            self.threshPendulum2 = thresholds[1][1]
            self.threshPendulum3 = thresholds[2][1]
            self.threshPendulum4 = thresholds[3][1]
            self.threshPendulum5 = thresholds[4][1]
            self.threshPendulum6 = thresholds[5][1]
            self.threshPendulum7 = thresholds[6][1]
            self.threshPendulum8 = thresholds[7][1]
            self.threshPendulum9 = thresholds[8][1]
            self.threshPendulum10 = thresholds[9][1]


            self.threshold_idxs.append(self.threshPendulum1)
            self.threshold_idxs.append(self.threshPendulum2)
            self.threshold_idxs.append(self.threshPendulum3)
            self.threshold_idxs.append(self.threshPendulum4)
            self.threshold_idxs.append(self.threshPendulum5)
            self.threshold_idxs.append(self.threshPendulum6)
            self.threshold_idxs.append(self.threshPendulum7)
            self.threshold_idxs.append(self.threshPendulum8)
            self.threshold_idxs.append(self.threshPendulum9)
            self.threshold_idxs.append(self.threshPendulum10)