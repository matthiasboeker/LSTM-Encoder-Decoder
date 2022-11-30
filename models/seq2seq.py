from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from tqdm.auto import tqdm
import numpy as np

from utils.plotting_funcs import plot_training_loss, plot_ts_result_eval


class Encoder(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.n_hidden, num_layers=self.n_layers,
                            batch_first=True, dropout=dropout)
        self.linear = nn.Linear(self.n_features, out_features=1)

    def forward(self, batches):
        ht = torch.zeros(self.n_layers, batches.shape[0], self.n_hidden, dtype=torch.float).requires_grad_()
        ct = torch.zeros(self.n_layers, batches.shape[0], self.n_hidden, dtype=torch.float).requires_grad_()
        lstm_out, (h_out, c_out) = self.lstm(batches, (ht, ct))
        last_observation = batches[:, -1, :]  # shape: (batch, feat)
        decoder_input = self.linear(last_observation).unsqueeze(1)
        return decoder_input, (h_out, c_out)


class Decoder(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers=1, dropout=0):
        super(Decoder, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.n_hidden, num_layers=self.n_layers,
                            batch_first=True, dropout=dropout)
        self.linear_out = nn.Linear(self.n_hidden, out_features=1)

    def forward(self, batches, hidden_states):
        lstm_out, hidden = self.lstm(batches, hidden_states)
        out = self.linear_out(lstm_out.squeeze(1))
        return out, hidden


class SEQ2SEQ(nn.Module):
    def __init__(self, n_features, hidden_size, loss_function, target_len,  num_layers=1, dropout=0):

        super(SEQ2SEQ, self).__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loss_function = loss_function
        self.target_len = target_len

        self.encoder = Encoder(n_features=self.n_features, n_hidden=hidden_size, n_layers=num_layers, dropout=dropout)
        self.decoder = Decoder(n_features=self.n_features, n_hidden=hidden_size, n_layers=num_layers, dropout=dropout)

    def train_model(self, training_loader, validation_loader, n_epochs, learning_rate, l2_reg, plot_loss: Path):

        # initialize array of losses
        training_losses = np.full(n_epochs, np.nan)
        validation_losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=l2_reg)

        with trange(n_epochs) as tr:
            for it in tr:

                batch_training_loss = 0.
                self.train(True)
                for X, y in training_loader:
                    # zero the gradient
                    optimizer.zero_grad()
                    predictions = self.predict(X)
                    # compute the loss
                    loss = self.loss_function(predictions, y)
                    batch_training_loss += loss.item()

                    # backpropagation
                    loss.backward()
                    optimizer.step()
                # loss for epoch
                batch_training_loss /= X.shape[0]
                training_losses[it] = batch_training_loss

                batch_validation_loss = 0.
                self.train(False)
                for X, y in validation_loader:

                    predictions = self.predict(X)
                    # compute the loss
                    loss = self.loss_function(predictions, y)
                    batch_validation_loss += loss.item()
                # loss for epoch
                batch_validation_loss /= X.shape[0]
                validation_losses[it] = batch_validation_loss

                # progress bar
                tr.set_postfix(train_loss="{0:.3f}".format(batch_training_loss),
                               val_loss="{0:.3f}".format(batch_validation_loss))

            if plot_loss:
                plot_training_loss(training_losses, validation_losses, plot_loss)

        return None

    def predict(self, input_batch):
        predictions = []
        decoder_input, (ht, ct) = self.encoder(input_batch)
        decoder_hidden = (ht, ct)

        for _ in range(self.target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output.unsqueeze(1)
            predictions.append(decoder_output)
        return torch.cat(predictions, dim=1)

    def evaluate(self, data_loader,  plot_results: Path, reversed_transformer=None):
        batch_loss = 0.
        loss_list = []
        prediction = []
        gt = []
        with tqdm(data_loader) as tr:
            for X, y in tr:
                pred = self.predict(X)
                prediction.append(pred[:,-1])
                gt.append(y[:,-1])
                loss = self.loss_function(pred, y)
                batch_loss += loss.item()
                batch_loss /= X.shape[0]
                loss_list.append(batch_loss)
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))

            predictions = torch.cat(prediction, dim=0).detach().numpy()
            ground_truth = torch.cat(gt, dim=0)
            if reversed_transformer:
                predictions = reversed_transformer.reverse_transform(predictions)
                ground_truth = reversed_transformer.reverse_transform(ground_truth)
            if plot_results:
                plot_ts_result_eval(ground_truth, predictions, plot_results)
        return None
