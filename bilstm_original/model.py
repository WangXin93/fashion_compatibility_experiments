import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import inception


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, need_rep=False):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.need_rep = need_rep

        cnn = inception.inception_v3(pretrained=True)
        cnn.fc = nn.Linear(cnn.fc.in_features, embed_size)
        nn.init.xavier_uniform_(cnn.fc.weight)
        nn.init.constant_(cnn.fc.bias, 0)
        self.cnn = cnn
        self.dropout = nn.Dropout(p=0.3)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        if self.cnn.training:
            features, representations, _ = self.cnn(images)
        else:
            features, representations = self.cnn(images)

        if self.need_rep:
            return features, representations.squeeze()
        else:
            return features


class GRUModel(nn.Module):
    # Our model

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 device,
                 n_layers=1,
                 bidirectional=True):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.n_directions = int(bidirectional) + 1

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return hidden.to(self.device)

    def forward(self, input, seq_lengths):
        batch_size = input.size(0)

        # Make a hidden
        hidden = self._init_hidden(batch_size)

        # Pack them up nicely
        gru_input = pack_padded_sequence(
            input, seq_lengths.data.cpu().numpy(), batch_first=True)

        # To compact weights again call flatten_parameters().
        self.gru.flatten_parameters()
        output, hidden = self.gru(gru_input, hidden)

        return output


class LSTMModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 device,
                 n_layers=1,
                 bidirectional=True):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.n_directions = int(bidirectional) + 1
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=0.3)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.1)

    def _init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers*self.n_directions,
                            batch_size,
                            self.hidden_size).to(self.device),
                torch.zeros(self.n_layers*self.n_directions,
                            batch_size,
                            self.hidden_size).to(self.device))

    def forward(self, input, seq_lengths):
        batch_size = input.size(0)
        hidden = self._init_hidden(batch_size)
        lstm_input = pack_padded_sequence(
            input, seq_lengths.data.cpu().numpy(), batch_first=True)

        self.lstm.flatten_parameters()
        output, hidden = self.lstm(lstm_input, hidden)

        return output[0]
