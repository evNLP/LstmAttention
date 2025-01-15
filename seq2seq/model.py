import torch.nn as nn
import torch
import torch.nn.functional as F

import torchtext

import random



class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__() 
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x, hidden=None, cell=None):
        if hidden is None:
            output, (hidden, cell) = self.rnn(x)

        else:
            output, (hidden, cell) = self.rnn(x, (hidden, cell))

        return output, (hidden, cell)
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention, output_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder                           
        self.es_embeddings = torchtext.vocab.FastText(language='es')
        self.M = self.es_embeddings.vectors
        self.M = torch.cat((self.M, torch.zeros((4, self.M.shape[1]))), 0)
        self.attention = attention
        self.fc_out = nn.Linear(decoder.hidden_dim, output_dim)

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        target_len = target.shape[1]
        batch_size = target.shape[0]

        outputs = torch.zeros(batch_size, target_len, 985671)

        enc_outputs, (hidden, cell) = self.encoder(source)
        x = target[:, 0, :]

        for t in range(1, target_len):
            output, (hidden, cell) = self.decoder(x.unsqueeze(1), hidden, cell)
            output = output.squeeze(1)

            attention_vect = self.attention(enc_outputs, output)
            output = attention_vect + output

            output = self.fc_out(output)
            
            outputs[:, t, :] = output.squeeze(1)

            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force:
                x = target[:, t, :]
            else:
                x = torch.matmul(output.squeeze(1), self.M)

        return outputs