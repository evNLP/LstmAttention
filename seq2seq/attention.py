import torch.nn as nn
import torch
import torch.nn.functional as F


class DotAttention(nn.Module):
    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, encoded_outputs, decoder_output):
        score = torch.bmm(encoded_outputs, decoder_output.unsqueeze(2)).squeeze()
        score = torch.softmax(score.float(), dim=1)
        
        return torch.bmm(score.unsqueeze(1).float(), encoded_outputs.float()).squeeze()
    

class LuAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LuAttention, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, encoded_outputs, decoder_output):
        score = self._get_scores(encoded_outputs.float(), decoder_output.float())
        return torch.bmm(
                        score.unsqueeze(1).float(),
                        encoded_outputs.float()
                ).squeeze()
    
    def _get_scores(self, encoded_outputs, decoder_output):
        ew = torch.matmul(encoded_outputs, self.W)
        score = torch.bmm(ew, decoder_output.unsqueeze(2)).squeeze()
        return torch.softmax(score, dim=1)
    

class BaAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BaAttention, self).__init__()
        self.W1 = nn.Parameter(torch.randn(2*hidden_dim, 2*hidden_dim))
        self.w2 = nn.Parameter(torch.randn(1, 2*hidden_dim))
        self.hidden_dim = hidden_dim

    def forward(self, encoded_outputs, decoder_output):
        score = self._get_scores(encoded_outputs.float(), decoder_output.float())
        
        return torch.bmm(
                        score.unsqueeze(1).float(),
                        encoded_outputs.float()
                ).squeeze()
    
    def _get_scores(self, encoded_outputs, decoder_output):
        decoder_resized_output = decoder_output.unsqueeze(1).repeat(1, encoded_outputs.shape[1], 1)
        h = torch.cat((encoded_outputs, decoder_resized_output), dim=2)

        batch_size = encoded_outputs.shape[0]
        h = h.reshape((batch_size, self.hidden_dim*2, -1))
        w1_batch = self.W1.unsqueeze(0).repeat(batch_size, 1, 1)

        wh = torch.bmm(w1_batch, h)
        wh = torch.tanh(wh)

        w2_batch = self.w2.unsqueeze(0).repeat(batch_size, 1, 1)
        score = torch.bmm(w2_batch, wh).squeeze().reshape(batch_size, -1)

        return torch.softmax(score, dim=1)