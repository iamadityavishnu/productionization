import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(RMM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedding = self.embedding(text)
        output, hidden = self.rnn(embedding)
        assert torch.equal(output[1, :, :], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))