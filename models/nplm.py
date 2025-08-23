import torch
import torch.nn as nn
from torch.nn import functional as F


class NPLM(nn.Module):
    '''
    Neural Probablistic Language Model  ~ 2003
    e = [e1, e2, e3, ... , en] where n  - sequence length
    h = f(W_e e + b1) where f - activation function
    final_output = softmax(W_x x + b2)
    '''

    def __init__(self, mconfig):
        super().__init__()
        self.vocab_size = mconfig.vocab_size
        self.seq_len = mconfig.seq_len
        self.embed_size = mconfig.embed_size
        self.hidden_size = mconfig.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.h = nn.Linear(self.seq_len * self.embed_size, self.hidden_size)
        self.h_to_logits = nn.Linear(self.hidden_size, self.vocab_size)


    def forward(self,x):
        '''
        x would be a Tensor of shape (batch size, seq_len)
        Lookup the embedding vector for each word.
        In NPLM, we concat the word vectors together before we pass it to the linear layer
        Flow: 
        x (batch size, seq len)
        -> embedding lookup (batch size, seq len, embedding size) (reshape to (batch size,  seq len *  embedding size))
        -> hidden linear layer (batch size, hidden size )
        -> activation (batch size, hidden size) 
        -> Final Linear layer (batch size, vocab size)
        '''
        batch_size = x.size()[0]
        x = self.embedding(x)
        x = x.view(batch_size, self.seq_len * self.embed_size)
        x = self.h(x)
        x = F.relu(x)
        logits = self.h_to_logits(x)
        return logits