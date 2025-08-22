import torch
import torch.nn as nn
from torch.nn import functional as F


class RNNState(nn.Module):

  def __init__(self, hidden_size, embed_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.embed_size = embed_size
    self.input_fc = nn.Linear(self.embed_size, self.hidden_size)
    self.hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

  def forward(self,x, h_prev):
    input_x = self.input_fc(x)
    #print(f"input block - shape: {input_x.shape}")
    h = self.hidden(h_prev)
    #print(f"hidden block - shape: {h.shape}")
    output = F.tanh(input_x + h)
    return output, h
 