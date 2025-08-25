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
 
 # hidden_curr = tanh(W_e * x + b_e + W_h * h_prev + b_h)
class RNN(nn.Module):
  def __init__(self,hidden_size, context_size, vocab_size, embed_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.context_size = context_size
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
    self.block = RNNState(self.hidden_size, self.embed_size)


  def forward(self, x):
    x = self.embedding(x)
    self.h_0 = torch.randn((x.size()[0], self.hidden_size))
    #print(f"Input shape after embedding lookup: {x.shape}")
    # batch_size, context_size, embed_size


    # shape (batch_size, hidden_size)

    '''
    For one iteration
    x -> batch_size , 1, embed_size && h_prev -> batch_size , hidden_size
    W_e -> hidden_size * embed_size
    -> input_x = self.input_fc(x) -> x * W_e.T + b_e -> (batch_size, embed_size) * (embed_size, hidden_size) + (hidden_size) -> (batch_size, hidden_size)


    h_prev -> (batch_size, hidden_size)
    W_h -> (hidden_size, hidden_size)
    -> h = self.hidden(h_prev) -> h_prev * W_h.T + b_h -> (batch_size, hidden_size)  * (hidden_size, hidden_size) + (hidden_size) -> (batch_size, hidden_size)

    To concat before we pass it to the activation function, we need to match the hidden's shape - (batch_size, 1, hidden_size) - can be ignored

    output = F.tanh(input_x + h) -> (batch_size, hidden_size)

    Now, after all the iterations, we need to stack it to one variable so that we can use it for loss calculation

    Note: We have assumed the num_layers as 1. Need to extend the thought to variable input of num_layers

    '''
    main_output = None
    h_prev = self.h_0
    for i in range(self.context_size):

    # each batch can run parallel
    # Take word wise by looping the context length and pass it to the block
    # x[:,i,:] - take the ith word of all batches shape - (batch_size, 1, embed_size) - is it right?

      output, h_curr = self.block(x[:,i,:] , h_prev)
      h_prev = h_curr
      output = torch.unsqueeze(output, dim=1)
      #print(i, output.shape)
      h_prev = h_curr
      if main_output == None:
        main_output = output
      else:
       main_output = torch.cat((main_output, output), dim=1)
    #print(f"The main output's shape : {main_output.shape}")
    #print(f"The last hidden output's shape : {h_prev.shape}")
    # return main_output, h_prev # here h_prev will have the last hidden state
    # We need to have all the output as single variable of size (batch_size, context_size, hidden_size)
    return main_output, h_prev
   

