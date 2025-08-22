import torch 
import torch.nn as nn


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 4e-4

    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_data, test_data, config):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        pass

    def save_checkpoint(self):
        pass
        
    def train(self):
        pass