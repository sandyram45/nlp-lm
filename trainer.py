import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.utils.dataloader as Dataloader

class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 4e-4

    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_data, test_data, config, device):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.device = device

    def save_checkpoint(self):
        pass
        
    def train(self):
        config = self.config
        model = self.model
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

        def run_epoch(is_train):
            data = self.train_data if is_train else self.test_data
            loader = Dataloader(data, batch_size=config.batch_size, num_workers=config.num_workers)

            progress_bar  =  tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for idx, (x, y) in progress_bar:
                x = x.to(self.device)
                y = y.to(self.device)
