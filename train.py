from model import *
from data import create_dataloader
from utils import load_constants
import argparse
from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_path: str = 'data/train.txt'
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
    d_model: int = 128
    num_heads: int = 4
    num_blocks: int = 2
    dropout_rate: float = 0.2
    model_type: str = 'LSTM'  # or 'Transformer'


def train_epoch(model, dataloader, criterion, optimizer, device):
    pass

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path):
    pass

def main():

    configs = TrainConfig()
    

    pass