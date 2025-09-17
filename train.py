from model import *
from data import TextDataset, TextAudioDataset, create_dataloader
from utils import *
import argparse
from dataclasses import dataclass

@dataclass
class TrainConfig:

    # data related
    data_path: str = 'data/train.txt'
    asr_data_path: str = 'data/asr_train.txt'  # for text + audio model

    # training related
    device: str = 'cuda'  # or 'cpu'
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001

    # model related
    d_model: int = 128
    num_heads: int = 4
    num_blocks: int = 2
    dropout_rate: float = 0.2
    model_type: str = 'Transformer'  # or  'LSTM' 'Transformer'

    # output related
    save_freq: int = 1
    eval_freq: int = 1
    save_path: str = 'model_checkpoints/'


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    training_loss = 0.0
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    return training_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            validation_loss += loss.item()

    return validation_loss / len(dataloader)

def train(config, model, train_loader, val_loader, criterion, optimizer):

    model.to(config.device)
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Training Loss: {train_loss:.4f}")

        if (epoch + 1) % config.eval_freq == 0:
            val_loss = evaluate(model, val_loader, criterion, config.device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{config.save_path}/best_model.pth")
            print(f"Epoch {epoch+1}/{config.num_epochs}, Validation Loss: {val_loss:.4f}")

        if (epoch + 1) % config.save_freq == 0:
            torch.save(model.state_dict(), f"{config.save_path}/model_epoch_{epoch+1}.pth")

    return (best_val_loss, model)

def main():

    # load constants
    load_constants('./constants')
    expanded_vocab = expand_vocabulary(constants.characters_mapping, constants.classes_mapping)
    configs = TrainConfig()

    # setup dataset
    # read the text files
    with open(configs.data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]

    # read asr text files (for text + audio model)
    with open(configs.asr_data_path, 'r', encoding='utf-8') as f:
        asr_lines = f.readlines()
    asr_lines = [line.strip() for line in asr_lines if line.strip()]
    
    assert len(lines) == len(asr_lines), "Text and ASR data must have the same number of lines."

    train_lines, val_lines = split_into_training_validation(lines)
    train_data = TextDataset(train_lines)
    val_data = TextDataset(val_lines)
    train_loader = create_dataloader(train_data, configs.batch_size)
    val_loader = create_dataloader(val_data, configs.batch_size)

    # setup model
    if configs.model_type == 'Transformer':
        model = ModifiedTransformerModel(
            maxlen=100,
            vocab_size=len(constants.characters_mapping),
            asr_vocab_size=len(expanded_vocab),
            d_model=configs.d_model,
            num_heads=configs.num_heads,
            dff=4 * configs.d_model,
            num_blocks=configs.num_blocks,
            dropout_rate=configs.dropout_rate
        )
    elif configs.model_type == 'LSTM':
        model = LSTMModel(
            vocab_size=len(constants.characters_mapping),
            asr_vocab_size=len(expanded_vocab),
            embedding_dim=configs.d_model,
            hidden_dim=configs.d_model,
            num_layers=configs.num_blocks,
            dropout_rate=configs.dropout_rate
        )
    else:
        raise ValueError(f"Unknown model type: {configs.model_type}")



    criterion = nn.CrossEntropyLoss(ignore_index=constants.classes_mapping.get('<PAD>', 0))
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)
    os.makedirs(configs.save_path, exist_ok=True)
    best_val_loss, trained_model = train(configs, model, train_loader, val_loader, criterion, optimizer)
    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")
    torch.save(trained_model.state_dict(), f"{configs.save_path}/final_model.pth")

    pass