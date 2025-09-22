
from model import *
from data import TextDataset, TextAudioDataset, create_dataloader
from utils import *
import argparse
from dataclasses import dataclass
from tqdm import tqdm
import wandb

@dataclass
class TrainConfig:

    # data related
    train_path: str = 'Diac/data/clartts/clartts_asr_train.tsv'
    test_path: str = 'Diac/data/clartts/clartts_asr_test.tsv'

    # training related
    device: str = 'cpu'  # or 'cpu'
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001

    # model related
    maxlen: int = 1000
    d_model: int = 128
    num_heads: int = 4
    num_blocks: int = 2
    dropout_rate: float = 0.2
    model_type: str = 'Transformer'  # or  'LSTM' 'Transformer'

    # output related
    save_freq: int = 30
    eval_freq: int = 1
    save_path: str = 'Diac/checkpoints/'

    @classmethod
    def from_yml(cls, yml_path: str):
        import yaml
        with open(yml_path, 'r') as f:
            yml_config = yaml.safe_load(f)
        for key, value in yml_config.items():
            if hasattr(cls, key):
                setattr(cls, key, value)


def train_epoch(epoch, model, dataloader, criterion, optimizer, device):
    model.train()
    training_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Training : Epoch {epoch}", leave=False):
        inputs, inputs_asr, targets = batch
        inputs, inputs_asr, targets = inputs.to(device), inputs_asr.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, inputs_asr)
        
        loss = criterion(outputs.permute(0,2,1), targets)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    return training_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    validation_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, input_asr, targets = batch
            inputs, input_asr, targets = inputs.to(device), input_asr.to(device), targets.to(device)

            outputs = model(inputs, input_asr)
            pred = outputs.argmax(dim=-1)
            total_correct += (pred == targets).sum().item()
            loss = criterion(outputs.permute(0, 2, 1), targets)
            validation_loss += loss.item()

    return validation_loss / len(dataloader), total_correct / (len(dataloader.dataset) * targets.size(1))

def train(config, model, train_loader, val_loader, criterion, optimizer):

    model.to(config.device)
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(epoch, model, train_loader, criterion, optimizer, config.device)
        wandb.log({"train/loss": train_loss, "epoch": epoch+1})

        if (epoch + 1) % config.eval_freq == 0:
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, config.device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{config.save_path}/best_model.pth")
            wandb.log({"val/loss": val_loss, "val/accuracy": val_accuracy, "epoch": epoch+1})

        if (epoch + 1) % config.save_freq == 0:
            torch.save(model.state_dict(), f"{config.save_path}/model_epoch_{epoch+1}.pth")

    return best_val_loss, model

def main():

    # load constants
    load_constants('./Diac/constants')
    expanded_vocab = expand_vocabulary(constants.characters_mapping, constants.classes_mapping)
    configs = TrainConfig()

    os.makedirs(configs.save_path, exist_ok=True)

    # Initialize wandb
    wandb.init(project="diacritization", config=configs.__dict__)
    wandb.run.name = f"{configs.model_type}_bs{configs.batch_size}_lr{configs.learning_rate}"
    
    # prepare data
    train_data = TextAudioDataset(configs.train_path, expanded_vocab)
    test_data = TextAudioDataset(configs.test_path, expanded_vocab)
    # split train dataset into training and validation sets
    train_size = int(0.9 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    train_loader = create_dataloader(train_data, configs.batch_size)
    val_loader = create_dataloader(val_data, configs.batch_size)
    # test_loader = create_dataloader(test_data, configs.batch_size)
    
    print("Data loaders created.")
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")
    print("Setting up model...")
    # setup model
    if configs.model_type == 'Transformer':
        model = ModifiedTransformerModel(
            maxlen=configs.maxlen,
            vocab_size=len(constants.characters_mapping),
            asr_vocab_size=len(expanded_vocab),
            d_model=configs.d_model,
            num_heads=configs.num_heads,
            dff=4 * configs.d_model,
            num_blocks=configs.num_blocks,
            dropout_rate=configs.dropout_rate,
            output_size=len(constants.classes_mapping)
        )
        model.to(configs.device)
    elif configs.model_type == 'LSTM':
        model = LSTMModel(
            vocab_size=len(constants.characters_mapping),
            asr_vocab_size=len(expanded_vocab),
            embedding_dim=configs.d_model,
            hidden_dim=configs.d_model,
            num_layers=configs.num_blocks,
            dropout_rate=configs.dropout_rate,
            output_size=len(constants.classes_mapping)
        )
        model.to(configs.device)
    else:
        raise ValueError(f"Unknown model type: {configs.model_type}")



    criterion = nn.CrossEntropyLoss(ignore_index=constants.classes_mapping.get('<PAD>', 0))
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)

    print("Starting training...")
    best_val_loss, trained_model = train(configs, model, train_loader, val_loader, criterion, optimizer)
    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")
    torch.save(trained_model.state_dict(), f"{configs.save_path}/final_model.pth")
    wandb.finish()

if __name__ == "__main__":
    main()