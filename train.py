

import argparse
from dataclasses import dataclass
from tqdm import tqdm
import wandb

from model import *
from data import TextDataset, TextAudioDataset, create_dataloader
from utils import *
from config import TrainConfig

def train_epoch(epoch, model, dataloader, criterion, optimizer, device, use_asr=True):
    model.train()
    training_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Training : Epoch {epoch}", leave=False):
        inputs, inputs_asr, targets = batch
        inputs, inputs_asr, targets = inputs.to(device), inputs_asr.to(device), targets.to(device)

        optimizer.zero_grad()
        if not use_asr:
            inputs_asr = None
        outputs = model(inputs, inputs_asr=inputs_asr)
        
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

    model.to(config.train.device)
    best_val_loss = float('inf')
    for epoch in range(config.train.num_epochs):
        train_loss = train_epoch(epoch, model, train_loader, criterion, optimizer, config.train.device, use_asr=config.model.use_asr)
        wandb.log({"train/loss": train_loss, "epoch": epoch+1})
        
        if (epoch + 1) % config.train.eval_freq == 0:
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, config.train.device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = f"{config.train.save_dir}/best_model.pth"
                torch.save(model.state_dict(), best_model_path)
                wandb.save(best_model_path)
            wandb.log({"val/loss": val_loss, "val/accuracy": val_accuracy, "epoch": epoch+1})

        if (epoch + 1) % config.train.save_freq == 0:
            checkpoint_path = f"{config.train.save_dir}/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

    return best_val_loss, model

def main(configs):

    # setup config
    constants = load_constants(configs.constants_path)
    expanded_vocab = expand_vocabulary(constants.characters_mapping, constants.classes_mapping)

    configs.model.vocab_size = len(constants.characters_mapping)
    configs.model.asr_vocab_size = len(expanded_vocab)
    configs.model.output_size = len(constants.classes_mapping)

    # save config to yml
    with open(f"{configs.train.save_dir}/config.yml", 'w') as f:
        import yaml
        yaml.dump(configs.__dict__, f)
    os.makedirs(configs.train.save_dir, exist_ok=True)
    breakpoint()
    # Initialize wandb
    wandb.init(project="diacritization", config=configs.__dict__, dir=configs.train.save_dir)
    wandb.run.name = f"{configs.model.type}_bs{configs.train.batch_size}_lr{configs.train.learning_rate}"
    
    # prepare data
    train_data = TextAudioDataset(configs.data.train_path, expanded_vocab)
    test_data = TextAudioDataset(configs.data.test_path, expanded_vocab)

    # split train dataset into training and validation sets
    train_size = int(0.9 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    train_loader = create_dataloader(train_data, configs.train.batch_size)
    val_loader = create_dataloader(val_data, configs.train.batch_size)
    # test_loader = create_dataloader(test_data, configs.batch_size)
    
    print("Data loaders created.")
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")
    print("Setting up model...")


    # setup model
    if configs.model.type == 'Transformer':
        model = ModifiedTransformerModel(
            **configs.model.__dict__
        )
        if configs.model.pretrained_path:
            model.load_pretrained(configs.model.pretrained_path)

        model.to(configs.train.device)
    elif configs.model.type == 'LSTM':
        model = ModifiedLSTMModel(
            **configs.model.__dict__
        )
        model.to(configs.train.device)
    else:
        raise ValueError(f"Unknown model type: {configs.model.type}")

    # setup loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=constants.classes_mapping.get('<PAD>', 0))
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.train.learning_rate)

    # TODO: Add LR scheduler if needed

    print("Starting training...")
    best_val_loss, _ = train(configs, model, train_loader, val_loader, criterion, optimizer)
    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")

    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser(description="Train Diacritization Model")
    parser.add_argument('--config', type=str, default='configs/transformer.clartts.yml', help='Path to the YAML config file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig.from_yml(args.config)
    main(config)