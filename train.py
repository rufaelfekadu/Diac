

import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import torch
import torch.nn as nn
import yaml

from model import AVAILABLE_MODELS
from data import TextAudioDataset, create_dataloader
from utils import load_cfg, dump_cfg, load_constants, expand_vocabulary
from config import _to_dict

def train_epoch(epoch, model, dataloader, criterion, optimizer, device, use_asr=True):
    model.train()
    training_loss = 0.0
    data_iter = tqdm(dataloader, desc=f"Training : Epoch {epoch}", leave=False)
    for batch in data_iter:
        inputs, inputs_asr, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        if use_asr and inputs_asr is not None:
            inputs_asr = inputs_asr.to(device)
    
        optimizer.zero_grad()

        outputs = model(inputs, inputs_asr=inputs_asr)

        loss = criterion(outputs.permute(0,2,1), targets)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        data_iter.set_postfix(loss=loss.item())

    return training_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, epoch=0):
    model.eval()
    validation_loss = 0.0
    total_correct = 0
    total_tokens = 0
    data_iter = tqdm(dataloader, desc=f"Evaluating : Epoch {epoch}")
    with torch.no_grad():
        for batch in data_iter:
            inputs, input_asr, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            if input_asr is not None:
                input_asr = input_asr.to(device)

            outputs = model(inputs, input_asr)
            pred = outputs.argmax(dim=-1)

            total_correct += (pred == targets).sum().item()
            total_tokens += targets.numel()

            loss = criterion(outputs.permute(0,2,1), targets)
            validation_loss += loss.item()
        # update progress bar with accuracy and loss
        data_iter.set_postfix(accuracy=100.0 * total_correct / total_tokens, loss=validation_loss / len(dataloader))
    return validation_loss / len(dataloader), total_correct / total_tokens

def train(config, model, train_loader, val_loader, criterion, optimizer, writer):

    model.to(config.TRAIN.DEVICE)
    best_val_loss = float('inf')
    for epoch in range(config.TRAIN.NUM_EPOCHS):
        train_loss = train_epoch(epoch, model, train_loader, criterion, optimizer, config.TRAIN.DEVICE, use_asr=config.MODEL.USE_ASR)
        writer.add_scalar('train/loss', train_loss, epoch+1)
        
        if (epoch + 1) % config.TRAIN.EVAL_FREQ == 0:
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, config.TRAIN.DEVICE, epoch=epoch+1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = f"{config.TRAIN.SAVE_DIR}/best_model.pth"
                torch.save(model.state_dict(), best_model_path)
            writer.add_scalar('val/loss', val_loss, epoch+1)
            writer.add_scalar('val/accuracy', val_accuracy, epoch+1)

        if (epoch + 1) % config.TRAIN.SAVE_FREQ == 0:
            checkpoint_path = f"{config.TRAIN.SAVE_DIR}/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)

    return best_val_loss, model

def main(configs):

    os.makedirs(configs.TRAIN.SAVE_DIR, exist_ok=True)

    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{configs.TRAIN.SAVE_DIR}/training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # setup config
    constants = load_constants(configs.CONSTANTS_PATH)
    expanded_vocab = expand_vocabulary(constants.characters_mapping, constants.classes_mapping)

    assert configs.MODEL.VOCAB_SIZE == len(constants.characters_mapping), f"model vocab size {configs.MODEL.VOCAB_SIZE} does not match characters mapping size {len(constants.characters_mapping)}" 
    assert configs.MODEL.ASR_VOCAB_SIZE == len(expanded_vocab), f"model asr vocab size {configs.MODEL.ASR_VOCAB_SIZE} does not match expanded vocab size {len(expanded_vocab)}"
    assert configs.MODEL.OUTPUT_SIZE == len(constants.classes_mapping), f"model output size {configs.MODEL.OUTPUT_SIZE} does not match classes mapping size {len(constants.classes_mapping)}"


    # save config to yml
    dump_cfg(configs, os.path.join(configs.TRAIN.SAVE_DIR, 'config.yml'))
    

    # Initialize tensorboard
    tensorboard_dir = f"{configs.TRAIN.SAVE_DIR}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # prepare data
    train_data = TextAudioDataset(configs.DATA.TRAIN_PATH, expanded_vocab, max_length=configs.DATA.MAX_LENGTH)
    test_data = TextAudioDataset(configs.DATA.TEST_PATH, expanded_vocab, max_length=configs.DATA.MAX_LENGTH)

    # # split train dataset into training and validation sets
    if not configs.DATA.VAL_PATH:
        val_size = int(0.1 * len(train_data))
        train_size = len(train_data) - val_size
        train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
    else:
        val_data = TextAudioDataset(configs.DATA.VAL_PATH, expanded_vocab, max_length=configs.DATA.MAX_LENGTH)

    train_loader = create_dataloader(train_data, configs.TRAIN.BATCH_SIZE)
    val_loader = create_dataloader(val_data, configs.TRAIN.BATCH_SIZE)
    
    logger.info("Data loaders created.")
    logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")
    logger.info("Setting up model...")

    # setup model
    if configs.MODEL.TYPE not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model type: {configs.MODEL.TYPE}")
    
    model = AVAILABLE_MODELS[configs.MODEL.TYPE].from_config(configs)
    if configs.MODEL.PRETRAINED_PATH:
        model.load_pretrained(configs.MODEL.PRETRAINED_PATH, text_branch_only=configs.MODEL.LOAD_TEXT_BRANCH_ONLY)
        logger.info(f"Loaded pretrained weights from {configs.MODEL.PRETRAINED_PATH}")
        model.to(configs.TRAIN.DEVICE)
    
    logger.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")

    # setup loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='mean', 
                                    # ignore_index=constants.classes_mapping.get('<PAD>')
                                    )
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.TRAIN.LEARNING_RATE)

    logger.info("Starting training...")
    best_val_loss, _ = train(configs, model, train_loader, val_loader, criterion, optimizer, writer)
    logger.info(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")

    writer.close()


if __name__ == "__main__":

    config = load_cfg()
    main(config)