#!/usr/bin/env python3

import argparse
import os

import torch
import lightning as L

from tokenizer import ArabicDiacritizationTokenizer
from model import DiacritizationModule
from utils import load_cfg, load_constants, setup_loggers, setup_callbacks, setup_data_loaders

os.environ["PYTHONIOENCODING"] = "utf-8"

def main(args):

    # Load configuration
    config = load_cfg(args)

    global constants
    constants = load_constants(config.CONSTANTS_PATH)
    
    os.makedirs(config.TRAIN.SAVE_DIR, exist_ok=True)
    
    pl_loggers, logger = setup_loggers(config)
    
    tokenizer = ArabicDiacritizationTokenizer(constants_path=config.CONSTANTS_PATH)
    
    # Setup data
    train_loader, val_loader, test_loader, train_size, val_size, test_size = setup_data_loaders(
        config, tokenizer
    )
    
    logger.info(f"Data loaded - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    model = DiacritizationModule(config, tokenizer)
    # print model and number of parameters
    logger.info(f"Model: {model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}")
    callbacks = setup_callbacks(config)

    trainer = L.Trainer(
        max_epochs=config.TRAIN.NUM_EPOCHS,
        callbacks=callbacks,
        logger=pl_loggers,
        accelerator='auto',  # Automatically detect GPU/CPU
        devices=1,     
        precision='16-mixed' if torch.cuda.is_available() else 32,  # Mixed precision for faster training
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=getattr(config.TRAIN, 'ACCUMULATE_GRAD_BATCHES', 1),
        val_check_interval=getattr(config.TRAIN, 'VAL_CHECK_INTERVAL', 1.0),
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    logger.info("Starting Lightning training...")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path='best')
    
    # Save final metrics
    if trainer.logged_metrics:
        logger.info("Final metrics:")
        for key, value in trainer.logged_metrics.items():
            logger.info(f"  {key}: {value}")
    
    logger.info(f"Training complete. Best model saved to: {config.TRAIN.SAVE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train diacritization model with PyTorch Lightning')
    parser.add_argument('--config', type=str, default='configs/lstm.yml',
                        help="Path to the config file")
    parser.add_argument('--opts', default=[], nargs=argparse.REMAINDER,
                        help="Override config: KEY VALUE pairs")
    args = parser.parse_args()
    
    main(args)