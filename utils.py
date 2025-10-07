import pickle as pkl
import os
from typing import List, Tuple, Dict, Optional
import re
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    DeviceStatsMonitor
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from data import TextAudioDataset, create_dataloader
import logging

from constants import constants, Constants

def load_constants(aux_dataset_path: str, with_extra_train: bool = False) -> Constants:
    """
    Load constants from pickle files and update the global constants instance.

    Inputs:
    - aux_dataset_path (str): Path to the directory containing pickle files
    - with_extra_train (bool): Whether to use extra training data mappings

    Outputs:
    - Constants: The updated constants object
    """
    global constants
    if with_extra_train:
        constants.characters_mapping = pkl.load(open(os.path.join(aux_dataset_path, 'RNN_BIG_CHARACTERS_MAPPING.pickle'), 'rb'))
    else:
        constants.characters_mapping = pkl.load(open(os.path.join(aux_dataset_path, 'RNN_SMALL_CHARACTERS_MAPPING.pickle'), 'rb'))
    constants.arabic_letters_list = pkl.load(open(os.path.join(aux_dataset_path, 'ARABIC_LETTERS_LIST.pickle'), 'rb'))
    constants.diacritics_list = pkl.load(open(os.path.join(aux_dataset_path, 'DIACRITICS_LIST.pickle'), 'rb'))
    constants.classes_mapping = pkl.load(open(os.path.join(aux_dataset_path, 'RNN_CLASSES_MAPPING.pickle'), 'rb'))
    constants.rev_classes_mapping = pkl.load(open(os.path.join(aux_dataset_path, 'RNN_REV_CLASSES_MAPPING.pickle'), 'rb'))
    return constants

def remove_diacritics(data_raw: str) -> str:
    """
    Remove diacritics from text.

    Inputs:
    - data_raw (str): Input text with diacritics

    Outputs:
    - str: Text without diacritics
    """
    if not constants or not constants.diacritics_list:
        raise ValueError("Constants not loaded. Call load_constants first.")
    return data_raw.translate(str.maketrans('', '', ''.join(constants.diacritics_list)))

def remove_special_chars(data_raw: str) -> str:
    _punctuations = ".,!?;:"
    return data_raw.translate(str.maketrans('', '', _punctuations))

# def split_data_tashkeela(data_raw: List[str], max_length: int = 270) -> List[str]:
#     """
#     Split data into lines shorter than max_length characters (without diacritics).

#     Inputs:
#     - data_raw (List[str]): List of input lines
#     - max_length (int): Maximum length of undiacritized text

#     Outputs:
#     - List[str]: List of split lines
#     """
#     data_new = []
#     for line in data_raw:
#         for sub_line in line.split('\n'):
#             stripped = remove_diacritics(sub_line).strip()
#             if len(stripped) == 0:
#                 continue
#             if len(stripped) <= max_length:
#                 data_new.append(sub_line.strip())
#             else:
#                 words = sub_line.split()
#                 tmp_line = ''
#                 for word in words:
#                     word_stripped = remove_diacritics(word).strip()
#                     tmp_stripped = remove_diacritics(tmp_line).strip()
#                     if len(tmp_stripped) + len(word_stripped) + 1 > max_length:
#                         if len(tmp_stripped) > 0:
#                             data_new.append(tmp_line.strip())
#                         tmp_line = word_stripped
#                     else:
#                         tmp_line = word_stripped if tmp_line == '' else tmp_line + ' ' + word_stripped
#                 if len(remove_diacritics(tmp_line).strip()) > 0:
#                     data_new.append(tmp_line.strip())
#     return data_new

def map_data(data_raw: List[str]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Convert text lines to sequences of character indices and diacritic class indices.

    Inputs:
    - data_raw (List[str]): List of input text lines

    Outputs:
    - Tuple[List[List[int]], List[List[int]]]: (X sequences, Y sequences)
    """
    if not constants:
        raise ValueError("Constants not loaded. Call load_constants first.")
    
    X = []
    Y = []
    for line in data_raw:
        x = [constants.characters_mapping['<SOS>']]
        y = [constants.classes_mapping['<SOS>']]
        for idx, char in enumerate(line):
            if char in constants.diacritics_list:
                continue
            x.append(constants.characters_mapping.get(char, constants.characters_mapping['<PAD>']))
            if char not in constants.arabic_letters_list:
                y.append(constants.classes_mapping[''])
            else:
                char_diac = ''
                if idx + 1 < len(line) and line[idx + 1] in constants.diacritics_list:
                    char_diac = line[idx + 1]
                    if idx + 2 < len(line) and line[idx + 2] in constants.diacritics_list:
                        if char_diac + line[idx + 2] in constants.classes_mapping:
                            char_diac += line[idx + 2]
                        elif line[idx + 2] + char_diac in constants.classes_mapping:
                            char_diac = line[idx + 2] + char_diac
                y.append(constants.classes_mapping.get(char_diac, constants.classes_mapping['']))
        x.append(constants.characters_mapping['<EOS>'])
        y.append(constants.classes_mapping['<EOS>'])
        X.append(x)
        Y.append(y)
    return X, Y

def split_data(data_raw: List[str], n: int, max_length: int = 100) -> List[str]:
    """
    Split data into lines shorter than max_length and return first n.

    Inputs:
    - data_raw (List[str]): List of input lines
    - n (int): Number of lines to return
    - max_length (int): Maximum length

    Outputs:
    - List[str]: First n split lines
    """
    data_new = split_data_tashkeela(data_raw, max_length)
    return data_new[:n]

def expand_vocabulary(CHARACTERS_MAPPING: Dict[str, int], CLASSES_MAPPING: Dict[str, int]) -> Dict[str, int]:
    """
    Expand vocabulary by adding diacritics not in characters mapping.

    Inputs:
    - CHARACTERS_MAPPING (Dict[str, int]): Character to index mapping
    - CLASSES_MAPPING (Dict[str, int]): Diacritic class to index mapping

    Outputs:
    - Dict[str, int]: Expanded vocabulary
    """
    ExpandedVocabulary = CHARACTERS_MAPPING.copy()
    max_id = max(ExpandedVocabulary.values())
    index = 1
    for diacritic in CLASSES_MAPPING:
        if diacritic not in ExpandedVocabulary and diacritic.strip() not in ["", "<N/A>"]:
            ExpandedVocabulary[diacritic] = max_id + index
            index += 1
    return ExpandedVocabulary

def map_asr_data(data_raw: List[str], expanded_vocabulary: Dict[str, int]) -> List[List[int]]:
    """
    Convert ASR text lines to sequences of indices using expanded vocabulary.

    Inputs:
    - data_raw (List[str]): List of ASR text lines
    - expanded_vocabulary (Dict[str, int]): Vocabulary mapping

    Outputs:
    - List[List[int]]: List of sequences
    """
    X = []
    for line in data_raw:
        x = [expanded_vocabulary['<SOS>']]
        for char in line:
            x.append(expanded_vocabulary.get(char, expanded_vocabulary.get('<PAD>', 0)))
        x.append(expanded_vocabulary['<EOS>'])
        X.append(x)
    return X

def batch_decode_predictions(batch_predictions: List[List[int]], texts: List[str]) -> List[str]:
    return [decode_predictions(pred, text) for pred, text in zip(batch_predictions, texts)]

def decode_predictions(predictions: List[int], text: str) -> str:
    """
    Decode sequence of predicted class indices to text with diacritics.

    Inputs:
    - predictions (List[int]): List of predicted class indices

    Outputs:
    - str: Decoded text with diacritics
    """
    if not constants or not constants.rev_classes_mapping:
        raise ValueError("Constants not loaded. Call load_constants first.")
    
    decoded_text = ""
    text = remove_diacritics(text)
    for idx, char in zip(predictions, text):
        decoded_text += char

        if char not in constants.arabic_letters_list:
            continue

        if idx in [constants.classes_mapping.get('<SOS>'), constants.classes_mapping.get('<EOS>'), constants.classes_mapping.get('<PAD>')]:
            continue
        
        decoded_text += constants.rev_classes_mapping.get(idx, '')

    return decoded_text



def load_cfg(args):
    from config import _C as cfg
    if args.config:
        cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def dump_cfg(config, path):
    import yaml 
    from config import _to_dict
    with open(path, 'w') as f:
        yaml.dump(_to_dict(config), f)


def setup_data_loaders(config, tokenizer):
    """Setup train, validation, and test data loaders."""
    # Create datasets
    train_data = TextAudioDataset(
        config.DATA.TRAIN_PATH, 
        tokenizer,
        max_length=config.DATA.MAX_LENGTH
    )       

    test_data = TextAudioDataset(
        config.DATA.TEST_PATH, 
        tokenizer,  
        max_length=config.DATA.MAX_LENGTH
    )

    # Handle validation data
    if hasattr(config.DATA, 'VAL_PATH') and config.DATA.VAL_PATH:
        val_data = TextAudioDataset(
            config.DATA.VAL_PATH, 
            tokenizer, 
            max_length=config.DATA.MAX_LENGTH
        )
    else:
        # Split train dataset into training and validation sets
        val_size = int(0.1 * len(train_data))
        train_size = len(train_data) - val_size
        train_data, val_data = torch.utils.data.random_split(
            train_data, [train_size, val_size]
        )
    
    # Create data loaders
    train_loader = create_dataloader(train_data, config.TRAIN.BATCH_SIZE)
    val_loader = create_dataloader(val_data, config.TRAIN.BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(test_data, config.TRAIN.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, len(train_data), len(val_data), len(test_data)

def setup_callbacks(config):
    """Setup Lightning callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        filename='best_model',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=getattr(config.TRAIN, 'EARLY_STOPPING_PATIENCE', 10),
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # # Learning rate monitoring
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # callbacks.append(lr_monitor)
    
    # Device stats monitoring 
    # if torch.cuda.is_available():
    #     device_stats = DeviceStatsMonitor()
    #     callbacks.append(device_stats)
    
    return callbacks

def setup_loggers(config):
    """Setup Lightning loggers."""
    pl_loggers = []

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{config.TRAIN.SAVE_DIR}/training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config.TRAIN.SAVE_DIR,
        name="tensorboard",
    )
    pl_loggers.append(tb_logger)

    # Forward PyTorch Lightning / lightning logs to our logger so that
    # messages produced by the library are written to the same handlers
    # (file + stdout) that we configured above.
    lightning_logger_names = [
        'lightning',
        'pytorch_lightning',
        'lightning.pytorch',
        'pytorch_lightning.core',
    ]

    # Get handlers from the root logger (where basicConfig placed them)
    root_handlers = logging.getLogger().handlers

    for name in lightning_logger_names:
        lib_logger = logging.getLogger(name)
        lib_logger.setLevel(logging.INFO)
        # Attach root handlers to the library logger if not already attached
        for h in root_handlers:
            if all(type(h) != type(existing) or getattr(h, 'stream', None) != getattr(existing, 'stream', None) for existing in lib_logger.handlers):
                lib_logger.addHandler(h)
        # Prevent propagation to avoid duplicate messages while ensuring
        # the logger writes to the same handlers as the root logger.
        lib_logger.propagate = False

    # Capture warnings and route them through logging
    logging.captureWarnings(True)


    # CSV logger for easy metric analysis (optional)
    # csv_logger = CSVLogger(
    #     save_dir=config.TRAIN.SAVE_DIR,
    #     name="csv_logs"
    # )
    # pl_loggers.append(csv_logger)

    return pl_loggers, logger

