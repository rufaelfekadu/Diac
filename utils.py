import torch
import pickle as pkl
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class Constants:
    characters_mapping: Dict[str, int]
    arabic_letters_list: List[str]
    diacritics_list: List[str]
    classes_mapping: Dict[str, int]
    rev_classes_mapping: Dict[int, str]

# Global constants instance
constants: Constants = Constants({}, [], [], {}, {})

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

def split_data_tashkeela(data_raw: List[str], max_length: int = 270) -> List[str]:
    """
    Split data into lines shorter than max_length characters (without diacritics).

    Inputs:
    - data_raw (List[str]): List of input lines
    - max_length (int): Maximum length of undiacritized text

    Outputs:
    - List[str]: List of split lines
    """
    data_new = []
    for line in data_raw:
        for sub_line in line.split('\n'):
            stripped = remove_diacritics(sub_line).strip()
            if len(stripped) == 0:
                continue
            if len(stripped) <= max_length:
                data_new.append(sub_line.strip())
            else:
                words = sub_line.split()
                tmp_line = ''
                for word in words:
                    word_stripped = remove_diacritics(word).strip()
                    tmp_stripped = remove_diacritics(tmp_line).strip()
                    if len(tmp_stripped) + len(word_stripped) + 1 > max_length:
                        if len(tmp_stripped) > 0:
                            data_new.append(tmp_line.strip())
                        tmp_line = word
                    else:
                        tmp_line = word if tmp_line == '' else tmp_line + ' ' + word
                if len(remove_diacritics(tmp_line).strip()) > 0:
                    data_new.append(tmp_line.strip())
    return data_new

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
    print("splited_data_fun ", len(data_new))
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

def split_into_training_validation(lines: List[str], split_index: int = 9000, val_split_index: int = 1000) -> Tuple[List[str], List[str]]:
    """
    Split data into training and validation sets.

    Inputs:
    - lines (List[str]): All data lines
    - split_index (int): Index to split training data
    - val_split_index (int): Number of validation samples

    Outputs:
    - Tuple[List[str], List[str]]: (training lines, validation lines)
    """
    train_raw_c = lines[:split_index]
    val_raw_c = lines[split_index:]
    train_split_c = split_data(train_raw_c, split_index)
    val_split_c = split_data(val_raw_c, val_split_index)
    return train_split_c, val_split_c


