import pickle as pkl
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from model import *
import re
import argparse

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

def split_data_tashkeela(
    data_raw: List[str],
    max_length: int = 270,
    return_undiacritized: bool = False,
) -> List[str]:
    """
    Split Arabic text into chunks whose *undiacritized* length is <= max_length.

    Args:
        data_raw: List of input strings (can contain newlines).
        max_length: Maximum length measured on text with diacritics removed.
        return_undiacritized: If True, return chunks without diacritics.
                              If False (default), return original text (with diacritics).

    Returns:
        List[str]: Chunks of text whose undiacritized length <= max_length.
    """

    out: List[str] = []

    # split on any whitespace blocks to "tokens" (words/punctuation)
    def tokenize(s: str) -> List[str]:
        return re.findall(r"\S+", s.strip())

    # split a single long token by characters so that each chunk (undiacritized) <= max_length
    def split_long_token(token: str) -> List[str]:
        chunks = []
        buf_chars = []
        buf_len = 0
        for ch in token:
            ch_len = len(remove_diacritics(ch))
            # if adding this char would exceed limit and we already have some chars, flush
            if buf_chars and buf_len + ch_len > max_length:
                chunks.append("".join(buf_chars))
                buf_chars = [ch]
                buf_len = ch_len
            else:
                buf_chars.append(ch)
                buf_len += ch_len
        if buf_chars:
            chunks.append("".join(buf_chars))
        return chunks

    for line in data_raw:
        for sub in line.split("\n"):
            base = sub.strip()
            if not base:
                continue

            # quick accept if whole line fits
            if len(remove_diacritics(base)) <= max_length:
                out.append(remove_diacritics(base) if return_undiacritized else base)
                continue

            # otherwise, word-wise packing
            tokens = tokenize(base)
            cur = []                 # list of original tokens
            cur_len = 0              # undiacritized length of current line
            for tok in tokens:
                tok_len = len(remove_diacritics(tok))
                sep = 1 if cur else 0  # space if we already have content

                # fits as a whole token
                if cur_len + sep + tok_len <= max_length:
                    if sep:
                        cur.append(" ")
                        cur_len += 1
                    cur.append(tok)
                    cur_len += tok_len
                    continue

                # flush current line if it has content
                if cur:
                    joined = "".join(cur)
                    out.append(remove_diacritics(joined) if return_undiacritized else joined)
                    cur, cur_len = [], 0

                # token itself is too long -> split by characters
                if tok_len > max_length:
                    pieces = split_long_token(tok)
                    for i, piece in enumerate(pieces):
                        piece_len = len(remove_diacritics(piece))
                        # each piece is guaranteed <= max_length; flush immediately
                        out.append(remove_diacritics(piece) if return_undiacritized else piece)
                    # after splitting a long token, we start fresh
                    cur, cur_len = [], 0
                else:
                    # start new line with this token
                    cur = [tok]
                    cur_len = tok_len

            # flush remainder
            if cur:
                joined = "".join(cur)
                out.append(remove_diacritics(joined) if return_undiacritized else joined)

    return out


def load_cfg():
    from config import _C as cfg
    parser = argparse.ArgumentParser(description="Train Diacritization Model")
    parser.add_argument('--config', type=str, default='configs/transformer.tashkeela.yml', help='Path to the YAML config file')
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER,
                        help="Override config: KEY VALUE pairs")
    args = parser.parse_args()
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