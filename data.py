import torch
import re
from torch.utils.data import Dataset, DataLoader
from constants import constants
from typing import List, Tuple, Optional
import csv
from tokenizer import ArabicDiacritizationTokenizer

def remove_diacritics(data_raw: str) -> str:
    """
    Remove diacritics from text.
    """
    if not constants or not constants.diacritics_list:
        raise ValueError("Constants not loaded. Call load_constants first.")
    return data_raw.translate(str.maketrans('', '', ''.join(constants.diacritics_list)))

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

# class TextDataset(Dataset):
#     """
#     Dataset class for text-only data.
#     """
#     def __init__(self, data_path: List[str]):
#         # load the tsv file
#         with open(data_path, 'r', encoding='utf-8') as f:
#             reader = csv.reader(f, delimiter='\t')
#             lines = [row[0] for row in reader if row]  # Assuming text is in the first column
#         self.lines = lines

#     def __len__(self) -> int:
#         return len(self.lines)

#     def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
#         line = self.lines[idx]
#         X, Y = map_data([line])
#         return {
#             "text": X[0],
#             "label": Y[0]
#             }

class TextAudioDataset(Dataset):
    """
    Dataset class for text + audio (ASR) data.
    """
    def __init__(self, data_path: str, tokenizer: ArabicDiacritizationTokenizer, max_length=None):

        self.tokenizer = tokenizer
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            lines = []
            asr_lines = []
            for row in reader:
                if len(row) >= 2:
                    lines.append(row[0])      
                    asr_lines.append(row[1]) 
                elif len(row) == 1:
                    if max_length:
                        lines.extend(split_data_tashkeela([row[0]], max_length))
                    else:
                        lines.append(row[0])
                else:
                    continue  # Skip empty lines
                    
        self.lines, self.asr_lines, self.labels = self.tokenizer.encode_batch(lines, asr_lines, padding=False)
    
        assert len(self.lines) == len(self.labels), "Mismatch in data lengths"

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[int]]:
        output = {
            "text": self.lines[idx],
            "label": self.labels[idx]
        }
        if len(self.asr_lines) > 0:
            output["asr"] = self.asr_lines[idx]
        return output

def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Custom collate function for variable length sequences.
    """
    if len(batch[0]) == 2:  # Text-only
        X_batch = [item['text'] for item in batch]
        Y_batch = [item['label'] for item in batch]
        X_batch = torch.nn.utils.rnn.pad_sequence(X_batch, batch_first=True, padding_value=constants.characters_mapping.get('<PAD>', 0))
        Y_batch = torch.nn.utils.rnn.pad_sequence(Y_batch, batch_first=True, padding_value=constants.classes_mapping.get('<PAD>', 0))
        return X_batch, None, Y_batch
    elif len(batch[0]) == 3:  # Text + Audio
        X_batch = [item['text'] for item in batch]
        X_asr_batch = [item['asr'] for item in batch]
        Y_batch = [item['label'] for item in batch]
        X_batch = torch.nn.utils.rnn.pad_sequence(X_batch, batch_first=True, padding_value=constants.characters_mapping.get('<PAD>', 0))
        X_asr_batch = torch.nn.utils.rnn.pad_sequence(X_asr_batch, batch_first=True, padding_value=constants.characters_mapping.get('<PAD>', 0))
        Y_batch = torch.nn.utils.rnn.pad_sequence(Y_batch, batch_first=True, padding_value=constants.classes_mapping.get('<PAD>', 0))
        return X_batch, X_asr_batch, Y_batch
    else:
        raise ValueError("Unexpected batch structure")


if __name__ == "__main__":

    data_path = "data/clartts/test_no_special.txt"
    tokenizer = ArabicDiacritizationTokenizer(constants_path='constants')

    # text_dataset = TextDataset(data_path)
    text_audio_dataset = TextAudioDataset(data_path, tokenizer, max_length=100)

    # text_loader = create_dataloader(text_dataset, batch_size=2)
    text_audio_loader = create_dataloader(text_audio_dataset, batch_size=2)

    for X, X_asr, Y in text_audio_loader:
        print("X:", X.dtype)
        print("Y:", Y)
        print("X_asr:", X_asr)
        break
