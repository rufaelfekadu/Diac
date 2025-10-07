"""
Tokenizer class for Arabic diacritization model.
Handles encoding and decoding of text to/from token indices.
"""

import pickle as pkl
import os
from typing import List, Dict, Optional, Union
import torch
from dataclasses import dataclass


@dataclass
class TokenizerConstants:
    """Container for tokenizer constants and mappings."""
    characters_mapping: Dict[str, int]
    arabic_letters_list: List[str]
    diacritics_list: List[str]
    classes_mapping: Dict[str, int]
    rev_classes_mapping: Dict[int, str]
    expanded_vocabulary: Dict[str, int]


class ArabicDiacritizationTokenizer:
    """
    Tokenizer for Arabic diacritization models.
    Handles encoding of text to token indices and decoding of predictions back to text.
    """
    
    def __init__(self, constants_path: str, with_extra_train: bool = False):
        """
        Initialize tokenizer with constants from pickle files.
        
        Args:
            constants_path: Path to directory containing pickle files
            with_extra_train: Whether to use extended character mapping
        """
        self.constants_path = constants_path
        self.with_extra_train = with_extra_train
        self.constants = self._load_constants()
        
    def _load_constants(self) -> TokenizerConstants:
        """Load constants from pickle files."""
        # Load character mappings
        if self.with_extra_train:
            chars_file = 'RNN_BIG_CHARACTERS_MAPPING.pickle'
        else:
            chars_file = 'RNN_SMALL_CHARACTERS_MAPPING.pickle'
            
        characters_mapping = pkl.load(open(
            os.path.join(self.constants_path, chars_file), 'rb'))
        
        # Load other constants
        arabic_letters_list = pkl.load(open(
            os.path.join(self.constants_path, 'ARABIC_LETTERS_LIST.pickle'), 'rb'))
        diacritics_list = pkl.load(open(
            os.path.join(self.constants_path, 'DIACRITICS_LIST.pickle'), 'rb'))
        classes_mapping = pkl.load(open(
            os.path.join(self.constants_path, 'RNN_CLASSES_MAPPING.pickle'), 'rb'))
        rev_classes_mapping = pkl.load(open(
            os.path.join(self.constants_path, 'RNN_REV_CLASSES_MAPPING.pickle'), 'rb'))
        
        # Create expanded vocabulary for ASR
        expanded_vocabulary = self._expand_vocabulary(characters_mapping, classes_mapping)
        
        return TokenizerConstants(
            characters_mapping=characters_mapping,
            arabic_letters_list=arabic_letters_list,
            diacritics_list=diacritics_list,
            classes_mapping=classes_mapping,
            rev_classes_mapping=rev_classes_mapping,
            expanded_vocabulary=expanded_vocabulary
        )
    
    def _expand_vocabulary(self, characters_mapping: Dict[str, int], 
                          classes_mapping: Dict[str, int]) -> Dict[str, int]:
        """Expand vocabulary by adding diacritics not in character mapping."""
        expanded_vocab = characters_mapping.copy()
        max_id = max(expanded_vocab.values())
        index = 1
        
        for diacritic in classes_mapping:
            if (diacritic not in expanded_vocab and 
                diacritic.strip() not in ["", "<N/A>"]):
                expanded_vocab[diacritic] = max_id + index
                index += 1
                
        return expanded_vocab
    
    def remove_diacritics(self, text: str) -> str:
        """Remove diacritics from text."""
        return text.translate(str.maketrans('', '', ''.join(self.constants.diacritics_list)))
    
    def encode_text(self, text: str) -> List[int]:
        """
        Encode text to sequence of character indices.
        
        Args:
            text: Input text string
            
        Returns:
            List of character indices
        """
        # Remove diacritics for encoding
        text_clean = self.remove_diacritics(text)
        
        # Encode with SOS and EOS tokens
        encoded = [self.constants.characters_mapping['<SOS>']]
        
        for char in text_clean:
            char_id = self.constants.characters_mapping.get(
                char, self.constants.characters_mapping.get('<PAD>', 0))
            encoded.append(char_id)
            
        encoded.append(self.constants.characters_mapping['<EOS>'])
        
        return encoded
    
    def encode_asr_text(self, text_asr: str) -> List[int]:
        """
        Encode ASR text using expanded vocabulary.
        
        Args:
            text_asr: ASR text string
            
        Returns:
            List of indices using expanded vocabulary
        """
        encoded = [self.constants.expanded_vocabulary['<SOS>']]
        
        for char in text_asr:
            char_id = self.constants.expanded_vocabulary.get(
                char, self.constants.expanded_vocabulary.get('<PAD>', 0))
            encoded.append(char_id)
            
        encoded.append(self.constants.expanded_vocabulary['<EOS>'])
        
        return encoded
    
    def encode_labels(self, text: str) -> List[int]:
        """
        Extract diacritic labels from text.
        
        Args:
            text: Text with diacritics
            
        Returns:
            List of diacritic class indices
        """
        labels = [self.constants.classes_mapping['<SOS>']]
        
        idx = 0
        while idx < len(text):
            char = text[idx]
            
            # Skip diacritics in main loop
            if char in self.constants.diacritics_list:
                idx += 1
                continue
                
            # For non-Arabic letters, assign empty class
            if char not in self.constants.arabic_letters_list:
                labels.append(self.constants.classes_mapping.get('', 0))
            else:
                # Look for diacritics after this character
                char_diac = ''
                if (idx + 1 < len(text) and 
                    text[idx + 1] in self.constants.diacritics_list):
                    char_diac = text[idx + 1]
                    
                    # Check for double diacritics
                    if (idx + 2 < len(text) and 
                        text[idx + 2] in self.constants.diacritics_list):
                        # Try both orders
                        double_diac1 = char_diac + text[idx + 2]
                        double_diac2 = text[idx + 2] + char_diac
                        
                        if double_diac1 in self.constants.classes_mapping:
                            char_diac = double_diac1
                        elif double_diac2 in self.constants.classes_mapping:
                            char_diac = double_diac2
                
                diac_id = self.constants.classes_mapping.get(
                    char_diac, self.constants.classes_mapping.get('', 0))
                labels.append(diac_id)
            
            idx += 1
        
        labels.append(self.constants.classes_mapping['<EOS>'])
        return labels
    
    def decode(self, predictions: List[int], original_text: str) -> str:
        """
        Decode prediction indices to diacritized text.
        
        Args:
            predictions: List of predicted diacritic class indices
            original_text: Original text without diacritics
            
        Returns:
            Text with predicted diacritics
        """
        decoded_text = ""
        clean_text = self.remove_diacritics(original_text)
        
        # Skip SOS token at the beginning if present
        start_idx = 1 if (len(predictions) > 0 and 
                         predictions[0] == self.constants.classes_mapping.get('<SOS>')) else 0
        
        # Process character by character
        for i, char in enumerate(clean_text):
            decoded_text += char
            
            # Skip non-Arabic letters for diacritization
            if char not in self.constants.arabic_letters_list:
                continue
            
            # Get prediction index (accounting for SOS token)
            pred_idx = start_idx + i
            if pred_idx < len(predictions):
                class_id = predictions[pred_idx]
                
                # Skip special tokens
                special_tokens = [
                    self.constants.classes_mapping.get('<SOS>'),
                    self.constants.classes_mapping.get('<EOS>'),
                    self.constants.classes_mapping.get('<PAD>')
                ]
                
                if class_id not in special_tokens:
                    diacritic = self.constants.rev_classes_mapping.get(class_id, '')
                    decoded_text += diacritic
        
        return decoded_text

    def encode_batch(self, texts: List[str], asr_texts: List[str]=[], max_length: Optional[int] = None, padding: bool = False) -> torch.Tensor:
        """
        Encode a batch of texts to tensor.
        
        Args:
            texts: List of text strings
            asr_texts: List of ASR text strings
            max_length: Maximum sequence length for padding
            
        Returns:
            Padded tensor of shape (batch_size, max_seq_len)
        """
        encoded_texts = [self.encode_text(text) for text in texts]
        labels = [self.encode_labels(text) for text in texts]
        encoded_asr_texts = [self.encode_asr_text(text) for text in asr_texts]

        # Convert to tensors
        tensors = [torch.tensor(encoded, dtype=torch.long) for encoded in encoded_texts]
        labels_tensors = [torch.tensor(encoded, dtype=torch.long) for encoded in labels]
        asr_tensors = [torch.tensor(encoded, dtype=torch.long) for encoded in encoded_asr_texts]

        if not padding:
            return tensors, asr_tensors, labels_tensors

        # Pad sequences
        pad_value = self.constants.characters_mapping.get('<PAD>', 0)
        padded = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=pad_value)

        labels_pad_value = self.constants.classes_mapping.get('<PAD>', 0)
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_tensors, batch_first=True, padding_value=labels_pad_value)

        if len(asr_tensors) > 0:
            asr_pad_value = self.constants.expanded_vocabulary.get('<PAD>', 0)
            padded_asr = torch.nn.utils.rnn.pad_sequence(
                asr_tensors, batch_first=True, padding_value=asr_pad_value)
        else:
            padded_asr = []

        # Truncate if max_length specified
        if max_length and padded.size(1) > max_length:
            padded = padded[:, :max_length]
            padded_labels = padded_labels[:, :max_length]

        if max_length and len(asr_tensors) > 0 and padded_asr.size(1) > max_length:
            padded_asr = padded_asr[:, :max_length]
        
        return padded, padded_asr, padded_labels

    def encode_asr_batch(self, texts_asr: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        """
        Encode a batch of ASR texts to tensor.
        
        Args:
            texts_asr: List of ASR text strings
            max_length: Maximum sequence length for padding
            
        Returns:
            Padded tensor of shape (batch_size, max_seq_len)
        """
        encoded_texts = [self.encode_asr_text(text) for text in texts_asr]
        
        # Convert to tensors
        tensors = [torch.tensor(encoded, dtype=torch.long) for encoded in encoded_texts]
        
        # Pad sequences
        pad_value = self.constants.expanded_vocabulary.get('<PAD>', 0)
        padded = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=pad_value)
        
        # Truncate if max_length specified
        if max_length and padded.size(1) > max_length:
            padded = padded[:, :max_length]
            
        return padded
    
    def decode_batch(self, predictions_batch: torch.Tensor, 
                    original_texts: List[str]) -> List[str]:
        """
        Decode a batch of predictions to diacritized texts.
        
        Args:
            predictions_batch: Tensor of shape (batch_size, seq_len)
            original_texts: List of original texts without diacritics
            
        Returns:
            List of diacritized texts
        """
        # Convert tensor to list
        if isinstance(predictions_batch, torch.Tensor):
            predictions_list = predictions_batch.cpu().numpy().tolist()
        else:
            predictions_list = predictions_batch
            
        # Decode each sequence
        results = []
        for predictions, original_text in zip(predictions_list, original_texts):
            decoded = self.decode(predictions, original_text)
            results.append(decoded)
            
        return results
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size for text encoding."""
        return len(self.constants.characters_mapping)
    
    @property
    def asr_vocab_size(self) -> int:
        """Get vocabulary size for ASR encoding."""
        return len(self.constants.expanded_vocabulary)
    
    @property
    def num_classes(self) -> int:
        """Get number of diacritic classes."""
        return len(self.constants.classes_mapping)
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            'pad': self.constants.characters_mapping.get('<PAD>', 0),
            'sos': self.constants.characters_mapping.get('<SOS>', 1),
            'eos': self.constants.characters_mapping.get('<EOS>', 2),
        }

if __name__ == "__main__":

    tokenizer = ArabicDiacritizationTokenizer('constants/')
    sample_asr_text = "َذَاالتَّسْجِيل مِنْ طَرَفِ لربّري بُوكْس جَمِيع تَسَّجِيلَات لربّري بُوكْس"
    sample_txt = "هَذَاالتَّسْجِيل مِنْ طَرَفِ لربّري بُوكْس جَمِيع تَسَّجِيلَات لربّري بُوكْس"
    

    encoded, encoded_asr, labels = tokenizer.encode_batch([sample_txt], [sample_asr_text], padding=False)

    print("Encoded text:", encoded[0].shape)
    print("Encoded ASR text:", encoded_asr[0].shape)
    print("Labels:", labels[0].shape)
