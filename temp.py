import os
import pickle as pkl

def _load_constants(self):
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
    
    return {
        'characters_mapping': characters_mapping,
        'arabic_letters_list': arabic_letters_list,
        'diacritics_list': diacritics_list,
        'classes_mapping': classes_mapping,
        'rev_classes_mapping': rev_classes_mapping
    }