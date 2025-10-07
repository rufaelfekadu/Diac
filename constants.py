from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Constants:
    characters_mapping: Dict[str, int]
    arabic_letters_list: List[str]
    diacritics_list: List[str]
    classes_mapping: Dict[str, int]
    rev_classes_mapping: Dict[int, str]

# Global constants instance
constants: Constants = Constants({}, [], [], {}, {})
