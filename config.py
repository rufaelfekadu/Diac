from yacs.config import CfgNode as CN

# Create root config node
_C = CN()

# Model configuration
_C.MODEL = CN()
_C.MODEL.TYPE = 'Transformer'  # or 'LSTM'
_C.MODEL.MAXLEN = 1000
_C.MODEL.VOCAB_SIZE = 1000
_C.MODEL.ASR_VOCAB_SIZE = 1000
_C.MODEL.D_MODEL = 128
_C.MODEL.NUM_HEADS = 4
_C.MODEL.DFF = 512
_C.MODEL.NUM_BLOCKS = 2
_C.MODEL.DROPOUT_RATE = 0.1
_C.MODEL.OUTPUT_SIZE = 15
_C.MODEL.USE_ASR = True
_C.MODEL.PRETRAINED_PATH = None  # Path to pretrained model if any
_C.MODEL.LOAD_TEXT_BRANCH_ONLY = False
_C.MODEL.WITH_CONN = False

# Training configuration
_C.TRAIN = CN()
_C.TRAIN.DEVICE = 'cuda'  # or 'cpu'
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.LEARNING_RATE = 0.0001
_C.TRAIN.SAVE_FREQ = 30
_C.TRAIN.EVAL_FREQ = 1
_C.TRAIN.SAVE_DIR = 'checkpoints/'

# Inference configuration
_C.INFERENCE = CN()
_C.INFERENCE.DEVICE = 'cuda'
_C.INFERENCE.BATCH_SIZE = 16
_C.INFERENCE.MODEL_PATH = 'checkpoints/best_model.pth'
_C.INFERENCE.ASR_MODEL_NAME = 'sashat/whisper-medium-ClassicalAr'
_C.INFERENCE.USE_ASR = True
_C.INFERENCE.FORCED_IDS = None  # List of tuples for forced decoder ids
_C.INFERENCE.OUTPUT_PATH = 'results/predictions.txt'

# Data configuration
_C.DATA = CN()
_C.DATA.TRAIN_PATH = 'data/clartts/clartts_asr_train.tsv'
_C.DATA.VAL_PATH = 'data/clartts/clartts_asr_val.tsv'
_C.DATA.TEST_PATH = 'data/clartts/clartts_asr_test.tsv'
_C.DATA.MAX_LENGTH = None  # Maximum length of undiacritized text, None means no limit

# Constants path
_C.CONSTANTS_PATH = 'constants/'

def _to_dict(cfg_node):
    """Convert a yacs CfgNode to a regular dictionary."""
    cfg_dict = {}
    for key in cfg_node:
        if hasattr(cfg_node[key], 'items'):  # if it's a sub-config
            cfg_dict[key] = _to_dict(cfg_node[key])
        else:
            cfg_dict[key] = cfg_node[key]
    return cfg_dict
