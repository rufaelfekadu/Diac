from utils import *
from model import DiacritizationModule
from tokenizer import ArabicDiacritizationTokenizer

os.environ["PYTHONIOENCODING"] = "utf-8"

def main(config):

    config = load_cfg(args)

    os.makedirs(os.path.dirname(config.INFERENCE.OUTPUT_PATH), exist_ok=True)

    constants = load_constants(config.CONSTANTS_PATH)

    tokenizer = ArabicDiacritizationTokenizer(constants_path=config.CONSTANTS_PATH)
    model = DiacritizationModule.load_from_checkpoint(
        config=config,
        checkpoint_path=config.INFERENCE.MODEL_PATH,
        tokenizer=tokenizer,
    )

    model.predict_file(
        input_file=config.DATA.TEST_PATH,
        output_file=config.INFERENCE.OUTPUT_PATH
    )

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Diacritization Inference")
    parser.add_argument('--config', type=str, default='configs/lstm.yml', help="Path to the config file")
    parser.add_argument('--opts', default=[], nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    args = parser.parse_args()
    
    main(args)