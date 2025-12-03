from diac.models import DiacritizationModule
from datasets import load_dataset
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="rufaelfekadu/diac-transformer-text-asr-tashkeela-clartts")
    parser.add_argument("--test_path", type=str, default="data/clartts/test.txt")
    parser.add_argument("--output_path", type=str, default="outputs/hf_inference_output.txt")
    args = parser.parse_args()

    model = DiacritizationModule.from_pretrained(
        args.model_name,
        tokenizer_constants_path="constants/"
    )

    model.predict_file(
        input_file=args.test_path, output_file=args.output_path
    )
