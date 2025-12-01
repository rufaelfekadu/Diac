from diac.models import DiacritizationModule
from datasets import load_dataset


if __name__ == "__main__":

    test_path = "data/clartts/test.txt"
    model = DiacritizationModule.from_pretrained(
        "rufaelfekadu/diac-transformer-text-asr-tashkeela-clartts"
    )

    model.predict_file(
        input_file=test_path, output_file="outputs/hf_inference_output.txt"
    )
