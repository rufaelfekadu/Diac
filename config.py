
from dataclasses import dataclass

@dataclass
class ModelParams:
    type: str = 'Transformer'  # or 'LSTM'
    maxlen: int = 1000
    vocab_size: int = 1000
    asr_vocab_size: int = 1000
    d_model: int = 128
    num_heads: int = 4
    dff: int = 512
    num_blocks: int = 2
    dropout_rate: float = 0.1
    output_size: int = 15
    use_asr: bool = True
    pretrained_path: str = None  # Path to pretrained model if any
    with_conn: bool = False

@dataclass
class TrainParams:
    device: str = 'cuda'  # or 'cpu'
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.0001
    save_freq: int = 30
    eval_freq: int = 1
    save_dir: str = 'checkpoints/'

@dataclass
class InferenceParams:
    device: str = 'cuda'
    batch_size: int = 16
    model_path: str = 'checkpoints/best_model.pth'
    asr_model_name: str = 'sashat/whisper-medium-ClassicalAr'
    use_asr: bool = True
    forced_ids: list = None  # List of tuples for forced decoder ids
    output_path: str = 'results/predictions.txt'

@dataclass
class DataParams:
    train_path: str = 'data/clartts/clartts_asr_train.tsv'
    test_path: str = 'data/clartts/clartts_asr_test.tsv'

@dataclass
class TrainConfig:
    constants_path: str = 'constants/'
    train: TrainParams = TrainParams()
    data: DataParams = DataParams()
    model: ModelParams = ModelParams()

    @classmethod
    def from_yml(cls, yml_path: str):
        import yaml
        from dacite import from_dict
        with open(yml_path, 'r') as f:
            yml_config = yaml.safe_load(f)
        return from_dict(cls, yml_config)

@dataclass
class InferenceConfig:
    constants_path: str = 'constants/'
    inference: InferenceParams = InferenceParams()
    model: ModelParams = ModelParams()
    data: DataParams = DataParams()

    @classmethod
    def from_yml(cls, yml_path: str):
        import yaml
        from dacite import from_dict
        with open(yml_path, 'r') as f:
            yml_config = yaml.safe_load(f)
        return from_dict(cls, yml_config)