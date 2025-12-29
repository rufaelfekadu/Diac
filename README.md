<h1 align="center">Automatic Restoration of Diacritics for Speech Data Sets</h1>

<div align="center">

<!-- [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) -->
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://aclanthology.org/2024.naacl-long.233.pdf)
[![Project](https://img.shields.io/badge/Project-Diacritization-red)](https://github.com/SaraShatnawi/Diacritization)
</div>


> **Accepted at NAACL 2024**

## Abstract
Automatic text-based diacritic restoration models often exhibit high diacritic error rates when applied to speech transcripts due to domain and style shifts in spoken language. In this work, we investigate improving automatic diacritic restoration for speech data by leveraging parallel spoken utterances. Specifically, we fine-tune the pre-trained Whisper ASR model on a relatively small amount of diacritized Arabic speech to produce rough diacritized transcripts for the utterances, which we then use as an additional input to diacritic restoration models. The proposed framework consistently improves performance over text-only baselines. Our results highlight the inadequacy of current text-based diacritic restoration models for speech datasets and establish a new baseline for speech-based diacritic restoration.


## Installation

```bash
git clone https://github.com/rufaelfekadu/Diac
cd Diac
conda create -n diac python=3.12
conda activate diac
pip install -r requirements.txt
```

## Getting Started

### 1. Prepare input data

Download the prepared CLArTTS and Tashkeela datasets:
```bash
git clone https://github.com/rufaelfekadu/arabic-diacritization-data.git data/
```

Alternatively, to prepare CLArTTS from scratch, run:

```bash 
python prep_clartts.py
```

This script generates train+asr.txt, test+asr.txt, and test.txt for the train and test splits in data/clartts/.


### 2. Training

#### a. Sweep
To run all experiments described in the paper, execute:
```bash
bash scripts/sweep.sh #[Options] --max-jobs <max number of jobs> --start <0,1,2> --stop_stage <0,1,2>
```

This will produce a folder structure similar to:
```text
results/
├── lstm-text+asr/
│   ├── clartts/
│   │   ├── logs/
│   │   │   ├── decode-*.log
│   │   │   ├── eval-*.log
│   │   │   └── train-*.log
│   │   ├── tensorboard/
│   │   │   └── version_0/
│   │   ├── predictions.txt
│   │   ├── training.done
│   │   └── inference.done
│   └── ...
├── transformer-text+asr/
│   └── ...
└── ...
```
The eval-*.log files contain the evaluation results.

#### b. Manual training

To train a model manually, run:
```bash
python train_lightning.py --config configs/transformer.yml --opts \
        DATA.TRAIN_PATH "data/tashkeela/train.txt" \
        DATA.TEST_PATH "data/tashkeela/test.txt" \
        MODEL.USE_ASR False \
        MODEL.LOAD_TEXT_BRANCH_ONLY False \
        TRAIN.SAVE_DIR "nadi-results/transformer-text+asr/tashkeela+nadi" \
        MODEL.PRETRAINED_PATH "outputs/results/transformer-text-only/tashkeela/tensorboard/version_0/checkpoints/best_model.ckpt"

```


Note: Set DATA.VAL_PATH only if a validation set is available; otherwise, the training set will be split automatically.

### Inference

To run inference, use the following. The test file can be one of:
- `.tsv`: a TSV file in either format: `(audio_paths\tundiacritized_text)` or `(undiacritized_text\tASR_output)`
- `.txt`: plain text file with one line of undiacritized text per line

```bash
python inference.py --config configs/transformer.yml --opts \
        DATA.TEST_PATH "data/clartts/test.txt" \
        MODEL.USE_ASR True \
        INFERENCE.MODEL_PATH "results-final/transformer-text+asr/tashkeela+clartts+clartts_aug/tensorboard/version_0/checkpoints/best_model.ckpt" \
        INFERENCE.OUTPUT_PATH "results-final/transformer-text+asr/tashkeela+clartts+clartts_aug/predictions.txt" \
        INFERENCE.USE_ASR True
```
This will create a text file with the predicted values at INFERENCE.OUTPUT_PATH. To evaluate, run:

```bash
python eval.py -ofp results-final/transformer-text+asr/tashkeela+clartts+clartts_aug/predictions.txt -tfp data/clartts/test_only.txt
```

### Hugging Face pretrained models

You can run inference using one of the pretrained models published on Hugging Face with the included `inference_hf.py` script.

Basic usage:

```bash
python inference_hf.py \
        --model_name rufaelfekadu/diac-transformer-text-asr-tashkeela-clartts \
        --test_path data/clartts/test.txt \
        --output_path outputs/hf_inference_output.txt
```

Notes:
- `--model_name` should be one of the available model IDs (see list below).
- `--test_path` can be a `.txt` file with one undiacritized sentence per line or a `.tsv` in the formats described above.
- The script uses `DiacritizationModule.from_pretrained(..., tokenizer_constants_path="constants/")`, so keep the repository `constants/` folder available in the working directory.

Available Hugging Face models:
1. `rufaelfekadu/diac-transforemer-text-only-tashkeela`
2. `rufaelfekadu/diac-transformer-text-asr-tashkeela-clartts`
3. `rufaelfekadu/diac-transformer-text-asr-tashkeela-clartts-kssa`


## Acknowledgments

This project builds upon or utilizes code and resources from:
- [Evaluation helper](https://github.com/AliOsm/arabic-text-diacritization)
- [CLArTTS Dataset](https://github.com/arabicsspeech/clarttscorpus)
- [Tashkeela Corpus](https://github.com/AliOsm/arabic-text-diacritization)

We thank all contributors to these resources for making their work available to the community.

## Citation

If you find this repository helpful, please cite our paper:

```bibtex
@inproceedings{shatnawi2024automatic,
  title={Automatic Restoration of Diacritics for Speech Data Sets},
  author={Shatnawi, Sara and Alqahtani, Sawsan and Aldarmaki, Hanan},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={4166--4176},
  year={2024}
  }
```
