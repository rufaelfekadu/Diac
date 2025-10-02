<h1 align="center"> Automatic Restoration of Diacritics for Speech Data Set </h1>

<div align="center">

<!-- [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) -->
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://aclanthology.org/2024.naacl-long.233.pdf)
[![Project](https://img.shields.io/badge/Project-Diacritization-red)](https://github.com/SaraShatnawi/Diacritization)
</div>


> **Accepted at NAACL 2024**

## Abstract
Automatic text-based diacritic restoration models generally have high diacritic error rates when applied to speech transcripts as a result of domain and style shifts in spoken language. In this work, we explore the possibility of improving the performance of automatic diacritic restoration when applied to speech data by utilizing parallel spoken utterances. In particular, we use the pre-trained Whisper ASR model fine-tuned on relatively small amounts of diacritized Arabic speech data to produce rough diacritized transcripts for the speech utterances, which we then use as an additional input for diacritic restoration models. The proposed framework consistently improves diacritic restoration performance compared to text-only baselines. Our results highlight the inadequacy of current text-based diacritic restoration models for speech data sets and provide a new baseline for speech-based diacritic restoration.


## 📦 Installation

```bash
git clone https://github.com/rufaelfekadu/Diac
cd Diacritization
conda create -n diac python=3.12
conda activate diac
pip install -r requirements.txt
```

## 📁 Data Preparation

### 1. Prepare Input Data

Download the prepared CLArTTS and Tashkeel datasets or

Run the following to download and prepare the CLArTTS data

```bash 
python prep_clartts.py
```

The script generates a .tsv file for train and test splits of the dataset in ./data/clartts dir

## Models

### Training
To train a text-only model on the tashkeela corpus run
```bash
python train.py --config configs/transformer.tashkeela.yml # path to the configuration file
```

To train text+asr model on the clartts dataset initialised from the tashkeela text-only model run,
```bash
python train.py --config configs/transformer.tashkeela+clartts.yml # path to the configuration file
```

### Inference

To Run inference update the test_path and output_path in the configuration yaml file and run the following.
Test file can be one of the following:
- `.tsv`: tsv file with format (audio_paths \t undiacritized_text) or (undiacritized_text \t ASR_output)
- `.txt`: lines of undiacritized text

```bash
python inference.py --config configs/transformer.clartts.yml
```
This will create a text file with the predicted values at output_path. To run the evaluation use

```bash
python eval.py -ofp /path/to/original-text -tfp /path/to/target-text
```

## Acknowledgments

This project builds upon or utilizes code and resources from:
- [Evaluation helper](https://github.com/AliOsm/arabic-text-diacritization)
- [CLArTTS Dataset](https://github.com/arabicsspeech/clarttscorpus)
- [Tashkeela Corpus](https://github.com/AliOsm/arabic-text-diacritization)

We thank all contributors to these resources for making their work available to the community.

## Citation

If you find this data annotations helpful, please cite our paper:

```bibtex
@inproceedings{shatnawi2024automatic,
  title={Automatic Restoration of Diacritics for Speech Data Sets},
  author={Shatnawi, Sara and Alqahtani, Sawsan and Aldarmaki, Hanan},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={4166--4176},
  year={2024}
  }
```
