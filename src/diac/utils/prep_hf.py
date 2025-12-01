import os, sys, time
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import load_dataset, load_from_disk, Dataset
import numpy as np
import argparse

from torchcodec.decoders import AudioDecoder

os.environ["PYTHONIOENCODING"] = "utf-8"


OUT_DIR = "."

AVAILABLE_DATASETS = {
    'nadi-all': {
        'dataset_name': 'MBZUAI/NADI-2025-Sub-task-3-all',
            'audio_column': 'audio',
            'text_column': 'transcription',
            'splits': {
                'train' : ['train', 'augment'],
                'val' : ['validation'],
                'test' : ['test', 'evaluation', 'eval']
                }
    },
    'clartts': {
        'dataset_name': 'MBZUAI/ClArTTS',
        'audio_column': 'audio',
        'text_column': 'transcription',
        'splits': {
            'train' : ['train', 'augment'],
            'val' : ['validation', 'dev'],
            'test' : ['test', 'evaluation', 'eval']
        }
    },
    'tunswitch': {
        'dataset_name': 'MBZUAI/TunSwitch',
        'audio_column': 'audio',
        'text_column': 'transcription',
        'splits': {
            'train' : ['train', 'augment'],
            'val' : ['validation', 'dev'],
            'test' : ['test', 'evaluation', 'eval']
        }
    },
    'qasr': {
        'dataset_name': '/l/QASR_TTS/QASRTTS_HF',
        'audio_column': 'audio',
        'text_column': 'arabic_text',
        'splits': {
            # 'train' : ['train'],
            'test' : ['validation']
        }
    },
    'nadi-test':{
        'dataset_name': 'MBZUAI/NADI-2025-Sub-task-3-test',
        'audio_column': 'audio',
        'text_column': 'transcription',
        'splits': {
            'test' : ['test']
        }
    },
    # 'mdaspc': {'dataset_name': 'herwoww/mdaspc',
    #            'audio_column': 'audio',
    #            'text_column': 'transcription',
    #            'splits': ['train','dev']
    #            },
}

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("sashat/whisper-medium-ClassicalAr")
model = AutoModelForSpeechSeq2Seq.from_pretrained("sashat/whisper-medium-ClassicalAr").to(device).eval()
forced_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")

def _get_Audio(audio_item):
    """
    Given an audio item from a Huggingface dataset, return the audio array and sampling rate
    """
    if isinstance(audio_item, AudioDecoder) or (isinstance(audio_item, dict) and "array" in audio_item):
        audio_array = audio_item["array"]
        sampling_rate = audio_item["sampling_rate"]
    else:
        audio_array = np.array(audio_item, dtype=np.float32)
        sampling_rate = 16000  # default assumption
    return audio_array, sampling_rate

def transcribe_from_dataset(dataset, audio_column="audio", text_column="transcription", limit=None, batch_size=32):
    """
    Run batch transcription on audio files from a Huggingface dataset
    """
    results = []
    text_inputs = []
    samples = dataset if limit is None else dataset.select(range(min(limit, len(dataset))))
    
    # Process in batches
    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples.select(range(i, min(i + batch_size, len(samples))))
        batch_audio = []
        batch_texts = []
        
        # Prepare batch data
        for item in batch:
            # Extract audio array and sampling rate from dataset
            audio, sr = _get_Audio(item[audio_column])

            # resample if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            batch_audio.append(audio)
            batch_texts.append(item.get(text_column, ""))
        
        # Process batch through model
        inputs = processor(batch_audio,
                          sampling_rate=16000,
                          return_tensors="pt",
                          padding=True,
                          return_attention_mask=True)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_ids = model.generate(**inputs,
                                   forced_decoder_ids=forced_ids,
                                   pad_token_id=processor.tokenizer.pad_token_id)
        
        # Decode batch results
        transcriptions = processor.batch_decode(gen_ids, skip_special_tokens=True)
        
        # Store results
        text_inputs.extend(batch_texts)
        results.extend(transcriptions)
    
    return text_inputs, results

def main(args):

    for ds in args.datasets:
        if ds not in AVAILABLE_DATASETS:
            print(f"Dataset '{ds}' not recognized. Available datasets: {', '.join(AVAILABLE_DATASETS.keys())}")
            continue
        
        dataset_info = AVAILABLE_DATASETS[ds]
        dataset_name = dataset_info['dataset_name']
        audio_column = dataset_info['audio_column']
        text_column = dataset_info['text_column']
        all_splits = dataset_info['splits']
        output_dir = os.path.join(args.out_dir, ds)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing dataset: {ds} ({dataset_name})")

        for split_name, splits in all_splits.items():
            text_inputs = []
            results = []
            for split in splits:
                try:
                    dataset = load_dataset(dataset_name, split=split)
                except:
                    print(f"Failed to load dataset '{dataset_name}' with split '{split}' trying locally.")
                    try: 
                        dataset = load_from_disk(dataset_name)[split]
                        # dataset = Dataset.from_dict(dataset)

                    except:
                        print(f"Also failed to load from disk '{dataset_name}' with split '{split}'. Skipping...")
                        continue
                breakpoint()
                try:
                    temp_text_inputs, temp_results = transcribe_from_dataset(dataset, audio_column=audio_column, text_column=text_column, limit=None, batch_size=32)
                except Exception as e:
                    print(f"Error during transcription for dataset '{ds}' split '{split}': {e}")
                    continue
                text_inputs.extend(temp_text_inputs)
                results.extend(temp_results)
            # write results to tsv file of form <original text> \t <asr text>
            out_path = os.path.join(output_dir, f"{split_name}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                for original, asr in zip(text_inputs, results):
                    f.write(f"{original}\t{asr}\n")

            if split_name == 'test':
                # also write original text only to a separate file for evaluation
                asr_out_path = os.path.join(output_dir, f"{split_name}_only.txt")
                with open(asr_out_path, "w", encoding="utf-8") as f:
                    for original in text_inputs:
                        f.write(f"{original}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Dataset for diacritization")
    parser.add_argument('--out_dir', type=str, default='data/', help="Output directory to save the processed files")
    parser.add_argument('--datasets', type=str, nargs='+', default=['clartts'], help="List of dataset names to process. Available: " + ", ".join(AVAILABLE_DATASETS.keys()))
    args = parser.parse_args()
    
    main(args)