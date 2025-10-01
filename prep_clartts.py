import os, sys, time
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import load_dataset

OUT_DIR = "./data/clartts"


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("sashat/whisper-medium-ClassicalAr")
model = AutoModelForSpeechSeq2Seq.from_pretrained("sashat/whisper-medium-ClassicalAr").to(device).eval()
forced_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")

def transcribe_from_dataset(dataset, audio_column="audio", limit=None, batch_size=32):
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
            audio = item[audio_column]["array"]
            sr = item[audio_column]["sampling_rate"]

            # resample if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            batch_audio.append(audio)
            batch_texts.append(item.get("transcription", ""))
        
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


def main():
    splits = ['train','test']
    for split in splits:
        dataset = load_dataset("AtharvA7k/ClArTTS", split=split, cache_dir=OUT_DIR)
        print(f"Processing split: {split} with {len(dataset)} samples")
        text_inputs, results = transcribe_from_dataset(dataset)

        # write results to tsv file of form <original text> \t <asr text>
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, f"clartts_asr_{split}.tsv")
        with open(out_path, "w", encoding="utf-8") as f:
            for original, asr in zip(text_inputs, results):
                f.write(f"{original}\t{asr}\n")


if __name__ == "__main__":
    main()