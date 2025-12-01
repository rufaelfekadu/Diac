from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import librosa


class AsrModel:
    def __init__(self, model_name, device="cpu", forced_ids=None):

        self.device = device
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.forced_ids = forced_ids

    def transcribe(self, audio):
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=16000)
        else:
            sr = 16000  # assume audio is already loaded and resampled

        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                forced_decoder_ids=self.forced_ids,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        transcription = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[
            0
        ]
        return transcription

    def transcribe_batch(self, audio_list, output_file=None):
        transcriptions = []
        for audio in audio_list:
            transcription = self.transcribe(audio)
            transcriptions.append(transcription)
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                for line in transcriptions:
                    f.write(line + "\n")
        return transcriptions
