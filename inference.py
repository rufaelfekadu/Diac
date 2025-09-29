from model import *
import torch
from utils import map_data, map_asr_data, decode_predictions, load_constants
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from config import InferenceConfig
from model import ModifiedTransformerModel
import csv
import pandas as pd

class AsrModel:
    def __init__(self, model, processor, device='cpu', forced_ids=None):
        self.device = device
        self.model = model
        self.model.to(device)
        self.processor = processor
        self.forced_ids = forced_ids

    def transcribe(self, audio):
        if isinstance(audio, str):
            import librosa
            audio, sr = librosa.load(audio, sr=16000)
        else:
            sr = 16000  # assume audio is already loaded and resampled

        inputs = self.processor(audio,
                                sampling_rate=sr,
                                return_tensors="pt",
                                padding=True,
                                return_attention_mask=True)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs,
                                         forced_decoder_ids=self.forced_ids,
                                         pad_token_id=self.processor.tokenizer.pad_token_id)
        
        transcription = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return transcription

    def transcribe_batch(self, audio_list, output_file=None):
        transcriptions = []
        for audio in audio_list:
            transcription = self.transcribe(audio)
            transcriptions.append(transcription)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in transcriptions:
                    f.write(line + '\n')
        return transcriptions
    
class Diacritize:
    def __init__(self, model_path, model_class, device='cpu', asr_model=None, **kwargs):
        self.device = device
        model_args = kwargs.get('model', {})
        self.model = model_class(**model_args.__dict__)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)

        self.asr_model = asr_model
        self.model.eval()
    
    def prep_data(self, text, audio_path=None, text_asr=None):
        X, _ = map_data([text])
        inputs = torch.tensor(X, dtype=torch.long)

        if text_asr:
            X_asr = map_asr_data([text_asr], self.asr_model.processor.tokenizer.get_vocab())
            input_asr = torch.tensor(X_asr, dtype=torch.long).unsqueeze(0).to(self.device)
            return inputs.to(self.device), input_asr
        
        if self.asr_model and audio_path:
            asr_transcription = self.asr_model.transcribe(audio_path)
            X_asr = map_asr_data([asr_transcription], self.asr_model.processor.tokenizer.get_vocab())
            input_asr = torch.tensor(X_asr, dtype=torch.long).unsqueeze(0).to(self.device)
            return inputs.to(self.device), input_asr
        
        return inputs.to(self.device), None

    def predict(self, text, audio_path=None, text_asr=None):
        self.model.eval()
        with torch.no_grad():
            inputs, input_asr = self.prep_data(text, audio_path=audio_path, text_asr=text_asr)
            outputs = self.model(inputs, input_asr=input_asr)
            predictions = outputs.argmax(dim=-1).squeeze(0).cpu().tolist()

        # decode predictions
        decoded_text = decode_predictions(predictions, text)
        return decoded_text
    @staticmethod
    def is_audio(file_path):
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
        return any(file_path.lower().endswith(ext) for ext in audio_extensions)
    
    def predict_file(self, input_file, output_file):

        # determine the structure of the input file| if first col is audio paths or text
        with open(input_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            is_audio_file = self.is_audio(first_line.split('\t')[0])
            is_asr_text = len(first_line.split('\t')) > 1 and not is_audio_file

        if is_audio_file:
            # input file contains audio paths
            print("using audio files...")
            with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                reader = csv.reader(f_in, delimiter='\t')
                for line in reader:
                    if not line:
                        continue
                    audio_path = line[0]
                    diacritized_line = self.predict(line[1], audio_path=audio_path)
                    f_out.write(diacritized_line + '\n')
            return

        elif is_asr_text:
            # input file contains ASR text in second column
            print("using ASR text...")
            with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                reader = csv.reader(f_in, delimiter='\t')
                for line in reader:
                    if not line:
                        continue
                    diacritized_line = self.predict(line[0], text_asr=line[1])
                    f_out.write(diacritized_line + '\n')
            return
        
        else:
            print("without ASR text...")
            with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                reader = csv.reader(f_in, delimiter='\t')
                for line in reader:
                    if not line:
                        continue
                    diacritized_line = self.predict(line[0])
                    f_out.write(diacritized_line + '\n')
            return


def main():
    # playground for testing diacritization
    config = InferenceConfig.from_yml('configs/transformer.clartts.yml')
    constants = load_constants(config.constants_path)

    # setup ASR model if needed
    if config.inference.use_asr:
        processor = AutoProcessor.from_pretrained("sashat/whisper-medium-ClassicalAr")
        model = AutoModelForSpeechSeq2Seq.from_pretrained("sashat/whisper-medium-ClassicalAr")
        forced_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
        asr_model = AsrModel(model, processor, device=config.inference.device, forced_ids=forced_ids)
    else:
        asr_model = None

    diacritizer = Diacritize(model_path=config.inference.model_path,
                            model_class=ModifiedTransformerModel,
                            asr_model=asr_model,
                            **config.__dict__)
    
    diacritizer.predict_file(config.data.train_path, config.inference.output_path)

if __name__ == "__main__":
    main()