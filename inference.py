from model import *
import torch
from utils import *
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from model import ModifiedTransformerModel
import csv
from tqdm import tqdm


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
    def __init__(self, model_path, model_class, device='cpu', asr_model=None, asr_vocab=None, **kwargs):
        self.device = device
        model_args = kwargs.get('model', {})
        self.max_length = kwargs.get('data', {}).max_length
        self.model = model_class(**model_args.__dict__)

        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)

        self.asr_model = asr_model
        self.asr_vocab = asr_vocab
        self.model.eval()
    
    def prep_data(self, text, audio_path=None, text_asr=None):
        if isinstance(text, str):
            text = [text]

        if isinstance(text_asr, str):
            text_asr = [text_asr]

        
        X, _ = map_data(text)

        # pad sequences to the max length in the batch
        X = [seq + [constants.characters_mapping.get('<PAD>', 0)] * (self.max_length+2 - len(seq)) for seq in X]

        inputs = torch.tensor(X, dtype=torch.long)
    
        if text_asr:
            X_asr = map_asr_data(text_asr, self.asr_vocab)
            input_asr = torch.tensor(X_asr, dtype=torch.long).unsqueeze(0).to(self.device)
            return inputs.to(self.device), input_asr
        
        if self.asr_model and audio_path:
            asr_transcription = self.asr_model.transcribe(audio_path)
            X_asr = map_asr_data([asr_transcription], self.asr_vocab)
            input_asr = torch.tensor(X_asr, dtype=torch.long).unsqueeze(0).to(self.device)
            return inputs.to(self.device), input_asr
        
        return inputs.to(self.device), None

    def predict(self,text, audio_path=None, text_asr=None):
        text = remove_diacritics(text).strip()
        with torch.no_grad():
            inputs, input_asr = self.prep_data(text, audio_path=audio_path, text_asr=text_asr)
            outputs = self.model(inputs, input_asr=input_asr)
            predictions = outputs.argmax(dim=-1).squeeze(0).cpu().tolist()

        # decode predictions
        decoded_text = decode_predictions(predictions[1:-1], text)
        return decoded_text
    
    def predict_long(self, text, audio_path=None, text_asr=None):
        self.model.eval()
        text = remove_diacritics(text).strip()
        output = ""
        if len(text) > self.max_length:
            start_idx = 0
            end_idx = 50

            while end_idx < len(text):
                start = max(0, start_idx - 25)
                end_idx = min(len(text), start_idx + 50)
                end = min(len(text), end_idx + 25)

                chunk = text[start:end]
                with torch.no_grad():
                    inputs, _ = self.prep_data(chunk, audio_path=audio_path, text_asr=text_asr)
                    outputs = self.model(inputs)
                    res = outputs.argmax(dim=-1).squeeze(0).cpu().tolist()
                    res = res[1:-1]  # remove <SOS> and <EOS>
                for i in range (start_idx-start, end_idx-start):
                    #print (i, _line)
                    char=chunk[i]
                    prediction=res[i]
                    output+= char
                    if char not in constants.arabic_letters_list:
                        continue
                    if '<' in constants.rev_classes_mapping[prediction]:
                        continue
                    output += constants.rev_classes_mapping[prediction]

                start_idx = end_idx
            return output
        else:
            output = self.predict(text, audio_path=audio_path, text_asr=text_asr)
            return output

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
                for line in tqdm(reader, desc="Processing lines"):
                    if not line:
                        continue
                    diacritized_line = self.predict_long(line[0])
                    f_out.write(diacritized_line + '\n')
            return


def main(config):

    global constants
    constants = load_constants(config.CONSTANTS_PATH)
    expanded_vocab = expand_vocabulary(constants.characters_mapping, constants.classes_mapping)
    # setup ASR model if needed
    if config.INFERENCE.USE_ASR:
        processor = AutoProcessor.from_pretrained(config.INFERENCE.ASR_MODEL_NAME)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(config.INFERENCE.ASR_MODEL_NAME)
        forced_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
        asr_model = AsrModel(model, processor, device=config.INFERENCE.DEVICE, forced_ids=forced_ids)
    else:
        asr_model = None

    inference_params = dict(config.INFERENCE)
    inference_kwargs = {k.lower(): v for k, v in inference_params.items()}
    diacritizer = Diacritize(model_path=config.INFERENCE.MODEL_PATH,
                            model_class=ModifiedTransformerModel,
                            asr_model=asr_model,
                            asr_vocab=expanded_vocab,
                            **inference_kwargs
                            )

    diacritizer.predict_file(config.DATA.TEST_PATH, config.INFERENCE.OUTPUT_PATH)

if __name__ == "__main__":
    config = load_cfg()
    main(config)