from sys import exception
import torch
import torch.nn as nn
import math
import lightning as L
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


from tokenizer import ArabicDiacritizationTokenizer

class AsrModel:
    def __init__(self, model_name, device='cpu', forced_ids=None):

        self.device = device
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
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
    
class SinePositionEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        
        # Create the sinusoidal positional encodings once in log space
        position = torch.arange(max_len).unsqueeze(1)          # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * 
                             -(math.log(10000.0) / embed_dim)) # (embed_dim/2,)

        pe = torch.zeros(max_len, embed_dim)                   # (max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim) for broadcasting
        self.register_buffer("pe", pe)  # not a parameter, but moves with .to(device)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
    
class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = SinePositionEncoding(embed_dim, maxlen)
        # self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, inputs):
        x = self.token_emb(inputs)
        b, t, e = x.size()
        if isinstance(self.pos_emb, nn.Embedding):
            x_pos = torch.arange(t, device=inputs.device).unsqueeze(0).expand_as(inputs)
            return x + self.pos_emb(x_pos)
        
        x = self.pos_emb(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.5):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        # self.in_proj = nn.Linear(d_model, d_model * num_heads)
        self.multi_head_attention = nn.MultiheadAttention(d_model*num_heads, num_heads, dropout=rate, batch_first=True)
        self.dropout1 = nn.Dropout(rate)
        self.out_proj = nn.Linear(d_model*num_heads, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.dropout2 = nn.Dropout(rate)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, mask=None):
        # Self-attention
        # inputs = self.in_proj(inputs)
        # brodcast the last dim of inputs to (batch_size, seq_len, d_model, num_heads)
        inputs_pre = inputs
        inputs = inputs.repeat(1, 1, self.num_heads)  # (batch_size, seq_len, d_model * num_heads)
        inputs = inputs.view(inputs.size(0), inputs.size(1), self.d_model*self.num_heads)

        attention_output, _ = self.multi_head_attention(inputs, inputs, inputs, attn_mask=mask)
        attention_output = self.dropout1(attention_output)
        attention_output = self.out_proj(attention_output)
        attention_output = self.layer_norm1(inputs_pre + attention_output)

        # Feed-forward
        ffn_output = self.ffn(attention_output)
        ffn_output = self.dropout2(ffn_output)
        block_output = self.layer_norm2(attention_output + ffn_output)

        return block_output

class LSTMModel(nn.Module):
    @classmethod
    def from_config(cls, config):
        return cls(
            maxlen=config.MODEL.MAXLEN,
            vocab_size=config.MODEL.VOCAB_SIZE,
            asr_vocab_size=config.MODEL.ASR_VOCAB_SIZE,
            output_size=config.MODEL.OUTPUT_SIZE,
            d_model=config.MODEL.D_MODEL,
            num_heads=config.MODEL.NUM_HEADS,
            dff=config.MODEL.DFF,
            num_blocks=config.MODEL.NUM_BLOCKS,
            dropout_rate=config.MODEL.DROPOUT_RATE,
            with_conn=config.MODEL.WITH_CONN,
            use_asr=config.MODEL.USE_ASR
        )
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def __init__(self, maxlen, vocab_size, asr_vocab_size, output_size, d_model, num_heads, dff, num_blocks, dropout_rate=0.5, with_conn=False, use_asr=True, **kwargs):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.asr_vocab_size = asr_vocab_size
        self.output_size = output_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.with_conn = with_conn
        self.use_asr = use_asr
        self.num_layers = num_blocks

        # Text branch
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.text_lstm = nn.LSTM(d_model, d_model, num_layers=num_blocks, dropout=dropout_rate, bidirectional=True, batch_first=True)
        self.text_ffn = nn.Sequential(
            nn.Linear(2*d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, dff),
            nn.ReLU()
        )

        # ASR branch
        if use_asr:
            self.asr_embedding = nn.Embedding(asr_vocab_size, d_model)
            self.asr_lstm = nn.LSTM(d_model, d_model, bidirectional=True, batch_first=True, num_layers=num_blocks, dropout=dropout_rate)
            self.asr_ffn = nn.Sequential(
                nn.Linear(2*d_model, dff),
                nn.ReLU(),
                nn.Linear(dff, d_model),
                nn.ReLU()
            )

            # Cross-attention
            self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
            combined_dim = d_model * 2 if with_conn else d_model

            self.final_dense = nn.Linear(combined_dim, output_size)
        else:
            self.final_dense = nn.Linear(dff, output_size)
            
        self.init_params()

    def forward(self, inputs, inputs_asr=None, **kwargs):
        # Text branch
        text_emb = self.text_embedding(inputs)
        text_out, _ = self.text_lstm(text_emb)
        text_out = self.text_ffn(text_out)

        if not self.use_asr:
            outputs = self.final_dense(text_out)
            return outputs
        
        # ASR branch
        asr_emb = self.asr_embedding(inputs_asr)
        asr_out, _ = self.asr_lstm(asr_emb)
        asr_out = self.asr_ffn(asr_out)

        # Cross-attention
        cross_out, _ = self.cross_attention(text_out.transpose(0, 1), asr_out.transpose(0, 1), asr_out.transpose(0, 1))
        cross_out = cross_out.transpose(0, 1)

        # Combine
        if self.with_conn:
            combined = torch.cat([text_out, cross_out], dim=-1)
        else:
            combined = cross_out

        outputs = self.final_dense(combined)

        return outputs

    def load_pretrained(self, pretrained_model_path, text_branch_only=False):
        if not pretrained_model_path:
            print("No pretrained model path provided, skipping loading pretrained weights.")
            return self
        
        try:
            # Load Lightning checkpoint
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                # Extract model weights from Lightning checkpoint
                pretrained_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() 
                                   if k.startswith('model.')}
            else:
                # Handle plain state dict
                pretrained_dict = checkpoint
            
            model_dict = self.state_dict()
            
            if text_branch_only:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                  if k.startswith('text_') or k.startswith('final_dense')}
            
            # Update the current model's state dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Loaded pretrained weights from {pretrained_model_path}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
        
        return self

class TransformerModel(nn.Module):
    
    @classmethod
    def from_config(cls, config):
        return cls(
            maxlen=config.MODEL.MAXLEN,
            vocab_size=config.MODEL.VOCAB_SIZE,
            asr_vocab_size=config.MODEL.ASR_VOCAB_SIZE,
            d_model=config.MODEL.D_MODEL,
            num_heads=config.MODEL.NUM_HEADS,
            dff=config.MODEL.DFF,
            num_blocks=config.MODEL.NUM_BLOCKS,
            output_size=config.MODEL.OUTPUT_SIZE,
            dropout_rate=config.MODEL.DROPOUT_RATE,
            with_conn=config.MODEL.WITH_CONN,
            use_asr=config.MODEL.USE_ASR
        )
    
    def __init__(self, maxlen, vocab_size, asr_vocab_size, d_model, num_heads, dff, num_blocks, output_size, with_conn=False, dropout_rate=0.5, use_asr=True, **kwargs):
        super(TransformerModel, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.asr_vocab_size = asr_vocab_size
        self.output_size = output_size
        self.with_conn = with_conn
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_asr = use_asr

        # Text branch
        self.text_embedding = TokenAndPositionEmbedding(maxlen, vocab_size, d_model)
        self.text_transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_blocks)
        ])

        # ASR branch
        if use_asr:
            self.asr_embedding = TokenAndPositionEmbedding(maxlen, asr_vocab_size, d_model*num_heads)
            self.asr_transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_blocks)
            ])

            # Cross-attention
            self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
            if with_conn:
                self.final_dense = nn.Linear(d_model * 2, output_size)
            else:
                self.final_dense = nn.Linear(d_model, output_size)
        else:
            self.final_dense = nn.Linear(d_model, output_size)

    def forward(self, inputs, inputs_asr=None, **kwargs):
        # Text branch
        x = self.text_embedding(inputs)
        for block in self.text_transformer_blocks:
            x = block(x)

        if not self.use_asr:
            outputs = self.final_dense(x)
            return outputs
        
        # ASR branch
        asr_emb = self.asr_embedding(inputs_asr)
        for block in self.asr_transformer_blocks:
            asr_emb = block(asr_emb)

        # Cross-attention
        cross_out, _ = self.cross_attention(x.transpose(0, 1), asr_emb.transpose(0, 1), asr_emb.transpose(0, 1))
        cross_out = cross_out.transpose(0, 1)

        # Combine
        if self.with_conn:
            combined = torch.cat([x, cross_out], dim=-1)
        else:
            combined = cross_out

        outputs = self.final_dense(combined)

        return outputs

    def load_pretrained(self, pretrained_model_path, text_branch_only=False):

        if not pretrained_model_path:
            print("No pretrained model path provided, skipping loading pretrained weights.")
            return self
        
        try:
            # Load Lightning checkpoint
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                # Extract model weights from Lightning checkpoint
                pretrained_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() 
                                   if k.startswith('model.')}
            else:
                # Handle plain state dict
                pretrained_dict = checkpoint
            
            model_dict = self.state_dict()
            
            if text_branch_only:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                  if k.startswith('text_') or k.startswith('final_dense')}
            
            # Update the current model's state dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Loaded pretrained weights from {pretrained_model_path}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
        
    def predict(self, inputs, inputs_asr=None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs, inputs_asr=inputs_asr)
            predictions = outputs.argmax(dim=-1)
        return predictions

class DiacritizationModule(L.LightningModule):
    """PyTorch Lightning module for diacritization models."""

    def __init__(self, config, tokenizer: ArabicDiacritizationTokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer

        if config.MODEL.TYPE not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {config.MODEL.TYPE}")
        
        model_class = AVAILABLE_MODELS[config.MODEL.TYPE]
        
        self.model = model_class.from_config(config)
        
        # Load pretrained weights if specified
        if hasattr(config.MODEL, 'PRETRAINED_PATH') and config.MODEL.PRETRAINED_PATH:
            self.model.load_pretrained(
                config.MODEL.PRETRAINED_PATH, 
                text_branch_only=getattr(config.MODEL, 'LOAD_TEXT_BRANCH_ONLY', False)
            )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        
        # Metrics
        self.train_accuracy = []
        self.val_accuracy = []
    
    def forward(self, inputs, inputs_asr=None):
        return self.model(inputs, inputs_asr=inputs_asr)
    
    def training_step(self, batch, batch_idx):
        inputs, inputs_asr, targets = batch
        
        # Forward pass
        outputs = self.forward(inputs, inputs_asr=inputs_asr)
        
        # Calculate loss
        loss = self.criterion(outputs.permute(0, 2, 1), targets)
        
        # Calculate accuracy
        pred = outputs.argmax(dim=-1)
        correct = (pred == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, inputs_asr, targets = batch
        
        # Forward pass
        outputs = self.forward(inputs, inputs_asr=inputs_asr)
        
        # Calculate loss
        loss = self.criterion(outputs.permute(0, 2, 1), targets)
        
        # Calculate accuracy
        pred = outputs.argmax(dim=-1)
        correct = (pred == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': accuracy}
    
    def test_step(self, batch, batch_idx):
        inputs, inputs_asr, targets = batch
        
        # Forward pass
        outputs = self.forward(inputs, inputs_asr=inputs_asr)
        
        # Calculate loss
        loss = self.criterion(outputs.permute(0, 2, 1), targets)
        
        # Calculate accuracy
        pred = outputs.argmax(dim=-1)
        correct = (pred == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'test_loss': loss, 'test_acc': accuracy}
    
    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), 
            lr=getattr(self.config.TRAIN, 'LEARNING_RATE', 1e-4)
        )
        
        # Optional: Add learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def predict_step(self, batch, batch_idx):
        inputs, inputs_asr, _ = batch
        outputs = self.forward(inputs, inputs_asr=inputs_asr)
        predictions = outputs.argmax(dim=-1)
        return predictions
    
    def predict_text(self, text, asr_text=[]):
        self.model.eval()
        
        if isinstance(text, str):
            text = [text]
        
        if isinstance(asr_text, str):
            asr_text = [asr_text]
        
        encoded_text, encoded_asr, _ = self.tokenizer.encode_batch(text, asr_text, padding=True)
        encoded_text = encoded_text.to(self.device)
        encoded_asr = encoded_asr.to(self.device) if self.config.INFERENCE.USE_ASR else None

        with torch.no_grad():
            outputs = self.model(encoded_text, inputs_asr=encoded_asr)
            predictions = outputs.argmax(dim=-1).cpu().tolist()

        decoded_texts = self.tokenizer.decode_batch(predictions, text)

        return decoded_texts

    def predict_sliding_window(self, text, text_asr=[]):
        self.model.eval()
        original_text = text


        text = self.remove_diacritics(text).strip()
        if len(text) <= self.config.INFERENCE.MAX_LENGTH:
            output = self.predict_text(text, asr_text=text_asr)[0]
        else:
            # Sliding window
            window_size = self.config.INFERENCE.WINDOW_SIZE
            buffer_size = getattr(self.config.INFERENCE, 'BUFFER_SIZE', 25)
            start_idx = 0
            end_idx = window_size
            output = ""
            
            while end_idx < len(text):
                start = max(0, start_idx - buffer_size)
                end = min(len(text), end_idx + window_size + buffer_size)
                end_idx = min(len(text), start_idx + window_size)

                chunk = text[start:end]
                encoded_chunk, encoded_asr_chunk, _ = self.tokenizer.encode_batch(
                    [chunk], 
                    padding=True
                )
                encoded_chunk = encoded_chunk.to(self.device)
                encoded_asr_chunk = encoded_asr_chunk.to(self.device) if text_asr else None
                
                with torch.no_grad():
                    outputs = self.model(encoded_chunk, inputs_asr=encoded_asr_chunk).squeeze(0)
                    predictions = outputs.argmax(dim=-1).cpu().tolist()[1:-1]  # remove <sos> and <eos> 

                decoded_chunk = self.tokenizer.decode(predictions[start_idx:end_idx], chunk[start_idx:end_idx])
                
                output += decoded_chunk  

                start_idx = end_idx
            
        return output
    
    def remove_diacritics(self, text:str) -> str:
        return text.translate(str.maketrans('', '', ''.join(self.tokenizer.constants.diacritics_list)))

    @staticmethod
    def is_audio(path: str) -> bool:
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
        return any(path.lower().endswith(ext) for ext in audio_extensions)
    
    def predict_file(self, input_file, output_file):

        # clear output file if exists
        open(output_file, 'w').close()

        # determine the structure of the input file| if first col is audio paths or text
        with open(input_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            is_audio_file = self.is_audio(first_line.split('\t')[0])
            is_asr_text = len(first_line.split('\t')) > 1 and not is_audio_file

        if is_audio_file and self.config.INFERENCE.USE_ASR:
            # input file contains audio paths
            asr_model = AsrModel(
                model_name=self.config.INFERENCE.ASR_MODEL_NAME,
                device=self.config.INFERENCE.DEVICE,
                forced_ids=None
            ) 
            print("using audio files...")
            with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                reader = csv.reader(f_in, delimiter='\t')
                for line in reader:
                    if not line:
                        continue
                    audio_path = line[0]
                    asr_text  = asr_model.transcribe(audio_path) if asr_model else None
                    diacritized_line = self.predict_sliding_window(line[1], text_asr=asr_text)
                    f_out.write(diacritized_line + '\n')
            return

        elif is_asr_text and self.config.INFERENCE.USE_ASR:
            # input file contains ASR text in second column
            print("using ASR text...")
            with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                reader = csv.reader(f_in, delimiter='\t')
                for line in tqdm(reader, desc="Processing lines"):
                    if not line:
                        continue
                    diacritized_line = self.predict_sliding_window(line[0], text_asr=line[1])[0]
                    f_out.write(diacritized_line + '\n')
            return
        
        else:
            print("without ASR text...")
            with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                reader = csv.reader(f_in, delimiter='\t')
                for line in tqdm(reader, desc="Processing lines"):
                    if not line:
                        continue
                    diacritized_line = self.predict_sliding_window(line[0])
                    f_out.write(diacritized_line + '\n')
            return
        
AVAILABLE_MODELS = {
    'Transformer': TransformerModel,
    'LSTM': LSTMModel
}

if __name__ == "__main__":


    # model = LSTMModel(
    #         maxlen=100, 
    #         vocab_size=1000,
    #         asr_vocab_size=1200,
    #         output_size=15,
    #         d_model=128,
    #         num_heads=4,
    #         dff=128,
    #         num_blocks=2,
    #         dropout_rate=0.2,
    #         with_conn=False
    #     )

    model = TransformerModel(
            maxlen=100, 
            vocab_size=1000, 
            asr_vocab_size=1200, 
            d_model=128, 
            num_heads=4, 
            dff=128, 
            num_blocks=2, 
            output_size=19,
            use_asr=False,
        )

    input_text = torch.randint(0, 1000, (32, 80))  # Batch of 32 samples, each of length 100
    input_asr = torch.randint(0, 1200, (32, 98))
    output = model(inputs=input_text, inputs_asr=input_asr)
    print(output.shape)  # Should be (32, 100, 15)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # visualize the computational graph
    # from torchviz import make_dot
    # make_dot(output, params=dict(model.named_parameters())).render("rnn_torchviz_asr", format="png")
