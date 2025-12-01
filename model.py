from sys import exception
import torch
import torch.nn as nn
import math
import lightning as L
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
from tqdm import tqdm
from pyarabic import araby
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import os
import json
import tempfile
from types import SimpleNamespace

# Optional huggingface-hub helpers. If not installed, hub-related helpers will raise
# a clear error when used.
try:
    from huggingface_hub import hf_hub_download, snapshot_download, create_repo, Repository
except Exception:
    hf_hub_download = None
    snapshot_download = None
    create_repo = None
    Repository = None


from diac.tokenizer import ArabicDiacritizationTokenizer

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
        
        # implement SinePositionEncoding
        position = torch.arange(t, device=inputs.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, e, 2, device=inputs.device) *
                             -(math.log(10000.0) / e))
        
        pe = torch.zeros(t, e, device=inputs.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, t, embed_dim) for broadcasting

        x = x + pe

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.5):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        # self.in_proj = nn.Linear(d_model, d_model * num_heads)
        self.multi_head_attention = nn.MultiheadAttention(d_model, num_heads, dropout=rate, batch_first=True)
        self.dropout1 = nn.Dropout(rate)
        # self.out_proj = nn.Linear(d_model*num_heads, d_model)
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
        # inputs_pre = inputs
        # inputs = inputs.repeat(1, 1, self.num_heads)  # (batch_size, seq_len, d_model * num_heads)
        # inputs = inputs.view(inputs.size(0), inputs.size(1), self.d_model*self.num_heads)

        attention_output, _ = self.multi_head_attention(inputs, inputs, inputs, attn_mask=mask)
        attention_output = self.dropout1(attention_output)
        # attention_output = self.out_proj(attention_output)
        attention_output = self.layer_norm1(inputs + attention_output)

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
            checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
            
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
                                  if k.startswith('text_')
                                  }
            
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
            self.asr_embedding = TokenAndPositionEmbedding(maxlen, asr_vocab_size, d_model)
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
        
        self._init_params()

    def _init_params(self):
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
            checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
            
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
                                  if k.startswith('text_')}
            
            # Update the current model's state dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Loaded pretrained weights from {pretrained_model_path} with text_branch_only={text_branch_only}")
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

    @classmethod
    def from_pretrained(cls, repo_or_dir, config=None, tokenizer=None, tokenizer_constants_path=None,
                        map_location='cpu', device=None, hf_token=None, strict=False,
                        text_branch_only=False):
        """Instantiate DiacritizationModule from a local directory or Hugging Face repo.

        Args:
            repo_or_dir: local path or HF repo id (e.g. 'username/repo').
            config: optional config object expected by the module. If not provided,
                the method will attempt to load 'config.json' from the repo_or_dir
                and convert it to a SimpleNamespace with nested attributes.
            tokenizer: optional instance of `ArabicDiacritizationTokenizer`. If not
                provided, the method will try to locate a `constants/` directory in
                the repo_or_dir and construct a tokenizer from it. Alternatively,
                provide `tokenizer_constants_path` pointing to that directory.
            tokenizer_constants_path: optional local path to tokenizer constants.
            map_location: device map for torch.load.
            device: optional torch device to move the model to (e.g. 'cpu' or 'cuda').
            hf_token: huggingface token if required to access private repos.
            strict: whether to require exact state dict key match.
            text_branch_only: if True, only load state dict keys starting with 'text_'.

        Returns:
            An instance of DiacritizationModule with weights loaded.
        """

        def _download_file(fname):
            # local path first
            local_candidate = os.path.join(repo_or_dir, fname)
            if os.path.exists(local_candidate):
                return local_candidate
            # try HF hub
            if hf_hub_download is None:
                return None
            try:
                return hf_hub_download(repo_id=repo_or_dir, filename=fname, use_auth_token=hf_token)
            except Exception:
                return None

        # find checkpoint
        candidates = ['pytorch_model.bin', 'model.pt', 'model.pth', 'state_dict.pt', 'checkpoint.pt']
        ckpt_path = None
        for c in candidates:
            p = _download_file(c)
            if p is not None:
                ckpt_path = p
                break

        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in '{repo_or_dir}'. Tried: {candidates}")

        # load or build config
        if config is None:
            cfg_path = _download_file('config.json')
            if cfg_path:
                try:
                    with open(cfg_path, 'r', encoding='utf-8') as fh:
                        cfg_dict = json.load(fh)

                    def _dict_to_ns(d):
                        ns = SimpleNamespace()
                        for k, v in d.items():
                            if isinstance(v, dict):
                                setattr(ns, k, _dict_to_ns(v))
                            else:
                                setattr(ns, k, v)
                        return ns

                    config = _dict_to_ns(cfg_dict)
                except Exception:
                    config = None

        # build or load tokenizer
        if tokenizer is None:
            # use provided constants path if available
            if tokenizer_constants_path and os.path.isdir(tokenizer_constants_path):
                tokenizer = ArabicDiacritizationTokenizer(tokenizer_constants_path)
            else:
                # look for a local constants/ dir
                local_constants = os.path.join(repo_or_dir, 'constants')
                if os.path.isdir(local_constants):
                    tokenizer = ArabicDiacritizationTokenizer(local_constants)
                else:
                    # try snapshot_download to fetch constants/ from HF repo
                    if snapshot_download is not None:
                        try:
                            snap_dir = snapshot_download(repo_or_dir, allow_patterns=['constants/*'], use_auth_token=hf_token)
                            # find 'constants' folder inside snap_dir
                            candidate = os.path.join(snap_dir, 'constants')
                            if os.path.isdir(candidate):
                                tokenizer = ArabicDiacritizationTokenizer(candidate)
                        except Exception:
                            tokenizer = None

        if tokenizer is None:
            raise ValueError('Tokenizer instance not provided and constants directory not found in the repo.\n'
                             'Either pass a tokenizer instance or provide tokenizer_constants_path pointing to the constants folder.')

        if config is None:
            raise ValueError('Config not provided and config.json not found in the repo.\n'
                             'Please pass a config object to from_pretrained.')

        # instantiate module
        module = cls(config=config, tokenizer=tokenizer)

        # load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=map_location)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # Lightning checkpoint: extract model.* entries
            sd = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
        else:
            sd = checkpoint

        if text_branch_only:
            sd = {k: v for k, v in sd.items() if k.startswith('text_')}

        # load into internal model
        model_state = module.model.state_dict()
        # only take keys that exist in the current model
        filtered = {k: v for k, v in sd.items() if k in model_state}
        model_state.update(filtered)
        module.model.load_state_dict(model_state, strict=strict)

        if device is not None:
            module.to(device)

        return module

    def push_to_hub(self, repo_id, hf_token=None, files_to_include=None, commit_message="upload model checkpoint"):
        """Save the module's model weights and push to Hugging Face Hub.

        This saves the internal `self.model` state dict as 'pytorch_model.bin' and
        writes a lightweight 'config.json' (if possible). The tokenizer constants
        directory is not uploaded automatically — include it via files_to_include
        or push it separately.
        """
        if Repository is None or create_repo is None:
            raise RuntimeError('huggingface_hub is required for push_to_hub; please pip install huggingface_hub')

        tmpdir = tempfile.mkdtemp()

        # create repo if necessary
        try:
            create_repo(repo_id=repo_id, exist_ok=True, token=hf_token)
        except Exception:
            pass

        repo = Repository(tmpdir, clone_from=repo_id, use_auth_token=hf_token)

        # save model weights
        model_path = os.path.join(tmpdir, 'pytorch_model.bin')
        torch.save(self.model.state_dict(), model_path)

        # try to dump a minimal config.json if the config is simple
        try:
            def _ns_to_dict(ns):
                if isinstance(ns, dict):
                    return {k: _ns_to_dict(v) for k, v in ns.items()}
                if isinstance(ns, SimpleNamespace):
                    return {k: _ns_to_dict(v) for k, v in ns.__dict__.items()}
                if hasattr(ns, '__dict__'):
                    return {k: _ns_to_dict(v) for k, v in ns.__dict__.items()}
                return ns

            cfg_dict = _ns_to_dict(self.config)
            with open(os.path.join(tmpdir, 'config.json'), 'w', encoding='utf-8') as fh:
                json.dump(cfg_dict, fh, ensure_ascii=False, indent=2)
        except Exception:
            # non-fatal
            pass

        # include extra files
        if files_to_include:
            for fname, content in files_to_include.items():
                target = os.path.join(tmpdir, fname)
                mode = 'wb' if isinstance(content, (bytes, bytearray)) else 'w'
                with open(target, mode, encoding='utf-8' if mode == 'w' else None) as fh:
                    fh.write(content)

        repo.git_add(pattern='*')
        repo.git_commit(commit_message)
        repo.push_to_hub()
        return tmpdir

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
            try:
                outputs = self.model(encoded_text, inputs_asr=encoded_asr)
                predictions = outputs.argmax(dim=-1).cpu().tolist()
            except Exception as e:
                print(f"Error during prediction: {e}")
                return [""] * len(text)

        decoded_texts = self.tokenizer.decode_batch(predictions, text)

        return decoded_texts

    def predict_sliding_window(self, text, text_asr=[]):
        self.model.eval()
        original_text = text

        text = self.remove_diacritics(text).strip()

        _len = len(text)

        if _len == 0:
            return original_text

        r = len(text_asr) / _len if text_asr else 1

        # Sliding window
        if len(text) <= self.config.INFERENCE.MAX_LENGTH:
            output = self.predict_text(text, asr_text=text_asr)
        else:
            
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
                chunk_asr = text_asr[int(start*r):int(end*r)] if text_asr else []
                encoded_chunk, encoded_asr_chunk, _ = self.tokenizer.encode(
                    chunk,
                    chunk_asr, return_tensor=True
                )
                encoded_chunk = encoded_chunk.to(self.device)
                encoded_asr_chunk = encoded_asr_chunk.to(self.device) if chunk_asr else None

                with torch.no_grad():
                    outputs = self.model(encoded_chunk, inputs_asr=encoded_asr_chunk).squeeze(0)
                    predictions = outputs.argmax(dim=-1).cpu().tolist()  # remove <sos> and <eos> 
                
                # if end_idx > len(text) - buffer_size:
                #     decoded_chunk += self.tokenizer.decode(predictions[end_idx:], chunk[end_idx:])
                #     output += decoded_chunk
                #     break
                # else:
                
                decoded_chunk = self.tokenizer.decode(predictions[start_idx-start:end_idx-start], chunk[start_idx-start:end_idx-start])
                
                output += decoded_chunk  
                start_idx = end_idx
            output = [output]
        return output
    
    def remove_diacritics(self, text:str) -> str:
        return araby.strip_diacritics(text)

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
                    diacritized_line = self.predict_sliding_window(line[0])[0]
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
            dff=512, 
            num_blocks=2, 
            output_size=19,
            use_asr=True,
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
