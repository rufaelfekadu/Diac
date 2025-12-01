
from torch import nn
import torch

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