
from torch import nn
import torch
import math

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
        # self.pos_emb = SinePositionEncoding(embed_dim, maxlen)
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

        if not self.use_asr or inputs_asr is None:
            outputs = self.final_dense(x)
            return outputs
        
        assert inputs_asr is not None, "ASR inputs must be provided when use_asr is True"
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