import torch
import torch.nn as nn
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
        self.pos_emb = SinePositionEncoding(embed_dim, maxlen)
        # self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, inputs):
        x = self.token_emb(inputs)
        x = self.pos_emb(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.5):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.multi_head_attention = nn.MultiheadAttention(d_model, num_heads, dropout=rate, batch_first=True)
        self.dropout1 = nn.Dropout(rate)
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
        attention_output, _ = self.multi_head_attention(inputs, inputs, inputs, attn_mask=mask)
        attention_output = self.dropout1(attention_output)
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
            nn.Linear(dff, d_model),
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
            self.final_dense = nn.Linear(d_model, output_size)

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

        pretrained_dict = torch.load(pretrained_model_path, map_location='cpu', weights_only=True)
        model_dict = self.state_dict()

        if text_branch_only:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('text_') or k.startswith('final_dense')}
        
        # Update the current model's state dict
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print(f"Loaded pretrained weights from {pretrained_model_path}")

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
        
        pretrained_dict = torch.load(pretrained_model_path, map_location='cpu', weights_only=True)
        model_dict = self.state_dict()

        if text_branch_only:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('text_') or k.startswith('final_dense')}
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        
    def predict(self, inputs, inputs_asr=None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs, inputs_asr=inputs_asr)
            predictions = outputs.argmax(dim=-1)
        return predictions

AVAILABLE_MODELS = {
    'Transformer': TransformerModel,
    'LSTM': LSTMModel
}

if __name__ == "__main__":


    model = LSTMModel(
            maxlen=100, 
            vocab_size=1000,
            asr_vocab_size=1200,
            output_size=15,
            d_model=128,
            num_heads=4,
            dff=128,
            num_blocks=2,
            dropout_rate=0.2,
            with_conn=False
        )

    # model = TransformerModel(
    #         maxlen=100, 
    #         vocab_size=1000, 
    #         asr_vocab_size=1200, 
    #         d_model=128, 
    #         num_heads=4, 
    #         dff=512, 
    #         num_blocks=2, 
    #         output_size=15,
    #         use_asr=True,
    #     )

    input_text = torch.randint(0, 1000, (32, 80))  # Batch of 32 samples, each of length 100
    input_asr = torch.randint(0, 1200, (32, 98))
    output = model(inputs=input_text, inputs_asr=input_asr)
    print(output.shape)  # Should be (32, 100, 15)

    # visualize the computational graph
    # from torchviz import make_dot
    # make_dot(output, params=dict(model.named_parameters())).render("rnn_torchviz_asr", format="png")
