import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.5):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.multi_head_attention = nn.MultiheadAttention(d_model, num_heads, dropout=rate)
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

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, inputs):
        seq_len = inputs.size(1)
        positions = torch.arange(0, seq_len, device=inputs.device).unsqueeze(0)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings

class TransformerModel(nn.Module):
    def __init__(self, maxlen, vocab_size, d_model, num_heads, dff, num_blocks, dropout_rate=0.5):
        super(TransformerModel, self).__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        for block in self.transformer_blocks:
            x = block(x)
        outputs = self.output_layer(x)
        return F.softmax(outputs, dim=-1)

class LSTMModel(nn.Module):
    def __init__(self, maxlen, vocab_size, asr_vocab_size, output_size, d_model, num_heads, dff, num_blocks, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.asr_vocab_size = asr_vocab_size
        self.output_size = output_size
        self.d_model = d_model

        # Text branch
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.text_lstm1 = nn.LSTM(d_model, dff, bidirectional=True, batch_first=True)
        self.text_dropout1 = nn.Dropout(dropout_rate)
        self.text_lstm2 = nn.LSTM(dff*2, dff, bidirectional=True, batch_first=True)
        self.text_dropout2 = nn.Dropout(dropout_rate)
        self.text_dense1 = nn.Linear(dff*2, dff)
        self.text_dense2 = nn.Linear(dff, dff)
        self.text_output = nn.Linear(dff, d_model)

        # ASR branch
        self.asr_embedding = nn.Embedding(asr_vocab_size, d_model)
        self.asr_lstm1 = nn.LSTM(d_model, dff, bidirectional=True, batch_first=True)
        self.asr_dropout1 = nn.Dropout(dropout_rate)
        self.asr_lstm2 = nn.LSTM(dff*2, dff, bidirectional=True, batch_first=True)
        self.asr_dropout2 = nn.Dropout(dropout_rate)
        self.asr_dense1 = nn.Linear(dff*2, dff)
        self.asr_dense2 = nn.Linear(dff, dff)
        self.asr_output = nn.Linear(dff, d_model)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
        self.final_dense = nn.Linear(d_model, output_size)

    def forward(self, inputs, input_asr, with_conn=True):
        # Text branch
        text_emb = self.text_embedding(inputs)
        text_out, _ = self.text_lstm1(text_emb)
        text_out = self.text_dropout1(text_out)
        text_out, _ = self.text_lstm2(text_out)
        text_out = self.text_dropout2(text_out)
        text_out = F.relu(self.text_dense1(text_out))
        text_out = F.relu(self.text_dense2(text_out))
        text_out = self.text_output(text_out)

        # ASR branch
        asr_emb = self.asr_embedding(input_asr)
        asr_out, _ = self.asr_lstm1(asr_emb)
        asr_out = self.asr_dropout1(asr_out)
        asr_out, _ = self.asr_lstm2(asr_out)
        asr_out = self.asr_dropout2(asr_out)
        asr_out = F.relu(self.asr_dense1(asr_out))
        asr_out = F.relu(self.asr_dense2(asr_out))
        asr_out = self.asr_output(asr_out)

        breakpoint()
        # Cross-attention
        cross_out, _ = self.cross_attention(text_out.transpose(0, 1), asr_out.transpose(0, 1), asr_out.transpose(0, 1))
        cross_out = cross_out.transpose(0, 1)

        # Combine
        if with_conn:
            combined = torch.cat([text_out, cross_out], dim=-1)
        else:
            combined = cross_out

        outputs = self.final_dense(combined)
        return F.softmax(outputs, dim=-1)

class ModifiedTransformerModel(nn.Module):
    
    def __init__(self, maxlen, vocab_size, asr_vocab_size, d_model, num_heads, dff, num_blocks, output_size, with_conn=False, dropout_rate=0.5):
        super(ModifiedTransformerModel, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.asr_vocab_size = asr_vocab_size
        self.output_size = output_size
        self.with_conn = with_conn
        self.d_model = d_model
        self.num_heads = num_heads

        # Text branch
        self.text_embedding = TokenAndPositionEmbedding(maxlen, vocab_size, d_model)
        self.text_transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(2)
        ])
        self.text_dense = nn.Linear(d_model, d_model)

        # ASR branch
        self.asr_embedding = TokenAndPositionEmbedding(maxlen, asr_vocab_size, d_model)
        self.asr_transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_blocks)
        ])
        self.asr_dense = nn.Linear(d_model, d_model)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
        if with_conn:
            self.final_dense = nn.Linear(d_model * 2, output_size)
        else:
            self.final_dense = nn.Linear(d_model, output_size)

    def forward(self, inputs, input_asr):
        # Text branch
        x = self.text_embedding(inputs)
        for block in self.text_transformer_blocks:
            x = block(x)
        x = self.text_dense(x)

        # ASR branch
        asr_emb = self.asr_embedding(input_asr)
        for block in self.asr_transformer_blocks:
            asr_emb = block(asr_emb)
        asr_attention = self.asr_dense(asr_emb)

        # Cross-attention
        cross_out, _ = self.cross_attention(x.transpose(0, 1), asr_attention.transpose(0, 1), asr_attention.transpose(0, 1))
        cross_out = cross_out.transpose(0, 1)

        # Combine
        if self.with_conn:
            combined = torch.cat([x, cross_out], dim=-1)
        else:
            combined = cross_out

        outputs = self.final_dense(combined)
        return F.softmax(outputs, dim=-1)

    def load_pretrained(self, pretrained_model_path):
        pass

if __name__ == "__main__":

    # model = ModifiedTransformerModel(
    #         maxlen=100, 
    #         vocab_size=1000, 
    #         asr_vocab_size=1200, 
    #         d_model=128, 
    #         num_heads=4, 
    #         dff=512, 
    #         num_blocks=2, 
    #         output_size=15
    #     )

    model = LSTMModel(
            maxlen=100, 
            vocab_size=1000, 
            asr_vocab_size=1200, 
            output_size=15,
            d_model=128, 
            num_heads=4, 
            dff=512, 
            num_blocks=2, 
            dropout_rate=0.5
        )
    input_text = torch.randint(0, 1000, (32, 80))  # Batch of 32 samples, each of length 100
    input_asr = torch.randint(0, 1200, (32, 98))
    output = model(input_text, input_asr, with_conn=False)
    print(output.shape)  # Should be (32, 100, 15)

    # visualize the computational graph
    # from torchviz import make_dot
    # make_dot(output, params=dict(model.named_parameters())).render("rnn_torchviz", format="png")
