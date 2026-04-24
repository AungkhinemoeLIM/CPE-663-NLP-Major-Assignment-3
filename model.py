import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    def forward(self, tokens):
        return self.embedding(tokens)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=20):
        super().__init__()
        self.dropout = nn.Dropout(0.1)

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Fill with a very small number for masked positions
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(dropout_rate)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply mask to attention scores
        if mask is not None:
            # mask shape: (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(1) # For broadcasting across heads and query positions

        attn_output, _ = self.attention(q, k, v, mask=mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_linear(attn_output)
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)
        self.feed_forward = PositionWiseFeedForward(embed_dim, ff_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # Self-attention part
        attn_output = self.self_attn(self.norm1(x), mask=mask)
        x = x + self.dropout(attn_output) # Residual connection

        # Feed-forward part
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output) # Residual connection
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout_rate, use_positional_encoding=True):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim) if use_positional_encoding else None
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim) # Final layer norm
        self.classification_head = nn.Linear(embed_dim, 1)

    def forward(self, tokens, mask=None):
        x = self.token_embedding(tokens)
        if self.positional_encoding:
            x = self.positional_encoding(x)

        for block in self.encoder_blocks:
            x = block(x, mask=mask)

        # Mean pooling over non-padding tokens
        # mask shape: (batch_size, seq_len)
        if mask is not None:
            # Expand mask to match embedding dimension
            expanded_mask = mask.unsqueeze(-1).float()
            # Apply mask to x, setting padded positions to 0
            x_masked = x * expanded_mask
            # Sum over sequence length and divide by count of non-padding tokens
            # Ensure sum is not zero for sequences that are entirely padded (though our data generation prevents this)
            sum_embeddings = x_masked.sum(dim=1)
            num_non_padding_tokens = expanded_mask.sum(dim=1)
            # Avoid division by zero for empty sequences, though again, our data generation prevents this
            pooled_output = sum_embeddings / (num_non_padding_tokens.clamp(min=1e-9))
        else:
            pooled_output = x.mean(dim=1) # Simple mean pooling if no mask

        logits = self.classification_head(pooled_output)
        return logits.squeeze(-1) # Squeeze to get (batch_size,) for binary classification
