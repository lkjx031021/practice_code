import torch
import torch.nn.functional as F
import math
from torch import nn

torch.set_default_device('cuda')

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute the scaled dot-product attention.

    Args:
        Q: Query tensor of shape (..., seq_len_q, depth)
        K: Key tensor of shape (..., seq_len_k, depth)
        V: Value tensor of shape (..., seq_len_v, depth_v)
        mask: Optional mask tensor broadcastable to (..., seq_len_q, seq_len_k)

    Returns:
        output: Attention output tensor of shape (..., seq_len_q, depth_v)
        attention_weights: Attention weights tensor of shape (..., seq_len_q, seq_len_k)
    """
    embedding_dim = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(embedding_dim)
    print(K.transpose(-2, -1).shape)
    print(K.permute(-2, -1, 0).shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attention_weights, V)

    return output, attention_weights


class Attention(nn.Module):
    def __init__(self,embedding_size):
        super(Attention, self).__init__()
        self.embedding_size = embedding_size
        self.wq = nn.Linear(embedding_size, embedding_size)
        self.wk = nn.Linear(embedding_size, embedding_size)
        self.wv = nn.Linear(embedding_size, embedding_size)

    def forward(self, q, k, v, mask=None):
        Q = self.wq(q)
        K = self.wk(k)
        V = self.wv(v)

        outputs, socres = scaled_dot_product_attention(Q, K, V, mask)

        return output, socres

class CrossAttention(nn.Module):
    def __init__(self, embedding_size):
        super(CrossAttention, self).__init__()
        self.embedding_size = embedding_size
        self.wq = nn.Linear(embedding_size, embedding_size)
        self.wk = nn.Linear(embedding_size, embedding_size)
        self.wv = nn.Linear(embedding_size, embedding_size)

        self.attention = Attention(embedding_size)

    def forward(self, q, kv, mask=None):
        Q = self.wq(q)
        K = self.wk(kv)
        V = self.wv(kv)
        outputs, scores = self.attention(Q, K, V, mask)

        return outputs, scores

 class MultiHeadAttention(nn.Module):
    def __init__(self,embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.mum_heads = num_heads

        self.w_q = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_heads)])
        self.w_k = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_heads)])
        self.w_v = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_heads)])

        self.fc_out = nn.Linear(self.num_heads * embed_size, embed_size)


if __name__ == "__main__":
    inputs = torch.randn(2, 4, 512)  # (batch_size, seq_len, embedding_dim)
    wQ = torch.randn(512, 512)  # (batch_size, seq_len_q, depth)
    wK = torch.randn(512, 512)  # (batch_size, seq_len_k, depth)
    wV = torch.randn(512, 512)  # (batch_size, seq_len_k, depth)

    Q = torch.matmul(inputs, wQ)
    K = torch.matmul(inputs, wK)
    V = torch.matmul(inputs, wV)

    output, weights = scaled_dot_product_attention(Q, K, V)
    print(output)

