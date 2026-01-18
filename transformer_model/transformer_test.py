import math

import torch
from torch import nn
import torch.nn.functional as F

src_vocab_size = 5000
tgt_vocab_size = 5000

d_model = 512
N = 6
hid_dim = 2048
dropout = 0.1

def create_padding_mask(seq, pad_token=0):
    """
    Creates a padding mask for the given sequence.
    seq:(batch_size, seq_len, d_model)

    Args:
        seq (torch.Tensor): Input sequence of shape (batch_size, seq_len).
        pad_token (int, optional): The token used for padding. Default is 0.
    """
    padding_mask = seq.eq(pad_token).all(dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.pe = torch.zeros(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # position = torch.arange(0, max_len).reshape(-1, 1)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe[:, ::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = self.pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        # 问题：模型输入的x不是已经通过padding填充了吗，为什么还要取x.size(1)
        # 答：因为pe中的序列长度是最大序列长度，训练过程中，输入的x的序列长度是设置的默认序列长度，非max_len，max_len是预分配一个足够大的位置编码缓,以支持不超过该长度的序列，训练和推理时的序列长度都应小于max_len
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_model = Q.size(-1)

    score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)

    if mask is not None:
        score = score.masked_fill(mask == 0, float('-inf'))

    attention_scores = F.softmax(score, dim=-1)
    
    output = torch.matmul(score, V)
    return output, attention_scores

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.wq(q)
        K = self.wk(k)
        V = self.wv(v)

        # 交叉注意力层，Q来自decoder，K和V来自encoder，所以q,k,v的seq_len可能不一样
        seq_len_q = Q.size(1)
        seq_len_k = K.size(1)

        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        
        scaled_attention, _ = scaled_dot_product_attention(Q, K, V, mask)
        attention_out = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc_out(attention_out)
        return output

            



if __name__ == '__main__':
    a = torch.arange(0, 8, 2).unsqueeze(1)
    b = torch.tensor([1, 2, 3])
    print(a * b)
    batch_size = 32
    seq_len_src = 10
    seq_len_tgt = 15

    src = torch.randint(0, 100, (batch_size, seq_len_src))
    tgt = torch.randint(0, 100, (batch_size, seq_len_tgt))

    # 创造一个示例数据
    batch_size = 4
    seq_len = 10
    embedding_dim = 8
    seq = torch.randint(0, 5, (batch_size, seq_len, embedding_dim))  # 随机生成一些数据

    #填充部分
    pad_token = 0
    seq[0, 7:, :] = pad_token  # 设置填充值
    seq[1, 9:, :] = pad_token  # 设置填充值
    seq[3, 5:, :] = pad_token  # 设置填充值
    print(seq.eq(pad_token).all(dim=-1).unsqueeze(1).unsqueeze(2).shape)  # 输出掩码形状
