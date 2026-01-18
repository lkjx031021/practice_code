import math
import struct
import inspect
import time

import LMConfig
from typing import Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(torch.nn.Module):
    # 初始化函数，接受参数：
    # dim: 归一化的维度大小
    # eps: 防止除零的非常小的数值
    def __init__(self, dim: int, eps: float):
        super().__init__()  # 调用父类的初始化方法
        self.eps = eps  # 将 eps 存储为类的属性
        # 初始化可学习的参数 weight，初始值为全1，形状为(dim,)
        # 这是每个维度的缩放系数
        self.weight = nn.Parameter(torch.ones(dim))  

    # 定义一个内部方法 _norm，用于对输入 x 进行归一化操作
    def _norm(self, x):
        # 使用平方的均值作为输入的标准差，并加上 eps 以防止除零
        # torch.rsqrt 是计算平方根的倒数，即 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # 定义前向传播的操作
    def forward(self, x):
        # 首先调用 _norm 方法对输入 x 进行归一化，并确保类型一致性
        # x.float() 将输入转换为浮点数进行精度较高的计算
        output = self._norm(x.float()).type_as(x)  
        # 将归一化后的输出乘以可学习的参数 weight，调整每个维度的缩放
        return output * self.weight

#定义频率计算
def precompute_pos_cis(dim: int, max_position: int, theta: float = 10000.0):
    #频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    #位置编码m
    m = torch.arange(max_position, device=freqs.device)

    #频率乘以位置编码、外积
    freqs = torch.outer(m, freqs).float()

    #
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return pos_cis

#将频率用于q、k矩阵
def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        print(pos_cis.shape)
        print(x.shape[1])
        print(x.shape[-1])
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # 定义函数 repeat_kv，接受两个参数：张量 x 和重复次数 n_rep
    # x 是一个形状为 (bs, slen, n_kv_heads, head_dim) 的张量，分别代表：
    # bs: 批次大小 (batch size)
    # slen: 序列长度 (sequence length)
    # n_kv_heads: KV 头的数量 (number of key-value heads)
    # head_dim: 每个头的维度大小 (dimension size of each head)
    # n_rep: 重复次数

    # 获取张量的形状 (bs: 批次大小, slen: 序列长度, n_kv_heads: KV 头的数量, head_dim: 每个头的维度)
    bs, slen, n_kv_heads, head_dim = x.shape

    # 如果 n_rep 为 1，表示不需要重复，直接返回原始张量
    if n_rep == 1:
        return x

    # 执行以下操作以重复 KV 头：
    # 1. 在第 4 维度 (即 None) 上扩展 x，使其形状为 (bs, slen, n_kv_heads, 1, head_dim)
    # 2. 使用 expand 函数将第 4 维度扩展为 n_rep，得到形状 (bs, slen, n_kv_heads, n_rep, head_dim)
    # 3. 最后通过 reshape 将形状重新调整为 (bs, slen, n_kv_heads * n_rep, head_dim)
    # 这会将每个 KV 头重复 n_rep 次
    return (
        x[:, :, :, None, :]                       # 扩展张量，在 n_kv_heads 后增加一个维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 扩展 n_rep 次
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 调整形状为新的维度
    )

class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()

        # 先确定n_kv_heads的值，如果设置了单独的n_kv_heads，就执行多头共享机制
        # 如果没设置kv_heads，就意味着全部的头都要执行kv缓存，此时n_kv_heads = n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # 检验，n_heads能否被n_kv_heads除尽
        assert args.n_heads % self.n_kv_heads == 0

        # 设置头数、kv缓存头数和重复次数
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        # 设置每个头上的特征维度
        self.head_dim = args.dim // args.n_heads

        # 设置权重层，当 x 的结构为 (seq_len, d_model)时
        # 常规的Q、K、V矩阵的结构应该与 X 一致，也是 (seq_len, d_model)
        # 因此常规的 w 应该是 (d_model,d_model)结构
        # 在多头注意力中，w 应该是 (d_model, d_model/n_heads)
        # 在具有kv缓存的情况下，我们是对所有头上的注意力并行计算
        # 因此Q的权重应该是(d_model, d_model)
        # K和V的权重应该是(d_model, d_model/n_heads * n_kv_heads)
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        # 输出层上的O的权重不受影响，是(d_model, d_model)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 设置kv缓存初始值
        self.k_cache, self.v_cache = None, None

        # 设置注意力和残差连接上的dropout层和dropout比例
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # flash attention
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        # 设置decoder专用前瞻掩码
        # 注意，前瞻掩码是用于QK.T矩阵的
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)

        # buffer用于保存神经网络中除了权重之外、需要被保存的静态数据们
        # 比如掩码矩阵、比如位置编码中的频率等等编码表
        # "mask"我们指定的buffer名称，我们可以通过self.mask来调出掩码矩阵
        self.register_buffer("mask", mask, persistent=False)

    # 设置旋转位置编码中的频率计算
    def _precompute_pos_cis(self, dim: int, max_position = 10000, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        m = torch.arange(max_position, device=freqs.device)
        freqs = torch.outer(m, freqs).float()
        pos_cis = torch.polar(torch.ones_like(freqs), freqs)
        return pos_cis
    
    def forward(self, x: torch.Tensor, kv_cache=False):

        # 作为注意力机制，被输入的x就是原始数据x
        # 结构为 (bs, seq_len, d_model)
        bsz, seqlen, _ = x.shape

        # 无论是否执行KV缓存，Q的求解是不变的
        xq = self.wq(x)

        # 如果是训练模式下，K和V照常求解
        if self.train():
            # 将x输入线性层、转换为初始的K和V
            # 但是只需要n_kv_heads个头的部分
            xk, xv = self.wk(x), self.wv(x)

        # 如果是推理模式，且kv_cache设置是打开的
        # 那要判断现在是否是初次预测
        if kv_cache and self.eval():
            # kv缓存是否还是None？已经存在了吗？
            if all(cache is not None for cache in (self.k_cache, self.v_cache)):
                # 如果不是None，说明不是初次预测了，此时需要的是缓存更新
                xk_new_token = self.wk(x[:,-1,:]).unsqueeze(1)
                xv_new_token = self.wv(x[:,-1,:]).unsqueeze(1)
                xk = torch.cat((self.k_cache, xk_new_token), dim=1)
                xv = torch.cat((self.v_cache, xv_new_token), dim=1)
            else:
                # 如果k和v缓存中有一个为None，说明是初次预测
                xk, xv = self.wk(x), self.wv(x)
            #生成xk和xv后，把结果保存到缓存中
            self.k_cache, self.v_cache = xk, xv

        # 为了更省内存，我们要将数据结构重新整理后适应位置编码的结构
        # 可以将该流程命名为“多头旋转位置编码”
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 在Q和K上执行旋转位置编码
        pos_cis = self._precompute_pos_cis(self.head_dim, seqlen)
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # 将k矩阵和v矩阵进行重复
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # 矩阵乘法计算注意力分数时，要将n_heads作为第二维度
        # 因为实际要进行乘法的应该时 (seqlen, head_dim) 这样的二维表
        # transpose交换维度，结构变为(bs, n_local_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 如果使用flash attention的话
        # 就调用nn.functional下面的点乘注意力计算方法
        if self.flash and seqlen != 1:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv
                                                                      , attn_mask=None #这里是padding掩码
                                                                      , dropout_p=self.dropout if self.training else 0.0
                                                                      , is_causal=True #这里是自动化的前瞻掩码
                                                                     )
        else:
            # 不使用flash attention，就自己计算
            # 这里的transpose是对最后两个维度的转置
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            # 在注意力分数上放上掩码
            # 如果有kv缓存的话，现在我们的kv矩阵可能会比掩码矩阵要大了
            # 获取缓存的长度
            cache_len = self.k_cache.shape[1] if self.k_cache is not None else 0
            total_len = cache_len + 1  # 当前总长度，等于历史缓存长度 + 当前序列长度

            # 检查是否需要扩展掩码矩阵
            if total_len > self.mask.shape[-1]:
                # 动态生成新的掩码，大小为 (seq_len + cache_len, seq_len + cache_len)
                new_mask = torch.full((1, 1, total_len, total_len), float("-inf")).to(x.device)
                new_mask = torch.triu(new_mask, diagonal=1)  # 生成前瞻掩码
                self.mask = new_mask  # 更新掩码矩阵
            
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
                
            # 对最后一个维度求解softmax
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # 最后再将结构转回来，并且将n_heads中的所有信息合并
        # contiguous() 用于确保张量在内存中的存储是连续的
        # 特别是在经过某些操作（如 transpose）后，这对后续的 view() 等操作至关重要，以避免错误
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 注意力机制的输出
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        """
        初始化 MoEGate 类，用于混合专家模型中的门控机制。

        参数：
        - config: LMConfig 对象，包含模型的配置信息，如专家数量、得分函数、辅助损失等。
        """
        super().__init__()
        self.config = config  # 保存配置信息
        self.top_k = config.num_experts_per_tok  # 每次选择的 top-k 个专家数量
        self.n_routed_experts = config.n_routed_experts  # 总的专家数量

        self.scoring_func = config.scoring_func  # 路由器使用的得分函数（如 softmax）
        self.alpha = config.aux_loss_alpha  # 辅助损失的系数
        self.seq_aux = config.seq_aux  # 是否启用基于序列的辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否对 top-k 权重进行归一化
        # 每个专家被给与的权重的维度
        self.gating_dim = config.dim  # 输入维度
        # 路由器的权重矩阵：用于计算每个专家的得分
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()  # 初始化权重参数

    def reset_parameters(self) -> None:
        """
        使用 Kaiming 初始化方法对权重矩阵进行初始化，确保模型在深层网络中有较好的梯度流动。
        """
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Kaiming 初始化

    def forward(self, hidden_states):
        """
        前向传播：根据输入的隐状态（hidden_states）计算专家的得分，选择 top-k 个专家，
        并在训练时计算辅助损失。

        参数：
        - hidden_states: Tensor，形状为 (batch_size, seq_len, hidden_dim) 的输入张量。

        返回：
        - topk_idx: 被选中的 top-k 个专家的索引。
        - topk_weight: 这些专家对应的权重。
        - aux_loss: 在训练模式下返回的辅助损失（否则为 None）。
        """
        # 获取 batch 大小、序列长度和隐藏维度
        bsz, seq_len, h = hidden_states.shape

        # 将输入重塑为二维张量 (batch_size * seq_len, hidden_dim)
        hidden_states = hidden_states.view(-1, h)

        # 使用线性层计算每个专家的得分，token-level (n_routed_experts, gating_dim) -> (batch_size * seq_len, n_routed_experts)
        logits = F.linear(hidden_states, self.weight, None)

        # 根据配置，使用 softmax 对得分进行归一化处理
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)  # 按照最后一维计算 softmax
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 获取 top-k 专家的权重和对应的索引
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 如果启用归一化，则对 top-k 权重进行归一化处理
        if self.top_k > 1 and self.norm_topk_prob:
            # 避免除以 0，添加小数项 1e-20
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 如果在训练模式下且辅助损失系数 alpha > 0，则计算辅助损失
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores  # 获取所有专家的得分用于辅助损失计算
            aux_topk = self.top_k  # 辅助损失中使用的 top-k 专家数量
            # 将 top-k 专家的索引重塑为 (batch_size, top_k * seq_len)
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            if self.seq_aux:
                # 如果启用了序列级别的辅助损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)

                # 初始化交叉熵损失的张量
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)

                # 按照每个序列来计算，计算每个序列所对应的所有专家权重
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
                ).div_(seq_len * aux_topk / self.n_routed_experts)

                # 计算序列级别的辅助损失
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 如果没有启用序列级别的辅助损失
                # 使用 one-hot 编码标记被选中的专家
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)  # 计算每个专家的平均使用率

                Pi = scores_for_aux.mean(0)  # 所有专家的平均分配概率
                fi = ce * self.n_routed_experts  # 专家使用频率
                aux_loss = (Pi * fi).sum() * self.alpha  # 计算辅助损失
        else:
            aux_loss = None  # 如果不在训练模式或 alpha=0，则不计算辅助损失

        # 返回 top-k 专家的索引、对应权重和辅助损失
        return topk_idx, topk_weight, aux_loss

class MOEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        """
        初始化 MOEFeedForward 类。
        
        参数：
        - config: 包含模型配置的 LMConfig 对象，包括专家数量、维度、隐藏层维度和 dropout 参数。
        """
        super().__init__()
        self.config = config
        
        # 创建多个专家网络的列表 (ModuleList)，每个专家是一个 FeedForward 层
        self.experts = nn.ModuleList([
            FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )
            for _ in range(config.n_routed_experts)  # n_routed_experts：总专家数量
        ])

        # 创建门控 (Gate) 对象，用于选择哪些专家参与计算
        self.gate = MoEGate(config)

        # 如果配置指定了共享专家，则代表允许MoE与FeedForward并联
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )

    def forward(self, x):
        """
        前向传播逻辑。

        参数：
        - x: 输入张量，形状为 (batch_size, seq_len, hidden_dim)。

        返回：
        - y: 输出张量，经过专家网络和共享专家（如果存在）的计算。
        """
        identity = x  # 保存原始输入（用于后续的残差连接）
        orig_shape = x.shape  # 保存原始输入的形状信息
        bsz, seq_len, _ = x.shape  # 获取批次大小、序列长度和隐藏层维度

        # 使用门控机制选择参与计算的专家
        topk_idx, topk_weight, aux_loss = self.gate(x)  # topk_idx: 选中的专家索引，topk_weight: 选中的专家权重

        # 将输入数据重塑为 (batch_size * seq_len, hidden_dim)
        x = x.view(-1, x.shape[-1])  
        flat_topk_idx = topk_idx.view(-1)  # 将专家索引展平成一维

        if self.training:
            # 训练模式下，重复输入数据以适应专家数量
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)

            # 创建用于存储专家输出的张量
            y = torch.empty_like(x, dtype=torch.float16)

            # 遍历每个专家，将符合条件的 token 输入到对应专家中
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])  # 仅处理属于该专家的 token

            # 计算每个 token 的加权输出
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)  # 恢复为原始输入的形状
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 如果有共享专家，将共享专家的输出与 y 相加（残差连接）
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理模式下的专家计算逻辑。

        参数：
        - x: 输入张量，形状为 (batch_size * seq_len, hidden_dim)。
        - flat_expert_indices: 展平后的专家索引，用于指示哪些 token 属于哪些专家。
        - flat_expert_weights: 展平后的专家权重，用于加权专家的输出。

        返回：
        - expert_cache: 经过专家计算后的输出张量。
        """
        # 创建一个与输入形状相同的张量，用于存储专家输出
        expert_cache = torch.zeros_like(x)

        # 对专家索引进行排序，以便批量处理属于同一专家的 token
        idxs = flat_expert_indices.argsort()

        # 计算每个专家需要处理的 token 数量，并累积求和以找到每个专家的范围
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # 将排序后的索引映射回 token 的原始位置
        token_idxs = idxs // self.config.num_experts_per_tok

        # 遍历每个专家，处理属于该专家的 token
        for i, end_idx in enumerate(tokens_per_expert):
            # 获取每个专家的 token 范围
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]

            # 如果该专家没有处理任何 token，则跳过
            if start_idx == end_idx:
                continue

            # 获取该专家对象
            expert = self.experts[i]

            # 获取属于该专家的 token 索引和对应的 token
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]

            # 通过专家计算 token 的输出
            expert_out = expert(expert_tokens)

            # 使用专家的权重对输出进行加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # 将加权后的输出累加到 expert_cache 中
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LMConfig):
        """
        TransformerBlock 是 Transformer 模型的基础构件，包含自注意力机制、前馈网络，并根据配置使用 Mixture of Experts (MoE) 或常规前馈网络。

        参数：
        - layer_id: 当前层的编号，用于标识层。
        - args: LMConfig 配置类，包含模型的超参数配置，如注意力头数、维度等。
        """
        super().__init__()
        self.n_heads = args.n_heads  # 注意力头的数量
        self.dim = args.dim  # 总的模型维度
        self.head_dim = args.dim // args.n_heads  # 每个注意力头的维度
        self.attention = Attention(args)  # 自注意力机制模块

        self.layer_id = layer_id  # 当前层的编号
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 注意力层前的 RMS 归一化
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 前馈网络层前的 RMS 归一化

        # 根据配置判断是否使用 Mixture of Experts (MoE) 作为前馈网络
        if args.use_moe:
            self.feed_forward = MOEFeedForward(args)  # 使用 Mixture of Experts (MoE) 前馈网络
        else:
            self.feed_forward = FeedForward(  # 使用常规的前馈网络
                dim=args.dim,  # 模型的总维度
                hidden_dim=args.hidden_dim,  # 前馈网络隐藏层的维度
                multiple_of=args.multiple_of,  # 隐藏层维度应为该数的倍数
                dropout=args.dropout,  # dropout 概率，用于正则化
            )

    def forward(self, x, pos_cis, kv_cache=False):
        """
        TransformerBlock 的前向传播函数。

        参数：
        - x: 输入张量。
        - pos_cis: 位置嵌入或旋转嵌入，用于加入位置信息。
        - kv_cache: 是否使用键值缓存（用于加速推理时）。

        返回：
        - out: Transformer Block 的输出张量。
        """
        # 输入经过注意力归一化后通过自注意力层，并叠加输入
        h = x + self.attention(self.attention_norm(x), pos_cis, kv_cache)
        # 注意力层输出经过前馈网络归一化后通过前馈网络，并叠加输出
        out = h + self.feed_forward(self.ffn_norm(h))
        return out  # 返回最终输出

class Transformer(PreTrainedModel):
    # 定义模型类 Transformer，继承自 PreTrainedModel,支持huggingface, deepspeed的预训练模型接口
    config_class = LMConfig  # 定义模型使用的配置类
    last_loss: Optional[torch.Tensor]  # 用于记录最后计算的损失值

    def __init__(self, params: LMConfig = None):
        """
        Transformer 是一个基于 Transformer 架构的语言模型，继承自 PreTrainedModel。

        参数：
        - params: 配置对象 LMConfig，包含模型的超参数配置。
        """
        super().__init__(params)
        if not params:
            params = LMConfig()  # 如果没有提供配置，则使用默认配置
        self.params = params  # 保存模型参数配置
        self.vocab_size = params.vocab_size  # 词汇表大小
        self.n_layers = params.n_layers  # Transformer 的层数

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)  # 词嵌入层
        self.dropout = nn.Dropout(params.dropout)  # Dropout 层用于正则化
        self.layers = torch.nn.ModuleList()  # TransformerBlock 的容器
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))  # 逐层添加 TransformerBlock
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)  # 最后的 RMS 正则化
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)  # 输出层，线性映射到词汇表大小
        self.tok_embeddings.weight = self.output.weight  # 共享词嵌入层和输出层的权重
        pos_cis = precompute_pos_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("pos_cis", pos_cis, persistent=False)  # 注册位置嵌入（或旋转嵌入），不参与训练

        self.apply(self._init_weights)  # 初始化模型权重

        # 特殊初始化部分参数权重
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        self.last_loss = None  # 初始化最后的损失为 None
        self.OUT = CausalLMOutputWithPast()  # 初始化输出类
        self._no_split_modules = [name for name, _ in self.named_modules()]  # 保存不拆分的模块名称

    def _init_weights(self, module):
        """
        初始化模块权重。
        - 对线性层使用正态分布初始化权重，均值为 0，标准差为 0.02。
        - 对词嵌入层也使用正态分布初始化权重，均值为 0，标准差为 0.02。
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # 如果存在偏置，将偏置初始化为 0
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None,
                kv_cache=False, **keyargs):
        """
        Transformer 的前向传播函数。

        参数：
        - tokens: 输入的 token 张量，表示输入的词序列。
        - targets: 目标张量，用于计算交叉熵损失。
        - kv_cache: 是否使用键值缓存（用于加速推理）。
        - keyargs: 其他可选参数，如 'input_ids' 和 'attention_mask'。

        返回：
        - 输出的 logits 和 loss（如果有目标）。
        """
        current_idx = 0  # 当前索引初始化为 0
        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']  # 从关键字参数中提取 'input_ids'
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']  # 从关键字参数中提取 'attention_mask'
        if 'current_idx' in keyargs:
            current_idx = int(keyargs['current_idx'])  # 更新当前索引

        _bsz, seqlen = tokens.shape  # 获取输入 tokens 的 batch 大小和序列长度
        h = self.tok_embeddings(tokens)  # 将输入 tokens 通过词嵌入层进行嵌入
        h = self.dropout(h)  # 通过 dropout 进行正则化
        pos_cis = self.pos_cis[current_idx:current_idx + seqlen]  # 获取当前位置的旋转嵌入

        # 逐层通过 TransformerBlock
        for idx, layer in enumerate(self.layers):
            h = layer(h, pos_cis, kv_cache)  # 调用每个 TransformerBlock 的前向传播

        h = self.norm(h)  # 最后的 RMS 正则化处理

        if targets is not None:
            logits = self.output(h)  # 通过线性输出层生成 logits
            # 计算交叉熵损失，忽略 index 为 0 的位置，reduction 为 'none'，即不自动求平均
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                             ignore_index=0, reduction='none')
        else:
            logits = self.output(h[:, [-1], :])  # 如果没有目标，只返回最后一个时间步的 logits
            self.last_loss = None  # 没有损失

        self.OUT.__setitem__('logits', logits)  # 设置输出的 logits
        self.OUT.__setitem__('last_loss', self.last_loss)  # 设置最后的 loss
        return self.OUT  # 返回输出对象

    @torch.inference_mode()
    def generate(self, idx, eos, max_new_tokens, temperature=0.7, top_k=8, stream=True, rp=1., kv_cache=True):
        """
        推理模式下的文本生成函数。

        参数：
        - idx: 输入的 tokens。
        - eos: 结束标志符号，当生成到 eos 时停止生成。
        - max_new_tokens: 最大生成的新 token 数量。
        - temperature: 控制生成的随机性，温度越高，生成越多样化。
        - top_k: 限制 top-k 采样，控制只选择概率最高的 k 个 token。
        - stream: 是否进行流式输出。
        - rp: 重复惩罚系数，控制重复 token 的惩罚。
        - kv_cache: 是否使用键值缓存来加速推理。

        返回：
        - 生成的 tokens（可能是流式返回）。
        """
        index = idx.shape[1]  # 获取输入 token 序列的长度
        init_inference = True  # 初始化推理标志
        while idx.shape[1] < max_new_tokens - 1:  # 当生成的 tokens 长度小于最大 tokens 数时继续生成
            if init_inference or not kv_cache:
                inference_res, init_inference = self(idx, kv_cache=kv_cache), False  # 第一次推理，或不使用缓存
            else:
                inference_res = self(idx[:, -1:], kv_cache=kv_cache, current_idx=idx.shape[1] - 1)  # 仅使用最后一个 token 推理

            logits = inference_res.logits  # 获取推理结果的 logits
            logits = logits[:, -1, :]  # 只选择最后一个 token 的 logits

            # 对生成的 token 进行重复惩罚
            for token in set(idx.tolist()[0]):
                logits[:, token] /= rp  # 对每个重复的 token 施加惩罚

            if temperature == 0.0:  # 如果温度为 0，使用贪心算法选择下一个 token
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature  # 根据温度调整 logits
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # 使用 top-k 采样
                    logits[logits < v[:, [-1]]] = -float('Inf')  # 排除 top-k 之外的 logits

                probs = F.softmax(logits, dim=-1)  # 计算概率分布
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)  # 根据概率进行采样

            if idx_next == eos:  # 如果生成了 eos token，停止生成
                break

            idx = torch.cat((idx, idx_next), dim=1)  # 将生成的 token 拼接到输入序列中
            if stream:  # 如果启用了流式输出
                yield idx[:, index:]  # 输出当前生成的 tokens

        if not stream:  # 如果未启用流式输出
            yield idx[:, index:]  # 返回生成的完整序列

    @torch.inference_mode()
    def eval_answer(self, idx):
        idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
        inference_res = self(idx_cond)
        logits = inference_res.logits
        logits = logits[:, -1, :]
        return logits