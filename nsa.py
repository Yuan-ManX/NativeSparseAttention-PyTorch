import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from einops import einsum, repeat
from einops.layers.torch import Rearrange

from local_attention import LocalAttention

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass


def exists(v):
    """
    检查变量是否存在（即不为None）。

    参数:
        v: 任意变量

    返回:
        bool: 如果v不为None，返回True；否则返回False。
    """
    return v is not None


def default(v, d):
    """
    如果变量存在，则返回变量本身；否则，返回默认值。

    参数:
        v: 任意变量
        d: 默认值

    返回:
        变量v或默认值d
    """
    return v if exists(v) else d


def round_down_mult(n, mult):
    """
    将数字n向下取整到最接近mult的倍数。

    参数:
        n (int): 要取整的数字
        mult (int): 倍数

    返回:
        int: 向下取整后的结果
    """
    return n // mult * mult


class Attention(Module):
    """
    Attention 类，实现了一个基于滑动窗口和压缩策略的注意力机制。

    该类结合了局部注意力（sliding window）和压缩注意力（compress block）两种策略，
    通过学习权重将它们结合起来，以实现高效的注意力计算。

    参数:
        dim (int): 输入特征的维度大小。
        dim_head (int): 每个注意力头的维度。
        heads (int): 注意力头的数量。
        sliding_window_size (int): 滑动窗口的大小，用于局部注意力。
        compress_block_size (int): 压缩块的尺寸，用于压缩注意力。
        norm (bool, optional): 是否使用归一化层，默认为True。
    """
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        sliding_window_size,
        compress_block_size,
        norm = True,
    ):
        super().__init__()

        # 计算注意力头的内部维度
        dim_inner = dim_head * heads

        # 定义归一化层，如果 norm 为 True，则使用 RMS 归一化；否则，使用恒等映射
        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()

        # 定义线性层，将输入特征映射到查询（Q）、键（K）和值（V）
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        # 定义滑动窗口注意力机制
        self.sliding_window = LocalAttention(
            dim = dim_head,   # 每个注意力头的维度
            window_size = sliding_window_size,  # 滑动窗口的大小
            causal = True,   # 是否使用因果掩码
            exact_window_size = True  # 是否严格使用窗口大小
        )

        # 定义压缩块的大小
        self.compress_block_size = compress_block_size

        # 定义压缩注意力中键（K）和值（V）的位置编码参数
        self.k_intrablock_positions = nn.Parameter(torch.zeros(heads, compress_block_size, dim_head))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(heads, compress_block_size, dim_head))

        # 定义键（K）的压缩模块
        self.k_compress = nn.Sequential(
            Rearrange('b h n d -> b (h d) n'),  # 重排张量形状
            nn.Conv1d(dim_head * heads, dim_head * heads, compress_block_size, stride = compress_block_size, groups = heads),  # 使用1D卷积进行压缩
            Rearrange('b (h d) nc -> b h nc d', h = heads)  # 重新调整张量形状
        )

        # 定义值（V）的压缩模块
        self.v_compress = nn.Sequential(
            Rearrange('b h n d -> b (h d) n'),  # 重排张量形状
            nn.Conv1d(dim_head * heads, dim_head * heads, compress_block_size, stride = compress_block_size, groups = heads),  # 使用1D卷积进行压缩
            Rearrange('b (h d) nc -> b h nc d', h = heads)  # 重新调整张量形状
        )

        # 定义策略组合模块，通过学习权重将三种稀疏分支结合起来，并使用Sigmoid激活
        self.to_strategy_combine = nn.Sequential(
            nn.Linear(dim, 3),  # 线性层，将维度映射到3
            nn.Sigmoid()  # Sigmoid激活
        )

        # 定义重排操作，用于分割和合并注意力头
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # 定义合并注意力头的线性层
        self.combine_heads = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        inp
    ):
        """
        前向传播方法，应用注意力机制处理输入特征。

        参数:
            inp (Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)

        返回:
            Tensor: 输出张量，形状为 (batch_size, sequence_length, dim)
        """
        # 获取序列长度
        seq_len = inp.shape[-2]
        # 计算可以被压缩块整除的序列长度
        block_divisible_seq_len = round_down_mult(seq_len, self.compress_block_size)

        # 应用归一化
        inp = self.norm(inp)

        # 分割输入为查询（Q）、键（K）和值（V）
        q, k, v = self.to_qkv(inp).chunk(3, dim = -1)

        # 分割注意力头
        q, k, v = map(self.split_heads, (q, k, v))

        # 压缩键（K）和值（V）的位置编码
        k_pos = repeat(self.k_intrablock_positions, 'h n d -> h (r n) d', r = block_divisible_seq_len // self.compress_block_size)
        v_pos = repeat(self.v_intrablock_positions, 'h n d -> h (r n) d', r = block_divisible_seq_len // self.compress_block_size)

        # 应用键（K）和值（V）的压缩
        ck = self.k_compress(k[..., :block_divisible_seq_len, :] + k_pos)
        cv = self.v_compress(v[..., :block_divisible_seq_len, :] + v_pos)

        # 应用滑动窗口注意力
        local_attn_out = self.sliding_window(q, k, v)

        # 结合不同的注意力策略
        strategy_weighted_combine = self.to_strategy_combine(inp)

        # 将压缩后的输出和滑动窗口的输出堆叠起来，并根据策略权重进行加权组合
        out = (strategy_weighted_combine * stack((compress_out, local_attn_out), dim = -1)).sum(dim = -1)

        # 合并注意力头
        out = self.merge_heads(out)

        # 合并输出
        return self.combine_heads(out)
