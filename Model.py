import math
import torch
import torch.nn as nn
import torch.nn.functional as nnfunc
# from torchviz import make_dot

# 本地模块
import config
import utils


class Embeddings(nn.Module):
    """
    生成所有输入词汇的词向量，并通过除以词向量维数的平方根进行缩放。

    参数:
        vocab_size (int): 词汇表的总大小。
        d_model (int): 词向量的维度。

    返回:
        torch.Tensor: 输入序列对应的词向量张量，形状为 (batch_size, seq_len, d_model)。
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    绝对位置编码，每个词向量生成对应的位置编码，绝对位置向量不参与学习。

    参数:
        seq_len (int): 一句话的词向量个数。
        d_model (int): 词向量的维度。

    返回:
        torch.Tensor: 加上位置编码后的张量，形状为 (batch_size, seq_len, d_model)。
    """

    def __init__(self, d_model: int, seq_len: int = config.SEQ_LEN) -> None:
        super().__init__()
        self.pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", self.pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class MultiHeadAttentionWithRelativePosition(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.e_r = nn.Parameter(torch.randn(config.MAX_SEQ_LEN, self.head_dim), requires_grad=False)
        self.dropout = nn.Dropout(dropout)
        # 输入检查
        assert self.head_dim * num_heads == d_model, f"词向量维度 {d_model} 应该被头数 {num_heads} 整除"

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        batch_size = q.size(0)

        # 计算注意力权重矩阵
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算相对位置编码
        q_len = q.size(2)
        e_sub = self.e_r[max(0, config.MAX_SEQ_LEN - q_len) :, :]
        qe_relative = torch.einsum("bhqd,kd->bhqk", q, e_sub)
        position_mask = utils.mask_relative_position(qe_relative)
        qe_relative = qe_relative * position_mask.unsqueeze(0).unsqueeze(0)
        S_rel = self.skew(qe_relative)

        # 按batch和sequence分批次计算
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores + S_rel, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attn_output), S_rel, qe_relative

    def skew(self, qe_relative: torch.Tensor) -> torch.Tensor:
        """
        倾斜相对位置注意力矩阵。
        """
        S_rel = nnfunc.pad(qe_relative, [1, 0, 0, 0, 0, 0, 0, 0])
        S_rel = S_rel.reshape(S_rel.size(0), S_rel.size(1), S_rel.size(3), S_rel.size(2))
        S_rel = S_rel[:, :, 1:, :]
        # TODO:q_len与k_len不一致时，可能会报错
        return S_rel


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 多头注意力模块，带相对位置编码
        self.self_attn = MultiHeadAttentionWithRelativePosition(d_model, num_heads, dropout)

        # 前馈网络：两层线性+GELU激活
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

        # 层归一化与 Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + LayerNorm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x


class DecoderLayer(nn.Module):
    "暂时没用"

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithRelativePosition(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttentionWithRelativePosition(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 交叉注意力
        cross_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, input_vocab_size, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # 词嵌入模块
        self.embedding = Embeddings(d_model=d_model, vocab_size=input_vocab_size)

        # 位置编码模块
        self.pos_encoding = PositionalEncoding(d_model=d_model, seq_len=seq_len)

        # 多层编码器层
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads=d_model // 64, d_ff=d_model // 2, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 词嵌入 + 位置编码 + Dropout
        x = self.embedding(x.long())  # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)  # 添加位置编码
        x = self.dropout(x)

        # 通过多个编码层
        for layer in self.encoder_layers:
            x = layer(x, mask)

        return x  # 输出形状: (batch_size, seq_len, d_model)


class MusicTransformer(torch.nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=388 + 2, num_layer=6, max_seq=2048, dropout=0.1):
        super().__init__()
        self.max_seq = max_seq
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.Decoder = Encoder(
            num_layers=self.num_layer,
            d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size,
            max_len=max_seq,
            rate=dropout,
        )
        self.fc = torch.nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x, length=None):
        if self.training:
            mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x, config.PAD_TOKEN)
            decoder = self.Decoder(x, mask)
            return self.fc(decoder).contiguous()
