import math
import torch
import torch.nn as nn
import torch.nn.functional as nnfunc
# from torchviz import make_dot

# 本地模块
import Config
import Utils


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

    def __init__(self, d_model: int, seq_len: int = Config.SEQ_LEN) -> None:
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
        self.e_r = nn.Parameter(torch.randn(Config.MAX_SEQ_LEN, self.head_dim), requires_grad=False)
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
        e_sub = self.e_r[max(0, Config.MAX_SEQ_LEN - q_len) :, :]
        qe_relative = torch.einsum("bhqd,kd->bhqk", q, e_sub)
        position_mask = Utils.mask_relative_position(qe_relative)
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
        self.self_attn = MultiHeadAttentionWithRelativePosition(d_model, num_heads, dropout)
        # ffn的目的？
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        # layer Norm 的目的？
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
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


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=6,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
    ):
        super().__init__()
        # self.encoder_embed = nn.Sequential(Embeddings(d_model, src_vocab_size), PositionalEncoding(d_model))
        self.encoder_embed = Embeddings(d_model, src_vocab_size)
        # self.decoder_embed = nn.Sequential(Embeddings(d_model, tgt_vocab_size), PositionalEncoding(d_model))
        self.decoder_embed = Embeddings(d_model, tgt_vocab_size)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器
        enc_output = self.encoder_embed(src)
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        # 解码器
        dec_output = self.decoder_embed(tgt)
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.final_linear(dec_output)
