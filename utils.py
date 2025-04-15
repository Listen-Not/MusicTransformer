import os
import random
import numpy as np
import torch
import torch.nn.functional as nnfunc
from torch.nn.modules.loss import _Loss
import matplotlib.pyplot as plt
from typing import Union, List


def mask_relative_position(tensor: torch.Tensor) -> torch.Tensor:
    """
    对于相对位置权重矩阵，后面的位置不能看到前面的位置信息。
    例如，位置0只能看到位置0，位置1只能看到位置0和1，位置2只能看到位置0、1和2。
    """
    j = torch.arange(tensor.size(2))

    # 生成行掩码：j >= (n - i - 1)
    mask = j >= (tensor.size(2) - torch.arange(tensor.size(2)).unsqueeze(1) - 1)

    return mask


def plot_and_save_losses(losses, save_path="loss/loss_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 创建目录（如果需要）
    plt.savefig(save_path)
    plt.close()


def get_masked_with_pad_tensor(
    size: int, src: torch.Tensor, trg: torch.Tensor, pad_token: Union[int, List[int]]
) -> tuple:
    """
    构造带 pad_token 的 mask 张量，支持多个 pad_token 值。

    :param size: int, decoder输入的长度
    :param src: (batch_size, seq_len) 的源张量
    :param trg: (batch_size, seq_len) 的目标张量
    :param pad_token: int 或 List[int]，表示 pad 的 token ID(s)
    :return: src_mask, trg_pad_mask, look_ahead_mask
    """
    if isinstance(pad_token, int):
        pad_values = torch.tensor([pad_token], device=src.device)
    else:
        pad_values = torch.tensor(pad_token, device=src.device)

    # src_pad_mask: (batch_size, 1, 1, seq_len)
    src_mask = torch.isin(src, pad_values).unsqueeze(1).unsqueeze(2)

    # trg_pad_mask: (batch_size, 1, 1, seq_len)
    trg_mask = torch.isin(trg, pad_values).unsqueeze(1).unsqueeze(2) if trg is not None else None

    look_ahead_mask = None
    if trg is not None:
        # look-ahead mask: (1, 1, size, size)
        seq_mask = torch.triu(torch.ones((size, size), dtype=torch.bool, device=trg.device), diagonal=1)
        look_ahead_mask = seq_mask.unsqueeze(0).unsqueeze(0)

        # dec_pad_mask: (batch_size, 1, 1, seq_len)
        dec_pad_mask = torch.isin(trg, pad_values).unsqueeze(1).unsqueeze(2)
        look_ahead_mask = look_ahead_mask | dec_pad_mask[:, :, :, :size]

    return src_mask, trg_mask, look_ahead_mask


def get_lr_lambda(d_model, warmup_steps):
    def lr_lambda(step):  # 这个 step 会在 scheduler.step() 时自动传入
        step = max(step, 1)  # 避免除以 0
        return (d_model**-0.5) * min(step**-0.5, step * (warmup_steps**-1.5))

    return lr_lambda


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SmoothCrossEntropyLoss(_Loss):
    """
    带标签平滑的交叉熵损失（来自 https://arxiv.org/abs/1512.00567）
    支持通过 ignore_index 忽略部分标签（如 PAD）
    """

    __constants__ = ["label_smoothing", "vocab_size", "ignore_index", "reduction"]

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, reduction="mean"):
        assert 0.0 <= label_smoothing <= 1.0
        super().__init__(reduction=reduction)

        self.label_smoothing = label_smoothing  # 标签平滑系数
        self.vocab_size = vocab_size  # 词表大小（类别数）
        self.ignore_index = ignore_index  # 忽略计算的标签编号（通常是 PAD）

    def forward(self, logits, target):
        """
        参数：
            logits: 模型输出的原始分数，形状为 [B * T, V]
            target: 标签索引，形状为 [B * T]
        返回：
            单个标量损失值
        """
        # 忽略 padding 标签的部分
        mask = target != self.ignore_index
        logits = logits[mask]
        target = target[mask]

        # 计算 log_softmax，得到对数概率分布
        log_probs = nnfunc.log_softmax(logits, dim=-1)

        if self.label_smoothing > 0:
            # 构建平滑后的目标分布
            n_classes = self.vocab_size
            true_dist = torch.zeros_like(log_probs).scatter(1, target.unsqueeze(1), 1)
            smooth_dist = (1.0 - self.label_smoothing) * true_dist + self.label_smoothing / n_classes

            # KL散度等价的计算形式（只计算 loss）
            loss = -torch.sum(smooth_dist * log_probs, dim=-1)
        else:
            # 如果没有平滑，就直接用 NLLLoss
            loss = nnfunc.nll_loss(log_probs, target, reduction="none")

        # 根据 reduction 返回平均或总和
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # 不进行归约，返回每个样本的 loss
