import torch
import torch.nn.functional as nnfunc
from torchmetrics.classification import MulticlassAccuracy
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch + 1, 1)
        scale = self.d_model**-0.5
        warmup_factor = min(step**-0.5, step * (self.warmup_steps**-1.5))
        absolute_lr = scale * warmup_factor
        return [absolute_lr for _ in self.optimizer.param_groups]


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
            logits: 模型输出的原始分数，形状为 [B, T, V]
            target: 标签索引，形状为 [B, T]
        返回：
            单个标量损失值
        """
        # 忽略 padding 标签的部分
        logits = logits.view(-1, logits.size(2))
        target = target.view(-1)
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


class CategoricalAccuracy(MulticlassAccuracy):
    """
    用于多分类（logits）输入的准确率计算。
    输入形状: [B, T, V]，target: [B, T]
    """
    def __init__(self, num_classes: int):
        super().__init__(num_classes=num_classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, T, V] → [B*T, V]
        # target: [B, T] → [B*T]
        logits = logits.view(-1,logits.size(2))
        target = target.view(-1)

        return super().forward(logits, target)
