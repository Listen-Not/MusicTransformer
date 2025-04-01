import torch


def mask_relative_position(tensor: torch.Tensor) -> torch.Tensor:
    """
    对于相对位置权重矩阵，后面的位置不能看到前面的位置信息。
    例如，位置0只能看到位置0，位置1只能看到位置0和1，位置2只能看到位置0、1和2。
    """
    j = torch.arange(tensor.size(2))

    # 生成行掩码：j >= (n - i - 1)
    mask = j >= (tensor.size(2) - torch.arange(tensor.size(2)).unsqueeze(1) - 1)

    return mask
