import glob
import os
import random
import numpy as np
import torch
from typing import Union, List

import config


def mask_relative_position(tensor: torch.Tensor) -> torch.Tensor:
    """
    对于相对位置权重矩阵，后面的位置不能看到前面的位置信息。
    例如，位置0只能看到位置0，位置1只能看到位置0和1，位置2只能看到位置0、1和2。
    """
    j = torch.arange(tensor.size(2))

    # 生成行掩码：j >= (n - i - 1)
    mask = j >= (tensor.size(2) - torch.arange(tensor.size(2)).unsqueeze(1) - 1)

    return mask


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model_parameters(model, save_dir=config.CHECK_DIR, tag=config.REPO_NAME):
    os.makedirs(save_dir, exist_ok=True)
    clear_folder(save_dir)
    for name, param in model.named_parameters():
        if param.requires_grad:
            file_name = f"{tag}_param_{name.replace('.', '_')}.pt"
            torch.save(param.data, os.path.join(save_dir, file_name))
            print(f"[Saved] {file_name}")


def clear_folder(folder_path):
    # 确保路径存在
    pt_files = glob.glob(os.path.join(folder_path, "*.pt"))
    for file_path in pt_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
