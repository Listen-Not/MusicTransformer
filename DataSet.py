import torch
from torch.utils.data import Dataset

class MIDIDataset(Dataset):
    """改进版数据集类，支持动态填充"""
    def __init__(self, sequences, seq_length=512, pad_token=383):
        self.seq_length = seq_length
        self.pad_token = pad_token
        
        # 将序列分割为固定长度片段（改进：动态填充）
        self.chunks = []
        for seq in sequences:
            for i in range(0, len(seq)):
                chunk = seq[i:i+seq_length]
                if len(chunk) < seq_length:
                    continue
                self.chunks.append(chunk)
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        # 将chunk转换为PyTorch张量，分为输入序列与预测序列
        chunk_x = torch.tensor(chunk[:-1], dtype=torch.long)
        chunk_y = torch.tensor(chunk[1:], dtype=torch.long)
        return chunk_x, chunk_y