import os
import random
import pickle
import numpy as np
from torch.utils.data import Dataset
from glob import glob


class MIDIDataset(Dataset):
    def __init__(self, dir_path, split_ratio=(0.8, 0.1, 0.1), mode="train"):
        """
        初始化 MIDI 数据集
        :param dir_path: 存放 .pickle 文件的路径
        :param split_ratio: 训练、验证、测试划分比例
        :param mode: 当前使用的数据划分（train/eval/test）
        """
        self.files = sorted(glob(os.path.join(dir_path, '*.pickle')))
        total = len(self.files)
        train_end = int(total * split_ratio[0])
        eval_end = train_end + int(total * split_ratio[1])

        self.file_dict = {
            "train": self.files[:train_end],
            "eval": self.files[train_end:eval_end],
            "test": self.files[eval_end:],
        }

        self.mode = mode
        self._seq_file_name_idx = 0
        self._seq_idx = 0

    def __len__(self):
        return len(self.file_dict[self.mode])

    def __getitem__(self, idx):
        return self._get_seq(self.file_dict[self.mode][idx])

    def __repr__(self):
        return f'<MIDIDataset mode="{self.mode}", files={len(self.file_dict[self.mode])}>'

    def _get_seq(self, fname, max_length=None):
        with open(fname, "rb") as f:
            data = pickle.load(f)

        if max_length is not None:
            if max_length <= len(data):
                start = random.randint(0, len(data) - max_length)
                return data[start : start + max_length]
            else:
                raise IndexError("数据长度不足指定长度")
        return data

    def batch(self, batch_size, length):
        batch_files = random.sample(self.file_dict[self.mode], k=batch_size)
        batch_data = [self._get_seq(f, length) for f in batch_files]
        return np.array(batch_data)

    def seq2seq_batch(self, batch_size, length):
        data = self.batch(batch_size, length * 2)
        return data[:, :length], data[:, length:]

    def slide_seq2seq_batch(self, batch_size, length):
        data = self.batch(batch_size, length + 1)
        return data[:, :-1], data[:, 1:]

    def smallest_encoder_batch(self, batch_size, length):
        data = self.batch(batch_size, length * 2)
        return data[:, : length // 100], data[:, length // 100 : length // 100 + length]

    def random_sequential_batch(self, batch_size, length):
        """
        从不同文件中截取长度为 `length` 的片段组成一个 batch
        """
        batch_files = random.sample(self.files, k=batch_size)
        batch_data = []
        for path in batch_files:
            data = self._get_seq(path)
            for i in range(len(data) - length):
                batch_data.append(data[i : i + length])
                if len(batch_data) == batch_size:
                    return np.array(batch_data)

    def sequential_batch(self, batch_size, length):
        """
        顺序迭代所有文件并生成固定长度的 batch，跨文件维护索引
        """
        batch_data = []
        data = self._get_seq(self.files[self._seq_file_name_idx])

        while len(batch_data) < batch_size:
            while self._seq_idx < len(data) - length:
                batch_data.append(data[self._seq_idx : self._seq_idx + length])
                self._seq_idx += 1
                if len(batch_data) == batch_size:
                    return np.array(batch_data)

            # 当前文件结束，切换到下一个
            self._seq_idx = 0
            self._seq_file_name_idx = (self._seq_file_name_idx + 1) % len(self.files)
            data = self._get_seq(self.files[self._seq_file_name_idx])
            if self._seq_file_name_idx == 0:
                print("所有文件轮换完成，开始新一轮顺序读取。")
