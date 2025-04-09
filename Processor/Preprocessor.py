import os
import sys
import pickle
from glob import glob
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from Processor.ProcessUtils import encode_midi


def preprocess_midi(file_path: str):
    """编码单个 MIDI 文件为事件序列"""
    return encode_midi(file_path)


def preprocess_all_midi_files(midi_dir: str, save_dir: str):
    """串行预处理指定目录下的所有 MIDI 文件"""
    os.makedirs(save_dir, exist_ok=True)
    midi_files = glob(os.path.join(midi_dir, "**", "*.mid"), recursive=True) + glob(
        os.path.join(midi_dir, "**", "*.midi"), recursive=True
    )

    for midi_path in tqdm(midi_files, desc="Processing MIDI files"):
        file_name = os.path.splitext(os.path.basename(midi_path))[0]
        output_path = os.path.join(save_dir, f"{file_name}.pickle")

        try:
            encoded_data = preprocess_midi(midi_path)
            with open(output_path, "wb") as f:
                pickle.dump(encoded_data, f)
        except Exception as e:
            print(f"\n[!] 处理文件出错: {midi_path}\n    错误信息: {e}")


if __name__ == "__main__":
    preprocess_all_midi_files(config.MIDI_DIR, config.SAVE_DIR)
