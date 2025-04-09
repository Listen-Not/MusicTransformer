import os

"文件路径"
# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 定义输入 MIDI 文件目录，使用 os.path.join 保证跨平台路径兼容性
MIDI_DIR = os.path.join(BASE_DIR, "TrainFiles", "Input")
# 定义输出 PICKLE 文件目录，使用 os.path.join 保证跨平台路径兼容性
SAVE_DIR = os.path.join(BASE_DIR, "TrainFiles", "Output")

"默认训练参数"
# 单次训练批数
BATCH_SIZE = 32
# 训练轮数
EPOCHS = 5
# 乐句最大长度
MAX_SEQ_LEN = 256
# 乐句长度
SEQ_LEN = 32
