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
BATCH_SIZE = 2
# 训练轮数
EPOCHS = 50
# 学习率
LEARNING_RATE = 0.0001
# 学习率衰减
LEARNING_RATE_DECAY = 0.5
# 标签光滑
LABEL_SMOOTHING = 0.1
# 残差
DEBUG = True

"输入数据设置长度"
# 事件个数
VOCAB_SIZE = 391
# 掩码标记
PAD_TOKEN = 388
# 乐句最大长度
MAX_SEQ_LEN = 2048
# 乐句长度
SEQ_LEN = 2048
# 词向量维度
EMBEDDING_DIM = 256
# 遗忘率
DROP_OUT = 0.1
# 编码器层数
NUM_LAYERS = 6
