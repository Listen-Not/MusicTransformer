import os

"各种路径"
# 项目名称
REPO_NAME = "MusicTransformer"
# 输入 MIDI 文件的目录，使用 os.path.join 保证跨平台路径兼容性
MIDI_DIR = os.path.join("TrainFiles", "Input")
# 输出 PICKLE 文件的目录，使用 os.path.join 保证跨平台路径兼容性
SAVE_DIR = os.path.join("TrainFiles", "Output")
# 导出中间变量至文件夹，检查Tensor值
CHECK_ROOT = "Check"
# 模型权重保存目录
CHECK_DIR = os.path.join(CHECK_ROOT, REPO_NAME)

"默认训练参数"
# 训练设备
DEVICE = "cpu"
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
# 生成乐句长度
TARGET_LEN = 4096
# 滑动窗口长度
SLIDE_LEN = 1
# 词向量维度
EMBEDDING_DIM = 256
# 遗忘率
DROP_OUT = 0.1
# 编码器层数
NUM_LAYERS = 6
