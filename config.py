import os
import glob

"文件路径"
# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 定义 MIDI 文件目录，使用 os.path.join 保证跨平台路径兼容性
MIDIFILES_DIR = os.path.join(BASE_DIR, "MidiFiles")
# 匹配MIDI文件夹下的所有文件
MIDIFILES = glob.glob(os.path.join(MIDIFILES_DIR, "*.mid"))

"默认训练参数"
# 单次训练批数
BATCH_SIZE = 32
# 训练轮数
EPOCHS = 5
# 乐句最大长度
MAX_SEQ_LEN = 256
# 乐句长度
SEQ_LEN = 32
