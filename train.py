import dataSet
import model
import config
import utils
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

"预设随机种子"
utils.set_seed(42)

"数据集预处理"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = dataSet.MIDIDataset(config.SAVE_DIR)
print(dataset)

"模型实例化"
music_model = model.MusicTransformer(
    embedding_dim=config.EMBEDDING_DIM,
    vocab_size=config.VOCAB_SIZE,
    num_layer=config.NUM_LAYERS,
    max_seq=config.MAX_SEQ_LEN,
    dropout=config.DROP_OUT,
).to(device)
print(music_model)

"优化器，学习率衰减，损失函数"
optimizer = optim.Adam(music_model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=utils.get_lr_lambda(d_model=config.EMBEDDING_DIM, warmup_steps=4000)
)

loss_func = utils.SmoothCrossEntropyLoss(
    label_smoothing=config.LABEL_SMOOTHING, vocab_size=config.VOCAB_SIZE, ignore_index=config.PAD_TOKEN
)
losses = []  # 用于存储每个 epoch 的损失值

"开训"
print(">> 开训...")
for epoch in range(config.EPOCHS):
    print(f">>> Epoch {epoch + 1} / {config.EPOCHS}")
    start_time = time.time()
    for batch in range(len(dataset.files) // config.BATCH_SIZE):
        scheduler.optimizer.zero_grad()

        try:
            # 通过自定义的 slide_seq2seq_batch 方法获取批次
            batch_x, batch_y = dataset.slide_seq2seq_batch(config.BATCH_SIZE, config.MAX_SEQ_LEN)
            batch_x = torch.from_numpy(batch_x).contiguous().to(device, non_blocking=True, dtype=torch.long)
            batch_y = torch.from_numpy(batch_y).contiguous().to(device, non_blocking=True, dtype=torch.long)
        except IndexError:
            continue  # 如果没有足够的数据则跳过当前批次

        music_model.train()

        # 模型前向传播
        output = music_model(batch_x)

        # 计算损失
        loss = loss_func(output.view(-1, output.size(2)), batch_y.view(-1))
        loss.backward()

        # 更新参数
        optimizer.step()

        # 学习率调度
        scheduler.step()
        losses.append(loss.item())  # 记录每一个 batch 的 loss

    end_time = time.time()

    if config.DEBUG:
        print(f"[Loss]: {loss.item():.4f} (Time: {end_time - start_time:.4f}s)")

    # utils.plot_and_save_losses(losses, save_path="loss/loss_curve.png")

    # 每个 epoch 后可以在验证集上计算一次损失并更新学习率
    # val_loss = compute_validation_loss()  # 这里需要你定义一个计算验证集损失的方法
    # scheduler.step(val_loss)
