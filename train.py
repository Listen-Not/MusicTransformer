import dataSet
import model
import config
import utils
import tweak
import time
import torch

"预设随机种子，复现结果"
utils.set_seed(42)

"数据集预处理"
config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(config.DEVICE)

dataset = dataSet.MIDIDataset(config.SAVE_DIR)
print(dataset)

"模型实例化"
music_model = model.MusicTransformer(
    embedding_dim=config.EMBEDDING_DIM,
    vocab_size=config.VOCAB_SIZE,
    num_layer=config.NUM_LAYERS,
    max_seq=config.MAX_SEQ_LEN,
    dropout=config.DROP_OUT,
).to(config.DEVICE)
print(music_model)


"优化器，学习率衰减，损失函数"
optimizer = torch.optim.Adam(music_model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
scheduler = tweak.CustomLRScheduler(optimizer, d_model=config.EMBEDDING_DIM)

loss_func = tweak.SmoothCrossEntropyLoss(
    label_smoothing=config.LABEL_SMOOTHING, vocab_size=config.VOCAB_SIZE, ignore_index=config.PAD_TOKEN
)
accuracy_func = tweak.CategoricalAccuracy(config.VOCAB_SIZE).to(config.DEVICE)
losses = []  # 用于存储每个 epoch 的损失值

"开训"
print(">>> 开训...")
for epoch in range(config.EPOCHS):
    print(f">>> Epoch {epoch + 1} / {config.EPOCHS}")
    start_time = time.time()
    for batch in range(len(dataset.files) // config.BATCH_SIZE):
        scheduler.optimizer.zero_grad()

        try:
            # 通过自定义的 slide_seq2seq_batch 方法获取批次
            batch_x, batch_y = dataset.slide_seq2seq_batch(config.BATCH_SIZE, config.SEQ_LEN, predict=config.SLIDE_LEN)
            batch_x = torch.from_numpy(batch_x).contiguous().to(config.DEVICE, non_blocking=True, dtype=torch.long)
            batch_y = torch.from_numpy(batch_y).contiguous().to(config.DEVICE, non_blocking=True, dtype=torch.long)
        except IndexError:
            continue  # 如果没有足够的数据则跳过当前批次

        music_model.train()

        output = music_model(batch_x)  # 模型前向传播

        loss = loss_func(output, batch_y)  # 计算损失
        loss.backward()

        optimizer.step()  # 迭代，学习率优化
        scheduler.step()

        losses.append(loss.item())  # 记录每一个 batch 的 loss

    end_time = time.time()
    if config.DEBUG:
        music_model.eval()
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(config.BATCH_SIZE, config.MAX_SEQ_LEN, predict=config.SLIDE_LEN, mode="eval")
            batch_x = torch.from_numpy(batch_x).contiguous().to(config.DEVICE, non_blocking=True, dtype=torch.long)
            batch_y = torch.from_numpy(batch_y).contiguous().to(config.DEVICE, non_blocking=True, dtype=torch.long)
        except IndexError:
            continue
        with torch.no_grad():
            prediction = music_model(batch_x)
            accuracy = accuracy_func(prediction, batch_y)
            print(f"[Loss]: {loss.item():.4f} [Accuracy]: {accuracy.item():.4f} (Time: {end_time - start_time:.4f}s)")

if True:
    torch.save(music_model.state_dict(), "music_model.pth")
