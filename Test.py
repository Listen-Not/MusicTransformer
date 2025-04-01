import Model 
import matplotlib.pyplot as plt
import math

# 初始化嵌入层
vocab_size = 100  # 假设词汇表大小为 100
d_model = 64      # 嵌入维度为 64
embedding_layer = Model.Embeddings(d_model, vocab_size)

# 提取嵌入向量
embeddings = embedding_layer.lut.weight.detach().numpy()  # (vocab_size, d_model)

# 选择 n 个词向量
n = 5  # 选择前 5 个词向量
selected_embeddings = embeddings[:n]  # (n, d_model)
selected_embeddings=selected_embeddings/math.sqrt(d_model)

# 绘制每个词向量的分量
plt.figure(figsize=(12, 6))
for i, embedding in enumerate(selected_embeddings):
    plt.plot(range(d_model), embedding, label=f"Word {i}")

# 添加图例和标签
plt.title(f"Visualization of {n} Word Embeddings")
plt.xlabel("Embedding Dimension")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()


model = Model.PositionalEncoding(d_model=512,max_len=30)
print(model.pe.requires_grad)  # 输出: False