from matplotlib import pyplot as plt
import torch

# 验证不同序列长度时的行为
import Model

if __name__ == "__main__":
    q = torch.randn(2, 10, 512)  # q_len=10
    k = torch.randn(2, 10, 512)  # k_len=10
    attn = Model.MultiHeadAttentionWithRelativePosition(d_model=512, num_heads=8)
    output, S_rel, qe_relative = attn(q, k, k)  # 应触发skew中的ValueError

    # 创建一个包含1行2列子图的画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制第一个头的注意力权重
    axes[0].imshow(S_rel[0, 0].detach().numpy(), cmap="viridis")
    axes[0].set_title("Attention Weights (Head 0, Sample 0)")
    axes[0].set_xlabel("Key Positions")
    axes[0].set_ylabel("Query Positions")

    # 绘制第二个头的注意力权重
    axes[1].imshow(S_rel[0, 1].detach().numpy(), cmap="viridis")
    axes[1].set_title("Attention Weights (Head 1, Sample 0)")
    axes[1].set_xlabel("Key Positions")
    axes[1].set_ylabel("Query Positions")

    # 优化布局并显示图像
    plt.tight_layout()
    plt.show()
