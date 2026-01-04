import json
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置 =================
# 填入trainer_state_qw_v1.json 路径
JSON_PATH = "trainer_state_qw_v2.json"


def plot_optimized(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file}")
        return

    history = data.get("log_history", [])

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for item in history:
        if "loss" in item and "step" in item:
            train_steps.append(item["step"])
            train_losses.append(item["loss"])
        if "eval_loss" in item and "step" in item:
            eval_steps.append(item["step"])
            eval_losses.append(item["eval_loss"])

    if not train_steps:
        print("未读取到数据")
        return

    # ================= 开始画图 (上下两张图) =================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # --- 图1：对数坐标 (Log Scale) ---
    # 作用：同时展示初始的高 Loss 和后期的低 Loss，不会被压缩
    ax1.plot(train_steps, train_losses, label="Train Loss", alpha=0.6, color='blue')
    if eval_steps:
        ax1.plot(eval_steps, eval_losses, label="Eval Loss", color='red', linewidth=2, marker='o', markersize=4)

    ax1.set_yscale('log')  # <--- 关键：设置对数坐标
    ax1.set_title("1. Logarithmic Scale (Global Trend)", fontsize=14)
    ax1.set_ylabel("Loss (Log Scale)")
    ax1.grid(True, which="both", linestyle='--', alpha=0.5)
    ax1.legend()

    # --- 图2：线性坐标 + 截断 (Zoomed View) ---
    # 作用：强行切掉头部高 Loss，只看 Loss < 1.5 的区域

    # 自动计算裁剪阈值：取后 80% 数据的最大值，或者强制设为 1.5
    # 如果强制看 1.0 以下，把下面的 y_limit 改成 1.0
    y_limit = 0.7

    ax2.plot(train_steps, train_losses, label="Train Loss", alpha=0.5, color='blue')

    # 平滑曲线 (Moving Average)，让趋势更清晰
    window_size = 5
    if len(train_losses) > window_size:
        smooth_loss = np.convolve(train_losses, np.ones(window_size) / window_size, mode='valid')
        # 补齐 x 轴长度
        smooth_steps = train_steps[window_size - 1:]
        ax2.plot(smooth_steps, smooth_loss, label=f"Train (Smoothed {window_size})", color='darkblue', linewidth=1.5)

    if eval_steps:
        ax2.plot(eval_steps, eval_losses, label="Eval Loss", color='red', linewidth=2, marker='o')

    ax2.set_ylim(0, y_limit)  # 强制设置 Y 轴范围为 0 到 1.5
    ax2.set_title(f"2. Zoomed Linear Scale (Focus on Loss < {y_limit})", fontsize=14)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    save_path = "loss_curve_qw_4_v2.png.png"
    plt.savefig(save_path, dpi=300)
    print(f"优化后的曲线图已保存为: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_optimized(JSON_PATH)

