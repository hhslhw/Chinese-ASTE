import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置：在这里直接指定三个文件的路径 =================
TARGET_FILES = [
    #"results/eval_Qwen_1.7B_fewshot.json",
    #"results/eval_Qwen_4B_fewshot.json",
    #"results/eval_Gemma_4B_fewshot.json"
    #"results/eval_Qwen_1.7B_lora.json",
    "results/eval_Qwen_4B_lora.json",
    "results/eval_Qwen_4B_lora_v2.json"
    #"results/eval_Gemma_4B_lora.json"
    #"results/eval_Qwen_1.7B_zeroshot.json",
    #"results/eval_Qwen_4B_zeroshot.json",
    #"results/eval_Gemma_4B_zeroshot.json"
    #"results/eval_Qwen_4B_fewshot.json",
    #"results/eval_Qwen_4B_fewshot_v2.json",
    #"results/eval_Qwen_4B_fewshot_v3.json",
]

# 这里定义在图表中显示的友好名称（Key 是文件名，不带 .json）
MODEL_NAME_MAP = {
    "eval_Qwen_1.7B_fewshot": "Qwen3-1.7B (Few-Shot)",
    "eval_Qwen_4B_fewshot": "Qwen3-4B (Few-Shot)",
    # 注意：这里修正了你原代码中的拼写错误 (Qemma -> Gemma) 以便正确匹配
    "eval_Gemma_4B_fewshot": "Gemma-3-4B (Few-Shot)"
    #"eval_Qwen_1.7B_zeroshot1": "Qwen3-1.7B (Zero-Shot)",
    #"eval_Qwen_4B_zeroshot": "Qwen3-4B (Zero-Shot)",
    #"eval_Gemma_4B_zeroshot": "Gemma-3-4B (Zero-Shot)"
    #"eval_Qwen_1.7B_lora": "Qwen3-1.7B (Lora)",
    #"eval_Qwen_4B_lora": "Qwen3-4B (Lora)",
    #"eval_Gemma_4B_lora": "Gemma-3-4B (Lora)"
}


# ================= 1. 数据读取 =================
def load_selected_data(file_list):
    all_data = []
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)

        file_key = os.path.splitext(os.path.basename(file_path))[0]
        display_name = MODEL_NAME_MAP.get(file_key, file_key)

        # 映射到英文标签
        for mode_key, mode_name in [("strict", "Strict Match"), ("soft", "Soft Match")]:
            for metric in ["accuracy", "precision", "recall", "f1"]:
                all_data.append({
                    "Model": display_name,
                    "Mode": mode_name,
                    "Metric": metric.upper(),
                    "Score": content[mode_key][metric]
                })
    return pd.DataFrame(all_data)


# ================= 2. 绘图逻辑 (分开保存) =================
def plot_comparison_separate(df):
    if df.empty:
        print("Error: No data to plot.")
        return

    # 设置 Seaborn 风格
    sns.set_theme(style="whitegrid", font="sans-serif")

    modes = ["Strict Match", "Soft Match"]
    metrics_order = ["ACCURACY", "PRECISION", "RECALL", "F1"]

    # 颜色组合
    custom_palette = ["#4A90E2", "#50E3C2", "#F5A623", "#E35B5B"]

    # --- 循环处理每种模式，单独绘图并保存 ---
    for mode in modes:
        # 1. 创建单独的画布
        plt.figure(figsize=(10, 7))

        # 2. 筛选数据
        sub_df = df[df["Mode"] == mode]

        # 3. 绘图
        ax = sns.barplot(
            data=sub_df,
            x="Model",
            y="Score",
            hue="Metric",
            hue_order=metrics_order,
            palette=custom_palette,
            edgecolor='black',
            linewidth=0.8
        )

        # 4. 设置标题和标签 (去掉 suptitle，只保留当前图标题)
        plt.title(f"Performance: {mode}", fontsize=16, fontweight='bold', pad=20)
        plt.ylim(0, 1.15)  # 稍微调高一点，给图例留空间
        plt.ylabel("Score (0-1)", fontsize=12)
        plt.xlabel("")  # 去掉X轴标题

        # 5. 数值标注
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9, fontweight='bold')

        # 6. 图例设置 (每张图都需要图例)
        plt.legend(title="Metrics", loc='upper right', bbox_to_anchor=(1, 1), ncol=4, fontsize=10)

        # 7. 定义输出文件名
        # 将 "Strict Match" 转换为 "strict_match"
        file_suffix = mode.lower().replace(" ", "_")
        output_filename = f"model_comparison_l_{file_suffix}.png"

        # 8. 保存并关闭
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭当前画布，防止重叠

        print(f"Success! Saved image: {output_filename}")


if __name__ == "__main__":
    df = load_selected_data(TARGET_FILES)
    if not df.empty:
        # 打印摘要表格
        print("\n" + "=" * 60)
        print("                 Evaluation Summary Table")
        print("=" * 60)
        summary = df.pivot_table(index=['Model', 'Mode'], columns='Metric', values='Score')
        print(summary)
        print("=" * 60)

        # 执行分开绘图
        plot_comparison_separate(df)

