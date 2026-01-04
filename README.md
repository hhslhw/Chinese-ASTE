# 基于轻量化 LLM 的中文属性级情感三元组抽取 (ASTE)

> **自然语言处理课程设计**
>
> **模型：** Qwen-1.7B / Qwen-4B / Gemma-4B
> **方法：** Zero-shot / Few-shot ICL / LoRA Fine-tuning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)](https://github.com/huggingface/peft)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow)](https://huggingface.co/docs/transformers/index)

## 项目简介 (Introduction)

本项目针对 **中文属性级情感分析 (Aspect Sentiment Triplet Extraction, ASTE)** 任务，探索了轻量级大语言模型（<7B）的性能边界。任务目标是从非结构化评论中抽取 `(评价对象 Aspect, 观点词 Opinion, 情感极性 Sentiment)` 三元组。

我们基于 **Qwen (通义千问)** 和 **Gemma** 系列模型，系统对比了零样本推理 (Zero-shot)、少样本学习 (Few-shot) 以及 **LoRA 指令微调**的效果。

原数据集下载地址：https://github.com/chiawen0104/chn_review_aste
模型下载自hugging face平台

## 核心工作 (Core Work)

*   **多模型对比**：横向测评 Qwen-1.7B, Qwen-4B 与 Google Gemma-4B，验证了国产模型在中文语境下的优势。
*   **三项对比内容**：涵盖 Prompt Engineering (Zero/Few-shot) 与 Parameter-Efficient Fine-Tuning (LoRA)。
*   **深度消融实验**：探究不同模块对任务结果的影响。
*   **双重评估体系**：设计 `Strict` (严格匹配) 与 `Soft` (模糊匹配) 双指标，参考原数据集代码构建评估指标的构建方法”。

## 文件结构 (File Structure)

```bash
.
├── data/                   # 数据集文件夹 (chn_review_aste)
├── result/                 # 实验结果 (.jsonl) 及其可视化
├── src/                    # 数据集自带文件
├── 1_convert.py            # [步骤1] 繁体转简体
├── 2_visualize_data.py     # [步骤2] 数据集探索性分析与可视化
├── 3_clean_data.py         # [步骤3] 数据清洗
├── 4_zero_shot.py          # [实验A] 零样本 (Zero-shot) 推理脚本
├── 5_few_shot.py           # [实验B] 少样本 (Few-shot) 推理脚本
├── 6_train_lora.py         # [实验C] LoRA 微调训练主程序
├── 7_predict_lora.py       # [步骤7] 加载微调权重进行预测
├── 8_evaluation.py         # [步骤8] 核心评估脚本 (计算 Strict/Soft F1)
├── 9_visualize_results.py  # [步骤9] 实验指标对比可视化绘图
├── 10_visualize_train.py   # [步骤10] 绘制 Loss 训练曲线
├── download_model.py       # 模型权重自动下载工具（也可手动从hugging face下载）
└── json_analyze.py         # JSON 结构分析

