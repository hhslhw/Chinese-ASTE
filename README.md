# Chinese-ASTE
个人nlp课程大作业，选用Qwen3与Gemini的轻量化模型完成中文ASTE任务
🎭 基于 LLM 的中文属性级情感三元组抽取 (ASTE)
自然语言处理课程设计 | 上海海事大学

模型： Qwen-1.7B / Qwen-4B / Gemma-4B
方法： Zero-shot / Few-shot ICL / LoRA Fine-tuning





📖 项目简介 (Introduction)
本项目针对 中文属性级情感分析 (Aspect Sentiment Triplet Extraction, ASTE) 任务，探索了轻量级大语言模型（<7B）的性能边界。任务目标是从非结构化评论中抽取 (评价对象 Aspect, 观点词 Opinion, 情感极性 Sentiment) 三元组。

我们基于 Qwen (通义千问) 和 Gemma 系列模型，系统对比了零样本推理 (Zero-shot)、少样本学习 (Few-shot) 以及 LoRA 指令微调的效果。实验证明，通过参数高效微调，1.7B 的小模型在垂直领域的结构化抽取任务上可以达到工业级可用水平。

✨ 核心特性 (Key Features)
多模型对比：横向测评 Qwen-1.7B, Qwen-4B 与 Google Gemma-4B，验证了国产模型在中文语境下的优势。
全流程范式：涵盖 Prompt Engineering (Zero/Few-shot) 与 Parameter-Efficient Fine-Tuning (LoRA)。
深度消融实验：
Prompt 消融：探究任务定义与角色扮演对指令遵循的影响。
Rank 消融：对比 
𝑟
=
16
r=16 与 
𝑟
=
64
r=64，发现低秩设置能有效防止过拟合。
双重评估体系：设计 Strict (严格匹配) 与 Soft (模糊匹配) 双指标，并深入分析了数据集标注滞后带来的“假阳性悖论”。
📂 文件结构 (File Structure)
<BASH>
.
├── data/                   # 数据集文件夹 (chn_review_aste)
│   ├── train.json          # 训练集
│   ├── test.json           # 测试集
│   └── dev.json            # 验证集
├── output/                 # 模型推理输出结果 (.jsonl)
├── src/                    # 源代码
│   ├── train.py            # LoRA 微调训练脚本
│   ├── inference.py        # 模型推理脚本
│   ├── evaluation.py       # 评估脚本 (计算 Precision/Recall/F1)
│   └── utils.py            # 数据预处理与Prompt构建工具
├── report/                 # 课程设计报告与分析图表
├── requirements.txt        # 环境依赖
└── README.md               # 项目说明文档
🚀 快速开始 (Getting Started)
1. 环境安装
建议使用 Conda 创建虚拟环境：

<BASH>
conda create -n aste_llm python=3.10
conda activate aste_llm
pip install -r requirements.txt
主要依赖库：torch, transformers, peft, accelerate, bitsandbytes.

2. LoRA 微调 (Training)
运行以下命令启动 Qwen-4B 的 LoRA 微调：

<BASH>
python src/train.py \
    --model_name_or_path "Qwen/Qwen-4B" \
    --data_path "data/train.json" \
    --output_dir "checkpoints/qwen_lora" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --batch_size 4 \
    --gradient_accumulation_steps 4
3. 推理与评估 (Evaluation)
微调后进行推理并生成评估报告：

<BASH>
# 生成推理结果
python src/inference.py --model_path "checkpoints/qwen_lora" --test_data "data/test.json"
# 计算指标
python src/evaluation.py --pred_file "output/output_Qwen_4B_lora.jsonl"
📊 实验结果 (Results)
我们在 chn_review_aste 数据集上进行了全面测试。以下是 Qwen-4B 模型的主要指标对比：

方法 (Method)	Strict Precision	Strict Recall	Strict F1	Soft F1
Zero-shot	17.61%	26.31%	20.19%	53.27%
Few-shot (4-shot)	19.89%	29.45%	22.46%	53.26%
LoRA (r=16)	71.78%	84.77%	75.64%	79.40%
结论： LoRA 微调后的 Strict F1 相比 Zero-shot 提升了约 3.7倍。

🔍 消融分析亮点
LoRA Rank 选择：实验发现 
𝑟
=
16
r=16 (F1=0.7564) 优于 
𝑟
=
64
r=64 (F1=0.7150)。证明了 ASTE 任务具有低内在维度，过高的秩会导致过拟合。
模型选型：在相同微调设置下，Qwen-1.7B (F1=0.71) 显著优于 Gemma-4B (F1=0.64)，表明基座模型的语言分布（中文语料占比）对中文任务至关重要。
📝 引用与致谢 (Credits)
本项目是上海海事大学《自然语言处理》课程设计作品。

作者： 黑向阳 (算法/训练), 王俊皓 (评估/工程)
指导教师： 谢雨波
数据集来源： Automated Construction of Chinese ASTE Dataset
如有任何问题，欢迎提交 Issue 或联系作者。
