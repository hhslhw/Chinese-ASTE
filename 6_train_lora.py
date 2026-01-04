import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback  # 1. 确保导入了这个
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import math

# ================= 配置区域 =================
#MODEL_PATH = "/root/autodl-tmp/model/Qwen3-1.7B"
MODEL_PATH = "/root/autodl-tmp/model/Qwen3-4B-Instruct-2507"
#MODEL_PATH = "/root/autodl-tmp/model/Gemma-3-4b"
DATA_FILE = "/root/autodl-tmp/nlp/data_/train/train_cleaned.json"
OUTPUT_DIR = "lora_checkpoints_qwen_4b_r=64"
DEVICE = "cuda"

# 为了防止 OOM，这里使用了更安全的配置
PER_DEVICE_BATCH_SIZE = 4  # 单卡 Batch 改小
GRADIENT_ACCUMULATION = 4  # 累积步数改大，保持总 Batch=16 不变


# ================= 核心逻辑复用 =================

def clean_input(raw_question):
    if "context:" in raw_question:
        return raw_question.split("context:")[-1].strip()
    return raw_question


def get_system_instruction():
    return """你是一个专业的情感分析助手。请从文本中提取“评价对象(Aspect)”、“观点词(Opinion)”和“情感极性(Sentiment)”。

要求：
1. 识别出文本中用户评价的具体对象。
2. 识别出对该对象的具体描述词。
3. 判断情感极性，只能从 [正面, 负面, 中性] 中选择。
4. 请务必以严格的 JSON 列表格式输出，不要包含任何解释性文字。
   格式例如：[{"aspect": "...", "opinion": "...", "sentiment": "..."}]
   如果没有观点，输出 []。
"""


def format_user_input(text):
    return f"文本内容：\n\"{text}\""


# ================= 数据处理 =================

def process_func(example, tokenizer):
    MAX_LENGTH = 1024  # 限制长度防止显存爆炸

    context = clean_input(example['question'])
    instruction = format_user_input(context)
    ground_truth = example['answer']

    messages = [
        {"role": "system", "content": get_system_instruction()},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": ground_truth}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )[0]

    # 截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]

    labels = input_ids.clone()

    # 重新计算 prompt 长度用于 mask
    prompt_messages = [
        {"role": "system", "content": get_system_instruction()},
        {"role": "user", "content": instruction}
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )[0]

    prompt_len = len(prompt_ids)
    safe_len = min(prompt_len, len(labels))
    labels[:safe_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels
    }


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def main():
    print(f"正在加载 Tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("正在加载并处理数据...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f]

    ds = Dataset.from_list(data_list)
    # 划分验证集 (早停必须要有验证集)
    ds = ds.train_test_split(test_size=0.1)

    train_dataset = ds['train'].map(lambda x: process_func(x, tokenizer), remove_columns=ds['train'].column_names)
    eval_dataset = ds['test'].map(lambda x: process_func(x, tokenizer), remove_columns=ds['test'].column_names)

    # ================= 2. 动态计算 0.1 Epoch 的步数 =================
    total_samples = len(train_dataset)
    # 计算一个 epoch 有多少个 step (样本数 / (batch * accum))
    # 这里假设单卡，如果是多卡需要除以显卡数量
    steps_per_epoch = math.ceil(total_samples / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION))

    # 设定检测间隔为 0.1 个 Epoch
    eval_steps = int(steps_per_epoch * 0.1)
    if eval_steps < 1: eval_steps = 1

    print(f"训练集数量: {total_samples}")
    print(f"每个 Epoch 步数: {steps_per_epoch}")
    print(f"早停检测间隔 (0.1 Epoch): {eval_steps} steps")

    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    # 开启梯度检查点以防 OOM
    model.enable_input_require_grads()

    target_modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ================= 3. 训练参数 =================
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,  # 使用防OOM的配置
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,  # 使用防OOM的配置
        num_train_epochs=3,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,

        # --- 早停核心配置 ---
        eval_strategy="steps",  # 按步数评估，而不是按 Epoch
        eval_steps=eval_steps,  # 每 0.1 个 Epoch 评估一次
        save_strategy="steps",  # 必须与 eval_strategy 一致
        save_steps=eval_steps,  # 每次评估时也保存 Checkpoint

        load_best_model_at_end=True,  # 训练结束后自动加载验证集 Loss 最低的模型
        metric_for_best_model="loss",  # 监控 Loss
        greater_is_better=False,  # Loss 是越小越好
        save_total_limit=2,  # 只保留最近2个模型，省硬盘

        bf16=True,
        gradient_checkpointing=True,  # 开启显存优化
        logging_dir=f"{OUTPUT_DIR}/logs",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        # --- 4. 添加早停回调 ---
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        # patience=3 表示：如果连续 3 次评估（即跑了 0.3 个 Epoch） Loss 都没有下降，就停止。
        # 如果非常严格想要 0.1 Epoch 没下降就立刻停，把这个改成 1。但建议至少设为 2 或 3 以防波动。
    )

    print("开始训练...")
    trainer.train()

    final_save_path = f"{OUTPUT_DIR}/final_adapter"
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"训练完成！模型已保存至 {final_save_path}")


if __name__ == "__main__":
    main()
