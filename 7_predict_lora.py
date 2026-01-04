import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ================= 配置区域 (请与训练脚本保持一致) =================
# 1. 基座模型 (必须与训练时一致)
BASE_MODEL_PATH = "/root/autodl-tmp/model/Qwen3-4B-Instruct-2507"
#/root/autodl-tmp/model/Gemma-3-4b
#/root/autodl-tmp/model/Qwen3-4B-Instruct-2507
# 2. LoRA 权重路径 (训练脚本输出的 final_adapter 目录)
LORA_ADAPTER_PATH = "lora_checkpoints_qwen_4b_r=64/final_adapter"
#lora_checkpoints_qwen_4b
# 3. 数据路径
INPUT_FILE = "/root/autodl-tmp/nlp/data_/test/test_gold300_cleaned.json"
OUTPUT_FILE = "output_Qwen_4B_lora_v2.jsonl"

DEVICE = "cuda"
DTYPE = torch.bfloat16  # 推荐与训练时保持一致


# ================= 核心 Prompt 逻辑 (必须与训练脚本 100% 一致) =================

def clean_input(raw_question):
    """清洗输入，去除 context: 前缀"""
    if "context:" in raw_question:
        return raw_question.split("context:")[-1].strip()
    return raw_question


def get_system_instruction():
    """【关键】必须复制训练脚本里的 System Prompt"""
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
    """【关键】必须复制训练脚本里的 User Format"""
    return f"文本内容：\n\"{text}\""


# ================= 主程序 =================

def main():
    print(f"1. 正在加载基座模型: {BASE_MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=DTYPE,
        trust_remote_code=True
    )

    print(f"2. 正在加载 LoRA 权重: {LORA_ADAPTER_PATH} ...")
    try:
        # 将 LoRA 权重挂载到基座模型上
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        model.eval()  # 切换到评估模式 (关闭 Dropout 等)
    except Exception as e:
        print(f"LoRA 加载失败，请检查路径！错误信息: {e}")
        return

    # 加载数据
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    #data = data[:50] # 调试时可以只跑前50条

    print(f"3. 开始 LoRA 推理，共 {len(data)} 条...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data):
            # 1. 准备输入
            context = clean_input(item['question'])

            # 2. 构建 Prompt (严格遵循训练格式)
            messages = [
                {"role": "system", "content": get_system_instruction()},
                {"role": "user", "content": format_user_input(context)}
            ]

            # 3. 转化为 Input IDs
            # add_generation_prompt=True 很重要！
            # 它会自动添加 assistant 的起始符 (如 <|im_start|>assistant\n)，引导模型开始输出
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

            # 4. 生成
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,  # 输出长度限制
                    temperature=0.1,  # 低温度，让 JSON 格式更稳定
                    top_p=0.9,
                    do_sample=True,  # 如果 temperature=0，这里可以设为 False
                    eos_token_id=tokenizer.eos_token_id
                )

            # 5. 解码 (去掉 Input 部分，只留 Output)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 6. 保存结果
            result_item = {
                "id": item.get('id', 'unknown'),  # 兼容没有id的情况
                "original_context": context,
                "ground_truth_str": item['answer'],
                "model_prediction": response.strip()
            }
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            f_out.flush()  # 实时写入硬盘

    print(f"LoRA 推理完成！结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
