import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# ================= 配置区域 =================
# 模型路径
MODEL_PATH = "/root/autodl-tmp/model/Qwen3-4B-Instruct-2507"
#MODEL_PATH = "/root/autodl-tmp/model/Gemma-3-4b"  # 跑完一个换另一个
#MODEL_PATH = "/root/autodl-tmp/model/Qwen3-1.7B"
# 输入数据路径
INPUT_FILE = "/root/autodl-tmp/nlp/data_/test/test_gold300_cleaned.json"  # 原始数据文件
# 输出文件路径
OUTPUT_FILE = "output_Qwen_4B_zeroshot_v3.jsonl"

# 设备配置 (RTX 5090 支持 bfloat16，速度更快且省显存)
DEVICE = "cuda"
DTYPE = torch.bfloat16


# ================= 核心逻辑 =================

def load_data(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        # jsonl 格式(一行一个json)用这个，如果是大列表用 json.load
        data = [json.loads(line) for line in f]
    """加载数据，只取前 50 条测试代码[:50]"""
    return data


def clean_input(raw_question):
    """
    【关键步骤】清洗数据，防止答案泄露。
    输入: "question: 请问... context: 这里的牛肉面很好吃"
    输出: "这里的牛肉面很好吃"
    """
    if "context:" in raw_question:
        return raw_question.split("context:")[-1].strip()
    return raw_question


"""def build_prompt(context_text):
    prompt = f",""你是一个专业的情感分析助手。请从以下文本中提取“评价对象(Aspect)”、“观点词(Opinion)”和“情感极性(Sentiment)”

文本内容：
"{context_text}"

要求：
1. 识别出文本中用户评价的具体对象。
2. 识别出对该对象的具体描述词。
3. 判断情感极性，只能从 [正面, 负面, 中性] 中选择。
4. 请务必以严格的 JSON 列表格式输出，不要包含任何解释性文字。
   格式例如：[{{"aspect": " ", "opinion": "", "sentiment": ""}}]
   如果没有观点，输出 []。
",""
    return prompt"""

"""def build_prompt(context_text):
    prompt = f"1""你是一个专业的情感分析助手。请从以下文本中提取“评价对象(Aspect)”、“观点词(Opinion)”和“情感极性(Sentiment)”

文本内容：
"{context_text}"

要求：

1. 情感极性只能从 [正面, 负面, 中性] 中选择。
4. 请务必以严格的 JSON 列表格式输出，不要包含任何解释性文字。
   格式例如：[{{"aspect": " ", "opinion": "", "sentiment": ""}}]
   如果没有观点，输出 []。
"1""
#去处详细要求
    return prompt
"""
def build_prompt(context_text):
    prompt = f"""请按要求从文本中完成抽取任务”

文本内容：
"{context_text}"

要求：
1. 识别出文本中用户评价的具体对象。
2. 识别出对该对象的具体描述词。
3. 判断情感极性，只能从 [正面, 负面, 中性] 中选择。
4. 请务必以严格的 JSON 列表格式输出，不要包含任何解释性文字。
   格式例如：[{{"aspect": " ", "opinion": "", "sentiment": ""}}]
   如果没有观点，输出 []。
"""
#去除角色扮演
    return prompt


def main():
    print(f"正在加载模型: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=DTYPE,
        trust_remote_code=True
    )

    data = load_data(INPUT_FILE)
    results = []

    print(f"开始推理，共 {len(data)} 条数据...")

    # 打开文件准备写入 (JSONL 格式，写一条存一条，防止中断)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data):
            # 1. 数据清洗
            context = clean_input(item['question'])

            # 2. 构建对话消息
            user_prompt = build_prompt(context)
            messages = [
                {"role": "system", "content": "你是一个严谨的情感分析专家，只输出 JSON 数据。"},
                {"role": "user", "content": user_prompt}
            ]

            # 3. 转化为模型输入
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking = False
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

            # 4. 生成 (设置 max_new_tokens 防止废话太多)
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature=0.1,  # 低温度，让输出更稳定、格式更规范
                    top_p=0.9
                )

            # 5. 解码
            # 只取生成的部分，去掉输入的 Prompt
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 6. 保存结果
            result_item = {
                "id": item['id'],
                "original_context": context,
                "ground_truth_str": item['answer'],  # 保留原始答案方便后续评估
                "model_prediction": response
            }

            # 写入文件
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            f_out.flush()  # 强制刷入硬盘

    print(f"推理完成！结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()