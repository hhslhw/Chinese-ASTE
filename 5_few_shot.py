import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# ================= 配置区域 =================
# 模型路径
#MODEL_PATH = "/root/autodl-tmp/model/Gemma-3-4b"
#MODEL_PATH = "/root/autodl-tmp/model/Qwen3-1.7B"
MODEL_PATH = "/root/autodl-tmp/model/Qwen3-4B-Instruct-2507"  # 模型路径

# 输入数据路径
INPUT_FILE = "/root/autodl-tmp/nlp/data_/test/test_gold300_cleaned.json"
# 输出文件路径
OUTPUT_FILE = "output_Qwen_4B_fewshot_v2.jsonl"

# 设备配置
DEVICE = "cuda"
DTYPE = torch.bfloat16

# ================= 少样本示例 (Few-Shot Examples) =================
# 这里定义 2-3 个高质量的示例，引导模型学习人工标注的风格
# 注意：示例的格式必须完全符合 Prompt 中的 JSON 要求
FEW_SHOT_EXAMPLES = [

    {
        "context": "环境不错，很适合拍照。就是牛肉稍微有点老，配菜也不新鲜。",
        "answer": '[{"aspect": "环境", "opinion": "不错", "sentiment": "正面"}, '
                  '{"aspect": "牛肉", "opinion": "有点老", "sentiment": "负面"}, '
                  '{"aspect": "配菜", "opinion": "不新鲜", "sentiment": "负面"}]'
    },

]
"""{
        "context": "这家的炸鸡很好吃，但是可乐没气了，服务员态度也很冷淡。",
        "answer": '[{"aspect": "炸鸡", "opinion": "好吃", "sentiment": "正面"}, '
                  '{"aspect": "服务员态度", "opinion": "冷淡", "sentiment": "负面"}]'
    },
{
    "context": "餐点好吃，餐厅气氛佳。",
    "answer": '[{"aspect":"餐点","opinion":"好吃”,"sentiment": "正面"},'
              '{"aspect": "气氛", "opinion": "佳", "sentiment": "正面"}]'
},
{
    "context": "猪脚（可选肉或蹄）很大只又好吃，手工面条很Q。",
    "answer": '[{"aspect": "猪脚", "opinion": "大", "sentiment": "正面"}, '
              '{"aspect": "猪脚", "opinion": "好吃", "sentiment": "正面"}, '
              '{"aspect": "面条", "opinion": "Q", "sentiment": "正面"}]'
}
"""
# ================= 核心逻辑 =================

def load_data(file_path):
    """加载数据，逻辑与零样本脚本保持一致"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 只取前 50 条进行测试[:50]
    return data


def clean_input(raw_question):
    """数据清洗，与零样本脚本保持一致"""
    if "context:" in raw_question:
        return raw_question.split("context:")[-1].strip()
    return raw_question


def get_system_instruction():
    """
    获取系统指令。
    这里直接复用 Zero-shot 脚本中的高通过率提示词，
    但将其作为系统指令，统领整个对话。
    """
    return """你是一个专业的情感分析助手。请从文本中提取“评价对象(Aspect)”、“观点词(Opinion)”和“情感极性(Sentiment)”。

要求：
1. 识别出文本中用户评价的具体对象。
2. 识别出对该对象的具体描述词。
3. 判断情感极性，只能从 [正面, 负面, 中性] 中选择。
4. 请务必以严格的 JSON 列表格式输出，不要包含任何解释性文字。
   格式例如：[{"aspect": "...", "opinion": "...", "sentiment": "..."}]
   如果没有观点，输出 []。
5.禁止输出示例的Aspect、Opinion及Sentiment。
"""


def format_user_input(text):
    """格式化用户输入，保持格式统一"""
    return f"文本内容：\n\"{text}\""


def build_messages(current_context):
    """
    构建完整的对话历史：System指令 -> Few-Shot示例 -> 当前用户提问
    """
    # 1. 系统指令
    messages = [{"role": "system", "content": get_system_instruction()}]

    # 2. 插入少样本示例 (In-Context Learning)
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": format_user_input(example['context'])})
        messages.append({"role": "assistant", "content": example['answer']})

    # 3. 插入当前测试数据
    messages.append({"role": "user", "content": format_user_input(current_context)})

    return messages


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

    print(f"开始 Few-Shot 推理，共 {len(data)} 条数据...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data):
            # 1. 数据清洗
            context = clean_input(item['question'])

            # 2. 构建对话消息 (包含指令、示例和当前问题)
            messages = build_messages(context)

            # 3. 转化为模型输入
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

            # 4. 生成 (参数与 Zero-Shot 保持一致)
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature=0.1,  # 保持低温，确保 JSON 格式稳定
                    top_p=0.9
                )

            # 5. 解码
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 6. 保存结果
            result_item = {
                "id": item['id'],
                "original_context": context,
                "ground_truth_str": item['answer'],
                "model_prediction": response
            }

            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"推理完成！结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
