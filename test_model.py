import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 检测可用设备（优先GPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅使用设备: {device.upper()} | GPU名称: {torch.cuda.get_device_name(0) if device=='cuda' else 'N/A'}")

# 直接加载本地模型
model_path = "/root/autodl-tmp/model/Qwen3-4B"  # 当前目录下的Qwen3-8B文件夹

# 初始化
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if (device=="cuda" and torch.cuda.is_bf16_supported()) else torch.float16,
    device_map="auto" if device=="cuda" else "cpu",  # GPU自动分配，CPU强制指定
    trust_remote_code=True
)

# 简易对话函数（增强输出格式）
def chat(prompt: str, enable_thinking: bool = True) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.6 if enable_thinking else 0.7,
        top_p=0.95 if enable_thinking else 0.8
    )

    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response.split("</think>")[-1].strip()

# 优化后的交互示例
if __name__ == "__main__":
    # 非思维模式示例
    user_input = "你好！"
    print(f"\n{'='*50}")
    print(f"用户输入: {user_input}")
    print(f"模型输出: {chat(user_input, enable_thinking=False)}")
    print(f"{'='*50}\n")

    # 思维模式示例（取消注释测试）
    """
    user_input = "如何用Python计算1到1000的质数？"
    print(f"\n{'='*50}")
    print(f"用户输入 (思维模式): {user_input}")
    print(f"思考过程 + 最终回答:")
    print(chat(user_input, enable_thinking=True))
    print(f"{'='*50}\n")
    """