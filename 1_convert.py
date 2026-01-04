import json
import os
from opencc import OpenCC

def convert_file(input_path, output_path):
    """转换单个文件"""
    cc = OpenCC('t2s')  # 繁体转简体

    # 读取文件并转换
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    converted_lines = []
    for line in lines:
        line = line.strip()
        if line:  # 跳过空行
            try:
                # 尝试解析JSON
                data = json.loads(line)
                # 转换字符串字段
                for key in data:
                    if isinstance(data[key], str):
                        data[key] = cc.convert(data[key])
                converted_lines.append(json.dumps(data, ensure_ascii=False))
            except json.JSONDecodeError:
                # 如果不是JSON格式，直接转换整行
                converted_lines.append(cc.convert(line))

    # 保存转换后的文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in converted_lines:
            f.write(line + '\n')

def batch_convert(input_dir, output_dir):
    """批量转换目录下所有文件"""
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"转换: {input_path} -> {output_path}")
            convert_file(input_path, output_path)

if __name__ == "__main__":
    # 转换test目录
    batch_convert('data/test', 'data_/test')
    batch_convert('data/train', 'data_/train')
    batch_convert('data/valid', 'data_/valid')
    print("转换完成！")