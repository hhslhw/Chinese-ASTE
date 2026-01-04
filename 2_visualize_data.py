import json
import re
from collections import Counter

def load_jsonl_file(filepath):
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"跳过无效JSON行: {line[:50]}...")
    return data

def extract_context_from_question(question):
    """从question字段中提取context"""
    match = re.search(r'context:\s*(.*)', question)
    if match:
        return match.group(1).strip()
    return ""

def analyze_dataset(filepath, dataset_name):
    """分析数据集"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} 数据分析")
    print(f"{'='*60}")

    data = load_jsonl_file(filepath)
    total_samples = len(data)

    if total_samples == 0:
        print("数据集为空或格式错误")
        return

    # 统计空值
    empty_answers = sum(1 for item in data if not item.get('answer', ''))

    # 从question中提取context
    contexts = []
    for item in data:
        context = extract_context_from_question(item.get('question', ''))
        contexts.append(context)

    empty_contexts = sum(1 for ctx in contexts if not ctx)

    print(f"总样本数: {total_samples}")
    print(f"answer为空的样本数: {empty_answers} ({empty_answers/total_samples*100:.1f}%)")
    print(f"context为空的样本数: {empty_contexts} ({empty_contexts/total_samples*100:.1f}%)")

    # 统计评价对象个数
    aspect_counts = []
    all_aspects = []

    for item in data:
        if item.get('answer', ''):
            # 提取评价对象（aspect）
            match = re.search(r'(.+?)。', item['answer'])
            if match:
                aspects = [a.strip() for a in match.group(1).split('、') if a.strip()]
                aspect_counts.append(len(aspects))
                all_aspects.extend(aspects)

    if aspect_counts:
        avg_aspects = sum(aspect_counts) / len(aspect_counts)
        print(f"平均评价对象个数: {avg_aspects:.2f}")

        # 评价对象个数分布（4个以下单独分，5个及以上为一组）
        count_dist = Counter(aspect_counts)
        print("评价对象个数分布:")

        # 1-4个单独统计
        for i in range(1, 5):
            if i in count_dist:
                freq = count_dist[i]
                percentage = freq / len(aspect_counts) * 100
                print(f"  {i}个: {freq}个样本 ({percentage:.1f}%)")

        # 5个及以上合并统计
        count_5_plus = sum(count_dist[i] for i in range(5, max(aspect_counts) + 1) if i in count_dist)
        if count_5_plus > 0:
            percentage_5_plus = count_5_plus / len(aspect_counts) * 100
            print(f"  5个及以上: {count_5_plus}个样本 ({percentage_5_plus:.1f}%)")

    # 统计评价对象频率（前7个）
    aspect_freq = Counter(all_aspects)
    print(f"\n前7个最常见的评价对象:")
    for i, (aspect, count) in enumerate(aspect_freq.most_common(7), 1):
        print(f"  {i:2d}. {aspect:<7} ({count}次)")

    # 统计context长度
    context_lengths = [len(ctx) for ctx in contexts if ctx]  # 只统计非空context
    if context_lengths:
        avg_context_len = sum(context_lengths) / len(context_lengths)
        print(f"\ncontext长度统计:")
        print(f"  平均长度: {avg_context_len:.1f}字符")
        print(f"  最长: {max(context_lengths)}字符")
        print(f"  最短: {min(context_lengths)}字符")
        print(f"  有效context数量: {len(context_lengths)}")

    # 显示前5个数据示例（完整显示，不省略）
    print(f"\n前5个数据示例:")
    for i, item in enumerate(data[:5]):
        context = extract_context_from_question(item.get('question', ''))
        print(f"  样本{i+1}:")
        print(f"    ID: {item.get('id', 'N/A')}")
        print(f"    Context: {context}")
        print(f"    Answer: {item.get('answer', '')}")
        print()

def visualize_data():
    """可视化数据集"""
    datasets = [
        ('data_/test/test.json', 'test/test.json'),
        ('data_/test/test_gold300.json', 'test/test_gold300.json'),
        ('data_/train/train.json', 'train/train.json'),
        ('data_/valid/valid.json', 'valid/valid.json')
    ]
    """
    datasets = [
        ('data_/test/test_cleaned.json', 'test/test_cleaned.json'),
        ('data_/test/test_gold300_cleaned.json', 'test/test_gold300_cleaned.json'),
        ('data_/train/train_cleaned.json', 'train/train_cleaned.json'),
        ('data_/valid/valid_cleaned.json', 'valid/valid_cleaned.json')
    ]
    """
    for filepath, name in datasets:
        try:
            analyze_dataset(filepath, name)
        except FileNotFoundError:
            print(f"\n{'='*60}")
            print(f"文件不存在: {filepath}")
            print(f"{'='*60}")
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"处理文件 {filepath} 时出错: {str(e)}")
            print(f"{'='*60}")

if __name__ == "__main__":
    visualize_data()
    print(f"\n{'='*60}")
    print("数据可视化完成！")
    print(f"{'='*60}")