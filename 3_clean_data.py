import json
import re
import os
from typing import List, Dict, Set

class ASTECleaner:
    def __init__(self):
        # 黑名单：不合理的方面词（支持后续扩展）
        self.blacklist = {
            "地表", "不一样", "整体", "全部", "一切"
            # 黑名单继续扩展
        }

    def add_to_blacklist(self, items: List[str]):
        """向黑名单添加新项目"""
        self.blacklist.update(set(items))

    def remove_from_blacklist(self, items: List[str]):
        """从黑名单移除项目"""
        for item in items:
            self.blacklist.discard(item)

    def extract_context_from_question(self, question: str) -> str:
        """从question字段中提取context"""
        match = re.search(r'context:\s*(.*)', question)
        if match:
            return match.group(1).strip()
        return ""

    def is_valid_context(self, sample: Dict, min_length: int = 5) -> bool:
        """
        检查context是否有效（从question中提取）

        Args:
            sample: 样本字典
            min_length: 最小长度阈值
        """
        context = self.extract_context_from_question(sample.get('question', ''))
        if not context or len(context.strip()) < min_length:
            return False
        return True

    def is_valid_answer(self, sample: Dict) -> bool:
        """
        检查answer是否有效

        Args:
            sample: 样本字典
        """
        answer = sample.get('answer', '')
        if not answer or answer.strip() == "":
            return False
        return True

    def contains_blacklist_aspect(self, sample: Dict) -> bool:
        """
        检查answer中是否包含黑名单中的方面词

        Args:
            sample: 样本字典
        """
        answer = sample.get('answer', '')
        if not answer:
            return False

        # 提取方面词
        match = re.search(r'(.+?)。', answer)
        if match:
            aspects = [a.strip() for a in match.group(1).split('、') if a.strip()]
            for aspect in aspects:
                if aspect in self.blacklist:
                    return True
        return False

    def clean_sample(self, sample: Dict) -> bool:
        """
        检查单个样本是否需要清洗
        sample: 单个样本字典
        True: 保留该样本
        False: 删除该样本
        """
        # 1. 删除context过短的样本（从question中提取context）
        if not self.is_valid_context(sample):
            return False

        # 2. 删除answer为空的样本
        if not self.is_valid_answer(sample):
            return False

        # 3. 删除包含黑名单方面词的样本
        if self.contains_blacklist_aspect(sample):
            return False

        return True

    def clean_dataset(self, input_file: str, output_file: str) -> Dict:
        """
        input_file: 输入文件路径
        output_file: 输出文件路径
        Returns:
        清洗统计信息
        """
        # 读取原始数据（JSONL格式）
        original_data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        original_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"跳过无效JSON行: {line[:50]}...")

        original_count = len(original_data)

        # 清洗数据
        cleaned_data = []
        removed_samples = []

        for i, sample in enumerate(original_data):
            if self.clean_sample(sample):
                cleaned_data.append(sample)
            else:
                removed_samples.append((i, sample))

        cleaned_count = len(cleaned_data)
        removed_count = len(removed_samples)

        # 保存清洗后的数据
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 返回统计信息
        stats = {
            'original_count': original_count,
            'cleaned_count': cleaned_count,
            'removed_count': removed_count,
            'removal_rate': removed_count / original_count if original_count > 0 else 0,
            'removed_samples': removed_samples
        }

        return stats

def batch_clean_data():
    """批量清洗数据集"""
    cleaner = ASTECleaner()

    # 定义需要清洗的数据集
    datasets = {
        'data_/train/train.json': 'data_/train/train_cleaned.json',
        'data_/valid/valid.json': 'data_/valid/valid_cleaned.json',
        'data_/test/test.json': 'data_/test/test_cleaned.json',
        'data_/test/test_gold300.json': 'data_/test/test_gold300_cleaned.json'
    }

    all_stats = {}

    for input_path, output_path in datasets.items():
        if os.path.exists(input_path):
            print(f"清洗数据集: {input_path}")
            stats = cleaner.clean_dataset(input_path, output_path)

            print(f"  原始样本数: {stats['original_count']}")
            print(f"  清洗后样本数: {stats['cleaned_count']}")
            print(f"  删除样本数: {stats['removed_count']}")
            removal_rate = stats['removal_rate']
            print(f"  删除比例: {removal_rate:.2%}")
            print("-" * 50)

            all_stats[input_path] = stats
        else:
            print(f"文件不存在: {input_path}")

    # 输出总体统计
    print(f"\n{'='*50}")
    print("总体清洗统计")
    print(f"{'='*50}")
    total_original = sum(stats['original_count'] for stats in all_stats.values())
    total_cleaned = sum(stats['cleaned_count'] for stats in all_stats.values())
    total_removed = sum(stats['removed_count'] for stats in all_stats.values())

    print(f"总体原始样本数: {total_original}")
    print(f"总体清洗后样本数: {total_cleaned}")
    print(f"总体删除样本数: {total_removed}")

    if total_original > 0:
        total_removal_rate = total_removed / total_original
        print(f"总体删除比例: {total_removal_rate:.2%}")
    else:
        print("总体删除比例: 0.00%")

    return all_stats

if __name__ == "__main__":
    # 执行批量清洗
    stats = batch_clean_data()
    print(f"\n{'='*50}")
    print("数据清洗完成！")
    print(f"{'='*50}")

    # 添加新的黑名单项目
    cleaner = ASTECleaner()
    cleaner.add_to_blacklist(["新的黑名单项目", "另一个黑名单项目"])
    print("黑名单已更新")