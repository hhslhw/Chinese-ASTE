import json
import re
import os

# ================= 配置 =================
# 输入文件列表
INPUT_FILES = [
    "output_Qwen_4B_lora_v2.jsonl",
    #"output_Qwen_4B_fewshot.jsonl",
    #"output_Gemma_4B_zeroshot.jsonl"
    #"output_Qwen_1.7B_zeroshotx.jsonl"
]
RESULTS_DIR = "results"


# ================= 1. 数据解析模块 =================
def parse_string_to_triples(text):
    triples = set()
    if not text: return triples
    try:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        segments = text.split('。')
        for seg in segments:
            if '：' not in seg: continue
            parts = seg.split('：')
            if len(parts) < 2: continue
            aspect = parts[0].strip()
            details = parts[1]
            ops = re.split(r'[、，,]', details)
            for op in ops:
                match = re.search(r'(.*?)[\(（](正面|负面|中性)[\)）]', op)
                if match:
                    opinion = match.group(1).strip()
                    sentiment = match.group(2).strip()
                    if aspect and opinion:
                        triples.add((aspect, opinion, sentiment))
    except Exception as e:
        print(f"解析文本格式出错: {e}")
    return triples


def parse_prediction(pred_content):
    if not pred_content: return []
    try:
        clean_json = pred_content.replace("```json", "").replace("```", "").strip()
        clean_json = re.sub(r'<think>.*?</think>', '', clean_json, flags=re.DOTALL).strip()
        data = json.loads(clean_json)
        triples = set()
        if isinstance(data, list):
            for item in data:
                a = item.get("aspect", "").strip()
                o = item.get("opinion", "").strip()
                s = item.get("sentiment", "").strip()
                if a and o and s: triples.add((a, o, s))
            if triples: return list(triples)
    except:
        pass
    return list(parse_string_to_triples(pred_content))


def parse_ground_truth(gt_str):
    return list(parse_string_to_triples(gt_str))


# ================= 2. 核心匹配逻辑 =================
def check_match(pred_item, gt_item, mode='strict'):
    p_a, p_o, p_s = pred_item
    g_a, g_o, g_s = gt_item
    if p_s != g_s: return False
    if mode == 'strict':
        return (p_a == g_a) and (p_o == g_o)
    elif mode == 'soft':
        a_match = (p_a in g_a) or (g_a in p_a)
        o_match = (p_o in g_o) or (g_o in p_o)
        return a_match and o_match
    return False


def calculate_sample_score(pred_list, gt_list, mode='strict'):
    correct_count = 0
    if mode == 'strict':
        p_set = set(pred_list)
        g_set = set(gt_list)
        correct_count = len(p_set.intersection(g_set))
        union_len = len(p_set.union(g_set))
    else:
        matched_gt_indices = set()
        for p in pred_list:
            for idx, g in enumerate(gt_list):
                if idx not in matched_gt_indices:
                    if check_match(p, g, mode='soft'):
                        correct_count += 1
                        matched_gt_indices.add(idx)
                        break
        union_len = len(pred_list) + len(gt_list) - correct_count

    accuracy = correct_count / union_len if union_len > 0 else 0
    precision = correct_count / len(pred_list) if len(pred_list) > 0 else 0
    recall = correct_count / len(gt_list) if len(gt_list) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1


# ================= 3. 主流程 =================
def evaluate_file(filepath):
    print(f"正在评估: {filepath} ...")
    results = {
        "strict": {"acc": [], "p": [], "r": [], "f1": []},
        "soft": {"acc": [], "p": [], "r": [], "f1": []}
    }
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            gt = parse_ground_truth(item.get('ground_truth_str', ''))
            pred = parse_prediction(item.get('model_prediction', ''))
            sa, sp, sr, sf = calculate_sample_score(pred, gt, mode='strict')
            results['strict']['acc'].append(sa)
            results['strict']['p'].append(sp)
            results['strict']['r'].append(sr)
            results['strict']['f1'].append(sf)
            oa, op, or_, of = calculate_sample_score(pred, gt, mode='soft')
            results['soft']['acc'].append(oa)
            results['soft']['p'].append(op)
            results['soft']['r'].append(or_)
            results['soft']['f1'].append(of)

    final_scores = {}
    for mode in ['strict', 'soft']:
        n = len(results[mode]['acc'])
        final_scores[mode] = {
            "accuracy": sum(results[mode]['acc']) / n if n > 0 else 0,
            "precision": sum(results[mode]['p']) / n if n > 0 else 0,
            "recall": sum(results[mode]['r']) / n if n > 0 else 0,
            "f1": sum(results[mode]['f1']) / n if n > 0 else 0
        }
    return final_scores


def save_results(filename, scores):
    """
    保存逻辑：
    输入: output_Qwen_1.7B_zeroshot.jsonl
    输出: eval_Qwen_1.7B_zeroshot.json
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # 1. 获取纯文件名 (例如: output_Qwen_1.7B_zeroshot)
    base_name = os.path.splitext(os.path.basename(filename))[0]

    # 2. 替换前缀：如果以 output_ 开头，则换成 eval_
    if base_name.startswith("output_"):
        new_name = base_name.replace("output_", "eval_", 1)
    else:
        new_name = "eval_" + base_name

    out_path = os.path.join(RESULTS_DIR, new_name + ".json")

    output_data = {
        "model_file": filename,
        "strict": scores['strict'],
        "soft": scores['soft']
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"结果已保存至: {out_path}")


def main():
    for f_path in INPUT_FILES:
        if os.path.exists(f_path):
            scores = evaluate_file(f_path)
            save_results(f_path, scores)
            print(f"--- {f_path} 评估完成 ---")
        else:
            print(f"警告: 找不到文件 {f_path}")


if __name__ == "__main__":
    main()
