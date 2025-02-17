import json
import pandas as pd 
import os 


def fillna(path, output_path):
    """
    空行填充，并保存回jsonl文件格式
    """
    df = pd.read_json(path, lines=True, orient="records")
    df.fillna(value={
        'fastest_agg_runtime': 0,
        'fastest_agg_runtime_updated': 0,
        'target_reward_updated': 0,
        'src_reward_updated': 0,
        'user_id': 'unknown'
    }, inplace=True)
    print(df.isnull().sum())

    df.to_json(output_path, orient='records', lines=True, force_ascii=False)


def analysis_target_code_same(path):
    """
    分析数据集中 target code相似性情况，数据集中有多少target code是一样
    """
    df = pd.read_json(path, lines=True, orient="records")

    # 统计每个 tgt_code 出现的次数
    tgt_code_counts = df['tgt_code'].value_counts()
    
    # 输出出现次数大于1的 tgt_code（即重复的 target code）
    duplicate_codes = tgt_code_counts[tgt_code_counts > 1]
    
    sorted_counts = duplicate_codes.sort_values(ascending=False).tolist()
    
    print(sorted_counts)
    print(f"length = {len(sorted_counts)}")

    # 返回重复代码的数量和对应的 tgt_code
    return duplicate_codes

def analysis_problem_view_speedup(path):
    df = pd.read_json(path, lines=True, orient="records")

    avg_speedup = df['speedup'].mean()
    print(f"avg_speedup = {avg_speedup}")

if __name__ == "__main__":
    dir_path = "/ainative/codefuse/user/448207/pie_dataset/pie/"

    # path = os.path.join(dir_path, f"train_with_synthetic.jsonl")
    # output_path = os.path.join(dir_path, f"train_with_synthetic_v1.jsonl")
    # fillna(path, output_path)   

    # train_problem_oriented_same_count_backup1.jsonl train_problem_oriented_100_percent_count_0115.jsonl
    analysis_target_code_same(os.path.join(dir_path, "train.jsonl"))

    analysis_problem_view_speedup(os.path.join(dir_path, "train.jsonl"))


