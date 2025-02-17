import os 
import pandas as pd
import logging 
from itertools import combinations
from tqdm import tqdm 

logging.basicConfig(filename='problem_oriented.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def combine_codes(row):
    """
    将每个row处理成统一的格式，并尽可能保留所有原始信息
    """
    src_part = {
        "id": row["src_id"],
        "agg_runtime": row["src_agg_runtime"],
        "problem_id": row["problem_id"], 
        "code": row["src_code"],
        "status": row["src_status"],
        "reward_updated": row["src_reward_updated"],
        "reward_updated_pct_bin": row["src_reward_updated_pct_bin"]

    }

    tgt_part = {
        "id": row["tgt_id"],
        "agg_runtime": row["tgt_agg_runtime"],
        "problem_id": row["problem_id"],
        "code": row["tgt_code"],
        "status": row["tgt_status"],
        "reward_updated": row["target_reward_updated"],
        "reward_updated_pct_bin": row["target_reward_updated_pct_bin"]
    }
    return [src_part, tgt_part]

def problem_oriented(path, speedup_save_threshold):
    """
    得到 问题导向视角下的 快慢pair
     ['src_id', 'tgt_id', 'src_agg_runtime', 'tgt_agg_runtime', 
     'problem_id', 'speedup', 'src_code', 'tgt_code', 'fastest_agg_runtime', 
     'src_status', 'tgt_status', 'fastest_agg_runtime_updated', 
     'target_reward_updated', 'src_reward_updated', 
     'target_reward_updated_pct_bin', 'src_reward_updated_pct_bin']

     { "problem_id":{"src_id1": {"code": xxx, "agg_runtime": xxx, "pct_bin": xxx} 
                     }
     }
    """
    df = pd.read_json(path, lines=True, orient="records")
    logging.info(f"dataset shape = {df.shape}")
    logging.info(f"dataset columns = {list(df.columns)}")

    grouped_df = df.groupby("problem_id")
    print(f"length of group = {len(grouped_df)}")
    
    problem_oriented_pairs = []
    # [pairs1, paris2, ...]
    for name, group in tqdm(grouped_df):
        # print(f'Group: {name}')
        # print(f"lenght = {len(group)}")

        code_with_runtime = []
        for index, row in group.iterrows():
            code_with_runtime.extend(combine_codes(row))
    
        # 去重，避免同一个提交的代码出现多次
        code_with_runtime = [dict(t) for t in {tuple(d.items()) for d in code_with_runtime}]

        # 按照运行时间进行排序 （从小到大，即从快到慢的排序）
        sorted_codes = sorted(code_with_runtime, key=lambda x: x['agg_runtime'])


        for fast, slow in combinations(sorted_codes, 2):
            if fast["agg_runtime"] < slow["agg_runtime"] and (slow["agg_runtime"]/fast["agg_runtime"] > speedup_save_threshold):
                problem_oriented_pairs.append({
                    "src_id": slow["id"],
                    "tgt_id": fast["id"],
                    "src_agg_runtime": slow["agg_runtime"],
                    "tgt_agg_runtime": fast["agg_runtime"],
                    "problem_id": fast["problem_id"],
                    "speedup": slow["agg_runtime"] / fast["agg_runtime"],
                    "src_code": slow["code"],
                    "tgt_code": fast["code"],
                    "src_status": slow["status"],
                    "tgt_status": fast["status"],
                    "target_reward_updated": fast["reward_updated"],
                    "src_reward_updated": slow["reward_updated"],
                    "target_reward_updated_pct_bin": fast["reward_updated_pct_bin"],
                    "src_reward_updated_pct_bin": slow["reward_updated_pct_bin"]
                })
        
    problem_oriented_df = pd.DataFrame(problem_oriented_pairs)
    print(f"lenght of final problem oriented dataset = {len(problem_oriented_df)}")

    return problem_oriented_df


def problem_oriented_same_count(path, percent=1.0):
    """
    统计原始数据集每个问题下的数量;
    然后再问题导向的数据集下抽取每个问题同样数据量的最快样本
    或者是 特定比例的 最快样本， 由percent决定
    """
    df = pd.read_json(path, lines=True, orient="records")
    grouped_df = df.groupby("problem_id")
    print(f"length of group = {len(grouped_df)}")

    whole_problem_oriented_certain_percent_count = []
    for name, group in tqdm(grouped_df):
        group_count = len(group)
        certain_percent_group_count = int(group_count * percent)

        code_with_runtime = []
        for index, row in group.iterrows():
            code_with_runtime.extend(combine_codes(row))

        # 去重，避免同一个提交的代码出现多次
        code_with_runtime = [dict(t) for t in {tuple(d.items()) for d in code_with_runtime}]

        # 按照运行时间进行排序 （从小到大，即从快到慢的排序）
        sorted_codes = sorted(code_with_runtime, key=lambda x: x['agg_runtime'])

        problem_oriented_pairs = []
        for fast, slow in combinations(sorted_codes, 2):
            if fast["agg_runtime"] < slow["agg_runtime"]:
                problem_oriented_pairs.append({
                    "src_id": slow["id"],
                    "tgt_id": fast["id"],
                    "src_agg_runtime": slow["agg_runtime"],
                    "tgt_agg_runtime": fast["agg_runtime"],
                    "problem_id": fast["problem_id"],
                    "speedup": slow["agg_runtime"] / fast["agg_runtime"],
                    "src_code": slow["code"],
                    "tgt_code": fast["code"],
                    "src_status": slow["status"],
                    "tgt_status": fast["status"],
                    "target_reward_updated": fast["reward_updated"],
                    "src_reward_updated": slow["reward_updated"],
                    "target_reward_updated_pct_bin": fast["reward_updated_pct_bin"],
                    "src_reward_updated_pct_bin": slow["reward_updated_pct_bin"]
                })

        problem_oriented_pairs.sort(key=lambda x: x['speedup'], reverse=True)

        # FIXME 特定数量并不一定是speedup最快的，这样可能会导致target重复性比较大，可以再想想
        problem_oriented_certain_percent_count_pairs = []
        select_speedup_max = False
        select_overlap = True

        if select_speedup_max:
            problem_oriented_certain_percent_count_pairs = problem_oriented_pairs[:certain_percent_group_count]
        elif select_overlap:
            used_tgt_codes = {}
            for current_pair in problem_oriented_pairs:
                tgt_code = current_pair['tgt_code']

                if used_tgt_codes.get(tgt_code, 0) < 100:
                    used_tgt_codes[tgt_code] = used_tgt_codes.get(tgt_code, 0) + 1
                    problem_oriented_certain_percent_count_pairs.append(current_pair)
                
                if len(problem_oriented_certain_percent_count_pairs)  >= certain_percent_group_count:
                    break

        whole_problem_oriented_certain_percent_count.extend(problem_oriented_certain_percent_count_pairs)

    print(f"length of whole problem oriented same count = {len(whole_problem_oriented_certain_percent_count)}")
    whole_problem_oriented_certain_percent_count_df = pd.DataFrame(whole_problem_oriented_certain_percent_count)

    return whole_problem_oriented_certain_percent_count_df


if __name__ == "__main__":
    dir_path = "/ainative/codefuse/user/448207/pie_dataset/pie"
    output_path = "/ainative/codefuse/user/448207/pie_dataset/remake"
    dataset_name = 'train'
    path = os.path.join(dir_path, f"{dataset_name}.jsonl")

    ## Step 1: 得到问题导向的数据集（指定加速比情况下的）
    # speedup_save_threshold = 25.0
    # problem_oriented_df = problem_oriented(path, speedup_save_threshold)

    # problem_oriented_df.to_json(os.path.join(output_path, f"train_problem_oriented_speedup_{speedup_save_threshold}.jsonl"), orient='records', lines=True)


    ## Step 2: 统计原始数据集每个问题下的数量，然后再问题导向的数据集下抽取同样数据的最快样本
    percent_now = 1.0
    whole_problem_oriented_certain_percent_count_df = problem_oriented_same_count(path, percent=percent_now)
    whole_problem_oriented_certain_percent_count_df.to_json(os.path.join(output_path, f"train_problem_oriented_{int(percent_now*100)}_percent_count_0121.jsonl"), orient="records", lines=True)
