import os 
import pandas as pd
import logging 
from tqdm import tqdm 

logging.basicConfig(filename='uer_oriented.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def user_oriented(path):
    """
    用户导向视角下的 快慢pair
     ['src_id', 'tgt_id', 'src_agg_runtime', 'tgt_agg_runtime', 
     'problem_id', 'speedup', 'src_code', 'tgt_code', 'fastest_agg_runtime', 
     'src_status', 'tgt_status', 'fastest_agg_runtime_updated', 
     'target_reward_updated', 'src_reward_updated', 
     'target_reward_updated_pct_bin', 'src_reward_updated_pct_bin']

    """
    df = pd.read_json(path, lines=True, orient="records")
    logging.info(f"dataset shape = {df.shape}")
    logging.info(f"dataset columns = {list(df.columns)}")

    average_speedup = df['speedup'].mean()
    print(f"Average speedup = {average_speedup}")


def user_oriented_certain_percent_count(path, percent=1.0):
    """
    得到每个问题下的数量，并同样得到特定比例的快慢对
    """
    df = pd.read_json(path, lines=True, orient="records")
    print(f"length of original = {len(df)}")
    print(f"speedup original = {df['speedup'].mean()}")
    grouped_df = df.groupby("problem_id")
    print(f"length of group = {len(grouped_df)}")

    whole_user_oriented_certain_percent_count = []
    for name, group in tqdm(grouped_df):
        group_count = len(group)
        certain_percent_group_count = int(group_count * percent)

        # 按照speedup从大到小排序
        sorted_samples = group.sort_values(by="speedup", ascending=False)

        # 获取特定比例的样本
        certain_percent_samples = sorted_samples.head(certain_percent_group_count)

        # 打印前几行作为样本
        # certain_percent_samples.to_csv("test.csv")
        # assert False

        whole_user_oriented_certain_percent_count.append(certain_percent_samples)

    # 将所有分组结果合并
    result_df = pd.concat(whole_user_oriented_certain_percent_count)

    print(f"length of certain = {len(result_df)}")
    print(f"speedup certain percent = {result_df['speedup'].mean()}")
    return result_df



if __name__ == "__main__":

    dir_path = f"/ainative/codefuse/448207/pie_dataset/original_dataset"
    output_path = f"/ainative/codefuse/448207/pie_dataset/zour_dataset"
    dataset_name = 'train'

    path = os.path.join(dir_path, f"{dataset_name}.jsonl")
    # user_oriented(path)
    result_df = user_oriented_certain_percent_count(path, percent=0.1)
    result_df.to_json(os.path.join(output_path, f"original_train_10_percent_count.jsonl"), orient="records", lines=True)