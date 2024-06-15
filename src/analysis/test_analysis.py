import os 
import json 
from tqdm import tqdm 

fast_path = "/largespace/tydata/code_optimization/cpp/saved_models/models_merge/deepseekcoder-33b-0523/merge_moe_user_problem_linear3_0523/fast_solutions_and_correct_10.json"

with open(fast_path, 'r') as reader:
    fast_result = json.load(reader)

print(type(fast_result))
print(f"fast length = {len(fast_result)}")


test_path = "/largespace/tydata/code_optimization/cpp/dataset/by_user/test_out_pair_in_original_description.json"
with open(test_path, 'r') as test_reader:
    test_dataset = json.load(test_reader)
print(f"test length = {len(test_dataset)}")


human_fast = dict()
human_speedup = []
for test_item in tqdm(test_dataset):
    problem_id = test_item["problem_id"]
    slow_item = test_item["slow_time"]
    fast_time = test_item["fast_time"]
    human_speedup.append(slow_item/fast_time)
    if problem_id not in human_fast:
        human_fast[problem_id] = fast_time
    else:
        last_fast_time = human_fast[problem_id]
        if fast_time < last_fast_time:
            human_fast[problem_id] = fast_time

print(len(human_fast))
print(f"average human speedup = {sum(human_speedup)/len(human_speedup)}")

count = 0
llm_fast = dict()
for fast_item_key, fast_item_value in tqdm(fast_result.items()):
    problem_id = fast_item_key.split("_")[0]
    average_time = list(fast_item_value.values())[0]['average_time']

    # if problem_id in human_fast:
    #     if average_time < human_fast[problem_id]:
    #         count += 1

    if problem_id not in llm_fast:
        llm_fast[problem_id] = average_time
    else:
        last_fast_time = llm_fast[problem_id]
        if average_time < last_fast_time:
            llm_fast[problem_id] = average_time

print(f"llm fast length = {len(llm_fast)}")

hit = 0
for llm_fast_item_key, llm_fast_item_value in llm_fast.items():
    if llm_fast_item_key in human_fast:
        if llm_fast_item_value < human_fast[llm_fast_item_key]:
            hit += 1

print(f"hit = {hit}")
