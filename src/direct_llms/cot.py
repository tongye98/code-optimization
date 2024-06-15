import os 
import json 
from tqdm import tqdm 

test_path = "/largespace/tydata/code_optimization/cpp/dataset/by_user/test_out_pair_in_original_description.json"
stray = "/largespace/tydata/code_optimization/cpp/baselines/cot/deepseek-coder-33b-instruct-strategy.json"

with open(test_path, 'r') as reader:
    test_dataset = json.load(reader)

print(f"length of test dataset = {len(test_dataset)}")

with open(stray, 'r') as reader2:
    stary_dataset = json.load(reader2)
print(f"length of stray dataset = {len(stary_dataset)}")

new_dataset = []
for item1, item2 in zip(test_dataset, stary_dataset):
    detail_stary = item2["maybe_faster"][0]
    item1["strategy"] = detail_stary

    new_dataset.append(item1)

with open("/largespace/tydata/code_optimization/cpp/dataset/by_user/test_out_pair_in_original_description_strategy.json", 'w') as writer:
    json.dump(new_dataset, writer, indent=4)