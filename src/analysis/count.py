import os 
import json 
from tqdm import tqdm 

user_oriented_path = "/largespace/tydata/code_optimization/cpp/dataset/by_user/train_out_pair_improvement10_tag_description.json"
problem_oriented_path = "/largespace/tydata/code_optimization/cpp/dataset/by_problem/train_out_pair_improvement90_code.json"

with open(user_oriented_path,'r') as user_reader, \
    open(problem_oriented_path, 'r') as problem_reader:
    user_dataset = json.load(user_reader)
    problem_dataset = json.load(problem_reader)

print(f"length of user dataset = {len(user_dataset)}")
print(f"length of problem dataset = {len(problem_dataset)}")

with open("same_identifier.txt", 'r') as f:
    lines = f.readlines()
lines = [line.rstrip('\n') for line in lines]

final_count = 0
with open("small_same_identifier.txt", 'w') as f:
    for problem_item in tqdm(problem_dataset):
        slow_identifer = problem_item["slow_identifier"]
        if slow_identifer in lines:
            slow_code = problem_item["slow_code"]
            fast_code = problem_item["fast_code"]
            slow_count = slow_code.count("\n\n")
            fast_count = fast_code.count("\n\n")
            if slow_count < 30 and fast_count < 30:
                f.write(f"{slow_identifer}\n")
                final_count += 1

print(f"final count = {final_count}")
    