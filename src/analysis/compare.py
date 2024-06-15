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


user_slow_identifier_set = set()
for user_item in tqdm(user_dataset):
    problem_id = user_item["problem_id"]
    user_id = user_item["user_id"]
    slow_submission_id = user_item["slow_submission_id"]
    fast_submission_id = user_item["fast_submission_id"]

    user_slow_identifier = f"{problem_id}_{user_id}_{slow_submission_id}"
    user_fast_identifier = f"{problem_id}_{user_id}_{fast_submission_id}"

    if user_slow_identifier not in user_slow_identifier_set:
        user_slow_identifier_set.add(user_slow_identifier)

print(f"length of user slow identifier set = {len(user_slow_identifier_set)}")

problem_slow_identifier_list = []
for problem_item in tqdm(problem_dataset):
    problem_id = problem_item["problem_id"]
    problem_slow_identifier = problem_item["slow_identifier"]
    problem_fast_identifier = problem_item["fast_identifier"]

    problem_slow_user_id = problem_slow_identifier.split("_")[1]
    problem_slow_submission_id = problem_slow_identifier.split("_")[2]
    problem_fast_user_id = problem_fast_identifier.split("_")[1]
    problem_fast_submission_id = problem_fast_identifier.split("_")[2]

    problem_slow_identifier_list.append(problem_slow_identifier)

print(f"length of problem slow identifier list = {len(problem_slow_identifier_list)}")


with open("same_identifier.txt", 'w') as f:
    count = 0
    for item in problem_slow_identifier_list:
        if item in user_slow_identifier_set:
            count += 1
            f.write(f"{item}\n")

print(f"count = {count}")