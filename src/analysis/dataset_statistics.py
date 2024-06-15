import os 
import json 
from tqdm import tqdm 

user_oriented_path = "/largespace/tydata/code_optimization/cpp/dataset/by_user/train_out_pair_improvement10_tag_description.json"
problem_oriented_path = "/largespace/tydata/code_optimization/cpp/dataset/by_problem/train_out_pair_improvement90_code.json"

with open(user_oriented_path, 'r') as user_reader:
    user_dataset = json.load(user_reader)
print(f"length of user dataset = {len(user_dataset)}")

with open(problem_oriented_path, 'r') as problem_reader:
    problem_dataset = json.load(problem_reader)
print(f"length of problem dataset = {len(problem_dataset)}")

problem_id_set_user = set()
for user_item in tqdm(user_dataset):
    problem_id = user_item["problem_id"]
    problem_id_set_user.add(problem_id)

print(f"length of user dataset problem id = {len(problem_id_set_user)}")

problem_id_set_problem = set()
for problem_item in tqdm(problem_dataset):
    problem_id =  problem_item["problem_id"]
    problem_id_set_problem.add(problem_id)

print(f"length of problem dataset problem id = {len(problem_id_set_problem)}")


test_path = "/largespace/tydata/code_optimization/cpp/dataset/by_user/test_out_pair_in_original.json"
with open(test_path, 'r') as test_reader:
    test_dataset = json.load(test_reader)
print(f"test dataset length = {len(test_dataset)}")

problem_id_set_test = set()
for test_item in tqdm(test_dataset):
    problem_id = test_item["problem_id"]
    if problem_id not in problem_id_set_test:
        problem_id_set_test.add(problem_id)
print(f"length of test dataset problem id = {len(problem_id_set_test)}")


val_path = "/home/tongye/code_generation/pie-perf/data/cpp_original/val.jsonl"
val_dataset = []
with open(val_path, 'r') as val_reader:
    for line in val_reader:
        val_dataset.append(json.loads(line))

print(f"val dataset = {len(val_dataset)}")