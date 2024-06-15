import os 
import json 
from tqdm import tqdm 

dataset_path = "/largespace/tydata/code_optimization/cpp/dataset/benchmark_gem5_testcases3/train_out_times.json"

def tag_solutions(solutions):
    """
    solutions:{
        "identifier1": time1,
        "identifier2": time2,
        ...
        "identifiern": timen
    }
    """
    sorted_solutions = sorted(solutions, key=solutions.get, reverse=False)

    tags = {}
    num_solutions = len(sorted_solutions)
    tag_size = num_solutions // 10
    tag_index = 10

    if num_solutions >= 10:
        for i in range(0, num_solutions, tag_size):
            tag = f"{tag_index}/10"
            for solution_identifier in sorted_solutions[i:i+tag_size]:
                tags[solution_identifier] = tag
            tag_index -= 1

            if tag_index == 0:
                tag_index = 1
    else: # num_solutions < 10
        for idx, solution_identifier in enumerate(sorted_solutions):
            tag = f"{num_solutions-idx}/10"
            tags[solution_identifier] = tag

    return tags


def get():
    with open(dataset_path, 'r') as reader:
        dataset = json.load(reader)

    print(f"There are {len(dataset)} problems in dataset.")

    all_tags = {}
    count = 0

    for problem_id, value in tqdm(dataset.items()):
    # for each problem_id 
        solutions = {}  # all solutions for this problem
        for user_id, submissions in value.items():
            for submission_id, submission_time in submissions.items():
                count += 1
                identifier = f"{problem_id}_{user_id}_{submission_id}"
                solutions[identifier] = submission_time
        
        tags = tag_solutions(solutions)
        all_tags.update(tags)

    print(f"all count = {count}")
    print(f"all solutions length = {len(all_tags)}")

    with open("/largespace/tydata/code_optimization/cpp/dataset/benchmark_gem5_testcases3/train_out_tags.json", 'w') as writer:
        json.dump(all_tags, writer, indent=4)
    return None 


def make_tag():
    dataset_path = "/largespace/tydata/code_optimization/cpp/dataset/by_user/train_out_pair_improvement10_description.json"
    tag_path = "/largespace/tydata/code_optimization/cpp/dataset/benchmark_gem5_testcases3/train_out_tags.json"

    with open(dataset_path, 'r') as reader:
        train_dataset = json.load(reader)
    print(f"There are {len(train_dataset)} items in train improvement10.")

    with open(tag_path, 'r') as reader:
        tags = json.load(reader)
    print(f"There are {len(tags)} items in tags.")

    for data in tqdm(train_dataset):
        problem_id = data["problem_id"]
        user_id = data["user_id"]
        slow_submission_id = data["slow_submission_id"]
        fast_submission_id = data["fast_submission_id"]

        identifier_slow = f"{problem_id}_{user_id}_{slow_submission_id}"
        identifier_fast = f"{problem_id}_{user_id}_{fast_submission_id}"

        if identifier_slow in tags:
            slow_code_tag = tags[identifier_slow]
        else:
            assert False

        if identifier_fast in tags:
            fast_code_tag = tags[identifier_fast]
        else:
            assert False
        
        data["slow_code_tag"] = slow_code_tag
        data["fast_code_tag"] = fast_code_tag

    with open("/largespace/tydata/code_optimization/cpp/dataset/by_user/train_out_pair_improvement10_tag_description.json", 'w') as writer:
        json.dump(train_dataset, writer, indent=4)


if __name__ == "__main__":
    # get()

    make_tag()