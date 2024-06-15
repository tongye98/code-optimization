import os 
import json
import re
from tqdm import tqdm 

test_out_times_path = "/largespace/tydata/code_optimization/cpp/dataset/benchmark_gem5_testcases3/train_out_times.json"
with open(test_out_times_path, 'r') as f:
    datasets = json.load(f)

datasets = datasets
print(type(datasets))
print(len(datasets))

def make():
    def relative_improvement(slow:float, fast:float):
        return round((slow - fast) / slow, 4)

    def make_pair(sorted_problem_all_solutions):
        length = len(sorted_problem_all_solutions)
        pairs = []
        for i in range(length):
            for j in range(i+1, length):
                slow = sorted_problem_all_solutions[i]
                fast = sorted_problem_all_solutions[j]
                pair = (slow[0], fast[0])
                pairs.append(pair)
                
        return pairs

    all_items = []
    for problem_id, problem_solutions in tqdm(datasets.items()):
        problem_all_solutions = {}
        for user_id, user_solutions in problem_solutions.items():
            for solution_id, time in user_solutions.items():
                identifier = f"{problem_id}_{user_id}_{solution_id}"
                problem_all_solutions[identifier] = time

        # for all solutions for the problem
        sorted_problem_all_solutions = sorted(problem_all_solutions.items(), key=lambda x: x[1], reverse=True) # List
        pairs = make_pair(sorted_problem_all_solutions)
        
        for pair in pairs:
            slow_identifier = pair[0]
            fast_identifier = pair[1]
            slow_time = problem_all_solutions[slow_identifier]
            fast_time = problem_all_solutions[fast_identifier]
            improvement = relative_improvement(slow_time, fast_time)
            if improvement > 0.9:
                item = {
                    "problem_id": problem_id,
                    "slow_identifier": slow_identifier,
                    "fast_identifier": fast_identifier,
                    "slow_time": slow_time,
                    "fast_time": fast_time,
                    "improvement": improvement,
                }

                all_items.append(item)

    print(f"all pairs count = {len(all_items)}")
    # with open("/largespace/tydata/code_optimization/cpp/dataset/by_problem/train_out_pair_3items_improvement.json", 'w') as writer:
    #     json.dump(all_items, writer, indent=4)    


def make2():
    def relative_improvement(slow:float, fast:float):
        return round((slow - fast) / slow, 4)

    def make_pair_3item(sorted_problem_all_solutions):
        length = len(sorted_problem_all_solutions)
        pairs = []
        if length < 3:
            return pairs
        for i in range(length):
            for j in range(i+1, length):
                for k in range(j+1, length):
                    slow = sorted_problem_all_solutions[i]
                    current = sorted_problem_all_solutions[j]
                    fast = sorted_problem_all_solutions[k]
                    pair = (slow[0], current[0], fast[0])
                    pairs.append(pair)
                
        return pairs

    all_items = []
    for problem_id, problem_solutions in tqdm(datasets.items()):
        problem_all_solutions = {}
        for user_id, user_solutions in problem_solutions.items():
            for solution_id, time in user_solutions.items():
                identifier = f"{problem_id}_{user_id}_{solution_id}"
                problem_all_solutions[identifier] = time

        # for all solutions for the problem
        sorted_problem_all_solutions = sorted(problem_all_solutions.items(), key=lambda x: x[1], reverse=True) # List
        pairs = make_pair_3item(sorted_problem_all_solutions)
        
        for pair in pairs:
            slow_identifier = pair[0]
            current_identifier = pair[1]
            fast_identifier = pair[2]
            slow_time = problem_all_solutions[slow_identifier]
            current_time = problem_all_solutions[current_identifier]
            fast_time = problem_all_solutions[fast_identifier]
            improvement1 = relative_improvement(current_time, fast_time)
            improvement2 = relative_improvement(slow_time, current_time)
            if improvement1 > 0.9 and improvement2 > 0.5:
                item = {
                    "problem_id": problem_id,
                    "slow_identifier": slow_identifier,
                    "current_identifier": current_identifier,
                    "fast_identifier": fast_identifier,
                    "slow_time": slow_time,
                    "current_time": current_time,
                    "fast_time": fast_time,
                    "improvement1": improvement1,
                    "improvement2": improvement2
                }

                all_items.append(item)

    print(f"all pairs count = {len(all_items)}")
    with open("/largespace/tydata/code_optimization/cpp/dataset/by_problem/train_out_pair_3items_improvement.json", 'w') as writer:
        json.dump(all_items, writer, indent=4)  

def make3():
    def relative_improvement(slow:float, fast:float):
        return round((slow - fast) / slow, 4)

    def make_pair_3item(sorted_problem_all_solutions):
        length = len(sorted_problem_all_solutions)
        pairs = []
        if length < 3:
            return pairs
        for i in range(length):
            for j in range(i+1, length):
                for k in range(j+1, length):
                    slow = sorted_problem_all_solutions[i]
                    middle = sorted_problem_all_solutions[j]
                    fast = sorted_problem_all_solutions[k]
                    pair = (slow[0], middle[0], fast[0])
                    pairs.append(pair)
                
        return pairs

    all_items = []
    for problem_id, problem_solutions in tqdm(datasets.items()):
        problem_all_solutions = {}
        for user_id, user_solutions in problem_solutions.items():
            for solution_id, time in user_solutions.items():
                identifier = f"{problem_id}_{user_id}_{solution_id}"
                problem_all_solutions[identifier] = time

        # for all solutions for the problem
        sorted_problem_all_solutions = sorted(problem_all_solutions.items(), key=lambda x: x[1], reverse=True) # List
        pairs = make_pair_3item(sorted_problem_all_solutions)
        
        for pair in pairs:
            slow_identifier = pair[0]
            middle_identifier = pair[1]
            fast_identifier = pair[2]
            slow_time = problem_all_solutions[slow_identifier]
            middle_time = problem_all_solutions[middle_identifier]
            fast_time = problem_all_solutions[fast_identifier]
            improvement2middle = relative_improvement(slow_time, middle_time)  # slow -> middle 
            improvement2fast = relative_improvement(slow_time, fast_time)  # slow -> fast: > 0.9
            if improvement2fast > 0.9 and 0.2 > improvement2middle > 0.1:
                item = {
                    "problem_id": problem_id,
                    "slow_identifier": slow_identifier,
                    "middle_identifier": middle_identifier,
                    "fast_identifier": fast_identifier,
                    "slow_time": slow_time,
                    "middle_time": middle_time,
                    "fast_time": fast_time,
                    "improvement2middle": improvement2middle,
                    "improvement2fast": improvement2fast
                }

                all_items.append(item)

    print(f"all pairs count = {len(all_items)}")
    with open("/largespace/tydata/code_optimization/cpp/dataset/by_problem/train_out_pair_3items_improvement12.json", 'w') as writer:
        json.dump(all_items, writer, indent=4)  



def make_code():
    with open("/largespace/tydata/code_optimization/cpp/dataset/by_problem/train_out_pair_3items_improvement12_sm_sole.json", 'r') as reader:
        all_items_train = json.load(reader)

    print(f"all items count = {len(all_items_train)}")

    for item in tqdm(all_items_train):
        problem_id = item["problem_id"]

        slow_identifier = item["slow_identifier"]
        slow_parts = slow_identifier.split("_")
        slow_user_id = slow_parts[1]
        slow_submission_id = slow_parts[2]

        middle_identifier = item["middle_identifier"]
        middle_parts = middle_identifier.split("_")
        middle_user_id = middle_parts[1]
        middle_submission_id = middle_parts[2]


        fast_identifier = item["fast_identifier"]
        fast_parts = fast_identifier.split("_")
        fast_user_id = fast_parts[1]
        fast_submission_id = fast_parts[2]

        slow_cpp_file_path = os.path.join("/largespace/tydata/code_optimization/cpp/dataset/cpp_code/train", f"{problem_id}_{slow_submission_id}_{slow_user_id}.cpp")
        with open(slow_cpp_file_path, 'r') as f_slow:
            slow_cpp_content = f_slow.read()
        slow_code = slow_cpp_content

        current_code_file_path = os.path.join("/largespace/tydata/code_optimization/cpp/dataset/cpp_code/train", f"{problem_id}_{middle_submission_id}_{middle_user_id}.cpp")
        with open(current_code_file_path, 'r') as f_current:
            middle_cpp_content = f_current.read()
        middle_code = middle_cpp_content

        fast_cpp_file_path = os.path.join("/largespace/tydata/code_optimization/cpp/dataset/cpp_code/train", f"{problem_id}_{fast_submission_id}_{fast_user_id}.cpp")
        with open(fast_cpp_file_path, 'r') as f_fast:
            fast_cpp_content = f_fast.read()
        fast_code = fast_cpp_content

        item["slow_code"] = slow_code
        item["middle_code"] = middle_code
        item["fast_code"] = fast_code

    with open("/largespace/tydata/code_optimization/cpp/dataset/by_problem/train_out_pair_3items_improvement12_sm_sole_code.json", 'w') as writer:
        json.dump(all_items_train, writer, indent=4)


def prepare_for_rankloss():
    with open("/largespace/tydata/code_optimization/cpp/dataset/by_problem/train_out_pair_3items_improvement_code.json", 'r') as reader:
        all_items_train = json.load(reader)

    results = []
    for item in tqdm(all_items_train):
        current_code = item["current_code"]
        slow_code = item["slow_code"]
        fast_code = item["fast_code"]

        new_item = {
            "current_code": current_code,
            "code_solution": [fast_code, slow_code],
            "system": ""
        }

        results.append(new_item)

    with open("/largespace/tydata/code_optimization/cpp/dataset/by_problem/train_out_pair_3items_improvement_code_rankloss.json", 'w') as writer:
        json.dump(results, writer, indent=4)


if __name__ == "__main__":
    # prepare_for_rankloss()
    # make3()
    make_code()