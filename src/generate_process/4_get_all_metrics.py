import os 
import json 


BASE = "/largespace/tydata/code_optimization/cpp/saved_models/continue/pie-gem5-by-problem-codellama-34b_continue_sft_0615/full_model"
generated_model_id = ""
generate_project = ""
test_problem_nums = N = 1422
IDX = 8

with open(os.path.join(BASE, generated_model_id, generate_project, f"merge_{IDX}_generates.json"), 'r') as reader:
    original = json.load(reader)

original_slow = {}
for each_original in original:
    problem_id = each_original["problem_id"]
    user_id = each_original["user_id"]
    slow_submission_id = each_original["slow_submission_id"]
    fast_submission_id = each_original["fast_submission_id"]
    slow_time = each_original["slow_time"]
    identifer = f"{problem_id}_{user_id}_{slow_submission_id}_{fast_submission_id}"
    original_slow[identifer] = slow_time


with open(os.path.join(BASE, generated_model_id, generate_project, f"fast_solutions_and_correct_{IDX}.json"), 'r') as reader:
    solutions = json.load(reader)


T = len(solutions)
print(f"There are {N} original test problems.")
print(f"There are (T) {T} solutions can pass compile, pass benchmark and have right solution.")


def relative_improve(slow:float, fast:float):
    return round((slow - fast) / slow, 4)

Q = even_slower_count = 0
R = speedup_small_ten_percent_count = 0
P = speedup_surpass_ten_percent_count = 0

speedups = []
for problem_identifier, solution in solutions.items():
    solution_name = list(solution.keys())[0]
    average_itme = solution[solution_name]["average_time"]
    slow_time = original_slow[problem_identifier]

    if average_itme > slow_time:
        even_slower_count += 1
    elif relative_improve(slow_time, average_itme) < 0.1:
        speedup_small_ten_percent_count += 1
        speedups.append(round(slow_time/average_itme, 4))
    else:
        speedup_surpass_ten_percent_count += 1
        speedups.append(round(slow_time/average_itme, 4))
    
print(f"even slower count (Q) = {even_slower_count}")
print(f"speedup small ten percent count (R) = {speedup_small_ten_percent_count}")
print(f"speedup surpass ten percent count (P) = {speedup_surpass_ten_percent_count}")

final_speedup = round((sum(speedups) + even_slower_count + N - T) / N, 4)
print(f"speedup = {final_speedup}")
print(f"%Opt = {round(speedup_surpass_ten_percent_count/N, 4)}")
print(f"%Correct = {round(T/N, 4)}")

print(f"Only consider speedup = {(sum(speedups)/len(speedups))}")