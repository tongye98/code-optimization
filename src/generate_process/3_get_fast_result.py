import os 
import json 

BASE = "/largespace/tydata/code_optimization/cpp/saved_models/continue/pie-gem5-by-problem-codellama-34b_continue_sft_0615/full_model"
generated_model_id = ""
generate_project = ""
IDX = 8

with open(os.path.join(BASE, generated_model_id, generate_project, f"analysis_results_{IDX}.json"), 'r') as reader:
    overall_result = json.load(reader)

print(f"There are {len(overall_result)} unique test problem answer.")

recode_fast = dict()
for test_problem_identifier, solutions in overall_result.items():
    solutions_count = len(solutions)
    pass_testcases_and_answer_correct_solution = []

    for current_solution, current_solution_result in solutions.items():
        pass_testcases_and_answer_correct = current_solution_result["pass_testcases_and_answer_correct"]
        if pass_testcases_and_answer_correct:
            pass_testcases_and_answer_correct_solution.append(current_solution)
    
    fast = None
    if len(pass_testcases_and_answer_correct_solution) == 1:
        fast = pass_testcases_and_answer_correct_solution[0]
    elif len(pass_testcases_and_answer_correct_solution) > 1:
        time_rank = {}
        for candidata in pass_testcases_and_answer_correct_solution:
            time_rank[candidata] = solutions[candidata]["average_time"]
        min_pair = min((time, candidata) for candidata, time in time_rank.items())
        fast = min_pair[1]
    else:
        fast = None

    if fast is not None:
        recode_fast[test_problem_identifier] = {fast: solutions[fast]}

with open(os.path.join(BASE, generated_model_id, generate_project, f"fast_solutions_and_correct_{IDX}.json"), 'w') as writer:
    json.dump(recode_fast, writer, indent=4)