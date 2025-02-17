"""
vllm generate. Fast generation
"""

from vllm import LLM, SamplingParams
import argparse
import os 
import json 
from typing import Iterable, Dict
import gzip 
from tqdm import tqdm 
import transformers
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest

# EOS = [
#     "<|endoftext|>",
#     "<|endofmask|>",
#     "</s>",
#     "\nif __name__",
#     "\ndef main(",
#     "\nprint(",
#     "\n#",
#     "<｜end▁of▁sentence｜>"
# ]

EOS = ["<｜end▁of▁sentence｜>"]

def write_jsonl(filename: str, data: Iterable[Dict]):
    """
    Writes an iterable of dictionaries to jsonl
    """
    with open(filename, 'w') as fp:
        for x in data:
            fp.write((json.dumps(x) + "\n"))


def get_dataset(dataset_path):
    dataset_list = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            dataset_list.append(data)
    
    # dataset_list = dataset_list[:30] # for debug
    dataset_dict = {}
    for id, data in enumerate(dataset_list):
        dataset_dict[id] = data
        # dataset_dict[data["task_id"]] = data   # for humaneval and mbpp

    return dataset_dict

def construct_prompt_for_humaneval_mbpp(text):
    # prompt = "Below is a program. Optimize the program and provide a more efficient version.\n\n### Program:\n{src_code}\n\n### Optimized Version:\n"
    # prompt = "### Instruction:\n{src_code}\n### Response:\n"
    # prompt = "Please complete the following Python function in a markdown style code block:\n```python\n{text}\n```"
    input = f"""<|im_start|>system
You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|>
<|im_start|>user
Can you complete the following Python function?
```python
{text}
```
<|im_end|>
<|im_start|>assistant
```python
"""
    return input 

def construct_prompt_for_pie(text):
    prompt = "Below is a program. Optimize the program and provide a more efficient version.\n\n### Program:\n{content}\n\n### Optimized Version:\n"
    input = prompt.format(content=text)
    return input 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,type=str)
    parser.add_argument("--model_path", required=True, type=str, help="path to base large language model")
    parser.add_argument("--samples_dir", required=True, type=str,help="path to save the samples")
    parser.add_argument("--max_new_tokens", default=1024, type=int, help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.")
    parser.add_argument("--use_lora", action="store_true", default=False, help="use lora")
    parser.add_argument("--base_model", type=str, help="For lora, base model path")
    args = parser.parse_args()
    return args


def vllm_setup(model_path, args):
    sampling_params = SamplingParams(
        n=8,
        best_of=8,
        temperature=0.5,
        max_tokens=1024,
        top_p=0.9,
        stop=EOS
    )

    if args.use_lora:
        print("use lora...")
        llm = LLM(model=args.base_model,
                gpu_memory_utilization=0.98,
                enable_lora=True,
                max_lora_rank=64,
                dtype="bfloat16")
    else:
        print("use full...")
        llm = LLM(model=model_path,
                gpu_memory_utilization=0.98,
                dtype="bfloat16")
    
    return llm, sampling_params

def vllm_generate_solution(llm, sampling_params, prompt, args):
    if args.use_lora:
        lora_request = LoRARequest("adapter", 1, lora_local_path=args.model_path)
        raw_generates = llm.generate(prompt, sampling_params, lora_request=lora_request)
    else:
        raw_generates = llm.generate(prompt, sampling_params)

    # gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in raw_generates]
    gen_strs = []
    for x in raw_generates:
        for out in x.outputs:
            gen_strs.append(out.text.replace("\t", "    ")) 

    return gen_strs

def main():
    args = get_args()
    print(args)

    if not os.path.isdir(args.samples_dir):
        os.makedirs(args.samples_dir, exist_ok=True)

    dataset = get_dataset(args.dataset)

    llm, sampling_params = vllm_setup(args.model_path, args)
    
    samples = []
    for task_id, task in tqdm(dataset.items()):
        prompt = construct_prompt_for_pie(task["src_code"])  # pie: src_code; humaneval: prompt
        solution = vllm_generate_solution(llm, sampling_params, prompt, args)
        samples.append(dict(task_id=task_id, solution=solution))
    
    # save samples
    write_jsonl(os.path.join(args.samples_dir, "pie_test_8_samples.jsonl"), samples)
    print(f'Solutions are saved at {os.path.join(args.samples_dir, "pie_test_8_samples.jsonl")}')

if __name__ == "__main__":
    main()