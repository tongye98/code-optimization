import os
import argparse
from tqdm import tqdm
from typing import List, Optional

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import json 
from typing import Iterable, Dict
import gzip 


EOS = [
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\nassert",
]


def write_jsonl(
    filename: str, data: Iterable[Dict], append: bool = False, drop_builtin: bool = True
):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    if drop_builtin:
                        x = {k: v for k, v in x.items() if not k.startswith("_")}
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                if drop_builtin:
                    x = {k: v for k, v in x.items() if not k.startswith("_")}
                fp.write((json.dumps(x) + "\n").encode("utf-8"))

def get_dataset(dataset_path):
    dataset_list = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            dataset_list.append(data)
    
    dataset_dict = {}
    for id, data in enumerate(dataset_list):
        dataset_dict[id] = data

    return dataset_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True,
        type=str
    )
    parser.add_argument(
        "--model_path", required=True,
        type=str, help="path to base large language model"
    )
    parser.add_argument(
        "--instruct", action="store_true", 
        help="Whether or not it is a instruction model"
    )
    parser.add_argument(
        "--adapter_path", default=None, 
        type=str, help="path to lora/qlora adapter"
    )
    parser.add_argument(
        "--samples_dir", required=True,
        type=str,help="path to save the samples"
    )
    parser.add_argument(
        "--gpu_id", default=None, 
        type=str, help="single GPU id for evaluation"
    )
    parser.add_argument(
        "--dtype", default="bfloat16", 
        type=str, choices=["float16", "bfloat16"]
    )
    parser.add_argument(
        "--max_new_tokens", default=1024, type=int, 
        help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."
    )
    parser.add_argument(
        "--do_sample", action="store_true",
        help="Whether or not to use sampling; use greedy decoding otherwise."
    )
    parser.add_argument(
        "--temperature", default=0.6, type=float,
        help="The value used to modulate the next token probabilities."
    )
    parser.add_argument(
        "--top_p", default=0.9, type=float, 
        help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."
    )
    parser.add_argument(
        "--num_return_sequences", default=8, type=int,
        help="number of generate."
    )
    args = parser.parse_args()
    return args


def load_tokenizer_and_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"loading model from {args.model_path}")
    kwargs = {}
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id else "auto"
    kwargs["device_map"] = device
    kwargs["torch_dtype"] = getattr(torch, args.dtype)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    if args.adapter_path:
        print(f"loading adapter from {args.adapter_path}")
        peft_model = PeftModel.from_pretrained(
            model=base_model, model_id=args.adapter_path,
        )
        model = peft_model.merge_and_unload()
    else:
        model = base_model
    
    return tokenizer, model


def generate_solution(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompt: str, 
    dataset: str,
    eos: Optional[List[str]] = None,
    max_new_tokens: Optional[int] = 2048,
    do_sample: Optional[bool] = False,
    temperature: Optional[float] = 0.6,
    top_p: Optional[float] = 0.9,
    instruct: Optional[bool] = False,
    num_return_sequences: Optional[int] = 8,
) -> str:
    add_special_tokens = not instruct
    inputs = tokenizer(prompt, add_special_tokens=add_special_tokens, return_tensors="pt").to(model.device)
    if do_sample:
        outputs = model.generate(
            # **inputs,
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            tokenizer=tokenizer,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stop_strings=eos,
            num_return_sequences=num_return_sequences
        )
    else:
        outputs = model.generate(
            # **inputs,
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stop_strings=eos,
        )
    solution = ""
    if instruct:
        if num_return_sequences == 1:
            solution = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        elif num_return_sequences > 1:
            solution = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        else:
            assert False
    else:
        assert False
    return solution


def construct_prompt(
    text: str, dataset: str, instruct: bool,
    tokenizer: transformers.PreTrainedTokenizerBase,
):
    if not instruct:
        return text
    
    # if it is a instruction model
    # if dataset == "humaneval":
    #     instruction = f"Please complete the following Python function in a markdown style code block:\n```python\n{text}\n```"
    # elif dataset == "mbpp":
    #     instruction = f"Please write the Python function in a markdown style code block that comply with the following instruction.\n{text}"

    instruction = text

    msg = [
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    # prompt = prompt + "```python"
    return prompt


def main():
    args = get_args()
    print(args)

    if not os.path.isdir(args.samples_dir):
        os.makedirs(args.samples_dir, exist_ok=True)
    
    # load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(args)

    dataset = get_dataset(args.dataset)
    
    samples = []
    for task_id, task in tqdm(dataset.items()):
        prompt = construct_prompt(task["src_code"], args.dataset, args.instruct, tokenizer)
        solution = generate_solution(
            model, tokenizer, prompt, args.dataset, None, args.max_new_tokens,
            args.do_sample, args.temperature, args.top_p, args.instruct, args.num_return_sequences
        )
        samples.append(dict(task_id=task_id, solution=solution))
    

    # save samples
    write_jsonl(os.path.join(args.samples_dir, f"generated_8_samples.jsonl"), samples)
    print(f'Solutions are saved at {os.path.join(args.samples_dir, f"generated_8_samples.jsonl")}')


if __name__ == "__main__":
    main()