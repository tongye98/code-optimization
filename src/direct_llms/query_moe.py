import argparse
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path",type=str,default="/largespace/tydata/code_optimization/cpp/dataset/by_problem/test_out_pair_in_original.json")
    parser.add_argument("--model_checkpoint",type=str,default="/largespace/tydata/models-hf/deepseek-coder-33b-instruct")
    parser.add_argument("--use_beam_search",type=bool,default=False)
    parser.add_argument("--n",type=int,default=1)
    parser.add_argument("--temperature",type=float,default=0.7)
    parser.add_argument("--top_p",type=float,default=0.95)
    parser.add_argument("--max_tokens",type=int,default=1024)
    parser.add_argument("--result_pth",type=str,default="/largespace/tydata/code_optimization/cpp/baselines/cot/codellama-7b-instruct/generate_1_samples-strategy.json")
    args = parser.parse_args()
    
    return args

def prepare_input1(data):

    prompt = "### Slower program:"
    slow_code = data["slow_code"]

    input = f"{prompt}\n{slow_code}\nAre there any effective strategies to improve the time efficiency of a given code? I'm asking for strategies, not code."

    return input

def vllm_generate(args):
    llm = LLM(args.model_checkpoint, swap_space=128, gpu_memory_utilization=0.9)
    #, max_model_len=14192
    if args.use_beam_search:
        gen_param = SamplingParams(use_beam_search=True, n=args.n, max_tokens=args.max_tokens)
    else:
        gen_param = SamplingParams(use_beam_search=False, n=args.n, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)

    
    test_data = json.load(open(args.test_data_path,'r'))

    chunk_size = 400
    all_out = []

    for i in tqdm(range(0, len(test_data), chunk_size)):
        chunks = [prepare_input1(data) for data in test_data[i:i+chunk_size]]
        out_seqs = llm.generate(prompts=chunks, sampling_params=gen_param)
        all_out += out_seqs

    results = []
    for in_data, out_data in zip(test_data, all_out):
        new_data = {}
        new_data['slow_code'] = in_data['slow_code']
        new_data['fast_code'] = in_data['fast_code']
        new_data['maybe_faster'] = [out.text for out in out_data.outputs]
        results.append(new_data)

    json.dump(results, open(args.result_pth,'w'), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = parse_args()

    vllm_generate(args)

    


