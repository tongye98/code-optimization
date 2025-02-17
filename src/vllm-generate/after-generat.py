import json 


def main(input_path, test_path, output_path, num_return_sequences):
    with open(input_path, 'r') as f, \
        open(test_path, 'r') as g, \
        open(output_path, 'w') as h:

        generated_samples = []
        for line in f:
            generated_sample = json.loads(line.strip())
            generated_samples.append(generated_sample)
        
        test_samples = []
        for line in g:
            test_sample = json.loads(line.strip())
            test_samples.append(test_sample)

        assert len(generated_samples) == len(test_samples), \
            f"generated samples lenght = {len(generated_samples)} || test samples lenght = {len(test_samples)}"
        
        output_samples = []
        for generated_sample, test_sample in zip(generated_samples, test_samples):
            if num_return_sequences == 1:
                test_sample["generated_answers"] = [generated_sample["solution"]]
            else:
                test_sample["generated_answers"] = generated_sample["solution"]
            output_samples.append(test_sample)
        
        for output_sample in output_samples:
            h.write(json.dumps(output_sample) + "\n")

import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", required=True,
        type=str
    )
    parser.add_argument(
        "--test_path", required=True,
        type=str
    )
    parser.add_argument(
        "--output_path", required=True,
        type=str
    )    
    parser.add_argument(
        "--num_return_sequences", required=True,
        type=int
    )

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    main(input_path=args.input_path,
         test_path=args.test_path,
         output_path=args.output_path,
         num_return_sequences=args.num_return_sequences)