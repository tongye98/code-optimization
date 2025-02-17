from transformers import AutoModel, AutoTokenizer
import json 
import torch 
from tqdm import tqdm 
import numpy as np

def get_embedding(code_dataset):
    checkpoint = "/models-hf/codet5-110m-embedding"
    device = "cuda"  # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    embeddings = []
    for code_item in tqdm(code_dataset):
        inputs = tokenizer.encode(code_item, return_tensors="pt", truncation=True, max_length=600).to(device)
        with torch.no_grad():
            embedding = model(inputs)[0]
        embeddings.append(embedding.cpu().numpy())

        del inputs
        torch.cuda.empty_cache()

    # print(f'Dimension of the embedding: {embedding.size()[0]}, with norm={embedding.norm().item()}')
    # Dimension of the embedding: 256, with norm=1.0
    embeddings = np.vstack(embeddings)
    np.save('problem_embeddings.npy', embeddings)
    return embeddings

def get_code_snippet(code_file):
    with open(code_file, 'r') as f:
        code_dataset = []
        for line in f:
            code_item = json.loads(line)
            code_src_tgt = code_item['src_code'] + '\n' +  code_item['tgt_code']
            code_dataset.append(code_src_tgt)
        print(len(code_dataset))
    return code_dataset

code_file = "train_problem_oriented_100_percent_count_0115.jsonl"
code_dataset = get_code_snippet(code_file)

embeddings = get_embedding(code_dataset)

print(f"Saved {embeddings.shape[0]} embeddings.")


