from transformers import AutoTokenizer

model_name_or_path = "/largespace/tydata/code_optimization/cpp/saved_models/pie-gem5-by-user-cpp_deepseekcoder-7b-instruct_sft_score_tag_0505/full_model/"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)