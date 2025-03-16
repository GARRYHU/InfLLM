# https://github.com/THUDM/LongBench/blob/main/pred.py
import os
from datasets import load_from_disk
import torch
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from inf_llm.utils import patch_hf, GreedySearch, patch_model_center
from transformers import AutoModelForCausalLM, AutoTokenizer
import inf_llm.utils.debug_var as debug_var


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--model_center", action="store_true", default=False)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--datafile", type=str, required=True)
    parser.add_argument("--output_length", type=int, default=100)
    parser.add_argument("--force_load", action="store_true", default=False)
    parser.add_argument("--force_no_load", action="store_true", default=False)
    parser.add_argument("--output_hit_log", action="store_true", default=False)
    parser.add_argument("--output_miss_rate", action="store_true", default=False)
    parser.add_argument("--output_load_time", action="store_true", default=False)
    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.model.model_center = args.model_center
    conf.rank = args.rank
    conf.world_size = args.world_size
    conf.verbose = args.verbose
    conf.datafile = args.datafile
    conf.output_length = args.output_length
    conf.force_load = args.force_load
    conf.force_no_load = args.force_no_load
    conf.output_hit_log = args.output_hit_log
    conf.output_miss_rate = args.output_miss_rate
    conf.output_load_time = args.output_load_time
    if not hasattr(conf.model, "tokenizer_path"):
        conf.model.tokenizer_path = conf.model.path
    if not hasattr(conf, "truncation"):
        conf.truncation = None

    return conf


def get_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    if config.model_center:
        import bmtrain as bmt
        bmt.init_distributed(seed=233)
        from model_center.model import Llama, LlamaConfig
        model_config = LlamaConfig.from_pretrained(config.path)
        model_config.dtype = torch.bfloat16
        model = Llama(model_config)
        bmt.load(model, os.path.join(config.path, "pytorch_model.pt"), strict=False)
        model = patch_model_center(model, config.type, **config)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.path, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                     device_map="cuda")
        model = patch_hf(model, config.type, **config)
    return model, tokenizer


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    model_name = model_name.strip().lower()
    if model_name == "vicuna":
        from fastchat.conversation import get_conv_template
        conv = get_conv_template("vicuna_v1.1")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif model_name in ["mistral-inst", "qwen", "minicpm", "llama-3-inst"]:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        raise NotImplementedError

    return prompt


def load_infinite_bench(path, data_name) -> str:
    import re
    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
    """
    print(f"read {data_name}.jsonl")
    fin = open(os.path.join(path, data_name + ".jsonl"), "r")
    lines = fin.readlines()
    fin.close()
    data = [json.loads(line) for line in lines]

    def get_answer(inp: dict):
        if data_name in ["code_debug", "longbook_choice_eng"]:
            OPTIONS = "ABCD"
            if isinstance(inp["answer"], str):
                ret = [inp["answer"], OPTIONS[inp['options'].index(inp["answer"])]]
            elif isinstance(inp["answer"], list):
                if len(inp["answer"]) == 1:
                    ret = [inp["answer"][0], OPTIONS[inp['options'].index(inp["answer"][0])]]
                elif len(inp["answer"]) == 2 and inp["answer"][1] in ['A', 'B', 'C', 'D']:
                    ret = inp['answer']
                else:
                    raise ValueError
            else:
                raise ValueError
            return ret
        return inp["answer"]

    ret = []
    for eg in data:
        # ================= Code tasks
        if data_name == "code_run":
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
            func_call = find_result[0]
            func = func_call.split("(")[0]
            instance = {"func": func, "func_call": func_call, "context": eg["context"]}
        elif data_name in ["code_debug", "code_debug_qa"]:
            # Load source code
            instance = {"context": eg["context"]}
            if data_name == "code_debug":
                instance.update({
                    "OPTION_A": eg["options"][0],
                    "OPTION_B": eg["options"][1],
                    "OPTION_C": eg["options"][2],
                    "OPTION_D": eg["options"][3]})
        # ================= Code tasks
        elif data_name == "longdialogue_qa_eng":
            instance = {"context": eg["context"]}
        # ==================== Long book tasks
        elif data_name in [
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_sum_eng",
            "longbook_qa_chn",
        ]:
            instance = {"context": eg["context"]}
            if data_name == "longbook_choice_eng":
                instance.update({
                    "question": eg["input"],
                    "OPTION_A": eg["options"][0],
                    "OPTION_B": eg["options"][1],
                    "OPTION_C": eg["options"][2],
                    "OPTION_D": eg["options"][3],
                })
            elif data_name in ["longbook_qa_eng", "longbook_qa_chn"]:
                instance.update({
                    "question": eg["input"],
                })
        elif data_name == "math_calc":
            instance = {"context": eg["context"]}
        elif data_name == "math_find":
            prompt = eg['input']
            context = eg['context']
            # Find "the * number" from the prompt
            find_result = re.findall(r"The .+ of", prompt)
            assert find_result, f"Cannot find the target number in {prompt}"
            target_number = find_result[0].lower()[:-3]
            # Replace the number with the answer
            prefix = f"What is {target_number} in the following list?"
            instance = {"prefix": prefix, "context": context, "input": prompt}
        elif data_name == "kv_retrieval":
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
                "key": eg["input"][6:44]
            }
            assert eg['input'][6] == '"'
            assert eg['input'][43] == '"'
        else:
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
            }
        ans = get_answer(eg)
        instance["answers"] = ans if isinstance(ans, list) else [ans]
        instance["length"] = len(instance["context"].split())
        instance["all_classes"] = None

        ret.append(instance)
        # if len(ret) > 4:
        #     break
    return ret


def post_process(pred, model_name, dataset):
    if model_name == "qwen":
        pred = pred.split("<|im_end|>")[0]

    if dataset == "samsum":
        pred = pred.split("\n")[0].strip()

    return pred


def inference(
        searcher, tokenizer, max_length,
        max_gen, prompt,
        model_name: str,
        gen_chunk_size=None, truncation: str = None,
        verbose: bool = False
):
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=True).input_ids[0]

    if truncation is None:
        if len(tokenized_prompt) > max_length - max_gen:
            if verbose:
                print(f"Length {len(tokenized_prompt)}. Skipped.")
    else:
        if truncation == "suffix":
            length = len(tokenized_prompt)
            if length > max_length - max_gen:
                if verbose:
                    print("over length")
                init_token_num = 128
                prompt = tokenizer.decode(tokenized_prompt[:init_token_num].tolist() + tokenized_prompt[- (
                            max_length - max_gen - init_token_num):].tolist())
                tokenized_prompt = \
                tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=True).input_ids[0]
        else:
            raise NotImplementedError

    extra_end_token_ids = []
    if model_name == "llama-3-inst":
        extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])

    if model_name == "qwen":
        extra_end_token_ids.append(tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])

    print(f"{tokenized_prompt.size(0)}")

    output = searcher.generate(
        input_ids=tokenized_prompt,
        max_length=max_gen,
        chunk_size=gen_chunk_size,
        extra_end_token_ids=extra_end_token_ids
    )

    searcher.clear()
    return output


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model
    model, tokenizer = get_model_and_tokenizer(args.model)
    debug_var.force_load = args.force_load
    debug_var.force_no_load = args.force_no_load
    debug_var.output_hit_log = args.output_hit_log
    debug_var.output_miss_rate = args.output_miss_rate
    debug_var.output_load_time = args.output_load_time
    assert debug_var.force_no_load & debug_var.force_load is False
    searcher = GreedySearch(model, tokenizer)

    prompts = list()
    sys_prompt = "You are a helpful assistant. You can help me by answering my questions."
    with open(args.datafile, "r") as f:
        for i in range(10):
            line = f.readline()
            try:
                data = json.loads(line)
                # 提取 input 和 content 字段
                input_question = data.get('input')
                context = data.get('context')
                prompt = sys_prompt + context + "\n\n回答问题:" + input_question
                prompts.append(prompt)
            except json.JSONDecodeError:
                print("第一行内容不是有效的 JSON 格式。")

    i = 5
    # prompt = prompts[i]
    # with open("/home/garry/llm/vllm_ini/text5400.txt", "r") as f:
    #     content = f.read()
    # prompt = content
    prompt_token_ids = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=True).input_ids[0]
    print(f"prompt length = {prompt_token_ids.size(0)}")
    output = inference(
        searcher=searcher,
        tokenizer=tokenizer,
        max_length=args.max_len,
        max_gen=args.output_length,
        prompt=prompt,
        gen_chunk_size=args.chunk_size,
        truncation=args.truncation,
        model_name=args.model_name
    )
    searcher.clear()

    with open(f"req_{i}.txt", 'w') as f:
        f.write(prompt)
        f.write("\n answer: ")
        f.write(output[0])
