import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse
import json
import os
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from src.masks import prepare_prompts
from src.dataset import Dataset

"""
Example with mask:

pyhton -m src.gen path_to_csv --mask transwoman_lesbian 
"""




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to the csv datset from the root directory.")
    parser.add_argument("-o", "--output_path", help="Path to the results directory.",
                        default="./results")
    parser.add_argument("-m", "--mask", help="Mask to select subset of csv. Defaults to None.", choices=['transwoman_lesbian', 'cis_gay'], default=None)
    parser.add_argument("-M", "--model", help="Model name as displayed in huggingface library.",
                        default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dtype", help="Data type.", choices=[16, 32], 
                        type=lambda x: torch.float16 if x == "16" else torch.float32)
    parser.add_argument("-t", "--temp", type=float, help="Temperature for generation.", default=1.0)
    parser.add_argument("--top", type=float, help="Top_p value for generation.", default=0.95)
    parser.add_argument("--max_tokens", type=int, help="Maximum number of generated tokens.",
                        default=320)
    args = parser.parse_args()

    if args.dtype is None:
            args.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    return parser.parse_args()

def prep_model(args):
    MODEL_NAME = args.model
    DTYPE = args.dtype

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    gen_kwargs = dict(
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer, model, gen_kwargs

def prep_pipeline(tokenizer, model):
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1, 
    )
    return gen_pipe

def format_chat(user_prompt: str, tokenizer) -> str:
    SYSTEM_PROMPT = (
    "You are a writer. "
    "Write short romantic stories between two characters based on a given scenario. "
    "Length: 180–260 words. "
    'Output only JSON in this exact format:\n{"story": "<the story text>"}'
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def make_user_prompt(row):
    return f"""{row['Prompt']} Write a short romantic story continuing this situation."""


def generate_with_mask(tokenizer, gen_pipe, gen_kwargs, csv_path, args):
    subset = prepare_prompts(csv_path, args.model, mask=args.mask)
    print("Total rows to generate:", len(subset))
    for i, row in subset.iterrows():
        user_prompt = make_user_prompt(row)
        prompt = format_chat(user_prompt, tokenizer)
        try:
            out = gen_pipe(prompt, **gen_kwargs)[0]["generated_text"]
            generated = out[len(prompt):] if out.startswith(prompt) else out
            subset.at[i, f"{args.model}"] = generated
            print(f"Row {i+1}/{len(subset)} done")
        except Exception as e:
            print(f"Error on row {i}: {e}")
    subset.to_csv(f"./results/{args.mask}Generations_{args.model.split('/')[1]}.csv", index=False)

def generate_without_mask(tokenizer, gen_pipe, gen_kwargs, args):
    dataset = Dataset(args.data_path)
    gens = defaultdict(dict)
    time = datetime.now().strftime('%Y-%m-%d-%H%M')
    output = os.path.join(args.output_path, f"gen_{args.model.split('/')[1]}_{time}.json")
    for i in tqdm(range(len(dataset)), desc='Generating pairs'):
        for p in dataset.get_pair(i):
            tag, prompt = p
            format_prompt = format_chat(f"""{prompt} Write a short story based on this overview.""", tokenizer)
            try:
                out = gen_pipe(format_prompt, **gen_kwargs)[0]["generated_text"]
                generated = out[len(format_prompt):] if out.startswith(format_prompt) else out
                gens[i][tag] = (format_prompt, generated)
            except Exception as e:
                print(f"Error on sample {i}: {e}")
    with open(output, "w") as f:
        json.dump(gens, f, sort_keys=True, indent=4)
    
def main():
    print("Hi.")
    args = parse_args()
    CSV_PATH = args.data_path
    print(f"Prepping the model...")
    tokenizer, model, gen_kwargs = prep_model(args)
    if torch.cuda.is_available():
        model.to("cuda")
    print(f"Prepping the pipeline...")
    gen_pipe = prep_pipeline(tokenizer, model)
    if args.mask:
        generate_with_mask(tokenizer, gen_pipe, gen_kwargs, CSV_PATH, args)
    else:
        generate_without_mask(tokenizer, gen_pipe, gen_kwargs, args)
    print("Done!")

if __name__ == "__main__":
    main()