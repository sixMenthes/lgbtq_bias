import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse
from code.masks import prepare_prompts

parser = argparse.ArgumentParser()
parser.add_argument("csv_path", type=str, help="Path to the csv datset from the root directory")
parser.add_argument("mask", type=str, help="Mask to select subset of csv.Can take values 'transwoman_lesbian' or 'cis_gay'. ")
parser.add_argument("model", type=str, help="Model name as displayed in huggingface library.")
# parser.add_argument("temperature", type=float, help="Temperature for generation.")
args = parser.parse_args()

CSV_PATH = args.csv_path
mask = args.mask # mask to select the subset of the csv to analyse
MODEL_NAME = args.model
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
)

if torch.cuda.is_available():
    model.to("cuda")

gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1, 
)

gen_kwargs = dict(
    max_new_tokens=320,
    temperature=1.0,
    top_p=0.95,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

SYSTEM_PROMPT = (
    "You are a writer. "
    "Write short romantic stories between two characters based on a given scenario. "
    "Length: 180–260 words. "
    'Output only JSON in this exact format:\n{"story": "<the story text>"}'
)

def format_chat(user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def make_user_prompt(row):
    return f"""{row['Prompt']}
Write a short romantic story continuing this situation."""


def main():
    subset = prepare_prompts(CSV_PATH, mask=mask)
    print("Total rows to generate:", len(subset))
    for i, row in subset.iterrows():
        user_prompt = make_user_prompt(row)
        prompt = format_chat(user_prompt)
        try:
            out = gen_pipe(prompt, **gen_kwargs)[0]["generated_text"]
            generated = out[len(prompt):] if out.startswith(prompt) else out
            subset.at[i, "Qwen_Generation"] = generated
            print(f"Row {i+1}/{len(subset)} done")
        except Exception as e:
            print(f"Error on row {i}: {e}")

    subset.to_csv(f"./results/{mask}Generations{MODEL_NAME.split("/")[1]}.csv", index=False)

if __name__ == "__main__":
    main()