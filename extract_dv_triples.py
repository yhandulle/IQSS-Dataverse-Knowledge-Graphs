import subprocess
from datasets import load_dataset
import os
import json
import re

def extract_json_keywords(output: str):
    print(output)
    try:
        match = re.search(r'\[\s*"(.*?)"(?:\s*[,;]\s*"(.*?)")+\s*\]', output, re.DOTALL)
        if match:
            raw = match.group(0).replace(";", ",") 
            cleaned = raw.split("]")[0] + "]" 
            return json.loads(cleaned)

        match = re.search(r"\[.*?\]", output, re.DOTALL)
        if match:
            raw = match.group(0).replace(";", ",")
            raw = re.sub(r'"(\s*)"', '", "', raw)
            cleaned = raw.split("]")[0] + "]"
            return json.loads(cleaned)

    except json.JSONDecodeError as e:
        print("RETURNING NOTHING")
        return None

def old_extract_json_keywords(output: str):
    # Doesn't work very well
    try:
        match = re.search(r"```json\s*(\[.*?\])\s*```", output, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"\[.*?\]", output)
        if match:
            raw = match.group(0)
            raw = raw.replace(";", ",")
            cleaned = raw.split("]")[0] + "]"
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None
    except json.JSONDecodeError:
        return None

ds = load_dataset("rtreacy/dv_keywords_desc_full", split="test")

LLAMA_CLI_PATH = "/Users/yh/IQSS/llama.cpp/build/bin/llama-cli"
GGUF_MODEL_PATH = "/Users/yh/IQSS/models/llama32-3B-mig-en-es-full-Q4_K_M.gguf"

N = 12
print(f"Running on {N} samples from the dataset...\n")

for i in range(N):
    prompt = ds[i]["Prompt"]

    print("\n--- LLM Response ---\n")
    env = os.environ.copy()
    env["LLAMA_DISABLE_CHAT"] = "1"


    process = subprocess.Popen(
        [
            LLAMA_CLI_PATH,
            "-m", GGUF_MODEL_PATH,
            "--n-predict", "256",
            "--temp", "0.7",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    # Send the prompt and simulate pressing return and EOF
    stdout, _ = process.communicate(prompt + "\n")
    keywords = extract_json_keywords(stdout)
    print(f"\n✅ Extracted Keywords for sample {i}: {keywords}\n")
