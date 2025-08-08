# salloc -p gpu_test -N 1 -n 4 --gres=gpu:1 -t 2:0:0  --mem=21g
# module load nvhpc/23.7-fasrc01
# module load cuda/12.2.0-fasrc01 
# module load gcc/12.2.0-fasrc01
# conda activate cuda

from llama_cpp import Llama
import re
import json
import pandas as pd
from datasets import load_dataset

# Path to your quantized model
MODEL_PATH = "/n/siacus_lab/Lab/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=3072,        # context length
    n_gpu_layers=-1,   # use GPU offloading
    verbose=False
)

dv = load_dataset("rtreacy/dv_keywords_desc_full")
    
data = dv["train"]



# Global system prompt (changed for DataCite)
system_prompt = (
    "You are an expert in research data curation. Extract subject-predicate-object "
    "triplets from dataset metadata and map them to DataCite Metadata Schema properties. "
    "Output only valid JSON as described."
)

# User prompt template (changed for DataCite)
user_prompt_template = """
Guidelines:
- The subject of each triplet must always be the dataset title.
- Map the `Subject` field to predicate `subject`.
- Map the `Keywords` field to predicate `subject` (one triplet per keyword, do not concatenate).
- For measurable variables in Description → predicate `other` with type `variableMeasured` in the object text.
- Time ranges/dates in Description → predicate `date` with `dateType` "Coverage".
- Geographic names/locations in Description → predicate `geoLocation`.
- Do not create or infer unrelated properties (like creators, identifiers, or publisher) unless explicitly given.
- Only include triplets that can be directly derived from the provided metadata fields.

Return JSON in this format:
[
  {"subject": "<Dataset>", "predicate": "<DataCite property>", "object": "<value>"}
]

Metadata:
- Title: {title}
- Description: {description}
- Subject: {subject}
- Keywords: {keywords}
"""

def extract_json_to_df(output_text: str) -> pd.DataFrame:
    # Find JSON array
    json_match = re.search(r"\[\s*{.*}\s*\]", output_text, re.S)
    if not json_match:
        # Fallback: find first '{' and last '}'
        brace_match = re.search(r"\{.*\}", output_text, re.S)
        if brace_match:
            json_str = "[" + brace_match.group(0) + "]"
        else:
            raise ValueError(f"No JSON found in model output:\n{output_text[:500]}...")
    else:
        json_str = json_match.group(0)
    # Clean and parse
    json_str = json_str.replace("\n", "").replace("\r", "")
    data = json.loads(json_str)
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Normalize column names
    df.columns = df.columns.str.lower().str.replace('"', '').str.strip()
    return df

def post_process_triplets(df: pd.DataFrame, subject_metadata: str) -> pd.DataFrame:
    """
    Filter out unwanted predicates for DataCite.
    """
    # Allowed predicates for DataCite
    allowed_predicates = {"subject", "date", "geolocation", "other"}
    df = df[df["predicate"].str.lower().isin(allowed_predicates)]
    return df.reset_index(drop=True)

def generate_triplets(entry: dict) -> pd.DataFrame:
    """
    Generate subject–predicate–object triplets from a single dataset metadata entry.
    """
    # Extract metadata fields
    pid = entry.get("pid", "")
    title = entry.get("Title", "")
    description = entry.get("Description", "")
    subject = entry.get("Subject", "")
    keywords = entry.get("Keywords", "")
    # Prepare user prompt
    user_prompt = user_prompt_template.format(
        title=title,
        description=description,
        subject=subject,
        keywords=keywords
    )
    # Construct prompt
    prompt = (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    # Call model
    response = llm(
        prompt,
        max_tokens=1000,
        temperature=0.1,
        stop=["<|eot_id|>"]
    )
    # Extract and normalize JSON
    output_text = response["choices"][0]["text"]
    df = extract_json_to_df(output_text)
    # Force subject = Title
    df["subject"] = title if title else "this dataset"
    # Post-process (filter unwanted predicates)
    df = post_process_triplets(df, subject)
    # Add PID as first column
    df.insert(0, "pid", pid)
    return df

target_pid = "doi:10.7910/DVN/26448"
entry = next(row for row in data if row["pid"] == target_pid)
print(f"Entry found at{entry}")
print(generate_triplets(entry))


