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

# Show first entry
print(data[0])




# Global system prompt
system_prompt = (
    "You are an expert in research data curation. Extract subject-predicate-object "
    "triplets from dataset metadata and map them to schema.org/Dataset properties. "
    "Output only valid JSON as described."
)

user_prompt_template = """
Guidelines:
- The subject of each triplet must always be the dataset title.
- The `about` predicate must always use the metadata field "Subject".
- The `keyword` predicate must always use the metadata field "Keywords".
- If there are multiple keywords separated by commas or semicolons, output one `keyword` triplet per keyword (do not keep them concatenated).
- Use `variableMeasured` only for measurable variables explicitly mentioned in the Description.
- Use `temporalCoverage` only if explicit time ranges or dates appear in the Description.
- Use `spatialCoverage` only if explicit geographic names or locations appear in the Description (otherwise do not include it).
- Do not create or infer properties like `creator`, `datePublished`, or `isBasedOn`.
- Valid predicates are: about, keyword, variableMeasured, temporalCoverage, spatialCoverage.
- Only include triplets that can be directly derived from the provided metadata fields (Title, Description, Subject, Keywords).

Return JSON in this format:
[
  {{"subject": "<Dataset>", "predicate": "<schema.org property>", "object": "<value>"}}
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
    Force `about` to equal Subject metadata and filter out unwanted predicates.
    """
    # Force about = Subject metadata
    df.loc[df["predicate"].str.lower() == "about", "object"] = subject_metadata
    # Allowed predicates
    allowed_predicates = {"about", "keyword", "variablemeasured", "temporalcoverage", "spatialcoverage"}
    df = df[df["predicate"].str.lower().isin(allowed_predicates)]
    # Optionally drop spatialCoverage if nonsense (no location-like text)
    spatial_mask = df["predicate"].str.lower() == "spatialcoverage"
    df = df[~(spatial_mask & ~df["object"].str.contains(r"[A-Z][a-z]+", na=False))]
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
    # Construct prompt (no duplicate <|begin_of_text|>)
    prompt = (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    print(prompt)
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
    # Post-process (fix about, filter nonsense)
    df = post_process_triplets(df, subject)
    # Add PID as first column
    df.insert(0, "pid", pid)
    return df


# try this


generate_triplets(data[0])

generate_triplets(data[1])

generate_triplets(data[100])










