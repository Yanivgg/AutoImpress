import os
import time
import pandas as pd
from openai import AzureOpenAI
from bert_score import score
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# ---------- Azure OpenAI ----------

def setup_azure_client_gpt41():
    """
    Initialize Azure OpenAI client for GPT-4.1 model access.
    Retrieves credentials from environment variables and returns configured client.
    
    Returns:
        AzureOpenAI: Configured client for GPT-4.1 model access
    """
    endpoint = os.getenv("AZURE_ENDPOINT_GPT41")
    api_key = os.getenv("AZURE_API_KEY_GPT41")
    api_version = os.getenv("AZURE_API_VERSION_GPT41", "2024-12-01-preview")
    return AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)

def setup_azure_client_gpt4o():
    """
    Initialize Azure OpenAI client for GPT-4 model access.
    Used for evaluation and judgment of model outputs.
    
    Returns:
        AzureOpenAI: Configured client for GPT-4 model access
    """
    endpoint = os.getenv("AZURE_ENDPOINT_GPT4O")
    api_key = os.getenv("AZURE_API_KEY_GPT4O")
    api_version = os.getenv("AZURE_API_VERSION_GPT4O", "2024-12-01-preview")
    return AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)


def build_prompt(row):
    """
    Build a prompt for radiology report analysis.
    Constructs a prompt asking for identification of acute/abnormal findings from a structured report.
    
    Args:
        row (pd.Series): Row from DataFrame containing radiology report fields
        
    Returns:
        str: Formatted prompt for the model
    """
    return (
        f"You are a radiologist. Given the following structured radiology report (without the original impression), "
        f"state explicitly whether there are any acute or abnormal findings. "
        f"If there are, briefly list them. If none, write: 'No acute findings.' Only provide this answer, no summary or explanation.\n\n"
        f"Image: {row['image']}\n"
        f"Indication: {row['indication']}\n"
        f"Comparison: {row['comparison']}\n"
        f"Findings: {row['findings']}\n"
        f"MeSH: {row['MeSH']}\n"
        f"Problems: {row['Problems']}\n"
    )

def process_samples(df, client, model_name="gpt-4.1", sample_fraction=0.5, random_state=42):
    """
    Process a sample of radiology reports using GPT model to generate impressions.
    
    Args:
        df (pd.DataFrame): DataFrame containing radiology reports
        client: Azure OpenAI client
        model_name (str): Name of the model to use (default: "gpt-4.1")
        sample_fraction (float): Fraction of dataset to process (default: 0.5)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        pd.DataFrame: DataFrame with generated impressions and ground truth
    """
    sample_df = df.sample(frac=sample_fraction, random_state=random_state)
    results = []

    for idx, row in sample_df.iterrows():
        prompt = build_prompt(row)

        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional radiologist assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.5,
                top_p=0.95,
                model=model_name
            )
            output = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error on idx {idx}: {e}")
            output = "ERROR"

        results.append({
            "uid": row.get("uid", idx),
            "acute_findings_report": output,
            "ground_truth_impression": row["impression"]
        })

        time.sleep(0.3)

    return pd.DataFrame(results)


# ---------- Clinical Equivalence with "Yes"/"No" ----------

def ask_gpt_yesno_equivalence(client, generated, reference, model_name="gpt-4o"):
    """
    Compare generated and reference impressions using GPT-4 for clinical equivalence.
    
    Args:
        client: Azure OpenAI client
        generated (str): Generated impression text
        reference (str): Reference (ground truth) impression text
        model_name (str): Model to use for comparison (default: "gpt-4o")
        
    Returns:
        str: "Yes" or "No" indicating clinical equivalence
    """
    prompt = f"""
You are a senior radiologist.
Given the following two impressions, compare only their acute (urgent) findings.

Reference Impression:
{reference}

Generated Impression:
{generated}

Important instructions:
- Treat phrases like “No acute findings” and “No acute cardiopulmonary disease” as clinically equivalent.
- Ignore wording or phrasing differences if the clinical meaning is the same.

Answer with "Yes" or "No" only.
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return "Error"

def run_yesno_equivalence_judgment(df, client, model_name="gpt-4o", save_path=None):
    """
    Run clinical equivalence judgment on a DataFrame of impressions.
    
    Args:
        df (pd.DataFrame): DataFrame containing generated and ground truth impressions
        client: Azure OpenAI client
        model_name (str): Model to use for judgment (default: "gpt-4o")
        save_path (str, optional): Path to save results CSV
        
    Returns:
        pd.DataFrame: Original DataFrame with added equivalence judgments
    """
    equiv_results = []
    for idx, row in df.iterrows():
        result = ask_gpt_yesno_equivalence(client, row["acute_findings_report"], row["ground_truth_impression"], model_name)
        print(f"Processing row {idx + 1}/{len(df)}")
        equiv_results.append(result)

    df["gpt_equivalence"] = equiv_results
    correct = sum(x.lower().startswith("yes") for x in equiv_results)
    total = len(equiv_results)
    print(f"Clinical Equivalence Rate: {correct}/{total} = {correct / total:.1%}")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"✅ Saved equivalence results to '{save_path}'")

    return df

# ---------- DeepSeek ----------

def setup_deepseek_client():
    """
    Initialize DeepSeek model client using Azure credentials.
    Retrieves endpoint and API key from environment variables.
    
    Returns:
        ChatCompletionsClient: Configured client for DeepSeek model
    """
    endpoint = os.getenv("DEEPSEEK_ENDPOINT")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_version = os.getenv("DEEPSEEK_API_VERSION", "2024-05-01-preview")
    return ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
        api_version=api_version
    )

def process_deepseek_samples(df, client, model_name="DeepSeek-V3-0324", sample_fraction=0.1, random_state=42):
    sample_df = df.sample(frac=sample_fraction, random_state=random_state)
    results = []

    for idx, row in sample_df.iterrows():
        prompt = build_prompt(row)

        try:
            response = client.complete(
                messages=[
                    SystemMessage(content="You are a professional radiologist assistant."),
                    UserMessage(content=prompt)
                ],
                max_tokens=300,
                temperature=0.5,
                top_p=0.95,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                model=model_name
            )
            output = response.choices[0].message.content.strip()
        except Exception as e:
            #print(f"❌ Error on idx {idx}: {e}")
            output = "ERROR"

        results.append({
            "uid": row.get("uid", idx),
            "acute_findings_report": output,
            "ground_truth_impression": row["impression"]
        })

        #time.sleep(1.2)

    return pd.DataFrame(results)



def compute_bertscore(df, candidate_col, reference_col, name_prefix, lang="en"):
    """
    Compute BERTScore for a DataFrame and return results as a dictionary.
    """
    cands = df[candidate_col].tolist()
    refs = df[reference_col].tolist()

    P, R, F1 = score(cands, refs, lang=lang, verbose=1)
    mean_f1 = F1.mean().item()
    std_f1 = F1.std().item()

    print(f"BERTScore {name_prefix} F1 mean ± std: {mean_f1:.4f} ± {std_f1:.4f}")

    return {
        f"{name_prefix}_P": P,
        f"{name_prefix}_R": R,
        f"{name_prefix}_F1": F1,
        f"{name_prefix}_mean": mean_f1,
        f"{name_prefix}_std": std_f1
    }

def compute_llmjudge_accuracy(df, result_column="gpt_equivalence"):
    """
    Compute LLM Judge accuracy (% of 'yes') from a DataFrame.
    """
    total = len(df)
    correct = sum(str(x).lower().startswith("yes") for x in df[result_column])
    accuracy = (correct / total * 100) if total > 0 else 0.0

    print(f"Recomputed LLM Judge: {correct}/{total} = {accuracy:.1f}%")
    return correct, total, accuracy

#Focus only on whether the acute findings match.


#--------------Colab utilities--------------

def build_few_shot_prompt(image, indication, comparison, findings, mesh=None, problems=None):
    example = (
        "Image: Chest X-ray\n"
        "Indication: Persistent cough\n"
        "Comparison: None\n"
        "Findings: Lungs are clear. No pleural effusion or pneumothorax. Heart size normal.\n"
        "MeSH: Normal\n"
        "Problems: Normal\n"
        "Impression: Normal chest x-ray.\n\n"
    )
    target = (
        f"Image: {image}\n"
        f"Indication: {indication}\n"
        f"Comparison: {comparison}\n"
        f"Findings: {findings}\n"
    )
    if mesh:
        target += f"MeSH: {mesh}\n"
    if problems:
        target += f"Problems: {problems}\n"
    return (
        "You are a radiologist. Generate the Impression section based on the structured report. See the example first:\n\n"
        + example + target + "Impression:"
    )

def generate_impression(prompt, tokenizer, model, max_input_length=512, max_new_tokens=60, num_beams=4):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_generation_on_dataframe(df, tokenizer, model, sample_size=100):
    results = []
    sample_df = df.iloc[:sample_size]

    for _, row in sample_df.iterrows():
        prompt = build_few_shot_prompt(
            image=row['image'],
            indication=row['indication'],
            comparison=row['comparison'],
            findings=row['findings'],
            mesh=row.get('MeSH', ''),
            problems=row.get('Problems', '')
        )
        generated = generate_impression(prompt, tokenizer, model)
        results.append({
            "uid": row['uid'],
            "generated_impression": generated,
            "true_impression": row['impression']
        })

    return pd.DataFrame(results)

def ask_gpt_clinical_equivalence_flan(client, generated, reference, model_name="gpt-4o"):
    prompt = f"""You are a medical expert. Compare the following two radiology impressions.

Determine if they are clinically equivalent in meaning, and verify that the generated impression is written in appropriate radiology language without including non-clinical prompt elements (such as 'Image:', 'Impression:', 'Findings:', 'MeSH:', etc.).

Reference Impression: {reference}
Generated Impression: {generated}

If the generated text includes non-clinical formatting or prompt tokens, consider it NOT clinically equivalent.

Answer with "Yes" or "No" only."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return "Error"

def run_equivalence_judgment_flan(df, client, model_name="gpt-4o", save_path=None):
    equiv_results = []
    for idx, row in df.iterrows():
        result = ask_gpt_clinical_equivalence_flan(client, row["generated_impression"], row["true_impression"], model_name)
        equiv_results.append(result)

    df["gpt_equivalence"] = equiv_results
    correct = sum(str(x).lower().startswith("yes") for x in equiv_results)
    total = len(equiv_results)
    accuracy = (correct / total * 100) if total > 0 else 0.0

    print(f"Clinical Equivalence Rate: {correct}/{total} = {accuracy:.1f}%")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"✅ Saved equivalence results to '{save_path}'")

    return df, correct, total, accuracy



def build_finetuning_prompt(row):
    example = (
        "Image: Chest X-ray\n"
        "Indication: Persistent cough\n"
        "Comparison: None\n"
        "Findings: Lungs are clear. No pleural effusion or pneumothorax. Heart size normal.\n"
        "MeSH: Normal\n"
        "Problems: Normal\n"
        "Impression: Normal chest x-ray.\n\n"
    )

    structured = (
        f"Image: {row['image']}\n"
        f"Indication: {row['indication']}\n"
        f"Comparison: {row['comparison']}\n"
        f"Findings: {row['findings']}\n"
    )
    if pd.notna(row["MeSH"]):
        structured += f"MeSH: {row['MeSH']}\n"
    if pd.notna(row["Problems"]):
        structured += f"Problems: {row['Problems']}\n"

    return (
        "You are a radiologist. Generate the Impression section based on the structured report. See the example first:\n\n"
        + example + structured + "Impression:"
    )



def preprocess_finetuning(example, tokenizer, max_input_length=512, max_target_length=128):
    model_input = tokenizer(
        example["input_text"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True
    )
    target = tokenizer(
        example["target_text"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True
    )
    model_input["labels"] = target["input_ids"]
    return model_input


def generate_predictions_from_test_dataframe(test_df, tokenizer, model, max_input_length=512, max_new_tokens=60):
    """
    Generate predictions on a DataFrame converted from a test Hugging Face dataset.

    Parameters:
        test_df (pd.DataFrame): DataFrame with 'input_text' and 'target_text' columns.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face Seq2Seq model.
        max_input_length (int): Maximum length for input tokens.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        pd.DataFrame: DataFrame with 'uid', 'generated_impression', 'true_impression'.
    """
    results = []
    for _, row in test_df.iterrows():
        inputs = tokenizer(
            row["input_text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "uid": row.get("uid", None),  # include uid if exists
            "generated_impression": decoded,
            "true_impression": row["target_text"]
        })
    return pd.DataFrame(results)


def load_project_csv(relative_path):
    """
    Load a CSV file based on project root, no matter where the notebook runs from.

    Parameters:
        relative_path (str): Path relative to the project root (e.g., 'outputs/flan-t5-base/file.csv')

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    # Resolve project root (assumes notebooks/ is inside project)
    current_dir = os.getcwd()
    if 'notebooks' in os.path.basename(current_dir):
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    full_path = os.path.join(project_root, relative_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ File not found: {full_path}")

    df = pd.read_csv(full_path)
    return df

def analyze_yes_without_no_acute_pattern(df, name=""):
    yes_judged = df["gpt_equivalence"].str.strip().str.lower().str.startswith("yes")

    # Dynamically find the ground truth column
    if "true_impression" in df.columns:
        ground_truth_col = "true_impression"
    elif "ground_truth_impression" in df.columns:
        ground_truth_col = "ground_truth_impression"
    else:
        print(f"❌ Could not find ground truth column in {name}")
        return

    ground_truth_no_acute_pattern = df[ground_truth_col].str.lower().str.contains(r"\bno acute\b", regex=True)

    # Filter: cases where ground truth does NOT have the pattern, but judge said Yes
    tricky_yes = (~ground_truth_no_acute_pattern) & (yes_judged)

    count_tricky_yes = tricky_yes.sum()
    total_yes = yes_judged.sum()

    print(f"===== {name} =====")
    print(f"Total YES judgments: {total_yes}")
    print(f"YES without 'no acute' pattern in ground truth: {count_tricky_yes} ({count_tricky_yes / total_yes * 100:.1f}%)")
    print()

# ---------- Analysis Utilities ----------

def count_yes_no(df, col="gpt_equivalence"):
    """
    Count the number of 'Yes' and 'No' responses in a DataFrame column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the judgment results
        col (str): Name of the column containing Yes/No responses, defaults to 'gpt_equivalence'
    
    Returns:
        tuple: (yes_count, no_count, total_count) containing:
            - Number of 'Yes' responses
            - Number of 'No' responses
            - Total number of responses
    """
    yes_count = sum(str(x).strip().lower().startswith("yes") for x in df[col])
    no_count = sum(str(x).strip().lower().startswith("no") for x in df[col])
    total = len(df)
    return yes_count, no_count, total


def analyze_yes_without_no_acute_pattern(df, name=""):
    """
    Analyze cases where the model judged 'Yes' without the typical 'no acute' pattern.
    
    This function identifies cases where the model found clinical equivalence in reports
    that don't use the standard 'no acute' phrasing, helping to understand the model's
    ability to recognize semantic equivalence beyond simple pattern matching.
    
    Notes:
        - Automatically detects whether to use 'true_impression' or 'ground_truth_impression'
        - Case-insensitive pattern matching for 'no acute'
        - Considers only definitive Yes/No judgments
    """
    yes_judged = df["gpt_equivalence"].str.strip().str.lower().str.startswith("yes")

    # Dynamically find the ground truth column
    if "true_impression" in df.columns:
        ground_truth_col = "true_impression"
    elif "ground_truth_impression" in df.columns:
        ground_truth_col = "ground_truth_impression"
    else:
        print(f"❌ Could not find ground truth column in {name}")
        return

    ground_truth_no_acute_pattern = df[ground_truth_col].str.lower().str.contains(r"\bno acute\b", regex=True)

    # Filter: cases where ground truth does NOT have the pattern, but judge said Yes
    tricky_yes = (~ground_truth_no_acute_pattern) & (yes_judged)

    count_tricky_yes = tricky_yes.sum()
    total_yes = yes_judged.sum()

    print(f"===== {name} =====")
    print(f"Total YES judgments: {total_yes}")
    print(f"YES without 'no acute' pattern in ground truth: {count_tricky_yes} ({count_tricky_yes / total_yes * 100:.1f}%)")
    print()

