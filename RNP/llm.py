"""
Review Necessity Prediction (RNP) using Large Language Models (LLMs)

This script uses large language models (CodeLlama, DeepSeek-Coder, Yi-Coder, Qwen2.5-Coder)
to predict whether a code change requires review via few-shot prompting.
"""

import json
import random
import re
import os
import argparse
import traceback

# ================= Configuration =================

# Model configurations - paths will be provided via command line arguments
MODEL_CONFIGS = {
    "CodeLlama-7b": {
        "model_name": "CodeLlama-7b-Instruct-hf"
    },
    "DeepSeek-Coder-7b": {
        "model_name": "deepseek-coder-7b-instruct-v1.5"
    },
    "Yi-Coder-9b": {
        "model_name": "Yi-Coder-9B-Chat"
    },
    "Qwen2.5-Coder-7b": {
        "model_name": "Qwen2.5-Coder-7B-Instruct"
    }
}

# Supported input modes
TARGET_MODES = ["original", "msg", "cc", "all"]

# =================================================


def parse_chronological_data(text):
    """
    Parse XML-formatted commit data to extract current code and history timeline.
    
    Args:
        text: Raw text containing code change data with XML-like tags
        
    Returns:
        current_code: The current code change snippet
        history_timeline: List of historical items (commit messages and code changes)
    """
    current_match = re.search(r'<CURRENT_CODE_CHNAGE>(.*?)</CURRENT_CODE_CHNAGE>', text, re.DOTALL)
    if not current_match:
        return text.strip(), []

    current_code = current_match.group(1).strip()
    history_text = text[:current_match.start()]
    
    pattern = re.compile(
        r'<HISTORY_MSG>(.*?)</HISTORY_MSG>|'
        r'<MSG>(.*?)</MSG>|'
        r'<HOSTORY_CODE_CHNAGE>(.*?)</HOSTORY_CODE_CHNAGE>',
        re.DOTALL
    )
    
    history_timeline = []
    for match in pattern.finditer(history_text):
        if match.group(1): 
            history_timeline.append({'type': 'msg', 'content': match.group(1).strip()})
        elif match.group(2): 
            history_timeline.append({'type': 'msg', 'content': match.group(2).strip()})
        elif match.group(3): 
            history_timeline.append({'type': 'code', 'content': match.group(3).strip()})
            
    return current_code, history_timeline


def get_few_shot_examples(train_file, mode, k=3):
    """
    Construct few-shot examples with label conversion logic (0->NO, 1->YES).
    
    Args:
        train_file: Path to training data file
        mode: Input mode ('original', 'msg', 'cc', 'all')
        k: Number of examples to sample
        
    Returns:
        examples_str: Formatted string containing few-shot examples
    """
    with open(train_file, 'r') as f:
        lines = f.readlines()
    
    # Randomly sample k examples
    sample_size = min(len(lines), k)
    random_lines = random.sample(lines, sample_size)
    
    examples_str = ""
    for line in random_lines:
        data = json.loads(line)
        c_code, timeline = parse_chronological_data(data['code_change'])
        
        # Label conversion: 1 -> YES (needs review), 0 -> NO (no review needed)
        label_val = data['label']
        label_str = "Review Necessity: YES" if label_val == 1 else "Review Necessity: NO"
        
        context_str = ""
        if mode != 'original' and timeline:
            temp_str = "Here is the chronological history leading up to the current code:\n"
            step_count = 1
            has_content = False
            for item in timeline:
                if mode == 'msg' and item['type'] != 'msg': continue
                if mode == 'cc' and item['type'] != 'code': continue
                label = "Commit Message" if item['type'] == 'msg' else "History Code Change"
                temp_str += f"[Step {step_count} - {label}]:\n{item['content']}\n\n"
                step_count += 1
                has_content = True
            if has_content:
                context_str = temp_str + "After the history above, here is the [Current Code Snippet] to be judged:\n"
            else:
                context_str = "Given the [Current Code Snippet] to be judged:\n"
        else:
            context_str = "Given the [Current Code Snippet] to be judged:\n"

        ex_text = (
            f"{context_str}{c_code}\n"
            f"Should the code snippet above be reviewed by other code reviewers? This is the response:\n{label_str}"
        )
        examples_str += ex_text + "\n\n"
        
    return examples_str


def construct_safe_prompt(tokenizer, mode, current_code, history_timeline, examples_str, max_total_tokens=3800):
    """
    Construct prompt with intelligent truncation to fit within token limit.
    
    If the prompt exceeds max_total_tokens, it truncates the history content
    while preserving the instruction, examples, current code, and suffix.
    
    Args:
        tokenizer: Tokenizer for encoding/decoding text
        mode: Input mode ('original', 'msg', 'cc', 'all')
        current_code: The current code snippet to review
        history_timeline: List of historical context items
        examples_str: Few-shot examples string
        max_total_tokens: Maximum allowed tokens for the prompt
        
    Returns:
        final_prompt: The constructed prompt within token limit
    """
    
    # 1. Build fixed prefix - customized for binary classification task
    base_instruction = (
        "You are a code reviewer, expert in detecting low-quality code. "
        "Determine whether the following code snippet requires a review by other developers.\n"
        "Provide response only in the following format: Review Necessity: <YES or NO>. "
        "Do not include anything else in the response.\n"
    )
    prefix_str = f"{base_instruction}Here are three examples:\n{examples_str}\n"
    
    # 2. Build fixed suffix - trigger phrase
    suffix_str = "Should the code snippet above be reviewed by other code reviewers? This is the response:"
    
    # 3. Build truncatable content (history + current code)
    context_str = ""
    if mode != 'original' and history_timeline:
        context_str += "Here is the chronological history leading up to the current code:\n"
        step_count = 1
        has_content = False
        for item in history_timeline:
            if mode == 'msg' and item['type'] != 'msg': continue
            if mode == 'cc' and item['type'] != 'code': continue
            
            label = "Commit Message" if item['type'] == 'msg' else "History Code Change"
            context_str += f"[Step {step_count} - {label}]:\n{item['content']}\n\n"
            step_count += 1
            has_content = True
            
        if has_content:
            context_str += "After the history above, here is the [Current Code Snippet] to be judged:\n"
        else:
            context_str = "Given the [Current Code Snippet] to be judged:\n"
    else:
        context_str = "Given the [Current Code Snippet] to be judged:\n"

    # Content = history context + current code
    content_str = f"{context_str}{current_code}\n"

    # 4. Calculate token lengths and perform intelligent truncation
    ids_prefix = tokenizer.encode(prefix_str)
    ids_suffix = tokenizer.encode(suffix_str, add_special_tokens=False)
    ids_content = tokenizer.encode(content_str, add_special_tokens=False)
    
    len_fixed = len(ids_prefix) + len(ids_suffix)
    budget = max_total_tokens - len_fixed
    
    # Ensure minimum budget for content
    if budget < 100: 
        budget = 100 

    if len(ids_content) > budget:
        # Keep the last 'budget' tokens of content to preserve current code
        ids_content = ids_content[-budget:]
        content_str = tokenizer.decode(ids_content, skip_special_tokens=False)
    
    # 5. Concatenate final prompt
    final_prompt = prefix_str + content_str + suffix_str
    return final_prompt


def run_experiment(llm_engine, sampling_params, model_name, language, mode, dataset_base, epochs=10):
    """
    Run LLM inference experiment for review necessity prediction.
    
    Args:
        llm_engine: The LLM engine for inference
        sampling_params: Sampling parameters for generation
        model_name: Name of the model being used
        language: Programming language ('cpp' or 'java')
        mode: Input mode ('original', 'msg', 'cc', 'all')
        dataset_base: Base path to the dataset directory
        epochs: Number of experiment epochs
    """
    # Construct dataset path from provided base
    data_dir = os.path.join(dataset_base, language, "RNP", "all")
    train_path = os.path.join(data_dir, "train.json")
    test_path = os.path.join(data_dir, "test.json")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Skipping {language} - {mode}: Data files not found in {data_dir}")
        return

    print(f"--- Task: Model=[{model_name}], Lang=[{language}], Mode=[{mode}] ---")

    test_data_list = []
    with open(test_path, 'r') as f:
        for line in f:
            test_data_list.append(json.loads(line))

    # Output directory for RNP results
    output_dir = os.path.join("llm_results", model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Get tokenizer for length calculation
    tokenizer = llm_engine.get_tokenizer()

    for epoch in range(epochs):
        examples_str = get_few_shot_examples(train_path, mode)
        prompts = []
        gold_references = []
        
        for data in test_data_list:
            c_code, timeline = parse_chronological_data(data['code_change'])
            
            # Construct prompt with intelligent truncation
            prompt = construct_safe_prompt(
                tokenizer=tokenizer,
                mode=mode,
                current_code=c_code,
                history_timeline=timeline,
                examples_str=examples_str,
                max_total_tokens=3800
            )
            
            prompts.append(prompt)
            # Save converted label text for comparison
            label_str = "Review Necessity: YES" if data['label'] == 1 else "Review Necessity: NO"
            gold_references.append(label_str)

        # Run inference
        try:
            outputs = llm_engine.generate(prompts, sampling_params)
        except Exception as e:
            print(f"Error during generation in epoch {epoch}: {e}")
            traceback.print_exc()
            continue

        output_filename = os.path.join(output_dir, f"result_{language}_{mode}_{epoch}.json")
        
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            for prompt, gold, output_item in zip(prompts, gold_references, outputs):
                generated_text = output_item.outputs[0].text.replace('\n', ' ').strip()
                record = {
                    "source": prompt,
                    "target": gold,
                    "output": generated_text
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"[{model_name}|{language}] Epoch {epoch} finished. Saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review Necessity Prediction using LLMs")
    parser.add_argument("--model_name", type=str, required=True, 
                        choices=list(MODEL_CONFIGS.keys()), 
                        help="Model name from available configurations")
    parser.add_argument("--language", type=str, required=True, 
                        choices=["cpp", "java"], 
                        help="Programming language for experiment")
    parser.add_argument("--gpu_id", type=str, required=True, 
                        help="Specific GPU ID to use (e.g., '0')")
    parser.add_argument("--models_base", type=str, required=True, 
                        help="Base directory containing model folders")
    parser.add_argument("--dataset_base", type=str, required=True, 
                        help="Base directory containing dataset folders")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of epochs to run (with temperature=0, only 1 epoch is needed)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    from vllm import LLM, SamplingParams

    model_name = args.model_name
    language = args.language
    
    if model_name not in MODEL_CONFIGS:
        print(f"Error: Model {model_name} not found in config.")
        exit(1)

    config = MODEL_CONFIGS[model_name]
    model_path = os.path.join(args.models_base, config["model_name"])

    print(f"\n{'='*30}\nLaunching RNP: {model_name} | Lang: {language} | GPU: {args.gpu_id}\n{'='*30}")
    
    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.9, 
            max_model_len=4096
        )
        
        # Binary classification task doesn't need long output, 32 tokens is sufficient
        # Use temperature=0 for deterministic output (greedy decoding)
        sampling_params = SamplingParams(temperature=0, max_tokens=32)
        
        for mode in TARGET_MODES:
            run_experiment(llm, sampling_params, model_name, language, mode, args.dataset_base, args.epochs)
            
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"CRITICAL ERROR on GPU {args.gpu_id} ({model_name}-{language}):\n{error_msg}")