import random
from statistics import mean, stdev
from typing import List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


def fetch_hf_dataset(dataset_name: str, max_samples: int = 1000) -> Optional[list]:
    """Helper method to fetch a HuggingFace dataset."""
    splits_to_try = ["train", "test"]
    
    for attempt_split in splits_to_try:
        try:
            dataset = load_dataset(dataset_name, split=attempt_split, streaming=False)
            if len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            return list(dataset)
        except Exception as e:
            if attempt_split == splits_to_try[-1]:
                print(f"Warning: Could not load dataset {dataset_name} with splits {splits_to_try}: {e}")
                return None
            continue
    
    return None


def estimate_prompt_len(model_name: str) -> Tuple[float, float]:
    """Estimate normal distribution (mean, stddev) of prompt length from 3 popular HF datasets (chat, code, math/science)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    datasets_config = [
        ("Open-Orca/OpenOrca", "question"),  # Chat dataset
        ("Muennighoff/mbpp", "content"),  # Code dataset
        ("HuggingFaceH4/MATH-500", "problem"),  # Math/Science dataset
    ]
    
    all_prompt_lengths = []
    
    for dataset_name, prompt_field in datasets_config:
        dataset = fetch_hf_dataset(dataset_name, max_samples=500)
        if dataset is None:
            continue
            
        for example in dataset:
            if prompt_field in example:
                prompt = str(example[prompt_field])
                prompt_tokens = len(tokenizer.encode(prompt))
                all_prompt_lengths.append(prompt_tokens)
    
    if not all_prompt_lengths:
        return (512.0, 100.0)
    
    prompt_mean = mean(all_prompt_lengths)
    prompt_stddev = stdev(all_prompt_lengths) if len(all_prompt_lengths) > 1 else 100.0
    
    print(f"Estimated prompt length: mean={prompt_mean:.1f}, stddev={prompt_stddev:.1f} tokens for dataset {dataset_name}")
    return (prompt_mean, prompt_stddev)


def estimate_output_len(model_name: str) -> Tuple[float, float]:
    """Estimate normal distribution (mean, stddev) of output length from the same 3 datasets."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    datasets_config = [
        ("Open-Orca/OpenOrca", "response"),  # Chat dataset
        ("Muennighoff/mbpp", "content"),  # Code dataset 
        ("HuggingFaceH4/MATH-500", "solution"),  # Math/Science dataset
    ]
    
    all_output_lengths = []
    
    for dataset_name, output_field in datasets_config:
        dataset = fetch_hf_dataset(dataset_name, max_samples=500)
        if dataset is None:
            continue
            
        for example in dataset:
            if output_field in example:
                output = str(example[output_field])
                output_tokens = len(tokenizer.encode(output))
                all_output_lengths.append(output_tokens)
    
    if not all_output_lengths:
        return (256.0, 50.0)
    
    output_mean = mean(all_output_lengths)
    output_stddev = stdev(all_output_lengths) if len(all_output_lengths) > 1 else 50.0
    
    print(f"Estimated output length: mean={output_mean:.1f}, stddev={output_stddev:.1f} tokens for dataset {dataset_name}")
    return (output_mean, output_stddev)


def synthetic_prompts(batch_size: int, prompt_len_mean: float, prompt_len_stddev: float) -> List[str]:
    base = "You are a helpful assistant. " + ("Please respond succinctly. " * 50)
    prompts = []
    for _ in range(batch_size):
        prompt_len = max(1, int(random.gauss(prompt_len_mean, prompt_len_stddev)))
        prompts.append(base[:prompt_len * 4]) # assume 4 chars per token
    return prompts


def agentic_prompts(batch_size: int, prompt_len_mean: float, prompt_len_stddev: float) -> List[str]:
    """
    Generate agentic prompts where each group of 5 prompts forms a trajectory.
    In a trajectory, each prompt is a prefix of the next one.
    """
    base = "You are a helpful assistant. " + ("Please respond succinctly. " * 50)
    prompts = []
    trajectory_size = 5
    num_trajectories = (batch_size + trajectory_size - 1) // trajectory_size
    
    for _ in range(num_trajectories):
        # Generate base prompt for this trajectory
        base_prompt_len = max(1, int(random.gauss(prompt_len_mean, prompt_len_stddev)))
        current_prompt = base[:base_prompt_len * 4]
        
        # Generate prompts in this trajectory (up to 5, or remaining if last trajectory)
        prompts_in_traj = min(trajectory_size, batch_size - len(prompts))
        for i in range(prompts_in_traj):
            prompts.append(current_prompt)
            
            # Generate suffix for next prompt (if not last in trajectory)
            if i < prompts_in_traj - 1:
                # Generate suffix length using mean/stddev (suffix is typically smaller)
                suffix_token_len = max(1, int(random.gauss(prompt_len_mean * 0.2, prompt_len_stddev * 0.2)))
                suffix_char_len = suffix_token_len * 4
                
                # Create random suffix from base text starting after current prompt
                start_idx = len(current_prompt)
                end_idx = min(len(base), start_idx + suffix_char_len)
                suffix = base[start_idx:end_idx]
                
                # Append suffix to current prompt for next iteration
                current_prompt = current_prompt + suffix
    
    return prompts

