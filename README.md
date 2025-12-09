# Inference Benchmarking

Benchmark LLM inference performance with vLLM across different batch sizes. Automatically estimates prompt and output lengths from popular HuggingFace datasets and provides separate cost analysis for prefill (input) and generation (output) phases.

## Notes

Design choices:
- Uses normal distributions (mean, stddev) for prompt and output token lengths rather than fixed values, providing more realistic variation
- Estimates distributions from 3 popular HuggingFace datasets: chat (Open-Orca), code (bigcode/python_code), and math/science (lighteval/MATH)
- Separates cost analysis for prefill (input tokens) and generation (output tokens) phases
- Ignores
  -- Reasoning vs non-reasoning tokens
  -- Tool use
  -- Agentic worklows (we generate prompts randomly which means that successive prompts might not have the same prefix)
- Uses sensible defaults sourced from the internet (eg 512 for prompt-len similar to llm-perf)
- For a batch, we take metrics batch_generation_time as the max of the generation_times for all the prompts in the batch, even though some might terminate early.

## Prerequisites (Ubuntu Server Setup)

For rented GPU instances, first ensure Python 3.12+ and pip are installed:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip
```

Verify installation:
```bash
python3 --version  # Should be 3.12 or higher
pip3 --version
```

## Setup

```bash
pip install datasets transformers torch accelerate
pip install vllm --torch-backend=auto
pip install huggingface_hub
```

Login to HuggingFace (required for some datasets):
```bash
huggingface-cli login
```

You'll need a HuggingFace account and access token. Get your token from https://huggingface.co/settings/tokens

## Usage

```bash
python benchmark_vllm.py --model <model-id> --gpu-price <price>
```

Example:
```bash
python benchmark_vllm.py --model meta-llama/Meta-Llama-3-8B-Instruct --gpu-price 2.5
```

## Arguments

- `--model`: HuggingFace model ID (required)
- `--gpu-price`: GPU price per hour in USD (required)
- `--batch-sizes`: Comma-separated batch sizes (default: "8,16,32,64,128")
- `--num-batches`: Number of batches to run (default: 20)

## Features

- **Automatic length estimation**: Estimates prompt and output token lengths from 3 popular HuggingFace datasets (chat, code, math/science)
- **Separate cost analysis**: Provides distinct cost calculations for input tokens (prefill phase) and output tokens (generation phase)
- **Performance metrics**: Reports tokens/second, requests/second, p50/p95 latency for each batch size

## Output

Reports tokens/second, requests/second, p50/p95 latency, and cost per million tokens for each batch size. Includes separate metrics for prefill and generation phases with individual cost breakdowns.

![Mac](https://ibb.co/r2ndYtF8)

