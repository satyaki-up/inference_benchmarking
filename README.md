# Inference Benchmarking

Benchmark LLM inference performance with vLLM across different batch sizes. Automatically estimates prompt and output lengths from popular HuggingFace datasets and provides separate cost analysis for prefill (input) and generation (output) phases.

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
pip install datasets transformers torch
pip install vllm --torch-backend=auto
```

## Usage

```bash
python benchmark.py --model <model-id> --gpu-price <price>
```

Example:
```bash
python benchmark.py --model meta-llama/Meta-Llama-3-8B-Instruct --gpu-price 2.5
```

## Arguments

- `--model`: HuggingFace model ID (required)
- `--gpu-price`: GPU price per hour in USD (required)
- `--batch-sizes`: Comma-separated batch sizes (default: "8,16,32,64,128")
- `--num-batches`: Number of batches to run (default: 20)
- `--max-p95-latency`: Maximum acceptable p95 latency in seconds (default: 10.0)

## Features

- **Automatic length estimation**: Estimates prompt and output token lengths from 3 popular HuggingFace datasets (chat, code, math/science)
- **Separate cost analysis**: Provides distinct cost calculations for input tokens (prefill phase) and output tokens (generation phase)
- **Performance metrics**: Reports tokens/second, requests/second, p50/p95 latency for each batch size
- **Optimization**: Selects the best batch size configuration within latency constraints

## Output

Reports tokens/second, requests/second, p50/p95 latency, and cost per million tokens for each batch size. Includes separate metrics for prefill and generation phases with individual cost breakdowns. Selects the best configuration within the latency constraint.

