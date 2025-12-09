# Inference Benchmarking

Benchmark LLM inference performance with vLLM across different batch sizes. Automatically estimates prompt and output lengths from popular HuggingFace datasets and provides separate cost analysis for prefill (input) and generation (output) phases.

## Notes

Design choices:
- Uses normal distributions (mean, stddev) for prompt and output token lengths rather than fixed values, providing more realistic variation
- Estimates distributions from 3 popular HuggingFace datasets: chat (Open-Orca), code (Muennighoff/mbpp), and math/science (HuggingFaceH4/MATH-500)
- Separates cost analysis for prefill (input tokens) and generation (output tokens) phases
- Ignores
  -- Reasoning vs non-reasoning tokens
  -- Tool use
  -- Agentic worklows (we generate prompts randomly which means that successive prompts might not have the same prefix)
- Uses sensible defaults sourced from the internet (eg 512 for prompt-len similar to llm-perf)
- For a batch, we take metrics batch_generation_time as the max of the generation_times for all the prompts in the batch, even though some might terminate early.
-  Did not handle concurrent requests - I saw that as commandline param in aiperf and llmperf.

## Repro (via Lambda Labs GPU)

Get an A10 or A100 instance. Be sure to pick the Lambda Stack as the base image because that has Python installed, and you get the Cloud IDE (via the Terminal). Can also view the youtube video below.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install datasets transformers torch accelerate
uv pip install vllm --torch-backend=auto
git clone https://github.com/satyaki-up/inference_benchmarking
cd inference_benchmarking/
python benchmark_vllm.py
```

[Repro on youtube](https://www.youtube.com/watch?v=TIMWmMeE5Co)

## Arguments

- `--model`: HuggingFace model ID (required)
- `--gpu-price`: GPU price per hour in USD (required)
- `--batch-sizes`: Comma-separated batch sizes (default: "4,8,16,32,64,128")
- `--num-batches`: Number of batches to run (default: 20)

## Features

- **Automatic length estimation**: Estimates prompt and output token lengths from 3 popular HuggingFace datasets (chat, code, math/science)
- **Separate cost analysis**: Provides distinct cost calculations for input tokens (prefill phase) and output tokens (generation phase)
- **Performance metrics**: Reports tokens/second, requests/second, p50/p95 latency for each batch size

## Output

Reports tokens/second, requests/second, p50/p95 latency, and cost per million tokens for each batch size. Includes separate metrics for prefill and generation phases with individual cost breakdowns.

