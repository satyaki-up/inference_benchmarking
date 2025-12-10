# Inference Benchmarking

Benchmark LLM inference performance with vLLM across different batch sizes. Automatically estimates prompt and output lengths from popular HuggingFace datasets and provides separate cost analysis for prefill (input) and generation (output) phases.

## Notes

Design choices:
- Uses normal distributions (mean, stddev) for prompt and output token lengths rather than fixed values, providing more realistic variation
- Estimates distributions from 3 popular HuggingFace datasets: chat (Open-Orca), code (Muennighoff/mbpp), and math/science (HuggingFaceH4/MATH-500)
- Separates cost analysis for prefill (input tokens) and generation (output tokens) phases
- Estimate prefill and total time first, then subtract to get generation time
  -- [PREFERRED] benchmark_vllm_simulate.py calls the LLM with max_output_token=1 and uses that as proxy for prefill time instead of looking at metrics.time_to_first_token
  -- benchmark_vllm_use_metric.py uses metrics.time_to_first_token inside vllm output to get TTFT as a proxy for prefill time (got errors for this on my GPU - always empty)
- Ignores
  -- Reasoning vs non-reasoning tokens
  -- Tool use
  -- Agentic worklows (we generate prompts randomly which means that successive prompts might not have the same prefix)
- Uses sensible defaults sourced from the internet (eg 512 for prompt-len similar to llm-perf)
- For a batch, we take metrics batch_generation_time as the max of the generation_times for all the prompts in the batch, even though some might terminate early.
- Did not handle concurrent requests - I saw that as commandline param in aiperf and llmperf. (I think that uses vllm AsyncLLMEngine or concurrency inside SamplingParams)
- Used default values for many params (eg tensor_parallel_size and gpu_memory_utilization) from the vLLM wiki


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
python benchmark_vllm_simulate.py
```

[Repro on youtube](https://www.youtube.com/watch?v=TIMWmMeE5Co)

## Arguments

- `--model`: HuggingFace model ID (required)
- `--gpu-price`: GPU price per hour in USD (required)
- `--batch-sizes`: Comma-separated batch sizes (default: "4,8,16,32,64,128,256,512,1024")
- `--num-batches`: Number of batches to run (default: 20)

## Features

- **Automatic length estimation**: Estimates prompt and output token lengths from 3 popular HuggingFace datasets (chat, code, math/science)
- **Separate cost analysis**: Provides distinct cost calculations for input tokens (prefill phase) and output tokens (generation phase)
- **Performance metrics**: Reports tokens/second, requests/second, p50/p95 latency for each batch size

## Output

Reports tokens/second, requests/second, p50/p95 latency, and cost per million tokens for each batch size. Includes separate metrics for prefill and generation phases with individual cost breakdowns.


## Sample vllm response

out: RequestOutput(request_id=55, prompt='You are a helpful assistant. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. Please respond succinctly. \n\nAssistant: I\'ll use a tool to help with this.\nTool call: {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}\nTool result: {"temperature": 22, "condition": "sunny", "humidity": 65}\nAssistant: Based on the tool result, ', prompt_token_ids=[2610, 525, 264, 10950, 17847, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 5209, 5889, 98632, 398, 13, 4710, 71703, 25, 358, 3278, 990, 264, 5392, 311, 1492, 448, 419, 624, 7740, 1618, 25, 5212, 606, 788, 330, 455, 30541, 9040, 497, 330, 16370, 788, 5212, 18785, 788, 330, 82916, 43, 95642, 7740, 1102, 25, 5212, 34558, 788, 220, 17, 17, 11, 330, 9056, 788, 330, 82, 27297, 497, 330, 93046, 788, 220, 21, 20, 532, 71703, 25, 20205, 389, 279, 5392, 1102, 11, 220], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None, outputs=[CompletionOutput(index=0, text='22 degrees Celsius and 65% humidity, the stock price of Apple Inc. (AAPL) is $100.00.', token_ids=[17, 17, 12348, 61347, 323, 220, 21, 20, 4, 37093, 11, 279, 5591, 3349, 315, 8162, 4848, 13, 320, 82916, 43, 8, 374, 400, 16, 15, 15, 13, 15, 15, 13, 151643], cumulative_logprob=None, logprobs=None, finish_reason=stop, stop_reason=None)], finished=True, metrics=None, lora_request=None, num_cached_tokens=320, multi_modal_placeholders={})

