import argparse
import random
import time
from statistics import median

from vllm import LLM, SamplingParams

from datagen import agentic_prompts, estimate_output_len, estimate_prompt_len


def run_single_config(
    llm: LLM,
    batch_size: int,
    prompt_len_mean: float,
    prompt_len_stddev: float,
    max_new_tokens_mean: float,
    max_new_tokens_stddev: float,
    num_batches: int,
) -> dict:
    prompts = agentic_prompts(batch_size, prompt_len_mean, prompt_len_stddev)
    max_new_tokens = max(1, int(random.gauss(max_new_tokens_mean, max_new_tokens_stddev)))
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    
    # Sampling params for prefill estimation (max_tokens=1)
    prefill_sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
    )

    # GPU warmup: First inference can be slower due to CUDA kernel compilation, memory allocation, and other one-time setup.
    # We run a dummy inference to warm up the GPU. Excludes cold-start overhead from benchmark measurements.
    _ = llm.generate(prompts, sampling_params)

    latencies = []
    prefill_times = []
    generation_times = []
    total_output_tokens = 0
    total_input_tokens = 0
    total_requests = batch_size * num_batches

    tokenizer = llm.llm_engine.tokenizer
    for prompt in prompts:
        total_input_tokens += len(tokenizer.encode(prompt))

    start_overall = time.time()
    for batch_idx in range(num_batches):
        # First call: estimate prefill time with max_tokens=1
        t_prefill_start = time.time()
        _ = llm.generate(prompts, prefill_sampling_params)
        t_prefill_end = time.time()
        batch_prefill_time = t_prefill_end - t_prefill_start
        
        # Second call: get total time
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        t1 = time.time()
        batch_latency = t1 - t0
        latencies.append(batch_latency)
        
        # Calculate generation time as total time minus prefill time
        batch_generation_time = batch_latency - batch_prefill_time
        
        for out in outputs:
            output_tokens = len(out.outputs[0].token_ids)
            total_output_tokens += output_tokens            
            print(f"  out: {out}")
        
        prefill_times.append(batch_prefill_time)
        generation_times.append(batch_generation_time)

    total_time = time.time() - start_overall
    
    total_input_tokens_all_batches = total_input_tokens * num_batches
    
    total_prefill_time = sum(prefill_times)
    total_generation_time = sum(generation_times)
    
    tps = total_output_tokens / total_time if total_time > 0 else 0.0
    rps = total_requests / total_time if total_time > 0 else 0.0
    
    prefill_tps = total_input_tokens_all_batches / total_prefill_time if total_prefill_time > 0 else 0.0
    generation_tps = total_output_tokens / total_generation_time if total_generation_time > 0 else 0.0

    latencies_sorted = sorted(latencies)
    p50 = median(latencies_sorted)
    p95 = latencies_sorted[int(0.95 * len(latencies_sorted)) - 1]

    return {
        "batch_size": batch_size,
        "tokens_per_second": tps,
        "requests_per_second": rps,
        "p50_latency": p50,
        "p95_latency": p95,
        "total_output_tokens": total_output_tokens,
        "total_input_tokens": total_input_tokens_all_batches,
        "total_time": total_time,
        "prefill_tokens_per_second": prefill_tps,
        "generation_tokens_per_second": generation_tps,
        "total_prefill_time": total_prefill_time,
        "total_generation_time": total_generation_time,
    }


def cost_per_million_tokens(gpu_price_per_hour: float, tokens_per_second: float) -> float:
    if tokens_per_second == 0:
        return float("inf")
    tokens_per_hour = tokens_per_second * 3600
    return gpu_price_per_hour / tokens_per_hour * 1_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HF model ID (default: Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--gpu-price", type=float, default=2.0,
                        help="GPU price per hour in USD (default: 2.0)")
    parser.add_argument("--batch-sizes", type=str, default="8,16,32,64,128,256,512,1024")
    parser.add_argument("--num-batches", type=int, default=20)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]

    print(f"Estimating prompt and output lengths from HF datasets...")
    prompt_len_mean, prompt_len_stddev = estimate_prompt_len(args.model)
    max_new_tokens_mean, max_new_tokens_stddev = estimate_output_len(args.model)
    print(f"Estimated prompt length: mean={prompt_len_mean:.1f}, stddev={prompt_len_stddev:.1f} tokens")
    print(f"Estimated max new tokens: mean={max_new_tokens_mean:.1f}, stddev={max_new_tokens_stddev:.1f} tokens")

    print(f"Loading model {args.model} with vLLM...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    results = []
    for bs in batch_sizes:
        print(f"\n=== Benchmarking batch_size={bs} ===")
        res = run_single_config(
            llm=llm,
            batch_size=bs,
            prompt_len_mean=prompt_len_mean,
            prompt_len_stddev=prompt_len_stddev,
            max_new_tokens_mean=max_new_tokens_mean,
            max_new_tokens_stddev=max_new_tokens_stddev,
            num_batches=args.num_batches,
        )
        
        price_per_million_total = cost_per_million_tokens(
            args.gpu_price, res["tokens_per_second"]
        )
        price_per_million_input = cost_per_million_tokens(
            args.gpu_price, res["prefill_tokens_per_second"]
        )
        price_per_million_output = cost_per_million_tokens(
            args.gpu_price, res["generation_tokens_per_second"]
        )
        
        res["usd_per_million_tokens"] = price_per_million_total
        res["usd_per_million_input_tokens"] = price_per_million_input
        res["usd_per_million_output_tokens"] = price_per_million_output
        results.append(res)

        print(
            f"bs={bs} | tps={res['tokens_per_second']:.1f} tok/s | "
            f"rps={res['requests_per_second']:.2f} req/s | "
            f"p50={res['p50_latency']:.3f}s | p95={res['p95_latency']:.3f}s"
        )
        print(
            f"  Prefill: {res['prefill_tokens_per_second']:.1f} tok/s | "
            f"$/{1_000_000} input={price_per_million_input:.4f}"
        )
        print(
            f"  Generation: {res['generation_tokens_per_second']:.1f} tok/s | "
            f"$/{1_000_000} output={price_per_million_output:.4f}"
        )


if __name__ == "__main__":
    main()

