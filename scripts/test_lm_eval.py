# SPDX-License-Identifier: Apache-2.0
"""
LM eval harness to test correctness of LLaMa4 implementation.
Tasks definitions are in correctness/tasks/.
Existing lm_eval tasks are supported. We can add more Meta specific stuff there.

This script supports both starting up an vLLM offline engine locally or
hitting an OpenAI compatible HTTP endpoint.

Environment setups:
    export VLLM_USE_V1=1
    export SAFETENSORS_FAST_GPU=1

Example usage (offline vLLM):
    python3 test_lm_eval.py
    python3 test_lm_eval.py --limit 10  # faster check
    python3 test_lm_eval.py --task gsm8k  # use other tasks

Example usage (OpenAI compatible HTTP server):
    PORT=12345 vllm serve $LLAMA_DIR
    --swap-space 16 \
    --served-model-name llama4_17b \
    -tp 8 \
    --host :: --port $PORT \
    --seed 0 (--disable-log-requests)

    python3 test_lm_eval.py --backend http
"""

import argparse
import logging
import os
from functools import partial
from pathlib import Path

import lm_eval
from lm_eval.loggers import EvaluationTracker
from lm_eval.utils import make_table, simple_parse_args_string

HOSTNAME = os.environ.get("HOSTNAME", "localhost")
PORT = os.environ.get("PORT", "12345")
# TODO: be smarter about this depends on 16E or 128E
GPU_MEMORY_UTILIZATION = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9")
MODEL_PATH = os.environ.get("LLAMA_OUT_DIR",
                            "meta-llama/Llama-4-Scout-17B-16E-Instruct")
MAX_OUTPUT_LEN = 2048


def setup_llama_tasks(args):
    tasks_dir = Path(__file__).parent / "tasks"
    kwargs = {}
    if any(["ruler_" in task or "niah_" in task for task in args.tasks]):
        kwargs["metadata"] = {
            # for constructing long context eval datasets
            "max_seq_lengths": args.ruler_seq_lens,
            "pretrained": MODEL_PATH,
        }
    task_manager = lm_eval.tasks.TaskManager(
        include_path=str(tasks_dir.resolve()),
        include_defaults=True,
        **kwargs,
    )
    return task_manager


def model_args_dict_to_str(model_args):
    return ",".join(f"{k}={v}" for k, v in model_args.items())


def http_args(args):
    args_dict = {
        "model": args.model,
        "base_url": f"http://{HOSTNAME}:{PORT}/v1/chat/completions",
        "num_concurrent": 96,
        "max_retries": 3,
        "tokenized_requests": False,
        "verify_certificate": False,
        "max_gen_toks": MAX_OUTPUT_LEN,
        "max_length": args.max_seq_len,
    }
    return model_args_dict_to_str(args_dict)


def vllm_args_dict(args):
    args_dict = {
        "pretrained": MODEL_PATH,
        "max_length": args.max_seq_len,
        "tensor_parallel_size": 8,
        "dtype": "auto",
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_gen_toks": MAX_OUTPUT_LEN,
        "seed": 0,
    }
    if args.quantization is not None:
        args_dict["quantization"] = args.quantization
    if args.max_seq_len > 32768:
        args_dict["override_generation_config"] = {
            "attn_temperature_tuning": True,
        }
    return args_dict


def vllm_args(args):
    args_dict = vllm_args_dict(args)
    if args.kv_cache_dtype:
        args_dict["kv_cache_dtype"] = args.kv_cache_dtype
    return model_args_dict_to_str(args_dict)


def vllm_vlm_args(args):
    args_dict = vllm_args_dict(args)
    if args.kv_cache_dtype:
        args_dict["kv_cache_dtype"] = args.kv_cache_dtype
    args_dict.update({"max_images": 10})
    return model_args_dict_to_str(args_dict)


def launch_lm_eval(args):
    output_path = os.path.expanduser(args.output_path)
    tracker_args = simple_parse_args_string(f"output_path={output_path}")
    tracker = EvaluationTracker(**tracker_args)
    task_manager = setup_llama_tasks(args)

    simple_evaluate = partial(
        lm_eval.simple_evaluate,
        tasks=args.tasks,
        apply_chat_template=True,
        log_samples=True,
        evaluation_tracker=tracker,
        task_manager=task_manager,
        write_out=True,
        limit=args.limit,
    )

    if args.backend == "vllm":
        results = simple_evaluate(
            model="vllm",
            model_args=vllm_args(args),
            batch_size="auto",
        )

    elif args.backend == "vllm-vlm":
        results = simple_evaluate(
            model="vllm-vlm",
            model_args=vllm_vlm_args(args),
            batch_size="auto",
        )
    else:
        # args.backend = "http"
        results = simple_evaluate(
            model="local-chat-completions",
            model_args=http_args(args),
        )

    # Print results like lm-eval script
    print(make_table(results))
    print(f"----------from model {MODEL_PATH}----------------")

    # Save results for examination
    # TODO: would be nice if tracker returns the real path
    model_path = tracker.general_config_tracker.model_name_sanitized
    log_path = f"{tracker.output_path}/{model_path}/"
    logging.info("Dumping eval results to %s", log_path)
    samples = results.pop("samples")
    tracker.save_results_aggregated(results=results, samples=samples)
    for task_name, task_results in samples.items():
        tracker.save_results_samples(task_name=task_name, samples=task_results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="llama4_17b",
        help="Model name specified on vLLM server",
    )
    parser.add_argument("--quantization",
                        "-q",
                        type=str,
                        help="online quantization schema")
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        choices=["vllm", "vllm-vlm", "http"],
        default="vllm",
        help="start vllm locally or hit an http endpoint",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="~/lm_eval",
        help="Path for saving eval results and samples",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        nargs="+",
        type=str,
        default="meta_gpqa_diamond_cot_zeroshot",
        help="lm_eval tasks to run",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        "data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. "
        "ROCm (AMD GPU) supports fp8 (=fp8_e4m3)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Number of samples to run. Default is the entire dataset.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=32768,
        help="Max sequence length to pass to the model",
    )
    parser.add_argument(
        "--ruler-seq-lens",
        "-s",
        nargs="*",
        default=[4096, 8192, 16384, 32768, 65536, 128000],
        type=int,
        help="sequence lengths for long context eval dataset",
    )

    args = parser.parse_args()
    launch_lm_eval(args)
