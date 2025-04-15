import argparse
import logging
import os
from functools import partial

import lm_eval
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from lm_eval.loggers import EvaluationTracker
from lm_eval.utils import make_table, simple_parse_args_string
from transformers import AutoTokenizer, Llama4ForConditionalGeneration

logging.basicConfig(
    filename="auto_quantize.log",  # Log output file
    filemode="a",  # Append mode
    level=logging.CRITICAL,  # Log messages with level INFO and above
    format="%(asctime)s - %(message)s",  # Log message format
)
from datetime import datetime

GPU_MEMORY_UTILIZATION = 0.9
MAX_OUTPUT_LEN = 2048
MAX_MODEL_LENGTH = 32768


def model_args_dict_to_str(model_args):
    return ",".join(f"{k}={v}" for k, v in model_args.items())


def vllm_args_dict(outpur_model_dir):
    args_dict = {
        "pretrained": outpur_model_dir,
        "max_length": MAX_MODEL_LENGTH,
        "tensor_parallel_size": 8,
        "dtype": "auto",
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_gen_toks": MAX_OUTPUT_LEN,
        "seed": 0,
    }
    return args_dict


def vllm_args(outpur_model_dir):
    args_dict = vllm_args_dict(outpur_model_dir)
    return model_args_dict_to_str(args_dict)


def vllm_vlm_args(outpur_model_dir):
    args_dict = vllm_args_dict(outpur_model_dir)
    args_dict.update({"max_images": 10})
    return model_args_dict_to_str(args_dict)


def run_eval(outpur_model_dir, model_name):
    text_evals = ["mmlu_pro"]
    mm_evals = ["chartqa"]

    output_path = os.path.expanduser(args.output_path)
    tracker_args = simple_parse_args_string(f"output_path={output_path}")
    tracker = EvaluationTracker(**tracker_args)
    task_manager = lm_eval.tasks.TaskManager(
        include_defaults=True,
    )

    for task in text_evals:
        simple_evaluate = partial(
            lm_eval.simple_evaluate,
            tasks=task,
            apply_chat_template=True,
            log_samples=True,
            evaluation_tracker=tracker,
            task_manager=task_manager,
            write_out=True,
        )

        logging.critical(f"==========Text eval for {model_name}: {task}==========")
        results = simple_evaluate(
            model="vllm",
            model_args=vllm_args(outpur_model_dir),
            batch_size="auto",
        )

        # Print results like lm-eval script
        logging.critical(make_table(results))

        samples = results.pop("samples")
        tracker.save_results_aggregated(results=results, samples=samples)
        for task_name, task_results in samples.items():
            tracker.save_results_samples(task_name=task_name, samples=task_results)

    for task in mm_evals:
        simple_evaluate = partial(
            lm_eval.simple_evaluate,
            tasks=task,
            apply_chat_template=True,
            log_samples=True,
            evaluation_tracker=tracker,
            task_manager=task_manager,
            write_out=True,
        )

        logging.critical(f"==========MM eval for {model_name}: {task}==========")
        results = simple_evaluate(
            model="vllm-vlm",
            model_args=vllm_vlm_args(outpur_model_dir),
            batch_size="auto",
        )

        # Print results like lm-eval script
        logging.critical(f"\n{make_table(results)}")

        samples = results.pop("samples")
        tracker.save_results_aggregated(results=results, samples=samples)
        for task_name, task_results in samples.items():
            tracker.save_results_samples(task_name=task_name, samples=task_results)


def sample_generate(model, tokenizer, model_name):
    logging.critical(f"========== SAMPLE GENERATION {model_name} ==============")
    prompts = [
        "The color of the sky is blue but sometimes it can also be",
        "The capital of France is",
    ]

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        output = model.generate(input_ids, max_new_tokens=256)
        logging.critical(tokenizer.decode(output[0]))
    logging.critical("==========================================")


def get_recipe(observer, dampening_frac):
    return GPTQModifier(
        targets="Linear",
        config_groups={
            "config_group": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=128,
                    symmetric=True,
                    dynamic=False,
                    actorder="weight",
                    observer=observer,
                ),
            ),
        },
        ignore=[
            "re:.*lm_head",
            "re:.*self_attn",
            "re:.*router",
            "re:.*vision_model",
            "re:.*multi_modal_projector",
        ],
        dampening_frac=dampening_frac,
    )


def quantize(args, model, tokenizer):
    calibration_samples = [1024, 2048]
    max_seq_length = [2048]
    dampening_fracs = [0.01]
    model_name = os.path.basename(args.base_model_path)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example[args.text_col_name], tokenize=False, add_generation_prompt=True
            )
        }

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            truncation=True,
            add_special_tokens=False,
        )

    ds = load_dataset(args.dataset, split=args.dataset_split).shuffle(seed=args.seed)

    idx = 0
    for num_calibration_samples in calibration_samples:
        for max_length in max_seq_length:
            selected_ds = ds.select(range(num_calibration_samples))
            selected_ds = selected_ds.map(preprocess).map(
                tokenize, remove_columns=ds.column_names
            )
            for dampening_frac in dampening_fracs:
                recipe = get_recipe(args.observer, dampening_frac)
                new_model_name = (
                    f"{model_name}_{datetime.now().strftime("%Y-%m-%d-%H:%M")}"
                )
                model_out_dir = os.path.join(
                    args.outpur_model_dir,
                    new_model_name,
                )
                os.makedirs(model_out_dir, exist_ok=True)
                logging.critical(
                    "===================================================================="
                )
                logging.critical(f"Loop{idx}: Quantize {model_name} with settings:")
                logging.critical(
                    f"\n    - {args.observer=}\n    - {num_calibration_samples=}\n  - {max_length=}\n   - {dampening_frac=}\n   - {args.seed=}"
                )
                logging.critical(f"Output dir: {model_out_dir}")
                oneshot(
                    model=model,
                    dataset=selected_ds,
                    recipe=recipe,
                    max_seq_length=max_length,
                    num_calibration_samples=num_calibration_samples,
                    save_compressed=True,
                    trust_remote_code_model=True,
                    output_dir=model_out_dir,
                )
                sample_generate(model, tokenizer, model_name)
                run_eval(model_out_dir, new_model_name)
                idx += 1
                logging.critical(
                    "===================================================================="
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--base_model_path",
        "-b",
        type=str,
    )
    parser.add_argument(
        "--outpur_model_dir",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="~/lm_eval",
        help="Path for saving eval results and samples",
    )
    parser.add_argument(
        "--seed",
        "-s",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceH4/ultrachat_200k",
        type=str,
    )
    parser.add_argument(
        "--dataset_split",
        default="train_sft",
        type=str,
    )
    parser.add_argument(
        "--text_col_name",
        default="messages",
        type=str,
    )
    parser.add_argument(
        "--observer",
        default="mse",
        type=str,
    )

    args = parser.parse_args()

    model = Llama4ForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    quantize(args, model, tokenizer)
