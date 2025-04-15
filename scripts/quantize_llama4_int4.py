# SPDX-License-Identifier: Apache-2.0
import os

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoTokenizer, Llama4ForConditionalGeneration


def open_full_model(file):
    from safetensors import safe_open

    # Open the safetensor file using a context manager
    with safe_open(file, framework="pt") as f:
        # Iterate over all keys (parameters) in the file
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"Key: {key} | Dtype: {tensor.dtype} | Shape: {tuple(tensor.shape)}")


def print_full_model_info(model):
    print("=== Module Hierarchy with Parameters ===")
    for module_name, module in model.named_modules():
        display_name = module_name if module_name else "(root)"
        print(f"Module: {display_name:<20} Type: {type(module).__name__}")

        for param_name, param in module.named_parameters(recurse=False):
            full_param_name = (
                f"{display_name}.{param_name}"
                if display_name != "(root)"
                else param_name
            )
            print(
                f"    Parameter: {full_param_name:<40} Type: {type(param).__name__:<10} Dtype: {param.dtype}"
            )


def get_device_map():
    import torch
    from llmcompressor.transformers.compression.helpers import (
        calculate_offload_device_map,
    )

    return calculate_offload_device_map(
        LLAMA_DIR,
        reserve_for_hessians=True,
        num_gpus=8,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        model_cls=Llama4ForConditionalGeneration,
    )


LLAMA_DIR = os.environ.get("LLAMA_DIR", "")
LLAMA_OUT_DIR = os.environ.get("LLAMA_OUT_DIR", "")
assert (
    LLAMA_DIR != ""
), "Please set the environment variable LLAMA_DIR to the path of the model directory."
assert (
    LLAMA_OUT_DIR != ""
), "Please set the environment variable LLAMA_OUT_DIR to the path of the model output directory."

print(f"Load model from {LLAMA_DIR} and save model to {LLAMA_OUT_DIR}")


model = Llama4ForConditionalGeneration.from_pretrained(
    LLAMA_DIR,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 2048

# # Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

recipe = GPTQModifier(
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
                observer="mse",
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
    dampening_frac=0.05,
)
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=True,
    trust_remote_code_model=True,
    output_dir=LLAMA_OUT_DIR,
)

# # Confirm generations of the quantized model look sane.
# # Generation is broken for deepseek models when using the latest transformers package
print("========== SAMPLE GENERATION ==============")
prompts = [
    "The color of the sky is blue but sometimes it can also be",
    "The capital of France is",
]

for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=256)
    print(tokenizer.decode(output[0]))
print("==========================================")
print(f"saved model to {LLAMA_OUT_DIR}")
print("==========================================")
