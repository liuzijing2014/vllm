# SPDX-License-Identifier: Apache-2.0
import os

from datasets import load_dataset
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

# Select model and load it.
LLAMA_DIR = os.environ.get("LLAMA_DIR", "")
LLAMA_OUT_DIR = os.environ.get("LLAMA_OUT_DIR", "")
assert (
    LLAMA_DIR != ""
), "Please set the environment variable LLAMA_DIR to the path of the model directory."
assert (
    LLAMA_OUT_DIR != ""
), "Please set the environment variable LLAMA_OUT_DIR to the path of the model output directory."

# Load model.
print(f"Loading model... from {LLAMA_DIR}")
print(f"Will save model... to {LLAMA_OUT_DIR}")

model = AutoModelForCausalLM.from_pretrained(
    LLAMA_DIR,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID,
                  split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def process_and_tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per-tensor scales
#   * quantize the activations to fp8 with per-tensor scales
#   * quantize the kv cache to fp8 with per-tensor scales
recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore:
                - re:.*lm_head
                - re:.*self_attn
                - re:.*router
                - re:.*vision_model
                - re:.*multi_modal_projector
                - re:.*shared_expert
                - re:.*feed_forward.gate_proj
                - re:.*feed_forward.up_proj
                - re:.*feed_forward.down_proj
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: channel
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets: ["Linear"]
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""

# Apply algorithms.
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

print(
    "Running sample generation. ",
    "Note: Inference with the quantized kv_cache is not supported. ",
    "Please use vLLM for inference with the quantized kv_cache.",
)
# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is",
                      return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")
