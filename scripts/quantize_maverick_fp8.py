# SPDX-License-Identifier: Apache-2.0
import os

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers.finetune import oneshot
from transformers import AutoTokenizer, Llama4ForCausalLM

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

model = Llama4ForCausalLM.from_pretrained(LLAMA_DIR,
                                          device_map="auto",
                                          torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)
# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
ignore_list = [
    "re:.*lm_head",
    "re:.*self_attn",
    "re:.*router",
    "re:.*vision_model",
    "re:.*multi_modal_projector",
    "re:.*shared_expert",
    "re:.*feed_forward.gate_proj",
    "re:.*feed_forward.up_proj",
    "re:.*feed_forward.down_proj",
]
recipe = QuantizationModifier(targets="Linear",
                              scheme="FP8_DYNAMIC",
                              ignore=ignore_list)

# Apply quantization.
oneshot(
    model=model,
    tokenizer=tokenizer,
    recipe=recipe,
    save_compressed=True,
    output_dir=LLAMA_OUT_DIR,
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is",
                      return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("===========================================")
