# SPDX-License-Identifier: Apache-2.0
import os
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

GPU_MEMORY_UTILIZATION = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", 0.9)
LLAMA_DIR = os.environ.get("LLAMA_DIR", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
MAX_MODEL_LENGTH = os.environ.get("MAX_MODEL_LENGTH", 8192)

# context = "Hi " * 1000
# context2 = "Hey " * 500
# prompts = [
#     context + "Hello, my name is",
#     context + "The capital of France is",
#     context2 + "Your name is",
#     context2 + "The capital of China is",
# ]
prompts = [
    "The capital of France is",
    "The color of the sky is blue but sometimes it can also be",
]

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

llm = LLM(model=LLAMA_DIR,
          enforce_eager=True,
          tensor_parallel_size=4,
          gpu_memory_utilization=float(GPU_MEMORY_UTILIZATION),
          max_model_len=MAX_MODEL_LENGTH,
          kv_transfer_config=KVTransferConfig.from_cli(
              '{"kv_connector":"SharedStorageConnector","kv_role":"kv_both", '
              '"kv_connector_extra_config": '
              '{"shared_storage_path": "local_storage"}}')
          )  #, max_model_len=2048, max_num_batched_tokens=2048)

# 1ST generation (prefill instance)
outputs = llm.generate(
    prompts,
    sampling_params,
)

new_prompts = []
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    new_prompts.append(prompt + generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# Write new_prompts to output.txt
with open("output.txt", "w") as f:
    for prompt in new_prompts:
        f.write(prompt + "\n")
print(f"Saved {len(new_prompts)} prompts to output.txt")
