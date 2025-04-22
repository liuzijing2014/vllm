# SPDX-License-Identifier: Apache-2.0
import os
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

GPU_MEMORY_UTILIZATION = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", 0.9)
LLAMA_DIR = os.environ.get("LLAMA_DIR", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
MAX_MODEL_LENGTH = os.environ.get("MAX_MODEL_LENGTH", 8192)

# Read prompts from output.txt
prompts = []
try:
    with open("output.txt") as f:
        for line in f:
            prompts.append(line.strip())
    print(f"Loaded {len(prompts)} prompts from output.txt")
except FileNotFoundError:
    print("Error: output.txt file not found")
    exit(-1)

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

llm = LLM(
    model=LLAMA_DIR,
    enforce_eager=True,
    gpu_memory_utilization=float(GPU_MEMORY_UTILIZATION),
    max_num_batched_tokens=64,
    max_num_seqs=16,
    tensor_parallel_size=4,
    max_model_len=MAX_MODEL_LENGTH,
    kv_transfer_config=KVTransferConfig.from_cli(
        '{"kv_connector":"SharedStorageConnector","kv_role":"kv_both",'
        '"kv_connector_extra_config": {"shared_storage_path": "local_storage"}}'
    ))  #, max_model_len=2048, max_num_batched_tokens=2048)

# 1ST generation (prefill instance)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
