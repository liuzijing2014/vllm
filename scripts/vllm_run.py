# SPDX-License-Identifier: Apache-2.0

import os

from vllm import LLM, SamplingParams

LLAMA_OUT_DIR = os.environ.get("LLAMA_OUT_DIR", "")
assert (
    LLAMA_OUT_DIR != ""
), "Please set the environment variable LLAMA_OUT_DIR to the path of the model output directory."

print(f"zjl: Load model from {LLAMA_OUT_DIR}")


def test():
    # Sample prompts.
    prompts = [
        "The color of the sky is blue but sometimes it can also be",
        "The capital of France is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.9, top_p=0.6, max_tokens=256)

    # Create an LLM.
    llm = LLM(
        model=LLAMA_OUT_DIR,
        tensor_parallel_size=8,
        max_model_len=32768,
    )
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("========== SAMPLE GENERATION ==============")
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("==========================================")
    print(f"from model {LLAMA_OUT_DIR}")
    print("==========================================")


if __name__ == "__main__":
    test()
