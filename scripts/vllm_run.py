# SPDX-License-Identifier: Apache-2.0

import argparse
import os

from vllm import LLM, SamplingParams

LLAMA_OUT_DIR = os.environ.get("LLAMA_OUT_DIR", "")
assert (
    LLAMA_OUT_DIR != ""
), "Please set the environment variable LLAMA_OUT_DIR to the path of the model output directory."

print(f"Load model from {LLAMA_OUT_DIR}")


def test(args):
    # Sample prompts.
    prompts = [
        "The capital of France is ",
        "The color of the sky is blue but sometimes it can also be",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    # Create an LLM.
    llm = LLM(
        model=LLAMA_OUT_DIR,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=8196,
        enforce_eager=True,
    )
    print(f"========== Prompt ==============")
    print(f"Prompt: {prompts[0]!r}")
    print("==========================================")

    for batch_size in args.batch_sizes:
        batched_prompts = prompts * batch_size
        outputs = llm.generate(batched_prompts, sampling_params)
        print(f"========== Batch{(batch_size)} ==============")
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"\n\nGenerated text: {generated_text!r}")
        print("==========================================")

    print(f"from model {LLAMA_OUT_DIR}")
    print("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--batch-sizes",
        "-b",
        nargs="*",
        default=[1, 2, 4, 8, 16, 32],
        type=int,
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        default=8,
        type=int,
    )
    args = parser.parse_args()
    test(args)
