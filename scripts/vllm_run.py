# SPDX-License-Identifier: Apache-2.0

import os

from vllm import LLM, SamplingParams

LLAMA_OUT_DIR = "/data/users/zijingliu/cp/llama4/16e-int4-gw82" # os.environ.get("LLAMA_OUT_DIR", "")
assert (
    LLAMA_OUT_DIR != ""
), "Please set the environment variable LLAMA_OUT_DIR to the path of the model output directory."

print(f"zjl: Load model from {LLAMA_OUT_DIR}")


def test():
    # Sample prompts.
    prompts = [
        "<|begin_of_text|><|header_start|>system<|header_end|>\n\nThe following are multiple choice questions (with answers) about business. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.<|eot|><|header_start|>user<|header_end|>\n\nQuestion:\nIn contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .\nOptions:\nA. Boycotts, Buyalls, Blockchain technology, Increased Sales\nB. Buycotts, Boycotts, Digital technology, Decreased Sales\nC. Boycotts, Buycotts, Digital technology, Decreased Sales\nD. Buycotts, Boycotts, Blockchain technology, Charitable donations\nE. Boycotts, Buyalls, Blockchain technology, Charitable donations\nF. Boycotts, Buycotts, Digital technology, Increased Sales\nG. Buycotts, Boycotts, Digital technology, Increased Sales\nH. Boycotts, Buycotts, Physical technology, Increased Sales\nI. Buycotts, Buyalls, Blockchain technology, Charitable donations\nJ. Boycotts, Buycotts, Blockchain technology, Decreased Sales\nAnswer: Let's think step by step. We refer to Wikipedia articles on business ethics for help. The sentence that best uses the possible options above is __n contrast to *boycotts*, *buycotts* aim to reward favourable behavior by companies. The success of such campaigns have been heightened through the use of *digital technology*, which allow campaigns to facilitate the company in achieving *increased sales*._ The answer is (F).\n\nQuestion:\n_______ is the direct attempt to formally or informally manage ethical issues or problems, through specific policies, practices and programmes.\nOptions:\nA. Operational management\nB. Corporate governance\nC. Environmental management\nD. Business ethics management\nE. Sustainability\nF. Stakeholder management\nG. Social marketing\nH. Human resource management\nI. N/A\nJ. N/A\nAnswer: Let's think step by step. We refer to Wikipedia articles on business ethics for help. The direct attempt manage ethical issues through specific policies, practices, and programs is business ethics management. The answer is (D).\n\nQuestion:\nHow can organisational structures that are characterised by democratic and inclusive styles of management be described?\nOptions:\nA. Flat\nB. Bureaucratic\nC. Autocratic\nD. Hierarchical\nE. Functional\nF. Decentralized\nG. Matrix\nH. Network\nI. Divisional\nJ. Centralized\nAnswer: Let's think step by step. We refer to Wikipedia articles on management for help. Flat organizational structures are characterized by democratic and inclusive styles of management, and have few (if any) levels of management between the workers and managers.  The answer is (A).\n\nQuestion:\nAlthough the content and quality can be as controlled as direct mail, response rates of this medium are lower because of the lack of a personal address mechanism. This media format is known as:\nOptions:\nA. Online banners.\nB. Television advertising.\nC. Email marketing.\nD. Care lines.\nE. Direct mail.\nF. Inserts.\nG. Door to door.\nH. Radio advertising.\nI. Billboards.\nJ. Social media advertising.\nAnswer: Let's think step by step. We refer to Wikipedia articles on marketing for help. Door to door marketing delivers non-addressed items within all buildings within a geographic area. While it can control the content and quality as well as direct mail marketing, its response rate is lower because of the lack of a personal address mechanism. The answer is (G).\n\nQuestion:\nIn an organization, the group of people tasked with buying decisions is referred to as the _______________.\nOptions:\nA. Procurement centre.\nB. Chief executive unit.\nC. Resources allocation group.\nD. Marketing department.\nE. Purchasing department.\nF. Supply chain management team.\nG. Outsourcing unit.\nH. Decision-making unit.\nI. Operations unit.\nJ. Financial management team.\nAnswer: Let's think step by step. We refer to Wikipedia articles on marketing for help. In an organization, the group of the people tasked with buying decision is referred to as the decision-making unit. The answer is (H).\n\nQuestion:\nTypical advertising regulatory bodies suggest, for example that adverts must not: encourage _________, cause unnecessary ________ or _____, and must not cause _______ offence.\nOptions:\nA. Safe practices, Fear, Jealousy, Trivial\nB. Unsafe practices, Distress, Joy, Trivial\nC. Safe practices, Wants, Jealousy, Trivial\nD. Safe practices, Distress, Fear, Trivial\nE. Unsafe practices, Wants, Jealousy, Serious\nF. Safe practices, Distress, Jealousy, Serious\nG. Safe practices, Wants, Fear, Serious\nH. Unsafe practices, Wants, Fear, Trivial\nI. Unsafe practices, Distress, Fear, Serious\nAnswer: Let's think step by step.<|eot|><|header_start|>assistant<|header_end|>\n\n"
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    # Create an LLM.
    llm = LLM(
        model=LLAMA_OUT_DIR,
        tensor_parallel_size=1,
        max_model_len=4096,
        enforce_eager=True,
    )
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("========== SAMPLE GENERATION ==============")
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"\n\nGenerated text: {generated_text!r}")
    print("==========================================")
    print(f"from model {LLAMA_OUT_DIR}")
    print("==========================================")


if __name__ == "__main__":
    test()
