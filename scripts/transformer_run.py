import os

from transformers import AutoTokenizer, Llama4ForConditionalGeneration


LLAMA_OUT_DIR = os.environ.get("LLAMA_OUT_DIR", "")
assert (
    LLAMA_OUT_DIR != ""
), "Please set the environment variable LLAMA_OUT_DIR to the path of the model output directory."

print(f"zjl: Load model from {LLAMA_OUT_DIR}")

model = Llama4ForConditionalGeneration.from_pretrained(
    LLAMA_OUT_DIR,
    torch_dtype="auto",
    device_map="auto",
)
model.generation_config.top_p = 0.9
model.generation_config.temperature = 0.6
tokenizer = AutoTokenizer.from_pretrained(LLAMA_OUT_DIR)


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
print(f"from model {LLAMA_OUT_DIR}")
print("==========================================")
