#!/bin/bash
export HTTPS_PROXY=http://fwdproxy:8080
export HTTP_PROXY=http://fwdproxy:8080
export FTP_PROXY=http://fwdproxy:8080
export https_proxy=http://fwdproxy:8080
export http_proxy=http://fwdproxy:8080
export ftp_proxy=http://fwdproxy:8080
export no_proxy=".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fb,127.0.0.1,localhost"

GIT_BASE=${GIT_BASE:-/data/users/$USER/gitrepos}
MODEL_DIR=${LLAMA_OUT_DIR:-"meta-llama/Llama-4-Scout-17B-16E-Instruct"}
TP=${TP:-"8"}
model_name=$(basename "$MODEL_DIR")

base_command="python $GIT_BASE/liuzijing2014/vllm/scripts/test_lm_eval.py -tp ${TP}"

# Run texst eval
mmlu_pro_command="$base_command --tasks mmlu_pro 2>&1 | tee ${model_name}_text_eval_$(date +%Y%m%d_%H%M).log"

echo "$mmlu_pro_command"
eval "$mmlu_pro_command"

# Run mm eval
chartqa_command="$base_command --backend vllm-vlm --tasks chartqa 2>&1 | tee ${model_name}_mm_eval_$(date +%Y%m%d_%H%M).log"

echo "$chartqa_command"
eval "$chartqa_command"
