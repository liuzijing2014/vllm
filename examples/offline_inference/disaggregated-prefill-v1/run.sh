rm -rf local_storage/
rm output.txt

VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 prefill_example.py
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=4,5,6,7 python3 decode_example.py
