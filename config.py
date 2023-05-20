LLM_HOST = "159.224.1.168"
LLM_PORT = "8887"

UVICORN_RELOAD = True
UVICORN_LOGLEVEL = "info"
UVICORN_WORKERS = 16

MILVUS_HOST = "159.224.1.168"
MILVUS_PORT = "19530"

# # chinese vicuna
MODEL_DIR = "/data02/it/models/chinese_vicuna"
CHECKPOINT = (
    "/data02/it/models/chinese_vicuna/vicuna-13B-1.1-Chinese-GPTQ-4bit-128g.safetensors"
)

# belle bloom
# MODEL_DIR = "/data02/it/models/BELLE_BLOOM_GPTQ_4BIT"
# CHECKPOINT = (
#     "/data02/it/models/BELLE_BLOOM_GPTQ_4BIT/bloom7b-2m-4bit-128g.pt"
# )

# GPT4-X-Alpasta
# MODEL_DIR = "/data02/it/models/GPT4-X-Alpasta-30b-4bit"
# CHECKPOINT = (
#     "/data02/it/gptq4llama_langchain/models/GPT4-X-Alpasta-30b-4bit/gpt4-x-alpasta-30b-128g-4bit.safetensors"
# )

# moss
# MODEL_DIR = "/data02/it/gptq4llama_langchain/models/moss-moon-003-sft-int4"
# CHECKPOINT = (
#     "/data02/it/gptq4llama_langchain/models/moss-moon-003-sft-int4/pytorch_model.bin"
# )

GENERATE_PARAMS = dict(
    min_length=0, max_length=200, num_beams=10, temperature=0.1, top_p=0.75, top_k=50, repetition_penalty=1.2
)