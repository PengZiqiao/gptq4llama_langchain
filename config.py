LLM_HOST = "159.224.1.168"
LLM_PORT = "8887"

UVICORN_RELOAD = True
UVICORN_LOGLEVEL = "info"
UVICORN_WORKERS = 16

MILVUS_HOST = "159.224.1.168"
MILVUS_PORT = "19530"

MODEL_DIR = "/data02/it/models/chinese_vicuna"
CHECKPOINT = (
    "/data02/it/models/chinese_vicuna/vicuna-13B-1.1-Chinese-GPTQ-4bit-128g.safetensors"
)

GENERATE_PARAMS = dict(
    min_length=0, max_length=4096, temperature=0.1, top_p=0.75, top_k=40
)