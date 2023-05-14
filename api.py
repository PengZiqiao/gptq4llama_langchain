from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from model import Vicuna
from langchain.prompts import load_prompt
from config import MODEL_DIR, CHECKPOINT


app = FastAPI()

# 解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 载入模型
vic = Vicuna(
    model_dir=MODEL_DIR,
    checkpoint=CHECKPOINT,
)


class GenerateParams(BaseModel):
    prompt: str
    params: Union[dict, None] = None


@app.post("/generate/")
async def generate(item: GenerateParams):
    return vic(item.prompt, streaming=False, **item.params)


@app.post("/streaming_generate/")
async def streaming_generate(item: GenerateParams):
    return EventSourceResponse(
        vic(item.prompt, streaming=True, **item.params), media_type="text/event-stream"
    )


@app.post("/chat/")
async def chat(history: list[list[str]]):
    # 构建prompt
    history_text = "\n".join(
        [f"USER: {x[0]}\nASSISTANT: {x[1]}" for x in history[-8:-1]]
    )
    content = history[-1][0]

    template = load_prompt("prompts/conversation.json")
    prompt = template.format(history=history_text, input=content)

    # 构建生成参数
    params = dict(min_length=0, max_length=2048, temperature=0.1, top_p=0.75, top_k=40)

    return EventSourceResponse(
        vic(prompt, streaming=True, **params), media_type="text/event-stream"
    )


@app.post("/embed/")
async def embed(item: GenerateParams):
    return vic.embed(item.prompt)