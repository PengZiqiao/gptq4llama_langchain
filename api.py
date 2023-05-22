from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from model import GPTQModel
from langchain.prompts import load_prompt
from config import HUMAN_PREFIX, AI_PREFIX
from langchain.memory import ConversationBufferMemory

app = FastAPI()

# 解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 载入模型
from config import AUTO_TYPE, MODEL_PARAMS
gptq = GPTQModel(AUTO_TYPE, **MODEL_PARAMS)


class GenerateParams(BaseModel):
    prompt: str
    params: Union[dict, None] = None


@app.post("/generate/")
async def generate(item: GenerateParams):
    return gptq(item.prompt, streaming=False, **item.params)


@app.post("/streaming_generate/")
async def streaming_generate(item: GenerateParams):
    return EventSourceResponse(
        gptq(item.prompt, streaming=True, **item.params), media_type="text/event-stream"
    )


@app.post("/chat/")
async def chat(history: list[list[str]]):
    # 构建prompt
    memory = ConversationBufferMemory(human_prefix=HUMAN_PREFIX, ai_prefix=AI_PREFIX)
    for human_text, ai_text in history[-10:-1]:
        memory.save_context({'input':human_text}, {'output':ai_text})
    history_text = memory.buffer

    template = load_prompt("prompts/conversation.json", )
    prompt = template.format(human_prefix=HUMAN_PREFIX, ai_prefix=AI_PREFIX, history=history_text, input=history[-1][0])

    # 构建生成参数
    params = dict(min_length=0, max_length=2048, num_beams=10, temperature=0.1, top_p=0.75, top_k=40)

    return EventSourceResponse(
        gptq(prompt, streaming=True, **params), media_type="text/event-stream"
    )


@app.post("/embed/")
async def embed(item: GenerateParams):
    return gptq.embed(item.prompt)