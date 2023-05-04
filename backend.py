from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from utils import StreamlitVicuna

app = FastAPI()

# 解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 载入模型
vic = StreamlitVicuna(
        model_dir = 'models/chinese_vicuna',
        checkpoint='models/chinese_vicuna/vicuna-13B-1.1-Chinese-GPTQ-4bit-128g.safetensors')


# 提供一个流式返回回复的api
@app.post("/chat/")
async def chat(history:list[list[str]]):
    return EventSourceResponse(vic.bot_predict(history), media_type='text/event-stream')
