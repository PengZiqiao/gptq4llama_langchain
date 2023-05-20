import streamlit as st
import requests
from sseclient import SSEClient
import json
from pathlib import Path

import sys
sys.path.append('..')
from config import LLM_HOST, LLM_PORT

def request_chat(history):
    # 构建请求参数
    url = f"http://{LLM_HOST}:{LLM_PORT}/chat/"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(history)

    # 请求后端api, 并反回流式响应
    res = requests.post(url, headers=headers, data=data, stream=True)
    client = SSEClient(res).events()
    return client


def chat():
    # 获得用户输入，回显在聊天记录中，并清空输入栏
    content = st.session_state["content"]
    st.session_state["history"].append([content, ""])
    st.session_state["content"] = ""

    # 请求后端，获得回复
    st.session_state["reply"] = request_chat(
        st.session_state["history"]
    )


def clear():
    st.session_state["content"] = ""
    st.session_state["history"] = []
    st.session_state["reply"] = []


# 初始化
if "content" not in st.session_state:
    st.session_state["log"] = Path('log.txt')
    clear()

# 输入框和清空按钮
user_input = st.text_input("请输入问题，按回车发送", on_change=chat, key="content")
clear_btn = st.button("新建对话", on_click=clear)

# 显示聊天历史记录
for each in st.session_state["history"][:-1]:
    st.write(f':orange[{each[0]}]')
    st.write(each[1])

if st.session_state["history"]:
    # 显示最后一次用户输入
    user_input = st.session_state['history'][-1][0]
    st.write(f":orange[{user_input}]") 
    # 流式输出AI回复
    final_reply = st.empty()
    for each in st.session_state["reply"]:
        reply = each.data
        final_reply.write(reply)
        st.session_state["history"][-1][1] = reply

    # 写入日志
    text = st.session_state["log"].read_text()
    text += f'{user_input}\n{reply}\n'
    st.session_state["log"].write_text(text)