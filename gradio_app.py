import gradio as gr
from utils import Vicuna

class GradioVicuna(Vicuna):
    def user_input(self, user_message, history):
        # 清空输入框，历史记录增加一组（用户消息，回复=None）
        return "", history + [[user_message, None]]
    
    def bot_predict(self, history):
        # 取history录前几条作为历史记录，取history最后一条用户消息作为输入，构建prompt
        history_text = '\n'.join([f'USER: {x[0]}\nASSISTANT: {x[1]}' for x in history[-5:-1]])
        content = history[-1][0]
        prompt = self.prompt(content, history_text)
        
        # 生成调用streaming_generate 方法得到生成器
        generate_params = dict(
            input_ids=self.tokenizer.encode(prompt, return_tensors="pt").to(self.cuda),
            min_length=0,
            max_length=2048,
            temperature=0.1,
            top_p=0.75,
            top_k=40)
        
        reply_stream = self.streaming_generate(**generate_params)
        
        # 流式更新history最后一条回复，并返回
        for each in reply_stream:
            reply = each.split('ASSISTANT: ')[-1].strip()
            history[-1][1] = reply
            yield history
    
vic = GradioVicuna(
    model_dir = 'models/chinese_vicuna',
    checkpoint='models/chinese_vicuna/vicuna-13B-1.1-Chinese-GPTQ-4bit-128g.safetensors'
)
    
with gr.Blocks() as app:
    # 聊天气泡组件
    chatbot = gr.Chatbot()
    
    # 文本输入框
    msg = gr.Textbox(placeholder = "请输入问题，按回车发送")
    
    # 按钮：清空
    clear = gr.Button("新建对话")
    
    # 提交逻辑：调用vic.user_input，将msg中文本与chatbot中历史记录输入，得到""清空msg，增加用户消息的history更新chatbot
    # 再调用vic.bot_predict，将chatbot中历史记录（含最后一次用户消息）作为输入，得到增加回复的history更新chatbot
    msg.submit(fn=vic.user_input, inputs=[msg, chatbot], outputs=[msg, chatbot]).then(
        fn=vic.bot_predict, inputs=chatbot, outputs=chatbot)
    
    # 点击清空按钮，chatbot的记录设为空列表
    clear.click(fn=lambda:[], outputs=chatbot)
    
app.queue().launch(server_name='0.0.0.0', server_port=8888)