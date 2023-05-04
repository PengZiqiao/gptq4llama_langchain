import torch
from safetensors.torch import load_file as safe_load
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from gptq4llama.utils.modelutils import find_layers
from gptq4llama import quant
from callbacks import Iteratorize, Stream


def load_quant(model, checkpoint, wbits, groupsize=-1, faster_kernel=False, exclude_layers=['lm_head'], kernel_switch_threshold=128, eval=True):

    def noop(*args, **kwargs):
        pass

    config = AutoConfig.from_pretrained(model)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model.eval()
    
    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]


    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    model.load_state_dict(safe_load(checkpoint), strict=False)

    model.seqlen = 2048
    print('Done.')

    return model


def load_model(self): 
    model_dir = 'models/chinese_vicuna'
    checkpoint = 'models/chinese_vicuna/vicuna-13B-1.1-Chinese-GPTQ-4bit-128g.safetensors'
    wbits = 4
    groupsize = 128   

    model = load_quant(model_dir, checkpoint, wbits, groupsize).to(self.cuda)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer


class Vicuna:
    def __init__(self, model_dir, checkpoint):
        self.cuda = torch.device('cuda:0')
        self.model, self.tokenizer = self.load_model(model_dir, checkpoint)
        
    def load_model(self, model_dir, checkpoint): 
        wbits = 4
        groupsize = 128   

        model = load_quant(model_dir, checkpoint, wbits, groupsize).to(self.cuda)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

        return model, tokenizer
    
    def prompt(self, content, history):
        prompt = f"""A chat between a user and an assistant.
        
        {history}
        USER: {content}
        ASSISTANT: """
        
        return prompt
        
    def generate(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.cuda)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                min_length=0,
                max_length=512,
                top_p=0.1,
                top_k=40,
                temperature=0.7
        )
            
        output_ids = generated_ids[0]
        reply_ids = output_ids[len(input_ids[0]):]

        reply = self.tokenizer.decode(reply_ids, skip_special_tokens=True)
        return reply
        
    def streaming_generate(self, **generate_params):
        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", transformers.StoppingCriteriaList()
            )
            kwargs["stopping_criteria"].append(
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                self.model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                decoded_output = self.tokenizer.decode(output)

                if output[-1] in [self.tokenizer.eos_token_id]:
                    break

                yield decoded_output

class StreamlitVicuna(Vicuna):
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
                yield reply

