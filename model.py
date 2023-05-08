
import traceback
from queue import Queue
from threading import Thread

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file as safe_load

"""
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
ln -s GPTQ-for-LLaMa ${YOUR_PYTHON_SITEPACKAGES_DIR}/gptq4llama
"""
from gptq4llama.utils.modelutils import find_layers
from gptq4llama import quant

"""
Helpers to load GPTQ LLaMa model and support streaming generate output.
Borrowed from https://github.com/oobabooga/text-generation-webui/
modules/GPTQ_loader.py
modules/callbacks.py
"""

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True


def load_quant(model, checkpoint, wbits=4, groupsize=128, exclude_layers=["lm_head"]):
    """
    This function is a replacement for the load_quant function in the
    GPTQ-for-LLaMa repository. It supports more models and branches.
    """

    def noop(*args, **kwargs):
        pass

    # 覆盖了一些随机初始化权重的torch函数，因为它们对于量化是不需要的。
    config = AutoConfig.from_pretrained(model)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    # 默认数据类型设置为半精度（torch.half），以减少内存使用
    # 创建模型后，恢复为单精度（torch.float），设置为评估模式（model.eval()）
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model.eval()

    # 找到模型中的所有线性层，排除一些不需要量化的层
    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]
    # 对线性层进行量化
    quant.make_quant_linear(model, layers, wbits, groupsize)
    del layers

    print("Loading model ...")
    model.load_state_dict(safe_load(checkpoint), strict=False)
    quant.make_quant_attn(model)
    model.seqlen = 2048
    print("Done.")

    return model


class Vicuna:
    def __init__(self, model_dir, checkpoint, **kwargs):
        self.model, self.tokenizer = self.load_model(model_dir, checkpoint, **kwargs)

    def load_model(self, model_dir, checkpoint, **kwargs):
        model = load_quant(model_dir, checkpoint, **kwargs).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

        return model, tokenizer
    
    def __call__(self, prompt, streaming=False, **kwargs):
        # 构建参数
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        generate_params = dict(input_ids=input_ids, **kwargs)

        if streaming:
            # 定义一个流式生成方法，
            def streaming_generate(**generate_params):
                def generate_with_callback(callback=None, **kwargs):
                    kwargs.setdefault("stopping_criteria", transformers.StoppingCriteriaList())
                    kwargs["stopping_criteria"].append(Stream(callback_func=callback))
                    with torch.no_grad():
                        self.model.generate(**kwargs)

                def generate_with_streaming(**kwargs):
                    return Iteratorize(generate_with_callback, kwargs, callback=None)

                with generate_with_streaming(**generate_params) as generator:
                    for output_ids in generator:
                        # 生成结束符时退出
                        if output_ids[-1] in [self.tokenizer.eos_token_id]:
                            break

                        # 去掉结果中的prompt，只保留回复部分
                        output_ids = output_ids[len(input_ids[0]):]
                        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                        yield output

            # 调用流式生成方法，返回生成器
            output_stream = streaming_generate(**generate_params)
            return output_stream
        else:
            # 不使用流式生成，直接调用模型的 generate 方法
            with torch.no_grad():
                output_ids = self.model.generate(**generate_params)
           
            # 去掉结果中的prompt，只保留回复部分
            output_ids = output_ids[0]
            output_ids = output_ids[len(input_ids[0]):]
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return output
        
    def embed(self, input_text):
        input_ids = self. tokenizer(input_text, return_tensors='pt').to(torch.device('cuda'))

        with torch.no_grad():
            model_output = self.model(**input_ids, output_hidden_states=True)

        # 获得attention_mask和token_embeddings，扩展attention_mask匹配token_embeddings大小
        attention_mask = input_ids.attention_mask
        token_embeddings = model_output.hidden_states[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # 以attention_mask为权重，计算token_embeddings的加权平均作为embeddings
        # 平均token_embeddings可以捕捉文本的整体语义信息，也可以减少计算量和内存消耗
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return embeddings.tolist()[0]


"""
使用requests调用fastapi接口
"""

import requests
import json
from sseclient import SSEClient

from config import LLM_HOST, LLM_PORT

def generate(prompt):
    url = f"http://{LLM_HOST}:{LLM_PORT}/generate/"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(dict(prompt=prompt, params=dict(min_length=0, max_length=2048, temperature=0.1, top_p=0.75, top_k=40)))

    res = requests.post(url, headers=headers, data=data)
    return res.text

def streaming_generate(prompt):
    url = f"http://{LLM_HOST}:{LLM_PORT}/streaming_generate/"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(dict(prompt=prompt, params=dict(min_length=0, max_length=2048, temperature=0.1, top_p=0.75, top_k=40)))

    res = requests.post(url, headers=headers, data=data, stream=True)
    client = SSEClient(res).events()
    # 解析返回数据，使用：each.data for each in client
    return client

def embed(prompt):
    url = f"http://{LLM_HOST}:{LLM_PORT}/embed/"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(dict(prompt=prompt))

    res = requests.post(url, headers=headers, data=data)
    return json.loads(res.text)


"""
使用langchain定义一些常用类
"""
from typing import List
from langchain.embeddings.base import Embeddings

class VicunaEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return embed(text)


from langchain.llms.base import LLM

class VicunaLLM(LLM):
    streaming: bool = False
    """Whether to stream the results or not."""

    @property
    def _llm_type(self) -> str:
        return "Vicuna"
    
    def _call(self, prompt: str, stop: list = None, run_manager = None) -> str:
        if self.streaming:
            last_output = '' # 供 run_manager 获得 new_token 使用
            for each in streaming_generate(prompt):
                output = each.data
                if run_manager:
                    # 删除之前的输出，只保留新增内容
                    new_token = output.replace(last_output, '') 
                    run_manager.on_llm_new_token(new_token)
                    last_output = output
            return output
        else:
            return generate(prompt)