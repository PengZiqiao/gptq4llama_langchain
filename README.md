# GPTQ4LLaMa-langchain

Use 4bit GPTQ models whith langchain.

## Getting Start

It is recommended to use conda to create a virtual environment for Python3.9. Then set up the environment according to the following steps：

**Step 1:**\
Use git to download this project

```bash
git clone https://github.com/PengZiqiao/gptq4llama_langchain.git
```

**Step 2:**\
Install dependencies using pip

```bash
pip install -r requirements.txt
```

**Step 3:**\
The project depends on [GTPQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa). You need to copy it or softlink it into the Python environment

```bash
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
pip install -r GPTQ-for-LLaMa/requirements.txt
ln -s GPTQ-for-LLaMa ENVIROMENT_DIR/lib/python3.9/site-packages/gptq4llama
```

3. Prepare the model files, you can use (but not limited to) the following models: 
- [vicuna-13b-GPTQ-4bit-128g](https://huggingface.co/anon8231489123/vicuna-13b-GPTQ-4bit-128g)
- [wizardLM-7B-GPTQ](https://huggingface.co/TheBloke/wizardLM-7B-GPTQ)
- [stable-vicuna-13B-GPTQ](https://huggingface.co/TheBloke/stable-vicuna-13B-GPTQ)
- [koala-13B-GPTQ-4bit-128g](https://huggingface.co/TheBloke/koala-13B-GPTQ-4bit-128g)
- [BELLE-7B-gptq](https://huggingface.co/BelleGroup/BELLE-7B-gptq)

## Using Vicuna model class

I used the model [vicuna-13b-GPTQ-4bit-128g](https://huggingface.co/anon8231489123/vicuna-13b-GPTQ-4bit-128g), so I named the class `Vicuna`. In fact, this class (perhaps) could also be used to load other models mentioned before.

1. import the `Vicuna` class and create an instance.

```python
from model import Vicuna

model_dir = 'YOUR_MODEL_DIR'
checkpoint = 'YOUR_MODEL_DIR/checkpoint_file'

vic = Vicuna(model_dir, checkpoint)
```

2. We define a method called `generate()`. This method has two parameters: `prompt` and `streaming`, which indicate the input prompt and whether streaming generation is enabled.

```python
content = input()

prompt = f"""A chat between a user and an assistant.
USER: {content}
ASSISTANT: """

print(vic.generate(prompt))
```

3. Using streaming output(learn a lot from [text-generation-webui](https://github.com/oobabooga/text-generation-webui)), you get a generator. The full content is output every time. If you only want to keep the newly generated content, you need to manually remove the last output.

```python
last_output = ''
for output in vic.generate(prompt, streaming=True)：
    print(output.replace(last_output, '')) 
    last_output = output
```

4. We also define a method called `embed()` for representing the input text conversion vector. Used for similar search, classification, clustering, and other operations.

```python
embeddings = vic.embed(prompt)
```

## Using API

1. If you are using this project on the server, we also use fastapi to enclose the above methods into APIs. Start the service using the following command:

```bash
python run_server.py
```

2. You can change the host, port in `config.py`

```python
LLM_HOST = "0.0.0.0"
LLM_PORT = "8080"
```

3. `/generate/` to get reply text.

|  |  |
| --- | --- |
| Method | `POST` |
| Requst body | <code>{"prompt": "string", <br>"params": {}}</code> |

use requests.post to call the api
```python
import requests
import json
from config import LLM_HOST, LLM_PORT

GENERATE_PARAMS = dict(
    min_length=0, max_length=4096, temperature=0.1, top_p=0.75, top_k=40
)

def generate(prompt):
    url = f"http://{LLM_HOST}:{LLM_PORT}/generate/"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(dict(prompt=prompt, params=GENERATE_PARAMS))

    res = requests.post(url, headers=headers, data=data)
    return res.text

print(generate('Hello, bot!'))
```

4. `/streaming_generate/` to get the streaming reply.

|  |  |
| --- | --- |
| Method | `POST` |
| Requst body | <code>{"prompt": "string", <br>"params": {}}</code> |

use requests.post to call the api
```python
from sseclient import SSEClient
import requests
import json
from config import LLM_HOST, LLM_PORT

def streaming_generate(prompt):
    url = f"http://{LLM_HOST}:{LLM_PORT}/streaming_generate/"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(dict(prompt=prompt, params=GENERATE_PARAMS))

    res = requests.post(url, headers=headers, data=data, stream=True)
    client = SSEClient(res).events()
    return client

form each in streaming_gerate('Hello, bot!'):
    print(each.data)
```

5. `/embed/` to get the embeddings

|  |  |
| --- | --- |
| Method | `POST` |
| Requst body | `{"prompt": "string"}` |

use requests.post to call the api
```python
import requests
import json
from config import LLM_HOST, LLM_PORT
def embed(prompt):
    url = f"http://{LLM_HOST}:{LLM_PORT}/embed/"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(dict(prompt=prompt))

    res = requests.post(url, headers=headers, data=data)
    return json.loads(res.text)

print(embed('Hello, bot!'))
```

6. `/chat/` to make conversations

|  |  |
| --- | --- |
| Method | `POST` |
| Requst body | <code>[['USER MESSAGE 1', 'ASSISTANT MESSAGE 1'], <br>['USER MESSAGE 2', 'ASSISTANT MESSAGE 2'], <br>... <br>['USER MESSAGE n', '']]</code> |

use requests.post to call the api
```python
from sseclient import SSEClient
import requests
import json
from config import LLM_HOST, LLM_PORT

def chat(history):
    url = f"http://{LLM_HOST}:{LLM_PORT}/chat/"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(history)

    res = requests.post(url, headers=headers, data=data, stream=True)
    client = SSEClient(res).events()
    return client

for each in chat([['Hello, bot!', '']]):
    print(each.data)
```

# Using with langchain

1. We provide a custom [langchain](https://github.com/hwchase17/langchain) LLM wrapper name VicunaLLM.

```python
from model import VicunaLLM
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
vic = VicunaLLM(streaming=True, callback_manager=callback_manager)
vic("Hello, bot!")
```

For more details on how to use LLMs within LangChain, see the [LLM getting started guide](https://python.langchain.com/en/latest/modules/models/llms/getting_started.html).

2. We also provie a custom Embeddings Model

```python
from model import VicunaEmbeddings

document = "This is a content of the document"
query = "What is the contnt of the document?"

embeddings = VicunaEmbeddings()

doc_result = embeddings.embed_documents([document])
query_result = embeddings.embed_query(query)
```

## Acknowledgements

- https://github.com/hwchase17/langchain
- https://github.com/qwopqwop200/GPTQ-for-LLaMa
- https://github.com/oobabooga/text-generation-webui


## TODO

- [] loras using support
- [] custom some langchain chains
- [] custom some langchain agents

