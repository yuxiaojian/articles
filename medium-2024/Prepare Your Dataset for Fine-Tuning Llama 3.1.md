# [Prepare Your Dataset for Fine-Tuning Llama 3.1](https://medium.com/@yuxiaojian/prepare-your-dataset-for-fine-tuning-llama-3-1-46fd3c78f6fd)


<p align="center">
  <img src="img/prepare-dataset-for-llama31-1.jpg">
</p>

In my previous story, I walked through the process of  [Fine-Tuning Ollama Models with Unsloth](https://medium.com/@yuxiaojian/fine-tuning-ollama-models-with-unsloth-a504ff9e8002). With the Supervised Fine-Tuning Trainer (SFTT) and Unsloth, fine-tuning Llama models becomes a breeze. Once your data is ready, the next crucial step is to prepare your dataset for fine-tuning. In this story, I’ll guide you on how to prepare your dataset for fine-tuning Llama 3.1.

Large Language Models (LLMs) are essentially text predictors that generate output based on given input. The input and output need to follow a specific format known as the “Prompt Format.” Different LLMs have different prompt formats, and it’s essential to customize your prompts accordingly. The  [Llama 3.1 prompt format](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1#prompt-format)  specifies special tokens that the model uses to distinguish different parts of a prompt.

## Llama 3.1 Chat Prompt

Here are the ones used in a chat template.

`<|begin_of_text|>`  Specifies the start of the prompt

`<|start_header_id|>`  and  `<|end_header_id|>`  These tokens enclose the role of a particular message. The possible roles are:  `[system, user, assistant and ipython]`

`<|eot_id|>`  End of turn.

If you are tuning a llama-based model, you can use this simple template. It starts with text and allows the model to generate new content based on the user-provided input.

```
<|begin_of_text|>{{ USER_INPUT }}
```

A base model is trained on a large corpus of general text data using unsupervised learning. This means the model learns patterns in language and how to predict the next word or sequence without explicit task-related guidance.

An instruct model, on the other hand, starts with a pre-trained base model and is further trained on a dataset containing pairs of instructions and the desired corresponding outputs. Instruct models generally perform better on specific tasks when given clear instructions, making them the preferred choice for many applications.

The chat template for the instruct model also includes the  `system`  role and corresponding message. Special tokens are used to guide the model. Tokens such as  `<|begin_of_text|>`  and  `<|eot_id|>`  are optional but can be included to enhance the model's understanding of the input structure.
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>  
  
{{SYSTEM}}  
<|eot_id|>  
<|start_header_id|>user<|end_header_id|>  
  
{{ USER }}  
<|eot_id|>  
<|start_header_id|>assistant<|end_header_id|>  
<|eot_id|>
```
## Format Dataset to Prompt

Before feeding data to the Llama 3.1 model, we need to format it according to the Llama 3.1 prompt format. Let’s take the  `yahma/alpaca-cleaned`  dataset as an example and print out the 22nd row in the required format.
```
from datasets import load_dataset  
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")  
print(dataset[22])  
  
{'output': 'She will play the piano beautifully for hours and then stop as it will be midnight.',  
 'input': 'She played the piano beautifully for hours and then stopped as it was midnight.',  
 'instruction': 'Based on the information provided, rewrite the sentence by changing its tense from past to future.'}
```
The dataset has three columns  `[‘instruction’, ‘input’, ‘output’]`  . They will be mapped to  `system`,  `user`, and  `assistant`  fields
```
llama31_prompt="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>  
  
{}<|eot_id|><|start_header_id|>user<|end_header_id|>  
  
{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>  
  
{}<|eot_id|>"""  
  
def formatting_prompts_func(examples):  
    instructions = examples["instruction"]  
    inputs       = examples["input"]  
    outputs      = examples["output"]  
    texts = []  
    for instruction, input, output in zip(instructions, inputs, outputs):  
        text = llama31_prompt.format(instruction, input, output)  
        texts.append(text)  
    return { "text" : texts, }  
pass  
dataset = dataset.map(formatting_prompts_func, batched = True,)  
print(dataset[22])  
  
{'output': 'She will play the piano beautifully for hours and then stop as it will be midnight.',  
 'input': 'She played the piano beautifully for hours and then stopped as it was midnight.',  
 'instruction': 'Based on the information provided, rewrite the sentence by changing its tense from past to future.',  
 'text': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBased on the information provided, rewrite the sentence by changing its tense from past to future.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nShe played the piano beautifully for hours and then stopped as it was midnight.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nShe will play the piano beautifully for hours and then stop as it will be midnight.<|eot_id|>'}  
  
  
trainer = SFTTrainer(  
...  
    train_dataset = dataset,  
    dataset_text_field = "text",  
...  
)
```
The  `text`  column will be the training data. The 22nd row  `text`  after formatting is like
```
<|begin_of_text|>  
<|start_header_id|>system<|end_header_id|>  
  
Based on the information provided, rewrite the sentence by changing its tense from past to future.  
<|eot_id|>  
<|start_header_id|>user<|end_header_id|>  
  
She played the piano beautifully for hours and then stopped as it was midnight.  
<|eot_id|>  
<|start_header_id|>assistant<|end_header_id|>  
  
She will play the piano beautifully for hours and then stop as it will be midnight.  
<|eot_id|>
```
## Ollama Modelfile

[Ollama](https://github.com/ollama/ollama)  specifies the prompt for a model in  `[TEMPLATE](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#template)`of a  [Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md). You can find the full template for llama3.1  [here](https://ollama.com/library/llama3.1:latest/blobs/11ce4ee3e170). The majority part of the template is tool-calling related, the chat prompt part is at the bottom.
```
{{- if .System }}<|start_header_id|>system<|end_header_id|>  
  
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>  
  
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>  
  
{{ end }}{{ .Response }}{{ if .Response }}<|eot_id|>{{ end }}
```
It uses GO template syntax to structure the conversation. Let’s break it down:

-   `{{ if .System }}...{{ end }}`: If a system message is provided, it's formatted like this:
```
<|start_header_id|>system<|end_header_id|> [System Message]<|eot_id|>
```
-   `{{ if .Prompt }}...{{ end }}`: If a user prompt is provided:
```
<|start_header_id|>user<|end_header_id|> [User Prompt]<|eot_id|>
```
-   The assistant’s response is always included:
```
<|start_header_id|>assistant<|end_header_id|> [Assistant Response]<|eot_id|>
```
Let’s look at an API call to Ollama
```
curl http://localhost:11434/api/chat -d '{  
  "model": "llama3.1",  
  "stream": false,  
  "messages": [  
    {  
      "role": "user",  
      "content": "why is the sky blue?"  
    },  
    {  
      "role": "system",  
      "content": "you are a helpful assistant"  
    },  
    {  
      "role": "assistant",  
      "content": ""  
    }  
  ]  
}'
```
Ollama rendered prompt is
```
<|start_header_id|>system<|end_header_id|>  
  
You are a helpful assistant  
<|eot_id|>  
<|start_header_id|>user<|end_header_id|>  
  
Why is the sky blue?  
<|eot_id|>  
<|start_header_id|>assistant<|end_header_id|>
```
You can see that the Ollama-rendered prompt follows the same format used during the training of the model.

Since fine-tuning only adjusts the model’s parameters, you can use the same  [template](https://ollama.com/library/llama3.1:latest/blobs/11ce4ee3e170)  when deploying your fine-tuned Llama 3.1 model to Ollama.

## Conclusion

The formatted dataset is essential for fine-tuning with the SFTTrainer. This guide details how to prepare datasets for fine-tuning Llama 3.1 models. You can find the fine-tuning process in the story  [Fine-Tuning Ollama Models with Unsloth](https://medium.com/@yuxiaojian/fine-tuning-ollama-models-with-unsloth-a504ff9e8002).