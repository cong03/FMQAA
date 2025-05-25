from transformers import AutoTokenizer
import pandas as pd
import os
import re
import json
import argparse
import torch
from datetime import datetime
# 设置CUDA设备
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
from modelscope import AutoModelForCausalLM, AutoTokenizer

def generate_llama(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def generate_Qwen(prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # 构建输入文本（假设没有 apply_chat_template 方法）
    text = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nUser: " + prompt + "\nAssistant:"
    
    # 构建模型输入并移动到设备
    model_inputs = {k: v.to(model.device) for k, v in tokenizer([text], return_tensors="pt").items()}
    
    # 生成文本
    generated_output = model.generate(
        **model_inputs,
        max_new_tokens=512,
        output_scores=True,
        return_dict_in_generate=True
    )

    # 提取生成的 token IDs
    generated_ids = generated_output.sequences
    
    # 去掉输入部分，只保留新生成的内容
    input_length = model_inputs['input_ids'].shape[1]  # 获取输入长度
    generated_ids = generated_ids[:, input_length:]  # 切片去掉输入部分
    
    # 解码生成的文本
    response = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)[0]
    
    return response.strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hallucination Generation")
    parser.add_argument("--model_name", default="Meta-Llama-3.2-3B-Instruct-step1ft", help="model name")
    args = parser.parse_args()
    model_name = args.model_name
    model_dir =  ""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    prompt = '''For the Commonsense Question Answering (CommonsenseQA) task, you need to choose the correct choice from the given "choices" as the answer according to the given "question". Give your answer in a sentence that states "So the answer is <s>.", where <s> is the correct choice.'''
    #上传数据
    test_dir = ''
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f""


    data = []
    # 读取 JSON 文件
    with open(test_dir, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            data.append(item)
    sum = len(data)
    correct = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            question = item['question']  # 提取问题
            choices = item['choices']   # 提取选项
            answer = item['answer']   # 提取答案
            input_text = f"{prompt}\nQuestion: {question}\nChoices: {', '.join(choices)}\nAnswer:"
            predictd_answer = generate_Qwen(input_text)
            f.write(f"Predicted Answer: {predictd_answer}, Answer: {answer}\n")
            print("Predicted Answer:", predictd_answer)
            
            # 遍历选项，匹配生成的答案
            matched_choices = [choice for choice in choices if choice in predictd_answer]

            if len(matched_choices) == 1:
                result = matched_choices[0]
            elif len(matched_choices) > 1:
                result = "Error: Multiple matches found"
            else:
                result = "No match found"
            if result == answer:
                correct = correct + 1
            print(f"correct: {correct}\n")
        acc = correct/sum
        print(f"acc: {acc}")
        f.write(f"acc: {acc}")
