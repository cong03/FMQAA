import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datetime import datetime
from modelscope import AutoModelForCausalLM, AutoTokenizer
# 设置CUDA设备
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6,7"

model_name = "Qwen2.5-14B-Instruct"
model_path = ""
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_length=2048, num_beams=1, early_stopping=False, repetition_penalty=1.0, temperature=1.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_response_Qwen(prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

prompt = '''

'''   
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

t = 5 
input_file_name = ''
output_file_name = ""
os.makedirs(os.path.dirname(output_file_name), exist_ok=True)


with open(input_file_name, 'r', encoding='utf-8') as file, open(output_file_name, 'w', encoding='utf-8') as output_file:
    total_lines = sum(1 for _ in file)
    file.seek(0) 
    for idx, line in enumerate(tqdm(file, total=total_lines, desc="Processing lines")):

        data = json.loads(line.strip())

        id = data['id']
        question = data['question']
        choices = data['choices']
        answer = data['answer']
        
        
        change = question + choices + answer

        output = [f'"{key}": {value}' for key, value in data.items()]
        data = ", ".join(output)

        # print(data)
        t_prompt = prompt.format(input = data)

        hallucinated_questions = []
        for i in range(1, t + 1): 
            hallucinated_question = generate_response_Qwen(t_prompt)
            hallucinated_question_cleaned = hallucinated_question.replace("\"hallucinated_question\":", "").strip()
            hallucinated_questions.append(hallucinated_question_cleaned.replace(t_prompt, '').strip())

        updated_data = {
            "id": id,
            "question": question,
            "choices": choices,
            "answer": answer
        }

        
        for i in range(t):
            updated_data[f"hallucinated_question{i+1}"] = hallucinated_questions[i]

        output_file.write(json.dumps(updated_data, ensure_ascii=False) + '\n')

        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1} lines...")


