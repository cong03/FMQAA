import random
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import argparse
import os

# 设置CUDA设备
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6,7"

def generate_response_Qwen(prompt, tokenizer, model):
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
        output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def generate_response_llama(prompt, tokenizer, model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") 
    ]

    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_response(prompt, tokenizer, model, model_type="llama"):
    if model_type == "qwen":
        return generate_response_Qwen(prompt, tokenizer, model)
    elif model_type == "llama":
        return generate_response_llama(prompt, tokenizer, model)
    else:
        raise ValueError("Unsupported model type")

# QA评估函数
def evaluation_qa_dataset(model, tokenizer, data, batch_size, instruction, output_path, model_type="llama"):
    correct = 0
    incorrect = 0
    for i in tqdm(range(0, len(data), batch_size), desc="Evaluating QA"):
        batch_data = data[i:i+batch_size]
        for entry in batch_data:
            question = entry["question"]
            hallucinated_answer = entry["hallucinated_answer"]
            right_answer = entry["right_answer"]
            ground_truth = "Yes" if random.random() > 0.5 else "No"
            answer = hallucinated_answer if ground_truth == "Yes" else right_answer
            
            # 获取生成的回答
            prompt = instruction + "\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
            response = generate_response(prompt, tokenizer, model, model_type)
            ans = response.replace(".", "").strip()

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                result = "failed"
                incorrect += 1
            else:
                result = "Yes" if "Yes" in response else "No"
                if result == ground_truth:
                    correct += 1
                else:
                    incorrect += 1

            dump_jsonl({
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "judgement": result
            }, output_path, append=True)

    return correct, incorrect


def evaluation_dialogue_dataset(model, tokenizer, data, batch_size, instruction, output_path, model_type="llama"):
    correct = 0
    incorrect = 0
    for i in tqdm(range(0, len(data), batch_size), desc="Evaluating Dialogue"):
        batch_data = data[i:i+batch_size]
        for entry in batch_data:
            dialog = entry["dialogue_history"]
            hallucinated_response = entry["hallucinated_response"]
            right_response = entry["right_response"]
            ground_truth = "Yes" if random.random() > 0.5 else "No"
            response = hallucinated_response if ground_truth == "Yes" else right_response

            prompt = instruction + "\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#:"
            result = generate_response(prompt, tokenizer, model, model_type)
            ans = response.replace(".", "").strip()

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                result = "failed"
                incorrect += 1
            else:
                result = "Yes" if "Yes" in result else "No"
                if result == ground_truth:
                    correct += 1
                else:
                    incorrect += 1

            dump_jsonl({
                "dialogue_history": dialog,
                "response": response,
                "ground_truth": ground_truth,
                "judgement": result
            }, output_path, append=True)

    return correct, incorrect

def evaluation_summarization_dataset(model, tokenizer, data, batch_size, instruction, output_path, model_type="llama"):
    correct = 0
    incorrect = 0
    for i in tqdm(range(0, len(data), batch_size), desc="Evaluating Summarization"):
        batch_data = data[i:i+batch_size]
        for entry in batch_data:
            document = entry["document"]
            hallucinated_summary = entry["hallucinated_summary"]
            right_summary = entry["right_summary"]
            ground_truth = "Yes" if random.random() > 0.5 else "No"
            summary = hallucinated_summary if ground_truth == "Yes" else right_summary

            prompt = instruction + "\n#Document#: " + document + "\n#Summary#: " + summary + "\n#Your Judgement#:"
            result = generate_response(prompt, tokenizer, model, model_type)
            ans = result.replace(".", "").strip()

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                result = "failed"
                incorrect += 1
            else:
                result = "Yes" if "Yes" in result else "No"
                if result == ground_truth:
                    correct += 1
                else:
                    incorrect += 1

            dump_jsonl({
                "document": document,
                "summary": summary,
                "ground_truth": ground_truth,
                "judgement": result
            }, output_path, append=True)

    return correct, incorrect

def dump_jsonl(data, output_path, append=False):
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hallucination Generation")
    parser.add_argument("--task", default="qa", help="qa, dialogue, or summarization")
    parser.add_argument("--model", default="Qwen2-7B-Instruct", help="model name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--model_type", default="qwen", help="Type of model: qwen or llama")
    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batch_size
    model_type = args.model_type
    model_path = ""
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 加载prompt
    instruction_file = ""
    f = open(instruction_file, 'r', encoding="utf-8")
    instruction = f.read()
    # 加载任务数据
    with open("") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = ''
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.task == "qa":
        correct, incorrect = evaluation_qa_dataset(model, tokenizer, data, batch_size, instruction, output_path, model_type)
    elif args.task == "dialogue":
        correct, incorrect = evaluation_dialogue_dataset(model, tokenizer, data, batch_size, instruction, output_path, model_type)
    elif args.task == "summarization":
        correct, incorrect = evaluation_summarization_dataset(model, tokenizer, data, batch_size, instruction, output_path, model_type)
    else:
        print(f"Unknown task: {args.task}")

    acc = correct / (correct + incorrect)
    with open(output_path, 'a') as f:
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'Correct: {correct}\n')
        f.write(f'Incorrect: {incorrect}\n')