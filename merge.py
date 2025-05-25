from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6"
# 基础模型
base_model_path = ""
# lora
lora_adapter_path = ""

model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

model = PeftModel.from_pretrained(model, lora_adapter_path)

model = model.merge_and_unload()

output_path = ""
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
