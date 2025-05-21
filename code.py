import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})


from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=5,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

from datasets import load_dataset
import json
from datasets import Dataset

# data = open('/content/nutrition_prompts.json', 'r+')
# data = json.load(data)
# # Load the JSON file
# Read your local JSONL file manually
data = []
with open('nutrition_prompts.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if line:  # skip empty lines
            data.append(json.loads(line))

# Convert to Hugging Face dataset
dataset = Dataset.from_list(data)
#dataset = load_dataset("json", data_files='/content/nutrition_prompts.jsonl', split="train")


# Define how to combine fields into a single input string
def preprocess(example):
    combined_input = (
        f"Instruction: {example['instruction']}\n"
        f"Context: {example['context']}\n"
        f"Prompt: {example['prompt']}"
    )
    return tokenizer(combined_input, truncation=True, padding="max_length", max_length=8192)

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess, batched=False)  # reduce from 8 or 16




import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

import transformers


tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
