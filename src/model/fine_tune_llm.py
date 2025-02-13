import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Load Pre-trained Model & Tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # You can replace with "mistralai/Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16,  # Use float16 for efficiency
    device_map="auto"  # Automatically assigns GPU if available
)

# Load Training Dataset
dataset = load_dataset("OpenAssistant/oasst1", split="train[:1%]")  # Using 1% for testing

# Tokenize Data
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    evaluation_strategy="epoch",
    fp16=True  # Enable mixed precision for faster training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Start Fine-Tuning
if __name__ == "__main__":
    print("Starting Fine-Tuning...")
    trainer.train()
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Fine-Tuning Completed & Model Saved!")
 
