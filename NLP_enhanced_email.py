import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification, 
                          GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, 
                          TextDataset, DataCollatorForLanguageModeling)

def load_dataset(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def prepare_dataset(data):
    return [
        f"{entry['prompt']} [Intent: {entry['intent']}] [Tone: {entry['tone']}] Subject: {entry['subject']} Body: {entry['body']}"
        for entry in data
    ]

def encode_labels(df):
    intent_encoder = LabelEncoder()
    tone_encoder = LabelEncoder()

    df['intent'] = intent_encoder.fit_transform(df['intent'])
    df['tone'] = tone_encoder.fit_transform(df['tone'])

    return intent_encoder, tone_encoder

def tokenize_function(examples, tokenizer):
    prompts = examples['prompt'].tolist()
    return tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt")

data = load_dataset('datasets.json')
df = pd.DataFrame(data)
intent_encoder, tone_encoder = encode_labels(df)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_intent = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=len(df['intent'].unique())
)
model_tone = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=len(df['tone'].unique())
)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_intent = model_intent.to(device)
model_tone = model_tone.to(device)
gpt2_model = gpt2_model.to(device)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_encodings_intent = tokenize_function(train_df, distilbert_tokenizer)
val_encodings_intent = tokenize_function(val_df, distilbert_tokenizer)
train_encodings_tone = tokenize_function(train_df, distilbert_tokenizer)
val_encodings_tone = tokenize_function(val_df, distilbert_tokenizer)

train_dataset_intent = CustomDataset(train_encodings_intent, train_df['intent'].tolist())
val_dataset_intent = CustomDataset(val_encodings_intent, val_df['intent'].tolist())
train_dataset_tone = CustomDataset(train_encodings_tone, train_df['tone'].tolist())
val_dataset_tone = CustomDataset(val_encodings_tone, val_df['tone'].tolist())

training_args = TrainingArguments(
    output_dir='./results_distilbert',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer_intent = Trainer(
    model=model_intent,
    args=training_args,
    train_dataset=train_dataset_intent,
    eval_dataset=val_dataset_intent
)

trainer_tone = Trainer(
    model=model_tone,
    args=training_args,
    train_dataset=train_dataset_tone,
    eval_dataset=val_dataset_tone
)

trainer_intent.train()
trainer_tone.train()

# Evaluate models
print("Evaluating Intent Model...")
eval_result_intent = trainer_intent.evaluate()
print("Intent Model Evaluation Results:")
print(eval_result_intent)

print("Evaluating Tone Model...")
eval_result_tone = trainer_tone.evaluate()
print("Tone Model Evaluation Results:")
print(eval_result_tone)

def load_text_dataset(formatted_examples):
    with open('temp_dataset.txt', 'w') as file:
        for example in formatted_examples:
            file.write(example + "\n")
    return TextDataset(
        tokenizer=gpt2_tokenizer,
        file_path='temp_dataset.txt',
        block_size=128
    )

formatted_examples = prepare_dataset(data)
train_dataset_gpt2 = load_text_dataset(formatted_examples)
data_collator = DataCollatorForLanguageModeling(tokenizer=gpt2_tokenizer, mlm=False)

training_args_gpt2 = TrainingArguments(
    output_dir='./results_gpt2',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

trainer_gpt2 = Trainer(
    model=gpt2_model,
    args=training_args_gpt2,
    data_collator=data_collator,
    train_dataset=train_dataset_gpt2,
)

trainer_gpt2.train()

def generate_email(prompt, intent, tone, tokenizer, model, max_length=500):
    input_text = f"{prompt} [Intent: {intent}] [Tone: {tone}]"
    
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    
    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=inputs.device)
    
    inputs = inputs.to(model.device)
    attention_mask = attention_mask.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_email = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_email

def predict(model, tokenizer, text, device='cpu'):
    model.to(device)
    encoding = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt", max_length=128)
    encoding = {key: val.to(device) for key, val in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    return predictions.item()

def email_pipeline(user_prompt):
    intent = predict(model_intent, distilbert_tokenizer, user_prompt, device=device)
    tone = predict(model_tone, distilbert_tokenizer, user_prompt, device=device)
    generated_email = generate_email(user_prompt, intent_encoder.inverse_transform([intent])[0], tone_encoder.inverse_transform([tone])[0], gpt2_tokenizer, gpt2_model)
    return generated_email

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    generated_email = email_pipeline(user_prompt)
    print("Generated Email:")
    print(generated_email)
