import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import joblib
import os


train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")


train_df = train_df.sample(n=30000, random_state=42) 
val_df = val_df.sample(n=5000, random_state=42)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['label'])
val_labels = label_encoder.transform(val_df['label'])


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MAX_LEN = 128
BATCH_SIZE = 32

class ArxivDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]['title'] + " [SEP] " + self.texts[idx]['abstract']
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


train_texts = train_df[['title', 'abstract']].to_dict('records')
val_texts = val_df[['title', 'abstract']].to_dict('records')

train_dataset = ArxivDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = ArxivDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label_encoder.classes_)
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
EPOCHS = 3

history = {'loss': [], 'accuracy': []}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc='  Training')
    for batch in progress:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    avg_loss = total_loss / len(train_loader)
    history['loss'].append(avg_loss)


    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='  Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    history['accuracy'].append(accuracy)
    
    print(f"\nРезультаты на val:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")



os.makedirs("arxiv_classifier", exist_ok=True)
model.save_pretrained("arxiv_classifier")
tokenizer.save_pretrained("arxiv_classifier")
joblib.dump(label_encoder, "arxiv_classifier/label_encoder.pkl")

