import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from bs4 import BeautifulSoup
import requests

# 📁 Local dataset paths
FAKE_PATH = 'data/Fake.csv'
REAL_PATH = 'data/True.csv'

# 📰 Google News Scraper (optional)
def fetch_google_news(query="current events", pages=1):
    print("Fetching real news from Google News...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    articles = []
    for page in range(pages):
        start = page * 10
        url = f"https://www.google.com/search?q={query}&tbm=nws&start={start}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.select('div.BNeawe.vvjwJb.AP7Wnd')
        descriptions = soup.select('div.BNeawe.s3v9rd.AP7Wnd')
        for i in range(min(len(headlines), len(descriptions))):
            title = headlines[i].text
            desc = descriptions[i].text
            articles.append(f"{title} {desc}")
    print(f"Fetched {len(articles)} real news articles from Google News.")
    return articles

# ✅ Load local CSVs
print("Loading local Fake and True datasets...")
df_fake = pd.read_csv(FAKE_PATH).head(250)  # lightweight: limit to 250 samples
df_real = pd.read_csv(REAL_PATH).head(250)

# ✅ Google News Integration (real news boost)
google_real_news = fetch_google_news()
google_real_df = pd.DataFrame({'text': google_real_news, 'label': [1] * len(google_real_news)})

# ✅ Combine datasets
df_fake['text'] = df_fake['title'].astype(str) + " " + df_fake['text'].astype(str)
df_real['text'] = df_real['title'].astype(str) + " " + df_real['text'].astype(str)
df_fake['label'] = 0
df_real['label'] = 1

df = pd.concat([df_fake[['text', 'label']], df_real[['text', 'label']], google_real_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ✅ Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# ✅ Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):  # reduce max_len for speed
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)
        self.labels = list(map(int, labels))  # Ensure int labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # 🟢 Correct dtype
        return item
    def __len__(self):
        return len(self.labels)
# developed by Shashwat Malhotra
train_dataset = NewsDataset(X_train, y_train, tokenizer)
test_dataset = NewsDataset(X_test, y_test, tokenizer)

# ✅ Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# ✅ Training Args - lightweight mode
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=0,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy='no',
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print('Training DistilBERT...')
trainer.train()

print('Evaluating...')
preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

# ✅ Save model and tokenizer
model.save_pretrained('./bert_fakenews_model')
tokenizer.save_pretrained('./bert_fakenews_model')
print('✅ Model and tokenizer saved to ./bert_fakenews_model')
