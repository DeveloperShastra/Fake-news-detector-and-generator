import os
import requests
import feedparser
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re
import time

model_path = "./distilbert-base-uncased"
NEWSAPI_KEY = '7d298abfbeb34a678caec8ce9e3303db'

# 1. Fetch Real News from NewsAPI (no changes made)
def fetch_real_news(api_key, query="latest", page_size=100, pages=5):
    all_articles = []
    for page in range(1, pages+1):
        url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize={page_size}&q={query}&apiKey={api_key}&page={page}"
        response = requests.get(url)
        data = response.json()
        articles = data.get("articles", [])
        all_articles.extend([article["title"] + " " + str(article.get("description", "")) for article in articles])
    return all_articles

# 2. Fetch Fake News
FAKE_FEEDS = [
    "https://www.theonion.com/rss",
    "https://worldnewsdailyreport.com/feed/",
    "https://babylonbee.com/feed",
    "https://www.clickhole.com/rss",
    "https://www.thespoof.com/rss",
]
def fetch_fake_news_rss(feed_url):
    feed = feedparser.parse(feed_url)
    return [str(entry.title) + " " + str(getattr(entry, 'summary', '')) for entry in feed.entries]
def fetch_all_fake_news():
    fake_news = []
    for url in FAKE_FEEDS:
        try:
            fake_news.extend(fetch_fake_news_rss(url))
        except Exception as e:
            print(f"Error fetching from {url}: {e}")
    return fake_news

# 3. Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9 .,!?:;\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 4. Build dataset
def build_dataset():
    print("Fetching real news from multiple topics...")
    real_news = []
    topics = ["latest", "politics", "world", "business", "science", "technology", "sports", "entertainment", "health"]
    for topic in topics:
        try:
            topic_news = fetch_real_news(NEWSAPI_KEY, query=topic, page_size=100, pages=2)
            real_news.extend(topic_news)
            print(f"Fetched {len(topic_news)} articles for topic: {topic}")
        except Exception as e:
            print(f"Error fetching {topic}: {e}")
    print(f"Total real news articles: {len(real_news)}")

    print("Fetching fake news...")
    fake_news = fetch_all_fake_news()
    print(f"Fetched {len(fake_news)} fake news articles.")

    texts = [clean_text(t) for t in real_news + fake_news]
    labels = [1]*len(real_news) + [0]*len(fake_news)
    df = pd.DataFrame({"text": texts, "label": labels})
    df = df.drop_duplicates(subset=["text"]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Final dataset size: {len(df)} articles")
    return df

# 5. Train model
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # ✅ FIXED TOKENIZER AND MODEL LOADING
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased'
)
#developed by Shashwat Malhotra
    class NewsDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=256):
            self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)
            self.labels = list(labels)
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    test_dataset = NewsDataset(X_test, y_test, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
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
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    save_dir = f'./bert_fakenews_model_{int(time.time())}'
    try:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f'Model and tokenizer saved to {save_dir}')
        with open('model_path.txt', 'w') as f:
            f.write(save_dir)
    except Exception as e:
        print(f"Error saving model: {e}")
        save_dir = f'./model_backup_{int(time.time())}'
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f'Model and tokenizer saved to {save_dir} (backup)')
        with open('model_path.txt', 'w') as f:
            f.write(save_dir)

if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    train_model(df)
