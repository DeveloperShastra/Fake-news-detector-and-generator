import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    pipeline,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import warnings
import os
import re
import torch
import requests
from difflib import SequenceMatcher  

warnings.filterwarnings("ignore")

# -----------------------------
# Utility Functions
# -----------------------------
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def get_model_path():
    try:
        with open('model_path.txt', 'r') as f:
            return f.read().strip()
    except:
        return './bert_fakenews_model'

# -----------------------------
# Load DistilBERT Model & Tokenizer
# -----------------------------
MODEL_PATH = get_model_path()
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z ]', '', text)
    return text

def predict_news(text):
    clean = clean_text(text)
    inputs = tokenizer(clean, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)
        confidence = float(np.max(probs))
    if confidence < 0.6:
        label = 'Uncertain'
    else:
        label = 'Real News' if pred == 1 else 'Fake News'
    explanation = f"Confidence: {confidence*100:.2f}%"
    return label, confidence, explanation

# -----------------------------
# Fake News Generator using GPT-2
# -----------------------------
def load_generator():
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
    import os

    try:
        local_model_path = r"C:\Users\HP\OneDrive\Desktop\Fake News Generator and Detector\gpt2-local"
        if not os.path.isdir(local_model_path):
            raise ValueError(f"Local model folder not found at: {local_model_path}")

        print("👉 Loading GPT2 model and tokenizer from local folder...")
        tokenizer = GPT2Tokenizer.from_pretrained(local_model_path)
        model = GPT2LMHeadModel.from_pretrained(local_model_path)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("✅ Model and tokenizer loaded successfully.")
        return generator
    except Exception as e:
        print(f"❌ Failed to load GPT2: {e}")
        exit(1)

generator = load_generator()

def generate_fake_news(prompt, style='news', length=150):
    bulletins_prompt = (
        f"Generate 5 realistic, short news bulletins that sound true but are actually fake. "
        f"Each bulletin should be 1-2 sentences and look like a real news headline or breaking news update.\n"
        f"Topic: {prompt}\n"
        f"Bulletins:\n1."
    )
    fake_news = generator(bulletins_prompt, max_length=length, num_return_sequences=1)
    text = fake_news[0]['generated_text']
    bulletins = text.split('\n')
    bulletins = [b for b in bulletins if b.strip() and b.strip()[0] in '12345']
    return '\n'.join(bulletins[:5]) if len(bulletins) >= 5 else text

# -----------------------------
# Flask Web App
# -----------------------------
app = Flask(__name__)
app.secret_key = 'supersecretkey'

NEWSAPI_KEY = '7d298abfbeb34a678caec8ce9e3303db'
GOOGLE_API_KEY = 'AIzaSyCAS08JgrG_Ty8TfSTGlC8qXtrKkaoRODU'
GOOGLE_CX = '33b40d3c785ea4091'

def search_newsapi(query):
    try:
        url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&apiKey={NEWSAPI_KEY}&pageSize=5"
        r = requests.get(url)
        data = r.json()
        articles = data.get('articles', [])
        return [f"{a['title']} - {a.get('description', '')}" for a in articles]
    except:
        return []

def search_google(query):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={requests.utils.quote(query)}&cx={GOOGLE_CX}&key={GOOGLE_API_KEY}"
        r = requests.get(url)
        data = r.json()
        items = data.get("items", [])
        return [f"{i['title']} - {i.get('snippet', '')}" for i in items]
    except:
        return []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form.get('news', '').strip()
    if not news:
        flash('Please enter news text to check.')
        return redirect(url_for('home'))

    google_results = search_google(news)
    newsapi_results = search_newsapi(news)

    def is_news_matched(news, results, threshold=0.5):
        news_tokens = set(news.lower().split())
        if not news_tokens:
            return False
        for result in results:
            result_tokens = set(result.lower().split())
            common_tokens = news_tokens & result_tokens
            match_ratio = len(common_tokens) / len(news_tokens)
            if match_ratio >= threshold and len(common_tokens) > 5:
                return True
        return False



    matched_google = is_news_matched(news, google_results)
    matched_newsapi = is_news_matched(news, newsapi_results)


    if matched_google or matched_newsapi:
        label = 'Real News'
        confidence = 1.0
        explanation = 'Verified via Google or NewsAPI.'
        sources_info = {'Google': google_results if matched_google else [], 'NewsAPI': newsapi_results if matched_newsapi else []}
        return render_template('result.html', prediction=label, confidence=confidence, explanation=explanation, news=news, sources_info=sources_info)

    label, confidence, explanation = predict_news(news)
    return render_template('result.html', prediction=label, confidence=confidence, explanation=explanation, news=news, sources_info={})

@app.route('/generate', methods=['POST'])
def generate():
    try:
        prompt = request.form.get('prompt', '').strip()
        style = request.form.get('style', 'news')
        length = request.form.get('length', 'medium')  # short, medium, long
        mode = request.form.get('mode', 'story')       # story or headlines

        if not prompt:
            flash('Please enter a prompt to generate fake news.')
            return redirect(url_for('home'))

        # Length map
        length_map = {
            'short': 100,
            'medium': 250,
            'long': 500
        }
        max_tokens = length_map.get(length, 250)

        # ----------------------------
        # HEADLINES mode
        # ----------------------------
        if mode == 'headlines':
            full_prompt = (
                f"Write 5 fake news headlines that sound real but are fictional.\n"
                f"Topic: {prompt}\n\n1."
            )

            generated = generator(
                full_prompt,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                temperature=0.9,
                top_k=50,
                top_p=0.95
            )

            text = generated[0]['generated_text']
            headlines = [line.strip() for line in text.split('\n') if line.strip().startswith(('1.', '2.', '3.', '4.', '5.'))]
            output = '\n'.join(headlines[:5]) if headlines else text

        # ----------------------------
        # STORY mode
        # ----------------------------
        else:
            if style == "satire":
                article_prompt = (
                    f"SATIRICAL REPORT: {prompt}\n\n"
                    f"In an absurd twist of events, sources claim that {prompt.lower()} — though experts remain skeptical.\n\n"
                )
            elif style == "blog":
                article_prompt = (
                    f"{prompt}\n\n"
                    f"Hey folks! Today I want to talk about something wild I heard: {prompt}. Here's what it could mean...\n\n"
                )
            elif style == "headline":
                article_prompt = (
                    f"1. {prompt} shocks the world\n2."
                )
            else:  # default to "news"
                article_prompt = (
                    f"BREAKING NEWS: {prompt}\n\n"
                    f"In a surprising development, experts from NeoScience Times reported that {prompt.lower()} is now becoming reality. "
                    f"\"We’re seeing unprecedented change,\" said Dr. Maya Iyer, lead analyst at the International Tech Bureau.\n\n"
                )

            generated = generator(
                article_prompt,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                temperature=0.9,
                top_k=50,
                top_p=0.95
            )

            output = generated[0]['generated_text'].strip()

        return render_template('generated.html', output=output, prompt=prompt, style=style, length=length, mode=mode)

    except Exception as e:
        print("Error during generation:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
