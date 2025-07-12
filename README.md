
##  `README.md`

```md
# 📰 Autonews Pro

**Autonews Pro** is an AI-powered web application that lets users detect fake news and also generate convincing fake news based on custom prompts. Built with a Flask backend, DistilBERT for classification, and GPT-2 for generation, the project features a modern glassmorphic UI with light/dark themes.

---

## 🚀 Features

-  **Fake News Detection** using DistilBERT
-  **Fake News Generation** with GPT-2 (locally hosted)
-  Responsive UI with light/dark theme and animations
-  Share or download results instantly
-  Runs offline — no external model dependencies

---

## 📦 Tech Stack

| Layer        | Tech                              |
|--------------|-----------------------------------|
| Frontend     | HTML, CSS, JS, FontAwesome        |
| Backend      | Python, Flask                     |
| Detection ML | HuggingFace DistilBERT            |
| Generation   | GPT-2 (Text Generation)           |
| Data Source  | NewsAPI + RSS Feeds + Google News |

---

## 📁 Project Structure

```

├── templates/
│   ├── home.html
│   ├── result.html
│   │── generated.html
│   └── requirements.txt  
│
├── fake\_news\_bert.py   # Model training for classification
├── fetch\_and\_train.py  # Scrapes data & trains models
├── fakenewsdetector.py # Main Flask app
├
└── README.md

````

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/autonews-pro.git
cd autonews-pro
````

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

```
because it works best on python 3.10.0

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Model Setup

### Fetch and Train with Latest News

```bash
python fetch_and_train.py
```
### Train BERT Model

```bash
python fake_news_bert.py
```
---

## 🔁 Run the Application

```bash
python fakenewsdetector.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---


## 📌 Future Improvements

* Add source citations for real news
* Token-level explanation using LIME/SHAP
* Multilingual support
* Integration with mobile clients

---

## 👨‍💻 Author

**Shashwat Malhotra**
AI Enthusiast | Developer 

---
