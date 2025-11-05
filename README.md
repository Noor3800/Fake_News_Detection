# 📰 Fake News Detection Using NLP & Machine Learning

## 🔍 Overview

This project detects **fake vs real news** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.
It applies **TF-IDF vectorization**, **Logistic Regression**, and **Support Vector Machine (SVM)** models to classify news articles with high accuracy.

---

## 🧠 What It Does

* Preprocesses and cleans news text (removing stopwords, lemmatizing, lowercasing).
* Converts text into numerical features using **TF-IDF Vectorizer**.
* Trains two models: **Logistic Regression** and **Linear SVM**.
* Evaluates both models using **accuracy**, **F1-score**, and **classification report**.
* Generates **WordClouds** for visualizing frequent terms in fake vs real news.

---

## ⚙️ Tech Stack

* **Python**
* **Pandas, NumPy, Matplotlib**
* **NLTK (for text cleaning & lemmatization)**
* **Scikit-learn (for ML models)**
* **WordCloud (for visualization)**

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install pandas numpy matplotlib nltk scikit-learn wordcloud
```

### 2️⃣ Download NLTK resources

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 3️⃣ Run the script

Place your `Fake.csv` and `True.csv` datasets in the same directory and execute:

```bash
python fake_news_detection.py
```

---

## 📊 Results

| Model               | Accuracy  | F1 Score  |
| ------------------- | --------- | --------- |
| Logistic Regression | 98.8%     | 98.7%     |
| Linear SVM          | **99.4%** | **99.4%** |

Both models perform extremely well, with **SVM slightly outperforming** Logistic Regression.

---

## 🌈 Visualization

WordClouds show the most frequent words in **fake** and **real** news datasets:

* 🟠 **Fake News Word Cloud**
* 🔵 **Real News Word Cloud**

---

## 🧩 Skills Practiced

* Text preprocessing (tokenization, stopword removal, lemmatization)
* Feature extraction using **TF-IDF**
* Model building with **Logistic Regression** & **SVM**
* Performance evaluation & visualization

---

