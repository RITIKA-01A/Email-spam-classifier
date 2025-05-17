# 📧 Spam Classifier using Naive Bayes

A clean and effective machine learning project that detects **spam messages** using the **Naive Bayes** algorithm. This project focuses on text classification through natural language processing (NLP) and probabilistic modeling to differentiate between **spam** and **ham (not spam)** messages.

---

## 🔍 Overview

This classifier was built using a dataset of SMS messages labeled as *spam* or *ham*. The pipeline includes preprocessing, feature extraction, model training, and performance evaluation — all implemented with clarity and precision in Python.

---

## 🚀 Key Features

- 🧠 **Multinomial Naive Bayes** for fast and effective spam detection  
- 🧹 Full **text preprocessing pipeline**: lowercase, punctuation removal, stopword removal, stemming  
- 🔢 **TF-IDF** or **CountVectorizer** for feature extraction  
- 📊 Model performance metrics: accuracy, precision, recall, F1-score  
- ☁️ Optional visualizations like word clouds and class distribution  

---

## 🗂️ Project Structure


├── .gitignore # Ignoring sensitive files like model/data


├── LICENSE # Open-source license

├── model-training.ipynb # Jupyter notebook with full training workflow

├── model.pkl # Saved trained Naive Bayes model

├── requirements.txt # Project dependencies

├── spam.csv # SMS dataset (ham/spam)

└── vectorizer.pkl # Saved TF-IDF vectorizer

## ⚙️ How It Works

1. **Load the dataset** (`spam.csv`)
2. **Preprocess the text** (clean, tokenize, remove stopwords, stem)
3. **Vectorize** using `TfidfVectorizer`
4. **Train** a `MultinomialNB` model
5. **Evaluate** the model using standard metrics
6. **Save** the trained model and vectorizer with `pickle`

## 🛠️ Installation

1. **Clone the repository:
```
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier
```
2. **Install dependencies:
```
pip install -r requirements.txt
```
3. **Run the notebook:
```
jupyter notebook model-training.ipynb
```

## 💡 Future Improvements
- Add a user interface (e.g., Streamlit)
- Deploy the model as a web app or API
- Extend to other languages or email-based datasets

## 📝 License
- This project is open-source under the MIT License.



