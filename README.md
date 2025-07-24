# 📩 Spam Message Classifier (Multinomial Naive Bayes)

This is an end-to-end Natural Language Processing (NLP) project that classifies SMS messages as either **spam** or **ham** (not spam). The model is built using **Multinomial Naive Bayes** and optimized to improve **recall** to ensure spam messages are detected effectively.

---

## 🧠 Features

- Cleaned and preprocessed text data
- Implemented stemming and punctuation removal
- TF-IDF Vectorization of text messages
- Trained with Multinomial Naive Bayes
- Threshold tuning for better recall performance
- Precision vs Recall visualization for threshold selection
- Exported trained model and vectorizer using `pickle`

---

## 📊 Project Structure

```
.
├── model/
│   ├── spam_classifier.pkl
│   └── vectorizer.pkl
│
├── notebooks/
│   └── spam_classifier.ipynb
│
├── README.md
```

---

## 🔧 Tech Stack

- Python 3.x
- Pandas
- Scikit-learn
- NLTK
- Matplotlib & Seaborn

---

## 🚀 How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/Rushi-12-11/spam-classifier.git
   ```

2. Navigate to the project folder and open the notebook:
   ```bash
   cd spam-classifier
   ```

3. Run the Jupyter/Colab notebook in the `notebooks/` folder:
   - It walks through loading the dataset, preprocessing, training, and evaluation.
   - You can load the saved model/vectorizer from the `model/` directory if needed.

4. (Optional) Use the `.pkl` files for inference in a web app or deployment later.

---

## ✅ Results

- **Model**: Multinomial Naive Bayes
- **Recall Score** (after tuning): `0.8758`
- **Precision vs Recall** curve used for threshold optimization

---

## 🚧 Future Improvements

- Add Streamlit or Gradio frontend for interactive predictions
- Deploy using Flask or FastAPI for real-time classification
- Integrate more robust NLP features (lemmatization, POS tagging, etc.)
