# 📩 SMS Spam Detection with ML + LLM Explanation

## 📌 Overview
This project uses **Machine Learning** to classify SMS messages as **Spam** or **Not Spam**, and integrates a **Large Language Model (LLM)** to provide human-readable explanations for each prediction. It's deployed as a **Streamlit web app** for real-time user interaction.

---

## 🛠️ Technologies Used
- Python
- scikit-learn
- pandas, numpy
- Streamlit
- OpenAI GPT / Ollama (LLM)
- matplotlib, seaborn, wordcloud

---

## 🔍 Features
- SMS classification using ML
- LLM-based natural language explanation
- Exploratory data analysis and visualizations
- Web interface with Streamlit
- Multiple classifier evaluation

---

## 📂 Dataset
- **Source:** [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- ~5,500 labeled SMS messages
- Two classes: `spam`, `ham`

---

## 📊 Model Building
Tested:
- Naive Bayes ✅
- Random Forest, KNN, SVM, Logistic Regression

**Final Model:**  
- **Naive Bayes** with TF-IDF  
- **Precision:** 100% (Spam class)

---

## 🤖 LLM Integration (New!)
Each ML prediction is followed by a natural explanation generated using **OpenAI GPT** or **Ollama**.

**Example:**

> **Input Message:** “You’ve won a free gift card. Click to claim.”  
> **ML Prediction:** Spam  
> **LLM Explanation:**  
> “This message uses promotional language and urges immediate action, which is typical of spam.”

---

## 🚀 Web Deployment

### ▶️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/hariharang25/SMS-spamDetection.git
cd SMS-spamDetection

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API Key (or configure Ollama)
export OPENAI_API_KEY=your-api-key

# Run the Streamlit app
streamlit run app.py
