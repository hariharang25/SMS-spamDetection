from flask import Flask, render_template, request
import pickle
from explain_llm import generate_explanation
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
ps = PorterStemmer()

# Load ML model and vectorizer
model = pickle.load(open("model.pkl", 'rb'))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    y = []
    for i in tokens:
        if i.isalnum():
            y.append(i)

    tokens = y[:]
    y.clear()

    for i in tokens:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    tokens = y[:]
    y.clear()

    for i in tokens:
        y.append(ps.stem(i))

    return " ".join(y)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    explanation = ""
    if request.method == "POST":
        input_sms = request.form["sms"]
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        prediction = "Spam" if result == 1 else "Not Spam"
        explanation = generate_explanation(input_sms, prediction)
    return render_template("index.html", prediction=prediction, explanation=explanation)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

