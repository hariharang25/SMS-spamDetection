from flask import Flask, request, render_template
import pickle
from explain_llm import generate_explanation

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/', methods=["GET", "POST"])
def index():
    explanation = prediction = ""
    if request.method == "POST":
        message = request.form["message"]
        vec_msg = vectorizer.transform([message])
        prediction = model.predict(vec_msg)[0]
        prediction = "Spam" if prediction == 1 else "Not Spam"
        explanation = generate_explanation(message, prediction)
    return render_template("index.html", prediction=prediction, explanation=explanation)

if __name__ == '__main__':
    app.run(debug=True)
