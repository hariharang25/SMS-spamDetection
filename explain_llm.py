import requests

def generate_explanation(message, prediction):
    prompt = f"""
You are an SMS classification assistant. A machine learning model predicted the following SMS as '{prediction}'.

Message: "{message}"

Give a simple 2–3 sentence explanation for why this prediction makes sense.
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",  # or mistral, phi, etc.
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()['response'].strip()
    except Exception as e:
        return f"⚠️ LLM error: {str(e)}"
