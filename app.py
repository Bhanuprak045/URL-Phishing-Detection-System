import os
import pickle
import re
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, session, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.secret_key = 'your_secret_key'

MODEL_PATH = "BhanuPrakash-2210636-RNN.h5"
model = load_model(MODEL_PATH)

with open("tokenized_data.pkl", "rb") as f:
    tokenized_data = pickle.load(f)
tokenizer = tokenized_data["tokenizer"]

MAX_LEN = 200

def is_valid_url(url):
    """Checks if the given string is a valid URL format."""
    regex = re.compile(
        r'^(https?:\/\/)' 
        r'([a-zA-Z0-9.-]+)' 
        r'(\.[a-zA-Z]{2,})'
        r'(:\d+)?' 
        r'(\/.*)?$'
    )
    return re.match(regex, url) is not None


def preprocess_url(url):
    """Tokenizes and pads the URL."""
    seq = tokenizer.texts_to_sequences([url])
    padded_seq = pad_sequences(seq, maxlen=MAX_LEN)
    return padded_seq

@app.route("/")
def home():
    """Render the main page."""
    return render_template("index.html")

@app.route("/input", methods=["GET", "POST"])
def input_url():
    """Page to enter URL or upload a file."""
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        file = request.files.get("file")
        valid_urls = []
        invalid_urls = []  # Store invalid URLs from file input

        if url:
            if not is_valid_url(url):
                return render_template("input.html", error="Invalid URL format. Please enter a valid URL.")
            valid_urls.append(url)

        if file:
            filename = file.filename.lower()
            
            # Check if file is empty
            if file.content_length == 0 or file.read(1) == b"":  
                return render_template("input.html", error="Uploaded file is empty. Please upload a valid file.")

            file.seek(0)  # Reset file pointer after checking for emptiness
            
            if filename.endswith(".txt"):
                urls = [line.decode("utf-8").strip() for line in file.readlines()]
            elif filename.endswith(".csv"):
                df = pd.read_csv(file, header=None, dtype=str)
                urls = df[0].dropna().tolist()
            else:
                return render_template("input.html", error="Invalid file format. Upload a .txt or .csv file.")

            # Separate valid and invalid URLs
            for u in urls:
                if is_valid_url(u):
                    valid_urls.append(u)
                else:
                    invalid_urls.append(u)

        if not valid_urls and not invalid_urls:
            return render_template("input.html", error="No valid URLs provided.")

        session["urls"] = valid_urls  # Store valid URLs
        session["invalid_urls"] = invalid_urls  # Store invalid URLs separately
        return redirect(url_for("confirm"))

    return render_template("input.html")


@app.route("/confirm")
def confirm():
    urls = session.get("urls", [])
    invalid_urls = session.get("invalid_urls", [])
    return render_template("confirm.html", urls=urls, invalid_urls=invalid_urls)


@app.route("/predict")
def predict():
    """Classifies stored URLs and redirects to output page."""
    urls = session.get("urls", [])
    invalid_urls = session.get("invalid_urls", [])  # Get stored invalid URLs

    if not urls and not invalid_urls:
        return redirect(url_for("home"))

    results = []

    # Predict only for valid URLs
    for url in urls:
        processed_url = preprocess_url(url)
        prediction = model.predict(processed_url)
        predicted_label = "Malicious" if prediction[0][0] > 0.5 else "Benign"
        results.append((url, predicted_label))  # Store valid URLs with predictions

    # Append invalid URLs with "Incorrect URL" label
    for invalid_url in invalid_urls:
        results.append((invalid_url.strip(), "Incorrect URL"))

    session["output"] = results  # Store for display
    return redirect(url_for("output"))


@app.route("/output")
def output():
    """Displays URLs and prediction results, including invalid URLs."""
    results = session.get("output", [])
    return render_template("output.html", results=results)

@app.route("/store_output")
def store_output():
    """Generate and return a CSV file containing valid and invalid URLs."""
    results = session.get("output", [])

    if not results:
        return redirect(url_for("home"))

    def generate_csv():
        """Generator function to stream CSV data."""
        yield "URL,Prediction\n"
        for url, result in results:
            yield f"{url},{result}\n"

    response = Response(generate_csv(), content_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    return response

if __name__ == "__main__":
    app.run(debug=True)
