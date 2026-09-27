from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def output_file(filename):
    return send_from_directory(
        os.path.join(app.root_path, "outputs"),
        filename
    )


@app.route("/health")
def health():
    return {
        "status": "healthy",
        "project": "NeuralPy",
        "test_accuracy": "97.97%",
        "framework": "NumPy"
    }


if __name__ == "__main__":
    app.run(debug=True)