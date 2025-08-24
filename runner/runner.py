from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for Chrome extension

@app.route("/run", methods=["POST"])
def run_pipeline():
    data = request.json
    match_id = data.get("match_id")

    if not match_id:
        return jsonify({"error": "No match_id provided"}), 400

    try:
        # 1. Run matchstater.py with match_id
        subprocess.run(["python", "matchstater.py", match_id], check=True)

        # 2. Run statschecker 3.py
        subprocess.run(["python", "statschecker 3.py"], check=True)

        # 3. Run machine learning model.py
        subprocess.run(["python", "machine learning model.py"], check=True)

        # 4. Load predictions.json and return as response
        with open('predictions.json', 'r') as f:
            predictions = json.load(f)

        return jsonify(predictions)

    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)