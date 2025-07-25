from model import DistilBERTWithCustomHead
from transformers import DistilBertTokenizerFast
import torch
from flask import Flask, request, jsonify
import os
from transformers import logging
logging.set_verbosity_error()

os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATH = "/opt/ml/model"

print("🔥 Starting inference container...")
print("📦 Looking for model in:", MODEL_PATH)
print("📄 Files in model dir:", os.listdir(MODEL_PATH))

# Load model
model = DistilBERTWithCustomHead(num_labels=5)
model.load_state_dict(torch.load(f"{MODEL_PATH}/model.pt", map_location=torch.device("cpu")))
model.eval()

assert os.path.exists(os.path.join(MODEL_PATH, "tokenizer.json")), "❌ tokenizer.json missing"
assert os.path.exists(os.path.join(MODEL_PATH, "vocab.txt")), "❌ vocab.txt missing"

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

# Init Flask app
app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return "OK", 200

@app.route("/invocations", methods=["POST"])
def predict():
    data = request.get_json()
    inputs = tokenizer(data["inputs"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs["logits"], dim=1).tolist()
    return jsonify(probs)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

    
