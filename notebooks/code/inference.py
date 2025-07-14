import joblib
import torch
import numpy as np
import os
import json
from transformers import DistilBertTokenizer, DistilBertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "logreg_model.joblib"))
    le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    bert_dir = os.path.join(model_dir, "bert")
    
    tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(model_dir, "bert"))
    bert = DistilBertModel.from_pretrained(os.path.join(model_dir, "bert")).to(device)

    bert.eval()

    return {"clf": clf, "le": le, "bert": bert, "tokenizer": tokenizer}

def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        data = json.loads(request_body)
        return data["text"]
    raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(text, model_artifacts):
    tokenizer = model_artifacts["tokenizer"]
    bert = model_artifacts["bert"]
    clf = model_artifacts["clf"]
    le = model_artifacts["le"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = bert(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    vec = output.reshape(1, -1)

    probs = clf.predict_proba(vec)[0]
    pred_idx = np.argmax(probs)
    label = le.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx]

    return {"label": label, "confidence": float(confidence)}

def output_fn(prediction, content_type='application/json'):
    return json.dumps(prediction)
