# Real-Time Complaint Classifier (BERT + Logistic Regression)

Classifies consumer complaints in real-time using a BERT-based NLP pipeline deployed on AWS.

---

## 🚀 Overview

- **Model**: DistilBERT embeddings + Logistic Regression classifier (scikit-learn)
- **Serving**: SageMaker real-time endpoint with custom inference logic
- **Streaming**: AWS Kinesis triggers Lambda for live classification
- **Logging**: Predictions are logged to S3 for retraining loop (Agentify-ready)
- **Latency**: <2s end-to-end from ingestion to classification

---

## 🛠️ Tech Stack

- AWS SageMaker (script mode PyTorch endpoint)
- AWS Kinesis Data Stream
- AWS Lambda (Python 3.9)
- AWS S3 (CSV-based logging)
- HuggingFace Transformers
- Scikit-learn
- Joblib

---

## 📦 Model Artifacts

- `logreg_model.joblib` – Trained classifier
- `label_encoder.joblib` – Label encoder
- `bert/` – Tokenizer and DistilBERT weights
- `model.tar.gz` – Final deployable archive

---
### 📊 NLP Model Evaluation (DistilBERT + Logistic Regression)

- Evaluated on 32,484 real-world consumer complaint narratives
- Encoder: DistilBERT (mean pooled, frozen)
- Classifier: Logistic Regression
- Per-record identification and audit loop implemented
- Retraining pipeline is already in place

**Metrics (No Fine-Tuning):**
- Precision: 45.1%
- Recall: 52.2%
- F1 Score: 40.8%
- Avg Inference Latency: 0.0088 sec per transaction (batchless, GPU)

This baseline reflects the limitations of a frozen transformer + linear classifier approach.  
Real-time infra is pre-wired but gated behind accuracy thresholds.  
**Next step:** upgrade classifier from Logistic Regression to XGBoost or shallow MLP to improve semantic discrimination and reduce class imbalance effects.


---
## 🧪 Sample Inference

```bash
Input:
"I was charged extra interest after my loan was closed."

Output:
{
  "label": "mortgages_and_loans",
  "confidence": 0.9758
}
```

---

## 📁 Directory Structure

```

├── code/
│   └── inference.py
├── bert/                 ← (optional if you store locally)
├── logreg_model.joblib
├── label_encoder.joblib
├── model.tar.gz
├── lambda/
│   └── lambda_handler.py
├── notebooks/
│   └── 01_data_cleaning.ipynb
│   └── 02_model_training.ipynb
│   └── 03_inference_testing.ipynb
│   └── 05_deploy_sagemaker.ipynb
├── README.md


---

## 👤 Author

Jayshree Pillai – Machine Learning Engineer  
