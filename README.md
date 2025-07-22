# 🧠 Real-Time NLP Complaint Classifier

A production-grade **multi-class complaint classification system** built using **DistilBERT + custom classification head**, fine-tuned end-to-end for imbalanced classes in financial complaints.  
Trained on **129K real-world complaints** with a **32K validation set**, deployed with full audit logging, and real-time inference using AWS infrastructure— containerized, deployed to SageMaker, and built with production-ready retraining hooks.

---

## 🎬 Demo

▶️ [Watch 3-min Video Demo](https://hatketech-demos.s3.amazonaws.com/distilbert-demo.mp4)  
*End-to-end pipeline with DistilBERT fine-tuning, Dockerized Flask API, and SageMaker deployment.*

---

## 🚨 Problem Statement

Given a customer complaint, predict which of the following categories it belongs to:
- Credit Reporting
- Debt Collection
- Mortgages/Loans
- Credit Card
- Retail Banking

Challenges include:
- **Severe class imbalance**
- **Semantic overlap across labels**
- **Precision-critical downstream usage**

---

## ⚙️ Project Overview

- **Model**: DistilBERT fine-tuned with a PyTorch-based custom classification head
- **Serving**: Real-time endpoint deployed via AWS SageMaker (script mode)
- **Inference Logic**: AWS Lambda (Python 3.9) triggered via Kinesis
- **Streaming**: New complaints flow through AWS Kinesis → Lambda → SageMaker
- **Logging**: Predictions stored in S3 (CSV format) for audit and retraining
- **Retraining Hooks**: Supports S3-based retraining loop with Agentify integration
- **Latency**: End-to-end classification in **<2 seconds**

---

## 🧰 Tech Stack

- AWS SageMaker (real-time endpoint, PyTorch script mode)
- AWS Kinesis Data Stream
- AWS Lambda (Python 3.9)
- AWS S3 (CSV-based prediction logging)
- HuggingFace Transformers (DistilBERT + Trainer)
- Scikit-learn (metrics & eval)
- Joblib (model serialization)
- Docker (custom image for model + tokenizer)
- Flask API (containerized and deployed via SageMaker)

---


## NLP Model Evaluation 

This classifier was trained on 129K labeled complaints, tested on a 32K holdout set, and evaluated on a 5-class imbalanced classification task using a custom DistilBERT classification head.

Final Model (Fine-Tuned DistilBERT + Custom Head)
| Metric             | Score                                      |
| ------------------ | ------------------------------------------ |
| Macro F1           | 0.85                                       |
| Weighted F1        | 0.88                                       |
| Precision          | 0.89                                       |
| Recall             | 0.86                                       |
| Avg Inference Time | \~0.22 sec (real-time, SageMaker endpoint) |

Per-Class Highlights:
- Credit Reporting: 0.93 precision, 0.87 recall
- Debt Collection: 0.86 precision, 0.73 recall
- Retail Banking: 0.89 precision, 0.88 recall

Baseline Comparison (Frozen DistilBERT + Logistic Regression)
| Metric             | Score                       |
| ------------------ | --------------------------- |
| F1 Score           | 0.408                       |
| Precision          | 45.1%                       |
| Recall             | 52.2%                       |
| Avg Inference Time | 0.0088 sec (batchless, GPU) |

**Metrics (No Fine-Tuning):**
- Precision: 45.1%
- Recall: 52.2%
- F1 Score: 40.8%

**Key Issues:**
- Severe underperformance on minority classes
- No task-specific adaptation
- Overfit to dominant label (Credit Reporting)

---

## Directory Structure

```

real-time-nlp-complaint-classifier/
├── notebooks/
│   ├── 01_data_split_and_embed.ipynb              # Tokenization and CLS vector extraction
│   ├── 02_model_train_distilbert_logreg.ipynb     # Baseline: frozen DistilBERT + Logistic Regression
│   ├── 03_model_finetune_distilbert.ipynb         # End-to-end fine-tuning of DistilBERT
│   ├── 04_model_log_agentify.ipynb                # Log predictions for Agentify integration
│   ├── 05_model_deploy_sagemaker.ipynb            # Deploy fine-tuned model to SageMaker endpoint
│   ├── 06_model_retrain_agent_loop.ipynb          # Simulated retraining loop using prediction logs
│   ├── 07_real_time_complaint_inference_pipeline.ipynb  # Real-time simulation: ingest → predict → log
│   ├── 99_utils_explore.ipynb                     # Ad hoc tools and utility functions
│   ├── evaluate_nlp_metrics.ipynb                 # Final evaluation: precision, recall, F1 (with confusion matrix)│
├── data/
│   ├── complaints_train.csv
│   └── complaints_test.csv
├── src/
│   ├── train_classifier.py        # Fine-tunes DistilBERT with custom head
│   ├── predict.py                 # Real-time inference logic
│   ├── model_utils.py             # Custom head definition & metrics
│   └── config.json                # Hyperparams & model settings
├── scripts/
│   └── evaluate_baseline.py       # Runs frozen BERT + logistic regression
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview & instructions
└── .gitignore                    # Ignored files (e.g., logs, checkpoints)


---

## Author

Jayshree Pillai – Machine Learning Engineer  
📫 [LinkedIn](https://linkedin.com/in/jspillai)

---

## 🧩 Next Steps

- Migrate to FastAPI + gunicorn
- CI/CD integration for automated deployment
- Inference monitoring with Prometheus/Grafana

---