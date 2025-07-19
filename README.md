# Real-Time NLP Complaint Classifier

A production-grade multi-class complaint classification system built using DistilBERT + custom classifier head, fine-tuned end-to-end for imbalanced classes in financial complaints.
Trained on 129K real-world complaints with a 32K validation set, deployed with full audit logging, and designed for real-time inference.
---

## Problem Statement 

Given a customer complaint, predict which of the following categories it belongs to:
- Credit Reporting
- Debt Collection
- Mortgages/Loans
- Credit Card
- Retail Banking
The challenge: imbalanced classes, semantic overlap, and high precision requirements for downstream routing.

---

## Overview

- Model: DistilBERT fine-tuned on 129K complaints with a custom classification head (PyTorch)
- Serving: Deployed to AWS SageMaker (script mode, real-time endpoint)
- Inference Logic: Lambda (Python 3.9) pulls from Kinesis, calls model, formats prediction
- Streaming: AWS Kinesis triggers Lambda on new complaint ingestion
- Logging: Predictions logged to AWS S3 (CSV format) for retraining, auditing, and Agentify integration
- Retraining Hooks: S3-stored predictions + labels support scheduled retraining pipelines
- Latency: End-to-end pipeline completes in under 2 seconds from ingestion to classification

---

## Tech Stack

- AWS SageMaker (PyTorch script mode endpoint)
- AWS Lambda (Python 3.9)
- AWS Kinesis Data Stream
- AWS S3 (CSV-based prediction logging)
- HuggingFace Transformers (DistilBERT, custom classification head)
- Scikit-learn (evaluation, metrics)
- Joblib (model serialization)

---

## NLP Model Evaluation (DistilBERT + Logistic Regression)

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
Key Issues:
- Severe underperformance on minority classes
- No task-specific adaptation
- Overfit to dominant label (Credit Reporting)

---

## Directory Structure

```

real-time-nlp-complaint-classifier/
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
