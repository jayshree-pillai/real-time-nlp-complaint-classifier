import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import ComplaintDataset
from sklearn.preprocessing import LabelEncoder
from model import DistilBERTWithCustomHead
import pandas as pd
from tqdm import tqdm

# ==== Config ====
MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "../models/best_model"

import os
os.makedirs(SAVE_PATH, exist_ok=True)
# ==== Data ====
df = pd.read_csv("../notebooks/downloads/complaints_train.csv")
df_clean = df[["narrative", "product"]].dropna().astype(str)

le = LabelEncoder()
le.fit(df_clean["product"])

from sklearn.model_selection import train_test_split

texts = df_clean["narrative"].tolist()
labels = le.transform(df_clean["product"].tolist())
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)

# ==== Dataset & Loader ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = ComplaintDataset(X_train, y_train, tokenizer)
val_dataset = ComplaintDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# ==== Model ====
model = DistilBERTWithCustomHead(num_labels=len(le.classes_))
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # === TRAIN ===
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # === VALIDATION ===
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"{SAVE_PATH}/model.pt")
        tokenizer.save_pretrained(SAVE_PATH)
        print("✅ Best model saved.")

