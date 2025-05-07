import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataset import load_dataset
from models import CNNStagnationClassifier

# Training configuration
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "stagnation_classifier/models/cnn_classifier_IV.pth"

# Load dataset
manual_dir = "stagnation_classifier/labeled_episodes_manual"

train_dataset, test_dataset = load_dataset(manual_dir, augment=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Get input shapes from a sample
sample = train_dataset[0]
in_channels_pred = sample["predictable"].shape[0]
in_channels_nonp = sample["nonpredictable"].shape[0]
seq_len = sample["predictable"].shape[1]

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNNStagnationClassifier(in_channels_pred + in_channels_nonp, seq_len).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    all_preds = []
    all_labels = []

    for batch in train_loader:
        pred = batch["predictable"].to(device)
        nonp = batch["nonpredictable"].to(device)
        inputs = torch.cat([pred, nonp], dim=1).to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    print(
        f"Epoch {epoch + 1}/{EPOCHS} - Loss: {sum(train_losses) / len(train_losses):.4f} - Train Acc: {train_acc:.3f}"
    )

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        pred = batch["predictable"].to(device)
        nonp = batch["nonpredictable"].to(device)
        inputs = torch.cat([pred, nonp], dim=1).to(device)
        labels = batch["label"].to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
print(f"\n✅ Test Accuracy: {test_acc:.3f}")

# Save model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"🧠 Model saved to: {MODEL_SAVE_PATH}")
