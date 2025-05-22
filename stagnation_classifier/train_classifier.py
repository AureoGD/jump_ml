import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, root_mean_squared_error

from dataset import load_dataset
from models import CNNStagnationClassifier

# --------------------
# Configuration
# --------------------
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "stagnation_classifier/models/cnn_regressor.pth"

# --------------------
# Load dataset
# --------------------
manual_dir = "stagnation_classifier/labeled_episodes_manual"
train_dataset, test_dataset = load_dataset(manual_dir, augment=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# --------------------
# Get input shapes
# --------------------
sample = train_dataset[0]
in_channels_base = sample["base_past"].shape[0]  # ✔️ Should be 6
in_channels_joint = sample["joint_past"].shape[0]  # ✔️ Should be 12
in_channels_comp = sample["comp_past"].shape[0]  # ✔️ Should be 5
seq_len = sample["base_past"].shape[1]  # Time steps (e.g., 30)

input_channels = in_channels_base + in_channels_joint + in_channels_comp
print(f"Input channels: {input_channels} | Sequence length: {seq_len}")

# --------------------
# Initialize model
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNNStagnationClassifier(input_channels, seq_len).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------
# Training loop
# --------------------
for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    preds_list = []
    labels_list = []

    for batch in train_loader:
        base = batch["base_past"].to(device)
        joint = batch["joint_past"].to(device)
        comp = batch["comp_past"].to(device)

        inputs = torch.cat([base, joint, comp], dim=1).to(device)  # (B, C, T)
        labels = batch["label"].to(device).float().unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        preds_list.extend(outputs.squeeze().detach().cpu().numpy())
        labels_list.extend(labels.squeeze().cpu().numpy())

    # Metrics
    r2 = r2_score(labels_list, preds_list)
    rmse = root_mean_squared_error(labels_list, preds_list)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {sum(train_losses) / len(train_losses):.4f} "
          f"- RMSE: {rmse:.4f} - R2: {r2:.4f}")

# --------------------
# Evaluation
# --------------------
model.eval()
preds_list = []
labels_list = []

with torch.no_grad():
    for batch in test_loader:
        base = batch["base_past"].to(device)
        joint = batch["joint_past"].to(device)
        comp = batch["comp_past"].to(device)

        inputs = torch.cat([base, joint, comp], dim=1).to(device)
        labels = batch["label"].to(device).float().unsqueeze(1)

        outputs = model(inputs)
        preds_list.extend(outputs.squeeze().cpu().numpy())
        labels_list.extend(labels.squeeze().cpu().numpy())

rmse = root_mean_squared_error(labels_list, preds_list)
r2 = r2_score(labels_list, preds_list)
print(f"\n✅ Test RMSE: {rmse:.4f} | Test R2: {r2:.4f}")

# --------------------
# Save model
# --------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"🧠 Model saved to: {MODEL_SAVE_PATH}")
