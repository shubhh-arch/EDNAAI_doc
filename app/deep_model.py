# app/deep_model.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Where the trained weights will live later
MODEL_PATH = os.path.join("models", "cnn_dbn.pth")


class CnnDbnClassifier(nn.Module):
    """
    Simple 1D CNN + DBN-style (sigmoid) hidden layer + classifier.
    This model works purely on k-mer vectors (sequence only, no Entrez).
    Input shape for forward(): [batch, 1, L] where L = vector length.
    """

    def __init__(self, input_len: int, num_classes: int = 2):
        super().__init__()

        # 1D convolution layers (feature extraction over k-mer axis)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # After two pool(2): length roughly L/4
        reduced_len = max(input_len // 4, 1)
        self.flat_dim = reduced_len * 32

        # DBN-style hidden layer (sigmoid = like RBM/DBN activations)
        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, 1, L]
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # flatten

        # DBN-like latent representation
        x = torch.sigmoid(self.fc1(x))
        logits = self.fc2(x)
        return logits


def load_cnn_dbn(input_len: int):
    """
    Try to load CNN+DBN model from disk.
    If weights not found, return (None, device) so the pipeline can skip deep scoring.

    This keeps the app working even before we train the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnDbnClassifier(input_len=input_len)

    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        print(f"✅ Loaded CNN+DBN model from {MODEL_PATH}")
        return model, device
    else:
        print("⚠️ CNN+DBN weights not found, skipping deep scoring for now.")
        return None, device


def predict_cnn_dbn(model, device, sample_matrix: np.ndarray):
    """
    Run inference on sample_matrix (numpy array [N, L]).
    Returns 1D numpy array of probabilities for class 1 (e.g., Marine / Deep-sea).
    If model is None, returns zeros so pipeline continues safely.
    """
    if model is None:
        # No trained model yet → return zeros (no contribution to HybridScore)
        return np.zeros(sample_matrix.shape[0], dtype=np.float32)

    # Convert to float32 and standardize
    X = sample_matrix.astype(np.float32)
    mean = X.mean()
    std = X.std() + 1e-6
    X = (X - mean) / std

    # Shape into [B, 1, L]
    X_tensor = torch.from_numpy(X).unsqueeze(1).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1]  # probability of class 1

    return probs.cpu().numpy()
