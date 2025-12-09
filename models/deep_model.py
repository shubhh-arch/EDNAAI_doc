import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================
# 1. DNA One-Hot Encoder
# =============================================================
class DNAEncoder:
    """
    Convert raw DNA sequence (string) -> one-hot tensor.
    A, C, G, T = 4 channels.
    """
    def __init__(self, seq_len=1500):
        self.seq_len = seq_len
        self.map = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3
        }

    def encode(self, seq):
        seq = seq.upper()

        tensor = torch.zeros(4, self.seq_len)

        for i in range(min(len(seq), self.seq_len)):
            base = seq[i]
            if base in self.map:
                tensor[self.map[base], i] = 1.0

        return tensor


# =============================================================
# 2. Restricted Boltzmann Machine (RBM) for DBN
# =============================================================
class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden_units, visible_units) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_units))
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))

    def sample_h(self, v):
        # sigmoid(Wv + b)
        h_prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return h_prob, torch.bernoulli(h_prob)

    def sample_v(self, h):
        # sigmoid(W^T h + c)
        v_prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return v_prob, torch.bernoulli(v_prob)

    def forward(self, v):
        h_prob, h_sample = self.sample_h(v)
        return h_prob


# =============================================================
# 3. CNN + DBN Hybrid Feature Extractor
# =============================================================
class CNN_DBM_Model(nn.Module):
    def __init__(self, seq_len=1500):
        super().__init__()

        self.seq_len = seq_len

        # Convolution layers
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Output length after 3 maxpools
        cnn_out_len = seq_len // 8

        self.flat_dim = 128 * cnn_out_len

        # DBN (2 RBM layers)
        self.rbm1 = RBM(self.flat_dim, 512)
        self.rbm2 = RBM(512, 256)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # marine vs non-marine
        )

    def forward(self, x):
        # x: (batch, 4, seq_len)
        feat = self.cnn(x)
        feat = feat.reshape(feat.size(0), -1)

        # RBM feature extraction (no Gibbs sampling during inference)
        h1 = self.rbm1(feat)
        h2 = self.rbm2(h1)

        logits = self.classifier(h2)
        return logits

    def get_embedding(self, x):
        """
        Returns the 256-D deep feature — used for:
        - Novelty detection
        - UMAP / HDBSCAN
        - Visualization
        """
        feat = self.cnn(x)
        feat = feat.reshape(feat.size(0), -1)

        h1 = self.rbm1(feat)
        h2 = self.rbm2(h1)

        return h2


# =============================================================
# 4. Utility function to load pretrained model
# =============================================================
def load_marine_classifier(path="models/pretrained/marine_classifier.pt",
                           seq_len=1500):

    model = CNN_DBM_Model(seq_len=seq_len)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

