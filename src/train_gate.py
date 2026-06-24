import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import string
from collections import defaultdict
import math
import os

# -------------------- Configuration --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Vocabulary (must match the one used during expert training)
chars = string.ascii_lowercase
char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
char_to_idx['_'] = 27
char_to_idx['<PAD>'] = 0
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx)

# Load the list of words (same as before)
with open('words_250000_train.txt', 'r') as f:
    words = f.read().splitlines()
words = [w.lower() for w in words if w.isalpha() and len(w) > 2]

# Use the same train/eval split (seed=42)
random.seed(42)
random.shuffle(words)
split_idx = int(len(words) * 0.8)
train_words = words[:split_idx]
eval_words = words[split_idx:]

print(f"Train words: {len(train_words)}, Eval words: {len(eval_words)}")

# -------------------- Load the five pre‑trained experts --------------------
class HangmanBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=char_to_idx['<PAD>'])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, 26)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(out)

# Expert names and their corresponding file names
expert_names = ['short', 'medium', 'long', 'common', 'rare']
experts = {}

# Hyperparameters (must match training)
EMBEDDING_DIM = 128
HIDDEN_DIM = 512

for name in expert_names:
    model = HangmanBiLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    path = f"expert_{name}_bilstm.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expert file {path} not found. Please train experts first.")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    experts[name] = model
    print(f"Loaded expert: {name}")

# -------------------- Helper functions for game state and expert probabilities --------------------
def get_expert_probs(models, input_tensor, device):
    """
    For a given input tensor (1, seq_len), return a list of 5 probability vectors (each of length 26),
    where each vector is the expert's average probability over the masked positions.
    """
    mask = (input_tensor == char_to_idx['_']).float()  # (1, seq_len)
    probs_list = []
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)                     # (1, seq_len, 26)
            probs = torch.softmax(logits, dim=-1)            # (1, seq_len, 26)
            # Average over masked positions
            masked_probs = (probs * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            probs_list.append(masked_probs.squeeze(0))       # (26)
    return probs_list  # list of 5 tensors of shape (26) on device

def get_state_features(input_seq, guessed_letters, incorrect_count, device):
    """
    Build a fixed-size feature vector from the current game state.
    input_seq: list of character indices (length L)
    guessed_letters: set of letters already guessed
    incorrect_count: int (0-6)
    device: torch device
    Returns a tensor of shape (29) on the given device.
    """
    L = len(input_seq)
    blanks = sum(1 for idx in input_seq if idx == char_to_idx['_'])
    # Normalized scalars
    length_norm = L / 20.0                      # assume max length 20
    blanks_norm = blanks / L if L > 0 else 0.0
    incorrect_norm = incorrect_count / 6.0

    # Guessed letters binary vector (26)
    guessed_vec = torch.zeros(26, device=device)
    for ch in guessed_letters:
        guessed_vec[char_to_idx[ch] - 1] = 1.0

    base = torch.tensor([length_norm, blanks_norm, incorrect_norm], dtype=torch.float32, device=device)
    return torch.cat([base, guessed_vec])  # size 29

# -------------------- Generate training data for the gating network --------------------
def simulate_and_collect(models, words, device, max_games=None):
    """
    Play games using the uniform ensemble on the given words.
    For each turn, record:
      - state features (29-dim base + 5*26 expert probs)
      - binary labels for each expert (1 if that expert's top guess would be correct)
    Returns:
      features: list of torch tensors (each 159-dim, on CPU)
      labels: list of torch tensors (each 5-dim binary, on CPU)
    """
    features = []
    labels = []
    if max_games is None:
        max_games = len(words)
    words_sample = random.sample(words, min(max_games, len(words)))

    for word in words_sample:
        guessed_letters = set()
        incorrect = 0
        current_board = ['_'] * len(word)
        # Play until game over
        while incorrect < 6 and '_' in current_board:
            input_seq = [char_to_idx[c] for c in current_board]
            input_tensor = torch.tensor([input_seq]).to(device)

            # Get expert probabilities (averaged over blanks)
            probs_list = get_expert_probs(models, input_tensor, device)  # list of 5 tensors (26) on device

            # For each expert, determine its top guess after masking guessed letters
            expert_correct = []
            for prob in probs_list:
                masked_prob = prob.clone()
                for ch in guessed_letters:
                    masked_prob[char_to_idx[ch] - 1] = -float('inf')
                top_letter_idx = torch.argmax(masked_prob).item()
                top_letter = idx_to_char[top_letter_idx + 1]
                # Check if this letter is in the word and not yet guessed
                if top_letter in word and top_letter not in guessed_letters:
                    expert_correct.append(1.0)
                else:
                    expert_correct.append(0.0)

            # Build state features
            base_feat = get_state_features(input_seq, guessed_letters, incorrect, device)
            expert_feat = torch.cat(probs_list)  # 5*26 = 130, on device
            full_feat = torch.cat([base_feat, expert_feat])  # 29 + 130 = 159, on device

            # Move to CPU for storage
            features.append(full_feat.cpu())
            labels.append(torch.tensor(expert_correct, dtype=torch.float32).cpu())

            # Now make the actual guess using the uniform ensemble to advance the game
            def uniform_ensemble_predict(models, input_tensor, guessed_letters):
                probs_list = get_expert_probs(models, input_tensor, device)
                avg_probs = torch.stack(probs_list).mean(dim=0)
                for ch in guessed_letters:
                    avg_probs[char_to_idx[ch] - 1] = -float('inf')
                return torch.argmax(avg_probs).item()

            guess_idx = uniform_ensemble_predict(models, input_tensor, guessed_letters)
            guess_char = idx_to_char[guess_idx + 1]
            guessed_letters.add(guess_char)

            if guess_char in word:
                for i, c in enumerate(word):
                    if c == guess_char:
                        current_board[i] = guess_char
            else:
                incorrect += 1

    return features, labels

# Generate data (use e.g., 10,000 games to keep it manageable)
print("Generating training data from simulated games...")
train_features, train_labels = simulate_and_collect(list(experts.values()), train_words, device, max_games=10000)
print(f"Collected {len(train_features)} turns.")

# -------------------- Dataset for gating network --------------------
class GateDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Split into train/val for the gate (from the generated data)
random.seed(42)
indices = list(range(len(train_features)))
random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]
train_feat = [train_features[i] for i in train_idx]
train_lab = [train_labels[i] for i in train_idx]
val_feat = [train_features[i] for i in val_idx]
val_lab = [train_labels[i] for i in val_idx]

train_dataset = GateDataset(train_feat, train_lab)
val_dataset = GateDataset(val_feat, val_lab)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# -------------------- Gating Network --------------------
class GatingNetwork(nn.Module):
    def __init__(self, input_dim=159, hidden_dims=[256, 128], output_dim=5):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # raw logits

gate = GatingNetwork().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(gate.parameters(), lr=0.001)

# -------------------- Training Loop --------------------
epochs = 20
best_val_loss = float('inf')
for epoch in range(epochs):
    gate.train()
    train_loss = 0.0
    for feat, lab in train_loader:
        feat, lab = feat.to(device), lab.to(device)
        optimizer.zero_grad()
        logits = gate(feat)
        loss = criterion(logits, lab)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    gate.eval()
    val_loss = 0.0
    with torch.no_grad():
        for feat, lab in val_loader:
            feat, lab = feat.to(device), lab.to(device)
            logits = gate(feat)
            loss = criterion(logits, lab)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(gate.state_dict(), "best_gating_network.pt")
        print("  -> Saved best gate")

print("Gating network training complete.")

# -------------------- Evaluation with Gated Ensemble --------------------
def play_hangman_gated(models, gate, word, device):
    gate.eval()
    guessed_letters = set()
    incorrect = 0
    current_board = ['_'] * len(word)

    while incorrect < 6 and '_' in current_board:
        input_seq = [char_to_idx[c] for c in current_board]
        input_tensor = torch.tensor([input_seq]).to(device)

        # Get expert probabilities
        probs_list = get_expert_probs(models, input_tensor, device)

        # Build features
        base_feat = get_state_features(input_seq, guessed_letters, incorrect, device)
        expert_feat = torch.cat(probs_list)
        full_feat = torch.cat([base_feat, expert_feat]).unsqueeze(0).to(device)  # (1,159)

        # Get gate weights (softmax over logits)
        with torch.no_grad():
            gate_logits = gate(full_feat).squeeze(0)  # (5)
            gate_weights = torch.softmax(gate_logits, dim=0)  # (5)

        # Weighted average of expert probabilities
        weighted_probs = torch.zeros(26, device=device)
        for w, p in zip(gate_weights, probs_list):
            weighted_probs += w * p

        # Mask guessed letters
        for ch in guessed_letters:
            weighted_probs[char_to_idx[ch] - 1] = -float('inf')

        guess_idx = torch.argmax(weighted_probs).item()
        guess_char = idx_to_char[guess_idx + 1]
        guessed_letters.add(guess_char)

        if guess_char in word:
            for i, c in enumerate(word):
                if c == guess_char:
                    current_board[i] = guess_char
        else:
            incorrect += 1

    return '_' not in current_board

def evaluate_gated_ensemble(models, gate, eval_words, device, num_samples=500):
    samples = random.sample(eval_words, min(num_samples, len(eval_words)))
    wins = 0
    for w in samples:
        if play_hangman_gated(models, gate, w, device):
            wins += 1
    return wins / len(samples)

# Load best gate
gate.load_state_dict(torch.load("best_gating_network.pt", map_location=device))
print("\nEvaluating gated ensemble on eval set...")
win_rate = evaluate_gated_ensemble(list(experts.values()), gate, eval_words, device, num_samples=500)
print(f"Gated Ensemble Win Rate: {win_rate:.4f}")