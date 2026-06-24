import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import string
import matplotlib.pyplot as plt
from collections import defaultdict
import math

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('words_250000_train.txt', 'r') as f: 
    words = f.read().splitlines()
words = [w.lower() for w in words if w.isalpha() and len(w) > 2]

# Split into completely disjoint Train and Eval sets
random.seed(42)
random.shuffle(words)
split_idx = int(len(words) * 0.8)
train_words = words[:split_idx]
eval_words = words[split_idx:]

print(f"Train words: {len(train_words)}, Eval words: {len(eval_words)}")

# Vocabulary mappings
chars = string.ascii_lowercase
char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
char_to_idx['_'] = 27  # Mask token
char_to_idx['<PAD>'] = 0  # Padding token
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx)

# ------------------------------------------------------------
# 1. Create expert-specific subsets from the training words
# ------------------------------------------------------------

# Length‑based experts
short_words   = [w for w in train_words if 3 <= len(w) <= 5]
medium_words  = [w for w in train_words if 6 <= len(w) <= 9]
long_words    = [w for w in train_words if len(w) >= 10]

# Common / rare experts based on bigram frequencies
# Compute bigram counts (with smoothing) from the entire training set
bigram_counts = defaultdict(int)
for w in train_words:
    for i in range(len(w)-1):
        bigram = w[i:i+2]
        bigram_counts[bigram] += 1
total_bigrams = sum(bigram_counts.values())
V = 26  # number of letters
# For each word, compute its average log bigram probability (using add‑one smoothing)
def word_commonality_score(word):
    log_prob_sum = 0.0
    n = len(word)-1
    if n == 0:
        return 0.0  # single‑letter words don't exist here (len>2)
    for i in range(n):
        bigram = word[i:i+2]
        # add‑one smoothing: (count+1) / (total_bigrams + V*V)
        prob = (bigram_counts[bigram] + 1) / (total_bigrams + V*V)
        log_prob_sum += math.log(prob)
    return log_prob_sum / n

word_scores = [(w, word_commonality_score(w)) for w in train_words]
word_scores.sort(key=lambda x: x[1])  # ascending
# Split at median
median_idx = len(word_scores) // 2
rare_words = [w for w, _ in word_scores[:median_idx]]
common_words = [w for w, _ in word_scores[median_idx:]]

print(f"Expert sizes: Short={len(short_words)}, Medium={len(medium_words)}, Long={len(long_words)}, "
      f"Common={len(common_words)}, Rare={len(rare_words)}")

# ------------------------------------------------------------
# 2. Dataset and model definitions (unchanged from original)
# ------------------------------------------------------------
class HangmanDataset(Dataset):
    def __init__(self, words, mask_range=(0.1, 0.25)):
        self.words = words
        self.mask_range = mask_range

    def set_mask_range(self, new_range):
        self.mask_range = new_range

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        word_len = len(word)
        
        min_mask, max_mask = self.mask_range
        mask_ratio = random.uniform(min_mask, max_mask)
        num_mask = max(1, int(word_len * mask_ratio))
        num_mask = min(num_mask, word_len - 1)
        
        mask_indices = random.sample(range(word_len), num_mask)
        
        input_seq = []
        target_seq = []
        mask_locations = []
        
        for i, char in enumerate(word):
            target_seq.append(char_to_idx[char] - 1)  # 0-25
            if i in mask_indices:
                input_seq.append(char_to_idx['_'])
                mask_locations.append(1)
            else:
                input_seq.append(char_to_idx[char])
                mask_locations.append(0)
                
        return torch.tensor(input_seq), torch.tensor(target_seq), torch.tensor(mask_locations)

def collate_fn(batch):
    inputs, targets, masks = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=char_to_idx['<PAD>'])
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-100)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded, masks_padded

class HangmanBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(HangmanBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=char_to_idx['<PAD>'])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, 26)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

# ------------------------------------------------------------
# 3. Ensemble prediction function
# ------------------------------------------------------------
def ensemble_predict(models, input_tensor, guessed_letters, device):
    """
    Given a list of models, average their softmax probabilities over masked positions.
    Returns the index of the best letter (0-25).
    """
    probs_list = []
    mask = (input_tensor == char_to_idx['_']).float()  # shape (1, seq_len)
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)  # (1, seq_len, 26)
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (1, seq_len, 26)
            # Average over masked positions
            masked_probs = (probs * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (1, 26)
            probs_list.append(masked_probs.squeeze(0))  # (26)
    # Average probabilities across models
    avg_probs = torch.stack(probs_list).mean(dim=0)  # (26)
    # Mask already guessed letters
    for char in guessed_letters:
        avg_probs[char_to_idx[char] - 1] = -float('inf')
    return torch.argmax(avg_probs).item()

def play_hangman_ensemble(models, word, device):
    """
    Play one game using the ensemble of models.
    """
    guessed_letters = set()
    incorrect_guesses = 0
    current_word = ['_'] * len(word)
    
    while incorrect_guesses < 6 and '_' in current_word:
        input_seq = [char_to_idx[c] for c in current_word]
        input_tensor = torch.tensor([input_seq]).to(device)
        
        guess_idx = ensemble_predict(models, input_tensor, guessed_letters, device)
        guess_char = idx_to_char[guess_idx + 1]
        guessed_letters.add(guess_char)
        
        if guess_char in word:
            for i, c in enumerate(word):
                if c == guess_char:
                    current_word[i] = guess_char
        else:
            incorrect_guesses += 1
            
    return '_' not in current_word  # True if won

def evaluate_ensemble(models, eval_words, device, num_samples=500):
    samples = random.sample(eval_words, min(num_samples, len(eval_words)))
    wins = sum([play_hangman_ensemble(models, w, device) for w in samples])
    return wins / len(samples)

# ------------------------------------------------------------
# 4. Training hyperparameters (same as original)
# ------------------------------------------------------------
EMBEDDING_DIM = 128
HIDDEN_DIM = 512
BATCH_SIZE = 512
EPOCHS = 60

# Curriculum phases
curriculum = {
    0: (0.10, 0.25),
    10: (0.10, 0.50),
    20: (0.25, 0.75),
    30: (0.50, 0.75),
    40: (0.70, 0.90),
    50: (0.10, 0.90)
}

# ------------------------------------------------------------
# 5. Train each expert
# ------------------------------------------------------------
experts = [
    {"name": "short",   "words": short_words},
    {"name": "medium",  "words": medium_words},
    {"name": "long",    "words": long_words},
    {"name": "common",  "words": common_words},
    {"name": "rare",    "words": rare_words},
]

# Dictionary to store trained models
trained_models = {}

for expert in experts:
    name = expert["name"]
    word_list = expert["words"]
    if len(word_list) == 0:
        print(f"Warning: No words for expert '{name}'. Skipping.")
        continue
    
    print(f"\n===== Training Expert: {name} (size: {len(word_list)}) =====")
    model = HangmanBiLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    dataset = HangmanDataset(word_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    train_losses = []
    eval_accuracies = []  # optional, we could evaluate each expert individually but not necessary
    
    for epoch in range(EPOCHS):
        if epoch in curriculum:
            dataset.set_mask_range(curriculum[epoch])
            print(f"   Curriculum phase: {curriculum[epoch]}")
            
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets, masks) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            logits = logits.view(-1, 26)
            targets = targets.view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        # Optional: evaluate this expert alone (not required for ensemble)
        # eval_acc = evaluate_model(model, eval_words, device)  # we'd need a single-model eval function
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
    
    # Save expert model
    expert_path = f"expert_{name}_bilstm.pt"
    torch.save(model.state_dict(), expert_path)
    print(f"Saved {name} expert to {expert_path}")
    trained_models[name] = model

# ------------------------------------------------------------
# 6. Evaluate the full ensemble on the eval set
# ------------------------------------------------------------
# Collect all trained models (skip any that might be missing)
ensemble_models = [model for model in trained_models.values() if model is not None]
if len(ensemble_models) == 0:
    print("No models trained!")
else:
    print(f"\n===== Evaluating Ensemble with {len(ensemble_models)} experts =====")
    ensemble_win_rate = evaluate_ensemble(ensemble_models, eval_words, device, num_samples=500)
    print(f"Ensemble Win Rate on 500 random eval words: {ensemble_win_rate:.4f}")

# (Optional) Plotting training curves for each expert could be added, but omitted for brevity.
