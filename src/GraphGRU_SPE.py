import pickle
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the dataset
train_smiles_set = set()

with open('train_smiles_canonical.pkl', 'rb') as f:
    train_smiles_set = pickle.load(f)

# %%
from SmilesPE.learner import learn_SPE
import codecs

# Save the train SMILES to a text file (one per line)
with open('qm9_train_smiles.txt', 'w') as f:
    for smi in train_smiles_set:
        f.write(f"{smi}\n")

# Train the SPE algorithm by deciding the maximum number of tokens
infile = codecs.open('qm9_train_smiles.txt', 'r', 'utf-8')
outfile = codecs.open('qm9_spe_vocab_100.txt', 'w', 'utf-8')

learn_SPE(
    infile=infile, 
    outfile=outfile, 
    num_symbols=100,      
    min_frequency=2, 
    verbose=False
)

infile.close()
outfile.close()

from SmilesPE.tokenizer import SPE_Tokenizer
import codecs

spe_vocab = codecs.open('qm9_spe_vocab_100.txt', 'r', 'utf-8')
tokenizer = SPE_Tokenizer(spe_vocab)

# Tokenize all sequences in the dataset
tokenized_dataset = []
for smi in train_smiles_set:
    # tokenizer.tokenize(smi): takes a SMILES string and converts it into a sequence where the identified fragments are separated by a space.
    # split(): Splits the resulting string at spaces, turning it into a list of distinct tokens.
    tokens = tokenizer.tokenize(smi).split()
    # Salva la lista di token di ciascuna molecola
    tokenized_dataset.append(tokens)

# It passes through all the token sequences present and picks the unique tokens in the dataset
unique_tokens = sorted(list({token for seq in tokenized_dataset for token in seq}))

token_to_idx = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3
}

# It scrolls through the list of tokens extracted from SPE and assigns each one a progressive numerical index
for token in unique_tokens:
    if token not in token_to_idx:
        token_to_idx[token] = len(token_to_idx)

# Creates a reversed dictionary where the key is the numeric index and the value is the corresponding textual token.
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

# %%
# Add <BOS> at the beginning and <EOS> at the end
def encode_sequence(tokens, token_to_idx):
    seq = [token_to_idx["<BOS>"]] + [token_to_idx.get(t, token_to_idx["<UNK>"]) for t in tokens] + [token_to_idx["<EOS>"]]
    return seq

# Convert the entire dataset to numbers
encoded_sequences = [encode_sequence(seq, token_to_idx) for seq in tokenized_dataset]

# Find the maximum length and apply padding
max_len_seq = max(len(seq) for seq in encoded_sequences)
padded_dataset = []
for seq in encoded_sequences:
    pad_len = max_len_seq - len(seq)
    padded_seq = seq + [token_to_idx["<PAD>"]] * pad_len
    padded_dataset.append(padded_seq)

dataset_tensor = torch.tensor(padded_dataset, dtype=torch.long)

import torch.nn as nn

# GRU
class GraphGRU(nn.Module):
    def __init__(self, dim_vocab, dim_embedding=256, dim_hidden=512, num_layers=3, dropout=0.2):
        super().__init__()
 
        #  Embeddig Layer
        self.embedding = nn.Embedding(dim_vocab, dim_embedding, padding_idx=0)
        self.drop = nn.Dropout(dropout)

        # GRU Layer
        self.rnn = nn.GRU(
            input_size=dim_embedding, 
            hidden_size=dim_hidden, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )

        # Linear Layer
        self.linear = nn.Linear(dim_hidden, dim_vocab)
    
    def forward(self, x, h=None):
        # Compute embedding
        embeds = self.drop(self.embedding(x))
        # Execute GRU
        out, hidden = self.rnn(embeds, h)
        # Apply droput
        out = self.drop(out)
        # Compute logits
        logits = self.linear(out)

        return logits, hidden

import os
import json

# Saving settings for generation
os.makedirs('GRU_qm9_model_BPE', exist_ok=True)

with open('GRU_qm9_model_BPE/vocabulary_qm9_t2i.pkl', 'wb') as f:
    pickle.dump(token_to_idx, f)

with open('GRU_qm9_model_BPE/vocabulary_qm9_i2t.pkl', 'wb') as f:
    pickle.dump(idx_to_token, f)

config = {
    'vocab_size': len(token_to_idx),
    'dim_embedding': 256,
    'dim_hidden' :512,
    'num_layers': 3,
    'dropout' : 0.1,
    'max_len' : max_len_seq
}

with open('GRU_qm9_model_BPE/model_config.json', 'w') as f:
    json.dump(config, f)

import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader

# Model definition
model = GraphGRU(
    dim_vocab=len(token_to_idx),
    dim_embedding=256,
    dim_hidden=512,
    num_layers=3,
    dropout=0.2
)
model = model.to(device)

# Loss
criterion = nn.CrossEntropyLoss(ignore_index=token_to_idx["<PAD>"])

# Dataset split in train and validation sets
train_size = int(0.8 * len(dataset_tensor))
val_size = len(dataset_tensor) - train_size
train_dataset, val_dataset = random_split(dataset_tensor, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

#Early stopping 
best_val_loss = float('inf')
patience = 5
patience_counter = 0

val_loss_history = []
train_loss_history = []

epochs = 50
for epoch in range(epochs):

    # Training
    model.train()
    total_loss = 0
    for batch_idx, batch_data in enumerate(train_loader):
        
        if isinstance(batch_data, (list, tuple)):
            sequence, _ = batch_data
        else:
            sequence = batch_data

        batch_x = sequence[:, :-1].to(device)
        batch_y = sequence[:, 1:].to(device)

        optimizer.zero_grad()

        # GRU output
        out, _ = model(batch_x)

        # Compute loss
        loss = criterion(out.view(-1, out.size(-1)), batch_y.view(-1))

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(validation_loader):
            
            if isinstance(batch_data, (list, tuple)):
                sequence, _ = batch_data
            else:
                sequence = batch_data

            batch_x = sequence[:, :-1].to(device)
            batch_y = sequence[:, 1:].to(device)
            
            # GRU output
            out, _ = model(batch_x)

            # Compute loss
            loss_val = criterion(out.view(-1, out.size(-1)), batch_y.view(-1))
            val_loss += loss_val.item()

    avg_loss = total_loss / len(train_loader)
    avg_loss_val = val_loss / len(validation_loader)
    scheduler.step(avg_loss_val)

    # Early stopping and saving optimal weights
    if avg_loss_val < best_val_loss:
        best_val_loss = avg_loss_val
        torch.save(model.state_dict(), 'GRU_qm9_model_BPE/GRU_weights_qm9.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping attivato! Il modello stava iniziando a overfittare.")
            break
    
    train_loss_history.append(avg_loss)
    val_loss_history.append(avg_loss_val)
    print(f"Epoca [{epoch+1}/{epochs}] - Loss Train: {avg_loss:.4f} - Loss validation: {avg_loss_val:.4f}")

with open('GRU_qm9_model_BPE/train_loss_history.pkl', 'wb') as f:
    pickle.dump(train_loss_history, f)

with open('GRU_qm9_model_BPE/validation_loss_history.pkl', 'wb') as f:
    pickle.dump(val_loss_history, f)

import matplotlib.pyplot as plt
import pickle 

with open('GRU_qm9_model_BPE/train_loss_history.pkl', 'rb') as f:
    train_loss = pickle.load(f)

with open('GRU_qm9_model_BPE/validation_loss_history.pkl', 'rb') as f:
    val_loss = pickle.load(f)

epochs = range(1, len(train_loss) + 1)

# Loss curve
plt.figure(figsize=(8, 5), dpi=300) 
plt.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--')

plt.title('Learning Curve - GraphGRU', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='upper right')

plt.xlim(1, len(train_loss))

plt.tight_layout()

plt.savefig('GRU_qm9_model_BPE/learning_curve_qm9.png', bbox_inches='tight', dpi=300)
plt.show()



