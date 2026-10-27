# %%
import pickle
import torch
from torch_geometric.datasets import QM9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Loading the dataset
dataset = QM9(root='./data/QM9')
train_smiles_set = set()

with open('train_smiles_canonical.pkl', 'rb') as f:
    train_smiles_set = pickle.load(f)

# %%
from rdkit import Chem
import re
from rdkit import RDLogger
from rdkit.Chem import BRICS

RDLogger.DisableLog('rdApp.*')

# Smiles fragmentation following BRICS Rules
def smiles_fragmentation(start_smiles):
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(start_smiles))
    if not mol:
        return None

    # Find breakable BRICS bonds
    brics_bonds = list(BRICS.FindBRICSBonds(mol))

    # If there are no breakable bonds, the whole molecule was returned.
    if brics_bonds:
        bond_indices = [mol.GetBondBetweenAtoms(a1, a2).GetIdx() for ((a1, a2), _) in brics_bonds]
    else:
        return [Chem.MolToSmiles(mol)]
    
    # Fragment on all BRICS ties identified
    frag_mol = Chem.FragmentOnBonds(mol, bond_indices, addDummies=True)
    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)

    # Adds fragments to the dictionary
    fragments = []
    for frag in frags:
        smi = Chem.MolToSmiles(frag)
        clean = re.sub(r'\[\d*\*\]', '*', smi)
        fragments.append(clean)

    return fragments

# %%
from collections import Counter

all_sequences = []

# Dictionaries for Text-to-Number Conversion and Conversely
token_to_idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "*<UNK>*": 3}
idx_to_token = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "*<UNK>*"}

# Inserts the fragments obtained for each molecule into the dictionary
for smiles in train_smiles_set:
    smiles_fragments = smiles_fragmentation(smiles)
    all_sequences.append(smiles_fragments)

all_fragments = []
for seq in all_sequences:
    for frag in seq:
        all_fragments.append(frag)

# Count how many times each fragment appears
counter = Counter(all_fragments)
MIN_FREQ = 2
rare_fragments = []

# Fragment sorting, assign an ID to each fragment except rare ones which are grouped in a dictionary
for fragment, freq in counter.items():
    if freq >= MIN_FREQ:
        current_idx = len(token_to_idx)
        token_to_idx[fragment] = current_idx
        idx_to_token[current_idx] = fragment
    else:
        rare_fragments.append(fragment)

# For each fragment that appears less than MIN_FREQ times, replace it with the <UKN> token.
dataset_id_sequences = []
for seq in all_sequences:
    masked_seq = [frag if frag in token_to_idx else "*<UNK>*" for frag in seq]
    dataset_id_sequences.append(masked_seq)


# %%
# Maximum length of a fragment sequence
max_len_seq = max(len(seq) for seq in dataset_id_sequences) + 2

dataset_transformer = []

# Final creation of the dataset by inserting the start and end molecule tokens <BOS>, <EOS> and padding
for seq in dataset_id_sequences:
    
    id_sequence = [token_to_idx[frag] for frag in seq]
    numerical_seq = [1] + id_sequence + [2]
    padded_seq = numerical_seq + [0] * (max_len_seq - len(numerical_seq))
    tensor_seq = torch.tensor(padded_seq, dtype=torch.long)

    x = tensor_seq[:-1]
    y = tensor_seq[1:]

    dataset_transformer.append([x,y])

# %%
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


# %%
import os
import json

# Saving settings for generation
os.makedirs('GRU_qm9_model', exist_ok=True)

with open('GRU_qm9_model/vocabulary_qm9_t2i.pkl', 'wb') as f:
    pickle.dump(token_to_idx, f)

with open('GRU_qm9_model/vocabulary_qm9_i2t.pkl', 'wb') as f:
    pickle.dump(idx_to_token, f)

with open('GRU_qm9_model/rare_fragments.pkl', 'wb') as f:
    pickle.dump(rare_fragments, f)

config = {
    'vocab_size': len(token_to_idx),
    'dim_embedding': 256,
    'dim_hidden' :512,
    'num_layers': 3,
    'dropout' : 0.2,
    'max_len' : max_len_seq
}

with open('GRU_qm9_model/model_config.json', 'w') as f:
    json.dump(config, f)

# %%
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
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Dataset split in train and validation sets
train_size = int(0.8 * len(dataset_transformer))
val_size = len(dataset_transformer) - train_size
train_dataset, val_dataset = random_split(dataset_transformer, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

#Early stopping 
best_val_loss = float('inf')
patience = 7
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
        torch.save(model.state_dict(), 'GRU_qm9_model/GRU_weights_qm9.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping attivato! Il modello stava iniziando a overfittare.")
            break
    
    train_loss_history.append(avg_loss)
    val_loss_history.append(avg_loss_val)
    print(f"Epoca [{epoch+1}/{epochs}] - Loss Train: {avg_loss:.4f} - Loss validation: {avg_loss_val:.4f}")

with open('GRU_qm9_model/train_loss_history.pkl', 'wb') as f:
    pickle.dump(train_loss_history, f)

with open('GRU_qm9_model/validation_loss_history.pkl', 'wb') as f:
    pickle.dump(val_loss_history, f)

# %%
import matplotlib.pyplot as plt
import pickle

with open('GRU_qm9_model/train_loss_history.pkl', 'rb') as f:
    train_loss = pickle.load(f)

with open('GRU_qm9_model/validation_loss_history.pkl', 'rb') as f:
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

plt.savefig('GRU_qm9_model/learning_curve_qm9.png', bbox_inches='tight', dpi=300)
plt.show()



