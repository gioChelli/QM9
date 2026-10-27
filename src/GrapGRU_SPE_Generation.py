# %%
import torch.nn as nn
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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
import torch
import json
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading training settings
train_smiles_set = set()

with open('train_smiles_canonical.pkl', 'rb') as f:
    train_smiles_set = pickle.load(f)

with open('GRU_qm9_model_BPE/model_config.json', 'r') as f:
    config = json.load(f)

with open('GRU_qm9_model_BPE/vocabulary_qm9_t2i.pkl', 'rb') as f:
    token_to_idx = pickle.load(f)

with open('GRU_qm9_model_BPE/vocabulary_qm9_i2t.pkl', 'rb') as f:
    idx_to_token = pickle.load(f)

# Model definition
model = GraphGRU(
    dim_vocab=config['vocab_size'],
    dim_embedding=config['dim_embedding'],
    dim_hidden=config['dim_hidden'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
)

max_len_seq = config['max_len']

model = model.to(device)
state_dict = torch.load('GRU_qm9_model_BPE/GRU_weights_qm9.pth')
model.load_state_dict(state_dict)

import torch.nn.functional as F

def generate_molecule(model, max_len, device, temperature=1.0):
    model.eval()

    start_idx = token_to_idx.get('<BOS>', 1)
    end_idx = token_to_idx.get('<EOS>', 2)

    generated = []
    current_token = torch.tensor([[start_idx]], dtype=torch.long).to(device)
    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            # Logits sampling
            out, hidden = model.rnn(model.embedding(current_token), hidden)
            logits = model.linear(out[:, -1, :]) / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Multinomial Sampling
            next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == end_idx:
                break

            # Add the generated token if it is not PAD/BOS/UNO
            if next_id not in [token_to_idx.get('<PAD>', 0), start_idx, token_to_idx.get('<UNK>', 3)]:
                generated.append(next_id)

            current_token = torch.tensor([[next_id]], dtype=torch.long).to(device)
            
    return generated

from rdkit import Chem

def reconstruct_smiles(token_ids):
    if not token_ids:
        return None

    # Reconstructs the smile by joining the token pieces
    raw_smiles = "".join([idx_to_token[idx] for idx in token_ids])
    
    # Conversion and validation with RDKit
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        return None 

    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

# Function to calculate validity, novelty and uniqueness values
def check_uniqueness_and_novelty(generated, train_smiles_set, num_samples):
    
    if not generated:
        print("Nessuna molecola generata.")
        return
    
    # Validity
    valid_count = len(generated)
    validity_percentage = (valid_count / num_samples) * 100 

    if valid_count == 0:
        print("Nessuna molecola valida.")
        return

    # Uniqueness
    unique_smiles = set(generated)
    uniqueness_percentage = (len(unique_smiles) / valid_count) * 100

    # Novelty
    novel_count = sum(1 for s in unique_smiles if s not in train_smiles_set)
    novel_percentage = (novel_count / len(unique_smiles)) * 100

    return validity_percentage, uniqueness_percentage, novel_percentage

import random
import re
import numpy as np
import torch
from rdkit.Chem import AllChem, Draw

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

num_samples = 1000                  # Number of molecules generated
seeds = [0, 1, 2, 3, 4]             # Seeds
temperatures = [0.5, 1.0, 1.5, 2.0]      # Temperature tested
results = {}
generated_smiles_per_temp = {}

# Special indexes to filter before passing the sequence for smile reconstruction
pad_idx = token_to_idx.get('<PAD>', 0)
bos_idx = token_to_idx.get('<BOS>', 1)
eos_idx = token_to_idx.get('<EOS>', 2)
unk_idx = token_to_idx.get('<UNK>', 3)
special_indices = {pad_idx, bos_idx, eos_idx, unk_idx}

# Generation of molecules for different temperatures and seeds
for temp in temperatures:
    results[temp] = {"validity": [], "uniqueness": [], "novelty": []}
    generated_smiles_per_temp[temp] = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        generated_sequences = []
        for i in range(num_samples):
            seq_ids = generate_molecule(model, max_len_seq, device, temp)
            generated_sequences.append(seq_ids)

        # Reconstruction of SMILES from IDs
        smiles_mol_generated = []
        for seq_ids in generated_sequences:
            # Filter out any special tokens before passing IDs
            clean_ids = [idx for idx in seq_ids if idx not in special_indices]
            smiles_mol = reconstruct_smiles(clean_ids)
            
            if smiles_mol is not None and "." not in smiles_mol:
                mol_check = Chem.MolFromSmiles(smiles_mol)
                if mol_check is not None and mol_check.GetNumAtoms() <= 9:
                    smiles_mol_generated.append(smiles_mol)

        val, uniq, novel = check_uniqueness_and_novelty(
            smiles_mol_generated, train_smiles_set, len(generated_sequences)
        )

        results[temp]["validity"].append(val)
        results[temp]["uniqueness"].append(uniq)
        results[temp]["novelty"].append(novel)

        generated_smiles_per_temp[temp].extend(smiles_mol_generated)

    # -- DRAW OF EXAMPLES OF GENERATED MOLECULES --
    draw_mols = smiles_mol_generated[:10]

    mols = []
    legende = []

    for i, smiles in enumerate(draw_mols):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
            legende.append(f"Mol {i+1}")

    if mols:
        dopts = Draw.rdMolDraw2D.MolDrawOptions()

        img = Draw.MolsToGridImage(
            mols, molsPerRow=5, subImgSize=(300, 300), legends=legende, useSVG=True, drawOptions=dopts
        )

        svg_data = img.data

        svg_data = re.sub(
            r'(<svg[^>]*>)',
            r'\1<rect width="100%" height="100%" fill="white"/>',
            svg_data
        )

        filename = f"GRU_qm9_model_BPE/generated_mol_GRU_t={temp}.svg"

        with open(filename, 'w') as f:
            f.write(svg_data)

    # -----------------------------------

for temp in temperatures:
    v = results[temp]["validity"]
    u = results[temp]["uniqueness"]
    n = results[temp]["novelty"]
    print(f"\nT = {temp}:")
    print(f"  Validity   = {np.mean(v):.2f}% ± {np.std(v):.2f}")
    print(f"  Uniqueness = {np.mean(u):.2f}% ± {np.std(u):.2f}")
    print(f"  Novelty    = {np.mean(n):.2f}% ± {np.std(n):.2f}")

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors
from scipy.stats import wasserstein_distance
import random

def extract_molecular_properties(smiles_list):

    properties = {
        'MW': [],
        'LogP': [],
        'TPSA': [],
        'QED': [],
        'NumRings': []
    }
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        properties['MW'].append(Descriptors.MolWt(mol))
        properties['LogP'].append(Crippen.MolLogP(mol))
        properties['TPSA'].append(Descriptors.TPSA(mol))
        properties['QED'].append(QED.qed(mol))
        properties['NumRings'].append(rdMolDescriptors.CalcNumRings(mol))
        
    return {k: np.array(v) for k, v in properties.items()}

# Calculate properties and compare via Wasserstein Distance (Earth Mover's Distance).
def evaluate_distribution_similarity(real_smiles, gen_smiles):

    # Compute properties of QM9 molecules
    real_props = extract_molecular_properties(real_smiles)
    # Compute properties of generated molecules
    gen_props = extract_molecular_properties(gen_smiles)

    # Compute Wasserstein Distance between real and generated molecules
    distances = {}
    print("\n WASSERSTEIN DISTANCE")
    for prop in real_props.keys():
        w_dist = wasserstein_distance(real_props[prop], gen_props[prop])
        distances[prop] = w_dist
        print(f"Dist. {prop:10s}: {w_dist:.4f}")
        
    # -- PLOT OF WASSERSTEIN DISTANCE --
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=200)
    axes = axes.flatten()
    
    for i, prop in enumerate(real_props.keys()):
        ax = axes[i]
        
        ax.hist(real_props[prop], bins=30, alpha=0.5, density=True, label='QM9 Train', color='blue')
        ax.hist(gen_props[prop], bins=30, alpha=0.5, density=True, label='GraphVAE Generated', color='orange')
        
        ax.set_title(f"Distribuzione {prop} (W-Dist: {distances[prop]:.3f})")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig('GraphVAE_model/property_distributions.png', bbox_inches='tight')
    plt.show()

    # --------------------------------

    return distances

gen_smiles_t1 = generated_smiles_per_temp[1.5] 
real_sample = random.sample(list(train_smiles_set), 1000)
w_distances = evaluate_distribution_similarity(real_sample, gen_smiles_t1)


