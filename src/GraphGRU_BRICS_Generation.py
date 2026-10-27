# %%
import torch
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
import torch.nn.functional as F
import json
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading training settings
train_smiles_set = set()

with open('train_smiles_canonical.pkl', 'rb') as f:
    train_smiles_set = pickle.load(f)

with open('GRU_qm9_model/model_config.json', 'r') as f:
    config = json.load(f)

# 2. Rileggi i dizionari
with open('GRU_qm9_model/vocabulary_qm9_t2i.pkl', 'rb') as f:
    token_to_idx = pickle.load(f)

with open('GRU_qm9_model/vocabulary_qm9_i2t.pkl', 'rb') as f:
    idx_to_token = pickle.load(f)

with open('GRU_qm9_model/rare_fragments.pkl', 'rb') as f:
    rare_fragments = pickle.load(f)

# GRU definition
model = GraphGRU(
    dim_vocab=config['vocab_size'],
    dim_embedding=config['dim_embedding'],
    dim_hidden=config['dim_hidden'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
)

max_len_seq = config['max_len']

model = model.to(device)
state_dict = torch.load('GRU_qm9_model/GRU_weights_qm9_naive.pth')
model.load_state_dict(state_dict)

# Molecules generations
def generate_molecule(model, max_len, device, temperature=1):
    model.eval()

    # Sequence start and end indices
    start_idx = token_to_idx.get('<BOS>', 1)
    end_idx = token_to_idx.get('<EOS>', 2)

    generated = [start_idx]

    current_token = torch.tensor([[start_idx]], dtype=torch.long).to(device)
    hidden = None
    current_atom_count = 0

    with torch.no_grad():

        # Generation of molecule fragments
        for _ in range(max_len):

            # Pass the current token into the Embedding -> GRU -> Linear pipeline
            out, hidden = model.rnn(model.embedding(current_token), hidden)
            logits = model.linear(out[:, -1, :])

            # Apply the temperature factor to scale the distribution
            logits = logits / temperature
            probs = F.softmax(logits, dim=1)

            # Stochastically sample the new fragment from the multinomial distribution
            fragment_id = torch.multinomial(probs, num_samples=1)
            next_id = fragment_id.item()

            # If it is EOS token the model has chosen for the end of the sequence
            if next_id == end_idx:
                generated.append(next_id)
                break

            # Estimate how many atoms this fragment would add
            if next_id not in [0, 1, 2, 3]:

                frag_smiles = idx_to_token[next_id]
                # Replace the '*' attachment points with hydrogens [H] to make RDKit interpret the fragment
                frag_mol = Chem.MolFromSmiles(frag_smiles.replace('*', '[H]')) 
                # Compute the heavy atoms added by the fragment
                n_atoms_frag = frag_mol.GetNumAtoms() - 1 if frag_mol else 0  

                # If adding fragment exceeds maximum capacity, force exit
                if current_atom_count + n_atoms_frag > max_len:
                    generated.append(end_idx) 
                    break

                current_atom_count += n_atoms_frag

            generated.append(next_id)
            current_token = torch.tensor([[next_id]], dtype=torch.long).to(device)
    
    return generated

# %%
import random
from rdkit import Chem

# Reconstruction of smiles from fragments
def reconstruct_smiles(fragments):
    if not fragments:
        return None

    first = fragments[0]
    if '*' not in first:
        first = '*' + first

    # Converts the string to an RDKit molecular object
    current_mol = Chem.MolFromSmiles(first)
    if current_mol is None:
        return None

    # It takes the molecule accumulated so far and the new fragment and combines them into a single object 
    # containing two detached fragments in the same space.
    for frag_smiles in fragments[1:]:
        next_mol = Chem.MolFromSmiles(frag_smiles)
        if next_mol is None:
            continue
        
        n_atoms_current = current_mol.GetNumAtoms()
        combined = Chem.CombineMols(current_mol, next_mol)

        # Locate the attachment point on the first and second fragment
        dummy_indices = [a.GetIdx() for a in combined.GetAtoms() if a.GetAtomicNum() == 0]

        d1_idx = next((i for i in dummy_indices if i < n_atoms_current), None)
        d2_idx = next((i for i in dummy_indices if i >= n_atoms_current), None)
        
        if d1_idx is not None and d2_idx is not None:
            neighbor1 = combined.GetAtomWithIdx(d1_idx).GetNeighbors()[0].GetIdx()
            neighbor2 = combined.GetAtomWithIdx(d2_idx).GetNeighbors()[0].GetIdx()

            # Retrieve the original binding type of the dummy
            bond1 = combined.GetBondBetweenAtoms(d1_idx, neighbor1)
            bond2 = combined.GetBondBetweenAtoms(d2_idx, neighbor2)
            bond_type1 = bond1.GetBondType() if bond1 else Chem.BondType.SINGLE
            bond_type2 = bond2.GetBondType() if bond2 else Chem.BondType.SINGLE

            # If the two dummies have different types, discard them to be safe
            if bond_type1 != bond_type2:
                current_mol = None
                break

            rw_mol = Chem.RWMol(combined)
            rw_mol.AddBond(neighbor1, neighbor2, bond_type1)

            # Remove dummy atoms starting from the highest index
            for idx in sorted([d1_idx, d2_idx], reverse=True):
                rw_mol.RemoveAtom(idx)
                
            current_mol = rw_mol.GetMol()
        else:
            continue

    if current_mol is None:
        return None

    # If there are any unused attachment points left at the ends of the accumulated molecule, 
    # it removes them altogether to leave only real atoms.
    rw_mol = Chem.RWMol(current_mol)
    dummy_residui = [a.GetIdx() for a in rw_mol.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(dummy_residui, reverse=True):
        rw_mol.RemoveAtom(idx)
    current_mol = rw_mol.GetMol()

    # Check the correctness of the valences and aromaticity
    try:
        Chem.SanitizeMol(current_mol)
        return Chem.MolToSmiles(current_mol, canonical=True)
    except:
        return None



# %%
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

# %%
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import re

num_samples = 1000                  # Number of molecules generated
seeds = [0, 1, 2, 3, 4]             # Seeds
temperatures = [0.5, 1.0, 1.5, 2.0]      # Temperature tested
results = {}
generated_smiles_per_temp = {}

# Generation of molecules for different temperatures and seeds
for temp in temperatures:
    results[temp] = {"validity": [], "uniqueness": [], "novelty": []}
    generated_smiles_per_temp[temp] = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        generated = []
        for i in range (num_samples):
            generated_mol = generate_molecule(model, max_len_seq, device, temp)
            generated.append(generated_mol)

        fragments_list = []

        # Modifies molecules with token <UKN> with a random rare fragment
        for sequence in generated:
            smiles = []
            for id in sequence:
                if id not in [0, 1, 2]:
                    if id == 3:
                        real_fragment = random.choice(rare_fragments)
                    else:
                        real_fragment = idx_to_token[id]        
                    smiles.append(real_fragment)
            
            fragments_list.append(smiles)

        # Reconstruction of the molecular smiles
        smiles_mol_generated = []
        for fragments in fragments_list:
            smiles_mol = reconstruct_smiles(fragments)
            if smiles_mol is not None:
                if not "." in smiles_mol:
                    smiles_mol_generated.append(smiles_mol)


        val, uniq, novel = check_uniqueness_and_novelty(smiles_mol_generated, train_smiles_set, len(fragments_list))

        results[temp]["validity"].append(val)
        results[temp]["uniqueness"].append(uniq)
        results[temp]["novelty"].append(novel)

        generated_smiles_per_temp[temp].extend(smiles_mol_generated)

    # -- DRAW OF EXAMPLES OF GENERATED MOLECULES --
    draw_mols = smiles_mol_generated[0:10]

    mols = []
    legende = []

    for i, smiles in enumerate(draw_mols):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
            legende.append(f"Mol {i+1}")

    dopts = Draw.rdMolDraw2D.MolDrawOptions()

    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300), legends=legende, useSVG=True, drawOptions=dopts)

    svg_data = img.data

    svg_data = re.sub(
        r'(<svg[^>]*>)',
        r'\1<rect width="100%" height="100%" fill="white"/>',
        svg_data
    )

    filename = f"GRU_qm9_model/generated_mol_GRU_t={temp}.svg"

    with open(filename, 'w') as f:
        f.write(svg_data)

    # ----------------------------------

for temp in temperatures:
    v = results[temp]["validity"]
    u = results[temp]["uniqueness"]
    n = results[temp]["novelty"]
    print(f"T={temp}:")
    print(f"  Validity   = {np.mean(v):.2f} ± {np.std(v):.2f}")
    print(f"  Uniqueness = {np.mean(u):.2f} ± {np.std(u):.2f}")
    print(f"  Novelty    = {np.mean(n):.2f} ± {np.std(n):.2f}")

# %%
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


