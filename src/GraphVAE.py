# %%
import torch
from torch_geometric.datasets import QM9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the dataset
dataset = QM9(root='./data/QM9')

# %%
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# Vocabulary of atoms and arcs
# C: Carbon
# N: Nitrogen
# O: Oxygen
# F: Fluorine
atoms = {'C': 1, 'N': 2, 'O': 3, 'F': 4}
edges = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}


# Construction of dense graph representation through edge adjacency matrix
def dense_rapresentation(smiles, max_atoms=9):
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        # Performs a first check of the valences and state of the bonds in non-strict mode
        mol.UpdatePropertyCache(strict=False)
        
        # Removal of explicit hydrogens (the molecular skeleton is the heavy atoms)
        mol = Chem.RemoveHs(mol, updateExplicitCount=False)
        
        # Recalculate real valences only on heavy atoms
        mol.UpdatePropertyCache(strict=False)
        
        # Apply standard sanitization: applies a set of controls on valence, charges, aromaticity, etc.
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    # Find the indices of heavy atoms in a molecule
    heavy_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H']

    if len(heavy_idx) > max_atoms:
        return None
    
    # If hydrogens were present, with their removal the indices could be spaced, so I assign them new contiguous positions
    old_to_new = {old_idx: new_pos for new_pos, old_idx in enumerate(heavy_idx)}

    # Create node representation as vectors of integers (If the molecule has less than 9 heavy atoms, the last cells will remain value 0)
    dense_nodes = torch.zeros(max_atoms, dtype=torch.long)  
    for new_pos, old_idx in enumerate(heavy_idx):
        atom = mol.GetAtomWithIdx(old_idx)
        dense_nodes[new_pos] = atoms[atom.GetSymbol()]
    
    # Create 9 x 9 adjacency matrix to represent the bonds
    dense_edges = torch.zeros((max_atoms, max_atoms), dtype=torch.long)
    for bond in mol.GetBonds():
        i_old, j_old = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Ignore bonds involving H residues
        if i_old not in old_to_new or j_old not in old_to_new:
            continue  
        i, j = old_to_new[i_old], old_to_new[j_old]

        bond_type = edges[bond.GetBondType()]
        dense_edges[i, j] = bond_type
        dense_edges[j, i] = bond_type

    return dense_nodes, dense_edges
            

# %%
import torch.nn.functional as F

# Creating the dataset with dense graph representation for training
dataset_dense = []
for data in dataset:

    result = dense_rapresentation(data.smiles, max_atoms=9)
    if result is None:
        continue
    dense_nodes, dense_edges = result
    data.dense_nodes = dense_nodes
    data.dense_edges = dense_edges

    # One-hot encoding representation for both nodes and bonds, to avoid bias
    row, col = torch.where(dense_edges > 0)
    data.edge_index = torch.stack([row, col], dim=0)

    bond_types = dense_edges[row, col] - 1
    data.edge_attr = F.one_hot(bond_types, num_classes=4).float()

    data.x = F.one_hot(dense_nodes, num_classes=5).float()
    dataset_dense.append(data)

dataset = dataset_dense

# %%
import torch.nn as nn
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool

# GraphVAE Encoder
class Encoder(nn.Module):
    def __init__(self, dim_latent, node_features=5, edge_features=4):
        super().__init__()

        # Networks used by NNConv layers to weight the connections between the nodes of the graph
        edge_nn1 = nn.Sequential(
            nn.Linear(edge_features, 32), nn.ReLU(),
            nn.Linear(32, node_features * 64)
        )
        edge_nn2 = nn.Sequential(
            nn.Linear(edge_features, 32), nn.ReLU(),
            nn.Linear(32, 64 * 128)
        )
        edge_nn3 = nn.Sequential(
            nn.Linear(edge_features, 64), nn.ReLU(),
            nn.Linear(64, 128 * 128)
        )

        # Message-passing layer with weighted connections
        self.conv1 = NNConv(node_features, 64,  edge_nn1)
        self.conv2 = NNConv(64,            128, edge_nn2)
        self.conv3 = NNConv(128,           128, edge_nn3)

        # Layer<Norm
        self.norm1 = nn.LayerNorm(64)   
        self.norm2 = nn.LayerNorm(128)
        self.norm3 = nn.LayerNorm(128)

        # Activation function
        self.activation = nn.ReLU()

        # Linear layers to compute mu and logvar
        self.mean   = nn.Linear(256, dim_latent)
        self.logvar = nn.Linear(256, dim_latent)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x.float()

        # Message-passing execution
        h1 = self.norm1(self.activation(self.conv1(x,  edge_index, edge_attr)))
        h2 = self.norm2(self.activation(self.conv2(h1, edge_index, edge_attr))) 
        h3 = self.norm3(self.activation(self.conv3(h2, edge_index, edge_attr))) + h2

        # Computing the embedding using the combination of max_pool and mean_pool
        embedding = torch.cat([
            global_mean_pool(h3, batch),  
            global_max_pool(h3, batch)    
        ], dim=-1)                       

        return self.mean(embedding), self.logvar(embedding)

# %%
# GraphVAE Decoder
class Decoder(nn.Module):
    def __init__(self, dim_latent, dim_embedding, max_nodes=9, node_features=5, edge_features=5, node_dim=64, heads=4):
        super().__init__()
        self.max_nodes = max_nodes

        self.node_dim = node_dim

        # Creating an embedding for each of the nodes
        self.z_to_nodes = nn.Sequential(
            nn.Linear(dim_latent, dim_embedding),
            nn.LayerNorm(dim_embedding),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim_embedding, max_nodes * self.node_dim)
        )

        # Multihead attention layer to compute the embedding by exchanging information between nodes
        self.attention = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=heads, batch_first=True
        )

        # Normalization Layer
        self.norm = nn.LayerNorm(node_dim)

        # Linear layer to compute atom from embedding
        self.node_head = nn.Linear(node_dim, node_features)

        # Layer to compute the type of connections using the embeddings of nodes i and j
        self.edge_head = nn.Sequential(
            nn.Linear(node_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, edge_features)
        )

    def forward(self, z):
        B = z.size(0)

        # Nodes embedding
        h = self.z_to_nodes(z).view(B, self.max_nodes, self.node_dim)  

        # Multihead attention
        h_att, _ = self.attention(h, h, h)

        # Residual connection and normalization
        h = self.norm(h + h_att)  

        # Compute nodes logits 
        nodes_logits = self.node_head(h) 

        # Predicting edges from embeddings of nodes i and j
        hi = h.unsqueeze(2).expand(-1, -1, self.max_nodes, -1) 
        hj = h.unsqueeze(1).expand(-1, self.max_nodes, -1, -1) 
        hij = torch.cat([hi, hj, hi * hj], dim=-1) 

        # Compute bonds logits
        edges_logits = self.edge_head(hij) 

        # Symmetric Logits
        edges_logits = (edges_logits + edges_logits.permute(0, 2, 1, 3)) / 2

        return nodes_logits, edges_logits

# %%
# Loss Function
def compute_loss(nodes_logits, edges_logits, nodes_target, edges_target):
    
    B, K, _ = nodes_logits.size() 
    device = nodes_logits.device

    # Nodes loss
    loss_nodes = F.cross_entropy(nodes_logits.view(-1, 5), nodes_target.view(-1), reduction='mean')

    # Bond loss  
    edges_logits_tr = edges_logits.permute(0, 3, 1, 2) 
    edge_weights = torch.tensor([0.5, 1.0, 1.2, 1.3, 1.2], device=device)
    loss_edges_raw = F.cross_entropy(edges_logits_tr, edges_target, reduction='none', weight=edge_weights)

    # Mask to exclude the diagonal from self-loops
    mask_diagonal = ~torch.eye(K, device=device).bool().unsqueeze(0)
    loss_edges_masked = loss_edges_raw * mask_diagonal.float()
    loss_edges = loss_edges_masked.sum() / (B * (K * K - K))

    # Valency loss
    edges_probs = F.softmax(edges_logits_tr, dim=1)

    # Bond type 
    bond_orders = torch.tensor([0.0, 1.0, 2.0, 3.0, 1.5], device=device).view(1, 5, 1, 1)
    expected_valency_matrix = (edges_probs * bond_orders).sum(dim=1)

    # Diagonal mask to eliminate self-loops
    expected_valency_matrix = expected_valency_matrix * mask_diagonal.float()
    pred_valency = expected_valency_matrix.sum(dim=-1) 

    # Definition of the maximum permitted valences
    max_valencies = torch.tensor([0.0, 4.0, 3.0, 2.0, 1.0], device=device)
    target_max_val = max_valencies[nodes_target] 

    # ReLU penalization
    is_real          = (nodes_target > 0).float()

    over_val = F.relu(pred_valency - target_max_val) * is_real
    loss_valency = over_val.mean()

    # Balanced loss
    return loss_nodes + 2.0 * loss_edges + 1 * loss_valency 

# %%
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def match_and_reorder(nodes_logits, edges_logits, nodes_target, edges_target, alpha=1.0, beta=1.0):
    
    B, K = nodes_logits.shape
    C_edges = edges_logits.shape[-1]
    device = nodes_logits.device

    # From One-Hot to Indices
    if nodes_target.dim() == 3:
        nodes_target_idx = nodes_target.argmax(dim=-1)
    else:
        nodes_target_idx = nodes_target

    if edges_target.dim() == 4:
        edges_target_idx = edges_target.argmax(dim=-1)
    else:
        edges_target_idx = edges_target

    with torch.no_grad():
        # Compute the Cross-Entropy between each predicted node i and each possible target node j
        node_log_probs = F.log_softmax(nodes_logits, dim=-1) 
        target_nodes_exp = nodes_target_idx.unsqueeze(1).expand(B, K, K) 
        cost_nodes = -node_log_probs.gather(2, target_nodes_exp) 
        
        edge_log_probs = F.log_softmax(edges_logits, dim=-1) 
        target_edges_oh = F.one_hot(edges_target_idx, num_classes=C_edges).float()
        
        # # Evaluate how closely the link topology of the predicted node i matches the topology of the target node j
        E = edge_log_probs.unsqueeze(2).unsqueeze(4) 
        T = target_edges_oh.unsqueeze(1).unsqueeze(3)  
        
        pair_ce = -(T * E).sum(dim=-1) 
        cost_edges = pair_ce.diagonal(dim1=3, dim2=4).mean(dim=-1)

        # Add the two costs weighted by the parameters alpha and beta
        total_cost_tensor = alpha * cost_nodes + beta * cost_edges
        total_cost_np = total_cost_tensor.detach().cpu().numpy()

        perms = torch.empty(B, K, dtype=torch.long, device=device)
        for b in range(B):
            row_ind, col_ind = linear_sum_assignment(total_cost_np[b])
            perm = torch.empty(K, dtype=torch.long, device=device)
            perm[col_ind] = torch.from_numpy(row_ind).to(device)
            perms[b] = perm

    # Reapply the perms found permutation directly to the original logits on GPU
    matched_nodes_logits = torch.stack([nodes_logits[b, perms[b]] for b in range(B)])
    matched_edges_logits = torch.stack([edges_logits[b][perms[b]][:, perms[b]] for b in range(B)])

    return matched_nodes_logits, matched_edges_logits

# %%
import os
import pickle
import json

# Saved settings to recover the model during generation
os.makedirs('GraphVAE_model', exist_ok=True)

config = {
    'dim_latent': 64,
    'dim_embedding': 256
}

with open('GraphVAE_model/model_config.json', 'w') as f:
    json.dump(config, f)

# %%
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

ENCODER_PATH = "encoder_weights_qm9.pth"
DECODER_PATH = "decoder_weights_qm9.pth"

# Beta annealing parameters
beta_max = 0.1
epochs_ann = 20

# Latent and embedding dimension
dim_latent = 64
dim_embedding = 256

encoder = Encoder(dim_latent, node_features=5).to(device)
decoder = Decoder(dim_latent, dim_embedding).to(device)

# Dataset split in train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

batch = 32
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False) 

# Optimizer and scheduler
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5, min_lr=1e-5
)

# Early stopping parameters
best_val_loss = float('inf')
patience = 10
patience_counter = 0

val_loss_history = []
train_loss_history = []

# Training cycle
epochs = 200
for epoch in range(0, epochs):

    # Train
    encoder.train()
    decoder.train()

    running_loss = 0.0

    # Beta annealing
    beta = min(beta_max, beta_max * (epoch / epochs_ann))
    
    for data in train_loader:
        data = data.to(device)
        B = data.num_graphs
        dense_nodes = data.dense_nodes.view(B, 9)
        dense_edges = data.dense_edges.view(B, 9, 9)

        optimizer.zero_grad()

        # Encoding
        mu, logvar = encoder(data.x, data.edge_index, data.edge_attr, data.batch)
    
        # Reparameterization Trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
    
        # Decoding
        nodes_logits, edges_logits = decoder(z)

        # Hungarian algorithm to find optimal permutation
        nodes_logits, edges_logits = match_and_reorder(nodes_logits, edges_logits, dense_nodes, dense_edges, beta=0)

        # GraphVAE Loss 
        recon_loss = compute_loss(nodes_logits, edges_logits, dense_nodes, dense_edges)
        
        # KL Divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta * kl_loss
        
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()

        running_loss += total_loss.item()
    
    # Validation
    encoder.eval()
    decoder.eval()   
    val_loss = 0.0

    with torch.no_grad():
        for data in validation_loader:
            data = data.to(device)
            B = data.num_graphs
            dense_nodes = data.dense_nodes.view(B, 9)
            dense_edges = data.dense_edges.view(B, 9, 9)

            # Encoding
            mu, logvar = encoder(data.x, data.edge_index, data.edge_attr, data.batch)

            # Reparametrization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            # Decoding
            nodes_logits, edges_logits = decoder(z)

            # Hungarian algorithm to find optimal permutation
            nodes_logits, edges_logits = match_and_reorder(nodes_logits, edges_logits, dense_nodes, dense_edges, beta=0)

            # Loss computation
            recon_loss_val = compute_loss(nodes_logits, edges_logits, dense_nodes, dense_edges)
            kl_loss_val = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            current_val_loss = recon_loss_val + beta * kl_loss_val
            val_loss += current_val_loss.item()
            
     
    epoch_loss = running_loss / len(train_loader)
    avg_loss_val = val_loss / len(validation_loader)
    
    # Learning rate update
    scheduler.step(avg_loss_val)

    # Early stopping and saving optimal weights
    if avg_loss_val < best_val_loss:
        best_val_loss = avg_loss_val
        torch.save(encoder.state_dict(), 'GraphVAE_model/' + ENCODER_PATH)
        torch.save(decoder.state_dict(), 'GraphVAE_model/' + DECODER_PATH)
        patience_counter = 0
    else:
        patience_counter += 1
    
    train_loss_history.append(epoch_loss)
    val_loss_history.append(avg_loss_val)

    print(f"Epoca {epoch+1}/{epochs} completata. Loss Train: {epoch_loss:.4f} - Loss Validation: {avg_loss_val:.4f}")

    if patience_counter >= patience:
        print("Early stopping attivato.")
        break

with open('GraphVAE_model/train_loss_history.pkl', 'wb') as f:
    pickle.dump(train_loss_history, f)

with open('GraphVAE_model/validation_loss_history.pkl', 'wb') as f:
    pickle.dump(val_loss_history, f) 

# %%
import matplotlib.pyplot as plt
import pickle

with open('GraphVAE_model/train_loss_history.pkl', 'rb') as f:
    train_loss = pickle.load(f)

with open('GraphVAE_model/validation_loss_history.pkl', 'rb') as f:
    val_loss = pickle.load(f)

epochs = range(1, len(train_loss) + 1)

# Loss curve
plt.figure(figsize=(8, 5), dpi=300) 
plt.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--')

plt.title('Learning Curve - GraphVAE', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='upper right')

plt.xlim(1, len(train_loss))

plt.tight_layout()

plt.savefig('GraphVAE_model/learning_curve_qm9.png', bbox_inches='tight', dpi=300)
plt.show()



