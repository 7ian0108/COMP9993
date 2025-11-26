
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import math, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

dataset = QM9(root="./data/QM9")
N_total = len(dataset)
N_train = int(0.85 * N_total)
N_val   = int(0.05  * N_total)
train_set = dataset[:N_train]
val_set   = dataset[N_train:N_train+N_val]
test_set  = dataset[N_train+N_val:]

ATOM_Z = [1,6,7,8,9]
Z2ID = {z:i for i,z in enumerate(ATOM_Z)}
N_ATOM = len(ATOM_Z)

def get_bond_dim(sample):
    return sample.edge_attr.size(-1)
N_BOND = get_bond_dim(dataset[0])

def to_atom_id(z_tensor):
    ids = torch.zeros_like(z_tensor)
    for z, idx in Z2ID.items():
        ids[z_tensor==z] = idx
    return ids

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)

def linear_beta_schedule(T, beta_start=1e-3, beta_end=0.05):
    return torch.linspace(beta_start, beta_end, T)

class DiscreteDiffusion:
    def __init__(self, T=200, n_atom=5, n_bond=4, device="cpu"):
        self.T = T
        self.device = device
        self.n_atom = n_atom
        self.n_bond = n_bond
        self.betas = linear_beta_schedule(T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)  # [T]

    def q_sample_nodes(self, y0, t_node):
        keep_prob = self.alpha_bar[t_node]                 # [N]
        replace = (torch.rand_like(keep_prob) > keep_prob).long()  
        noise = torch.randint(low=0, high=self.n_atom, size=y0.shape, device=y0.device)
        return y0*(1-replace) + noise*replace

    def q_sample_edges(self, e0, t_edge):
        keep_prob = self.alpha_bar[t_edge]
        replace = (torch.rand_like(keep_prob) > keep_prob).long()
        noise = torch.randint(low=0, high=self.n_bond, size=e0.shape, device=e0.device)
        return e0*(1-replace) + noise*replace

diff = DiscreteDiffusion(T=200, n_atom=N_ATOM, n_bond=N_BOND, device=device)

class SinTimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim%2==0
        self.dim = dim
    def forward(self, t_graph):
        half = self.dim//2
        freqs = torch.exp(torch.arange(half, device=t_graph.device, dtype=torch.float32)
                          * (-math.log(10000.0)/half))
        args = t_graph[:,None].float()*freqs[None,:]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class DiGressGNN(nn.Module):
    def __init__(self, n_atom, n_bond, hidden=256, tdim=64):
        super().__init__()
        self.n_atom = n_atom
        self.n_bond = n_bond
        in_dim = n_atom
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.temb = SinTimeEmbed(tdim)
        self.t_mlp = nn.Sequential(nn.Linear(tdim, hidden), nn.SiLU())
        self.node_head = nn.Linear(hidden, n_atom)
        self.edge_mlp  = nn.Sequential(nn.Linear(2*hidden, hidden), nn.SiLU(),
                                       nn.Linear(hidden, n_bond))

    def forward(self, node_x_oh, edge_index, t_graph, batch_vec):
        h = F.relu(self.conv1(node_x_oh, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        temb = self.t_mlp(self.temb(t_graph))             # [G, hidden]
        h = h + temb[batch_vec]                         
        node_logits = self.node_head(h)                   # [N, n_atom]
        src, dst = edge_index
        edge_h = torch.cat([h[src], h[dst]], dim=-1)      # [E, 2H]
        edge_logits = self.edge_mlp(edge_h)               # [E, n_bond]
        return node_logits, edge_logits

model = DiGressGNN(N_ATOM, N_BOND).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-4)

def one_hot(labels, num_classes):
    return F.one_hot(labels.long(), num_classes=num_classes).float()

@torch.no_grad()
def node_edge_accuracy(logits_n, y_n, logits_e, y_e):
    n_acc = (logits_n.argmax(-1) == y_n).float().mean().item()
    e_acc = (logits_e.argmax(-1) == y_e).float().mean().item()
    return n_acc, e_acc

def run_epoch(loader, train=True, epoch=0):
    model.train() if train else model.eval()
    phase = "Train" if train else "Val"
    total_loss = 0.0
    total_nacc, total_eacc, total_cnt = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"[{phase}] Epoch {epoch}", leave=False, mininterval=0.5)
    with torch.set_grad_enabled(train):
        for data in pbar:
            data = data.to(device)
            batch_vec  = data.batch
            G = data.num_graphs

            y_node = to_atom_id(data.z).to(device)           # [N]
            y_edge = data.edge_attr.argmax(-1).to(device)    # [E]

            t_graph = torch.randint(0, diff.T, (G,), device=device).long()
            t_node  = t_graph[batch_vec]                     # [N]
            src, dst = data.edge_index
            t_edge = batch_vec[src]                          # [E]


            x_node_t = diff.q_sample_nodes(y_node, t_node)   # [N]
            # x_edge_t = diff.q_sample_edges(y_edge, t_edge)  

            node_oh = one_hot(x_node_t, N_ATOM)

            node_logits, edge_logits = model(node_oh, data.edge_index, t_graph, batch_vec)
            loss_node = F.cross_entropy(node_logits, y_node)
            loss_edge = F.cross_entropy(edge_logits, y_edge)
            loss = loss_node + loss_edge

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            nacc, eacc = node_edge_accuracy(node_logits, y_node, edge_logits, y_edge)
            total_loss += loss.item()
            total_nacc += nacc
            total_eacc += eacc
            total_cnt  += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "n_acc": f"{nacc:.3f}", "e_acc": f"{eacc:.3f}"})
    return total_loss/max(1,total_cnt), total_nacc/max(1,total_cnt), total_eacc/max(1,total_cnt)


def sample_greedy_from_batch(model, data, T=None):

    model.eval()
    data = data.to(device)
    batch_vec = data.batch
    G = data.num_graphs
    N = data.x.size(0)
    T = diff.T if T is None else T

    x_node = torch.randint(0, N_ATOM, (N,), device=device)

    edge_logits = None
    for t in reversed(range(T)):
        t_graph = torch.full((G,), t, device=device, dtype=torch.long)
        node_oh = one_hot(x_node, N_ATOM)
        node_out, edge_out = model(node_oh, data.edge_index, t_graph, batch_vec)
        x_node = node_out.argmax(-1)
        edge_logits = edge_out  
    return x_node.detach().cpu(), edge_logits.detach().cpu()


ALLOWED_VALENCE = {1:1, 6:4, 7:3, 8:2, 9:1}  # H,C,N,O,F

BOND_ORDER = {0:1, 1:2, 2:3, 3:1}            # SINGLE,DOUBLE,TRIPLE,AROMATIC≈1

BOND_TYPE = {
    0: rdchem.BondType.SINGLE,
    1: rdchem.BondType.DOUBLE,
    2: rdchem.BondType.TRIPLE,
    3: rdchem.BondType.AROMATIC,
}

def build_mol_repair(atom_cls, edge_index, edge_logits):

    rw = Chem.RWMol()
    idx_map = {}
    try:
  
        Zs = []
        for i, c in enumerate(atom_cls):
            Z = ATOM_Z[int(c)]
            a = Chem.Atom(int(Z))
            idx = rw.AddAtom(a)
            idx_map[i] = idx
            Zs.append(Z)

        remain = [ALLOWED_VALENCE.get(Z, 4) for Z in Zs]


        with torch.no_grad():
            probs = torch.softmax(edge_logits, dim=-1).cpu().numpy()  # [E, n_bond]
            scores = probs.max(axis=1)
            pred   = probs.argmax(axis=1)

        src, dst = edge_index
        E = src.size(0)
        edges = list(range(E))

        edges = [e for e in edges if int(src[e]) < int(dst[e])]

        edges.sort(key=lambda e: float(scores[e]), reverse=True)

        for e in edges:
            u = int(src[e]); v = int(dst[e])
            want = int(pred[e])

    
            candidates = [want, 0, 1, 3, 2]
            placed = False
            for cand in candidates:
                order = BOND_ORDER[cand]
                if remain[u] >= order and remain[v] >= order:
                    rw.AddBond(idx_map[u], idx_map[v], BOND_TYPE[cand])
                    remain[u] -= order; remain[v] -= order
                    placed = True
                    break
     
            _ = placed

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)  
        return mol
    except Exception:
        return None


@torch.no_grad()
def evaluate_valid_unique(model, loader, max_batches=30):
    mols = []
    for i, data in enumerate(loader):
        if i >= max_batches: break
        atom_cls, edge_logits = sample_greedy_from_batch(model, data, T=diff.T)
        mol = build_mol_repair(atom_cls, data.edge_index.cpu(), edge_logits)
        mols.append(mol)

    valid_mols = [m for m in mols if m is not None]
    valid = len(valid_mols) / max(1, len(mols))

    smis = []
    for m in valid_mols:
        try:
            s = Chem.MolToSmiles(m, canonical=True)
            smis.append(s)
        except Exception:
            pass
    unique = len(set(smis)) / max(1, len(valid_mols))
    return {"valid": valid, "unique": unique}

EPOCHS = 200
logs = {"tr_loss":[], "va_loss":[], "tr_nacc":[], "va_nacc":[], "tr_eacc":[], "va_eacc":[]}

for ep in range(1, EPOCHS+1):
    tr_loss, tr_nacc, tr_eacc = run_epoch(train_loader, True, ep)
    va_loss, va_nacc, va_eacc = run_epoch(val_loader,   False, ep)
    logs["tr_loss"].append(tr_loss); logs["va_loss"].append(va_loss)
    logs["tr_nacc"].append(tr_nacc); logs["va_nacc"].append(va_nacc)
    logs["tr_eacc"].append(tr_eacc); logs["va_eacc"].append(va_eacc)
    tqdm.write(f"Epoch {ep:02d} | "
               f"Train {tr_loss:.4f} (n_acc {tr_nacc:.3f}, e_acc {tr_eacc:.3f}) | "
               f"Val {va_loss:.4f} (n_acc {va_nacc:.3f}, e_acc {va_eacc:.3f}) | "
               f"LR {scheduler.get_last_lr()[0]:.5f}")
    scheduler.step()

vu = evaluate_valid_unique(model, val_loader, max_batches=30)
print("\n[Val Sampling] Valid: {:.3f} | Unique: {:.3f}".format(vu["valid"], vu["unique"]))


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(logs["tr_loss"], label="train")
plt.plot(logs["va_loss"], label="val")
plt.title("CE loss"); plt.grid(True); plt.legend()
plt.subplot(1,2,2)
plt.plot(logs["va_nacc"], label="node acc")
plt.plot(logs["va_eacc"], label="edge acc")
plt.title("Val accuracy"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

