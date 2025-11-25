import os, math, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset = QM9(root='./data/QM9')
node_dim = dataset.num_features
N_total = len(dataset)
N_train = int(0.85 * N_total)
N_val   = int(0.05  * N_total)
train_set = dataset[:N_train]
val_set   = dataset[N_train:N_train+N_val]
test_set  = dataset[N_train+N_val:]
print(f"QM9 loaded. total={N_total}, train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
print("Node feature dim:", node_dim)


with torch.no_grad():
    s = torch.zeros(node_dim)
    ss = torch.zeros(node_dim)
    n_nodes = 0
    for d in train_set:
        x = d.x
        s  += x.sum(0)
        ss += (x**2).sum(0)
        n_nodes += x.size(0)
    mean = s / n_nodes
    var  = ss / n_nodes - mean**2
    std  = torch.clamp(var, min=1e-12).sqrt()
print("Feature mean/std ready.")

def norm_x(x):   return (x - mean.to(x.device)) / std.to(x.device)
def denorm_x(x): return x * std.to(x.device) + mean.to(x.device)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 1e-4
    beta_end   = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class Diffusion:
    def __init__(self, timesteps=400, device="cpu"):
        self.timesteps = timesteps
        self.betas  = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.ac     = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ac = torch.sqrt(self.ac)
        self.sqrt_om = torch.sqrt(1.0 - self.ac)

    def q_sample(self, x0, t_node, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa  = self.sqrt_ac[t_node].unsqueeze(-1)
        som = self.sqrt_om[t_node].unsqueeze(-1)
        return sa * x0 + som * noise

diffusion = Diffusion(timesteps=400, device=device)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "tdim must be even"
        self.dim = dim
    def forward(self, t_graph):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t_graph.device, dtype=torch.float32)
                          * (-math.log(10000.0)/half))
        args = t_graph[:,None].float() * freqs[None,:]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class GCN_DDPM(nn.Module):
    def __init__(self, in_dim, hidden=256, tdim=64):
        super().__init__()
        self.temb = SinusoidalTimeEmbedding(tdim)
        self.t_mlp = nn.Sequential(nn.Linear(tdim, hidden), nn.SiLU())
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, in_dim)
        )
    def forward(self, x, edge_index, t_graph, batch_vec):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        temb = self.t_mlp(self.temb(t_graph))   # [G, hidden]
        h = h + temb[batch_vec]
        return self.mlp(h)

model = GCN_DDPM(in_dim=node_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-4)

def _cos_sim(a, b, eps=1e-8):
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return (an * bn).sum(-1)

@torch.no_grad()
def evaluate_regression(loader):
    model.eval()
    y_true, y_pred = [], []

    for data in tqdm(loader, desc="Evaluating", leave=False, mininterval=0.5):
        data = data.to(device)
        x0 = norm_x(data.x)
        edge_index = data.edge_index
        batch_vec  = data.batch
        G = data.num_graphs
        t_graph = torch.randint(0, diffusion.timesteps, (G,), device=device).long()
        t_node  = t_graph[batch_vec]

        xt = diffusion.q_sample(x0, t_node)
        pred_noise = model(xt, edge_index, t_graph, batch_vec)
        x_recon = (xt - diffusion.sqrt_om[t_node].unsqueeze(-1) * pred_noise) / \
                   diffusion.sqrt_ac[t_node].unsqueeze(-1)

        y_true.append(denorm_x(x0).detach().cpu().numpy())
        y_pred.append(denorm_x(x_recon).detach().cpu().numpy())

    Yt = np.concatenate(y_true, axis=0)
    Yp = np.concatenate(y_pred, axis=0)
    r2 = r2_score(Yt.ravel(), Yp.ravel())
    cos = _cos_sim(Yt.reshape(Yt.shape[0], -1),
                   Yp.reshape(Yp.shape[0], -1)).mean()
    return {"R2": float(r2), "Cosine": float(cos)}

def run_epoch(loader, train=True, epoch=None):
    model.train() if train else model.eval()
    phase = "Train" if train else "Val"
    total = 0.0
    pbar = tqdm(loader, desc=f"[{phase}] Epoch {epoch}", leave=False, mininterval=0.5)
    with torch.set_grad_enabled(train):
        for data in pbar:
            data = data.to(device)
            x0 = norm_x(data.x)
            edge_index = data.edge_index
            batch_vec  = data.batch
            G = data.num_graphs
            t_graph = torch.randint(0, diffusion.timesteps, (G,), device=device).long()
            t_node  = t_graph[batch_vec]
            noise = torch.randn_like(x0)
            xt = diffusion.q_sample(x0, t_node, noise)
            pred_noise = model(xt, edge_index, t_graph, batch_vec)
            loss = F.mse_loss(pred_noise, noise)    
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total / max(1, len(loader))

EPOCHS = 800
logs = {"train_loss": [], "val_loss": [], "val_r2": [], "val_cos": []}

for ep in range(1, EPOCHS+1):
    tr = run_epoch(train_loader, True,  ep)
    va = run_epoch(val_loader,   False, ep)
    regm = evaluate_regression(val_loader)
    logs["train_loss"].append(tr)
    logs["val_loss"].append(va)
    logs["val_r2"].append(regm["R2"])
    logs["val_cos"].append(regm["Cosine"])
    tqdm.write(f"Epoch {ep:02d} | Train {tr:.4f} | Val {va:.4f} | "
               f"R2 {regm['R2']:.4f} | Cos {regm['Cosine']:.4f} | "
               f"LR {scheduler.get_last_lr()[0]:.5f}")
    scheduler.step()

torch.save(model.state_dict(), "graph_ddpm_qm9.pt")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(logs["train_loss"], label="Train Loss")
plt.plot(logs["val_loss"],   label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss per Epoch")
plt.grid(True); plt.legend()

plt.subplot(1,2,2)
plt.plot(logs["val_r2"],  label="Val R²")
plt.plot(logs["val_cos"], label="Val Cosine")
plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("R² & Cosine per Epoch")
plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()


final = evaluate_regression(val_loader)
print("\n=== Final (Val) ===")
for k,v in final.items():
    print(f"{k}: {v:.4f}")
print("\nSaved model to: graph_ddpm_qm9.pt")

