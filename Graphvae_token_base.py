import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from tqdm import tqdm

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import to_dense_adj, to_dense_batch

def sinusoidal_pe(num_nodes: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(num_nodes, dim, device=device)
    position = torch.arange(0, num_nodes, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(np.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

@torch.no_grad()
def calc_metrics(pred_prob_1d: torch.Tensor, target_1d: torch.Tensor) -> Tuple[float, float, float, float]:
    pred_bin = (pred_prob_1d > 0.5).float()
    tp = (pred_bin * target_1d).sum()
    fp = (pred_bin * (1 - target_1d)).sum()
    fn = ((1 - pred_bin) * target_1d).sum()
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    acc       = (pred_bin == target_1d).float().mean()
    return precision.item(), recall.item(), f1.item(), acc.item()

def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

class GCNEncoderWithPool(nn.Module):
    def __init__(self, in_dim: int, pe_dim: int, hidden_dim: int, latent_dim: int, pool_ratio: float = 0.5):
        super().__init__()
        self.gc1 = GCNConv(in_dim + pe_dim, hidden_dim)
        self.gc_mu = GCNConv(hidden_dim, latent_dim)
        self.gc_logvar = GCNConv(hidden_dim, latent_dim)
        self.pool = TopKPooling(latent_dim, ratio=pool_ratio)

    def forward(self, x_raw, sincos_pe, edge_index, batch):
        x_in = torch.cat([x_raw, sincos_pe], dim=-1)              # [sumN, in+pe]
        h = F.relu(self.gc1(x_in, edge_index))                    # [sumN, H]
        mu = self.gc_mu(h, edge_index)                            # [sumN, Z]
        logvar = self.gc_logvar(h, edge_index)                    # [sumN, Z]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z_pool, edge_index_pool, _, batch_pool, perm, _ = self.pool(z, edge_index, batch=batch)
        mu_pool = mu[perm]
        logvar_pool = logvar[perm]
        return z_pool, mu_pool, logvar_pool, batch_pool


class TransformerDecoderCross(nn.Module):
    def __init__(self, latent_dim: int, pe_dim: int = 8, num_heads: int = 4, num_layers: int = 2, feat_dim: int = 11):
        super().__init__()
        self.input_proj = nn.Linear(pe_dim + feat_dim, latent_dim)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=num_heads,
            dim_feedforward=latent_dim * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.edge_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),   # logits
        )

    def forward(self, mask_nodes, sincos_pe, h_pool, node_mask, kv_mask):
        B, Nmax, _ = mask_nodes.shape
        q = torch.cat([mask_nodes, sincos_pe], dim=-1)            # [B, Nmax, F+pe]
        q_proj = self.input_proj(q)                               # [B, Nmax, D]

        tgt_kpm = (~node_mask)                                    # [B, Nmax]
        mem_kpm = (~kv_mask)                                      # [B, Kmax]

        h_dec = self.decoder(tgt=q_proj, memory=h_pool,
                             tgt_key_padding_mask=tgt_kpm,
                             memory_key_padding_mask=mem_kpm)     # [B, Nmax, D]

        hi = h_dec.unsqueeze(2).expand(B, Nmax, Nmax, h_dec.size(-1))
        hj = h_dec.unsqueeze(1).expand(B, Nmax, Nmax, h_dec.size(-1))
        pair = torch.cat([hi, hj], dim=-1)                        # [B, Nmax, Nmax, 2D]
        logits = self.edge_mlp(pair).squeeze(-1)                  # [B, Nmax, Nmax]

        eye = torch.eye(Nmax, device=logits.device, dtype=torch.bool).unsqueeze(0)  # [1,Nmax,Nmax]
        logits = logits.masked_fill(eye, float('-inf'))
        return logits

class GraphVAEPooled(nn.Module):
    def __init__(self, feat_dim, pe_dim, hidden_dim, latent_dim, pool_ratio=0.5):
        super().__init__()
        self.pe_dim = pe_dim
        self.feat_dim = feat_dim
        self.encoder = GCNEncoderWithPool(feat_dim, pe_dim, hidden_dim, latent_dim, pool_ratio)
        self.decoder = TransformerDecoderCross(latent_dim, pe_dim=pe_dim, feat_dim=feat_dim)

    def forward(self, x_all, edge_index_all, batch_vec, pe_dense, node_mask):
        z_pool, mu_pool, logvar_pool, batch_pool = self.encoder(x_all, pe_dense.view(-1, self.pe_dim)[node_mask.view(-1)], edge_index_all, batch_vec)

        h_pool, kv_mask = to_dense_batch(z_pool, batch_pool)      # [B, Kmax, D], [B, Kmax]
        mu_pool_d, _ = to_dense_batch(mu_pool, batch_pool)        # [B, Kmax, D]
        logvar_pool_d, _ = to_dense_batch(logvar_pool, batch_pool)

        B, Nmax = node_mask.size()
        mask_nodes = x_all.new_zeros((B, Nmax, self.feat_dim))
        logits = self.decoder(mask_nodes, pe_dense, h_pool, node_mask, kv_mask)  # [B,Nmax,Nmax]
        return logits, mu_pool_d, logvar_pool_d, kv_mask

def train_model(epochs=200, batch_size=128, pe_dim=8, hidden_dim=64, latent_dim=64, pool_ratio=0.5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = QM9(root="data/QM9")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,            
        pin_memory=True,
        persistent_workers=True
    )

    model = GraphVAEPooled(11, pe_dim, hidden_dim, latent_dim, pool_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    warmup_ratio = 0.3
    beta_max = 0.5
    bce = nn.BCEWithLogitsLoss(reduction='mean')  

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        p_sum = r_sum = f1_sum = acc_sum = 0.0
        denom_loader = max(1, len(loader))

        progress = min(1.0, epoch / max(1, int(epochs * warmup_ratio)))
        beta = beta_max * progress

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", mininterval=0.5, dynamic_ncols=True)
        for data in pbar:
            data = data.to(device)
            x_all, edge_index_all = data.x, data.edge_index
            batch_vec = data.batch                               

   
            x_dense, node_mask = to_dense_batch(x_all, batch_vec)  # [B,Nmax,F], [B,Nmax]
            B, Nmax, Fdim = x_dense.shape
            
            pe_list = []
            for b in range(B):
                n_b = int(node_mask[b].sum())
                pe_b = sinusoidal_pe(n_b, pe_dim, device)
                pe_pad = torch.zeros(Nmax, pe_dim, device=device)
                pe_pad[:n_b] = pe_b
                pe_list.append(pe_pad)
            pe_dense = torch.stack(pe_list, dim=0)                 # [B, Nmax, pe_dim]

 
            adj_dense = to_dense_adj(edge_index_all, batch=batch_vec)  # [B, Nmax, Nmax]

            triu_mask = torch.triu(torch.ones(Nmax, Nmax, device=device, dtype=torch.bool), diagonal=1)  # [Nmax,Nmax]
            valid_pair_mask = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1)) & triu_mask.unsqueeze(0) # [B,Nmax,Nmax]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
              
                logits, mu_pool_d, logvar_pool_d, kv_mask = model(
                    x_all, edge_index_all, batch_vec, pe_dense, node_mask
                )                                                  

               
                logits_u = logits[valid_pair_mask]                  # [M]
                target_u = adj_dense[valid_pair_mask]               # [M]

                pos = target_u.sum()
                total_u = target_u.numel()
                neg = total_u - pos
                if pos.item() < 0.5:
                    pos_weight_val = 1.0
                else:
                    pos_weight_val = float(torch.clamp(neg / (pos + 1e-6), 1.0, 20.0).item())
 
                bce.pos_weight = torch.tensor([pos_weight_val], device=device)

                recon = bce(logits_u, target_u)

          
                kv_mask_flat = kv_mask.unsqueeze(-1).expand_as(mu_pool_d)      # [B,Kmax,D]
                mu_v = mu_pool_d[kv_mask_flat].view(-1, mu_pool_d.size(-1))
                logvar_v = logvar_pool_d[kv_mask_flat].view(-1, logvar_pool_d.size(-1))
                if mu_v.numel() == 0:
                    kl = torch.tensor(0.0, device=device)
                else:
                    kl = kl_normal(mu_v, logvar_v)

                loss = recon + beta * kl

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_epoch += loss.item()

            with torch.no_grad():
                prob_u = torch.sigmoid(logits_u)
                p, r, f1, acc = calc_metrics(prob_u, target_u)
                p_sum += p; r_sum += r; f1_sum += f1; acc_sum += acc

            pbar.set_postfix(loss=float(loss.item()), f1=float(f1))

        print(f"[Epoch {epoch:03d}] beta={beta:.3f} | loss={total_loss_epoch/denom_loader:.4f} | "
              f"P={p_sum/denom_loader:.4f} R={r_sum/denom_loader:.4f} F1={f1_sum/denom_loader:.4f} Acc={acc_sum/denom_loader:.4f}")

if __name__ == "__main__":
    train_model()

