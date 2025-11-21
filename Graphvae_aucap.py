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

# =============== SinCos 位置编码（单图） ===============
def sinusoidal_pe(num_nodes: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(num_nodes, dim, device=device)
    position = torch.arange(0, num_nodes, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(np.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# =============== KL ===============
def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# =============== AUC & AP（1D 上三角向量） ===============
@torch.no_grad()
def calc_auc_ap(pred_scores_1d: torch.Tensor, target_1d: torch.Tensor) -> Tuple[float, float]:
    """
    计算 ROC-AUC 与 AP（Average Precision）。
    pred_scores_1d: 预测分数，越大越倾向为正类。可用 sigmoid(logits) 概率，也可直接用 logits。
    target_1d:      {0,1} 的真值
    返回: (auc, ap)
    备注：
      - AUC 用 Mann–Whitney U 等价实现（基于排序求秩），不做 ties 校正（概率基本无完全相等）。
      - AP 用“正样本处的 precision 平均值”的等价定义，和 sklearn 的 AP 一致（无插值时）。
    """
    y = target_1d.float().view(-1)
    s = pred_scores_1d.view(-1)

    n = y.numel()
    n_pos = int(y.sum().item())
    n_neg = n - n_pos

    # AUC：极端情况处理
    if n_pos == 0 or n_neg == 0:
        auc = 0.5  # 无法定义时给 0.5（中性）
    else:
        # 使用秩和实现 AUC： (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        order = torch.argsort(s)  # 从小到大
        ranks = torch.empty_like(order, dtype=torch.float)
        ranks[order] = torch.arange(1, n + 1, device=s.device, dtype=torch.float)
        R_pos = ranks[y == 1].sum()
        auc = (R_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg + 1e-12)
        auc = float(auc.item())

    # AP：极端情况处理
    if n_pos == 0:
        ap = 0.0
    elif n_neg == 0:
        ap = 1.0
    else:
        # 按分数从大到小排序
        desc = torch.argsort(s, descending=True)
        y_sorted = y[desc]
        # 累计 TP，位置从 1 开始计
        tp_cum = torch.cumsum(y_sorted, dim=0)
        idx = torch.arange(1, n + 1, device=s.device, dtype=torch.float)
        # 只在正样本处取 precision
        pos_mask = (y_sorted == 1)
        precision_at_k = tp_cum[pos_mask] / idx[pos_mask]
        # AP = 正样本处 precision 的均值
        ap = float(precision_at_k.mean().item()) if precision_at_k.numel() > 0 else 0.0

    return auc, ap

# ==========================
# Encoder
# ==========================
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

# ==========================
# Decoder（批处理版，输出 logits）
# ==========================
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

        tgt_kpm = (~node_mask)                                    # True=忽略
        mem_kpm = (~kv_mask)

        h_dec = self.decoder(tgt=q_proj, memory=h_pool,
                             tgt_key_padding_mask=tgt_kpm,
                             memory_key_padding_mask=mem_kpm)     # [B, Nmax, D]

        hi = h_dec.unsqueeze(2).expand(B, Nmax, Nmax, h_dec.size(-1))
        hj = h_dec.unsqueeze(1).expand(B, Nmax, Nmax, h_dec.size(-1))
        pair = torch.cat([hi, hj], dim=-1)                        # [B, Nmax, Nmax, 2D]
        logits = self.edge_mlp(pair).squeeze(-1)                  # [B, Nmax, Nmax]

        eye = torch.eye(Nmax, device=logits.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(eye, float('-inf'))
        return logits

# ==========================
# GraphVAE（批处理）
# ==========================
class GraphVAEPooled(nn.Module):
    def __init__(self, feat_dim, pe_dim, hidden_dim, latent_dim, pool_ratio=0.5):
        super().__init__()
        self.pe_dim = pe_dim
        self.feat_dim = feat_dim
        self.encoder = GCNEncoderWithPool(feat_dim, pe_dim, hidden_dim, latent_dim, pool_ratio)
        self.decoder = TransformerDecoderCross(latent_dim, pe_dim=pe_dim, feat_dim=feat_dim)

    def forward(self, x_all, edge_index_all, batch_vec, pe_dense, node_mask):
        pe_flat = pe_dense.view(-1, self.pe_dim)
        eff_idx = node_mask.view(-1)
        z_pool, mu_pool, logvar_pool, batch_pool = self.encoder(
            x_all, pe_flat[eff_idx], edge_index_all, batch_vec
        )

        h_pool, kv_mask = to_dense_batch(z_pool, batch_pool)      # [B, Kmax, D], [B, Kmax]
        mu_pool_d, _ = to_dense_batch(mu_pool, batch_pool)        # [B, Kmax, D]
        logvar_pool_d, _ = to_dense_batch(logvar_pool, batch_pool)

        B, Nmax = node_mask.size()
        mask_nodes = x_all.new_zeros((B, Nmax, self.feat_dim))
        logits = self.decoder(mask_nodes, pe_dense, h_pool, node_mask, kv_mask)
        return logits, mu_pool_d, logvar_pool_d, kv_mask

# ==========================
# 训练（批处理 + AMP）
# ==========================
def train_model(epochs=200, batch_size=128, pe_dim=8, hidden_dim=64, latent_dim=64, pool_ratio=0.5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = QM9(root="data/QM9")
    # dataset = dataset[:5000]  # 如需快速验证

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
        auc_sum = 0.0
        ap_sum  = 0.0
        denom_loader = max(1, len(loader))

        progress = min(1.0, epoch / max(1, int(epochs * warmup_ratio)))
        beta = beta_max * progress

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", mininterval=0.5, dynamic_ncols=True)
        for data in pbar:
            data = data.to(device)
            x_all, edge_index_all = data.x, data.edge_index
            batch_vec = data.batch

            x_dense, node_mask = to_dense_batch(x_all, batch_vec)   # [B,Nmax,F], [B,Nmax]
            B, Nmax, _ = x_dense.shape

            # 构造每图 PE（原版），如需“度数感知分配”，在这里替换
            pe_list = []
            for b in range(B):
                n_b = int(node_mask[b].sum())
                pe_b = sinusoidal_pe(n_b, pe_dim, device)
                pe_pad = torch.zeros(Nmax, pe_dim, device=device)
                pe_pad[:n_b] = pe_b
                pe_list.append(pe_pad)
            pe_dense = torch.stack(pe_list, dim=0)                  # [B, Nmax, pe_dim]

            adj_dense = to_dense_adj(edge_index_all, batch=batch_vec)  # [B, Nmax, Nmax]

            triu_mask = torch.triu(torch.ones(Nmax, Nmax, device=device, dtype=torch.bool), diagonal=1)
            valid_pair_mask = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1)) & triu_mask.unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, mu_pool_d, logvar_pool_d, kv_mask = model(
                    x_all, edge_index_all, batch_vec, pe_dense, node_mask
                )

                logits_u = logits[valid_pair_mask]      # [M]
                target_u = adj_dense[valid_pair_mask]   # [M]

                pos = target_u.sum()
                total_u = target_u.numel()
                neg = total_u - pos
                if pos.item() < 0.5:
                    pos_weight_val = 1.0
                else:
                    pos_weight_val = float(torch.clamp(neg / (pos + 1e-6), 1.0, 20.0).item())
                bce.pos_weight = torch.tensor([pos_weight_val], device=device)

                recon = bce(logits_u, target_u)

                kv_mask_flat = kv_mask.unsqueeze(-1).expand_as(mu_pool_d)
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

            # === 指标：AUC + AP（建议用 logits 直接喂；也可用 sigmoid 后概率）===
            with torch.no_grad():
                # 用 logits 计算更加稳定（阈值无关）
                auc, ap = calc_auc_ap(logits_u, target_u)
                auc_sum += auc
                ap_sum  += ap

            pbar.set_postfix(loss=float(loss.item()), auc=float(auc), ap=float(ap))

        print(f"[Epoch {epoch:03d}] beta={beta:.3f} | loss={total_loss_epoch/denom_loader:.4f} | "
              f"AUC={auc_sum/denom_loader:.4f} AP={ap_sum/denom_loader:.4f}")

if __name__ == "__main__":
    train_model()
