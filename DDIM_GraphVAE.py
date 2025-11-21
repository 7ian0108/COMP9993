import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import math
import warnings
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import to_dense_adj, to_dense_batch

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 0. 基础工具
# ============================================================

def sinusoidal_pe(num_nodes: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(num_nodes, dim, device=device)
    position = torch.arange(0, num_nodes, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


@torch.no_grad()
def calc_binary_metrics(prob: torch.Tensor, target: torch.Tensor):
    pred_bin = (prob > 0.5).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    acc       = (pred_bin == target).float().mean()

    t_cpu = target.detach().cpu().numpy()
    p_cpu = prob.detach().cpu().numpy()
    try:
        auc_roc = roc_auc_score(t_cpu, p_cpu)
    except ValueError:
        auc_roc = float("nan")
    try:
        ap_val = average_precision_score(t_cpu, p_cpu)
    except ValueError:
        ap_val = float("nan")

    return (precision.item(), recall.item(), f1.item(), acc.item(), float(auc_roc), float(ap_val))


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ============================================================
# 1. Encoder with TopKPooling
# ============================================================

class GCNEncoderWithPool(nn.Module):
    def __init__(self, in_dim: int, pe_dim: int, hidden_dim: int,
                 latent_dim: int, pool_ratio: float = 0.5):
        super().__init__()
        self.gc1 = GCNConv(in_dim + pe_dim, hidden_dim)
        self.gc_mu = GCNConv(hidden_dim, latent_dim)
        self.gc_logvar = GCNConv(hidden_dim, latent_dim)
        self.pool = TopKPooling(latent_dim, ratio=pool_ratio)

    def forward(self, x_raw, pe_nodefeat, edge_index, batch_vec):
        x_in = torch.cat([x_raw, pe_nodefeat], dim=-1)
        h = F.relu(self.gc1(x_in, edge_index))
        mu = self.gc_mu(h, edge_index)
        logvar = self.gc_logvar(h, edge_index)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z_pool, edge_index_pool, _, batch_pool, perm, _ = self.pool(z, edge_index, batch=batch_vec)
        mu_pool = mu[perm]
        logvar_pool = logvar[perm]
        return z_pool, mu_pool, logvar_pool, edge_index_pool, batch_pool


# ============================================================
# 2. SmallGraphHead
# ============================================================

class SmallGraphHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_pool, batch_pool):
        h_all = self.node_proj(z_pool)  # [sumK, H]
        h_small, h_mask = to_dense_batch(h_all, batch_pool)  # [B,Kmax,H], [B,Kmax]

        B, Kmax, H = h_small.shape
        hi = h_small.unsqueeze(2).expand(B, Kmax, Kmax, H)
        hj = h_small.unsqueeze(1).expand(B, Kmax, Kmax, H)
        pair = torch.cat([hi, hj], dim=-1)
        logits_small_clean = self.edge_mlp(pair).squeeze(-1)  # [B,Kmax,Kmax]

        eye = torch.eye(Kmax, device=logits_small_clean.device, dtype=torch.bool).unsqueeze(0)
        logits_small_clean = logits_small_clean.masked_fill(eye, float('-inf'))
        return h_small, h_mask, logits_small_clean


# ============================================================
# 3. DDIM Components
# ============================================================

class SinTimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                          * (-math.log(10000.0)/half))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DDIMDiffusionSchedule:
    """
    训练阶段与 DDPM 相同（用 q(x_t | x_0) 加噪，loss 预测 ε）；
    推理阶段使用 DDIM 的确定性步进：
      x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_hat
                + sqrt(1 - alpha_bar_{t-1}) * eps_hat
      其中 x0_hat = (x_t - sqrt(1 - alpha_bar_t)*eps_hat) / sqrt(alpha_bar_t)
    """
    def __init__(self, T: int = 1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)   # [T]
        self.alphas = 1.0 - self.betas                                        # [T]
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)                    # [T]

    def sample_xt(self, x0, t):
        B, K, _ = x0.shape
        alpha_bar_t = self.alpha_bar[t].view(B, 1, 1)
        eps = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps
        return x_t, eps

    @torch.no_grad()
    def ddim_step(self, x_t, t, t_prev, eps_hat):
        """
        单步 DDIM (eta=0 的确定性版本)：
          x0_hat = (x_t - sqrt(1 - a_t)*eps_hat) / sqrt(a_t)
          x_{t_prev} = sqrt(a_{t_prev}) * x0_hat + sqrt(1 - a_{t_prev}) * eps_hat
        """
        B = x_t.size(0)
        a_t = self.alpha_bar[t].view(B, 1, 1)         # [B,1,1]
        a_prev = self.alpha_bar[t_prev].view(B, 1, 1) # [B,1,1]

        x0_hat = (x_t - torch.sqrt(1.0 - a_t) * eps_hat) / torch.sqrt(a_t + 1e-12)
        x_prev = torch.sqrt(a_prev) * x0_hat + torch.sqrt(1.0 - a_prev) * eps_hat
        return x_prev.clamp_(0.0, 1.0)  # 我们把小图当概率矩阵，截断到[0,1]

    @torch.no_grad()
    def ddim_sample(self, eps_net, h_small, h_small_mask, Kmax: int,
                    steps: int = 50, device="cpu", time_index: Optional[List[int]] = None):
        """
        从纯噪声出发的 DDIM 采样，返回 [B,Kmax,Kmax] 概率矩阵。
        - eps_net: εθ(h_small, t, x_t) -> eps
        - h_small/h_small_mask: 条件
        - steps: 采样步数（<= T）
        - time_index: 可选的时间子序列（降序），否则均匀子采样
        """
        B = h_small.size(0)
        if time_index is None:
            # 均匀子采样 T/steps 个点（含0）
            full = torch.linspace(self.T-1, 0, steps, device=device).long()
            time_index = full.tolist()
        else:
            time_index = [int(x) for x in time_index]
            assert time_index[0] < self.T and time_index[-1] >= 0
            assert all(time_index[i] > time_index[i+1] for i in range(len(time_index)-1))

        # 初始从N(0,1)出发，再映射近似到[0,1]（用sigmoid也可，这里直接标准化+截断）
        x_t = torch.randn(B, Kmax, Kmax, device=device)
        x_t = (x_t - x_t.min()) / (x_t.max() - x_t.min() + 1e-8)

        for i in range(len(time_index)-1):
            t = torch.full((B,), time_index[i], device=device, dtype=torch.long)
            t_prev = torch.full((B,), time_index[i+1], device=device, dtype=torch.long)
            eps_hat = eps_net(h_small, h_small_mask, x_t, t)
            x_t = self.ddim_step(x_t, t, t_prev, eps_hat)

            # 屏蔽无效小图节点对与自环
            valid_row = h_small_mask.unsqueeze(2)
            valid_col = h_small_mask.unsqueeze(1)
            valid_mat = (valid_row & valid_col)
            x_t = x_t.masked_fill(~valid_mat, 0.0)
            eye = torch.eye(Kmax, device=device, dtype=torch.bool).unsqueeze(0)
            x_t = x_t.masked_fill(eye, 0.0)

        return x_t  # [B,Kmax,Kmax] 概率矩阵


class SmallGraphDDIMDenoiser(nn.Module):
    """
    与 DDPM 版本相同的 εθ 结构；DDIM 与 DDPM 的训练目标相同（预测噪声 ε）。
    """
    def __init__(self, node_dim: int, time_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.time_embed = SinTimeEmbed(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_small, h_small_mask, x_t, t):
        B, K, H = h_small.shape
        t_emb = self.time_mlp(self.time_embed(t))  # [B,H]
        h_small_cond = h_small + t_emb.unsqueeze(1)  # [B,K,H]

        hi = h_small_cond.unsqueeze(2).expand(B, K, K, H)
        hj = h_small_cond.unsqueeze(1).expand(B, K, K, H)
        pair = torch.cat([hi, hj], dim=-1)          # [B,K,K,2H]

        eps_pred = self.edge_mlp(pair).squeeze(-1)  # [B,K,K]

        valid_row = h_small_mask.unsqueeze(2)
        valid_col = h_small_mask.unsqueeze(1)
        valid_mat = (valid_row & valid_col)
        eps_pred = eps_pred.masked_fill(~valid_mat, 0.0)

        eye = torch.eye(K, device=h_small.device, dtype=torch.bool).unsqueeze(0)
        eps_pred = eps_pred.masked_fill(eye, 0.0)
        return eps_pred


# ============================================================
# 4. TransformerDecoderCross (大图解码)
# ============================================================

class TransformerDecoderCross(nn.Module):
    def __init__(self, latent_dim: int, pe_dim: int,
                 num_heads: int = 4, num_layers: int = 2,
                 feat_dim: int = 11):
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
            nn.Linear(latent_dim, 1)
        )

    def forward(self, node_mask_big, pe_dense_big, h_small_memory, h_small_mask):
        B, Nmax, pe_dim = pe_dense_big.shape
        feat_dim = self.input_proj.in_features - pe_dim
        device = pe_dense_big.device

        zero_feat = torch.zeros(B, Nmax, feat_dim, device=device)
        q = torch.cat([zero_feat, pe_dense_big], dim=-1)
        q_proj = self.input_proj(q)

        tgt_kpm = (~node_mask_big)
        mem_kpm = (~h_small_mask)

        h_dec = self.decoder(
            tgt=q_proj,
            memory=h_small_memory,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=mem_kpm
        )

        Hi = h_dec.unsqueeze(2).expand(B, Nmax, Nmax, h_dec.size(-1))
        Hj = h_dec.unsqueeze(1).expand(B, Nmax, Nmax, h_dec.size(-1))
        pair_big = torch.cat([Hi, Hj], dim=-1)
        logits_big = self.edge_mlp(pair_big).squeeze(-1)

        valid_row = node_mask_big.unsqueeze(2)
        valid_col = node_mask_big.unsqueeze(1)
        valid_mat = (valid_row & valid_col)
        logits_big = logits_big.masked_fill(~valid_mat, float('-inf'))
        eye_big = torch.eye(Nmax, device=device, dtype=torch.bool).unsqueeze(0)
        logits_big = logits_big.masked_fill(eye_big, float('-inf'))
        return logits_big


# ============================================================
# 5. 总模型 GraphGenModel（DDIM 版本）
# ============================================================

class GraphGenModel(nn.Module):
    def __init__(self,
                 in_dim_nodefeat: int,
                 pe_dim: int,
                 enc_hidden_dim: int,
                 latent_dim: int,
                 pool_ratio: float,
                 small_hidden_dim: int,
                 dec_heads: int,
                 dec_layers: int,
                 ddpm_T: int = 1000,  # 时间轴总步数，沿用命名
                 device: torch.device = torch.device("cpu")):
        super().__init__()

        self.encoder = GCNEncoderWithPool(
            in_dim=in_dim_nodefeat,
            pe_dim=pe_dim,
            hidden_dim=enc_hidden_dim,
            latent_dim=latent_dim,
            pool_ratio=pool_ratio
        )
        self.small_head = SmallGraphHead(in_dim=latent_dim, hidden_dim=small_hidden_dim)

        # —— 核心替换：DDIM Schedule & Denoiser ——
        self.ddim_schedule = DDIMDiffusionSchedule(
            T=ddpm_T, beta_start=1e-4, beta_end=2e-2, device=device
        )
        self.ddim_denoiser = SmallGraphDDIMDenoiser(
            node_dim=small_hidden_dim, time_dim=64, hidden_dim=128
        )

        self.decoder = TransformerDecoderCross(
            latent_dim=small_hidden_dim,
            pe_dim=pe_dim,
            num_heads=dec_heads,
            num_layers=dec_layers,
            feat_dim=in_dim_nodefeat
        )

        self.pe_dim = pe_dim
        self.in_dim_nodefeat = in_dim_nodefeat
        self.device = device

    def forward(self, data):
        device = data.x.device
        x_all, edge_index_all, batch_vec = data.x, data.edge_index, data.batch

        x_dense_big, node_mask_big = to_dense_batch(x_all, batch_vec)
        B, Nmax, _ = x_dense_big.shape

        # 共享 SinCosPE（与 encoder/decoder 统一）
        pe_list_flat = []
        for b in range(B):
            n_b = int(node_mask_big[b].sum())
            pe_b = sinusoidal_pe(n_b, self.pe_dim, device)
            pe_list_flat.append(pe_b)
        pe_concat = torch.cat(pe_list_flat, dim=0)

        # Encoder + Pool
        z_pool, mu_pool, logvar_pool, edge_index_pool, batch_pool = self.encoder(
            x_raw=x_all, pe_nodefeat=pe_concat,
            edge_index=edge_index_all, batch_vec=batch_vec
        )

        # 小图骨架（干净 x0 概率）
        h_small, h_small_mask, logits_small_clean = self.small_head(z_pool, batch_pool)
        A_clean = torch.sigmoid(logits_small_clean)  # x0

        # DDIM 训练等价于 DDPM 的噪声预测训练：随机 t，加噪，MSE(ε_pred, ε_true)
        t = torch.randint(low=0, high=self.ddim_schedule.T, size=(B,), device=device)
        x_t, noise_true = self.ddim_schedule.sample_xt(A_clean, t)
        noise_pred = self.ddim_denoiser(h_small, h_small_mask, x_t, t)

        mu_pool_d, _ = to_dense_batch(mu_pool, batch_pool)
        logvar_pool_d, _ = to_dense_batch(logvar_pool, batch_pool)

        # 还原大图（注意：训练阶段仍使用 clean h_small 作为 memory）
        # （也可尝试把 x_t -> x0_hat 的一步反推作为 memory，但先保持稳定版）
        pe_dense_big = []
        idx = 0
        for b in range(B):
            n_b = int(node_mask_big[b].sum())
            pad = torch.zeros(Nmax, self.pe_dim, device=device)
            pad[:n_b] = pe_concat[idx:idx+n_b]
            pe_dense_big.append(pad)
            idx += n_b
        pe_dense_big = torch.stack(pe_dense_big, dim=0)  # [B,Nmax,pe_dim]

        logits_big = self.decoder(
            node_mask_big=node_mask_big,
            pe_dense_big=pe_dense_big,
            h_small_memory=h_small,
            h_small_mask=h_small_mask
        )

        return (logits_big, node_mask_big, mu_pool_d, logvar_pool_d,
                h_small_mask, noise_pred, noise_true, t, A_clean, h_small)

    # ---------------- 推理：DDIM 采样（可选在验证/测试时调用） ----------------
    @torch.no_grad()
    def ddim_sample_small(self, h_small, h_small_mask, steps: int = 50,
                          time_index: Optional[List[int]] = None):
        """
        用 DDIM 从纯噪声确定性生成小图概率矩阵 A_small ∈ [0,1]^(B×K×K)
        然后你可以把它交给 decoder 还原成大图。
        """
        device = h_small.device
        B, Kmax, _ = h_small.shape
        A_small = self.ddim_schedule.ddim_sample(
            eps_net=self.ddim_denoiser,
            h_small=h_small,
            h_small_mask=h_small_mask,
            Kmax=Kmax,
            steps=steps,
            device=device,
            time_index=time_index
        )
        return A_small  # [B,Kmax,Kmax]


# ============================================================
# 6. 训练
# ============================================================

def train_model(
    epochs=50,
    batch_size=64,
    pe_dim=8,
    enc_hidden_dim=64,
    latent_dim=64,
    pool_ratio=0.5,
    small_hidden_dim=64,
    lr=1e-3,
    num_workers=2
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = QM9(root="data/QM9")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    model = GraphGenModel(
        in_dim_nodefeat=dataset.num_node_features,
        pe_dim=pe_dim,
        enc_hidden_dim=enc_hidden_dim,
        latent_dim=latent_dim,
        pool_ratio=pool_ratio,
        small_hidden_dim=small_hidden_dim,
        dec_heads=4,
        dec_layers=2,
        ddpm_T=1000,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    warmup_ratio = 0.3
    beta_max = 0.5

    bce = nn.BCEWithLogitsLoss(reduction='mean')
    mse = nn.MSELoss(reduction='mean')

    for epoch in range(1, epochs + 1):
        model.train()

        progress = min(1.0, epoch / max(1, int(epochs * warmup_ratio)))
        beta_kl = beta_max * progress

        total_loss, total_auc, total_ap, total_f1, total_acc, total_cnt = 0.0, 0.0, 0.0, 0.0, 0.0, 0

        pbar = tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]", mininterval=0.5, dynamic_ncols=True)
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                (logits_big, node_mask_big, mu_pool_d, logvar_pool_d,
                 h_small_mask, noise_pred, noise_true, t_batch,
                 A_clean, h_small) = model(data)

                # 大图 BCE
                B, Nmax, _ = logits_big.shape
                adj_dense_big = to_dense_adj(data.edge_index, batch=data.batch)
                triu_mask_big = torch.triu(torch.ones(Nmax, Nmax, device=device, dtype=torch.bool), 1).unsqueeze(0)
                valid_pairs_big = (node_mask_big.unsqueeze(2) &
                                   node_mask_big.unsqueeze(1) & triu_mask_big)
                logits_big_u = logits_big[valid_pairs_big]
                target_big_u = adj_dense_big[valid_pairs_big]
                loss_big = bce(logits_big_u, target_big_u)

                # KL
                B2, Kmax, Zdim = mu_pool_d.shape
                mask3 = h_small_mask.unsqueeze(-1).expand(-1, -1, Zdim)
                if mask3.any():
                    mu_v = mu_pool_d[mask3].view(-1, Zdim)
                    logvar_v = logvar_pool_d[mask3].view(-1, Zdim)
                    loss_kl = kl_normal(mu_v, logvar_v)
                else:
                    loss_kl = torch.tensor(0.0, device=device)

                # DDIM (训练仍为 MSE ε)
                triu_mask_small = torch.triu(torch.ones(Kmax, Kmax, device=device, dtype=torch.bool), 1).unsqueeze(0)
                valid_pairs_small = (h_small_mask.unsqueeze(2) &
                                     h_small_mask.unsqueeze(1) & triu_mask_small)
                eps_pred_u = noise_pred[valid_pairs_small]
                eps_true_u = noise_true[valid_pairs_small]
                loss_ddim = torch.tensor(0.0, device=device) if eps_pred_u.numel() == 0 else mse(eps_pred_u, eps_true_u)

                loss = loss_big + beta_kl * loss_kl + loss_ddim

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_cnt += 1

            with torch.no_grad():
                prob_big_u = torch.sigmoid(logits_big_u)
                p, r, f1, acc, auc_v, ap_v = calc_binary_metrics(prob_big_u, target_big_u)
                total_auc += 0.0 if np.isnan(auc_v) else auc_v
                total_ap  += 0.0 if np.isnan(ap_v) else ap_v
                total_f1  += f1
                total_acc += acc
                pbar.set_postfix({
                    "loss": float(loss.item()),
                    "recon": float(loss_big.item()),
                    "kl": float(loss_kl.item()),
                    "ddim": float(loss_ddim.item()),
                    "auc": auc_v, "ap": ap_v, "f1": f1
                })

        avg_loss = total_loss / max(1, total_cnt)
        print(f"[Epoch {epoch:03d}] betaKL={beta_kl:.3f} | loss={avg_loss:.4f} | "
              f"AUC={total_auc/max(1,total_cnt):.4f} AP={total_ap/max(1,total_cnt):.4f} "
              f"F1={total_f1/max(1,total_cnt):.4f} Acc={total_acc/max(1,total_cnt):.4f}")


# ============================================================
# 7. main
# ============================================================

if __name__ == "__main__":
    train_model(
        epochs=200,
        batch_size=64,
        pe_dim=8,
        enc_hidden_dim=64,
        latent_dim=64,
        pool_ratio=0.5,
        small_hidden_dim=64,
        lr=1e-3,
        num_workers=2
    )
