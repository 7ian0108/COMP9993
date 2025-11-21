import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import math
import warnings
from typing import Tuple

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
    """
    为单个图生成 [num_nodes, dim] 的 Sin/Cos 位置编码。
    同一批次内 encoder / decoder 共享生成方式。
    """
    pe = torch.zeros(num_nodes, dim, device=device)
    position = torch.arange(0, num_nodes, device=device).unsqueeze(1)  # [N,1]
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


@torch.no_grad()
def calc_binary_metrics(prob: torch.Tensor, target: torch.Tensor):
    """
    输入:
      prob   : [M] 预测概率(sigmoid后)
      target : [M] 0/1
    返回:
      precision, recall, f1, acc, auc_roc, ap
    """
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

    return (precision.item(),
            recall.item(),
            f1.item(),
            acc.item(),
            float(auc_roc),
            float(ap_val))


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    标准 VAE KL 散度：KL[q(z|x) || N(0, I)]
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ============================================================
# 1. Encoder with TopKPooling
# ============================================================

class GCNEncoderWithPool(nn.Module):
    """
    大图 -> GCN -> (mu, logvar) -> reparam z -> TopKPooling
    输出小图 latent 节点 z_pool + mu/logvar 以及 pool 后的图结构信息
    """
    def __init__(self, in_dim: int, pe_dim: int, hidden_dim: int,
                 latent_dim: int, pool_ratio: float = 0.5):
        super().__init__()
        self.gc1 = GCNConv(in_dim + pe_dim, hidden_dim)
        self.gc_mu = GCNConv(hidden_dim, latent_dim)
        self.gc_logvar = GCNConv(hidden_dim, latent_dim)
        self.pool = TopKPooling(latent_dim, ratio=pool_ratio)

    def forward(self, x_raw, pe_nodefeat, edge_index, batch_vec):
        """
        x_raw:       [sumN, F]
        pe_nodefeat: [sumN, pe_dim] 与 x_raw 对齐的 SinCosPE
        edge_index:  [2, sumE]
        batch_vec:   [sumN]
        返回:
          z_pool:         [sumK, latent_dim]
          mu_pool:        [sumK, latent_dim]
          logvar_pool:    [sumK, latent_dim]
          edge_index_pool:[2, sumE_pool]  (没直接用到decoder，但可留着以后用)
          batch_pool:     [sumK]
        """
        x_in = torch.cat([x_raw, pe_nodefeat], dim=-1)  # [sumN, F+PE]
        h = F.relu(self.gc1(x_in, edge_index))          # [sumN, H]

        mu = self.gc_mu(h, edge_index)                  # [sumN, Z]
        logvar = self.gc_logvar(h, edge_index)          # [sumN, Z]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std                              # reparameterized z

        z_pool, edge_index_pool, _, batch_pool, perm, _ = self.pool(
            z, edge_index, batch=batch_vec
        )
        mu_pool = mu[perm]
        logvar_pool = logvar[perm]

        return z_pool, mu_pool, logvar_pool, edge_index_pool, batch_pool


# ============================================================
# 2. SmallGraphHead
#    - 把 pooled latent 节点 z_pool -> h_small (小图节点表示)
#    - 基于 pair-wise MLP 得到小图骨架 logits_small_clean
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
        """
        z_pool:     [sumK, latent_dim]
        batch_pool: [sumK]
        返回:
          h_small:             [B, Kmax, H]
          h_small_mask:        [B, Kmax]  True=节点有效
          logits_small_clean:  [B, Kmax, Kmax]  小图骨架的logits
        """
        h_all = self.node_proj(z_pool)  # [sumK, H]
        h_small, h_mask = to_dense_batch(h_all, batch_pool)  # [B,Kmax,H], [B,Kmax]

        B, Kmax, H = h_small.shape
        hi = h_small.unsqueeze(2).expand(B, Kmax, Kmax, H)
        hj = h_small.unsqueeze(1).expand(B, Kmax, Kmax, H)
        pair = torch.cat([hi, hj], dim=-1)  # [B,Kmax,Kmax,2H]

        logits_small_clean = self.edge_mlp(pair).squeeze(-1)  # [B,Kmax,Kmax]

        # 屏掉自环
        eye = torch.eye(Kmax, device=logits_small_clean.device, dtype=torch.bool).unsqueeze(0)
        logits_small_clean = logits_small_clean.masked_fill(eye, float('-inf'))

        return h_small, h_mask, logits_small_clean


# ============================================================
# 3. DDPM Components
#    - 时间嵌入 SinTimeEmbed
#    - DDPMDiffusionSchedule (alpha_bar 等放到指定 device 上)
#    - SmallGraphDDPMDenoiser: εθ 预测噪声
# ============================================================

class SinTimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
    def forward(self, t):
        """
        t: [B] long
        return: [B, dim] 正弦/余弦时间编码
        """
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-math.log(10000.0)/half)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B,half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B,dim]


class DDPMDiffusionSchedule:
    """
    标准 DDPM 调度器:
    - 线性 beta_t
    - alpha_t = 1 - beta_t
    - alpha_bar_t = prod_{s<=t} alpha_s
    提供 q(x_t | x_0) 采样: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
    """
    def __init__(self, T: int = 1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)   # [T]
        self.alphas = 1.0 - self.betas                                        # [T]
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)                    # [T]

    def sample_xt(self, x0, t):
        """
        x0: [B,K,K] in [0,1] (干净小图的概率矩阵)
        t:  [B] long, 0..T-1
        return:
          x_t:        [B,K,K]
          noise_eps:  [B,K,K] 实际加的高斯噪声
        """
        B, K, _ = x0.shape
        # alpha_bar_t shape: [B,1,1]
        alpha_bar_t = self.alpha_bar[t].view(B, 1, 1)
        eps = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps
        return x_t, eps


class SmallGraphDDPMDenoiser(nn.Module):
    """
    εθ(h_small, t, x_t) -> 预测噪声 ε_hat
    我们用 h_small 给节点级上下文 + 时间条件。
    这里简化为“node-only + time”条件，不显式拼 x_t；如果想更强可以把 x_t 也拼到pair特征中。
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
        """
        h_small:      [B,K,H]
        h_small_mask: [B,K] True=有效节点
        x_t:          [B,K,K]  (目前没拼进MPL特征，但保留参数位，之后可增强)
        t:            [B] long

        return:
          eps_pred:   [B,K,K]  预测噪声 ε_hat
        """
        B, K, H = h_small.shape
        # time conditioning
        t_emb = self.time_mlp(self.time_embed(t))  # [B,H]
        h_small_cond = h_small + t_emb.unsqueeze(1)  # [B,K,H]

        hi = h_small_cond.unsqueeze(2).expand(B, K, K, H)
        hj = h_small_cond.unsqueeze(1).expand(B, K, K, H)
        pair = torch.cat([hi, hj], dim=-1)          # [B,K,K,2H]

        eps_pred = self.edge_mlp(pair).squeeze(-1)  # [B,K,K]

        # mask无效节点对 -> 置0
        valid_row = h_small_mask.unsqueeze(2)       # [B,K,1]
        valid_col = h_small_mask.unsqueeze(1)       # [B,1,K]
        valid_mat = (valid_row & valid_col)         # [B,K,K]
        eps_pred = eps_pred.masked_fill(~valid_mat, 0.0)

        # 自环无意义 -> 置0
        eye = torch.eye(K, device=h_small.device, dtype=torch.bool).unsqueeze(0)
        eps_pred = eps_pred.masked_fill(eye, 0.0)

        return eps_pred


# ============================================================
# 4. TransformerDecoderCross
#    - 用小图节点表征当 memory
#    - 用相同 SinCosPE 构建大图 query
#    - 输出还原大图邻接 logits_big
# ============================================================

class TransformerDecoderCross(nn.Module):
    def __init__(self, latent_dim: int, pe_dim: int,
                 num_heads: int = 4, num_layers: int = 2,
                 feat_dim: int = 11):
        super().__init__()
        self.input_proj = nn.Linear(pe_dim + feat_dim, latent_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 2,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.edge_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)   # edge logits
        )

    def forward(self, node_mask_big, pe_dense_big, h_small_memory, h_small_mask):
        """
        node_mask_big:   [B,Nmax] True=有效节点
        pe_dense_big:    [B,Nmax,pe_dim]  (与 encoder 一致生成的 SinCosPE)
        h_small_memory:  [B,Kmax,H] 小图节点memory (refined小图结构感知)
        h_small_mask:    [B,Kmax] True=有效小图节点
        """
        B, Nmax, pe_dim = pe_dense_big.shape
        feat_dim = self.input_proj.in_features - pe_dim
        device = pe_dense_big.device

        zero_feat = torch.zeros(B, Nmax, feat_dim, device=device)
        q = torch.cat([zero_feat, pe_dense_big], dim=-1)    # [B,Nmax,feat_dim+pe_dim]
        q_proj = self.input_proj(q)                         # [B,Nmax,H]

        tgt_kpm = (~node_mask_big)   # True=pad位置
        mem_kpm = (~h_small_mask)    # True=pad位置

        h_dec = self.decoder(
            tgt=q_proj,
            memory=h_small_memory,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=mem_kpm
        )  # [B,Nmax,H]

        Hi = h_dec.unsqueeze(2).expand(B, Nmax, Nmax, h_dec.size(-1))
        Hj = h_dec.unsqueeze(1).expand(B, Nmax, Nmax, h_dec.size(-1))
        pair_big = torch.cat([Hi, Hj], dim=-1)              # [B,Nmax,Nmax,2H]
        logits_big = self.edge_mlp(pair_big).squeeze(-1)    # [B,Nmax,Nmax]

        # mask无效区域
        valid_row = node_mask_big.unsqueeze(2)
        valid_col = node_mask_big.unsqueeze(1)
        valid_mat = (valid_row & valid_col)
        logits_big = logits_big.masked_fill(~valid_mat, float('-inf'))

        # 去掉自环
        eye_big = torch.eye(Nmax, device=device, dtype=torch.bool).unsqueeze(0)
        logits_big = logits_big.masked_fill(eye_big, float('-inf'))

        return logits_big


# ============================================================
# 5. 总模型 GraphGenModel
# ============================================================

class GraphGenModel(nn.Module):
    """
    主流程:
    1. 大图 -> Encoder+TopKPooling -> 小图 latent 节点
    2. small_head 得到 h_small & 小图骨架 logits_small_clean
    3. 把 logits_small_clean 过 sigmoid 当作 x0 (干净小图概率矩阵)
    4. DDPM: 随机时间步 t, 根据 q(x_t|x0) 采样 x_t 和 真噪声 eps_true
       再用 SmallGraphDDPMDenoiser 预测 eps_pred (εθ)
    5. Decoder: 用 h_small 作为 memory，配合同一份 SinCosPE，重建大图邻接 logits_big
    """
    def __init__(self,
                 in_dim_nodefeat: int,
                 pe_dim: int,
                 enc_hidden_dim: int,
                 latent_dim: int,
                 pool_ratio: float,
                 small_hidden_dim: int,
                 dec_heads: int,
                 dec_layers: int,
                 ddpm_T: int = 1000,
                 device: torch.device = torch.device("cpu")):
        super().__init__()

        self.encoder = GCNEncoderWithPool(
            in_dim=in_dim_nodefeat,
            pe_dim=pe_dim,
            hidden_dim=enc_hidden_dim,
            latent_dim=latent_dim,
            pool_ratio=pool_ratio
        )

        self.small_head = SmallGraphHead(
            in_dim=latent_dim,
            hidden_dim=small_hidden_dim
        )

        # <<< 修复点：把schedule放到正确device上 >>>
        self.ddpm_schedule = DDPMDiffusionSchedule(
            T=ddpm_T,
            beta_start=1e-4,
            beta_end=2e-2,
            device=device
        )

        self.ddpm_denoiser = SmallGraphDDPMDenoiser(
            node_dim=small_hidden_dim,
            time_dim=64,
            hidden_dim=128
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

    def forward(self, data):
        """
        data: torch_geometric 的 batch Data
        返回:
          logits_big:    [B,Nmax,Nmax] 大图预测边logits
          node_mask_big: [B,Nmax]      大图有效节点mask
          mu_pool_d:     [B,Kmax,Z]
          logvar_pool_d: [B,Kmax,Z]
          h_small_mask:  [B,Kmax]
          noise_pred:    [B,Kmax,Kmax]  预测噪声 ε_hat
          noise_true:    [B,Kmax,Kmax]  实际噪声 ε
          t:             [B]            这次sample的时间步
          A_clean:       [B,Kmax,Kmax]  干净小图概率矩阵 x0 (仅用于debug/分析)
        """
        device = data.x.device

        # ---------- 1) Dense化原图 + SinCosPE (共享) ----------
        x_all = data.x                     # [sumN, F]
        edge_index_all = data.edge_index   # [2,sumE]
        batch_vec = data.batch             # [sumN]

        x_dense_big, node_mask_big = to_dense_batch(x_all, batch_vec)  # [B,Nmax,F], [B,Nmax]
        B, Nmax, _ = x_dense_big.shape

        # 逐图生成 PE，拼成 pe_concat (与 x_all 顺序对齐)
        pe_list_flat = []
        for b in range(B):
            n_b = int(node_mask_big[b].sum())
            pe_b = sinusoidal_pe(n_b, self.pe_dim, device)  # [n_b, pe_dim]
            pe_list_flat.append(pe_b)
        pe_concat = torch.cat(pe_list_flat, dim=0)          # [sumN, pe_dim]

        # ---------- 2) Encoder + TopKPooling 变小图 ----------
        z_pool, mu_pool, logvar_pool, edge_index_pool, batch_pool = self.encoder(
            x_raw=x_all,
            pe_nodefeat=pe_concat,
            edge_index=edge_index_all,
            batch_vec=batch_vec
        )
        # z_pool: [sumK, latent_dim]
        # batch_pool: [sumK]

        # ---------- 3) SmallGraphHead 得到小图骨架 ----------
        h_small, h_small_mask, logits_small_clean = self.small_head(
            z_pool, batch_pool
        )
        # h_small:            [B,Kmax,H]
        # h_small_mask:       [B,Kmax]
        # logits_small_clean: [B,Kmax,Kmax]

        # 把小图骨架 logits -> 概率，作为 DDPM 的 x0
        A_clean = torch.sigmoid(logits_small_clean)  # [B,Kmax,Kmax], in [0,1]

        # ---------- 4) DDPM 前向扩散采样随机 t ----------
        t = torch.randint(
            low=0,
            high=self.ddpm_schedule.T,
            size=(B,),
            device=device
        )  # [B]
        x_t, noise_true = self.ddpm_schedule.sample_xt(A_clean, t)  # [B,Kmax,Kmax] each

        # ---------- 5) DDPM 噪声预测 εθ ----------
        noise_pred = self.ddpm_denoiser(
            h_small, h_small_mask, x_t, t
        )  # [B,Kmax,Kmax]

        # ---------- 6) pad mu/logvar 给 KL ----------
        mu_pool_d, _ = to_dense_batch(mu_pool, batch_pool)          # [B,Kmax,Z]
        logvar_pool_d, _ = to_dense_batch(logvar_pool, batch_pool)  # [B,Kmax,Z]

        # ---------- 7) Decoder：同一 SinCosPE 还原大图 ----------
        # 把 pe_concat 切回每个大图并 pad 到 Nmax -> pe_dense_big
        pe_dense_big = []
        idx_start = 0
        for b in range(B):
            n_b = int(node_mask_big[b].sum())
            pe_b = pe_concat[idx_start:idx_start+n_b]  # [n_b, pe_dim]
            idx_start += n_b

            pad_pe = torch.zeros(Nmax, self.pe_dim, device=device)
            pad_pe[:n_b] = pe_b
            pe_dense_big.append(pad_pe)
        pe_dense_big = torch.stack(pe_dense_big, dim=0)  # [B,Nmax,pe_dim]

        logits_big = self.decoder(
            node_mask_big=node_mask_big,
            pe_dense_big=pe_dense_big,
            h_small_memory=h_small,        # 目前直接用h_small当memory
            h_small_mask=h_small_mask
        )  # [B,Nmax,Nmax]

        return (
            logits_big,
            node_mask_big,
            mu_pool_d,
            logvar_pool_d,
            h_small_mask,
            noise_pred,
            noise_true,
            t,
            A_clean,
        )


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
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
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
        device=device  # <<< 关键：把GPU/CPU传进去给DDPM schedule
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 使用新的 torch.amp API，消掉 FutureWarning
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # KL warmup
    warmup_ratio = 0.3
    beta_max = 0.5

    bce = nn.BCEWithLogitsLoss(reduction='mean')
    mse = nn.MSELoss(reduction='mean')  # DDPM 噪声预测 MSE

    for epoch in range(1, epochs + 1):
        model.train()

        progress = min(1.0, epoch / max(1, int(epochs * warmup_ratio)))
        beta_kl = beta_max * progress  # KL anneal

        total_loss_epoch = 0.0
        total_auc = 0.0
        total_ap = 0.0
        total_f1 = 0.0
        total_acc = 0.0
        total_cnt = 0

        pbar = tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]", mininterval=0.5, dynamic_ncols=True)

        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            # autocast 新写法
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                (logits_big,
                 node_mask_big,
                 mu_pool_d,
                 logvar_pool_d,
                 h_small_mask,
                 noise_pred,
                 noise_true,
                 t_batch,
                 A_clean) = model(data)

                # ========== 大图重建 BCE ==========
                B, Nmax, _ = logits_big.shape
                adj_dense_big = to_dense_adj(data.edge_index, batch=data.batch)  # [B,Nmax,Nmax]

                triu_mask_big = torch.triu(
                    torch.ones(Nmax, Nmax, device=device, dtype=torch.bool),
                    diagonal=1
                ).unsqueeze(0)  # [1,Nmax,Nmax]

                valid_pairs_big = (node_mask_big.unsqueeze(2) &
                                   node_mask_big.unsqueeze(1) &
                                   triu_mask_big)  # [B,Nmax,Nmax]

                logits_big_u = logits_big[valid_pairs_big]        # [M_big]
                target_big_u = adj_dense_big[valid_pairs_big]     # [M_big]

                loss_big_recon = bce(logits_big_u, target_big_u)

                # ========== KL Loss ==========
                B2, Kmax, Zdim = mu_pool_d.shape
                valid_small_mask = h_small_mask.unsqueeze(-1).expand(-1, -1, Zdim)  # [B,Kmax,Z]
                if valid_small_mask.any():
                    mu_v = mu_pool_d[valid_small_mask].view(-1, Zdim)
                    logvar_v = logvar_pool_d[valid_small_mask].view(-1, Zdim)
                    loss_kl = kl_normal(mu_v, logvar_v)
                else:
                    loss_kl = torch.tensor(0.0, device=device)

                # ========== DDPM 噪声预测 MSE ==========
                # 只在小图有效节点对 (i<j) 上算
                triu_mask_small = torch.triu(
                    torch.ones(Kmax, Kmax, device=device, dtype=torch.bool),
                    diagonal=1
                ).unsqueeze(0)  # [1,Kmax,Kmax]

                valid_pairs_small = (h_small_mask.unsqueeze(2) &
                                     h_small_mask.unsqueeze(1) &
                                     triu_mask_small)  # [B,Kmax,Kmax]

                eps_pred_u = noise_pred[valid_pairs_small]  # [M_small]
                eps_true_u = noise_true[valid_pairs_small]  # [M_small]

                if eps_pred_u.numel() == 0:
                    loss_ddpm = torch.tensor(0.0, device=device)
                else:
                    loss_ddpm = mse(eps_pred_u, eps_true_u)

                # 总Loss
                loss = loss_big_recon + beta_kl * loss_kl + loss_ddpm

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss_epoch += loss.item()
            total_cnt += 1

            # 监控指标（针对大图重建）
            with torch.no_grad():
                prob_big_u = torch.sigmoid(logits_big_u)
                precision, recall, f1, acc, auc_val, ap_val = calc_binary_metrics(
                    prob_big_u, target_big_u
                )
                total_auc += 0.0 if np.isnan(auc_val) else auc_val
                total_ap  += 0.0 if np.isnan(ap_val) else ap_val
                total_f1  += f1
                total_acc += acc

                pbar.set_postfix({
                    "loss": float(loss.item()),
                    "recon": float(loss_big_recon.item()),
                    "kl": float(loss_kl.item()),
                    "ddpm": float(loss_ddpm.item()),
                    "auc": auc_val,
                    "ap": ap_val,
                    "f1": f1
                })

        avg_loss = total_loss_epoch / max(1, total_cnt)
        avg_auc  = total_auc / max(1, total_cnt)
        avg_ap   = total_ap / max(1, total_cnt)
        avg_f1   = total_f1 / max(1, total_cnt)
        avg_acc  = total_acc / max(1, total_cnt)

        print(f"[Epoch {epoch:03d}] betaKL={beta_kl:.3f} | "
              f"loss={avg_loss:.4f} | "
              f"AUC={avg_auc:.4f} AP={avg_ap:.4f} "
              f"F1={avg_f1:.4f} Acc={avg_acc:.4f}")


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
        num_workers=2   # Windows 下如果卡 dataloader，可以降成 0 或 1
    )
