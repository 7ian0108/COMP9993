import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import to_dense_adj, to_dense_batch


# =========================================================
# 0. 小工具
# =========================================================

def pairwise_dist(coords: torch.Tensor) -> torch.Tensor:
    """
    coords: [N,2]
    return: [N,N] 欧氏距离矩阵
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-9)


def rbf(dist: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    RBF 权函数（距离越大，权越小）
    """
    return torch.exp(-dist / max(tau, 1e-6))


# =========================================================
# 1. SinCos 位置编码 & 工具函数
# =========================================================

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


# =========================================================
# 1.1 依据 TopKPooling 的选择构造 dense 保留掩码
# =========================================================
@torch.no_grad()
def build_keep_mask_dense(node_mask: torch.Tensor,
                          data_ptr: torch.Tensor,
                          perm: torch.Tensor,
                          batch_pool: torch.Tensor) -> torch.Tensor:
    B, Nmax = node_mask.size()
    keep_mask = torch.zeros_like(node_mask, dtype=torch.bool)
    for b in range(B):
        start = int(data_ptr[b].item())
        keep_global_b = perm[(batch_pool == b)]
        if keep_global_b.numel() == 0:
            continue
        dense_j = (keep_global_b - start).clamp(min=0, max=Nmax-1)
        valid = node_mask[b]
        keep_mask[b, dense_j] = True
        keep_mask[b] &= valid
    return keep_mask


# =========================================================
# 2. GraphVAE 模型（Encoder/Decoder 强化 + 原图提示）
# =========================================================

class ResidualGCN(nn.Module):
    """
    两层 GCN + 残差 + BN：支持 edge_weight
    """
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim, cached=False, normalize=True)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim, cached=False, normalize=True)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = self.bn1(h)
        h = F.relu(h, inplace=True)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = self.bn2(h)
        return F.relu(h + self.proj(x), inplace=True)


class GCNEncoderWithPool(nn.Module):
    """
    Encoder:
    - 输入: concat(coords, pe)
    - 两层 ResidualGCN
    - 采样后得到 z，并返回 graph_token（未池化 z 的 mean）
    """
    def __init__(self, in_dim: int, pe_dim: int, hidden_dim: int, latent_dim: int, pool_ratio: float = 0.5):
        super().__init__()
        self.in_dim = in_dim + pe_dim
        self.backbone = ResidualGCN(self.in_dim, hidden_dim, hidden_dim)
        self.gc_mu = GCNConv(hidden_dim, latent_dim)
        self.gc_logvar = GCNConv(hidden_dim, latent_dim)
        self.pool = TopKPooling(latent_dim, ratio=pool_ratio)

    def forward(self, x_raw, sincos_pe, edge_index, batch, edge_weight=None):
        x_in = torch.cat([x_raw, sincos_pe], dim=-1)                     # [sumN, in+pe]
        h = self.backbone(x_in, edge_index, edge_weight=edge_weight)     # [sumN, H]

        mu = self.gc_mu(h, edge_index, edge_weight=edge_weight)          # [sumN, Z]
        logvar = self.gc_logvar(h, edge_index, edge_weight=edge_weight)  # [sumN, Z]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std                                               # 未池化 latent

        # 全图 token：每图对未池化 z 做 mean
        z_dense, z_mask = to_dense_batch(z, batch)                       # [B,Nmax,Z], [B,Nmax]
        graph_token = (z_dense.sum(dim=1) / (z_mask.sum(dim=1, keepdim=True) + 1e-6))  # [B,Z]

        # TopKPooling（在 z 上）
        z_pool, edge_index_pool, _, batch_pool, perm, _ = self.pool(z, edge_index, batch=batch)
        mu_pool = mu[perm]
        logvar_pool = logvar[perm]
        return z_pool, mu_pool, logvar_pool, batch_pool, perm, graph_token


class TransformerDecoderCross(nn.Module):
    """
    Decoder:
    - 输入节点特征: [x, y, deg_prior]，与 pe 拼接后做投影
    - Memory: concat([h_pool, graph_token]) 作为 cross-attention 的 key/value
    - Pair-wise Head: 计算边 logits
    - 距离先验 bias: 对 logits 叠加一个 -gamma * dist_norm 的偏置（仅用于 kept 子图）
    """
    def __init__(self, latent_dim: int, pe_dim: int = 8, num_heads: int = 4, num_layers: int = 2, feat_dim: int = 3,
                 dist_bias_gamma: float = 1.0):
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
            nn.Linear(latent_dim, 1),
        )
        self.dist_bias_gamma = dist_bias_gamma

    def forward(self, mask_nodes_aug, sincos_pe, h_pool, graph_token, node_mask, kv_mask, coords_dense):
        """
        mask_nodes_aug: [B, Nmax, 3] = [x, y, deg_prior]
        sincos_pe:      [B, Nmax, pe]
        h_pool:         [B, Kmax, D]
        graph_token:    [B, D]  -> 作为额外的一个 memory token
        node_mask:      [B, Nmax]  True 表示 tgt 有效
        kv_mask:        [B, Kmax]  True 表示 memory 有效（来自 pooling 结果）
        coords_dense:   [B, Nmax, 2]  用于距离先验
        """
        B, Nmax, _ = mask_nodes_aug.shape

        # === 输入投影 ===
        q = torch.cat([mask_nodes_aug, sincos_pe], dim=-1)               # [B, Nmax, 3+pe]
        q_proj = self.input_proj(q)                                      # [B, Nmax, D]

        # === 组装 Memory: [h_pool, graph_token] ===
        graph_tok = graph_token.unsqueeze(1)                             # [B,1,D]
        mem = torch.cat([h_pool, graph_tok], dim=1)                      # [B, Kmax+1, D]
        mem_mask = torch.cat([kv_mask, torch.ones(B, 1, dtype=torch.bool, device=kv_mask.device)], dim=1)  # True=valid

        # PyTorch Transformer 的 *key_padding_mask* 中 True=要MASK掉；当前 True 表示“有效”，需要取反
        tgt_kpm = (~node_mask)                                           # [B, Nmax]
        mem_kpm = (~mem_mask)                                            # [B, Kmax+1]

        # === Transformer 解码 ===
        h_dec = self.decoder(tgt=q_proj, memory=mem,
                             tgt_key_padding_mask=tgt_kpm,
                             memory_key_padding_mask=mem_kpm)            # [B, Nmax, D]

        # === Pairwise logits ===
        hi = h_dec.unsqueeze(2).expand(B, Nmax, Nmax, h_dec.size(-1))
        hj = h_dec.unsqueeze(1).expand(B, Nmax, Nmax, h_dec.size(-1))
        pair = torch.cat([hi, hj], dim=-1)                               # [B,Nmax,Nmax,2D]
        logits = self.edge_mlp(pair).squeeze(-1)                         # [B,Nmax,Nmax]

        # === 距离先验 bias（仅在 kept 子图内有效）===
        with torch.no_grad():
            D_all = []
            for b in range(B):
                coords_b = coords_dense[b]                                # [Nmax,2]
                nb = int(node_mask[b].sum().item())
                if nb == 0:
                    D_all.append(torch.zeros(Nmax, Nmax, device=coords_dense.device))
                    continue
                c = coords_b[:nb]
                d = pairwise_dist(c)                                      # [nb,nb]
                m = (d.sum() / (nb * nb) + 1e-6)
                d_norm = d / m
                pad = torch.zeros(Nmax, Nmax, device=coords_dense.device)
                pad[:nb, :nb] = d_norm
                D_all.append(pad)
            D_all = torch.stack(D_all, dim=0)                             # [B,Nmax,Nmax]

        logits = logits - self.dist_bias_gamma * D_all

        # 不允许自环
        eye = torch.eye(Nmax, device=logits.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(eye, float('-inf'))
        return logits


class GraphVAEPooled(nn.Module):
    def __init__(self, feat_dim, pe_dim, hidden_dim, latent_dim, pool_ratio=0.5,
                 dist_bias_gamma: float = 1.0):
        super().__init__()
        self.pe_dim = pe_dim
        self.feat_dim = feat_dim
        self.encoder = GCNEncoderWithPool(feat_dim, pe_dim, hidden_dim, latent_dim, pool_ratio)
        # 注意：decoder 现在期望 feat_dim=3（x, y, deg_prior）
        self.decoder = TransformerDecoderCross(latent_dim, pe_dim=pe_dim, feat_dim=3,
                                              dist_bias_gamma=dist_bias_gamma)

    def forward(self, x_all, edge_index_all, batch_vec, pe_dense, node_mask, data_ptr, edge_weight_all):
        """
        x_all:          [sumN, F]
        edge_index_all: [2, E]
        batch_vec:      [sumN]
        pe_dense:       [B, Nmax, pe]
        node_mask:      [B, Nmax]
        data_ptr:       [B+1]
        edge_weight_all:[E]  RBF 距离权
        """
        B, Nmax = node_mask.size()

        # Encoder 的 SinCos：从 dense 拉平后与 x_all 对齐
        pe_flat = pe_dense.view(-1, self.pe_dim)[node_mask.view(-1)]   # [sumN_valid, pe_dim]

        # === Encoder ===
        z_pool, mu_pool, logvar_pool, batch_pool, perm, graph_token = self.encoder(
            x_all, pe_flat, edge_index_all, batch_vec, edge_weight=edge_weight_all
        )

        # 池化后 dense
        h_pool, kv_mask = to_dense_batch(z_pool, batch_pool)
        mu_pool_d, _ = to_dense_batch(mu_pool, batch_pool)
        logvar_pool_d, _ = to_dense_batch(logvar_pool, batch_pool)

        # === 构造 keep_mask_dense：只在被保留节点上进行解码与监督 ===
        keep_mask_dense = build_keep_mask_dense(node_mask, data_ptr, perm, batch_pool)

        # Decoder 的节点特征：coords + deg_prior（dense）
        x_dense, _ = to_dense_batch(x_all, batch_vec)                   # [B, Nmax, 2]
        deg_prior = torch.zeros(B, Nmax, 1, device=x_all.device)        # 先占位

        # 基于坐标的 RBF，计算“子图度先验”
        for b in range(B):
            mask_b = keep_mask_dense[b]
            nb = int(mask_b.sum().item())
            if nb < 2:
                continue
            ids = torch.arange(Nmax, device=x_all.device)[mask_b]
            coords_b = x_dense[b, ids]                                  # [nb,2]
            dist_b = pairwise_dist(coords_b)                            # [nb,nb]
            w_b = rbf(dist_b, tau=0.1)                                  # [nb,nb]
            deg_b = w_b.sum(dim=1, keepdim=True)                        # [nb,1]
            deg_prior[b, ids] = deg_b

        mask_nodes_aug = torch.cat([x_dense, deg_prior], dim=-1)        # [B,Nmax,3]

        # Decoder 用的 SinCos & Memory
        logits = self.decoder(mask_nodes_aug, pe_dense, h_pool, graph_token,
                              keep_mask_dense, kv_mask, x_dense)

        return logits, mu_pool_d, logvar_pool_d, kv_mask, keep_mask_dense, perm, batch_pool


# =========================================================
# 3. TSP txt → PyG Dataset  （新增 tour_order + edge_weight）
# =========================================================

class TSPTxtDataset(torch.utils.data.Dataset):
    """
    读取 generate_data.py 生成的 txt：
    x0 y0 x1 y1 ... x{N-1} y{N-1} output i1 i2 ... iN i1
    """
    def __init__(self, txt_path: str):
        super().__init__()
        self.data_list = []

        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            tokens = line.strip().split()
            if not tokens:
                continue

            sep = tokens.index("output")
            coord_tokens = tokens[:sep]
            tour_tokens = tokens[sep + 1:]

            coords_flat = list(map(float, coord_tokens))
            coords = torch.tensor(coords_flat, dtype=torch.float32).view(-1, 2)  # [N,2]
            N = coords.size(0)

            # tour: i1 i2 ... iN i1 (1-based) -> 0-based
            tour_1 = torch.tensor(list(map(int, tour_tokens)), dtype=torch.long)  # [N+1]
            tour0 = tour_1 - 1
            if tour0[0] == tour0[-1]:
                tour0 = tour0[:-1]
            assert tour0.numel() == N

            # 原图：完全图（无向） + RBF 边权
            row, col = torch.triu_indices(N, N, offset=1)
            edge_index_full = torch.cat(
                [torch.stack([row, col], dim=0),
                 torch.stack([col, row], dim=0)],
                dim=1
            )
            # 计算两端点的距离并转为 RBF 权
            with torch.no_grad():
                dmat = pairwise_dist(coords)                               # [N,N]
                ew_upper = dmat[row, col]                                  # [E/2]
                ew_full = torch.cat([ew_upper, ew_upper], dim=0)           # [E]
                edge_weight = rbf(ew_full, tau=0.1).to(torch.float32)      # [E]

            # GT tour 子图（无向）
            src = tour0
            dst = torch.cat([tour0[1:], tour0[:1]], dim=0)
            tour_edge_dir = torch.stack([src, dst], dim=0)
            tour_edge_undir = torch.cat(
                [tour_edge_dir, tour_edge_dir.flip(0)],
                dim=1
            )

            # 最优长度（欧氏距离）
            pts = coords[tour0]
            diff = pts[1:] - pts[:-1]
            seg_len = torch.sqrt((diff ** 2).sum(dim=-1))
            last_leg = torch.sqrt(((pts[0] - pts[-1]) ** 2).sum())
            length_opt = (seg_len.sum() + last_leg).unsqueeze(0)

            data = Data(
                x=coords,
                edge_index=edge_index_full,
                edge_weight=edge_weight,
                tour_edge_index=tour_edge_undir,
                length_opt=length_opt,
                tour_order=tour0.clone()
            )
            self.data_list.append(data)

        print(f"Loaded {len(self.data_list)} TSP instances from {txt_path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# =========================================================
# 4. 纯贪心解码（无 opt）
# =========================================================

@torch.no_grad()
def decode_tour_from_logits(logits_b: torch.Tensor,
                            node_mask_b: torch.Tensor) -> torch.Tensor:
    N = int(node_mask_b.sum().item())
    logits_N = logits_b[:N, :N]
    prob = torch.sigmoid(logits_N)
    prob = (prob + prob.t()) / 2.0
    prob = prob - torch.eye(N, device=prob.device) * 1e9
    start = 0
    tour = [start]
    used = set([start])
    for _ in range(N - 1):
        i = tour[-1]
        cand = prob[i].clone()
        cand[list(used)] = -1e9
        j = int(cand.argmax().item())
        tour.append(j); used.add(j)
    return torch.tensor(tour, device=logits_b.device, dtype=torch.long)


@torch.no_grad()
def compute_tour_length(coords_b: torch.Tensor, tour_idx: torch.Tensor) -> float:
    pts = coords_b[tour_idx]  # [N,2]
    diff = pts[1:] - pts[:-1]
    seg_len = torch.sqrt((diff ** 2).sum(dim=-1))
    last_leg = torch.sqrt(((pts[0] - pts[-1]) ** 2).sum())
    return float((seg_len.sum() + last_leg).item())


# =========================================================
# 5. 评估函数（方案A：SubGap，对齐保留子图的 GT；严格无 opt）
# =========================================================

@torch.no_grad()
def evaluate_tsp_vae(
    model,
    dataset,
    batch_size=8,
    pe_dim=8,
    device=None
):
    if device is None:
        device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_p = all_r = all_f1 = all_acc = 0.0
    all_len_pred = []
    all_len_gt_keep = []
    all_gap_sub = []

    model.eval()
    for data in tqdm(loader, desc="Eval (greedy, no-opt)", dynamic_ncols=True):
        data = data.to(device)
        x_all, edge_index_all = data.x, data.edge_index
        edge_weight_all = data.edge_weight
        batch_vec = data.batch

        # dense 化
        x_dense, node_mask = to_dense_batch(x_all, batch_vec)  # [B,Nmax,2],[B,Nmax]
        B, Nmax, _ = x_dense.shape

        # 共享 PE
        pe_list = []
        for b in range(B):
            n_b = int(node_mask[b].sum())
            pe_b = sinusoidal_pe(n_b, pe_dim, device)
            pe_pad = torch.zeros(Nmax, pe_dim, device=device)
            pe_pad[:n_b] = pe_b
            pe_list.append(pe_pad)
        pe_dense = torch.stack(pe_list, dim=0)  # [B,Nmax,pe]

        # GT 邻接（边级指标）
        tour_edge_index_all = data.tour_edge_index
        adj_dense_gt = to_dense_adj(tour_edge_index_all, batch=batch_vec)  # [B,Nmax,Nmax]

        # 前向，拿到 keep_mask_dense
        logits, _, _, kv_mask, keep_mask_dense, perm, batch_pool = model(
            x_all, edge_index_all, batch_vec, pe_dense, node_mask, data.ptr, edge_weight_all
        )

        # ---- 边级指标：仅保留×保留上三角 ----
        triu_mask = torch.triu(torch.ones(Nmax, Nmax, device=device, dtype=torch.bool), diagonal=1)
        valid_pair_mask = keep_mask_dense.unsqueeze(2) & keep_mask_dense.unsqueeze(1) & triu_mask.unsqueeze(0)
        logits_u = logits[valid_pair_mask]
        target_u = adj_dense_gt[valid_pair_mask]
        prob_u = torch.sigmoid(logits_u)
        p, r, f1, acc = calc_metrics(prob_u, target_u)
        all_p += p; all_r += r; all_f1 += f1; all_acc += acc

        # ---- 路径评估（子图，纯贪心，无 opt）----
        coords_dense, _ = to_dense_batch(data.x, data.batch)  # [B,Nmax,2]

        for b in range(B):
            mask_b = keep_mask_dense[b]
            N_keep = int(mask_b.sum().item())
            if N_keep < 2:
                continue

            # 子图索引（dense）
            dense_ids_b = torch.arange(Nmax, device=mask_b.device)[mask_b]  # [N_keep]

            # 子图 logits、坐标
            logits_b_full = logits[b]
            logits_sub = logits_b_full.index_select(0, dense_ids_b).index_select(1, dense_ids_b)  # [N_keep,N_keep]
            coords_b_full = coords_dense[b]
            coords_b = coords_b_full.index_select(0, dense_ids_b).cpu()  # [N_keep,2]

            # 预测（纯贪心）
            tour_idx_sub = decode_tour_from_logits(
                logits_sub, torch.ones(N_keep, dtype=torch.bool, device=logits.device)
            )
            len_pred = compute_tour_length(coords_b, tour_idx_sub.cpu())

            # === 计算“GT在同一子集上的参考长度”（SubGT）===
            b_start = int(data.ptr[b].item())
            b_end   = int(data.ptr[b+1].item())
            tour_order_local = data.tour_order[b_start:b_end].detach().cpu()  # [N_b]

            global_ids_keep = (dense_ids_b + b_start).detach().cpu().tolist()
            keep_set = set(global_ids_keep)

            tour_order_keep_global = []
            for local_idx in tour_order_local.tolist():
                g = b_start + int(local_idx)
                if g in keep_set:
                    tour_order_keep_global.append(g)
            if len(tour_order_keep_global) < 2:
                continue

            dense_id_map = {int(g): k for k, g in enumerate(global_ids_keep)}
            tour_order_keep_dense = torch.tensor([dense_id_map[int(g)] for g in tour_order_keep_global],
                                                 dtype=torch.long)

            len_gt_keep = compute_tour_length(coords_b, tour_order_keep_dense)
            gap_sub = (len_pred - len_gt_keep) / (len_gt_keep + 1e-8)

            all_len_pred.append(float(len_pred))
            all_len_gt_keep.append(float(len_gt_keep))
            all_gap_sub.append(float(gap_sub))

    n_batches = max(1, len(loader))

    metrics = {
        "P": all_p / n_batches,
        "R": all_r / n_batches,
        "F1": all_f1 / n_batches,
        "Acc": all_acc / n_batches,
        "len_pred_mean": float(np.mean(all_len_pred)) if all_len_pred else float('nan'),
        "len_gt_keep_mean": float(np.mean(all_len_gt_keep)) if all_len_gt_keep else float('nan'),
        "gap_sub_mean": float(np.mean(all_gap_sub)) if all_gap_sub else float('nan'),
        "len_opt_mean": float(np.mean(all_len_gt_keep)) if all_len_gt_keep else float('nan'),
        "gap_mean": float(np.mean(all_gap_sub)) if all_gap_sub else float('nan'),
    }
    return metrics


# =========================================================
# 6. 训练函数（加入度约束损失 deg_loss）
# =========================================================

def train_tsp_vae(
    txt_path: str,
    epochs: int = 20,
    batch_size: int = 4,
    pe_dim: int = 8,
    hidden_dim: int = 64,
    latent_dim: int = 64,
    pool_ratio: float = 0.5,
    lr: float = 1e-3,
    neg_ratio: int = 3,      # 负样本数量 = neg_ratio * 正样本
    dist_bias_gamma: float = 1.0,  # 距离先验系数（可调 0.5~2.0）
    lambda_deg: float = 0.3        # 新增：度约束损失系数
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = TSPTxtDataset(txt_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    feat_dim = 2
    model = GraphVAEPooled(feat_dim, pe_dim, hidden_dim, latent_dim, pool_ratio,
                           dist_bias_gamma=dist_bias_gamma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    warmup_ratio = 0.3
    beta_max = 0.3
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        p_sum = r_sum = f1_sum = acc_sum = 0.0
        denom_loader = max(1, len(loader))

        warmup_epochs = max(1, int(epochs * warmup_ratio))
        progress = min(1.0, epoch / warmup_epochs)
        beta = beta_max * progress

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", dynamic_ncols=True, mininterval=0.5)
        for data in pbar:
            data = data.to(device)
            x_all, edge_index_all = data.x, data.edge_index
            edge_weight_all = data.edge_weight
            batch_vec = data.batch

            x_dense, node_mask = to_dense_batch(x_all, batch_vec)  # [B,Nmax,2]
            B, Nmax, _ = x_dense.shape

            # PE（Encoder & Decoder 共用）
            pe_list = []
            for b in range(B):
                n_b = int(node_mask[b].sum())
                pe_b = sinusoidal_pe(n_b, pe_dim, device)
                pe_pad = torch.zeros(Nmax, pe_dim, device=device)
                pe_pad[:n_b] = pe_b
                pe_list.append(pe_pad)
            pe_dense = torch.stack(pe_list, dim=0)  # [B,Nmax,pe]

            # GT 邻接
            tour_edge_index_all = data.tour_edge_index
            adj_dense_gt = to_dense_adj(tour_edge_index_all, batch=batch_vec)  # [B,Nmax,Nmax]

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.amp.autocast("cuda", enabled=torch.cuda.is_available())
            with autocast_ctx:
                # === 前向（Encoder/Decoder 升级已集成）===
                logits, mu_pool_d, logvar_pool_d, kv_mask, keep_mask_dense, perm, batch_pool = model(
                    x_all, edge_index_all, batch_vec, pe_dense, node_mask, data.ptr, edge_weight_all
                )

                # 只在保留子图上三角监督
                triu_mask = torch.triu(
                    torch.ones(Nmax, Nmax, device=device, dtype=torch.bool),
                    diagonal=1
                )
                valid_pair_mask = (
                    keep_mask_dense.unsqueeze(2) &
                    keep_mask_dense.unsqueeze(1) &
                    triu_mask.unsqueeze(0)
                )

                logits_u = logits[valid_pair_mask]
                target_u = adj_dense_gt[valid_pair_mask]

                # 正负均衡
                pos_mask = (target_u == 1)
                neg_mask = ~pos_mask
                pos_logits = logits_u[pos_mask]
                neg_logits = logits_u[neg_mask]

                num_pos = pos_logits.numel()
                if num_pos == 0:
                    continue
                num_neg = neg_logits.numel()
                k = min(num_neg, neg_ratio * num_pos)
                perm_idx = torch.randperm(num_neg, device=device)[:k]
                neg_logits_sample = neg_logits[perm_idx]

                pos_target = torch.ones_like(pos_logits)
                neg_target = torch.zeros_like(neg_logits_sample)
                logits_bal = torch.cat([pos_logits, neg_logits_sample], dim=0)
                target_bal = torch.cat([pos_target, neg_target], dim=0)

                recon = bce(logits_bal, target_bal)

                # ====== 新增：度约束损失 deg_loss ======
                eye = torch.eye(Nmax, device=device, dtype=torch.bool).unsqueeze(0)
                prob_full = torch.sigmoid(logits)                              # [B,Nmax,Nmax]
                prob_full = prob_full * keep_mask_dense.unsqueeze(2)
                prob_full = prob_full * keep_mask_dense.unsqueeze(1)
                prob_full = prob_full.masked_fill(eye, 0.0)

                deg_pred = prob_full.sum(dim=-1)                               # [B,Nmax]
                deg_target = 2.0 * keep_mask_dense.float()                     # [B,Nmax]
                mask_deg = keep_mask_dense

                if mask_deg.any():
                    deg_loss = F.mse_loss(
                        deg_pred[mask_deg],
                        deg_target[mask_deg],
                        reduction="mean"
                    )
                else:
                    deg_loss = torch.tensor(0.0, device=device)
                # =====================================

                # KL on pooled nodes
                mu_v = mu_pool_d[kv_mask]
                logvar_v = logvar_pool_d[kv_mask]
                if mu_v.numel() == 0:
                    kl = torch.tensor(0.0, device=device)
                else:
                    kl = kl_normal(mu_v, logvar_v)

                loss = recon + beta * kl + lambda_deg * deg_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_epoch += loss.item()

            # 训练日志指标
            with torch.no_grad():
                prob_u = torch.sigmoid(logits_u)
                p, r, f1, acc = calc_metrics(prob_u, target_u)
                p_sum += p; r_sum += r; f1_sum += f1; acc_sum += acc

            pbar.set_postfix(loss=float(loss.item()), kl=float(kl.item()), f1=float(f1))

        print(f"[Epoch {epoch:03d}] beta={beta:.3f} | "
              f"loss={total_loss_epoch/denom_loader:.4f} | "
              f"P={p_sum/denom_loader:.4f} R={r_sum/denom_loader:.4f} "
              f"F1={f1_sum/denom_loader:.4f} Acc={acc_sum/denom_loader:.4f}")

    # ===== 评估：严格无 opt =====
    print("\n=== Evaluation (greedy, no-opt, SubGap) ===")
    metrics = evaluate_tsp_vae(
        model,
        dataset,
        batch_size=8,
        pe_dim=pe_dim,
        device=device
    )
    print(f"Edge metrics: P={metrics['P']:.4f}  R={metrics['R']:.4f}  "
          f"F1={metrics['F1']:.4f}  Acc={metrics['Acc']:.4f}")
    print(f"[SubGraph|no-opt] Length (pred/gt_keep): {metrics['len_pred_mean']:.3f} / {metrics['len_gt_keep_mean']:.3f}")
    print(f"[SubGraph|no-opt] SubGap (mean): {metrics['gap_sub_mean']*100:.2f}%")

    return model, metrics


# =========================================================
# 7. main
# =========================================================

if __name__ == "__main__":
    # 这里改成你实际的 TSP-500 数据文件
    txt_path = "tsp500_train_lkh_100.txt"

    model, metrics = train_tsp_vae(
        txt_path,
        epochs=100,
        batch_size=4,
        pe_dim=8,
        hidden_dim=64,
        latent_dim=64,
        pool_ratio=0.5,
        lr=1e-3,
        neg_ratio=3,
        dist_bias_gamma=1.0,   # 可调 0.5~2.0
        lambda_deg=0.3         # 度约束损失权重，可试 0.1~0.5
    )
