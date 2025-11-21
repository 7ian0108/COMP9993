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
# 1.1 依据 TopKPooling 的选择构造 dense 保留掩码（关键改动）
# =========================================================
@torch.no_grad()
def build_keep_mask_dense(node_mask: torch.Tensor,
                          data_ptr: torch.Tensor,
                          perm: torch.Tensor,
                          batch_pool: torch.Tensor) -> torch.Tensor:
    """
    根据 TopKPooling 返回的 perm（全局 idx）与 batch_pool（所属图）
    在 dense 空间 [B, Nmax] 构造保留掩码。
    node_mask: [B, Nmax]  原始有效节点掩码
    data_ptr:  [B+1]      每个图在 x_all 里的全局起止偏移
    perm:      [K_all]    被保留的全局节点索引
    batch_pool:[K_all]    这些保留节点所属的图编号
    -> keep_mask_dense: [B, Nmax] True 表示该 dense 位置属于被保留节点
    """
    B, Nmax = node_mask.size()
    keep_mask = torch.zeros_like(node_mask, dtype=torch.bool)
    for b in range(B):
        start = int(data_ptr[b].item())
        # end = int(data_ptr[b+1].item())  # 如需校验范围可启用
        keep_global_b = perm[(batch_pool == b)]
        if keep_global_b.numel() == 0:
            continue
        dense_j = (keep_global_b - start).clamp(min=0, max=Nmax-1)
        valid = node_mask[b]
        keep_mask[b, dense_j] = True
        keep_mask[b] &= valid
    return keep_mask


# =========================================================
# 2. GraphVAE 模型（保持你的框架 & 同一 SinCos PE）
# =========================================================

class GCNEncoderWithPool(nn.Module):
    def __init__(self, in_dim: int, pe_dim: int, hidden_dim: int, latent_dim: int, pool_ratio: float = 0.5):
        super().__init__()
        self.gc1 = GCNConv(in_dim + pe_dim, hidden_dim)
        self.gc_mu = GCNConv(hidden_dim, latent_dim)
        self.gc_logvar = GCNConv(hidden_dim, latent_dim)
        self.pool = TopKPooling(latent_dim, ratio=pool_ratio)

    def forward(self, x_raw, sincos_pe, edge_index, batch):
        # x_raw: [sumN, F]
        # sincos_pe: [sumN, pe_dim]（和 x_raw 一一对应）
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
        return z_pool, mu_pool, logvar_pool, batch_pool, perm   # 返回 perm


class TransformerDecoderCross(nn.Module):
    def __init__(self, latent_dim: int, pe_dim: int = 8, num_heads: int = 4, num_layers: int = 2, feat_dim: int = 2):
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

    def forward(self, mask_nodes, sincos_pe, h_pool, node_mask, kv_mask):
        """
        mask_nodes: [B, Nmax, F]  这里传 coords
        sincos_pe:  [B, Nmax, pe]
        h_pool:     [B, Kmax, D]
        node_mask:  [B, Nmax]  True 表示**tgt 端有效节点**
        kv_mask:    [B, Kmax]  True 表示**memory 端有效节点**
        """
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
        pair = torch.cat([hi, hj], dim=-1)                        # [B,Nmax,Nmax,2D]
        logits = self.edge_mlp(pair).squeeze(-1)                  # [B,Nmax,Nmax]

        eye = torch.eye(Nmax, device=logits.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(eye, float('-inf'))
        return logits


class GraphVAEPooled(nn.Module):
    def __init__(self, feat_dim, pe_dim, hidden_dim, latent_dim, pool_ratio=0.5):
        super().__init__()
        self.pe_dim = pe_dim
        self.feat_dim = feat_dim
        self.encoder = GCNEncoderWithPool(feat_dim, pe_dim, hidden_dim, latent_dim, pool_ratio)
        self.decoder = TransformerDecoderCross(latent_dim, pe_dim=pe_dim, feat_dim=feat_dim)

    def forward(self, x_all, edge_index_all, batch_vec, pe_dense, node_mask, data_ptr):
        """
        x_all:   [sumN, F]
        edge_index_all: [2, E]
        batch_vec: [sumN]
        pe_dense: [B, Nmax, pe]  # 同一 SinCos PE 用于 Encoder & Decoder
        node_mask: [B, Nmax]     # 原始有效节点
        data_ptr: [B+1]          # 每个图在 x_all 中的起止（来自 batch.ptr）
        """
        B, Nmax = node_mask.size()

        # Encoder 的 SinCos：从 dense PE 拉平后，根据 node_mask 对齐到 x_all
        pe_flat = pe_dense.view(-1, self.pe_dim)[node_mask.view(-1)]   # [sumN_valid, pe_dim]

        # === Encoder ===
        z_pool, mu_pool, logvar_pool, batch_pool, perm = self.encoder(
            x_all, pe_flat, edge_index_all, batch_vec
        )

        # 池化后 dense
        h_pool, kv_mask = to_dense_batch(z_pool, batch_pool)
        mu_pool_d, _ = to_dense_batch(mu_pool, batch_pool)
        logvar_pool_d, _ = to_dense_batch(logvar_pool, batch_pool)

        # === 构造 keep_mask_dense：只在被保留节点上进行解码与监督 ===
        keep_mask_dense = build_keep_mask_dense(node_mask, data_ptr, perm, batch_pool)

        # Decoder 的节点特征：用 coords（dense）
        x_dense, _ = to_dense_batch(x_all, batch_vec)  # [B, Nmax, F]
        mask_nodes = x_dense

        # Decoder 用的 SinCos：直接用同一个 pe_dense
        logits = self.decoder(mask_nodes, pe_dense, h_pool, keep_mask_dense, kv_mask)

        # 返回 keep_mask_dense / perm / batch_pool 以便外层评估/日志
        return logits, mu_pool_d, logvar_pool_d, kv_mask, keep_mask_dense, perm, batch_pool


# =========================================================
# 3. TSP txt → PyG Dataset  （新增 tour_order 保存 GT 巡回顺序）
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

            # 原图：完全图（无向）
            row, col = torch.triu_indices(N, N, offset=1)
            edge_index_full = torch.cat(
                [torch.stack([row, col], dim=0),
                 torch.stack([col, row], dim=0)],
                dim=1
            )

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
                tour_edge_index=tour_edge_undir,
                length_opt=length_opt,
                tour_order=tour0.clone()   # NEW: 保存 GT 巡回顺序（0-based, len=N）
            )
            self.data_list.append(data)

        print(f"Loaded {len(self.data_list)} TSP instances from {txt_path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# =========================================================
# 4. 解 tour + 2-opt + length / gap（解码保留子图）
# =========================================================

@torch.no_grad()
def decode_tour_from_logits(logits_b: torch.Tensor,
                            node_mask_b: torch.Tensor) -> torch.Tensor:
    """
    logits_b:   [Nmax,Nmax] （此处传子图 logits）
    node_mask_b:[Nmax]      True 表示有效（子图里可用）
    返回一个 0-based 的 tour 索引 [N]
    """
    N = int(node_mask_b.sum().item())
    logits_N = logits_b[:N, :N]

    prob = torch.sigmoid(logits_N)
    # 无向图，对称化
    prob = (prob + prob.t()) / 2.0
    # 不允许自环
    prob = prob - torch.eye(N, device=prob.device) * 1e9

    start = 0
    tour = [start]
    used = set([start])

    for _ in range(N - 1):
        i = tour[-1]
        cand = prob[i].clone()
        cand[list(used)] = -1e9
        j = int(cand.argmax().item())
        tour.append(j)
        used.add(j)

    return torch.tensor(tour, device=logits_b.device, dtype=torch.long)


@torch.no_grad()
def compute_tour_length(coords_b: torch.Tensor, tour_idx: torch.Tensor) -> float:
    pts = coords_b[tour_idx]  # [N,2]
    diff = pts[1:] - pts[:-1]
    seg_len = torch.sqrt((diff ** 2).sum(dim=-1))
    last_leg = torch.sqrt(((pts[0] - pts[-1]) ** 2).sum())
    return float((seg_len.sum() + last_leg).item())


@torch.no_grad()
def two_opt(coords: torch.Tensor, tour: torch.Tensor, max_iters: int = 1) -> Tuple[torch.Tensor, float]:
    """
    简单 2-opt，本地优化 tour。
    coords: [N,2] (cpu)
    tour:   [N]   (cpu, 0-based)
    max_iters: 迭代轮数，越大越慢
    """
    N = tour.size(0)
    best = tour.clone()
    best_len = compute_tour_length(coords, best)

    def dist(i, j):
        p = coords[i]
        q = coords[j]
        return float(torch.sqrt(((p - q) ** 2).sum()).item())

    for _ in range(max_iters):
        improved = False
        for i in range(1, N - 2):
            a = best[i - 1].item()
            b = best[i].item()
            for j in range(i + 1, N - 1):
                c = best[j].item()
                d = best[(j + 1) % N].item()

                old = dist(a, b) + dist(c, d)
                new = dist(a, c) + dist(b, d)
                if new + 1e-9 < old:
                    # 反转 i..j 段
                    best[i:j+1] = torch.flip(best[i:j+1], dims=[0])
                    best_len = best_len - old + new
                    improved = True
        if not improved:
            break

    return best, best_len


# =========================================================
# 5. 评估函数（方案A：SubGap，对齐保留子图的 GT）
# =========================================================

@torch.no_grad()
def evaluate_tsp_vae(
    model,
    dataset,
    batch_size=8,
    pe_dim=8,
    device=None,
    use_2opt: bool = False,
    max_2opt_iters: int = 1
):
    """
    方案A（SubGap）：仅在 Encoder 保留的节点子图上解码，并与“GT巡回顺序过滤到同一子集”的长度做对照。
    为兼容你的原打印逻辑：
      - len_opt_mean 将返回 len_gt_keep_mean（子图口径下的对照长度均值）
      - gap_mean     将返回 gap_sub_mean  （子图口径下的 SubGap 均值）
    """
    if device is None:
        device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_p = all_r = all_f1 = all_acc = 0.0
    all_len_pred = []
    all_len_gt_keep = []
    all_gap_sub = []

    model.eval()
    desc = "Eval (greedy)" if not use_2opt else "Eval (+2-opt)"
    for data in tqdm(loader, desc=desc, dynamic_ncols=True):
        data = data.to(device)
        x_all, edge_index_all = data.x, data.edge_index
        batch_vec = data.batch

        # dense 化
        x_dense, node_mask = to_dense_batch(x_all, batch_vec)  # [B,Nmax,2],[B,Nmax]
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

        # GT 邻接（tour 子图，用于边级指标）
        tour_edge_index_all = data.tour_edge_index
        adj_dense_gt = to_dense_adj(tour_edge_index_all, batch=batch_vec)  # [B,Nmax,Nmax]

        # ===== 前向：内部会根据 TopKPooling 计算 keep_mask_dense（关键） =====
        logits, _, _, kv_mask, keep_mask_dense, perm, batch_pool = model(
            x_all, edge_index_all, batch_vec, pe_dense, node_mask, data.ptr
        )

        # 边级指标：只在“保留节点 × 保留节点”的上三角对上统计
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
        prob_u = torch.sigmoid(logits_u)
        p, r, f1, acc = calc_metrics(prob_u, target_u)
        all_p += p; all_r += r; all_f1 += f1; all_acc += acc

        # 路径评估（SubGap）：在“保留子图”上解码，并与“GT巡回顺序过滤到同一子集”的长度对比
        coords_dense, _ = to_dense_batch(data.x, data.batch)  # [B,Nmax,2]

        for b in range(B):
            mask_b = keep_mask_dense[b]        # [Nmax]，True 表示该 dense 位置被保留
            N_keep = int(mask_b.sum().item())
            if N_keep < 2:
                continue  # 极端：保留节点太少，跳过

            # 子图索引（dense 序号）
            dense_ids_b = torch.arange(Nmax, device=mask_b.device)[mask_b]  # [N_keep]

            # 子图 logits
            logits_b_full = logits[b]  # [Nmax,Nmax]
            logits_sub = logits_b_full.index_select(0, dense_ids_b).index_select(1, dense_ids_b)  # [N_keep,N_keep]

            # 子图坐标
            coords_b_full = coords_dense[b]    # [Nmax,2]
            coords_b = coords_b_full.index_select(0, dense_ids_b).cpu()  # [N_keep,2]

            # === 解码（子图）===
            tour_idx_sub = decode_tour_from_logits(
                logits_sub, torch.ones(N_keep, dtype=torch.bool, device=logits.device)
            )
            tour_idx_sub_cpu = tour_idx_sub.cpu()
            len_pred = compute_tour_length(coords_b, tour_idx_sub_cpu)

            # 可选：子图上再跑 2-opt
            if use_2opt:
                tour_best, len_best = two_opt(coords_b, tour_idx_sub_cpu, max_iters=max_2opt_iters)
                len_pred = len_best

            # === 计算“GT在同一子集上的参考长度”（SubGT）===
            # 根据 ptr 截出本图的 tour_order（局部 0..N_b-1）
            b_start = int(data.ptr[b].item())
            b_end   = int(data.ptr[b+1].item())
            tour_order_local = data.tour_order[b_start:b_end].detach().cpu()  # [N_b]

            # dense -> global 映射（dense位置 + b_start）
            global_ids_keep = (dense_ids_b + b_start).detach().cpu().tolist()
            keep_set = set(global_ids_keep)

            # 将 GT 的局部索引转为全局索引，再按“是否在保留集合中”过滤
            tour_order_keep_global = []
            for local_idx in tour_order_local.tolist():
                g = b_start + int(local_idx)
                if g in keep_set:
                    tour_order_keep_global.append(g)
            if len(tour_order_keep_global) < 2:
                continue

            # 建立 全局ID -> 子图denseID 的字典，映射到子图的 0..N_keep-1 序号
            dense_id_map = {int(g): k for k, g in enumerate(global_ids_keep)}
            tour_order_keep_dense = torch.tensor([dense_id_map[int(g)] for g in tour_order_keep_global],
                                                 dtype=torch.long)

            # 计算 GT 子图长度（可选：也可对 GT 子序列跑 2-opt 形成更紧上界）
            len_gt_keep = compute_tour_length(coords_b, tour_order_keep_dense)

            # SubGap
            gap_sub = (len_pred - len_gt_keep) / (len_gt_keep + 1e-8)

            all_len_pred.append(float(len_pred))
            all_len_gt_keep.append(float(len_gt_keep))
            all_gap_sub.append(float(gap_sub))

    n_batches = max(1, len(loader))

    # 组装指标（兼容旧字段名 + 新字段名）
    metrics = {
        "P": all_p / n_batches,
        "R": all_r / n_batches,
        "F1": all_f1 / n_batches,
        "Acc": all_acc / n_batches,

        # 新字段（子图口径）
        "len_pred_mean": float(np.mean(all_len_pred)) if all_len_pred else float('nan'),
        "len_gt_keep_mean": float(np.mean(all_len_gt_keep)) if all_len_gt_keep else float('nan'),
        "gap_sub_mean": float(np.mean(all_gap_sub)) if all_gap_sub else float('nan'),

        # 兼容旧打印（把“子图GT”复用到旧字段名上）
        "len_opt_mean": float(np.mean(all_len_gt_keep)) if all_len_gt_keep else float('nan'),
        "gap_mean": float(np.mean(all_gap_sub)) if all_gap_sub else float('nan'),
    }
    return metrics


# =========================================================
# 6. 训练函数（KL warm-up 到 0.3，默认 neg_ratio=3；监督对齐保留节点）
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
    neg_ratio: int = 3      # 负样本数量 = neg_ratio * 正样本（推荐 2~3）
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = TSPTxtDataset(txt_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    feat_dim = 2
    model = GraphVAEPooled(feat_dim, pe_dim, hidden_dim, latent_dim, pool_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())  # 新写法

    # KL warm-up：0 -> beta_max
    warmup_ratio = 0.3
    beta_max = 0.3
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        p_sum = r_sum = f1_sum = acc_sum = 0.0
        denom_loader = max(1, len(loader))

        # KL warm-up 进度
        warmup_epochs = max(1, int(epochs * warmup_ratio))
        progress = min(1.0, epoch / warmup_epochs)
        beta = beta_max * progress

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", dynamic_ncols=True, mininterval=0.5)
        for data in pbar:
            data = data.to(device)
            x_all, edge_index_all = data.x, data.edge_index
            batch_vec = data.batch

            x_dense, node_mask = to_dense_batch(x_all, batch_vec)  # [B,Nmax,2]
            B, Nmax, _ = x_dense.shape

            # PE（Encoder & Decoder 共用同一 SinCos）
            pe_list = []
            for b in range(B):
                n_b = int(node_mask[b].sum())
                pe_b = sinusoidal_pe(n_b, pe_dim, device)
                pe_pad = torch.zeros(Nmax, pe_dim, device=device)
                pe_pad[:n_b] = pe_b
                pe_list.append(pe_pad)
            pe_dense = torch.stack(pe_list, dim=0)  # [B,Nmax,pe]

            # GT tour 邻接
            tour_edge_index_all = data.tour_edge_index
            adj_dense_gt = to_dense_adj(tour_edge_index_all, batch=batch_vec)  # [B,Nmax,Nmax]

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.amp.autocast("cuda", enabled=torch.cuda.is_available())
            with autocast_ctx:
                # === 前向：内部完成 TopK 与 keep_mask_dense 一致化（关键改动）===
                logits, mu_pool_d, logvar_pool_d, kv_mask, keep_mask_dense, perm, batch_pool = model(
                    x_all, edge_index_all, batch_vec, pe_dense, node_mask, data.ptr
                )

                # 只在保留节点组成的上三角对上监督
                triu_mask = torch.triu(
                    torch.ones(Nmax, Nmax, device=device, dtype=torch.bool),
                    diagonal=1
                )
                valid_pair_mask = (
                    keep_mask_dense.unsqueeze(2) &
                    keep_mask_dense.unsqueeze(1) &
                    triu_mask.unsqueeze(0)
                )

                logits_u = logits[valid_pair_mask]           # [M]
                target_u = adj_dense_gt[valid_pair_mask]     # [M]

                # ===== 1:neg_ratio 正负采样 =====
                pos_mask = (target_u == 1)
                neg_mask = ~pos_mask

                pos_logits = logits_u[pos_mask]
                neg_logits = logits_u[neg_mask]

                num_pos = pos_logits.numel()
                if num_pos == 0:
                    continue  # 极端情况：该 batch 恰好没有正样本

                num_neg = neg_logits.numel()
                k = min(num_neg, neg_ratio * num_pos)
                perm_idx = torch.randperm(num_neg, device=device)[:k]
                neg_logits_sample = neg_logits[perm_idx]

                pos_target = torch.ones_like(pos_logits)
                neg_target = torch.zeros_like(neg_logits_sample)

                logits_bal = torch.cat([pos_logits, neg_logits_sample], dim=0)
                target_bal = torch.cat([pos_target, neg_target], dim=0)

                recon = bce(logits_bal, target_bal)

                # KL on pooled nodes（与原逻辑一致）
                mu_v = mu_pool_d[kv_mask]
                logvar_v = logvar_pool_d[kv_mask]
                if mu_v.numel() == 0:
                    kl = torch.tensor(0.0, device=device)
                else:
                    kl = kl_normal(mu_v, logvar_v)

                loss = recon + beta * kl

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_epoch += loss.item()

            # 指标：在保留子图的全体边上算
            with torch.no_grad():
                prob_u = torch.sigmoid(logits_u)
                p, r, f1, acc = calc_metrics(prob_u, target_u)
                p_sum += p; r_sum += r; f1_sum += f1; acc_sum += acc

            pbar.set_postfix(loss=float(loss.item()), kl=float(kl.item()), f1=float(f1))

        print(f"[Epoch {epoch:03d}] beta={beta:.3f} | "
              f"loss={total_loss_epoch/denom_loader:.4f} | "
              f"P={p_sum/denom_loader:.4f} R={r_sum/denom_loader:.4f} "
              f"F1={f1_sum/denom_loader:.4f} Acc={acc_sum/denom_loader:.4f}")

    # 训练结束后评估：先用 greedy 快速评估，再选用 2-opt 精评
    print("\n=== Evaluation (greedy, fast, SubGap) ===")
    metrics_greedy = evaluate_tsp_vae(
        model,
        dataset,
        batch_size=8,
        pe_dim=pe_dim,
        device=device,
        use_2opt=False
    )
    print(f"Edge metrics: P={metrics_greedy['P']:.4f}  R={metrics_greedy['R']:.4f}  "
          f"F1={metrics_greedy['F1']:.4f}  Acc={metrics_greedy['Acc']:.4f}")
    print(f"[SubGraph] Length (pred/gt_keep): {metrics_greedy['len_pred_mean']:.3f} / {metrics_greedy['len_gt_keep_mean']:.3f}")
    print(f"[SubGraph] SubGap (mean): {metrics_greedy['gap_sub_mean']*100:.2f}%")

    print("\n=== Evaluation (+2-opt, SubGap) ===")
    metrics_2opt = evaluate_tsp_vae(
        model,
        dataset,
        batch_size=4,      # 2-opt 慢一点，可以适当减小 batch_size
        pe_dim=pe_dim,
        device=device,
        use_2opt=True,
        max_2opt_iters=1   # 先用 1 轮 2-opt，想更猛可以改 2
    )
    print(f"Edge metrics: P={metrics_2opt['P']:.4f}  R={metrics_2opt['R']:.4f}  "
          f"F1={metrics_2opt['F1']:.4f}  Acc={metrics_2opt['Acc']:.4f}")
    print(f"[SubGraph] Length (pred/gt_keep): {metrics_2opt['len_pred_mean']:.3f} / {metrics_2opt['len_gt_keep_mean']:.3f}")
    print(f"[SubGraph] SubGap (mean): {metrics_2opt['gap_sub_mean']*100:.2f}%")

    return model, metrics_greedy, metrics_2opt


# =========================================================
# 7. main
# =========================================================

if __name__ == "__main__":
    # 这里改成你实际的 TSP-500 数据文件
    # 比如 "tsp500_train_lkh_500.txt" 或 "tsp500_train_lkh_100.txt"
    txt_path = "tsp500_train_lkh_100.txt"

    model, metrics_greedy, metrics_2opt = train_tsp_vae(
        txt_path,
        epochs=200,          # 可以先跑 10~20 看趋势
        batch_size=4,       # TSP-500 比较大，不要太大 batch
        pe_dim=8,
        hidden_dim=64,
        latent_dim=64,
        pool_ratio=0.5,
        lr=1e-3,
        neg_ratio=3        # 推荐 2 或 3，提高 precision
    )
