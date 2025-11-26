import os
import math
import re
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



class ConcordeTSPDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path: str, max_instances: int = None):
        assert os.path.isfile(txt_path), f"Dataset file not found: {txt_path}"
        fname = os.path.basename(txt_path)
        m = re.search(r"tsp(\d+)", fname)
        if m is None:
            raise ValueError(f"Cannot parse TSP size N from filename: {fname}")
        self.N = int(m.group(1))
        print(f"[Data] Parsed TSP size N={self.N} from '{fname}'")
        self.data_list = self._load_file(txt_path, max_instances)

    def _load_file(self, txt_path: str, max_instances: int):
        data_list = []
        N = self.N

        with open(txt_path, "r") as f:
            for line_id, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                L = len(parts)
                if L < 2 * N + N:
                    raise ValueError(
                        f"Line {line_id} too short: len={L}, need >= {2*N+N}"
                    )


                coords_flat = list(map(float, parts[: 2 * N]))
                coords = np.asarray(coords_flat, dtype=np.float32).reshape(N, 2)


                tour_str = parts[-N:]
                tour = np.asarray(tour_str, dtype=int)
                if tour.max() >= N: 
                    tour = tour - 1
                    
                rows, cols = [], []
                for i in range(N):
                    for j in range(N):
                        if i == j:
                            continue
                        rows.append(i)
                        cols.append(j)
                rows = np.asarray(rows, dtype=np.int64)
                cols = np.asarray(cols, dtype=np.int64)

                coord_i = coords[rows]
                coord_j = coords[cols]
                dists = np.linalg.norm(coord_i - coord_j, axis=1, keepdims=True).astype(
                    np.float32
                )


                tour_edges = set()
                for k in range(N):
                    u = int(tour[k])
                    v = int(tour[(k + 1) % N])
                    tour_edges.add((u, v))
                    tour_edges.add((v, u))

                y = np.zeros(rows.shape[0], dtype=np.float32)
                for eid, (u, v) in enumerate(zip(rows, cols)):
                    if (int(u), int(v)) in tour_edges:
                        y[eid] = 1.0

                data = Data(
                    x=torch.from_numpy(coords),  # [N,2]
                    edge_index=torch.from_numpy(np.stack([rows, cols], axis=0)),
                    edge_attr=torch.from_numpy(dists),  # [E,1]
                    y=torch.from_numpy(y),  # [E]
                    n_nodes=N,
                )
                data_list.append(data)

                if max_instances is not None and len(data_list) >= max_instances:
                    break

        print(
            f"[Data] Loaded {len(data_list)} TSP instances from {txt_path}, "
            f"N (per graph) = {data_list[0].n_nodes}"
        )
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def sinusoidal_pe(num_nodes: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(num_nodes, dim, device=device)
    position = torch.arange(0, num_nodes, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [N, dim]


class SinTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / half)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


@torch.no_grad()
def edge_metrics_from_logits(
    logits: torch.Tensor, target_sym: torch.Tensor
) -> Tuple[float, float, float, float]:
    """
    logits: [B,N,N]   VAE 解码出的邻接 logits
    target_sym: [B,N,N] 对称 0/1 邻接（无向 tour）
    """
    B, N, _ = logits.shape
    device = logits.device
    triu_mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    log_u = logits[:, triu_mask].reshape(-1)
    tgt_u = target_sym[:, triu_mask].reshape(-1)

    prob = torch.sigmoid(log_u)
    pred = (prob > 0.5).float()

    tp = (pred * tgt_u).sum()
    fp = (pred * (1 - tgt_u)).sum()
    fn = ((1 - pred) * tgt_u).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    acc = (pred == tgt_u).float().mean()

    return precision.item(), recall.item(), f1.item(), acc.item()


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def recover_tour_from_adj(A_sym: torch.Tensor) -> List[int]:
    """
    根据真实 tour 的对称邻接矩阵（0/1）恢复一个结点顺序。
    假设每个点度数=2（TSP tour）。
    """
    N = A_sym.size(0)
    neighbors = {i: [] for i in range(N)}
    for i in range(N):
        for j in range(N):
            if i != j and A_sym[i, j] > 0.5:
                neighbors[i].append(j)

    if any(len(nb) != 2 for nb in neighbors.values()):
        prob_mat = A_sym
        return nearest_neighbor_tour(prob_mat)

    tour = [0]
    prev = -1
    cur = 0
    for _ in range(N - 1):
        nb = neighbors[cur]
        nxt = nb[0] if nb[0] != prev else nb[1]
        tour.append(nxt)
        prev, cur = cur, nxt
    return tour


def nearest_neighbor_tour(prob_mat: torch.Tensor) -> List[int]:
    N = prob_mat.size(0)
    visited = [False] * N
    tour = [0]
    visited[0] = True
    cur = 0
    for _ in range(N - 1):
        scores = prob_mat[cur].clone()
        mask = torch.tensor(visited, device=prob_mat.device)
        scores[mask] = -1e9
        nxt = int(scores.argmax().item())
        tour.append(nxt)
        visited[nxt] = True
        cur = nxt
    return tour  


def tour_length(coords: torch.Tensor, tour: List[int]) -> float:

    N = len(tour)
    length = 0.0
    for i in range(N):
        u = tour[i]
        v = tour[(i + 1) % N]
        length += torch.norm(coords[u] - coords[v]).item()
    return float(length)


class GraphVAEEncoder(nn.Module):
    def __init__(self, node_in_dim: int, pe_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.pe_dim = pe_dim
        self.gcn1 = GCNConv(node_in_dim + pe_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_weight, batch, pe_dense):

        B, N, P = pe_dense.shape
        pe_flat = pe_dense.view(B * N, P) 

        x_in = torch.cat([x, pe_flat], dim=-1)  # [sumN, 2+pe]
        h = F.relu(self.gcn1(x_in, edge_index, edge_weight=edge_weight))
        h = F.relu(self.gcn2(h, edge_index, edge_weight=edge_weight))

        h_graph = global_mean_pool(h, batch)  # [B, hidden_dim]

        mu = self.fc_mu(h_graph)
        logvar = self.fc_logvar(h_graph)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class GraphVAEDecoder(nn.Module):
    def __init__(self, pe_dim: int, latent_dim: int, hidden_dim: int, coord_dim: int = 2):
        super().__init__()
        self.pe_dim = pe_dim
        self.latent_to_node = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pe_proj = nn.Linear(pe_dim, hidden_dim)
        self.coord_proj = nn.Linear(coord_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, pe_dense, coord_dense):
       
        B, N, P = pe_dense.shape
        device = pe_dense.device

        z_node = self.latent_to_node(z).unsqueeze(1).expand(B, N, -1)  # [B,N,H]
        pe_emb = self.pe_proj(pe_dense)                                # [B,N,H]
        coord_emb = self.coord_proj(coord_dense)                       # [B,N,H]

        node_repr = z_node + pe_emb + coord_emb                        # [B,N,H]

        hi = node_repr.unsqueeze(2).expand(B, N, N, -1)
        hj = node_repr.unsqueeze(1).expand(B, N, N, -1)
        pair = torch.cat([hi, hj], dim=-1)                             # [B,N,N,2H]

        logits = self.edge_mlp(pair).squeeze(-1)                       # [B,N,N]
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(eye, float("-inf"))
        return logits


class GraphVAE(nn.Module):
    def __init__(self, node_in_dim=2, pe_dim=8, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.pe_dim = pe_dim
        self.encoder = GraphVAEEncoder(node_in_dim, pe_dim, hidden_dim, latent_dim)
        self.decoder = GraphVAEDecoder(pe_dim, latent_dim, hidden_dim, coord_dim=2)

    def forward(self, data, pe_dense, coord_dense):
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr.squeeze(-1)
        batch = data.batch

        z, mu, logvar = self.encoder(x, edge_index, edge_weight, batch, pe_dense)
        logits_A = self.decoder(z, pe_dense, coord_dense)
        return logits_A, mu, logvar, z


class GaussianDDPMScheduleLatent:
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, z0: torch.Tensor, t_graph: torch.Tensor):
        eps = torch.randn_like(z0)
        alpha_bar_t = self.alpha_bar[t_graph]  # [B]
        sqrt_ab = torch.sqrt(alpha_bar_t).unsqueeze(-1)
        sqrt_1m = torch.sqrt(1.0 - alpha_bar_t).unsqueeze(-1)
        z_t = sqrt_ab * z0 + sqrt_1m * eps
        return z_t, eps


class LatentDenoiseNet(nn.Module):
    def __init__(self, latent_dim=64, time_dim=64, hidden_dim=256):
        super().__init__()
        self.time_embed = SinTimeEmbed(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t, t_graph):
        t_emb = self.time_embed(t_graph)
        h = torch.cat([z_t, t_emb], dim=-1)
        return self.mlp(h)


@torch.no_grad()
def sample_z0_prior_ddim(denoise_net, diff_sched, batch_size, latent_dim):
    device = next(denoise_net.parameters()).device
    T = diff_sched.T
    z_t = torch.randn(batch_size, latent_dim, device=device)

    for t in reversed(range(T)):
        t_graph = torch.full((batch_size,), t, device=device, dtype=torch.long)
        z0_hat = denoise_net(z_t, t_graph)  

        alpha_bar_t = diff_sched.alpha_bar[t]  # scalar
        if t > 0:
            alpha_bar_prev = diff_sched.alpha_bar[t - 1]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        alpha_bar_t_sqrt = torch.sqrt(alpha_bar_t)
        one_minus_t_sqrt = torch.sqrt(1.0 - alpha_bar_t + 1e-8)

        eps_hat = (z_t - alpha_bar_t_sqrt * z0_hat) / one_minus_t_sqrt
        z_t = torch.sqrt(alpha_bar_prev) * z0_hat + torch.sqrt(
            1.0 - alpha_bar_prev
        ) * eps_hat

    return z_t  



def train_model(
    data_path: str = "./data/tsp/tsp20_train_concorde.txt",
    max_instances: int = 2000,
    epochs: int = 30,
    batch_size: int = 64,
    pe_dim: int = 8,
    hidden_dim: int = 128,
    latent_dim: int = 64,
    lr_vae: float = 1e-3,
    lr_diff: float = 1e-3,
    T_latent: int = 100,
    beta_kl: float = 0.1,
    lambda_diff: float = 0.5,
    pretrain_epochs: int = 10,  
):
    dataset = ConcordeTSPDataset(data_path, max_instances=max_instances)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    N = dataset.N

    vae = GraphVAE(
        node_in_dim=2, pe_dim=pe_dim, hidden_dim=hidden_dim, latent_dim=latent_dim
    ).to(device)
    diff_sched = GaussianDDPMScheduleLatent(T=T_latent, device=device)
    denoise_net = LatentDenoiseNet(
        latent_dim=latent_dim, time_dim=64, hidden_dim=256
    ).to(device)

    opt_vae = torch.optim.Adam(vae.parameters(), lr=lr_vae)
    opt_diff = torch.optim.Adam(denoise_net.parameters(), lr=lr_diff)

    for epoch in range(1, epochs + 1):
        vae.train()
        denoise_net.train()

        total_vae_loss = 0.0
        total_diff_loss = 0.0
        p_sum = r_sum = f1_sum = acc_sum = 0.0
        denom = max(1, len(loader))

        if epoch <= pretrain_epochs:
            lambda_diff_eff = 0.0
        else:
            lambda_diff_eff = lambda_diff

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", mininterval=0.5)
        for batch_data in pbar:
            batch_data = batch_data.to(device)
            x = batch_data.x
            edge_index = batch_data.edge_index
            y_edge = batch_data.y
            batch_vec = batch_data.batch
            B = batch_data.num_graphs

      
            adj_dense = to_dense_adj(
                edge_index, batch=batch_vec, edge_attr=y_edge.unsqueeze(-1)
            ).squeeze(1)  # [B,N,N]
            A_sym = ((adj_dense + adj_dense.transpose(1, 2)) > 0).float()

       
            pe_list = [sinusoidal_pe(N, pe_dim, device) for _ in range(B)]
            pe_dense = torch.stack(pe_list, dim=0)  # [B,N,pe]

 
            x_dense, _ = to_dense_batch(x, batch_vec)  # [B,N,2]


            logits_A, mu, logvar, z0 = vae(batch_data, pe_dense, x_dense)  # logits_A: [B,N,N]

            triu_mask = torch.triu(
                torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1
            )
            logits_u = logits_A[:, triu_mask].reshape(-1)
            target_u = A_sym[:, triu_mask].reshape(-1)


            pos = target_u.sum()
            total = target_u.numel()
            neg = total - pos
            if pos.item() < 0.5:
                pos_weight = torch.tensor(1.0, device=device)
            else:
                pos_weight = torch.clamp(neg / (pos + 1e-6), 1.0, 20.0)

            recon_loss = F.binary_cross_entropy_with_logits(
                logits_u, target_u, pos_weight=pos_weight
            )
            kl = kl_normal(mu, logvar)
            vae_loss = recon_loss + beta_kl * kl

 
            z0_det = z0.detach()
            t_graph = torch.randint(
                0, diff_sched.T, (B,), device=device, dtype=torch.long
            )
            z_t, _ = diff_sched.q_sample(z0_det, t_graph)
            z0_hat = denoise_net(z_t, t_graph)
            diff_loss = F.mse_loss(z0_hat, z0_det)

            loss = vae_loss + lambda_diff_eff * diff_loss

            opt_vae.zero_grad()
            opt_diff.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(denoise_net.parameters(), 1.0)
            opt_vae.step()
            opt_diff.step()

            total_vae_loss += vae_loss.item()
            total_diff_loss += diff_loss.item()

            with torch.no_grad():
                p, r, f1, acc = edge_metrics_from_logits(logits_A, A_sym)
                p_sum += p
                r_sum += r
                f1_sum += f1
                acc_sum += acc

            pbar.set_postfix(
                vae_loss=float(vae_loss.item()),
                diff_loss=float(diff_loss.item()),
                f1=float(f1),
                lam=float(lambda_diff_eff),
            )

        print(
            f"[Epoch {epoch:03d}] "
            f"VAE_loss={total_vae_loss/denom:.4f} | Diff_loss={total_diff_loss/denom:.4f} | "
            f"P={p_sum/denom:.4f} R={r_sum/denom:.4f} F1={f1_sum/denom:.4f} Acc={acc_sum/denom:.4f}"
        )

    return vae, denoise_net, diff_sched, dataset


@torch.no_grad()
def evaluate_sampling(
    vae: GraphVAE,
    denoise_net: LatentDenoiseNet,
    diff_sched: GaussianDDPMScheduleLatent,
    dataset: ConcordeTSPDataset,
    pe_dim: int,
    latent_dim: int,
    num_graphs: int = 200,
):

    vae.eval()
    denoise_net.eval()

    N = dataset.N
    sub_indices = list(range(min(num_graphs, len(dataset))))
    sub_loader = DataLoader(
        torch.utils.data.Subset(dataset, sub_indices),
        batch_size=32,
        shuffle=False,
    )

    total_true_len = 0.0
    total_sample_len = 0.0
    count_graph = 0

    for batch_data in tqdm(sub_loader, desc="[Eval Sampling]"):
        batch_data = batch_data.to(device)
        x = batch_data.x
        edge_index = batch_data.edge_index
        y_edge = batch_data.y
        batch_vec = batch_data.batch
        B = batch_data.num_graphs


        adj_dense = to_dense_adj(
            edge_index, batch=batch_vec, edge_attr=y_edge.unsqueeze(-1)
        ).squeeze(1)  # [B,N,N]
        A_sym_true = ((adj_dense + adj_dense.transpose(1, 2)) > 0).float()


        coords_dense, _ = to_dense_batch(x, batch_vec)  # [B,N,2]

        # SinCos PE
        pe_list = [sinusoidal_pe(N, pe_dim, device) for _ in range(B)]
        pe_dense = torch.stack(pe_list, dim=0)  # [B,N,pe]


        true_lengths = []
        for b in range(B):
            tour_true = recover_tour_from_adj(A_sym_true[b])
            L_true = tour_length(coords_dense[b], tour_true)
            true_lengths.append(L_true)


        z0_sample = sample_z0_prior_ddim(denoise_net, diff_sched, B, latent_dim)


        logits_A_sample = vae.decoder(z0_sample, pe_dense, coords_dense)  # [B,N,N]
        prob_A_sample = torch.sigmoid(
            (logits_A_sample + logits_A_sample.transpose(1, 2)) / 2.0
        )  

        sample_lengths = []
        for b in range(B):
            tour_pred = nearest_neighbor_tour(prob_A_sample[b])
            L_pred = tour_length(coords_dense[b], tour_pred)
            sample_lengths.append(L_pred)

        total_true_len += float(np.sum(true_lengths))
        total_sample_len += float(np.sum(sample_lengths))
        count_graph += B

    avg_true = total_true_len / max(1, count_graph)
    avg_sample = total_sample_len / max(1, count_graph)
    ratio = avg_sample / (avg_true + 1e-8)

    print(
        f"\n[Sampling Eval] On {count_graph} graphs | "
        f"Avg true tour length = {avg_true:.3f} | "
        f"Avg sampled tour length = {avg_sample:.3f} | "
        f"Avg ratio (sample/true) = {ratio:.3f}"
    )


if __name__ == "__main__":
    pe_dim = 8
    latent_dim = 64

    vae, denoise_net, diff_sched, dataset = train_model(
        data_path="./data/tsp/tsp20_train_concorde.txt",
        max_instances=100000,
        epochs=200,
        batch_size=64,
        pe_dim=pe_dim,
        hidden_dim=128,
        latent_dim=latent_dim,
        lr_vae=1e-3,
        lr_diff=1e-3,
        T_latent=100,
        beta_kl=0.1,
        lambda_diff=0.5,
        pretrain_epochs=10,
    )

 
    evaluate_sampling(
        vae,
        denoise_net,
        diff_sched,
        dataset,
        pe_dim=pe_dim,
        latent_dim=latent_dim,
        num_graphs=200,
    )

