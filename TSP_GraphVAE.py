import os
import math
import re
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

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
                        f"Line {line_id} too short: len={L}, "
                        f"but need at least 2N+N={2*N+N}."
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
                for e_idx, (u, v) in enumerate(zip(rows, cols)):
                    if (int(u), int(v)) in tour_edges:
                        y[e_idx] = 1.0

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


class SinTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] 整数时间步
        返回: [B, dim] Sin/Cos 位置编码
        """
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / half)
        )  # [half]
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B,half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


@torch.no_grad()
def calc_metrics_from_hat(
    pred_hat: torch.Tensor, target: torch.Tensor
) -> Tuple[float, float, float, float]:

    prob = (pred_hat.clamp(-1.0, 1.0) + 1.0) / 2.0  # [-1,1] -> [0,1]
    pred_bin = (prob > 0.5).float()

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    acc = (pred_bin == target).float().mean()

    return (
        precision.item(),
        recall.item(),
        f1.item(),
        acc.item(),
    )


class GaussianDDPMSchedule:
    def __init__(self, T: int = 200, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0_hat: torch.Tensor, t_edge: torch.Tensor):
        """
        x0_hat: [E]   真实的 {-1,1} 边变量
        t_edge: [E]   每条边的时间步（根据所在图的 t_graph 复制而来）
        返回:
          x_t_hat: [E]
          noise : [E]
        """
        noise = torch.randn_like(x0_hat)
        alpha_bar_t = self.alpha_bar[t_edge]  # [E]
        sqrt_ab = torch.sqrt(alpha_bar_t)
        sqrt_1m = torch.sqrt(1.0 - alpha_bar_t)
        x_t_hat = sqrt_ab * x0_hat + sqrt_1m * noise
        return x_t_hat, noise

class EdgeDenoiseGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 2,
        edge_in_dim: int = 2,  # [dist, x_t_hat]
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = SinTimeEmbed(time_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU())

        self.node_in = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # 边 MLP： [h_i, h_j, edge_attr(dist, x_t_hat), t_emb] -> x0_hat
        edge_feat_dim = 2 * hidden_dim + edge_in_dim + hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(), 
        )

    def forward(
        self,
        x: torch.Tensor,          # [N_total,2] 
        edge_index: torch.Tensor, # [2,E_total]
        edge_attr: torch.Tensor,  # [E_total,1] 
        x_t_hat: torch.Tensor,    # [E_total] 
        t_graph: torch.Tensor,    # [B] 
        batch: torch.Tensor,      # [N_total] 
    ):

        h = self.node_in(x)
        h = F.relu(h)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))


        t_emb_graph = self.time_embed(t_graph)          # [B,time_dim]
        t_emb_graph = self.time_mlp(t_emb_graph)        # [B,hidden]

        row, col = edge_index
        edge_batch = batch[row]                         # 每条边属于哪个图
        t_edge = t_emb_graph[edge_batch]                # [E,hidden]

 
        hi = h[row]
        hj = h[col]
        x_t_hat_col = x_t_hat.unsqueeze(-1)             # [E,1]
        edge_feat = torch.cat(
            [hi, hj, edge_attr, x_t_hat_col, t_edge], dim=-1
        )                                               # [E, 2H + 1 + 1 + H]

        x0_hat_pred = self.edge_mlp(edge_feat).squeeze(-1)  # [E] in [-1,1]
        return x0_hat_pred


def train_model(
    data_path: str = "./data/tsp/tsp20_train_concorde.txt",
    max_instances: int = 2000,
    epochs: int = 30,
    batch_size: int = 64,
    T: int = 200,
    lr: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    dataset = ConcordeTSPDataset(data_path, max_instances=max_instances)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model = EdgeDenoiseGNN(
        node_in_dim=2,
        edge_in_dim=2,
        hidden_dim=128,
        time_dim=64,
        num_layers=3,
    ).to(device)
    ddpm = GaussianDDPMSchedule(T=T, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        p_sum = r_sum = f1_sum = acc_sum = 0.0
        denom = max(1, len(loader))

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", mininterval=0.5)
        for batch_data in pbar:
            batch_data = batch_data.to(device)

            x = batch_data.x               # [N_tot,2]
            edge_index = batch_data.edge_index
            edge_attr = batch_data.edge_attr  # [E_tot,1]
            y = batch_data.y               # [E_tot], 0/1
            batch_vec = batch_data.batch   # [N_tot]
            B = batch_data.num_graphs


            x0_hat = y * 2.0 - 1.0         # 0 -> -1, 1 -> 1


            t_graph = torch.randint(0, ddpm.T, (B,), device=device, dtype=torch.long)


            row, _ = edge_index
            edge_batch = batch_vec[row]
            t_edge = t_graph[edge_batch]   # [E_tot]


            x_t_hat, _ = ddpm.q_sample(x0_hat, t_edge)


            x0_hat_pred = model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                x_t_hat=x_t_hat,
                t_graph=t_graph,
                batch=batch_vec,
            )

          
            loss = F.mse_loss(x0_hat_pred, x0_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


            with torch.no_grad():
                p, r, f1, acc = calc_metrics_from_hat(x0_hat_pred, y)
                p_sum += p
                r_sum += r
                f1_sum += f1
                acc_sum += acc

            pbar.set_postfix(loss=float(loss.item()), f1=float(f1))

        print(
            f"[Epoch {epoch:03d}] loss={total_loss/denom:.4f} | "
            f"P={p_sum/denom:.4f} R={r_sum/denom:.4f} "
            f"F1={f1_sum/denom:.4f} Acc={acc_sum/denom:.4f}"
        )


if __name__ == "__main__":
    train_model(
        data_path="./data/tsp/tsp20_train_concorde.txt",
        max_instances=2000,  
        epochs=30,
        batch_size=64,
        T=200,
        lr=1e-3,
    )

