# ============================================================
# DiGress-style Categorical Diffusion on QM9 (Minimal, Practical)
# - 节点原子类型 & 边键类型：离散扩散（D3PM/DiGress 思路）
# - 训练：预测 x0 的类别分布（CE 损失），监控 n_acc / e_acc
# - 采样：T->0 贪心，同时输出 edge_logits
# - 解码：约束解码（价键修复，按置信度排序，超价则降级/跳过）
# - 评估：RDKit Valid / Unique
# ============================================================

# =========== 如遇 OpenMP 冲突，可取消注释以下两行 ==============
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 临时规避 OpenMP 冲突，不建议长期使用
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

# ====================== 1) 数据 & 标签 ======================
dataset = QM9(root="./data/QM9")
N_total = len(dataset)
N_train = int(0.85 * N_total)
N_val   = int(0.05  * N_total)
train_set = dataset[:N_train]
val_set   = dataset[N_train:N_train+N_val]
test_set  = dataset[N_train+N_val:]

# 节点类别：H,C,N,O,F -> 5类
ATOM_Z = [1,6,7,8,9]
Z2ID = {z:i for i,z in enumerate(ATOM_Z)}
N_ATOM = len(ATOM_Z)

# 边类别：QM9 edge_attr 一般是 4 维 one-hot [SINGLE, DOUBLE, TRIPLE, AROMATIC]
def get_bond_dim(sample):
    return sample.edge_attr.size(-1)
N_BOND = get_bond_dim(dataset[0])

def to_atom_id(z_tensor):
    # z: 原子序号 [N] -> 0..4
    ids = torch.zeros_like(z_tensor)
    for z, idx in Z2ID.items():
        ids[z_tensor==z] = idx
    return ids

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)

# ====================== 2) 离散扩散调度（DiGress风格） ======================
def linear_beta_schedule(T, beta_start=1e-3, beta_end=0.05):
    return torch.linspace(beta_start, beta_end, T)

class DiscreteDiffusion:
    """
    类别扩散：每个时间步以概率 beta_t 将标签替换为均匀噪声（其余保持原值）。
    α_t = 1 - β_t；ᾱ_t = ∏_s<=t α_s。
    """
    def __init__(self, T=200, n_atom=5, n_bond=4, device="cpu"):
        self.T = T
        self.device = device
        self.n_atom = n_atom
        self.n_bond = n_bond
        self.betas = linear_beta_schedule(T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)  # [T]

    def q_sample_nodes(self, y0, t_node):
        """
        y0: [N] 节点原始标签
        t_node: [N] 每个节点的时间步
        返回 xt（被替换为随机类别的地方用均匀噪声）
        """
        keep_prob = self.alpha_bar[t_node]                 # [N]
        replace = (torch.rand_like(keep_prob) > keep_prob).long()  # 1=替换
        noise = torch.randint(low=0, high=self.n_atom, size=y0.shape, device=y0.device)
        return y0*(1-replace) + noise*replace

    def q_sample_edges(self, e0, t_edge):
        keep_prob = self.alpha_bar[t_edge]
        replace = (torch.rand_like(keep_prob) > keep_prob).long()
        noise = torch.randint(low=0, high=self.n_bond, size=e0.shape, device=e0.device)
        return e0*(1-replace) + noise*replace

diff = DiscreteDiffusion(T=200, n_atom=N_ATOM, n_bond=N_BOND, device=device)

# ====================== 3) 时间嵌入 & 模型 ======================
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
    """
    输入：节点 one-hot（x_t）；输出：节点 logits [N,n_atom]、边 logits [E,n_bond]。
    （最小实现：不把边 x_t 作为输入特征；如需更强，可把边特征融入消息传递）
    """
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
        h = h + temb[batch_vec]                           # 注入时间
        node_logits = self.node_head(h)                   # [N, n_atom]
        src, dst = edge_index
        edge_h = torch.cat([h[src], h[dst]], dim=-1)      # [E, 2H]
        edge_logits = self.edge_mlp(edge_h)               # [E, n_bond]
        return node_logits, edge_logits

model = DiGressGNN(N_ATOM, N_BOND).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-4)

# ====================== 4) 训练与评估（CE：预测 x0） ======================
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

            # 原始标签
            y_node = to_atom_id(data.z).to(device)           # [N]
            y_edge = data.edge_attr.argmax(-1).to(device)    # [E]

            # 图级 t -> 节点/边
            t_graph = torch.randint(0, diff.T, (G,), device=device).long()
            t_node  = t_graph[batch_vec]                     # [N]
            src, dst = data.edge_index
            t_edge = batch_vec[src]                          # [E]

            # 前向离散加噪：得到 x_t
            x_node_t = diff.q_sample_nodes(y_node, t_node)   # [N]
            # x_edge_t = diff.q_sample_edges(y_edge, t_edge)  # 本最小实现未作为输入

            node_oh = one_hot(x_node_t, N_ATOM)

            # 预测 x0 的分布
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

# ====================== 5) 采样（T->0 贪心） ======================
def sample_greedy_from_batch(model, data, T=None):
    """
    返回：
      atom_cls: [N] 节点类别
      edge_logits: [E, n_bond] 最后一步的边 logits（用于约束解码排序/降级）
    """
    model.eval()
    data = data.to(device)
    batch_vec = data.batch
    G = data.num_graphs
    N = data.x.size(0)
    T = diff.T if T is None else T

    # x_T：节点均匀随机
    x_node = torch.randint(0, N_ATOM, (N,), device=device)

    edge_logits = None
    for t in reversed(range(T)):
        t_graph = torch.full((G,), t, device=device, dtype=torch.long)
        node_oh = one_hot(x_node, N_ATOM)
        node_out, edge_out = model(node_oh, data.edge_index, t_graph, batch_vec)
        x_node = node_out.argmax(-1)
        edge_logits = edge_out  # 始终保留最新一步的边 logits
    return x_node.detach().cpu(), edge_logits.detach().cpu()

# ====================== 6) 约束解码（价键修复） ======================
# 允许价（简化表）
ALLOWED_VALENCE = {1:1, 6:4, 7:3, 8:2, 9:1}  # H,C,N,O,F
# 键阶
BOND_ORDER = {0:1, 1:2, 2:3, 3:1}            # SINGLE,DOUBLE,TRIPLE,AROMATIC≈1
# RDKit 键类型映射
BOND_TYPE = {
    0: rdchem.BondType.SINGLE,
    1: rdchem.BondType.DOUBLE,
    2: rdchem.BondType.TRIPLE,
    3: rdchem.BondType.AROMATIC,
}

def build_mol_repair(atom_cls, edge_index, edge_logits):
    """
    按边置信度从高到低尝试加键；若会导致任一端超价则降级（如三键->双键->单键->芳香），仍不行就跳过。
    """
    rw = Chem.RWMol()
    idx_map = {}
    try:
        # 加原子
        Zs = []
        for i, c in enumerate(atom_cls):
            Z = ATOM_Z[int(c)]
            a = Chem.Atom(int(Z))
            idx = rw.AddAtom(a)
            idx_map[i] = idx
            Zs.append(Z)

        # 每个原子的剩余价
        remain = [ALLOWED_VALENCE.get(Z, 4) for Z in Zs]

        # 边分数与预测
        with torch.no_grad():
            probs = torch.softmax(edge_logits, dim=-1).cpu().numpy()  # [E, n_bond]
            scores = probs.max(axis=1)
            pred   = probs.argmax(axis=1)

        src, dst = edge_index
        E = src.size(0)
        edges = list(range(E))
        # 只处理无向一次（u < v）
        edges = [e for e in edges if int(src[e]) < int(dst[e])]
        # 按分数从高到低
        edges.sort(key=lambda e: float(scores[e]), reverse=True)

        for e in edges:
            u = int(src[e]); v = int(dst[e])
            want = int(pred[e])

            # 候选类型顺序：先试想要的，然后降级到单键/双键/芳香/三键（可按需调整策略）
            candidates = [want, 0, 1, 3, 2]
            placed = False
            for cand in candidates:
                order = BOND_ORDER[cand]
                if remain[u] >= order and remain[v] >= order:
                    rw.AddBond(idx_map[u], idx_map[v], BOND_TYPE[cand])
                    remain[u] -= order; remain[v] -= order
                    placed = True
                    break
            # 放弃该边（不加）以避免超价
            _ = placed

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)  # 若失败会抛异常
        return mol
    except Exception:
        return None

# ====================== 7) 评估：Valid / Unique ======================
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
    # 唯一性：对合法分子做 canonical SMILES 去重
    smis = []
    for m in valid_mols:
        try:
            s = Chem.MolToSmiles(m, canonical=True)
            smis.append(s)
        except Exception:
            pass
    unique = len(set(smis)) / max(1, len(valid_mols))
    return {"valid": valid, "unique": unique}

# ====================== 8) 训练主循环 ======================
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

# ====================== 9) 采样评估（Valid/Unique） ======================
vu = evaluate_valid_unique(model, val_loader, max_batches=30)
print("\n[Val Sampling] Valid: {:.3f} | Unique: {:.3f}".format(vu["valid"], vu["unique"]))

# ====================== 10) 简单可视化 ======================
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
