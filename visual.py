import matplotlib.pyplot as plt
import numpy as np

# ====== 1️⃣ 读取文件 ======
file_path = "tsp500_train_lkh_100.txt"  # 你的文件路径

with open(file_path, "r") as f:
    lines = f.readlines()

# ====== 2️⃣ 解析第一个样本 ======
sample_idx = 0  # 改成 1, 2, 3 可查看其他样本
line = lines[sample_idx].strip().split("output")

coords_part, tour_part = line[0].strip(), line[1].strip()

# 坐标部分
coords = np.array(list(map(float, coords_part.split()))).reshape(-1, 2)

# tour 部分（1-based → 0-based）
tour = np.array(list(map(int, tour_part.split()))) - 1

# ====== 3️⃣ 生成封闭路径 ======
tour_coords = coords[tour]

# ====== 4️⃣ 绘制图像 ======
plt.figure(figsize=(6, 6))
plt.scatter(coords[:, 0], coords[:, 1], c="blue", s=8, label="Cities")
plt.plot(tour_coords[:, 0], tour_coords[:, 1], c="red", linewidth=1.5, label="Optimal tour")
plt.title(f"TSP-500 Sample #{sample_idx} (GT Tour from LKH)")
plt.legend()
plt.axis("equal")
plt.show()
