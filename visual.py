import matplotlib.pyplot as plt
import numpy as np

file_path = "tsp500_train_lkh_100.txt"

with open(file_path, "r") as f:
    lines = f.readlines()


sample_idx = 0  
line = lines[sample_idx].strip().split("output")

coords_part, tour_part = line[0].strip(), line[1].strip()


coords = np.array(list(map(float, coords_part.split()))).reshape(-1, 2)

tour = np.array(list(map(int, tour_part.split()))) - 1


tour_coords = coords[tour]


plt.figure(figsize=(6, 6))
plt.scatter(coords[:, 0], coords[:, 1], c="blue", s=8, label="Cities")
plt.plot(tour_coords[:, 0], tour_coords[:, 1], c="red", linewidth=1.5, label="Optimal tour")
plt.title(f"TSP-500 Sample #{sample_idx} (GT Tour from LKH)")
plt.legend()
plt.axis("equal")
plt.show()

