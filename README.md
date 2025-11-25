# COMP9993 - GraphVAE + Diffusion for Combinatorial Optimization

> Graph generative modelling for molecules (QM9, ZINC) and TSP using GraphVAE and diffusion models.  


---

## 1. Project Overview 

This repository contains my COMP9993 research project on **graph-based generative models** for **combinatorial optimization (CO)**.  
The main goals are:

- Compress graph-structured solutions (molecular graphs, TSP tours) into a **latent representation** with **GraphVAE**.
- Train **diffusion models** (DDPM/DDIM style) in this latent space to generate new valid structures.
- Evaluate the pipeline on:
  - **QM9, ZINC** molecular datasets (graph reconstruction).
  - **TSP-20, TSP-50, TSP-100** (tour generation, GAP metric).

---

## 2. Generative Model Architecture 

### 2.1 GraphVAE

For both molecules and TSP tours, I use a **GraphVAE** with:

- **Encoder**
  - Multi-layer **GCNConv** backbone.
  - Optional **TopKPooling** for graph-level pooling.
  - Outputs mean \(\mu\) and log-variance \(\log \sigma^2\) for the latent vector.
  - Node features = original attributes + **SinCos positional encoding** (Random PE vs. Degree-based PE).

- **Latent space**
  - Continuous latent vector \( z \in \mathbb{R}^d \).
  - Trained with standard **VAE loss** (reconstruction + KL divergence with warm-up).

- **Decoder**
  - Transformer-style decoder for molecular graphs (QM9/ZINC).
  - Binary adjacency decoder for TSP tours (three VAE versions V1–V3).
  - Produces reconstructed adjacency matrices or edge probabilities.

---

### 2.2 Diffusion Model in Latent Space

For TSP, the final pipeline uses **V3 GraphVAE** + **DDPM**:

- The VAE encodes the **TSP binary adjacency matrix** into a low-dimensional latent vector.
- A **DDPM-style diffusion model** is trained **only in the latent space**:
  - Forward process: gradually adds Gaussian noise to \( z \).
  - Reverse process: denoises \( z_t \rightarrow z_0 \) using a U-Net/MLP-style network.
- The denoised latent is decoded back to a TSP tour by the GraphVAE decoder.

---

## 3. Method Overview 

### 3.1 Molecule GraphVAE (QM9, ZINC)

1. **Data preprocessing**
   - Load QM9, ZINC molecules and convert them to graphs (atoms = nodes, bonds = edges).
   - Build node features (atom type one-hot) and adjacency matrices.

2. **Positional Encoding Experiments**
   - Introduce **SinCos positional encoding (PE)** to break node permutation symmetry.
   - Compare:
     - **Random PE assignment** (shuffle node order).
     - **Degree-based PE** (sort nodes by degree before assigning PE).
   - Metrics: node-level **F1**, **Accuracy**, and edge-level **AUC, AP**.

3. **Findings**
   - Random SinCos PE works better than degree-based PE.
   - GraphVAE prefers **weak, permutation-breaking signals** rather than strong structural ordering.
   - On ZINC, **AUC ≈ 0.99**, **AP ≈ 0.98**, showing strong reconstruction ability on large, complex graphs.


---

### 3.2 TSP GraphVAE (V1–V3) and Latent Compression

1. **TSP Representation**
   - Represent each TSP solution as a **binary adjacency matrix** (tour edges = 1, others = 0).
   - Train GraphVAE to reconstruct this matrix.

2. **Three VAE Versions**
   - **V1 – Initial compression**  
     Shallow GCN encoder + simple decoder; verifies that TSP tours can be compressed.
   - **V2 – Enhanced encoder**  
     Deeper GCN, residual message passing, larger hidden dims → better global path structure.
   - **V3 – Diffusion-ready**  
     Tuned latent dimension, pooling ratio, and decoder; reduces noise in binary reconstruction and aligns latent space with diffusion model needs.

3. **Evaluation**
   - Metrics: **Precision, Recall, F1, Accuracy**, and **GAP (\%)** between model tour length and Concorde optimal.
   - V3 achieves the best trade-off (GAP ≈ 10.22% on TSP-500 GT compression), providing a stable latent representation for later diffusion.


---

### 3.3 VAE + Diffusion Pipeline for TSP

1. **Step 1 – VAE Encoding**  
   - Use V3 GraphVAE to encode TSP adjacency matrices into low dimensional latent vectors.

2. **Step 2 – Diffusion in Latent Space**  
   - Train a DDPM in the VAE latent space to model the distribution of valid tours.
   - Sampling: start from Gaussian noise, run reverse diffusion to obtain a new latent tour.

3. **Step 3 – VAE Decoding and Path Extraction**  
   - Decode latent vectors back to adjacency matrices.
   - Extract Hamiltonian paths and compute tour length and GAP.

4. **Results (TSP-20, 50, 100)**
   - The pipeline achieves:
     - TSP-20: Length 3.89, GAP 1.09%.
     - TSP-50: GAP 1.89%.
     - TSP-100: GAP 2.23%.
   - Competitive with many learning-based baselines, though still behind large graph-diffusion models like **DIFUSCO**.
