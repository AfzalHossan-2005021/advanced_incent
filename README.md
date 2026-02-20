# advanced_incent

A production-quality implementation of the **INCENT** (_Integrated Cellular Niche and Expression Transport_) algorithm for spatial-transcriptomics slice alignment.

---

## Table of Contents

1. [Background](#1-background)
2. [Algorithm in Full](#2-algorithm-in-full)
3. [Mathematical Formulation](#3-mathematical-formulation)
4. [Implementation Walk-through](#4-implementation-walk-through)
5. [Improvements over the Baseline](#5-improvements-over-the-baseline)
6. [Quick Start](#6-quick-start)
7. [API Reference](#7-api-reference)
8. [Algorithm Review Q & A](#8-algorithm-review-q--a)
9. [References](#9-references)

---

## 1. Background

Spatial transcriptomics technologies (e.g., MERFISH, Visium, Slide-seq) measure gene expression at single-cell resolution while retaining physical location. Comparing slices from different time-points, donors, or conditions requires **spatial alignment**: finding a probabilistic mapping $\pi$ between cells of slice A and cells of slice B that respects both _what genes are expressed_ and _how cells are spatially organised_.

INCENT solves this with a **Fused Gromov-Wasserstein (FGW)** transport problem that jointly minimises three complementary costs:

| Term  | What it measures                                                                               |
| ----- | ---------------------------------------------------------------------------------------------- |
| $M_1$ | Gene-expression dissimilarity (cosine distance, augmented with cell-type one-hot)              |
| $M_2$ | Cellular-niche dissimilarity (Jensen-Shannon / cosine / MSD of neighbourhood type-composition) |
| GW    | Preservation of spatial structure (pairwise Euclidean distances within each slice)             |

---

## 2. Algorithm in Full

```
Input:  sliceA, sliceB  (AnnData with .X, .obsm['spatial'], .obs['cell_type_annot'])
        AlignConfig     (all hyperparameters)

Step 1  Build M1 – gene-expression cost
        ─────────────────────────────────
        For each cell i in A, j in B:
          s_i = [expr_i | beta * one_hot(type_i)]     # concatenate
          s_j = [expr_j | beta * one_hot(type_j)]
          M1[i,j] = cosine_distance(s_i, s_j)
        Cache result to disk as cosine_dist_gene_expr_A_B.npy

Step 2  Build neighbourhood distributions
        ──────────────────────────────────
        For each cell i in slice S:
          neighbours_i = {k : ||coord_i - coord_k|| ≤ radius}
          nd_i = histogram over cell types in neighbours_i   (k-vector)
        (Parallelised with joblib over all cells; cached per slice)

Step 3  Build M2 – niche dissimilarity
        ─────────────────────────────────
        For each (i, j):
          M2[i,j] = JSD(nd_A_i, nd_B_j)       [default]
                  | cosine_dist(nd_A_i, nd_B_j)
                  | MSD(nd_A_i, nd_B_j)
        JSD is computed in chunks of 'jsd_chunk_size' rows on GPU/CPU
        Cache result to disk as js_dist_neighborhood_A_B.npy

Step 4  Build spatial structure matrices
        ─────────────────────────────────
        D_A[i,k] = ||coord_i - coord_k||   (within sliceA)
        D_B[j,l] = ||coord_j - coord_l||   (within sliceB)

Step 5  Initialise transport plan G0
        ────────────────────────────
        If cell_type_init:
          S[i,j] = weight  if cell_type(i) == cell_type(j)
                 = 1       otherwise
          G0 = outer(a, b) * S  /  sum(outer(a, b) * S)
        Else:
          G0 = outer(a, b)      (uniform)

Step 6  Solve FGW via Frank-Wolfe (Conditional Gradient)
        ─────────────────────────────────────────────────
        Minimise (over G ≥ 0):
          F(G) = (1-α)·[<M1,G> + γ·<M2,G>]
               + α·[<C1, GGᵀ> + <C2, GᵀG>]
               + ρ1·KL(G1 ‖ a)  +  ρ2·KL(Gᵀ1 ‖ b)

        repeat until convergence:
          a)  Linearise: Mi = ∇_G [(1-α)<M1,G> + α·f(G)]
          b)  Inner LP:  Gc = mm_unbalanced(a, b, Mi; ρ1, ρ2)
                              (or exact EMD when ρ1,ρ2 ≥ threshold)
          c)  Line-search: α* = argmin_α F(G + α·(Gc - G))
                           (quadratic formula for square_loss)
          d)  Update:  G ← G + α* · (Gc - G)

Output: π = optimal transport plan (ns × nt matrix)
        Logged: initial & final objectives, marginal violations, runtime
```

---

## 3. Mathematical Formulation

### 3.1 Primal Objective

$$
\pi^* = \underset{\pi \geq 0}{\arg\min}
    \underbrace{(1-\alpha)\bigl[\langle M_1,\pi\rangle + \gamma\langle M_2,\pi\rangle\bigr]}_{\text{feature alignment}}
    + \underbrace{\alpha\,\bigl[\langle C_1, \pi\pi^\top\rangle + \langle C_2, \pi^\top\pi\rangle\bigr]}_{\text{structural alignment (GW)}}
    + \underbrace{\rho_1\,\mathrm{KL}(\pi\mathbf{1}\,\|\,\mathbf{a}) + \rho_2\,\mathrm{KL}(\pi^\top\mathbf{1}\,\|\,\mathbf{b})}_{\text{unbalanced relaxation}}
$$

**Balanced limit:** $\rho_1, \rho_2 \to \infty$ recovers the hard marginal constraints $\pi\mathbf{1}=\mathbf{a}$ and $\pi^\top\mathbf{1}=\mathbf{b}$.

### 3.2 Generalised KL Divergence

$$
\mathrm{KL}(p \,\|\, q) = \sum_i \Bigl[p_i \log\frac{p_i}{q_i} - p_i + q_i\Bigr]
$$

This is the _unnormalised_ (Csiszár) KL, which is finite and smooth even when $p$ is not a probability simplex.

### 3.3 Linearised Subproblem (inner LP)

At iteration $t$ with current plan $G$:

$$
G_c = \underset{P \geq 0}{\arg\min}\;\langle M_i^{(t)}, P\rangle
    + \rho_1\,\mathrm{KL}(P\mathbf{1}\,\|\,\mathbf{a}) + \rho_2\,\mathrm{KL}(P^\top\mathbf{1}\,\|\,\mathbf{b})
$$

where

$$
M_i^{(t)} = M_1 + \alpha\,\nabla_G f(G^{(t)}), \qquad
\nabla_G f = 2\bigl(C_1 G + G C_2\bigr)
$$

Solved by **multiplicative updates** (no entropic blur):

$$
P^{(k+1)}_{ij} \;\propto\; P^{(k)}_{ij}\exp\!\Bigl(-\tfrac{M^{(t)}_{ij}}{\rho}\Bigr)
$$

### 3.4 Quadratic Line-search

$$
F(G + s\,\Delta G) = \underbrace{F(G)}_{\text{cost\_G}} + b\,s + a\,s^2
$$

$$
a = -2\,\langle C_1\Delta G C_2^\top,\,\Delta G\rangle, \quad
b = \langle M,\Delta G\rangle - 2\,\bigl[\langle C_1\Delta G C_2^\top, G\rangle + \langle C_1 G C_2^\top,\Delta G\rangle\bigr]
$$

$$
s^* = -\tfrac{b}{2a} \quad \text{(clipped to } [0,1]\text{)}
$$

### 3.5 Jensen-Shannon Distance (vectorised chunk form)

For chunk rows $P \in \mathbb{R}^{c \times k}$ and all target rows $Q \in \mathbb{R}^{m \times k}$:

$$
M = \frac{P + Q}{2},\qquad
\mathrm{JSD}(P_i, Q_j) = \sqrt{\frac{\mathrm{KL}(P_i\|M_{ij}) + \mathrm{KL}(Q_j\|M_{ij})}{2}}
$$

Using broadcasting $(c, 1, k)$ vs $(1, m, k)$ — one GPU kernel per chunk.

---

## 4. Implementation Walk-through

```
advanced_incent/
├── config.py              # AlignConfig dataclass  — all hyperparameters
├── alignment.py           # pairwise_align()        — main entry point
├── initialization.py      # cell_type_aware_init()  — G0 via cell-type prior
├── distances.py           # M1 (cosine) + M2 (JSD/cosine/MSD)
├── neighborhood.py        # neighbourhood_distribution()  — parallelised
├── scoring.py             # age_progression_score()
├── transport/
│   ├── fgw.py             # fused_gromov_wasserstein_incent()
│   ├── conditional_gradient.py  # generic_cg + cg_incent
│   ├── linesearch.py      # solve_gromov_linesearch
│   └── __init__.py
└── utils/
    ├── device.py          # DeviceManager (GPU picker + tensor helpers)
    ├── array_utils.py     # extract_data_matrix, to_dense_array, to_numpy
    └── __init__.py
```

### Module responsibilities

| File                                | Key function                       | Improvement                                                  |
| ----------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| `config.py`                         | `AlignConfig`                      | Replaces 20+ positional args with a clean dataclass          |
| `utils/device.py`                   | `DeviceManager`                    | Picks best GPU by free memory; unified `to()` / `to_numpy()` |
| `initialization.py`                 | `cell_type_aware_init`             | Fills original `# todo` — vectorised cell-type prior on G₀   |
| `distances.py`                      | `jensenshannon_divergence_backend` | Chunked 3-D tensor JSD; no Python for-loop                   |
| `neighborhood.py`                   | `neighborhood_distribution`        | `joblib.Parallel` with thread backend                        |
| `transport/conditional_gradient.py` | `cg_incent`                        | KL cost terms in `cost(G)` for consistent line-search        |
| `transport/fgw.py`                  | `fused_gromov_wasserstein_incent`  | `log.get('u', None)` prevents KeyError                       |
| `alignment.py`                      | `pairwise_align`                   | `logging` module, `AlignConfig`, `DeviceManager`             |

---

## 5. Improvements over the Baseline

| #   | Issue in original                                                 | Fix in advanced_incent                                            |
| --- | ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| 1   | 20+ positional arguments in `pairwise_align`                      | `AlignConfig` dataclass                                           |
| 2   | Manual `open(log.txt, 'w')` file writes                           | Python `logging` module with file handler                         |
| 3   | `# todo: cell-type aware init` left unimplemented                 | `cell_type_aware_init` fills it                                   |
| 4   | O(n) Python for-loop for JSD                                      | Chunked `(c, m, k)` tensor broadcast on GPU                       |
| 5   | Sequential neighbourhood computation                              | `joblib.Parallel(prefer='threads')`                               |
| 6   | `cost(G)` had no KL terms → line-search minimised wrong objective | KL penalty added to `cost(G)`                                     |
| 7   | `log['u'] = log['u']` crashes with mm_unbalanced                  | `log.get('u', None)` guards safely                                |
| 8   | GPU picked as `cuda:0` regardless of memory                       | `DeviceManager._pick_device()` selects GPU with most free memory  |
| 9   | Scattered `isinstance(x, torch.Tensor)` checks everywhere         | `DeviceManager.to()` / `to_numpy()` handle all backends uniformly |

---

## 6. Quick Start

```python
import scanpy as sc
from advanced_incent import pairwise_align, AlignConfig, age_progression_score

# Load slices
sliceA = sc.read_h5ad("data/Mouse_brain_MERFISH/adata4wk_donor_id_4_slice_0.h5ad")
sliceB = sc.read_h5ad("data/Mouse_brain_MERFISH/adata24wk_donor_id_11_slice_0.h5ad")

# Configure
config = AlignConfig(
    alpha=0.1,          # GW (spatial structure) weight
    beta=0.8,           # Cell-type one-hot weight in gene expression
    gamma=0.8,          # Neighbourhood term weight (M2 vs M1)
    radius=100.0,       # Neighbourhood radius (microns)
    rho1=1.0,           # Source KL penalty (unbalanced; set >>1 for balanced)
    rho2=1.0,           # Target KL penalty
    cell_type_init=True,           # Use cell-type aware G0
    cell_type_init_weight=10.0,    # Boost for matching cell types
    neighborhood_dissimilarity='jsd',
    jsd_chunk_size=64,  # GPU chunk size; reduce if CUDA OOM
    use_gpu=True,       # Use best available GPU
    n_jobs=-1,          # All CPUs for neighbourhood computation
)

# Align
pi = pairwise_align(
    sliceA, sliceB, config,
    sliceA_name="4wk", sliceB_name="24wk",
    filePath="outputs/",
)

# Score age progression
sliceA, sliceB = age_progression_score(sliceA, sliceB, "4wk", "24wk", "outputs/")
```

---

## 7. API Reference

### `AlignConfig`

```python
@dataclass
class AlignConfig:
    alpha: float = 0.1              # GW weight (0 = pure feature OT, 1 = pure GW)
    beta: float = 0.8               # Cell-type one-hot scale in M1
    gamma: float = 0.8              # M2 weight relative to M1
    radius: float = 100.0           # Neighbourhood radius
    rho1: float = 1.0               # Source KL penalty (∞ → balanced)
    rho2: float = 1.0               # Target KL penalty (∞ → balanced)
    balanced_fallback_threshold: float = 1e6   # Use EMD above this rho
    numItermax: int = 6000          # Max FW iterations
    tol_rel: float = 1e-9           # Relative convergence tolerance
    tol_abs: float = 1e-9           # Absolute convergence tolerance
    neighborhood_dissimilarity: str = 'jsd'    # 'jsd' | 'cosine' | 'msd'
    jsd_chunk_size: int = 64        # Rows per GPU kernel in JSD computation
    cell_type_init: bool = True     # Use cell-type aware G0
    cell_type_init_weight: float = 10.0  # Boost for matching cell types
    use_gpu: bool = False           # Enable CUDA
    n_jobs: int = -1                # Parallel jobs for neighbourhood
    use_rep: str = None             # obsm key for expression (None → X)
    overwrite: bool = False         # Recompute even if cache exists
    verbose: bool = False           # Detailed iteration log
    norm: bool = False              # Normalise spatial distance matrices
```

### `pairwise_align(sliceA, sliceB, config, *, sliceA_name, sliceB_name, filePath, G_init, a_distribution, b_distribution, return_obj)`

Main alignment function. Returns the (ns, nt) transport plan `pi`.

### `age_progression_score(sliceA, sliceB, data1, data2, filePath)`

Adds `obs['age_progression_score']` to both slices using cached `pi_matrix` and `cosine_dist_gene_expr` files.

### `cell_type_aware_init(sliceA, sliceB, a, b, weight)`

Returns a cell-type biased initial plan. Called automatically by `pairwise_align` when `config.cell_type_init = True`.

---

## 8. Algorithm Review Q & A

**Q1: Why use Fused Gromov-Wasserstein instead of plain OT?**

Plain (Monge-Kantorovich) OT matches cells purely by feature similarity and can map spatially distant cells together if their features happen to match. FGW adds a _structural penalty_: if cell $i_1$ is close to $i_2$ in slice A, then their images $j_1, j_2$ in slice B should also be close. This is the Gromov-Wasserstein term $\langle C_1,\pi\pi^\top\rangle + \langle C_2,\pi^\top\pi\rangle$.

**Q2: Why add neighbourhood distributions (M2)?**

Gene expression alone may not distinguish spatially distinct but transcriptomically similar cell types (e.g., excitatory neurons in different cortical layers). The neighbourhood distribution (the histogram of nearby cell types within radius $r$) encodes the _spatial context_: a cell embedded in a mostly-GABAergic niche is distinguished from a chemically identical cell surrounded by glutamatergic neighbours. $M_2$ captures this without requiring an explicit graph architecture.

**Q3: Why is the JSD used instead of KL or cosine for the niche term?**

- **KL** is asymmetric and blows up when one distribution has zero mass.
- **Cosine** ignores the absolute scale of the histogram (a cell with 1 neighbour and a cell with 100 neighbours in the same proportions get the same score).
- **JSD** is symmetric, bounded in $[0,1]$, and defined even for distributions with non-overlapping support. This makes it the most robust choice for comparing neighbourhood histograms of varying sample sizes.

**Q4: Why use unbalanced OT (KL marginal relaxation)?**

Real datasets have:

- **Batch effects** — the number of detected cells per cell type differs across slices.
- **Dropout / missing cells** — some cells captured in one slice have no counterpart in another.

Hard marginal constraints $\pi\mathbf{1}=\mathbf{a}$ force all source mass to be transported even if the target has no good match. KL relaxation lets the plan _partially ignore_ outlier cells (those with no near counterpart pay a penalty $\rho \cdot \mathrm{KL}$ instead). Setting $\rho_1 = \rho_2 = 10^6$ recovers the fully balanced solution.

**Q5: Why must the KL terms appear in `cost(G)` (the objective), not only in the LP direction?**

The Frank-Wolfe convergence analysis requires that the _line-search evaluates the true objective_. If `cost(G)` omits the KL marginal terms, the step-size $s^*$ is computed for the balanced objective while the LP direction $G_c$ is computed for the unbalanced objective. These two are inconsistent — the algorithm will not converge to the true unbalanced optimum and may oscillate. Adding the KL terms to `cost(G)` fixes this.

**Q6: Why `mm_unbalanced` instead of `sinkhorn_unbalanced` for the inner LP?**

`sinkhorn_unbalanced` adds _entropic regularisation_ $\varepsilon H(\pi)$, which blurs the transport plan. To recover a sharp plan you must anneal $\varepsilon \to 0$, which requires additional tuning. `mm_unbalanced` (multiplicative updates) solves the exact KL-marginal problem with _no_ extra regularisation, making it a drop-in replacement for `emd` in the conditional-gradient loop.

**Q7: How does cell-type aware initialisation help?**

The FGW objective is non-convex; Frank-Wolfe may converge to different local minima depending on $G_0$. Starting from $G_0 \propto \mathbf{a}\mathbf{b}^\top$ (uniform) gives equal prior probability to all matches. The cell-type prior boosts $G_0[i,j]$ by `weight` whenever $\text{type}(i) = \text{type}(j)$, pushing the algorithm toward the biologically plausible region of the solution space. This typically reduces the number of FW iterations needed and improves final alignment quality.

**Q8: What is the computational complexity?**

| Step                       | Complexity                                     |
| -------------------------- | ---------------------------------------------- |
| Neighbourhood distribution | $O(n^2 d)$ with $d=2$ spatial dims             |
| M1 (cosine distance)       | $O(n_A \cdot n_B \cdot g)$ with $g$ genes      |
| M2 (JSD)                   | $O(n_A \cdot n_B \cdot k)$ with $k$ cell types |
| Each FW iteration          | $O(n_A^2 + n_B^2 + n_A n_B)$                   |
| Total FW                   | $O(T \cdot (n_A^2 + n_B^2 + n_A n_B))$         |

The chunked JSD and joblib neighbourhood parallelisation reduce the constant factors on M2 and neighbourhood computation — the two steps that dominate for large slices.

**Q9: How should `alpha`, `gamma`, `rho1`/`rho2` be tuned?**

| Hyperparameter | Low value                  | High value               | Suggested starting point                             |
| -------------- | -------------------------- | ------------------------ | ---------------------------------------------------- |
| `alpha`        | Gene/niche driven          | Spatial structure driven | 0.1                                                  |
| `gamma`        | Gene-only                  | Niche-heavy              | 0.8                                                  |
| `rho1`/`rho2`  | Highly unbalanced          | Fully balanced           | 1.0 (start); raise to 1e3 if marginals drift too far |
| `radius`       | Local niche (fine-grained) | Diffuse niche            | 100 µm for MERFISH                                   |

**Q10: Is the algorithm guaranteed to converge?**

The Frank-Wolfe algorithm converges sub-linearly ($O(1/T)$) for non-convex objectives. For the FGW problem specifically, convergence to a _stationary point_ is guaranteed under mild conditions on the line-search (Vayer et al., 2019). The unbalanced extension inherits this guarantee because the KL terms are smooth and convex, so the overall objective remains continuously differentiable — the only requirement for FW convergence.

---

## 9. References

1. **Vayer T., Chapel L., Flamary R., Tavenard R., Courty N.** (2019). _Optimal Transport for Structured Data with Application on Graphs._ ICML 2019.

2. **Chapel L., Flamary R., Wu H., Févotte C., Peyré G.** (2021). _Unbalanced Optimal Transport through Non-negative Penalized Linear Regression._ NeurIPS 2021.

3. **Peyré G., Cuturi M.** (2019). _Computational Optimal Transport._ Foundations and Trends in ML.

4. **Flamary R. et al.** (2021). _POT: Python Optimal Transport._ J. Machine Learning Research.

5. **Bhowmik A.** _INCENT: Integrated Cellular Niche and Expression Transport for Spatial Transcriptomics Alignment._ BUET CSE.
