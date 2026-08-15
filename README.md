# SVM from Scratch — Primal, Dual & Kernel Formulations

> Three complete Support Vector Machine implementations built from mathematical first principles — no sklearn for core algorithms.

---

## Overview

This project implements and compares three SVM variants on the [UCI Banknote Authentication dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) (1,372 samples, 4 wavelet-transform features). Each model is built entirely from scratch in NumPy, covering the full mathematical pipeline from loss derivation to gradient computation to decision boundary visualization.

| Model | Loss Function | Optimization |
|---|---|---|
| Primal SVM | Hinge loss + L2 regularization | Batch subgradient descent |
| Dual SVM (RBF Kernel) | Dual objective (Lagrangian) | Gradient ascent on α |
| Squared Hinge SVM | Differentiable squared hinge | Batch gradient descent |

---

## Mathematical Formulations

### 1. Primal SVM — Subgradient Descent

Minimizes the regularized hinge loss:

$$\mathcal{L}(w, b) = \frac{\lambda}{2}\|w\|^2 + \frac{1}{m}\sum_{i=1}^{m} \max\left(0,\ 1 - y_i(w \cdot x_i + b)\right)$$

The subgradient update rule for samples violating the margin ($y_i(w \cdot x_i + b) < 1$):

$$\nabla_w \mathcal{L} = \lambda w - \frac{1}{m}\sum_{i \in \mathcal{V}} y_i x_i \qquad \nabla_b \mathcal{L} = -\frac{1}{m}\sum_{i \in \mathcal{V}} y_i$$

Labels are mapped $\{0, 1\} \to \{-1, +1\}$ to conform to the hinge loss formulation.

Support vectors are identified as points satisfying $y_i(w \cdot x_i + b) \leq 1$.

---

### 2. Dual SVM — RBF Kernel

Solves the dual optimization problem by maximizing the Lagrangian:

$$\max_{\alpha}\ \sum_{i=1}^{m} \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

subject to $0 \leq \alpha_i \leq C$.

The **RBF kernel** maps data implicitly to infinite-dimensional feature space:

$$K(x_i, x_j) = e^{-\gamma \|x_i - x_j\|^2}$$

The decision function uses only support vectors (where $\alpha_i > 0$):

$$f(x) = \sum_{i \in \mathcal{SV}} \alpha_i y_i K(x_i, x) + b$$

Bias $b$ is computed from support vectors on the margin via the KKT conditions.

---

### 3. Squared Hinge Loss SVM — Gradient Descent

Replaces the non-differentiable hinge loss with a differentiable squared version:

$$\mathcal{L}(w, b) = \frac{\lambda}{2}\|w\|^2 + \frac{1}{m}\sum_{i=1}^{m} \max\left(0,\ 1 - y_i(w \cdot x_i + b)\right)^2$$

Closed-form gradients (fully differentiable):

$$\nabla_w \mathcal{L} = \lambda w - \frac{2}{m}\sum_{i \in \mathcal{V}} y_i x_i \cdot \max(0, 1 - y_i f(x_i))$$

$$\nabla_b \mathcal{L} = -\frac{2}{m}\sum_{i \in \mathcal{V}} y_i \cdot \max(0, 1 - y_i f(x_i))$$

Trained with a 60/20/20 train/validation/test split, tracking both curves simultaneously.

---

## Dataset

**UCI Banknote Authentication** — features extracted from banknote images via Wavelet Transform:

| Feature | Description |
|---|---|
| `variance` | Variation in pixel intensity |
| `skewness` | Asymmetry in the wavelet-transformed image |
| `curtosis` | Sharpness/flatness of distribution |
| `entropy` | Randomness in the image |
| `class` | 0 = forged, 1 = genuine |

- **1,372 samples** after deduplication (24 duplicates removed)
- **Standardized** with `StandardScaler` before training
- **Class balance:** ~55% forged, ~45% genuine

---

## Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Primal SVM (Hinge) | ~98% | ~98% |
| Dual SVM (RBF Kernel) | ~97% | ~97% |
| Squared Hinge SVM | ~98% | ~98% |

All three custom implementations match or approach sklearn's SVC performance on this dataset.

---

## Hyperparameter Tuning

Each model includes **k-fold cross-validation from scratch** using a combined accuracy-loss scoring function:

$$\text{score} = \text{accuracy} - \alpha \cdot \text{val\_loss}$$

This penalizes models that achieve high accuracy at the cost of poor generalization.

**Primal SVM:** best `lr=0.1`, `λ=0.0001`  
**Dual SVM:** best `lr=0.001`, `C=1`, `γ=10`  
**Squared Hinge:** best `lr=0.01`, `λ=0.01`

---

## Visualizations

- **2D decision boundary** with margin lines and support vector highlighting
- **3D scatter plots** colored by predicted class
- **PCA projections** (2D and 3D) for dimensionality-reduced visualization
- **Loss and accuracy curves** over training epochs
- **Convergence comparison** across all three models
- **Confusion matrices** and **inference time** comparison

---

## Project Structure

```
├── coceez.ipynb                        # Main notebook
├── data_banknote_authentication.csv    # Dataset
└── README.md
```

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
plotly
```

---

## Key Concepts Demonstrated

- Convex optimization and subgradient methods
- Primal vs. dual SVM formulations
- The kernel trick and Mercer's theorem
- KKT conditions and support vector identification
- Regularization and the bias-variance tradeoff
- Cross-validation design and combined scoring metrics
- Decision boundary geometry in 2D and 3D
