# Assignment 3: Experimentation and Expansion
## FAST-NUCES — Deep Learning (DS/AI)
**Authors:** Yahya Abdul Hakeem (23I-2627) · Mustafa Kashif (23I-2583) · Ishmal Faheem (23I-5032)  
**Instructor:** Mr. Ubaid Ul Rehman

---

## Overview

This assignment extends Assignment 2 (RecDCL reproduction) in two directions:

1. **Mandatory — Cross-dataset evaluation** on Amazon Beauty
2. **Proposed method — AdaptRecDCL** with GradNorm-based adaptive loss weighting

---

## Proposed Method: AdaptRecDCL

### Motivation
In Assignment 2, we observed that RecDCL's fixed loss weights (λ_BCL=0.05, λ_FCL=0.05)
caused training collapse at full scale. The root cause: BCL's InfoNCE loss produces
gradients ~7× larger than BPR at batch_size=2048. With the wrong λ, the contrastive
objective overwhelms the ranking objective.

### Solution
Replace fixed λ with **learnable log-scale parameters** balanced via **GradNorm**
(Chen et al., ICML 2018).

**How it works:**
1. λ_BCL and λ_FCL are initialised to 0.05 but treated as trainable parameters
2. Every 5 training steps, a GradNorm auxiliary loss is computed:
   - Measure gradient norm each task places on the shared projection layer
   - Penalise any task whose gradient deviates from the balanced average
3. A separate small-LR optimiser updates only the λ parameters
4. λ values are clamped to [1e-4, 1.0] to stay in a reasonable range

**Reference:** Z. Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss
Balancing in Deep Multitask Networks," ICML 2018.

---

## Datasets

| Dataset | Users | Items | Density | Split |
|---------|-------|-------|---------|-------|
| Gowalla (subset) | 3,000 | ~33K | 0.08% | 80/20 |
| Amazon Beauty | ~22K | ~12K | ~0.05% | 80/20 |

Both use 10-core filtering (≥10 interactions per user and item), matching the
preprocessing used in the RecDCL paper.

---

## Models

| Model | Description |
|-------|-------------|
| MF | Matrix Factorization + BPR loss |
| LightGCN | 3-layer graph convolution + BPR loss |
| RecDCL | LightGCN + BCL + FCL (fixed λ=0.05) |
| **AdaptRecDCL** | LightGCN + BCL + FCL (**adaptive λ via GradNorm**) |

---

## How to Run

1. Open `Assignment3_AdaptRecDCL_FIXED.ipynb` in Google Colab
2. Set runtime: Runtime → Change runtime type → **T4 GPU**
3. Run all cells in order (Runtime → Run all)
4. Final cell downloads all results automatically

**Expected runtime on T4:** ~55 minutes total

---

## Output Files

| File | Description |
|------|-------------|
| `all_results.json` | All metrics and training histories |
| `curves_Gowalla.png` | Training curves — all 4 models on Gowalla |
| `curves_Amazon_Beauty.png` | Training curves — all 4 models on Beauty |
| `lambda_evolution_Gowalla.png` | How λ_BCL and λ_FCL adapt during training (Gowalla) |
| `lambda_evolution_Amazon_Beauty.png` | Same for Beauty |
| `cross_dataset_comparison.png` | Bar chart comparing all models across both datasets |

---

## Key Hyperparameters

| Parameter | Gowalla | Beauty |
|-----------|---------|--------|
| Embedding size | 64 | 64 |
| GNN layers | 3 | 3 |
| Batch size | 2048 | 2048 |
| Max epochs | 200 | 300 |
| Early stop patience | 40 | 40 |
| Eval every | 5 epochs | 5 epochs |
| LR (MF) | 5e-3 | 1e-3 |
| LR (LightGCN) | 1e-3 | 1e-3 |
| LR (RecDCL) | 3e-3 | 1e-3 |
| LR (AdaptRecDCL) | 3e-3 | 1e-3 |
| λ_BCL (RecDCL fixed) | 0.05 | 0.05 |
| λ_FCL (RecDCL fixed) | 0.05 | 0.05 |
| GradNorm α | 1.5 | 1.5 |
| Adapt LR | 0.01 | 0.01 |

---

## Repository Structure

```
Assignment3/
├── Assignment3_AdaptRecDCL_FIXED.ipynb   ← Main notebook (run this)
├── README.md                              ← This file
└── results_a3/                            ← Generated after running
    ├── all_results.json
    ├── curves_Gowalla.png
    ├── curves_Amazon_Beauty.png
    ├── lambda_evolution_Gowalla.png
    ├── lambda_evolution_Amazon_Beauty.png
    ├── cross_dataset_comparison.png
    └── *_best.pt                          ← Best model checkpoints
```

---

## References

1. Zhang et al., "RecDCL: Dual Contrastive Learning for Recommendation," WWW 2024
2. He et al., "LightGCN: Simplifying and Powering GCN for Recommendation," SIGIR 2020
3. Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing," ICML 2018
4. Cho et al., "Friendship and Mobility," KDD 2011 (Gowalla dataset)
5. McAuley et al., "Inferring Networks of Substitutable and Complementary Products," KDD 2015 (Amazon Beauty)
