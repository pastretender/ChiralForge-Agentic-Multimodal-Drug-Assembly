# Multimodal Flow Matching (MFM) — Structure-Phenotype Drug Generation

A research codebase for *de novo* drug candidate generation that jointly conditions on 3D protein structure (Cryo-EM) and cellular phenotype (High-Content Screening), using E(3)-equivariant conditional flow matching as the generative backbone.

---

## Overview

Classical structure-based drug design conditions generation on a target protein alone. MFM extends this by fusing two complementary biological signals:

| Modality | Source | Encoder | Produces |
|---|---|---|---|
| 3D structural density | Cryo-EM voxel maps | 3D ResNet CNN | `c_struct` |
| Cellular phenotype | HCS multi-channel images | Vision Transformer (ViT) | `c_pheno` |

These are fused into a unified conditioning vector **c** that drives a time-conditioned E(3)-equivariant graph network (EGNN) trained with the Conditional Flow Matching (CFM) objective. At inference, an ODE sampler integrates the learned drift field from Gaussian noise to a valid molecular geometry — with optional fragment constraints and classifier-free guidance (CFG).

---

## Architecture

```
Cryo-EM Map  ──► CryoEMEncoder (3D ResNet) ──► c_struct ──┐
                                                            ├──► CrossModalFusionModule ──► c
HCS Image    ──► HCSViTEncoder (ViT-Tiny)  ──► c_pheno  ──┘
                                                            │
                                                            ▼
Gaussian Noise x_0  ──► TimeConditionedEGNN ◄── t, c ──► v̂(x_t, t, c)
                              │
                              ▼
                    ODE Integrator (Heun 2nd Order)
                              │
                              ▼
                    Generated Molecule (pos, h)
```

### Key components

**`multimodal_encoders.py`** — `CryoEMEncoder`, `HCSViTEncoder`, `CrossModalFusionModule`
Encodes each biological modality independently into a shared `hidden_dim`-dimensional space, then merges them via a two-layer MLP with LayerNorm.

**`flow_matching_engine.py`** — `TimeConditionedEGNN`, `ConditionalFlowMatcher`
The generative core. Uses sinusoidal time embeddings concatenated with the fused context vector to condition each EGNN message-passing step. Trained with the simulation-free CFM objective:

```
L_CFM = E[ ||v̂(x_t, t, c) − (x_1 − x_0)||² ]
where x_t = t·x_1 + (1−t)·x_0,  t ~ U(0,1)
```

**`egnn_baseline.py`** — `EGNNLayer`, `EGNNModel`
Standalone E(3)-equivariant baseline with an SE(3) chirality extension: cross-product of the pairwise relative vector and center-of-mass displacement is appended to each edge message to break reflection symmetry.

**`ode_inference_sampler.py`** — `sample_molecule_euler`, `sample_molecule_heun_constrained`
Two ODE solvers for inference. The Heun 2nd-order method reduces integration steps from 100 → 20 with equivalent quality. Supports static fragment constraints (known active scaffolds) applied at each integration step, and CFG with guidance scale **w**:

```
v = v_uncond + w · (v_cond − v_uncond)
```

**`data_pipeline.py`** — `CryoEMDataset`, `HCSDataset`, `PairedMultimodalDataset`
PyTorch `Dataset` wrappers for both modalities with a built-in fail-safe that generates mock tensors to disk when no real data directory is found, keeping the full pipeline runnable in isolation.

**`agent_orchestrator.py`** — `MockBiologyAgent`, `generate_targeted_drug`
A LangChain-compatible tool-calling layer. The `generate_targeted_drug` tool accepts biological objectives as natural language and routes structured arguments (protein ID, phenotype condition, fragment coordinates, guidance scale) to the inference server. Swap `MockBiologyAgent` for a live `ChatOpenAI` instance with an API key to enable real LLM orchestration.

**`main_training_loop.py`** — `train_end_to_end`
End-to-end training over the full encoder → fusion → flow-matching stack with AdamW, mixed-precision (`torch.amp`), gradient clipping, and 10% unconditional dropout for CFG training.

---

## Installation

```bash
# Python 3.10+ recommended
pip install torch torchvision torch-geometric
pip install timm einops jaxtyping
pip install langchain langchain-openai
pip install ogb  # for the molhiv baseline dataset (optional)
```

---

## Quickstart

### Validate each module independently

```bash
# Data pipeline
python data_pipeline.py

# Multimodal encoders
python multimodal_encoders.py

# EGNN baseline (downloads ogbg-molhiv or falls back to mock)
python egnn_baseline.py

# Flow matching forward/backward pass
python flow_matching_engine.py

# Full training loop (mock data, 3 epochs)
python main_training_loop.py

# ODE inference sampler
python ode_inference_sampler.py

# Agent orchestrator
python agent_orchestrator.py
```

### End-to-end training

```python
from main_training_loop import train_end_to_end

train_end_to_end(num_epochs=10, batch_size=4, hidden_dim=256)
# Checkpoints saved to ./checkpoints/
```

### Inference from a trained checkpoint

```python
import torch
from flow_matching_engine import TimeConditionedEGNN
from ode_inference_sampler import sample_molecule_heun_constrained

model = TimeConditionedEGNN(in_node_dim=9, c_dim=256, hidden_dim=128, num_layers=3)
model.load_state_dict(torch.load("./checkpoints/conditioned_egnn.pt"))

# c comes from your CryoEMEncoder + HCSViTEncoder + CrossModalFusionModule pipeline
c = torch.randn(1, 256)  # replace with real fused context

fragment_coords = torch.tensor([[1.2, -0.5, 3.4]])
fragment_mask = torch.tensor([0])  # node indices to constrain

pos, h = sample_molecule_heun_constrained(
    model=model,
    c=c,
    num_nodes=15,
    fragment_coords=fragment_coords,
    fragment_mask=fragment_mask,
    guidance_scale=5.0,
    num_steps=20,
)
```

### Agent-driven generation

```python
# With a real LLM (requires OPENAI_API_KEY)
from agent_orchestrator import get_production_agent, generate_targeted_drug
from langchain_core.messages import HumanMessage

agent = get_production_agent()
response = agent.invoke([HumanMessage(content=(
    "Target the binding pocket of 7XYZ, preserve the healthy phenotype, "
    "and anchor generation to the known fragment at [1.2, -0.5, 3.4]."
))])
```

---

## Data

Place real biological data in the following directories before training (the pipeline auto-generates mock tensors if the directories are empty):

```
./data/
├── cryo_em/          # *.pt files — shape (1, 32, 32, 32) float32 voxel grids
└── hcs_images/       # *.pt files — shape (3, 224, 224) float32 multi-channel images
```

Files are sorted lexicographically and paired by index via `PairedMultimodalDataset`, so filenames must correspond across both directories (e.g., `sample_0000.pt` ↔ `sample_0000.pt`).

For the EGNN baseline, the `ogbg-molhiv` dataset from the Open Graph Benchmark is fetched automatically on first run.

---

## Configuration

All key hyperparameters are set inline at module entry points. The main ones:

| Parameter | Default | Location |
|---|---|---|
| `hidden_dim` | 256 | `main_training_loop.py` |
| `num_layers` | 3 | `flow_matching_engine.py` |
| `num_steps` (Heun) | 20 | `ode_inference_sampler.py` |
| `guidance_scale` (w) | 5.0 | `ode_inference_sampler.py` |
| CFG dropout rate | 0.10 | `main_training_loop.py` |
| `batch_size` | 4 | `main_training_loop.py` |
| `lr` | 1e-4 | `main_training_loop.py` |

---

## Research Notes

- **CFG training**: 10% of training steps zero out `c_fused` to train the unconditional drift field, enabling guidance-scale extrapolation at inference without a separately trained unconditional model.
- **Fragment constraints**: Hard-pinning known active fragments at each Heun step is a greedy enforcement strategy. A softer alternative is to add a constraint energy term to the CFM loss.
- **Chirality**: The SE(3) cross-product extension in `egnn_baseline.py` breaks the reflection equivariance of the base EGNN, making it sensitive to molecular chirality — important for drug binding.
- **kNN graph**: The inference sampler rebuilds the molecular graph dynamically each step using `knn_graph(pos, k=6)`, so topology adapts as the structure coalesces rather than being fixed at initialization.

---

## Project Structure

```
.
├── agent_orchestrator.py       # LangChain tool-calling layer
├── data_pipeline.py            # Cryo-EM and HCS dataset loaders
├── egnn_baseline.py            # Standalone E(3)-equivariant GNN baseline
├── flow_matching_engine.py     # CFM objective + time-conditioned EGNN
├── main_training_loop.py       # End-to-end training script
├── multimodal_encoders.py      # 3D CNN, ViT, and cross-modal fusion
├── ode_inference_sampler.py    # Euler and Heun ODE samplers
├── data/                       # Auto-created; holds Cryo-EM and HCS tensors
└── checkpoints/                # Auto-created; holds saved model weights
```
