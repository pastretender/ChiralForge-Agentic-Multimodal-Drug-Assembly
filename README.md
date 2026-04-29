
# ChiralForge: Agentic Multimodal Drug Assembly

ChiralForge is an advanced computational pipeline for generating targeted drug candidates. It combines geometric deep learning, multimodal biological data fusion, and agentic LLM orchestration to synthesize molecules constrained by 3D structural data (Cryo-EM) and 2D phenotypic imaging (High-Content Screening).

## 🚀 Key Features

* **Intelligent Agentic Orchestration:** Utilizes LangChain to interpret complex natural language biological queries (e.g., target proteins, desired phenotypes, spatial coordinates) and dynamically map them to the generative inference server.
* **Multimodal Representation Fusion:** Integrates structural context from 3D voxel density maps via a 3D-CNN (`CryoEMEncoder`) and biological context from multi-channel cellular images via a Vision Transformer (`HCSViTEncoder`), merged using Cross-Attention mechanisms.
* **SE(3)-Equivariant Flow Matching:** Powers the generation using a Time-Conditioned E(3)-Equivariant Graph Neural Network (EGNN), patched with cross-product operations for critical chirality awareness. 
* **Constrained ODE Sampling:** Synthesizes final molecular coordinates and node features using Heun's 2nd Order ODE integration, seamlessly applying spatial fragment constraints and Classifier-Free Guidance (CFG).

## 🗂️ Repository Structure

* `src/agent_orchestrator.py`: LangChain-powered mock/production agent that maps biological objectives to Flow Matching API calls.
* `src/data_pipeline.py`: Synchronized PyTorch DataLoaders for Cryo-EM and HCS datasets (includes a fail-safe mock data generator).
* `src/multimodal_encoders.py`: 3D-CNN, timm-based ViT, and Cross-Modal Fusion modules.
* `src/egnn_baseline.py`: Foundational EGNN baseline architecture.
* `src/flow_matching_engine.py`: Defines the `TimeConditionedEGNN` and `ConditionalFlowMatcher` for Simulation-Free Flow Matching.
* `src/main_training_loop.py`: End-to-end multi-epoch training script integrating all encoders and the generative engine.
* `src/ode_inference_sampler.py`: High-efficiency constrained molecule samplers (Euler and Heun methods) resolving the learned continuous drift field.

## 🛠️ Installation

The core dependencies include PyTorch (`2.6.0`), Torch Geometric, e3nn, timm, and LangChain.

**Option 1: Using Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate mfm_drug_gen
```

**Option 2: Using pip**
```bash
pip install -r requirements.txt
```

## 💻 Usage Pipeline

**1. End-to-End Training**
To train the multimodal encoders alongside the flow matching drift field, run the main training loop. The data pipeline will automatically generate mock tensors if the `/data` directories are empty.
```bash
python src/main_training_loop.py
```

**2. Agentic Orchestration**
Prompt the intelligent orchestrator with biological constraints to configure generation parameters dynamically:
```bash
python src/agent_orchestrator.py
```

**3. ODE Generation & Inference**
Sample the generated molecules by running the inference ODE solver, which reads the saved checkpoints and extrapolates spatial coordinates:
```bash
python src/ode_inference_sampler.py
```

## 📄 License

This project is open-sourced under the **Apache License, Version 2.0**. You may not use the files except in compliance with the License. For full terms and conditions, see the `LICENSE` file.
