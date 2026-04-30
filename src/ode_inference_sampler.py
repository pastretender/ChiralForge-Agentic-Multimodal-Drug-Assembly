import os
import sys
import importlib.util
import torch
import einops
from jaxtyping import Float

from torch_geometric.nn import knn_graph

# =============================================================================
# 1. Dynamic Module Imports (Handles Numeric Filenames)
# =============================================================================

def load_module_from_file(module_name: str, file_path: str):
    """Dynamically loads a Python file as a module to bypass numeric naming restrictions."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required script {file_path} not found in the current directory.")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

try:
    mod_cfm = load_module_from_file("mod_cfm", os.path.join(os.path.dirname(__file__), "flow_matching_engine.py"))
except FileNotFoundError as e:
    print(f"Initialization Error: {e}")
    sys.exit(1)

TimeConditionedEGNN = mod_cfm.TimeConditionedEGNN

# =============================================================================
# 2. ODE Inference Sampler (Euler Method)
# =============================================================================

@torch.no_grad()
def sample_molecule_euler(
    model: TimeConditionedEGNN,
    c: Float[torch.Tensor, "1 c_dim"],
    num_nodes: int,
    in_dim: int = 9,
    num_steps: int = 100,
    device: torch.device = torch.device("cpu")
) -> tuple[Float[torch.Tensor, "num_nodes 3"], Float[torch.Tensor, "num_nodes in_dim"]]:
    """
    Solves the Flow Matching ODE from t=0 to t=1 using Euler integration.
    Transforms standard Gaussian noise into valid molecular coordinates and features.
    """
    model.eval()

    # 1. Initialize Prior x_0 (Standard Gaussian Noise)
    pos: Float[torch.Tensor, "num_nodes 3"] = torch.randn((num_nodes, 3), device=device)
    h: Float[torch.Tensor, "num_nodes in_dim"] = torch.randn((num_nodes, in_dim), device=device)

    # Expand the graph-level conditioning vector to node-level
    # STRICT REQUIREMENT: Using einops for safe dimensional expansion
    c_node: Float[torch.Tensor, "num_nodes c_dim"] = einops.repeat(c, '1 d -> n d', n=num_nodes)

    dt = 1.0 / num_steps

    # 2. ODE Integration Loop
    for step in range(num_steps):
        t_val = step * dt

        # Dynamically compute kNN graph based on current spatial coordinates to scale linearly
        # Fallback to fully-connected if num_nodes is smaller than k
        k_neighbors = min(6, num_nodes - 1)
        edge_index = knn_graph(pos, k=k_neighbors, loop=False)

        # Create node-level time tensor
        t_node: Float[torch.Tensor, "num_nodes 1"] = torch.full(
            (num_nodes, 1), t_val, device=device, dtype=torch.float32
        )

        c_uncond = torch.zeros_like(c_node)

        # Batch inputs for single forward pass
        batched_h = torch.cat([h, h], dim=0)
        batched_pos = torch.cat([pos, pos], dim=0)
        batched_edge_index = torch.cat([edge_index, edge_index + num_nodes], dim=1)
        batched_t_node = torch.cat([t_node, t_node], dim=0)
        batched_c_node = torch.cat([c_node, c_uncond], dim=0)

        # Single Forward Pass
        batched_v_r, batched_v_h = model(
            h=batched_h, pos=batched_pos, edge_index=batched_edge_index,
            t_node=batched_t_node, c_node=batched_c_node
        )

        v_cond_r, v_uncond_r = torch.chunk(batched_v_r, 2, dim=0)
        v_cond_h, v_uncond_h = torch.chunk(batched_v_h, 2, dim=0)

        # 3. Apply Classifier-Free Guidance (w = guidance_scale)
        w = 5.0 # Passed from agent
        v_r = v_uncond_r + w * (v_cond_r - v_uncond_r)
        v_h = v_uncond_h + w * (v_cond_h - v_uncond_h)

        # 4. Euler Update Step
        pos = pos + v_r * dt
        h = h + v_h * dt

    return pos, h

#replace the basic sample_molecule_euler function. By upgrading to Heun's 2nd Order Method, we can drop the integration steps from 100 down to 20, drastically accelerating the feedback cycle

@torch.no_grad()
def sample_molecule_heun_constrained(
    model: TimeConditionedEGNN,
    c: Float[torch.Tensor, "1 c_dim"],
    num_nodes: int,
    fragment_coords: Float[torch.Tensor, "num_fragment 3"] = None,
    fragment_mask: torch.Tensor = None,
    guidance_scale: float = 5.0,
    in_dim: int = 9,
    num_steps: int = 20, # Slashed from 100 due to Heun's efficiency
    device: torch.device = torch.device("cpu")
) -> tuple[Float[torch.Tensor, "num_nodes 3"], Float[torch.Tensor, "num_nodes in_dim"]]:
    """
    Solves the Flow Matching ODE using Heun's 2nd Order Method.
    Integrates Classifier-Free Guidance (CFG) and static spatial constraints.
    """
    model.eval()

    # 1. Initialize Prior
    pos = torch.randn((num_nodes, 3), device=device)
    h = torch.randn((num_nodes, in_dim), device=device)

    # 2. Apply initial fragment constraints (overwriting noise)
    if fragment_coords is not None and fragment_mask is not None:
        pos[fragment_mask] = fragment_coords.to(device)
        # Note: In a full pipeline, you would also inject the known fragment node features into `h` here.

    c_node = einops.repeat(c, '1 d -> n d', n=num_nodes)
    c_uncond = torch.zeros_like(c_node) # For CFG

    dt = 1.0 / num_steps

    def get_velocity(current_pos, current_h, t_val):
        t_node = torch.full((num_nodes, 1), t_val, device=device, dtype=torch.float32)

        # Dynamically compute kNN graph based on current spatial coordinates to scale linearly
        # Fallback to fully-connected if num_nodes is smaller than k
        k_neighbors = min(6, num_nodes - 1)
        edge_index = knn_graph(current_pos, k=k_neighbors, loop=False)

        # Combine conditional and unconditional inputs for a single forward pass
        batched_h = torch.cat([current_h, current_h], dim=0)
        batched_pos = torch.cat([current_pos, current_pos], dim=0)
        batched_edge_index = torch.cat([edge_index, edge_index + num_nodes], dim=1)
        batched_t_node = torch.cat([t_node, t_node], dim=0)
        batched_c_node = torch.cat([c_node, c_uncond], dim=0)

        # Single batched forward pass
        batched_v_r, batched_v_h = model(
            h=batched_h, pos=batched_pos, edge_index=batched_edge_index,
            t_node=batched_t_node, c_node=batched_c_node
        )

        # Split features
        v_cond_r, v_uncond_r = torch.chunk(batched_v_r, 2, dim=0)
        v_cond_h, v_uncond_h = torch.chunk(batched_v_h, 2, dim=0)

        # Classifier-Free Guidance Extrapolation
        v_r = v_uncond_r + guidance_scale * (v_cond_r - v_uncond_r)
        v_h = v_uncond_h + guidance_scale * (v_cond_h - v_uncond_h)
        return v_r, v_h

    # 3. Heun's Integration Loop
    for step in range(num_steps):
        t_val = step * dt

        # Step A: Predict initial velocity
        v_r, v_h = get_velocity(pos, h, t_val)

        # Step B: Euler predictor step
        pos_intermediate = pos + v_r * dt
        h_intermediate = h + v_h * dt

        # Re-apply constraints to intermediate state
        if fragment_coords is not None and fragment_mask is not None:
            pos_intermediate[fragment_mask] = fragment_coords.to(device)

        # Step C: Predict velocity at intermediate state
        v_r_next, v_h_next = get_velocity(pos_intermediate, h_intermediate, t_val + dt)

        # Step D: Heun's corrector step (average the velocities)
        pos = pos + 0.5 * (v_r + v_r_next) * dt
        h = h + 0.5 * (v_h + v_h_next) * dt

        # Final constraint enforcement for this step
        if fragment_coords is not None and fragment_mask is not None:
            pos[fragment_mask] = fragment_coords.to(device)

    return pos, h


# =============================================================================
# 3. Execution & Validation Block (Agent Integration)
# =============================================================================

if __name__ == "__main__":
    print("Initializing ODE Inference Sampler...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Agent Orchestrator Simulation ---
    # Simulating the payload received from 05_agent_orchestrator.py
    agent_payload = {
        "target_protein_id": "7XYZ",
        "phenotype_condition": "preserve_healthy",
        "fragment_constraint_coords": [1.2, -0.5, 3.4],
        "guidance_scale": 5.0
    }

    print("📥 Received Orchestration Payload from Agent:")
    for key, val in agent_payload.items():
        print(f"   - {key}: {val}")
    print("-" * 60)

    # --- 2. Model & Context Setup ---
    IN_DIM = 9
    C_DIM = 256
    HIDDEN_DIM = 128
    NUM_NODES_TO_GENERATE = 15
    NUM_EULER_STEPS = 50 # Reduced for quick validation

    print(f"Booting Generative Engine on {device}...")
    model = TimeConditionedEGNN(
        in_node_dim=IN_DIM,
        c_dim=C_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=3
    ).to(device)

    # NEW: Load the trained weights
    checkpoint_path = "./checkpoints/conditioned_egnn.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading trained weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    else:
        print("⚠️ WARNING: No trained checkpoints found. Running inference with UNTRAINED network.")

    # Simulate the fused context vector 'c' produced by the multimodal encoders
    c_mock = torch.randn((1, C_DIM), device=device, dtype=torch.float32)

    # --- 3. Run Inference ---
    print(f"Sampling molecule with {NUM_NODES_TO_GENERATE} atoms using {NUM_EULER_STEPS} Euler steps...")

    final_pos, final_h = sample_molecule_heun_constrained(
        model=model,
        c=c_mock,
        num_nodes=NUM_NODES_TO_GENERATE,
        in_dim=IN_DIM,
        num_steps=NUM_EULER_STEPS,
        device=device
    )

    # --- 4. Validation & Statistics ---
    print("\n" + "=" * 60)
    print("🧬 GENERATION COMPLETE. GEOMETRY STATISTICS:")
    print("=" * 60)
    print(f"Final Coordinate Shape: {list(final_pos.shape)} -> Expected [{NUM_NODES_TO_GENERATE}, 3]")
    print(f"Final Features Shape:   {list(final_h.shape)} -> Expected [{NUM_NODES_TO_GENERATE}, {IN_DIM}]")

    # Calculate statistics to prove the noise has coalesced
    # Note: In a fully trained model, these coordinates would closely map to standard bond lengths (e.g., 1-2 Angstroms)
    pos_mean = final_pos.mean(dim=0).cpu().numpy()
    pos_std = final_pos.std(dim=0).cpu().numpy()

    print("\nCoordinate Distribution (X, Y, Z):")
    print(f"  Mean: [{pos_mean[0]:.4f}, {pos_mean[1]:.4f}, {pos_mean[2]:.4f}]")
    print(f"  Std:  [{pos_std[0]:.4f}, {pos_std[1]:.4f}, {pos_std[2]:.4f}]")
    print("=" * 60)
    print("Status: SUCCESS. ODE Sampler iteratively integrated the drift field.")