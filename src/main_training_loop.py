import os
import sys
import importlib.util
import torch
from torch.optim import AdamW

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
    mod_enc = load_module_from_file("mod_enc", os.path.join(os.path.dirname(__file__), "multimodal_encoders.py"))
    mod_data = load_module_from_file("mod_data", os.path.join(os.path.dirname(__file__), "data_pipeline.py"))
    mod_cfm = load_module_from_file("mod_cfm", os.path.join(os.path.dirname(__file__), "flow_matching_engine.py"))
except FileNotFoundError as e:
    print(f"Initialization Error: {e}")
    sys.exit(1)

# Extract required classes and functions
CryoEMEncoder = mod_enc.CryoEMEncoder
HCSViTEncoder = mod_enc.HCSViTEncoder
CrossModalFusionModule = mod_enc.CrossModalFusionModule

get_multimodal_dataloaders = mod_data.get_multimodal_dataloaders

TimeConditionedEGNN = mod_cfm.TimeConditionedEGNN
ConditionalFlowMatcher = mod_cfm.ConditionalFlowMatcher
get_mock_batch_and_context = mod_cfm.get_mock_batch_and_context

# =============================================================================
# 2. Main Training Loop Setup & Execution
# =============================================================================

def train_end_to_end(num_epochs: int = 5, batch_size: int = 4, hidden_dim: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing End-to-End Training on device: {device}\n")

    # 1. Initialize Encoders and Fusion Layer
    cryo_encoder = CryoEMEncoder(in_channels=1, hidden_dim=hidden_dim).to(device)
    hcs_encoder = HCSViTEncoder(in_channels=3, hidden_dim=hidden_dim).to(device)
    fusion_module = CrossModalFusionModule(hidden_dim=hidden_dim, num_heads=4).to(device)

    # 2. Initialize Core Generative Engine (Flow Matching)
    # Using in_dim=9 as default for OGB node features
    conditioned_egnn = TimeConditionedEGNN(
        in_node_dim=9,
        c_dim=hidden_dim,
        hidden_dim=128,
        num_layers=3
    ).to(device)

    flow_matcher = ConditionalFlowMatcher(model=conditioned_egnn)

    # 3. Configure Single Optimizer for End-to-End Training
    all_parameters = (
        list(cryo_encoder.parameters()) +
        list(hcs_encoder.parameters()) +
        list(fusion_module.parameters()) +
        list(conditioned_egnn.parameters())
    )
    optimizer = AdamW(all_parameters, lr=1e-4, weight_decay=1e-5)

    # 4. Initialize DataLoaders
    # This triggers the fail-safe mock data generation if real data isn't present
    cryo_loader, hcs_loader = get_multimodal_dataloaders(batch_size=batch_size)

    # Initialize GradScaler for mixed precision training
    scaler = torch.amp.GradScaler()

    print("=" * 60)
    print("🚀 STARTING TRAINING LOOP")
    print("=" * 60)

    # 5. The Training Loop
    cryo_encoder.train()
    hcs_encoder.train()
    fusion_module.train()
    conditioned_egnn.train()

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # Zip the multimodal dataloaders to synchronize structural and phenotypic batches
        for step, (cryo_batch, hcs_batch) in enumerate(zip(cryo_loader, hcs_loader)):

            # Move inputs to device
            cryo_batch = cryo_batch.to(device)
            hcs_batch = hcs_batch.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # --- FORWARD PASS: Context & Encoders ---
                # 1. Extract structural conditioning
                c_struct = cryo_encoder(cryo_batch)

                # 2. Extract phenotypic conditioning
                c_pheno = hcs_encoder(hcs_batch)

                # 3. Cross-modal fusion into unified vector c
                c_fused = fusion_module(c_struct, c_pheno)

                # INNOVATION PATCH: 10% Unconditional Dropout for CFG
                if torch.rand(1).item() < 0.1:
                    c_fused = torch.zeros_like(c_fused)

                # --- FORWARD PASS: Generative Engine ---
                # Fetch a geometric target graph batch.
                # In a full pipeline, these would be the ground truth molecules paired with the cryo/hcs data.
                # Here, we use the mock generator from script 04 to simulate target topologies.
                current_batch_size = c_fused.shape[0]
                mol_batch, _ = get_mock_batch_and_context(
                    num_graphs=current_batch_size,
                    nodes_per_graph=12,
                    in_dim=9,
                    c_dim=hidden_dim
                )
                mol_batch = mol_batch.to(device)

                # 4. Compute Conditional Flow Matching Loss
                loss = flow_matcher.compute_cfm_loss(batch=mol_batch, c=c_fused)

            # --- BACKWARD PASS & OPTIMIZATION ---
            scaler.scale(loss).backward()

            # Gradient clipping to stabilize deep geometric/multimodal architectures
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # --- LOGGING ---
            if step % 2 == 0 or step == len(cryo_loader) - 1:
                print(f"Step [{step:03d}/{len(cryo_loader):03d}] | L_CFM (Flow Matching Loss): {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("💾 SAVING MODEL CHECKPOINTS")
    print("=" * 60)

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(cryo_encoder.state_dict(), "./checkpoints/cryo_encoder.pt")
    torch.save(hcs_encoder.state_dict(), "./checkpoints/hcs_encoder.pt")
    torch.save(fusion_module.state_dict(), "./checkpoints/fusion_module.pt")
    torch.save(conditioned_egnn.state_dict(), "./checkpoints/conditioned_egnn.pt")

    print("Checkpoints successfully saved to ./checkpoints/")

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE.")


if __name__ == "__main__":
    # Execute the training sequence for 3 epochs to validate backward flow
    train_end_to_end(num_epochs=3, batch_size=4)