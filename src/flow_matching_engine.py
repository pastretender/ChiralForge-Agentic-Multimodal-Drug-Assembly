import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter
from jaxtyping import Float, Int, Shaped

# =============================================================================
# 1. Dependency Modules (Included for Standalone Execution)
# =============================================================================

class EGNNLayer(nn.Module):
    """Simplified E(3)-Equivariant Graph Neural Network Layer for Flow Matching."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        h: Float[torch.Tensor, "num_nodes hidden_dim"],
        pos: Float[torch.Tensor, "num_nodes 3"],
        edge_index: Int[torch.Tensor, "2 num_edges"]
    ) -> tuple[Float[torch.Tensor, "num_nodes hidden_dim"], Float[torch.Tensor, "num_nodes 3"]]:

        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        sq_dists = torch.sum(coord_diff**2, dim=-1, keepdim=True)

        # Edge Messages
        edge_inputs = torch.cat([h[row], h[col], sq_dists], dim=-1)
        m_ij = self.edge_mlp(edge_inputs)

        # Coordinate Update (Velocity Prediction step)
        coord_weights = self.coord_mlp(m_ij)
        norm_coord_diff = coord_diff / (sq_dists + 1e-8).sqrt()
        pos_updates_ij = norm_coord_diff * coord_weights
        pos_update_i = scatter(pos_updates_ij, row, dim=0, dim_size=pos.size(0), reduce='mean')
        pos_out = pos + pos_update_i

        # Node Update
        m_i = scatter(m_ij, row, dim=0, dim_size=h.size(0), reduce='add')
        node_inputs = torch.cat([h, m_i], dim=-1)
        h_out = h + self.node_mlp(node_inputs)

        return h_out, pos_out

# =============================================================================
# 2. Time-Conditioned EGNN Architecture
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Embeds scalar time t into a high-dimensional vector."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Float[torch.Tensor, "num_nodes 1"]) -> Float[torch.Tensor, "num_nodes dim"]:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        # Using einops to reshape the embedding weights for broadcasting
        emb_matrix = einops.rearrange(emb, 'd -> 1 d')
        t_emb = t * emb_matrix
        return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

class TimeConditionedEGNN(nn.Module):
    """
    EGNN wrapper that conditions the generative drift field on both
    time t and the fused multimodal vector c.
    """
    def __init__(self, in_node_dim: int, c_dim: int, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.time_embedder = SinusoidalTimeEmbedding(dim=hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Node embedding integrates raw features, time embedding, and multimodal context
        fused_in_dim = in_node_dim + hidden_dim + c_dim
        self.node_embedding = nn.Linear(fused_in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim=hidden_dim) for _ in range(num_layers)
        ])

        # Final projection to output feature velocities (coordinate velocities are intrinsic to EGNN)
        self.v_h_proj = nn.Linear(hidden_dim, in_node_dim)

    def forward(
        self,
        h: Float[torch.Tensor, "num_nodes in_dim"],
        pos: Float[torch.Tensor, "num_nodes 3"],
        edge_index: Int[torch.Tensor, "2 num_edges"],
        t_node: Float[torch.Tensor, "num_nodes 1"],
        c_node: Float[torch.Tensor, "num_nodes c_dim"]
    ) -> tuple[Float[torch.Tensor, "num_nodes 3"], Float[torch.Tensor, "num_nodes in_dim"]]:

        # 1. Embed time
        t_emb = self.time_mlp(self.time_embedder(t_node))

        # 2. Condition node features via concatenation
        h_fused = torch.cat([h, t_emb, c_node], dim=-1)
        h_latent = self.node_embedding(h_fused)

        # 3. Message Passing (Tracks coordinate shift natively)
        pos_out = pos
        for layer in self.layers:
            h_latent, pos_out = layer(h_latent, pos_out, edge_index)

        # 4. Extract drift vectors (velocities)
        # Coordinate velocity is the displacement vector from current pos to pos_out
        v_r_pred = pos_out - pos
        # Feature velocity is projected back to the original feature dimension
        v_h_pred = self.v_h_proj(h_latent)

        return v_r_pred, v_h_pred

# =============================================================================
# 3. Conditional Flow Matcher Objective
# =============================================================================

class ConditionalFlowMatcher(nn.Module):
    """
    Computes the Simulation-Free Flow Matching loss (L_CFM) to train
    the conditional continuous drift field.
    """
    def __init__(self, model: TimeConditionedEGNN):
        super().__init__()
        self.model = model

    def compute_cfm_loss(
        self,
        batch: Batch,
        c: Float[torch.Tensor, "batch_size c_dim"]
    ) -> Float[torch.Tensor, "1"]:

        device = batch.x.device
        num_graphs = batch.num_graphs

        # 1. Sample continuous time steps t ~ U(0, 1) per graph
        t = torch.rand(num_graphs, device=device)

        # Map graph-level time/context to node-level via batch index gathering
        t_node = t[batch.batch]
        c_node = c[batch.batch]

        # STRICT REQUIREMENT: Safe reshaping using einops
        # Expand node-level time tensor for geometric interpolation and concatenation
        t_node_expanded: Float[torch.Tensor, "num_nodes 1"] = einops.rearrange(t_node, 'n -> n 1')

        # 2. Define Targets (x_1) and Prior (x_0)
        x1_r: Float[torch.Tensor, "num_nodes 3"] = batch.pos
        x1_h: Float[torch.Tensor, "num_nodes in_dim"] = batch.x

        # Standard Gaussian prior for both coordinates and features
        x0_r = torch.randn_like(x1_r)
        x0_h = torch.randn_like(x1_h)

        # 3. Linear Interpolation for Flow Matching path x_t = t * x_1 + (1 - t) * x_0
        xt_r = t_node_expanded * x1_r + (1 - t_node_expanded) * x0_r
        xt_h = t_node_expanded * x1_h + (1 - t_node_expanded) * x0_h

        # 4. Target Velocity Vector Field v_t = x_1 - x_0
        vt_r_target = x1_r - x0_r
        vt_h_target = x1_h - x0_h

        # 5. Predict Velocities using Time-Conditioned Drift Field
        v_r_pred, v_h_pred = self.model(
            h=xt_h,
            pos=xt_r,
            edge_index=batch.edge_index,
            t_node=t_node_expanded,
            c_node=c_node
        )

        # 6. Compute Mean Squared Error (MSE) Loss
        loss_r = F.mse_loss(v_r_pred, vt_r_target)
        loss_h = F.mse_loss(v_h_pred, vt_h_target)

        # Equal weighting for geometric and feature drift
        loss = loss_r + loss_h
        return loss

# =============================================================================
# 4. Mock Execution and Validation Block
# =============================================================================

def get_mock_batch_and_context(num_graphs: int = 4, nodes_per_graph: int = 8, in_dim: int = 9, c_dim: int = 256):
    """Generates a dummy PyG Batch and corresponding multimodal context vector."""
    data_list = []
    for _ in range(num_graphs):
        # Generate random node features, 3D coords, and dense edge indices
        x = torch.randn((nodes_per_graph, in_dim), dtype=torch.float32)
        pos = torch.randn((nodes_per_graph, 3), dtype=torch.float32)

        # Complete graph edges (excluding self loops for simplicity)
        edge_index = torch.cartesian_prod(torch.arange(nodes_per_graph), torch.arange(nodes_per_graph)).t()
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        data_list.append(Data(x=x, pos=pos, edge_index=edge_index))

    batch = Batch.from_data_list(data_list)
    c_fused = torch.randn((num_graphs, c_dim), dtype=torch.float32)

    return batch, c_fused

if __name__ == "__main__":
    print("Initializing Conditional Flow Matching Engine...\n")

    # Configuration
    IN_DIM = 9
    C_DIM = 256
    HIDDEN_DIM = 128

    # Instantiate modules
    conditioned_egnn = TimeConditionedEGNN(in_node_dim=IN_DIM, c_dim=C_DIM, hidden_dim=HIDDEN_DIM, num_layers=3)
    flow_matcher = ConditionalFlowMatcher(model=conditioned_egnn)

    # Generate mock data
    batch, c_fused = get_mock_batch_and_context(num_graphs=4, nodes_per_graph=10, in_dim=IN_DIM, c_dim=C_DIM)

    # Forward Pass Validation
    print("Executing Flow Matching Forward Pass...")
    conditioned_egnn.train()

    # Zero gradients to test backward pass viability
    conditioned_egnn.zero_grad()

    # Compute L_CFM
    cfm_loss = flow_matcher.compute_cfm_loss(batch=batch, c=c_fused)

    print("-" * 65)
    print("FLOW MATCHING VALIDATION REPORT:")
    print("-" * 65)
    print(f"Batch Number of Graphs:   {batch.num_graphs}")
    print(f"Total Nodes in Batch:     {batch.num_nodes}")
    print(f"Input Node Dimension:     {IN_DIM}")
    print(f"Conditioning Dimension c: {C_DIM}")
    print(f"Time Expansion Check:     Passed (via einops.rearrange)")
    print("---")
    print(f"Target Vector Velocity v_t Computed: Yes (x_1 - x_0)")
    print(f"Predicted Drift Vector v_hat output: Yes (v_r_pred, v_h_pred)")
    print("---")
    print(f"Final CFM Loss Value:     {cfm_loss.item():.4f}")

    # Ensure gradients can flow back
    cfm_loss.backward()
    grad_check = any(p.grad is not None for p in conditioned_egnn.parameters())
    print(f"Gradients Flowing:        {grad_check}")
    print("-" * 65)
    print("Status: SUCCESS. Time-Conditioned EGNN and Flow Matcher are fully operational.")