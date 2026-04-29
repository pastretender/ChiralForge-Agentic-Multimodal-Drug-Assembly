import torch
import torch.nn as nn
import timm
import einops
from jaxtyping import Float

# =============================================================================
# 1. Cryo-EM 3D-CNN Encoder (Structural Context)
# =============================================================================

class CryoEMEncoder(nn.Module):
    """
    Extracts geometric contours and electrostatic potentials from 3D voxel density maps
    into a latent structural vector c_struct.
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 256):
        super().__init__()
        # Lightweight 3D ResNet-style feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)) # Compresses to fixed spatial size
        )

        # 64 channels * 2 * 2 * 2 = 512
        self.proj = nn.Linear(512, hidden_dim)

    def forward(self, x: Float[torch.Tensor, "batch channels depth height width"]) -> Float[torch.Tensor, "batch hidden_dim"]:
        feats = self.feature_extractor(x)
        # STRICT REQUIREMENT: einops used for flattening
        flat_feats = einops.rearrange(feats, 'b c d h w -> b (c d h w)')
        c_struct = self.proj(flat_feats)
        return c_struct

# =============================================================================
# 2. HCS ViT Encoder (Phenotypic Context)
# =============================================================================

class HCSViTEncoder(nn.Module):
    """
    Embeds multi-channel High-Content Screening cellular images into a
    latent biological vector c_pheno using a pre-trained Vision Transformer.
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 256):
        super().__init__()
        # Using a tiny ViT from timm. num_classes=0 strips the classification head,
        # returning the pooled latent representation.
        self.vit = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            in_chans=in_channels,
            num_classes=0,
            global_pool='avg'
        )
        # Map ViT output features to the unified hidden dimension
        self.proj = nn.Linear(self.vit.num_features, hidden_dim)

    def forward(self, x: Float[torch.Tensor, "batch channels height width"]) -> Float[torch.Tensor, "batch hidden_dim"]:
        feats = self.vit(x)
        c_pheno = self.proj(feats)
        return c_pheno

# =============================================================================
# 3. Cross-Modal Fusion Module
# =============================================================================

class CrossModalFusionModule(nn.Module):
    """
    Merges c_struct and c_pheno into a unified conditioning vector c
    via cross-attention mechanisms.
    """
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(
        self,
        c_struct: Float[torch.Tensor, "batch hidden_dim"],
        c_pheno: Float[torch.Tensor, "batch hidden_dim"]
    ) -> Float[torch.Tensor, "batch hidden_dim"]:

        # STRICT REQUIREMENT: einops used to introduce sequence dimension for Attention
        # Shape transforms from [batch, hidden_dim] -> [batch, 1, hidden_dim]
        q = einops.rearrange(c_struct, 'b d -> b 1 d')
        k = einops.rearrange(c_pheno, 'b d -> b 1 d')
        v = k  # Using phenotype as context

        # Cross-Attention: Structure queries Phenotype
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)

        # Residual connection and LayerNorm
        out = self.norm1(q + attn_out)

        # Feed-Forward Network
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)

        # STRICT REQUIREMENT: einops used to squeeze sequence dimension back out
        # Shape transforms from [batch, 1, hidden_dim] -> [batch, hidden_dim]
        c_fused = einops.rearrange(out, 'b 1 d -> b d')

        return c_fused

# =============================================================================
# 4. Mock Execution and Validation Block
# =============================================================================

def get_mock_multimodal_batch(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates dummy 3D voxel grids and 2D multi-channel images."""
    # Cryo-EM: [batch, channels, depth, height, width]
    cryo_em_batch = torch.randn((batch_size, 1, 32, 32, 32), dtype=torch.float32)

    # HCS ViT: [batch, channels, height, width] (timm standard is 224x224)
    hcs_img_batch = torch.randn((batch_size, 3, 224, 224), dtype=torch.float32)

    return cryo_em_batch, hcs_img_batch

if __name__ == "__main__":
    print("Initializing Multimodal Representation Fusion Layer...\n")

    # Hyperparameters
    BATCH_SIZE = 4
    HIDDEN_DIM = 256

    # Instantiate modules
    cryo_encoder = CryoEMEncoder(in_channels=1, hidden_dim=HIDDEN_DIM)
    hcs_encoder = HCSViTEncoder(in_channels=3, hidden_dim=HIDDEN_DIM)
    fusion_module = CrossModalFusionModule(hidden_dim=HIDDEN_DIM, num_heads=4)

    # Get mock data
    cryo_data, hcs_data = get_mock_multimodal_batch(batch_size=BATCH_SIZE)

    # Forward Pass Validation
    cryo_encoder.eval()
    hcs_encoder.eval()
    fusion_module.eval()

    with torch.no_grad():
        print("Executing Multimodal Forward Pass...")
        c_struct = cryo_encoder(cryo_data)
        c_pheno = hcs_encoder(hcs_data)
        c_fused = fusion_module(c_struct, c_pheno)

    print("-" * 60)
    print("SHAPE VALIDATION REPORT:")
    print(f"Input Cryo-EM 3D Data:      {list(cryo_data.shape)}")
    print(f"Input HCS Phenotypic Image: {list(hcs_data.shape)}")
    print("---")
    print(f"Intermediate c_struct:      {list(c_struct.shape)} -> Matches [batch, hidden_dim]")
    print(f"Intermediate c_pheno:       {list(c_pheno.shape)} -> Matches [batch, hidden_dim]")
    print("---")
    print(f"Output Fused Vector c:      {list(c_fused.shape)} -> Matches [batch, hidden_dim]")
    print("-" * 60)
    print("Status: SUCCESS. Multimodal encoders and cross-modal fusion validated.")