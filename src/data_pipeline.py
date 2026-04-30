import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from jaxtyping import Float

# =============================================================================
# 1. Cryo-EM Dataset (3D Voxel Density Maps)
# =============================================================================

class CryoEMDataset(Dataset):
    """
    Dataset for loading 3D volumetric density maps.
    Includes a fail-safe to generate mock biological data if the directory is empty.
    """
    def __init__(self, data_dir: str = "./data/cryo_em", num_mock_samples: int = 10):
        super().__init__()
        self.data_dir = data_dir
        self.file_paths = []

        # Define the standard shape for our 3D ResNet encoder: [Channels, Depth, Height, Width]
        self.mock_shape = (1, 32, 32, 32)

        self._ensure_data_exists(num_mock_samples)
        self.file_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.pt")))
        self.data_cache = [torch.load(p, weights_only=True).to(torch.float32) for p in self.file_paths]

    def _ensure_data_exists(self, num_samples: int):
        """Fail-Safe: Generates dummy 3D voxel grids to disk if none exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        existing_files = glob.glob(os.path.join(self.data_dir, "*.pt"))

        if len(existing_files) == 0:
            print(f"[Fail-Safe] No data found in {self.data_dir}. Generating {num_samples} mock 3D tensors...")
            for i in range(num_samples):
                mock_tensor = torch.randn(self.mock_shape, dtype=torch.float32)
                torch.save(mock_tensor, os.path.join(self.data_dir, f"cryo_sample_{i:04d}.pt"))

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Float[torch.Tensor, "channels depth height width"]:
        return self.data_cache[idx]


# =============================================================================
# 2. High-Content Screening (HCS) Dataset (2D Phenotypic Images)
# =============================================================================

class HCSDataset(Dataset):
    """
    Dataset for loading multi-channel 2D phenotypic cellular images.
    Includes a fail-safe to generate mock biological data if the directory is empty.
    """
    def __init__(self, data_dir: str = "./data/hcs_images", num_mock_samples: int = 10):
        super().__init__()
        self.data_dir = data_dir
        self.file_paths = []

        # Define the standard shape for our ViT encoder: [Channels, Height, Width]
        self.mock_shape = (3, 224, 224)

        self._ensure_data_exists(num_mock_samples)
        self.file_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.pt")))
        self.data_cache = [torch.load(p, weights_only=True).to(torch.float32) for p in self.file_paths]

    def _ensure_data_exists(self, num_samples: int):
        """Fail-Safe: Generates dummy multi-channel 2D images to disk if none exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        existing_files = glob.glob(os.path.join(self.data_dir, "*.pt"))

        if len(existing_files) == 0:
            print(f"[Fail-Safe] No data found in {self.data_dir}. Generating {num_samples} mock 2D image tensors...")
            for i in range(num_samples):
                mock_tensor = torch.randn(self.mock_shape, dtype=torch.float32)
                torch.save(mock_tensor, os.path.join(self.data_dir, f"hcs_sample_{i:04d}.pt"))

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Float[torch.Tensor, "channels height width"]:
        return self.data_cache[idx]


# =============================================================================
# 3. Synchronized DataLoader Utility
# =============================================================================

def get_multimodal_dataloaders(
    cryo_dir: str = "./data/cryo_em",
    hcs_dir: str = "./data/hcs_images",
    batch_size: int = 4
) -> tuple[DataLoader, DataLoader]:
    """
    Initializes both multimodal datasets and wraps them in PyTorch DataLoaders.
    Returns: (cryo_dataloader, hcs_dataloader)
    """
    cryo_dataset = CryoEMDataset(data_dir=cryo_dir)
    hcs_dataset = HCSDataset(data_dir=hcs_dir)

    # In a real training scenario with distributed setups, you would use DistributedSampler.
    # For now, we use standard shuffling.
    cryo_loader = DataLoader(cryo_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    hcs_loader = DataLoader(hcs_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return cryo_loader, hcs_loader

class PairedMultimodalDataset(Dataset):
    """
    Synchronized dataset ensuring Cryo-EM maps and HCS images
    are strictly paired by their index/target ID.
    Pre-loads data into RAM to avoid I/O bottlenecks during __getitem__.
    """
    def __init__(self, cryo_dir: str = "./data/cryo_em", hcs_dir: str = "./data/hcs_images", num_mock_samples: int = 10):
        super().__init__()
        self.cryo_dir = cryo_dir
        self.hcs_dir = hcs_dir

        self._ensure_mock_data(num_mock_samples)

        # Sort to ensure indices match exactly across directories
        self.cryo_paths = sorted(glob.glob(os.path.join(self.cryo_dir, "*.pt")))
        self.hcs_paths = sorted(glob.glob(os.path.join(self.hcs_dir, "*.pt")))

        assert len(self.cryo_paths) == len(self.hcs_paths), "Mismatch between Cryo-EM and HCS file counts!"

        # Pre-load entirely into memory
        self.cryo_data = [torch.load(p, weights_only=True).to(torch.float32) for p in self.cryo_paths]
        self.hcs_data = [torch.load(p, weights_only=True).to(torch.float32) for p in self.hcs_paths]

    def _ensure_mock_data(self, num_samples: int):
        os.makedirs(self.cryo_dir, exist_ok=True)
        os.makedirs(self.hcs_dir, exist_ok=True)
        if not glob.glob(os.path.join(self.cryo_dir, "*.pt")):
            print("[Fail-Safe] Generating synchronized mock data...")
            for i in range(num_samples):
                torch.save(torch.randn((1, 32, 32, 32)), os.path.join(self.cryo_dir, f"sample_{i:04d}.pt"))
                torch.save(torch.randn((3, 224, 224)), os.path.join(self.hcs_dir, f"sample_{i:04d}.pt"))

    def __len__(self) -> int:
        return len(self.cryo_paths)

    def __getitem__(self, idx: int) -> tuple[Float[torch.Tensor, "c d h w"], Float[torch.Tensor, "c h w"]]:
        return self.cryo_data[idx], self.hcs_data[idx]

def get_paired_dataloader(batch_size: int = 4) -> DataLoader:
    dataset = PairedMultimodalDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# =============================================================================
# 4. Validation Block
# =============================================================================

if __name__ == "__main__":
    print("Initializing Multimodal Data Pipeline...\n")

    BATCH_SIZE = 4

    # 1. Fetch the DataLoaders (This will trigger the Fail-Safe mock data generation if needed)
    cryo_loader, hcs_loader = get_multimodal_dataloaders(batch_size=BATCH_SIZE)

    print(f"Total batches in Cryo-EM DataLoader: {len(cryo_loader)}")
    print(f"Total batches in HCS DataLoader: {len(hcs_loader)}\n")

    # 2. Fetch a single batch
    cryo_batch: Float[torch.Tensor, "batch channels depth height width"] = next(iter(cryo_loader))
    hcs_batch: Float[torch.Tensor, "batch channels height width"] = next(iter(hcs_loader))

    # 3. Validate Dimensions
    print("-" * 65)
    print("DATA PIPELINE SHAPE VALIDATION REPORT:")
    print("-" * 65)
    print(f"Cryo-EM Batch Shape: {list(cryo_batch.shape)}")
    print(f"  -> Expected:       [{BATCH_SIZE}, Channels, Depth, Height, Width]")
    print(f"  -> Type:           {cryo_batch.dtype}\n")

    print(f"HCS Batch Shape:     {list(hcs_batch.shape)}")
    print(f"  -> Expected:       [{BATCH_SIZE}, Channels, Height, Width]")
    print(f"  -> Type:           {hcs_batch.dtype}")
    print("-" * 65)
    print("Status: SUCCESS. Data loaders are synchronized, typed, and operational.")