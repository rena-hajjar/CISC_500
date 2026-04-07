import torch
import torch.nn as nn
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import json

# Load the nnUNet predictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)

predictor.initialize_from_trained_model_folder(
    r'nnUNet_Data\nnUNet_results\Dataset556\nnUNetTrainer__nnUNetPlans__2d',
    use_folds=(0,1, 2, 3, 4),  # or whichever fold(s) you trained
    checkpoint_name='checkpoint_final.pth',
)

# Get the underlying network
network = predictor.network
network.eval()

# Create a wrapper that accepts a single tensor and returns logits or softmax
class nnUNetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (1, C, H, W) for 2D or (1, C, D, H, W) for 3D
        with torch.no_grad():
            output = self.model(x)
            # Return softmax probabilities or argmax
            return torch.softmax(output, dim=1)

wrapped = nnUNetWrapper(network)
wrapped.eval()

# Your current setup
dummy_input = torch.randn(1, 1, 512, 768)

with torch.no_grad():
    traced_model = torch.jit.trace(wrapped, dummy_input)

test_output = traced_model(dummy_input)
print("Output shape:", test_output.shape)

# Create config with your model's expected shape
config = {
    "shape": [1, 1, 512, 768],  # [batch, channels, height, width]
    "use_tracking_layer": False,
    "tracking_method": "none"
}

# Save with config as extra files
extra_files = {"config.json": json.dumps(config)}
torch.jit.save(traced_model, "nnunet_model2.pt", _extra_files=extra_files)
print("Saved TorchScript model with config!")

# Verification
loaded = torch.jit.load("my_nnunet_model.pt")
loaded.eval()
out = loaded(dummy_input)
print("Verification output shape:", out.shape)