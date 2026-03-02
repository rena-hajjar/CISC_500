import torch
import torch.nn as nn
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

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
    'nnUNet_Data/nnUNet_results/Dataset555/nnUNetTrainer__nnUNetPlans__2d',
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

# Create a dummy input matching your model's expected input shape
# For 2D nnUNet: (batch, channels, height, width)
dummy_input = torch.randn(1, 1, 512, 768)  # adjust C, H, W to your data

# Try tracing first (usually works better than scripting for nnUNet)
with torch.no_grad():
    traced_model = torch.jit.trace(wrapped, dummy_input)

# Validate it works
test_output = traced_model(dummy_input)
print("Output shape:", test_output.shape)

# Save
traced_model.save("my_nnunet_model.pt")
print("Saved TorchScript model!")

loaded = torch.jit.load("my_nnunet_model.pt")
loaded.eval()
out = loaded(dummy_input)
print("Verification output shape:", out.shape)