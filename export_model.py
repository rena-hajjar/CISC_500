import torch
import h5py

fold_paths = [
    "path/to/fold_0/checkpoint_best.pth",
    "path/to/fold_1/checkpoint_best.pth",
    "path/to/fold_2/checkpoint_best.pth",
    "path/to/fold_3/checkpoint_best.pth",
    "path/to/fold_4/checkpoint_best.pth",
]

state_dicts = [torch.load(p, map_location="cpu")["state_dict"] for p in fold_paths]

# Average weights across folds
avg_state_dict = {}
for key in state_dicts[0].keys():
    avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)

# Save averaged model to .h5
with h5py.File("nnunet_ensemble_weights.h5", "w") as f:
    for key, value in avg_state_dict.items():
        f.create_dataset(key, data=value.cpu().numpy())

print("✅ Ensemble model saved as nnunet_ensemble_weights.h5")
