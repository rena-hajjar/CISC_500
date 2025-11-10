import torch
import h5py

fold_paths = [
    r"P:\data\BCSCavityScan\nnUNetTrainer__nnUNetPlans__2d\fold_0\checkpoint_best.pth",
    r"P:\data\BCSCavityScan\nnUNetTrainer__nnUNetPlans__2d\fold_1\checkpoint_best.pth",
    r"P:\data\BCSCavityScan\nnUNetTrainer__nnUNetPlans__2d\fold_2\checkpoint_best.pth",
    r"P:\data\BCSCavityScan\nnUNetTrainer__nnUNetPlans__2d\fold_3\checkpoint_best.pth",
    r"P:\data\BCSCavityScan\nnUNetTrainer__nnUNetPlans__2d\fold_4\checkpoint_best.pth",
]


# Load all state_dicts from 'network_weights'
state_dicts = [torch.load(p, map_location="cpu", weights_only=False)['network_weights'] for p in fold_paths]

# Average weights across folds
avg_state_dict = {}
for key in state_dicts[0].keys():
    print('key')
    avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)

# Save to .h5
print('Saving file...')
with h5py.File("nnunet_ensemble_weights.h5", "w") as f:

    for key, value in avg_state_dict.items():
        f.create_dataset(key, data=value.cpu().numpy())

print("✅ Ensemble model saved as nnunet_ensemble_weights.h5")