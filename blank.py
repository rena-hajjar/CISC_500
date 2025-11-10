import os
import numpy as np
from PIL import Image
import nibabel as nib

# # Path to your folder
# input_folder = r"C:\Users\Rena\Documents\BCS Cavity Scanning\USFrames2"
# output_folder = r"C:\Users\Rena\Documents\BCS Cavity Scanning\USFrames2ForInference"

# os.makedirs(output_folder, exist_ok=True)

# for filename in sorted(os.listdir(input_folder)):
#     if filename.endswith(".png") and filename.startswith("frame_"):
#         # Extract original number and remove leading zeros
#         num = int(filename.split('_')[1].split('.')[0])
#         new_name = f"frame_{num:03d}_0000.nii.gz"
        
#         # Load PNG and convert to numpy array
#         img_path = os.path.join(input_folder, filename)
#         img = Image.open(img_path).convert("L")  # convert to grayscale
#         data = np.array(img)
        
#         # Add extra dimension for NIfTI (3D)
#         data = data[np.newaxis, :, :]  # shape: (1, H, W)
        
#         # Create NIfTI image
#         nii_img = nib.Nifti1Image(data, affine=np.eye(4))
#         nib.save(nii_img, os.path.join(output_folder, new_name))

# print("Conversion done!")

import os
import nibabel as nib
import numpy as np
from PIL import Image

pred_folder = r"C:\Users\Rena\Documents\BCS Cavity Scanning\Frames2InferenceResults"      # NIfTI prediction files
out_folder = r"C:\Users\Rena\Documents\BCS Cavity Scanning\USFrames2Segmentations"   # Output PNG folder

os.makedirs(out_folder, exist_ok=True)

for f in sorted(os.listdir(pred_folder)):
    if f.endswith(".nii.gz"):
        nii_path = os.path.join(pred_folder, f)
        img = nib.load(nii_path)
        data = img.get_fdata()   # shape: (1, H, W) or (H, W)

        # If 3D or channel-first, squeeze extra dimensions
        if data.ndim == 3:
            data = data[0]       # take first channel

        # Convert to binary mask: 0 = background, 255 = foreground
        # Assumes 0 = background, anything else = foreground
        binary_mask = np.where(data > 0, 255, 0).astype(np.uint8)

        # Save as PNG
        out_path = os.path.join(out_folder, f.replace(".nii.gz", ".png"))
        Image.fromarray(binary_mask).save(out_path)

print(f"Converted {len(os.listdir(out_folder))} NIfTI masks to 0/255 PNGs!")