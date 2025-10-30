import os
import cv2
import SimpleITK as sitk

# --- SETTINGS ---
# image_dir = r"C:\Users\Rena\Documents\BCS Cavity Scanning\USFrameSegmentations\images"
# mask_dir  = r"C:\Users\Rena\Documents\BCS Cavity Scanning\USFrameSegmentations\masks"
# output_images = r"C:\Users\Rena\Documents\BCS Cavity Scanning\UNetData\nnUNet_raw\Dataset011\imagesTr"
# output_labels = r"C:\Users\Rena\Documents\BCS Cavity Scanning\UNetData\nnUNet_raw\Dataset011\labelsTr"

# os.makedirs(output_images, exist_ok=True)
# os.makedirs(output_labels, exist_ok=True)

# # Get sorted list of frames
# frame_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")],
#                      key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

# for f in frame_files:
#     # Extract frame number
#     frame_num = ''.join(filter(str.isdigit, f))
#     # Build corresponding mask filename
#     mask_name = f"mask_{frame_num}.png"
    
#     # Read image and mask
#     img = cv2.imread(os.path.join(image_dir, f), cv2.IMREAD_GRAYSCALE)
#     msk = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
    
#     if img is None:
#         print(f"⚠️ Image not found: {f}")
#         continue
#     if msk is None:
#         print(f"⚠️ Mask not found: {mask_name}")
#         continue
    
#     # Convert to SimpleITK and save
#     img_itk = sitk.GetImageFromArray(img)
#     msk_itk = sitk.GetImageFromArray(msk)
    
#     sitk.WriteImage(img_itk, os.path.join(output_images, f"{os.path.splitext(f)[0]}_0000.nii.gz"))
#     sitk.WriteImage(msk_itk, os.path.join(output_labels, f"{os.path.splitext(f)[0]}.nii.gz"))

# print("✅ Dataset converted to nnU-Net format.")

import os
import re

print("HELLO")

# Folder containing your files
folder = r"C:\Users\Rena\Documents\BCS Cavity Scanning\UNetData\nnUNet_raw\Dataset011\labelsTr"


for filename in os.listdir(folder):
    if filename.endswith(".nii.gz"):
        # Match 'frame_' followed by a digit we want to remove, then the rest
        match = re.match(r"(frame_)0(\d+\.nii\.gz)", filename)
        if match:
            new_name = match.group(1) + match.group(2)
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)
            print(f"Renaming: {filename} → {new_name}")
            os.rename(old_path, new_path)