import os
import cv2
import SimpleITK as sitk
import os
import shutil
import argparse
import numpy as np
import re


def getargs():
    p = argparse.ArgumentParser(
    description="Folder names for nnUNet preprocessing data",
    )
    p.add_argument(
        "--folder1",
        required=True,
        help="Location of existing images and masks folders")
    p.add_argument(
        "--folder2",
        required=True)
    p.add_argument(
        "--folder3",
        required=True
    )
    p.add_argument(
        "--output_folder",
        required=True
    )

    return p.parse_args()

args = getargs()

folder1 = args.folder1
folder2 = args.folder2
folder3 = args.folder3
output_folder = args.output_folder

os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)

def copy_with_sequential_numbering(folders, start_index=0):
    counter = start_index

    for folder in folders:
        src_images = os.path.join(folder, "images")
        src_masks = os.path.join(folder, "masks")

        filenames = sorted([
            f for f in os.listdir(src_images) if f.startswith("frame_")
        ])

        for filename in filenames:
            number_part = filename.replace("frame_", "")
            mask_filename = f"mask_{number_part}"

            src_img_path = os.path.join(src_images, filename)
            src_mask_path = os.path.join(src_masks, mask_filename)

            new_img_name = f"frame_{counter:04d}.png"
            new_mask_name = f"mask_{counter:04d}.png"

            dst_img_path = os.path.join(output_folder, "images", new_img_name)
            dst_mask_path = os.path.join(output_folder, "masks", new_mask_name)

            shutil.copy2(src_img_path, dst_img_path)

            if os.path.exists(src_mask_path):
                shutil.copy2(src_mask_path, dst_mask_path)
            else:
                print(f"No matching mask for {filename}")

            counter += 1

    print(f"Done merging datasets. Total files: {counter - start_index}")

copy_with_sequential_numbering([folder1, folder2, folder3])


output_labels = r"C:\Users\Rena\Documents\BCS Cavity Scanning\UNetData\nnUNet_raw\Dataset557\labelsTr"
output_images = r"C:\Users\Rena\Documents\BCS Cavity Scanning\UNetData\nnUNet_raw\Dataset557\imagesTr"

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

image_dir = os.path.join(output_folder, "images")
mask_dir = os.path.join(output_folder, "masks")

# Get sorted list of frames
frame_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")],
                     key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

for f in frame_files:
    # Extract frame number
    frame_num = ''.join(filter(str.isdigit, f))
    # Build corresponding mask filename
    mask_name = f"mask_{frame_num}.png"
    
    # Read image and mask
    img = cv2.imread(os.path.join(image_dir, f), cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Image not found: {f}")
        continue
    if msk is None:
        print(f"Mask not found: {mask_name}")
        continue

    # Normalize mask from 0-255 to 0-1
    msk = (msk > 127).astype(np.uint8)  # threshold to binary 0 or 1

    # Convert to SimpleITK and save
    img_itk = sitk.GetImageFromArray(img)
    msk_itk = sitk.GetImageFromArray(msk)
    
    sitk.WriteImage(img_itk, os.path.join(output_images, f"{os.path.splitext(f)[0]}_0000.nii.gz"))
    sitk.WriteImage(msk_itk, os.path.join(output_labels, f"{os.path.splitext(f)[0]}.nii.gz"))

print("Dataset converted to nnU-Net format.")
