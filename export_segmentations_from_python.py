import slicer
import os
import SimpleITK as sitk
from PIL import Image

# Path to your .mrb files
input_files = [
    "/Users/renahajjar/Desktop/YEAR 4/CISC_500/USSegmentationSequence.mrb",
    "/Users/renahajjar/Desktop/YEAR 4/CISC_500/USSegmentationSequence2.mrb"
]

output_dir = "/Users/renahajjar/Desktop/YEAR 4/CISC_500/SegmentationData"  # root output folder
png_dir = os.path.join(output_dir, "png_slices")
nnunet_dir = os.path.join(output_dir, "nnunet_ready")
os.makedirs(png_dir, exist_ok=True)
os.makedirs(nnunet_dir, exist_ok=True)
for mrb_path in input_files:
    slicer.mrmlScene.Clear(0)
    slicer.util.loadScene(mrb_path)

    # Get segmentation and volume nodes
    segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    seg = list(segmentation_nodes.values())[0]
    vol = list(volume_nodes.values())[0]

    # Export segmentation to labelmap
    labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(seg, labelmap, vol)

    base_name = os.path.splitext(os.path.basename(mrb_path))[0]
    nii_image_path = os.path.join(nnunet_dir, f"{base_name}_0000.nii.gz")  # nnU-Net input naming convention
    nii_mask_path = os.path.join(nnunet_dir, f"{base_name}.nii.gz")

    # Save .nii.gz files (for nnU-Net)
    slicer.util.saveNode(vol, nii_image_path)
    slicer.util.saveNode(labelmap, nii_mask_path)

    # Convert to PNG slices (for visualization)
    # Read the just-saved NIfTI files back into SimpleITK
    image_itk = sitk.ReadImage(nii_image_path)
    mask_itk = sitk.ReadImage(nii_mask_path)

    image_np = sitk.GetArrayFromImage(image_itk)
    mask_np = sitk.GetArrayFromImage(mask_itk)

    case_png_dir = os.path.join(png_dir, base_name)
    os.makedirs(os.path.join(case_png_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(case_png_dir, "masks"), exist_ok=True)

    for i in range(image_np.shape[0]):
        img_slice = (image_np[i] - np.min(image_np[i])) / (np.ptp(image_np[i]) + 1e-8)
        img_slice = (img_slice * 255).astype(np.uint8)
        mask_slice = (mask_np[i] > 0).astype(np.uint8) * 255

        Image.fromarray(img_slice).save(os.path.join(case_png_dir, "images", f"slice_{i:03d}.png"))
        Image.fromarray(mask_slice).save(os.path.join(case_png_dir, "masks", f"slice_{i:03d}.png"))

    print(f" Saved: {base_name}_0000.nii.gz (image), {base_name}.nii.gz (mask), and PNG slices.")

print("🎯 All MRB files exported successfully.")