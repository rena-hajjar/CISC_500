import os
import slicer
import SimpleITK as sitk
import sitkUtils

# === CONFIG ===
sequence_name = "Sequence"       
seg_sequence_name = "Sequence_2"    
output_img_dir = r"C:\Users\Rena\Documents\BCS Cavity Scanning\USFrameSegmentations\images"
output_mask_dir = r"C:\Users\Rena\Documents\BCS Cavity Scanning\USFrameSegmentations\masks"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# === GET SEQUENCE NODES ===
sequence_node = slicer.util.getNode(sequence_name)
seg_seq_node = slicer.util.getNode(seg_sequence_name)

n_frames = sequence_node.GetNumberOfDataNodes()
print(f"Found {n_frames} frames.")

# --- EXPORT LOOP ---
# for i in range(n_frames):
for i in range(132):
    # Get per-frame nodes
    vol_node = sequence_node.GetNthDataNode(i)
    seg_node = seg_seq_node.GetNthDataNode(i)
    frame_id = f"{i:04d}"

    if vol_node is None or seg_node is None:
        print(f"Skipping frame {i} (missing node)")
        continue

    # Convert segmentation for this frame to labelmap
    labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
        seg_node, labelmap_node, vol_node
    )

    # Convert to SimpleITK
    image = sitkUtils.PullVolumeFromSlicer(vol_node)
    mask = sitkUtils.PullVolumeFromSlicer(labelmap_node)

    # Rescale & cast for PNG
    image = sitk.Cast(sitk.RescaleIntensity(image, 0, 255), sitk.sitkUInt8)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Save
    img_path = os.path.join(output_img_dir, f"frame_{frame_id}.png")
    mask_path = os.path.join(output_mask_dir, f"mask_{frame_id}.png")
    sitk.WriteImage(image, img_path)
    sitk.WriteImage(mask, mask_path)

    # Clean up
    slicer.mrmlScene.RemoveNode(labelmap_node)

print("✅ All frames and corresponding masks exported successfully!")