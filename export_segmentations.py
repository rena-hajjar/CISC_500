

import slicer
import SimpleITK as sitk
import sitkUtils
import numpy as np
import os
import cv2


sequence_node = slicer.util.getNode("")
segmentation_node = slicer.util.getNode("")
output_img_dir = r"C:\Users\Rena\Documents\BCS Cavity Scanning\UltraSAM\~\images"
output_mask_dir = r"C:\Users\Rena\Documents\BCS Cavity Scanning\UltraSAM\~\masks"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

n_frames = sequence_node.GetNumberOfDataNodes()
print(f"Found {n_frames} frames.")

start_frame = 0
end_frame= 0

browserNode = slicer.util.getNode("~")
# for i in range(n_frames):
for i in range(start_frame, end_frame + 1):
    browserNode.SetSelectedItemNumber(i)
    vol_node = sequence_node.GetNthDataNode(i)
    seg_node = segmentation_node.GetNthDataNode(i)
    frame_id = f"{i:04d}"

    if vol_node is None or seg_node is None:
        print(f"Skipping frame {i} (missing node)")
        continue

    segmentation = seg_node.GetSegmentation()
    segment_id = segmentation.GetSegmentIdBySegmentName("Cavity")
    if not segment_id:
        print(f"Frame {i}: 'Cavity' segment not found")
        continue

    labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        seg_node,
        [segment_id],
        labelmap_node,
        vol_node # Match to og size
    )

    arr = slicer.util.arrayFromVolume(labelmap_node)
    mask_2d = (arr[0] > 0).astype(np.uint8) * 255
    mask_path = os.path.join(output_mask_dir, f"mask_{frame_id}.png")
    cv2.imwrite(mask_path, mask_2d)

    slicer.mrmlScene.RemoveNode(labelmap_node)

    image = sitkUtils.PullVolumeFromSlicer(vol_node)
    img_path = os.path.join(output_img_dir, f"frame_{frame_id}.png")
    sitk.WriteImage(image, img_path)