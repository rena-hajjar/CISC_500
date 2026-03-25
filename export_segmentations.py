# import os
# import slicer
# import SimpleITK as sitk
# import sitkUtils

# import cv2
# import numpy as np
# from pathlib import Path

import slicer
import SimpleITK as sitk
import sitkUtils
import numpy as np
import os

import vtk
from vtk.util import numpy_support

seg_seq_node = slicer.util.getNode("Sequence")
vol_seq_node = slicer.util.getNode("Sequence_6")
output_mask_dir = r"C:\Users\Rena\Documents\BCS Cavity Scanning\UltrasoundPhantomSweep\masks"

n_frames = seg_seq_node.GetNumberOfDataNodes()
print(f"Exporting {n_frames} frames...")

for i in range(n_frames):
    seg_node = seg_seq_node.GetNthDataNode(i)
    vol_node = vol_seq_node.GetNthDataNode(i)
    frame_id = f"{i:04d}"

    if seg_node is None or vol_node is None:
        print(f"Skipping frame {i}")
        continue

    # Get binary labelmap directly from segment
    segmentation = seg_node.GetSegmentation()
    segID = segmentation.GetNthSegmentID(0)  # CavityWall is segment 0
    segment = segmentation.GetSegment(segID)

    # Get the binary labelmap representation directly
    binaryLabelmap = segment.GetRepresentation("Binary labelmap")

    if binaryLabelmap is None:
        print(f"Frame {i}: no binary labelmap representation")
        continue

    # Convert vtkOrientedImageData to numpy
    dims = binaryLabelmap.GetDimensions()  # (x, y, z)
    vtk_array = binaryLabelmap.GetPointData().GetScalars()
    arr = numpy_support.vtk_to_numpy(vtk_array).reshape(dims[2], dims[1], dims[0])

    # Take the 2D slice (z=0 since extent shows z: 0-0)
    mask_2d = arr[0].astype(np.uint8) * 255

    # Save
    import cv2
    mask_path = os.path.join(output_mask_dir, f"mask_{frame_id}.png")
    cv2.imwrite(mask_path, mask_2d)

    if i % 50 == 0:
        print(f"Frame {i}: shape={mask_2d.shape}, max={mask_2d.max()}")

print("Done!")


sequence_node = slicer.util.getNode("Sequence")
output_img_dir = r"C:\Users\Rena\Documents\BCS Cavity Scanning\UltraSAM\CavitySweep2"
os.makedirs(output_img_dir, exist_ok=True)

n_frames = sequence_node.GetNumberOfDataNodes()
print(f"Found {n_frames} frames.")

for i in range(n_frames):
    vol_node = sequence_node.GetNthDataNode(i)
    frame_id = f"{i:03d}"
    if vol_node is None:
        print(f"Skipping frame {i} (missing node)")

    image = sitkUtils.PullVolumeFromSlicer(vol_node)
    img_path = os.path.join(output_img_dir, f"frame_{frame_id}.png")
    sitk.WriteImage(image, img_path)
