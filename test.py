import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Change these paths ---
frame_index = 42  # any frame between 0 and 130
base_path = r"C:\Users\Rena\Documents\BCS Cavity Scanning\USFrameSegmentations"
image_path = os.path.join(base_path, "images", f"frame_{frame_index:04d}.png")
mask_path = os.path.join(base_path, "masks", f"mask_{frame_index:04d}.png")

print(mask_path)

# --- Load image and mask ---
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# --- Normalize or scale mask ---
if mask.max() <= 1:
    mask = (mask * 255).astype(np.uint8)
else:
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- Convert binary mask to boolean for overlay ---
mask_bool = mask > 127

# --- Create overlay ---
overlay = image.copy()
overlay_color = (0, 255, 0)  # bright green
alpha = 0.6  # transparency

overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + np.array(overlay_color) * alpha).astype(np.uint8)

# --- Display side-by-side ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()