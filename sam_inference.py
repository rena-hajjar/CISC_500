import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from segment_anything import sam_model_registry, SamPredictor

class SimpleSegmenter(QWidget):
    def __init__(self, input_dir, output_dir, sam_checkpoint, model_type="vit_b"):
        super().__init__()
        self.setWindowTitle("MedSAM Simple Segmenter")
        self.setGeometry(50, 50, 1200, 800)  # Larger window

        # Paths
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.image_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(".png")]
        self.index = 0

        # SAM setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

        # GUI elements
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expand with window
        self.next_btn = QPushButton("Next Image")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.next_btn)
        self.setLayout(layout)

        # Brush & mask
        self.drawing = False
        self.erasing = False
        self.brush_size = 10  # Bigger brush for visibility
        self.mask = None
        self.image = None

        # Connect
        self.next_btn.clicked.connect(self.next_image)
        self.label.mousePressEvent = self.mouse_press
        self.label.mouseMoveEvent = self.mouse_move
        self.label.mouseReleaseEvent = self.mouse_release

        # Load first image
        if self.image_files:
            self.load_image()

    # ---------------- Mouse events ----------------
    def mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            x, y = self.label_to_image_coords(event.x(), event.y())
            self.paint_mask(x, y)

    def mouse_move(self, event):
        if self.drawing:
            x, y = self.label_to_image_coords(event.x(), event.y())
            self.paint_mask(x, y)

    def mouse_release(self, event):
        self.drawing = False

    def label_to_image_coords(self, lx, ly):
        h, w = self.image.shape[:2]
        label_w, label_h = self.label.width(), self.label.height()
        scale_w = w / label_w
        scale_h = h / label_h
        x = int(lx * scale_w)
        y = int(ly * scale_h)
        return np.clip(x, 0, w-1), np.clip(y, 0, h-1)

    def paint_mask(self, x, y):
        color = 255 if not self.erasing else 0
        cv2.circle(self.mask, (x, y), self.brush_size, color, -1)
        self.update_display()

    # ---------------- Image & SAM ----------------
    def load_image(self):
        fname = self.image_files[self.index]
        path = os.path.join(self.input_dir, fname)
        self.image = cv2.imread(path)
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Draw bounding box to prompt SAM
        bbox = cv2.selectROI("Draw bounding box", self.image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Draw bounding box")
        if bbox == (0,0,0,0):
            print(f"Skipping {fname}")
            self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        else:
            x, y, w, h = bbox
            bbox_array = np.array([x, y, x+w, y+h])
            self.predictor.set_image(image_rgb)
            masks, scores, logits = self.predictor.predict(box=bbox_array[None,:], multimask_output=False)
            self.mask = (masks[np.argmax(scores)]*255).astype(np.uint8)

        self.update_display()

    def update_display(self):
        # Overlay mask on image
        overlay = cv2.addWeighted(self.image, 0.7, cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR), 0.3, 0)

        # Get window size
        win_w = self.label.width()
        win_h = self.label.height()

        # Resize overlay to fit label while keeping aspect ratio
        img_h, img_w = overlay.shape[:2]
        scale = min(win_w / img_w, win_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert to QImage and display
        h, w, ch = resized_overlay.shape
        bytes_per_line = ch * w
        qt_image = QImage(resized_overlay.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_image)
        self.label.setPixmap(pixmap)

    # ---------------- Next image ----------------
    def next_image(self):
        # Save mask
        fname = self.image_files[self.index]
        out_path = os.path.join(self.output_dir, fname.replace(".png", "_mask.png"))
        cv2.imwrite(out_path, self.mask)
        print(f"Saved {out_path}")

        # Move to next image
        if self.index < len(self.image_files)-1:
            self.index += 1
            self.load_image()
        else:
            print("All images segmented!")
            self.close()

# ---------------- Main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    input_dir = r"C:\Users\Rena\Documents\BCS Cavity Scanning\USFrames"
    output_dir = r"C:\Users\Rena\Documents\BCS Cavity Scanning\USFrameSegmentations"
    sam_checkpoint = r"C:\Users\Rena\Documents\YEAR4\CISC500\CISC_500\MedSAM\medsam_vit_b.pth"
    gui = SimpleSegmenter(input_dir, output_dir, sam_checkpoint)
    gui.showMaximized()  # Open window maximized
    sys.exit(app.exec_())
