# CISC_500

Undergraduate Thesis Project Files

Included:

- script to export segmentations from slicer as PNGs
- script to export trained nnUNet model to a PyTorch model for TorchSequenceSegmentation
- updated TorchSequenceSegmentation script (Fixed to use nnUNetv2)
- script to convert images and masks to NIfTI for nnUNet training

The rest of the Other Work can be found in Pdrive under data/BCSCavityScan, including the following:

- An updated version of Olivia Radcliffe's module, to fix point cloud reconstruction to be compatible with the new registration and real models. The nnUNet reconstruction section is still under construction.
- The Galaxy retractor, TRUS probe, and TRUS probe EM clip fusion files and .STL files.
- All training and testing data used for the nnUNet.
- The final model weights, for testing in TorchSequenceSegmentation.
