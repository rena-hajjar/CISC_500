#!/bin/bash

DATASET_ID=557
CONFIG="2d"
FOLDS=(1 2 3 4)

for fold in "${FOLDS[@]}"; do
    echo "=========================================="
    echo "Training fold $fold..."
    echo "=========================================="
    
    nnUNetv2_train $DATASET_ID $CONFIG $fold --npz -tr nnUNetTrainer_250epochs -device cuda --c
    
    if [ $? -ne 0 ]; then
        echo "Fold $fold failed! Exiting."
        exit 1
    fi
    
    echo "Fold $fold complete."
done

echo "All folds complete!"