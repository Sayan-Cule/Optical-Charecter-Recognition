# Handwritten Text Recognition (OCR)

A deep-learning optical character recognition (OCR) project for recognizing handwritten English words from images.

The implementation uses a residual CNN feature extractor followed by a Bidirectional LSTM and CTC-based sequence decoding. A Tkinter desktop GUI is provided for image-based inference through an exported ONNX model.

## Project Overview

The system is designed for word-level handwritten text recognition rather than isolated character classification.

### Pipeline

```text
IAM Words dataset
       ↓
Image loading and resizing
       ↓
Data augmentation
       ↓
Residual CNN feature extraction
       ↓
Bidirectional LSTM
       ↓
CTC loss during training
       ↓
H5 / ONNX model
       ↓
CTC decoding
       ↓
Recognized handwritten word
```

## Model Architecture

- Input: `128 × 32 × 3` image
- Pixel normalization inside the model
- Residual CNN blocks with increasing channel depth
- Bidirectional LSTM with 128 units
- Dropout
- Dense softmax output over the project vocabulary
- CTC loss for variable-length word transcription

The architecture is implemented in `src/model.py`.

## Dataset

The training pipeline uses the **IAM Words** handwritten English word dataset. The dataset itself is not included in this repository.

The training script builds image paths and transcription labels from the IAM Words directory structure and creates the vocabulary and maximum text length from the available samples.

## Training

The training pipeline in `src/train.py` includes:

- IAM Words dataset preparation
- Image resizing to `128 × 32`
- Label indexing and padding
- Random brightness augmentation
- Random erode/dilate augmentation
- Random sharpening
- Random rotation
- 90/10 training-validation split
- Adam optimizer
- CTC loss
- CWER metric
- Early stopping
- Model checkpointing
- Learning-rate reduction
- TensorBoard logging
- H5 model checkpointing
- ONNX model export

The saved project configuration records a batch size of 64, learning rate of 0.001, maximum text length of 16, and 1000 training epochs.

## Inference

`src/inferenceModel.py` provides ONNX inference and CTC decoding. It can also evaluate predictions from a validation CSV using Character Error Rate (CER).

## Desktop GUI

`src/gui.py` provides a simple Tkinter interface with:

- **Import Image** — select a handwritten word image
- **Recognize** — run ONNX inference and decode the prediction
- **Clear** — reset the interface

The GUI uses OpenCV for image loading/resizing, ONNX Runtime through MLTU's inference wrapper, and CTC decoding for the final text output.

## Repository Structure

```text
Optical-Charecter-Recognition/
├── README.md
├── LICENSE
├── .gitignore
├── config/
│   └── configs.yaml
├── data/
│   └── README.md
├── models/
│   ├── model.h5
│   └── model.onnx
└── src/
    ├── configs.py
    ├── gui.py
    ├── inferenceModel.py
    ├── model.py
    └── train.py
```

## Requirements

Main Python dependencies used by the source include:

- TensorFlow / Keras
- MLTU
- OpenCV
- NumPy
- Pillow
- Tkinter
- ONNX Runtime
- pandas
- tqdm

## Notes

- The academic project report is intentionally not included.
- The public source has been cleaned of local machine-specific paths.
- The IAM Words dataset is not redistributed here.
- Trained H5 and ONNX model files are included in the `models/` directory.
