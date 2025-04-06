# FitzToneClassifier
**This repository provides a comprehensive application for image preprocessing, training, and inference of PyTorch classifiers, including skin tone classification based on the Fitzpatrick scale.**

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)

---

## Installation

### Requirements
```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yaml
conda activate FitzToneClassifier

# Docker build
docker build -t fitztone-classifier -f Dockerfile .
```

---

## Project Structure

```bash
FitzToneClassifier/
├── classification/          			# Model training and inference logic
│   ├── model_training_inference.py  	
│   └── pipeline.py          			
├── preprocess/              			# Image preprocessing utilities
│   ├── data_split.py        			
│   ├── preprocess.py        			
│   └── run_preprocess.py    			
├── settings/                			# Configurations and resources
│   └── info/
│       ├── face_landmarks/  			
│       │   └── shape_predictor_68_face_landmarks.dat
│       ├── train.json      			# Training hyperparameters
│       ├── test.json        			# Inference configuration
│       └── further_train.json 		# Fine-tuning configuration
└── results/                 			# Auto-generated results
    ├── metrics/             			
    └── predictions/         			
```

---

## Usage

### 1. Image Preprocessing
```bash
# For a single image
python ./preprocess/run_preprocess.py \
  --input ./images/sample.jpg \
  --output ./preprocessed_images/

# For an entire folder
python ./preprocess/run_preprocess.py \
  --input ./images/class1/ \
  --output ./preprocessed_images/class1/
```
**Functionality includes:**  
- Face alignment via 68-point landmarks (MTCNN + dlib)  
- Lighting normalization (CLAHE + denoising)  
- Cropping to 224x224 px for model input

---

### 2. Dataset Splitting
```bash
python ./preprocess/data_split.py \
  --input ./preprocessed_images/ \
  --output ./dataset/
```
This splits data into `train` and `val` sets (default: 80/20).

---

### 3. Model Training
```bash
# Initial training
python classification/pipeline.py \
  --process train \
  --data_folder ./dataset/ \
  --params_path ./settings/info_about_start/train.json

# Fine-tuning
python classification/pipeline.py \
  --process further_train \
  --data_folder ./dataset/ \
  --params_path ./settings/info_about_start/further_train.json
```

---

### 4. Inference
```bash
python classification/pipeline.py \
  --process inference \
  --data_folder ./test_images/ \
  --params_path ./settings/info_about_start/inference.json
```
**Outputs:**  
- `./results/predictions/` — CSV file with predictions

---

## Configuration Example

### `train.json`
```json
{
  "model": "vit_base_patch16",
  "learning_rate": 0.0005,
  "scheduler": "CosineAnnealingLR",
  "augmentations": {
    "flip": true,
    "rotation": 15,
    "color_jitter": 0.2
  }
}
```

### Supported Models
Supports all available PyTorch classification models  
(e.g., ResNet, AlexNet, ViT, MobileNet, EfficientNet, etc.)

---

## Dataset Format

Recommended folder layout:
```
images/
├── class1/
│   ├── img1.jpg
│   └── img2.jpg
└── class2/
    ├── img3.jpg
    └── img4.jpg
```
📎 [Download example dataset](https://drive.google.com/drive/folders/1ww_i0yUb3gqbqMPvxnUKHfRhr1zUsttZ)

---

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

💡 *Feel free to submit issues, feature requests, and ideas!*
