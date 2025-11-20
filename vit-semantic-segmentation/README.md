# Vision Transformer Fine-Tuning with LoRA: Food Classification

A comprehensive implementation demonstrating how to efficiently fine-tune a Vision Transformer (ViT) model using Low-Rank Adaptation (LoRA) for image classification on the Food101 dataset.

## Overview

This project showcases modern techniques for parameter-efficient fine-tuning of large vision models. By leveraging LoRA, we reduce the number of trainable parameters by over 98% while maintaining high accuracy, making fine-tuning accessible even with limited computational resources.

## Architecture Components

### 1. Vision Transformer (ViT)
- **Base Model**: google/vit-base-patch16-224-in21k
- **Purpose**: Pre-trained vision model that processes images as sequences of patches
- **Architecture**: Transformer encoder with self-attention mechanisms
- **Innovation**: Treats image patches like tokens in NLP

### 2. Low-Rank Adaptation (LoRA)
- **Purpose**: Parameter-efficient fine-tuning technique
- **Efficiency**: Reduces trainable parameters from 85M to 1.2M (1.44% of original)
- **Mechanism**: Injects trainable rank decomposition matrices into attention layers
- **Benefits**:
  - Faster training
  - Lower memory footprint
  - Easier deployment and sharing

### 3. Data Augmentation Pipeline
- **Training Augmentations**:
  - Random resized cropping
  - Random horizontal flipping
  - Normalization with ImageNet statistics
- **Validation/Test Transforms**:
  - Center cropping
  - Resizing
  - Normalization

### 4. Fine-Tuning Strategy
- **Target Modules**: Query and Value attention matrices
- **LoRA Rank**: 32
- **LoRA Alpha**: 16
- **Dropout**: 0.1
- **Preserved Modules**: Classification head

## Dataset

The project uses the [Food101 dataset](https://huggingface.co/datasets/ethz/food101):
- **Classes**: 101 food categories
- **Total Images**: 101,000
- **Training Split**: 75,750 images
- **Validation Split**: 17,675 images (70% of original validation)
- **Test Split**: 7,575 images (30% of original validation)

## Training Configuration

### Hyperparameters
```python
Learning Rate: 5e-3
Batch Size: 64
Gradient Accumulation Steps: 4
Epochs: 5
Precision: FP16 (mixed precision)
```

### Optimization
- **Strategy**: Evaluation and saving at each epoch
- **Best Model Selection**: Based on validation accuracy
- **Logging**: Integration with Weights & Biases for experiment tracking

## Results

The fine-tuned model achieves:
- **Test Accuracy**: 90.5%
- **Trainable Parameters**: 1.26M (1.44% of total)
- **Total Parameters**: 87.1M

## Usage

### Environment Setup
```bash
pip install transformers accelerate evaluate datasets peft wandb
```

### Model Training
The notebook demonstrates how to:
1. Load and preprocess the Food101 dataset
2. Configure the Vision Transformer model
3. Apply LoRA for parameter-efficient fine-tuning
4. Train with data augmentation and mixed precision
5. Evaluate on held-out test data
6. Track experiments with Weights & Biases

### Authentication
```python
# HuggingFace Hub
from huggingface_hub import notebook_login
notebook_login()

# Weights & Biases
import wandb
wandb.login()
```

### Model Inference
```python
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Load fine-tuned model
model = AutoModelForImageClassification.from_pretrained("your-model-path")
processor = AutoImageProcessor.from_pretrained("your-model-path")

# Process and classify image
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(-1)
```

## Key Learning Objectives

- Understanding Vision Transformer architecture for image classification
- Implementing parameter-efficient fine-tuning with LoRA
- Designing effective data augmentation strategies
- Managing dataset splits for robust evaluation
- Tracking experiments with modern MLOps tools
- Optimizing training with mixed precision and gradient accumulation

## Technical Highlights

- **Modern Architecture**: Vision Transformer (ViT) with attention mechanisms
- **Efficiency**: LoRA reduces trainable parameters by 98.56%
- **Performance**: Achieves 90.5% accuracy on Food101 test set
- **Scalability**: Mixed precision training for faster computation
- **Reproducibility**: Comprehensive experiment tracking with W&B
- **Production-Ready**: Easy model sharing via HuggingFace Hub

## Requirements

```bash
pip install torch transformers accelerate evaluate datasets peft wandb
```

## Model Checkpoints

The trained model is automatically pushed to HuggingFace Hub with the naming convention:
```
vit-base-patch16-224-in21k-l{lr}_b{batch}_e{epochs}_finetuned-lora-food101-{timestamp}
```
