# Deep Learning Projects

A collection of deep learning implementations covering transformer architectures, multimodal systems, and retrieval-augmented generation.

## Projects

### [Qwen3 Language Model from Scratch](./qwen-from-scratch)
Building a modern Transformer-based language model from the ground up. This project implements all core components of the Qwen3 architecture including:
- Grouped Query Attention mechanism
- Root Mean Square Layer Normalization
- Feed Forward networks
- Key-Value caching for efficient inference
- Complete Transformer blocks

**[View Project →](./qwen-from-scratch/README.md)**

### [Multimodal RAG System](./retrieval-augmented)
An end-to-end Retrieval-Augmented Generation pipeline that processes PDF documents containing both text and images. Features:
- Multimodal embeddings with Jina-CLIP
- Vector database storage with ChromaDB
- Image and text extraction from PDFs
- Question-answering with Phi-3-Vision
- Interactive chat interface

**[View Project →](./retrieval-augmented/README.md)**

### [Vision Transformer Fine-Tuning with LoRA](./vit-semantic-segmentation)
Parameter-efficient fine-tuning of Vision Transformer models using Low-Rank Adaptation for food image classification. Features:
- LoRA integration reducing trainable parameters by 98.56%
- Vision Transformer (ViT) architecture
- Food101 dataset with 101 food categories
- Data augmentation pipeline
- Mixed precision training
- Experiment tracking with Weights & Biases

**[View Project →](./vit-semantic-segmentation/README.md)**