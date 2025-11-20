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