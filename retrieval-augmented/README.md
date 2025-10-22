# Multimodal RAG System with Jina-CLIP and Phi-3-Vision

A Retrieval-Augmented Generation (RAG) system that processes PDF documents containing both text and images, enabling intelligent question-answering over multimodal content.

## Overview

This project implements an end-to-end multimodal RAG pipeline that:
- Extracts text and images from PDF documents
- Generates embeddings using state-of-the-art multimodal models
- Stores and retrieves relevant context from a vector database
- Generates contextually-aware responses using a vision-language model

## Architecture

### 1. Document Ingestion and Extraction
- **PDF Fetching**: `fetch_pdf_from_url()` downloads PDFs from URLs and optionally saves them locally
- **Content Extraction**: `extract_content_from_pdf()` uses PyMuPDF (successor to the deprecated `fitz` library) to extract:
  - Text blocks with page metadata
  - Images with positional information
  
### 2. Multimodal Embedding Generation
- **Model**: Jina-CLIP v1 - a state-of-the-art multimodal embedding model
- **Implementation**: `generate_content_embeddings()` produces 768-dimensional vectors for both text and images
- **Shared Embedding Space**: Text and image embeddings exist in the same semantic space, enabling cross-modal retrieval

### 3. Vector Database Storage
- **Database**: ChromaDB with HNSW indexing and cosine similarity
- **Collections**:
  - Text embeddings stored via `store_text_embeddings()`
  - Image embeddings stored via `store_image_embeddings()`
- **Metadata**: Each entry includes page numbers, source URLs, and content type

### 4. Retrieval Component
- **Query Method**: `query_with_text()` converts natural language queries into embeddings
- **Similarity Search**: Returns top-k most relevant text blocks and images
- **Filtering**: Optional type filtering (text-only or image-only retrieval)

### 5. Generation Component
- **Model**: Microsoft Phi-3-Vision-128K-Instruct
- **Context Formatting**: `format_context_for_phi3()` structures retrieved content for zero-shot inference
- **Flash Attention**: Optimized inference using Flash Attention 2

### 6. End-to-End Pipeline
- **Unified Interface**: `rag_query_and_generate()` orchestrates the complete flow:
  1. Query embedding and retrieval
  2. Context formatting
  3. Response generation with Phi-3-Vision
- **Interactive Chat**: `chat_with_rag()` enables conversational Q&A with chat history

## Installation
```bash
pip install transformers==4.48.0 # Some problems with earlier versions of Phi-3-Vision model https://stackoverflow.com/questions/79769295/attributeerror-dynamiccache-object-has-no-attribute-seen-tokens
pip install chromadb pymupdf
pip install flash-attn --no-build-isolation # Phi-3-Vision dependency
```

## Usage

### Basic RAG Query
```python
# Complete RAG pipeline
result = rag_query_and_generate(
    jina_model=jina_model,
    phi3_model=phi3_model,
    phi3_processor=phi3_processor,
    collection=collection,
    query="Explain the attention mechanism in transformers",
    n_results=5
)

print(result["response"])
```

### Interactive Chat
```python
# Start a conversation
chat_history = []
response = chat_with_rag(
    "What is the main architecture proposed in the 'Attention is All You Need' paper?",
    chat_history=chat_history
)
```

### Performance Optimizations

- **Flash Attention 2**: 2-3x faster inference for Phi-3-Vision
- **Batch Processing**: Efficient embedding generation for multiple documents
- **Vector Indexing**: HNSW algorithm for sub-linear retrieval time

## Example Results

The system successfully answers complex questions about the "Attention Is All You Need" paper, including:
- Architecture explanations
- Technical formulas and their components
- Comparisons with other models (RNNs, CNNs)
- Visual diagram descriptions
