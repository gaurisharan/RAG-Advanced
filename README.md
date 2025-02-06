# 📚 RAG Chat: Document-based Question Answering System

## Overview

This Streamlit application implements a Retrieval-Augmented Generation (RAG) system for intelligent document-based question answering, enabling users to upload PDFs and interactively query their contents.

[./Screenshot (1697)]

## 🌟 Features

- PDF document upload and processing
- Advanced text chunking and embedding
- Vector storage using Pinecone
- AI-powered question answering with Mistral
- Interactive chat interface

## 🛠 Technologies Used

- Streamlit
- Pinecone
- LangChain
- Mistral AI
- HuggingFace Embeddings

## 🚀 How It Works

### Document Processing
- Upload PDF files through Streamlit interface
- Extract and chunk text using advanced splitters
- Generate high-dimensional embeddings
- Store vectorized documents in Pinecone index

### Question Answering Pipeline
- Retrieve contextually relevant document chunks
- Generate precise answers using Mistral AI
- Provide source document references

## 📦 Dependencies

```bash
streamlit
pinecone-client
langchain
transformers
mistralai
```

## 🔧 Configuration

### Required API Keys
- Pinecone API Key
- Mistral AI API Key

### Embedding Model
- Model: `BAAI/bge-large-en-v1.5`
- Dimensions: 1024
- Device: CPU/CUDA

## 💻 Usage Instructions

1. Upload PDF documents
2. Click "Process Documents"
3. Ask questions in chat interface
4. Receive AI-generated answers with source references

## 🔍 Example Workflow

User uploads research papers ➡️ Documents are chunked and embedded ➡️ User asks: "What are the key findings?" ➡️ AI retrieves relevant sections ➡️ Generates comprehensive answer


## 🔒 Security Notes

- Secrets managed via Streamlit
- Temporary file handling
- Secure API key management

## 🚧 Potential Improvements

- Multi-language support
- Enhanced embedding models
- More granular source tracking
- Advanced filtering options

## 📝 License

[MIT]

## 👥 Contributors

- [Gauri Sharan]

## 🙏 Acknowledgements

- Streamlit Community
- Pinecone
- Mistral AI
- LangChain Team
