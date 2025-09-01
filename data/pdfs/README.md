# PDF Files Directory

This directory contains PDF files that will be processed by the Multi-Agent RAG system.

## Usage

1. Place your PDF files in this directory
2. The system will automatically process them when you run the retrieval agent
3. Supported formats: PDF files (.pdf extension)

## Processing

- PDFs are automatically chunked into smaller segments for better retrieval
- Text is extracted and embedded using sentence transformers
- A vector store is created for fast similarity search

## Notes

- Ensure PDF files are readable and contain extractable text
- Large files may take longer to process
- The system will create a vector store cache to speed up subsequent runs
