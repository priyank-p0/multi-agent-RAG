"""
PDF processing utilities for the Multi-Agent RAG System
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from sentence_transformers import SentenceTransformer

from config import CONFIG

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF processing and embedding for the RAG system"""
    
    def __init__(self, pdf_directory: str = None, embedding_model: str = None):
        self.pdf_directory = Path(pdf_directory or CONFIG["rag"].pdf_directory)
        self.embedding_model_name = embedding_model or CONFIG["rag"].embedding_model
        self.chunk_size = CONFIG["rag"].chunk_size
        self.chunk_overlap = CONFIG["rag"].chunk_overlap
        
        # Initialize embedding model
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace embeddings: {e}")
            try:
                # Fallback to sentence-transformers directly
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            except Exception as e2:
                logger.error(f"Failed to load any embedding model: {e2}")
                raise
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.vector_store = None
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file using available libraries"""
        
        # Try PyMuPDF first (better text extraction)
        if fitz:
            try:
                doc = fitz.open(str(pdf_path))
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except Exception as e:
                logger.warning(f"PyMuPDF failed for {pdf_path}: {e}")
        
        # Fallback to PyPDF2
        if PyPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    return text
            except Exception as e:
                logger.warning(f"PyPDF2 failed for {pdf_path}: {e}")
        
        raise RuntimeError(f"Could not extract text from {pdf_path}. No PDF library available.")
    
    def load_pdfs_from_directory(self) -> List[Document]:
        """Load all PDF files from the configured directory"""
        
        if not self.pdf_directory.exists():
            logger.warning(f"PDF directory {self.pdf_directory} does not exist. Creating it...")
            self.pdf_directory.mkdir(parents=True, exist_ok=True)
            return []
        
        documents = []
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in pdf_files:
            try:
                logger.info(f"Processing {pdf_path.name}...")
                text = self.extract_text_from_pdf(pdf_path)
                
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": str(pdf_path),
                            "filename": pdf_path.name,
                            "file_type": "pdf"
                        }
                    )
                    documents.append(doc)
                    logger.info(f"Successfully processed {pdf_path.name}")
                else:
                    logger.warning(f"No text extracted from {pdf_path.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                continue
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for better retrieval"""
        
        if not documents:
            return []
        
        logger.info(f"Chunking {len(documents)} documents...")
        
        chunked_docs = []
        for doc in documents:
            try:
                chunks = self.text_splitter.split_documents([doc])
                chunked_docs.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking document {doc.metadata.get('filename', 'unknown')}: {e}")
                continue
        
        # Remove duplicate chunks
        unique_chunks = []
        seen_content = set()
        
        for chunk in chunked_docs:
            content_hash = hash(chunk.page_content.strip())
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
        
        logger.info(f"Created {len(unique_chunks)} unique chunks from {len(chunked_docs)} total chunks")
        return unique_chunks
    
    def create_vector_store(self, documents: List[Document] = None) -> FAISS:
        """Create FAISS vector store from documents"""
        
        if documents is None:
            documents = self.load_pdfs_from_directory()
        
        if not documents:
            logger.warning("No documents to create vector store from")
            return None
        
        # Chunk the documents
        chunked_docs = self.chunk_documents(documents)
        
        if not chunked_docs:
            logger.warning("No chunks created from documents")
            return None
        
        logger.info(f"Creating vector store from {len(chunked_docs)} chunks...")
        
        try:
            self.vector_store = FAISS.from_documents(
                documents=chunked_docs,
                embedding=self.embedding_model,
                distance_strategy=DistanceStrategy.COSINE
            )
            logger.info("Vector store created successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def save_vector_store(self, path: str = None):
        """Save the vector store to disk"""
        if not self.vector_store:
            raise ValueError("No vector store to save")
        
        save_path = path or CONFIG["rag"].vector_store_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        self.vector_store.save_local(save_path)
        logger.info(f"Vector store saved to {save_path}")
    
    def load_vector_store(self, path: str = None) -> FAISS:
        """Load vector store from disk"""
        load_path = path or CONFIG["rag"].vector_store_path
        
        if not os.path.exists(load_path):
            logger.warning(f"Vector store not found at {load_path}")
            return None
        
        try:
            self.vector_store = FAISS.load_local(
                load_path, 
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from {load_path}")
            return self.vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None
    
    def get_or_create_vector_store(self, force_recreate: bool = False) -> FAISS:
        """Get existing vector store or create new one"""
        
        if not force_recreate:
            # Try to load existing vector store
            vector_store = self.load_vector_store()
            if vector_store:
                return vector_store
        
        # Create new vector store
        logger.info("Creating new vector store from PDF files...")
        vector_store = self.create_vector_store()
        
        if vector_store:
            # Save for future use
            try:
                self.save_vector_store()
            except Exception as e:
                logger.warning(f"Could not save vector store: {e}")
        
        return vector_store
    
    def search_documents(self, query: str, k: int = None) -> List[Document]:
        """Search for relevant documents"""
        if not self.vector_store:
            self.vector_store = self.get_or_create_vector_store()
        
        if not self.vector_store:
            logger.error("No vector store available for search")
            return []
        
        k = k or CONFIG["rag"].max_retrieval_docs
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_pdf_count(self) -> int:
        """Get the number of PDF files in the directory"""
        if not self.pdf_directory.exists():
            return 0
        
        return len(list(self.pdf_directory.glob("*.pdf")))
    
    def list_pdf_files(self) -> List[str]:
        """List all PDF files in the directory"""
        if not self.pdf_directory.exists():
            return []
        
        return [f.name for f in self.pdf_directory.glob("*.pdf")]


def create_pdf_readme():
    """Create a README file for the PDF directory"""
    pdf_dir = Path(CONFIG["rag"].pdf_directory)
    readme_path = pdf_dir / "README.md"
    
    readme_content = """# PDF Files Directory

This directory contains PDF files that will be processed by the Multi-Agent RAG system.

## Usage

1. Place your PDF files in this directory
2. The system will automatically process them when you run the retrieval agent
3. Supported formats: PDF files (.pdf extension)

## Processing

- PDFs are automatically chunked into smaller segments for better retrieval
- Text is extracted and embedded using sentence transformers
- A vector store is created for fast similarity search

## Files

Currently contains the following PDF files:
(This will be updated automatically when you add files)

## Notes

- Ensure PDF files are readable and contain extractable text
- Large files may take longer to process
- The system will create a vector store cache to speed up subsequent runs
"""
    
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    return readme_path
