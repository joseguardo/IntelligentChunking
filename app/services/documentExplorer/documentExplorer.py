"""
Backend-ready Document Explorer
Modular implementation with data models, repository pattern, and service layer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import re
from abc import ABC, abstractmethod


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class DocumentInfo:
    """Document metadata model."""
    id: str
    chunks_count: int
    final_chunks_count: int
    path: Path
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "chunks_count": self.chunks_count,
            "final_chunks_count": self.final_chunks_count,
            "path": str(self.path)
        }


@dataclass
class ChunkInfo:
    """Chunk metadata model."""
    filename: str
    title: str
    path: Path
    size: int
    document_id: str
    chunk_type: str  # 'chunks' or 'final_chunks'
    
    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "title": self.title,
            "path": str(self.path),
            "size": self.size,
            "document_id": self.document_id,
            "chunk_type": self.chunk_type
        }


@dataclass
class ChunkContent:
    """Chunk content model."""
    chunk_info: ChunkInfo
    content: str
    clean_content: str  # Content without front-matter
    
    def to_dict(self) -> dict:
        return {
            "chunk_info": self.chunk_info.to_dict(),
            "content": self.content,
            "clean_content": self.clean_content
        }


# =============================================================================
# EXCEPTIONS
# =============================================================================

class DocumentExplorerError(Exception):
    """Base exception for document explorer."""
    pass


class DocumentNotFoundError(DocumentExplorerError):
    """Raised when a document is not found."""
    pass


class ChunkNotFoundError(DocumentExplorerError):
    """Raised when a chunk is not found."""
    pass


class ProcessedDirectoryNotFoundError(DocumentExplorerError):
    """Raised when processed directory doesn't exist."""
    pass


# =============================================================================
# REPOSITORY INTERFACES
# =============================================================================

class DocumentRepository(ABC):
    """Abstract repository for document operations."""
    
    @abstractmethod
    def get_all_documents(self) -> List[DocumentInfo]:
        """Get all available documents."""
        pass
    
    @abstractmethod
    def get_document_by_id(self, document_id: str) -> Optional[DocumentInfo]:
        """Get document by ID."""
        pass
    
    @abstractmethod
    def get_document_chunks(self, document_id: str, chunk_type: str = "chunks") -> List[ChunkInfo]:
        """Get chunks for a specific document."""
        pass
    
    @abstractmethod
    def get_chunk_content(self, chunk_info: ChunkInfo) -> ChunkContent:
        """Get content of a specific chunk."""
        pass


# =============================================================================
# FILE SYSTEM REPOSITORY IMPLEMENTATION
# =============================================================================

class FileSystemDocumentRepository(DocumentRepository):
    """File system implementation of document repository."""
    
    def __init__(self, base_data_dir: Union[str, Path] = "data"):
        self.base_data_dir = Path(base_data_dir)
        self.processed_dir = self.base_data_dir / "processed"
    
    def get_all_documents(self) -> List[DocumentInfo]:
        """Get all available documents from processed directory."""
        if not self.processed_dir.exists():
            raise ProcessedDirectoryNotFoundError(f"Processed directory not found: {self.processed_dir}")
        
        documents = []
        for doc_dir in self.processed_dir.iterdir():
            if doc_dir.is_dir():
                doc_info = self._create_document_info(doc_dir)
                documents.append(doc_info)
        
        return sorted(documents, key=lambda x: x.id)
    
    def get_document_by_id(self, document_id: str) -> Optional[DocumentInfo]:
        """Get specific document by ID."""
        doc_dir = self.processed_dir / document_id
        if doc_dir.exists() and doc_dir.is_dir():
            return self._create_document_info(doc_dir)
        return None
    
    def get_document_chunks(self, document_id: str, chunk_type: str = "chunks") -> List[ChunkInfo]:
        """Get chunks for a specific document."""
        doc_info = self.get_document_by_id(document_id)
        if not doc_info:
            raise DocumentNotFoundError(f"Document not found: {document_id}")
        
        chunks_dir = doc_info.path / chunk_type
        if not chunks_dir.exists():
            return []
        
        chunk_files = list(chunks_dir.glob("*.md"))
        chunk_files.sort(key=self._natural_sort_key)
        
        chunks = []
        for chunk_file in chunk_files:
            chunk_info = self._create_chunk_info(chunk_file, document_id, chunk_type)
            chunks.append(chunk_info)
        
        return chunks
    
    def get_chunk_content(self, chunk_info: ChunkInfo) -> ChunkContent:
        """Get content of a specific chunk."""
        try:
            content = chunk_info.path.read_text(encoding="utf-8")
            clean_content = self._remove_front_matter(content)
            
            return ChunkContent(
                chunk_info=chunk_info,
                content=content,
                clean_content=clean_content
            )
        except Exception as e:
            raise ChunkNotFoundError(f"Error reading chunk {chunk_info.filename}: {e}")
    
    def _create_document_info(self, doc_dir: Path) -> DocumentInfo:
        """Create DocumentInfo from directory."""
        chunks_count = 0
        final_chunks_count = 0
        
        chunks_dir = doc_dir / "chunks"
        if chunks_dir.exists():
            chunks_count = len(list(chunks_dir.glob("*.md")))
        
        final_chunks_dir = doc_dir / "final_chunks"
        if final_chunks_dir.exists():
            final_chunks_count = len(list(final_chunks_dir.glob("*.md")))
        
        return DocumentInfo(
            id=doc_dir.name,
            chunks_count=chunks_count,
            final_chunks_count=final_chunks_count,
            path=doc_dir
        )
    
    def _create_chunk_info(self, chunk_file: Path, document_id: str, chunk_type: str) -> ChunkInfo:
        """Create ChunkInfo from file."""
        title = self._extract_title_from_file(chunk_file)
        
        try:
            content = chunk_file.read_text(encoding="utf-8")
            size = len(content)
        except Exception:
            size = 0
        
        return ChunkInfo(
            filename=chunk_file.name,
            title=title,
            path=chunk_file,
            size=size,
            document_id=document_id,
            chunk_type=chunk_type
        )
    
    def _extract_title_from_file(self, file_path: Path) -> str:
        """Extract title from markdown file or filename."""
        try:
            content = file_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("# "):
                    return line[2:].strip()
        except Exception:
            pass
        
        # Fallback to filename
        filename = file_path.stem
        clean_name = re.sub(r'^(chunk_\d+_|section_\d+_)', '', filename)
        return clean_name.replace('_', ' ').title()
    
    def _remove_front_matter(self, content: str) -> str:
        """Remove front-matter comments from content."""
        lines = content.splitlines()
        content_start = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith("<!-- ") and line.strip().endswith(" -->"):
                content_start = i + 1
            else:
                break
        
        return "\n".join(lines[content_start:]).strip()
    
    def _natural_sort_key(self, file_path: Path) -> list:
        """Sort key that handles numbers in filenames naturally."""
        parts = re.split(r'(\d+)', file_path.name)
        return [int(part) if part.isdigit() else part.lower() for part in parts]


# =============================================================================
# SERVICE LAYER
# =============================================================================

class DocumentService:
    """Service layer for document operations."""
    
    def __init__(self, repository: DocumentRepository):
        self.repository = repository
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents as dictionaries."""
        documents = self.repository.get_all_documents()
        return [doc.to_dict() for doc in documents]
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """Get specific document by ID."""
        document = self.repository.get_document_by_id(document_id)
        return document.to_dict() if document else None
    
    def get_document_chunks(self, document_id: str, chunk_type: str = "chunks") -> List[Dict]:
        """Get chunks for a document."""
        chunks = self.repository.get_document_chunks(document_id, chunk_type)
        return [chunk.to_dict() for chunk in chunks]
    
    def get_chunk_content(self, document_id: str, filename: str, chunk_type: str = "chunks") -> Optional[Dict]:
        """Get content of a specific chunk."""
        chunks = self.repository.get_document_chunks(document_id, chunk_type)
        
        # Find chunk by filename
        target_chunk = None
        for chunk in chunks:
            if chunk.filename == filename:
                target_chunk = chunk
                break
        
        if not target_chunk:
            return None
        
        chunk_content = self.repository.get_chunk_content(target_chunk)
        return chunk_content.to_dict()
    
    def search_documents(self, query: str) -> List[Dict]:
        """Search documents by name/ID."""
        all_documents = self.repository.get_all_documents()
        query_lower = query.lower()
        
        matching_docs = [
            doc for doc in all_documents 
            if query_lower in doc.id.lower()
        ]
        
        return [doc.to_dict() for doc in matching_docs]
    
    def search_chunks(self, document_id: str, query: str, chunk_type: str = "chunks") -> List[Dict]:
        """Search chunks by title."""
        chunks = self.repository.get_document_chunks(document_id, chunk_type)
        query_lower = query.lower()
        
        matching_chunks = [
            chunk for chunk in chunks 
            if query_lower in chunk.title.lower()
        ]
        
        return [chunk.to_dict() for chunk in matching_chunks]


# =============================================================================
# MAIN BACKEND API CLASS
# =============================================================================

class DocumentExplorerAPI:
    """
    Backend-ready API for document exploration.
    Provides clean interface for loading documents, chunks, and content.
    """
    
    def __init__(self, base_data_dir: Union[str, Path] = "data"):
        self.repository = FileSystemDocumentRepository(base_data_dir)
        self.service = DocumentService(self.repository)
    
    # Document operations
    def list_documents(self) -> Dict:
        """List all available documents."""
        try:
            documents = self.service.get_all_documents()
            return {
                "success": True,
                "data": documents,
                "count": len(documents)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    def get_document_info(self, document_id: str) -> Dict:
        """Get information about a specific document."""
        try:
            document = self.service.get_document(document_id)
            if document:
                return {
                    "success": True,
                    "data": document
                }
            else:
                return {
                    "success": False,
                    "error": f"Document '{document_id}' not found",
                    "data": None
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    # Chunk operations
    def list_chunks(self, document_id: str, chunk_type: str = "chunks") -> Dict:
        """List chunks for a specific document."""
        try:
            chunks = self.service.get_document_chunks(document_id, chunk_type)
            return {
                "success": True,
                "data": chunks,
                "count": len(chunks),
                "document_id": document_id,
                "chunk_type": chunk_type
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    def get_chunk_content(self, document_id: str, filename: str, chunk_type: str = "chunks") -> Dict:
        """Get content of a specific chunk."""
        try:
            chunk_content = self.service.get_chunk_content(document_id, filename, chunk_type)
            if chunk_content:
                return {
                    "success": True,
                    "data": chunk_content
                }
            else:
                return {
                    "success": False,
                    "error": f"Chunk '{filename}' not found in document '{document_id}'",
                    "data": None
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    # Search operations
    def search_documents(self, query: str) -> Dict:
        """Search documents by query."""
        try:
            documents = self.service.search_documents(query)
            return {
                "success": True,
                "data": documents,
                "count": len(documents),
                "query": query
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    def search_chunks(self, document_id: str, query: str, chunk_type: str = "chunks") -> Dict:
        """Search chunks by query."""
        try:
            chunks = self.service.search_chunks(document_id, query, chunk_type)
            return {
                "success": True,
                "data": chunks,
                "count": len(chunks),
                "document_id": document_id,
                "query": query,
                "chunk_type": chunk_type
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    # Batch operations
    def get_multiple_chunks_content(self, document_id: str, filenames: List[str], 
                                  chunk_type: str = "chunks") -> Dict:
        """Get content of multiple chunks."""
        results = []
        errors = []
        
        for filename in filenames:
            try:
                chunk_content = self.service.get_chunk_content(document_id, filename, chunk_type)
                if chunk_content:
                    results.append(chunk_content)
                else:
                    errors.append(f"Chunk '{filename}' not found")
            except Exception as e:
                errors.append(f"Error loading '{filename}': {str(e)}")
        
        return {
            "success": len(errors) == 0,
            "data": results,
            "count": len(results),
            "errors": errors,
            "document_id": document_id,
            "chunk_type": chunk_type
        }


# =============================================================================
# EXTERNAL DEPENDENCIES (flagged as requested)
# =============================================================================

# NOTE: The following functions/imports from the original code are external dependencies:
# - re module (used for regex operations) ✓ (standard library)
# - pathlib.Path ✓ (standard library)  
# - The embedding-related code at the end appears unrelated to the main class
# - All other dependencies are now properly contained within this module