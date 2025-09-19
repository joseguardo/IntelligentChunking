"""
llamaparse_processor.py - Enhanced PDF processing using LlamaIndex's LlamaParse service.

This module handles document processing using LlamaIndex's LlamaParse service with
improved error handling, logging, and integration with the orchestrator system.
"""

import json
import os
import time
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

try:
    import tiktoken
    from dotenv import load_dotenv
    from llama_cloud_services import LlamaParse
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False

# Import our utils
from services.utils import (
    sanitize_stem,
    validate_document_id,
    ensure_directory,
    safe_write_json,
    safe_read_json,
    get_file_size_safely,
    create_log_entry,
    append_to_log
)


# Constants
TOKEN_MODEL = "gpt-4"  # Default model for token counting
DEFAULT_RESULT_TYPE = "markdown"


class LlamaParseProcessor:
    """
    Handles document processing using LlamaIndex's LlamaParse service.
    Uses new data structure: data/documents/{document_id}/ and data/processed/{document_id}/
    
    Features:
    - Robust error handling with detailed logging
    - Automatic document ID generation and validation
    - Standardized metadata and logging format
    - Proper cleanup on failures
    - Integration with orchestrator system
    """
    
    def __init__(
        self, 
        base_data_dir: Path = Path("data"),
        result_type: str = DEFAULT_RESULT_TYPE,
        auto_load_env: bool = True
    ):
        """
        Initializes the LlamaParseProcessor.
        
        Args:
            base_data_dir: Base directory for data storage (default: "data")
            result_type: LlamaParse result type (default: "markdown")
            auto_load_env: Whether to automatically load environment variables
            
        Raises:
            ImportError: If LlamaParse library is not available
            ValueError: If API key is not found
        """
        if not LLAMAPARSE_AVAILABLE:
            raise ImportError(
                "LlamaParse library is not available. "
                "Install with: pip install llama-parse"
            )
        
        # Load environment if requested
        if auto_load_env:
            load_dotenv()
        
        # Get API key
        api_key = os.getenv("LLAMAPARSE_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing LlamaParse API key. Set LLAMAPARSE_API_KEY environment variable."
            )
        
        # Initialize parser
        self.parser = LlamaParse(
            api_key=api_key,
            result_type=result_type,
            verbose=True
        )
        
        # Setup directories
        self.base_data_dir = Path(base_data_dir)
        self.documents_dir = self.base_data_dir / "documents"
        self.processed_dir = self.base_data_dir / "processed"
        
        # Ensure base directories exist
        ensure_directory(self.documents_dir)
        ensure_directory(self.processed_dir)
        
        print(f"✅ LlamaParseProcessor initialized with base_data_dir: {self.base_data_dir}")

    def generate_document_id(self, file_path: Path, custom_id: Optional[str] = None) -> str:
        """
        Generates and validates a document ID.
        
        Args:
            file_path: Source file path
            custom_id: Custom document ID (optional)
            
        Returns:
            Valid document ID
            
        Raises:
            ValueError: If generated ID is invalid
        """
        if custom_id:
            document_id = custom_id
        else:
            document_id = sanitize_stem(file_path.name)
        
        # Validate document ID
        if not validate_document_id(document_id):
            # Fallback to safe default
            document_id = f"doc_{int(time.time())}"
        
        return document_id

    def get_document_paths(self, document_id: str) -> Dict[str, Path]:
        """
        Gets all relevant paths for a document.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            Dictionary with all document paths
        """
        doc_dir = self.documents_dir / document_id
        processed_dir = self.processed_dir / document_id
        
        return {
            # Document storage paths
            "document_dir": doc_dir,
            "original_file": doc_dir / "original.pdf",  # Will adjust extension as needed
            "metadata_file": doc_dir / "metadata.json",
            
            # Processed content paths  
            "processed_dir": processed_dir,
            "processed_content": processed_dir / "processed_content.md",
            "processing_log": processed_dir / "processing_log.json"
        }

    def _save_original_file(self, source_file: Path, document_id: str) -> bool:
        """
        Saves the original file to the documents directory.
        
        Args:
            source_file: Path to source file
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        try:
            paths = self.get_document_paths(document_id)
            ensure_directory(paths["document_dir"])
            
            # Determine target filename with correct extension
            target_file = paths["document_dir"] / f"original{source_file.suffix}"
            
            # Copy file
            shutil.copy2(source_file, target_file)
            
            # Update paths to reflect actual filename
            paths = self.get_document_paths(document_id)
            paths["original_file"] = target_file
            
            print(f"✅ Original file saved: {target_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving original file: {e}")
            return False

    def _save_document_metadata(self, metadata: Dict[str, Any], document_id: str) -> bool:
        """
        Saves document metadata to the documents directory.
        
        Args:
            metadata: Document metadata
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        try:
            paths = self.get_document_paths(document_id)
            
            # Create comprehensive metadata
            doc_metadata = {
                "document_id": document_id,
                "created_at": datetime.now().isoformat(),
                "file_info": {
                    "original_filename": metadata.get("original_filename", "unknown"),
                    "file_size": metadata.get("file_size", 0),
                    "file_type": metadata.get("file_type", "unknown")
                },
                "processing_info": {
                    "processor": "LlamaParse",
                    "result_type": DEFAULT_RESULT_TYPE,
                    "success": metadata.get("success", False),
                    "processing_time": metadata.get("processing_time", 0),
                    "pages_processed": metadata.get("pages_processed", 0),
                    "token_count": metadata.get("token_count", 0),
                    "content_length": metadata.get("content_length", 0)
                },
                "error_info": {
                    "error_message": metadata.get("error")
                } if metadata.get("error") else None
            }
            
            return safe_write_json(paths["metadata_file"], doc_metadata)
            
        except Exception as e:
            print(f"❌ Error saving document metadata: {e}")
            return False

    def _log_processing_step(
        self, 
        document_id: str, 
        operation: str, 
        success: bool,
        duration: float = 0.0,
        data: Dict[str, Any] = None,
        error: str = None
    ) -> None:
        """
        Logs a processing step using standardized format.
        
        Args:
            document_id: Document identifier
            operation: Operation name
            success: Whether operation succeeded
            duration: Operation duration in seconds
            data: Additional operation data
            error: Error message if failed
        """
        try:
            paths = self.get_document_paths(document_id)
            ensure_directory(paths["processed_dir"])
            
            log_entry = create_log_entry(
                operation=operation,
                success=success,
                duration=duration,
                data=data,
                error=error
            )
            
            # Add document context
            log_entry["document_id"] = document_id
            log_entry["processor"] = "LlamaParse"
            
            append_to_log(paths["processing_log"], log_entry)
            
        except Exception as e:
            print(f"⚠️ Error logging processing step: {e}")

    def count_tokens(self, text: str, model_name: str = TOKEN_MODEL) -> int:
        """
        Counts tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            model_name: Model name for tokenizer
            
        Returns:
            Number of tokens, or 0 if error
        """
        try:
            enc = tiktoken.encoding_for_model(model_name)
            return len(enc.encode(text))
        except Exception as e:
            print(f"⚠️ Error counting tokens: {e}")
            return 0

    def process_file(
        self, 
        file_path: Path, 
        document_id: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Processes a file using LlamaParse and returns markdown content with metadata.
        
        Args:
            file_path: Path to file to process
            document_id: Custom document ID (optional)
            
        Returns:
            Tuple of (markdown_content, metadata)
        """
        # Generate and validate document ID
        if document_id is None:
            document_id = self.generate_document_id(file_path)
        elif not validate_document_id(document_id):
            raise ValueError(f"Invalid document_id: {document_id}")
        
        print(f"🚀 Starting processing for document: {document_id}")
        
        start_time = time.time()
        
        # Initialize metadata
        base_metadata = {
            "document_id": document_id,
            "original_filename": file_path.name,
            "file_size": get_file_size_safely(file_path),
            "file_type": file_path.suffix.lower(),
            "start_time": datetime.now().isoformat()
        }
        
        try:
            # Log start of processing
            self._log_processing_step(
                document_id=document_id,
                operation="process_file_start",
                success=True,
                data=base_metadata.copy()
            )
            
            # Process with LlamaParse
            print(f"📄 Processing file: {file_path}")
            documents = self.parser.load_data(str(file_path))
            
            # Combine all document text
            markdown_content = "\n\n".join([doc.text for doc in documents])
            processing_time = time.time() - start_time
            
            # Count tokens
            token_count = self.count_tokens(markdown_content)
            
            # Create success metadata
            metadata = {
                **base_metadata,
                "success": True,
                "processing_time": processing_time,
                "token_count": token_count,
                "content_length": len(markdown_content),
                "pages_processed": len(documents),
                "error": None,
                "end_time": datetime.now().isoformat()
            }
            
            # Save original file and metadata
            file_saved = self._save_original_file(file_path, document_id)
            metadata_saved = self._save_document_metadata(metadata, document_id)
            
            # Log successful processing
            self._log_processing_step(
                document_id=document_id,
                operation="process_file_complete",
                success=True,
                duration=processing_time,
                data={
                    "token_count": token_count,
                    "content_length": len(markdown_content),
                    "pages_processed": len(documents),
                    "file_saved": file_saved,
                    "metadata_saved": metadata_saved
                }
            )
            
            print(f"✅ Processing completed successfully:")
            print(f"   - Duration: {processing_time:.2f}s")
            print(f"   - Pages: {len(documents)}")
            print(f"   - Tokens: {token_count:,}")
            print(f"   - Content length: {len(markdown_content):,} chars")
            
            return markdown_content, metadata

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            # Create error metadata
            metadata = {
                **base_metadata,
                "success": False,
                "processing_time": processing_time,
                "token_count": 0,
                "content_length": 0,
                "pages_processed": 0,
                "error": error_msg,
                "end_time": datetime.now().isoformat()
            }
            
            # Save metadata even on failure
            self._save_document_metadata(metadata, document_id)
            
            # Log error
            self._log_processing_step(
                document_id=document_id,
                operation="process_file_error",
                success=False,
                duration=processing_time,
                error=error_msg
            )
            
            print(f"❌ Processing failed: {error_msg}")
            traceback.print_exc()
            
            return "", metadata

    def save_processed_content(
        self, 
        content: str, 
        document_id: str
    ) -> bool:
        """
        Saves processed markdown content to the processed directory.
        
        Args:
            content: Processed markdown content
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        try:
            paths = self.get_document_paths(document_id)
            ensure_directory(paths["processed_dir"])
            
            # Write content
            with open(paths["processed_content"], "w", encoding="utf-8") as f:
                f.write(content)
            
            # Log the save operation
            self._log_processing_step(
                document_id=document_id,
                operation="save_processed_content",
                success=True,
                data={
                    "content_length": len(content),
                    "output_path": str(paths["processed_content"])
                }
            )
            
            print(f"✅ Processed content saved: {paths['processed_content']}")
            return True
            
        except Exception as e:
            error_msg = f"Error saving processed content: {e}"
            print(f"❌ {error_msg}")
            
            # Log the error
            self._log_processing_step(
                document_id=document_id,
                operation="save_processed_content",
                success=False,
                error=error_msg
            )
            
            return False

    def process_and_save(
        self, 
        file_path: Path, 
        document_id: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Complete processing workflow: process file and save results.
        
        This is the main method that orchestrators should use.
        
        Args:
            file_path: Path to file to process
            document_id: Custom document ID (optional)
            
        Returns:
            Tuple of (content, metadata) where metadata includes save status
        """
        # Process the file
        content, metadata = self.process_file(file_path, document_id)
        
        if metadata["success"]:
            # Save processed content
            save_success = self.save_processed_content(content, metadata["document_id"])
            metadata["content_saved"] = save_success
            metadata["output_path"] = str(self.get_document_paths(metadata["document_id"])["processed_content"])
            
            # Update overall success status
            metadata["success"] = save_success
            
            if not save_success:
                metadata["error"] = "Failed to save processed content"
        
        return content, metadata

    def get_processing_status(self, document_id: str) -> Dict[str, Any]:
        """
        Gets the current processing status for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary with processing status
        """
        paths = self.get_document_paths(document_id)
        
        status = {
            "document_id": document_id,
            "exists": {
                "document_dir": paths["document_dir"].exists(),
                "original_file": paths.get("original_file", Path("")).exists() if paths.get("original_file") else False,
                "metadata_file": paths["metadata_file"].exists(),
                "processed_dir": paths["processed_dir"].exists(),
                "processed_content": paths["processed_content"].exists(),
                "processing_log": paths["processing_log"].exists()
            },
            "metadata": safe_read_json(paths["metadata_file"]),
            "last_processing": None
        }
        
        # Get last processing entry from log
        if paths["processing_log"].exists():
            try:
                with open(paths["processing_log"], 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                if isinstance(logs, list) and logs:
                    status["last_processing"] = logs[-1]
                elif isinstance(logs, dict):
                    status["last_processing"] = logs
            except:
                pass
        
        return status

    def cleanup_document(self, document_id: str, keep_original: bool = True) -> bool:
        """
        Cleans up processing artifacts for a document.
        
        Args:
            document_id: Document identifier
            keep_original: Whether to keep original file and metadata
            
        Returns:
            True if successful
        """
        try:
            paths = self.get_document_paths(document_id)
            
            # Always clean processed directory
            if paths["processed_dir"].exists():
                shutil.rmtree(paths["processed_dir"])
                print(f"🗑️ Cleaned processed directory: {paths['processed_dir']}")
            
            # Optionally clean document directory
            if not keep_original and paths["document_dir"].exists():
                shutil.rmtree(paths["document_dir"])
                print(f"🗑️ Cleaned document directory: {paths['document_dir']}")
            
            print(f"✅ Cleanup completed for document: {document_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error cleaning up document {document_id}: {e}")
            return False

    def list_documents(self) -> List[str]:
        """
        Lists all processed document IDs.
        
        Returns:
            List of document IDs
        """
        try:
            if not self.documents_dir.exists():
                return []
            
            return [
                d.name for d in self.documents_dir.iterdir() 
                if d.is_dir() and validate_document_id(d.name)
            ]
            
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
            return []


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# For backward compatibility with existing code
def process_file_legacy(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Legacy interface for processing files.
    
    Args:
        file_path: String path to file
        
    Returns:
        Tuple of (content, metadata)
    """
    processor = LlamaParseProcessor()
    return processor.process_and_save(Path(file_path))