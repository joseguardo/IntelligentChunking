"""
Complete FastAPI Backend with Document Processing Orchestrator Integration

This backend integrates all the modular components:
- utils.py: Helper functions
- llamaparse_processor.py: PDF processing
- document_indexing_pipeline.py: Structure extraction and chunking
- document_processing_orchestrator.py: Pipeline orchestration
- DocumentExplorerAPI: Backend-ready document access

File structure expected:
main.py (this file)
utils.py
llamaparse_processor.py  
document_indexing_pipeline.py
document_processing_orchestrator.py
static/
  index.html
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Dict, List, Optional
import traceback

# Base directory (where this file is located)
BASE_DIR = Path(__file__).resolve().parent

# Import all our modular components
from services.utils import sanitize_stem, validate_document_id
from services.llamaparseProcessor import LlamaParseProcessor
from services.documentIndexingPipeline import DocumentIndexingPipeline
from services.orchestrator import (
    DocumentProcessingOrchestrator, 
    PipelineConfig,
    create_orchestrator_with_defaults
)

# Import the DocumentExplorerAPI we created earlier
from services.documentExplorer.documentExplorer import DocumentExplorerAPI

# Initialize FastAPI app
app = FastAPI(title="Document Processing Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONSTANTS
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_MODEL = "text-embedding-3-small"

# Mount static files (using absolute path)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Initialize global components
document_explorer = DocumentExplorerAPI(DATA_DIR)


# =============================================================================
# MAIN ROUTES
# =============================================================================

@app.get("/")
def root():
    """Serve the main HTML page."""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "data_dir": str(DATA_DIR),
        "data_dir_exists": DATA_DIR.exists()
    }


# =============================================================================
# DOCUMENT PROCESSING ROUTES
# =============================================================================

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process PDF using the new orchestrator system.
    
    This replaces the original monolithic endpoint with a clean orchestrator-based approach
    while maintaining perfect compatibility with the original response format.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Generate document_id from filename (maintaining original logic)
    document_id = sanitize_stem(file.filename)
    
    # Handle duplicate document_ids (maintaining original logic)
    base_document_id = document_id
    counter = 1
    while True:
        doc_paths = DATA_DIR / "documents" / document_id
        if not doc_paths.exists():
            break
        document_id = f"{base_document_id}_{counter}"
        counter += 1

    print(f"Processing document: {document_id} (original: {file.filename})")

    try:
        # Create orchestrator with default configuration
        orchestrator = create_orchestrator_with_defaults(
            base_data_dir=DATA_DIR,
            # You can customize these settings:
            target_chunk_size=10000,
            chunk_tolerance=1500,
            enable_rechunking=True,
            enable_embeddings=False,  # Set to True when VectorStore is ready
            enable_search_test=False  # Set to True when VectorStore is ready
        )
        
        # Read file content
        file_content = await file.read()
        
        # Process document through orchestrator
        result = await orchestrator.process_document(
            file_content=file_content,
            filename=file.filename,
            custom_document_id=document_id
        )
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=result.state.error_summary or "Document processing failed"
            )
        
        # Print success summary (maintaining original format)
        state = result.state
        print(f"""
        PROCESSING COMPLETED SUCCESSFULLY!
        
        Summary for document: {document_id}
        ├── Original file: {file.filename}
        ├── Total processing time: {state.total_execution_time:.2f}s
        ├── Completed steps: {len(state.completed_steps)}
        ├── Failed steps: {len(state.failed_steps)}
        └── Status: {state.status}
        
        Data structure created:
        data/
        ├── documents/{document_id}/
        │   ├── original.pdf
        │   └── metadata.json
        └── processed/{document_id}/
            ├── processed_content.md
            ├── chunks/ (initial chunks)
            ├── final_chunks/ (re-chunked)
            ├── document_structure.json
            ├── indexing_log.json
            └── pipeline_log.json
        """)
        
        # Return response in exact original format
        return result.to_response_dict()

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error during document processing: {e}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )


@app.get("/documents/")
def list_documents():
    """List all processed documents."""
    try:
        result = document_explorer.list_documents()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}")
def get_document_info(document_id: str):
    """Get information about a specific document."""
    try:
        result = document_explorer.get_document_info(document_id)
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}/chunks")
def list_document_chunks(document_id: str, chunk_type: str = "chunks"):
    """List chunks for a specific document."""
    try:
        if chunk_type not in ["chunks", "final_chunks"]:
            raise HTTPException(status_code=400, detail="chunk_type must be 'chunks' or 'final_chunks'")
        
        result = document_explorer.list_chunks(document_id, chunk_type)
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}/chunks/{filename}")
def get_chunk_content(document_id: str, filename: str, chunk_type: str = "chunks"):
    """Get content of a specific chunk."""
    try:
        if chunk_type not in ["chunks", "final_chunks"]:
            raise HTTPException(status_code=400, detail="chunk_type must be 'chunks' or 'final_chunks'")
        
        result = document_explorer.get_chunk_content(document_id, filename, chunk_type)
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/{document_id}/chunks/batch")
def get_multiple_chunks_content(
    document_id: str, 
    filenames: List[str],
    chunk_type: str = "chunks"
):
    """Get content of multiple chunks."""
    try:
        if chunk_type not in ["chunks", "final_chunks"]:
            raise HTTPException(status_code=400, detail="chunk_type must be 'chunks' or 'final_chunks'")
        
        result = document_explorer.get_multiple_chunks_content(document_id, filenames, chunk_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SEARCH ROUTES
# =============================================================================

@app.get("/search/documents")
def search_documents(query: str):
    """Search documents by query."""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        result = document_explorer.search_documents(query)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/documents/{document_id}/chunks")
def search_document_chunks(document_id: str, query: str, chunk_type: str = "chunks"):
    """Search chunks within a specific document."""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if chunk_type not in ["chunks", "final_chunks"]:
            raise HTTPException(status_code=400, detail="chunk_type must be 'chunks' or 'final_chunks'")
        
        result = document_explorer.search_chunks(document_id, query, chunk_type)
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STATUS AND MONITORING ROUTES
# =============================================================================

@app.get("/status/pipeline/{document_id}")
def get_pipeline_status(document_id: str):
    """Get comprehensive pipeline status for a document."""
    try:
        # Create a temporary orchestrator to get status
        orchestrator = create_orchestrator_with_defaults(base_data_dir=DATA_DIR)
        status = orchestrator.get_pipeline_status(document_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/processed-documents")
def list_processed_documents():
    """List all processed document IDs."""
    try:
        orchestrator = create_orchestrator_with_defaults(base_data_dir=DATA_DIR)
        documents = orchestrator.list_processed_documents()
        return {
            "success": True,
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ADVANCED PROCESSING ROUTES
# =============================================================================

@app.post("/reprocess/{document_id}/chunks")
def reprocess_chunks(
    document_id: str,
    target_size: int = 10000,
    tolerance: int = 1500
):
    """Re-process chunks for a document with new parameters."""
    try:
        if not validate_document_id(document_id):
            raise HTTPException(status_code=400, detail="Invalid document_id")
        
        # Initialize indexing pipeline
        pipeline = DocumentIndexingPipeline(base_data_dir=DATA_DIR)
        
        # Check if document exists
        status = pipeline.get_document_status(document_id)
        if not status["exists"]["chunks_dir"]:
            raise HTTPException(status_code=404, detail="Document chunks not found")
        
        # Re-chunk with new parameters
        final_chunks, metadata = pipeline.rechunk_directory(
            document_id=document_id,
            target_chunk_size=target_size,
            tolerance=tolerance,
            save=True
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "final_chunks_count": len(final_chunks),
            "parameters": {
                "target_size": target_size,
                "tolerance": tolerance
            },
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, keep_original: bool = True):
    """Delete processed document data."""
    try:
        if not validate_document_id(document_id):
            raise HTTPException(status_code=400, detail="Invalid document_id")
        
        # Clean up using both processors
        llamaparse_processor = LlamaParseProcessor(base_data_dir=DATA_DIR)
        indexing_pipeline = DocumentIndexingPipeline(base_data_dir=DATA_DIR)
        
        # Cleanup processed data
        llamaparse_success = llamaparse_processor.cleanup_document(document_id, keep_original)
        pipeline_success = indexing_pipeline.cleanup_document(document_id, keep_structure=True)
        
        return {
            "success": llamaparse_success and pipeline_success,
            "document_id": document_id,
            "keep_original": keep_original,
            "cleanup_results": {
                "llamaparse_processor": llamaparse_success,
                "indexing_pipeline": pipeline_success
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CONFIGURATION ROUTES
# =============================================================================

@app.get("/config/pipeline")
def get_pipeline_config():
    """Get current pipeline configuration."""
    config = PipelineConfig()
    return {
        "base_data_dir": str(config.base_data_dir),
        "openai_model": config.openai_model,
        "max_preview_pages": config.max_preview_pages,
        "target_chunk_size": config.target_chunk_size,
        "chunk_tolerance": config.chunk_tolerance,
        "enable_rechunking": config.enable_rechunking,
        "enable_embeddings": config.enable_embeddings,
        "enable_search_test": config.enable_search_test,
        "cleanup_on_error": config.cleanup_on_error,
        "max_retries": config.max_retries
    }


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Not found", "detail": str(exc.detail)}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc.detail)}


# =============================================================================
# DEVELOPMENT/DEBUG ROUTES (Remove in production)
# =============================================================================

@app.get("/debug/data-structure")
def debug_data_structure():
    """Debug endpoint to examine data directory structure."""
    try:
        def scan_directory(path: Path, max_depth: int = 3, current_depth: int = 0):
            if current_depth > max_depth or not path.exists():
                return {}
            
            result = {"type": "directory" if path.is_dir() else "file"}
            
            if path.is_file():
                result["size"] = path.stat().st_size
            elif path.is_dir() and current_depth < max_depth:
                result["contents"] = {}
                try:
                    for item in path.iterdir():
                        result["contents"][item.name] = scan_directory(
                            item, max_depth, current_depth + 1
                        )
                except PermissionError:
                    result["contents"] = {"error": "Permission denied"}
            
            return result
        
        structure = scan_directory(DATA_DIR)
        
        return {
            "data_directory": str(DATA_DIR),
            "exists": DATA_DIR.exists(),
            "structure": structure
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/components-status")
def debug_components_status():
    """Debug endpoint to check component initialization status."""
    try:
        status = {
            "data_dir": {
                "path": str(DATA_DIR),
                "exists": DATA_DIR.exists(),
                "writable": os.access(DATA_DIR, os.W_OK) if DATA_DIR.exists() else False
            },
            "components": {}
        }
        
        # Test LlamaParseProcessor
        try:
            processor = LlamaParseProcessor(base_data_dir=DATA_DIR)
            status["components"]["llamaparse_processor"] = {
                "status": "ok",
                "base_data_dir": str(processor.base_data_dir)
            }
        except Exception as e:
            status["components"]["llamaparse_processor"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test DocumentIndexingPipeline
        try:
            pipeline = DocumentIndexingPipeline(base_data_dir=DATA_DIR)
            status["components"]["indexing_pipeline"] = {
                "status": "ok",
                "base_data_dir": str(pipeline.base_data_dir),
                "has_openai_client": pipeline.client is not None
            }
        except Exception as e:
            status["components"]["indexing_pipeline"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test DocumentExplorerAPI
        try:
            status["components"]["document_explorer"] = {
                "status": "ok",
                "repository_type": type(document_explorer.repository).__name__
            }
        except Exception as e:
            status["components"]["document_explorer"] = {
                "status": "error", 
                "error": str(e)
            }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STARTUP MESSAGE
# =============================================================================

@app.on_event("startup")
async def startup_event():
    print(f"""
    Document Processing Pipeline Backend Started!
    
    Data Directory: {DATA_DIR}
    Available Endpoints:
    - GET  / : Main HTML page
    - POST /upload/ : Upload and process PDF
    - GET  /documents/ : List all documents
    - GET  /documents/{{id}} : Get document info
    - GET  /documents/{{id}}/chunks : List document chunks
    - GET  /documents/{{id}}/chunks/{{filename}} : Get chunk content
    - GET  /search/documents?query= : Search documents
    - GET  /status/pipeline/{{id}} : Get pipeline status
    - GET  /config/pipeline : Get configuration
    - GET  /debug/data-structure : Debug data structure
    
    Ready to process documents!
    """)


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Add missing import for os.access check
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )