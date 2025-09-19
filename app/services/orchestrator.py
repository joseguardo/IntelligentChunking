"""
document_processing_orchestrator.py - Complete document processing pipeline orchestrator.

This module provides a centralized orchestrator that manages the complete document processing 
pipeline with modular steps, error recovery, progress tracking, and perfect compatibility 
with all existing components.
"""

import time
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import traceback
import shutil

# Import our components
from services.utils import (
    sanitize_stem,
    validate_document_id,
    ensure_directory,
    create_log_entry,
    append_to_log,
    calculate_chunk_statistics
)
from services.llamaparseProcessor import LlamaParseProcessor
from services.documentIndexingPipeline import DocumentIndexingPipeline


# =============================================================================
# CONFIGURATION AND STATE MODELS
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the document processing pipeline."""
    
    # Base settings
    base_data_dir: Path = field(default_factory=lambda: Path("data"))
    
    # API Keys (will be loaded from environment)
    llamaparse_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Processing settings
    openai_model: str = "gpt-4.1-2025-04-14"
    max_preview_pages: int = 10
    
    # Chunking settings
    target_chunk_size: int = 10000
    chunk_tolerance: int = 1500
    
    # Step controls
    enable_pdf_processing: bool = True
    enable_structure_extraction: bool = True
    enable_initial_chunking: bool = True
    enable_rechunking: bool = True
    enable_embeddings: bool = True  # For future VectorStore integration
    enable_search_test: bool = True  # For future VectorStore integration
    
    # Error handling
    cleanup_on_error: bool = True
    max_retries: int = 2
    continue_on_optional_failure: bool = True
    
    # Performance
    max_workers: int = 8


@dataclass 
class StepResult:
    """Result of a pipeline step execution."""
    step_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    retry_count: int = 0


@dataclass
class PipelineState:
    """Complete state of the document processing pipeline."""
    document_id: str
    original_filename: str
    status: str  # "running", "completed", "failed", "cancelled"
    current_step: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Step tracking
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    
    # Overall results
    total_execution_time: float = 0.0
    success: bool = False
    error_summary: Optional[str] = None


@dataclass
class ProcessingResult:
    """Final result of document processing."""
    success: bool
    document_id: str
    state: PipelineState
    
    # Compatibility with original endpoint response format
    response_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_response_dict(self) -> Dict[str, Any]:
        """Convert to response format compatible with original endpoint."""
        return self.response_data


# =============================================================================
# MAIN ORCHESTRATOR CLASS  
# =============================================================================

class DocumentProcessingOrchestrator:
    """
    Centralized orchestrator for the complete document processing pipeline.
    
    Features:
    - Modular step execution with enable/disable controls
    - Comprehensive error handling and recovery
    - Progress tracking and status reporting
    - Perfect compatibility with existing components
    - Resource management and cleanup
    - Detailed logging and metrics
    
    Pipeline Steps:
    1. PDF Processing (Required) - Convert PDF to markdown
    2. Structure Extraction (Required) - Extract document structure 
    3. Initial Chunking (Required) - Create initial chunks
    4. Re-chunking (Optional) - Advanced re-chunking 
    5. Embeddings (Optional) - Generate vector embeddings
    6. Search Test (Optional) - Test search functionality
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.llamaparse_processor: Optional[LlamaParseProcessor] = None
        self.indexing_pipeline: Optional[DocumentIndexingPipeline] = None
        # self.vector_store: Optional[VectorStore] = None  # Future implementation
        
        # State tracking
        self.current_state: Optional[PipelineState] = None
        
        # Setup directories
        ensure_directory(self.config.base_data_dir)
        ensure_directory(self.config.base_data_dir / "temp")
        
        print(f"DocumentProcessingOrchestrator initialized with base_data_dir: {self.config.base_data_dir}")

    def _initialize_components(self) -> Tuple[bool, str]:
        """
        Initialize all required components.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Initialize LlamaParseProcessor
            if self.config.enable_pdf_processing:
                try:
                    self.llamaparse_processor = LlamaParseProcessor(
                        base_data_dir=self.config.base_data_dir,
                        auto_load_env=True
                    )
                    print("LlamaParseProcessor initialized successfully")
                except Exception as e:
                    return False, f"Failed to initialize LlamaParseProcessor: {e}"
            
            # Initialize DocumentIndexingPipeline
            if (self.config.enable_structure_extraction or 
                self.config.enable_initial_chunking or 
                self.config.enable_rechunking):
                try:
                    self.indexing_pipeline = DocumentIndexingPipeline(
                        model=self.config.openai_model,
                        base_data_dir=self.config.base_data_dir,
                        max_preview_pages=self.config.max_preview_pages,
                        auto_env=True
                    )
                    print("DocumentIndexingPipeline initialized successfully")
                except Exception as e:
                    return False, f"Failed to initialize DocumentIndexingPipeline: {e}"
            
            # Future: Initialize VectorStore
            # if (self.config.enable_embeddings or self.config.enable_search_test):
            #     try:
            #         self.vector_store = VectorStore(base_data_dir=self.config.base_data_dir)
            #         print("VectorStore initialized successfully")
            #     except Exception as e:
            #         return False, f"Failed to initialize VectorStore: {e}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Component initialization failed: {e}"

    def _generate_document_id(self, filename: str, custom_id: Optional[str] = None) -> str:
        """
        Generate and validate document ID, handling collisions.
        
        Args:
            filename: Original filename
            custom_id: Custom document ID (optional)
            
        Returns:
            Valid, unique document ID
        """
        # Use custom ID or generate from filename
        if custom_id:
            base_document_id = custom_id
        else:
            base_document_id = sanitize_stem(filename)
        
        # Ensure valid document ID
        if not validate_document_id(base_document_id):
            base_document_id = f"doc_{int(time.time())}"
        
        # Handle collisions by checking if document already exists
        document_id = base_document_id
        counter = 1
        
        while True:
            doc_paths = self.config.base_data_dir / "documents" / document_id
            if not doc_paths.exists():
                break
            document_id = f"{base_document_id}_{counter}"
            counter += 1
            
            # Safety check to prevent infinite loop
            if counter > 1000:
                document_id = f"doc_{int(time.time())}"
                break
        
        return document_id

    def _create_temp_file(self, file_content: bytes, document_id: str, original_filename: str) -> Path:
        """
        Create temporary file for processing.
        
        Args:
            file_content: File content bytes
            document_id: Document identifier
            original_filename: Original filename for extension
            
        Returns:
            Path to temporary file
        """
        temp_dir = self.config.base_data_dir / "temp"
        ensure_directory(temp_dir)
        
        # Preserve original file extension
        original_ext = Path(original_filename).suffix
        temp_filename = f"{document_id}_temp{original_ext}"
        temp_path = temp_dir / temp_filename
        
        # Write file content
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        return temp_path

    def _cleanup_temp_file(self, temp_path: Path) -> None:
        """Safely cleanup temporary file."""
        try:
            if temp_path.exists():
                temp_path.unlink()
                print(f"Cleaned up temporary file: {temp_path}")
        except Exception as e:
            print(f"Warning: Could not cleanup temp file {temp_path}: {e}")

    def _log_pipeline_event(self, document_id: str, event: str, data: Dict[str, Any] = None) -> None:
        """
        Log pipeline-level events.
        
        Args:
            document_id: Document identifier
            event: Event name
            data: Event data
        """
        try:
            # Create pipeline log path
            processed_dir = self.config.base_data_dir / "processed" / document_id
            ensure_directory(processed_dir)
            pipeline_log = processed_dir / "pipeline_log.json"
            
            log_entry = create_log_entry(
                operation=f"pipeline_{event}",
                success=True,
                data=data or {}
            )
            log_entry["document_id"] = document_id
            log_entry["processor"] = "DocumentProcessingOrchestrator"
            
            append_to_log(pipeline_log, log_entry)
            
        except Exception as e:
            print(f"Warning: Could not log pipeline event: {e}")

    def _execute_step_with_retry(
        self, 
        step_name: str, 
        step_func, 
        *args,
        **kwargs
    ) -> StepResult:
        """
        Execute a pipeline step with retry logic.
        
        Args:
            step_name: Name of the step
            step_func: Function to execute
            *args: Positional arguments to pass to step function
            **kwargs: Keyword arguments (including 'required' flag)
            
        Returns:
            StepResult with execution details
        """
        # Extract required flag from kwargs, default to True
        required = kwargs.pop('required', True)
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                print(f"{'[RETRY]' if attempt > 0 else ''} Executing step: {step_name}")
                
                # Execute step function
                result = step_func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Determine success based on result type
                if isinstance(result, tuple):
                    success = bool(result[0]) if len(result) > 0 else True
                    data = result[1] if len(result) > 1 else None
                elif isinstance(result, dict):
                    success = result.get('success', True)
                    data = result
                else:
                    success = result is not None
                    data = result
                
                return StepResult(
                    step_name=step_name,
                    success=success,
                    data=data,
                    execution_time=execution_time,
                    retry_count=attempt
                )
                
            except Exception as e:
                last_error = str(e)
                print(f"Step {step_name} failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.max_retries:
                    print(f"Retrying step {step_name} in 1 second...")
                    time.sleep(1)
                else:
                    print(f"Step {step_name} failed after {self.config.max_retries + 1} attempts")
        
        execution_time = time.time() - start_time
        
        return StepResult(
            step_name=step_name,
            success=False,
            error=last_error,
            execution_time=execution_time,
            retry_count=self.config.max_retries
        )

    # =============================================================================
    # INDIVIDUAL STEP IMPLEMENTATIONS
    # =============================================================================

    def _step_pdf_processing(self, temp_pdf_path: Path, document_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Step 1: Process PDF to markdown using LlamaParseProcessor.
        
        Args:
            temp_pdf_path: Path to temporary PDF file
            document_id: Document identifier
            
        Returns:
            Tuple of (success, result_data)
        """
        if not self.llamaparse_processor:
            raise RuntimeError("LlamaParseProcessor not initialized")
        
        print(f"Step 1: Processing PDF to markdown...")
        
        # Process and save content
        content, metadata = self.llamaparse_processor.process_and_save(temp_pdf_path, document_id)
        
        if not metadata["success"]:
            return False, {
                "error": metadata.get("error", "PDF processing failed"),
                "metadata": metadata
            }
        
        # Get document paths for response
        paths = self.llamaparse_processor.get_document_paths(document_id)
        
        result_data = {
            "content": content,
            "metadata": metadata,
            "paths": {k: str(v) for k, v in paths.items()},
            "processing_time": metadata["processing_time"],
            "token_count": metadata["token_count"],
            "pages_processed": metadata["pages_processed"],
            "content_length": metadata["content_length"]
        }
        
        print(f"[OK] PDF processed successfully:")
        print(f"   - Processing time: {metadata['processing_time']:.2f}s")
        print(f"   - Token count: {metadata['token_count']:,}")
        print(f"   - Content length: {metadata['content_length']:,} chars")
        print(f"   - Pages processed: {metadata['pages_processed']}")
        
        return True, result_data

    def _step_structure_extraction(
        self, 
        temp_pdf_path: Path, 
        markdown_path: Path, 
        document_id: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Step 2: Extract document structure and create initial chunks.
        
        Args:
            temp_pdf_path: Path to temporary PDF file
            markdown_path: Path to processed markdown
            document_id: Document identifier
            
        Returns:
            Tuple of (success, result_data)
        """
        if not self.indexing_pipeline:
            raise RuntimeError("DocumentIndexingPipeline not initialized")
        
        print(f"Step 2: Extracting document structure and creating chunks...")
        
        # Run the complete indexing and chunking pipeline
        doc_structure = self.indexing_pipeline.run_index_and_chunking_pipeline(
            str(temp_pdf_path),
            str(markdown_path),
            document_id
        )
        
        if not doc_structure:
            return False, {"error": "Failed to extract document structure"}
        
        # Get paths and count created chunks
        paths = self.indexing_pipeline.get_document_paths(document_id)
        chunks_created = list(paths["chunks_dir"].glob("*.md")) if paths["chunks_dir"].exists() else []
        
        result_data = {
            "structure": doc_structure,
            "paths": {k: str(v) for k, v in paths.items()},
            "sections_count": len(doc_structure.sections),
            "chunks_created": len(chunks_created),
            "chunks_files": [c.name for c in chunks_created]
        }
        
        print(f"[OK] Document structure extracted:")
        print(f"   - Total sections: {len(doc_structure.sections)}")
        
        # Log section details
        for i, section in enumerate(doc_structure.sections, 1):
            sub_count = len(section.sub_sections) if section.sub_sections else 0
            print(f"   {i}. {section.title} ({sub_count} subsections)")
        
        print(f"   - Initial chunks created: {len(chunks_created)} files")
        
        return True, result_data

    def _step_rechunking(
        self, 
        document_id: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Step 3: Advanced re-chunking of initial chunks.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Tuple of (success, result_data)
        """
        if not self.indexing_pipeline:
            raise RuntimeError("DocumentIndexingPipeline not initialized")
        
        print(f"Step 3: Advanced re-chunking...")
        
        try:
            final_chunks, chunk_metadata = self.indexing_pipeline.rechunk_directory(
                document_id=document_id,
                target_chunk_size=self.config.target_chunk_size,
                tolerance=self.config.chunk_tolerance,
                save=True
            )
            
            if not final_chunks:
                # This is not necessarily an error - some documents might not need re-chunking
                print("No final chunks generated - using initial chunks")
                return True, {
                    "final_chunks": [],
                    "chunk_metadata": [],
                    "final_chunks_count": 0,
                    "message": "No re-chunking needed - initial chunks are sufficient"
                }
            
            # Calculate statistics
            chunk_stats = calculate_chunk_statistics(final_chunks)
            
            result_data = {
                "final_chunks": final_chunks,
                "chunk_metadata": chunk_metadata,
                "final_chunks_count": len(final_chunks),
                "chunk_statistics": chunk_stats
            }
            
            print(f"[OK] Re-chunking completed:")
            print(f"   - Final chunks: {len(final_chunks)}")
            print(f"   - Average chunk length: {chunk_stats['avg_length']:.0f} chars")
            print(f"   - Length range: {chunk_stats['min_length']}-{chunk_stats['max_length']} chars")
            
            return True, result_data
            
        except Exception as e:
            print(f"Re-chunking failed: {e}")
            # Since this is optional, we can continue without re-chunking
            return False, {"error": str(e), "message": "Continuing with initial chunks"}

    def _step_embeddings_generation(self, document_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Step 4: Generate embeddings (placeholder for future VectorStore integration).
        
        Args:
            document_id: Document identifier
            
        Returns:
            Tuple of (success, result_data)
        """
        print(f"Step 4: Generating embeddings...")
        
        # Placeholder implementation - will be replaced when VectorStore is provided
        print("Embeddings generation not yet implemented (waiting for VectorStore class)")
        
        return True, {
            "embeddings_count": 0,
            "source_type": "none",
            "message": "Embeddings generation not yet implemented"
        }

    def _step_search_test(self, document_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Step 5: Test search functionality (placeholder for future VectorStore integration).
        
        Args:
            document_id: Document identifier
            
        Returns:
            Tuple of (success, result_data)
        """
        print(f"Step 5: Testing search functionality...")
        
        # Placeholder implementation - will be replaced when VectorStore is provided
        print("Search test not yet implemented (waiting for VectorStore class)")
        
        return True, {
            "query": "test_query",
            "results_found": 0,
            "working": False,
            "message": "Search test not yet implemented"
        }

    # =============================================================================
    # MAIN PROCESSING METHOD
    # =============================================================================

    async def process_document(
        self, 
        file_content: bytes,
        filename: str,
        custom_document_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a complete document through the pipeline.
        
        This is the main entry point that replaces the original upload endpoint logic.
        
        Args:
            file_content: Raw file content bytes
            filename: Original filename
            custom_document_id: Custom document ID (optional)
            
        Returns:
            ProcessingResult with complete pipeline results
        """
        # Validate file type
        if not filename.lower().endswith(".pdf"):
            return ProcessingResult(
                success=False,
                document_id="",
                state=PipelineState(
                    document_id="",
                    original_filename=filename,
                    status="failed",
                    error_summary="Only PDF files are allowed"
                )
            )
        
        # Generate document ID
        document_id = self._generate_document_id(filename, custom_document_id)
        
        print(f"Processing document: {document_id} (original: {filename})")
        
        # Initialize pipeline state
        self.current_state = PipelineState(
            document_id=document_id,
            original_filename=filename,
            status="running",
            start_time=datetime.now()
        )
        
        # Initialize components
        init_success, init_error = self._initialize_components()
        if not init_success:
            self.current_state.status = "failed"
            self.current_state.error_summary = init_error
            self.current_state.end_time = datetime.now()
            
            return ProcessingResult(
                success=False,
                document_id=document_id,
                state=self.current_state
            )
        
        # Create temporary file
        temp_pdf_path = self._create_temp_file(file_content, document_id, filename)
        
        try:
            # Log pipeline start
            self._log_pipeline_event(document_id, "start", {
                "original_filename": filename,
                "document_id": document_id,
                "config": {
                    "enable_rechunking": self.config.enable_rechunking,
                    "enable_embeddings": self.config.enable_embeddings,
                    "target_chunk_size": self.config.target_chunk_size
                }
            })
            
            # Execute pipeline steps
            step_results = {}
            
            # Step 1: PDF Processing (Required)
            if self.config.enable_pdf_processing:
                self.current_state.current_step = "pdf_processing"
                result = self._execute_step_with_retry(
                    "pdf_processing",
                    self._step_pdf_processing,
                    temp_pdf_path,
                    document_id,
                    required=True
                )
                step_results["pdf_processing"] = result
                
                if not result.success:
                    raise Exception(f"PDF processing failed: {result.error}")
                
                self.current_state.completed_steps.append("pdf_processing")
                
                # Get processed markdown path for next steps
                processed_content_path = (
                    self.config.base_data_dir / "processed" / document_id / "processed_content.md"
                )
            
            # Step 2: Structure Extraction and Initial Chunking (Required)
            if self.config.enable_structure_extraction and self.config.enable_initial_chunking:
                self.current_state.current_step = "structure_extraction"
                result = self._execute_step_with_retry(
                    "structure_extraction",
                    self._step_structure_extraction,
                    temp_pdf_path,
                    processed_content_path,
                    document_id,
                    required=True
                )
                step_results["structure_extraction"] = result
                
                if not result.success:
                    raise Exception(f"Structure extraction failed: {result.error}")
                
                self.current_state.completed_steps.append("structure_extraction")
            
            # Step 3: Re-chunking (Optional)
            if self.config.enable_rechunking:
                self.current_state.current_step = "rechunking"
                result = self._execute_step_with_retry(
                    "rechunking", 
                    self._step_rechunking,
                    document_id,
                    required=False
                )
                step_results["rechunking"] = result
                
                if result.success:
                    self.current_state.completed_steps.append("rechunking")
                elif not self.config.continue_on_optional_failure:
                    raise Exception(f"Re-chunking failed: {result.error}")
                else:
                    self.current_state.failed_steps.append("rechunking")
                    print(f"Warning: Re-chunking failed but continuing: {result.error}")
            
            # Step 4: Embeddings (Optional - Placeholder)
            if self.config.enable_embeddings:
                self.current_state.current_step = "embeddings"
                result = self._execute_step_with_retry(
                    "embeddings",
                    self._step_embeddings_generation,
                    document_id,
                    required=False
                )
                step_results["embeddings"] = result
                
                if result.success:
                    self.current_state.completed_steps.append("embeddings")
                elif not self.config.continue_on_optional_failure:
                    raise Exception(f"Embeddings generation failed: {result.error}")
                else:
                    self.current_state.failed_steps.append("embeddings")
            
            # Step 5: Search Test (Optional - Placeholder)
            if self.config.enable_search_test:
                self.current_state.current_step = "search_test"
                result = self._execute_step_with_retry(
                    "search_test",
                    self._step_search_test,
                    document_id,
                    required=False
                )
                step_results["search_test"] = result
                
                if result.success:
                    self.current_state.completed_steps.append("search_test")
                elif not self.config.continue_on_optional_failure:
                    raise Exception(f"Search test failed: {result.error}")
                else:
                    self.current_state.failed_steps.append("search_test")
            
            # Pipeline completed successfully
            self.current_state.status = "completed"
            self.current_state.success = True
            self.current_state.step_results = step_results
            
        except Exception as e:
            # Pipeline failed
            self.current_state.status = "failed"
            self.current_state.error_summary = str(e)
            self.current_state.success = False
            
            print(f"Pipeline failed: {e}")
            traceback.print_exc()
            
            # Cleanup on error if requested
            if self.config.cleanup_on_error:
                self._cleanup_on_error(document_id)
        
        finally:
            # Always cleanup temp file and finalize state
            self._cleanup_temp_file(temp_pdf_path)
            self.current_state.current_step = None
            self.current_state.end_time = datetime.now()
            
            # Calculate total execution time
            if self.current_state.start_time and self.current_state.end_time:
                delta = self.current_state.end_time - self.current_state.start_time
                self.current_state.total_execution_time = delta.total_seconds()
            
            # Log pipeline completion
            self._log_pipeline_event(document_id, "complete", {
                "status": self.current_state.status,
                "success": self.current_state.success,
                "total_execution_time": self.current_state.total_execution_time,
                "completed_steps": self.current_state.completed_steps,
                "failed_steps": self.current_state.failed_steps,
                "error_summary": self.current_state.error_summary
            })
        
        # Create response data compatible with original endpoint
        response_data = self._create_response_data(self.current_state, step_results)
        
        return ProcessingResult(
            success=self.current_state.success,
            document_id=document_id,
            state=self.current_state,
            response_data=response_data
        )

    def _cleanup_on_error(self, document_id: str) -> None:
        """Clean up partial processing artifacts on error."""
        try:
            print(f"Cleaning up partial processing artifacts for document: {document_id}")
            
            # Clean up processed directory
            processed_dir = self.config.base_data_dir / "processed" / document_id
            if processed_dir.exists():
                shutil.rmtree(processed_dir)
                print(f"Cleaned up processed directory: {processed_dir}")
            
            # Optionally clean up documents directory (keep original file by default)
            # documents_dir = self.config.base_data_dir / "documents" / document_id  
            # if documents_dir.exists():
            #     shutil.rmtree(documents_dir)
            
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

    def _create_response_data(
        self, 
        state: PipelineState, 
        step_results: Dict[str, StepResult]
    ) -> Dict[str, Any]:
        """
        Create response data compatible with original endpoint format.
        
        Args:
            state: Pipeline state
            step_results: Results from each step
            
        Returns:
            Response data dict compatible with original endpoint
        """
        # Base response structure
        response_data = {
            "status": "success" if state.success else "error",
            "message": f"Document '{state.original_filename}' {'processed successfully' if state.success else 'processing failed'}",
            "document_id": state.document_id,
            "processing_summary": {},
            "document_structure": {},
            "file_locations": {},
            "data_structure": {}
        }
        
        # Add error information if failed
        if not state.success:
            response_data["error"] = state.error_summary
            return response_data
        
        # Processing summary from step results
        pdf_result = step_results.get("pdf_processing")
        if pdf_result and pdf_result.success:
            pdf_data = pdf_result.data
            response_data["processing_summary"]["pdf_processing"] = {
                "success": True,
                "processing_time": pdf_data.get("processing_time", 0),
                "token_count": pdf_data.get("token_count", 0),
                "pages_processed": pdf_data.get("pages_processed", 0)
            }
        
        structure_result = step_results.get("structure_extraction")
        if structure_result and structure_result.success:
            structure_data = structure_result.data
            response_data["processing_summary"]["structure_extraction"] = {
                "success": True,
                "sections_count": structure_data.get("sections_count", 0),
                "initial_chunks": structure_data.get("chunks_created", 0)
            }
            
            # Document structure details
            structure = structure_data.get("structure")
            if structure:
                response_data["document_structure"] = {
                    "total_sections": len(structure.sections),
                    "sections": [
                        {
                            "title": section.title,
                            "subsections_count": len(section.sub_sections) if section.sub_sections else 0
                        }
                        for section in structure.sections
                    ]
                }
        
        rechunk_result = step_results.get("rechunking")
        if rechunk_result:
            rechunk_data = rechunk_result.data or {}
            response_data["processing_summary"]["rechunking"] = {
                "success": rechunk_result.success,
                "final_chunks_count": rechunk_data.get("final_chunks_count", 0)
            }
        
        embeddings_result = step_results.get("embeddings")
        if embeddings_result:
            embeddings_data = embeddings_result.data or {}
            response_data["processing_summary"]["embeddings"] = {
                "success": embeddings_result.success,
                "vectors_count": embeddings_data.get("embeddings_count", 0),
                "source_type": embeddings_data.get("source_type", "none")
            }
        
        # File locations
        if self.llamaparse_processor:
            paths = self.llamaparse_processor.get_document_paths(state.document_id)
            response_data["file_locations"]["original_pdf"] = str(paths.get("original_file", ""))
            response_data["file_locations"]["processed_markdown"] = str(paths.get("processed_content", ""))
        
        if self.indexing_pipeline:
            paths = self.indexing_pipeline.get_document_paths(state.document_id)
            response_data["file_locations"]["chunks_directory"] = str(paths.get("chunks_dir", ""))
            response_data["file_locations"]["final_chunks_directory"] = str(paths.get("final_chunks_dir", ""))
            response_data["file_locations"]["embeddings_directory"] = str(paths.get("chunks_dir", "").parent / "embeddings")
        
        # Data structure status
        response_data["data_structure"] = {
            "documents_path": f"data/documents/{state.document_id}/",
            "processed_path": f"data/processed/{state.document_id}/", 
            "structure_exists": structure_result.success if structure_result else False,
            "embeddings_ready": embeddings_result.success if embeddings_result else False
        }
        
        # Add search test results if available
        search_result = step_results.get("search_test")
        if search_result and search_result.success:
            search_data = search_result.data or {}
            response_data["search_test"] = {
                "query": search_data.get("query", ""),
                "results_found": search_data.get("results_found", 0),
                "working": search_data.get("working", False)
            }
        
        return response_data

    # =============================================================================
    # UTILITY AND STATUS METHODS
    # =============================================================================

    def get_pipeline_status(self, document_id: str) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Pipeline status information
        """
        status = {
            "document_id": document_id,
            "processors": {}
        }
        
        # Get status from each processor
        if self.llamaparse_processor:
            status["processors"]["llamaparse"] = self.llamaparse_processor.get_processing_status(document_id)
        
        if self.indexing_pipeline:
            status["processors"]["indexing_pipeline"] = self.indexing_pipeline.get_document_status(document_id)
        
        # Current pipeline state
        if self.current_state and self.current_state.document_id == document_id:
            status["current_pipeline_state"] = {
                "status": self.current_state.status,
                "current_step": self.current_state.current_step,
                "completed_steps": self.current_state.completed_steps,
                "failed_steps": self.current_state.failed_steps,
                "start_time": self.current_state.start_time.isoformat() if self.current_state.start_time else None,
                "success": self.current_state.success
            }
        
        return status

    def list_processed_documents(self) -> List[str]:
        """List all processed document IDs."""
        try:
            processed_dir = self.config.base_data_dir / "processed"
            if not processed_dir.exists():
                return []
            
            return [
                d.name for d in processed_dir.iterdir() 
                if d.is_dir() and validate_document_id(d.name)
            ]
            
        except Exception as e:
            print(f"Error listing processed documents: {e}")
            return []


# =============================================================================
# CONVENIENCE FUNCTIONS FOR INTEGRATION
# =============================================================================

def create_orchestrator_with_defaults(
    base_data_dir: Path = Path("data"),
    **config_overrides
) -> DocumentProcessingOrchestrator:
    """
    Create orchestrator with default configuration and optional overrides.
    
    Args:
        base_data_dir: Base data directory
        **config_overrides: Configuration parameters to override
        
    Returns:
        Configured DocumentProcessingOrchestrator
    """
    config = PipelineConfig(base_data_dir=base_data_dir)
    
    # Apply any overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return DocumentProcessingOrchestrator(config)


async def process_uploaded_file(
    file_content: bytes,
    filename: str,
    base_data_dir: Path = Path("data"),
    custom_document_id: Optional[str] = None,
    **config_overrides
) -> ProcessingResult:
    """
    Convenience function to process an uploaded file with default settings.
    This directly replaces the original upload endpoint logic.
    
    Args:
        file_content: Raw file content bytes
        filename: Original filename
        base_data_dir: Base data directory
        custom_document_id: Custom document ID
        **config_overrides: Configuration overrides
        
    Returns:
        ProcessingResult
    """
    orchestrator = create_orchestrator_with_defaults(
        base_data_dir=base_data_dir,
        **config_overrides
    )
    
    return await orchestrator.process_document(
        file_content=file_content,
        filename=filename,
        custom_document_id=custom_document_id
    )