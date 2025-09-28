"""
document_indexing_pipeline.py - Enhanced document structure extraction and chunking pipeline.

This module handles document structure extraction, initial chunking, and advanced re-chunking
with improved error handling, logging, and integration with the orchestrator system.
"""

import os
import re
import json
import time
from pathlib import Path
from textwrap import dedent
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

try:
    from dotenv import load_dotenv
    from openai import OpenAI
    from pydantic import BaseModel
    from PyPDF2 import PdfReader, PdfWriter
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

# Import our utils
from services.utils import (
    normalize_chunk_markdown,
    sanitize_filename,
    sanitize_stem,
    validate_document_id,
    ensure_directory,
    safe_write_json,
    safe_read_json,
    get_file_size_safely,
    create_log_entry,
    append_to_log,
    split_sections_by_titles,
    split_chunks_with_metadata,
    calculate_chunk_statistics,
    extract_markdown_title,
    validate_chunk_parameters
)


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class IndexRetrieval(BaseModel):
    """Schema for OpenAI index extraction response."""
    
    class IndexSubIndex(BaseModel):
        index_element: str
        sub_index_elements: List[str]

    index_elements: List[str]
    sub_index_elements: List[IndexSubIndex]


class SectionNode(BaseModel):
    """Schema for hierarchical document sections."""
    title: str
    sub_sections: Optional[List["SectionNode"]] = None

# Rebuild model to handle forward references
SectionNode.model_rebuild()


class DocumentStructure(BaseModel):
    """Schema for complete document structure."""
    sections: List[SectionNode]


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class DocumentIndexingPipeline:
    """
    End-to-end pipeline for document structure extraction and chunking.
    
    Features:
    1. Preview first pages from PDF/DOCX
    2. Extract TOC/Index using OpenAI (fallback to H1 in markdown)
    3. Build hierarchical structure
    4. Split markdown file into subsection files
    5. (Optional) Re-chunk files using advanced chunking logic
    
    Uses structure: data/processed/{document_id}/chunks/ and data/processed/{document_id}/final_chunks/
    """

    def __init__(
        self,
        model: str = "gpt-4.1-2025-04-14",
        base_data_dir: Path = Path("data"),
        max_preview_pages: int = 10,
        openai_client: Optional[OpenAI] = None,
        auto_env: bool = True,
    ):
        """
        Initialize the pipeline.
        
        Args:
            model: OpenAI model to use for structure extraction
            base_data_dir: Base directory for data storage
            max_preview_pages: Maximum pages to extract for preview
            openai_client: Pre-initialized OpenAI client (optional)
            auto_env: Whether to auto-load environment variables
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Required dependencies not available. Install: "
                "pip install openai python-dotenv PyPDF2 pydantic"
            )
        
        self.model = model
        self.base_data_dir = Path(base_data_dir)
        self.processed_dir = self.base_data_dir / "processed"
        self.max_preview_pages = max_preview_pages
        
        # Initialize OpenAI client
        if openai_client:
            self.client = openai_client
        elif auto_env:
            self.client = self._init_openai_client()
        else:
            self.client = None
        
        # Ensure base directories exist
        ensure_directory(self.processed_dir)
        
        print(f"DocumentIndexingPipeline initialized with base_data_dir: {self.base_data_dir}")

    def _init_openai_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI client with environment variables."""
        try:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("OPENAI_API_KEY not found in environment variables")
                return None
            print("OpenAI client initialized successfully")
            return OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            return None

    def get_document_paths(self, document_id: str) -> Dict[str, Path]:
        """
        Get all relevant paths for a document.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            Dictionary with all document paths
        """
        if not validate_document_id(document_id):
            raise ValueError(f"Invalid document_id: {document_id}")
        
        processed_doc_dir = self.processed_dir / document_id
        
        return {
            "processed_dir": processed_doc_dir,
            "chunks_dir": processed_doc_dir / "chunks",
            "final_chunks_dir": processed_doc_dir / "final_chunks",
            "temp_dir": processed_doc_dir / "temp",
            "structure_file": processed_doc_dir / "document_structure.json",
            "indexing_log": processed_doc_dir / "indexing_log.json",
            "preview_pdf": processed_doc_dir / "temp" / "preview.pdf"
        }

    def _log_operation(
        self, 
        document_id: str, 
        operation: str, 
        success: bool,
        duration: float = 0.0,
        data: Dict[str, Any] = None,
        error: str = None
    ) -> None:
        """
        Log an operation using standardized format.
        
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
            log_entry["processor"] = "DocumentIndexingPipeline"
            
            append_to_log(paths["indexing_log"], log_entry)
            
        except Exception as e:
            print(f"Error logging operation: {e}")

    def _save_document_structure(self, document_id: str, structure: DocumentStructure) -> bool:
        """
        Save document structure to JSON file.
        
        Args:
            document_id: Document identifier
            structure: Document structure to save
            
        Returns:
            True if successful
        """
        try:
            paths = self.get_document_paths(document_id)
            success = safe_write_json(paths["structure_file"], structure.model_dump())
            
            if success:
                print(f"Document structure saved: {paths['structure_file']}")
            
            return success
            
        except Exception as e:
            print(f"Error saving document structure: {e}")
            return False

    def load_document_structure(self, document_id: str) -> Optional[DocumentStructure]:
        """
        Load document structure from JSON file.
        
        Args:
            document_id: Document identifier
            
        Returns:
            DocumentStructure if exists, None otherwise
        """
        try:
            paths = self.get_document_paths(document_id)
            
            if not paths["structure_file"].exists():
                return None
                
            data = safe_read_json(paths["structure_file"])
            if not data:
                return None
                
            return DocumentStructure.model_validate(data)
            
        except Exception as e:
            print(f"Error loading document structure: {e}")
            return None

    # =============================================================================
    # FILE PROCESSING UTILITIES
    # =============================================================================

    def convert_docx_to_pdf(self, docx_path: str, document_id: str) -> Optional[str]:
        """
        Convert DOCX to PDF and save in document temp directory.
        
        Args:
            docx_path: Path to DOCX file
            document_id: Document identifier
            
        Returns:
            Path to converted PDF or None if failed
        """
        start_time = time.time()
        
        try:
            # Note: This requires python-docx2pdf which wasn't in original dependencies
            # For now, we'll return None and log that conversion is not available
            duration = time.time() - start_time
            
            self._log_operation(
                document_id=document_id,
                operation="convert_docx_to_pdf", 
                success=False,
                duration=duration,
                error="DOCX to PDF conversion not implemented (requires python-docx2pdf)"
            )
            
            print("DOCX to PDF conversion not available - requires additional dependency")
            return None
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"DOCX to PDF conversion failed: {e}"
            
            self._log_operation(
                document_id=document_id,
                operation="convert_docx_to_pdf",
                success=False,
                duration=duration,
                error=error_msg
            )
            
            print(f"Error in DOCX conversion: {e}")
            return None

    def extract_first_pages_pdf(self, input_path: str, document_id: str) -> Optional[str]:
        """
        Extract first pages of PDF for preview.
        Saves to: data/processed/{document_id}/temp/preview.pdf
        
        Args:
            input_path: Path to input PDF
            document_id: Document identifier
            
        Returns:
            Path to preview PDF or None if failed
        """
        start_time = time.time()
        ext = Path(input_path).suffix.lower()

        # Handle DOCX files
        if ext == ".docx":
            converted_path = self.convert_docx_to_pdf(input_path, document_id)
            if not converted_path:
                return None
            input_path = converted_path
            ext = ".pdf"

        # Process PDF files
        if ext == ".pdf":
            try:
                paths = self.get_document_paths(document_id)
                ensure_directory(paths["temp_dir"])
                
                output_path = paths["preview_pdf"]
                
                reader = PdfReader(input_path)
                writer = PdfWriter()
                
                pages_extracted = min(self.max_preview_pages, len(reader.pages))
                for i in range(pages_extracted):
                    writer.add_page(reader.pages[i])

                with open(output_path, "wb") as f:
                    writer.write(f)
                
                duration = time.time() - start_time
                
                self._log_operation(
                    document_id=document_id,
                    operation="extract_pdf_preview",
                    success=True,
                    duration=duration,
                    data={
                        "input_path": input_path,
                        "output_path": str(output_path),
                        "pages_extracted": pages_extracted,
                        "total_pages": len(reader.pages)
                    }
                )
                
                print(f"PDF preview saved to: {output_path}")
                return str(output_path)
                
            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"PDF preview extraction failed: {e}"
                
                self._log_operation(
                    document_id=document_id,
                    operation="extract_pdf_preview",
                    success=False,
                    duration=duration,
                    error=error_msg
                )
                
                print(f"Error extracting PDF preview: {e}")
                return None

        print("Unsupported file type for preview extraction")
        return None

    # =============================================================================
    # OPENAI INTEGRATION
    # =============================================================================

    def extract_index_with_openai(self, pdf_path: str, document_id: str) -> Optional[IndexRetrieval]:
        """
        Extract document index/TOC using OpenAI.
        
        Args:
            pdf_path: Path to PDF file (or preview)
            document_id: Document identifier
            
        Returns:
            IndexRetrieval object or None if failed
        """
        if not self.client:
            print("OpenAI client not available for index extraction")
            return None

        start_time = time.time()
        
        prompt = dedent("""
            Your task is to locate and extract any section of the document that serves as a Table of Contents or Index.
            This includes any part that lists sections, chapters, clauses, headings, or navigational elements, even if it's not explicitly labeled "Table of Contents" or "Index".
            Carefully analyze the structure and layout of the text to infer whether such content exists.
            Guidelines:
            - Differentiate between main sections and subsections. 
            - ONLY include numeration of indexes if the document explicitly uses it.
            - Do not hallucinate entries — only extract content that clearly exists in the document.
            - Be strict: avoid interpreting summaries, descriptions, or general text as a TOC unless it's structured accordingly.
        """)

        try:
            with open(pdf_path, "rb") as file:
                uploaded_file = self.client.files.create(file=file, purpose="user_data")

            print("File uploaded to OpenAI for index extraction")

            completion = self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": uploaded_file.id},
                            {"type": "input_text", "text": prompt}
                        ]
                    }
                ],
                text_format=IndexRetrieval,
            )

            result = completion.output_parsed
            duration = time.time() - start_time
            
            self._log_operation(
                document_id=document_id,
                operation="extract_index_openai",
                success=True,
                duration=duration,
                data={
                    "pdf_path": pdf_path,
                    "model": self.model,
                    "index_elements_count": len(result.index_elements) if result else 0,
                    "sub_index_elements_count": len(result.sub_index_elements) if result else 0
                }
            )
            
            print(f"OpenAI index extraction completed successfully")
            return result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"OpenAI index extraction failed: {e}"
            
            self._log_operation(
                document_id=document_id,
                operation="extract_index_openai",
                success=False,
                duration=duration,
                error=error_msg
            )
            
            print(f"Error during OpenAI index extraction: {e}")
            return None

    def build_structure_from_titles_with_openai(
        self, 
        titles: List[str], 
        document_id: str
    ) -> Optional[DocumentStructure]:
        """
        Fallback: infer hierarchical structure from H1 titles using OpenAI.
        
        Args:
            titles: List of H1 titles from markdown
            document_id: Document identifier
            
        Returns:
            DocumentStructure or None if failed
        """
        if not self.client:
            print("OpenAI client not available for structure building")
            return None

        start_time = time.time()
        
        prompt = dedent(f"""
            The following is a list of markdown H1 section titles extracted from a document:

            {json.dumps(titles, indent=2)}

            Your task is to analyze these titles and reconstruct a plausible hierarchical index structure
            (like a Table of Contents). Group related entries under common parents if it makes sense.

            Guidelines:
            - Do not hallucinate section names. Use only what's in the list.
            - Preserve the order of appearance.
            - Prefer grouping based on numeric prefixes, topic similarity, or known legal patterns.

            Return only structured data matching this schema:
            {{
                "index_elements": [...],
                "sub_index_elements": [
                    {{
                        "index_element": "...",
                        "sub_index_elements": ["...", "..."]
                    }}
                ]
            }}
        """)

        try:
            completion = self.client.responses.parse(
                model=self.model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                text_format=IndexRetrieval
            )

            parsed_result: IndexRetrieval = completion.output_parsed
            structure = self.build_document_structure(
                parsed_result.index_elements, 
                parsed_result.sub_index_elements
            )
            
            duration = time.time() - start_time
            
            self._log_operation(
                document_id=document_id,
                operation="build_structure_fallback",
                success=True,
                duration=duration,
                data={
                    "h1_titles_count": len(titles),
                    "titles": titles,
                    "index_elements_count": len(parsed_result.index_elements)
                }
            )
            
            print("Fallback structure building completed successfully")
            return structure

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Fallback structure building failed: {e}"
            
            self._log_operation(
                document_id=document_id,
                operation="build_structure_fallback",
                success=False,
                duration=duration,
                error=error_msg
            )
            
            print(f"Error in fallback structure building: {e}")
            return None

    # =============================================================================
    # MARKDOWN PROCESSING
    # =============================================================================

    @staticmethod
    def extract_h1_titles_from_markdown(markdown_path: str) -> List[str]:
        """
        Extract H1 titles from markdown file.
        
        Args:
            markdown_path: Path to markdown file
            
        Returns:
            List of H1 titles
        """
        try:
            with open(markdown_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            h1_titles = []
            for line in lines:
                if line.startswith("# "):
                    title = line.strip("# ").strip()
                    if title:
                        h1_titles.append(title)
            
            return h1_titles
            
        except Exception as e:
            print(f"Error extracting H1 titles: {e}")
            return []

    @staticmethod
    def build_document_structure(
        index_elements: List[str],
        sub_index_elements: List[IndexRetrieval.IndexSubIndex]
    ) -> DocumentStructure:
        """
        Build DocumentStructure from index elements.
        
        Args:
            index_elements: Main section titles
            sub_index_elements: Subsection mappings
            
        Returns:
            DocumentStructure object
        """
        section_nodes: List[SectionNode] = []
        
        # Create mapping of main sections to subsections
        sub_index_map = {}
        for sub in sub_index_elements:
            # Remove numeric prefixes for matching
            clean_title = re.sub(r"^\d+(\.\d+)*\s*", "", sub.index_element).strip()
            sub_index_map[clean_title] = sub.sub_index_elements

        # Build section nodes
        for raw_title in index_elements:
            # Try to find subsections for this title
            sub_titles = sub_index_map.get(raw_title, [])
            
            node = SectionNode(
                title=raw_title,
                sub_sections=[SectionNode(title=sub) for sub in sub_titles] if sub_titles else None
            )
            section_nodes.append(node)

        return DocumentStructure(sections=section_nodes)

    @staticmethod
    def build_title_map(structure: DocumentStructure) -> Dict[str, Dict]:
        """
        Build a mapping of titles to their hierarchy information.
        
        Args:
            structure: Document structure
            
        Returns:
            Dictionary mapping titles to hierarchy info
        """
        title_map: Dict[str, Dict] = {}
        
        for section in structure.sections:
            title_map[section.title] = {"parent": None, "is_subsection": False}
            
            if section.sub_sections:
                for sub in section.sub_sections:
                    title_map[sub.title] = {
                        "parent": section.title, 
                        "is_subsection": True
                    }
        
        return title_map

    def split_markdown_to_subsection_files(
        self,
        markdown_path: str,
        structure: DocumentStructure,
        document_id: str,
        output_dir: Optional[str] = None
    ) -> List[Path]:
        """
        Split markdown file into subsection files based on structure.
        Saves to: data/processed/{document_id}/chunks/
        
        Args:
            markdown_path: Path to full markdown file
            structure: Document structure for splitting
            document_id: Document identifier
            output_dir: Custom output directory (optional)
            
        Returns:
            List of created chunk file paths
        """
        start_time = time.time()
        
        try:
            title_map = self.build_title_map(structure)

            with open(markdown_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            # Use structure-based output directory
            if output_dir:
                output_path = Path(output_dir)
            else:
                paths = self.get_document_paths(document_id)
                output_path = paths["chunks_dir"]
                
            ensure_directory(output_path)

            current_buffer: List[str] = []
            current_title: Optional[str] = None
            current_parent: Optional[str] = None
            chunks: List[Path] = []
            intro_buffer: List[str] = []  # Buffer for intro content
            found_first_section = False  # Track if we've found our first indexed section

            def flush_buffer(title: Optional[str], parent: Optional[str], buffer: List[str]):
                if not buffer or not title:
                    return
                    
                # Normalize and clean content
                content = normalize_chunk_markdown("".join(buffer))
                
                # Create filename with proper structure
                section_part = f"{sanitize_filename(parent)}_" if parent else ""
                filename = f"{section_part}{sanitize_filename(title)}.md"
                filepath = output_path / filename
                
                # Write content
                with open(filepath, "w", encoding="utf-8") as out:
                    out.write(content)
                
                chunks.append(filepath)
                print(f"Saved chunk: {filepath}")

            # Process lines and split by headers
            for line in lines:
                header_match = re.match(r"^# (.+)", line)
                if header_match:
                    title = header_match.group(1).strip()
                    if title in title_map:
                        # If this is our first indexed section and we have intro content, save it
                        if not found_first_section and intro_buffer:
                            intro_content = normalize_chunk_markdown("".join(intro_buffer))
                            intro_filename = "0_Intro.md"
                            intro_filepath = output_path / intro_filename
                            with open(intro_filepath, "w", encoding="utf-8") as out:
                                out.write(intro_content)
                            chunks.append(intro_filepath)
                            print(f"Saved intro chunk: {intro_filepath}")
                            intro_buffer = []  # Clear intro buffer
                    
                        found_first_section = True
                        
                        # Flush previous buffer
                        flush_buffer(current_title, current_parent, current_buffer)
                        
                        # Start new buffer
                        current_title = title
                        current_parent = title_map[title]["parent"]
                        current_buffer = [line]
                    else:
                        if not found_first_section:
                            intro_buffer.append(line)  # Add to intro if we haven't found first section yet
                        else:
                            current_buffer.append(line)  # Add to current section otherwise
                else:
                    if not found_first_section:
                        intro_buffer.append(line)  # Add to intro if we haven't found first section yet
                    else:
                        current_buffer.append(line)  # Add to current section otherwise

            # Handle remaining content
            # Save intro if we never found any indexed sections
            if not found_first_section and intro_buffer:
                intro_content = normalize_chunk_markdown("".join(intro_buffer))
                intro_filename = "0_Intro.md"
                intro_filepath = output_path / intro_filename
                with open(intro_filepath, "w", encoding="utf-8") as out:
                    out.write(intro_content)
                chunks.append(intro_filepath)
                print(f"Saved intro chunk: {intro_filepath}")

            # Don't forget the last buffer
            flush_buffer(current_title, current_parent, current_buffer)
            
            duration = time.time() - start_time
            
            self._log_operation(
                document_id=document_id,
                operation="split_markdown_to_chunks",
                success=True,
                duration=duration,
                data={
                    "markdown_path": markdown_path,
                    "output_dir": str(output_path),
                    "chunks_created": len(chunks),
                    "structure_sections": len(structure.sections)
                }
            )
            
            print(f"Split markdown into {len(chunks)} chunk files")
            return chunks
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error splitting markdown: {e}"
            
            self._log_operation(
                document_id=document_id,
                operation="split_markdown_to_chunks",
                success=False,
                duration=duration,
                error=error_msg
            )
            
            print(f"Error splitting markdown: {e}")
            return []

    # =============================================================================
    # MAIN PIPELINE ORCHESTRATOR
    # =============================================================================

    def run_index_and_chunking_pipeline(
        self, 
        pdf_path: str, 
        markdown_path: str, 
        document_id: str
    ) -> Optional[DocumentStructure]:
        """
        Complete pipeline: preview -> index extraction -> split markdown.
        
        Args:
            pdf_path: Path to PDF file
            markdown_path: Path to processed markdown
            document_id: Document identifier
            
        Returns:
            DocumentStructure if successful, None otherwise
        """
        if not self.client:
            print("OpenAI client not available - cannot run pipeline")
            return None

        print(f"Starting indexing pipeline for document: {document_id}")
        start_time = time.time()

        try:
            # Step 1: Extract preview
            preview_path = self.extract_first_pages_pdf(pdf_path, document_id)
            if not preview_path:
                print("Failed to create PDF preview - continuing with original file")
                preview_path = pdf_path

            # Step 2: Extract index with OpenAI
            index_data = self.extract_index_with_openai(preview_path, document_id)

            # Step 3: Build document structure
            if index_data and index_data.index_elements and len(index_data.index_elements) >= 2:
                doc_structure = self.build_document_structure(
                    index_data.index_elements, 
                    index_data.sub_index_elements
                )
                print("Document structure built from extracted index")
            else:
                print("No index detected - falling back to markdown H1 structure")
                h1_titles = self.extract_h1_titles_from_markdown(markdown_path)
                
                if not h1_titles:
                    print("No H1 headings found in markdown - cannot proceed")
                    return None
                
                doc_structure = self.build_structure_from_titles_with_openai(h1_titles, document_id)
                if not doc_structure:
                    print("Fallback structure could not be created")
                    return None

            # Step 4: Save structure
            structure_saved = self._save_document_structure(document_id, doc_structure)
            if not structure_saved:
                print("Warning: Could not save document structure")

            # Step 5: Split markdown into chunk files
            chunks = self.split_markdown_to_subsection_files(
                markdown_path, 
                doc_structure, 
                document_id
            )
            
            duration = time.time() - start_time
            
            self._log_operation(
                document_id=document_id,
                operation="run_index_and_chunking_pipeline",
                success=True,
                duration=duration,
                data={
                    "pdf_path": pdf_path,
                    "markdown_path": markdown_path,
                    "structure_sections": len(doc_structure.sections),
                    "chunks_created": len(chunks),
                    "structure_saved": structure_saved
                }
            )
            
            print(f"Pipeline completed successfully for document {document_id}")
            print(f"Created {len(chunks)} chunk files")
            
            return doc_structure
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Pipeline failed: {e}"
            
            self._log_operation(
                document_id=document_id,
                operation="run_index_and_chunking_pipeline",
                success=False,
                duration=duration,
                error=error_msg
            )
            
            print(f"Pipeline failed: {e}")
            return None

    # =============================================================================
    # ADVANCED RE-CHUNKING
    # =============================================================================

    def load_chunks_from_directory(self, chunks_dir: str, document_id: str) -> List[Tuple[str, str]]:
        """
        Load all markdown chunks and extract title-content pairs.
        
        Args:
            chunks_dir: Directory containing chunk files
            document_id: Document identifier
            
        Returns:
            List of (title, content) tuples
        """
        start_time = time.time()
        
        try:
            chunks_path = Path(chunks_dir)
            if not chunks_path.exists():
                print(f"Chunks directory not found: {chunks_dir}")
                return []

            sections: List[Tuple[str, str]] = []
            
            for md_file in sorted(chunks_path.glob("*.md")):
                try:
                    content = md_file.read_text(encoding="utf-8").strip()
                    title = extract_markdown_title(content)
                    
                    if title and content:
                        sections.append((title, content))
                        print(f"Loaded section: {title}")
                    else:
                        print(f"Warning: No title found in file: {md_file.name}")
                        
                except Exception as e:
                    print(f"Error reading {md_file}: {e}")

            duration = time.time() - start_time
            
            self._log_operation(
                document_id=document_id,
                operation="load_chunks_from_directory",
                success=True,
                duration=duration,
                data={
                    "chunks_dir": chunks_dir,
                    "sections_loaded": len(sections)
                }
            )

            return sections
            
        except Exception as e:
            duration = time.time() - start_time
            
            self._log_operation(
                document_id=document_id,
                operation="load_chunks_from_directory",
                success=False,
                duration=duration,
                error=str(e)
            )
            
            print(f"Error loading chunks: {e}")
            return []

    @staticmethod
    def extract_all_titles_from_sections(sections: List[Tuple[str, str]]) -> List[str]:
        """Extract titles from section tuples."""
        return [title for title, _ in sections]

    def process_chunks_directory(
        self,
        document_id: str,
        chunks_dir: Optional[str] = None,
        target_size: int = 1000,
        tolerance: int = 150
    ) -> Tuple[List[str], List[Dict]]:
        """
        Re-chunk a folder of subsection files using advanced chunking logic.
        
        Args:
            document_id: Document identifier
            chunks_dir: Directory of chunks (uses default if None)
            target_size: Target chunk size in characters
            tolerance: Size tolerance
            
        Returns:
            Tuple of (final_chunks, metadata)
        """
        start_time = time.time()
        
        # Validate parameters
        valid, error_msg = validate_chunk_parameters(target_size, tolerance)
        if not valid:
            print(f"Invalid chunking parameters: {error_msg}")
            return [], []
        
        # Use default chunks directory if not specified
        if chunks_dir is None:
            paths = self.get_document_paths(document_id)
            chunks_dir = str(paths["chunks_dir"])
            
        print(f"Processing chunks directory for document {document_id}: {chunks_dir}")
        
        try:
            # Load sections from chunk files
            sections = self.load_chunks_from_directory(chunks_dir, document_id)
            if not sections:
                print("No sections loaded for re-chunking")
                return [], []

            print(f"Loaded {len(sections)} sections")
            
            # Extract titles and combine text
            titles = self.extract_all_titles_from_sections(sections)
            combined_text = "\n\n".join([content for _, content in sections])
            
            print(f"Combined text length: {len(combined_text):,} characters")

            # Use advanced chunking functions from utils
            sections_for_chunking = split_sections_by_titles(combined_text, titles)
            print(f"Found {len(sections_for_chunking)} sections for chunking")

            final_chunks, metadata = split_chunks_with_metadata(
                sections_for_chunking,
                target=target_size,
                tol=tolerance
            )

            # Calculate statistics
            chunk_stats = calculate_chunk_statistics(final_chunks)
            
            duration = time.time() - start_time
            
            self._log_operation(
                document_id=document_id,
                operation="process_chunks_rechunk",
                success=True,
                duration=duration,
                data={
                    "chunks_dir": chunks_dir,
                    "input_sections": len(sections),
                    "target_size": target_size,
                    "tolerance": tolerance,
                    "final_chunks_count": len(final_chunks),
                    "chunk_statistics": chunk_stats
                }
            )

            print(f"Generated {len(final_chunks)} final chunks")
            if final_chunks:
                print("Chunk statistics:")
                print(f"   - Average length: {chunk_stats['avg_length']:.0f} characters")
                print(f"   - Min length: {chunk_stats['min_length']} characters")
                print(f"   - Max length: {chunk_stats['max_length']} characters")
                print(f"   - Target range: {target_size-tolerance}-{target_size+tolerance} characters")

            return final_chunks, metadata
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error in re-chunking process: {e}"
            
            self._log_operation(
                document_id=document_id,
                operation="process_chunks_rechunk",
                success=False,
                duration=duration,
                error=error_msg
            )
            
            print(f"Re-chunking failed: {e}")
            return [], []

    def save_final_chunks(
        self,
        chunks: List[str],
        metadata: List[Dict],
        document_id: str,
        output_dir: Optional[str] = None
    ) -> bool:
        """
        Save re-chunked results to disk with front-matter comments.
        Uses: data/processed/{document_id}/final_chunks/
        
        Args:
            chunks: List of chunk contents
            metadata: List of chunk metadata
            document_id: Document identifier
            output_dir: Custom output directory (optional)
            
        Returns:
            True if successful
        """
        start_time = time.time()
        
        try:
            # Use default final_chunks directory if not specified
            if output_dir is None:
                paths = self.get_document_paths(document_id)
                output_path = paths["final_chunks_dir"]
            else:
                output_path = Path(output_dir)
                
            ensure_directory(output_path)

            # Save each chunk with front-matter
            for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
                title = meta.get('title', 'unknown')
                safe_title = sanitize_filename(title)
                filename = f"chunk_{i:03d}_{safe_title}.md"
                filepath = output_path / filename
                
                # Create front-matter
                front_matter = [
                    f"<!-- Document ID: {document_id} -->",
                    f"<!-- Title: {title} -->", 
                    f"<!-- Chunk: {i+1}/{len(chunks)} -->",
                    f"<!-- Length: {len(chunk)} characters -->",
                    ""
                ]
                
                # Write file
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("\n".join(front_matter))
                    f.write(chunk)

            duration = time.time() - start_time
            
            self._log_operation(
                document_id=document_id,
                operation="save_final_chunks",
                success=True,
                duration=duration,
                data={
                    "output_dir": str(output_path),
                    "chunks_saved": len(chunks),
                    "total_characters": sum(len(c) for c in chunks)
                }
            )

            print(f"Saved {len(chunks)} final chunks to: {output_path}")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error saving final chunks: {e}"
            
            self._log_operation(
                document_id=document_id,
                operation="save_final_chunks",
                success=False,
                duration=duration,
                error=error_msg
            )
            
            print(f"Failed to save final chunks: {e}")
            return False

    def rechunk_directory(
        self,
        document_id: str,
        chunks_dir: Optional[str] = None,
        target_chunk_size: int = 10000,
        tolerance: int = 1500,
        save: bool = False,
        save_dir: Optional[str] = None
    ) -> Tuple[List[str], List[Dict]]:
        """
        Complete re-chunking workflow: process -> optionally save.
        
        Args:
            document_id: Document identifier
            chunks_dir: Input chunks directory (uses default if None)
            target_chunk_size: Target chunk size
            tolerance: Size tolerance
            save: Whether to save final chunks
            save_dir: Where to save (uses default if None)
            
        Returns:
            Tuple of (final_chunks, metadata)
        """
        # Process chunks
        final_chunks, metadata = self.process_chunks_directory(
            document_id=document_id,
            chunks_dir=chunks_dir,
            target_size=target_chunk_size,
            tolerance=tolerance
        )
        
        # Save if requested and chunks were generated
        if save and final_chunks:
            save_success = self.save_final_chunks(
                final_chunks, 
                metadata, 
                document_id, 
                save_dir
            )
            if not save_success:
                print("Warning: Final chunks generated but could not be saved")
                
        return final_chunks, metadata

    # =============================================================================
    # STATUS AND UTILITY METHODS
    # =============================================================================

    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """
        Get comprehensive processing status for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary with document status information
        """
        try:
            paths = self.get_document_paths(document_id)
            
            status = {
                "document_id": document_id,
                "exists": {
                    "processed_dir": paths["processed_dir"].exists(),
                    "chunks_dir": paths["chunks_dir"].exists(),
                    "final_chunks_dir": paths["final_chunks_dir"].exists(),
                    "structure_file": paths["structure_file"].exists(),
                    "indexing_log": paths["indexing_log"].exists()
                },
                "counts": {
                    "chunks": 0,
                    "final_chunks": 0
                },
                "structure": None,
                "last_operation": None
            }
            
            # Count chunks
            if paths["chunks_dir"].exists():
                status["counts"]["chunks"] = len(list(paths["chunks_dir"].glob("*.md")))
                
            if paths["final_chunks_dir"].exists():
                status["counts"]["final_chunks"] = len(list(paths["final_chunks_dir"].glob("*.md")))
            
            # Load structure if exists
            if paths["structure_file"].exists():
                structure = self.load_document_structure(document_id)
                if structure:
                    status["structure"] = {
                        "sections_count": len(structure.sections),
                        "has_subsections": any(s.sub_sections for s in structure.sections)
                    }
            
            # Get last operation from log
            if paths["indexing_log"].exists():
                try:
                    with open(paths["indexing_log"], "r", encoding="utf-8") as f:
                        logs = json.load(f)
                    if isinstance(logs, list) and logs:
                        status["last_operation"] = logs[-1]
                    elif isinstance(logs, dict):
                        status["last_operation"] = logs
                except:
                    pass
            
            return status
            
        except Exception as e:
            print(f"Error getting document status: {e}")
            return {"document_id": document_id, "error": str(e)}

    def list_documents(self) -> List[str]:
        """
        List all document IDs that have processing data.
        
        Returns:
            List of document IDs
        """
        try:
            if not self.processed_dir.exists():
                return []
                
            return [
                d.name for d in self.processed_dir.iterdir() 
                if d.is_dir() and validate_document_id(d.name)
            ]
            
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []

    def cleanup_document(self, document_id: str, keep_structure: bool = True) -> bool:
        """
        Clean up processing artifacts for a document.
        
        Args:
            document_id: Document identifier
            keep_structure: Whether to preserve structure file
            
        Returns:
            True if successful
        """
        try:
            paths = self.get_document_paths(document_id)
            
            # Clean chunks
            if paths["chunks_dir"].exists():
                import shutil
                shutil.rmtree(paths["chunks_dir"])
                print(f"Cleaned chunks directory: {paths['chunks_dir']}")
            
            # Clean final chunks
            if paths["final_chunks_dir"].exists():
                import shutil
                shutil.rmtree(paths["final_chunks_dir"])
                print(f"Cleaned final chunks directory: {paths['final_chunks_dir']}")
            
            # Clean temp
            if paths["temp_dir"].exists():
                import shutil
                shutil.rmtree(paths["temp_dir"])
                print(f"Cleaned temp directory: {paths['temp_dir']}")
            
            # Optionally clean structure
            if not keep_structure and paths["structure_file"].exists():
                paths["structure_file"].unlink()
                print(f"Removed structure file: {paths['structure_file']}")
            
            self._log_operation(
                document_id=document_id,
                operation="cleanup_document",
                success=True,
                data={
                    "keep_structure": keep_structure,
                    "cleaned": ["chunks", "final_chunks", "temp"] + ([] if keep_structure else ["structure"])
                }
            )
            
            print(f"Cleanup completed for document: {document_id}")
            return True
            
        except Exception as e:
            self._log_operation(
                document_id=document_id,
                operation="cleanup_document",
                success=False,
                error=str(e)
            )
            
            print(f"Error cleaning document {document_id}: {e}")
            return False

    # =============================================================================
    # LEGACY COMPATIBILITY INTERFACES
    # =============================================================================

    def run_index_and_chunking_pipeline_legacy(
        self, 
        pdf_path: str, 
        markdown_path: str
    ) -> Optional[DocumentStructure]:
        """Legacy interface - generates document_id from filename."""
        document_id = sanitize_stem(Path(pdf_path).name)
        return self.run_index_and_chunking_pipeline(pdf_path, markdown_path, document_id)

    def process_chunks_directory_legacy(
        self,
        chunks_dir: str,
        target_size: int = 1000,
        tolerance: int = 150
    ) -> Tuple[List[str], List[Dict]]:
        """Legacy interface - infers document_id from path."""
        chunks_path = Path(chunks_dir)
        
        # Try to extract document_id from path structure
        if "processed" in chunks_path.parts:
            try:
                processed_idx = chunks_path.parts.index("processed")
                if processed_idx + 1 < len(chunks_path.parts):
                    document_id = chunks_path.parts[processed_idx + 1]
                else:
                    document_id = chunks_path.parent.name
            except:
                document_id = chunks_path.parent.name
        else:
            document_id = chunks_path.parent.name
            
        print(f"Inferred document_id: {document_id}")
        return self.process_chunks_directory(document_id, chunks_dir, target_size, tolerance)

    def rechunk_directory_legacy(
        self,
        chunks_dir: str,
        target_chunk_size: int = 10000,
        tolerance: int = 1500,
        save: bool = False,
        save_dir: str = "final_chunks"
    ) -> Tuple[List[str], List[Dict]]:
        """Legacy interface - maintains exact compatibility."""
        chunks_path = Path(chunks_dir)
        
        # Infer document_id from path
        if "processed" in chunks_path.parts:
            try:
                processed_idx = chunks_path.parts.index("processed")
                if processed_idx + 1 < len(chunks_path.parts):
                    document_id = chunks_path.parts[processed_idx + 1]
                else:
                    document_id = chunks_path.parent.name
            except:
                document_id = chunks_path.parent.name
        else:
            document_id = chunks_path.parent.name
            
        print(f"Inferred document_id for rechunk: {document_id}")
        
        return self.rechunk_directory(
            document_id=document_id,
            chunks_dir=chunks_dir,
            target_chunk_size=target_chunk_size,
            tolerance=tolerance,
            save=save,
            save_dir=save_dir if save else None
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pipeline_with_env(base_data_dir: Path = Path("data")) -> DocumentIndexingPipeline:
    """
    Create a pipeline with environment auto-loading.
    
    Args:
        base_data_dir: Base data directory
        
    Returns:
        Configured DocumentIndexingPipeline
    """
    return DocumentIndexingPipeline(base_data_dir=base_data_dir, auto_env=True)