"""
utils.py - Helper functions for document processing pipeline.

This module contains utility functions for text processing, chunking, and file operations
that are shared across the document processing pipeline components.
"""

import re
import html
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime


# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================

def normalize_chunk_markdown(content: str) -> str:
    """
    Normalizes markdown content by cleaning up HTML entities, quotes, and whitespace.
    
    Args:
        content: Raw markdown content
        
    Returns:
        Normalized markdown content
    """
    # Decode HTML entities
    content = html.unescape(content)
    
    # Normalize quotes and dashes
    content = (content.replace("\u201c", '"').replace("\u201d", '"')
                     .replace("\u2018", "'").replace("\u2019", "'")
                     .replace("\u2013", "-").replace("\u2014", "-"))
    
    # Normalize unicode
    content = unicodedata.normalize("NFKC", content)
    
    # Clean up excessive newlines
    content = re.sub(r"\n{3,}", "\n\n", content)
    
    return content.strip()


def sanitize_filename(title: str) -> str:
    """
    Converts a title into a safe filename by removing invalid characters.
    
    Args:
        title: Original title text
        
    Returns:
        Sanitized filename (without extension)
    """
    title = title.lower().strip()
    # Remove invalid characters
    title = re.sub(r"[^\w\s-]", "", title)
    # Replace spaces and underscores with single underscores
    title = re.sub(r"[\s_-]+", "_", title)
    return title


def sanitize_stem(filename: str) -> str:
    """
    Sanitizes a filename stem for use as document_id.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized stem suitable for use as document_id
    """
    if not filename:
        return "unknown_document"
    
    # Get stem without extension
    stem = Path(filename).stem
    
    # Basic sanitization
    stem = stem.lower().strip()
    # Remove or replace problematic characters
    stem = re.sub(r"[^\w\s-]", "", stem)
    stem = re.sub(r"[\s_-]+", "_", stem)
    
    # Ensure it's not empty
    if not stem:
        return "unknown_document"
    
    return stem


# =============================================================================
# TEXT CHUNKING UTILITIES
# =============================================================================

def split_sections_by_titles(text: str, titles: List[str]) -> List[Tuple[str, str]]:
    """
    Divide el texto en secciones basadas en los títulos dados.
    Devuelve una lista de tuplas: (título, contenido de la sección).
    
    Args:
        text: Full text content
        titles: List of section titles to split on
        
    Returns:
        List of tuples (title, section_content)
    """
    if not titles:
        return []
    
    # Escape special regex characters in titles
    pattern_titles = [re.escape(t) for t in titles]
    regex = re.compile(rf"({'|'.join(pattern_titles)})", flags=re.MULTILINE)

    matches = list(regex.finditer(text))
    sections = []

    for i, match in enumerate(matches):
        start = match.start()
        title = match.group(1)
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        content = text[start:end].strip()
        
        if content:  # Only add non-empty sections
            sections.append((title, content))

    return sections


def split_paragraphs_and_tables(section_text: str) -> List[str]:
    """
    Divide una sección en bloques de párrafos y tablas.
    
    Args:
        section_text: Text content of a section
        
    Returns:
        List of text blocks (paragraphs and tables)
    """
    if not section_text.strip():
        return []
    
    lines = section_text.split('\n')
    blocks = []
    i = 0
    n = len(lines)

    while i < n:
        if lines[i].strip().startswith('|'):
            # Table block
            tbl_lines = []
            while i < n and lines[i].strip().startswith('|'):
                tbl_lines.append(lines[i])
                i += 1
            table_content = '\n'.join(tbl_lines).strip()
            if table_content:
                blocks.append(table_content)
        else:
            # Paragraph block
            para_lines = []
            while i < n and lines[i].strip() != '' and not lines[i].strip().startswith('|'):
                para_lines.append(lines[i])
                i += 1
            paragraph_content = '\n'.join(para_lines).strip()
            if paragraph_content:
                blocks.append(paragraph_content)
            
            # Skip empty lines
            while i < n and lines[i].strip() == '':
                i += 1

    return [b for b in blocks if b.strip()]


def split_chunks_with_metadata(
    sections: List[Tuple[str, str]], 
    target: int = 1000, 
    tol: int = 150
) -> Tuple[List[str], List[Dict]]:
    """
    Divide el contenido en chunks de tamaño aproximado al `target`, conservando tablas y párrafos enteros.
    
    Args:
        sections: List of (title, content) tuples
        target: Target chunk size in characters
        tol: Tolerance for chunk size variation
        
    Returns:
        Tuple of (chunks_list, metadata_list)
    """
    if not sections:
        return [], []
    
    lower = max(1, target - tol)  # Ensure positive lower bound
    upper = target + tol
    chunks = []
    metadata = []

    for title, section_text in sections:
        if not section_text.strip():
            continue
            
        blocks = split_paragraphs_and_tables(section_text)
        if not blocks:
            continue
            
        current = ""

        for b in blocks:
            if not current:
                current = b
            else:
                candidate_length = len(current) + 2 + len(b)  # +2 for "\n\n"
                
                if candidate_length <= upper:
                    current = current + "\n\n" + b
                else:
                    # Current chunk would exceed upper limit
                    if len(current) < lower:
                        # Current is too small, add this block anyway
                        current = current + "\n\n" + b
                    else:
                        # Current is good size, save it and start new chunk
                        chunks.append(current.strip())
                        metadata.append({'title': title, 'chunk_size': len(current)})
                        current = b

        # Don't forget the last chunk
        if current.strip():
            chunks.append(current.strip())
            metadata.append({'title': title, 'chunk_size': len(current)})

    return chunks, metadata


# =============================================================================
# FILE AND PATH UTILITIES
# =============================================================================

def ensure_directory(path: Path) -> bool:
    """
    Ensures a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        True if directory exists/was created successfully
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        return False


def safe_read_json(file_path: Path) -> Dict[str, Any]:
    """
    Safely reads a JSON file, returning empty dict on error.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data or empty dict
    """
    try:
        if not file_path.exists():
            return {}
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Error reading JSON from {file_path}: {e}")
        return {}


def safe_write_json(file_path: Path, data: Any) -> bool:
    """
    Safely writes data to a JSON file.
    
    Args:
        file_path: Path to write to
        data: Data to serialize
        
    Returns:
        True if successful
    """
    try:
        ensure_directory(file_path.parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error writing JSON to {file_path}: {e}")
        return False


def get_file_size_safely(file_path: Path) -> int:
    """
    Gets file size safely, returning 0 on error.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes, or 0 if error
    """
    try:
        return file_path.stat().st_size if file_path.exists() else 0
    except:
        return 0


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def create_log_entry(
    operation: str,
    success: bool,
    duration: float = 0.0,
    data: Dict[str, Any] = None,
    error: str = None
) -> Dict[str, Any]:
    """
    Creates a standardized log entry.
    
    Args:
        operation: Name of the operation
        success: Whether operation succeeded
        duration: Time taken in seconds
        data: Additional data to log
        error: Error message if failed
        
    Returns:
        Standardized log entry dict
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        "success": success,
        "duration": duration
    }
    
    if data:
        entry["data"] = data
        
    if error:
        entry["error"] = error
    
    return entry


def append_to_log(log_file: Path, entry: Dict[str, Any]) -> bool:
    """
    Appends an entry to a log file (JSON array format).
    
    Args:
        log_file: Path to log file
        entry: Log entry to append
        
    Returns:
        True if successful
    """
    try:
        # Load existing logs
        logs = []
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if isinstance(existing, list):
                logs = existing
            elif isinstance(existing, dict):
                logs = [existing]
        
        # Append new entry
        logs.append(entry)
        
        # Write back
        return safe_write_json(log_file, logs)
        
    except Exception as e:
        print(f"Error appending to log {log_file}: {e}")
        return False


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_document_id(document_id: str) -> bool:
    """
    Validates that a document_id is safe for use in file paths.
    
    Args:
        document_id: Document identifier to validate
        
    Returns:
        True if valid
    """
    if not document_id or not isinstance(document_id, str):
        return False
    
    # Check for dangerous characters
    dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
    if any(char in document_id for char in dangerous_chars):
        return False
    
    # Check length
    if len(document_id) > 100:  # Reasonable limit
        return False
    
    return True


def validate_chunk_parameters(target: int, tolerance: int) -> Tuple[bool, str]:
    """
    Validates chunking parameters.
    
    Args:
        target: Target chunk size
        tolerance: Size tolerance
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(target, int) or target <= 0:
        return False, "Target must be a positive integer"
    
    if not isinstance(tolerance, int) or tolerance < 0:
        return False, "Tolerance must be a non-negative integer"
    
    if tolerance >= target:
        return False, "Tolerance should be less than target"
    
    return True, ""


# =============================================================================
# MARKDOWN UTILITIES
# =============================================================================

def extract_markdown_title(content: str) -> str:
    """
    Extracts the first H1 title from markdown content.
    
    Args:
        content: Markdown content
        
    Returns:
        Title string, or empty string if no title found
    """
    if not content:
        return ""
    
    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith('# '):
            return line[2:].strip()
    
    return ""


def remove_front_matter_comments(content: str) -> str:
    """
    Removes front-matter HTML comments from markdown content.
    
    Args:
        content: Markdown content with possible front-matter
        
    Returns:
        Clean content without front-matter comments
    """
    if not content:
        return ""
    
    lines = content.splitlines()
    content_start = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("<!-- ") and stripped.endswith(" -->"):
            content_start = i + 1
        else:
            break
    
    return "\n".join(lines[content_start:]).strip()


# =============================================================================
# CHUNKING STATISTICS
# =============================================================================

def calculate_chunk_statistics(chunks: List[str]) -> Dict[str, Any]:
    """
    Calculates statistics for a list of chunks.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            "count": 0,
            "total_length": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0
        }
    
    lengths = [len(chunk) for chunk in chunks]
    
    return {
        "count": len(chunks),
        "total_length": sum(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths)
    }