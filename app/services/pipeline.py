from importlib import metadata
import os
import time
import traceback
import tiktoken
import re
import html
import unicodedata
import json
import numpy as np
import faiss
import shutil
import threading
from textwrap import dedent
from pydantic import BaseModel
from PyPDF2 import PdfReader, PdfWriter
from docx2pdf import convert
from openai import OpenAI
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from concurrent.futures import ThreadPoolExecutor, as_completed

import voyageai

TOKEN_MODEL = "gpt-4" #
LLAMAPARSE_AVAILABLE = True #
EMBEDDING_DIM = 1536  # Dimension for text-embedding-3-small
EMBEDDING_MODEL = "text-embedding-3-small"  #




def split_sections_by_titles(text: str, titles: List[str]) -> List[Tuple[str, str]]:
    """
    Divide el texto en secciones basadas en los títulos dados.
    Devuelve una lista de tuplas: (título, contenido de la sección).
    """
    pattern_titles = [re.escape(t) for t in titles]
    regex = re.compile(rf"({'|'.join(pattern_titles)})", flags=re.MULTILINE)

    matches = list(regex.finditer(text))
    sections = []

    for i, match in enumerate(matches):
        start = match.start()
        title = match.group(1)
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append((title, content))

    return sections


def split_paragraphs_and_tables(section_text: str) -> List[str]:
    """
    Divide una sección en bloques de párrafos y tablas.
    """
    lines = section_text.split('\n')
    blocks = []
    i = 0
    n = len(lines)

    while i < n:
        if lines[i].startswith('|'):
            tbl_lines = []
            while i < n and lines[i].startswith('|'):
                tbl_lines.append(lines[i])
                i += 1
            blocks.append('\n'.join(tbl_lines).strip())
        else:
            para_lines = []
            while i < n and lines[i].strip() != '' and not lines[i].startswith('|'):
                para_lines.append(lines[i])
                i += 1
            blocks.append('\n'.join(para_lines).strip())
            while i < n and lines[i].strip() == '':
                i += 1

    return [b for b in blocks if b.strip()]


def split_chunks_with_metadata(sections: List[Tuple[str, str]], target=1000, tol=150) -> Tuple[List[str], List[Dict]]:
    """
    Divide el contenido en chunks de tamaño aproximado al `target`, conservando tablas y párrafos enteros.
    Devuelve:
        - lista de chunks
        - lista de metadatos con el título asociado a cada chunk
    """
    lower = target - tol
    upper = target + tol
    chunks = []
    metadata = []

    for title, section_text in sections:
        blocks = split_paragraphs_and_tables(section_text)
        current = ""

        for b in blocks:
            if not current:
                current = b
            else:
                candidate_length = len(current) + 2 + len(b)
                if candidate_length <= upper:
                    current = current + "\n\n" + b
                else:
                    if len(current) < lower:
                        current = current + "\n\n" + b
                    else:
                        chunks.append(current.strip())
                        metadata.append({'title': title})
                        current = b

        if current:
            chunks.append(current.strip())
            metadata.append({'title': title})

    return chunks, metadata


class LlamaParseProcessor:
    """
    Handles document processing using LlamaIndex's LlamaParse service.
    Uses new data structure: data/documents/{document_id}/ and data/processed/{document_id}/
    """
    
    def __init__(self, base_data_dir: Path = Path("data")):
        """
        Initializes the LlamaParseProcessor.
        
        Args:
            base_data_dir: Base directory for data storage (default: "data")
        """
        if not LLAMAPARSE_AVAILABLE:
            raise ImportError("LlamaParse library is not available or API key not found")
            
        load_dotenv()
        api_key = os.getenv("LLAMAPARSE_API_KEY")
        if not api_key:
            raise ValueError("Missing LlamaParse API key in environment variables")
            
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",  # type: ignore
            verbose=True
        )
        
        self.base_data_dir = Path(base_data_dir)
        self.documents_dir = self.base_data_dir / "documents"
        self.processed_dir = self.base_data_dir / "processed"
        
        # Crear directorios base
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_file(self, file_path: Path, document_id: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Processes a file using LlamaParse and returns markdown content with metadata.
        
        Args:
            file_path: Path to the file to process
            document_id: Unique identifier for the document. If None, uses filename without extension
            
        Returns:
            Tuple of (markdown_content, metadata)
        """
        # Generar document_id si no se proporciona
        if document_id is None:
            document_id = file_path.stem
        
        # Configurar paths para este documento
        doc_dir = self.documents_dir / document_id
        processed_doc_dir = self.processed_dir / document_id
        
        start_time = time.time()
        
        try:
            # Procesar el archivo
            documents = self.parser.load_data(str(file_path))
            markdown_content = "\n\n".join([doc.text for doc in documents])
            processing_time = time.time() - start_time
            token_count = self.count_tokens(markdown_content)

            # Crear metadata completa
            metadata = {
                "document_id": document_id,
                "original_file": str(file_path),
                "processing_time": processing_time,
                "token_count": token_count,
                "content_length": len(markdown_content),
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat(),
                "pages_processed": len(documents),
                "processor": "LlamaParse",
                "file_size": file_path.stat().st_size if file_path.exists() else 0
            }
            
            # Guardar documento original y metadata si el procesamiento fue exitoso
            self._save_document_data(file_path, doc_dir, metadata)
            
            # Crear log de procesamiento
            self._log_processing(processed_doc_dir, metadata)
            
            return markdown_content, metadata

        except Exception as e:
            processing_time = time.time() - start_time
            metadata = {
                "document_id": document_id,
                "original_file": str(file_path),
                "library": "LlamaParse",
                "processing_time": processing_time,
                "token_count": 0,
                "content_length": 0,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "pages_processed": 0,
                "processor": "LlamaParse",
                "file_size": file_path.stat().st_size if file_path.exists() else 0
            }
            
            # Log del error
            self._log_processing(processed_doc_dir, metadata)
            
            return "", metadata

    def save_output(self, content: str, output_dir: Path, name: str) -> bool:
        """
        Saves processed content to specified directory.
        Mantiene la interfaz pública original pero mejorada internamente.
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / name
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error saving output: {e}")
            traceback.print_exc()
            return False

    def process_and_save(self, file_path: Path, document_id: Optional[str] = None) -> Tuple[bool, Dict]:
        """
        Procesa un archivo y guarda el resultado en la nueva estructura.
        Método de conveniencia que combina procesamiento y guardado.
        
        Args:
            file_path: Path al archivo a procesar
            document_id: ID único del documento
            
        Returns:
            Tuple of (success, metadata)
        """
        if document_id is None:
            document_id = file_path.stem
            
        # Procesar archivo
        content, metadata = self.process_file(file_path, document_id)
        
        if not metadata["success"]:
            return False, metadata
            
        # Guardar contenido procesado
        processed_doc_dir = self.processed_dir / document_id
        markdown_file = processed_doc_dir / "processed_content.md"
        
        success = self.save_output(content, processed_doc_dir, "processed_content.md")
        
        # Actualizar metadata con información de guardado
        metadata["output_saved"] = success
        metadata["output_path"] = str(markdown_file) if success else None
        
        # Actualizar log
        self._log_processing(processed_doc_dir, metadata, update=True)
        
        return success, metadata

    def get_document_paths(self, document_id: str) -> Dict[str, Path]:
        """
        Obtiene los paths relevantes para un documento específico.
        
        Args:
            document_id: ID del documento
            
        Returns:
            Diccionario con paths del documento
        """
        doc_dir = self.documents_dir / document_id
        processed_dir = self.processed_dir / document_id
        
        return {
            "document_dir": doc_dir,
            "processed_dir": processed_dir,
            "original_pdf": doc_dir / "original.pdf",
            "metadata_file": doc_dir / "metadata.json",
            "processed_content": processed_dir / "processed_content.md",
            "processing_log": processed_dir / "processing_log.json"
        }

    def _save_document_data(self, source_file: Path, doc_dir: Path, metadata: Dict) -> None:
        """
        Guarda el documento original y su metadata en la estructura de datos.
        
        Args:
            source_file: Archivo fuente original
            doc_dir: Directorio del documento
            metadata: Metadata del procesamiento
        """
        try:
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            # Copiar archivo original si existe
            if source_file.exists():
                original_file = doc_dir / f"original{source_file.suffix}"
                shutil.copy2(source_file, original_file)
            
            # Guardar metadata del documento
            doc_metadata = {
                "document_id": metadata["document_id"],
                "original_filename": source_file.name,
                "file_size": metadata["file_size"],
                "created_at": metadata["timestamp"],
                "file_type": source_file.suffix.lower(),
                "processing_metadata": {
                    "processor": metadata.get("processor", "LlamaParse"),
                    "success": metadata["success"],
                    "processing_time": metadata["processing_time"],
                    "pages_processed": metadata["pages_processed"]
                }
            }
            
            metadata_file = doc_dir / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(doc_metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving document data: {e}")
            traceback.print_exc()

    def _log_processing(self, processed_dir: Path, metadata: Dict, update: bool = False) -> None:
        """
        Crea o actualiza el log de procesamiento para un documento.
        
        Args:
            processed_dir: Directorio de datos procesados
            metadata: Metadata del procesamiento
            update: Si True, actualiza log existente
        """
        try:
            processed_dir.mkdir(parents=True, exist_ok=True)
            log_file = processed_dir / "processing_log.json"
            
            log_entry = {
                "timestamp": metadata["timestamp"],
                "processor": metadata.get("processor", "LlamaParse"),
                "success": metadata["success"],
                "processing_time": metadata["processing_time"],
                "token_count": metadata["token_count"],
                "content_length": metadata["content_length"],
                "pages_processed": metadata["pages_processed"],
                "error": metadata.get("error"),
                "output_saved": metadata.get("output_saved"),
                "output_path": metadata.get("output_path")
            }
            
            # Si es una actualización y el archivo existe, cargar logs existentes
            if update and log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        existing_logs = json.load(f)
                    if not isinstance(existing_logs, list):
                        existing_logs = [existing_logs]
                except:
                    existing_logs = []
                    
                existing_logs.append(log_entry)
                log_data = existing_logs
            else:
                log_data = [log_entry]
            
            # Guardar log
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error logging processing: {e}")
            traceback.print_exc()

    @staticmethod
    def count_tokens(text: str, model_name: str = TOKEN_MODEL) -> int:
        """
        Counts the number of tokens in the provided text.
        """
        try:
            enc = tiktoken.encoding_for_model(model_name)
            return len(enc.encode(text))
        except Exception as e:
            print(f"Error counting tokens: {e}")
            return 0

# --- External helpers (expected to exist in your codebase) ---
# pipline expects these to be available
import os
import re
import json
import html
import unicodedata
from pathlib import Path
from textwrap import dedent
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from PyPDF2 import PdfReader, PdfWriter

# Asumiendo imports de tu módulo de chunking
# from chunking import split_sections_by_titles, split_chunks_with_metadata


# -------------------- Pydantic Schemas --------------------
class IndexRetrieval(BaseModel):
    class IndexSubIndex(BaseModel):
        index_element: str
        sub_index_elements: List[str]

    index_elements: List[str]
    sub_index_elements: List[IndexSubIndex]


class SectionNode(BaseModel):
    title: str
    sub_sections: Optional[List["SectionNode"]] = None

SectionNode.model_rebuild()


class DocumentStructure(BaseModel):
    sections: List[SectionNode]


# -------------------- Pipeline Class --------------------
class DocumentIndexingPipeline:
    """
    End-to-end pipeline using new data structure:
      1) Preview first pages from PDF/DOCX.
      2) Ask OpenAI to extract TOC/Index (fallback to H1 in markdown).
      3) Build hierarchical structure.
      4) Split a markdown file into subsection files.
      5) (Optional) Re-chunk those files using custom 'chunking' module.
      
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
        self.model = model
        self.base_data_dir = Path(base_data_dir)
        self.processed_dir = self.base_data_dir / "processed"
        self.max_preview_pages = max_preview_pages
        self.client = openai_client or (self._init_env() if auto_env else None)
        
        # Crear directorios base
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Env / Clients ----------
    def _init_env(self) -> Optional[OpenAI]:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] OPENAI_API_KEY not found in .env")
            return None
        print("[OK] Environment initialized")
        return OpenAI(api_key=api_key)

    # ---------- Document Path Management ----------
    def get_document_paths(self, document_id: str) -> Dict[str, Path]:
        """
        Obtiene los paths estructurados para un documento específico.
        
        Args:
            document_id: ID único del documento
            
        Returns:
            Diccionario con todos los paths relevantes
        """
        processed_doc_dir = self.processed_dir / document_id
        
        return {
            "processed_dir": processed_doc_dir,
            "chunks_dir": processed_doc_dir / "chunks",
            "final_chunks_dir": processed_doc_dir / "final_chunks",
            "temp_dir": processed_doc_dir / "temp",
            "structure_file": processed_doc_dir / "document_structure.json",
            "index_log": processed_doc_dir / "indexing_log.json",
            "preview_pdf": processed_doc_dir / "temp" / "preview.pdf"
        }

    def _log_operation(self, document_id: str, operation: str, data: Dict[str, Any]) -> None:
        """
        Registra operaciones en el log de indexing por documento.
        
        Args:
            document_id: ID del documento
            operation: Tipo de operación realizada
            data: Datos de la operación
        """
        try:
            paths = self.get_document_paths(document_id)
            log_file = paths["index_log"]
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "document_id": document_id,
                "data": data
            }
            
            # Cargar logs existentes si existen
            logs = []
            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                    if not isinstance(logs, list):
                        logs = [logs]
                except:
                    logs = []
            
            logs.append(log_entry)
            
            # Guardar logs actualizados
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error logging operation: {e}")

    def _save_document_structure(self, document_id: str, structure: DocumentStructure) -> None:
        """
        Guarda la estructura del documento en formato JSON.
        
        Args:
            document_id: ID del documento
            structure: Estructura del documento
        """
        try:
            paths = self.get_document_paths(document_id)
            structure_file = paths["structure_file"]
            
            structure_file.parent.mkdir(parents=True, exist_ok=True)
            with open(structure_file, "w", encoding="utf-8") as f:
                json.dump(structure.model_dump(), f, indent=2, ensure_ascii=False)
                
            print(f"✅ Document structure saved: {structure_file}")
            
        except Exception as e:
            print(f"Error saving document structure: {e}")

    def load_document_structure(self, document_id: str) -> Optional[DocumentStructure]:
        """
        Carga la estructura del documento desde archivo JSON.
        
        Args:
            document_id: ID del documento
            
        Returns:
            DocumentStructure si existe, None en caso contrario
        """
        try:
            paths = self.get_document_paths(document_id)
            structure_file = paths["structure_file"]
            
            if not structure_file.exists():
                return None
                
            with open(structure_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            return DocumentStructure.model_validate(data)
            
        except Exception as e:
            print(f"Error loading document structure: {e}")
            return None

    # ---------- File Utilities ----------
    def convert_docx_to_pdf(self, docx_path: str, document_id: str) -> Optional[str]:
        """
        Convierte DOCX a PDF y guarda en directorio temporal del documento.
        """
        try:
            paths = self.get_document_paths(document_id)
            temp_dir = paths["temp_dir"]
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = temp_dir / f"{Path(docx_path).stem}.pdf"
            convert(docx_path, output_path)
            
            print(f"✅ Converted DOCX to PDF: {output_path}")
            
            self._log_operation(document_id, "convert_docx_to_pdf", {
                "input_path": docx_path,
                "output_path": str(output_path),
                "success": True
            })
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ DOCX to PDF conversion failed: {e}")
            
            self._log_operation(document_id, "convert_docx_to_pdf", {
                "input_path": docx_path,
                "error": str(e),
                "success": False
            })
            
            return None

    def extract_first_pages_pdf(self, input_path: str, document_id: str) -> Optional[str]:
        """
        Extrae las primeras páginas de un PDF/DOCX para preview.
        Guarda en estructura: data/processed/{document_id}/temp/preview.pdf
        """
        ext = Path(input_path).suffix.lower()

        if ext == ".docx":
            input_path = self.convert_docx_to_pdf(input_path, document_id)  # type: ignore
            if not input_path:
                return None
            ext = ".pdf"

        if ext == ".pdf":
            try:
                paths = self.get_document_paths(document_id)
                temp_dir = paths["temp_dir"]
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                output_path = paths["preview_pdf"]
                
                reader = PdfReader(input_path)
                writer = PdfWriter()
                
                pages_extracted = min(self.max_preview_pages, len(reader.pages))
                for i in range(pages_extracted):
                    writer.add_page(reader.pages[i])

                with open(output_path, "wb") as f:
                    writer.write(f)
                    
                print(f"✅ PDF preview saved to: {output_path}")
                
                self._log_operation(document_id, "extract_pdf_preview", {
                    "input_path": input_path,
                    "output_path": str(output_path),
                    "pages_extracted": pages_extracted,
                    "total_pages": len(reader.pages),
                    "success": True
                })
                
                return str(output_path)
                
            except Exception as e:
                print(f"❌ PDF preview error: {e}")
                
                self._log_operation(document_id, "extract_pdf_preview", {
                    "input_path": input_path,
                    "error": str(e),
                    "success": False
                })
                
                return None

        print("❌ Unsupported file type.")
        return None

    # ---------- OpenAI Calls ----------
    def extract_index_with_openai(self, pdf_path: str, document_id: str) -> Optional[IndexRetrieval]:
        if not self.client:
            print("❌ OpenAI client not initialized.")
            return None

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

            print("✅ File uploaded to OpenAI")

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
            print(f"✅ Completion parsed successfully: {result}")
            
            # Log de la extracción de índice
            self._log_operation(document_id, "extract_index_openai", {
                "pdf_path": pdf_path,
                "model": self.model,
                "index_elements_count": len(result.index_elements) if result else 0,
                "sub_index_elements_count": len(result.sub_index_elements) if result else 0,
                "success": True
            })
            
            return result

        except Exception as e:
            print(f"❌ Error during OpenAI call: {e}")
            
            self._log_operation(document_id, "extract_index_openai", {
                "pdf_path": pdf_path,
                "error": str(e),
                "success": False
            })
            
            return None

    # ---------- Markdown Utilities ----------
    @staticmethod
    def extract_h1_titles_from_markdown(markdown_path: str) -> List[str]:
        with open(markdown_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        h1_titles = [line.strip("# ").strip() for line in lines if line.startswith("# ")]
        return h1_titles

    def build_structure_from_titles_with_openai(
        self, titles: List[str], document_id: str
    ) -> Optional[DocumentStructure]:
        """
        Fallback path: infer plausible hierarchy from H1 titles using OpenAI.
        """
        if not self.client:
            print("❌ OpenAI client not initialized.")
            return None

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

            print("✅ Completion parsed successfully (fallback structure)")
            parsed_result: IndexRetrieval = completion.output_parsed # type: ignore
            
            structure = self.build_document_structure(parsed_result.index_elements, parsed_result.sub_index_elements)
            
            # Log del fallback
            self._log_operation(document_id, "build_structure_fallback", {
                "h1_titles_count": len(titles),
                "titles": titles,
                "index_elements_count": len(parsed_result.index_elements),
                "success": True
            })
            
            return structure

        except Exception as e:
            print(f"❌ Fallback OpenAI parse failed: {e}")
            
            self._log_operation(document_id, "build_structure_fallback", {
                "h1_titles_count": len(titles),
                "error": str(e),
                "success": False
            })
            
            return None

    # ---------- Structure Builders ----------
    @staticmethod
    def build_document_structure(
        index_elements: List[str],
        sub_index_elements: List[IndexRetrieval.IndexSubIndex]
    ) -> DocumentStructure:
        section_nodes: List[SectionNode] = []
        sub_index_map = {
            re.sub(r"^\d+(\.\d+)*\s*", "", sub.index_element).strip(): sub.sub_index_elements
            for sub in sub_index_elements
        }

        for raw_title in index_elements:
            sub_titles = sub_index_map.get(raw_title, [])
            node = SectionNode(
                title=raw_title,
                sub_sections=[SectionNode(title=sub) for sub in sub_titles] if sub_titles else None
            )
            section_nodes.append(node)

        return DocumentStructure(sections=section_nodes)

    # ---------- Chunking Utilities ----------
    @staticmethod
    def normalize_chunk_markdown(content: str) -> str:
        content = html.unescape(content)
        content = (content.replace("\u201c", '"').replace("\u201d", '"')
                           .replace("\u2018", "'").replace("\u2019", "'")
                           .replace("\u2013", "-").replace("\u2014", "-"))
        content = unicodedata.normalize("NFKC", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()

    @staticmethod
    def sanitize_filename(title: str) -> str:
        title = title.lower().strip()
        title = re.sub(r"[^\w\s-]", "", title)
        title = re.sub(r"[\s_-]+", "_", title)
        return title

    @staticmethod
    def build_title_map(structure: DocumentStructure) -> Dict[str, Dict]:
        title_map: Dict[str, Dict] = {}
        for section in structure.sections:
            title_map[section.title] = {"parent": None, "is_subsection": False}
            if section.sub_sections:
                for sub in section.sub_sections:
                    title_map[sub.title] = {"parent": section.title, "is_subsection": True}
        return title_map

    def split_markdown_to_subsection_files(
        self,
        markdown_path: str,
        structure: DocumentStructure,
        document_id: str,
        output_dir: Optional[str] = None
    ) -> List[Path]:
        """
        Splits a single full-document markdown into multiple files aligned with structure.
        Saves to: data/processed/{document_id}/chunks/
        """
        title_map = self.build_title_map(structure)

        with open(markdown_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Usar estructura de datos nueva
        if output_dir:
            output_path = Path(output_dir)
        else:
            paths = self.get_document_paths(document_id)
            output_path = paths["chunks_dir"]
            
        output_path.mkdir(parents=True, exist_ok=True)

        current_buffer: List[str] = []
        current_title: Optional[str] = None
        current_parent: Optional[str] = None
        chunks: List[Path] = []

        def flush_buffer(title: Optional[str], parent: Optional[str], buffer: List[str]):
            if not buffer or not title:
                return
            content = self.normalize_chunk_markdown("".join(buffer))
            section_part = f"{self.sanitize_filename(parent)}_" if parent else ""
            filename = f"{section_part}{self.sanitize_filename(title)}.md"
            filepath = output_path / filename
            with open(filepath, "w", encoding="utf-8") as out:
                out.write(content)
            print(f"📄 Saved chunk: {filepath}")
            chunks.append(filepath)

        for line in lines:
            header_match = re.match(r"^# (.+)", line)
            if header_match:
                title = header_match.group(1).strip()
                if title in title_map:
                    flush_buffer(current_title, current_parent, current_buffer)
                    current_title = title
                    current_parent = title_map[title]["parent"]
                    current_buffer = [line]
                else:
                    current_buffer.append(line)
            else:
                current_buffer.append(line)

        flush_buffer(current_title, current_parent, current_buffer)
        
        # Log de la operación de splitting
        self._log_operation(document_id, "split_markdown_to_chunks", {
            "markdown_path": markdown_path,
            "output_dir": str(output_path),
            "chunks_created": len(chunks),
            "structure_sections": len(structure.sections)
        })
        
        return chunks

    # ---------- Orchestrators ----------
    def run_index_and_chunking_pipeline(
        self, 
        pdf_path: str, 
        markdown_path: str, 
        document_id: str
    ) -> Optional[DocumentStructure]:
        """
        Full run: preview -> index extraction (or fallback) -> split markdown into files.
        Uses new data structure with document_id organization.
        
        Args:
            pdf_path: Path to PDF file
            markdown_path: Path to processed markdown
            document_id: Unique document identifier
            
        Returns:
            DocumentStructure if successful, None otherwise
        """
        if not self.client:
            print("❌ No OpenAI client available.")
            return None

        print(f"🚀 Starting indexing pipeline for document: {document_id}")

        # Extraer preview
        preview_path = self.extract_first_pages_pdf(pdf_path, document_id)
        if not preview_path:
            return None

        # Extraer índice con OpenAI
        index_data = self.extract_index_with_openai(preview_path, document_id)

        # Construir estructura del documento
        if index_data and index_data.index_elements and len(index_data.index_elements) >= 2:
            doc_structure = self.build_document_structure(index_data.index_elements, index_data.sub_index_elements)
            print("✅ Document structure built from extracted index")
        else:
            print("⚠️ No index detected. Falling back to markdown H1 structure...")
            h1_titles = self.extract_h1_titles_from_markdown(markdown_path)
            if not h1_titles:
                print("❌ No H1 headings found in markdown. Cannot proceed.")
                return None
            doc_structure = self.build_structure_from_titles_with_openai(h1_titles, document_id)
            if not doc_structure:
                print("❌ Fallback structure could not be created.")
                return None

        # Guardar estructura del documento
        self._save_document_structure(document_id, doc_structure)

        # Split markdown en archivos de subsección
        chunks = self.split_markdown_to_subsection_files(markdown_path, doc_structure, document_id)
        
        print(f"✅ Pipeline completed successfully for document {document_id}")
        print(f"📁 Created {len(chunks)} chunk files")
        
        return doc_structure

    # ---------- Post-processing / Re-chunking ----------
    def load_chunks_from_directory(self, chunks_dir: str, document_id: str) -> List[Tuple[str, str]]:
        """
        Loads all .md chunks and extracts the first '# ' title as (title, content).
        """
        chunks_path = Path(chunks_dir)
        if not chunks_path.exists():
            print(f"❌ Directory not found: {chunks_dir}")
            return []

        sections: List[Tuple[str, str]] = []
        for md_file in chunks_path.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                title = None
                for line in content.splitlines():
                    if line.strip().startswith('# '):
                        title = line.strip('# ').strip()
                        break
                if title and content:
                    sections.append((title, content))
                    print(f"✅ Loaded section: {title}")
                else:
                    print(f"⚠️ No title found in file: {md_file.name}")
            except Exception as e:
                print(f"❌ Error reading {md_file}: {e}")

        self._log_operation(document_id, "load_chunks_from_directory", {
            "chunks_dir": chunks_dir,
            "sections_loaded": len(sections)
        })

        return sections

    @staticmethod
    def extract_all_titles_from_sections(sections: List[Tuple[str, str]]) -> List[str]:
        return [title for title, _ in sections]

    def process_chunks_directory(
        self,
        document_id: str,
        chunks_dir: Optional[str] = None,
        target_size: int = 1000,
        tolerance: int = 150
    ) -> Tuple[List[str], List[Dict]]:
        """
        Re-chunk a folder of subsection files using 'chunking.py' logic.
        Uses document_id to organize data properly.
        """
        # Usar chunks_dir del documento si no se especifica
        if chunks_dir is None:
            paths = self.get_document_paths(document_id)
            chunks_dir = str(paths["chunks_dir"])
            
        print(f"🔄 Processing chunks directory for document {document_id}: {chunks_dir}")
        
        sections = self.load_chunks_from_directory(chunks_dir, document_id)
        if not sections:
            print("❌ No sections loaded")
            return [], []

        print(f"✅ Loaded {len(sections)} sections")
        titles = self.extract_all_titles_from_sections(sections)
        print(f"📋 Extracted titles: {titles}")

        combined_text = "\n\n".join([content for _, content in sections])
        print(f"📝 Combined text length: {len(combined_text)} characters")

        # Usar funciones de chunking (asumiendo que están importadas)
        sections_for_chunking = split_sections_by_titles(combined_text, titles)
        print(f"🔍 Found {len(sections_for_chunking)} sections for chunking")

        final_chunks, metadata = split_chunks_with_metadata(
            sections_for_chunking,
            target=target_size,
            tol=tolerance
        )

        print(f"✅ Generated {len(final_chunks)} final chunks")
        if final_chunks:
            lengths = [len(c) for c in final_chunks]
            avg_len = sum(lengths) / len(lengths)
            print("📊 Chunk statistics:")
            print(f"   - Average length: {avg_len:.0f} characters")
            print(f"   - Min length: {min(lengths)} characters")
            print(f"   - Max length: {max(lengths)} characters")
            print(f"   - Target range: {target_size-tolerance}-{target_size+tolerance} characters")

        # Log de re-chunking
        self._log_operation(document_id, "process_chunks_rechunk", {
            "chunks_dir": chunks_dir,
            "input_sections": len(sections),
            "target_size": target_size,
            "tolerance": tolerance,
            "final_chunks_count": len(final_chunks),
            "avg_chunk_length": sum(len(c) for c in final_chunks) / len(final_chunks) if final_chunks else 0
        })

        return final_chunks, metadata

    # ---------- Legacy Interface Compatibility ----------
    def run_index_and_chunking_pipeline_legacy(
        self, pdf_path: str, markdown_path: str
    ) -> Optional[DocumentStructure]:
        """
        Legacy interface wrapper - mantiene compatibilidad con código existente.
        Genera document_id automáticamente del nombre del archivo.
        """
        document_id = Path(pdf_path).stem
        return self.run_index_and_chunking_pipeline(pdf_path, markdown_path, document_id)

    def process_chunks_directory_legacy(
        self,
        chunks_dir: str,
        target_size: int = 1000,
        tolerance: int = 150
    ) -> Tuple[List[str], List[Dict]]:
        """
        Legacy interface wrapper para process_chunks_directory.
        Infiere document_id del path del directorio.
        """
        # Intentar extraer document_id del path
        chunks_path = Path(chunks_dir)
        
        # Buscar patrón data/processed/{document_id}/chunks
        if "processed" in chunks_path.parts:
            try:
                processed_idx = chunks_path.parts.index("processed")
                if processed_idx + 1 < len(chunks_path.parts):
                    document_id = chunks_path.parts[processed_idx + 1]
                else:
                    # Fallback: usar nombre del directorio padre
                    document_id = chunks_path.parent.name
            except:
                # Fallback: usar nombre del directorio padre
                document_id = chunks_path.parent.name
        else:
            # Fallback: usar nombre del directorio padre
            document_id = chunks_path.parent.name
            
        print(f"📋 Inferred document_id: {document_id}")
        return self.process_chunks_directory(document_id, chunks_dir, target_size, tolerance)

    def save_final_chunks_legacy(
        self,
        chunks: List[str],
        metadata: List[Dict],
        output_dir: str = "final_chunks"
    ) -> None:
        """
        Legacy interface wrapper para save_final_chunks.
        Infiere document_id del path o usa genérico.
        """
        # Intentar extraer document_id del output_dir
        output_path = Path(output_dir)
        
        if "processed" in output_path.parts:
            try:
                processed_idx = output_path.parts.index("processed")
                if processed_idx + 1 < len(output_path.parts):
                    document_id = output_path.parts[processed_idx + 1]
                else:
                    document_id = "unknown_document"
            except:
                document_id = "unknown_document"
        else:
            document_id = "unknown_document"
            
        print(f"📋 Inferred document_id for save: {document_id}")
        self.save_final_chunks(chunks, metadata, document_id, output_dir)

    def rechunk_directory_legacy(
        self,
        chunks_dir: str,
        target_chunk_size: int = 10000,
        tolerance: int = 1500,
        save: bool = False,
        save_dir: str = "final_chunks"
    ) -> Tuple[List[str], List[Dict]]:
        """
        Legacy interface wrapper para rechunk_directory.
        Mantiene compatibilidad exacta con la interfaz original.
        """
        # Inferir document_id del chunks_dir
        chunks_path = Path(chunks_dir)
        
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
            
        print(f"📋 Inferred document_id for rechunk: {document_id}")
        
        return self.rechunk_directory(
            document_id=document_id,
            chunks_dir=chunks_dir,
            target_chunk_size=target_chunk_size,
            tolerance=tolerance,
            save=save,
            save_dir=save_dir
        )

    # ---------- Utility Methods ----------
    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado completo de procesamiento de un documento.
        
        Args:
            document_id: ID del documento
            
        Returns:
            Diccionario con información del estado del documento
        """
        paths = self.get_document_paths(document_id)
        
        status = {
            "document_id": document_id,
            "exists": {
                "processed_dir": paths["processed_dir"].exists(),
                "chunks_dir": paths["chunks_dir"].exists(),
                "final_chunks_dir": paths["final_chunks_dir"].exists(),
                "structure_file": paths["structure_file"].exists(),
                "index_log": paths["index_log"].exists()
            },
            "counts": {
                "chunks": 0,
                "final_chunks": 0
            },
            "structure": None,
            "last_operation": None
        }
        
        # Contar chunks
        if paths["chunks_dir"].exists():
            status["counts"]["chunks"] = len(list(paths["chunks_dir"].glob("*.md")))
            
        if paths["final_chunks_dir"].exists():
            status["counts"]["final_chunks"] = len(list(paths["final_chunks_dir"].glob("*.md")))
        
        # Cargar estructura si existe
        if paths["structure_file"].exists():
            structure = self.load_document_structure(document_id)
            if structure:
                status["structure"] = {
                    "sections_count": len(structure.sections),
                    "has_subsections": any(s.sub_sections for s in structure.sections)
                }
        
        # Obtener última operación del log
        if paths["index_log"].exists():
            try:
                with open(paths["index_log"], "r", encoding="utf-8") as f:
                    logs = json.load(f)
                if isinstance(logs, list) and logs:
                    status["last_operation"] = logs[-1]
                elif isinstance(logs, dict):
                    status["last_operation"] = logs
            except:
                pass
        
        return status

    def list_documents(self) -> List[str]:
        """
        Lista todos los document_ids que tienen datos procesados.
        
        Returns:
            Lista de document_ids
        """
        if not self.processed_dir.exists():
            return []
            
        return [d.name for d in self.processed_dir.iterdir() if d.is_dir()]

    def cleanup_document(self, document_id: str, keep_structure: bool = True) -> bool:
        """
        Limpia los datos procesados de un documento.
        
        Args:
            document_id: ID del documento a limpiar
            keep_structure: Si mantener la estructura del documento
            
        Returns:
            True si la limpieza fue exitosa
        """
        try:
            paths = self.get_document_paths(document_id)
            
            # Limpiar chunks
            if paths["chunks_dir"].exists():
                import shutil
                shutil.rmtree(paths["chunks_dir"])
                print(f"🗑️ Cleaned chunks directory: {paths['chunks_dir']}")
            
            # Limpiar final chunks
            if paths["final_chunks_dir"].exists():
                import shutil
                shutil.rmtree(paths["final_chunks_dir"])
                print(f"🗑️ Cleaned final chunks directory: {paths['final_chunks_dir']}")
            
            # Limpiar temp
            if paths["temp_dir"].exists():
                import shutil
                shutil.rmtree(paths["temp_dir"])
                print(f"🗑️ Cleaned temp directory: {paths['temp_dir']}")
            
            # Limpiar estructura si se solicita
            if not keep_structure and paths["structure_file"].exists():
                paths["structure_file"].unlink()
                print(f"🗑️ Removed structure file: {paths['structure_file']}")
            
            # Log de limpieza
            self._log_operation(document_id, "cleanup_document", {
                "keep_structure": keep_structure,
                "cleaned_chunks": True,
                "cleaned_final_chunks": True,
                "cleaned_temp": True,
                "cleaned_structure": not keep_structure
            })
            
            return True
            
        except Exception as e:
            print(f"❌ Error cleaning document {document_id}: {e}")
            return False, metadata # type: ignore

    def save_final_chunks(
        self,
        chunks: List[str],
        metadata: List[Dict],
        document_id: str,
        output_dir: Optional[str] = None
    ) -> None:
        """
        Persist re-chunked results to disk with simple front-matter comments.
        Uses new data structure: data/processed/{document_id}/final_chunks/
        """
        # Usar final_chunks_dir del documento si no se especifica
        if output_dir is None:
            paths = self.get_document_paths(document_id)
            output_path = paths["final_chunks_dir"]
        else:
            output_path = Path(output_dir)
            
        output_path.mkdir(parents=True, exist_ok=True)

        for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
            title = meta.get('title', 'unknown')
            safe_title = (
                title.lower()
                .replace(' ', '_')
                .replace('/', '_')
                .replace('\\', '_')
            )
            safe_title = ''.join(c for c in safe_title if c.isalnum() or c in '_-')
            filename = f"chunk_{i:03d}_{safe_title}.md"
            filepath = output_path / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"<!-- Document ID: {document_id} -->\n")
                f.write(f"<!-- Title: {title} -->\n")
                f.write(f"<!-- Chunk: {i+1}/{len(chunks)} -->\n")
                f.write(f"<!-- Length: {len(chunk)} characters -->\n\n")
                f.write(chunk)

        print(f"💾 Saved {len(chunks)} final chunks to: {output_path}")
        
        # Log de guardado
        self._log_operation(document_id, "save_final_chunks", {
            "output_dir": str(output_path),
            "chunks_saved": len(chunks),
            "total_characters": sum(len(c) for c in chunks)
        })

    # Convenience wrapper for post-indexing step
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
        Wrapper conveniente para re-chunking con nueva estructura de datos.
        
        Args:
            document_id: ID único del documento
            chunks_dir: Directorio de chunks (usa automático si es None)
            target_chunk_size: Tamaño objetivo de chunks
            tolerance: Tolerancia en el tamaño
            save: Si guardar los chunks finales
            save_dir: Directorio donde guardar (usa automático si es None)
        """
        final_chunks, metadata = self.process_chunks_directory(
            document_id=document_id,
            chunks_dir=chunks_dir,
            target_size=target_chunk_size,
            tolerance=tolerance
        )
        
        if save and final_chunks:
            self.save_final_chunks(final_chunks, metadata, document_id, save_dir)
            
        return final_chunks # type: ignore

import os
import json
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
from openai import OpenAI

# Asumiendo que estas constantes están definidas
# EMBEDDING_MODEL, EMBEDDING_DIM


class VectorStore:
    """
    Vector store for document embeddings using new data structure.
    Uses: data/processed/{document_id}/embeddings/
    """
    
    def __init__(self, base_data_dir: Path = Path("data")) -> None:
        self.base_data_dir = Path(base_data_dir)
        self.processed_dir = self.base_data_dir / "processed"
        
        # Crear directorios base
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client once
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY")) # type: ignore

        # Thread-local storage for additional clients if needed
        self._local = threading.local()

    def get_document_paths(self, document_id: str) -> Dict[str, Path]:
        """
        Obtiene los paths de embeddings para un documento específico.
        
        Args:
            document_id: ID único del documento
            
        Returns:
            Diccionario con paths de embeddings del documento
        """
        processed_doc_dir = self.processed_dir / document_id
        embeddings_dir = processed_doc_dir / "embeddings"
        
        return {
            "processed_dir": processed_doc_dir,
            "embeddings_dir": embeddings_dir,
            "embeddings_file": embeddings_dir / "embeddings.npy",
            "metadata_file": embeddings_dir / "metadata.json",
            "faiss_index_file": embeddings_dir / "faiss_index",
            "embeddings_log": embeddings_dir / "embeddings_log.json"
        }

    def _log_operation(self, document_id: str, operation: str, data: Dict[str, Any]) -> None:
        """
        Registra operaciones de embeddings en el log específico del documento.
        
        Args:
            document_id: ID del documento
            operation: Tipo de operación realizada
            data: Datos de la operación
        """
        try:
            paths = self.get_document_paths(document_id)
            log_file = paths["embeddings_log"]
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "document_id": document_id,
                "data": data
            }
            
            # Cargar logs existentes si existen
            logs = []
            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                    if not isinstance(logs, list):
                        logs = [logs]
                except:
                    logs = []
            
            logs.append(log_entry)
            
            # Guardar logs actualizados
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error logging embeddings operation: {e}")

    def get_embedding(self, text: str, client: OpenAI, model: str = EMBEDDING_MODEL) -> List[float]:
        """
        Genera embedding para un texto usando OpenAI.
        Método estático mantenido para compatibilidad.
        """
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding

    def _get_client(self) -> OpenAI:
        """Get a client for the current thread"""
        if not hasattr(self._local, 'client'):
            self._local.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._local.client

    def _process_single_file(self, section_file: Path, document_id: str) -> Tuple[bool, Dict[str, Any], List[float]]:
        """Process a single file and return success status, metadata, and embedding"""
        try:
            with open(section_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Skip empty files
            if not content.strip():
                print(f"⚠️ Skipping empty file: {section_file.name}")
                return False, {}, []

            # Generate embedding using OpenAI with thread-local client
            client = self._get_client()
            response = client.embeddings.create(
                input=content,
                model=EMBEDDING_MODEL
            )
            embedding = response.data[0].embedding
            
            # Create enhanced metadata
            metadata = {
                "document_id": document_id,
                "section_file": section_file.name,
                "file_path": str(section_file),
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "content_length": len(content),
                "embedding_model": EMBEDDING_MODEL,
                "timestamp": datetime.now().isoformat(),
                "content_hash": hash(content.strip())  # Para detectar cambios
            }
            
            return True, metadata, embedding
            
        except Exception as e:
            print(f"❌ Failed to generate embedding for {section_file.name}: {e}")
            return False, {}, []

    def generate_embeddings(
        self, 
        document_id: str, 
        chunks_dir: Optional[Path] = None, 
        max_workers: int = 10,
        source_type: str = "chunks"
    ) -> Dict[str, Any]:
        """
        Generate embeddings for all section chunks using OpenAI with concurrency.
        
        Args:
            document_id: ID único del documento
            chunks_dir: Directorio de chunks (usa automático si es None)
            max_workers: Número máximo de workers concurrentes
            source_type: Tipo de source ("chunks" o "final_chunks")
            
        Returns:
            Diccionario con información sobre embeddings generados
        """
        # Determinar directorio de chunks si no se especifica
        if chunks_dir is None:
            processed_doc_dir = self.processed_dir / document_id
            if source_type == "final_chunks":
                chunks_dir = processed_doc_dir / "final_chunks"
            else:
                chunks_dir = processed_doc_dir / "chunks"
        else:
            chunks_dir = Path(chunks_dir)
        
        if not chunks_dir.exists():
            raise ValueError(f"Chunks directory not found: {chunks_dir}")
            
        section_files = list(chunks_dir.glob("*.md"))
        
        if not section_files: 
            raise ValueError(f"No markdown files found in the specified directory {chunks_dir}")
        
        print(f"🔄 Generating embeddings for {len(section_files)} files from {chunks_dir}")
        
        embeddings = []
        metadata = []

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, section_file, document_id): section_file 
                for section_file in section_files
            }
            
            # Process completed tasks in order of completion
            for future in as_completed(future_to_file):
                section_file = future_to_file[future]
                try:
                    success, file_metadata, embedding = future.result()
                    if success:
                        embeddings.append(embedding)
                        metadata.append(file_metadata)
                        print(f"✅ Generated embedding for: {section_file.name}")
                except Exception as e:
                    print(f"❌ Error processing {section_file.name}: {e}")
                    continue

        # Check if we have valid embeddings
        if not embeddings: 
            raise ValueError("No valid content found to embed")
        
        # Sort metadata and embeddings by section_file name to maintain consistent order
        # This ensures reproducible results across runs
        combined_data = list(zip(metadata, embeddings))
        combined_data.sort(key=lambda x: x[0]["section_file"])
        metadata, embeddings = zip(*combined_data)
        metadata = list(metadata)
        embeddings = list(embeddings)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Get document-specific embeddings paths
        paths = self.get_document_paths(document_id)
        embeddings_dir = paths["embeddings_dir"]
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Save embeddings and metadata 
        np.save(paths["embeddings_file"], embeddings_array)
        
        # Enhanced metadata with generation info
        full_metadata = {
            "document_id": document_id,
            "generation_timestamp": datetime.now().isoformat(),
            "source_directory": str(chunks_dir),
            "source_type": source_type,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIM,
            "total_files_processed": len(metadata),
            "files_metadata": metadata
        }
        
        with open(paths["metadata_file"], "w", encoding="utf-8") as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)

        # Create FAISS index for fast similarity search
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        # Normalize for cosine similarity
        embeddings_normalized = embeddings_array.copy()
        faiss.normalize_L2(embeddings_normalized)
        index.add(embeddings_normalized) # type: ignore

        # Save FAISS index
        faiss.write_index(index, str(paths["faiss_index_file"]))

        result = {
            "document_id": document_id,
            "embeddings_count": len(embeddings),
            "embedding_dim": EMBEDDING_DIM,
            "embeddings_path": str(paths["embeddings_file"]),
            "metadata_path": str(paths["metadata_file"]),
            "faiss_index_path": str(paths["faiss_index_file"]),
            "sections_processed": [m["section_file"] for m in metadata],
            "source_type": source_type,
            "source_directory": str(chunks_dir)
        }
        
        # Log de generación de embeddings
        self._log_operation(document_id, "generate_embeddings", {
            "source_directory": str(chunks_dir),
            "source_type": source_type,
            "files_processed": len(section_files),
            "embeddings_generated": len(embeddings),
            "max_workers": max_workers,
            "embedding_model": EMBEDDING_MODEL,
            "success": True
        })
        
        print(f"✅ Generated embeddings for document {document_id}: {len(embeddings)} vectors")
        return result

    def search_similar(
        self, 
        document_id: str, 
        query: str, 
        top_k: int = 5,
        include_content: bool = False
    ) -> Dict[str, Any]:
        """
        Search within a specific document using semantic similarity.
        
        Args:
            document_id: ID del documento
            query: Query de búsqueda
            top_k: Número de resultados a retornar
            include_content: Si incluir el contenido completo en resultados
            
        Returns:
            Diccionario con resultados de búsqueda
        """
        paths = self.get_document_paths(document_id)
        
        if not paths["embeddings_dir"].exists():
            raise ValueError(f"Document embeddings not found for document_id: {document_id}")
        
        # Load metadata
        with open(paths["metadata_file"], 'r', encoding='utf-8') as f:
            full_metadata = json.load(f)
        
        # Extraer metadata de archivos
        files_metadata = full_metadata.get("files_metadata", [])
        
        if not files_metadata:
            raise ValueError(f"No file metadata found for document: {document_id}")
        
        # Load FAISS index
        index = faiss.read_index(str(paths["faiss_index_file"]))
        
        # Generate query embedding using OpenAI
        try:
            response = self.client.embeddings.create(
                input=query,
                model=EMBEDDING_MODEL
            )
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
            faiss.normalize_L2(query_embedding)
        except Exception as e:
            raise RuntimeError(f"Failed to generate query embedding: {e}")
        
        # Search
        scores, indices = index.search(query_embedding, min(top_k, len(files_metadata)))
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(files_metadata):
                result = files_metadata[idx].copy()
                result["similarity_score"] = float(score)
                result["rank"] = i + 1
                
                # Incluir contenido completo si se solicita
                if include_content:
                    try:
                        file_path = Path(result["file_path"])
                        if file_path.exists():
                            with open(file_path, "r", encoding="utf-8") as f:
                                result["full_content"] = f.read()
                    except Exception as e:
                        result["content_error"] = str(e)
                
                results.append(result)
        
        search_result = {
            "document_id": document_id,
            "query": query,
            "results": results,
            "total_results": len(results),
            "embeddings_metadata": {
                "source_type": full_metadata.get("source_type", "unknown"),
                "embedding_model": full_metadata.get("embedding_model", "unknown"),
                "total_available": len(files_metadata)
            }
        }
        
        # Log de búsqueda
        self._log_operation(document_id, "search_similar", {
            "query": query,
            "top_k": top_k,
            "results_found": len(results),
            "include_content": include_content
        })
        
        return search_result

    def get_embeddings_status(self, document_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado de embeddings para un documento.
        
        Args:
            document_id: ID del documento
            
        Returns:
            Diccionario con estado de embeddings
        """
        paths = self.get_document_paths(document_id)
        
        status = {
            "document_id": document_id,
            "exists": {
                "embeddings_dir": paths["embeddings_dir"].exists(),
                "embeddings_file": paths["embeddings_file"].exists(),
                "metadata_file": paths["metadata_file"].exists(),
                "faiss_index": paths["faiss_index_file"].exists(),
                "log_file": paths["embeddings_log"].exists()
            },
            "metadata": None,
            "last_operation": None
        }
        
        # Cargar metadata si existe
        if paths["metadata_file"].exists():
            try:
                with open(paths["metadata_file"], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                status["metadata"] = {
                    "generation_timestamp": metadata.get("generation_timestamp"),
                    "source_type": metadata.get("source_type"),
                    "embedding_model": metadata.get("embedding_model"),
                    "total_files_processed": metadata.get("total_files_processed"),
                    "embedding_dimension": metadata.get("embedding_dimension")
                }
            except:
                pass
        
        # Cargar última operación del log
        if paths["embeddings_log"].exists():
            try:
                with open(paths["embeddings_log"], "r", encoding="utf-8") as f:
                    logs = json.load(f)
                if isinstance(logs, list) and logs:
                    status["last_operation"] = logs[-1]
                elif isinstance(logs, dict):
                    status["last_operation"] = logs
            except:
                pass
        
        return status

    def list_documents_with_embeddings(self) -> List[str]:
        """
        Lista todos los document_ids que tienen embeddings generados.
        
        Returns:
            Lista de document_ids con embeddings
        """
        if not self.processed_dir.exists():
            return []
        
        documents_with_embeddings = []
        for doc_dir in self.processed_dir.iterdir():
            if doc_dir.is_dir():
                embeddings_dir = doc_dir / "embeddings"
                if (embeddings_dir.exists() and 
                    (embeddings_dir / "embeddings.npy").exists() and
                    (embeddings_dir / "faiss_index").exists()):
                    documents_with_embeddings.append(doc_dir.name)
        
        return documents_with_embeddings

    def delete_embeddings(self, document_id: str) -> bool:
        """
        Elimina los embeddings de un documento específico.
        
        Args:
            document_id: ID del documento
            
        Returns:
            True si la eliminación fue exitosa
        """
        try:
            paths = self.get_document_paths(document_id)
            
            if paths["embeddings_dir"].exists():
                import shutil
                shutil.rmtree(paths["embeddings_dir"])
                print(f"🗑️ Deleted embeddings for document: {document_id}")
                
                # Log de eliminación
                self._log_operation(document_id, "delete_embeddings", {
                    "deleted_successfully": True
                })
                
                return True
            else:
                print(f"⚠️ No embeddings found for document: {document_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting embeddings for {document_id}: {e}")
            return False

    def update_embeddings(
        self, 
        document_id: str, 
        chunks_dir: Optional[Path] = None,
        source_type: str = "chunks",
        max_workers: int = 10
    ) -> Dict[str, Any]:
        """
        Actualiza embeddings existentes, regenerándolos si es necesario.
        
        Args:
            document_id: ID del documento
            chunks_dir: Directorio de chunks
            source_type: Tipo de source
            max_workers: Workers concurrentes
            
        Returns:
            Diccionario con información de actualización
        """
        print(f"🔄 Updating embeddings for document: {document_id}")
        
        # Eliminar embeddings existentes
        self.delete_embeddings(document_id)
        
        # Generar nuevos embeddings
        result = self.generate_embeddings(
            document_id=document_id,
            chunks_dir=chunks_dir,
            max_workers=max_workers,
            source_type=source_type
        )
        
        # Log de actualización
        self._log_operation(document_id, "update_embeddings", {
            "source_type": source_type,
            "embeddings_regenerated": result["embeddings_count"],
            "success": True
        })
        
        print(f"✅ Updated embeddings for document {document_id}")
        return result
    
    def natural_sort_key(metadata_item): # type: ignore
        """Sort key that handles numbers in filenames naturally."""
        section_file = metadata_item.get("section_file", "") # type: ignore
        parts = re.split(r'(\d+)', section_file)
        return [int(part) if part.isdigit() else part.lower() for part in parts]
    
    def extract_title_from_content(content): # type: ignore
        """Extract title from markdown content."""
        for line in content.splitlines(): # type: ignore
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return None

    # ---------- Legacy Interface Compatibility ----------
    def generate_embeddings_legacy(
        self, 
        document_id: str, 
        chunks_dir: Path, 
        max_workers: int = 10
    ) -> Dict[str, Any]:
        """
        Legacy interface wrapper para mantener compatibilidad.
        """
        return self.generate_embeddings(
            document_id=document_id,
            chunks_dir=chunks_dir,
            max_workers=max_workers,
            source_type="chunks"
        )
    

def test_document_indexing_pipeline(pdf_path: str, markdown_path: str, output_dir: str = "test_output"):
    """
    Simple test function to run the DocumentIndexingPipeline end-to-end.
    
    Args:
        pdf_path: Path to the original PDF file
        markdown_path: Path to the markdown file (converted from PDF)
        output_dir: Directory where chunked files will be saved (default: "test_output")
    
    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    try:
        # Initialize the pipeline
        print("🚀 Initializing DocumentIndexingPipeline...")
        pipeline = DocumentIndexingPipeline(
            output_dir=output_dir, # type: ignore
            max_preview_pages=10
        )
        
        # Check if files exist
        from pathlib import Path
        if not Path(pdf_path).exists():
            print(f"❌ PDF file not found: {pdf_path}")
            return False
            
        if not Path(markdown_path).exists():
            print(f"❌ Markdown file not found: {markdown_path}")
            return False
        
        print(f"✅ Input files found:")
        print(f"   📄 PDF: {pdf_path}")
        print(f"   📝 Markdown: {markdown_path}")
        
        # Run the main pipeline
        print("\n🔍 Running index and chunking pipeline...")
        doc_structure = pipeline.run_index_and_chunking_pipeline(pdf_path, markdown_path) # type: ignore
        
        if doc_structure:
            print("✅ Pipeline completed successfully!")
            print(f"\n📊 Document Structure Summary:")
            print(f"   - Total sections: {len(doc_structure.sections)}")
            
            for i, section in enumerate(doc_structure.sections, 1):
                sub_count = len(section.sub_sections) if section.sub_sections else 0
                print(f"   {i}. {section.title} ({sub_count} subsections)")
            
            print(f"\n📁 Output directory: {output_dir}")
            
            # Optional: Test re-chunking if you want
            print("\n🔄 Testing re-chunking functionality...")
            chunks_dir = Path(output_dir) / Path(markdown_path).stem
            if chunks_dir.exists():
                final_chunks, metadata = pipeline.rechunk_directory(
                    chunks_dir=str(chunks_dir),
                    target_chunk_size=1000,
                    tolerance=150,
                    save=True,
                    save_dir=f"{output_dir}/final_chunks"
                ) # type: ignore
                print(f"✅ Re-chunking complete! Generated {len(final_chunks)} final chunks")
            else:
                print("⚠️ Chunks directory not found, skipping re-chunking test")
            
            return True
            
        else:
            print("❌ Pipeline failed to generate document structure")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Example usage:
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    pdf_file = "data/Senior Facilities Agreement [Execution version](212684405_1)/Senior Facilities Agreement [Execution version](212684405_1).pdf"
    markdown_file = "data/Senior Facilities Agreement [Execution version](212684405_1)/Senior Facilities Agreement [Execution version](212684405_1).md"
    
    # Run the test
    success = test_document_indexing_pipeline(pdf_file, markdown_file)
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed - check the error messages above")