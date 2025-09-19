"""
Comprehensive tests for the Document Explorer Backend.
Tests all layers: models, repository, service, and API.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add the current directory to path so we can import our module
# In a real project, this would be handled by proper packaging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our classes (assuming they're in the same file or properly imported)
from documentExplorer import (
    DocumentInfo, ChunkInfo, ChunkContent,
    DocumentExplorerError, DocumentNotFoundError, ChunkNotFoundError,
    ProcessedDirectoryNotFoundError,
    FileSystemDocumentRepository, DocumentService, DocumentExplorerAPI
)


class TestDataModels(unittest.TestCase):
    """Test data models and their methods."""
    
    def test_document_info_creation(self):
        """Test DocumentInfo creation and to_dict method."""
        doc_path = Path("/test/path")
        doc_info = DocumentInfo(
            id="test_doc",
            chunks_count=5,
            final_chunks_count=3,
            path=doc_path
        )
        
        self.assertEqual(doc_info.id, "test_doc")
        self.assertEqual(doc_info.chunks_count, 5)
        self.assertEqual(doc_info.final_chunks_count, 3)
        self.assertEqual(doc_info.path, doc_path)
        
        # Test to_dict
        expected_dict = {
            "id": "test_doc",
            "chunks_count": 5,
            "final_chunks_count": 3,
            "path": "/test/path"
        }
        self.assertEqual(doc_info.to_dict(), expected_dict)
    
    def test_chunk_info_creation(self):
        """Test ChunkInfo creation and to_dict method."""
        chunk_path = Path("/test/chunk.md")
        chunk_info = ChunkInfo(
            filename="chunk_001.md",
            title="Test Chunk",
            path=chunk_path,
            size=1024,
            document_id="test_doc",
            chunk_type="chunks"
        )
        
        self.assertEqual(chunk_info.filename, "chunk_001.md")
        self.assertEqual(chunk_info.title, "Test Chunk")
        self.assertEqual(chunk_info.size, 1024)
        
        # Test to_dict
        result_dict = chunk_info.to_dict()
        self.assertEqual(result_dict["filename"], "chunk_001.md")
        self.assertEqual(result_dict["document_id"], "test_doc")
        self.assertEqual(result_dict["chunk_type"], "chunks")
    
    def test_chunk_content_creation(self):
        """Test ChunkContent creation and to_dict method."""
        chunk_info = ChunkInfo("test.md", "Test", Path("/test"), 100, "doc1", "chunks")
        chunk_content = ChunkContent(
            chunk_info=chunk_info,
            content="# Test\nContent with front-matter",
            clean_content="Content with front-matter"
        )
        
        self.assertEqual(chunk_content.chunk_info, chunk_info)
        self.assertEqual(chunk_content.content, "# Test\nContent with front-matter")
        self.assertEqual(chunk_content.clean_content, "Content with front-matter")
        
        # Test to_dict
        result_dict = chunk_content.to_dict()
        self.assertIn("chunk_info", result_dict)
        self.assertIn("content", result_dict)
        self.assertIn("clean_content", result_dict)


class TestFileSystemDocumentRepository(unittest.TestCase):
    """Test the file system repository implementation."""
    
    def setUp(self):
        """Set up test environment with temporary directories and files."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "data"
        self.processed_dir = self.test_data_dir / "processed"
        
        # Create test structure
        self._create_test_structure()
        
        # Initialize repository
        self.repository = FileSystemDocumentRepository(self.test_data_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_structure(self):
        """Create test directory structure with sample files."""
        # Create directories
        doc1_dir = self.processed_dir / "document_1"
        doc1_chunks = doc1_dir / "chunks"
        doc1_final = doc1_dir / "final_chunks"
        
        doc2_dir = self.processed_dir / "document_2"
        doc2_chunks = doc2_dir / "chunks"
        
        doc1_chunks.mkdir(parents=True)
        doc1_final.mkdir(parents=True)
        doc2_chunks.mkdir(parents=True)
        
        # Create test chunk files
        (doc1_chunks / "chunk_001_introduction.md").write_text(
            "<!-- Generated chunk -->\n# Introduction\nThis is the introduction section."
        )
        (doc1_chunks / "chunk_002_methodology.md").write_text(
            "# Methodology\nThis describes the methodology."
        )
        (doc1_chunks / "chunk_010_conclusion.md").write_text(
            "# Conclusion\nThis is the conclusion."
        )
        
        (doc1_final / "final_chunk_001.md").write_text(
            "# Final Introduction\nFinal version of introduction."
        )
        
        (doc2_chunks / "section_001_overview.md").write_text(
            "# Overview\nDocument 2 overview."
        )
    
    def test_get_all_documents(self):
        """Test retrieving all documents."""
        documents = self.repository.get_all_documents()
        
        self.assertEqual(len(documents), 2)
        
        # Check document 1
        doc1 = next((d for d in documents if d.id == "document_1"), None)
        self.assertIsNotNone(doc1)
        self.assertEqual(doc1.chunks_count, 3)
        self.assertEqual(doc1.final_chunks_count, 1)
        
        # Check document 2
        doc2 = next((d for d in documents if d.id == "document_2"), None)
        self.assertIsNotNone(doc2)
        self.assertEqual(doc2.chunks_count, 1)
        self.assertEqual(doc2.final_chunks_count, 0)
    
    def test_get_all_documents_no_processed_dir(self):
        """Test error when processed directory doesn't exist."""
        repo = FileSystemDocumentRepository(Path("/nonexistent"))
        
        with self.assertRaises(ProcessedDirectoryNotFoundError):
            repo.get_all_documents()
    
    def test_get_document_by_id(self):
        """Test retrieving specific document by ID."""
        # Existing document
        doc = self.repository.get_document_by_id("document_1")
        self.assertIsNotNone(doc)
        self.assertEqual(doc.id, "document_1")
        self.assertEqual(doc.chunks_count, 3)
        
        # Non-existing document
        doc = self.repository.get_document_by_id("nonexistent")
        self.assertIsNone(doc)
    
    def test_get_document_chunks(self):
        """Test retrieving chunks for a document."""
        # Test regular chunks
        chunks = self.repository.get_document_chunks("document_1", "chunks")
        self.assertEqual(len(chunks), 3)
        
        # Check natural sorting (chunk_010 should come last)
        filenames = [c.filename for c in chunks]
        expected_order = ["chunk_001_introduction.md", "chunk_002_methodology.md", "chunk_010_conclusion.md"]
        self.assertEqual(filenames, expected_order)
        
        # Check chunk titles
        titles = [c.title for c in chunks]
        self.assertIn("Introduction", titles)
        self.assertIn("Methodology", titles)
        self.assertIn("Conclusion", titles)
        
        # Test final chunks
        final_chunks = self.repository.get_document_chunks("document_1", "final_chunks")
        self.assertEqual(len(final_chunks), 1)
        self.assertEqual(final_chunks[0].title, "Final Introduction")
    
    def test_get_document_chunks_nonexistent(self):
        """Test error when document doesn't exist."""
        with self.assertRaises(DocumentNotFoundError):
            self.repository.get_document_chunks("nonexistent", "chunks")
    
    def test_get_chunk_content(self):
        """Test retrieving chunk content."""
        chunks = self.repository.get_document_chunks("document_1", "chunks")
        intro_chunk = next((c for c in chunks if "introduction" in c.filename), None)
        self.assertIsNotNone(intro_chunk)
        
        chunk_content = self.repository.get_chunk_content(intro_chunk)
        
        self.assertEqual(chunk_content.chunk_info, intro_chunk)
        self.assertIn("Generated chunk", chunk_content.content)
        self.assertNotIn("Generated chunk", chunk_content.clean_content)  # Should be removed
        self.assertIn("Introduction", chunk_content.clean_content)
    
    def test_natural_sort_key(self):
        """Test natural sorting of filenames."""
        repo = self.repository
        
        # Test with various filename patterns
        filenames = [
            Path("chunk_10.md"),
            Path("chunk_2.md"),
            Path("chunk_1.md"),
            Path("section_100.md")
        ]
        
        sorted_files = sorted(filenames, key=repo._natural_sort_key)
        expected_order = ["chunk_1.md", "chunk_2.md", "chunk_10.md", "section_100.md"]
        actual_order = [f.name for f in sorted_files]
        
        self.assertEqual(actual_order, expected_order)
    
    def test_extract_title_from_file(self):
        """Test title extraction from markdown files."""
        repo = self.repository
        
        # Test with actual file (has # header)
        chunks = repo.get_document_chunks("document_1", "chunks")
        intro_chunk = next((c for c in chunks if "introduction" in c.filename), None)
        title = repo._extract_title_from_file(intro_chunk.path)
        self.assertEqual(title, "Introduction")
        
        # Test fallback to filename
        temp_file = Path(self.temp_dir) / "chunk_005_test_section.md"
        temp_file.write_text("No header content")
        title = repo._extract_title_from_file(Path(temp_file))
        self.assertEqual(title, "Test Section")


class TestDocumentService(unittest.TestCase):
    """Test the document service layer."""
    
    def setUp(self):
        """Set up test with mock repository."""
        self.mock_repo = Mock()
        self.service = DocumentService(self.mock_repo)
        
        # Setup mock data
        self.sample_doc = DocumentInfo("doc1", 5, 2, Path("/test"))
        self.sample_chunk = ChunkInfo("chunk1.md", "Test Chunk", Path("/test"), 100, "doc1", "chunks")
        self.sample_content = ChunkContent(self.sample_chunk, "content", "clean content")
    
    def test_get_all_documents(self):
        """Test getting all documents through service."""
        self.mock_repo.get_all_documents.return_value = [self.sample_doc]
        
        result = self.service.get_all_documents()
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "doc1")
        self.mock_repo.get_all_documents.assert_called_once()
    
    def test_get_document(self):
        """Test getting specific document through service."""
        self.mock_repo.get_document_by_id.return_value = self.sample_doc
        
        result = self.service.get_document("doc1")
        
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "doc1")
        self.mock_repo.get_document_by_id.assert_called_once_with("doc1")
        
        # Test non-existent document
        self.mock_repo.get_document_by_id.return_value = None
        result = self.service.get_document("nonexistent")
        self.assertIsNone(result)
    
    def test_get_document_chunks(self):
        """Test getting document chunks through service."""
        self.mock_repo.get_document_chunks.return_value = [self.sample_chunk]
        
        result = self.service.get_document_chunks("doc1", "chunks")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["filename"], "chunk1.md")
        self.mock_repo.get_document_chunks.assert_called_once_with("doc1", "chunks")
    
    def test_get_chunk_content(self):
        """Test getting chunk content through service."""
        self.mock_repo.get_document_chunks.return_value = [self.sample_chunk]
        self.mock_repo.get_chunk_content.return_value = self.sample_content
        
        result = self.service.get_chunk_content("doc1", "chunk1.md", "chunks")
        
        self.assertIsNotNone(result)
        self.assertEqual(result["clean_content"], "clean content")
        
        # Test non-existent chunk
        result = self.service.get_chunk_content("doc1", "nonexistent.md", "chunks")
        self.assertIsNone(result)
    
    def test_search_documents(self):
        """Test document search functionality."""
        docs = [
            DocumentInfo("python_guide", 5, 2, Path("/test1")),
            DocumentInfo("java_tutorial", 3, 1, Path("/test2")),
            DocumentInfo("python_advanced", 8, 3, Path("/test3"))
        ]
        self.mock_repo.get_all_documents.return_value = docs
        
        result = self.service.search_documents("python")
        
        self.assertEqual(len(result), 2)
        ids = [doc["id"] for doc in result]
        self.assertIn("python_guide", ids)
        self.assertIn("python_advanced", ids)
        self.assertNotIn("java_tutorial", ids)
    
    def test_search_chunks(self):
        """Test chunk search functionality."""
        chunks = [
            ChunkInfo("chunk1.md", "Introduction to Python", Path("/test1"), 100, "doc1", "chunks"),
            ChunkInfo("chunk2.md", "Advanced Python Concepts", Path("/test2"), 200, "doc1", "chunks"),
            ChunkInfo("chunk3.md", "Java Basics", Path("/test3"), 150, "doc1", "chunks")
        ]
        self.mock_repo.get_document_chunks.return_value = chunks
        
        result = self.service.search_chunks("doc1", "python", "chunks")
        
        self.assertEqual(len(result), 2)
        titles = [chunk["title"] for chunk in result]
        self.assertIn("Introduction to Python", titles)
        self.assertIn("Advanced Python Concepts", titles)
        self.assertNotIn("Java Basics", titles)


class TestDocumentExplorerAPI(unittest.TestCase):
    """Test the main API class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary test structure
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "data"
        self.processed_dir = self.test_data_dir / "processed"
        
        # Create simple test structure
        doc_dir = self.processed_dir / "test_document"
        chunks_dir = doc_dir / "chunks"
        chunks_dir.mkdir(parents=True)
        
        (chunks_dir / "chunk_001.md").write_text("# Test Chunk\nTest content")
        
        # Initialize API
        self.api = DocumentExplorerAPI(self.test_data_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_list_documents_success(self):
        """Test successful document listing."""
        result = self.api.list_documents()
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["data"][0]["id"], "test_document")
    
    def test_list_documents_no_processed_dir(self):
        """Test document listing when processed directory doesn't exist."""
        api = DocumentExplorerAPI(Path("/nonexistent"))
        result = api.list_documents()
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertEqual(result["data"], [])
    
    def test_get_document_info_success(self):
        """Test successful document info retrieval."""
        result = self.api.get_document_info("test_document")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["id"], "test_document")
        self.assertEqual(result["data"]["chunks_count"], 1)
    
    def test_get_document_info_not_found(self):
        """Test document info for non-existent document."""
        result = self.api.get_document_info("nonexistent")
        
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])
        self.assertIsNone(result["data"])
    
    def test_list_chunks_success(self):
        """Test successful chunk listing."""
        result = self.api.list_chunks("test_document")
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["document_id"], "test_document")
        self.assertEqual(result["chunk_type"], "chunks")
    
    def test_get_chunk_content_success(self):
        """Test successful chunk content retrieval."""
        result = self.api.get_chunk_content("test_document", "chunk_001.md")
        
        self.assertTrue(result["success"])
        self.assertIn("chunk_info", result["data"])
        self.assertIn("content", result["data"])
        self.assertIn("Test Chunk", result["data"]["content"])
    
    def test_get_chunk_content_not_found(self):
        """Test chunk content for non-existent chunk."""
        result = self.api.get_chunk_content("test_document", "nonexistent.md")
        
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])
        self.assertIsNone(result["data"])
    
    def test_search_documents(self):
        """Test document search."""
        result = self.api.search_documents("test")
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["query"], "test")
    
    def test_search_chunks(self):
        """Test chunk search."""
        result = self.api.search_chunks("test_document", "test")
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["document_id"], "test_document")
        self.assertEqual(result["query"], "test")
    
    def test_get_multiple_chunks_content(self):
        """Test retrieving multiple chunks content."""
        # Add another chunk for testing
        chunks_dir = self.processed_dir / "test_document" / "chunks"
        (chunks_dir / "chunk_002.md").write_text("# Second Chunk\nSecond content")
        
        result = self.api.get_multiple_chunks_content(
            "test_document", 
            ["chunk_001.md", "chunk_002.md", "nonexistent.md"]
        )
        
        # Should succeed partially
        self.assertFalse(result["success"])  # Because one chunk is missing
        self.assertEqual(len(result["data"]), 2)  # But two chunks loaded
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["errors"]), 1)  # One error for missing chunk
        self.assertIn("nonexistent.md", result["errors"][0])


class TestIntegration(unittest.TestCase):
    """Integration tests that test the complete workflow."""
    
    def setUp(self):
        """Set up comprehensive test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "data"
        self.processed_dir = self.test_data_dir / "processed"
        
        # Create comprehensive test structure
        self._create_comprehensive_test_structure()
        
        self.api = DocumentExplorerAPI(self.test_data_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_comprehensive_test_structure(self):
        """Create a comprehensive test structure."""
        # Multiple documents with different chunk types
        for doc_id in ["python_guide", "machine_learning_basics"]:
            doc_dir = self.processed_dir / doc_id
            chunks_dir = doc_dir / "chunks" 
            final_chunks_dir = doc_dir / "final_chunks"
            
            chunks_dir.mkdir(parents=True)
            final_chunks_dir.mkdir(parents=True)
            
            # Regular chunks
            for i in range(1, 4):
                chunk_file = chunks_dir / f"chunk_{i:03d}_{doc_id}_section_{i}.md"
                content = f"<!-- Generated -->\n# Section {i} of {doc_id.replace('_', ' ').title()}\nContent for section {i}"
                chunk_file.write_text(content)
            
            # Final chunks
            final_file = final_chunks_dir / f"final_summary.md"
            final_file.write_text(f"# Final Summary\nSummary of {doc_id}")
    
    def test_complete_workflow(self):
        """Test a complete workflow from listing documents to getting content."""
        # 1. List all documents
        docs_result = self.api.list_documents()
        self.assertTrue(docs_result["success"])
        self.assertEqual(len(docs_result["data"]), 2)
        
        # 2. Get specific document
        doc_result = self.api.get_document_info("python_guide")
        self.assertTrue(doc_result["success"])
        self.assertEqual(doc_result["data"]["chunks_count"], 3)
        self.assertEqual(doc_result["data"]["final_chunks_count"], 1)
        
        # 3. List chunks for document
        chunks_result = self.api.list_chunks("python_guide")
        self.assertTrue(chunks_result["success"])
        self.assertEqual(len(chunks_result["data"]), 3)
        
        # 4. Get content of first chunk
        first_chunk_filename = chunks_result["data"][0]["filename"]
        content_result = self.api.get_chunk_content("python_guide", first_chunk_filename)
        self.assertTrue(content_result["success"])
        self.assertIn("Section 1 of Python Guide", content_result["data"]["content"])
        
        # 5. Search functionality
        search_result = self.api.search_documents("python")
        self.assertTrue(search_result["success"])
        self.assertEqual(len(search_result["data"]), 1)
        
        # 6. Get multiple chunks
        all_filenames = [chunk["filename"] for chunk in chunks_result["data"]]
        multi_result = self.api.get_multiple_chunks_content("python_guide", all_filenames)
        self.assertTrue(multi_result["success"])
        self.assertEqual(len(multi_result["data"]), 3)
    
    def test_error_handling_workflow(self):
        """Test error handling in various scenarios."""
        # Non-existent document
        result = self.api.get_document_info("nonexistent")
        self.assertFalse(result["success"])
        
        # Non-existent chunk
        result = self.api.get_chunk_content("python_guide", "nonexistent.md")
        self.assertFalse(result["success"])
        
        # Mixed batch with some errors
        result = self.api.get_multiple_chunks_content(
            "python_guide", 
            ["chunk_001_python_guide_section_1.md", "nonexistent.md"]
        )
        self.assertFalse(result["success"])  # Has errors
        self.assertEqual(len(result["data"]), 1)  # But one chunk succeeded
        self.assertEqual(len(result["errors"]), 1)  # One error


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataModels,
        TestFileSystemDocumentRepository,
        TestDocumentService,
        TestDocumentExplorerAPI,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == "__main__":
    print("Running Document Explorer Backend Tests...")
    print("=" * 60)
    
    result = run_tests()
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print(f"Tests run: {result.testsRun}")
    print("=" * 60)