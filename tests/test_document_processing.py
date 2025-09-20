"""
Comprehensive tests for document processing functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil

# Import modules to test
from datasets.document_processor import (
    DocumentProcessor,
    DocumentValidator,
    DocumentContent,
    DocumentMetadata,
    process_document,
    validate_document,
    get_supported_formats,
)
from datasets.document_structure import (
    extract_document_structure,
    TextAnalyzer,
    PDFStructureExtractor,
    DOCXStructureExtractor,
    HTMLStructureExtractor,
    DocumentStructure,
    HeaderInfo,
    TableData,
)


class TestDocumentValidator:
    """Test document validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DocumentValidator()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_test_file(
        self, filename: str, content: str = "test content", size: int = None
    ) -> str:
        """Create a test file with specified content."""
        file_path = os.path.join(self.temp_dir, filename)

        if size is not None:
            # Create file with specific size
            with open(file_path, "w") as f:
                f.write("x" * size)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        return file_path

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        is_valid, errors = self.validator.validate_document("/nonexistent/file.pdf")
        assert not is_valid
        assert "File does not exist" in errors

    def test_validate_empty_file(self):
        """Test validation of empty file."""
        file_path = self.create_test_file("empty.txt", "", 0)
        is_valid, errors = self.validator.validate_document(file_path)
        assert not is_valid
        assert "File is empty" in errors

    def test_validate_oversized_file(self):
        """Test validation of oversized file."""
        # Create a file larger than the limit
        file_path = self.create_test_file(
            "large.txt", size=self.validator.MAX_FILE_SIZE + 1
        )
        is_valid, errors = self.validator.validate_document(file_path)
        assert not is_valid
        assert "exceeds maximum" in str(errors)

    def test_validate_unsupported_format(self):
        """Test validation of unsupported file format."""
        file_path = self.create_test_file("test.xyz", "content")
        is_valid, errors = self.validator.validate_document(file_path)
        assert not is_valid
        assert "Unsupported file format" in str(errors)

    def test_validate_valid_text_file(self):
        """Test validation of valid text file."""
        file_path = self.create_test_file("valid.txt", "This is valid content.")
        is_valid, errors = self.validator.validate_document(file_path)
        assert is_valid
        assert not errors

    def test_validate_valid_markdown_file(self):
        """Test validation of valid markdown file."""
        content = "# Header\n\nThis is markdown content."
        file_path = self.create_test_file("valid.md", content)
        is_valid, errors = self.validator.validate_document(file_path)
        assert is_valid
        assert not errors


class TestDocumentProcessor:
    """Test document processing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_test_file(self, filename: str, content: str) -> str:
        """Create a test file with specified content."""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    def test_process_invalid_document(self):
        """Test processing invalid document."""
        result = self.processor.process_document("/nonexistent/file.pdf")
        assert result.text == ""
        assert result.processing_errors
        assert "File does not exist" in result.processing_errors

    def test_process_text_document(self):
        """Test processing plain text document."""
        content = "This is a test document.\n\nWith multiple paragraphs."
        file_path = self.create_test_file("test.txt", content)

        result = self.processor.process_document(file_path)
        assert result.text == content.strip()
        assert result.metadata.format_type == "Text"
        assert result.metadata.word_count == 8
        assert result.metadata.character_count == len(content.strip())
        assert not result.processing_errors

    def test_process_html_document(self):
        """Test processing HTML document."""
        content = """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Main Header</h1>
<p>This is a paragraph.</p>
</body>
</html>"""
        file_path = self.create_test_file("test.html", content)

        result = self.processor.process_document(file_path)

        # Should extract text properly
        assert isinstance(result, DocumentContent)
        assert result.metadata.format_type == "HTML"
        # BeautifulSoup should extract text content
        assert "Main Header" in result.text or not result.processing_errors

    def test_encoding_detection(self):
        """Test handling of different text encodings."""
        # Create file with special characters
        content = "Café résumé naïve"
        file_path = os.path.join(self.temp_dir, "encoding_test.txt")

        # Write with UTF-8
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        result = self.processor.process_document(file_path)
        assert result.text == content
        assert result.metadata.encoding == "utf-8"
        assert not result.processing_errors


class TestTextAnalyzer:
    """Test text analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TextAnalyzer()

    def test_extract_headers(self):
        """Test header extraction from text."""
        text = """INTRODUCTION

This is some content.

1. First Section

More content here.

1.1. Subsection

Subsection content.

CONCLUSION

Final thoughts."""

        headers = self.analyzer.extract_headers(text)
        assert len(headers) >= 2  # At least INTRODUCTION and CONCLUSION

        # Check for hierarchical structure
        for header in headers:
            assert header.level > 0
            assert header.text.strip()

    def test_extract_lists(self):
        """Test list extraction from text."""
        text = """Here are some items:

• First bullet point
• Second bullet point
• Third bullet point

And numbered items:

1. First numbered item
2. Second numbered item
3. Third numbered item"""

        lists = self.analyzer.extract_lists(text)
        assert len(lists) >= 2  # Should find bullet and numbered lists

        bullet_list = next((l for l in lists if l["type"] == "bullet"), None)
        assert bullet_list is not None
        assert len(bullet_list["items"]) == 3

        numbered_list = next((l for l in lists if l["type"] == "numbered"), None)
        assert numbered_list is not None
        assert len(numbered_list["items"]) == 3


class TestConvenienceFunctions:
    """Test convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = get_supported_formats()
        assert isinstance(formats, dict)
        assert ".pdf" in formats
        assert ".docx" in formats
        assert ".txt" in formats
        assert ".html" in formats

    def test_process_document_function(self):
        """Test standalone process_document function."""
        # Create test file
        file_path = os.path.join(self.temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("Test content")

        result = process_document(file_path)
        assert isinstance(result, DocumentContent)
        assert result.text == "Test content"


class TestErrorHandling:
    """Test error handling in document processing."""

    def test_missing_dependencies_graceful_degradation(self):
        """Test graceful degradation when dependencies are missing."""
        # Test with a mock file that will trigger validation errors
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("fake pdf content")
            temp_path = f.name

        try:
            processor = DocumentProcessor()
            result = processor.process_document(temp_path)
            # Should handle gracefully with validation or processing errors
            assert result.processing_errors or result.text == ""
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
