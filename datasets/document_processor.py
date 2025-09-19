"""
Document processing utilities for Vega 2.0 multi-modal learning.

This module provides:
- Document format support (PDF, DOCX, TXT, RTF, HTML)
- Text extraction with formatting preservation
- Document validation and sanitization
- Metadata extraction and analysis
- Document structure analysis
"""

import os
import tempfile
import shutil
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import json
import mimetypes
from datetime import datetime

# Document processing imports
try:
    import PyPDF2
    import pypdf

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning(
        "PDF libraries not available. Install with: pip install PyPDF2 pypdf"
    )

try:
    from docx import Document as DocxDocument

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logging.warning("python-docx not available. Install with: pip install python-docx")

try:
    from bs4 import BeautifulSoup

    HTML_AVAILABLE = True
except ImportError:
    HTML_AVAILABLE = False
    logging.warning(
        "BeautifulSoup not available. Install with: pip install beautifulsoup4"
    )

try:
    import chardet

    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logging.warning("chardet not available. Install with: pip install chardet")

try:
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logging.warning(
        "python-magic not available. Install with: pip install python-magic"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".pdf", ".docx", ".txt", ".rtf", ".html", ".htm", ".md"}


@dataclass
class DocumentMetadata:
    """Document metadata and properties."""

    file_path: str
    file_size: int
    format: str
    encoding: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    language: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class DocumentContent:
    """Extracted document content with structure."""

    text: str
    metadata: DocumentMetadata
    paragraphs: List[str]
    headers: List[Dict[str, Any]]
    tables: List[List[List[str]]]
    images: List[Dict[str, Any]]
    links: List[Dict[str, str]]
    structure: Dict[str, Any]
    extraction_time: float
    extraction_method: str


class DocumentValidator:
    """Validate and sanitize document files."""

    def __init__(self):
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_extensions = SUPPORTED_FORMATS

    def validate_document(self, file_path: str) -> Dict[str, Any]:
        """Validate document file and return validation results."""
        result = {"is_valid": False, "errors": [], "warnings": [], "file_info": {}}

        try:
            if not os.path.exists(file_path):
                result["errors"].append("File does not exist")
                return result

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                result["errors"].append(
                    f"File too large: {file_size} bytes (max: {self.max_file_size})"
                )
                return result

            # Check extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.allowed_extensions:
                result["errors"].append(f"Unsupported file format: {file_ext}")
                return result

            # Check mime type if magic is available
            if MAGIC_AVAILABLE:
                try:
                    mime_type = magic.from_file(file_path, mime=True)
                    result["file_info"]["mime_type"] = mime_type

                    # Validate mime type matches extension
                    expected_mimes = {
                        ".pdf": "application/pdf",
                        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        ".txt": "text/plain",
                        ".html": "text/html",
                        ".htm": "text/html",
                    }

                    if file_ext in expected_mimes:
                        if mime_type != expected_mimes[file_ext]:
                            result["warnings"].append(
                                f"Mime type mismatch: {mime_type} (expected: {expected_mimes[file_ext]})"
                            )

                except Exception as e:
                    result["warnings"].append(f"Could not determine mime type: {e}")

            # Basic file info
            result["file_info"]["size"] = file_size
            result["file_info"]["extension"] = file_ext
            result["file_info"]["path"] = file_path

            result["is_valid"] = len(result["errors"]) == 0

        except Exception as e:
            result["errors"].append(f"Validation error: {e}")

        return result


class PDFProcessor:
    """Process PDF documents."""

    def __init__(self):
        if not PDF_AVAILABLE:
            raise ImportError("PDF processing libraries not available")

    def extract_text(self, file_path: str) -> Tuple[str, DocumentMetadata]:
        """Extract text and metadata from PDF."""
        try:
            text_content = ""
            metadata_dict = {}

            # Try pypdf first (newer), then PyPDF2
            try:
                with open(file_path, "rb") as file:
                    reader = pypdf.PdfReader(file)

                    # Extract text
                    for page in reader.pages:
                        text_content += page.extract_text() + "\n\n"

                    # Extract metadata
                    if reader.metadata:
                        metadata_dict = {
                            "title": reader.metadata.get("/Title"),
                            "author": reader.metadata.get("/Author"),
                            "created_date": reader.metadata.get("/CreationDate"),
                            "modified_date": reader.metadata.get("/ModDate"),
                        }

                    page_count = len(reader.pages)

            except Exception:
                # Fallback to PyPDF2
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)

                    for page in reader.pages:
                        text_content += page.extract_text() + "\n\n"

                    if reader.metadata:
                        metadata_dict = {
                            "title": reader.metadata.get("/Title"),
                            "author": reader.metadata.get("/Author"),
                            "created_date": reader.metadata.get("/CreationDate"),
                            "modified_date": reader.metadata.get("/ModDate"),
                        }

                    page_count = len(reader.pages)

            # Create metadata
            file_size = os.path.getsize(file_path)
            metadata = DocumentMetadata(
                file_path=file_path,
                file_size=file_size,
                format="pdf",
                title=metadata_dict.get("title"),
                author=metadata_dict.get("author"),
                created_date=metadata_dict.get("created_date"),
                modified_date=metadata_dict.get("modified_date"),
                page_count=page_count,
                word_count=len(text_content.split()) if text_content else 0,
                character_count=len(text_content) if text_content else 0,
            )

            return text_content.strip(), metadata

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise


class DOCXProcessor:
    """Process DOCX documents."""

    def __init__(self):
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not available")

    def extract_text(self, file_path: str) -> Tuple[str, DocumentMetadata]:
        """Extract text and metadata from DOCX."""
        try:
            doc = DocxDocument(file_path)

            # Extract text
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)

            text_content = "\n\n".join(paragraphs)

            # Extract metadata
            props = doc.core_properties

            file_size = os.path.getsize(file_path)
            metadata = DocumentMetadata(
                file_path=file_path,
                file_size=file_size,
                format="docx",
                title=props.title,
                author=props.author,
                created_date=props.created.isoformat() if props.created else None,
                modified_date=props.modified.isoformat() if props.modified else None,
                word_count=len(text_content.split()) if text_content else 0,
                character_count=len(text_content) if text_content else 0,
            )

            return text_content, metadata

        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise


class TextProcessor:
    """Process plain text documents."""

    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        if CHARDET_AVAILABLE:
            try:
                with open(file_path, "rb") as file:
                    raw_data = file.read()
                    result = chardet.detect(raw_data)
                    return result.get("encoding", "utf-8")
            except Exception:
                pass
        return "utf-8"

    def extract_text(self, file_path: str) -> Tuple[str, DocumentMetadata]:
        """Extract text from plain text files."""
        try:
            encoding = self.detect_encoding(file_path)

            with open(file_path, "r", encoding=encoding) as file:
                text_content = file.read()

            file_size = os.path.getsize(file_path)
            file_ext = Path(file_path).suffix.lower()

            metadata = DocumentMetadata(
                file_path=file_path,
                file_size=file_size,
                format=file_ext[1:] if file_ext else "txt",
                encoding=encoding,
                word_count=len(text_content.split()) if text_content else 0,
                character_count=len(text_content) if text_content else 0,
            )

            return text_content, metadata

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise


class HTMLProcessor:
    """Process HTML documents."""

    def __init__(self):
        if not HTML_AVAILABLE:
            raise ImportError("BeautifulSoup not available")

    def extract_text(self, file_path: str) -> Tuple[str, DocumentMetadata]:
        """Extract text from HTML files."""
        try:
            # Detect encoding
            encoding = "utf-8"
            if CHARDET_AVAILABLE:
                try:
                    with open(file_path, "rb") as file:
                        raw_data = file.read()
                        result = chardet.detect(raw_data)
                        encoding = result.get("encoding", "utf-8")
                except Exception:
                    pass

            with open(file_path, "r", encoding=encoding) as file:
                html_content = file.read()

            soup = BeautifulSoup(html_content, "html.parser")

            # Extract title
            title = soup.title.string if soup.title else None

            # Extract text (remove scripts and styles)
            for script in soup(["script", "style"]):
                script.decompose()

            text_content = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = "\n".join(chunk for chunk in chunks if chunk)

            file_size = os.path.getsize(file_path)

            metadata = DocumentMetadata(
                file_path=file_path,
                file_size=file_size,
                format="html",
                encoding=encoding,
                title=title,
                word_count=len(text_content.split()) if text_content else 0,
                character_count=len(text_content) if text_content else 0,
            )

            return text_content, metadata

        except Exception as e:
            logger.error(f"HTML extraction failed: {e}")
            raise


class DocumentProcessor:
    """Main document processing pipeline."""

    def __init__(self):
        self.validator = DocumentValidator()
        self.processors = {}

        # Initialize available processors
        if PDF_AVAILABLE:
            self.processors["pdf"] = PDFProcessor()
        if DOCX_AVAILABLE:
            self.processors["docx"] = DOCXProcessor()

        self.processors["txt"] = TextProcessor()
        self.processors["md"] = TextProcessor()
        self.processors["rtf"] = TextProcessor()

        if HTML_AVAILABLE:
            self.processors["html"] = HTMLProcessor()
            self.processors["htm"] = HTMLProcessor()

    def process_document(self, file_path: str) -> DocumentContent:
        """Process a document and extract content."""
        import time

        start_time = time.time()

        try:
            # Validate document
            validation = self.validator.validate_document(file_path)
            if not validation["is_valid"]:
                raise ValueError(f"Document validation failed: {validation['errors']}")

            # Determine format
            file_ext = Path(file_path).suffix.lower()[1:]  # Remove the dot

            if file_ext not in self.processors:
                raise ValueError(f"No processor available for format: {file_ext}")

            # Extract content
            processor = self.processors[file_ext]
            text_content, metadata = processor.extract_text(file_path)

            # Basic structure analysis
            paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]

            # Simple header detection (lines that are short and followed by longer content)
            headers = []
            lines = text_content.split("\n")
            for i, line in enumerate(lines):
                line = line.strip()
                if (
                    len(line) < 100
                    and len(line) > 5
                    and i < len(lines) - 1
                    and len(lines[i + 1].strip()) > len(line)
                ):
                    headers.append(
                        {
                            "text": line,
                            "level": 1,  # Simple implementation
                            "line_number": i + 1,
                        }
                    )

            processing_time = time.time() - start_time

            return DocumentContent(
                text=text_content,
                metadata=metadata,
                paragraphs=paragraphs,
                headers=headers,
                tables=[],  # TODO: Implement table extraction
                images=[],  # TODO: Implement image extraction
                links=[],  # TODO: Implement link extraction
                structure={},  # TODO: Implement detailed structure analysis
                extraction_time=processing_time,
                extraction_method=f"{file_ext}_processor",
            )

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise


# Convenience functions
def process_document_file(file_path: str) -> DocumentContent:
    """Process a single document file."""
    processor = DocumentProcessor()
    return processor.process_document(file_path)


def validate_document_file(file_path: str) -> Dict[str, Any]:
    """Validate a document file."""
    validator = DocumentValidator()
    return validator.validate_document(file_path)


def extract_text_from_document(file_path: str) -> str:
    """Simple text extraction from document."""
    content = process_document_file(file_path)
    return content.text


def get_document_metadata(file_path: str) -> DocumentMetadata:
    """Get metadata from document."""
    content = process_document_file(file_path)
    return content.metadata


if __name__ == "__main__":
    # Example usage
    test_file = "sample_document.pdf"
    if os.path.exists(test_file):
        try:
            content = process_document_file(test_file)
            print(f"Extracted {len(content.text)} characters")
            print(f"Found {len(content.paragraphs)} paragraphs")
            print(f"Found {len(content.headers)} headers")
            print(f"Processing time: {content.extraction_time:.2f}s")
        except Exception as e:
            print(f"Processing failed: {e}")
    else:
        print("No test file found")
