"""
Text extraction pipeline for Vega 2.0 multi-modal learning.

This module provides:
- Advanced text extraction with formatting preservation
- Content preprocessing and cleaning
- Metadata enrichment and analysis
- Text structure detection and parsing
- Content quality assessment
"""

import re
import os
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from collections import defaultdict

# Import document processor
from .document_processor import DocumentProcessor, DocumentContent, DocumentMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of extracted text with metadata."""
    text: str
    chunk_type: str  # paragraph, header, table, list, etc.
    level: int  # hierarchy level for headers
    position: int  # position in document
    formatting: Dict[str, Any]  # bold, italic, font, etc.
    metadata: Dict[str, Any]  # additional context

@dataclass
class ExtractedContent:
    """Comprehensive extracted content with structure."""
    raw_text: str
    clean_text: str
    chunks: List[TextChunk]
    structure: Dict[str, Any]
    quality_metrics: Dict[str, float]
    preprocessing_steps: List[str]
    extraction_metadata: Dict[str, Any]

class TextCleaner:
    """Clean and preprocess extracted text."""
    
    def __init__(self):
        self.preprocessing_steps = []
        
    def clean_text(self, text: str, preserve_formatting: bool = True) -> str:
        """Clean text while optionally preserving formatting."""
        if not text:
            return ""
        
        self.preprocessing_steps = []
        cleaned = text
        
        # Remove excessive whitespace but preserve paragraph breaks
        if not preserve_formatting:
            cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
            self.preprocessing_steps.append("normalized_line_breaks")
        
        # Remove trailing whitespace from lines
        cleaned = re.sub(r'[ \t]+$', '', cleaned, flags=re.MULTILINE)
        self.preprocessing_steps.append("removed_trailing_whitespace")
        
        # Normalize unicode characters
        cleaned = cleaned.encode('utf-8', errors='ignore').decode('utf-8')
        self.preprocessing_steps.append("normalized_unicode")
        
        # Remove control characters but keep tabs and newlines
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
        self.preprocessing_steps.append("removed_control_chars")
        
        # Fix common OCR errors (optional)
        if not preserve_formatting:
            cleaned = self._fix_ocr_errors(cleaned)
            self.preprocessing_steps.append("fixed_ocr_errors")
        
        # Normalize quotation marks
        cleaned = re.sub(r'["""]', '"', cleaned)
        cleaned = re.sub(r'[''']', "'", cleaned)
        self.preprocessing_steps.append("normalized_quotes")
        
        return cleaned
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors."""
        # Common OCR substitutions
        ocr_fixes = {
            r'\bl\b': 'I',  # lowercase l mistaken for I
            r'\b0\b': 'O',  # zero mistaken for O in some contexts
            r'rn': 'm',     # rn mistaken for m
            r'vv': 'w',     # vv mistaken for w
            r'ii': 'll',    # ii mistaken for ll
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        return text

class StructureAnalyzer:
    """Analyze document structure and hierarchy."""
    
    def __init__(self):
        self.header_patterns = [
            r'^[A-Z][A-Z\s]{2,50}$',  # ALL CAPS headers
            r'^\d+\.\s+[A-Z]',        # Numbered headers
            r'^[IVX]+\.\s+[A-Z]',     # Roman numeral headers
            r'^[A-Z][a-z\s]{5,50}:?$', # Title case headers
        ]
        
    def analyze_structure(self, content: DocumentContent) -> Dict[str, Any]:
        """Analyze document structure and hierarchy."""
        structure = {
            'sections': [],
            'hierarchy': {},
            'toc': [],
            'metadata': {},
            'statistics': {}
        }
        
        # Analyze paragraphs
        structure['statistics']['paragraph_count'] = len(content.paragraphs)
        structure['statistics']['avg_paragraph_length'] = (
            sum(len(p) for p in content.paragraphs) / len(content.paragraphs)
            if content.paragraphs else 0
        )
        
        # Analyze headers
        headers = self._extract_headers(content.text)
        structure['hierarchy'] = self._build_hierarchy(headers)
        structure['toc'] = self._generate_toc(headers)
        
        # Detect sections
        structure['sections'] = self._detect_sections(content.text, headers)
        
        # Document statistics
        structure['statistics'].update({
            'header_count': len(headers),
            'section_count': len(structure['sections']),
            'text_density': len(content.text.replace(' ', '')) / len(content.text) if content.text else 0,
            'readability_score': self._calculate_readability(content.text)
        })
        
        return structure
    
    def _extract_headers(self, text: str) -> List[Dict[str, Any]]:
        """Extract headers from text."""
        headers = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if line matches header patterns
            for pattern in self.header_patterns:
                if re.match(pattern, line):
                    level = self._determine_header_level(line)
                    headers.append({
                        'text': line,
                        'level': level,
                        'line_number': i + 1,
                        'position': text.find(line)
                    })
                    break
        
        return headers
    
    def _determine_header_level(self, text: str) -> int:
        """Determine header hierarchy level."""
        # Simple heuristic based on formatting
        if re.match(r'^[A-Z][A-Z\s]{2,50}$', text):
            return 1  # ALL CAPS - top level
        elif re.match(r'^\d+\.\s+[A-Z]', text):
            return 2  # Numbered
        elif re.match(r'^[IVX]+\.\s+[A-Z]', text):
            return 2  # Roman numerals
        elif len(text) < 30:
            return 3  # Short titles
        else:
            return 4  # Longer headers
    
    def _build_hierarchy(self, headers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build document hierarchy from headers."""
        hierarchy = {'root': []}
        stack = [hierarchy['root']]
        current_level = 0
        
        for header in headers:
            level = header['level']
            
            # Adjust stack for current level
            while len(stack) > level:
                stack.pop()
            while len(stack) < level:
                stack.append([])
            
            # Add header to current level
            header_node = {
                'header': header,
                'children': []
            }
            stack[-1].append(header_node)
            stack.append(header_node['children'])
            current_level = level
        
        return hierarchy
    
    def _generate_toc(self, headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate table of contents."""
        toc = []
        for header in headers:
            toc.append({
                'title': header['text'],
                'level': header['level'],
                'page': None,  # TODO: Add page number detection
                'line': header['line_number']
            })
        return toc
    
    def _detect_sections(self, text: str, headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect document sections."""
        sections = []
        lines = text.split('\n')
        
        if not headers:
            # Single section if no headers
            return [{
                'title': 'Main Content',
                'start_line': 1,
                'end_line': len(lines),
                'content': text,
                'word_count': len(text.split())
            }]
        
        for i, header in enumerate(headers):
            start_line = header['line_number']
            end_line = headers[i + 1]['line_number'] if i + 1 < len(headers) else len(lines)
            
            section_lines = lines[start_line:end_line]
            section_content = '\n'.join(section_lines)
            
            sections.append({
                'title': header['text'],
                'level': header['level'],
                'start_line': start_line,
                'end_line': end_line,
                'content': section_content,
                'word_count': len(section_content.split())
            })
        
        return sections
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate simple readability score."""
        if not text:
            return 0.0
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        avg_chars_per_word = sum(len(word) for word in words) / len(words)
        
        # Simple readability score (inverse of complexity)
        complexity = (avg_words_per_sentence * 0.1) + (avg_chars_per_word * 0.2)
        readability = max(0, 10 - complexity)
        
        return min(10, readability)

class QualityAssessor:
    """Assess text quality and extraction confidence."""
    
    def assess_quality(self, content: DocumentContent, extracted: ExtractedContent) -> Dict[str, float]:
        """Assess text extraction quality."""
        metrics = {}
        
        # Text completeness
        metrics['completeness'] = self._assess_completeness(content, extracted)
        
        # Character distribution quality
        metrics['character_quality'] = self._assess_character_quality(extracted.clean_text)
        
        # Structure preservation
        metrics['structure_preservation'] = self._assess_structure_preservation(content, extracted)
        
        # Content coherence
        metrics['coherence'] = self._assess_coherence(extracted.clean_text)
        
        # Overall quality score
        metrics['overall_quality'] = sum(metrics.values()) / len(metrics)
        
        return metrics
    
    def _assess_completeness(self, original: DocumentContent, extracted: ExtractedContent) -> float:
        """Assess how complete the extraction is."""
        if not original.text or not extracted.clean_text:
            return 0.0
        
        # Compare lengths (simple heuristic)
        ratio = len(extracted.clean_text) / len(original.text)
        return min(1.0, ratio)
    
    def _assess_character_quality(self, text: str) -> float:
        """Assess character distribution quality."""
        if not text:
            return 0.0
        
        # Check for reasonable character distribution
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        digit_chars = sum(1 for c in text if c.isdigit())
        space_chars = sum(1 for c in text if c.isspace())
        punct_chars = sum(1 for c in text if c in '.,;:!?"\'()-')
        
        # Calculate ratios
        alpha_ratio = alpha_chars / total_chars
        space_ratio = space_chars / total_chars
        
        # Good text should be mostly alphabetic with reasonable spacing
        quality = 0.0
        if 0.6 <= alpha_ratio <= 0.9:  # 60-90% letters
            quality += 0.4
        if 0.1 <= space_ratio <= 0.2:  # 10-20% spaces
            quality += 0.3
        if (digit_chars + punct_chars) / total_chars <= 0.2:  # Not too many numbers/punct
            quality += 0.3
        
        return quality
    
    def _assess_structure_preservation(self, original: DocumentContent, extracted: ExtractedContent) -> float:
        """Assess how well structure is preserved."""
        # Compare paragraph counts
        orig_paragraphs = len(original.paragraphs)
        extracted_chunks = len([c for c in extracted.chunks if c.chunk_type == 'paragraph'])
        
        if orig_paragraphs == 0:
            return 1.0 if extracted_chunks == 0 else 0.5
        
        ratio = extracted_chunks / orig_paragraphs
        return min(1.0, ratio)
    
    def _assess_coherence(self, text: str) -> float:
        """Assess text coherence."""
        if not text:
            return 0.0
        
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) < 2:
            return 0.5
        
        # Check for reasonable line lengths
        avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        
        # Good coherence if lines are reasonable length
        if 20 <= avg_line_length <= 200:
            return 0.8
        elif 10 <= avg_line_length <= 300:
            return 0.6
        else:
            return 0.3

class TextExtractionPipeline:
    """Main text extraction pipeline."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.text_cleaner = TextCleaner()
        self.structure_analyzer = StructureAnalyzer()
        self.quality_assessor = QualityAssessor()
        
    def extract_text(self, file_path: str, preserve_formatting: bool = True, 
                    clean_text: bool = True) -> ExtractedContent:
        """Extract text with full pipeline processing."""
        try:
            # Process document
            document_content = self.document_processor.process_document(file_path)
            
            # Clean text
            raw_text = document_content.text
            if clean_text:
                cleaned_text = self.text_cleaner.clean_text(raw_text, preserve_formatting)
                preprocessing_steps = self.text_cleaner.preprocessing_steps
            else:
                cleaned_text = raw_text
                preprocessing_steps = []
            
            # Create text chunks
            chunks = self._create_text_chunks(document_content, cleaned_text)
            
            # Analyze structure
            structure = self.structure_analyzer.analyze_structure(document_content)
            
            # Create extracted content
            extracted = ExtractedContent(
                raw_text=raw_text,
                clean_text=cleaned_text,
                chunks=chunks,
                structure=structure,
                quality_metrics={},
                preprocessing_steps=preprocessing_steps,
                extraction_metadata={
                    'file_path': file_path,
                    'format': document_content.metadata.format,
                    'extraction_time': document_content.extraction_time,
                    'method': document_content.extraction_method
                }
            )
            
            # Assess quality
            extracted.quality_metrics = self.quality_assessor.assess_quality(
                document_content, extracted
            )
            
            return extracted
            
        except Exception as e:
            logger.error(f"Text extraction pipeline failed: {e}")
            raise
    
    def _create_text_chunks(self, document_content: DocumentContent, 
                          cleaned_text: str) -> List[TextChunk]:
        """Create structured text chunks."""
        chunks = []
        
        # Create paragraph chunks
        for i, paragraph in enumerate(document_content.paragraphs):
            if paragraph.strip():
                chunks.append(TextChunk(
                    text=paragraph,
                    chunk_type='paragraph',
                    level=0,
                    position=i,
                    formatting={},
                    metadata={'source': 'paragraph_extraction'}
                ))
        
        # Create header chunks
        for i, header in enumerate(document_content.headers):
            chunks.append(TextChunk(
                text=header['text'],
                chunk_type='header',
                level=header.get('level', 1),
                position=header.get('line_number', 0),
                formatting={},
                metadata={'source': 'header_extraction'}
            ))
        
        # Sort chunks by position
        chunks.sort(key=lambda x: x.position)
        
        return chunks

# Convenience functions
def extract_text_from_file(file_path: str, clean: bool = True, 
                          preserve_formatting: bool = True) -> str:
    """Simple text extraction."""
    pipeline = TextExtractionPipeline()
    result = pipeline.extract_text(file_path, preserve_formatting, clean)
    return result.clean_text

def extract_structured_content(file_path: str) -> ExtractedContent:
    """Extract structured content with full pipeline."""
    pipeline = TextExtractionPipeline()
    return pipeline.extract_text(file_path)

def assess_extraction_quality(file_path: str) -> Dict[str, float]:
    """Assess extraction quality for a file."""
    pipeline = TextExtractionPipeline()
    result = pipeline.extract_text(file_path)
    return result.quality_metrics

if __name__ == "__main__":
    # Example usage
    test_file = "sample_document.pdf"
    if os.path.exists(test_file):
        try:
            result = extract_structured_content(test_file)
            print(f"Extracted {len(result.clean_text)} characters")
            print(f"Quality score: {result.quality_metrics.get('overall_quality', 0):.2f}")
            print(f"Found {len(result.chunks)} chunks")
            print(f"Structure sections: {len(result.structure.get('sections', []))}")
        except Exception as e:
            print(f"Extraction failed: {e}")
    else:
        print("No test file found")