"""
Test fixtures and utilities for document intelligence testing
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import dataclass

# Import modules to test
from src.vega.document.base import (
    BaseDocumentProcessor,
    ProcessingContext,
    ProcessingResult,
    ConfigurableComponent,
    MetricsCollector,
    DocumentIntelligenceError,
    ProcessingError,
    ValidationError,
)


@dataclass
class TestDocument:
    """Test document for testing purposes"""

    content: str
    doc_type: str = "test"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MockDocumentProcessor(BaseDocumentProcessor):
    """Mock document processor for testing base functionality"""

    def __init__(self, config=None, should_fail=False, processing_time=0.1):
        super().__init__(config or {})
        self.should_fail = should_fail
        self.processing_time = processing_time
        self.processed_inputs = []

    async def _async_initialize(self) -> None:
        """Mock initialization"""
        await asyncio.sleep(0.01)  # Simulate initialization time

    async def _process_internal(
        self, input_data: Any, context: ProcessingContext
    ) -> Dict[str, Any]:
        """Mock processing that can be configured to fail"""
        self.processed_inputs.append(input_data)

        if self.should_fail:
            raise ProcessingError("Mock processing error")

        await asyncio.sleep(self.processing_time)

        return {
            "processed": True,
            "input_data": str(input_data),
            "processing_time": self.processing_time,
            "context_id": context.context_id if context else None,
        }


# Test fixtures for various document types
SAMPLE_DOCUMENTS = {
    "contract": """
        SERVICE AGREEMENT
        
        This Service Agreement ("Agreement") is entered into on [Date] between
        Company A ("Provider") and Company B ("Client").
        
        1. SERVICES
        Provider shall provide consulting services as described in Exhibit A.
        
        2. PAYMENT TERMS
        Client shall pay Provider within 30 days of invoice receipt.
        
        3. LIMITATION OF LIABILITY
        Provider's liability shall be limited to the amount paid under this Agreement.
        
        4. TERMINATION
        Either party may terminate this Agreement with 30 days written notice.
    """,
    "technical_doc": """
        # API Documentation
        
        ## Authentication
        
        This API uses API key authentication. Include your API key in the header:
        ```
        X-API-Key: your-api-key-here
        ```
        
        ## Endpoints
        
        ### GET /api/users/{id}
        
        Retrieve user information by ID.
        
        **Parameters:**
        - id (string): User identifier
        
        **Returns:**
        - User object with id, name, email
        
        **Example:**
        ```json
        {
          "id": "123",
          "name": "John Doe",
          "email": "john@example.com"
        }
        ```
    """,
    "research_paper": """
        # Advanced Machine Learning Techniques in Natural Language Processing
        
        ## Abstract
        
        This paper presents novel approaches to natural language processing using
        advanced machine learning techniques including transformer architectures
        and attention mechanisms.
        
        ## Introduction
        
        Natural language processing has seen significant advances with the
        introduction of transformer-based models. These models have achieved
        state-of-the-art results across various NLP tasks.
        
        ## Methodology
        
        We employ a multi-layered transformer architecture with self-attention
        mechanisms to process sequential text data. The model is trained on
        large-scale corpora using supervised learning techniques.
        
        ## Results
        
        Our approach achieves 95.2% accuracy on standard benchmarks,
        outperforming previous state-of-the-art methods by 3.1%.
        
        ## Conclusion
        
        The proposed method demonstrates significant improvements in NLP tasks
        and provides a foundation for future research in the field.
    """,
    "workflow_doc": """
        # Document Processing Workflow
        
        ## Overview
        This workflow describes the steps for processing incoming documents.
        
        ## Workflow Steps
        
        1. **Document Intake**
           - Receive document from client
           - Validate document format and size
           - Assign unique document ID
        
        2. **Initial Classification**
           - Analyze document type
           - Extract metadata
           - Route to appropriate processing queue
        
        3. **Content Analysis**
           - Extract text content
           - Identify key sections and entities
           - Perform sentiment analysis if applicable
        
        4. **Quality Review**
           - Validate extracted content
           - Check for completeness
           - Flag any quality issues
        
        5. **Final Processing**
           - Generate summary report
           - Store processed data
           - Notify stakeholders of completion
        
        ## Error Handling
        
        If any step fails:
        1. Log error details
        2. Notify system administrator
        3. Route document to manual review queue
        4. Update document status
    """,
    "understanding_test": """
        The impact of artificial intelligence on modern business operations
        cannot be overstated. Organizations across industries are leveraging
        AI technologies to automate processes, improve decision-making, and
        enhance customer experiences.
        
        Key benefits include:
        - Increased operational efficiency
        - Reduced human error
        - Enhanced data analysis capabilities
        - Improved customer service through chatbots and virtual assistants
        
        However, challenges remain:
        - Data privacy and security concerns
        - Need for skilled AI professionals
        - Integration with existing systems
        - Ethical considerations around AI decision-making
        
        Companies must carefully balance the benefits and risks when
        implementing AI solutions, ensuring proper governance and
        oversight throughout the process.
    """,
}

# Mock responses for different processors
MOCK_RESPONSES = {
    "understanding": {
        "document_type": "business_analysis",
        "key_topics": ["artificial intelligence", "business operations", "automation"],
        "sentiment": "neutral",
        "complexity_score": 0.7,
        "readability_score": 0.8,
        "entities": [
            {
                "text": "artificial intelligence",
                "label": "TECHNOLOGY",
                "confidence": 0.95,
            },
            {"text": "organizations", "label": "ORG", "confidence": 0.8},
        ],
    },
    "classification": {
        "primary_category": "business_document",
        "subcategories": ["analysis", "technology"],
        "confidence": 0.92,
        "classification_hierarchy": {
            "level_1": "business",
            "level_2": "analysis",
            "level_3": "technology_impact",
        },
    },
    "workflow": {
        "workflow_type": "document_processing",
        "steps_identified": 5,
        "automation_potential": 0.85,
        "estimated_duration": "2-3 hours",
        "complexity_score": 0.6,
    },
    "legal": {
        "document_type": "contract",
        "risk_assessment": {"overall_risk": "medium", "risk_score": 0.4},
        "clauses": [
            {"type": "liability", "confidence": 0.9},
            {"type": "termination", "confidence": 0.85},
        ],
    },
    "technical": {
        "documentation_type": "api_docs",
        "quality_score": 0.88,
        "completeness_score": 0.75,
        "code_examples": 2,
        "api_endpoints": 1,
    },
}


class TestFixtures:
    """Collection of test fixtures for document intelligence testing"""

    @staticmethod
    def get_sample_document(doc_type: str) -> str:
        """Get a sample document by type"""
        return SAMPLE_DOCUMENTS.get(doc_type, "Default test content")

    @staticmethod
    def get_mock_response(processor_type: str) -> Dict[str, Any]:
        """Get a mock response for a processor type"""
        return MOCK_RESPONSES.get(processor_type, {"mock": True})

    @staticmethod
    def create_processing_context(
        context_id: str = "test_context",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessingContext:
        """Create a test processing context"""
        return ProcessingContext(
            context_id=context_id,
            user_id=user_id or "test_user",
            session_id=session_id or "test_session",
            metadata=metadata or {},
        )

    @staticmethod
    def create_temp_file(content: str, suffix: str = ".txt") -> Path:
        """Create a temporary file with specified content"""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)


# Pytest fixtures
@pytest.fixture
def sample_documents():
    """Fixture providing sample documents"""
    return SAMPLE_DOCUMENTS


@pytest.fixture
def mock_responses():
    """Fixture providing mock responses"""
    return MOCK_RESPONSES


@pytest.fixture
def processing_context():
    """Fixture providing a test processing context"""
    return TestFixtures.create_processing_context()


@pytest.fixture
def temp_directory():
    """Fixture providing a temporary directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_processor():
    """Fixture providing a mock document processor"""
    return MockDocumentProcessor()


@pytest.fixture
def failing_processor():
    """Fixture providing a processor that fails"""
    return MockDocumentProcessor(should_fail=True)


@pytest.fixture
def slow_processor():
    """Fixture providing a slow processor for timeout testing"""
    return MockDocumentProcessor(processing_time=2.0)


@pytest.fixture
async def initialized_processor():
    """Fixture providing an initialized processor"""
    processor = MockDocumentProcessor()
    await processor.initialize()
    return processor


@pytest.fixture
def metrics_collector():
    """Fixture providing a metrics collector"""
    return MetricsCollector()


# Performance testing utilities
class PerformanceTestHelper:
    """Helper class for performance testing"""

    @staticmethod
    async def measure_processing_time(processor, input_data, context=None):
        """Measure processing time for a processor"""
        import time

        start_time = time.time()
        result = await processor.process(input_data, context)
        end_time = time.time()

        return {"processing_time": end_time - start_time, "result": result}

    @staticmethod
    async def run_load_test(processor, input_data, num_requests=100, concurrency=10):
        """Run a load test on a processor"""
        import time

        async def single_request():
            context = TestFixtures.create_processing_context()
            start_time = time.time()
            try:
                await processor.process(input_data, context)
                return time.time() - start_time, None
            except Exception as e:
                return time.time() - start_time, str(e)

        # Run requests in batches
        results = []
        for i in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - i)
            batch_tasks = [single_request() for _ in range(batch_size)]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

        # Analyze results
        processing_times = [r[0] for r in results]
        errors = [r[1] for r in results if r[1] is not None]

        return {
            "total_requests": num_requests,
            "successful_requests": num_requests - len(errors),
            "error_count": len(errors),
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "errors": errors[:10],  # First 10 errors
        }


# Test data generators
class TestDataGenerator:
    """Generates test data for various scenarios"""

    @staticmethod
    def generate_large_document(size_kb: int) -> str:
        """Generate a large document of specified size"""
        base_text = "This is a test document with repeated content. "
        target_size = size_kb * 1024
        current_size = 0
        content_parts = []

        while current_size < target_size:
            content_parts.append(base_text)
            current_size += len(base_text)

        return "".join(content_parts)

    @staticmethod
    def generate_malformed_json() -> str:
        """Generate malformed JSON for error testing"""
        return '{"key": "value", "incomplete": '

    @staticmethod
    def generate_special_characters_text() -> str:
        """Generate text with special characters for encoding testing"""
        return "Test with émojis 🚀 and spëcial çharaçters: àáâãäåæçèéêëìíîï"

    @staticmethod
    def generate_multilingual_text() -> str:
        """Generate multilingual text for language detection testing"""
        return """
        English: This is an English sentence.
        Spanish: Esta es una oración en español.
        French: Ceci est une phrase en français.
        German: Das ist ein deutscher Satz.
        中文: 这是一个中文句子。
        """


# Async test utilities
class AsyncTestUtils:
    """Utilities for async testing"""

    @staticmethod
    async def wait_for_condition(condition_func, timeout=5.0, poll_interval=0.1):
        """Wait for a condition to become true"""
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            if (
                await condition_func()
                if asyncio.iscoroutinefunction(condition_func)
                else condition_func()
            ):
                return True
            await asyncio.sleep(poll_interval)

        return False

    @staticmethod
    async def run_with_timeout(coro, timeout=5.0):
        """Run a coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
