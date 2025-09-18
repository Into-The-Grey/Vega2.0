#!/usr/bin/env python3
"""
Web-Based Error Resolution Research
==================================

Searches StackOverflow, GitHub issues, documentation, and other sources
for similar errors and proven solutions. Extracts and adapts fixes 
automatically using web scraping and LLM integration.

Features:
- Multi-source error research (StackOverflow, GitHub, docs)
- Semantic similarity matching for error patterns
- Solution extraction and ranking
- Code adaptation for local context
- Confidence scoring based on community validation
- Caching to avoid repeated searches
"""

import os
import re
import json
import sqlite3
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from urllib.parse import quote_plus, urljoin
import xml.etree.ElementTree as ET

# External dependencies
import httpx
from bs4 import BeautifulSoup

# Local imports
from error_tracker import ErrorRecord
from self_debugger import FixSuggestion, FixStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WebSolution:
    """Represents a solution found on the web"""
    id: str
    source: str  # stackoverflow, github, docs, etc.
    url: str
    title: str
    content: str
    code_snippets: List[str]
    votes: int
    views: int
    accepted: bool
    tags: List[str]
    similarity_score: float
    confidence_score: float
    extracted_at: datetime
    adapted_fix: Optional[FixSuggestion] = None

class ErrorSearchEngine:
    """Searches web sources for error solutions"""
    
    def __init__(self, cache_db: str = "autonomous_debug/web_solutions.db"):
        self.cache_db = cache_db
        self.http_client = None
        self.search_engines = {
            'stackoverflow': StackOverflowSearcher(),
            'github': GitHubSearcher(),
            'documentation': DocumentationSearcher(),
            'general': GeneralWebSearcher()
        }
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize web solutions cache database"""
        try:
            os.makedirs(os.path.dirname(self.cache_db), exist_ok=True)
            
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            # Web solutions cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS web_solutions (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    content TEXT,
                    code_snippets TEXT,
                    votes INTEGER DEFAULT 0,
                    views INTEGER DEFAULT 0,
                    accepted BOOLEAN DEFAULT FALSE,
                    tags TEXT,
                    similarity_score REAL,
                    confidence_score REAL,
                    extracted_at TEXT,
                    search_query TEXT,
                    error_signature TEXT
                )
            """)
            
            # Search queries cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT,
                    source TEXT,
                    results TEXT,
                    created_at TEXT,
                    expires_at TEXT
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_error_signature ON web_solutions (error_signature)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON web_solutions (confidence_score DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON web_solutions (source)")
            
            conn.commit()
            conn.close()
            
            logger.info(f"Web solutions cache initialized: {self.cache_db}")
            
        except Exception as e:
            logger.error(f"Failed to initialize web solutions cache: {e}")
    
    async def search_for_error(self, error: ErrorRecord, max_results: int = 10) -> List[WebSolution]:
        """Search web sources for solutions to an error"""
        solutions = []
        
        try:
            # Generate search queries
            queries = self._generate_search_queries(error)
            
            # Check cache first
            cached_solutions = self._get_cached_solutions(error)
            if cached_solutions:
                logger.info(f"Found {len(cached_solutions)} cached solutions for error")
                return cached_solutions[:max_results]
            
            # Search each source
            async with httpx.AsyncClient(timeout=30.0) as client:
                self.http_client = client
                
                search_tasks = []
                for source_name, searcher in self.search_engines.items():
                    for query in queries:
                        task = searcher.search(client, query, max_results=5)
                        search_tasks.append((source_name, task))
                
                # Execute searches in parallel
                for source_name, task in search_tasks:
                    try:
                        source_solutions = await task
                        
                        # Calculate similarity and confidence scores
                        for solution in source_solutions:
                            solution.similarity_score = self._calculate_similarity(error, solution)
                            solution.confidence_score = self._calculate_confidence(solution)
                        
                        solutions.extend(source_solutions)
                        
                    except Exception as e:
                        logger.error(f"Search failed for {source_name}: {e}")
            
            # Sort by combined score
            solutions.sort(key=lambda s: s.confidence_score * s.similarity_score, reverse=True)
            
            # Cache results
            self._cache_solutions(error, solutions)
            
            logger.info(f"Found {len(solutions)} solutions for error {error.id[:8]}")
            return solutions[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to search for error solutions: {e}")
            return []
    
    def _generate_search_queries(self, error: ErrorRecord) -> List[str]:
        """Generate search queries for the error"""
        queries = []
        
        # Basic error message query
        if error.message:
            clean_message = re.sub(r'[^\w\s]', ' ', error.message)
            clean_message = re.sub(r'\s+', ' ', clean_message).strip()
            queries.append(f'"{error.error_type}" "{clean_message}"')
        
        # Error type + context
        if error.error_type:
            queries.append(f"{error.error_type} python")
            if error.file_path and error.file_path.endswith('.py'):
                queries.append(f"{error.error_type} python error fix")
        
        # Specific patterns from traceback
        if error.full_traceback:
            # Extract module names from traceback
            modules = re.findall(r'File ".*?([^/\\]+\.py)"', error.full_traceback)
            for module in modules[:2]:  # Only first 2 modules
                module_name = module.replace('.py', '')
                queries.append(f"{error.error_type} {module_name} python")
        
        # Add language context
        queries = [f"{q} python" if 'python' not in q.lower() else q for q in queries]
        
        return queries[:5]  # Limit to 5 queries
    
    def _calculate_similarity(self, error: ErrorRecord, solution: WebSolution) -> float:
        """Calculate similarity between error and solution"""
        score = 0.0
        
        # Error type match
        if error.error_type.lower() in solution.title.lower():
            score += 0.3
        if error.error_type.lower() in solution.content.lower():
            score += 0.2
        
        # Message similarity
        if error.message:
            error_words = set(error.message.lower().split())
            solution_words = set(solution.content.lower().split())
            word_overlap = len(error_words & solution_words) / len(error_words | solution_words)
            score += word_overlap * 0.3
        
        # Python context
        if 'python' in solution.tags or 'python' in solution.title.lower():
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_confidence(self, solution: WebSolution) -> float:
        """Calculate confidence score for a solution"""
        score = 0.0
        
        # Source credibility
        source_weights = {
            'stackoverflow': 0.3,
            'github': 0.25,
            'documentation': 0.35,
            'general': 0.1
        }
        score += source_weights.get(solution.source, 0.1)
        
        # Community validation
        if solution.accepted:
            score += 0.3
        if solution.votes > 0:
            score += min(solution.votes / 100.0, 0.2)  # Max 0.2 from votes
        if solution.views > 1000:
            score += 0.1
        
        # Code presence
        if solution.code_snippets:
            score += 0.2
        
        return min(score, 1.0)
    
    def _get_cached_solutions(self, error: ErrorRecord) -> List[WebSolution]:
        """Get cached solutions for an error"""
        try:
            conn = sqlite3.connect(self.cache_db)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Generate error signature for cache lookup
            error_signature = self._generate_error_signature(error)
            
            # Check for cached solutions (valid for 24 hours)
            yesterday = (datetime.now() - timedelta(hours=24)).isoformat()
            
            cursor.execute("""
                SELECT * FROM web_solutions 
                WHERE error_signature = ? AND extracted_at > ?
                ORDER BY confidence_score DESC, similarity_score DESC
            """, (error_signature, yesterday))
            
            rows = cursor.fetchall()
            conn.close()
            
            solutions = []
            for row in rows:
                solution = WebSolution(
                    id=row['id'],
                    source=row['source'],
                    url=row['url'],
                    title=row['title'] or "",
                    content=row['content'] or "",
                    code_snippets=json.loads(row['code_snippets'] or '[]'),
                    votes=row['votes'],
                    views=row['views'],
                    accepted=bool(row['accepted']),
                    tags=json.loads(row['tags'] or '[]'),
                    similarity_score=row['similarity_score'],
                    confidence_score=row['confidence_score'],
                    extracted_at=datetime.fromisoformat(row['extracted_at'])
                )
                solutions.append(solution)
            
            return solutions
            
        except Exception as e:
            logger.error(f"Failed to get cached solutions: {e}")
            return []
    
    def _cache_solutions(self, error: ErrorRecord, solutions: List[WebSolution]):
        """Cache solutions for future use"""
        try:
            if not solutions:
                return
            
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            error_signature = self._generate_error_signature(error)
            
            for solution in solutions:
                cursor.execute("""
                    INSERT OR REPLACE INTO web_solutions (
                        id, source, url, title, content, code_snippets,
                        votes, views, accepted, tags, similarity_score,
                        confidence_score, extracted_at, error_signature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    solution.id, solution.source, solution.url, solution.title,
                    solution.content, json.dumps(solution.code_snippets),
                    solution.votes, solution.views, solution.accepted,
                    json.dumps(solution.tags), solution.similarity_score,
                    solution.confidence_score, solution.extracted_at.isoformat(),
                    error_signature
                ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Cached {len(solutions)} solutions for error")
            
        except Exception as e:
            logger.error(f"Failed to cache solutions: {e}")
    
    def _generate_error_signature(self, error: ErrorRecord) -> str:
        """Generate a signature for error caching"""
        content = f"{error.error_type}|{error.message}"
        return hashlib.md5(content.encode()).hexdigest()

class StackOverflowSearcher:
    """Searches StackOverflow for error solutions"""
    
    async def search(self, client: httpx.AsyncClient, query: str, max_results: int = 5) -> List[WebSolution]:
        """Search StackOverflow for solutions"""
        solutions = []
        
        try:
            # Use StackOverflow API
            api_url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                'order': 'desc',
                'sort': 'relevance',
                'q': query,
                'accepted': 'True',
                'site': 'stackoverflow',
                'filter': 'withbody',
                'pagesize': max_results
            }
            
            response = await client.get(api_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('items', []):
                    # Extract code snippets
                    code_snippets = self._extract_code_snippets(item.get('body', ''))
                    
                    solution = WebSolution(
                        id=f"so_{item['question_id']}",
                        source='stackoverflow',
                        url=item['link'],
                        title=item['title'],
                        content=self._clean_html(item.get('body', '')),
                        code_snippets=code_snippets,
                        votes=item.get('score', 0),
                        views=item.get('view_count', 0),
                        accepted=item.get('is_answered', False),
                        tags=item.get('tags', []),
                        similarity_score=0.0,  # Will be calculated later
                        confidence_score=0.0,  # Will be calculated later
                        extracted_at=datetime.now()
                    )
                    solutions.append(solution)
            
            logger.debug(f"Found {len(solutions)} StackOverflow solutions for: {query}")
            
        except Exception as e:
            logger.error(f"StackOverflow search failed: {e}")
        
        return solutions
    
    def _extract_code_snippets(self, html_content: str) -> List[str]:
        """Extract code snippets from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            code_blocks = soup.find_all(['code', 'pre'])
            
            snippets = []
            for block in code_blocks:
                code_text = block.get_text().strip()
                if len(code_text) > 10 and '\n' in code_text:  # Multi-line code
                    snippets.append(code_text)
            
            return snippets
            
        except Exception as e:
            logger.error(f"Failed to extract code snippets: {e}")
            return []
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content to plain text"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text().strip()
        except Exception:
            return html_content

class GitHubSearcher:
    """Searches GitHub issues for error solutions"""
    
    async def search(self, client: httpx.AsyncClient, query: str, max_results: int = 5) -> List[WebSolution]:
        """Search GitHub issues for solutions"""
        solutions = []
        
        try:
            # GitHub API search
            api_url = "https://api.github.com/search/issues"
            params = {
                'q': f"{query} is:issue is:closed",
                'sort': 'interactions',
                'order': 'desc',
                'per_page': max_results
            }
            
            response = await client.get(api_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('items', []):
                    # Extract code snippets from issue body
                    code_snippets = self._extract_code_snippets(item.get('body', ''))
                    
                    solution = WebSolution(
                        id=f"gh_{item['id']}",
                        source='github',
                        url=item['html_url'],
                        title=item['title'],
                        content=item.get('body', '') or "",
                        code_snippets=code_snippets,
                        votes=item.get('reactions', {}).get('total_count', 0),
                        views=0,  # GitHub doesn't provide view counts
                        accepted=item.get('state') == 'closed',
                        tags=item.get('labels', []),
                        similarity_score=0.0,
                        confidence_score=0.0,
                        extracted_at=datetime.now()
                    )
                    solutions.append(solution)
            
            logger.debug(f"Found {len(solutions)} GitHub solutions for: {query}")
            
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
        
        return solutions
    
    def _extract_code_snippets(self, markdown_content: str) -> List[str]:
        """Extract code snippets from Markdown content"""
        try:
            # Find code blocks in markdown
            code_pattern = r'```(?:python)?\n(.*?)\n```'
            matches = re.findall(code_pattern, markdown_content, re.DOTALL)
            
            snippets = []
            for match in matches:
                code = match.strip()
                if len(code) > 10:  # Meaningful code snippets
                    snippets.append(code)
            
            return snippets
            
        except Exception as e:
            logger.error(f"Failed to extract markdown code snippets: {e}")
            return []

class DocumentationSearcher:
    """Searches official documentation for error solutions"""
    
    DOCS_SOURCES = [
        'https://docs.python.org/3/search.html',
        'https://docs.djangoproject.com/en/stable/search/',
        'https://flask.palletsprojects.com/en/2.3.x/search/',
        'https://fastapi.tiangolo.com/search/'
    ]
    
    async def search(self, client: httpx.AsyncClient, query: str, max_results: int = 5) -> List[WebSolution]:
        """Search documentation sources"""
        solutions = []
        
        try:
            # For now, use a simple web search with site: filter
            search_query = f"site:docs.python.org OR site:docs.djangoproject.com {query}"
            
            # This would typically use a search API or scraping
            # For demonstration, returning empty results
            logger.debug(f"Documentation search for: {query}")
            
        except Exception as e:
            logger.error(f"Documentation search failed: {e}")
        
        return solutions

class GeneralWebSearcher:
    """Searches general web for error solutions"""
    
    async def search(self, client: httpx.AsyncClient, query: str, max_results: int = 5) -> List[WebSolution]:
        """Search general web sources"""
        solutions = []
        
        try:
            # This would use a search engine API like Google, Bing, or DuckDuckGo
            # For demonstration, returning empty results
            logger.debug(f"General web search for: {query}")
            
        except Exception as e:
            logger.error(f"General web search failed: {e}")
        
        return solutions

class SolutionAdaptor:
    """Adapts web solutions to local code context"""
    
    def __init__(self, llm_api_url: str = "http://127.0.0.1:11434"):
        self.llm_api_url = llm_api_url
    
    async def adapt_solution(self, solution: WebSolution, error: ErrorRecord) -> Optional[FixSuggestion]:
        """Adapt a web solution to the local error context"""
        try:
            if not solution.code_snippets:
                return None
            
            # Build adaptation prompt
            prompt = self._build_adaptation_prompt(solution, error)
            
            # Query LLM for adaptation
            adapted_code = await self._query_llm(prompt)
            
            if not adapted_code:
                return None
            
            # Parse adapted code into fix suggestion
            fix = self._parse_adapted_fix(adapted_code, solution, error)
            
            return fix
            
        except Exception as e:
            logger.error(f"Failed to adapt solution: {e}")
            return None
    
    def _build_adaptation_prompt(self, solution: WebSolution, error: ErrorRecord) -> str:
        """Build prompt for solution adaptation"""
        prompt = f"""Adapt this web solution to fix the local error:

LOCAL ERROR:
File: {error.file_path}:{error.line_number}
Type: {error.error_type}
Message: {error.message}

CODE CONTEXT:
{error.snippet}

WEB SOLUTION:
Source: {solution.source}
Title: {solution.title}
Code Snippets:
{chr(10).join(solution.code_snippets)}

Please adapt the solution to the local context and provide:
1. Specific file and line changes needed
2. Explanation of the adaptation
3. Why this solution applies to the local error

Respond in JSON format with exact code changes."""
        
        return prompt
    
    async def _query_llm(self, prompt: str) -> str:
        """Query LLM for solution adaptation"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.llm_api_url}/api/generate",
                    json={
                        "model": "llama3.1:8b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1}
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '')
        
        except Exception as e:
            logger.error(f"LLM adaptation query failed: {e}")
        
        return ""
    
    def _parse_adapted_fix(self, adapted_code: str, solution: WebSolution, 
                          error: ErrorRecord) -> Optional[FixSuggestion]:
        """Parse adapted code into fix suggestion"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', adapted_code, re.DOTALL)
            if json_match:
                fix_data = json.loads(json_match.group())
                
                fix = FixSuggestion(
                    id=f"web_{solution.id}_{error.id[:8]}",
                    error_id=error.id,
                    strategy=FixStrategy.QUICK_FIX,
                    description=f"Solution from {solution.source}: {solution.title}",
                    code_changes=fix_data.get('code_changes', []),
                    confidence_score=solution.confidence_score * 0.8,  # Slightly lower for web solutions
                    reasoning=fix_data.get('explanation', ''),
                    dependencies=fix_data.get('dependencies', []),
                    test_commands=[],
                    rollback_safe=True,
                    estimated_impact='low',
                    created_at=datetime.now()
                )
                
                # Store reference to original solution
                solution.adapted_fix = fix
                
                return fix
        
        except Exception as e:
            logger.error(f"Failed to parse adapted fix: {e}")
        
        return None

class WebErrorResolver:
    """Main orchestrator for web-based error resolution"""
    
    def __init__(self):
        self.search_engine = ErrorSearchEngine()
        self.adaptor = SolutionAdaptor()
    
    async def resolve_error(self, error: ErrorRecord, max_solutions: int = 5) -> Dict[str, Any]:
        """Find and adapt web solutions for an error"""
        result = {
            'success': False,
            'error_id': error.id,
            'web_solutions': [],
            'adapted_fixes': [],
            'best_solution': None
        }
        
        try:
            # Search for web solutions
            solutions = await self.search_engine.search_for_error(error, max_solutions)
            result['web_solutions'] = solutions
            
            if not solutions:
                logger.info(f"No web solutions found for error {error.id[:8]}")
                return result
            
            # Adapt solutions to local context
            adapted_fixes = []
            for solution in solutions[:3]:  # Only adapt top 3 solutions
                adapted_fix = await self.adaptor.adapt_solution(solution, error)
                if adapted_fix:
                    adapted_fixes.append(adapted_fix)
            
            result['adapted_fixes'] = adapted_fixes
            
            # Select best solution
            if solutions:
                best_solution = max(solutions, key=lambda s: s.confidence_score * s.similarity_score)
                result['best_solution'] = best_solution
            
            result['success'] = True
            logger.info(f"Found {len(solutions)} web solutions, adapted {len(adapted_fixes)} fixes")
            
        except Exception as e:
            logger.error(f"Failed to resolve error with web research: {e}")
            result['error'] = str(e)
        
        return result
    
    async def batch_resolve_errors(self, error_ids: List[str]) -> List[Dict[str, Any]]:
        """Resolve multiple errors using web research"""
        results = []
        
        # Import here to avoid circular imports
        from error_tracker import ErrorDatabase
        
        db = ErrorDatabase()
        
        try:
            for error_id in error_ids:
                # Get error from database
                cursor = db.conn.cursor()
                cursor.execute("SELECT * FROM errors WHERE id = ?", (error_id,))
                error_row = cursor.fetchone()
                
                if error_row:
                    # Convert to ErrorRecord (simplified)
                    error = ErrorRecord(
                        id=error_row['id'],
                        timestamp=datetime.fromisoformat(error_row['timestamp']),
                        file_path=error_row['file_path'],
                        line_number=error_row['line_number'] or 0,
                        error_type=error_row['error_type'],
                        message=error_row['message'],
                        traceback_hash=error_row['traceback_hash'],
                        frequency=error_row['frequency'],
                        snippet=error_row['snippet'] or "",
                        first_seen=datetime.fromisoformat(error_row['first_seen']),
                        last_seen=datetime.fromisoformat(error_row['last_seen']),
                        severity=error_row['severity'],
                        full_traceback=error_row['full_traceback'] or ""
                    )
                    
                    # Resolve error
                    result = await self.resolve_error(error)
                    results.append(result)
                    
                    # Small delay between requests
                    await asyncio.sleep(2)
        
        finally:
            db.close()
        
        return results

async def main():
    """Main function for web error resolution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Web-Based Error Resolution Research")
    parser.add_argument("--error-id", help="Resolve specific error ID")
    parser.add_argument("--batch", nargs='+', help="Resolve multiple error IDs")
    parser.add_argument("--search-test", help="Test search for a query")
    
    args = parser.parse_args()
    
    resolver = WebErrorResolver()
    
    try:
        if args.error_id:
            print(f"🔍 Researching web solutions for error {args.error_id}...")
            
            # Get error from database
            from error_tracker import ErrorDatabase
            db = ErrorDatabase()
            
            cursor = db.conn.cursor()
            cursor.execute("SELECT * FROM errors WHERE id = ?", (args.error_id,))
            error_row = cursor.fetchone()
            
            if error_row:
                # Convert to ErrorRecord (simplified)
                error = ErrorRecord(
                    id=error_row['id'],
                    timestamp=datetime.fromisoformat(error_row['timestamp']),
                    file_path=error_row['file_path'],
                    line_number=error_row['line_number'] or 0,
                    error_type=error_row['error_type'],
                    message=error_row['message'],
                    traceback_hash=error_row['traceback_hash'],
                    frequency=error_row['frequency'],
                    snippet=error_row['snippet'] or "",
                    first_seen=datetime.fromisoformat(error_row['first_seen']),
                    last_seen=datetime.fromisoformat(error_row['last_seen']),
                    severity=error_row['severity'],
                    full_traceback=error_row['full_traceback'] or ""
                )
                
                result = await resolver.resolve_error(error)
                
                if result['success']:
                    print(f"✅ Found {len(result['web_solutions'])} web solutions")
                    print(f"🔧 Adapted {len(result['adapted_fixes'])} fixes")
                    
                    if result['best_solution']:
                        best = result['best_solution']
                        print(f"🏆 Best solution from {best.source}: {best.title}")
                        print(f"   Confidence: {best.confidence_score:.2f}")
                        print(f"   URL: {best.url}")
                else:
                    print(f"❌ Failed: {result.get('error', 'No solutions found')}")
            
            db.close()
        
        elif args.batch:
            print(f"🔍 Batch researching {len(args.batch)} errors...")
            results = await resolver.batch_resolve_errors(args.batch)
            
            successful = sum(1 for r in results if r['success'])
            print(f"✅ Successfully researched {successful}/{len(results)} errors")
        
        elif args.search_test:
            print(f"🔍 Testing search for: {args.search_test}")
            # Create a test error for search
            test_error = ErrorRecord(
                id="test_id",
                timestamp=datetime.now(),
                file_path="test.py",
                line_number=1,
                error_type="ValueError",
                message=args.search_test,
                traceback_hash="test_hash",
                frequency=1,
                snippet="",
                first_seen=datetime.now(),
                last_seen=datetime.now()
            )
            
            solutions = await resolver.search_engine.search_for_error(test_error)
            print(f"Found {len(solutions)} solutions:")
            
            for i, solution in enumerate(solutions, 1):
                print(f"{i}. {solution.title} ({solution.source})")
                print(f"   Confidence: {solution.confidence_score:.2f}")
                print(f"   URL: {solution.url}")
        
        else:
            print("Specify --error-id, --batch, or --search-test")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())