"""
Educational Reasoning System
============================

Course monitoring, topic mapping, concept tracking, and automated educational
resource harvesting with exam/deadline awareness and academic performance prediction.
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import re
from collections import defaultdict
import networkx as nx
import spacy

from database.user_profile_schema import UserProfileDatabase, EducationProfile, Calendar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EducationConfig:
    """Configuration for educational reasoning"""

    syllabus_directory: str = "input_data/education"
    auto_resource_harvesting: bool = True
    deadline_prediction_days: int = 14
    concept_mapping: bool = True
    performance_tracking: bool = True
    study_optimization: bool = True
    exam_stress_monitoring: bool = True
    academic_calendar_sync: bool = True


@dataclass
class Course:
    """Course information structure"""

    id: str
    name: str
    code: str = ""
    instructor: str = ""
    credits: int = 0
    schedule: List[Dict] = None
    syllabus_content: str = ""
    topics: List[str] = None
    assignments: List[Dict] = None
    exams: List[Dict] = None
    grades: List[Dict] = None
    prerequisites: List[str] = None

    def __post_init__(self):
        if self.schedule is None:
            self.schedule = []
        if self.topics is None:
            self.topics = []
        if self.assignments is None:
            self.assignments = []
        if self.exams is None:
            self.exams = []
        if self.grades is None:
            self.grades = []
        if self.prerequisites is None:
            self.prerequisites = []


@dataclass
class Topic:
    """Academic topic/concept structure"""

    id: str
    name: str
    course_id: str
    description: str = ""
    difficulty_level: float = 0.0  # 0.0 to 1.0
    prerequisites: List[str] = None
    subtopics: List[str] = None
    resources: List[Dict] = None
    mastery_level: float = 0.0  # User's mastery 0.0 to 1.0
    time_to_master: int = 0  # Estimated hours

    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.subtopics is None:
            self.subtopics = []
        if self.resources is None:
            self.resources = []


class SyllabusParser:
    """Parse syllabus documents and extract structured information"""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model not available. Install with: python -m spacy download en_core_web_sm"
            )
            self.nlp = None

        self.academic_keywords = {
            "assignment": [
                "assignment",
                "homework",
                "project",
                "paper",
                "essay",
                "report",
            ],
            "exam": ["exam", "test", "quiz", "midterm", "final", "assessment"],
            "topic": ["chapter", "unit", "section", "topic", "module", "lesson"],
            "deadline": ["due", "deadline", "submit", "turn in", "by"],
            "grade": ["grade", "points", "percent", "%", "score", "marks"],
        }

    def parse_syllabus_file(self, file_path: str) -> Course:
        """Parse syllabus file and extract course information"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            course_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
            course_name = self._extract_course_name(content)

            course = Course(
                id=course_id,
                name=course_name,
                code=self._extract_course_code(content),
                instructor=self._extract_instructor(content),
                syllabus_content=content,
                topics=self._extract_topics(content),
                assignments=self._extract_assignments(content),
                exams=self._extract_exams(content),
                schedule=self._extract_schedule(content),
            )

            return course

        except Exception as e:
            logger.error(f"Error parsing syllabus {file_path}: {e}")
            return None

    def _extract_course_name(self, content: str) -> str:
        """Extract course name from syllabus content"""
        lines = content.split("\n")[:10]  # Check first 10 lines

        # Look for common course name patterns
        patterns = [
            r"course:?\s*(.+)",
            r"class:?\s*(.+)",
            r"^([A-Z]+\s+\d+.*)",
            r"title:?\s*(.+)",
        ]

        for line in lines:
            line = line.strip()
            if len(line) > 5 and len(line) < 100:  # Reasonable course name length
                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()

        return os.path.basename(file_path).replace(".txt", "").replace(".md", "")

    def _extract_course_code(self, content: str) -> str:
        """Extract course code (e.g., CS101, MATH201)"""
        # Look for patterns like CS101, MATH 201, etc.
        pattern = r"\b[A-Z]{2,4}\s*\d{3,4}\b"
        matches = re.findall(pattern, content)

        if matches:
            return matches[0].replace(" ", "")

        return ""

    def _extract_instructor(self, content: str) -> str:
        """Extract instructor name"""
        patterns = [
            r"instructor:?\s*(.+)",
            r"professor:?\s*(.+)",
            r"taught by:?\s*(.+)",
            r"dr\.?\s+([a-z\s]+)",
            r"prof\.?\s+([a-z\s]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up the name
                name = re.sub(r"[^\w\s]", "", name)
                if len(name) > 3 and len(name) < 50:
                    return name

        return ""

    def _extract_topics(self, content: str) -> List[str]:
        """Extract course topics and concepts"""
        topics = []

        # Look for numbered/bulleted lists
        lines = content.split("\n")

        for line in lines:
            line = line.strip()

            # Skip very short or very long lines
            if len(line) < 5 or len(line) > 200:
                continue

            # Look for topic indicators
            if any(
                keyword in line.lower() for keyword in self.academic_keywords["topic"]
            ):
                # Extract the topic
                topic = re.sub(r"^\d+\.?\s*", "", line)  # Remove numbers
                topic = re.sub(r"^[•\-\*]\s*", "", topic)  # Remove bullets
                topic = topic.strip()

                if len(topic) > 5:
                    topics.append(topic)

        # Use NLP to extract key concepts if available
        if self.nlp:
            doc = self.nlp(content)

            # Extract noun phrases that might be topics
            for chunk in doc.noun_chunks:
                if (
                    len(chunk.text) > 5
                    and len(chunk.text) < 100
                    and chunk.root.pos_ in ["NOUN", "PROPN"]
                ):

                    topic = chunk.text.strip()
                    if topic not in topics and len(topics) < 50:  # Limit topics
                        topics.append(topic)

        return topics[:20]  # Return top 20 topics

    def _extract_assignments(self, content: str) -> List[Dict]:
        """Extract assignment information"""
        assignments = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for assignment keywords
            if any(
                keyword in line.lower()
                for keyword in self.academic_keywords["assignment"]
            ):
                assignment = {
                    "title": line,
                    "description": "",
                    "due_date": None,
                    "points": 0,
                    "type": "assignment",
                }

                # Look for due date in current or next few lines
                for j in range(i, min(i + 3, len(lines))):
                    if any(
                        keyword in lines[j].lower()
                        for keyword in self.academic_keywords["deadline"]
                    ):
                        due_date = self._extract_date(lines[j])
                        if due_date:
                            assignment["due_date"] = due_date.isoformat()

                # Look for points
                points_match = re.search(r"(\d+)\s*points?", line, re.IGNORECASE)
                if points_match:
                    assignment["points"] = int(points_match.group(1))

                assignments.append(assignment)

        return assignments

    def _extract_exams(self, content: str) -> List[Dict]:
        """Extract exam information"""
        exams = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for exam keywords
            if any(
                keyword in line.lower() for keyword in self.academic_keywords["exam"]
            ):
                exam = {
                    "title": line,
                    "date": None,
                    "type": "exam",
                    "topics": [],
                    "weight": 0.0,
                }

                # Look for exam date
                exam_date = self._extract_date(line)
                if not exam_date:
                    # Check next few lines
                    for j in range(i + 1, min(i + 3, len(lines))):
                        exam_date = self._extract_date(lines[j])
                        if exam_date:
                            break

                if exam_date:
                    exam["date"] = exam_date.isoformat()

                # Look for weight/percentage
                weight_match = re.search(r"(\d+)%", line)
                if weight_match:
                    exam["weight"] = float(weight_match.group(1)) / 100

                exams.append(exam)

        return exams

    def _extract_schedule(self, content: str) -> List[Dict]:
        """Extract course schedule"""
        schedule = []

        # Look for schedule patterns
        lines = content.split("\n")

        for line in lines:
            # Look for day/time patterns
            day_pattern = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)"
            time_pattern = r"(\d{1,2}:\d{2}(?:\s*[ap]m)?)"

            if re.search(day_pattern, line, re.IGNORECASE) and re.search(
                time_pattern, line, re.IGNORECASE
            ):
                schedule_entry = {
                    "raw_text": line.strip(),
                    "day": None,
                    "time": None,
                    "location": None,
                }

                # Extract day
                day_match = re.search(day_pattern, line, re.IGNORECASE)
                if day_match:
                    schedule_entry["day"] = day_match.group(1).lower()

                # Extract time
                time_match = re.search(time_pattern, line, re.IGNORECASE)
                if time_match:
                    schedule_entry["time"] = time_match.group(1)

                schedule.append(schedule_entry)

        return schedule

    def _extract_date(self, text: str) -> Optional[datetime]:
        """Extract date from text"""
        # Common date patterns
        date_patterns = [
            r"(\d{1,2})/(\d{1,2})/(\d{4})",
            r"(\d{1,2})-(\d{1,2})-(\d{4})",
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})",
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?\s+(\d{1,2}),?\s+(\d{4})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if "/" in pattern or "-" in pattern:
                        month, day, year = match.groups()
                        return datetime(int(year), int(month), int(day))
                    else:
                        month_str, day, year = match.groups()
                        month_names = {
                            "january": 1,
                            "february": 2,
                            "march": 3,
                            "april": 4,
                            "may": 5,
                            "june": 6,
                            "july": 7,
                            "august": 8,
                            "september": 9,
                            "october": 10,
                            "november": 11,
                            "december": 12,
                            "jan": 1,
                            "feb": 2,
                            "mar": 3,
                            "apr": 4,
                            "may": 5,
                            "jun": 6,
                            "jul": 7,
                            "aug": 8,
                            "sep": 9,
                            "oct": 10,
                            "nov": 11,
                            "dec": 12,
                        }
                        month = month_names.get(month_str.lower())
                        if month:
                            return datetime(int(year), month, int(day))
                except ValueError:
                    continue

        return None


class ConceptMapper:
    """Create concept dependency graphs and learning paths"""

    def __init__(self):
        self.concept_graph = nx.DiGraph()

        # Predefined concept relationships for common subjects
        self.concept_dependencies = {
            "mathematics": {
                "calculus": ["algebra", "trigonometry"],
                "linear_algebra": ["algebra"],
                "differential_equations": ["calculus"],
                "statistics": ["probability", "algebra"],
                "discrete_math": ["logic", "set_theory"],
            },
            "computer_science": {
                "data_structures": ["programming_basics"],
                "algorithms": ["data_structures", "mathematics"],
                "machine_learning": ["statistics", "linear_algebra", "programming"],
                "databases": ["data_structures"],
                "computer_networks": ["operating_systems"],
                "software_engineering": ["programming_basics", "data_structures"],
            },
            "physics": {
                "mechanics": ["algebra", "trigonometry"],
                "thermodynamics": ["mechanics"],
                "electromagnetism": ["calculus", "mechanics"],
                "quantum_physics": ["linear_algebra", "calculus", "electromagnetism"],
            },
            "chemistry": {
                "organic_chemistry": ["general_chemistry"],
                "physical_chemistry": ["calculus", "physics"],
                "biochemistry": ["organic_chemistry", "biology"],
            },
        }

    def build_concept_map(self, courses: List[Course]) -> nx.DiGraph:
        """Build concept dependency graph from courses"""
        # Add concepts as nodes
        for course in courses:
            for topic in course.topics:
                topic_id = self._normalize_topic_name(topic)
                self.concept_graph.add_node(
                    topic_id,
                    name=topic,
                    course=course.name,
                    difficulty=self._estimate_difficulty(topic),
                )

        # Add edges based on predefined dependencies
        for subject, dependencies in self.concept_dependencies.items():
            for concept, prereqs in dependencies.items():
                if concept in self.concept_graph:
                    for prereq in prereqs:
                        if prereq in self.concept_graph:
                            self.concept_graph.add_edge(prereq, concept)

        # Infer additional dependencies from topic names
        self._infer_dependencies()

        return self.concept_graph

    def _normalize_topic_name(self, topic: str) -> str:
        """Normalize topic name for graph storage"""
        return re.sub(r"[^\w\s]", "", topic.lower()).replace(" ", "_")

    def _estimate_difficulty(self, topic: str) -> float:
        """Estimate topic difficulty based on keywords"""
        difficulty_indicators = {
            "advanced": 0.9,
            "intermediate": 0.6,
            "basic": 0.3,
            "introduction": 0.2,
            "fundamentals": 0.3,
            "theory": 0.7,
            "applied": 0.5,
            "analysis": 0.8,
        }

        topic_lower = topic.lower()
        for indicator, difficulty in difficulty_indicators.items():
            if indicator in topic_lower:
                return difficulty

        return 0.5  # Default medium difficulty

    def _infer_dependencies(self):
        """Infer concept dependencies from naming patterns"""
        nodes = list(self.concept_graph.nodes())

        for node in nodes:
            node_name = self.concept_graph.nodes[node]["name"].lower()

            # Look for prerequisite patterns
            if "advanced" in node_name:
                # Find corresponding basic version
                basic_name = node_name.replace("advanced", "basic").replace(
                    "advanced", "introduction"
                )
                for other_node in nodes:
                    if (
                        other_node != node
                        and basic_name
                        in self.concept_graph.nodes[other_node]["name"].lower()
                    ):
                        self.concept_graph.add_edge(other_node, node)

            # Mathematical dependencies
            if any(
                term in node_name for term in ["calculus", "differential", "integral"]
            ):
                for other_node in nodes:
                    other_name = self.concept_graph.nodes[other_node]["name"].lower()
                    if "algebra" in other_name:
                        self.concept_graph.add_edge(other_node, node)

    def get_learning_path(self, target_concept: str) -> List[str]:
        """Get optimal learning path to target concept"""
        target_id = self._normalize_topic_name(target_concept)

        if target_id not in self.concept_graph:
            return []

        # Use topological sort to find dependencies
        try:
            # Get all ancestors (prerequisites)
            ancestors = nx.ancestors(self.concept_graph, target_id)
            ancestors.add(target_id)

            # Create subgraph and topologically sort
            subgraph = self.concept_graph.subgraph(ancestors)
            path = list(nx.topological_sort(subgraph))

            return [self.concept_graph.nodes[node]["name"] for node in path]

        except nx.NetworkXError:
            # Graph has cycles, use alternative approach
            return [target_concept]

    def identify_knowledge_gaps(
        self, mastered_concepts: List[str], target_concepts: List[str]
    ) -> List[str]:
        """Identify concepts that need to be learned"""
        mastered_ids = {
            self._normalize_topic_name(concept) for concept in mastered_concepts
        }

        gaps = set()

        for target in target_concepts:
            target_id = self._normalize_topic_name(target)
            if target_id in self.concept_graph:
                # Get all prerequisites
                prereqs = nx.ancestors(self.concept_graph, target_id)

                # Find unmastered prerequisites
                unmastered = prereqs - mastered_ids
                gaps.update(unmastered)

        return [
            self.concept_graph.nodes[gap]["name"]
            for gap in gaps
            if gap in self.concept_graph
        ]


class ResourceHarvester:
    """Automatically harvest educational resources"""

    def __init__(self):
        self.resource_sources = {
            "khan_academy": "https://www.khanacademy.org/api/v1/search",
            "coursera": "https://www.coursera.org/api/courses.v1/courses",
            "mit_ocw": "https://ocw.mit.edu/search/",
            "wikipedia": "https://en.wikipedia.org/api/rest_v1/page/summary/",
        }

    async def harvest_resources(
        self, topic: str, resource_type: str = "all"
    ) -> List[Dict]:
        """Harvest educational resources for a topic"""
        resources = []

        try:
            # Search multiple sources
            if resource_type in ["all", "video"]:
                resources.extend(await self._search_khan_academy(topic))

            if resource_type in ["all", "article"]:
                resources.extend(await self._search_wikipedia(topic))

            # Add more resource sources as needed

        except Exception as e:
            logger.error(f"Error harvesting resources for {topic}: {e}")

        return resources

    async def _search_khan_academy(self, topic: str) -> List[Dict]:
        """Search Khan Academy for educational videos"""
        resources = []

        try:
            async with aiohttp.ClientSession() as session:
                # Note: This is a simplified example - actual Khan Academy API requires authentication
                search_url = f"https://www.khanacademy.org/api/internal/graphql/search"

                # For demonstration, return mock data
                resources.append(
                    {
                        "title": f"Khan Academy: {topic}",
                        "url": f"https://www.khanacademy.org/search?page_search_query={topic.replace(' ', '%20')}",
                        "type": "video",
                        "source": "khan_academy",
                        "description": f"Educational videos about {topic}",
                        "difficulty": "intermediate",
                    }
                )

        except Exception as e:
            logger.error(f"Khan Academy search error: {e}")

        return resources

    async def _search_wikipedia(self, topic: str) -> List[Dict]:
        """Search Wikipedia for articles"""
        resources = []

        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"

                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        if "extract" in data:
                            resources.append(
                                {
                                    "title": data.get("title", topic),
                                    "url": data.get("content_urls", {})
                                    .get("desktop", {})
                                    .get("page", ""),
                                    "type": "article",
                                    "source": "wikipedia",
                                    "description": data.get("extract", "")[:200]
                                    + "...",
                                    "difficulty": "general",
                                }
                            )

        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")

        return resources


class PerformancePredictor:
    """Predict academic performance and identify at-risk situations"""

    def __init__(self):
        self.performance_factors = {
            "assignment_completion_rate": 0.3,
            "exam_performance": 0.4,
            "attendance_rate": 0.1,
            "concept_mastery": 0.15,
            "time_management": 0.05,
        }

    def predict_exam_performance(
        self, course: Course, upcoming_exam: Dict, historical_performance: List[Dict]
    ) -> Dict[str, Any]:
        """Predict performance on upcoming exam"""
        prediction = {
            "expected_score": 0.0,
            "confidence": 0.0,
            "risk_level": "low",
            "recommendations": [],
            "study_time_needed": 0,
            "weak_areas": [],
        }

        if not historical_performance:
            prediction["confidence"] = 0.1
            prediction["expected_score"] = 0.75  # Assume average
            return prediction

        # Calculate average historical performance
        avg_score = sum(p.get("score", 0) for p in historical_performance) / len(
            historical_performance
        )

        # Adjust based on exam difficulty (estimated from topics)
        exam_topics = upcoming_exam.get("topics", [])
        difficulty_adjustment = self._estimate_exam_difficulty(exam_topics)

        predicted_score = avg_score * (1 - difficulty_adjustment * 0.2)

        prediction["expected_score"] = max(0.0, min(1.0, predicted_score))
        prediction["confidence"] = min(0.8, len(historical_performance) * 0.2)

        # Determine risk level
        if predicted_score < 0.6:
            prediction["risk_level"] = "high"
            prediction["recommendations"].append("Increase study time significantly")
            prediction["recommendations"].append("Focus on fundamental concepts")
            prediction["study_time_needed"] = 20
        elif predicted_score < 0.75:
            prediction["risk_level"] = "medium"
            prediction["recommendations"].append("Review key concepts")
            prediction["study_time_needed"] = 10
        else:
            prediction["risk_level"] = "low"
            prediction["study_time_needed"] = 5

        return prediction

    def _estimate_exam_difficulty(self, topics: List[str]) -> float:
        """Estimate exam difficulty based on topics"""
        difficulty_keywords = {
            "advanced": 0.8,
            "complex": 0.7,
            "analysis": 0.6,
            "theory": 0.6,
            "proof": 0.8,
            "application": 0.5,
            "basic": 0.2,
            "introduction": 0.2,
        }

        total_difficulty = 0
        topic_count = len(topics) if topics else 1

        for topic in topics:
            topic_lower = topic.lower()
            topic_difficulty = 0.5  # Default

            for keyword, difficulty in difficulty_keywords.items():
                if keyword in topic_lower:
                    topic_difficulty = max(topic_difficulty, difficulty)

            total_difficulty += topic_difficulty

        return total_difficulty / topic_count

    def analyze_study_patterns(self, study_sessions: List[Dict]) -> Dict[str, Any]:
        """Analyze study patterns and effectiveness"""
        analysis = {
            "total_study_time": 0,
            "average_session_length": 0,
            "study_frequency": 0,
            "peak_study_times": [],
            "effectiveness_score": 0.0,
            "recommendations": [],
        }

        if not study_sessions:
            return analysis

        # Calculate basic metrics
        total_time = sum(
            session.get("duration_minutes", 0) for session in study_sessions
        )
        analysis["total_study_time"] = total_time
        analysis["average_session_length"] = total_time / len(study_sessions)

        # Analyze frequency
        dates = {session.get("date", "").split("T")[0] for session in study_sessions}
        days_with_study = len(dates)
        date_range = 30  # Assume 30-day analysis period
        analysis["study_frequency"] = days_with_study / date_range

        # Calculate effectiveness (simplified)
        if (
            analysis["average_session_length"] > 25
            and analysis["average_session_length"] < 120
        ):
            analysis["effectiveness_score"] += 0.3  # Optimal session length

        if analysis["study_frequency"] > 0.5:
            analysis["effectiveness_score"] += 0.3  # Regular study

        # Add recommendations
        if analysis["average_session_length"] < 25:
            analysis["recommendations"].append(
                "Increase study session length for better retention"
            )
        elif analysis["average_session_length"] > 120:
            analysis["recommendations"].append(
                "Break long sessions into smaller chunks with breaks"
            )

        if analysis["study_frequency"] < 0.3:
            analysis["recommendations"].append(
                "Study more frequently for better retention"
            )

        return analysis


class EduPredictor:
    """Main educational reasoning and prediction engine"""

    def __init__(self, db: UserProfileDatabase, config: EducationConfig = None):
        self.db = db
        self.config = config or EducationConfig()
        self.syllabus_parser = SyllabusParser()
        self.concept_mapper = ConceptMapper()
        self.resource_harvester = ResourceHarvester()
        self.performance_predictor = PerformancePredictor()

    async def process_educational_data(self) -> Dict[str, Any]:
        """Process all educational data and generate insights"""
        results = {
            "courses_processed": 0,
            "topics_mapped": 0,
            "resources_found": 0,
            "upcoming_deadlines": [],
            "performance_predictions": [],
            "study_recommendations": [],
            "knowledge_gaps": [],
            "errors": [],
        }

        try:
            # Parse syllabus files
            courses = await self._parse_syllabi()
            results["courses_processed"] = len(courses)

            # Store course data
            if courses:
                await self._store_courses(courses)

            # Build concept map
            if courses:
                concept_graph = self.concept_mapper.build_concept_map(courses)
                results["topics_mapped"] = len(concept_graph.nodes())

            # Harvest resources for key topics
            if self.config.auto_resource_harvesting:
                resources = await self._harvest_topic_resources(courses)
                results["resources_found"] = len(resources)

            # Identify upcoming deadlines
            results["upcoming_deadlines"] = await self._identify_upcoming_deadlines(
                courses
            )

            # Generate performance predictions
            results["performance_predictions"] = (
                await self._generate_performance_predictions(courses)
            )

            # Provide study recommendations
            results["study_recommendations"] = self._generate_study_recommendations(
                courses
            )

        except Exception as e:
            error_msg = f"Educational processing error: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)

        return results

    async def _parse_syllabi(self) -> List[Course]:
        """Parse all syllabus files in directory"""
        courses = []
        syllabus_dir = Path(self.config.syllabus_directory)

        if not syllabus_dir.exists():
            logger.warning(f"Syllabus directory {syllabus_dir} does not exist")
            return courses

        for file_path in syllabus_dir.glob("*"):
            if file_path.suffix.lower() in [
                ".txt",
                ".md",
                ".pdf",
            ]:  # Add PDF support later
                course = self.syllabus_parser.parse_syllabus_file(str(file_path))
                if course:
                    courses.append(course)

        return courses

    async def _store_courses(self, courses: List[Course]):
        """Store course data in database"""
        session = self.db.get_session()

        try:
            for course in courses:
                # Check if course already exists
                existing = (
                    session.query(EducationProfile)
                    .filter(
                        EducationProfile.education_type == "course",
                        EducationProfile.program_name == course.name,
                    )
                    .first()
                )

                if not existing:
                    education_record = EducationProfile(
                        education_type="course",
                        institution_name="",  # Could be extracted from syllabus
                        program_name=course.name,
                        start_date=datetime.now(),  # Could be extracted
                        status="current",
                        course_details={
                            "code": course.code,
                            "instructor": course.instructor,
                            "topics": course.topics,
                            "assignments": course.assignments,
                            "exams": course.exams,
                            "schedule": course.schedule,
                        },
                        performance_metrics={
                            "current_grade": 0.0,
                            "assignment_completion": 0.0,
                            "attendance": 0.0,
                        },
                        skill_development=course.topics,
                        confidence_score=0.8,
                        data_source="syllabus_parser",
                    )
                    session.add(education_record)

            session.commit()
            logger.info(f"Stored {len(courses)} course records")

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing course data: {e}")
        finally:
            session.close()

    async def _harvest_topic_resources(self, courses: List[Course]) -> List[Dict]:
        """Harvest educational resources for course topics"""
        all_resources = []

        for course in courses:
            for topic in course.topics[:5]:  # Limit to top 5 topics per course
                resources = await self.resource_harvester.harvest_resources(topic)
                all_resources.extend(resources)

        return all_resources

    async def _identify_upcoming_deadlines(self, courses: List[Course]) -> List[Dict]:
        """Identify upcoming academic deadlines"""
        deadlines = []
        current_date = datetime.now()
        deadline_window = current_date + timedelta(
            days=self.config.deadline_prediction_days
        )

        for course in courses:
            # Check assignments
            for assignment in course.assignments:
                if assignment.get("due_date"):
                    try:
                        due_date = datetime.fromisoformat(assignment["due_date"])
                        if current_date <= due_date <= deadline_window:
                            deadlines.append(
                                {
                                    "type": "assignment",
                                    "course": course.name,
                                    "title": assignment["title"],
                                    "due_date": assignment["due_date"],
                                    "days_until": (due_date - current_date).days,
                                    "priority": (
                                        "high"
                                        if (due_date - current_date).days <= 3
                                        else "medium"
                                    ),
                                }
                            )
                    except ValueError:
                        continue

            # Check exams
            for exam in course.exams:
                if exam.get("date"):
                    try:
                        exam_date = datetime.fromisoformat(exam["date"])
                        if current_date <= exam_date <= deadline_window:
                            deadlines.append(
                                {
                                    "type": "exam",
                                    "course": course.name,
                                    "title": exam["title"],
                                    "due_date": exam["date"],
                                    "days_until": (exam_date - current_date).days,
                                    "priority": (
                                        "critical"
                                        if (exam_date - current_date).days <= 5
                                        else "high"
                                    ),
                                }
                            )
                    except ValueError:
                        continue

        return sorted(deadlines, key=lambda x: x["days_until"])

    async def _generate_performance_predictions(
        self, courses: List[Course]
    ) -> List[Dict]:
        """Generate academic performance predictions"""
        predictions = []

        for course in courses:
            for exam in course.exams:
                if exam.get("date"):
                    try:
                        exam_date = datetime.fromisoformat(exam["date"])
                        if exam_date > datetime.now():
                            # Mock historical performance (would come from actual data)
                            historical_performance = [
                                {"score": 0.85, "type": "exam"},
                                {"score": 0.78, "type": "assignment"},
                            ]

                            prediction = (
                                self.performance_predictor.predict_exam_performance(
                                    course, exam, historical_performance
                                )
                            )

                            prediction.update(
                                {
                                    "course": course.name,
                                    "exam_title": exam["title"],
                                    "exam_date": exam["date"],
                                }
                            )

                            predictions.append(prediction)
                    except ValueError:
                        continue

        return predictions

    def _generate_study_recommendations(self, courses: List[Course]) -> List[str]:
        """Generate personalized study recommendations"""
        recommendations = []

        # Generic recommendations (would be personalized based on user data)
        if courses:
            recommendations.extend(
                [
                    "Create a study schedule with regular review sessions",
                    "Focus on active learning techniques like practice problems",
                    "Form study groups for collaborative learning",
                    "Use spaced repetition for better retention",
                    "Take breaks every 45-60 minutes during study sessions",
                ]
            )

        return recommendations


async def run_educational_analysis(
    db_path: str = None, config_dict: Dict = None
) -> Dict[str, Any]:
    """Main function to run educational analysis"""
    db = UserProfileDatabase(db_path)

    config = EducationConfig()
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    predictor = EduPredictor(db, config)
    results = await predictor.process_educational_data()

    return results


if __name__ == "__main__":
    # Test educational analysis
    async def main():
        results = await run_educational_analysis()
        print("Educational Analysis Results:")
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())
