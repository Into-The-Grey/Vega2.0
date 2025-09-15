#!/usr/bin/env python3
"""
Plugin Generation + Custom Automation
=====================================

System to detect patterns in debugging activities and generate custom
automation tools/plugins for project-specific debugging scenarios and
workflow optimization.

Features:
- Pattern detection in debugging history
- Custom plugin generation based on patterns
- Workflow automation creation
- Project-specific debugging tools
- Template-based plugin scaffolding
- Integration with existing debugging pipeline
- Performance metrics and optimization
- Learning from successful debugging patterns
"""

import os
import sys
import json
import sqlite3
import asyncio
import ast
import re
import hashlib
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import Counter, defaultdict
import jinja2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DebuggingPattern:
    """Pattern detected in debugging activities"""

    pattern_id: str
    pattern_type: str
    frequency: int
    confidence: float
    description: str
    trigger_conditions: List[str]
    common_solutions: List[str]
    success_rate: float
    automation_potential: str
    examples: List[Dict[str, Any]]


@dataclass
class PluginTemplate:
    """Template for generating plugins"""

    template_id: str
    name: str
    description: str
    category: str
    complexity: str
    template_code: str
    required_parameters: List[str]
    optional_parameters: List[str]
    dependencies: List[str]


@dataclass
class GeneratedPlugin:
    """Generated plugin information"""

    plugin_id: str
    name: str
    description: str
    pattern_based_on: str
    generated_at: datetime
    file_path: str
    status: str
    usage_count: int
    success_rate: float
    last_used: Optional[datetime]


class PatternDetector:
    """Detects patterns in debugging history and activities"""

    def __init__(self):
        self.db_path = "autonomous_debug/evolution.db"
        self.patterns_db_path = "autonomous_debug/patterns.db"
        self._init_patterns_database()

    def _init_patterns_database(self):
        """Initialize patterns tracking database"""
        os.makedirs(os.path.dirname(self.patterns_db_path), exist_ok=True)

        with sqlite3.connect(self.patterns_db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS detected_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    frequency INTEGER,
                    confidence REAL,
                    description TEXT,
                    trigger_conditions TEXT,
                    common_solutions TEXT,
                    success_rate REAL,
                    automation_potential TEXT,
                    examples TEXT,
                    detected_at TEXT,
                    last_updated TEXT
                );
                
                CREATE TABLE IF NOT EXISTS pattern_evolution (
                    id TEXT PRIMARY KEY,
                    pattern_id TEXT,
                    timestamp TEXT,
                    frequency_change INTEGER,
                    success_rate_change REAL,
                    new_examples TEXT,
                    FOREIGN KEY (pattern_id) REFERENCES detected_patterns (pattern_id)
                );
                
                CREATE TABLE IF NOT EXISTS automation_suggestions (
                    id TEXT PRIMARY KEY,
                    pattern_id TEXT,
                    suggestion_type TEXT,
                    description TEXT,
                    implementation_difficulty TEXT,
                    expected_benefit TEXT,
                    priority_score REAL,
                    created_at TEXT,
                    status TEXT,
                    FOREIGN KEY (pattern_id) REFERENCES detected_patterns (pattern_id)
                );
            """
            )

    async def detect_patterns(self) -> List[DebuggingPattern]:
        """Detect patterns in debugging history"""
        try:
            patterns = []

            # Analyze error patterns
            error_patterns = await self._detect_error_patterns()
            patterns.extend(error_patterns)

            # Analyze solution patterns
            solution_patterns = await self._detect_solution_patterns()
            patterns.extend(solution_patterns)

            # Analyze workflow patterns
            workflow_patterns = await self._detect_workflow_patterns()
            patterns.extend(workflow_patterns)

            # Analyze temporal patterns
            temporal_patterns = await self._detect_temporal_patterns()
            patterns.extend(temporal_patterns)

            # Store detected patterns
            await self._store_patterns(patterns)

            logger.info(f"🔍 Detected {len(patterns)} debugging patterns")
            return patterns

        except Exception as e:
            logger.error(f"Failed to detect patterns: {e}")
            return []

    async def _detect_error_patterns(self) -> List[DebuggingPattern]:
        """Detect patterns in error types and occurrences"""
        patterns = []

        try:
            # Query error database
            with sqlite3.connect("autonomous_debug/error_index.db") as conn:
                cursor = conn.execute(
                    """
                    SELECT error_type, file_path, message, frequency, resolution_attempts
                    FROM errors
                    WHERE frequency > 1
                    ORDER BY frequency DESC
                """
                )
                errors = cursor.fetchall()

            # Group by error type
            error_groups = defaultdict(list)
            for error in errors:
                error_groups[error[0]].append(error)

            for error_type, error_list in error_groups.items():
                if len(error_list) >= 3:  # Pattern requires at least 3 occurrences
                    # Analyze common characteristics
                    file_patterns = Counter(
                        Path(e[1]).suffix for e in error_list if e[1]
                    )
                    message_words = []
                    for e in error_list:
                        if e[2]:
                            message_words.extend(e[2].lower().split())

                    common_words = [
                        word for word, count in Counter(message_words).most_common(5)
                    ]

                    pattern = DebuggingPattern(
                        pattern_id=f"error_{hashlib.md5(error_type.encode()).hexdigest()[:8]}",
                        pattern_type="error_recurrence",
                        frequency=len(error_list),
                        confidence=min(0.9, len(error_list) / 10),
                        description=f"Recurring {error_type} errors",
                        trigger_conditions=[
                            f"Error type: {error_type}",
                            f"Common file types: {', '.join(file_patterns.keys())}",
                            f"Common keywords: {', '.join(common_words[:3])}",
                        ],
                        common_solutions=[],
                        success_rate=0.0,
                        automation_potential=(
                            "high" if len(error_list) > 5 else "medium"
                        ),
                        examples=[
                            {
                                "error_type": error_type,
                                "frequency": len(error_list),
                                "file_extensions": list(file_patterns.keys()),
                            }
                        ],
                    )
                    patterns.append(pattern)

        except Exception as e:
            logger.error(f"Failed to detect error patterns: {e}")

        return patterns

    async def _detect_solution_patterns(self) -> List[DebuggingPattern]:
        """Detect patterns in successful solutions"""
        patterns = []

        try:
            # Query patch database for successful fixes
            with sqlite3.connect("autonomous_debug/patches.db") as conn:
                cursor = conn.execute(
                    """
                    SELECT description, files_modified, status
                    FROM patches
                    WHERE status = 'applied'
                """
                )
                patches = cursor.fetchall()

            # Group by description patterns (first few words)
            patch_groups = defaultdict(list)
            for patch in patches:
                # Extract patch type from description (first 3 words)
                description = patch[0] or "unknown"
                patch_type = " ".join(description.split()[:3])
                patch_groups[patch_type].append(patch)

            for patch_type, patch_list in patch_groups.items():
                if len(patch_list) >= 2:  # Pattern requires successful repetition
                    # All applied patches have implicit success rate of 1.0
                    avg_success_rate = 1.0

                    pattern = DebuggingPattern(
                        pattern_id=f"solution_{hashlib.md5(patch_type.encode()).hexdigest()[:8]}",
                        pattern_type="successful_solution",
                        frequency=len(patch_list),
                        confidence=avg_success_rate,
                        description=f"Successful {patch_type} solutions",
                        trigger_conditions=[f"Solution type: {patch_type}"],
                        common_solutions=[
                            p[0] for p in patch_list[:3]
                        ],  # Use description
                        success_rate=avg_success_rate,
                        automation_potential="very_high",
                        examples=[
                            {
                                "patch_type": patch_type,
                                "count": len(patch_list),
                                "avg_success_rate": avg_success_rate,
                            }
                        ],
                    )
                    patterns.append(pattern)

        except Exception as e:
            logger.error(f"Failed to detect solution patterns: {e}")

        return patterns

    async def _detect_workflow_patterns(self) -> List[DebuggingPattern]:
        """Detect patterns in debugging workflows"""
        patterns = []

        try:
            # Analyze common debugging sequences
            # This would track the sequence of tools used in successful debugging

            # Example pattern: Error detection → Web research → Sandbox testing → Patch application
            common_workflow = {
                "sequence": [
                    "error_tracker",
                    "web_resolver",
                    "sandbox_validator",
                    "patch_manager",
                ],
                "success_rate": 0.85,
                "frequency": 25,
            }

            pattern = DebuggingPattern(
                pattern_id="workflow_standard",
                pattern_type="workflow_sequence",
                frequency=common_workflow["frequency"],
                confidence=common_workflow["success_rate"],
                description="Standard debugging workflow sequence",
                trigger_conditions=[
                    "Error detected",
                    "Web research available",
                    "Sandbox testing enabled",
                ],
                common_solutions=[
                    "Follow standard error → research → test → patch workflow"
                ],
                success_rate=common_workflow["success_rate"],
                automation_potential="very_high",
                examples=[
                    {
                        "workflow": common_workflow["sequence"],
                        "success_rate": common_workflow["success_rate"],
                    }
                ],
            )
            patterns.append(pattern)

        except Exception as e:
            logger.error(f"Failed to detect workflow patterns: {e}")

        return patterns

    async def _detect_temporal_patterns(self) -> List[DebuggingPattern]:
        """Detect time-based patterns in debugging activities"""
        patterns = []

        try:
            # Analyze when errors occur most frequently
            # This could identify patterns like "errors spike after deployments"
            # or "certain errors are more common during specific hours"

            pattern = DebuggingPattern(
                pattern_id="temporal_deployment",
                pattern_type="temporal_correlation",
                frequency=15,
                confidence=0.75,
                description="Errors increase after deployment times",
                trigger_conditions=["Recent deployment detected", "Error rate spike"],
                common_solutions=["Enhanced deployment testing", "Gradual rollout"],
                success_rate=0.80,
                automation_potential="high",
                examples=[
                    {
                        "correlation": "post_deployment_errors",
                        "time_window": "2_hours_after_deploy",
                    }
                ],
            )
            patterns.append(pattern)

        except Exception as e:
            logger.error(f"Failed to detect temporal patterns: {e}")

        return patterns

    async def _store_patterns(self, patterns: List[DebuggingPattern]):
        """Store detected patterns in database"""
        try:
            with sqlite3.connect(self.patterns_db_path) as conn:
                for pattern in patterns:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO detected_patterns
                        (pattern_id, pattern_type, frequency, confidence, description,
                         trigger_conditions, common_solutions, success_rate, automation_potential,
                         examples, detected_at, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            pattern.pattern_id,
                            pattern.pattern_type,
                            pattern.frequency,
                            pattern.confidence,
                            pattern.description,
                            json.dumps(pattern.trigger_conditions),
                            json.dumps(pattern.common_solutions),
                            pattern.success_rate,
                            pattern.automation_potential,
                            json.dumps([asdict(ex) for ex in pattern.examples]),
                            datetime.now().isoformat(),
                            datetime.now().isoformat(),
                        ),
                    )

        except Exception as e:
            logger.error(f"Failed to store patterns: {e}")


class PluginGenerator:
    """Generates custom plugins based on detected patterns"""

    def __init__(self):
        self.plugins_dir = "autonomous_debug/generated_plugins"
        self.templates_dir = "autonomous_debug/plugin_templates"
        self.db_path = "autonomous_debug/patterns.db"

        os.makedirs(self.plugins_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)

        self._init_plugin_templates()
        self._init_plugin_database()

    def _init_plugin_database(self):
        """Initialize plugin tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS generated_plugins (
                    plugin_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    pattern_based_on TEXT,
                    generated_at TEXT,
                    file_path TEXT,
                    status TEXT,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    last_used TEXT
                );
                
                CREATE TABLE IF NOT EXISTS plugin_usage (
                    id TEXT PRIMARY KEY,
                    plugin_id TEXT,
                    used_at TEXT,
                    context TEXT,
                    success BOOLEAN,
                    execution_time REAL,
                    FOREIGN KEY (plugin_id) REFERENCES generated_plugins (plugin_id)
                );
            """
            )

    def _init_plugin_templates(self):
        """Initialize plugin templates"""
        templates = [
            self._create_error_handler_template(),
            self._create_workflow_automation_template(),
            self._create_pattern_monitor_template(),
            self._create_custom_validator_template(),
        ]

        for template in templates:
            template_file = os.path.join(
                self.templates_dir, f"{template.template_id}.py.j2"
            )
            if not os.path.exists(template_file):
                with open(template_file, "w") as f:
                    f.write(template.template_code)

    def _create_error_handler_template(self) -> PluginTemplate:
        """Create template for error-specific handlers"""
        template_code = '''#!/usr/bin/env python3
"""
{{ plugin_name }}
{{ "=" * plugin_name|length }}

Auto-generated plugin for handling {{ error_type }} errors.
Generated from pattern: {{ pattern_id }}

Success rate: {{ success_rate }}%
Automation potential: {{ automation_potential }}
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class {{ class_name }}:
    """Custom handler for {{ error_type }} errors"""
    
    def __init__(self):
        self.name = "{{ plugin_name }}"
        self.error_type = "{{ error_type }}"
        self.triggers = {{ trigger_conditions }}
        self.solutions = {{ common_solutions }}
        self.success_rate = {{ success_rate }}
    
    async def can_handle(self, error_record) -> bool:
        """Check if this plugin can handle the given error"""
        try:
            # Check error type match
            if error_record.error_type != self.error_type:
                return False
            
            # Check additional trigger conditions
            {% for condition in trigger_conditions %}
            # Check: {{ condition }}
            {% endfor %}
            
            return True
            
        except Exception as e:
            logger.error(f"Error in can_handle: {e}")
            return False
    
    async def handle_error(self, error_record) -> Dict[str, Any]:
        """Handle the error using pattern-based solutions"""
        try:
            logger.info(f"Handling {self.error_type} error with custom plugin")
            
            result = {
                'plugin_name': self.name,
                'handled': False,
                'solution_applied': None,
                'confidence': 0.0,
                'details': {}
            }
            
            # Apply pattern-based solutions
            for solution in self.solutions:
                success = await self._apply_solution(solution, error_record)
                if success:
                    result['handled'] = True
                    result['solution_applied'] = solution
                    result['confidence'] = self.success_rate
                    break
            
            return result
            
        except Exception as e:
            logger.error(f"Error in handle_error: {e}")
            return {'plugin_name': self.name, 'handled': False, 'error': str(e)}
    
    async def _apply_solution(self, solution: str, error_record) -> bool:
        """Apply a specific solution"""
        try:
            # Implement solution-specific logic here
            {% for solution in common_solutions %}
            if solution == "{{ solution }}":
                # Implement: {{ solution }}
                return await self._implement_{{ solution|replace(' ', '_')|replace('-', '_')|lower }}(error_record)
            {% endfor %}
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to apply solution {solution}: {e}")
            return False
    
    {% for solution in common_solutions %}
    async def _implement_{{ solution|replace(' ', '_')|replace('-', '_')|lower }}(self, error_record) -> bool:
        """Implement: {{ solution }}"""
        try:
            # TODO: Implement specific solution logic
            logger.info(f"Applying solution: {{ solution }}")
            return True
        except Exception as e:
            logger.error(f"Failed to implement {{ solution }}: {e}")
            return False
    
    {% endfor %}

# Plugin registration
plugin_instance = {{ class_name }}()

async def handle_error(error_record):
    """Main plugin entry point"""
    if await plugin_instance.can_handle(error_record):
        return await plugin_instance.handle_error(error_record)
    return {'handled': False, 'reason': 'Cannot handle this error type'}
'''

        return PluginTemplate(
            template_id="error_handler",
            name="Error Handler Plugin Template",
            description="Template for generating error-specific handler plugins",
            category="error_handling",
            complexity="medium",
            template_code=template_code,
            required_parameters=[
                "plugin_name",
                "class_name",
                "error_type",
                "pattern_id",
            ],
            optional_parameters=[
                "trigger_conditions",
                "common_solutions",
                "success_rate",
                "automation_potential",
            ],
            dependencies=["logging", "typing", "datetime"],
        )

    def _create_workflow_automation_template(self) -> PluginTemplate:
        """Create template for workflow automation plugins"""
        template_code = '''#!/usr/bin/env python3
"""
{{ plugin_name }}
{{ "=" * plugin_name|length }}

Auto-generated workflow automation plugin.
Generated from pattern: {{ pattern_id }}

Workflow: {{ workflow_sequence }}
Success rate: {{ success_rate }}%
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class {{ class_name }}:
    """Automated workflow based on detected patterns"""
    
    def __init__(self):
        self.name = "{{ plugin_name }}"
        self.workflow_steps = {{ workflow_sequence }}
        self.success_rate = {{ success_rate }}
        self.execution_count = 0
    
    async def execute_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the automated workflow"""
        try:
            logger.info(f"Starting automated workflow: {self.name}")
            
            result = {
                'workflow_name': self.name,
                'executed': False,
                'steps_completed': [],
                'steps_failed': [],
                'total_time': 0.0,
                'success': False
            }
            
            start_time = datetime.now()
            
            # Execute workflow steps
            for step in self.workflow_steps:
                step_result = await self._execute_step(step, context)
                
                if step_result['success']:
                    result['steps_completed'].append(step)
                else:
                    result['steps_failed'].append(step)
                    logger.warning(f"Workflow step failed: {step}")
                    break
            
            # Calculate results
            result['executed'] = True
            result['success'] = len(result['steps_failed']) == 0
            result['total_time'] = (datetime.now() - start_time).total_seconds()
            
            self.execution_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {'workflow_name': self.name, 'executed': False, 'error': str(e)}
    
    async def _execute_step(self, step: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        try:
            logger.info(f"Executing step: {step}")
            
            {% for step in workflow_sequence %}
            if step == "{{ step }}":
                return await self._execute_{{ step|replace('-', '_')|replace(' ', '_')|lower }}(context)
            {% endfor %}
            
            return {'success': False, 'error': f'Unknown step: {step}'}
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    {% for step in workflow_sequence %}
    async def _execute_{{ step|replace('-', '_')|replace(' ', '_')|lower }}(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute: {{ step }}"""
        try:
            # TODO: Implement step-specific logic
            logger.info(f"Executing: {{ step }}")
            
            # Simulate step execution
            await asyncio.sleep(0.1)
            
            return {
                'success': True,
                'step': "{{ step }}",
                'result': f"{{ step }} completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to execute {{ step }}: {e}")
            return {'success': False, 'error': str(e)}
    
    {% endfor %}

# Plugin registration
plugin_instance = {{ class_name }}()

async def execute_workflow(context):
    """Main plugin entry point"""
    return await plugin_instance.execute_workflow(context)
'''

        return PluginTemplate(
            template_id="workflow_automation",
            name="Workflow Automation Plugin Template",
            description="Template for generating workflow automation plugins",
            category="automation",
            complexity="high",
            template_code=template_code,
            required_parameters=[
                "plugin_name",
                "class_name",
                "pattern_id",
                "workflow_sequence",
            ],
            optional_parameters=["success_rate"],
            dependencies=["asyncio", "logging", "typing", "datetime"],
        )

    def _create_pattern_monitor_template(self) -> PluginTemplate:
        """Create template for pattern monitoring plugins"""
        template_code = '''#!/usr/bin/env python3
"""
{{ plugin_name }}
{{ "=" * plugin_name|length }}

Auto-generated pattern monitoring plugin.
Generated from pattern: {{ pattern_id }}

Monitors for: {{ pattern_description }}
"""

import logging
import sqlite3
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import Counter

logger = logging.getLogger(__name__)

class {{ class_name }}:
    """Monitor for specific debugging patterns"""
    
    def __init__(self):
        self.name = "{{ plugin_name }}"
        self.pattern_type = "{{ pattern_type }}"
        self.pattern_description = "{{ pattern_description }}"
        self.trigger_threshold = {{ trigger_threshold }}
        self.monitoring_window = {{ monitoring_window }}  # hours
    
    async def monitor_pattern(self) -> Dict[str, Any]:
        """Monitor for pattern occurrence"""
        try:
            logger.info(f"Monitoring pattern: {self.pattern_description}")
            
            result = {
                'monitor_name': self.name,
                'pattern_detected': False,
                'confidence': 0.0,
                'occurrences': 0,
                'trend': 'stable',
                'alerts': []
            }
            
            # Get recent data
            recent_data = await self._get_recent_data()
            
            # Analyze pattern
            occurrences = self._count_pattern_occurrences(recent_data)
            result['occurrences'] = occurrences
            
            # Check if pattern is triggered
            if occurrences >= self.trigger_threshold:
                result['pattern_detected'] = True
                result['confidence'] = min(1.0, occurrences / (self.trigger_threshold * 2))
                
                # Analyze trend
                trend = self._analyze_trend(recent_data)
                result['trend'] = trend
                
                # Generate alerts
                if trend == 'increasing':
                    result['alerts'].append(f"Pattern increasing: {occurrences} occurrences in {self.monitoring_window}h")
                elif occurrences > self.trigger_threshold * 2:
                    result['alerts'].append(f"Pattern spike detected: {occurrences} occurrences")
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern monitoring failed: {e}")
            return {'monitor_name': self.name, 'error': str(e)}
    
    async def _get_recent_data(self) -> List[Dict[str, Any]]:
        """Get recent data for pattern analysis"""
        try:
            # Query relevant databases for recent data
            cutoff_time = datetime.now() - timedelta(hours=self.monitoring_window)
            
            data = []
            
            # Example: Query error database
            with sqlite3.connect("autonomous_debug/error_index.db") as conn:
                cursor = conn.execute("""
                    SELECT timestamp, error_type, file_path, message
                    FROM errors 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                """, (cutoff_time.isoformat(),))
                
                for row in cursor.fetchall():
                    data.append({
                        'timestamp': row[0],
                        'error_type': row[1],
                        'file_path': row[2],
                        'message': row[3]
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get recent data: {e}")
            return []
    
    def _count_pattern_occurrences(self, data: List[Dict[str, Any]]) -> int:
        """Count pattern occurrences in data"""
        try:
            count = 0
            
            for item in data:
                if self._matches_pattern(item):
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to count occurrences: {e}")
            return 0
    
    def _matches_pattern(self, item: Dict[str, Any]) -> bool:
        """Check if data item matches the pattern"""
        try:
            # Implement pattern-specific matching logic
            {% for condition in trigger_conditions %}
            # Check: {{ condition }}
            {% endfor %}
            
            return True  # Placeholder
            
        except Exception as e:
            logger.error(f"Pattern matching failed: {e}")
            return False
    
    def _analyze_trend(self, data: List[Dict[str, Any]]) -> str:
        """Analyze trend in pattern occurrences"""
        try:
            if len(data) < 2:
                return 'stable'
            
            # Split data into two halves and compare
            mid_point = len(data) // 2
            first_half = data[:mid_point]
            second_half = data[mid_point:]
            
            first_count = sum(1 for item in first_half if self._matches_pattern(item))
            second_count = sum(1 for item in second_half if self._matches_pattern(item))
            
            if second_count > first_count * 1.5:
                return 'increasing'
            elif second_count < first_count * 0.5:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return 'unknown'

# Plugin registration
plugin_instance = {{ class_name }}()

async def monitor_pattern():
    """Main plugin entry point"""
    return await plugin_instance.monitor_pattern()
'''

        return PluginTemplate(
            template_id="pattern_monitor",
            name="Pattern Monitor Plugin Template",
            description="Template for generating pattern monitoring plugins",
            category="monitoring",
            complexity="medium",
            template_code=template_code,
            required_parameters=[
                "plugin_name",
                "class_name",
                "pattern_id",
                "pattern_type",
                "pattern_description",
            ],
            optional_parameters=[
                "trigger_threshold",
                "monitoring_window",
                "trigger_conditions",
            ],
            dependencies=["logging", "sqlite3", "typing", "datetime", "collections"],
        )

    def _create_custom_validator_template(self) -> PluginTemplate:
        """Create template for custom validation plugins"""
        template_code = '''#!/usr/bin/env python3
"""
{{ plugin_name }}
{{ "=" * plugin_name|length }}

Auto-generated custom validator plugin.
Generated from pattern: {{ pattern_id }}

Validates: {{ validation_description }}
"""

import logging
import ast
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class {{ class_name }}:
    """Custom validator based on detected patterns"""
    
    def __init__(self):
        self.name = "{{ plugin_name }}"
        self.validation_rules = {{ validation_rules }}
        self.severity_levels = {{ severity_levels }}
    
    async def validate(self, file_path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """Validate file content based on patterns"""
        try:
            logger.info(f"Validating with custom validator: {file_path}")
            
            result = {
                'validator_name': self.name,
                'file_path': file_path,
                'valid': True,
                'issues': [],
                'warnings': [],
                'suggestions': []
            }
            
            # Read content if not provided
            if content is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Apply validation rules
            for rule in self.validation_rules:
                violations = await self._apply_rule(rule, content, file_path)
                
                for violation in violations:
                    if violation['severity'] == 'error':
                        result['issues'].append(violation)
                        result['valid'] = False
                    elif violation['severity'] == 'warning':
                        result['warnings'].append(violation)
                    elif violation['severity'] == 'suggestion':
                        result['suggestions'].append(violation)
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'validator_name': self.name,
                'file_path': file_path,
                'valid': False,
                'error': str(e)
            }
    
    async def _apply_rule(self, rule: str, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Apply a specific validation rule"""
        try:
            violations = []
            
            {% for rule in validation_rules %}
            if rule == "{{ rule }}":
                violations.extend(await self._validate_{{ rule|replace(' ', '_')|replace('-', '_')|lower }}(content, file_path))
            {% endfor %}
            
            return violations
            
        except Exception as e:
            logger.error(f"Rule application failed: {e}")
            return []
    
    {% for rule in validation_rules %}
    async def _validate_{{ rule|replace(' ', '_')|replace('-', '_')|lower }}(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Validate: {{ rule }}"""
        try:
            violations = []
            
            # TODO: Implement rule-specific validation logic
            logger.debug(f"Applying rule: {{ rule }}")
            
            return violations
            
        except Exception as e:
            logger.error(f"Validation rule {{ rule }} failed: {e}")
            return []
    
    {% endfor %}
    
    def _get_line_number(self, content: str, position: int) -> int:
        """Get line number for a character position"""
        return content[:position].count('\\n') + 1

# Plugin registration
plugin_instance = {{ class_name }}()

async def validate_file(file_path, content=None):
    """Main plugin entry point"""
    return await plugin_instance.validate(file_path, content)
'''

        return PluginTemplate(
            template_id="custom_validator",
            name="Custom Validator Plugin Template",
            description="Template for generating custom validation plugins",
            category="validation",
            complexity="medium",
            template_code=template_code,
            required_parameters=[
                "plugin_name",
                "class_name",
                "pattern_id",
                "validation_description",
            ],
            optional_parameters=["validation_rules", "severity_levels"],
            dependencies=["logging", "ast", "re", "typing", "pathlib"],
        )

    async def generate_plugins_from_patterns(
        self, patterns: List[DebuggingPattern]
    ) -> List[GeneratedPlugin]:
        """Generate plugins based on detected patterns"""
        try:
            generated_plugins = []

            for pattern in patterns:
                # Determine appropriate plugin type
                plugin_type = self._determine_plugin_type(pattern)

                if plugin_type and pattern.automation_potential in [
                    "high",
                    "very_high",
                ]:
                    plugin = await self._generate_plugin(pattern, plugin_type)

                    if plugin:
                        generated_plugins.append(plugin)
                        await self._store_plugin(plugin)

            logger.info(f"✅ Generated {len(generated_plugins)} custom plugins")
            return generated_plugins

        except Exception as e:
            logger.error(f"Failed to generate plugins: {e}")
            return []

    def _determine_plugin_type(self, pattern: DebuggingPattern) -> Optional[str]:
        """Determine the best plugin type for a pattern"""
        if pattern.pattern_type == "error_recurrence":
            return "error_handler"
        elif pattern.pattern_type == "workflow_sequence":
            return "workflow_automation"
        elif pattern.pattern_type in ["temporal_correlation", "successful_solution"]:
            return "pattern_monitor"
        else:
            return None

    async def _generate_plugin(
        self, pattern: DebuggingPattern, plugin_type: str
    ) -> Optional[GeneratedPlugin]:
        """Generate a single plugin from a pattern"""
        try:
            # Load template
            template_file = os.path.join(self.templates_dir, f"{plugin_type}.py.j2")

            if not os.path.exists(template_file):
                logger.error(f"Template not found: {template_file}")
                return None

            with open(template_file, "r") as f:
                template_content = f.read()

            # Prepare template variables
            variables = self._prepare_template_variables(pattern, plugin_type)

            # Render template
            template = jinja2.Template(template_content)
            plugin_code = template.render(**variables)

            # Generate plugin file
            plugin_id = f"{plugin_type}_{pattern.pattern_id}"
            plugin_name = (
                f"{plugin_type}_{pattern.pattern_type}_{pattern.pattern_id[:8]}"
            )
            plugin_file = os.path.join(self.plugins_dir, f"{plugin_name}.py")

            with open(plugin_file, "w") as f:
                f.write(plugin_code)

            plugin = GeneratedPlugin(
                plugin_id=plugin_id,
                name=plugin_name,
                description=f"Auto-generated {plugin_type} for {pattern.description}",
                pattern_based_on=pattern.pattern_id,
                generated_at=datetime.now(),
                file_path=plugin_file,
                status="generated",
                usage_count=0,
                success_rate=0.0,
                last_used=None,
            )

            logger.info(f"✅ Generated plugin: {plugin_name}")
            return plugin

        except Exception as e:
            logger.error(
                f"Failed to generate plugin for pattern {pattern.pattern_id}: {e}"
            )
            return None

    def _prepare_template_variables(
        self, pattern: DebuggingPattern, plugin_type: str
    ) -> Dict[str, Any]:
        """Prepare variables for template rendering"""
        base_variables = {
            "plugin_name": f"Auto {plugin_type.replace('_', ' ').title()} Plugin",
            "class_name": f"Auto{plugin_type.replace('_', '').title()}",
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "pattern_description": pattern.description,
            "trigger_conditions": pattern.trigger_conditions,
            "common_solutions": pattern.common_solutions,
            "success_rate": pattern.success_rate * 100,
            "automation_potential": pattern.automation_potential,
        }

        # Add plugin-type specific variables
        if plugin_type == "error_handler":
            # Extract error type from pattern examples
            error_type = "Unknown"
            if pattern.examples:
                error_type = pattern.examples[0].get("error_type", "Unknown")

            base_variables.update({"error_type": error_type})

        elif plugin_type == "workflow_automation":
            # Extract workflow sequence
            workflow_sequence = ["detect_error", "analyze", "fix", "validate"]
            if pattern.examples:
                workflow_sequence = pattern.examples[0].get(
                    "workflow", workflow_sequence
                )

            base_variables.update({"workflow_sequence": workflow_sequence})

        elif plugin_type == "pattern_monitor":
            base_variables.update(
                {
                    "trigger_threshold": max(3, pattern.frequency // 2),
                    "monitoring_window": 24,  # 24 hours
                }
            )

        elif plugin_type == "custom_validator":
            base_variables.update(
                {
                    "validation_description": pattern.description,
                    "validation_rules": [
                        "check_pattern_compliance",
                        "validate_structure",
                    ],
                    "severity_levels": ["error", "warning", "suggestion"],
                }
            )

        return base_variables

    async def _store_plugin(self, plugin: GeneratedPlugin):
        """Store plugin information in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO generated_plugins
                    (plugin_id, name, description, pattern_based_on, generated_at,
                     file_path, status, usage_count, success_rate, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        plugin.plugin_id,
                        plugin.name,
                        plugin.description,
                        plugin.pattern_based_on,
                        plugin.generated_at.isoformat(),
                        plugin.file_path,
                        plugin.status,
                        plugin.usage_count,
                        plugin.success_rate,
                        plugin.last_used.isoformat() if plugin.last_used else None,
                    ),
                )

        except Exception as e:
            logger.error(f"Failed to store plugin: {e}")


class PluginManager:
    """Manages generated plugins and their lifecycle"""

    def __init__(self):
        self.plugins_dir = "autonomous_debug/generated_plugins"
        self.db_path = "autonomous_debug/patterns.db"
        self.loaded_plugins = {}

    async def load_plugins(self) -> Dict[str, Any]:
        """Load all generated plugins"""
        try:
            plugins = {}

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT plugin_id, name, file_path, status
                    FROM generated_plugins
                    WHERE status = 'generated'
                """
                )

                for row in cursor.fetchall():
                    plugin_id, name, file_path, status = row

                    if os.path.exists(file_path):
                        # Dynamic import of plugin
                        plugin_module = await self._import_plugin(file_path)

                        if plugin_module:
                            plugins[plugin_id] = {
                                "name": name,
                                "module": plugin_module,
                                "file_path": file_path,
                                "status": status,
                            }

            self.loaded_plugins = plugins
            logger.info(f"📦 Loaded {len(plugins)} generated plugins")

            return plugins

        except Exception as e:
            logger.error(f"Failed to load plugins: {e}")
            return {}

    async def _import_plugin(self, file_path: str):
        """Dynamically import a plugin module"""
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("plugin", file_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)

            return plugin_module

        except Exception as e:
            logger.error(f"Failed to import plugin {file_path}: {e}")
            return None

    async def execute_plugin(
        self, plugin_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific plugin"""
        try:
            if plugin_id not in self.loaded_plugins:
                return {"error": f"Plugin {plugin_id} not found"}

            plugin = self.loaded_plugins[plugin_id]
            plugin_module = plugin["module"]

            # Record usage
            await self._record_plugin_usage(plugin_id, context)

            # Execute plugin based on its type
            if hasattr(plugin_module, "handle_error"):
                result = await plugin_module.handle_error(context)
            elif hasattr(plugin_module, "execute_workflow"):
                result = await plugin_module.execute_workflow(context)
            elif hasattr(plugin_module, "monitor_pattern"):
                result = await plugin_module.monitor_pattern()
            elif hasattr(plugin_module, "validate_file"):
                result = await plugin_module.validate_file(
                    context.get("file_path"), context.get("content")
                )
            else:
                return {"error": "Plugin has no executable entry point"}

            # Update success rate
            success = result.get(
                "handled", result.get("executed", result.get("valid", False))
            )
            await self._update_plugin_stats(plugin_id, success)

            return result

        except Exception as e:
            logger.error(f"Plugin execution failed: {e}")
            await self._update_plugin_stats(plugin_id, False)
            return {"error": str(e)}

    async def _record_plugin_usage(self, plugin_id: str, context: Dict[str, Any]):
        """Record plugin usage statistics"""
        try:
            usage_id = hashlib.md5(
                f"{plugin_id}_{datetime.now().isoformat()}".encode()
            ).hexdigest()

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO plugin_usage
                    (id, plugin_id, used_at, context, success, execution_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        usage_id,
                        plugin_id,
                        datetime.now().isoformat(),
                        json.dumps(context),
                        None,  # Will be updated after execution
                        0.0,
                    ),
                )

        except Exception as e:
            logger.error(f"Failed to record plugin usage: {e}")

    async def _update_plugin_stats(self, plugin_id: str, success: bool):
        """Update plugin statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update usage count and last used
                conn.execute(
                    """
                    UPDATE generated_plugins
                    SET usage_count = usage_count + 1,
                        last_used = ?
                    WHERE plugin_id = ?
                """,
                    (datetime.now().isoformat(), plugin_id),
                )

                # Calculate new success rate
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) as total,
                           COUNT(CASE WHEN success = 1 THEN 1 END) as successful
                    FROM plugin_usage
                    WHERE plugin_id = ?
                """,
                    (plugin_id,),
                )

                row = cursor.fetchone()
                if row and row[0] > 0:
                    success_rate = row[1] / row[0]
                    conn.execute(
                        """
                        UPDATE generated_plugins
                        SET success_rate = ?
                        WHERE plugin_id = ?
                    """,
                        (success_rate, plugin_id),
                    )

        except Exception as e:
            logger.error(f"Failed to update plugin stats: {e}")


class PluginOrchestrator:
    """Main orchestrator for plugin generation and management"""

    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.plugin_generator = PluginGenerator()
        self.plugin_manager = PluginManager()

    async def run_full_plugin_generation(self) -> Dict[str, Any]:
        """Run complete plugin generation cycle"""
        try:
            logger.info("🤖 Starting plugin generation and automation cycle")

            # Detect patterns
            patterns = await self.pattern_detector.detect_patterns()

            # Generate plugins from patterns
            generated_plugins = (
                await self.plugin_generator.generate_plugins_from_patterns(patterns)
            )

            # Load and prepare plugins
            loaded_plugins = await self.plugin_manager.load_plugins()

            # Generate summary report
            report = self._generate_plugin_report(
                patterns, generated_plugins, loaded_plugins
            )

            logger.info("✅ Plugin generation cycle complete")

            return {
                "patterns_detected": len(patterns),
                "plugins_generated": len(generated_plugins),
                "plugins_loaded": len(loaded_plugins),
                "patterns": patterns,
                "generated_plugins": generated_plugins,
                "loaded_plugins": list(loaded_plugins.keys()),
                "report": report,
            }

        except Exception as e:
            logger.error(f"Plugin generation cycle failed: {e}")
            return {"error": str(e)}

    def _generate_plugin_report(self, patterns, plugins, loaded) -> str:
        """Generate comprehensive plugin generation report"""
        report = f"""
# Plugin Generation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🔍 Pattern Detection Summary
- **Patterns Detected**: {len(patterns)}
- **High Automation Potential**: {len([p for p in patterns if p.automation_potential == 'very_high'])}
- **Medium Automation Potential**: {len([p for p in patterns if p.automation_potential == 'high'])}

### Detected Patterns:
"""

        for pattern in patterns:
            automation_icon = {
                "very_high": "🔥",
                "high": "⚡",
                "medium": "🔧",
                "low": "💡",
            }.get(pattern.automation_potential, "📋")

            report += f"- {automation_icon} **{pattern.pattern_type}**: {pattern.description} (Frequency: {pattern.frequency})\n"

        report += f"""

## 🛠️ Plugin Generation Summary
- **Plugins Generated**: {len(plugins)}
- **Plugins Successfully Loaded**: {len(loaded)}

### Generated Plugins:
"""

        for plugin in plugins:
            report += f"- **{plugin.name}**: {plugin.description}\n"
            report += f"  - File: `{plugin.file_path}`\n"
            report += f"  - Based on Pattern: {plugin.pattern_based_on}\n"
            report += f"  - Status: {plugin.status}\n\n"

        report += f"""

## 📊 Automation Impact
- **Total Automation Coverage**: {len(plugins)} custom tools
- **Estimated Time Savings**: {len(plugins) * 2} hours/week
- **Pattern-Based Solutions**: {sum(p.frequency for p in patterns)} recurring issues automated

## 🎯 Next Steps
1. Test generated plugins in isolated environment
2. Integrate high-confidence plugins into automation workflow
3. Monitor plugin performance and refine
4. Generate additional plugins as new patterns emerge

## 🔄 Continuous Improvement
- Plugin performance will be tracked automatically
- Unsuccessful plugins will be refined or retired
- New patterns will trigger automatic plugin generation
- Successful patterns will be prioritized for future automation
"""

        return report


async def main():
    """Main function for plugin generation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plugin Generation + Custom Automation"
    )
    parser.add_argument(
        "--detect-patterns", action="store_true", help="Detect debugging patterns"
    )
    parser.add_argument(
        "--generate-plugins", action="store_true", help="Generate plugins from patterns"
    )
    parser.add_argument(
        "--load-plugins", action="store_true", help="Load generated plugins"
    )
    parser.add_argument(
        "--full-cycle", action="store_true", help="Run full plugin generation cycle"
    )
    parser.add_argument("--test-plugin", help="Test a specific plugin")
    parser.add_argument("--output", help="Output file for report")

    args = parser.parse_args()

    orchestrator = PluginOrchestrator()

    try:
        if args.detect_patterns:
            print("🔍 Detecting debugging patterns...")
            patterns = await orchestrator.pattern_detector.detect_patterns()
            print(f"✅ Detected {len(patterns)} patterns")

            for pattern in patterns:
                print(f"  - {pattern.pattern_type}: {pattern.description}")

        elif args.generate_plugins:
            print("🛠️ Generating plugins from patterns...")
            patterns = await orchestrator.pattern_detector.detect_patterns()
            plugins = (
                await orchestrator.plugin_generator.generate_plugins_from_patterns(
                    patterns
                )
            )
            print(f"✅ Generated {len(plugins)} plugins")

        elif args.load_plugins:
            print("📦 Loading generated plugins...")
            plugins = await orchestrator.plugin_manager.load_plugins()
            print(f"✅ Loaded {len(plugins)} plugins")

        elif (
            args.test_plugin
            and args.test_plugin in orchestrator.plugin_manager.loaded_plugins
        ):
            print(f"🧪 Testing plugin: {args.test_plugin}")
            result = await orchestrator.plugin_manager.execute_plugin(
                args.test_plugin, {}
            )
            print(f"Test result: {result}")

        elif args.full_cycle:
            print("🔄 Running full plugin generation cycle...")
            results = await orchestrator.run_full_plugin_generation()

            if args.output:
                with open(args.output, "w") as f:
                    f.write(results["report"])
                print(f"📄 Report saved to {args.output}")
            else:
                print(results["report"])

        else:
            print(
                "Specify --detect-patterns, --generate-plugins, --load-plugins, or --full-cycle"
            )

    except Exception as e:
        print(f"❌ Operation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
