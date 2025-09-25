"""
Vega 2.0 Document Workflow Automation Module

This module provides intelligent document workflow automation including:
- Smart document routing based on content and metadata
- Automated approval workflows with role-based access control
- Document processing pipelines with conditional logic
- Integration hooks for external systems and APIs
- Workflow monitoring, analytics, and optimization
- Event-driven processing with triggers and actions

Dependencies:
- asyncio: Asynchronous workflow execution
- json, yaml: Configuration and data serialization
- typing: Type hints for workflow definitions
- dataclasses: Structured workflow configurations
- pathlib: File system operations
- datetime: Timestamp and scheduling
"""

import asyncio
import logging
import json
import yaml
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Awaitable
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import hashlib

try:
    from croniter import croniter

    HAS_CRONITER = True
except ImportError:
    croniter = None
    HAS_CRONITER = False

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False

try:
    from pydantic import BaseModel, Field, validator

    HAS_PYDANTIC = True
except ImportError:
    BaseModel = object
    Field = validator = None
    HAS_PYDANTIC = False

logger = logging.getLogger(__name__)


class WorkflowError(Exception):
    """Custom exception for workflow errors"""

    pass


class WorkflowState(Enum):
    """Workflow execution states"""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ActionType(Enum):
    """Types of workflow actions"""

    CLASSIFY = "classify"
    ROUTE = "route"
    APPROVE = "approve"
    REJECT = "reject"
    EXTRACT = "extract"
    TRANSFORM = "transform"
    NOTIFY = "notify"
    WEBHOOK = "webhook"
    EMAIL = "email"
    API_CALL = "api_call"
    STORE = "store"
    ARCHIVE = "archive"
    DELETE = "delete"
    CUSTOM = "custom"


class TriggerType(Enum):
    """Types of workflow triggers"""

    DOCUMENT_UPLOADED = "document_uploaded"
    CLASSIFICATION_COMPLETE = "classification_complete"
    APPROVAL_NEEDED = "approval_needed"
    DEADLINE_APPROACHING = "deadline_approaching"
    SCHEDULE = "schedule"
    WEBHOOK_RECEIVED = "webhook_received"
    MANUAL = "manual"
    CONDITION_MET = "condition_met"


class Priority(Enum):
    """Task priority levels"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class WorkflowContext:
    """Context passed through workflow execution"""

    workflow_id: str
    document_id: str
    document_path: Optional[str] = None
    document_content: Optional[str] = None
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    classification_result: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ActionConfig:
    """Configuration for workflow actions"""

    action_type: ActionType
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 3
    retry_delay: int = 30
    on_success: Optional[str] = None
    on_failure: Optional[str] = None
    enabled: bool = True


@dataclass
class TriggerConfig:
    """Configuration for workflow triggers"""

    trigger_type: TriggerType
    name: str
    description: str = ""
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    schedule: Optional[str] = None  # Cron expression
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalConfig:
    """Configuration for approval workflows"""

    name: str
    description: str = ""
    approvers: List[str] = field(default_factory=list)
    approval_threshold: int = 1  # Number of approvals needed
    rejection_threshold: int = 1  # Number of rejections needed
    timeout_hours: int = 24
    escalation_hours: Optional[int] = None
    escalation_approvers: List[str] = field(default_factory=list)
    auto_approve_conditions: List[Dict[str, Any]] = field(default_factory=list)
    auto_reject_conditions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RoutingRule:
    """Document routing rules"""

    name: str
    description: str = ""
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    destination: str = ""
    priority: Priority = Priority.NORMAL
    metadata_updates: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""

    name: str
    description: str = ""
    version: str = "1.0.0"
    triggers: List[TriggerConfig] = field(default_factory=list)
    actions: List[ActionConfig] = field(default_factory=list)
    routing_rules: List[RoutingRule] = field(default_factory=list)
    approval_configs: List[ApprovalConfig] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    timeout_hours: int = 24
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowExecution:
    """Workflow execution instance"""

    execution_id: str
    workflow_name: str
    context: WorkflowContext
    state: WorkflowState = WorkflowState.PENDING
    current_action: Optional[str] = None
    completed_actions: List[str] = field(default_factory=list)
    failed_actions: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_at: Optional[datetime] = None


@dataclass
class ApprovalRequest:
    """Approval request instance"""

    request_id: str
    workflow_execution_id: str
    approval_config_name: str
    document_id: str
    requester_id: str
    approvers: List[str]
    approval_threshold: int
    rejection_threshold: int
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    rejections: List[Dict[str, Any]] = field(default_factory=list)
    state: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    timeout_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ConditionEvaluator:
    """
    Evaluates workflow conditions
    """

    @staticmethod
    def evaluate_condition(condition: Dict[str, Any], context: WorkflowContext) -> bool:
        """
        Evaluate a single condition

        Args:
            condition: Condition configuration
            context: Workflow context

        Returns:
            True if condition is met
        """
        try:
            condition_type = condition.get("type", "equals")
            field = condition.get("field", "")
            value = condition.get("value", "")
            operator = condition.get("operator", "equals")

            # Get field value from context
            field_value = ConditionEvaluator._get_field_value(field, context)

            # Evaluate based on operator
            if operator == "equals":
                return field_value == value
            elif operator == "not_equals":
                return field_value != value
            elif operator == "contains":
                return str(value).lower() in str(field_value).lower()
            elif operator == "not_contains":
                return str(value).lower() not in str(field_value).lower()
            elif operator == "starts_with":
                return str(field_value).lower().startswith(str(value).lower())
            elif operator == "ends_with":
                return str(field_value).lower().endswith(str(value).lower())
            elif operator == "greater_than":
                return float(field_value) > float(value)
            elif operator == "less_than":
                return float(field_value) < float(value)
            elif operator == "greater_equal":
                return float(field_value) >= float(value)
            elif operator == "less_equal":
                return float(field_value) <= float(value)
            elif operator == "in":
                return field_value in value if isinstance(value, list) else False
            elif operator == "not_in":
                return field_value not in value if isinstance(value, list) else True
            elif operator == "regex":
                import re

                pattern = re.compile(str(value), re.IGNORECASE)
                return bool(pattern.search(str(field_value)))
            elif operator == "exists":
                return field_value is not None
            elif operator == "not_exists":
                return field_value is None
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False

        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False

    @staticmethod
    def evaluate_conditions(
        conditions: List[Dict[str, Any]], context: WorkflowContext, logic: str = "and"
    ) -> bool:
        """
        Evaluate multiple conditions with AND/OR logic

        Args:
            conditions: List of condition configurations
            context: Workflow context
            logic: 'and' or 'or' logic

        Returns:
            True if conditions are met
        """
        try:
            if not conditions:
                return True

            results = [
                ConditionEvaluator.evaluate_condition(cond, context)
                for cond in conditions
            ]

            if logic.lower() == "or":
                return any(results)
            else:  # Default to 'and'
                return all(results)

        except Exception as e:
            logger.error(f"Conditions evaluation error: {e}")
            return False

    @staticmethod
    def _get_field_value(field: str, context: WorkflowContext) -> Any:
        """Get field value from context using dot notation"""
        try:
            if not field:
                return None

            # Handle dot notation (e.g., 'metadata.category')
            parts = field.split(".")
            value = context

            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None

            return value

        except Exception as e:
            logger.debug(f"Field value extraction error: {e}")
            return None


class ActionExecutor:
    """
    Executes workflow actions
    """

    def __init__(self):
        self.custom_actions: Dict[str, Callable] = {}

    def register_custom_action(
        self,
        name: str,
        handler: Callable[[Dict[str, Any], WorkflowContext], Awaitable[Dict[str, Any]]],
    ):
        """Register custom action handler"""
        self.custom_actions[name] = handler

    async def execute_action(
        self, action_config: ActionConfig, context: WorkflowContext
    ) -> Dict[str, Any]:
        """
        Execute a workflow action

        Args:
            action_config: Action configuration
            context: Workflow context

        Returns:
            Action execution results
        """
        try:
            # Check conditions
            if action_config.conditions:
                if not ConditionEvaluator.evaluate_conditions(
                    action_config.conditions, context
                ):
                    return {
                        "success": True,
                        "skipped": True,
                        "reason": "conditions_not_met",
                    }

            # Execute action based on type
            action_type = action_config.action_type

            if action_type == ActionType.CLASSIFY:
                return await self._execute_classify_action(action_config, context)
            elif action_type == ActionType.ROUTE:
                return await self._execute_route_action(action_config, context)
            elif action_type == ActionType.APPROVE:
                return await self._execute_approve_action(action_config, context)
            elif action_type == ActionType.NOTIFY:
                return await self._execute_notify_action(action_config, context)
            elif action_type == ActionType.WEBHOOK:
                return await self._execute_webhook_action(action_config, context)
            elif action_type == ActionType.EMAIL:
                return await self._execute_email_action(action_config, context)
            elif action_type == ActionType.API_CALL:
                return await self._execute_api_call_action(action_config, context)
            elif action_type == ActionType.STORE:
                return await self._execute_store_action(action_config, context)
            elif action_type == ActionType.TRANSFORM:
                return await self._execute_transform_action(action_config, context)
            elif action_type == ActionType.CUSTOM:
                return await self._execute_custom_action(action_config, context)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_type}",
                }

        except Exception as e:
            logger.error(f"Action execution error: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_classify_action(
        self, action_config: ActionConfig, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute document classification action"""
        try:
            # This would integrate with the DocumentClassifier
            # For demo, return mock classification

            classification_result = {
                "category": "contract",
                "confidence": 0.92,
                "probabilities": {"contract": 0.92, "legal": 0.85, "business": 0.78},
            }

            # Update context
            context.classification_result = classification_result
            context.variables["classification"] = classification_result

            return {"success": True, "result": classification_result}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_route_action(
        self, action_config: ActionConfig, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute document routing action"""
        try:
            destination = action_config.parameters.get("destination", "default")

            # Update context metadata
            context.document_metadata["routed_to"] = destination
            context.document_metadata["routed_at"] = datetime.now().isoformat()

            return {
                "success": True,
                "destination": destination,
                "routed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_approve_action(
        self, action_config: ActionConfig, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute approval action"""
        try:
            # Create approval request
            approval_request = {
                "request_id": str(uuid.uuid4()),
                "document_id": context.document_id,
                "workflow_id": context.workflow_id,
                "approvers": action_config.parameters.get("approvers", []),
                "created_at": datetime.now().isoformat(),
            }

            # In a real implementation, this would create an approval request
            # and wait for approver responses

            return {
                "success": True,
                "approval_request_id": approval_request["request_id"],
                "status": "pending_approval",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_notify_action(
        self, action_config: ActionConfig, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute notification action"""
        try:
            recipients = action_config.parameters.get("recipients", [])
            message = action_config.parameters.get("message", "Workflow notification")

            # Template substitution
            message = self._substitute_template_variables(message, context)

            # Log notification (in production, would send actual notifications)
            logger.info(f"Notification sent to {recipients}: {message}")

            return {
                "success": True,
                "recipients": recipients,
                "message": message,
                "sent_at": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_webhook_action(
        self, action_config: ActionConfig, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute webhook action"""
        try:
            if not HAS_AIOHTTP:
                return {"success": False, "error": "aiohttp not available for webhooks"}

            url = action_config.parameters.get("url", "")
            method = action_config.parameters.get("method", "POST").upper()
            headers = action_config.parameters.get("headers", {})
            payload = action_config.parameters.get("payload", {})

            # Substitute template variables in payload
            payload_str = json.dumps(payload)
            payload_str = self._substitute_template_variables(payload_str, context)
            payload = json.loads(payload_str)

            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, json=payload, headers=headers
                ) as response:
                    response_data = await response.text()

                    return {
                        "success": response.status < 400,
                        "status_code": response.status,
                        "response": response_data,
                        "sent_at": datetime.now().isoformat(),
                    }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_email_action(
        self, action_config: ActionConfig, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute email action"""
        try:
            recipients = action_config.parameters.get("recipients", [])
            subject = action_config.parameters.get("subject", "Workflow Notification")
            body = action_config.parameters.get(
                "body", "This is a workflow notification."
            )

            # Template substitution
            subject = self._substitute_template_variables(subject, context)
            body = self._substitute_template_variables(body, context)

            # Log email (in production, would send actual emails)
            logger.info(f"Email sent to {recipients}: {subject}")

            return {
                "success": True,
                "recipients": recipients,
                "subject": subject,
                "sent_at": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_api_call_action(
        self, action_config: ActionConfig, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute API call action"""
        try:
            if not HAS_AIOHTTP:
                return {
                    "success": False,
                    "error": "aiohttp not available for API calls",
                }

            url = action_config.parameters.get("url", "")
            method = action_config.parameters.get("method", "GET").upper()
            headers = action_config.parameters.get("headers", {})
            params = action_config.parameters.get("params", {})
            data = action_config.parameters.get("data", {})

            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, headers=headers, params=params, json=data
                ) as response:
                    response_data = await response.json()

                    return {
                        "success": response.status < 400,
                        "status_code": response.status,
                        "data": response_data,
                        "called_at": datetime.now().isoformat(),
                    }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_store_action(
        self, action_config: ActionConfig, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute document storage action"""
        try:
            storage_path = action_config.parameters.get("path", "/default/storage/")
            metadata = action_config.parameters.get("metadata", {})

            # Update context with storage information
            context.document_metadata.update(
                {
                    "stored_at": datetime.now().isoformat(),
                    "storage_path": storage_path,
                    "storage_metadata": metadata,
                }
            )

            return {
                "success": True,
                "storage_path": storage_path,
                "stored_at": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_transform_action(
        self, action_config: ActionConfig, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute document transformation action"""
        try:
            transformation = action_config.parameters.get("transformation", "none")

            # Mock transformation
            result = {
                "original_format": "pdf",
                "target_format": "text",
                "transformation": transformation,
                "transformed_at": datetime.now().isoformat(),
            }

            context.variables["transformation_result"] = result

            return {"success": True, "result": result}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_custom_action(
        self, action_config: ActionConfig, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute custom action"""
        try:
            action_name = action_config.parameters.get("handler", "")

            if action_name in self.custom_actions:
                handler = self.custom_actions[action_name]
                return await handler(action_config.parameters, context)
            else:
                return {
                    "success": False,
                    "error": f"Custom action handler not found: {action_name}",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _substitute_template_variables(
        self, text: str, context: WorkflowContext
    ) -> str:
        """Substitute template variables in text"""
        try:
            # Simple template substitution
            variables = {
                "workflow_id": context.workflow_id,
                "document_id": context.document_id,
                "user_id": context.user_id or "unknown",
                "created_at": context.created_at.isoformat(),
                "updated_at": context.updated_at.isoformat(),
            }

            # Add context variables
            variables.update(context.variables)

            # Add metadata
            if context.document_metadata:
                for key, value in context.document_metadata.items():
                    variables[f"metadata.{key}"] = value

            # Simple string substitution
            for key, value in variables.items():
                placeholder = f"{{{key}}}"
                text = text.replace(placeholder, str(value))

            return text

        except Exception as e:
            logger.debug(f"Template substitution error: {e}")
            return text


class DocumentRouter:
    """
    Smart document routing based on rules
    """

    def __init__(self):
        self.routing_rules: List[RoutingRule] = []

    def add_routing_rule(self, rule: RoutingRule):
        """Add a routing rule"""
        self.routing_rules.append(rule)

    async def route_document(self, context: WorkflowContext) -> Optional[RoutingRule]:
        """
        Route document based on rules

        Args:
            context: Workflow context

        Returns:
            Matching routing rule or None
        """
        try:
            # Sort rules by priority
            sorted_rules = sorted(
                [r for r in self.routing_rules if r.enabled],
                key=lambda x: x.priority.value,
                reverse=True,
            )

            for rule in sorted_rules:
                if ConditionEvaluator.evaluate_conditions(rule.conditions, context):
                    logger.info(f"Document routed using rule: {rule.name}")

                    # Apply metadata updates
                    if rule.metadata_updates:
                        context.document_metadata.update(rule.metadata_updates)

                    return rule

            return None

        except Exception as e:
            logger.error(f"Document routing error: {e}")
            return None


class ApprovalManager:
    """
    Manages approval workflows
    """

    def __init__(self):
        self.approval_requests: Dict[str, ApprovalRequest] = {}

    async def create_approval_request(
        self, config: ApprovalConfig, context: WorkflowContext
    ) -> ApprovalRequest:
        """Create a new approval request"""
        try:
            request = ApprovalRequest(
                request_id=str(uuid.uuid4()),
                workflow_execution_id=context.workflow_id,
                approval_config_name=config.name,
                document_id=context.document_id,
                requester_id=context.user_id or "system",
                approvers=config.approvers,
                approval_threshold=config.approval_threshold,
                rejection_threshold=config.rejection_threshold,
                timeout_at=datetime.now() + timedelta(hours=config.timeout_hours),
            )

            self.approval_requests[request.request_id] = request

            return request

        except Exception as e:
            logger.error(f"Approval request creation error: {e}")
            raise WorkflowError(f"Failed to create approval request: {e}")

    async def process_approval_response(
        self, request_id: str, approver_id: str, decision: str, comments: str = ""
    ) -> Dict[str, Any]:
        """Process approval response"""
        try:
            if request_id not in self.approval_requests:
                raise WorkflowError(f"Approval request not found: {request_id}")

            request = self.approval_requests[request_id]

            if request.state != "pending":
                raise WorkflowError(
                    f"Approval request not in pending state: {request.state}"
                )

            if approver_id not in request.approvers:
                raise WorkflowError(f"User not authorized to approve: {approver_id}")

            # Record response
            response = {
                "approver_id": approver_id,
                "decision": decision,
                "comments": comments,
                "timestamp": datetime.now().isoformat(),
            }

            if decision.lower() == "approve":
                request.approvals.append(response)
            elif decision.lower() == "reject":
                request.rejections.append(response)
            else:
                raise WorkflowError(f"Invalid decision: {decision}")

            # Check if approval is complete
            result = self._check_approval_completion(request)

            return result

        except Exception as e:
            logger.error(f"Approval response processing error: {e}")
            return {"success": False, "error": str(e)}

    def _check_approval_completion(self, request: ApprovalRequest) -> Dict[str, Any]:
        """Check if approval request is complete"""
        try:
            approval_count = len(request.approvals)
            rejection_count = len(request.rejections)

            if approval_count >= request.approval_threshold:
                request.state = "approved"
                request.completed_at = datetime.now()
                return {
                    "success": True,
                    "status": "approved",
                    "approval_count": approval_count,
                    "rejection_count": rejection_count,
                }
            elif rejection_count >= request.rejection_threshold:
                request.state = "rejected"
                request.completed_at = datetime.now()
                return {
                    "success": True,
                    "status": "rejected",
                    "approval_count": approval_count,
                    "rejection_count": rejection_count,
                }
            else:
                return {
                    "success": True,
                    "status": "pending",
                    "approval_count": approval_count,
                    "rejection_count": rejection_count,
                    "approvals_needed": request.approval_threshold - approval_count,
                }

        except Exception as e:
            logger.error(f"Approval completion check error: {e}")
            return {"success": False, "error": str(e)}


class WorkflowScheduler:
    """
    Schedules and triggers workflows
    """

    def __init__(self):
        self.scheduled_workflows: List[Dict[str, Any]] = []
        self.running = False

    def schedule_workflow(
        self, workflow_name: str, cron_expression: str, context_template: Dict[str, Any]
    ):
        """Schedule a workflow to run on cron schedule"""
        try:
            if not HAS_CRONITER:
                logger.warning("croniter not available for scheduling")
                return False

            # Validate cron expression
            try:
                cron = croniter(cron_expression)
                next_run = cron.get_next(datetime)
            except Exception as e:
                logger.error(f"Invalid cron expression: {cron_expression}")
                return False

            scheduled_workflow = {
                "workflow_name": workflow_name,
                "cron_expression": cron_expression,
                "context_template": context_template,
                "next_run": next_run,
                "enabled": True,
            }

            self.scheduled_workflows.append(scheduled_workflow)

            logger.info(
                f"Scheduled workflow '{workflow_name}' with cron '{cron_expression}'"
            )
            return True

        except Exception as e:
            logger.error(f"Workflow scheduling error: {e}")
            return False

    async def start_scheduler(self, workflow_manager):
        """Start the workflow scheduler"""
        self.running = True

        while self.running:
            try:
                current_time = datetime.now()

                for scheduled in self.scheduled_workflows:
                    if scheduled["enabled"] and scheduled["next_run"] <= current_time:

                        # Trigger workflow
                        await self._trigger_scheduled_workflow(
                            scheduled, workflow_manager
                        )

                        # Calculate next run time
                        if HAS_CRONITER:
                            cron = croniter(scheduled["cron_expression"])
                            scheduled["next_run"] = cron.get_next(datetime)

                # Sleep for 1 minute
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)

    async def _trigger_scheduled_workflow(
        self, scheduled: Dict[str, Any], workflow_manager
    ):
        """Trigger a scheduled workflow"""
        try:
            context_data = scheduled["context_template"].copy()
            context_data.update(
                {
                    "workflow_id": str(uuid.uuid4()),
                    "document_id": f"scheduled_{int(datetime.now().timestamp())}",
                    "triggered_by": "scheduler",
                    "scheduled_at": datetime.now().isoformat(),
                }
            )

            context = WorkflowContext(**context_data)

            await workflow_manager.execute_workflow(scheduled["workflow_name"], context)

            logger.info(f"Triggered scheduled workflow: {scheduled['workflow_name']}")

        except Exception as e:
            logger.error(f"Scheduled workflow trigger error: {e}")

    def stop_scheduler(self):
        """Stop the workflow scheduler"""
        self.running = False


class WorkflowManager:
    """
    Main workflow management system
    """

    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}

        # Components
        self.action_executor = ActionExecutor()
        self.document_router = DocumentRouter()
        self.approval_manager = ApprovalManager()
        self.scheduler = WorkflowScheduler()

        # Configuration
        self.workflow_config_dir = Path("./workflows")
        self.workflow_config_dir.mkdir(exist_ok=True)

    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a workflow definition"""
        self.workflows[workflow.name] = workflow
        logger.info(f"Registered workflow: {workflow.name}")

    async def load_workflows_from_directory(self, directory: Union[str, Path]) -> int:
        """Load workflow definitions from directory"""
        try:
            directory = Path(directory)
            loaded_count = 0

            for file_path in directory.glob("*.yaml"):
                try:
                    with open(file_path, "r") as f:
                        workflow_data = yaml.safe_load(f)

                    workflow = self._create_workflow_from_dict(workflow_data)
                    self.register_workflow(workflow)
                    loaded_count += 1

                except Exception as e:
                    logger.error(f"Failed to load workflow from {file_path}: {e}")

            for file_path in directory.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        workflow_data = json.load(f)

                    workflow = self._create_workflow_from_dict(workflow_data)
                    self.register_workflow(workflow)
                    loaded_count += 1

                except Exception as e:
                    logger.error(f"Failed to load workflow from {file_path}: {e}")

            logger.info(f"Loaded {loaded_count} workflows from {directory}")
            return loaded_count

        except Exception as e:
            logger.error(f"Workflow loading error: {e}")
            return 0

    async def execute_workflow(
        self, workflow_name: str, context: WorkflowContext
    ) -> WorkflowExecution:
        """
        Execute a workflow

        Args:
            workflow_name: Name of workflow to execute
            context: Workflow context

        Returns:
            Workflow execution instance
        """
        try:
            if workflow_name not in self.workflows:
                raise WorkflowError(f"Workflow not found: {workflow_name}")

            workflow = self.workflows[workflow_name]

            if not workflow.enabled:
                raise WorkflowError(f"Workflow disabled: {workflow_name}")

            # Create execution
            execution = WorkflowExecution(
                execution_id=str(uuid.uuid4()),
                workflow_name=workflow_name,
                context=context,
                state=WorkflowState.RUNNING,
                started_at=datetime.now(),
                timeout_at=datetime.now() + timedelta(hours=workflow.timeout_hours),
            )

            self.executions[execution.execution_id] = execution

            logger.info(f"Starting workflow execution: {execution.execution_id}")

            # Execute workflow steps
            try:
                # Execute actions in order
                for action_config in workflow.actions:
                    if not action_config.enabled:
                        continue

                    execution.current_action = action_config.name

                    logger.info(f"Executing action: {action_config.name}")

                    # Execute action with timeout and retry
                    result = await self._execute_action_with_retry(
                        action_config, context
                    )

                    execution.results[action_config.name] = result

                    if result.get("success", False):
                        execution.completed_actions.append(action_config.name)
                    else:
                        execution.failed_actions.append(action_config.name)

                        # Handle failure
                        if action_config.on_failure:
                            logger.info(
                                f"Action failed, executing failure handler: {action_config.on_failure}"
                            )
                        else:
                            # Stop execution on failure
                            execution.state = WorkflowState.FAILED
                            execution.error_message = result.get(
                                "error", "Action failed"
                            )
                            execution.completed_at = datetime.now()
                            return execution

                # Workflow completed successfully
                execution.state = WorkflowState.COMPLETED
                execution.completed_at = datetime.now()
                execution.current_action = None

                logger.info(f"Workflow execution completed: {execution.execution_id}")

            except asyncio.TimeoutError:
                execution.state = WorkflowState.TIMEOUT
                execution.error_message = "Workflow execution timeout"
                execution.completed_at = datetime.now()

            except Exception as e:
                execution.state = WorkflowState.FAILED
                execution.error_message = str(e)
                execution.completed_at = datetime.now()
                logger.error(f"Workflow execution error: {e}")

            return execution

        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            raise WorkflowError(f"Failed to execute workflow: {e}")

    async def _execute_action_with_retry(
        self, action_config: ActionConfig, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute action with retry logic"""
        try:
            for attempt in range(action_config.retry_count + 1):
                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        self.action_executor.execute_action(action_config, context),
                        timeout=action_config.timeout_seconds,
                    )

                    if result.get("success", False):
                        return result

                    # If not successful and retries available
                    if attempt < action_config.retry_count:
                        logger.info(
                            f"Action failed, retrying in {action_config.retry_delay}s (attempt {attempt + 1})"
                        )
                        await asyncio.sleep(action_config.retry_delay)
                    else:
                        return result

                except asyncio.TimeoutError:
                    if attempt < action_config.retry_count:
                        logger.info(
                            f"Action timeout, retrying in {action_config.retry_delay}s (attempt {attempt + 1})"
                        )
                        await asyncio.sleep(action_config.retry_delay)
                    else:
                        return {"success": False, "error": "Action timeout"}

            return {"success": False, "error": "Max retries exceeded"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_workflow_from_dict(self, data: Dict[str, Any]) -> WorkflowDefinition:
        """Create workflow definition from dictionary"""
        try:
            # Parse triggers
            triggers = []
            for trigger_data in data.get("triggers", []):
                trigger = TriggerConfig(
                    trigger_type=TriggerType(trigger_data["trigger_type"]),
                    name=trigger_data["name"],
                    description=trigger_data.get("description", ""),
                    conditions=trigger_data.get("conditions", []),
                    schedule=trigger_data.get("schedule"),
                    enabled=trigger_data.get("enabled", True),
                    parameters=trigger_data.get("parameters", {}),
                )
                triggers.append(trigger)

            # Parse actions
            actions = []
            for action_data in data.get("actions", []):
                action = ActionConfig(
                    action_type=ActionType(action_data["action_type"]),
                    name=action_data["name"],
                    description=action_data.get("description", ""),
                    parameters=action_data.get("parameters", {}),
                    conditions=action_data.get("conditions", []),
                    timeout_seconds=action_data.get("timeout_seconds", 300),
                    retry_count=action_data.get("retry_count", 3),
                    retry_delay=action_data.get("retry_delay", 30),
                    on_success=action_data.get("on_success"),
                    on_failure=action_data.get("on_failure"),
                    enabled=action_data.get("enabled", True),
                )
                actions.append(action)

            # Parse routing rules
            routing_rules = []
            for rule_data in data.get("routing_rules", []):
                rule = RoutingRule(
                    name=rule_data["name"],
                    description=rule_data.get("description", ""),
                    conditions=rule_data.get("conditions", []),
                    destination=rule_data.get("destination", ""),
                    priority=Priority(rule_data.get("priority", 2)),
                    metadata_updates=rule_data.get("metadata_updates", {}),
                    actions=rule_data.get("actions", []),
                    enabled=rule_data.get("enabled", True),
                )
                routing_rules.append(rule)

            # Parse approval configs
            approval_configs = []
            for approval_data in data.get("approval_configs", []):
                approval = ApprovalConfig(
                    name=approval_data["name"],
                    description=approval_data.get("description", ""),
                    approvers=approval_data.get("approvers", []),
                    approval_threshold=approval_data.get("approval_threshold", 1),
                    rejection_threshold=approval_data.get("rejection_threshold", 1),
                    timeout_hours=approval_data.get("timeout_hours", 24),
                    escalation_hours=approval_data.get("escalation_hours"),
                    escalation_approvers=approval_data.get("escalation_approvers", []),
                    auto_approve_conditions=approval_data.get(
                        "auto_approve_conditions", []
                    ),
                    auto_reject_conditions=approval_data.get(
                        "auto_reject_conditions", []
                    ),
                )
                approval_configs.append(approval)

            # Create workflow
            workflow = WorkflowDefinition(
                name=data["name"],
                description=data.get("description", ""),
                version=data.get("version", "1.0.0"),
                triggers=triggers,
                actions=actions,
                routing_rules=routing_rules,
                approval_configs=approval_configs,
                variables=data.get("variables", {}),
                timeout_hours=data.get("timeout_hours", 24),
                enabled=data.get("enabled", True),
            )

            return workflow

        except Exception as e:
            logger.error(f"Workflow creation error: {e}")
            raise WorkflowError(f"Failed to create workflow from data: {e}")

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status"""
        try:
            if execution_id not in self.executions:
                return None

            execution = self.executions[execution_id]

            return {
                "execution_id": execution.execution_id,
                "workflow_name": execution.workflow_name,
                "state": execution.state.value,
                "current_action": execution.current_action,
                "completed_actions": execution.completed_actions,
                "failed_actions": execution.failed_actions,
                "progress": len(execution.completed_actions)
                / max(
                    len(execution.completed_actions) + len(execution.failed_actions), 1
                )
                * 100,
                "started_at": (
                    execution.started_at.isoformat() if execution.started_at else None
                ),
                "completed_at": (
                    execution.completed_at.isoformat()
                    if execution.completed_at
                    else None
                ),
                "error_message": execution.error_message,
            }

        except Exception as e:
            logger.error(f"Execution status error: {e}")
            return None

    def create_demo_workflow(self) -> Dict[str, Any]:
        """Create demo workflow configuration"""
        try:
            demo_workflow_data = {
                "name": "contract_processing_workflow",
                "description": "Automated contract processing workflow with classification, routing, and approval",
                "version": "1.2.0",
                "triggers": [
                    {
                        "trigger_type": "document_uploaded",
                        "name": "contract_upload_trigger",
                        "description": "Trigger when contract document is uploaded",
                        "conditions": [
                            {
                                "field": "document_metadata.category",
                                "operator": "equals",
                                "value": "contract",
                            }
                        ],
                        "enabled": True,
                    }
                ],
                "actions": [
                    {
                        "action_type": "classify",
                        "name": "classify_contract",
                        "description": "Classify the contract document",
                        "parameters": {"confidence_threshold": 0.8},
                        "timeout_seconds": 120,
                        "enabled": True,
                    },
                    {
                        "action_type": "route",
                        "name": "route_to_legal",
                        "description": "Route contract to legal department",
                        "parameters": {"destination": "legal_department"},
                        "conditions": [
                            {
                                "field": "variables.classification.confidence",
                                "operator": "greater_than",
                                "value": 0.8,
                            }
                        ],
                        "enabled": True,
                    },
                    {
                        "action_type": "approve",
                        "name": "legal_approval",
                        "description": "Request legal approval for contract",
                        "parameters": {
                            "approvers": ["legal_manager", "compliance_officer"],
                            "approval_threshold": 1,
                            "timeout_hours": 48,
                        },
                        "enabled": True,
                    },
                    {
                        "action_type": "notify",
                        "name": "completion_notification",
                        "description": "Notify stakeholders of workflow completion",
                        "parameters": {
                            "recipients": ["document_owner", "legal_team"],
                            "message": "Contract processing workflow completed for document {document_id}",
                            "channel": "email",
                        },
                        "enabled": True,
                    },
                ],
                "routing_rules": [
                    {
                        "name": "high_value_contracts",
                        "description": "Route high-value contracts to senior management",
                        "conditions": [
                            {
                                "field": "document_metadata.contract_value",
                                "operator": "greater_than",
                                "value": 100000,
                            }
                        ],
                        "destination": "senior_management",
                        "priority": 4,
                        "metadata_updates": {
                            "priority": "high",
                            "requires_executive_approval": True,
                        },
                        "enabled": True,
                    },
                    {
                        "name": "standard_contracts",
                        "description": "Route standard contracts to legal team",
                        "conditions": [
                            {
                                "field": "document_metadata.contract_value",
                                "operator": "less_equal",
                                "value": 100000,
                            }
                        ],
                        "destination": "legal_team",
                        "priority": 2,
                        "enabled": True,
                    },
                ],
                "approval_configs": [
                    {
                        "name": "legal_review",
                        "description": "Legal department contract review",
                        "approvers": ["legal_manager", "senior_lawyer"],
                        "approval_threshold": 1,
                        "rejection_threshold": 1,
                        "timeout_hours": 24,
                        "auto_approve_conditions": [
                            {
                                "field": "document_metadata.contract_value",
                                "operator": "less_than",
                                "value": 10000,
                            },
                            {
                                "field": "variables.classification.confidence",
                                "operator": "greater_than",
                                "value": 0.95,
                            },
                        ],
                    }
                ],
                "variables": {
                    "department": "legal",
                    "process_type": "contract_review",
                    "sla_hours": 48,
                },
                "timeout_hours": 72,
                "enabled": True,
            }

            return demo_workflow_data

        except Exception as e:
            logger.error(f"Demo workflow creation error: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":

    async def demo():
        """Demo workflow automation"""
        try:
            # Create workflow manager
            workflow_manager = WorkflowManager()

            print("Document Workflow Automation Demo")
            print("=" * 50)
            print(f"Croniter available: {HAS_CRONITER}")
            print(f"Aiohttp available: {HAS_AIOHTTP}")
            print(f"Pydantic available: {HAS_PYDANTIC}")
            print()

            # Create demo workflow
            demo_workflow_data = workflow_manager.create_demo_workflow()
            demo_workflow = workflow_manager._create_workflow_from_dict(
                demo_workflow_data
            )
            workflow_manager.register_workflow(demo_workflow)

            print(f"Registered workflow: {demo_workflow.name}")
            print(f"  - Triggers: {len(demo_workflow.triggers)}")
            print(f"  - Actions: {len(demo_workflow.actions)}")
            print(f"  - Routing Rules: {len(demo_workflow.routing_rules)}")
            print(f"  - Approval Configs: {len(demo_workflow.approval_configs)}")
            print()

            # Create workflow context
            context = WorkflowContext(
                workflow_id=str(uuid.uuid4()),
                document_id="contract_001",
                document_path="/documents/contract_001.pdf",
                document_metadata={
                    "category": "contract",
                    "contract_value": 50000,
                    "department": "sales",
                    "urgency": "normal",
                },
                user_id="user123",
            )

            print("Executing workflow...")
            execution = await workflow_manager.execute_workflow(
                demo_workflow.name, context
            )

            print(f"Workflow execution: {execution.execution_id}")
            print(f"  - State: {execution.state.value}")
            print(f"  - Completed actions: {len(execution.completed_actions)}")
            print(f"  - Failed actions: {len(execution.failed_actions)}")

            if execution.completed_actions:
                print(
                    f"  - Actions completed: {', '.join(execution.completed_actions)}"
                )

            if execution.results:
                print("  - Action results:")
                for action_name, result in execution.results.items():
                    status = "✓" if result.get("success", False) else "✗"
                    print(f"    {status} {action_name}: {result.get('success', False)}")

            # Display execution status
            status = workflow_manager.get_execution_status(execution.execution_id)
            if status:
                print(f"  - Progress: {status['progress']:.1f}%")
                if status["error_message"]:
                    print(f"  - Error: {status['error_message']}")

            print("\nDocument Workflow Automation demo completed successfully!")

        except Exception as e:
            print(f"Demo error: {e}")

    # Run demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
