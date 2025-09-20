"""
Document Collaboration with Operational Transformation
=====================================================

Advanced document collaboration system with real-time editing,
conflict resolution using Operational Transformation (OT),
and comprehensive version control.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of document operations"""

    INSERT = "insert"
    DELETE = "delete"
    RETAIN = "retain"
    FORMAT = "format"


@dataclass
class Operation:
    """Single document operation"""

    type: OperationType
    position: int
    content: Optional[str] = None
    length: Optional[int] = None
    attributes: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "position": self.position,
            "content": self.content,
            "length": self.length,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Operation":
        return cls(
            type=OperationType(data["type"]),
            position=data["position"],
            content=data.get("content"),
            length=data.get("length"),
            attributes=data.get("attributes"),
        )


@dataclass
class DocumentChange:
    """Document change with operations and metadata"""

    id: str
    document_id: str
    user_id: str
    operations: List[Operation]
    base_version: int
    timestamp: datetime
    applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "user_id": self.user_id,
            "operations": [op.to_dict() for op in self.operations],
            "base_version": self.base_version,
            "timestamp": self.timestamp.isoformat(),
            "applied": self.applied,
        }


@dataclass
class Document:
    """Collaborative document with version control"""

    id: str
    title: str
    content: str
    version: int
    created_by: str
    created_at: datetime
    last_modified: datetime
    change_history: List[DocumentChange]
    active_editors: Dict[str, Dict[str, Any]]  # user_id -> cursor/selection info

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "version": self.version,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "change_history_count": len(self.change_history),
            "active_editors": self.active_editors,
        }


class OperationalTransform:
    """Operational Transformation engine for conflict resolution"""

    @staticmethod
    def transform_operations(
        op1: Operation, op2: Operation
    ) -> Tuple[Operation, Operation]:
        """Transform two concurrent operations"""
        # Simplified OT implementation - in production, use a full OT library

        if op1.type == OperationType.INSERT and op2.type == OperationType.INSERT:
            if op1.position <= op2.position:
                # op1 is before op2, adjust op2 position
                new_op2 = Operation(
                    type=op2.type,
                    position=op2.position + len(op1.content or ""),
                    content=op2.content,
                    attributes=op2.attributes,
                )
                return op1, new_op2
            else:
                # op2 is before op1, adjust op1 position
                new_op1 = Operation(
                    type=op1.type,
                    position=op1.position + len(op2.content or ""),
                    content=op1.content,
                    attributes=op1.attributes,
                )
                return new_op1, op2

        elif op1.type == OperationType.DELETE and op2.type == OperationType.DELETE:
            # Handle overlapping deletes
            if op1.position + (op1.length or 0) <= op2.position:
                # op1 is completely before op2
                new_op2 = Operation(
                    type=op2.type,
                    position=op2.position - (op1.length or 0),
                    length=op2.length,
                    attributes=op2.attributes,
                )
                return op1, new_op2
            elif op2.position + (op2.length or 0) <= op1.position:
                # op2 is completely before op1
                new_op1 = Operation(
                    type=op1.type,
                    position=op1.position - (op2.length or 0),
                    length=op1.length,
                    attributes=op1.attributes,
                )
                return new_op1, op2
            else:
                # Overlapping deletes - complex case, simplified handling
                return op1, op2

        elif op1.type == OperationType.INSERT and op2.type == OperationType.DELETE:
            if op1.position <= op2.position:
                new_op2 = Operation(
                    type=op2.type,
                    position=op2.position + len(op1.content or ""),
                    length=op2.length,
                    attributes=op2.attributes,
                )
                return op1, new_op2
            elif op1.position >= op2.position + (op2.length or 0):
                new_op1 = Operation(
                    type=op1.type,
                    position=op1.position - (op2.length or 0),
                    content=op1.content,
                    attributes=op1.attributes,
                )
                return new_op1, op2
            else:
                # Insert within delete range
                return op1, op2

        elif op1.type == OperationType.DELETE and op2.type == OperationType.INSERT:
            # Reverse of above case
            transformed_op2, transformed_op1 = (
                OperationalTransform.transform_operations(op2, op1)
            )
            return transformed_op1, transformed_op2

        return op1, op2

    @staticmethod
    def apply_operation(content: str, operation: Operation) -> str:
        """Apply a single operation to content"""
        if operation.type == OperationType.INSERT:
            pos = operation.position
            text = operation.content or ""
            return content[:pos] + text + content[pos:]

        elif operation.type == OperationType.DELETE:
            start = operation.position
            end = start + (operation.length or 0)
            return content[:start] + content[end:]

        elif operation.type == OperationType.RETAIN:
            # Retain operation doesn't change content
            return content

        return content


class DocumentCollaboration:
    """Document collaboration manager"""

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.pending_changes: Dict[str, List[DocumentChange]] = (
            {}
        )  # document_id -> pending changes

    def create_document(
        self, title: str, user_id: str, initial_content: str = ""
    ) -> str:
        """Create a new collaborative document"""
        doc_id = str(uuid.uuid4())

        document = Document(
            id=doc_id,
            title=title,
            content=initial_content,
            version=1,
            created_by=user_id,
            created_at=datetime.now(),
            last_modified=datetime.now(),
            change_history=[],
            active_editors={},
        )

        self.documents[doc_id] = document
        self.pending_changes[doc_id] = []

        logger.info(f"Created document {doc_id}: {title}")
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.documents.get(doc_id)

    def join_document_editing(
        self, doc_id: str, user_id: str, cursor_info: Dict[str, Any]
    ):
        """User joins document editing session"""
        if doc_id not in self.documents:
            return False

        document = self.documents[doc_id]
        document.active_editors[user_id] = {
            "cursor_position": cursor_info.get("position", 0),
            "selection": cursor_info.get("selection"),
            "joined_at": datetime.now().isoformat(),
        }

        logger.info(f"User {user_id} joined editing document {doc_id}")
        return True

    def leave_document_editing(self, doc_id: str, user_id: str):
        """User leaves document editing session"""
        if doc_id in self.documents:
            document = self.documents[doc_id]
            document.active_editors.pop(user_id, None)
            logger.info(f"User {user_id} left editing document {doc_id}")

    def update_cursor_position(
        self, doc_id: str, user_id: str, cursor_info: Dict[str, Any]
    ):
        """Update user's cursor position"""
        if (
            doc_id in self.documents
            and user_id in self.documents[doc_id].active_editors
        ):
            editor_info = self.documents[doc_id].active_editors[user_id]
            editor_info.update(
                {
                    "cursor_position": cursor_info.get("position", 0),
                    "selection": cursor_info.get("selection"),
                    "last_update": datetime.now().isoformat(),
                }
            )

    def apply_change(
        self, doc_id: str, change: DocumentChange
    ) -> Tuple[bool, Optional[DocumentChange]]:
        """Apply a document change with OT conflict resolution"""
        if doc_id not in self.documents:
            return False, None

        document = self.documents[doc_id]

        # Check if change is based on current version
        if change.base_version == document.version:
            # No conflicts, apply directly
            transformed_change = change
        else:
            # Need to transform against pending changes
            transformed_change = self._transform_change(doc_id, change)
            if not transformed_change:
                return False, None

        # Apply operations to document content
        new_content = document.content
        for operation in transformed_change.operations:
            new_content = OperationalTransform.apply_operation(new_content, operation)

        # Update document
        document.content = new_content
        document.version += 1
        document.last_modified = datetime.now()
        transformed_change.applied = True
        document.change_history.append(transformed_change)

        # Clean up old history (keep last 1000 changes)
        if len(document.change_history) > 1000:
            document.change_history = document.change_history[-500:]

        logger.info(
            f"Applied change {change.id} to document {doc_id}, new version {document.version}"
        )
        return True, transformed_change

    def _transform_change(
        self, doc_id: str, change: DocumentChange
    ) -> Optional[DocumentChange]:
        """Transform a change against other concurrent changes"""
        document = self.documents[doc_id]

        # Get changes since the base version
        concurrent_changes = [
            c
            for c in document.change_history
            if c.base_version >= change.base_version and c.user_id != change.user_id
        ]

        if not concurrent_changes:
            return change

        # Transform operations against each concurrent change
        transformed_operations = change.operations.copy()

        for concurrent_change in concurrent_changes:
            new_transformed_ops = []

            for op in transformed_operations:
                for concurrent_op in concurrent_change.operations:
                    transformed_op, _ = OperationalTransform.transform_operations(
                        op, concurrent_op
                    )
                    new_transformed_ops.append(transformed_op)

            transformed_operations = new_transformed_ops

        # Create transformed change
        transformed_change = DocumentChange(
            id=change.id,
            document_id=change.document_id,
            user_id=change.user_id,
            operations=transformed_operations,
            base_version=document.version,
            timestamp=change.timestamp,
        )

        return transformed_change

    def get_document_state(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get complete document state for synchronization"""
        if doc_id not in self.documents:
            return None

        document = self.documents[doc_id]
        return {
            "document": document.to_dict(),
            "recent_changes": [
                change.to_dict() for change in document.change_history[-10:]
            ],
        }

    def list_documents(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all documents, optionally filtered by user"""
        documents = []

        for document in self.documents.values():
            if (
                user_id
                and user_id not in document.active_editors
                and document.created_by != user_id
            ):
                continue

            documents.append(
                {
                    "id": document.id,
                    "title": document.title,
                    "version": document.version,
                    "created_by": document.created_by,
                    "last_modified": document.last_modified.isoformat(),
                    "active_editors": len(document.active_editors),
                    "editor_names": list(document.active_editors.keys()),
                }
            )

        return documents


# Global document collaboration instance
document_collaboration = DocumentCollaboration()
