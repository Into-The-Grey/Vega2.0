"""
Document version control and branching system
============================================

This module provides comprehensive version control for collaborative
documents, including branching, merging, and history management.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid


class BranchStatus(Enum):
    """Branch status types"""

    ACTIVE = "active"
    MERGED = "merged"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MergeStrategy(Enum):
    """Merge strategy types"""

    AUTO = "auto"
    MANUAL = "manual"
    FAST_FORWARD = "fast_forward"
    THREE_WAY = "three_way"


@dataclass
class DocumentVersion:
    """Represents a specific version of a document"""

    id: str
    document_id: str
    branch_id: str
    version_number: int
    content: str
    content_hash: str
    author_id: str
    commit_message: str
    timestamp: datetime
    parent_version_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "branch_id": self.branch_id,
            "version_number": self.version_number,
            "content": self.content,
            "content_hash": self.content_hash,
            "author_id": self.author_id,
            "commit_message": self.commit_message,
            "timestamp": self.timestamp.isoformat(),
            "parent_version_id": self.parent_version_id,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class DocumentBranch:
    """Represents a document branch"""

    id: str
    document_id: str
    name: str
    description: str
    created_by: str
    created_at: datetime
    parent_branch_id: Optional[str] = None
    parent_version_id: Optional[str] = None
    head_version_id: Optional[str] = None
    status: BranchStatus = BranchStatus.ACTIVE
    protected: bool = False
    auto_merge: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "name": self.name,
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "parent_branch_id": self.parent_branch_id,
            "parent_version_id": self.parent_version_id,
            "head_version_id": self.head_version_id,
            "status": self.status.value,
            "protected": self.protected,
            "auto_merge": self.auto_merge,
        }


@dataclass
class MergeRequest:
    """Represents a merge request between branches"""

    id: str
    document_id: str
    source_branch_id: str
    target_branch_id: str
    title: str
    description: str
    author_id: str
    created_at: datetime
    status: str = "open"  # open, approved, merged, rejected, closed
    merge_strategy: MergeStrategy = MergeStrategy.AUTO
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    approvals: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "source_branch_id": self.source_branch_id,
            "target_branch_id": self.target_branch_id,
            "title": self.title,
            "description": self.description,
            "author_id": self.author_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "merge_strategy": self.merge_strategy.value,
            "conflicts": self.conflicts,
            "reviewers": self.reviewers,
            "approvals": self.approvals,
        }


class DocumentVersionControl:
    """Document version control system"""

    def __init__(self):
        self.versions: Dict[str, DocumentVersion] = {}  # version_id -> version
        self.branches: Dict[str, DocumentBranch] = {}  # branch_id -> branch
        self.document_branches: Dict[str, List[str]] = {}  # document_id -> branch_ids
        self.document_versions: Dict[str, List[str]] = {}  # document_id -> version_ids
        self.merge_requests: Dict[str, MergeRequest] = {}  # request_id -> request
        self.tags: Dict[str, str] = {}  # tag_name -> version_id

    def create_initial_version(
        self,
        document_id: str,
        content: str,
        author_id: str,
        commit_message: str = "Initial version",
    ) -> Tuple[str, str]:
        """Create initial version and main branch"""
        # Create main branch
        main_branch_id = str(uuid.uuid4())
        main_branch = DocumentBranch(
            id=main_branch_id,
            document_id=document_id,
            name="main",
            description="Main branch",
            created_by=author_id,
            created_at=datetime.now(),
            protected=True,
        )
        self.branches[main_branch_id] = main_branch

        if document_id not in self.document_branches:
            self.document_branches[document_id] = []
        self.document_branches[document_id].append(main_branch_id)

        # Create initial version
        version_id = self.commit_version(
            document_id=document_id,
            branch_id=main_branch_id,
            content=content,
            author_id=author_id,
            commit_message=commit_message,
        )

        return main_branch_id, version_id

    def create_branch(
        self,
        document_id: str,
        name: str,
        author_id: str,
        description: str = "",
        parent_branch_id: Optional[str] = None,
        parent_version_id: Optional[str] = None,
    ) -> str:
        """Create a new branch"""
        branch_id = str(uuid.uuid4())

        # If no parent specified, use main branch
        if not parent_branch_id:
            main_branches = [
                b
                for b in self.document_branches.get(document_id, [])
                if self.branches[b].name == "main"
            ]
            if main_branches:
                parent_branch_id = main_branches[0]
                parent_branch = self.branches[parent_branch_id]
                parent_version_id = parent_branch.head_version_id

        branch = DocumentBranch(
            id=branch_id,
            document_id=document_id,
            name=name,
            description=description,
            created_by=author_id,
            created_at=datetime.now(),
            parent_branch_id=parent_branch_id,
            parent_version_id=parent_version_id,
        )

        self.branches[branch_id] = branch

        if document_id not in self.document_branches:
            self.document_branches[document_id] = []
        self.document_branches[document_id].append(branch_id)

        return branch_id

    def commit_version(
        self,
        document_id: str,
        branch_id: str,
        content: str,
        author_id: str,
        commit_message: str,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Commit a new version to a branch"""
        branch = self.branches.get(branch_id)
        if not branch or branch.document_id != document_id:
            raise ValueError("Invalid branch")

        # Calculate version number
        branch_versions = [
            v
            for v in self.document_versions.get(document_id, [])
            if self.versions[v].branch_id == branch_id
        ]
        version_number = len(branch_versions) + 1

        # Calculate content hash
        import hashlib

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        version_id = str(uuid.uuid4())
        version = DocumentVersion(
            id=version_id,
            document_id=document_id,
            branch_id=branch_id,
            version_number=version_number,
            content=content,
            content_hash=content_hash,
            author_id=author_id,
            commit_message=commit_message,
            timestamp=datetime.now(),
            parent_version_id=branch.head_version_id,
            tags=tags or [],
        )

        self.versions[version_id] = version

        if document_id not in self.document_versions:
            self.document_versions[document_id] = []
        self.document_versions[document_id].append(version_id)

        # Update branch head
        branch.head_version_id = version_id

        # Add tags
        for tag in tags or []:
            self.tags[tag] = version_id

        return version_id

    def get_version(self, version_id: str) -> Optional[DocumentVersion]:
        """Get a specific version"""
        return self.versions.get(version_id)

    def get_branch(self, branch_id: str) -> Optional[DocumentBranch]:
        """Get a specific branch"""
        return self.branches.get(branch_id)

    def get_document_branches(self, document_id: str) -> List[DocumentBranch]:
        """Get all branches for a document"""
        branch_ids = self.document_branches.get(document_id, [])
        return [self.branches[bid] for bid in branch_ids if bid in self.branches]

    def get_branch_versions(
        self, document_id: str, branch_id: str, limit: Optional[int] = None
    ) -> List[DocumentVersion]:
        """Get versions for a specific branch"""
        version_ids = self.document_versions.get(document_id, [])
        branch_versions = [
            self.versions[vid]
            for vid in version_ids
            if vid in self.versions and self.versions[vid].branch_id == branch_id
        ]

        # Sort by timestamp (newest first)
        branch_versions.sort(key=lambda v: v.timestamp, reverse=True)

        if limit:
            branch_versions = branch_versions[:limit]

        return branch_versions

    def get_version_history(
        self, document_id: str, version_id: str
    ) -> List[DocumentVersion]:
        """Get version history (ancestry) for a specific version"""
        history = []
        current_version = self.versions.get(version_id)

        while current_version:
            history.append(current_version)
            if current_version.parent_version_id:
                current_version = self.versions.get(current_version.parent_version_id)
            else:
                break

        return history

    def create_merge_request(
        self,
        document_id: str,
        source_branch_id: str,
        target_branch_id: str,
        title: str,
        description: str,
        author_id: str,
        reviewers: Optional[List[str]] = None,
        merge_strategy: MergeStrategy = MergeStrategy.AUTO,
    ) -> str:
        """Create a merge request"""
        request_id = str(uuid.uuid4())

        # Check for conflicts
        conflicts = self._detect_conflicts(source_branch_id, target_branch_id)

        merge_request = MergeRequest(
            id=request_id,
            document_id=document_id,
            source_branch_id=source_branch_id,
            target_branch_id=target_branch_id,
            title=title,
            description=description,
            author_id=author_id,
            created_at=datetime.now(),
            merge_strategy=merge_strategy,
            conflicts=conflicts,
            reviewers=reviewers or [],
        )

        self.merge_requests[request_id] = merge_request
        return request_id

    def approve_merge_request(
        self, request_id: str, approver_id: str, comments: str = ""
    ) -> bool:
        """Approve a merge request"""
        merge_request = self.merge_requests.get(request_id)
        if not merge_request or merge_request.status != "open":
            return False

        # Check if approver is in reviewers list
        if merge_request.reviewers and approver_id not in merge_request.reviewers:
            return False

        approval = {
            "approver_id": approver_id,
            "timestamp": datetime.now().isoformat(),
            "comments": comments,
        }
        merge_request.approvals.append(approval)

        # Check if all required approvals are met
        if len(merge_request.approvals) >= len(merge_request.reviewers):
            merge_request.status = "approved"

        return True

    def merge_branches(
        self, request_id: str, merger_id: str, merge_message: str = ""
    ) -> Optional[str]:
        """Merge branches based on merge request"""
        merge_request = self.merge_requests.get(request_id)
        if not merge_request or merge_request.status not in ["open", "approved"]:
            return None

        source_branch = self.branches.get(merge_request.source_branch_id)
        target_branch = self.branches.get(merge_request.target_branch_id)

        if not source_branch or not target_branch:
            return None

        # Get latest versions
        source_version = self.versions.get(source_branch.head_version_id)
        target_version = self.versions.get(target_branch.head_version_id)

        if not source_version or not target_version:
            return None

        # Simple merge strategy - take source content
        merged_content = source_version.content
        if merge_request.merge_strategy == MergeStrategy.THREE_WAY:
            # TODO: Implement three-way merge algorithm
            merged_content = self._three_way_merge(source_version, target_version)

        # Create merge commit
        merge_version_id = self.commit_version(
            document_id=merge_request.document_id,
            branch_id=merge_request.target_branch_id,
            content=merged_content,
            author_id=merger_id,
            commit_message=merge_message
            or f"Merge {source_branch.name} into {target_branch.name}",
        )

        # Update merge request status
        merge_request.status = "merged"

        return merge_version_id

    def checkout_version(self, document_id: str, version_id: str) -> Optional[str]:
        """Checkout a specific version (get content)"""
        version = self.versions.get(version_id)
        if not version or version.document_id != document_id:
            return None

        return version.content

    def tag_version(self, version_id: str, tag_name: str) -> bool:
        """Tag a specific version"""
        if version_id not in self.versions:
            return False

        self.tags[tag_name] = version_id
        version = self.versions[version_id]
        if tag_name not in version.tags:
            version.tags.append(tag_name)

        return True

    def get_version_by_tag(self, tag_name: str) -> Optional[DocumentVersion]:
        """Get version by tag"""
        version_id = self.tags.get(tag_name)
        if version_id:
            return self.versions.get(version_id)
        return None

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two versions"""
        version1 = self.versions.get(version_id1)
        version2 = self.versions.get(version_id2)

        if not version1 or not version2:
            return {"error": "Version not found"}

        # Simple diff - could be enhanced with proper diff algorithm
        return {
            "version1": version1.to_dict(),
            "version2": version2.to_dict(),
            "content_changed": version1.content_hash != version2.content_hash,
            "size_diff": len(version2.content) - len(version1.content),
            "time_diff": (version2.timestamp - version1.timestamp).total_seconds(),
        }

    def _detect_conflicts(
        self, source_branch_id: str, target_branch_id: str
    ) -> List[Dict[str, Any]]:
        """Detect potential merge conflicts"""
        # Simplified conflict detection
        conflicts = []

        source_branch = self.branches.get(source_branch_id)
        target_branch = self.branches.get(target_branch_id)

        if not source_branch or not target_branch:
            return conflicts

        source_version = self.versions.get(source_branch.head_version_id)
        target_version = self.versions.get(target_branch.head_version_id)

        if source_version and target_version:
            # Check if branches have diverged
            if (
                source_version.parent_version_id != target_version.parent_version_id
                and source_version.content_hash != target_version.content_hash
            ):
                conflicts.append(
                    {
                        "type": "content_conflict",
                        "description": "Both branches have conflicting changes",
                        "source_version": source_version.id,
                        "target_version": target_version.id,
                    }
                )

        return conflicts

    def _three_way_merge(
        self, source_version: DocumentVersion, target_version: DocumentVersion
    ) -> str:
        """Perform three-way merge"""
        # Simplified merge - in practice, would use proper diff/merge algorithms
        # For now, just return source content
        return source_version.content

    def get_merge_requests(
        self, document_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[MergeRequest]:
        """Get merge requests"""
        requests = list(self.merge_requests.values())

        if document_id:
            requests = [r for r in requests if r.document_id == document_id]

        if status:
            requests = [r for r in requests if r.status == status]

        return sorted(requests, key=lambda r: r.created_at, reverse=True)


# Global version control instance
document_version_control = DocumentVersionControl()
