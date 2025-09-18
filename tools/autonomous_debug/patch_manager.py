#!/usr/bin/env python3
"""
Patch Management + Rollback System
=================================

Manages file backups, diff generation, metadata tracking, rollback 
capabilities, and CLI commands for manual intervention. Provides 
comprehensive patch lifecycle management.

Features:
- Automated file backups before applying patches
- Git-style diff generation and storage
- Comprehensive metadata tracking
- Safe rollback operations
- CLI interface for manual intervention
- Patch history and versioning
- Atomic operations with transaction support
"""

import os
import sys
import shutil
import subprocess
import json
import sqlite3
import hashlib
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import difflib
import tempfile
import logging

# Local imports
from error_tracker import ErrorRecord
from self_debugger import FixSuggestion
from code_sandbox import SandboxResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PatchMetadata:
    """Metadata for a patch operation"""
    patch_id: str
    fix_id: str
    error_id: str
    description: str
    files_modified: List[str]
    backup_path: str
    diff_path: str
    applied_at: datetime
    applied_by: str  # user or autonomous
    status: str  # applied, rolled_back, failed
    rollback_safe: bool
    validation_result: Optional[str]  # sandbox validation ID
    dependencies: List[str]
    estimated_impact: str
    checksum: str  # For integrity verification

@dataclass 
class FileBackup:
    """Information about a backed up file"""
    file_path: str
    backup_path: str
    original_size: int
    original_hash: str
    backup_time: datetime
    compressed: bool = False

class PatchBackupManager:
    """Manages file backups for patch operations"""
    
    def __init__(self, backup_dir: str = "autonomous_debug/backups"):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        self.backup_db = self._init_backup_db()
    
    def _init_backup_db(self) -> sqlite3.Connection:
        """Initialize backup tracking database"""
        db_path = os.path.join(self.backup_dir, "backups.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_backups (
                backup_id TEXT PRIMARY KEY,
                patch_id TEXT,
                file_path TEXT NOT NULL,
                backup_path TEXT NOT NULL,
                original_size INTEGER,
                original_hash TEXT,
                backup_time TEXT,
                compressed BOOLEAN DEFAULT FALSE,
                restored BOOLEAN DEFAULT FALSE
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backup_sessions (
                session_id TEXT PRIMARY KEY,
                patch_id TEXT,
                created_at TEXT,
                file_count INTEGER,
                total_size INTEGER,
                completed BOOLEAN DEFAULT FALSE
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patch_id ON file_backups (patch_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON file_backups (file_path)")
        
        conn.commit()
        return conn
    
    def create_backup_session(self, patch_id: str, files: List[str]) -> str:
        """Create a backup session for multiple files"""
        session_id = f"backup_{patch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Calculate total size
            total_size = 0
            for file_path in files:
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
            
            # Create session record
            cursor = self.backup_db.cursor()
            cursor.execute("""
                INSERT INTO backup_sessions (
                    session_id, patch_id, created_at, file_count, total_size
                ) VALUES (?, ?, ?, ?, ?)
            """, (session_id, patch_id, datetime.now().isoformat(), len(files), total_size))
            
            self.backup_db.commit()
            logger.info(f"Created backup session {session_id} for {len(files)} files")
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create backup session: {e}")
            return ""
    
    def backup_file(self, file_path: str, patch_id: str, session_id: str = None) -> Optional[FileBackup]:
        """Create backup of a single file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            # Generate backup info
            file_hash = self._calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            
            # Create backup directory structure
            relative_path = os.path.relpath(file_path, os.getcwd())
            backup_subdir = os.path.join(self.backup_dir, patch_id)
            os.makedirs(backup_subdir, exist_ok=True)
            
            # Generate backup filename
            filename = os.path.basename(file_path)
            backup_filename = f"{filename}.{timestamp}.backup"
            backup_path = os.path.join(backup_subdir, backup_filename)
            
            # Decide whether to compress based on file size
            compress = file_size > 10240  # Compress files > 10KB
            
            if compress:
                backup_path += ".gz"
                self._compress_file(file_path, backup_path)
            else:
                shutil.copy2(file_path, backup_path)
            
            # Create backup record
            backup = FileBackup(
                file_path=file_path,
                backup_path=backup_path,
                original_size=file_size,
                original_hash=file_hash,
                backup_time=datetime.now(),
                compressed=compress
            )
            
            # Store in database
            backup_id = f"backup_{hashlib.md5(f'{file_path}{timestamp}'.encode()).hexdigest()}"
            cursor = self.backup_db.cursor()
            cursor.execute("""
                INSERT INTO file_backups (
                    backup_id, patch_id, file_path, backup_path,
                    original_size, original_hash, backup_time, compressed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                backup_id, patch_id, file_path, backup_path,
                file_size, file_hash, backup.backup_time.isoformat(), compress
            ))
            
            self.backup_db.commit()
            logger.debug(f"Backed up {file_path} to {backup_path}")
            
            return backup
            
        except Exception as e:
            logger.error(f"Failed to backup file {file_path}: {e}")
            return None
    
    def backup_multiple_files(self, files: List[str], patch_id: str) -> List[FileBackup]:
        """Backup multiple files as part of a patch operation"""
        session_id = self.create_backup_session(patch_id, files)
        backups = []
        
        for file_path in files:
            backup = self.backup_file(file_path, patch_id, session_id)
            if backup:
                backups.append(backup)
        
        # Mark session as completed
        cursor = self.backup_db.cursor()
        cursor.execute(
            "UPDATE backup_sessions SET completed = TRUE WHERE session_id = ?",
            (session_id,)
        )
        self.backup_db.commit()
        
        logger.info(f"Backed up {len(backups)}/{len(files)} files for patch {patch_id}")
        return backups
    
    def restore_file(self, file_path: str, patch_id: str) -> bool:
        """Restore a file from backup"""
        try:
            # Find the most recent backup for this file and patch
            cursor = self.backup_db.cursor()
            cursor.execute("""
                SELECT * FROM file_backups 
                WHERE file_path = ? AND patch_id = ?
                ORDER BY backup_time DESC 
                LIMIT 1
            """, (file_path, patch_id))
            
            backup_row = cursor.fetchone()
            if not backup_row:
                logger.error(f"No backup found for {file_path} in patch {patch_id}")
                return False
            
            backup_path = backup_row['backup_path']
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Restore the file
            if backup_row['compressed']:
                self._decompress_file(backup_path, file_path)
            else:
                shutil.copy2(backup_path, file_path)
            
            # Verify restoration
            if self._verify_restoration(file_path, backup_row):
                # Mark as restored
                cursor.execute(
                    "UPDATE file_backups SET restored = TRUE WHERE backup_id = ?",
                    (backup_row['backup_id'],)
                )
                self.backup_db.commit()
                
                logger.info(f"Successfully restored {file_path}")
                return True
            else:
                logger.error(f"Restoration verification failed for {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore file {file_path}: {e}")
            return False
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        hash_obj = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def _compress_file(self, source_path: str, dest_path: str):
        """Compress a file using gzip"""
        with open(source_path, 'rb') as f_in:
            with gzip.open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    def _decompress_file(self, source_path: str, dest_path: str):
        """Decompress a gzip file"""
        with gzip.open(source_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    def _verify_restoration(self, file_path: str, backup_row) -> bool:
        """Verify that file was restored correctly"""
        try:
            current_hash = self._calculate_file_hash(file_path)
            return current_hash == backup_row['original_hash']
        except Exception:
            return False
    
    def get_patch_backups(self, patch_id: str) -> List[Dict[str, Any]]:
        """Get all backups for a patch"""
        try:
            cursor = self.backup_db.cursor()
            cursor.execute(
                "SELECT * FROM file_backups WHERE patch_id = ? ORDER BY backup_time",
                (patch_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get patch backups: {e}")
            return []
    
    def cleanup_old_backups(self, days_old: int = 30):
        """Clean up backups older than specified days"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            
            cursor = self.backup_db.cursor()
            cursor.execute(
                "SELECT backup_path FROM file_backups WHERE backup_time < ?",
                (cutoff_date,)
            )
            
            old_backups = cursor.fetchall()
            
            # Delete backup files
            for backup in old_backups:
                backup_path = backup['backup_path']
                if os.path.exists(backup_path):
                    os.remove(backup_path)
            
            # Remove from database
            cursor.execute(
                "DELETE FROM file_backups WHERE backup_time < ?",
                (cutoff_date,)
            )
            
            self.backup_db.commit()
            logger.info(f"Cleaned up {len(old_backups)} old backups")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
    
    def close(self):
        """Close database connection"""
        if self.backup_db:
            self.backup_db.close()

class DiffManager:
    """Manages diff generation and storage"""
    
    def __init__(self, diff_dir: str = "autonomous_debug/patches"):
        self.diff_dir = diff_dir
        os.makedirs(diff_dir, exist_ok=True)
    
    def generate_diff(self, patch_id: str, changes: List[Dict[str, Any]]) -> str:
        """Generate unified diff for patch changes"""
        try:
            diff_lines = []
            diff_lines.append(f"# Patch ID: {patch_id}")
            diff_lines.append(f"# Generated: {datetime.now().isoformat()}")
            diff_lines.append("")
            
            for change in changes:
                file_path = change['file']
                line_num = change['line']
                old_code = change['old_code']
                new_code = change['new_code']
                
                # Generate file diff
                diff_lines.append(f"--- a/{file_path}")
                diff_lines.append(f"+++ b/{file_path}")
                diff_lines.append(f"@@ -{line_num},1 +{line_num},1 @@")
                diff_lines.append(f"-{old_code}")
                diff_lines.append(f"+{new_code}")
                diff_lines.append("")
            
            # Save diff to file
            diff_filename = f"{patch_id}.diff"
            diff_path = os.path.join(self.diff_dir, diff_filename)
            
            with open(diff_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(diff_lines))
            
            logger.debug(f"Generated diff: {diff_path}")
            return diff_path
            
        except Exception as e:
            logger.error(f"Failed to generate diff: {e}")
            return ""
    
    def generate_file_diff(self, file_path: str, original_content: str, 
                          modified_content: str) -> List[str]:
        """Generate diff for a single file"""
        try:
            original_lines = original_content.splitlines(keepends=True)
            modified_lines = modified_content.splitlines(keepends=True)
            
            diff = difflib.unified_diff(
                original_lines, modified_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm=''
            )
            
            return list(diff)
            
        except Exception as e:
            logger.error(f"Failed to generate file diff: {e}")
            return []
    
    def apply_diff(self, diff_path: str, target_dir: str = None) -> bool:
        """Apply a diff using patch command"""
        try:
            target_dir = target_dir or os.getcwd()
            
            # Use git apply for better handling
            result = subprocess.run([
                'git', 'apply', '--directory', target_dir, diff_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully applied diff: {diff_path}")
                return True
            else:
                logger.error(f"Failed to apply diff: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply diff: {e}")
            return False
    
    def reverse_diff(self, diff_path: str, target_dir: str = None) -> bool:
        """Reverse a diff (rollback)"""
        try:
            target_dir = target_dir or os.getcwd()
            
            # Use git apply --reverse
            result = subprocess.run([
                'git', 'apply', '--reverse', '--directory', target_dir, diff_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully reversed diff: {diff_path}")
                return True
            else:
                logger.error(f"Failed to reverse diff: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reverse diff: {e}")
            return False

class PatchManager:
    """Main patch management orchestrator"""
    
    def __init__(self, workspace_dir: str = "/home/ncacord/Vega2.0"):
        self.workspace_dir = workspace_dir
        self.backup_manager = PatchBackupManager()
        self.diff_manager = DiffManager()
        self.patch_db = self._init_patch_db()
        
    def _init_patch_db(self) -> sqlite3.Connection:
        """Initialize patch management database"""
        db_path = "autonomous_debug/patches.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patches (
                patch_id TEXT PRIMARY KEY,
                fix_id TEXT,
                error_id TEXT,
                description TEXT,
                files_modified TEXT,
                backup_path TEXT,
                diff_path TEXT,
                applied_at TEXT,
                applied_by TEXT,
                status TEXT,
                rollback_safe BOOLEAN,
                validation_result TEXT,
                dependencies TEXT,
                estimated_impact TEXT,
                checksum TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patch_history (
                history_id TEXT PRIMARY KEY,
                patch_id TEXT,
                action TEXT,  -- applied, rolled_back, failed
                timestamp TEXT,
                details TEXT,
                success BOOLEAN
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patch_status ON patches (status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_applied_at ON patches (applied_at)")
        
        conn.commit()
        return conn
    
    def apply_patch(self, fix: FixSuggestion, validation_result: SandboxResult = None, 
                   applied_by: str = "autonomous") -> PatchMetadata:
        """Apply a patch with full backup and tracking"""
        patch_id = f"patch_{fix.id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Applying patch {patch_id} for fix {fix.id[:8]}")
            
            # Extract files to modify
            files_to_modify = [change['file'] for change in fix.code_changes]
            
            # Create backups
            backups = self.backup_manager.backup_multiple_files(files_to_modify, patch_id)
            
            if len(backups) != len(files_to_modify):
                logger.error("Failed to backup all files")
                return self._create_failed_patch_metadata(patch_id, fix, "Backup failed")
            
            # Generate diff
            diff_path = self.diff_manager.generate_diff(patch_id, fix.code_changes)
            
            if not diff_path:
                logger.error("Failed to generate diff")
                return self._create_failed_patch_metadata(patch_id, fix, "Diff generation failed")
            
            # Apply changes
            if not self._apply_changes(fix.code_changes):
                logger.error("Failed to apply changes")
                # Restore backups
                self._rollback_patch(patch_id)
                return self._create_failed_patch_metadata(patch_id, fix, "Apply changes failed")
            
            # Create patch metadata
            metadata = PatchMetadata(
                patch_id=patch_id,
                fix_id=fix.id,
                error_id=fix.error_id,
                description=fix.description,
                files_modified=files_to_modify,
                backup_path=self.backup_manager.backup_dir,
                diff_path=diff_path,
                applied_at=datetime.now(),
                applied_by=applied_by,
                status="applied",
                rollback_safe=fix.rollback_safe,
                validation_result=validation_result.sandbox_id if validation_result else None,
                dependencies=fix.dependencies,
                estimated_impact=fix.estimated_impact,
                checksum=self._calculate_patch_checksum(fix.code_changes)
            )
            
            # Store patch metadata
            self._store_patch_metadata(metadata)
            
            # Record history
            self._record_patch_history(patch_id, "applied", "Patch applied successfully", True)
            
            logger.info(f"Successfully applied patch {patch_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to apply patch: {e}")
            # Attempt rollback
            self._rollback_patch(patch_id)
            return self._create_failed_patch_metadata(patch_id, fix, str(e))
    
    def rollback_patch(self, patch_id: str, reason: str = "Manual rollback") -> bool:
        """Rollback a previously applied patch"""
        try:
            logger.info(f"Rolling back patch {patch_id}")
            
            # Get patch metadata
            metadata = self.get_patch_metadata(patch_id)
            if not metadata:
                logger.error(f"Patch not found: {patch_id}")
                return False
            
            if metadata['status'] != 'applied':
                logger.error(f"Patch not in applied state: {metadata['status']}")
                return False
            
            # Restore files from backups
            for file_path in json.loads(metadata['files_modified']):
                if not self.backup_manager.restore_file(file_path, patch_id):
                    logger.error(f"Failed to restore {file_path}")
                    return False
            
            # Update patch status
            cursor = self.patch_db.cursor()
            cursor.execute(
                "UPDATE patches SET status = 'rolled_back' WHERE patch_id = ?",
                (patch_id,)
            )
            self.patch_db.commit()
            
            # Record history
            self._record_patch_history(patch_id, "rolled_back", reason, True)
            
            logger.info(f"Successfully rolled back patch {patch_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback patch {patch_id}: {e}")
            self._record_patch_history(patch_id, "rollback_failed", str(e), False)
            return False
    
    def _apply_changes(self, code_changes: List[Dict[str, Any]]) -> bool:
        """Apply code changes to files"""
        try:
            for change in code_changes:
                file_path = change['file']
                line_num = change['line']
                old_code = change['old_code']
                new_code = change['new_code']
                
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Verify and apply change
                if 1 <= line_num <= len(lines):
                    current_line = lines[line_num - 1].rstrip()
                    expected_line = old_code.rstrip()
                    
                    if current_line != expected_line:
                        logger.warning(f"Line mismatch in {file_path}:{line_num}")
                        logger.warning(f"Expected: {expected_line}")
                        logger.warning(f"Found: {current_line}")
                    
                    # Apply change
                    lines[line_num - 1] = new_code + '\n'
                    
                    # Write file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    
                    logger.debug(f"Applied change to {file_path}:{line_num}")
                else:
                    logger.error(f"Line number out of range: {file_path}:{line_num}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply changes: {e}")
            return False
    
    def _rollback_patch(self, patch_id: str) -> bool:
        """Internal rollback method"""
        try:
            # Get backup files for this patch
            backups = self.backup_manager.get_patch_backups(patch_id)
            
            for backup in backups:
                file_path = backup['file_path']
                if not self.backup_manager.restore_file(file_path, patch_id):
                    logger.error(f"Failed to restore {file_path} during rollback")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback patch {patch_id}: {e}")
            return False
    
    def _calculate_patch_checksum(self, code_changes: List[Dict[str, Any]]) -> str:
        """Calculate checksum for patch integrity"""
        content = json.dumps(code_changes, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _store_patch_metadata(self, metadata: PatchMetadata):
        """Store patch metadata in database"""
        try:
            cursor = self.patch_db.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO patches (
                    patch_id, fix_id, error_id, description, files_modified,
                    backup_path, diff_path, applied_at, applied_by, status,
                    rollback_safe, validation_result, dependencies,
                    estimated_impact, checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.patch_id, metadata.fix_id, metadata.error_id,
                metadata.description, json.dumps(metadata.files_modified),
                metadata.backup_path, metadata.diff_path,
                metadata.applied_at.isoformat(), metadata.applied_by,
                metadata.status, metadata.rollback_safe,
                metadata.validation_result, json.dumps(metadata.dependencies),
                metadata.estimated_impact, metadata.checksum
            ))
            self.patch_db.commit()
            
        except Exception as e:
            logger.error(f"Failed to store patch metadata: {e}")
    
    def _record_patch_history(self, patch_id: str, action: str, details: str, success: bool):
        """Record patch history event"""
        try:
            history_id = f"hist_{patch_id}_{action}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            cursor = self.patch_db.cursor()
            cursor.execute("""
                INSERT INTO patch_history (
                    history_id, patch_id, action, timestamp, details, success
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                history_id, patch_id, action, datetime.now().isoformat(),
                details, success
            ))
            self.patch_db.commit()
            
        except Exception as e:
            logger.error(f"Failed to record patch history: {e}")
    
    def _create_failed_patch_metadata(self, patch_id: str, fix: FixSuggestion, 
                                     reason: str) -> PatchMetadata:
        """Create metadata for failed patch"""
        metadata = PatchMetadata(
            patch_id=patch_id,
            fix_id=fix.id,
            error_id=fix.error_id,
            description=f"FAILED: {fix.description}",
            files_modified=[],
            backup_path="",
            diff_path="",
            applied_at=datetime.now(),
            applied_by="autonomous",
            status="failed",
            rollback_safe=True,
            validation_result=None,
            dependencies=fix.dependencies,
            estimated_impact=fix.estimated_impact,
            checksum=""
        )
        
        self._store_patch_metadata(metadata)
        self._record_patch_history(patch_id, "failed", reason, False)
        
        return metadata
    
    def get_patch_metadata(self, patch_id: str) -> Optional[Dict[str, Any]]:
        """Get patch metadata"""
        try:
            cursor = self.patch_db.cursor()
            cursor.execute("SELECT * FROM patches WHERE patch_id = ?", (patch_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get patch metadata: {e}")
            return None
    
    def list_patches(self, status: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """List patches with optional status filter"""
        try:
            cursor = self.patch_db.cursor()
            
            if status:
                cursor.execute("""
                    SELECT * FROM patches 
                    WHERE status = ? 
                    ORDER BY applied_at DESC 
                    LIMIT ?
                """, (status, limit))
            else:
                cursor.execute("""
                    SELECT * FROM patches 
                    ORDER BY applied_at DESC 
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to list patches: {e}")
            return []
    
    def get_patch_statistics(self) -> Dict[str, Any]:
        """Get patch statistics"""
        try:
            cursor = self.patch_db.cursor()
            
            # Total patches
            cursor.execute("SELECT COUNT(*) as total FROM patches")
            total = cursor.fetchone()['total']
            
            # By status
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM patches 
                GROUP BY status
            """)
            by_status = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Recent activity (last 7 days)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("""
                SELECT COUNT(*) as recent 
                FROM patches 
                WHERE applied_at > ?
            """, (week_ago,))
            recent = cursor.fetchone()['recent']
            
            return {
                'total_patches': total,
                'by_status': by_status,
                'recent_patches': recent,
                'success_rate': (by_status.get('applied', 0) / total * 100) if total > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get patch statistics: {e}")
            return {}
    
    def cleanup_old_patches(self, days_old: int = 30):
        """Clean up old patch data"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            
            # Get old patches
            cursor = self.patch_db.cursor()
            cursor.execute("""
                SELECT patch_id, diff_path FROM patches 
                WHERE applied_at < ? AND status != 'applied'
            """, (cutoff_date,))
            
            old_patches = cursor.fetchall()
            
            # Clean up files and database records
            for patch in old_patches:
                patch_id = patch['patch_id']
                diff_path = patch['diff_path']
                
                # Remove diff file
                if diff_path and os.path.exists(diff_path):
                    os.remove(diff_path)
                
                # Remove from database
                cursor.execute("DELETE FROM patches WHERE patch_id = ?", (patch_id,))
                cursor.execute("DELETE FROM patch_history WHERE patch_id = ?", (patch_id,))
            
            self.patch_db.commit()
            
            # Clean up old backups
            self.backup_manager.cleanup_old_backups(days_old)
            
            logger.info(f"Cleaned up {len(old_patches)} old patches")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old patches: {e}")
    
    def close(self):
        """Close all database connections"""
        if self.patch_db:
            self.patch_db.close()
        if self.backup_manager:
            self.backup_manager.close()

def main():
    """Main function for patch management CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Patch Management + Rollback System")
    parser.add_argument("--list", action="store_true", help="List all patches")
    parser.add_argument("--status", help="Filter patches by status")
    parser.add_argument("--rollback", help="Rollback specific patch ID")
    parser.add_argument("--stats", action="store_true", help="Show patch statistics")
    parser.add_argument("--cleanup", type=int, help="Cleanup patches older than N days")
    parser.add_argument("--backup-file", help="Backup a specific file")
    parser.add_argument("--patch-id", help="Patch ID for operations")
    
    args = parser.parse_args()
    
    patch_manager = PatchManager()
    
    try:
        if args.list:
            patches = patch_manager.list_patches(args.status)
            print(f"\n📋 Patches ({len(patches)}):")
            
            for patch in patches:
                status_icon = {"applied": "✅", "rolled_back": "↩️", "failed": "❌"}.get(patch['status'], "❓")
                print(f"\n{status_icon} {patch['patch_id']}")
                print(f"  Description: {patch['description']}")
                print(f"  Applied: {patch['applied_at']}")
                print(f"  Status: {patch['status']}")
                print(f"  Files: {len(json.loads(patch['files_modified']))}")
        
        elif args.rollback:
            print(f"↩️ Rolling back patch {args.rollback}...")
            success = patch_manager.rollback_patch(args.rollback, "Manual rollback via CLI")
            
            if success:
                print("✅ Rollback successful")
            else:
                print("❌ Rollback failed")
        
        elif args.stats:
            stats = patch_manager.get_patch_statistics()
            print("\n📊 Patch Statistics:")
            print(f"  Total Patches: {stats.get('total_patches', 0)}")
            print(f"  Success Rate: {stats.get('success_rate', 0):.1f}%")
            print(f"  Recent Activity: {stats.get('recent_patches', 0)} (last 7 days)")
            
            print("\n  By Status:")
            for status, count in stats.get('by_status', {}).items():
                print(f"    {status}: {count}")
        
        elif args.cleanup:
            print(f"🧹 Cleaning up patches older than {args.cleanup} days...")
            patch_manager.cleanup_old_patches(args.cleanup)
            print("✅ Cleanup completed")
        
        elif args.backup_file and args.patch_id:
            print(f"💾 Backing up {args.backup_file}...")
            backup = patch_manager.backup_manager.backup_file(args.backup_file, args.patch_id)
            
            if backup:
                print(f"✅ Backup created: {backup.backup_path}")
            else:
                print("❌ Backup failed")
        
        else:
            print("Specify an operation: --list, --rollback, --stats, --cleanup, or --backup-file")
    
    finally:
        patch_manager.close()

if __name__ == "__main__":
    main()