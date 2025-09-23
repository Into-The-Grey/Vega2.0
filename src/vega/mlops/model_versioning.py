"""
Model Versioning & Registry System

Provides Git-based model registry with experiment tracking, model lineage,
and version management for personal ML workflows.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import hashlib
import json
import pickle
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import tempfile
import sqlite3
from contextlib import asynccontextmanager
import aiosqlite

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - some model operations disabled")

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("Joblib not available - some serialization options disabled")

try:
    import git

    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    logger.warning("GitPython not available - Git operations disabled")


class ModelFramework(Enum):
    """Supported ML frameworks"""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SCIKIT_LEARN = "scikit_learn"
    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    CUSTOM = "custom"


class ModelStage(Enum):
    """Model lifecycle stages"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ExperimentStatus(Enum):
    """Experiment status"""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelMetadata:
    """Model metadata information"""

    model_id: str
    name: str
    version: str
    framework: ModelFramework
    stage: ModelStage
    description: str
    tags: List[str]
    author: str
    created_at: datetime
    updated_at: datetime
    model_size_mb: float
    parameters_count: Optional[int] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    training_dataset: Optional[str] = None
    validation_metrics: Optional[Dict[str, float]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class Experiment:
    """Experiment tracking information"""

    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    created_at: datetime
    updated_at: datetime
    author: str
    parent_experiment_id: Optional[str] = None
    tags: List[str] = None
    parameters: Dict[str, Any] = None
    metrics: Dict[str, float] = None
    artifacts: List[str] = None
    models: List[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if self.tags is None:
            self.tags = []
        if self.parameters is None:
            self.parameters = {}
        if self.metrics is None:
            self.metrics = {}
        if self.artifacts is None:
            self.artifacts = []
        if self.models is None:
            self.models = []


@dataclass
class ModelVersion:
    """Model version information"""

    version_id: str
    model_id: str
    version: str
    parent_version: Optional[str]
    commit_hash: Optional[str]
    created_at: datetime
    author: str
    message: str
    metrics: Dict[str, float]
    file_path: str
    file_size_mb: float
    checksum: str

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if self.metrics is None:
            self.metrics = {}


class ModelSerializer:
    """
    Universal model serialization handler
    """

    @staticmethod
    def serialize_model(
        model: Any, framework: ModelFramework, output_path: str
    ) -> Dict[str, Any]:
        """Serialize model to file"""
        metadata = {
            "framework": framework.value,
            "serialized_at": datetime.now().isoformat(),
            "file_size_mb": 0.0,
            "parameters_count": None,
        }

        try:
            if framework == ModelFramework.PYTORCH and TORCH_AVAILABLE:
                if hasattr(model, "state_dict"):
                    torch.save(model.state_dict(), output_path)
                    metadata["parameters_count"] = sum(
                        p.numel() for p in model.parameters()
                    )
                else:
                    torch.save(model, output_path)

            elif framework == ModelFramework.SCIKIT_LEARN and JOBLIB_AVAILABLE:
                joblib.dump(model, output_path)

            elif framework == ModelFramework.TENSORFLOW:
                # TensorFlow model saving
                if hasattr(model, "save"):
                    model.save(output_path)
                else:
                    # Fallback to pickle
                    with open(output_path, "wb") as f:
                        pickle.dump(model, f)

            else:
                # Generic pickle serialization
                with open(output_path, "wb") as f:
                    pickle.dump(model, f)

            # Get file size
            file_size = os.path.getsize(output_path)
            metadata["file_size_mb"] = file_size / (1024 * 1024)

            return metadata

        except Exception as e:
            logger.error(f"Model serialization failed: {e}")
            raise

    @staticmethod
    def deserialize_model(file_path: str, framework: ModelFramework) -> Any:
        """Deserialize model from file"""
        try:
            if framework == ModelFramework.PYTORCH and TORCH_AVAILABLE:
                return torch.load(file_path, map_location="cpu")

            elif framework == ModelFramework.SCIKIT_LEARN and JOBLIB_AVAILABLE:
                return joblib.load(file_path)

            elif framework == ModelFramework.TENSORFLOW:
                # TensorFlow model loading
                try:
                    import tensorflow as tf

                    return tf.keras.models.load_model(file_path)
                except ImportError:
                    # Fallback to pickle
                    with open(file_path, "rb") as f:
                        return pickle.load(f)

            else:
                # Generic pickle deserialization
                with open(file_path, "rb") as f:
                    return pickle.load(f)

        except Exception as e:
            logger.error(f"Model deserialization failed: {e}")
            raise


class GitModelRepository:
    """
    Git-based model repository manager
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)
        self.repo = None
        self._initialize_repository()

    def _initialize_repository(self):
        """Initialize Git repository"""
        if not GIT_AVAILABLE:
            logger.warning("Git not available - version control disabled")
            return

        try:
            # Try to open existing repository
            self.repo = git.Repo(self.repo_path)
        except git.exc.InvalidGitRepositoryError:
            # Initialize new repository
            self.repo = git.Repo.init(self.repo_path)

            # Create initial commit
            gitignore_content = """
# Model files
*.pkl
*.pt
*.pth
*.h5
*.pb
*.onnx

# Temporary files
*.tmp
*.temp
__pycache__/
.DS_Store

# Large files (use Git LFS)
*.bin
models/large/
"""
            gitignore_path = self.repo_path / ".gitignore"
            gitignore_path.write_text(gitignore_content.strip())

            self.repo.index.add([".gitignore"])
            self.repo.index.commit("Initial commit: Model repository setup")

            logger.info("Initialized new Git model repository")

    def commit_model(
        self,
        file_paths: List[str],
        message: str,
        author_name: str = "Vega2.0",
        author_email: str = "vega@local",
    ) -> str:
        """Commit model files to repository"""
        if not self.repo:
            return "no-git"

        try:
            # Add files to staging
            for file_path in file_paths:
                if os.path.exists(file_path):
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    self.repo.index.add([rel_path])

            # Commit changes
            commit = self.repo.index.commit(
                message,
                author=git.Actor(author_name, author_email),
                committer=git.Actor(author_name, author_email),
            )

            logger.info(f"Committed model files: {commit.hexsha[:8]}")
            return commit.hexsha

        except Exception as e:
            logger.error(f"Git commit failed: {e}")
            return "commit-failed"

    def create_branch(self, branch_name: str) -> bool:
        """Create new branch"""
        if not self.repo:
            return False

        try:
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
            return True
        except Exception as e:
            logger.error(f"Branch creation failed: {e}")
            return False

    def get_commit_history(
        self, file_path: Optional[str] = None, max_count: int = 10
    ) -> List[Dict[str, Any]]:
        """Get commit history"""
        if not self.repo:
            return []

        try:
            commits = []
            for commit in self.repo.iter_commits(max_count=max_count):
                commit_info = {
                    "hash": commit.hexsha,
                    "short_hash": commit.hexsha[:8],
                    "message": commit.message.strip(),
                    "author": str(commit.author),
                    "authored_date": datetime.fromtimestamp(commit.authored_date),
                    "files_changed": list(commit.stats.files.keys()),
                }
                commits.append(commit_info)

            return commits

        except Exception as e:
            logger.error(f"Failed to get commit history: {e}")
            return []

    def checkout_commit(self, commit_hash: str) -> bool:
        """Checkout specific commit"""
        if not self.repo:
            return False

        try:
            self.repo.git.checkout(commit_hash)
            return True
        except Exception as e:
            logger.error(f"Checkout failed: {e}")
            return False


class ModelRegistry:
    """
    Central model registry with database backend
    """

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.registry_path / "registry.db"
        self.models_path = self.registry_path / "models"
        self.experiments_path = self.registry_path / "experiments"
        self.artifacts_path = self.registry_path / "artifacts"

        # Create directories
        self.models_path.mkdir(exist_ok=True)
        self.experiments_path.mkdir(exist_ok=True)
        self.artifacts_path.mkdir(exist_ok=True)

        # Initialize Git repository
        self.git_repo = GitModelRepository(str(self.registry_path))

        # Initialize database
        asyncio.create_task(self._initialize_database())

    async def _initialize_database(self):
        """Initialize SQLite database schema"""
        async with aiosqlite.connect(self.db_path) as db:
            # Models table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    author TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    model_size_mb REAL,
                    parameters_count INTEGER,
                    input_schema TEXT,
                    output_schema TEXT,
                    training_dataset TEXT,
                    validation_metrics TEXT,
                    hyperparameters TEXT,
                    dependencies TEXT,
                    UNIQUE(name, version)
                )
            """
            )

            # Model versions table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    version TEXT,
                    parent_version TEXT,
                    commit_hash TEXT,
                    created_at TEXT,
                    author TEXT,
                    message TEXT,
                    metrics TEXT,
                    file_path TEXT,
                    file_size_mb REAL,
                    checksum TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """
            )

            # Experiments table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    author TEXT,
                    parent_experiment_id TEXT,
                    tags TEXT,
                    parameters TEXT,
                    metrics TEXT,
                    artifacts TEXT,
                    models TEXT,
                    notes TEXT
                )
            """
            )

            # Create indexes
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_models_name ON models(name)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_models_framework ON models(framework)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_models_stage ON models(stage)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)"
            )

            await db.commit()

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    async def register_model(
        self, model: Any, metadata: ModelMetadata, experiment_id: Optional[str] = None
    ) -> str:
        """Register new model"""

        # Create model directory
        model_dir = self.models_path / metadata.model_id
        model_dir.mkdir(exist_ok=True)

        # Serialize model
        model_file_path = model_dir / f"{metadata.name}_{metadata.version}.model"
        serialization_metadata = ModelSerializer.serialize_model(
            model, metadata.framework, str(model_file_path)
        )

        # Update metadata with serialization info
        metadata.model_size_mb = serialization_metadata["file_size_mb"]
        if serialization_metadata["parameters_count"]:
            metadata.parameters_count = serialization_metadata["parameters_count"]

        # Calculate checksum
        checksum = self._calculate_checksum(str(model_file_path))

        # Save metadata
        metadata_file_path = model_dir / f"{metadata.name}_{metadata.version}.json"
        with open(metadata_file_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)

        # Store in database
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO models 
                (model_id, name, version, framework, stage, description, tags, author, 
                 created_at, updated_at, model_size_mb, parameters_count, input_schema, 
                 output_schema, training_dataset, validation_metrics, hyperparameters, dependencies)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metadata.model_id,
                    metadata.name,
                    metadata.version,
                    metadata.framework.value,
                    metadata.stage.value,
                    metadata.description,
                    json.dumps(metadata.tags),
                    metadata.author,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                    metadata.model_size_mb,
                    metadata.parameters_count,
                    (
                        json.dumps(metadata.input_schema)
                        if metadata.input_schema
                        else None
                    ),
                    (
                        json.dumps(metadata.output_schema)
                        if metadata.output_schema
                        else None
                    ),
                    metadata.training_dataset,
                    (
                        json.dumps(metadata.validation_metrics)
                        if metadata.validation_metrics
                        else None
                    ),
                    (
                        json.dumps(metadata.hyperparameters)
                        if metadata.hyperparameters
                        else None
                    ),
                    (
                        json.dumps(metadata.dependencies)
                        if metadata.dependencies
                        else None
                    ),
                ),
            )
            await db.commit()

        # Create model version
        version_id = f"{metadata.model_id}_v{metadata.version}_{int(time.time())}"

        # Commit to Git
        commit_hash = self.git_repo.commit_model(
            [str(model_file_path), str(metadata_file_path)],
            f"Register model {metadata.name} v{metadata.version}",
            metadata.author,
        )

        # Store version info
        model_version = ModelVersion(
            version_id=version_id,
            model_id=metadata.model_id,
            version=metadata.version,
            parent_version=None,
            commit_hash=commit_hash,
            created_at=datetime.now(),
            author=metadata.author,
            message=f"Initial version of {metadata.name}",
            metrics=metadata.validation_metrics or {},
            file_path=str(model_file_path),
            file_size_mb=metadata.model_size_mb,
            checksum=checksum,
        )

        await self._store_model_version(model_version)

        # Link to experiment if provided
        if experiment_id:
            await self._link_model_to_experiment(experiment_id, metadata.model_id)

        logger.info(f"Registered model {metadata.name} v{metadata.version}")
        return metadata.model_id

    async def _store_model_version(self, version: ModelVersion):
        """Store model version in database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO model_versions 
                (version_id, model_id, version, parent_version, commit_hash, created_at, 
                 author, message, metrics, file_path, file_size_mb, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    version.version_id,
                    version.model_id,
                    version.version,
                    version.parent_version,
                    version.commit_hash,
                    version.created_at.isoformat(),
                    version.author,
                    version.message,
                    json.dumps(version.metrics),
                    version.file_path,
                    version.file_size_mb,
                    version.checksum,
                ),
            )
            await db.commit()

    async def load_model(
        self, model_id: str, version: Optional[str] = None
    ) -> Tuple[Any, ModelMetadata]:
        """Load model from registry"""
        # Get model metadata
        metadata = await self.get_model_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")

        # Find model file
        model_dir = self.models_path / model_id
        if version:
            model_file_path = model_dir / f"{metadata.name}_{version}.model"
        else:
            model_file_path = model_dir / f"{metadata.name}_{metadata.version}.model"

        if not model_file_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_file_path}")

        # Deserialize model
        model = ModelSerializer.deserialize_model(
            str(model_file_path), metadata.framework
        )

        return model, metadata

    async def get_model_metadata(
        self, model_id: str, version: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """Get model metadata"""
        async with aiosqlite.connect(self.db_path) as db:
            if version:
                cursor = await db.execute(
                    "SELECT * FROM models WHERE model_id = ? AND version = ?",
                    (model_id, version),
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM models WHERE model_id = ? ORDER BY updated_at DESC LIMIT 1",
                    (model_id,),
                )

            row = await cursor.fetchone()
            if not row:
                return None

            # Convert row to metadata
            columns = [description[0] for description in cursor.description]
            data = dict(zip(columns, row))

            # Parse JSON fields
            for field in [
                "tags",
                "input_schema",
                "output_schema",
                "validation_metrics",
                "hyperparameters",
                "dependencies",
            ]:
                if data[field]:
                    data[field] = json.loads(data[field])

            data["framework"] = ModelFramework(data["framework"])
            data["stage"] = ModelStage(data["stage"])

            return ModelMetadata(**data)

    async def list_models(
        self,
        framework: Optional[ModelFramework] = None,
        stage: Optional[ModelStage] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelMetadata]:
        """List models with optional filtering"""
        query = "SELECT * FROM models WHERE 1=1"
        params = []

        if framework:
            query += " AND framework = ?"
            params.append(framework.value)

        if stage:
            query += " AND stage = ?"
            params.append(stage.value)

        query += " ORDER BY updated_at DESC"

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            models = []
            for row in rows:
                columns = [description[0] for description in cursor.description]
                data = dict(zip(columns, row))

                # Parse JSON fields
                for field in [
                    "tags",
                    "input_schema",
                    "output_schema",
                    "validation_metrics",
                    "hyperparameters",
                    "dependencies",
                ]:
                    if data[field]:
                        data[field] = json.loads(data[field])

                data["framework"] = ModelFramework(data["framework"])
                data["stage"] = ModelStage(data["stage"])

                metadata = ModelMetadata(**data)

                # Filter by tags if specified
                if tags and not any(tag in metadata.tags for tag in tags):
                    continue

                models.append(metadata)

            return models

    async def create_experiment(self, experiment: Experiment) -> str:
        """Create new experiment"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO experiments 
                (experiment_id, name, description, status, created_at, updated_at, author,
                 parent_experiment_id, tags, parameters, metrics, artifacts, models, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment.experiment_id,
                    experiment.name,
                    experiment.description,
                    experiment.status.value,
                    experiment.created_at.isoformat(),
                    experiment.updated_at.isoformat(),
                    experiment.author,
                    experiment.parent_experiment_id,
                    json.dumps(experiment.tags),
                    json.dumps(experiment.parameters),
                    json.dumps(experiment.metrics),
                    json.dumps(experiment.artifacts),
                    json.dumps(experiment.models),
                    experiment.notes,
                ),
            )
            await db.commit()

        # Create experiment directory
        exp_dir = self.experiments_path / experiment.experiment_id
        exp_dir.mkdir(exist_ok=True)

        # Save experiment metadata
        metadata_file = exp_dir / "experiment.json"
        with open(metadata_file, "w") as f:
            json.dump(asdict(experiment), f, indent=2, default=str)

        logger.info(f"Created experiment: {experiment.name}")
        return experiment.experiment_id

    async def update_experiment(
        self,
        experiment_id: str,
        metrics: Optional[Dict[str, float]] = None,
        status: Optional[ExperimentStatus] = None,
        notes: Optional[str] = None,
    ):
        """Update experiment"""
        updates = []
        params = []

        if metrics:
            updates.append("metrics = ?")
            params.append(json.dumps(metrics))

        if status:
            updates.append("status = ?")
            params.append(status.value)

        if notes:
            updates.append("notes = ?")
            params.append(notes)

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())

        params.append(experiment_id)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f'UPDATE experiments SET {", ".join(updates)} WHERE experiment_id = ?',
                params,
            )
            await db.commit()

    async def _link_model_to_experiment(self, experiment_id: str, model_id: str):
        """Link model to experiment"""
        async with aiosqlite.connect(self.db_path) as db:
            # Get current models list
            cursor = await db.execute(
                "SELECT models FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            row = await cursor.fetchone()

            if row:
                current_models = json.loads(row[0]) if row[0] else []
                if model_id not in current_models:
                    current_models.append(model_id)

                    await db.execute(
                        "UPDATE experiments SET models = ? WHERE experiment_id = ?",
                        (json.dumps(current_models), experiment_id),
                    )
                    await db.commit()

    async def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get model lineage and relationships"""
        lineage = {
            "model_id": model_id,
            "versions": [],
            "experiments": [],
            "descendants": [],
            "ancestors": [],
        }

        # Get all versions
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM model_versions WHERE model_id = ? ORDER BY created_at",
                (model_id,),
            )
            rows = await cursor.fetchall()

            for row in rows:
                columns = [description[0] for description in cursor.description]
                version_data = dict(zip(columns, row))
                version_data["metrics"] = json.loads(version_data["metrics"])
                lineage["versions"].append(version_data)

            # Get linked experiments
            cursor = await db.execute(
                "SELECT * FROM experiments WHERE models LIKE ?", (f'%"{model_id}"%',)
            )
            rows = await cursor.fetchall()

            for row in rows:
                columns = [description[0] for description in cursor.description]
                exp_data = dict(zip(columns, row))
                lineage["experiments"].append(exp_data)

        return lineage

    async def promote_model(self, model_id: str, target_stage: ModelStage) -> bool:
        """Promote model to different stage"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE models SET stage = ?, updated_at = ? WHERE model_id = ?",
                (target_stage.value, datetime.now().isoformat(), model_id),
            )
            await db.commit()

        logger.info(f"Promoted model {model_id} to {target_stage.value}")
        return True

    async def delete_model(self, model_id: str) -> bool:
        """Delete model and all its versions"""
        try:
            # Remove from database
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "DELETE FROM model_versions WHERE model_id = ?", (model_id,)
                )
                await db.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
                await db.commit()

            # Remove model directory
            model_dir = self.models_path / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)

            logger.info(f"Deleted model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        stats = {
            "total_models": 0,
            "total_experiments": 0,
            "total_versions": 0,
            "models_by_framework": {},
            "models_by_stage": {},
            "total_storage_mb": 0,
            "last_updated": datetime.now().isoformat(),
        }

        async with aiosqlite.connect(self.db_path) as db:
            # Count models
            cursor = await db.execute("SELECT COUNT(*) FROM models")
            stats["total_models"] = (await cursor.fetchone())[0]

            # Count experiments
            cursor = await db.execute("SELECT COUNT(*) FROM experiments")
            stats["total_experiments"] = (await cursor.fetchone())[0]

            # Count versions
            cursor = await db.execute("SELECT COUNT(*) FROM model_versions")
            stats["total_versions"] = (await cursor.fetchone())[0]

            # Models by framework
            cursor = await db.execute(
                "SELECT framework, COUNT(*) FROM models GROUP BY framework"
            )
            rows = await cursor.fetchall()
            for framework, count in rows:
                stats["models_by_framework"][framework] = count

            # Models by stage
            cursor = await db.execute(
                "SELECT stage, COUNT(*) FROM models GROUP BY stage"
            )
            rows = await cursor.fetchall()
            for stage, count in rows:
                stats["models_by_stage"][stage] = count

            # Total storage
            cursor = await db.execute("SELECT SUM(model_size_mb) FROM models")
            total_size = (await cursor.fetchone())[0]
            stats["total_storage_mb"] = total_size or 0

        return stats


# Demo and testing functions
async def demo_model_versioning():
    """Demonstrate model versioning capabilities"""

    print("Model Versioning & Registry Demo")

    # Initialize registry
    registry = ModelRegistry("data/model_registry")
    await registry._initialize_database()

    # Create sample experiment
    experiment = Experiment(
        experiment_id="exp_001",
        name="Text Classification Experiment",
        description="Training a text classifier for sentiment analysis",
        status=ExperimentStatus.RUNNING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        author="Vega2.0",
        parameters={"learning_rate": 0.001, "batch_size": 32, "epochs": 10},
    )

    await registry.create_experiment(experiment)
    print(f"Created experiment: {experiment.name}")

    # Register a sample model
    if TORCH_AVAILABLE:
        import torch.nn as nn

        # Create simple model
        model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 2))

        metadata = ModelMetadata(
            model_id="text_classifier_v1",
            name="text_classifier",
            version="1.0.0",
            framework=ModelFramework.PYTORCH,
            stage=ModelStage.DEVELOPMENT,
            description="Simple text classification model",
            tags=["nlp", "classification", "pytorch"],
            author="Vega2.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model_size_mb=0.0,  # Will be calculated
            validation_metrics={
                "accuracy": 0.85,
                "f1_score": 0.82,
                "precision": 0.84,
                "recall": 0.80,
            },
            hyperparameters={
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
            },
            dependencies=["torch>=1.9.0", "numpy>=1.20.0"],
        )

        model_id = await registry.register_model(
            model, metadata, experiment.experiment_id
        )
        print(f"Registered model: {model_id}")

        # Load model back
        loaded_model, loaded_metadata = await registry.load_model(model_id)
        print(f"Loaded model: {loaded_metadata.name} v{loaded_metadata.version}")
        print(f"Model size: {loaded_metadata.model_size_mb:.2f} MB")
        print(f"Parameters: {loaded_metadata.parameters_count:,}")

    else:
        # Use scikit-learn model as fallback
        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(random_state=42)

            metadata = ModelMetadata(
                model_id="sklearn_classifier_v1",
                name="sklearn_classifier",
                version="1.0.0",
                framework=ModelFramework.SCIKIT_LEARN,
                stage=ModelStage.DEVELOPMENT,
                description="Scikit-learn logistic regression classifier",
                tags=["sklearn", "classification", "logistic_regression"],
                author="Vega2.0",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                model_size_mb=0.0,
                validation_metrics={"accuracy": 0.78, "f1_score": 0.75},
            )

            model_id = await registry.register_model(
                model, metadata, experiment.experiment_id
            )
            print(f"Registered scikit-learn model: {model_id}")

        except ImportError:
            print("Neither PyTorch nor scikit-learn available - using dummy model")

            # Create dummy model
            dummy_model = {"type": "dummy", "parameters": [1, 2, 3]}

            metadata = ModelMetadata(
                model_id="dummy_model_v1",
                name="dummy_model",
                version="1.0.0",
                framework=ModelFramework.CUSTOM,
                stage=ModelStage.DEVELOPMENT,
                description="Dummy model for demonstration",
                tags=["demo", "dummy"],
                author="Vega2.0",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                model_size_mb=0.0,
            )

            model_id = await registry.register_model(
                dummy_model, metadata, experiment.experiment_id
            )
            print(f"Registered dummy model: {model_id}")

    # Update experiment
    await registry.update_experiment(
        experiment.experiment_id,
        metrics={"final_accuracy": 0.87, "training_loss": 0.23},
        status=ExperimentStatus.COMPLETED,
        notes="Training completed successfully",
    )

    # List models
    models = await registry.list_models()
    print(f"\nRegistered models ({len(models)}):")
    for model_meta in models:
        print(
            f"- {model_meta.name} v{model_meta.version} ({model_meta.framework.value})"
        )
        print(f"  Stage: {model_meta.stage.value}")
        print(f"  Size: {model_meta.model_size_mb:.2f} MB")
        if model_meta.validation_metrics:
            print(f"  Metrics: {model_meta.validation_metrics}")

    # Get model lineage
    lineage = await registry.get_model_lineage(model_id)
    print(f"\nModel lineage for {model_id}:")
    print(f"- Versions: {len(lineage['versions'])}")
    print(f"- Experiments: {len(lineage['experiments'])}")

    # Get registry statistics
    stats = await registry.get_registry_stats()
    print(f"\nRegistry Statistics:")
    print(f"- Total models: {stats['total_models']}")
    print(f"- Total experiments: {stats['total_experiments']}")
    print(f"- Total storage: {stats['total_storage_mb']:.2f} MB")
    print(f"- Models by framework: {stats['models_by_framework']}")
    print(f"- Models by stage: {stats['models_by_stage']}")

    return registry


if __name__ == "__main__":
    asyncio.run(demo_model_versioning())
