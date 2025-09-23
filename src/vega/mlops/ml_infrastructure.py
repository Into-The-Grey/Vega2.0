"""
Personal ML Infrastructure

Complete ML infrastructure for personal use including feature stores,
model serving endpoints, batch inference pipelines, and workflow automation.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import json
import time
import threading
import warnings
import pickle
import joblib
import hashlib
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Protocol
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
from contextlib import asynccontextmanager
import aiosqlite
from abc import ABC, abstractmethod
import tempfile
import shutil
import os

logger = logging.getLogger(__name__)

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import httpx

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available - API serving disabled")

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - some caching features disabled")

try:
    import schedule

    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    logger.warning("Schedule not available - workflow automation limited")

try:
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        String,
        Float,
        DateTime,
        Text,
        Boolean,
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session

    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not available - advanced database features disabled")


class ModelFormat(Enum):
    """Supported model formats"""

    PICKLE = "pickle"
    JOBLIB = "joblib"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    SKLEARN = "sklearn"


class ServingStatus(Enum):
    """Model serving status"""

    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    STOPPED = "stopped"


class InferenceType(Enum):
    """Types of inference"""

    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAM = "stream"


class WorkflowStatus(Enum):
    """Workflow execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FeatureSpec:
    """Feature specification"""

    name: str
    dtype: str
    description: str
    source: Optional[str] = None
    transformation: Optional[str] = None
    validation_rules: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        elif isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)


@dataclass
class FeatureValue:
    """Feature value with metadata"""

    feature_name: str
    value: Any
    entity_id: str
    timestamp: datetime
    version: str = "1.0"

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class ModelEndpoint:
    """Model serving endpoint configuration"""

    endpoint_id: str
    model_id: str
    model_version: str
    endpoint_url: str
    status: ServingStatus
    inference_type: InferenceType
    created_at: datetime
    last_request_at: Optional[datetime] = None
    request_count: int = 0
    error_count: int = 0

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_request_at, str):
            self.last_request_at = datetime.fromisoformat(self.last_request_at)


@dataclass
class BatchInferenceJob:
    """Batch inference job configuration"""

    job_id: str
    model_id: str
    input_data_path: str
    output_data_path: str
    status: WorkflowStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processed_records: int = 0
    error_message: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.started_at, str):
            self.started_at = datetime.fromisoformat(self.started_at)
        if isinstance(self.completed_at, str):
            self.completed_at = datetime.fromisoformat(self.completed_at)


@dataclass
class MLWorkflow:
    """ML workflow definition"""

    workflow_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    schedule: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        elif isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_run_at, str):
            self.last_run_at = datetime.fromisoformat(self.last_run_at)


# Pydantic models for API
if FASTAPI_AVAILABLE:

    class PredictionRequest(BaseModel):
        model_id: str
        features: Dict[str, Any]
        return_confidence: bool = False

    class PredictionResponse(BaseModel):
        prediction: Any
        confidence: Optional[float] = None
        model_id: str
        timestamp: str

    class BatchJobRequest(BaseModel):
        model_id: str
        input_data_path: str
        output_data_path: str

    class BatchJobResponse(BaseModel):
        job_id: str
        status: str
        message: str


class FeatureStore:
    """Feature store for ML features"""

    def __init__(self, db_path: str = "data/feature_store.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.feature_specs = {}
        self.cache = {}

        asyncio.create_task(self._initialize_database())

    async def _initialize_database(self):
        """Initialize feature store database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_specs (
                    name TEXT PRIMARY KEY,
                    dtype TEXT,
                    description TEXT,
                    source TEXT,
                    transformation TEXT,
                    validation_rules TEXT,
                    created_at TEXT
                )
            """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT,
                    value TEXT,
                    entity_id TEXT,
                    timestamp TEXT,
                    version TEXT,
                    FOREIGN KEY (feature_name) REFERENCES feature_specs (name)
                )
            """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feature_entity_time 
                ON feature_values (feature_name, entity_id, timestamp)
            """
            )

            await db.commit()

    async def register_feature(self, feature_spec: FeatureSpec):
        """Register a new feature"""
        self.feature_specs[feature_spec.name] = feature_spec

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO feature_specs 
                (name, dtype, description, source, transformation, validation_rules, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    feature_spec.name,
                    feature_spec.dtype,
                    feature_spec.description,
                    feature_spec.source,
                    feature_spec.transformation,
                    (
                        json.dumps(feature_spec.validation_rules)
                        if feature_spec.validation_rules
                        else None
                    ),
                    feature_spec.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def store_feature_value(self, feature_value: FeatureValue):
        """Store a feature value"""
        if feature_value.feature_name not in self.feature_specs:
            raise ValueError(f"Feature {feature_value.feature_name} not registered")

        # Validate feature value
        await self._validate_feature_value(feature_value)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO feature_values 
                (feature_name, value, entity_id, timestamp, version)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    feature_value.feature_name,
                    json.dumps(feature_value.value),
                    feature_value.entity_id,
                    feature_value.timestamp.isoformat(),
                    feature_value.version,
                ),
            )
            await db.commit()

        # Update cache
        cache_key = f"{feature_value.feature_name}:{feature_value.entity_id}"
        self.cache[cache_key] = feature_value

    async def get_feature_value(
        self, feature_name: str, entity_id: str, timestamp: Optional[datetime] = None
    ) -> Optional[FeatureValue]:
        """Get feature value for entity"""

        # Check cache first
        cache_key = f"{feature_name}:{entity_id}"
        if cache_key in self.cache and timestamp is None:
            return self.cache[cache_key]

        # Query database
        query = """
            SELECT value, timestamp, version 
            FROM feature_values 
            WHERE feature_name = ? AND entity_id = ?
        """
        params = [feature_name, entity_id]

        if timestamp:
            query += " AND timestamp <= ?"
            params.append(timestamp.isoformat())

        query += " ORDER BY timestamp DESC LIMIT 1"

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            row = await cursor.fetchone()

            if row:
                value, ts, version = row
                return FeatureValue(
                    feature_name=feature_name,
                    value=json.loads(value),
                    entity_id=entity_id,
                    timestamp=datetime.fromisoformat(ts),
                    version=version,
                )

        return None

    async def get_feature_vector(
        self,
        feature_names: List[str],
        entity_id: str,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get feature vector for entity"""

        feature_vector = {}

        for feature_name in feature_names:
            feature_value = await self.get_feature_value(
                feature_name, entity_id, timestamp
            )
            if feature_value:
                feature_vector[feature_name] = feature_value.value
            else:
                # Use default value if available
                if feature_name in self.feature_specs:
                    spec = self.feature_specs[feature_name]
                    if spec.validation_rules and "default" in spec.validation_rules:
                        feature_vector[feature_name] = spec.validation_rules["default"]
                    else:
                        feature_vector[feature_name] = None

        return feature_vector

    async def _validate_feature_value(self, feature_value: FeatureValue):
        """Validate feature value against spec"""
        spec = self.feature_specs[feature_value.feature_name]

        if spec.validation_rules:
            rules = spec.validation_rules

            # Type validation
            if "type" in rules:
                expected_type = rules["type"]
                if not isinstance(feature_value.value, expected_type):
                    raise ValueError(
                        f"Feature {feature_value.feature_name} expects {expected_type}, got {type(feature_value.value)}"
                    )

            # Range validation
            if "min" in rules and feature_value.value < rules["min"]:
                raise ValueError(
                    f"Feature {feature_value.feature_name} value {feature_value.value} below minimum {rules['min']}"
                )

            if "max" in rules and feature_value.value > rules["max"]:
                raise ValueError(
                    f"Feature {feature_value.feature_name} value {feature_value.value} above maximum {rules['max']}"
                )

            # Enum validation
            if "choices" in rules and feature_value.value not in rules["choices"]:
                raise ValueError(
                    f"Feature {feature_value.feature_name} value {feature_value.value} not in allowed choices {rules['choices']}"
                )

    async def list_features(self) -> List[FeatureSpec]:
        """List all registered features"""
        return list(self.feature_specs.values())

    async def delete_feature(self, feature_name: str):
        """Delete a feature and all its values"""
        if feature_name in self.feature_specs:
            del self.feature_specs[feature_name]

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM feature_values WHERE feature_name = ?", (feature_name,)
            )
            await db.execute(
                "DELETE FROM feature_specs WHERE name = ?", (feature_name,)
            )
            await db.commit()


class ModelRegistry:
    """Registry for trained models"""

    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.load_models()

    def load_models(self):
        """Load available models from disk"""
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                try:
                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                        self.models[metadata["model_id"]] = {
                            "metadata": metadata,
                            "path": model_dir,
                            "model": None,  # Lazy load
                        }
                except Exception as e:
                    logger.warning(f"Failed to load model from {model_dir}: {e}")

    def register_model(self, model_id: str, model, metadata: Dict[str, Any]) -> str:
        """Register a new model"""
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)

        # Save model
        model_path = model_dir / "model.pkl"

        try:
            if hasattr(model, "save"):
                # TensorFlow/Keras model
                model.save(str(model_dir / "model"))
                metadata["format"] = ModelFormat.TENSORFLOW.value
            elif hasattr(model, "state_dict"):
                # PyTorch model
                import torch

                torch.save(model.state_dict(), model_path)
                metadata["format"] = ModelFormat.PYTORCH.value
            else:
                # Scikit-learn or other pickle-able model
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                metadata["format"] = ModelFormat.PICKLE.value
        except Exception as e:
            # Fallback to joblib
            joblib.dump(model, model_path)
            metadata["format"] = ModelFormat.JOBLIB.value

        # Save metadata
        metadata["model_id"] = model_id
        metadata["created_at"] = datetime.now().isoformat()
        metadata["file_path"] = str(model_path)

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update registry
        self.models[model_id] = {
            "metadata": metadata,
            "path": model_dir,
            "model": model,
        }

        return model_id

    def load_model(self, model_id: str):
        """Load model into memory"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model_info = self.models[model_id]

        if model_info["model"] is None:
            # Load model from disk
            metadata = model_info["metadata"]
            model_format = ModelFormat(metadata["format"])
            model_path = Path(metadata["file_path"])

            try:
                if model_format == ModelFormat.TENSORFLOW:
                    import tensorflow as tf

                    model = tf.keras.models.load_model(str(model_path.parent / "model"))
                elif model_format == ModelFormat.PYTORCH:
                    import torch

                    model = torch.load(model_path)
                elif model_format == ModelFormat.JOBLIB:
                    model = joblib.load(model_path)
                else:
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)

                model_info["model"] = model
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                raise

        return model_info["model"]

    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get model metadata"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        return self.models[model_id]["metadata"]

    def list_models(self) -> List[str]:
        """List available models"""
        return list(self.models.keys())

    def delete_model(self, model_id: str):
        """Delete a model"""
        if model_id in self.models:
            model_info = self.models[model_id]
            shutil.rmtree(model_info["path"])
            del self.models[model_id]


class ModelServingEngine:
    """Engine for serving models via API"""

    def __init__(self, model_registry: ModelRegistry, feature_store: FeatureStore):
        self.model_registry = model_registry
        self.feature_store = feature_store
        self.endpoints = {}
        self.app = None

        if FASTAPI_AVAILABLE:
            self._setup_api()

    def _setup_api(self):
        """Setup FastAPI application"""
        self.app = FastAPI(title="Personal ML Serving API", version="1.0.0")

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add endpoints
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            return await self._handle_prediction(request)

        @self.app.get("/models")
        async def list_models():
            return {"models": self.model_registry.list_models()}

        @self.app.get("/models/{model_id}/metadata")
        async def get_model_metadata(model_id: str):
            try:
                metadata = self.model_registry.get_model_metadata(model_id)
                return {"model_id": model_id, "metadata": metadata}
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @self.app.get("/endpoints")
        async def list_endpoints():
            return {"endpoints": [asdict(ep) for ep in self.endpoints.values()]}

        @self.app.post("/endpoints")
        async def create_endpoint(model_id: str, inference_type: str = "real_time"):
            endpoint = await self.create_model_endpoint(
                model_id, InferenceType(inference_type)
            )
            return {"endpoint": asdict(endpoint)}

        @self.app.delete("/endpoints/{endpoint_id}")
        async def delete_endpoint(endpoint_id: str):
            success = await self.delete_model_endpoint(endpoint_id)
            return {"success": success}

    async def _handle_prediction(
        self, request: PredictionRequest
    ) -> PredictionResponse:
        """Handle prediction request"""
        try:
            # Load model
            model = self.model_registry.load_model(request.model_id)

            # Prepare features
            features = request.features

            # Convert to appropriate format
            if isinstance(features, dict):
                # Convert to array for sklearn models
                feature_values = list(features.values())
                X = np.array(feature_values).reshape(1, -1)
            else:
                X = np.array(features).reshape(1, -1)

            # Make prediction
            prediction = model.predict(X)[0]

            # Get confidence if requested
            confidence = None
            if request.return_confidence and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))

            # Update endpoint stats
            endpoint_id = f"endpoint_{request.model_id}"
            if endpoint_id in self.endpoints:
                endpoint = self.endpoints[endpoint_id]
                endpoint.request_count += 1
                endpoint.last_request_at = datetime.now()

            return PredictionResponse(
                prediction=(
                    prediction.tolist() if hasattr(prediction, "tolist") else prediction
                ),
                confidence=confidence,
                model_id=request.model_id,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Update error count
            endpoint_id = f"endpoint_{request.model_id}"
            if endpoint_id in self.endpoints:
                self.endpoints[endpoint_id].error_count += 1

            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    async def create_model_endpoint(
        self, model_id: str, inference_type: InferenceType = InferenceType.REAL_TIME
    ) -> ModelEndpoint:
        """Create serving endpoint for model"""

        if model_id not in self.model_registry.models:
            raise ValueError(f"Model {model_id} not found")

        endpoint_id = f"endpoint_{model_id}"
        endpoint_url = f"/predict"  # In real deployment, this would be full URL

        endpoint = ModelEndpoint(
            endpoint_id=endpoint_id,
            model_id=model_id,
            model_version="1.0",
            endpoint_url=endpoint_url,
            status=ServingStatus.LOADING,
            inference_type=inference_type,
            created_at=datetime.now(),
        )

        try:
            # Load model to verify it works
            self.model_registry.load_model(model_id)
            endpoint.status = ServingStatus.READY
        except Exception as e:
            endpoint.status = ServingStatus.ERROR
            logger.error(f"Failed to load model {model_id}: {e}")

        self.endpoints[endpoint_id] = endpoint
        return endpoint

    async def delete_model_endpoint(self, endpoint_id: str) -> bool:
        """Delete serving endpoint"""
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
            return True
        return False

    def start_server(self, host: str = "127.0.0.1", port: int = 8080):
        """Start serving API"""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available - cannot start server")

        if self.app is None:
            raise RuntimeError("API not initialized")

        uvicorn.run(self.app, host=host, port=port)


class BatchInferenceEngine:
    """Engine for batch inference jobs"""

    def __init__(self, model_registry: ModelRegistry, feature_store: FeatureStore):
        self.model_registry = model_registry
        self.feature_store = feature_store
        self.jobs = {}
        self.job_queue = asyncio.Queue()
        self.worker_task = None

        # Start worker
        self.start_worker()

    def start_worker(self):
        """Start background worker for batch jobs"""
        self.worker_task = asyncio.create_task(self._batch_worker())

    async def _batch_worker(self):
        """Background worker for processing batch jobs"""
        while True:
            try:
                job_id = await self.job_queue.get()
                await self._process_batch_job(job_id)
            except Exception as e:
                logger.error(f"Batch worker error: {e}")

    async def submit_batch_job(
        self, model_id: str, input_data_path: str, output_data_path: str
    ) -> str:
        """Submit a batch inference job"""

        job_id = f"batch_{model_id}_{int(time.time())}"

        job = BatchInferenceJob(
            job_id=job_id,
            model_id=model_id,
            input_data_path=input_data_path,
            output_data_path=output_data_path,
            status=WorkflowStatus.PENDING,
            created_at=datetime.now(),
        )

        self.jobs[job_id] = job

        # Add to queue
        await self.job_queue.put(job_id)

        return job_id

    async def _process_batch_job(self, job_id: str):
        """Process a batch inference job"""
        job = self.jobs[job_id]

        try:
            job.status = WorkflowStatus.RUNNING
            job.started_at = datetime.now()

            # Load model
            model = self.model_registry.load_model(job.model_id)

            # Load input data
            input_path = Path(job.input_data_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            if input_path.suffix == ".csv":
                data = pd.read_csv(input_path)
            elif input_path.suffix == ".json":
                data = pd.read_json(input_path)
            else:
                raise ValueError(f"Unsupported input format: {input_path.suffix}")

            # Make predictions
            predictions = []

            for idx, row in data.iterrows():
                try:
                    # Convert row to feature array
                    features = row.values.reshape(1, -1)
                    prediction = model.predict(features)[0]

                    predictions.append(
                        {
                            "index": idx,
                            "prediction": (
                                prediction.tolist()
                                if hasattr(prediction, "tolist")
                                else prediction
                            ),
                        }
                    )

                    job.processed_records += 1

                except Exception as e:
                    logger.warning(f"Failed to predict for row {idx}: {e}")
                    predictions.append(
                        {"index": idx, "prediction": None, "error": str(e)}
                    )

            # Save results
            output_path = Path(job.output_data_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            results_df = pd.DataFrame(predictions)

            if output_path.suffix == ".csv":
                results_df.to_csv(output_path, index=False)
            elif output_path.suffix == ".json":
                results_df.to_json(output_path, orient="records")
            else:
                # Default to JSON
                results_df.to_json(output_path.with_suffix(".json"), orient="records")

            job.status = WorkflowStatus.COMPLETED
            job.completed_at = datetime.now()

        except Exception as e:
            job.status = WorkflowStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"Batch job {job_id} failed: {e}")

    def get_job_status(self, job_id: str) -> Optional[BatchInferenceJob]:
        """Get job status"""
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[BatchInferenceJob]:
        """List all jobs"""
        return list(self.jobs.values())


class WorkflowEngine:
    """Engine for ML workflow automation"""

    def __init__(
        self,
        model_registry: ModelRegistry,
        feature_store: FeatureStore,
        batch_engine: BatchInferenceEngine,
    ):
        self.model_registry = model_registry
        self.feature_store = feature_store
        self.batch_engine = batch_engine

        self.workflows = {}
        self.workflow_runs = {}

        if SCHEDULE_AVAILABLE:
            self.scheduler_thread = threading.Thread(
                target=self._run_scheduler, daemon=True
            )
            self.scheduler_thread.start()

    def _run_scheduler(self):
        """Run scheduled workflows"""
        while True:
            if SCHEDULE_AVAILABLE:
                schedule.run_pending()
            time.sleep(60)  # Check every minute

    async def create_workflow(self, workflow: MLWorkflow):
        """Create a new workflow"""
        self.workflows[workflow.workflow_id] = workflow

        # Schedule if needed
        if SCHEDULE_AVAILABLE and workflow.schedule:
            schedule.every().day.at(workflow.schedule).do(
                lambda: asyncio.create_task(self.run_workflow(workflow.workflow_id))
            )

    async def run_workflow(self, workflow_id: str) -> str:
        """Run a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]
        run_id = f"run_{workflow_id}_{int(time.time())}"

        run_info = {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "status": WorkflowStatus.RUNNING,
            "started_at": datetime.now(),
            "completed_at": None,
            "steps_completed": 0,
            "error_message": None,
        }

        self.workflow_runs[run_id] = run_info

        try:
            for i, step in enumerate(workflow.steps):
                await self._execute_workflow_step(step, run_info)
                run_info["steps_completed"] = i + 1

            run_info["status"] = WorkflowStatus.COMPLETED
            run_info["completed_at"] = datetime.now()

            # Update workflow last run
            workflow.last_run_at = datetime.now()

        except Exception as e:
            run_info["status"] = WorkflowStatus.FAILED
            run_info["completed_at"] = datetime.now()
            run_info["error_message"] = str(e)
            logger.error(f"Workflow {workflow_id} failed: {e}")

        return run_id

    async def _execute_workflow_step(
        self, step: Dict[str, Any], run_info: Dict[str, Any]
    ):
        """Execute a single workflow step"""
        step_type = step.get("type")

        if step_type == "batch_inference":
            # Submit batch inference job
            job_id = await self.batch_engine.submit_batch_job(
                step["model_id"], step["input_path"], step["output_path"]
            )

            # Wait for completion
            while True:
                job = self.batch_engine.get_job_status(job_id)
                if job.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                    break
                await asyncio.sleep(5)

            if job.status == WorkflowStatus.FAILED:
                raise Exception(f"Batch job failed: {job.error_message}")

        elif step_type == "data_processing":
            # Execute data processing script
            script_path = step.get("script_path")
            if script_path:
                result = subprocess.run(
                    ["python", script_path], capture_output=True, text=True
                )

                if result.returncode != 0:
                    raise Exception(f"Data processing failed: {result.stderr}")

        elif step_type == "model_training":
            # This would trigger model training
            # For now, just simulate
            await asyncio.sleep(1)

        elif step_type == "model_evaluation":
            # This would trigger model evaluation
            # For now, just simulate
            await asyncio.sleep(1)

        elif step_type == "feature_engineering":
            # Execute feature engineering
            feature_script = step.get("feature_script")
            if feature_script:
                # This would execute feature engineering logic
                await asyncio.sleep(1)

        else:
            logger.warning(f"Unknown workflow step type: {step_type}")

    def get_workflow(self, workflow_id: str) -> Optional[MLWorkflow]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)

    def list_workflows(self) -> List[MLWorkflow]:
        """List all workflows"""
        return list(self.workflows.values())

    def get_workflow_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow run info"""
        return self.workflow_runs.get(run_id)

    def list_workflow_runs(
        self, workflow_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List workflow runs"""
        runs = list(self.workflow_runs.values())

        if workflow_id:
            runs = [r for r in runs if r["workflow_id"] == workflow_id]

        return sorted(runs, key=lambda x: x["started_at"], reverse=True)


class PersonalMLInfrastructure:
    """Complete personal ML infrastructure"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize components
        self.feature_store = FeatureStore(
            self.config.get("feature_store_db", "data/feature_store.db")
        )

        self.model_registry = ModelRegistry(
            self.config.get("models_dir", "data/models")
        )

        self.serving_engine = ModelServingEngine(
            self.model_registry, self.feature_store
        )

        self.batch_engine = BatchInferenceEngine(
            self.model_registry, self.feature_store
        )

        self.workflow_engine = WorkflowEngine(
            self.model_registry, self.feature_store, self.batch_engine
        )

        logger.info("Personal ML Infrastructure initialized")

    async def setup_demo_environment(self):
        """Setup demo environment with sample data"""

        # Register sample features
        await self.feature_store.register_feature(
            FeatureSpec(
                name="age",
                dtype="int",
                description="Customer age",
                validation_rules={"min": 0, "max": 120, "type": int},
            )
        )

        await self.feature_store.register_feature(
            FeatureSpec(
                name="income",
                dtype="float",
                description="Annual income",
                validation_rules={"min": 0, "type": float},
            )
        )

        await self.feature_store.register_feature(
            FeatureSpec(
                name="credit_score",
                dtype="int",
                description="Credit score",
                validation_rules={"min": 300, "max": 850, "type": int},
            )
        )

        # Store sample feature values
        entities = ["customer_1", "customer_2", "customer_3"]

        for entity in entities:
            await self.feature_store.store_feature_value(
                FeatureValue(
                    feature_name="age",
                    value=np.random.randint(25, 65),
                    entity_id=entity,
                    timestamp=datetime.now(),
                )
            )

            await self.feature_store.store_feature_value(
                FeatureValue(
                    feature_name="income",
                    value=float(np.random.randint(30000, 150000)),
                    entity_id=entity,
                    timestamp=datetime.now(),
                )
            )

            await self.feature_store.store_feature_value(
                FeatureValue(
                    feature_name="credit_score",
                    value=np.random.randint(600, 800),
                    entity_id=entity,
                    timestamp=datetime.now(),
                )
            )

        # Create and register a dummy model
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, n_features=3, random_state=42)
        model = LogisticRegression()
        model.fit(X, y)

        model_id = self.model_registry.register_model(
            "demo_model",
            model,
            {
                "name": "Demo Logistic Regression",
                "type": "classification",
                "features": ["age", "income", "credit_score"],
                "target": "loan_approval",
                "accuracy": 0.85,
            },
        )

        # Create serving endpoint
        await self.serving_engine.create_model_endpoint(model_id)

        # Create sample workflow
        workflow = MLWorkflow(
            workflow_id="daily_batch_scoring",
            name="Daily Batch Scoring",
            description="Daily batch scoring of new customers",
            steps=[
                {"type": "data_processing", "script_path": "scripts/prepare_data.py"},
                {
                    "type": "batch_inference",
                    "model_id": model_id,
                    "input_path": "data/new_customers.csv",
                    "output_path": "data/scored_customers.csv",
                },
            ],
            schedule="09:00",
        )

        await self.workflow_engine.create_workflow(workflow)

        print("Demo environment setup complete!")
        print(f"Registered features: {len(await self.feature_store.list_features())}")
        print(f"Registered models: {len(self.model_registry.list_models())}")
        print(f"Active endpoints: {len(self.serving_engine.endpoints)}")
        print(f"Created workflows: {len(self.workflow_engine.list_workflows())}")

    def start_api_server(self, host: str = "127.0.0.1", port: int = 8080):
        """Start the API server"""
        self.serving_engine.start_server(host, port)

    async def get_infrastructure_status(self) -> Dict[str, Any]:
        """Get infrastructure status"""

        features = await self.feature_store.list_features()
        models = self.model_registry.list_models()
        endpoints = list(self.serving_engine.endpoints.values())
        workflows = self.workflow_engine.list_workflows()
        batch_jobs = self.batch_engine.list_jobs()

        return {
            "feature_store": {
                "features_count": len(features),
                "features": [f.name for f in features],
            },
            "model_registry": {"models_count": len(models), "models": models},
            "serving": {
                "endpoints_count": len(endpoints),
                "ready_endpoints": len(
                    [e for e in endpoints if e.status == ServingStatus.READY]
                ),
                "total_requests": sum(e.request_count for e in endpoints),
            },
            "batch_processing": {
                "total_jobs": len(batch_jobs),
                "completed_jobs": len(
                    [j for j in batch_jobs if j.status == WorkflowStatus.COMPLETED]
                ),
                "failed_jobs": len(
                    [j for j in batch_jobs if j.status == WorkflowStatus.FAILED]
                ),
            },
            "workflows": {
                "total_workflows": len(workflows),
                "scheduled_workflows": len([w for w in workflows if w.schedule]),
            },
            "system_status": "healthy" if len(models) > 0 else "no_models",
            "last_updated": datetime.now().isoformat(),
        }


# Demo and testing functions
async def demo_personal_ml_infrastructure():
    """Demonstrate personal ML infrastructure capabilities"""

    print("Personal ML Infrastructure Demo")

    # Initialize infrastructure
    config = {
        "feature_store_db": "data/demo_feature_store.db",
        "models_dir": "data/demo_models",
    }

    infrastructure = PersonalMLInfrastructure(config)

    print("\n1. Setting up demo environment...")
    await infrastructure.setup_demo_environment()

    print("\n2. Feature Store Operations")

    # Get feature vector
    feature_vector = await infrastructure.feature_store.get_feature_vector(
        ["age", "income", "credit_score"], "customer_1"
    )
    print(f"Feature vector for customer_1: {feature_vector}")

    # List features
    features = await infrastructure.feature_store.list_features()
    print(f"Registered features: {[f.name for f in features]}")

    print("\n3. Model Registry Operations")

    # List models
    models = infrastructure.model_registry.list_models()
    print(f"Available models: {models}")

    # Get model metadata
    if models:
        metadata = infrastructure.model_registry.get_model_metadata(models[0])
        print(f"Model metadata: {metadata}")

    print("\n4. Model Serving")

    # Test prediction (simulate API call)
    if FASTAPI_AVAILABLE and models:
        try:
            request = {
                "model_id": models[0],
                "features": {"age": 35, "income": 75000, "credit_score": 720},
                "return_confidence": True,
            }

            # This would be an actual API call in practice
            model = infrastructure.model_registry.load_model(models[0])
            features = np.array([[35, 75000, 720]])
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features)[0].max()

            print(f"Prediction result: {prediction}")
            print(f"Confidence: {confidence:.3f}")

        except Exception as e:
            print(f"Prediction test failed: {e}")

    print("\n5. Batch Inference")

    # Create sample input data
    input_data = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "income": [40000, 75000, 90000, 120000],
            "credit_score": [650, 720, 750, 800],
        }
    )

    input_path = Path("data/demo_input.csv")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_data.to_csv(input_path, index=False)

    # Submit batch job
    if models:
        job_id = await infrastructure.batch_engine.submit_batch_job(
            models[0], str(input_path), "data/demo_output.csv"
        )

        print(f"Submitted batch job: {job_id}")

        # Wait for completion
        for i in range(10):
            job = infrastructure.batch_engine.get_job_status(job_id)
            print(f"Job status: {job.status.value}")

            if job.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                break

            await asyncio.sleep(1)

        if job.status == WorkflowStatus.COMPLETED:
            print(f"Processed {job.processed_records} records")

            # Check output
            output_path = Path("data/demo_output.csv")
            if output_path.exists():
                results = pd.read_csv(output_path)
                print(f"Output shape: {results.shape}")
                print(f"Sample predictions: {results.head()}")

    print("\n6. Workflow Engine")

    # List workflows
    workflows = infrastructure.workflow_engine.list_workflows()
    print(f"Available workflows: {[w.name for w in workflows]}")

    # Run a workflow
    if workflows:
        workflow_id = workflows[0].workflow_id
        print(f"Running workflow: {workflow_id}")

        # Create dummy script file for demo
        script_path = Path("scripts/prepare_data.py")
        script_path.parent.mkdir(parents=True, exist_ok=True)

        with open(script_path, "w") as f:
            f.write('print("Data preparation complete")\n')

        try:
            run_id = await infrastructure.workflow_engine.run_workflow(workflow_id)
            print(f"Workflow run ID: {run_id}")

            # Check run status
            run_info = infrastructure.workflow_engine.get_workflow_run(run_id)
            print(f"Run status: {run_info['status'].value}")
            print(f"Steps completed: {run_info['steps_completed']}")

        except Exception as e:
            print(f"Workflow execution failed: {e}")

    print("\n7. Infrastructure Status")

    status = await infrastructure.get_infrastructure_status()
    print(f"System status: {status['system_status']}")
    print(f"Features: {status['feature_store']['features_count']}")
    print(f"Models: {status['model_registry']['models_count']}")
    print(f"Endpoints: {status['serving']['endpoints_count']}")
    print(f"Batch jobs: {status['batch_processing']['total_jobs']}")
    print(f"Workflows: {status['workflows']['total_workflows']}")

    print("\n8. API Server Information")

    if FASTAPI_AVAILABLE:
        print(
            "FastAPI server available - start with: infrastructure.start_api_server()"
        )
        print("API endpoints:")
        print("- POST /predict - Make predictions")
        print("- GET /models - List models")
        print("- GET /models/{model_id}/metadata - Get model metadata")
        print("- GET /endpoints - List serving endpoints")
        print("- POST /endpoints - Create serving endpoint")
    else:
        print("FastAPI not available - API serving disabled")

    return infrastructure


if __name__ == "__main__":
    asyncio.run(demo_personal_ml_infrastructure())
