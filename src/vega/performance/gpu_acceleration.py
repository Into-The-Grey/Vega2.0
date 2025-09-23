"""
GPU Acceleration System

Provides CUDA acceleration, GPU memory management, and optimized
computing for AI workloads in personal environments.
"""

import asyncio
import logging
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager, contextmanager
import json
import subprocess
import psutil

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.cuda

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU acceleration disabled")

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available - some optimizations disabled")

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.info("CuPy not available - using PyTorch for GPU operations")


class GPUBackend(Enum):
    """GPU computation backends"""

    CUDA = "cuda"
    OPENCL = "opencl"
    CPU_FALLBACK = "cpu"
    AUTO = "auto"


class ComputeType(Enum):
    """Types of GPU computations"""

    INFERENCE = "inference"
    TRAINING = "training"
    MATRIX_OPS = "matrix_ops"
    IMAGE_PROCESSING = "image_processing"
    VECTOR_OPS = "vector_ops"
    CUSTOM = "custom"


class GPUMemoryStrategy(Enum):
    """GPU memory management strategies"""

    CONSERVATIVE = "conservative"  # Minimal memory usage
    BALANCED = "balanced"  # Balance speed/memory
    AGGRESSIVE = "aggressive"  # Maximum performance
    CUSTOM = "custom"


@dataclass
class GPUDevice:
    """GPU device information"""

    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    temperature_c: Optional[int] = None
    power_usage_w: Optional[int] = None
    utilization_percent: Optional[int] = None
    is_available: bool = True


@dataclass
class ComputeTask:
    """GPU compute task definition"""

    task_id: str
    compute_type: ComputeType
    input_data: Any
    function: Callable
    priority: int = 1
    device_preference: Optional[int] = None
    memory_requirement_mb: Optional[int] = None
    estimated_duration_ms: Optional[float] = None


@dataclass
class ComputeResult:
    """GPU compute task result"""

    task_id: str
    result: Any
    execution_time_ms: float
    device_used: int
    memory_used_mb: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class GPUStats:
    """GPU performance statistics"""

    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    total_execution_time_ms: float
    average_execution_time_ms: float
    peak_memory_usage_mb: float
    current_memory_usage_mb: float
    device_utilization: Dict[int, float]
    last_updated: datetime

    def __post_init__(self):
        if isinstance(self.last_updated, str):
            self.last_updated = datetime.fromisoformat(self.last_updated)


class GPUDeviceManager:
    """
    Manages GPU devices and resource allocation
    """

    def __init__(self):
        self.devices: Dict[int, GPUDevice] = {}
        self.backend = GPUBackend.AUTO
        self.is_initialized = False

    def initialize(self) -> bool:
        """Initialize GPU device manager"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - using CPU fallback")
            self.backend = GPUBackend.CPU_FALLBACK
            return False

        # Detect CUDA availability
        if torch.cuda.is_available():
            self.backend = GPUBackend.CUDA
            self._detect_cuda_devices()
        else:
            logger.info("CUDA not available - using CPU fallback")
            self.backend = GPUBackend.CPU_FALLBACK
            return False

        self.is_initialized = True
        logger.info(f"GPU manager initialized with {len(self.devices)} devices")
        return True

    def _detect_cuda_devices(self):
        """Detect and catalog CUDA devices"""
        device_count = torch.cuda.device_count()

        for device_id in range(device_count):
            props = torch.cuda.get_device_properties(device_id)

            # Get memory info
            torch.cuda.set_device(device_id)
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            cached_memory = torch.cuda.memory_reserved(device_id)

            free_memory = total_memory - allocated_memory

            device = GPUDevice(
                device_id=device_id,
                name=props.name,
                compute_capability=(props.major, props.minor),
                total_memory_gb=total_memory / (1024**3),
                free_memory_gb=free_memory / (1024**3),
                used_memory_gb=allocated_memory / (1024**3),
            )

            # Try to get additional device info
            try:
                self._update_device_status(device)
            except Exception as e:
                logger.warning(
                    f"Could not get extended info for device {device_id}: {e}"
                )

            self.devices[device_id] = device
            logger.info(
                f"Detected GPU {device_id}: {device.name} ({device.total_memory_gb:.1f}GB)"
            )

    def _update_device_status(self, device: GPUDevice):
        """Update device status with current metrics"""
        try:
            # Try to get GPU utilization via nvidia-ml-py if available
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device.device_id)

            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            device.temperature_c = temp

            # Power
            power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert to watts
            device.power_usage_w = power

            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            device.utilization_percent = util.gpu

        except ImportError:
            logger.debug("pynvml not available - extended GPU monitoring disabled")
        except Exception as e:
            logger.debug(f"Could not get GPU status: {e}")

    def get_optimal_device(
        self,
        memory_requirement_mb: Optional[int] = None,
        compute_type: ComputeType = ComputeType.INFERENCE,
    ) -> Optional[int]:
        """Get optimal device for computation"""
        if not self.devices:
            return None

        available_devices = []

        for device_id, device in self.devices.items():
            if not device.is_available:
                continue

            # Check memory requirement
            if memory_requirement_mb:
                free_memory_mb = device.free_memory_gb * 1024
                if free_memory_mb < memory_requirement_mb:
                    continue

            # Calculate device score
            score = self._calculate_device_score(device, compute_type)
            available_devices.append((device_id, score))

        if not available_devices:
            return None

        # Return device with highest score
        available_devices.sort(key=lambda x: x[1], reverse=True)
        return available_devices[0][0]

    def _calculate_device_score(
        self, device: GPUDevice, compute_type: ComputeType
    ) -> float:
        """Calculate device suitability score"""
        score = 0.0

        # Base score from compute capability
        score += device.compute_capability[0] * 10 + device.compute_capability[1]

        # Memory availability
        memory_ratio = device.free_memory_gb / device.total_memory_gb
        score += memory_ratio * 20

        # Utilization (prefer less utilized devices)
        if device.utilization_percent is not None:
            score += (100 - device.utilization_percent) * 0.1

        # Temperature (prefer cooler devices)
        if device.temperature_c is not None:
            if device.temperature_c < 70:
                score += 5
            elif device.temperature_c > 85:
                score -= 10

        return score

    def update_device_memory(self, device_id: int):
        """Update device memory information"""
        if device_id not in self.devices:
            return

        device = self.devices[device_id]

        if self.backend == GPUBackend.CUDA and TORCH_AVAILABLE:
            torch.cuda.set_device(device_id)
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)

            device.used_memory_gb = allocated_memory / (1024**3)
            device.free_memory_gb = (total_memory - allocated_memory) / (1024**3)

    def get_device_info(
        self, device_id: Optional[int] = None
    ) -> Union[GPUDevice, Dict[int, GPUDevice]]:
        """Get device information"""
        if device_id is not None:
            return self.devices.get(device_id)
        return self.devices.copy()


class GPUMemoryManager:
    """
    Intelligent GPU memory management
    """

    def __init__(self, strategy: GPUMemoryStrategy = GPUMemoryStrategy.BALANCED):
        self.strategy = strategy
        self.memory_pools: Dict[int, List[torch.Tensor]] = {}
        self.allocation_history: List[Tuple[int, int, datetime]] = (
            []
        )  # device, size_mb, timestamp
        self.peak_usage: Dict[int, float] = {}

    def allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Allocate GPU tensor with memory management"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for GPU operations")

        if device_id is None:
            device_id = 0  # Default device

        device = torch.device(f"cuda:{device_id}")

        # Try to reuse pooled memory
        if self.strategy in [GPUMemoryStrategy.BALANCED, GPUMemoryStrategy.AGGRESSIVE]:
            tensor = self._try_reuse_memory(shape, dtype, device_id)
            if tensor is not None:
                return tensor

        # Allocate new tensor
        try:
            tensor = torch.empty(shape, dtype=dtype, device=device)

            # Track allocation
            size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            self.allocation_history.append((device_id, size_mb, datetime.now()))

            # Update peak usage
            current_usage = torch.cuda.memory_allocated(device_id) / (1024 * 1024)
            if (
                device_id not in self.peak_usage
                or current_usage > self.peak_usage[device_id]
            ):
                self.peak_usage[device_id] = current_usage

            return tensor

        except torch.cuda.OutOfMemoryError:
            # Try memory cleanup and retry
            self.cleanup_memory(device_id)
            torch.cuda.empty_cache()

            try:
                tensor = torch.empty(shape, dtype=dtype, device=device)
                return tensor
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(f"Insufficient GPU memory on device {device_id}")

    def _try_reuse_memory(
        self, shape: Tuple[int, ...], dtype: torch.dtype, device_id: int
    ) -> Optional[torch.Tensor]:
        """Try to reuse pooled memory"""
        if device_id not in self.memory_pools:
            return None

        pool = self.memory_pools[device_id]

        # Look for compatible tensor
        for i, tensor in enumerate(pool):
            if (
                tensor.shape == shape
                and tensor.dtype == dtype
                and tensor.device.index == device_id
            ):
                # Remove from pool and return
                return pool.pop(i)

        return None

    def release_tensor(self, tensor: torch.Tensor):
        """Release tensor back to pool or free memory"""
        if not TORCH_AVAILABLE or tensor.device.type != "cuda":
            return

        device_id = tensor.device.index

        if self.strategy == GPUMemoryStrategy.CONSERVATIVE:
            # Immediately free memory
            del tensor
            torch.cuda.empty_cache()
            return

        # Add to pool for reuse
        if device_id not in self.memory_pools:
            self.memory_pools[device_id] = []

        # Limit pool size
        pool = self.memory_pools[device_id]
        if len(pool) < 10:  # Max 10 tensors per device
            pool.append(tensor.detach())
        else:
            del tensor

    def cleanup_memory(self, device_id: Optional[int] = None):
        """Clean up GPU memory"""
        if not TORCH_AVAILABLE:
            return

        if device_id is not None:
            # Clean specific device
            if device_id in self.memory_pools:
                del self.memory_pools[device_id]
            torch.cuda.empty_cache()
        else:
            # Clean all devices
            self.memory_pools.clear()
            torch.cuda.empty_cache()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        stats = {
            "strategy": self.strategy.value,
            "peak_usage_mb": dict(self.peak_usage),
            "pool_sizes": {k: len(v) for k, v in self.memory_pools.items()},
            "total_allocations": len(self.allocation_history),
        }

        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Current memory usage
            stats["current_usage_mb"] = {}
            for device_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(device_id) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(device_id) / (1024 * 1024)
                stats["current_usage_mb"][device_id] = {
                    "allocated": allocated,
                    "reserved": reserved,
                }

        return stats


class GPUComputeEngine:
    """
    High-performance GPU compute engine
    """

    def __init__(self, memory_strategy: GPUMemoryStrategy = GPUMemoryStrategy.BALANCED):
        self.device_manager = GPUDeviceManager()
        self.memory_manager = GPUMemoryManager(memory_strategy)
        self.task_queue = asyncio.Queue()
        self.results: Dict[str, ComputeResult] = {}
        self.stats = GPUStats(
            total_tasks=0,
            successful_tasks=0,
            failed_tasks=0,
            total_execution_time_ms=0.0,
            average_execution_time_ms=0.0,
            peak_memory_usage_mb=0.0,
            current_memory_usage_mb=0.0,
            device_utilization={},
            last_updated=datetime.now(),
        )
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []

    def initialize(self):
        """Initialize GPU compute engine"""
        gpu_available = self.device_manager.initialize()

        if not gpu_available:
            logger.warning("No GPU available - compute engine will use CPU fallback")

        return gpu_available

    async def start_workers(self, num_workers: int = 2):
        """Start compute worker tasks"""
        self.is_running = True

        for i in range(num_workers):
            worker = asyncio.create_task(self._compute_worker(f"worker-{i}"))
            self.worker_tasks.append(worker)

        logger.info(f"Started {num_workers} GPU compute workers")

    async def stop_workers(self):
        """Stop compute worker tasks"""
        self.is_running = False

        # Cancel all workers
        for task in self.worker_tasks:
            task.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()

        logger.info("Stopped GPU compute workers")

    async def _compute_worker(self, worker_id: str):
        """GPU compute worker task"""
        logger.info(f"GPU worker {worker_id} started")

        while self.is_running:
            try:
                # Get next task
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # Execute task
                result = await self._execute_task(task)

                # Store result
                self.results[task.task_id] = result

                # Update statistics
                self._update_stats(result)

                # Mark task done
                self.task_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def _execute_task(self, task: ComputeTask) -> ComputeResult:
        """Execute compute task on GPU"""
        start_time = time.time()

        # Select optimal device
        device_id = self.device_manager.get_optimal_device(
            memory_requirement_mb=task.memory_requirement_mb,
            compute_type=task.compute_type,
        )

        if device_id is None:
            # Fallback to CPU
            device_id = -1  # CPU indicator

        try:
            # Execute task function
            if device_id >= 0 and TORCH_AVAILABLE:
                # GPU execution
                with torch.cuda.device(device_id):
                    result = await self._execute_gpu_task(task, device_id)
            else:
                # CPU fallback
                result = await self._execute_cpu_task(task)

            execution_time = (time.time() - start_time) * 1000

            # Calculate memory usage
            memory_used = 0.0
            if device_id >= 0:
                memory_used = torch.cuda.memory_allocated(device_id) / (1024 * 1024)

            return ComputeResult(
                task_id=task.task_id,
                result=result,
                execution_time_ms=execution_time,
                device_used=device_id,
                memory_used_mb=memory_used,
                success=True,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            return ComputeResult(
                task_id=task.task_id,
                result=None,
                execution_time_ms=execution_time,
                device_used=device_id,
                memory_used_mb=0.0,
                success=False,
                error_message=str(e),
            )

    async def _execute_gpu_task(self, task: ComputeTask, device_id: int) -> Any:
        """Execute task on GPU"""
        # Move input data to GPU if needed
        gpu_data = self._move_to_gpu(task.input_data, device_id)

        # Execute function
        if asyncio.iscoroutinefunction(task.function):
            result = await task.function(gpu_data)
        else:
            # Run in thread pool for non-async functions
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, task.function, gpu_data)

        # Move result back to CPU if needed
        cpu_result = self._move_to_cpu(result)

        return cpu_result

    async def _execute_cpu_task(self, task: ComputeTask) -> Any:
        """Execute task on CPU"""
        if asyncio.iscoroutinefunction(task.function):
            result = await task.function(task.input_data)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, task.function, task.input_data)

        return result

    def _move_to_gpu(self, data: Any, device_id: int) -> Any:
        """Move data to GPU"""
        if not TORCH_AVAILABLE:
            return data

        if isinstance(data, torch.Tensor):
            return data.to(f"cuda:{device_id}")
        elif isinstance(data, dict):
            return {k: self._move_to_gpu(v, device_id) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._move_to_gpu(item, device_id) for item in data)
        else:
            return data

    def _move_to_cpu(self, data: Any) -> Any:
        """Move data to CPU"""
        if not TORCH_AVAILABLE:
            return data

        if isinstance(data, torch.Tensor):
            return data.cpu()
        elif isinstance(data, dict):
            return {k: self._move_to_cpu(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._move_to_cpu(item) for item in data)
        else:
            return data

    def _update_stats(self, result: ComputeResult):
        """Update performance statistics"""
        self.stats.total_tasks += 1
        self.stats.total_execution_time_ms += result.execution_time_ms

        if result.success:
            self.stats.successful_tasks += 1
        else:
            self.stats.failed_tasks += 1

        self.stats.average_execution_time_ms = (
            self.stats.total_execution_time_ms / self.stats.total_tasks
        )

        if result.memory_used_mb > self.stats.peak_memory_usage_mb:
            self.stats.peak_memory_usage_mb = result.memory_used_mb

        self.stats.current_memory_usage_mb = result.memory_used_mb
        self.stats.last_updated = datetime.now()

    async def submit_task(self, task: ComputeTask) -> str:
        """Submit compute task"""
        await self.task_queue.put(task)
        return task.task_id

    async def get_result(
        self, task_id: str, timeout: float = 30.0
    ) -> Optional[ComputeResult]:
        """Get task result"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if task_id in self.results:
                return self.results.pop(task_id)
            await asyncio.sleep(0.1)

        return None  # Timeout

    async def matrix_multiply(
        self, a: Union[List, torch.Tensor], b: Union[List, torch.Tensor]
    ) -> torch.Tensor:
        """Optimized matrix multiplication"""
        task_id = f"matmul_{int(time.time() * 1000)}"

        def matmul_func(data):
            a_tensor, b_tensor = data
            if not isinstance(a_tensor, torch.Tensor):
                a_tensor = torch.tensor(a_tensor, dtype=torch.float32)
            if not isinstance(b_tensor, torch.Tensor):
                b_tensor = torch.tensor(b_tensor, dtype=torch.float32)

            return torch.matmul(a_tensor, b_tensor)

        task = ComputeTask(
            task_id=task_id,
            compute_type=ComputeType.MATRIX_OPS,
            input_data=(a, b),
            function=matmul_func,
            priority=1,
        )

        await self.submit_task(task)
        result = await self.get_result(task_id)

        return result.result if result and result.success else None

    async def vector_operations(
        self, vectors: List[torch.Tensor], operation: str = "sum"
    ) -> torch.Tensor:
        """Optimized vector operations"""
        task_id = f"vecops_{int(time.time() * 1000)}"

        def vecops_func(data):
            vecs, op = data

            if op == "sum":
                return torch.stack(vecs).sum(dim=0)
            elif op == "mean":
                return torch.stack(vecs).mean(dim=0)
            elif op == "max":
                return torch.stack(vecs).max(dim=0)[0]
            elif op == "min":
                return torch.stack(vecs).min(dim=0)[0]
            else:
                raise ValueError(f"Unsupported operation: {op}")

        task = ComputeTask(
            task_id=task_id,
            compute_type=ComputeType.VECTOR_OPS,
            input_data=(vectors, operation),
            function=vecops_func,
            priority=1,
        )

        await self.submit_task(task)
        result = await self.get_result(task_id)

        return result.result if result and result.success else None

    def get_compute_stats(self) -> Dict[str, Any]:
        """Get comprehensive compute statistics"""
        stats_dict = asdict(self.stats)

        # Add device information
        stats_dict["devices"] = {}
        for device_id, device in self.device_manager.devices.items():
            stats_dict["devices"][device_id] = asdict(device)

        # Add memory statistics
        stats_dict["memory"] = self.memory_manager.get_memory_stats()

        # Add queue status
        stats_dict["queue_size"] = self.task_queue.qsize()
        stats_dict["active_workers"] = len(self.worker_tasks)

        return stats_dict


# Demo and testing functions
async def demo_gpu_acceleration():
    """Demonstrate GPU acceleration capabilities"""

    print("GPU Acceleration Demo")

    # Initialize compute engine
    compute_engine = GPUComputeEngine(GPUMemoryStrategy.BALANCED)
    gpu_available = compute_engine.initialize()

    if gpu_available:
        print(f"GPU acceleration enabled")
        devices = compute_engine.device_manager.get_device_info()
        for device_id, device in devices.items():
            print(
                f"- Device {device_id}: {device.name} ({device.total_memory_gb:.1f}GB)"
            )
    else:
        print("GPU acceleration not available - using CPU fallback")

    # Start workers
    await compute_engine.start_workers(num_workers=2)

    # Test matrix multiplication
    print("\nTesting matrix multiplication...")
    if TORCH_AVAILABLE:
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)

        start_time = time.time()
        result = await compute_engine.matrix_multiply(a, b)
        execution_time = (time.time() - start_time) * 1000

        if result is not None:
            print(
                f"Matrix multiplication: {result.shape} result in {execution_time:.2f}ms"
            )
        else:
            print("Matrix multiplication failed")

    # Test vector operations
    print("\nTesting vector operations...")
    if TORCH_AVAILABLE:
        vectors = [torch.randn(10000) for _ in range(100)]

        start_time = time.time()
        result = await compute_engine.vector_operations(vectors, "mean")
        execution_time = (time.time() - start_time) * 1000

        if result is not None:
            print(f"Vector mean: {result.shape} result in {execution_time:.2f}ms")
        else:
            print("Vector operations failed")

    # Get statistics
    stats = compute_engine.get_compute_stats()
    print(f"\nCompute Statistics:")
    print(f"- Total tasks: {stats['total_tasks']}")
    print(f"- Success rate: {stats['successful_tasks']}/{stats['total_tasks']}")
    print(f"- Average execution time: {stats['average_execution_time_ms']:.2f}ms")
    print(f"- Peak memory usage: {stats['peak_memory_usage_mb']:.1f}MB")
    print(f"- Active workers: {stats['active_workers']}")

    # Stop workers
    await compute_engine.stop_workers()

    return compute_engine


if __name__ == "__main__":
    asyncio.run(demo_gpu_acceleration())
