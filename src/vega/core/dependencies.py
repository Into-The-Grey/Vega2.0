"""
dependencies.py - Dependency Injection Container

Provides a simple DI container for managing service dependencies.
Makes testing easier by allowing dependency overrides.

Usage:
    # Register services
    container = get_container()
    container.register("llm", LLMProvider())

    # Resolve services
    llm = container.resolve("llm")

    # Override for testing
    with container.override("llm", MockLLM()):
        # Tests run with mock
        pass
"""

from __future__ import annotations

import threading
from typing import Dict, Any, Optional, TypeVar, Type, Callable
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DependencyContainer:
    """
    Simple dependency injection container.

    Features:
    - Singleton by default
    - Factory support for lazy initialization
    - Thread-safe
    - Override context for testing
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._overrides: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def register(self, name: str, instance: Any) -> None:
        """Register a service instance"""
        with self._lock:
            self._services[name] = instance
            logger.debug(f"Registered service: {name}")

    def register_factory(self, name: str, factory: Callable[[], Any]) -> None:
        """Register a factory function for lazy initialization"""
        with self._lock:
            self._factories[name] = factory
            logger.debug(f"Registered factory: {name}")

    def resolve(self, name: str) -> Any:
        """
        Resolve a service by name.

        Resolution order:
        1. Overrides (for testing)
        2. Registered instances
        3. Factories (lazy init)
        """
        with self._lock:
            # Check overrides first
            if name in self._overrides:
                return self._overrides[name]

            # Check registered instances
            if name in self._services:
                return self._services[name]

            # Check factories
            if name in self._factories:
                instance = self._factories[name]()
                self._services[name] = instance  # Cache for future calls
                return instance

            raise KeyError(f"Service not found: {name}")

    def resolve_optional(self, name: str, default: T = None) -> Optional[T]:
        """Resolve a service, returning default if not found"""
        try:
            return self.resolve(name)
        except KeyError:
            return default

    @contextmanager
    def override(self, name: str, instance: Any):
        """
        Temporarily override a service (useful for testing).

        Usage:
            with container.override("llm", MockLLM()):
                # Tests run with mock
                result = container.resolve("llm")
        """
        with self._lock:
            self._overrides[name] = instance
        try:
            yield
        finally:
            with self._lock:
                del self._overrides[name]

    def clear(self) -> None:
        """Clear all registered services"""
        with self._lock:
            self._services.clear()
            self._factories.clear()
            self._overrides.clear()

    def list_services(self) -> list[str]:
        """List all registered service names"""
        with self._lock:
            return list(set(self._services.keys()) | set(self._factories.keys()))


# Global container instance
_container: Optional[DependencyContainer] = None
_container_lock = threading.Lock()


def get_container() -> DependencyContainer:
    """Get or create the global dependency container"""
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = DependencyContainer()
    return _container


def inject(service_name: str):
    """
    Decorator to inject a dependency into a function.

    Usage:
        @inject("llm")
        def my_function(prompt: str, llm=None):
            return llm.generate(prompt)
    """

    def decorator(func: Callable) -> Callable:
        import functools
        import inspect

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if service_name not in kwargs or kwargs[service_name] is None:
                kwargs[service_name] = get_container().resolve(service_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Register core services on import
def register_core_services():
    """Register core Vega services with the container"""
    container = get_container()

    # Register config (lazy)
    def config_factory():
        from .config import get_config

        return get_config()

    container.register_factory("config", config_factory)

    # Register database (lazy)
    def db_factory():
        from .db import engine

        return engine

    container.register_factory("database", db_factory)

    # Register LLM (lazy)
    def llm_factory():
        from .llm import query_llm

        return query_llm

    container.register_factory("llm", llm_factory)

    logger.info("Core services registered with DI container")
