"""
Dependency injection container for the IDP system.
"""
from typing import Dict, Type, Any, Optional, TypeVar, Generic, cast
from utils.logging_utils import logger

T = TypeVar('T')

class DIContainer:
    """
    Simple dependency injection container.
    """
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register(self, interface: Type[T], implementation: Type[T], singleton: bool = False) -> None:
        """
        Register a service implementation.
        
        Args:
            interface: The interface type
            implementation: The implementation type
            singleton: Whether the service should be a singleton
        """
        service_name = interface.__name__
        self._services[service_name] = {
            'implementation': implementation,
            'singleton': singleton
        }
        logger.debug(f"Registered service: {service_name}")
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """
        Register an existing instance.
        
        Args:
            interface: The interface type
            instance: The instance to register
        """
        service_name = interface.__name__
        self._singletons[service_name] = instance
        logger.debug(f"Registered instance: {service_name}")
    
    def register_factory(self, interface: Type[T], factory_func: Any) -> None:
        """
        Register a factory function.
        
        Args:
            interface: The interface type
            factory_func: Factory function that creates instances
        """
        service_name = interface.__name__
        self._factories[service_name] = factory_func
        logger.debug(f"Registered factory: {service_name}")
    
    def get(self, interface: Type[T]) -> T:
        """
        Get a service instance.
        
        Args:
            interface: The interface type
            
        Returns:
            An instance of the requested service
            
        Raises:
            KeyError: If the service is not registered
        """
        service_name = interface.__name__
        
        # Check for existing singleton
        if service_name in self._singletons:
            return cast(T, self._singletons[service_name])
        
        # Check for factory
        if service_name in self._factories:
            instance = self._factories[service_name](self)
            return cast(T, instance)
        
        # Check for registered service
        if service_name in self._services:
            service_info = self._services[service_name]
            implementation = service_info['implementation']
            singleton = service_info['singleton']
            
            # Create instance
            instance = implementation()
            
            # Store as singleton if required
            if singleton:
                self._singletons[service_name] = instance
            
            return cast(T, instance)
        
        raise KeyError(f"Service not registered: {service_name}")


# Global container instance
container = DIContainer()
